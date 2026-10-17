"""Build match emails and send them with Resend (default) or SMTP."""

from __future__ import annotations

import html
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import make_msgid

import httpx

from .rules import unit_name
from .settings import RESEND_DEFAULT_FROM

RESEND_URL = "https://api.resend.com/emails"
http_transport: httpx.BaseTransport | None = None  # tests swap in a mock transport


class EmailError(Exception):
    pass


def recipients(email_cfg: dict) -> list[str]:
    return [r.strip() for r in email_cfg["to"].replace(";", ",").split(",") if r.strip()]


def config_problem(email_cfg: dict) -> str:
    """Why email cannot be sent with this configuration, or '' if it looks usable."""
    if not email_cfg["enabled"]:
        return "Email notifications are turned off in Settings"
    if not recipients(email_cfg):
        return "No recipient address set"
    if email_cfg["method"] == "resend":
        return "" if email_cfg["resend_api_key"] else "Resend API key not set (Settings > Email, or RESEND_API_KEY)"
    if not email_cfg["smtp_host"]:
        return "SMTP server not set"
    if not email_cfg["smtp_username"] and not email_cfg["from_address"]:
        return "SMTP username / From address not set"
    if email_cfg["smtp_username"] and not email_cfg["smtp_password"]:
        return "SMTP password not set (for Gmail use an App Password)"
    return ""


def _attribute_summary(match: dict) -> str:
    a = match["attributes"]
    if match["tracker"] == "ram":
        parts = [
            f"{a.get('sticks', '?')} x {a.get('stick_gb', '?')}GB",
            str(a.get("module_type", "")),
            f"DDR4-{a['speed_mts']}" if isinstance(a.get("speed_mts"), int) else "speed not stated",
        ]
    else:
        parts = [str(a.get("model", "")), f"vendor lock: {a.get('vendor_lock', 'not stated')}"]
    if a.get("condition"):
        parts.append(str(a["condition"]))
    return " · ".join(p for p in parts if p)


def _money(value) -> str:
    return "?" if value is None else f"${value:,.2f}"


def _why(match: dict) -> str:
    if match["notify_reason"] == "price_drop":
        return f"Price drop (was {_money(match.get('previous_unit_price'))} per {unit_name(match['tracker'])})"
    return "New match"


def build_email(matches: list[dict], settings: dict, run_id: int | None, demo: bool = False) -> tuple[str, str, str]:
    """Return (subject, plain text, html) for a batch of matches."""
    trackers = settings["trackers"]
    by_tracker: dict[str, list[dict]] = {}
    for match in sorted(matches, key=lambda m: m["unit_price"] or 0):
        by_tracker.setdefault(match["tracker"], []).append(match)

    headline = []
    for key, items in by_tracker.items():
        unit = unit_name(key)
        short = "64GB DDR4" if key == "ram" else trackers[key]["model"]
        headline.append(f"{short} from {_money(items[0]['unit_price'])}/{unit}")
    count = len(matches)
    prefix = "[Price Tracker DEMO]" if demo else "[Price Tracker]"
    subject = f"{prefix} {count} match{'es' if count != 1 else ''}: " + ", ".join(headline)

    base_url = settings["general"]["public_base_url"]
    run_link = f"{base_url}/runs/{run_id}" if base_url and run_id else ""

    text_lines = []
    if demo:
        text_lines += ["DEMO DATA: these listings are fake and were produced by the demo provider.", ""]
    html_parts = [
        '<div style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1f2328;'
        'max-width:760px">'
    ]
    if demo:
        html_parts.append(
            '<p style="background:#fff8c5;padding:8px 12px;border-radius:6px">'
            "<b>Demo data:</b> these listings are fake and were produced by the demo provider.</p>"
        )
    for key, items in by_tracker.items():
        label = trackers[key]["label"]
        unit = unit_name(key)
        text_lines += [f"== {label} ==", ""]
        html_parts.append(f'<h2 style="font-size:18px;margin:20px 0 8px">{html.escape(label)}</h2>')
        for m in items:
            if m["shipping"] is None:
                shipping = "shipping not stated"
            else:
                shipping = "free shipping" if m["shipping"] == 0 else f"shipping {_money(m['shipping'])}"
            price_line = f"{_money(m['unit_price'])} per {unit}  (total {_money(m['total_price'])}: "
            price_line += f"item {_money(m['price'])}, {shipping})"
            text_lines += [
                f"* {m['title']}",
                f"  {_why(m)} · {m['merchant']}",
                f"  {price_line}",
                f"  {_attribute_summary(m)}",
                *[f"  ! {note}" for note in m["notes"]],
                f"  {m['link']}",
                "",
            ]
            notes_html = "".join(
                f'<div style="color:#9a6700;font-size:13px">&#9888; {html.escape(n)}</div>' for n in m["notes"]
            )
            html_parts.append(
                '<div style="border:1px solid #d0d7de;border-radius:8px;padding:12px 14px;margin:8px 0">'
                f'<div style="font-size:12px;color:#57606a">{html.escape(_why(m))} · {html.escape(m["merchant"])}'
                "</div>"
                f'<a href="{html.escape(m["link"], quote=True)}" style="font-size:15px;font-weight:600;color:#0969da;'
                f'text-decoration:none">{html.escape(m["title"])}</a>'
                f'<div style="font-size:20px;font-weight:700;margin:6px 0 2px">{_money(m["unit_price"])} '
                f'<span style="font-size:13px;font-weight:400;color:#57606a">per {unit}</span></div>'
                f'<div style="font-size:13px;color:#57606a">Total {_money(m["total_price"])} '
                f"(item {_money(m['price'])}, {html.escape(shipping)})</div>"
                f'<div style="font-size:13px;margin-top:4px">{html.escape(_attribute_summary(m))}</div>'
                f"{notes_html}</div>"
            )
    if run_link:
        text_lines.append(f"Scan details: {run_link}")
        html_parts.append(
            f'<p style="font-size:13px"><a href="{html.escape(run_link, quote=True)}">Scan details</a></p>'
        )
    html_parts.append(
        '<p style="font-size:12px;color:#57606a">Sent by your price tracker. Always check the listing '
        "description and seller feedback before buying.</p></div>"
    )
    return subject, "\n".join(text_lines).strip() + "\n", "".join(html_parts)


def sender_address(email_cfg: dict) -> str:
    if email_cfg["from_address"]:
        return email_cfg["from_address"]
    return RESEND_DEFAULT_FROM if email_cfg["method"] == "resend" else email_cfg["smtp_username"]


def send_email(email_cfg: dict, subject: str, text: str, html_body: str | None = None) -> None:
    if email_cfg["method"] == "resend":
        _send_resend(email_cfg, subject, text, html_body)
    else:
        _send_smtp(email_cfg, subject, text, html_body)


def _send_resend(email_cfg: dict, subject: str, text: str, html_body: str | None) -> None:
    payload = {"from": sender_address(email_cfg), "to": recipients(email_cfg), "subject": subject, "text": text}
    if html_body:
        payload["html"] = html_body
    headers = {"Authorization": f"Bearer {email_cfg['resend_api_key']}"}
    try:
        with httpx.Client(transport=http_transport, timeout=30) as client:
            response = client.post(RESEND_URL, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise EmailError(f"Could not reach Resend: {exc.__class__.__name__}: {exc}") from exc
    if response.status_code >= 400:
        try:
            message = response.json().get("message") or response.text
        except ValueError:
            message = response.text
        hint = ""
        if response.status_code == 403 and "resend.dev" in sender_address(email_cfg):
            hint = (
                " (the shared onboarding@resend.dev sender can only email the address your Resend account was "
                "created with; verify a domain in Resend to email anyone else)"
            )
        raise EmailError(f"Resend HTTP {response.status_code}: {str(message)[:300]}{hint}")


def _send_smtp(email_cfg: dict, subject: str, text: str, html_body: str | None) -> None:
    to = recipients(email_cfg)
    sender = sender_address(email_cfg)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    msg["Message-ID"] = make_msgid(domain=sender.split("@")[-1] if "@" in sender else None)
    msg.set_content(text)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    host, port, security = email_cfg["smtp_host"], int(email_cfg["smtp_port"]), email_cfg["smtp_security"]
    try:
        if security == "ssl":
            server = smtplib.SMTP_SSL(host, port, timeout=30, context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(host, port, timeout=30)
        with server:
            if security == "starttls":
                server.starttls(context=ssl.create_default_context())
            if email_cfg["smtp_username"]:
                server.login(email_cfg["smtp_username"], email_cfg["smtp_password"])
            server.send_message(msg)
    except (smtplib.SMTPException, OSError) as exc:
        raise EmailError(f"{exc.__class__.__name__}: {exc}") from exc
