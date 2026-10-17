"""Web UI: dashboard, scan history, listings, merchants, emails, rule playground and settings."""

from __future__ import annotations

import base64
import binascii
import copy
import json
import logging
import secrets
from collections import Counter
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode, urlparse
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import pass_context

from . import __version__, config
from .db import Database
from .merchants import judge_merchant, normalize_merchant
from .models import Offer
from .notifier import EmailError, config_problem, recipients, send_email
from .rules import REASON_LABELS, TRACKER_KINDS, evaluate, unit_name
from .scanner import ScanAlreadyRunning, is_running, replay_run, start_scan
from .scheduler import ScanScheduler
from .settings import (
    DEFAULT_SETTINGS,
    EMAIL_METHODS,
    LOCK_POLICIES,
    MODULE_TYPES,
    RESEND_DEFAULT_FROM,
    SMTP_SECURITY,
    effective_settings,
    format_trusted_merchants,
    load_settings,
    paid_requests_per_month,
    save_settings,
    settings_from_form,
    value_source,
)
from .sources import SOURCE_LABELS, ProviderError, build_sources, parse_price, parse_shipping

log = logging.getLogger(__name__)
HERE = Path(__file__).parent
_CONFIG_CHECK_CLIENT = httpx.Client()  # handed to sources built only to check their settings; never used
PAGE_SIZE = 100


def create_app(db_path: str | Path | None = None, start_scheduler: bool = True) -> FastAPI:
    db = Database(db_path or config.DB_PATH)
    scheduler = ScanScheduler(db)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        interrupted = db.mark_interrupted_runs()
        if interrupted:
            log.warning("Marked %s interrupted scan(s) as failed", interrupted)
        if start_scheduler:
            scheduler.start()
        yield
        scheduler.shutdown()

    app = FastAPI(title="Price Tracker", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.db = db
    app.state.scheduler = scheduler
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = _templates()

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if config.UI_PASSWORD and request.url.path != "/healthz" and not _authorized(request):
            return Response("Authentication required", 401, {"WWW-Authenticate": 'Basic realm="Price Tracker"'})
        if request.method == "POST" and not _same_origin(request):
            return Response("Cross-origin form posts are not allowed", 403)
        return await call_next(request)

    def render(request: Request, name: str, **context) -> HTMLResponse:
        settings = context.pop("settings", None) or load_settings(db)
        context.update(
            settings=settings,
            tz=settings["schedule"]["timezone"],
            demo=settings["sources"]["demo"]["enabled"],
            scan_running=is_running(),
            msg=request.query_params.get("msg", ""),
            err=request.query_params.get("err", ""),
            trackers=settings["trackers"],
            reason_labels=REASON_LABELS,
            version=__version__,
        )
        return templates.TemplateResponse(request, name, context)

    def back(path: str, **params) -> RedirectResponse:
        query = urlencode({k: v for k, v in params.items() if v})
        return RedirectResponse(path + (f"?{query}" if query else ""), status_code=303)

    # --- dashboard & scans ---

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        settings = load_settings(db)
        effective = effective_settings(settings)
        latest = db.latest_completed_run()
        good = db.latest_successful_run()
        observations = db.observations_for_run(good["id"]) if good else []
        current = [o for o in observations if o["decision"] == "accepted"]
        summary = {}
        for key in settings["trackers"]:
            rows = [o for o in observations if o["tracker"] == key]
            accepted = [o for o in rows if o["decision"] == "accepted"]
            summary[key] = {
                "seen": len(rows),
                "accepted": len(accepted),
                "best": min((o["unit_price"] for o in accepted), default=None),
            }
        sources, source_problems = build_sources(effective, client=_CONFIG_CHECK_CLIENT)
        if not sources and not source_problems:
            source_problems = ["No search sources are enabled"]
        return render(
            request,
            "dashboard.html",
            settings=settings,
            latest=latest,
            matches_run=good,
            running=db.running_run(),
            current=current,
            summary=summary,
            recent_runs=db.list_runs(limit=6),
            next_runs=scheduler.next_runs(),
            counts=db.dashboard_counts(),
            last_email=db.last_notification(),
            source_names=[SOURCE_LABELS[s.name] for s in sources],
            source_problems=source_problems,
            email_problem=config_problem(effective["email"]),
        )

    @app.post("/scan")
    async def scan(request: Request):
        form = await request.form()
        try:
            run_id = start_scan(db, trigger="manual", send_email="send_email" in form, background=True)
        except ScanAlreadyRunning:
            running = db.running_run()
            return back(f"/runs/{running['id']}" if running else "/", err="A scan is already running")
        return back(f"/runs/{run_id}", msg="Scan started")

    @app.get("/runs", response_class=HTMLResponse)
    def runs(request: Request, page: int = 1):
        page = max(page, 1)
        total = db.count_runs()
        return render(
            request,
            "runs.html",
            runs=db.list_runs(limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE),
            page=page,
            pages=max(1, -(-total // PAGE_SIZE)),
            total=total,
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: int, decision: str = "", tracker: str = "", reason: str = ""):
        run = db.get_run(run_id)
        if not run:
            raise HTTPException(404, "Scan not found")
        observations = db.observations_for_run(run_id)
        reason_counts = Counter(r["code"] for o in observations for r in o["reasons"])
        shown = [
            o
            for o in observations
            if (not decision or o["decision"] == decision)
            and (not tracker or o["tracker"] == tracker)
            and (not reason or any(r["code"] == reason for r in o["reasons"]))
        ]
        return render(
            request,
            "run_detail.html",
            run=run,
            queries=db.queries_for_run(run_id),
            observations=shown,
            total_observations=len(observations),
            reason_counts=reason_counts.most_common(),
            filters={"decision": decision, "tracker": tracker, "reason": reason},
        )

    @app.get("/runs/{run_id}/replay", response_class=HTMLResponse)
    def run_replay(request: Request, run_id: int, changed: int = 1):
        run = db.get_run(run_id)
        if not run:
            raise HTTPException(404, "Scan not found")
        settings = load_settings(db)
        rows = replay_run(db, run_id, effective_settings(settings))
        changed_rows = [r for r in rows if r["old"]["decision"] != r["new"]["decision"]]
        return render(
            request,
            "replay.html",
            settings=settings,
            run=run,
            rows=changed_rows if changed else rows,
            total=len(rows),
            changed_count=len(changed_rows),
            only_changed=bool(changed),
            now_accepted=sum(1 for r in rows if r["new"]["decision"] == "accepted"),
            was_accepted=sum(1 for r in rows if r["old"]["decision"] == "accepted"),
        )

    # --- listings ---

    @app.get("/listings", response_class=HTMLResponse)
    def listings(
        request: Request,
        tracker: str = "",
        decision: str = "",
        q: str = "",
        merchant: str = "",
        sort: str = "recent",
        page: int = 1,
    ):
        page = max(page, 1)
        rows, total = db.list_listings(tracker, decision, q, merchant, sort, PAGE_SIZE, (page - 1) * PAGE_SIZE)
        filters = {"tracker": tracker, "decision": decision, "q": q, "merchant": merchant, "sort": sort}
        return render(
            request,
            "listings.html",
            listings=rows,
            total=total,
            page=page,
            pages=max(1, -(-total // PAGE_SIZE)),
            filters=filters,
            page_query=urlencode({k: v for k, v in filters.items() if v}),
        )

    @app.get("/listings/{listing_id}", response_class=HTMLResponse)
    def listing_detail(request: Request, listing_id: int):
        listing = db.get_listing(listing_id)
        if not listing:
            raise HTTPException(404, "Listing not found")
        history = db.observations_for_listing(listing_id)
        latest_raw = json.dumps(json.loads(history[0]["raw"]), indent=2) if history and history[0]["raw"] else ""
        return render(request, "listing_detail.html", listing=listing, history=history, latest_raw=latest_raw)

    @app.get("/observations/{observation_id}/raw")
    def observation_raw(observation_id: int):
        obs = db.get_observation(observation_id)
        if not obs:
            raise HTTPException(404, "Observation not found")
        return JSONResponse(json.loads(obs["raw"]) if obs["raw"] else {})

    # --- merchants ---

    @app.get("/merchants", response_class=HTMLResponse)
    def merchants(request: Request, days: int = 30):
        settings = load_settings(db)
        since = (datetime.now(UTC) - timedelta(days=max(days, 1))).isoformat(timespec="seconds")
        rows = db.merchant_stats(since)
        trusted, blocked = settings["merchants"]["trusted"], settings["merchants"]["blocked"]
        for row in rows:
            merchant_url = f"https://{row['domain']}/" if row["domain"] else ""
            offer = Offer(title="", merchant=row["merchant"] or "", price=None, merchant_url=merchant_url)
            row["verdict"] = judge_merchant(offer, trusted, blocked, allow_untrusted=False)
        return render(request, "merchants.html", settings=settings, rows=rows, days=days)

    @app.post("/merchants/trust")
    async def merchant_trust(request: Request):
        form = await request.form()
        merchant, domain = str(form.get("merchant", "")).strip(), str(form.get("domain", "")).strip().lower()
        if not merchant and not domain:
            return back("/merchants", err="Nothing to trust")
        settings = load_settings(db)
        settings["merchants"]["trusted"].append(
            {
                "name": merchant or domain,
                "aliases": [normalize_merchant(merchant)] if merchant else [],
                "domains": [domain] if domain else [],
            }
        )
        settings["merchants"]["blocked"] = [
            b for b in settings["merchants"]["blocked"] if b.lower() not in {merchant.lower(), domain}
        ]
        save_settings(db, settings)
        return back("/merchants", msg=f"Added {merchant or domain} to trusted merchants")

    @app.post("/merchants/block")
    async def merchant_block(request: Request):
        form = await request.form()
        value = str(form.get("value", "")).strip()
        if not value:
            return back("/merchants", err="Nothing to block")
        settings = load_settings(db)
        if value not in settings["merchants"]["blocked"]:
            settings["merchants"]["blocked"].append(value)
        save_settings(db, settings)
        return back("/merchants", msg=f"Blocked {value}")

    @app.post("/merchants/unblock")
    async def merchant_unblock(request: Request):
        form = await request.form()
        value = str(form.get("value", "")).strip()
        settings = load_settings(db)
        settings["merchants"]["blocked"] = [b for b in settings["merchants"]["blocked"] if b != value]
        save_settings(db, settings)
        return back("/merchants", msg=f"Unblocked {value}")

    # --- emails ---

    @app.get("/emails", response_class=HTMLResponse)
    def emails(request: Request):
        settings = load_settings(db)
        return render(
            request,
            "emails.html",
            settings=settings,
            notifications=db.list_notifications(),
            email_problem=config_problem(effective_settings(settings)["email"]),
        )

    # --- playground ---

    @app.get("/playground", response_class=HTMLResponse)
    def playground(request: Request):
        sample = {
            "tracker": "ram",
            "title": "Samsung 64GB 2Rx4 PC4-3200AA-RB2 DDR4 ECC Registered RDIMM M393A8G40AB2-CWE",
            "merchant": "eBay - example_seller",
            "price": "189.99",
            "shipping": "Free delivery",
            "link": "https://www.ebay.com/itm/123456789012",
            "condition": "used",
            "snippet": "",
        }
        return render(request, "playground.html", form=sample, result=None)

    @app.post("/playground", response_class=HTMLResponse)
    async def playground_run(request: Request):
        form = {k: str(v) for k, v in (await request.form()).items()}
        tracker = form.get("tracker", "ram") if form.get("tracker") in TRACKER_KINDS else "ram"
        price, currency = parse_price(form.get("price", ""))
        shipping, shipping_text = parse_shipping(form.get("shipping", ""))
        if shipping is None and form.get("shipping", "").strip():
            amount, _ = parse_price(form["shipping"])
            shipping, shipping_text = amount, form["shipping"]
        link = form.get("link", "").strip()
        offer = Offer(
            title=form.get("title", ""),
            merchant=form.get("merchant", ""),
            price=price,
            price_text=form.get("price", ""),
            currency=currency,
            shipping=shipping,
            shipping_text=shipping_text,
            link=link,
            merchant_url=link,
            condition=form.get("condition", ""),
            snippet=form.get("snippet", ""),
        )
        settings = effective_settings(load_settings(db))
        result = evaluate(tracker, offer, settings)
        return render(request, "playground.html", form={**form, "tracker": tracker}, result=result, offer=offer)

    # --- settings ---

    def render_settings(request: Request, settings: dict, errors: dict | None = None) -> HTMLResponse:
        origins = {".".join(path): value_source(settings, path) for path in config.ENV_FALLBACKS}
        return render(
            request,
            "settings.html",
            settings=settings,
            errors=errors or {},
            origins=origins,
            env_vars={".".join(path): var for path, var in config.ENV_FALLBACKS.items()},
            email_methods=EMAIL_METHODS,
            resend_default_from=RESEND_DEFAULT_FROM,
            source_labels=SOURCE_LABELS,
            module_types=MODULE_TYPES,
            lock_policies=LOCK_POLICIES,
            smtp_security=SMTP_SECURITY,
            trusted_text=format_trusted_merchants(settings["merchants"]["trusted"]),
            paid_requests=paid_requests_per_month(settings),
            unit_name=unit_name,
        )

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request):
        return render_settings(request, load_settings(db))

    @app.post("/settings", response_class=HTMLResponse)
    async def settings_save(request: Request):
        form = await request.form()
        new, errors = settings_from_form(form, load_settings(db))
        if errors:
            return render_settings(request, new, errors)
        save_settings(db, new)
        scheduler.apply(new)
        return back("/settings", msg="Settings saved")

    @app.post("/settings/test-source")
    async def settings_test_source(request: Request):
        """Run one sample search against one source with the saved settings and report what came back."""
        name = str((await request.form()).get("source", ""))
        if name not in SOURCE_LABELS:
            return back("/settings", err="Unknown source")
        settings = effective_settings(load_settings(db))
        for key, cfg in settings["sources"].items():
            cfg["enabled"] = key == name.split(":")[0]
        searxng = settings["sources"]["searxng"]
        searxng.update(use_ebay_engine=name == "searxng:ebay", use_web_search=name == "searxng:web", max_page_fetches=3)
        with httpx.Client(timeout=settings["search"]["timeout_seconds"]) as http:
            return _test_source(settings, name, http)

    def _test_source(settings: dict, name: str, http: httpx.Client) -> RedirectResponse:
        sources, problems = build_sources(settings, client=http)
        label = SOURCE_LABELS[name]
        if problems or not sources:
            return back("/settings", err=f"{label}: " + ("; ".join(problems) or "not available"))
        query = settings["trackers"]["ram"]["queries"][0] if settings["trackers"]["ram"]["queries"] else "64GB DDR4"
        try:
            offers = sources[0].search(query, "ram")
        except ProviderError as exc:
            return back("/settings", err=f"{label} test search failed: {exc}")
        priced = [o for o in offers if o.price is not None]
        message = f"{label}: {len(offers)} results for {query!r}, {len(priced)} with a price"
        if priced:
            message += f", e.g. {priced[0].title[:60]!r} at ${priced[0].price:,.2f}"
        warnings = getattr(sources[0], "warnings", None)
        if warnings:
            message += ". Engines that failed: " + "; ".join(warnings)
        return back("/settings", msg=message)

    @app.post("/settings/test-email")
    def settings_test_email():
        email_cfg = effective_settings(load_settings(db))["email"]
        problem = config_problem({**email_cfg, "enabled": True})
        if problem:
            return back("/settings", err=f"Cannot send a test email: {problem}")
        to = ", ".join(recipients(email_cfg))
        subject = "[Price Tracker] Test email"
        body = "This is a test email from your price tracker. If you can read this, notifications work.\n"
        try:
            send_email(email_cfg, subject, body)
        except EmailError as exc:
            db.insert_notification(kind="test", recipient=to, subject=subject, status="failed", error=str(exc))
            return back("/settings", err=f"Test email failed: {exc}")
        db.insert_notification(kind="test", recipient=to, subject=subject, status="sent", body_text=body)
        return back("/settings", msg=f"Test email sent to {to}")

    @app.post("/settings/restore-merchants")
    def settings_restore_merchants():
        settings = load_settings(db)
        settings["merchants"]["trusted"] = copy.deepcopy(DEFAULT_SETTINGS["merchants"]["trusted"])
        save_settings(db, settings)
        return back("/settings", msg="Trusted merchant list restored to defaults")

    # --- machine-readable ---

    @app.get("/healthz")
    def healthz():
        return PlainTextResponse("ok")

    @app.get("/api/status")
    def api_status():
        latest = db.latest_completed_run()
        return {
            "version": __version__,
            "scan_running": is_running(),
            "latest_run": latest,
            "next_runs": [t.isoformat() for t in scheduler.next_runs()],
            "counts": db.dashboard_counts(),
        }

    return app


def _authorized(request: Request) -> bool:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        return False
    try:
        username, _, password = base64.b64decode(header[6:]).decode().partition(":")
    except (binascii.Error, UnicodeDecodeError):
        return False
    return secrets.compare_digest(username, config.UI_USERNAME) and secrets.compare_digest(password, config.UI_PASSWORD)


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin or origin == "null":
        return True
    return urlparse(origin).netloc == request.headers.get("host", "")


def _templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=HERE / "templates")
    env = templates.env

    def to_local(value, tz: str) -> datetime | None:
        if not value:
            return None
        moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment.astimezone(ZoneInfo(tz))

    @pass_context
    def dt(context, value, fmt: str = "%b %d, %Y %I:%M %p"):
        moment = to_local(value, context.get("tz", "UTC"))
        return moment.strftime(fmt + " %Z") if moment else "—"

    def ago(value):
        if not value:
            return ""
        moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        seconds = (datetime.now(UTC) - moment).total_seconds()
        future = seconds < 0
        seconds = abs(seconds)
        for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
            if seconds >= size:
                text = f"{int(seconds // size)}{unit}"
                break
        else:
            text = "<1m" if not future else "1m"
        return f"in {text}" if future else f"{text} ago"

    def safe_url(value):
        return value if str(value or "").lower().startswith(("http://", "https://")) else ""

    def money(value):
        return "—" if value is None else f"${value:,.2f}"

    def duration(start, end):
        if not start or not end:
            return ""
        seconds = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
        return f"{seconds:.0f}s" if seconds < 120 else f"{seconds / 60:.1f}m"

    env.filters.update(dt=dt, ago=ago, money=money, safe_url=safe_url)
    env.globals.update(duration=duration, unit_name=unit_name, urlencode=urlencode)
    return templates
