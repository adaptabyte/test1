"""Run a scan: search Google Shopping for every tracker, evaluate each offer, store, and email matches."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import traceback
from collections import Counter
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

from .db import Database, utcnow
from .merchants import domain_of
from .models import Evaluation, Offer
from .notifier import EmailError, build_email, config_problem, recipients, send_email
from .providers import ProviderError, get_provider, offer_from_serpapi, offer_from_serper
from .rules import evaluate
from .rules.common import norm
from .settings import effective_settings, load_settings

_scan_lock = threading.Lock()


class ScanAlreadyRunning(Exception):
    pass


def is_running() -> bool:
    return _scan_lock.locked()


def listing_keys(tracker: str, offer: Offer) -> tuple[str, str]:
    """(key, dedupe_key). `key` identifies one offer across runs; `dedupe_key` groups same seller + title."""
    seller = offer.merchant.strip().lower() or domain_of(offer.link)
    if offer.product_id:
        ident = "pid:" + offer.product_id
    elif offer.merchant_url:
        parsed = urlparse(offer.merchant_url)
        ident = "url:" + parsed.netloc.lower() + parsed.path
    else:
        ident = "title:" + norm(offer.title)

    def digest(value: str) -> str:
        return hashlib.sha1(value.encode()).hexdigest()[:24]

    return digest(f"{tracker}|{seller}|{ident}"), digest(f"{tracker}|{seller}|{norm(offer.title)}")


def offer_from_raw(provider_name: str, raw: dict) -> Offer:
    return offer_from_serper(raw) if provider_name == "serper" else offer_from_serpapi(raw)


def observation_record(tracker: str, offer: Offer, ev: Evaluation) -> dict:
    return {
        "tracker": tracker,
        "title": offer.title,
        "merchant": offer.merchant or ev.merchant_domain or "unknown seller",
        "merchant_canonical": ev.merchant,
        "merchant_domain": ev.merchant_domain,
        "link": offer.link,
        "price": offer.price,
        "shipping": offer.shipping,
        "total_price": ev.total_price,
        "units": ev.units,
        "unit_price": ev.unit_price,
        "decision": ev.decision,
        "reasons": [{"code": r.code, "message": r.message} for r in ev.reasons],
        "notes": ev.notes,
        "attributes": ev.attributes,
        "tier": ev.tier,
        "raw": offer.raw,
    }


class _RunLog:
    def __init__(self, db: Database, run_id: int):
        self.db, self.run_id, self.lines = db, run_id, []

    def __call__(self, message: str, flush: bool = False) -> None:
        self.lines.append(f"{datetime.now(UTC):%H:%M:%S} {message}")
        if flush:
            self.flush()

    def flush(self) -> None:
        self.db.update_run(self.run_id, log="\n".join(self.lines))

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def start_scan(db: Database, trigger: str, send_email: bool = True, provider=None, background: bool = False) -> int:
    """Start a scan and return its run id. With background=True the work happens in a thread."""
    if not _scan_lock.acquire(blocking=False):
        raise ScanAlreadyRunning("A scan is already running")
    try:
        # Another process (e.g. `python -m pricetracker scan` from cron) may be scanning the same database.
        other = db.running_run()
        recent = (datetime.now(UTC) - timedelta(minutes=30)).isoformat(timespec="seconds")
        if other and other["started_at"] > recent:
            raise ScanAlreadyRunning(f"Scan #{other['id']} is already running")
        settings = effective_settings(load_settings(db))
        run_id = db.create_run(trigger, settings["provider"]["name"], send_email)
    except BaseException:
        _scan_lock.release()
        raise

    def work() -> None:
        try:
            _execute(db, run_id, settings, send_email, provider)
        finally:
            _scan_lock.release()

    if background:
        threading.Thread(target=work, name=f"scan-{run_id}", daemon=True).start()
    else:
        work()
    return run_id


def _execute(db: Database, run_id: int, settings: dict, send_email_enabled: bool, provider) -> None:
    log = _RunLog(db, run_id)
    counts: Counter = Counter()
    try:
        provider = provider or get_provider(settings["provider"])
        log(f"Provider: {provider.name}; trigger email: {'yes' if send_email_enabled else 'no'}", flush=True)
        matches = _search_all(db, run_id, settings, provider, counts, log)
        email_status = _notify(db, run_id, settings, matches, send_email_enabled, provider.name == "demo", log)
        cutoff = datetime.now(UTC) - timedelta(days=int(settings["general"]["retention_days"]))
        pruned = db.prune(cutoff.isoformat(timespec="seconds"))
        if pruned:
            log(f"Pruned {pruned} observations older than {settings['general']['retention_days']} days")
        if counts["queries"] and counts["query_errors"] == counts["queries"]:
            status = "failed"
        elif counts["query_errors"]:
            status = "partial"
        else:
            status = "ok"
        error = f"{counts['query_errors']} of {counts['queries']} queries failed" if counts["query_errors"] else None
        log(f"Finished: {status}")
        db.update_run(
            run_id,
            status=status,
            finished_at=utcnow(),
            error=error,
            email_status=email_status,
            log=log.text,
            **_count_fields(counts),
        )
    except ProviderError as exc:
        log(f"Provider error: {exc}")
        db.update_run(run_id, status="failed", finished_at=utcnow(), error=str(exc), log=log.text)
    except Exception as exc:  # keep the scheduler alive and make the failure visible in the UI
        log("Unexpected error:\n" + traceback.format_exc())
        db.update_run(
            run_id,
            status="failed",
            finished_at=utcnow(),
            error=f"{exc.__class__.__name__}: {exc}",
            log=log.text,
            **_count_fields(counts),
        )


def _count_fields(counts: Counter) -> dict:
    return {
        "query_count": counts["queries"],
        "query_errors": counts["query_errors"],
        "result_count": counts["results"],
        "accepted_count": counts["accepted"],
        "rejected_count": counts["rejected"],
        "new_match_count": counts["matches"],
    }


def _search_all(db: Database, run_id: int, settings: dict, provider, counts: Counter, log) -> list[dict]:
    matches: list[dict] = []
    seen: set[str] = set()
    for tracker_key, tracker in settings["trackers"].items():
        if not tracker["enabled"]:
            log(f"[{tracker_key}] disabled, skipped")
            continue
        for query in tracker["queries"]:
            counts["queries"] += 1
            query_id = db.create_query(run_id, tracker_key, query, provider.name, provider.describe(query))
            started = time.monotonic()
            try:
                offers = provider.search(query)
            except Exception as exc:
                message = str(exc) if isinstance(exc, ProviderError) else f"{exc.__class__.__name__}: {exc}"
                db.finish_query(query_id, int((time.monotonic() - started) * 1000), None, message)
                counts["query_errors"] += 1
                log(f"[{tracker_key}] {query!r}: FAILED: {message}", flush=True)
                continue
            db.finish_query(query_id, int((time.monotonic() - started) * 1000), len(offers), None)
            counts["results"] += len(offers)

            per_query: Counter = Counter()
            for offer in offers:
                key, dedupe_key = listing_keys(tracker_key, offer)
                if key in seen:
                    per_query["duplicate"] += 1
                    continue
                seen.add(key)
                match = _process_offer(db, run_id, query_id, tracker_key, offer, key, dedupe_key, settings)
                per_query[match["decision"]] += 1
                counts[match["decision"]] += 1
                if match["notify_reason"] in ("new", "price_drop"):
                    counts["matches"] += 1
                    matches.append(match)
            log(
                f"[{tracker_key}] {query!r}: {len(offers)} results, {per_query['accepted']} accepted, "
                f"{per_query['rejected']} rejected, {per_query['duplicate']} already seen in this run",
                flush=True,
            )
    return matches


def _process_offer(db, run_id, query_id, tracker_key, offer, key, dedupe_key, settings) -> dict:
    ev = evaluate(tracker_key, offer, settings)
    obs = observation_record(tracker_key, offer, ev)
    previous = db.get_listing_by_key(key)
    listing_id = db.upsert_listing(key, dedupe_key, tracker_key, run_id, obs)
    previous_price = None
    if ev.accepted:
        obs["notify_reason"], previous_price = _notify_reason(db, previous, listing_id, dedupe_key, ev, settings)
    obs["observation_id"] = db.insert_observation(run_id, query_id, listing_id, obs)
    obs["listing_id"] = listing_id
    obs["previous_unit_price"] = previous_price
    obs.setdefault("notify_reason", None)
    return obs


def _notify_reason(db, previous, listing_id, dedupe_key, ev, settings) -> tuple[str, float | None]:
    email_cfg = settings["email"]
    if previous is None or previous["notified_unit_price"] is None:
        duplicate = db.notified_duplicate(dedupe_key, listing_id, ev.unit_price)
        if duplicate:
            return f"duplicate of listing #{duplicate['id']}", duplicate["notified_unit_price"]
        return "new", None
    notified = previous["notified_unit_price"]
    threshold = notified * (1 - float(email_cfg["price_drop_percent"]) / 100)
    if email_cfg["renotify_on_price_drop"] and ev.unit_price <= threshold and ev.unit_price < notified:
        return "price_drop", notified
    return "already_notified", notified


def _notify(db, run_id, settings, matches, send_email_enabled, demo, log) -> str:
    if not matches:
        log("No new matches to email")
        return "no new matches"
    if not send_email_enabled:
        log(f"{len(matches)} new matches; email skipped because this scan was started without email")
        return f"skipped ({len(matches)} matches will be emailed by the next scan with email on)"
    email_cfg = settings["email"]
    to = ", ".join(recipients(email_cfg))
    problem = config_problem(email_cfg)
    if problem:
        log(f"{len(matches)} new matches; email not sent: {problem}")
        db.insert_notification(
            run_id=run_id, kind="matches", recipient=to, subject="", status="skipped", error=problem,
            listing_count=len(matches),
        )  # fmt: skip
        return f"not sent: {problem}"
    subject, text, html_body = build_email(matches, settings, run_id, demo=demo)
    try:
        send_email(email_cfg, subject, text, html_body)
    except EmailError as exc:
        log(f"Email to {to} FAILED: {exc}")
        db.insert_notification(
            run_id=run_id, kind="matches", recipient=to, subject=subject, status="failed", error=str(exc),
            listing_count=len(matches), body_text=text,
        )  # fmt: skip
        return f"failed: {exc}"
    db.insert_notification(
        run_id=run_id, kind="matches", recipient=to, subject=subject, status="sent", listing_count=len(matches),
        body_text=text,
    )  # fmt: skip
    db.mark_notified([(m["listing_id"], m["observation_id"], m["unit_price"]) for m in matches])
    log(f"Emailed {len(matches)} matches to {to}")
    return f"sent {len(matches)} matches to {to}"


def replay_run(db: Database, run_id: int, settings: dict) -> list[dict]:
    """Re-evaluate a past run's stored results with the given settings, without saving or emailing."""
    run = db.get_run(run_id)
    rows = []
    for obs in db.observations_for_run(run_id):
        offer = offer_from_raw(run["provider"], json.loads(obs["raw"]) if obs["raw"] else {})
        ev = evaluate(obs["tracker"], offer, settings)
        rows.append({"old": obs, "new": observation_record(obs["tracker"], offer, ev)})
    return rows
