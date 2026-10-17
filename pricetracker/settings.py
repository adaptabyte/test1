"""User-adjustable settings: defaults, persistence, validation and form parsing."""

from __future__ import annotations

import copy
import json
import os
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import ENV_FALLBACKS
from .merchants import DEFAULT_TRUSTED_MERCHANTS

DEFAULT_SETTINGS: dict = {
    "sources": {
        "searxng": {
            "enabled": True,
            "base_url": "",  # docker-compose sets SEARXNG_URL=http://searxng:8080
            "use_ebay_engine": True,
            "use_web_search": True,
            "web_engines": "duckduckgo,brave,bing,google",
            "restrict_to_sites": [
                "ebay.com",
                "newegg.com",
                "amazon.com",
                "walmart.com",
                "bestbuy.com",
                "bhphotovideo.com",
                "microcenter.com",
                "poshmark.com",
                "mercari.com",
            ],
            "web_pages": 1,
            "fetch_product_pages": True,
            "max_page_fetches": 80,
        },
        "ebay_api": {"enabled": False, "client_id": "", "client_secret": ""},
        "serpapi": {"enabled": False, "api_key": ""},
        "serper": {"enabled": False, "api_key": ""},
        "demo": {"enabled": False},
    },
    "search": {"country": "us", "language": "en", "location": "", "results_per_query": 60, "timeout_seconds": 45},
    "schedule": {"enabled": True, "times": ["08:00", "20:00"], "timezone": "America/New_York"},
    "email": {
        "enabled": True,
        "method": "resend",  # resend | smtp
        "to": "adaptabyte@gmail.com",
        "from_address": "",  # blank: Resend's shared sender, or the SMTP username
        "resend_api_key": "",
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_security": "starttls",  # starttls | ssl | none
        "smtp_username": "",
        "smtp_password": "",
        "renotify_on_price_drop": True,
        "price_drop_percent": 5.0,
    },
    "general": {
        "public_base_url": "",
        "retention_days": 90,
        "allow_untrusted_merchants": False,
        "min_seller_feedback_percent": 97.0,
        "min_seller_feedback_score": 10,
    },
    "merchants": {"trusted": DEFAULT_TRUSTED_MERCHANTS, "blocked": []},
    "trackers": {
        "ram": {
            "enabled": True,
            "label": "64GB DDR4 DIMM (64GB per stick)",
            "queries": [
                "64GB DDR4 3200 RDIMM",
                "64GB DDR4 LRDIMM",
                "64GB DDR4 2933 ECC registered",
                "128GB 2x64GB DDR4 kit",
            ],
            "module_size_gb": 64,
            "max_price_per_stick": 300.0,
            "min_price_per_stick": 40.0,
            "min_speed_mts": 2666,
            "preferred_speed_mts": 3200,
            "allow_unknown_speed": False,
            "allowed_types": ["RDIMM", "LRDIMM", "UDIMM"],
            "allow_unknown_type": True,
            "max_sticks_per_listing": 2,
            "include_shipping": True,
            "reject_untested": True,
            "exclude_keywords": ["motherboard", "combo", "bundle", "barebone"],
            "ebay_category_id": "170083",  # Computer Components > Memory (RAM)
        },
        "cpu": {
            "enabled": True,
            "label": "AMD Threadripper PRO 3975WX",
            "queries": ["AMD Ryzen Threadripper PRO 3975WX", "3975WX not vendor locked"],
            "model": "3975WX",
            "model_aliases": ["100-000000086", "100-100000086"],
            "max_price": 500.0,
            "min_price": 150.0,
            "lock_policy": "reject_locked",  # reject_locked | require_unlocked
            "include_shipping": True,
            "reject_untested": True,
            "exclude_keywords": [],
            "ebay_category_id": "164",  # Computer Components > CPUs/Processors
        },
    },
}

EMAIL_METHODS = {"resend": "Resend (API key)", "smtp": "SMTP server (e.g. Gmail)"}
RESEND_DEFAULT_FROM = "Price Tracker <onboarding@resend.dev>"
MODULE_TYPES = ["RDIMM", "LRDIMM", "UDIMM"]
LOCK_POLICIES = {
    "reject_locked": "Reject listings that say or suggest they are vendor locked (recommended)",
    "require_unlocked": "Strict: only accept listings that state they are unlocked or new retail",
}
SMTP_SECURITY = {"starttls": "STARTTLS (port 587)", "ssl": "SSL/TLS (port 465)", "none": "None"}


def _merge(defaults: dict, stored: dict) -> dict:
    """Stored values over defaults, so settings added in newer versions get their defaults."""
    merged = copy.deepcopy(defaults)
    for key, value in stored.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict) and key != "trackers":
            merged[key] = _merge(merged[key], value)
        elif key == "trackers" and isinstance(value, dict):
            for tracker_key, tracker_value in value.items():
                if tracker_key in merged["trackers"] and isinstance(tracker_value, dict):
                    merged["trackers"][tracker_key] = _merge(merged["trackers"][tracker_key], tracker_value)
        else:
            merged[key] = value
    return merged


def load_settings(db) -> dict:
    blob = db.get_settings_blob()
    return _merge(DEFAULT_SETTINGS, json.loads(blob) if blob else {})


def save_settings(db, settings: dict) -> None:
    db.save_settings_blob(json.dumps(settings, indent=2, sort_keys=True))


def _get(settings: dict, path: tuple[str, ...]):
    value = settings
    for key in path:
        value = value[key]
    return value


def effective_settings(settings: dict) -> dict:
    """Settings with secrets and deployment values filled in from the environment where Settings is blank."""
    result = copy.deepcopy(settings)
    for path, env_var in ENV_FALLBACKS.items():
        parent = _get(result, path[:-1])
        if not parent.get(path[-1]) and os.environ.get(env_var):
            parent[path[-1]] = os.environ[env_var]
    return result


def value_source(settings: dict, path: tuple[str, ...]) -> str:
    """Where a value comes from: 'settings', 'env:<VAR>', or '' when unset."""
    if _get(settings, path):
        return "settings"
    env_var = ENV_FALLBACKS.get(path)
    if env_var and os.environ.get(env_var):
        return f"env:{env_var}"
    return ""


def paid_requests_per_month(settings: dict) -> int:
    """Requests a month sent to paid sources (SerpApi / Serper); the free sources have no quota to plan for."""
    queries = sum(len(t["queries"]) for t in settings["trackers"].values() if t["enabled"])
    runs_per_day = len(settings["schedule"]["times"]) if settings["schedule"]["enabled"] else 0
    paid = sum(1 for name in ("serpapi", "serper") if settings["sources"][name]["enabled"])
    return queries * runs_per_day * 30 * paid


# --- Form parsing -------------------------------------------------------------------------------


def _lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _keywords(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\n,]", value) if item.strip()]


def parse_trusted_merchants(text: str) -> list[dict]:
    """One merchant per line: `Name | alias one, alias two | domain.com, other.com`."""
    merchants = []
    for line in _lines(text):
        parts = [p.strip() for p in line.split("|")]
        name = parts[0]
        if not name:
            continue
        aliases = _keywords(parts[1]) if len(parts) > 1 else []
        domains = [d.lower() for d in _keywords(parts[2])] if len(parts) > 2 else []
        merchants.append({"name": name, "aliases": aliases, "domains": domains})
    return merchants


def format_trusted_merchants(merchants: list[dict]) -> str:
    return "\n".join(
        f"{m['name']} | {', '.join(m.get('aliases', []))} | {', '.join(m.get('domains', []))}" for m in merchants
    )


class _Form:
    """Reads typed values from a submitted form and collects validation errors."""

    def __init__(self, form):
        self.form = form
        self.errors: dict[str, str] = {}

    def text(self, name: str) -> str:
        return str(self.form.get(name, "")).strip()

    def flag(self, name: str) -> bool:
        return name in self.form

    def number(self, name: str, minimum: float, maximum: float, integer: bool = False) -> float:
        raw = self.text(name)
        try:
            value = int(raw) if integer else float(raw)
        except ValueError:
            self.errors[name] = "Enter a number"
            return minimum
        if not minimum <= value <= maximum:
            self.errors[name] = f"Must be between {minimum:g} and {maximum:g}"
        return value

    def choice(self, name: str, options) -> str:
        value = self.text(name)
        if value not in options:
            self.errors[name] = "Pick one of the options"
        return value


def settings_from_form(form, current: dict) -> tuple[dict, dict[str, str]]:
    """Build new settings from the Settings page form. Returns (settings, errors by field name)."""
    f = _Form(form)
    new = copy.deepcopy(current)

    sources = new["sources"]
    sx = sources["searxng"]
    sx["enabled"] = f.flag("sources.searxng.enabled")
    sx["base_url"] = f.text("sources.searxng.base_url").rstrip("/")
    if sx["base_url"] and not re.match(r"https?://", sx["base_url"]):
        f.errors["sources.searxng.base_url"] = "Enter a URL starting with http:// or https://"
    sx["use_ebay_engine"] = f.flag("sources.searxng.use_ebay_engine")
    sx["use_web_search"] = f.flag("sources.searxng.use_web_search")
    sx["web_engines"] = ",".join(_keywords(f.text("sources.searxng.web_engines")))
    if sx["enabled"] and sx["use_web_search"] and not sx["web_engines"]:
        f.errors["sources.searxng.web_engines"] = "List at least one engine, e.g. duckduckgo, brave"
    sx["restrict_to_sites"] = [d.lower() for d in _keywords(f.text("sources.searxng.restrict_to_sites"))]
    sx["web_pages"] = f.number("sources.searxng.web_pages", 1, 5, integer=True)
    sx["fetch_product_pages"] = f.flag("sources.searxng.fetch_product_pages")
    sx["max_page_fetches"] = f.number("sources.searxng.max_page_fetches", 0, 1000, integer=True)
    ebay = sources["ebay_api"]
    ebay["enabled"] = f.flag("sources.ebay_api.enabled")
    _apply_secret(f, ebay, "sources.ebay_api", "client_id", secret=False)
    _apply_secret(f, ebay, "sources.ebay_api", "client_secret")
    for name in ("serpapi", "serper"):
        sources[name]["enabled"] = f.flag(f"sources.{name}.enabled")
        _apply_secret(f, sources[name], f"sources.{name}", "api_key")
    sources["demo"]["enabled"] = f.flag("sources.demo.enabled")
    if not any(sources[name]["enabled"] for name in sources):
        f.errors["sources"] = "Enable at least one search source"

    search = new["search"]
    search["country"] = f.text("search.country").lower() or "us"
    search["language"] = f.text("search.language").lower() or "en"
    search["location"] = f.text("search.location")
    search["results_per_query"] = f.number("search.results_per_query", 10, 200, integer=True)
    search["timeout_seconds"] = f.number("search.timeout_seconds", 5, 300, integer=True)

    schedule = new["schedule"]
    schedule["enabled"] = f.flag("schedule.enabled")
    times = _keywords(f.text("schedule.times"))
    bad_times = [t for t in times if not re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", t)]
    if bad_times or not times:
        f.errors["schedule.times"] = "Use 24-hour HH:MM times separated by commas, e.g. 08:00, 20:00"
    schedule["times"] = sorted({f"{int(t.split(':')[0]):02d}:{t.split(':')[1]}" for t in times if t not in bad_times})
    timezone = f.text("schedule.timezone")
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        f.errors["schedule.timezone"] = "Unknown time zone; use a name like America/New_York"
    schedule["timezone"] = timezone

    email = new["email"]
    email["enabled"] = f.flag("email.enabled")
    email["to"] = f.text("email.to")
    recipients = _keywords(email["to"])
    if not recipients or any(not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", r) for r in recipients):
        f.errors["email.to"] = "Enter one or more email addresses separated by commas"
    email["method"] = f.choice("email.method", EMAIL_METHODS)
    _apply_secret(f, email, "email", "resend_api_key")
    email["from_address"] = f.text("email.from_address")
    email["smtp_host"] = f.text("email.smtp_host")
    email["smtp_port"] = f.number("email.smtp_port", 1, 65535, integer=True)
    email["smtp_security"] = f.choice("email.smtp_security", SMTP_SECURITY)
    _apply_secret(f, email, "email", "smtp_username", secret=False)
    _apply_secret(f, email, "email", "smtp_password")
    email["renotify_on_price_drop"] = f.flag("email.renotify_on_price_drop")
    email["price_drop_percent"] = f.number("email.price_drop_percent", 0, 90)

    general = new["general"]
    general["public_base_url"] = f.text("general.public_base_url").rstrip("/")
    general["retention_days"] = f.number("general.retention_days", 7, 3650, integer=True)
    general["allow_untrusted_merchants"] = f.flag("general.allow_untrusted_merchants")
    general["min_seller_feedback_percent"] = f.number("general.min_seller_feedback_percent", 0, 100)
    general["min_seller_feedback_score"] = f.number("general.min_seller_feedback_score", 0, 1_000_000, integer=True)

    new["merchants"]["trusted"] = parse_trusted_merchants(f.text("merchants.trusted"))
    new["merchants"]["blocked"] = _lines(f.text("merchants.blocked"))

    ram = new["trackers"]["ram"]
    p = "trackers.ram."
    ram["enabled"] = f.flag(p + "enabled")
    ram["queries"] = _lines(f.text(p + "queries"))
    ram["module_size_gb"] = f.number(p + "module_size_gb", 1, 512, integer=True)
    ram["max_price_per_stick"] = f.number(p + "max_price_per_stick", 1, 100000)
    ram["min_price_per_stick"] = f.number(p + "min_price_per_stick", 0, 100000)
    ram["min_speed_mts"] = f.number(p + "min_speed_mts", 1600, 6400, integer=True)
    ram["preferred_speed_mts"] = f.number(p + "preferred_speed_mts", 1600, 6400, integer=True)
    ram["allow_unknown_speed"] = f.flag(p + "allow_unknown_speed")
    ram["allowed_types"] = [t for t in MODULE_TYPES if t in form.getlist(p + "allowed_types")]
    ram["allow_unknown_type"] = f.flag(p + "allow_unknown_type")
    ram["max_sticks_per_listing"] = f.number(p + "max_sticks_per_listing", 1, 32, integer=True)
    ram["include_shipping"] = f.flag(p + "include_shipping")
    ram["reject_untested"] = f.flag(p + "reject_untested")
    ram["exclude_keywords"] = _keywords(f.text(p + "exclude_keywords"))
    if not ram["allowed_types"]:
        f.errors[p + "allowed_types"] = "Allow at least one module type"

    cpu = new["trackers"]["cpu"]
    p = "trackers.cpu."
    cpu["enabled"] = f.flag(p + "enabled")
    cpu["queries"] = _lines(f.text(p + "queries"))
    cpu["model"] = f.text(p + "model").upper()
    if not re.fullmatch(r"\d{3,5}\s?[A-Z]{0,3}", cpu["model"]):
        f.errors[p + "model"] = "Enter a model number like 3975WX"
    cpu["model_aliases"] = _keywords(f.text(p + "model_aliases"))
    cpu["max_price"] = f.number(p + "max_price", 1, 100000)
    cpu["min_price"] = f.number(p + "min_price", 0, 100000)
    cpu["lock_policy"] = f.choice(p + "lock_policy", LOCK_POLICIES)
    cpu["include_shipping"] = f.flag(p + "include_shipping")
    cpu["reject_untested"] = f.flag(p + "reject_untested")
    cpu["exclude_keywords"] = _keywords(f.text(p + "exclude_keywords"))

    for key, tracker in new["trackers"].items():
        label = f.text(f"trackers.{key}.label")
        tracker["label"] = label or tracker["label"]
        tracker["ebay_category_id"] = f.text(f"trackers.{key}.ebay_category_id")
        if tracker["ebay_category_id"] and not tracker["ebay_category_id"].isdigit():
            f.errors[f"trackers.{key}.ebay_category_id"] = "eBay category IDs are numbers (blank = all categories)"
        if tracker["enabled"] and not tracker["queries"]:
            f.errors[f"trackers.{key}.queries"] = "Add at least one search query (or disable the tracker)"

    return new, f.errors


def _apply_secret(f: _Form, section_values: dict, section: str, key: str, secret: bool = True) -> None:
    """Secret inputs are left blank in the form; blank keeps the stored value unless 'clear' is ticked."""
    name = f"{section}.{key}"
    value = f.text(name)
    if f.flag(name + ".clear"):
        section_values[key] = ""
    elif value or not secret:
        section_values[key] = value
