"""Decide whether a Google Shopping seller is one of the reputable merchants we allow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from .models import Offer

DEFAULT_TRUSTED_MERCHANTS = [
    {"name": "eBay", "aliases": ["ebay"], "domains": ["ebay.com"]},
    {"name": "Walmart", "aliases": ["walmart", "walmart marketplace"], "domains": ["walmart.com"]},
    {"name": "Best Buy", "aliases": ["best buy", "bestbuy"], "domains": ["bestbuy.com"]},
    {"name": "Newegg", "aliases": ["newegg", "newegg business"], "domains": ["newegg.com", "neweggbusiness.com"]},
    {"name": "Poshmark", "aliases": ["poshmark"], "domains": ["poshmark.com"]},
    {
        "name": "B&H Photo",
        "aliases": ["b&h", "b&h photo", "b&h photo video", "b&h photo-video-audio", "b and h photo", "bhphotovideo"],
        "domains": ["bhphotovideo.com"],
    },
    {"name": "Amazon", "aliases": ["amazon", "amazon marketplace"], "domains": ["amazon.com"]},
    {"name": "Micro Center", "aliases": ["micro center", "microcenter"], "domains": ["microcenter.com"]},
    {"name": "Adorama", "aliases": ["adorama"], "domains": ["adorama.com"]},
    {"name": "CDW", "aliases": ["cdw"], "domains": ["cdw.com"]},
    {"name": "Zones", "aliases": ["zones"], "domains": ["zones.com"]},
    {"name": "Insight", "aliases": ["insight"], "domains": ["insight.com"]},
    {"name": "Connection", "aliases": ["connection", "pc connection"], "domains": ["connection.com"]},
    {"name": "Provantage", "aliases": ["provantage"], "domains": ["provantage.com"]},
    {"name": "Crucial", "aliases": ["crucial"], "domains": ["crucial.com"]},
    {"name": "Kingston", "aliases": ["kingston", "kingston technology"], "domains": ["kingston.com"]},
    {"name": "Supermicro", "aliases": ["supermicro", "supermicro estore"], "domains": ["supermicro.com"]},
    {"name": "Dell", "aliases": ["dell", "dell technologies"], "domains": ["dell.com"]},
    {"name": "HP", "aliases": ["hp", "hp store"], "domains": ["hp.com"]},
    {"name": "Lenovo", "aliases": ["lenovo"], "domains": ["lenovo.com"]},
    {"name": "Mercari", "aliases": ["mercari"], "domains": ["mercari.com"]},
    {"name": "Back Market", "aliases": ["back market", "backmarket"], "domains": ["backmarket.com"]},
    {"name": "Woot", "aliases": ["woot"], "domains": ["woot.com"]},
    {"name": "Target", "aliases": ["target"], "domains": ["target.com"]},
    {"name": "Staples", "aliases": ["staples"], "domains": ["staples.com"]},
    {"name": "Office Depot", "aliases": ["office depot", "office depot officemax"], "domains": ["officedepot.com"]},
    {"name": "Costco", "aliases": ["costco"], "domains": ["costco.com"]},
    {"name": "Sam's Club", "aliases": ["sam's club", "sams club"], "domains": ["samsclub.com"]},
    {"name": "Antonline", "aliases": ["antonline"], "domains": ["antonline.com"]},
    {"name": "Central Computers", "aliases": ["central computers"], "domains": ["centralcomputers.com"]},
    {"name": "Server Monkey", "aliases": ["server monkey", "servermonkey"], "domains": ["servermonkey.com"]},
    {
        "name": "ServerPartDeals",
        "aliases": ["serverpartdeals", "server part deals"],
        "domains": ["serverpartdeals.com"],
    },
]

_GOOGLE_HOSTS = re.compile(r"(^|\.)(google\.[a-z.]+|googleadservices\.com|doubleclick\.net|g\.co)$")


def normalize_merchant(name: str) -> str:
    """'eBay - some_seller' -> 'ebay', 'Newegg.com' -> 'newegg', 'Amazon.com & more' -> 'amazon'."""
    name = name.strip().lower()
    name = re.sub(r"\s*(?:&|\+|and)\s*more\s*$", "", name)
    name = re.split(r"\s+[-–—|]\s+|\s+by\s+", name, maxsplit=1)[0]
    name = re.sub(r"^www\.", "", name)
    name = re.sub(r"\.(?:com|net|org|us|co)$", "", name)
    return re.sub(r"\s+", " ", name).strip()


def normalize_domain(domain: str) -> str:
    domain = domain.strip().lower()
    domain = re.sub(r"^[a-z]+://", "", domain).split("/")[0].split(":")[0]
    return re.sub(r"^(?:www|m|smile)\.", "", domain)


def domain_of(url: str) -> str:
    """Merchant domain for a URL, following Google redirect links (google.com/url?url=...)."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    host = normalize_domain(parsed.netloc)
    if not _GOOGLE_HOSTS.search(host):
        return host
    params = parse_qs(parsed.query)
    for key in ("url", "q", "adurl"):
        target = params.get(key, [""])[0]
        if target.startswith("http"):
            inner = normalize_domain(urlparse(target).netloc)
            if inner and not _GOOGLE_HOSTS.search(inner):
                return inner
    return ""


def _domain_matches(domain: str, allowed: str) -> bool:
    allowed = normalize_domain(allowed)
    return bool(allowed) and (domain == allowed or domain.endswith("." + allowed))


def _is_blocked(entry: str, merchant: str, name_norm: str, domain: str) -> bool:
    """A blocked entry is one marketplace seller ("ebay - badseller"), a whole store ("walmart") or a domain."""
    if not entry:
        return False
    if re.search(r"\s[-–—|]\s", entry):
        return re.sub(r"\s+", " ", merchant.strip().lower()) == re.sub(r"\s+", " ", entry)
    if domain and "." in entry and " " not in entry and _domain_matches(domain, entry):
        return True
    return bool(name_norm) and normalize_merchant(entry) == name_norm


@dataclass
class MerchantVerdict:
    trusted: bool
    canonical: str  # trusted merchant name, or the seller name as shown when untrusted
    domain: str
    code: str = ""  # rejection code when not trusted
    message: str = ""
    on_list: bool = False  # matched an entry on the trusted list


def judge_merchant(offer: Offer, trusted: list[dict], blocked: list[str], allow_untrusted: bool) -> MerchantVerdict:
    name_norm = normalize_merchant(offer.merchant)
    domain = domain_of(offer.merchant_url) or domain_of(offer.link)
    shown = offer.merchant or domain or "unknown seller"

    for entry in blocked:
        if _is_blocked(entry.strip().lower(), offer.merchant, name_norm, domain):
            return MerchantVerdict(False, shown, domain, "merchant_blocked", f"'{shown}' is on the blocked list")

    by_name = by_domain = None
    for entry in trusted:
        aliases = {normalize_merchant(entry["name"]), *(normalize_merchant(a) for a in entry.get("aliases", []))}
        if name_norm and name_norm in aliases:
            by_name = entry
        if domain and any(_domain_matches(domain, d) for d in entry.get("domains", [])):
            by_domain = entry

    if domain:
        if by_domain:
            return MerchantVerdict(True, by_domain["name"], domain, on_list=True)
        if by_name:
            return MerchantVerdict(
                False,
                shown,
                domain,
                "merchant_mismatch",
                f"Seller is shown as '{offer.merchant}' but the link goes to {domain} (possible impostor)",
            )
    elif by_name:
        return MerchantVerdict(True, by_name["name"], domain, on_list=True)

    where = f" ({domain})" if domain and domain != shown.lower() else ""
    if allow_untrusted:
        return MerchantVerdict(True, shown, domain)
    return MerchantVerdict(
        False, shown, domain, "merchant_untrusted", f"'{shown}'{where} is not on the trusted merchant list"
    )
