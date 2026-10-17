"""Read price, availability and condition from a store's product page.

Web search results (from SearXNG) only have a title, URL and snippet. Almost every store embeds
schema.org Product data (JSON-LD) or price meta tags for Google's rich results, so we fetch the
listing page and read those. Pages behind bot protection simply yield no price, and the listing is
then rejected with that explanation.
"""

from __future__ import annotations

import html
import json
import re
from urllib.parse import urlparse

import httpx

from ..merchants import domain_matches, domain_of
from .common import parse_price

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
MAX_PAGE_BYTES = 4_000_000

# Where each big store keeps its individual listings; anything else on these domains is a search,
# category, review or help page.
PRODUCT_PATHS = {
    "ebay.com": r"/itm/",
    "newegg.com": r"/p/(?!pl\b)|/product/",  # /p/pl?d=... is Newegg's search page
    "walmart.com": r"/ip/",
    "bestbuy.com": r"/site/.+\.p\b",
    "bhphotovideo.com": r"/c/product/",
    "amazon.com": r"/(?:dp|gp/product)/",
    "poshmark.com": r"/listing/",
    "mercari.com": r"/item/",
    "microcenter.com": r"/product/",
    "backmarket.com": r"/p/",
    "target.com": r"/-/a-\d+|/p/",
    "cdw.com": r"/product/",
}
_NON_LISTING_PATH = re.compile(
    r"/(?:search|sch|s|b|browse|shop|category|categories|c/buy|deals?|reviews?|questions?|blog|news|forums?"
    r"|community|help|support|compare)(?:/|$)",
    re.I,
)

_BLOCK_MARKERS = re.compile(
    r"captcha|robot or human|pardon our interruption|access denied|are you a robot|unusual traffic"
    r"|request blocked|verify you are a human|splashui/challenge|px-captcha|challenge-platform",
    re.I,
)
_SITE_SUFFIX = re.compile(r"\s*[|\-–—:]\s*(?:ebay|newegg(?:\.com)?|walmart(?:\.com)?|amazon(?:\.com)?|best buy"
                          r"|b&h photo.*|micro center|poshmark|mercari)\s*$", re.I)  # fmt: skip


def listing_page_problem(url: str) -> str:
    """'' if `url` looks like an individual listing, otherwise a short explanation."""
    domain = domain_of(url)
    path = urlparse(url).path
    for store, pattern in PRODUCT_PATHS.items():
        if domain_matches(domain, store):
            return "" if re.search(pattern, path) else f"{store} page that is not an individual listing"
    if path in ("", "/"):
        return "a home page, not a listing"
    if _NON_LISTING_PATH.search(path):
        return "a search, category or article page, not a listing"
    return ""


def clean_title(title: str) -> str:
    return _SITE_SUFFIX.sub("", html.unescape(title or "")).strip()


def fetch_product_page(client: httpx.Client, url: str) -> dict:
    """Fetch and parse a listing page. Always returns a dict; failures are described in 'error'."""
    page: dict = {"url": url}
    try:
        response = client.get(url, headers=BROWSER_HEADERS, follow_redirects=True)
    except httpx.HTTPError as exc:
        page["error"] = f"could not load the page ({exc.__class__.__name__})"
        return page
    page["status"] = response.status_code
    page["final_url"] = str(response.url)
    text = response.text[:MAX_PAGE_BYTES]
    page.update(parse_product_page(text))
    if "challenge" in page["final_url"]:  # e.g. eBay's splashui/challenge bot check
        text += " splashui/challenge"
    if response.status_code >= 400 or (page.get("price") is None and _BLOCK_MARKERS.search(text[:200_000])):
        blocked = response.status_code in (401, 403, 429, 503) or _BLOCK_MARKERS.search(text[:200_000])
        reason = "the store's bot protection blocked the request" if blocked else "the page returned an error"
        page["error"] = f"{reason} (HTTP {response.status_code})"
    elif page.get("price") is None:
        page["error"] = "no price found in the page's product data"
    return page


def parse_product_page(text: str) -> dict:
    """Extract product data from page HTML: JSON-LD first, then price meta tags."""
    found: dict = {}
    product = _json_ld_product(text)
    if product:
        found.update(product)
        found["method"] = "json-ld"
    meta = _meta_tags(text)
    if found.get("price") is None:
        amount = meta.get("product:price:amount") or meta.get("og:price:amount") or meta.get("price")
        if amount is None:
            match = re.search(r"itemprop=[\"']price[\"'][^>]*content=[\"']([\d.,]+)", text)
            amount = match.group(1) if match else None
        if amount is not None:
            found["price"], _ = parse_price(amount)
            found["currency"] = (
                meta.get("product:price:currency") or meta.get("og:price:currency") or meta.get("pricecurrency") or ""
            )
            found["method"] = "meta"
    found.setdefault("availability", meta.get("product:availability") or meta.get("og:availability") or "")
    found.setdefault("condition", meta.get("product:condition") or "")
    if not found.get("title"):
        title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
        found["title"] = meta.get("og:title") or (title_match.group(1).strip() if title_match else "")
    found["title"] = clean_title(found.get("title", ""))
    found["description"] = html.unescape(meta.get("og:description") or meta.get("description") or "")[:500]
    return {k: v for k, v in found.items() if v not in (None, "")}


def _json_ld_product(text: str) -> dict:
    for block in re.findall(r"<script[^>]+application/ld\+json[^>]*>(.*?)</script>", text, re.I | re.S):
        try:
            data = json.loads(html.unescape(block.strip()))
        except ValueError:
            continue
        for node in _walk(data):
            types = node.get("@type")
            types = types if isinstance(types, list) else [types]
            if "Product" in types or ("offers" in node and node.get("name")):
                product = _product_fields(node)
                if product.get("price") is not None:
                    return product
    return {}


def _walk(data):
    if isinstance(data, list):
        for item in data:
            yield from _walk(item)
    elif isinstance(data, dict):
        yield data
        for value in data.values():
            if isinstance(value, (dict, list)):
                yield from _walk(value)


def _product_fields(node: dict) -> dict:
    offers = node.get("offers")
    offers = offers if isinstance(offers, list) else [offers] if isinstance(offers, dict) else []
    for offer in offers:
        spec = offer.get("priceSpecification")
        spec = spec[0] if isinstance(spec, list) and spec else spec if isinstance(spec, dict) else {}
        raw_price = offer.get("price", offer.get("lowPrice", spec.get("price")))
        price, _ = parse_price(raw_price)
        if price is None:
            continue
        return {
            "title": str(node.get("name") or ""),
            "price": price,
            "currency": str(offer.get("priceCurrency") or spec.get("priceCurrency") or ""),
            "availability": _schema_value(offer.get("availability")),
            "condition": _schema_value(offer.get("itemCondition") or node.get("itemCondition")),
            "shipping": _shipping_cost(offer.get("shippingDetails")),
        }
    return {}


def _schema_value(value) -> str:
    """'https://schema.org/UsedCondition' -> 'Used', 'InStock' -> 'InStock'."""
    if not value:
        return ""
    value = str(value).rstrip("/").rsplit("/", 1)[-1]
    return value[: -len("Condition")] if value.endswith("Condition") else value


def _shipping_cost(details) -> float | None:
    details = details if isinstance(details, list) else [details] if isinstance(details, dict) else []
    costs = []
    for detail in details:
        rate = detail.get("shippingRate") if isinstance(detail, dict) else None
        if isinstance(rate, dict):
            amount, _ = parse_price(rate.get("value"))
            if amount is not None:
                costs.append(amount)
    return min(costs) if costs else None


def _meta_tags(text: str) -> dict:
    tags: dict = {}
    for tag in re.findall(r"<meta\s[^>]*>", text[:600_000], re.I):
        attrs = {
            k.lower(): html.unescape(v1 or v2)
            for k, v1, v2 in re.findall(r"([\w:-]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')", tag)
        }
        key = (attrs.get("property") or attrs.get("name") or attrs.get("itemprop") or "").lower()
        if key and "content" in attrs and key not in tags:
            tags[key] = attrs["content"].strip()
    return tags
