"""Google Shopping search providers.

Google does not offer a public Shopping search API and blocks automated scraping, so searches go
through a SERP API service that runs the Google Shopping query and returns structured JSON.
"""

from __future__ import annotations

import re
from urllib.parse import urlencode

import httpx

from .demo_data import demo_results
from .merchants import domain_of
from .models import Offer


class ProviderError(Exception):
    """A search failed; the message is shown in the run log."""


def parse_price(value) -> tuple[float | None, str]:
    """Return (amount, currency) from a number or a price string like '$1,299.99' or 'CA$120'."""
    if isinstance(value, (int, float)):
        return float(value), "USD"
    if not value:
        return None, "USD"
    text = str(value).strip()
    currency = "USD"
    if re.search(r"\b(?:CA|C|A|AU|NZ|HK|S)\$|CAD|AUD|£|€|¥|₹|EUR|GBP", text):
        currency = "OTHER"
    match = re.search(r"\d[\d,]*(?:\.\d{1,2})?", text)
    if not match:
        return None, currency
    return float(match.group(0).replace(",", "")), currency


def parse_shipping(*texts: str) -> tuple[float | None, str]:
    """Shipping cost from Google's delivery text ('Free delivery', '$7.99 delivery', '+$12 shipping')."""
    for text in texts:
        if not text:
            continue
        lowered = str(text).lower()
        if not re.search(r"ship|deliver|postage", lowered):
            continue
        if "free" in lowered:
            return 0.0, str(text)
        match = re.search(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)", lowered)
        if match:
            return float(match.group(1).replace(",", "")), str(text)
    return None, ""


def offer_from_serpapi(item: dict) -> Offer:
    """SerpApi `google_shopping` result item -> Offer. Also used for demo data."""
    price, currency = parse_price(item.get("extracted_price"))
    price_text = str(item.get("price") or "")
    if price is None or (price_text and parse_price(price_text)[1] != "USD"):
        price, currency = parse_price(price_text)
    extensions = [str(e) for e in item.get("extensions") or []]
    shipping, shipping_text = parse_shipping(item.get("delivery", ""), item.get("shipping", ""), *extensions)
    direct = item.get("link") or ""
    google_page = item.get("product_link") or ""
    merchant_url = direct if domain_of(direct) else ""
    return Offer(
        title=str(item.get("title") or ""),
        merchant=str(item.get("source") or item.get("seller") or ""),
        price=price,
        price_text=price_text,
        currency=currency,
        shipping=shipping,
        shipping_text=shipping_text,
        link=direct or google_page,
        merchant_url=merchant_url,
        product_id=str(item.get("product_id") or ""),
        condition=str(item.get("second_hand_condition") or item.get("condition") or ""),
        snippet=str(item.get("snippet") or ""),
        extensions=[e for e in extensions if not re.search(r"ship|deliver|return", e.lower())],
        thumbnail=str(item.get("thumbnail") or ""),
        multiple_sources=bool(item.get("multiple_sources")),
        position=int(item.get("position") or 0),
        raw=item,
    )


def offer_from_serper(item: dict) -> Offer:
    """Serper.dev `/shopping` result item -> Offer."""
    price, currency = parse_price(item.get("price"))
    shipping, shipping_text = parse_shipping(item.get("delivery", ""), item.get("shipping", ""))
    link = str(item.get("link") or "")
    return Offer(
        title=str(item.get("title") or ""),
        merchant=str(item.get("source") or ""),
        price=price,
        price_text=str(item.get("price") or ""),
        currency=currency,
        shipping=shipping,
        shipping_text=shipping_text,
        link=link,
        merchant_url=link if domain_of(link) else "",
        product_id=str(item.get("productId") or ""),
        condition=str(item.get("condition") or ""),
        snippet=str(item.get("snippet") or ""),
        thumbnail=str(item.get("imageUrl") or ""),
        position=int(item.get("position") or 0),
        raw=item,
    )


_NO_RESULTS = "hasn't returned any results"


class SerpApiProvider:
    name = "serpapi"
    endpoint = "https://serpapi.com/search.json"

    def __init__(self, cfg: dict, client: httpx.Client | None = None):
        if not cfg["serpapi_api_key"]:
            raise ProviderError("SerpApi API key is not set (Settings > Search provider, or SERPAPI_API_KEY)")
        self.cfg = cfg
        self.client = client or httpx.Client(timeout=cfg["timeout_seconds"])

    def _params(self, query: str) -> dict:
        params = {
            "engine": "google_shopping",
            "q": query,
            "gl": self.cfg["country"],
            "hl": self.cfg["language"],
            "num": self.cfg["results_per_query"],
        }
        if self.cfg["location"]:
            params["location"] = self.cfg["location"]
        return params

    def describe(self, query: str) -> str:
        return f"GET {self.endpoint}?{urlencode(self._params(query))}"

    def search(self, query: str) -> list[Offer]:
        params = {**self._params(query), "api_key": self.cfg["serpapi_api_key"]}
        try:
            data = _request_json(self.client, "GET", self.endpoint, params=params)
        except ProviderError as exc:
            if _NO_RESULTS in str(exc):
                return []
            raise
        error = data.get("error")
        if error:
            if _NO_RESULTS in error:
                return []
            raise ProviderError(f"SerpApi: {error}")
        items = list(data.get("shopping_results") or [])
        for group in data.get("categorized_shopping_results") or []:
            items.extend(group.get("shopping_results") or [])
        items.extend(data.get("inline_shopping_results") or [])
        return [offer_from_serpapi(item) for item in items if item.get("title")]


class SerperProvider:
    name = "serper"
    endpoint = "https://google.serper.dev/shopping"

    def __init__(self, cfg: dict, client: httpx.Client | None = None):
        if not cfg["serper_api_key"]:
            raise ProviderError("Serper API key is not set (Settings > Search provider, or SERPER_API_KEY)")
        self.cfg = cfg
        self.client = client or httpx.Client(timeout=cfg["timeout_seconds"])

    def _body(self, query: str) -> dict:
        body = {"q": query, "gl": self.cfg["country"], "hl": self.cfg["language"], "num": self.cfg["results_per_query"]}
        if self.cfg["location"]:
            body["location"] = self.cfg["location"]
        return body

    def describe(self, query: str) -> str:
        return f"POST {self.endpoint} {self._body(query)}"

    def search(self, query: str) -> list[Offer]:
        headers = {"X-API-KEY": self.cfg["serper_api_key"], "Content-Type": "application/json"}
        data = _request_json(self.client, "POST", self.endpoint, json=self._body(query), headers=headers)
        if data.get("message") and not data.get("shopping"):
            raise ProviderError(f"Serper: {data['message']}")
        return [offer_from_serper(item) for item in data.get("shopping") or [] if item.get("title")]


class DemoProvider:
    """Canned results that exercise every rule; lets you try the app without an API key."""

    name = "demo"

    def __init__(self, cfg: dict, client=None):
        self.cfg = cfg

    def describe(self, query: str) -> str:
        return f"demo data for {query!r} (no network request)"

    def search(self, query: str) -> list[Offer]:
        return [offer_from_serpapi(item) for item in demo_results(query)]


PROVIDER_CLASSES = {"serpapi": SerpApiProvider, "serper": SerperProvider, "demo": DemoProvider}


def get_provider(provider_cfg: dict, client: httpx.Client | None = None):
    try:
        cls = PROVIDER_CLASSES[provider_cfg["name"]]
    except KeyError:
        raise ProviderError(f"Unknown search provider {provider_cfg['name']!r}") from None
    return cls(provider_cfg, client=client)


def _request_json(client: httpx.Client, method: str, url: str, **kwargs) -> dict:
    try:
        response = client.request(method, url, **kwargs)
    except httpx.HTTPError as exc:
        raise ProviderError(f"Request failed: {exc.__class__.__name__}: {exc}") from exc
    try:
        data = response.json()
    except ValueError:
        raise ProviderError(f"HTTP {response.status_code}: response was not JSON: {response.text[:200]!r}") from None
    if response.status_code >= 400 and not (isinstance(data, dict) and (data.get("error") or data.get("message"))):
        raise ProviderError(f"HTTP {response.status_code}: {str(data)[:200]}")
    if response.status_code >= 400:
        message = data.get("error") or data.get("message")
        raise ProviderError(f"HTTP {response.status_code}: {message}")
    if not isinstance(data, dict):
        raise ProviderError("Unexpected response shape (expected a JSON object)")
    return data
