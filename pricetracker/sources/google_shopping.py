"""Paid Google Shopping sources: SerpApi and Serper.

Google has no public Shopping API and blocks scrapers, so these services run the Google Shopping
search and return structured JSON. Optional; the free sources are SearXNG and the eBay API.
"""

from __future__ import annotations

import re
from urllib.parse import urlencode

import httpx

from ..merchants import domain_of
from ..models import Offer
from .common import ProviderError, parse_price, parse_shipping, request_json

_NO_RESULTS = "hasn't returned any results"


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
    return Offer(
        title=str(item.get("title") or ""),
        merchant=str(item.get("source") or item.get("seller") or ""),
        price=price,
        price_text=price_text,
        currency=currency,
        shipping=shipping,
        shipping_text=shipping_text,
        link=direct or google_page,
        merchant_url=direct if domain_of(direct) else "",
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


class SerpApiSource:
    name = "serpapi"
    label = "SerpApi (Google Shopping)"
    endpoint = "https://serpapi.com/search.json"

    def __init__(self, settings: dict, client: httpx.Client | None = None):
        self.key = settings["sources"]["serpapi"]["api_key"]
        if not self.key:
            raise ProviderError("SerpApi API key is not set (Settings > Search sources, or SERPAPI_API_KEY)")
        self.search_cfg = settings["search"]
        self.client = client or httpx.Client(timeout=self.search_cfg["timeout_seconds"])

    def _params(self, query: str) -> dict:
        cfg = self.search_cfg
        params = {"engine": "google_shopping", "q": query, "gl": cfg["country"], "hl": cfg["language"]}
        params["num"] = cfg["results_per_query"]
        if cfg["location"]:
            params["location"] = cfg["location"]
        return params

    def describe(self, query: str, tracker: str) -> str:
        return f"GET {self.endpoint}?{urlencode(self._params(query))}"

    def search(self, query: str, tracker: str) -> list[Offer]:
        params = {**self._params(query), "api_key": self.key}
        try:
            data = request_json(self.client, "GET", self.endpoint, params=params)
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


class SerperSource:
    name = "serper"
    label = "Serper.dev (Google Shopping)"
    endpoint = "https://google.serper.dev/shopping"

    def __init__(self, settings: dict, client: httpx.Client | None = None):
        self.key = settings["sources"]["serper"]["api_key"]
        if not self.key:
            raise ProviderError("Serper API key is not set (Settings > Search sources, or SERPER_API_KEY)")
        self.search_cfg = settings["search"]
        self.client = client or httpx.Client(timeout=self.search_cfg["timeout_seconds"])

    def _body(self, query: str) -> dict:
        cfg = self.search_cfg
        body = {"q": query, "gl": cfg["country"], "hl": cfg["language"], "num": cfg["results_per_query"]}
        if cfg["location"]:
            body["location"] = cfg["location"]
        return body

    def describe(self, query: str, tracker: str) -> str:
        return f"POST {self.endpoint} {self._body(query)}"

    def search(self, query: str, tracker: str) -> list[Offer]:
        headers = {"X-API-KEY": self.key, "Content-Type": "application/json"}
        data = request_json(self.client, "POST", self.endpoint, json=self._body(query), headers=headers)
        if data.get("message") and not data.get("shopping"):
            raise ProviderError(f"Serper: {data['message']}")
        return [offer_from_serper(item) for item in data.get("shopping") or [] if item.get("title")]
