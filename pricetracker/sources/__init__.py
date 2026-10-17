"""Search sources. A scan sends every tracker query to every enabled source.

Free: SearXNG (eBay engine + web search with store-page prices) and the eBay Browse API.
Paid, optional: SerpApi and Serper (Google Shopping). Demo: canned data for trying the app.
"""

from __future__ import annotations

import httpx

from ..models import Offer
from .common import ProviderError, parse_price, parse_shipping
from .demo import DemoSource
from .ebay import EbayApiSource, ebay_item_id, offer_from_ebay_api
from .google_shopping import SerpApiSource, SerperSource, offer_from_serpapi, offer_from_serper
from .searxng import SearxngEbaySource, SearxngWebSource, offer_from_searxng_ebay, offer_from_searxng_web

__all__ = [
    "SOURCE_LABELS",
    "ProviderError",
    "build_sources",
    "ebay_item_id",
    "enabled_source_classes",
    "offer_from_raw",
    "parse_price",
    "parse_shipping",
]

# Order matters: when two sources return the same listing in one scan, the first one's data is kept,
# so the richest sources come first.
_ALL_SOURCES = [EbayApiSource, SearxngEbaySource, SearxngWebSource, SerpApiSource, SerperSource, DemoSource]
SOURCE_LABELS = {cls.name: cls.label for cls in _ALL_SOURCES}

_OFFER_PARSERS = {
    "ebay_api": offer_from_ebay_api,
    "searxng:ebay": offer_from_searxng_ebay,
    "searxng:web": offer_from_searxng_web,
    "serpapi": offer_from_serpapi,
    "serper": offer_from_serper,
    "demo": offer_from_serpapi,
}


def enabled_source_classes(settings: dict) -> list[type]:
    cfg = settings["sources"]
    enabled = {
        EbayApiSource: cfg["ebay_api"]["enabled"],
        SearxngEbaySource: cfg["searxng"]["enabled"] and cfg["searxng"]["use_ebay_engine"],
        SearxngWebSource: cfg["searxng"]["enabled"] and cfg["searxng"]["use_web_search"],
        SerpApiSource: cfg["serpapi"]["enabled"],
        SerperSource: cfg["serper"]["enabled"],
        DemoSource: cfg["demo"]["enabled"],
    }
    return [cls for cls in _ALL_SOURCES if enabled[cls]]


def build_sources(settings: dict, client: httpx.Client | None = None) -> tuple[list, list[str]]:
    """(usable sources, problems with enabled sources that could not be set up)."""
    sources, failures = [], {}
    for cls in enabled_source_classes(settings):
        try:
            sources.append(cls(settings, client=client))
        except ProviderError as exc:
            failures.setdefault(str(exc), []).append(cls.label)  # both SearXNG sources share one config
    return sources, [f"{' / '.join(labels)}: {message}" for message, labels in failures.items()]


def offer_from_raw(source_name: str, raw: dict) -> Offer:
    """Rebuild an Offer from a stored raw result (used to replay past scans with new settings)."""
    return _OFFER_PARSERS.get(source_name, offer_from_serpapi)(raw)
