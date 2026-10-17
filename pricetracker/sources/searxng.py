"""Free search through a self-hosted SearXNG instance (bundled in docker-compose.yml).

SearXNG has no Google Shopping engine, so it is used two ways:

* ``searxng:ebay``: SearXNG's eBay engine, which returns eBay search results with prices.
* ``searxng:web``: ordinary web search (DuckDuckGo, Brave, Bing, Google, …) restricted to the trusted
  stores. Web results have no price, so each listing page on a trusted store is fetched and its
  schema.org product data read (see pages.py).
"""

from __future__ import annotations

import re

import httpx

from ..merchants import domain_of, judge_merchant
from ..models import Offer
from .common import ProviderError, parse_price, parse_shipping, request_json
from .pages import clean_title, fetch_product_page, listing_page_problem


class _Searxng:
    def __init__(self, settings: dict, client: httpx.Client | None):
        cfg = settings["sources"]["searxng"]
        self.base_url = (cfg["base_url"] or "").rstrip("/")
        if not self.base_url:
            raise ProviderError("SearXNG URL is not set (Settings > Search sources, or SEARXNG_URL)")
        search = settings["search"]
        self.language = f"{search['language']}-{search['country'].upper()}"
        self.client = client or httpx.Client(timeout=search["timeout_seconds"])

    def params(self, query: str, engines: str, pageno: int = 1) -> dict:
        return {"q": query, "format": "json", "engines": engines, "language": self.language, "pageno": pageno,
                "safesearch": 0}  # fmt: skip

    def describe(self, query: str, engines: str) -> str:
        request = httpx.Request("GET", f"{self.base_url}/search", params=self.params(query, engines))
        return f"GET {request.url}"

    def search(self, query: str, engines: str, pages: int = 1) -> tuple[list[dict], list[str]]:
        """(results, warnings). Raises if nothing came back and engines reported errors."""
        results: list[dict] = []
        failures: dict[str, str] = {}
        for pageno in range(1, pages + 1):
            try:
                data = request_json(
                    self.client, "GET", f"{self.base_url}/search", params=self.params(query, engines, pageno)
                )
            except ProviderError as exc:
                if "HTTP 403" in str(exc):
                    raise ProviderError(
                        "SearXNG refused the JSON request (HTTP 403). Enable `json` under `search: formats:` "
                        "in SearXNG's settings.yml"
                    ) from exc
                raise ProviderError(f"SearXNG at {self.base_url}: {exc}") from exc
            results.extend(r for r in data.get("results") or [] if isinstance(r, dict) and r.get("url"))
            for item in data.get("unresponsive_engines") or []:
                if isinstance(item, (list, tuple)) and item:
                    failures[str(item[0])] = str(item[1]) if len(item) > 1 else "no response"
        warnings = [f"{engine}: {error}" for engine, error in failures.items()]
        if not results and failures:
            raise ProviderError("SearXNG engines failed: " + "; ".join(warnings))
        return results, warnings


def offer_from_searxng_ebay(result: dict) -> Offer:
    """A result from SearXNG's eBay engine (title, url, price, shipping, condition, source_country)."""
    price_text = str(result.get("price") or "")
    price, currency = parse_price(price_text)
    shipping_text = str(result.get("shipping") or "")
    shipping, _ = parse_shipping(shipping_text)
    if shipping is None and re.search(r"free", shipping_text, re.I):
        shipping = 0.0
    url = str(result.get("url") or "")
    return Offer(
        title=clean_title(str(result.get("title") or "")),
        merchant="eBay",
        price=price,
        price_text=price_text,
        currency=currency,
        shipping=shipping,
        shipping_text=shipping_text,
        link=url,
        merchant_url=url,
        condition=str(result.get("content") or "").strip(),
        thumbnail=str(result.get("thumbnail") or ""),
        ships_from=re.sub(r"^from\s+", "", str(result.get("source_country") or ""), flags=re.I).strip(),
        raw=result,
    )


def offer_from_searxng_web(raw: dict) -> Offer:
    """A web search result plus whatever was read from its listing page (raw['_page'])."""
    page = raw.get("_page") or {}
    url = str(raw.get("url") or "")
    price = page.get("price")
    currency = "USD" if str(page.get("currency") or "USD").upper() in ("USD", "US$", "$") else "OTHER"
    if price is not None:
        problem = ""
    elif raw.get("_not_fetched"):
        problem = raw["_not_fetched"]
    else:
        problem = page.get("error") or "no price found"
    snippet = " ".join(part for part in [str(raw.get("content") or ""), str(page.get("description") or "")] if part)
    return Offer(
        title=str(page.get("title") or clean_title(str(raw.get("title") or ""))),
        merchant=domain_of(url) or "unknown site",
        price=price,
        price_text=f"{price} {page.get('currency', '')}".strip() if price is not None else "",
        currency=currency,
        shipping=page.get("shipping"),
        link=url,
        merchant_url=url,
        condition=str(page.get("condition") or ""),
        availability=str(page.get("availability") or ""),
        snippet=snippet[:600],
        price_problem=problem,
        not_listing_reason=str(raw.get("_not_listing") or ""),
        raw=raw,
    )


class SearxngEbaySource:
    name = "searxng:ebay"
    label = "SearXNG · eBay"

    def __init__(self, settings: dict, client: httpx.Client | None = None):
        self.sx = _Searxng(settings, client)
        self.warnings: list[str] = []

    def describe(self, query: str, tracker: str) -> str:
        return self.sx.describe(query, "ebay")

    def search(self, query: str, tracker: str) -> list[Offer]:
        results, self.warnings = self.sx.search(query, "ebay")
        return [offer_from_searxng_ebay(r) for r in results if r.get("title")]


class SearxngWebSource:
    name = "searxng:web"
    label = "SearXNG · web search + store pages"

    def __init__(self, settings: dict, client: httpx.Client | None = None, page_client: httpx.Client | None = None):
        cfg = settings["sources"]["searxng"]
        self.sx = _Searxng(settings, client)
        self.engines = ",".join(e.strip() for e in cfg["web_engines"].split(",") if e.strip())
        if not self.engines:
            raise ProviderError("No web search engines set for SearXNG (Settings > Search sources)")
        self.sites = [s.strip().lower() for s in cfg["restrict_to_sites"] if s.strip()]
        self.pages = int(cfg["web_pages"])
        self.fetch_pages = cfg["fetch_product_pages"]
        self.fetch_budget = int(cfg["max_page_fetches"])
        self.merchants = settings["merchants"]
        self.allow_untrusted = settings["general"]["allow_untrusted_merchants"]
        self.page_client = page_client or self.sx.client
        self._pages: dict[str, dict] = {}  # one fetch per URL per scan
        self.warnings: list[str] = []

    def query_text(self, query: str) -> str:
        if not self.sites:
            return query
        return f"{query} ({' OR '.join('site:' + s for s in self.sites)})"

    def describe(self, query: str, tracker: str) -> str:
        return self.sx.describe(self.query_text(query), self.engines)

    def search(self, query: str, tracker: str) -> list[Offer]:
        results, self.warnings = self.sx.search(self.query_text(query), self.engines, self.pages)
        offers = []
        for result in results:
            raw = dict(result)
            problem = listing_page_problem(raw["url"])
            if problem:
                raw["_not_listing"] = problem
            else:
                self._attach_page(raw)
            offers.append(offer_from_searxng_web(raw))
        return offers

    def _attach_page(self, raw: dict) -> None:
        url = raw["url"]
        if not self.fetch_pages:
            raw["_not_fetched"] = "listing pages are not fetched (turned off in Settings)"
            return
        probe = Offer(title="", merchant=domain_of(url), price=None, link=url, merchant_url=url)
        verdict = judge_merchant(probe, self.merchants["trusted"], self.merchants["blocked"], self.allow_untrusted)
        if not verdict.trusted:
            raw["_not_fetched"] = "page not fetched because the site is not a trusted store"
            return
        if url not in self._pages:
            if self.fetch_budget <= 0:
                raw["_not_fetched"] = "page fetch limit for this scan reached (Settings > Search sources)"
                return
            self.fetch_budget -= 1
            self._pages[url] = fetch_product_page(self.page_client, url)
        raw["_page"] = self._pages[url]
