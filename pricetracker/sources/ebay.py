"""eBay's official Browse API: free (5,000 calls a day) with a developer account.

Optional, but the most reliable eBay source: structured prices, shipping, condition and seller
feedback, and no scraping. Auctions are excluded (only Buy It Now / Best Offer listings).
"""

from __future__ import annotations

import re
import time
from urllib.parse import urlencode

import httpx

from ..models import Offer
from .common import ProviderError, parse_price, request_json

TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
SCOPE = "https://api.ebay.com/oauth/api_scope"


def offer_from_ebay_api(item: dict) -> Offer:
    price_info = item.get("price") or {}
    price, _ = parse_price(price_info.get("value"))
    currency = "USD" if price_info.get("currency", "USD") == "USD" else "OTHER"
    costs = []
    for option in item.get("shippingOptions") or []:
        cost = option.get("shippingCost") or {}
        amount, _ = parse_price(cost.get("value"))
        if amount is not None:
            costs.append(amount)
    seller = item.get("seller") or {}
    percent = seller.get("feedbackPercentage")
    score = seller.get("feedbackScore")
    url = str(item.get("itemWebUrl") or "")
    return Offer(
        title=str(item.get("title") or ""),
        merchant=f"eBay - {seller['username']}" if seller.get("username") else "eBay",
        price=price,
        price_text=f"{price_info.get('value', '')} {price_info.get('currency', '')}".strip(),
        currency=currency,
        shipping=min(costs) if costs else None,
        link=url,
        merchant_url=url,
        product_id=str(item.get("itemId") or ""),
        condition=str(item.get("condition") or ""),
        thumbnail=str((item.get("image") or {}).get("imageUrl") or ""),
        seller_feedback_percent=float(percent) if percent not in (None, "") else None,
        seller_feedback_score=int(score) if score not in (None, "") else None,
        ships_from=str((item.get("itemLocation") or {}).get("country") or ""),
        raw=item,
    )


class EbayApiSource:
    name = "ebay_api"
    label = "eBay Browse API"

    def __init__(self, settings: dict, client: httpx.Client | None = None):
        cfg = settings["sources"]["ebay_api"]
        self.client_id, self.client_secret = cfg["client_id"], cfg["client_secret"]
        if not self.client_id or not self.client_secret:
            raise ProviderError(
                "eBay API client ID / secret not set "
                "(Settings > Search sources, or EBAY_CLIENT_ID / EBAY_CLIENT_SECRET)"
            )
        self.limit = max(1, min(int(settings["search"]["results_per_query"]), 200))
        self.country = settings["search"]["country"].upper()
        self.categories = {key: str(t.get("ebay_category_id") or "") for key, t in settings["trackers"].items()}
        self.client = client or httpx.Client(timeout=settings["search"]["timeout_seconds"])
        self._token = ""
        self._token_expires = 0.0

    def _params(self, query: str, tracker: str) -> dict:
        params = {
            "q": query,
            "limit": self.limit,
            "filter": f"buyingOptions:{{FIXED_PRICE|BEST_OFFER}},deliveryCountry:{self.country}",
        }
        if self.categories.get(tracker):
            params["category_ids"] = self.categories[tracker]
        return params

    def describe(self, query: str, tracker: str) -> str:
        return f"GET {SEARCH_URL}?{urlencode(self._params(query, tracker))}"

    def _access_token(self) -> str:
        if self._token and time.monotonic() < self._token_expires:
            return self._token
        data = request_json(
            self.client,
            "POST",
            TOKEN_URL,
            auth=(self.client_id, self.client_secret),
            data={"grant_type": "client_credentials", "scope": SCOPE},
        )
        if not data.get("access_token"):
            raise ProviderError("eBay did not return an access token; check the client ID and secret")
        self._token = data["access_token"]
        self._token_expires = time.monotonic() + int(data.get("expires_in", 7200)) - 120
        return self._token

    def search(self, query: str, tracker: str) -> list[Offer]:
        try:
            token = self._access_token()
        except ProviderError as exc:
            raise ProviderError(f"eBay sign-in failed: {exc}") from exc
        headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": f"EBAY_{self.country}"}
        data = request_json(self.client, "GET", SEARCH_URL, params=self._params(query, tracker), headers=headers)
        items = data.get("itemSummaries") or []
        return [offer_from_ebay_api(item) for item in items if item.get("title")]


def ebay_item_id(url: str) -> str:
    """The eBay item number in a listing URL, so the same item from different sources is one listing."""
    match = re.search(r"ebay\.[a-z.]+/itm/(?:[^/?#]+/)?(\d{9,15})", url or "", re.I)
    return match.group(1) if match else ""
