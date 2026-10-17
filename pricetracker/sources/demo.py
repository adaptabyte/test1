"""Demo source: canned results that exercise every rule, so the app can be tried without any setup."""

from __future__ import annotations

from ..models import Offer
from .demo_data import demo_results
from .google_shopping import offer_from_serpapi


class DemoSource:
    name = "demo"
    label = "Demo data"

    def __init__(self, settings: dict, client=None):
        pass

    def describe(self, query: str, tracker: str) -> str:
        return f"demo data for {query!r} (no network request)"

    def search(self, query: str, tracker: str) -> list[Offer]:
        return [offer_from_serpapi(item) for item in demo_results(query)]
