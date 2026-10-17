"""Helpers shared by every search source."""

from __future__ import annotations

import re

import httpx


class ProviderError(Exception):
    """A search failed or a source is misconfigured; the message is shown in the scan log."""


def parse_price(value) -> tuple[float | None, str]:
    """Return (amount, currency) from a number or a price string like '$1,299.99' or 'CA$120'."""
    if isinstance(value, bool):
        return None, "USD"
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
    """Shipping cost from delivery text ('Free delivery', '$7.99 delivery', '+$12 shipping')."""
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


def request_json(client: httpx.Client, method: str, url: str, **kwargs) -> dict:
    """Make a request and return the JSON object, turning every failure into a readable ProviderError."""
    try:
        response = client.request(method, url, **kwargs)
    except httpx.HTTPError as exc:
        raise ProviderError(f"Request failed: {exc.__class__.__name__}: {exc}") from exc
    try:
        data = response.json()
    except ValueError:
        raise ProviderError(f"HTTP {response.status_code}: response was not JSON: {response.text[:200]!r}") from None
    if response.status_code >= 400:
        message = _error_message(data) or str(data)[:200]
        raise ProviderError(f"HTTP {response.status_code}: {message}")
    if not isinstance(data, dict):
        raise ProviderError("Unexpected response shape (expected a JSON object)")
    return data


def _error_message(data) -> str:
    if not isinstance(data, dict):
        return ""
    errors = data.get("errors")
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        return str(errors[0].get("longMessage") or errors[0].get("message") or "")
    return str(data.get("error_description") or data.get("error") or data.get("message") or "")
