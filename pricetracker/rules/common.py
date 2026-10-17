"""Text helpers and checks shared by every tracker."""

from __future__ import annotations

import re

from ..models import Evaluation, Offer

_DASHES = str.maketrans({"–": "-", "—": "-", "×": "x", "✕": "x", " ": " "})


def norm(text: str) -> str:
    """Lowercase, unify dashes and the multiplication sign, collapse whitespace."""
    return re.sub(r"\s+", " ", text.translate(_DASHES).lower()).strip()


def find_terms(text_norm: str, terms: list[str]) -> list[str]:
    """Return the terms that occur in `text_norm` as whole words or phrases."""
    hits = []
    for term in terms:
        term_norm = norm(term)
        if not term_norm:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(term_norm) + r"(?![a-z0-9])"
        if re.search(pattern, text_norm):
            hits.append(term)
    return hits


def strip_matches(text: str, patterns: list[str]) -> tuple[str, list[str]]:
    """Remove every match of `patterns` from text; return the remaining text and what was removed."""
    removed: list[str] = []
    for pattern in patterns:
        removed.extend(m.group(0) for m in re.finditer(pattern, text))
        text = re.sub(pattern, " ", text)
    return text, removed


# Phrases that mention a defect only to deny it ("no bent pins", "not for parts").
_DENIED_DEFECTS = [
    r"\b(?:no|not|zero|without|never|free\s+of)\s+(?:any\s+)?(?:bent\s+pins?|missing\s+pins?|damaged?|damages"
    r"|broken|issues?|problems?|defects?|scratch(?:es)?|for\s+parts)\b",
]

_DEFECTS = [
    (r"\bfor\s+parts\b", "for parts"),
    (r"\bparts\s*(?:only|/\s*repair|or\s+repair|&\s*repair|and\s+repair|/\s*not\s+working)\b", "parts/repair"),
    (r"\bfor\s+repair\b", "for repair"),
    (r"\bnot\s+working\b", "not working"),
    (r"\bnon[\s-]?working\b", "non-working"),
    (r"\bdoes\s*n[o']?t\s+work\b", "doesn't work"),
    (r"\bbroken\b", "broken"),
    (r"\bdamaged?\b", "damaged"),
    (r"\bdefective\b", "defective"),
    (r"\bfaulty\b", "faulty"),
    (r"\bcracked\b", "cracked"),
    (r"\bbent\s+pins?\b", "bent pins"),
    (r"\bmissing\s+pins?\b", "missing pins"),
    (r"\bno[\s-]post\b", "no POST"),
    (r"\bdead\b", "dead"),
]

_RISKY = [
    (r"\buntested\b", "untested"),
    (r"\bas[\s-]is\b", "as-is"),
]


def check_condition(ev: Evaluation, offer: Offer, reject_untested: bool) -> None:
    """Reject for-parts / broken listings, and optionally untested / as-is ones."""
    text = norm(" | ".join([offer.spec_text, offer.condition]))
    text, _ = strip_matches(text, _DENIED_DEFECTS)
    defects = [label for pattern, label in _DEFECTS if re.search(pattern, text)]
    if defects:
        ev.reject("condition_defect", "Listed as " + ", ".join(defects))
    risky = [label for pattern, label in _RISKY if re.search(pattern, text)]
    if risky and reject_untested:
        ev.reject("condition_untested", "Listed as " + ", ".join(risky))
    elif risky:
        ev.note("Listed as " + ", ".join(risky))
    if offer.condition:
        ev.attributes["condition"] = offer.condition


def check_excluded_keywords(ev: Evaluation, offer: Offer, keywords: list[str]) -> None:
    hits = find_terms(norm(offer.spec_text), keywords)
    if hits:
        ev.reject("excluded_keyword", "Contains excluded keyword(s): " + ", ".join(hits))


def apply_price(
    ev: Evaluation,
    offer: Offer,
    *,
    units: int,
    include_shipping: bool,
    must_be_below: float,
    suspicious_below: float,
    unit_name: str,
) -> None:
    """Compute total and per-unit price and enforce the tracker's price window."""
    if offer.price is None:
        ev.reject("price_missing", "Listing has no price")
        return
    if offer.currency != "USD":
        ev.reject("price_currency", f"Price is not in USD ({offer.price_text or offer.currency})")
        return
    total = offer.price
    if include_shipping:
        if offer.shipping is None:
            ev.note("Shipping cost not stated; price shown excludes shipping")
        else:
            total += offer.shipping
    units = max(units, 1)
    ev.units = units
    ev.total_price = round(total, 2)
    ev.unit_price = round(total / units, 2)
    if ev.unit_price >= must_be_below:
        ev.reject(
            "price_too_high",
            f"${ev.unit_price:,.2f} per {unit_name} is not below the ${must_be_below:,.2f} limit",
        )
    elif ev.unit_price < suspicious_below:
        ev.reject(
            "price_suspicious",
            f"${ev.unit_price:,.2f} per {unit_name} is below the ${suspicious_below:,.2f} sanity floor "
            "(likely an accessory, a scam or a mis-parsed listing)",
        )
