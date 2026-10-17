"""Tracker rules: turn a normalized offer into an accept/reject decision with reasons."""

from __future__ import annotations

from ..merchants import judge_merchant
from ..models import Evaluation, Offer
from .cpu import evaluate_cpu
from .ram import evaluate_ram

TRACKER_KINDS = {
    "ram": {"evaluate": evaluate_ram, "unit": "stick"},
    "cpu": {"evaluate": evaluate_cpu, "unit": "CPU"},
}

REASON_LABELS = {
    "merchant_untrusted": "Seller not on trusted list",
    "merchant_blocked": "Seller blocked",
    "merchant_mismatch": "Seller name and link disagree",
    "price_missing": "No price",
    "price_currency": "Not priced in USD",
    "price_too_high": "Over price limit",
    "price_suspicious": "Suspiciously low price",
    "condition_defect": "For parts / broken",
    "condition_untested": "Untested / as-is",
    "excluded_keyword": "Excluded keyword",
    "ram_generation": "Not DDR4",
    "ram_sodimm": "Laptop / SO-DIMM",
    "ram_type": "Module type not allowed",
    "ram_type_unknown": "Module type not stated",
    "ram_module_size": "Not 64GB per stick",
    "ram_module_ambiguous": "Unclear stick configuration",
    "ram_too_many_sticks": "Too many sticks in listing",
    "ram_speed_low": "Speed below minimum",
    "ram_speed_unknown": "Speed not stated",
    "cpu_model": "Wrong CPU model",
    "cpu_other_model": "Mentions other CPU models",
    "cpu_accessory": "Accessory, not a CPU",
    "cpu_bundle": "Bundle / complete system",
    "cpu_vendor_locked": "Vendor locked",
    "cpu_lenovo_risk": "Likely Lenovo PSB-locked",
    "cpu_lock_unknown": "Lock status not stated",
}


def evaluate(tracker_key: str, offer: Offer, settings: dict) -> Evaluation:
    """Run the tracker's product rules, then the shared merchant-trust check."""
    tracker = settings["trackers"][tracker_key]
    ev = TRACKER_KINDS[tracker_key]["evaluate"](offer, tracker)
    merchants = settings["merchants"]
    verdict = judge_merchant(
        offer, merchants["trusted"], merchants["blocked"], settings["general"]["allow_untrusted_merchants"]
    )
    ev.merchant = verdict.canonical
    ev.merchant_domain = verdict.domain
    if not verdict.trusted:
        ev.reject(verdict.code, verdict.message)
    elif not verdict.on_list:
        ev.note("Seller is not on the trusted list (untrusted sellers are allowed in settings)")
    if offer.multiple_sources:
        ev.note("Google groups several sellers under this listing; the price shown is the lowest")
    return ev


def unit_name(tracker_key: str) -> str:
    return TRACKER_KINDS[tracker_key]["unit"]
