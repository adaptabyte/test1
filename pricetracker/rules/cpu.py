"""Rules for the Threadripper PRO CPU tracker (default model: 3975WX)."""

from __future__ import annotations

import re

from ..models import Evaluation, Offer
from .common import apply_price, check_condition, check_excluded_keywords, norm, strip_matches

# Phrases that say the CPU is NOT vendor locked. These are removed before looking for lock signals,
# so "not vendor locked" is never read as "vendor locked".
_UNLOCK_PHRASES = [
    r"\b(?:not|non|no|never|isn'?t|is\s+not|without|zero)[\s-]+(?:a\s+|any\s+)?"
    r"(?:(?:vendor|lenovo|dell|hp|oem|psb|platform)[\s-]*)*(?:lock(?:ed|ing)?|fused|bound|tied)\b",
    r"\b(?:vendor|psb|lenovo|oem)[\s-]*(?:un[\s-]?locked|lock[\s-]?free)\b",
    r"\bno[\s-]+psb\b",
    r"\bpsb[\s-]*(?:disabled|off|not\s+(?:enabled|fused|set|activated))\b",
    r"\bnot\s+(?:tied|bound)\s+to\b",
    r"\bunlocked\s+(?:for|to)\s+any\b",
]
_LOCK_SIGNALS = [
    (r"\b(?:vendor|lenovo|dell|hp|oem|psb)[\s-]*lock(?:ed)?\b", "says vendor locked"),
    (r"\blocked\s+to\b", "says locked to a vendor"),
    (r"\bpsb\b", "mentions PSB (AMD Platform Secure Boot)"),
    (r"\block(?:ed)?\b", "mentions 'locked'"),
]
# Lenovo ThinkStation P620 pulls are almost always PSB-locked to Lenovo boards.
_LENOVO_SIGNAL = r"\b(?:lenovo|thinkstation|p620)\b"
_NEW_RETAIL = r"\b(?:brand\s+new|new\s+in\s+box|nib|factory\s+sealed|sealed|retail\s+box(?:ed)?|boxed)\b"

# Accessories that mention the CPU model ("cooler for 3975WX"), after removing "no cooler" style denials.
_ACCESSORY_DENIALS = [
    r"\b(?:no|without|w/o|excluding|excludes|not\s+including)\s+(?:the\s+|a\s+|any\s+)?(?:cpu\s+)?"
    r"(?:cooler|fan|heatsink|heat\s+sink|box|bracket|torque\s+(?:tool|wrench)|screwdriver)\b",
    r"\b(?:cooler|fan|heatsink|heat\s+sink|box|bracket)\s+(?:is\s+)?(?:not\s+included|sold\s+separately)\b",
]
_ACCESSORY_TERMS = [
    r"\bcoolers?\b", r"\bheat[\s-]?sinks?\b", r"\baio\b", r"\bwater[\s-]?block\b", r"\bbrackets?\b",
    r"\bmounting\s+(?:kit|bracket)\b", r"\bretention\s+(?:kit|frame)\b", r"\bthermal\s+(?:paste|pads?|compound)\b",
    r"\bsocket\s+(?:protector|cover)\b", r"\bbox\s+only\b", r"\bempty\s+box\b", r"\bpackaging\s+only\b",
    r"\bdelid", r"\btorque\s+(?:tool|wrench|screwdriver)\b",
]  # fmt: skip
_BUNDLE_TERMS = [
    r"\bcombo\b", r"\bbundle\b", r"\bbarebones?\b", r"\bprecision\s+\d{4}",
    r"\bworkstation\s+(?:pc|computer|tower|desktop)\b", r"\b(?:desktop|gaming|tower)\s+(?:pc|computer)\b",
    r"\bcomplete\s+(?:system|build)\b",
    r"(?:\+|\bwith\b|\bw/|&|\band\b)\s*(?:an?\s+)?(?:[\w-]+\s+){0,5}?(?:motherboard|mobo|mainboard|gpu)\b",
    r"(?:\+|\bwith\b|\bw/|&|\band\b)\s*(?:an?\s+)?(?:[\w-]+\s+){0,2}?(?:ram|memory)\b",
]  # fmt: skip
_COMPATIBILITY_PHRASES = [r"\b(?:compatible\s+with|for\s+use\s+with|works\s+with|fits)\b[^,;|()]*"]
_CPU_MODEL = re.compile(r"\b(\d{4})[\s-]?(wx|x)\b")
_UNIT_COUNT = [r"\blot\s?(?:of\s)?\(?\s?(\d{1,2})\b", r"\b(\d{1,2})\s?(?:pcs|pieces|cpus|processors)\b"]


def _model_pattern(model: str) -> str:
    match = re.fullmatch(r"\s*(\d+)\s*([a-z]*)\s*", model.lower())
    if not match:
        return r"\b" + re.escape(model.lower()) + r"\b"
    digits, suffix = match.groups()
    return rf"\b{digits}[\s-]?{suffix}\b" if suffix else rf"\b{digits}\b"


def lock_status(full_text_norm: str) -> tuple[str, list[str]]:
    """Classify vendor-lock status.

    Returns (status, evidence) where status is one of: locked, lenovo_risk, lenovo_claims_unlocked,
    explicitly_unlocked, likely_unlocked, unknown.
    """
    remaining, unlock_phrases = strip_matches(full_text_norm, _UNLOCK_PHRASES)
    lock_hits = [label for pattern, label in _LOCK_SIGNALS if re.search(pattern, remaining)]
    if len(lock_hits) > 1 and "mentions 'locked'" in lock_hits:
        lock_hits.remove("mentions 'locked'")
    strong_lock = [hit for hit in lock_hits if hit != "mentions PSB (AMD Platform Secure Boot)"]
    lenovo = re.search(_LENOVO_SIGNAL, full_text_norm)

    if strong_lock or (lock_hits and not unlock_phrases):
        return "locked", lock_hits
    if lenovo and unlock_phrases:
        return "lenovo_claims_unlocked", [f"mentions '{lenovo.group(0)}'", *unlock_phrases]
    if lenovo:
        return "lenovo_risk", [f"mentions '{lenovo.group(0)}'"]
    if unlock_phrases:
        return "explicitly_unlocked", unlock_phrases
    retail = re.search(_NEW_RETAIL, full_text_norm)
    if retail:
        return "likely_unlocked", [f"'{retail.group(0)}' (new retail CPUs ship unlocked)"]
    if re.search(r"\bunlocked\b", full_text_norm):
        return "likely_unlocked", ["says 'unlocked' (can also mean multiplier-unlocked)"]
    return "unknown", []


LOCK_STATUS_LABELS = {
    "locked": "vendor locked",
    "lenovo_risk": "likely Lenovo PSB-locked",
    "lenovo_claims_unlocked": "Lenovo pull, seller says unlocked",
    "explicitly_unlocked": "listing says not vendor locked",
    "likely_unlocked": "probably unlocked",
    "unknown": "not stated",
}


def evaluate_cpu(offer: Offer, cfg: dict) -> Evaluation:
    ev = Evaluation()
    spec_norm = norm(offer.spec_text)
    full_norm = norm(offer.full_text)
    model = cfg["model"].strip()
    model_norm = norm(model).replace(" ", "")

    aliases = [norm(a) for a in cfg.get("model_aliases", []) if a.strip()]
    if re.search(_model_pattern(model), spec_norm) or any(a in spec_norm for a in aliases):
        ev.attributes["model"] = model.upper()
    else:
        ev.reject("cpu_model", f"Title does not mention {model.upper()}")

    others = sorted({"".join(m) for m in _CPU_MODEL.findall(spec_norm)} - {model_norm})
    if others:
        ev.reject("cpu_other_model", "Also mentions " + ", ".join(o.upper() for o in others))

    without_denials, _ = strip_matches(spec_norm, _ACCESSORY_DENIALS)
    accessory = [m.group(0) for p in _ACCESSORY_TERMS if (m := re.search(p, without_denials))]
    if accessory:
        ev.reject("cpu_accessory", "Looks like an accessory, not a CPU (" + ", ".join(accessory) + ")")

    bundle_text, _ = strip_matches(spec_norm, _COMPATIBILITY_PHRASES)
    bundle = [m.group(0).strip() for p in _BUNDLE_TERMS if (m := re.search(p, bundle_text))]
    if bundle:
        ev.reject("cpu_bundle", "Bundle or complete system (" + ", ".join(bundle) + ")")

    status, evidence = lock_status(full_norm)
    ev.attributes["vendor_lock"] = LOCK_STATUS_LABELS[status]
    detail = f" ({'; '.join(evidence)})" if evidence else ""
    if status == "locked":
        ev.reject("cpu_vendor_locked", "Vendor locked" + detail)
    elif status == "lenovo_risk":
        ev.reject("cpu_lenovo_risk", "Lenovo/P620 pull without an unlock statement; likely PSB-locked" + detail)
    elif status == "lenovo_claims_unlocked":
        ev.note("Lenovo pull but the seller says it is not locked; confirm before buying" + detail)
    elif status == "unknown":
        if cfg["lock_policy"] == "require_unlocked":
            ev.reject("cpu_lock_unknown", "Listing does not say the CPU is unlocked (strict lock policy)")
        else:
            ev.note("Vendor-lock status not stated; ask the seller if it was used in a Lenovo P620")
    elif status == "likely_unlocked":
        ev.note("Vendor lock: probably unlocked" + detail)

    check_condition(ev, offer, reject_untested=cfg["reject_untested"])
    check_excluded_keywords(ev, offer, cfg["exclude_keywords"])

    units = 1
    for pattern in _UNIT_COUNT:
        match = re.search(pattern, spec_norm)
        if match and int(match.group(1)) > 1:
            units = int(match.group(1))
            ev.note(f"Listing appears to contain {units} CPUs; price is divided per CPU")
            break
    apply_price(
        ev,
        offer,
        units=units,
        include_shipping=cfg["include_shipping"],
        must_be_below=float(cfg["max_price"]),
        suspicious_below=float(cfg["min_price"]),
        unit_name="CPU",
    )
    return ev
