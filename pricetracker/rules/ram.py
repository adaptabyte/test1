"""Rules for the "64GB-per-stick DDR4 DIMM" tracker."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Evaluation, Offer
from .common import apply_price, check_condition, check_excluded_keywords, norm

_GEN_PATTERNS = {
    "DDR4": r"\bddr[\s-]?4(?!\d*\s?gb?\b)|\bpc4(?:\b|[-\s]?\d{4})",
    "DDR5": r"\bddr[\s-]?5(?!\d*\s?gb?\b)|\bpc5(?:\b|[-\s]?\d{4})",
    "DDR3": r"\bddr[\s-]?3l?(?!\d*\s?gb?\b)|\bpc3l?(?:\b|[-\s]?\d{4})",
}

_SODIMM_PATTERNS = [
    (r"\bso[\s-]?dimm", "SO-DIMM"),
    (r"\b260[\s-]?pins?\b", "260-pin"),
    (r"\blaptops?\b", "laptop"),
    (r"\bnotebooks?\b", "notebook"),
    (r"\bimac\b", "iMac"),
    (r"\bmacbook", "MacBook"),
    (r"\bnuc\b", "NUC"),
    (r"\bmini[\s-]?pc\b", "mini PC"),
    (r"\bsmall\s+outline\b", "small outline"),
]

# JEDEC module label, e.g. PC4-3200AA-RB2 (R=RDIMM, L=LRDIMM, U=UDIMM, E=ECC UDIMM, S=SO-DIMM).
_JEDEC_LABEL = re.compile(r"\bpc4[-\s]?(\d{4})[a-z]{1,2}[-\s]([rlues])[a-z]\d")
_JEDEC_TYPES = {"r": "RDIMM", "l": "LRDIMM", "u": "UDIMM", "e": "UDIMM", "s": "SODIMM"}

_TYPE_PATTERNS = [
    ("LRDIMM", r"\blr[\s-]?dimm|\bload[\s-]?reduced"),
    ("RDIMM", r"\br[\s-]?dimm|\bregistered\b|\breg\b|\becc[\s-]?reg|\b3ds\b|\brecc\b|\bbuffered\b"),
    ("UDIMM", r"\bu[\s-]?dimm|\bunbuffered\b|\bnon[\s-]?ecc\b|\bnon[\s-]?registered\b"),
]

# Known DDR4 server-module part numbers: prefix -> (module type, GB per module).
_PART_NUMBERS = [
    (r"\bm393a8g40[a-z0-9]{3}", "RDIMM", 64),  # Samsung 64GB 2Rx4 RDIMM
    (r"\bm393a8k40[a-z0-9]{3}", "RDIMM", 64),  # Samsung 64GB 2Rx4 RDIMM (8Gb)
    (r"\bm386a8k40[a-z0-9]{3}", "LRDIMM", 64),  # Samsung 64GB 4Rx4 LRDIMM
    (r"\bhmaa8gr7[a-z0-9]{3}", "RDIMM", 64),  # SK hynix 64GB RDIMM
    (r"\bhmaa8gl7[a-z0-9]{3}", "LRDIMM", 64),  # SK hynix 64GB LRDIMM
    (r"\bmta36asf8g72pz", "RDIMM", 64),  # Micron 64GB 2Rx4 RDIMM
    (r"\bmta36asf8g72lz", "LRDIMM", 64),  # Micron 64GB LRDIMM
    (r"\bmta72ass8g72lz", "LRDIMM", 64),  # Micron 64GB 4Rx4 LRDIMM
    (r"\bm393a4k40[a-z0-9]{3}", "RDIMM", 32),  # Samsung 32GB
    (r"\bm393a4g40[a-z0-9]{3}", "RDIMM", 32),  # Samsung 32GB
    (r"\bhma84gr7[a-z0-9]{3}", "RDIMM", 32),  # SK hynix 32GB
    (r"\bmta36asf4g72pz", "RDIMM", 32),  # Micron 32GB
    (r"\bmta18asf4g72", "RDIMM", 32),  # Micron 32GB
    (r"\bm393aag40[a-z0-9]{3}", "RDIMM", 128),  # Samsung 128GB
    (r"\bm386aag40[a-z0-9]{3}", "LRDIMM", 128),  # Samsung 128GB
]

_SPEED_ALIASES = {2667: 2666, 2934: 2933, 2132: 2133}
_BANDWIDTH_TO_SPEED = {17000: 2133, 19200: 2400, 21300: 2666, 21333: 2666, 23400: 2933, 23466: 2933, 25600: 3200}
_PART_SPEED_CODES = {
    # Samsung, SK hynix and Micron speed-grade suffixes.
    "cpb": 2133, "crc": 2400, "ctd": 2666, "cvf": 2933, "cwe": 3200,
    "tf": 2133, "ug": 2400, "vk": 2666, "wm": 2933, "xn": 3200,
    "2g1": 2133, "2g3": 2400, "2g6": 2666, "2g9": 2933, "3g2": 3200,
}  # fmt: skip
_SPEED_PATTERNS = [
    r"\bddr[\s-]?4[-\s]?(\d{4})\b",
    r"\b(\d{4})\s?(?:mhz|mt/s|mts|mt|m/s)\b",
    r"\bpc4[-\s]?(\d{4})[a-z]{0,2}\b",
    r"\b(2133|2400|2666|2667|2933|3200)[a-z]{0,2}\b",
]
_BANDWIDTH_PATTERN = r"\bpc4[-\s]?(\d{5})\b"
_PART_SPEED_PATTERN = r"(?:\b(?:m3\w+|hma\w+|mta\w+))-(cpb|crc|ctd|cvf|cwe|tf|ug|vk|wm|xn|2g1|2g3|2g6|2g9|3g2)\b"

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "six": 6, "eight": 8, "twelve": 12, "sixteen": 16}
_COUNT = r"(\d{1,2}|one|two|three|four|six|eight|twelve|sixteen)"
_GB = r"\s?(?:gb|g)\b"
_VALID_SIZES = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}

# "2x64GB", "(2 x 64 GB)", "2*64G"
_NXM = re.compile(r"(?<![\w.])" + _COUNT + r"\s?[x*]\s?(\d{1,3})" + _GB)
# "64GBx2", "64GB x 2"
_MXN = re.compile(r"(?<![\w.])(\d{1,3})\s?(?:gb|g)\s?[x*]\s?(\d{1,2})\b(?!\s?(?:gb|g|rank|r)\b)")
# "2pcs 64GB", "2 sticks of 64GB", "two 64GB modules" is handled by count hints
_PCS_OF = re.compile(
    r"(?<![\w.])" + _COUNT + r"\s?(?:pcs?|pieces?|sticks?|modules?|dimms?)\s?(?:of\s|x\s?)?(\d{1,3})" + _GB
)
# Count words that refer to individual items: "lot of 2", "2 pcs", "qty 2", "2 sticks".
_ITEM_COUNT = [
    r"\blot\s?(?:of\s)?\(?\s?" + _COUNT + r"\b",
    r"\b" + _COUNT + r"[\s-]?(?:pcs|pieces|sticks|modules|dimms|lot)\b",
    r"\bqty[:\s]{0,2}(\d{1,2})\b",
]
# Count words that may refer to a kit whose total capacity is the advertised number.
_KIT_COUNT = [
    r"\b(?:kit|set|pack)\s?(?:of\s)?\(?\s?" + _COUNT + r"\b",
    r"\b" + _COUNT + r"[\s-]?(?:pack|pk)\b",
]
_KIT_WORDS = r"\bkit\b|\bpair\b|\bdual[\s-]?(?:channel\s)?kit\b|\btwin[\s-]?pack\b|\bmatched\b"
_SINGLE_WORDS = r"\bsingle\s?(?:stick|module|dimm)\b|\b1\s?pcs?\b|\b1\s?stick\b"


def _count_value(token: str) -> int:
    return _NUMBER_WORDS.get(token, 0) or int(token)


@dataclass
class ModuleInfo:
    """How many modules a listing contains and how big each one is."""

    stick_gb: int | None = None
    sticks: int | None = None
    source: str = ""  # human-readable explanation of how this was worked out
    problem: str = ""  # set when the configuration could not be determined safely


def _capacities(text: str) -> list[int]:
    """Capacities written as GB in the original (case-sensitive) text; "16Gb" is gigabits and ignored."""
    sizes = []
    for match in re.finditer(r"(?<![\w.\-])(\d{1,4})\s?(GB|gb|Gb|gB|G|g)(?![a-zA-Z])", text):
        if match.group(2) == "Gb":
            continue
        value = int(match.group(1))
        if value in _VALID_SIZES:
            sizes.append(value)
    return sizes


def _part_number_hint(text_norm: str) -> tuple[str, int, str] | None:
    for pattern, module_type, size in _PART_NUMBERS:
        match = re.search(pattern, text_norm)
        if match:
            return module_type, size, match.group(0).upper()
    return None


def parse_modules(text: str, target_gb: int) -> ModuleInfo:
    """Work out per-stick capacity and stick count from a listing title."""
    text_norm = norm(text)
    explicit: set[tuple[int, int]] = set()
    for match in _NXM.finditer(text_norm):
        explicit.add((_count_value(match.group(1)), int(match.group(2))))
    for match in _MXN.finditer(text_norm):
        explicit.add((int(match.group(2)), int(match.group(1))))
    for match in _PCS_OF.finditer(text_norm):
        explicit.add((_count_value(match.group(1)), int(match.group(2))))
    explicit = {(count, size) for count, size in explicit if count > 0 and size in _VALID_SIZES}

    if len(explicit) > 1:
        configs = ", ".join(f"{c}x{s}GB" for c, s in sorted(explicit))
        return ModuleInfo(problem=f"Conflicting module configurations in title ({configs})")
    if explicit:
        count, size = next(iter(explicit))
        return ModuleInfo(stick_gb=size, sticks=count, source=f"title states {count}x{size}GB")

    part = _part_number_hint(text_norm)
    capacities = set(_capacities(text.translate(str.maketrans({"×": "x"}))))
    item_count = _first_count(text_norm, _ITEM_COUNT)
    kit_count = _first_count(text_norm, _KIT_COUNT)
    is_single = bool(re.search(_SINGLE_WORDS, text_norm))
    mentions_kit = bool(re.search(_KIT_WORDS, text_norm)) or kit_count is not None

    if part:
        # A known part number pins the per-module size; the title only has to agree with it.
        _, size, part_number = part
        others = capacities - {size}
        source = f"part number {part_number} is a {size}GB module"
        if not others:
            sticks = item_count or kit_count or (1 if is_single or not mentions_kit else None)
            if sticks:
                return ModuleInfo(stick_gb=size, sticks=sticks, source=source)
        elif len(others) == 1 and next(iter(others)) % size == 0:
            total = next(iter(others))
            return ModuleInfo(stick_gb=size, sticks=total // size, source=f"{source}, {total}GB total")

    if not capacities:
        return ModuleInfo(problem="Capacity not stated in title")

    if len(capacities) == 1:
        (size,) = capacities
        if item_count:
            return ModuleInfo(stick_gb=size, sticks=item_count, source=f"{item_count} items of {size}GB")
        if kit_count and size % kit_count == 0 and (size // kit_count) in _VALID_SIZES:
            per_stick = size // kit_count
            return ModuleInfo(
                stick_gb=per_stick, sticks=kit_count, source=f"{size}GB kit of {kit_count} = {per_stick}GB per stick"
            )
        if is_single or not mentions_kit:
            return ModuleInfo(stick_gb=size, sticks=1, source=f"single {size}GB module")
        return ModuleInfo(problem=f"{size}GB kit without a per-stick breakdown (could be several smaller sticks)")

    # Several capacities and no NxM pattern, e.g. "128GB Kit 64GB DDR4": accept only if consistent.
    smallest, largest = min(capacities), max(capacities)
    if len(capacities) == 2 and largest % smallest == 0 and smallest == target_gb:
        sticks = largest // smallest
        if mentions_kit or item_count == sticks or kit_count == sticks:
            return ModuleInfo(stick_gb=smallest, sticks=sticks, source=f"{largest}GB total made of {smallest}GB sticks")
    sizes = ", ".join(f"{s}GB" for s in sorted(capacities))
    return ModuleInfo(problem=f"Several capacities in title ({sizes}) and no per-stick breakdown")


def _first_count(text_norm: str, patterns: list[str]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text_norm)
        if match:
            value = _count_value(match.group(1))
            if value > 0:
                return value
    return None


def detect_speed(text_norm: str) -> int | None:
    """Highest DDR4 data rate (MT/s) mentioned, from speeds, PC4 labels or part-number speed codes."""
    speeds: list[int] = []
    for pattern in _SPEED_PATTERNS:
        speeds.extend(int(m) for m in re.findall(pattern, text_norm))
    speeds.extend(_BANDWIDTH_TO_SPEED.get(int(m), 0) for m in re.findall(_BANDWIDTH_PATTERN, text_norm))
    speeds.extend(_PART_SPEED_CODES[m] for m in re.findall(_PART_SPEED_PATTERN, text_norm))
    speeds = [_SPEED_ALIASES.get(s, s) for s in speeds if 1600 <= s <= 5000]
    return max(speeds) if speeds else None


def detect_type(text_norm: str) -> tuple[str | None, str]:
    """Return (module type, evidence). Type is RDIMM, LRDIMM, UDIMM, SODIMM or None."""
    label = _JEDEC_LABEL.search(text_norm)
    if label:
        return _JEDEC_TYPES[label.group(2)], f"JEDEC label {label.group(0).upper()}"
    for module_type, pattern in _TYPE_PATTERNS:
        match = re.search(pattern, text_norm)
        if match:
            return module_type, f"'{match.group(0)}'"
    part = _part_number_hint(text_norm)
    if part:
        return part[0], f"part number {part[2]}"
    return None, ""


def evaluate_ram(offer: Offer, cfg: dict) -> Evaluation:
    ev = Evaluation()
    text = offer.spec_text
    text_norm = norm(text)
    target_gb = int(cfg["module_size_gb"])

    speed = detect_speed(text_norm)
    generations = [gen for gen, pattern in _GEN_PATTERNS.items() if re.search(pattern, text_norm)]
    if not generations and (_part_number_hint(text_norm) or (speed and 2400 <= speed <= 3200)):
        # DDR4 server parts and DDR4-2400..3200 speeds don't exist in other generations' RDIMMs.
        generations = ["DDR4"]
        ev.note("DDR4 not stated; inferred from the speed / part number")
    ev.attributes["generation"] = "/".join(generations) or "not stated"
    if generations != ["DDR4"]:
        if not generations:
            ev.reject("ram_generation", "DDR generation not stated (need DDR4)")
        elif "DDR4" in generations:
            ev.reject("ram_generation", f"Mentions several DDR generations ({', '.join(generations)})")
        else:
            ev.reject("ram_generation", f"{'/'.join(generations)}, not DDR4")

    sodimm = [label for pattern, label in _SODIMM_PATTERNS if re.search(pattern, text_norm)]
    module_type, type_evidence = detect_type(text_norm)
    if module_type == "SODIMM":
        sodimm.append(type_evidence)
    if sodimm:
        ev.attributes["module_type"] = "SO-DIMM"
        ev.reject("ram_sodimm", "Laptop / SO-DIMM module (" + ", ".join(sodimm) + ")")
    elif module_type:
        ev.attributes["module_type"] = module_type
        if module_type not in cfg["allowed_types"]:
            ev.reject("ram_type", f"{module_type} is not an allowed module type ({type_evidence})")
    else:
        ev.attributes["module_type"] = "not stated"
        if cfg["allow_unknown_type"]:
            ev.note("Module type (RDIMM/LRDIMM/UDIMM) not stated")
        else:
            ev.reject("ram_type_unknown", "Module type (RDIMM/LRDIMM/UDIMM) not stated")

    modules = parse_modules(text, target_gb)
    if modules.problem:
        ev.reject("ram_module_ambiguous", modules.problem)
    else:
        ev.attributes["stick_gb"] = modules.stick_gb
        ev.attributes["sticks"] = modules.sticks
        ev.attributes["module_config"] = modules.source
        if modules.stick_gb != target_gb:
            ev.reject("ram_module_size", f"{modules.stick_gb}GB per stick, need {target_gb}GB ({modules.source})")
        elif modules.sticks > int(cfg["max_sticks_per_listing"]):
            ev.reject(
                "ram_too_many_sticks",
                f"{modules.sticks} sticks in one listing (max {cfg['max_sticks_per_listing']})",
            )

    ev.attributes["speed_mts"] = speed or "not stated"
    if speed is None:
        if cfg["allow_unknown_speed"]:
            ev.note("Speed not stated; check before buying")
        else:
            ev.reject("ram_speed_unknown", "Speed not stated in title")
    elif speed < int(cfg["min_speed_mts"]):
        ev.reject("ram_speed_low", f"DDR4-{speed} is below the DDR4-{cfg['min_speed_mts']} minimum")
    elif speed >= int(cfg["preferred_speed_mts"]):
        ev.tier = "preferred"
    else:
        ev.tier = "acceptable"
        ev.note(f"DDR4-{speed}: acceptable, but below the preferred DDR4-{cfg['preferred_speed_mts']}")

    check_condition(ev, offer, reject_untested=cfg["reject_untested"])
    check_excluded_keywords(ev, offer, cfg["exclude_keywords"])
    apply_price(
        ev,
        offer,
        units=modules.sticks or 1,
        include_shipping=cfg["include_shipping"],
        must_be_below=float(cfg["max_price_per_stick"]),
        suspicious_below=float(cfg["min_price_per_stick"]),
        unit_name="stick",
    )
    return ev
