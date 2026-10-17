import pytest
from conftest import make_offer

from pricetracker.rules import evaluate
from pricetracker.rules.ram import detect_speed, parse_modules


def run(settings, title, price=150.0, **kwargs):
    return evaluate("ram", make_offer(title, price=price, **kwargs), settings)


def codes(ev):
    return {r.code for r in ev.reasons}


ACCEPTED = [
    # title, price, sticks, speed, module type
    ("Samsung 64GB 2Rx4 PC4-3200AA-RB2 DDR4 ECC Registered RDIMM", 150, 1, 3200, "RDIMM"),
    ("SK hynix 128GB (2 x 64GB) DDR4-3200 RDIMM Server Memory", 500, 2, 3200, "RDIMM"),
    ("128GB Kit 2x64GB DDR4 2933 LRDIMM", 400, 2, 2933, "LRDIMM"),
    ("64GBx2 DDR4 3200 RDIMM", 500, 2, 3200, "RDIMM"),
    ("Lot of 2 Samsung 64GB DDR4 3200 RDIMM", 400, 2, 3200, "RDIMM"),
    ("Micron 64GB DDR4 PC4-25600 Registered", 150, 1, 3200, "RDIMM"),
    ("Hynix HMAA8GR7AJR4N-XN 64GB DDR4 Server RAM", 150, 1, 3200, "RDIMM"),
    ("Micron MTA36ASF8G72PZ-2G9 DDR4 RDIMM", 150, 1, 2933, "RDIMM"),
    ("Kingston 64GB DDR4 3200MHz UDIMM Desktop Memory", 150, 1, 3200, "UDIMM"),
    ("64GB DDR4 3200 ECC Unbuffered DIMM", 150, 1, 3200, "UDIMM"),
    ("Samsung 64GB 4Rx4 PC4-2666V-LD2 Server Memory", 150, 1, 2666, "LRDIMM"),
    ("64GB DDR4 3200 RDIMM 16Gb B-die", 150, 1, 3200, "RDIMM"),
    ("Samsung 64GB DDR4 2933Y RDIMM", 150, 1, 2933, "RDIMM"),
    ("Samsung 64GB DDR4 3200 RDIMM tested, no damage", 150, 1, 3200, "RDIMM"),
    ("Samsung 64 GB DDR4-3200 288-pin Registered", 150, 1, 3200, "RDIMM"),
    ("Single 64GB DDR4 2666MHz ECC REG", 150, 1, 2666, "RDIMM"),
]


@pytest.mark.parametrize(("title", "price", "sticks", "speed", "module_type"), ACCEPTED)
def test_accepted_listings(settings, title, price, sticks, speed, module_type):
    ev = run(settings, title, price)
    assert ev.accepted, ev.reasons
    assert ev.attributes["sticks"] == sticks
    assert ev.attributes["stick_gb"] == 64
    assert ev.attributes["speed_mts"] == speed
    assert ev.attributes["module_type"] == module_type
    assert ev.unit_price == pytest.approx(price / sticks)


REJECTED = [
    ("64GB (2x32GB) DDR4 3200 RDIMM", "ram_module_size"),
    ("Crucial 64GB Kit (2 x 32GB) DDR4-3200 RDIMM", "ram_module_size"),
    ("128GB (4 x 32GB) DDR4 3200 RDIMM", "ram_module_size"),
    ("64GB Kit (8x8GB) DDR4 2666 RDIMM", "ram_module_size"),
    ("64GB Kit (4x16GB) DDR4 3200 Desktop", "ram_module_size"),
    ("Samsung 32GB DDR4 3200 RDIMM", "ram_module_size"),
    ("Samsung M393A4K40DB3-CWE 64GB Kit DDR4", "ram_module_size"),
    ("Crucial 64GB DDR4 3200 SODIMM", "ram_sodimm"),
    ("64GB DDR4 3200MHz Laptop Memory 260-pin", "ram_sodimm"),
    ("Samsung 64GB PC4-3200AA-SA2 DDR4", "ram_sodimm"),
    ("Samsung 64GB DDR4 2400 RDIMM", "ram_speed_low"),
    ("Samsung 64GB DDR4 RDIMM PC4-17000", "ram_speed_low"),
    ("Samsung 64GB DDR4 ECC Registered RDIMM", "ram_speed_unknown"),
    ("Samsung 64GB DDR5 4800 RDIMM", "ram_generation"),
    ("64GB DDR3 1600 RDIMM", "ram_generation"),
    ("64GB RDIMM Server Memory", "ram_generation"),
    ("Samsung 64GB DDR4 3200 RDIMM for parts not working", "condition_defect"),
    ("Samsung 64GB DDR4 3200 RDIMM - untested", "condition_untested"),
    ("Samsung 64GB DDR4 3200 RDIMM sold as-is", "condition_untested"),
    ("256GB (4x64GB) DDR4 3200 RDIMM", "ram_too_many_sticks"),
    ("Lot of 4 Samsung 64GB DDR4 3200 RDIMM", "ram_too_many_sticks"),
    ("Dell PowerEdge R740 768GB (12 x 64GB) DDR4 RAM", "ram_too_many_sticks"),
    ("64GB DDR4 3200 RDIMM Kit", "ram_module_ambiguous"),
    ("128GB DDR4 3200 RDIMM", "ram_module_size"),
    ("32GB/64GB DDR4 3200 RDIMM", "ram_module_ambiguous"),
    ("DDR4 3200 RDIMM server memory", "ram_module_ambiguous"),
    ("Samsung 64GB DDR4 3200 RDIMM motherboard combo", "excluded_keyword"),
]


@pytest.mark.parametrize(("title", "code"), REJECTED)
def test_rejected_listings(settings, title, code):
    ev = run(settings, title, price=600 if "4x64" in title or "12 x 64" in title else 150)
    assert not ev.accepted
    assert code in codes(ev), ev.reasons


def test_ddr4_inferred_from_speed(settings):
    ev = run(settings, "Samsung 64GB 3200MHz ECC RDIMM")
    assert ev.accepted
    assert any("inferred" in n for n in ev.notes)


def test_price_must_be_below_limit(settings):
    assert "price_too_high" in codes(run(settings, "Samsung 64GB DDR4 3200 RDIMM", price=300))
    assert run(settings, "Samsung 64GB DDR4 3200 RDIMM", price=299.99).accepted


def test_kit_price_is_per_stick_including_shipping(settings):
    ev = run(settings, "128GB (2x64GB) DDR4 3200 RDIMM", price=560, shipping=50.0)
    assert ev.unit_price == 305.0
    assert "price_too_high" in codes(ev)
    settings["trackers"]["ram"]["include_shipping"] = False
    assert run(settings, "128GB (2x64GB) DDR4 3200 RDIMM", price=560, shipping=50.0).accepted


def test_unknown_shipping_is_noted(settings):
    ev = run(settings, "Samsung 64GB DDR4 3200 RDIMM", shipping=None)
    assert ev.accepted
    assert any("Shipping" in n for n in ev.notes)


def test_suspiciously_cheap_is_rejected(settings):
    assert "price_suspicious" in codes(run(settings, "Samsung 64GB DDR4 3200 RDIMM", price=19.99))


def test_non_usd_rejected(settings):
    ev = run(settings, "Samsung 64GB DDR4 3200 RDIMM", currency="OTHER", price_text="CA$150")
    assert "price_currency" in codes(ev)


def test_speed_tiers(settings):
    assert run(settings, "Samsung 64GB DDR4 3200 RDIMM").tier == "preferred"
    ev = run(settings, "Samsung 64GB DDR4 2666 RDIMM")
    assert ev.accepted and ev.tier == "acceptable"
    assert any("preferred" in n for n in ev.notes)


def test_settings_change_rules(settings):
    ram = settings["trackers"]["ram"]
    ram["allow_unknown_speed"] = True
    assert run(settings, "Samsung 64GB DDR4 ECC Registered RDIMM").accepted
    ram["allowed_types"] = ["RDIMM"]
    assert "ram_type" in codes(run(settings, "Kingston 64GB DDR4 3200 UDIMM"))
    ram["allow_unknown_type"] = False
    assert "ram_type_unknown" in codes(run(settings, "Samsung 64GB DDR4 3200 Memory"))
    ram["min_speed_mts"] = 3200
    assert "ram_speed_low" in codes(run(settings, "Samsung 64GB DDR4 2933 RDIMM"))
    ram["max_sticks_per_listing"] = 4
    assert run(settings, "256GB (4x64GB) DDR4 3200 RDIMM", price=800).accepted


def test_all_failures_are_reported(settings):
    ev = run(
        settings,
        "Kingston 32GB DDR5 SODIMM for parts",
        price=500,
        merchant="sketchy.shop",
        link="https://sketchy.shop/x",
    )
    assert {"ram_generation", "ram_sodimm", "ram_module_size", "condition_defect", "merchant_untrusted"} <= codes(ev)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("DDR4 3200MHz", 3200),
        ("DDR4-2933", 2933),
        ("PC4-2666V-RB2", 2666),
        ("PC4-23400", 2933),
        ("2667MHz", 2666),
        ("M393A8G40AB2-CWE", 3200),
        ("MTA36ASF8G72PZ-2G6", 2666),
        ("supports 2400/2666/2933", 2933),
        ("DDR4 RDIMM", None),
    ],
)
def test_detect_speed(title, expected):
    assert detect_speed(title.lower()) == expected


def test_rank_notation_is_not_a_kit():
    info = parse_modules("Samsung 64GB 2Rx4 DDR4", 64)
    assert (info.stick_gb, info.sticks) == (64, 1)
    info = parse_modules("64GB 4Rx4 LRDIMM", 64)
    assert (info.stick_gb, info.sticks) == (64, 1)
