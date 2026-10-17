from conftest import make_offer

from pricetracker.rules import evaluate

RAM = "Samsung 64GB DDR4 3200 RDIMM"


def codes(ev):
    return {r.code for r in ev.reasons}


def test_out_of_stock_and_ended_listings_rejected(settings):
    for availability in ["OutOfStock", "SoldOut", "Discontinued", "out of stock"]:
        ev = evaluate("ram", make_offer(RAM, availability=availability), settings)
        assert "listing_unavailable" in codes(ev), availability
    assert evaluate("ram", make_offer(RAM, availability="InStock"), settings).accepted


def test_seller_feedback_limits(settings):
    good = make_offer(RAM, seller_feedback_percent=99.8, seller_feedback_score=4200)
    assert evaluate("ram", good, settings).accepted
    low = make_offer(RAM, seller_feedback_percent=93.0, seller_feedback_score=4200)
    assert "seller_feedback" in codes(evaluate("ram", low, settings))
    new_account = make_offer(RAM, seller_feedback_percent=100.0, seller_feedback_score=3)
    assert "seller_feedback" in codes(evaluate("ram", new_account, settings))
    settings["general"]["min_seller_feedback_score"] = 0
    assert evaluate("ram", new_account, settings).accepted


def test_sources_without_feedback_are_not_penalised(settings):
    assert evaluate("ram", make_offer(RAM), settings).accepted


def test_non_listing_pages_only_get_one_reason(settings):
    page = make_offer("64gb ddr4 for sale | eBay", price=None, not_listing_reason="ebay.com page that is not a listing")
    ev = evaluate("ram", page, settings)
    assert codes(ev) == {"not_a_listing"}


def test_missing_price_explains_why(settings):
    ev = evaluate("ram", make_offer(RAM, price=None, price_problem="bot protection blocked the request"), settings)
    assert any(r.code == "price_missing" and "bot protection" in r.message for r in ev.reasons)


def test_price_range_and_foreign_shipping_are_noted(settings):
    ev = evaluate("ram", make_offer(RAM, price=180.0, price_text="$180.00 to $240.00", ships_from="China"), settings)
    assert ev.accepted
    assert any("Price range" in n for n in ev.notes) and any("Ships from China" in n for n in ev.notes)


def test_for_parts_condition_from_ebay_api(settings):
    ev = evaluate(
        "cpu",
        make_offer("AMD Ryzen Threadripper PRO 3975WX", price=300, condition="For parts or not working"),
        settings,
    )
    assert "condition_defect" in codes(ev)
