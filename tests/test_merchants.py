import pytest
from conftest import make_offer

from pricetracker.merchants import DEFAULT_TRUSTED_MERCHANTS, domain_of, judge_merchant, normalize_merchant


def judge(merchant, link, blocked=(), allow_untrusted=False):
    offer = make_offer("x", merchant=merchant, link=link)
    offer.merchant_url = link if domain_of(link) else ""
    return judge_merchant(offer, DEFAULT_TRUSTED_MERCHANTS, list(blocked), allow_untrusted)


@pytest.mark.parametrize(
    ("merchant", "link", "canonical"),
    [
        ("eBay - servergear_outlet", "https://www.ebay.com/itm/1", "eBay"),
        ("eBay", "https://www.google.com/shopping/product/123", "eBay"),
        ("eBay", "", "eBay"),
        ("Newegg.com - Mega Memory", "https://www.newegg.com/p/1", "Newegg"),
        ("Walmart - TechSeller", "https://www.walmart.com/ip/1", "Walmart"),
        ("Best Buy", "https://www.bestbuy.com/site/1.p", "Best Buy"),
        ("B&H Photo-Video-Audio", "https://www.bhphotovideo.com/c/product/1", "B&H Photo"),
        ("Poshmark", "https://poshmark.com/listing/1", "Poshmark"),
        ("Amazon.com & more", "https://www.google.com/shopping/product/1", "Amazon"),
        ("Some Name", "https://pages.ebay.com/x", "eBay"),
        ("Newegg", "https://www.google.com/url?url=https://www.newegg.com/p/123&sa=U", "Newegg"),
    ],
)
def test_trusted(merchant, link, canonical):
    verdict = judge(merchant, link)
    assert verdict.trusted and verdict.on_list
    assert verdict.canonical == canonical


@pytest.mark.parametrize(
    ("merchant", "link", "code"),
    [
        ("memorydealz.shop", "https://memorydealz.shop/p/1", "merchant_untrusted"),
        ("Ebay Outlet Deals", "https://www.google.com/shopping/product/1", "merchant_untrusted"),
        ("eBay", "https://ebay-clearance-deals.shop/item/1", "merchant_mismatch"),
        ("Newegg", "https://www.newegg.com.cheap-parts.io/p/1", "merchant_mismatch"),
        ("", "", "merchant_untrusted"),
    ],
)
def test_untrusted(merchant, link, code):
    verdict = judge(merchant, link)
    assert not verdict.trusted
    assert verdict.code == code


def test_blocked_beats_trusted():
    assert judge("eBay - scammer", "https://www.ebay.com/itm/1", blocked=["ebay.com"]).code == "merchant_blocked"
    assert judge("Walmart - Seller", "", blocked=["walmart"]).code == "merchant_blocked"
    assert judge("x", "https://scam.shop/p", blocked=["scam.shop"]).code == "merchant_blocked"


def test_blocking_one_marketplace_seller_keeps_the_rest():
    blocked = ["eBay - scammer"]
    assert judge("eBay - scammer", "https://www.ebay.com/itm/1", blocked=blocked).code == "merchant_blocked"
    assert judge("eBay - honest", "https://www.ebay.com/itm/2", blocked=blocked).trusted


def test_allow_untrusted():
    verdict = judge("memorydealz.shop", "https://memorydealz.shop/p/1", allow_untrusted=True)
    assert verdict.trusted and not verdict.on_list


def test_normalize_merchant():
    assert normalize_merchant("eBay - seller - with dashes") == "ebay"
    assert normalize_merchant("Amazon.com") == "amazon"
    assert normalize_merchant("Micro Center") == "micro center"
