import copy
import json

import httpx
import pytest

from pricetracker.settings import DEFAULT_SETTINGS
from pricetracker.sources import ProviderError, build_sources, ebay_item_id, offer_from_raw, parse_price, parse_shipping
from pricetracker.sources.ebay import EbayApiSource
from pricetracker.sources.google_shopping import SerpApiSource, SerperSource
from pricetracker.sources.pages import listing_page_problem, parse_product_page
from pricetracker.sources.searxng import SearxngEbaySource, SearxngWebSource


def make_settings(**sources):
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    settings["sources"]["searxng"]["base_url"] = "http://searxng:8080"
    for name, cfg in sources.items():
        settings["sources"][name].update(cfg)
    return settings


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- SearXNG ---------------------------------------------------------------------------------------

EBAY_ENGINE_JSON = {
    "query": "64GB DDR4 RDIMM",
    "results": [
        {"url": "https://www.ebay.com/itm/123456789012?hash=x", "title": "Samsung 64GB DDR4 3200 RDIMM",
         "content": "Pre-Owned", "price": "$189.99", "shipping": "+$5.00 shipping",
         "source_country": "from United States", "engine": "ebay", "template": "products.html"},
        {"url": "https://www.ebay.com/itm/210987654321", "title": "Hynix 128GB (2x64GB) DDR4 2933 RDIMM",
         "content": "Pre-Owned", "price": "$400.00 to $450.00", "shipping": "Free delivery",
         "source_country": "from China", "engine": "ebay"},
    ],
    "unresponsive_engines": [],
}  # fmt: skip


def test_searxng_ebay_engine_results():
    seen = {}

    def handler(request):
        seen.update(request.url.params)
        return httpx.Response(200, json=EBAY_ENGINE_JSON)

    source = SearxngEbaySource(make_settings(), client(handler))
    offers = source.search("64GB DDR4 RDIMM", "ram")
    assert seen["format"] == "json" and seen["engines"] == "ebay" and seen["language"] == "en-US"
    first, second = offers
    assert (first.price, first.shipping, first.condition, first.ships_from) == (
        189.99,
        5.0,
        "Pre-Owned",
        "United States",
    )
    assert first.merchant == "eBay" and first.merchant_url.startswith("https://www.ebay.com/itm/")
    assert (second.price, second.shipping, second.ships_from) == (400.0, 0.0, "China")
    assert "engines=ebay" in source.describe("64GB DDR4 RDIMM", "ram")


def test_searxng_reports_engine_failures():
    def failed(request):
        return httpx.Response(200, json={"results": [], "unresponsive_engines": [["ebay", "HTTP connection error"]]})

    with pytest.raises(ProviderError, match="ebay: HTTP connection error"):
        SearxngEbaySource(make_settings(), client(failed)).search("q", "ram")

    def partial(request):
        return httpx.Response(200, json={**EBAY_ENGINE_JSON, "unresponsive_engines": [["google", "CAPTCHA"]]})

    source = SearxngEbaySource(make_settings(), client(partial))
    assert len(source.search("q", "ram")) == 2
    assert source.warnings == ["google: CAPTCHA"]


def test_searxng_json_disabled_hint():
    def forbidden(request):
        return httpx.Response(403, text="<html>Forbidden</html>")

    with pytest.raises(ProviderError, match="Enable `json`"):
        SearxngEbaySource(make_settings(), client(forbidden)).search("q", "ram")


def test_searxng_requires_url():
    settings = make_settings(searxng={"base_url": ""})
    with pytest.raises(ProviderError, match="SearXNG URL is not set"):
        SearxngEbaySource(settings, client(lambda r: None))


EBAY_ITEM_PAGE = """<html><head><title>Samsung 64GB DDR4 3200 RDIMM | eBay</title>
<script type="application/ld+json">{"@context":"https://schema.org","@type":"Product","name":"Samsung 64GB 2Rx4 PC4-3200AA RDIMM M393A8G40AB2-CWE",
"offers":{"@type":"Offer","price":"179.00","priceCurrency":"USD","availability":"https://schema.org/InStock",
"itemCondition":"https://schema.org/UsedCondition",
"shippingDetails":{"@type":"OfferShippingDetails","shippingRate":{"@type":"MonetaryAmount","value":"0.00","currency":"USD"}}}}</script>
</head><body></body></html>"""

NEWEGG_PAGE = """<html><head>
<meta property="og:title" content="SK hynix 64GB DDR4 3200 ECC RDIMM - Newegg.com">
<meta property="og:description" content="Server memory, registered, 2Rx4">
<script type="application/ld+json">[{"@type":"BreadcrumbList"},{"@graph":[{"@type":"Product","name":"SK hynix 64GB DDR4 3200 ECC RDIMM",
"offers":[{"@type":"Offer","priceSpecification":{"price":"259.99","priceCurrency":"USD"},"availability":"OutOfStock"}]}]}]</script>
</head></html>"""

BOT_WALL = "<html><head><title>Robot or human?</title></head><body>Activate and hold the button</body></html>"


def web_settings(**searxng):
    return make_settings(searxng={"web_engines": "duckduckgo,brave", **searxng})


def test_searxng_web_reads_listing_pages():
    results = [
        {
            "url": "https://www.ebay.com/itm/123456789012",
            "title": "Samsung 64GB DDR4 3200 RDIMM | eBay",
            "content": "x",
        },
        {"url": "https://www.newegg.com/sk-hynix-64gb/p/1X5-0001-00001", "title": "SK hynix 64GB", "content": ""},
        {"url": "https://www.walmart.com/ip/Samsung-64GB/123", "title": "Samsung 64GB DDR4 - Walmart.com"},
        {"url": "https://www.ebay.com/sch/i.html?_nkw=64gb+ddr4", "title": "64gb ddr4 for sale | eBay"},
        {"url": "https://www.reddit.com/r/homelab/comments/abc/64gb_rdimm_deals/", "title": "64GB RDIMM deals"},
        {"url": "https://cheap-ram-store.shop/products/64gb-ddr4", "title": "64GB DDR4 RDIMM $49"},
    ]
    queries, fetched = [], []

    def searxng(request):
        queries.append(request.url.params["q"])
        return httpx.Response(200, json={"results": results, "unresponsive_engines": []})

    def pages(request):
        fetched.append(str(request.url))
        body = {"www.ebay.com": EBAY_ITEM_PAGE, "www.newegg.com": NEWEGG_PAGE}.get(request.url.host, BOT_WALL)
        return httpx.Response(403 if body is BOT_WALL else 200, text=body)

    source = SearxngWebSource(web_settings(), client(searxng), client(pages))
    offers = source.search("64GB DDR4 RDIMM", "ram")
    assert "site:ebay.com OR site:newegg.com" in queries[0]
    by_url = {o.link: o for o in offers}
    by_host = {o.merchant: o for o in offers if "/sch/" not in o.link}

    ebay = by_url[results[0]["url"]]
    assert (ebay.price, ebay.shipping, ebay.condition, ebay.availability) == (179.0, 0.0, "Used", "InStock")
    assert ebay.title.startswith("Samsung 64GB 2Rx4")
    newegg = by_host["newegg.com"]
    assert (newegg.price, newegg.availability) == (259.99, "OutOfStock")
    assert newegg.snippet.endswith("Server memory, registered, 2Rx4")
    walmart = by_host["walmart.com"]
    assert walmart.price is None and "bot protection" in walmart.price_problem
    assert "not an individual listing" in by_url[results[3]["url"]].not_listing_reason
    assert "not a trusted store" in by_host["reddit.com"].price_problem
    scam = by_host["cheap-ram-store.shop"]
    assert scam.price is None and "not a trusted store" in scam.price_problem
    # only listing pages on trusted stores are opened
    assert sorted(fetched) == sorted(r["url"] for r in results[:3])

    # each page is fetched once per scan
    source.search("64GB DDR4 RDIMM", "ram")
    assert len(fetched) == 3


def test_searxng_web_fetch_budget_and_switch():
    results = [{"url": f"https://www.ebay.com/itm/10000000000{i}", "title": "x"} for i in range(3)]

    def searxng(request):
        return httpx.Response(200, json={"results": results})

    pages = client(lambda r: httpx.Response(200, text=EBAY_ITEM_PAGE))
    offers = SearxngWebSource(web_settings(max_page_fetches=2), client(searxng), pages).search("q", "ram")
    assert [o.price for o in offers] == [179.0, 179.0, None]
    assert "fetch limit" in offers[2].price_problem
    offers = SearxngWebSource(web_settings(fetch_product_pages=False), client(searxng), pages).search("q", "ram")
    assert all("turned off" in o.price_problem for o in offers)
    plain = SearxngWebSource(web_settings(restrict_to_sites=[]), client(searxng), pages)
    assert "site:" not in plain.query_text("q")


# --- Product pages ---------------------------------------------------------------------------------


def test_parse_product_page_variants():
    page = parse_product_page(EBAY_ITEM_PAGE)
    assert page["price"] == 179.0 and page["currency"] == "USD" and page["method"] == "json-ld"
    aggregate = '<script type="application/ld+json">{"@type":["Product"],"name":"X","offers":{"@type":"AggregateOffer","lowPrice":199.5,"priceCurrency":"USD"}}</script>'
    assert parse_product_page(aggregate)["price"] == 199.5
    meta_only = (
        '<meta content="149.95" property="product:price:amount"><meta property="product:price:currency" content="USD">'
        '<meta property="og:title" content="Micron 64GB RDIMM | Micro Center">'
    )
    page = parse_product_page(meta_only)
    assert (page["price"], page["currency"], page["method"], page["title"]) == (
        149.95,
        "USD",
        "meta",
        "Micron 64GB RDIMM",
    )
    microdata = '<span itemprop="price" content="210.00">$210.00</span>'
    assert parse_product_page(microdata)["price"] == 210.0
    broken = '<script type="application/ld+json">{not json</script><title>Thing</title>'
    assert "price" not in parse_product_page(broken)


@pytest.mark.parametrize(
    ("url", "is_listing"),
    [
        ("https://www.ebay.com/itm/123456789012", True),
        ("https://www.ebay.com/b/Server-Memory-RAM/11071/bn_7023302080", False),
        ("https://www.newegg.com/p/pl?d=64gb+ddr4", False),
        ("https://www.newegg.com/sk-hynix-64gb/p/N82E16820000001", True),
        ("https://www.bestbuy.com/site/crucial-64gb/6501234.p?skuId=6501234", True),
        ("https://www.bestbuy.com/site/searchpage.jsp?st=64gb", False),
        ("https://www.amazon.com/Samsung-64GB/dp/B08XYZ1234", True),
        ("https://www.amazon.com/s?k=64gb+ddr4", False),
        ("https://www.servethehome.com/forums/threads/64gb-rdimm.1234/", False),
        ("https://memory.net/product/m393a8g40ab2-cwe/", True),
        ("https://example.com/", False),
    ],
)
def test_listing_page_problem(url, is_listing):
    assert (listing_page_problem(url) == "") is is_listing


# --- eBay Browse API -------------------------------------------------------------------------------

EBAY_SEARCH = {
    "total": 2,
    "itemSummaries": [
        {
            "itemId": "v1|123456789012|0",
            "title": "Samsung 64GB DDR4 3200 RDIMM M393A8G40AB2-CWE",
            "price": {"value": "185.00", "currency": "USD"},
            "shippingOptions": [{"shippingCostType": "FIXED", "shippingCost": {"value": "7.50", "currency": "USD"}},
                                {"shippingCostType": "FIXED", "shippingCost": {"value": "4.99", "currency": "USD"}}],
            "condition": "Used",
            "itemWebUrl": "https://www.ebay.com/itm/123456789012",
            "seller": {"username": "server_parts_co", "feedbackPercentage": "99.6", "feedbackScore": 5321},
            "itemLocation": {"country": "US"},
        },
        {
            "itemId": "v1|222|0",
            "title": "AMD Ryzen Threadripper PRO 3975WX",
            "price": {"value": "420.00", "currency": "USD"},
            "condition": "For parts or not working",
            "itemWebUrl": "https://www.ebay.com/itm/222222222222",
            "seller": {"username": "newbie", "feedbackPercentage": "100.0", "feedbackScore": 2},
        },
    ],
}  # fmt: skip


def ebay_settings():
    return make_settings(ebay_api={"enabled": True, "client_id": "app-id", "client_secret": "cert-id"})


def test_ebay_api_search():
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/oauth2/token"):
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 7200})
        return httpx.Response(200, json=EBAY_SEARCH)

    source = EbayApiSource(ebay_settings(), client(handler))
    offers = source.search("64GB DDR4 RDIMM", "ram")
    source.search("3975WX", "cpu")
    token_calls = [c for c in calls if c.url.path.endswith("/token")]
    searches = [c for c in calls if c.url.path.endswith("/search")]
    assert len(token_calls) == 1 and token_calls[0].headers["authorization"].startswith("Basic ")
    assert searches[0].headers["authorization"] == "Bearer tok"
    assert searches[0].headers["x-ebay-c-marketplace-id"] == "EBAY_US"
    assert searches[0].url.params["category_ids"] == "170083" and searches[1].url.params["category_ids"] == "164"
    assert "buyingOptions:{FIXED_PRICE|BEST_OFFER}" in searches[0].url.params["filter"]

    first, second = offers
    assert (first.price, first.shipping, first.merchant) == (185.0, 4.99, "eBay - server_parts_co")
    assert (first.seller_feedback_percent, first.seller_feedback_score, first.ships_from) == (99.6, 5321, "US")
    assert second.shipping is None and second.condition == "For parts or not working"


def test_ebay_api_bad_credentials():
    def handler(request):
        return httpx.Response(
            401, json={"error": "invalid_client", "error_description": "client authentication failed"}
        )

    with pytest.raises(ProviderError, match=r"eBay sign-in failed.*client authentication failed"):
        EbayApiSource(ebay_settings(), client(handler)).search("q", "ram")


def test_ebay_item_id():
    assert ebay_item_id("https://www.ebay.com/itm/123456789012?hash=abc") == "123456789012"
    assert ebay_item_id("https://www.ebay.com/itm/samsung-64gb/123456789012") == "123456789012"
    assert ebay_item_id("https://www.newegg.com/p/123456789012") == ""


# --- Registry, paid sources ------------------------------------------------------------------------


def test_build_sources_reports_problems():
    settings = make_settings(searxng={"base_url": ""}, ebay_api={"enabled": True}, serpapi={"enabled": True})
    sources, problems = build_sources(settings)
    assert sources == []
    assert any("SearXNG URL" in p for p in problems)
    assert any("eBay API client ID" in p for p in problems)
    assert any("SerpApi API key" in p for p in problems)


def test_build_sources_order():
    settings = make_settings(ebay_api={"enabled": True, "client_id": "a", "client_secret": "b"}, demo={"enabled": True})
    sources, problems = build_sources(settings)
    assert [s.name for s in sources] == ["ebay_api", "searxng:ebay", "searxng:web", "demo"]
    assert problems == []


def test_offer_from_raw_round_trip():
    raw = {**EBAY_ENGINE_JSON["results"][0]}
    assert offer_from_raw("searxng:ebay", raw).price == 189.99
    web_raw = {
        "url": "https://www.ebay.com/itm/123456789012",
        "title": "t",
        "_page": {"price": 99.0, "currency": "USD"},
    }
    assert offer_from_raw("searxng:web", web_raw).price == 99.0
    assert offer_from_raw("ebay_api", EBAY_SEARCH["itemSummaries"][0]).seller_feedback_score == 5321


SERPAPI_RESPONSE = {
    "shopping_results": [
        {"position": 1, "title": "Samsung 64GB DDR4 3200 RDIMM", "product_link": "https://www.google.com/shopping/product/111",
         "product_id": "111", "source": "eBay - seller1", "price": "$189.99", "extracted_price": 189.99,
         "delivery": "$5.99 delivery", "second_hand_condition": "used", "extensions": ["DDR4", "Free 30-day returns"]},
        {"position": 2, "title": "Hynix 64GB DDR4 2933 RDIMM", "link": "https://www.newegg.com/p/abc", "product_id": "222",
         "source": "Newegg.com", "price": "$210.00", "extracted_price": 210.0, "delivery": "Free delivery"},
    ],
    "categorized_shopping_results": [{"shopping_results": [{"title": "Micron 64GB", "source": "Amazon.com", "price": "$199"}]}],
}  # fmt: skip


def test_serpapi_source():
    seen = {}

    def handler(request):
        seen.update(request.url.params)
        return httpx.Response(200, json=SERPAPI_RESPONSE)

    settings = make_settings(serpapi={"enabled": True, "api_key": "secret"})
    source = SerpApiSource(settings, client(handler))
    offers = source.search("64GB DDR4", "ram")
    assert seen["engine"] == "google_shopping" and seen["api_key"] == "secret"
    assert [o.price for o in offers] == [189.99, 210.0, 199.0]
    assert offers[0].extensions == ["DDR4"] and offers[0].shipping == 5.99
    assert "secret" not in source.describe("q", "ram")

    def no_results(request):
        return httpx.Response(200, json={"error": "Google hasn't returned any results for this query."})

    assert SerpApiSource(settings, client(no_results)).search("q", "ram") == []

    def bad_key(request):
        return httpx.Response(401, json={"error": "Invalid API key."})

    with pytest.raises(ProviderError, match="Invalid API key"):
        SerpApiSource(settings, client(bad_key)).search("q", "ram")


def test_serper_source():
    def handler(request):
        assert request.headers["x-api-key"] == "secret" and json.loads(request.content)["q"] == "3975WX"
        return httpx.Response(200, json={"shopping": [{"title": "AMD 3975WX", "source": "eBay", "price": "$449.00",
                                                        "link": "https://www.ebay.com/itm/1", "delivery": "Free shipping"}]})  # fmt: skip

    offers = SerperSource(make_settings(serper={"enabled": True, "api_key": "secret"}), client(handler)).search(
        "3975WX", "cpu"
    )
    assert (offers[0].price, offers[0].shipping) == (449.0, 0.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("$1,299.99", (1299.99, "USD")), ("US$120", (120.0, "USD")), ("CA$150.00", (150.0, "OTHER")), ("", (None, "USD"))],
)
def test_parse_price(value, expected):
    assert parse_price(value) == expected


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (("Free delivery",), 0.0),
        (("$7.99 delivery",), 7.99),
        (("+$12.50 shipping",), 12.5),
        (("Free 30-day returns",), None),
    ],
)
def test_parse_shipping(texts, expected):
    assert parse_shipping(*texts)[0] == expected
