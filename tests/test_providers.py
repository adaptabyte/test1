import json

import httpx
import pytest

from pricetracker.providers import (
    ProviderError,
    SerpApiProvider,
    SerperProvider,
    get_provider,
    parse_price,
    parse_shipping,
)
from pricetracker.settings import DEFAULT_SETTINGS

SERPAPI_RESPONSE = {
    "search_metadata": {"status": "Success"},
    "shopping_results": [
        {
            "position": 1,
            "title": "Samsung 64GB DDR4 3200 RDIMM",
            "product_link": "https://www.google.com/shopping/product/111",
            "product_id": "111",
            "source": "eBay - seller1",
            "price": "$189.99",
            "extracted_price": 189.99,
            "delivery": "$5.99 delivery",
            "second_hand_condition": "used",
            "extensions": ["DDR4", "Free 30-day returns"],
            "thumbnail": "https://example.com/t.jpg",
        },
        {
            "position": 2,
            "title": "Hynix 64GB DDR4 2933 RDIMM",
            "link": "https://www.newegg.com/p/abc",
            "product_id": "222",
            "source": "Newegg.com",
            "price": "$210.00",
            "extracted_price": 210.0,
            "delivery": "Free delivery",
        },
    ],
    "categorized_shopping_results": [
        {"title": "More", "shopping_results": [{"title": "Micron 64GB", "source": "Amazon.com", "price": "$199"}]}
    ],
}


def cfg(**overrides):
    return {**DEFAULT_SETTINGS["provider"], "serpapi_api_key": "secret", "serper_api_key": "secret", **overrides}


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_serpapi_parses_results():
    seen = {}

    def handler(request):
        seen.update(request.url.params)
        return httpx.Response(200, json=SERPAPI_RESPONSE)

    offers = SerpApiProvider(cfg(), client(handler)).search("64GB DDR4")
    assert seen["engine"] == "google_shopping" and seen["q"] == "64GB DDR4" and seen["api_key"] == "secret"
    assert len(offers) == 3
    first, second, third = offers
    assert first.price == 189.99 and first.shipping == 5.99 and first.condition == "used"
    assert first.link == "https://www.google.com/shopping/product/111" and first.merchant_url == ""
    assert first.extensions == ["DDR4"]  # delivery/returns chips are not product specs
    assert second.merchant_url == "https://www.newegg.com/p/abc" and second.shipping == 0
    assert third.price == 199.0 and third.shipping is None


def test_serpapi_describe_hides_key():
    assert "secret" not in SerpApiProvider(cfg(), client(lambda r: None)).describe("q")


def test_serpapi_errors():
    def error(request):
        return httpx.Response(401, json={"error": "Invalid API key."})

    with pytest.raises(ProviderError, match="Invalid API key"):
        SerpApiProvider(cfg(), client(error)).search("q")

    def no_results(request):
        return httpx.Response(200, json={"error": "Google hasn't returned any results for this query."})

    assert SerpApiProvider(cfg(), client(no_results)).search("q") == []

    def not_json(request):
        return httpx.Response(502, text="<html>Bad gateway</html>")

    with pytest.raises(ProviderError, match="502"):
        SerpApiProvider(cfg(), client(not_json)).search("q")

    def network(request):
        raise httpx.ConnectError("boom")

    with pytest.raises(ProviderError, match="ConnectError"):
        SerpApiProvider(cfg(), client(network)).search("q")


def test_missing_api_key():
    with pytest.raises(ProviderError, match="API key"):
        get_provider(cfg(serpapi_api_key=""))
    with pytest.raises(ProviderError, match="API key"):
        get_provider(cfg(name="serper", serper_api_key=""))


def test_serper_parses_results():
    seen = {}

    def handler(request):
        seen["key"] = request.headers["x-api-key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "shopping": [
                    {
                        "title": "AMD Ryzen Threadripper PRO 3975WX",
                        "source": "eBay",
                        "link": "https://www.ebay.com/itm/1",
                        "price": "$449.00",
                        "delivery": "Free shipping",
                        "productId": "999",
                        "position": 1,
                    }
                ]
            },
        )

    offers = SerperProvider(cfg(name="serper"), client(handler)).search("3975WX")
    assert seen["key"] == "secret" and seen["body"]["q"] == "3975WX"
    assert offers[0].price == 449.0 and offers[0].shipping == 0 and offers[0].product_id == "999"
    assert offers[0].merchant_url == "https://www.ebay.com/itm/1"


def test_serper_error_message():
    def handler(request):
        return httpx.Response(403, json={"message": "Unauthorized."})

    with pytest.raises(ProviderError, match="Unauthorized"):
        SerperProvider(cfg(name="serper"), client(handler)).search("q")


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
        (("", "Delivery: $5"), 5.0),
    ],
)
def test_parse_shipping(texts, expected):
    assert parse_shipping(*texts)[0] == expected
