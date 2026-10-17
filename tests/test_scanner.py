import threading

import pytest
from conftest import demo_only

from pricetracker.models import Offer
from pricetracker.scanner import ScanAlreadyRunning, _scan_lock, replay_run, start_scan
from pricetracker.sources import ProviderError
from pricetracker.sources.google_shopping import offer_from_serpapi


class FakeSource:
    """Returns whatever `results` holds for each query; raises for queries listed in `fail`."""

    def __init__(self, results=None, fail=(), name="serpapi"):
        self.results = results or {}
        self.fail = set(fail)
        self.calls = []
        self.name = name

    def describe(self, query, tracker):
        return f"fake {query}"

    def search(self, query, tracker):
        self.calls.append(query)
        if query in self.fail:
            raise ProviderError(f"quota exceeded for {query}")
        return [offer_from_serpapi(item) for item in self.results.get(query, [])]


def item(title, price, source="eBay - seller", product_id="p1", link="https://www.ebay.com/itm/1"):
    return {"title": title, "source": source, "extracted_price": price, "price": f"${price}", "link": link,
            "product_id": product_id, "delivery": "Free delivery"}  # fmt: skip


RAM = "Samsung 64GB DDR4 3200 RDIMM"


@pytest.fixture
def one_query(configure):
    def change(s):
        s["trackers"]["ram"]["queries"] = ["ram query"]
        s["trackers"]["cpu"]["queries"] = ["cpu query"]

    return configure(change)


def test_demo_scan_records_everything(db, configure):
    configure(demo_only)
    run_id = start_scan(db, "manual")
    run = db.get_run(run_id)
    assert run["status"] == "ok"
    assert run["query_count"] == 6
    assert run["accepted_count"] == 6 and run["rejected_count"] == 21
    assert run["new_match_count"] == 6
    assert run["email_status"].startswith("not sent: Resend API key not set")
    observations = db.observations_for_run(run_id)
    assert len(observations) == 27
    assert all(o["reasons"] for o in observations if o["decision"] == "rejected")
    assert db.list_notifications()[0]["status"] == "skipped"


def test_emails_once_then_only_on_price_drop(db, one_query, email_on, fake_smtp):
    provider = FakeSource({"ram query": [item(RAM, 250)]})
    run_id = start_scan(db, "manual", sources=[provider])
    assert db.get_run(run_id)["email_status"].startswith("sent 1")
    assert len(fake_smtp.sent) == 1
    message = fake_smtp.sent[0]
    assert message["To"] == "adaptabyte@gmail.com"
    assert "64GB DDR4 from $250.00/stick" in message["Subject"]
    assert "https://www.ebay.com/itm/1" in message.get_body(("plain",)).get_content()

    start_scan(db, "manual", sources=[provider])
    assert len(fake_smtp.sent) == 1  # same listing, same price: no repeat email

    provider.results["ram query"] = [item(RAM, 240)]  # 4% drop: below the 5% threshold
    start_scan(db, "manual", sources=[provider])
    assert len(fake_smtp.sent) == 1

    provider.results["ram query"] = [item(RAM, 200)]
    run_id = start_scan(db, "manual", sources=[provider])
    assert len(fake_smtp.sent) == 2
    assert "Price drop" in fake_smtp.sent[1].get_body(("plain",)).get_content()
    assert db.observations_for_run(run_id)[0]["notify_reason"] == "price_drop"


def test_same_seller_and_title_with_new_id_is_not_re_emailed(db, one_query, email_on, fake_smtp):
    provider = FakeSource({"ram query": [item(RAM, 250, product_id="a")]})
    start_scan(db, "manual", sources=[provider])
    provider.results["ram query"] = [item(RAM, 250, product_id="b")]
    run_id = start_scan(db, "manual", sources=[provider])
    assert len(fake_smtp.sent) == 1
    assert db.observations_for_run(run_id)[0]["notify_reason"].startswith("duplicate")


def test_failed_email_is_retried_next_scan(db, one_query, email_on, fake_smtp):
    provider = FakeSource({"ram query": [item(RAM, 250)]})
    fake_smtp.fail_with = OSError("connection refused")
    run_id = start_scan(db, "manual", sources=[provider])
    assert db.get_run(run_id)["email_status"].startswith("failed")
    assert db.list_notifications()[0]["status"] == "failed"

    fake_smtp.fail_with = None
    start_scan(db, "manual", sources=[provider])
    assert len(fake_smtp.sent) == 1


def test_scan_without_email_leaves_matches_for_next_scan(db, one_query, email_on, fake_smtp):
    provider = FakeSource({"ram query": [item(RAM, 250)]})
    run_id = start_scan(db, "manual", send_email=False, sources=[provider])
    assert db.get_run(run_id)["email_status"].startswith("skipped")
    assert not fake_smtp.sent
    start_scan(db, "scheduled", sources=[provider])
    assert len(fake_smtp.sent) == 1


def test_duplicate_results_within_a_scan_count_once(db, configure):
    configure(lambda s: (s["trackers"]["ram"].update(queries=["a", "b"]), s["trackers"]["cpu"].update(enabled=False)))
    provider = FakeSource({"a": [item(RAM, 250)], "b": [item(RAM, 250)]})
    run_id = start_scan(db, "manual", sources=[provider])
    run = db.get_run(run_id)
    assert run["result_count"] == 2 and run["accepted_count"] == 1
    assert provider.calls == ["a", "b"]


def test_query_failures_mark_run_partial_or_failed(db, one_query):
    run_id = start_scan(db, "manual", sources=[FakeSource({"ram query": [item(RAM, 250)]}, fail=["cpu query"])])
    run = db.get_run(run_id)
    assert run["status"] == "partial" and run["query_errors"] == 1
    assert "quota exceeded" in db.queries_for_run(run_id)[1]["error"]

    run_id = start_scan(db, "manual", sources=[FakeSource(fail=["ram query", "cpu query"])])
    assert db.get_run(run_id)["status"] == "failed"


def test_unconfigured_sources_fail_run_with_message(db, monkeypatch):
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    run_id = start_scan(db, "scheduled")
    run = db.get_run(run_id)
    assert run["status"] == "failed"
    assert "SearXNG URL is not set" in run["error"]


def test_broken_source_does_not_stop_the_others(db, one_query, configure, monkeypatch):
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    configure(lambda s: s["sources"]["demo"].update(enabled=True))  # SearXNG stays on but has no URL
    run = db.get_run(start_scan(db, "manual"))
    assert run["status"] == "partial" and run["accepted_count"] > 0
    assert "SearXNG URL is not set" in run["error"]


def test_unexpected_provider_exception_is_contained(db, one_query):
    class Broken(FakeSource):
        def search(self, query, tracker):
            raise RuntimeError("parser exploded")

    run_id = start_scan(db, "manual", sources=[Broken()])
    run = db.get_run(run_id)
    assert run["status"] == "failed"
    assert "RuntimeError: parser exploded" in db.queries_for_run(run_id)[0]["error"]


def test_only_one_scan_at_a_time(db):
    with _scan_lock, pytest.raises(ScanAlreadyRunning):
        start_scan(db, "manual")


def test_scan_running_in_another_process_blocks_new_scan(db):
    db.create_run("cli", "serpapi", True)  # left 'running' by another process
    with pytest.raises(ScanAlreadyRunning, match="already running"):
        start_scan(db, "manual")
    db.mark_interrupted_runs()
    start_scan(db, "manual")


def test_background_scan(db, configure):
    configure(demo_only)
    done = threading.Event()

    class Wrapped(FakeSource):
        def search(self, query, tracker):
            done.wait(5)
            return []

    run_id = start_scan(db, "manual", sources=[Wrapped()], background=True)
    assert db.get_run(run_id)["status"] == "running"
    done.set()
    for thread in threading.enumerate():
        if thread.name == f"scan-{run_id}":
            thread.join(5)
    assert db.get_run(run_id)["status"] == "ok"


def test_replay_uses_current_settings(db, one_query, configure):
    provider = FakeSource({"ram query": [item(RAM, 250)]})
    run_id = start_scan(db, "manual", sources=[provider])
    settings = configure(lambda s: s["trackers"]["ram"].update(max_price_per_stick=200))
    rows = replay_run(db, run_id, settings)
    assert rows[0]["old"]["decision"] == "accepted"
    assert rows[0]["new"]["decision"] == "rejected"
    assert rows[0]["new"]["reasons"][0]["code"] == "price_too_high"


def test_untrusted_merchants_never_email(db, one_query, email_on, fake_smtp):
    scam = item(RAM, 99, source="memorydealz.shop", link="https://memorydealz.shop/p/1")
    run_id = start_scan(db, "manual", sources=[FakeSource({"ram query": [scam]})])
    assert db.get_run(run_id)["accepted_count"] == 0
    assert not fake_smtp.sent


def test_offer_model_text_helpers():
    offer = Offer(title="T", merchant="m", price=1.0, extensions=["DDR4"], snippet="s")
    assert offer.spec_text == "T | DDR4"
    assert offer.full_text == "T | DDR4 | s"


def test_same_ebay_item_from_two_sources_counts_once(db, one_query, email_on, fake_smtp):
    url = "https://www.ebay.com/itm/123456789012"
    api = FakeSource({"ram query": [item(RAM, 250, source="eBay - seller", product_id="v1|123456789012|0", link=url)]},
                     name="ebay_api")  # fmt: skip
    engine = FakeSource({"ram query": [item(RAM, 250, source="eBay", product_id="", link=url + "?hash=x")]},
                        name="searxng:ebay")  # fmt: skip
    run_id = start_scan(db, "manual", sources=[api, engine])
    run = db.get_run(run_id)
    assert run["result_count"] == 2 and run["accepted_count"] == 1 and run["provider"] == "ebay_api, searxng:ebay"
    assert len(fake_smtp.sent) == 1


def test_resend_is_the_default_email_method(db, one_query, configure, resend):
    configure(lambda s: s["email"].update(resend_api_key="re_test_123"))
    run_id = start_scan(db, "manual", sources=[FakeSource({"ram query": [item(RAM, 250)]})])
    assert db.get_run(run_id)["email_status"].startswith("sent 1")
    call = resend.calls[0]
    assert call["headers"]["authorization"] == "Bearer re_test_123"
    assert call["json"]["to"] == ["adaptabyte@gmail.com"]
    assert call["json"]["from"] == "Price Tracker <onboarding@resend.dev>"
    assert "64GB DDR4" in call["json"]["subject"] and "<a href=" in call["json"]["html"]


def test_resend_error_is_recorded_and_retried(db, one_query, configure, resend):
    configure(lambda s: s["email"].update(resend_api_key="re_test_123"))
    resend.status, resend.body = 403, {"message": "You can only send testing emails to your own email address"}
    source = FakeSource({"ram query": [item(RAM, 250)]})
    run_id = start_scan(db, "manual", sources=[source])
    status = db.get_run(run_id)["email_status"]
    assert status.startswith("failed: Resend HTTP 403") and "onboarding@resend.dev" in status
    resend.status, resend.body = 200, {"id": "ok"}
    start_scan(db, "manual", sources=[source])
    assert len(resend.calls) == 2 and db.list_notifications()[0]["status"] == "sent"


def test_query_notes_record_failed_engines(db, one_query):
    class Flaky(FakeSource):
        def search(self, query, tracker):
            self.warnings = ["google: CAPTCHA"]
            return super().search(query, tracker)

    run_id = start_scan(db, "manual", sources=[Flaky({"ram query": [item(RAM, 250)]})])
    assert db.queries_for_run(run_id)[0]["note"] == "Some engines failed: google: CAPTCHA"
