import threading

import pytest

from pricetracker.models import Offer
from pricetracker.providers import ProviderError, offer_from_serpapi
from pricetracker.scanner import ScanAlreadyRunning, _scan_lock, replay_run, start_scan


class FakeProvider:
    """Returns whatever `results` holds for each query; raises for queries listed in `fail`."""

    name = "fake"

    def __init__(self, results=None, fail=()):
        self.results = results or {}
        self.fail = set(fail)
        self.calls = []

    def describe(self, query):
        return f"fake {query}"

    def search(self, query):
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
    configure(lambda s: s["provider"].update(name="demo"))
    run_id = start_scan(db, "manual")
    run = db.get_run(run_id)
    assert run["status"] == "ok"
    assert run["query_count"] == 6
    assert run["accepted_count"] == 6 and run["rejected_count"] == 21
    assert run["new_match_count"] == 6
    assert run["email_status"].startswith("not sent: SMTP")
    observations = db.observations_for_run(run_id)
    assert len(observations) == 27
    assert all(o["reasons"] for o in observations if o["decision"] == "rejected")
    assert db.list_notifications()[0]["status"] == "skipped"


def test_emails_once_then_only_on_price_drop(db, one_query, email_on, fake_smtp):
    provider = FakeProvider({"ram query": [item(RAM, 250)]})
    run_id = start_scan(db, "manual", provider=provider)
    assert db.get_run(run_id)["email_status"].startswith("sent 1")
    assert len(fake_smtp.sent) == 1
    message = fake_smtp.sent[0]
    assert message["To"] == "adaptabyte@gmail.com"
    assert "64GB DDR4 from $250.00/stick" in message["Subject"]
    assert "https://www.ebay.com/itm/1" in message.get_body(("plain",)).get_content()

    start_scan(db, "manual", provider=provider)
    assert len(fake_smtp.sent) == 1  # same listing, same price: no repeat email

    provider.results["ram query"] = [item(RAM, 240)]  # 4% drop: below the 5% threshold
    start_scan(db, "manual", provider=provider)
    assert len(fake_smtp.sent) == 1

    provider.results["ram query"] = [item(RAM, 200)]
    run_id = start_scan(db, "manual", provider=provider)
    assert len(fake_smtp.sent) == 2
    assert "Price drop" in fake_smtp.sent[1].get_body(("plain",)).get_content()
    assert db.observations_for_run(run_id)[0]["notify_reason"] == "price_drop"


def test_same_seller_and_title_with_new_id_is_not_re_emailed(db, one_query, email_on, fake_smtp):
    provider = FakeProvider({"ram query": [item(RAM, 250, product_id="a")]})
    start_scan(db, "manual", provider=provider)
    provider.results["ram query"] = [item(RAM, 250, product_id="b")]
    run_id = start_scan(db, "manual", provider=provider)
    assert len(fake_smtp.sent) == 1
    assert db.observations_for_run(run_id)[0]["notify_reason"].startswith("duplicate")


def test_failed_email_is_retried_next_scan(db, one_query, email_on, fake_smtp):
    provider = FakeProvider({"ram query": [item(RAM, 250)]})
    fake_smtp.fail_with = OSError("connection refused")
    run_id = start_scan(db, "manual", provider=provider)
    assert db.get_run(run_id)["email_status"].startswith("failed")
    assert db.list_notifications()[0]["status"] == "failed"

    fake_smtp.fail_with = None
    start_scan(db, "manual", provider=provider)
    assert len(fake_smtp.sent) == 1


def test_scan_without_email_leaves_matches_for_next_scan(db, one_query, email_on, fake_smtp):
    provider = FakeProvider({"ram query": [item(RAM, 250)]})
    run_id = start_scan(db, "manual", send_email=False, provider=provider)
    assert db.get_run(run_id)["email_status"].startswith("skipped")
    assert not fake_smtp.sent
    start_scan(db, "scheduled", provider=provider)
    assert len(fake_smtp.sent) == 1


def test_duplicate_results_within_a_scan_count_once(db, configure):
    configure(lambda s: (s["trackers"]["ram"].update(queries=["a", "b"]), s["trackers"]["cpu"].update(enabled=False)))
    provider = FakeProvider({"a": [item(RAM, 250)], "b": [item(RAM, 250)]})
    run_id = start_scan(db, "manual", provider=provider)
    run = db.get_run(run_id)
    assert run["result_count"] == 2 and run["accepted_count"] == 1
    assert provider.calls == ["a", "b"]


def test_query_failures_mark_run_partial_or_failed(db, one_query):
    run_id = start_scan(db, "manual", provider=FakeProvider({"ram query": [item(RAM, 250)]}, fail=["cpu query"]))
    run = db.get_run(run_id)
    assert run["status"] == "partial" and run["query_errors"] == 1
    assert "quota exceeded" in db.queries_for_run(run_id)[1]["error"]

    run_id = start_scan(db, "manual", provider=FakeProvider(fail=["ram query", "cpu query"]))
    assert db.get_run(run_id)["status"] == "failed"


def test_missing_api_key_fails_run_with_message(db):
    run_id = start_scan(db, "scheduled")
    run = db.get_run(run_id)
    assert run["status"] == "failed"
    assert "SerpApi API key is not set" in run["error"]


def test_unexpected_provider_exception_is_contained(db, one_query):
    class Broken(FakeProvider):
        def search(self, query):
            raise RuntimeError("parser exploded")

    run_id = start_scan(db, "manual", provider=Broken())
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
    configure(lambda s: s["provider"].update(name="demo"))
    done = threading.Event()

    class Wrapped(FakeProvider):
        def search(self, query):
            done.wait(5)
            return []

    run_id = start_scan(db, "manual", provider=Wrapped(), background=True)
    assert db.get_run(run_id)["status"] == "running"
    done.set()
    for thread in threading.enumerate():
        if thread.name == f"scan-{run_id}":
            thread.join(5)
    assert db.get_run(run_id)["status"] == "ok"


def test_replay_uses_current_settings(db, one_query, configure):
    provider = FakeProvider({"ram query": [item(RAM, 250)]})
    run_id = start_scan(db, "manual", provider=provider)
    settings = configure(lambda s: s["trackers"]["ram"].update(max_price_per_stick=200))
    rows = replay_run(db, run_id, settings)
    assert rows[0]["old"]["decision"] == "accepted"
    assert rows[0]["new"]["decision"] == "rejected"
    assert rows[0]["new"]["reasons"][0]["code"] == "price_too_high"


def test_untrusted_merchants_never_email(db, one_query, email_on, fake_smtp):
    scam = item(RAM, 99, source="memorydealz.shop", link="https://memorydealz.shop/p/1")
    run_id = start_scan(db, "manual", provider=FakeProvider({"ram query": [scam]}))
    assert db.get_run(run_id)["accepted_count"] == 0
    assert not fake_smtp.sent


def test_offer_model_text_helpers():
    offer = Offer(title="T", merchant="m", price=1.0, extensions=["DDR4"], snippet="s")
    assert offer.spec_text == "T | DDR4"
    assert offer.full_text == "T | DDR4 | s"
