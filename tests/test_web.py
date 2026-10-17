import base64
import threading

import pytest
from fastapi.testclient import TestClient

from pricetracker import config
from pricetracker.scanner import start_scan
from pricetracker.settings import load_settings, save_settings
from pricetracker.web import create_app


@pytest.fixture
def app(tmp_path):
    app = create_app(tmp_path / "web.db", start_scheduler=False)
    settings = load_settings(app.state.db)
    settings["provider"]["name"] = "demo"
    save_settings(app.state.db, settings)
    return app


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def settings_form(app, **overrides):
    """The Settings form as the browser would submit it for the stored settings."""
    client = TestClient(app)
    html = client.get("/settings").text
    s = load_settings(app.state.db)
    form = {
        "provider.name": s["provider"]["name"],
        "provider.country": "us",
        "provider.language": "en",
        "provider.location": "",
        "provider.results_per_query": "60",
        "provider.timeout_seconds": "45",
        "schedule.enabled": "on",
        "schedule.times": "08:00, 20:00",
        "schedule.timezone": "America/New_York",
        "email.enabled": "on",
        "email.to": "adaptabyte@gmail.com",
        "email.from_address": "",
        "email.smtp_host": "smtp.gmail.com",
        "email.smtp_port": "587",
        "email.smtp_security": "starttls",
        "email.smtp_username": "",
        "email.price_drop_percent": "5",
        "email.renotify_on_price_drop": "on",
        "general.public_base_url": "",
        "general.retention_days": "90",
        "merchants.trusted": "eBay | ebay | ebay.com\nNewegg | newegg | newegg.com",
        "merchants.blocked": "",
        "trackers.ram.enabled": "on",
        "trackers.ram.label": "RAM",
        "trackers.ram.queries": "64GB DDR4 RDIMM",
        "trackers.ram.module_size_gb": "64",
        "trackers.ram.max_price_per_stick": "300",
        "trackers.ram.min_price_per_stick": "40",
        "trackers.ram.min_speed_mts": "2666",
        "trackers.ram.preferred_speed_mts": "3200",
        "trackers.ram.allowed_types": ["RDIMM", "LRDIMM"],
        "trackers.ram.max_sticks_per_listing": "2",
        "trackers.ram.include_shipping": "on",
        "trackers.ram.reject_untested": "on",
        "trackers.ram.exclude_keywords": "combo, bundle",
        "trackers.cpu.enabled": "on",
        "trackers.cpu.label": "CPU",
        "trackers.cpu.queries": "3975WX",
        "trackers.cpu.model": "3975WX",
        "trackers.cpu.model_aliases": "100-000000086",
        "trackers.cpu.max_price": "500",
        "trackers.cpu.min_price": "150",
        "trackers.cpu.lock_policy": "reject_locked",
        "trackers.cpu.include_shipping": "on",
        "trackers.cpu.exclude_keywords": "",
    }
    assert 'name="trackers.ram.allowed_types"' in html
    form.update(overrides)
    return {k: v for k, v in form.items() if v is not None}


PAGES = [
    "/", "/runs", "/runs/1", "/runs/1?decision=accepted", "/runs/1?tracker=cpu&reason=cpu_vendor_locked",
    "/runs/1/replay", "/runs/1/replay?changed=0", "/listings", "/listings?decision=notified&sort=price&q=64GB",
    "/listings/1", "/observations/1/raw", "/merchants", "/merchants?days=7", "/emails", "/playground", "/settings",
    "/healthz", "/api/status",
]  # fmt: skip


def test_every_page_renders_after_a_scan(app, client):
    start_scan(app.state.db, "manual")
    for path in PAGES:
        response = client.get(path)
        assert response.status_code == 200, (path, response.text[:500])
    dashboard = client.get("/").text
    assert "Current matches" in dashboard and "Not Vendor Locked" in dashboard
    run_page = client.get("/runs/1").text
    assert "Seller not on trusted list" in run_page and "Likely Lenovo PSB-locked" in run_page


def test_pages_render_with_no_data(client):
    for path in ["/", "/runs", "/listings", "/merchants", "/emails", "/settings"]:
        assert client.get(path).status_code == 200


def test_missing_run_is_404(client):
    assert client.get("/runs/999").status_code == 404
    assert client.get("/listings/999").status_code == 404


def test_scan_button_starts_background_scan(app, client):
    response = client.post("/scan", data={"send_email": "on"}, follow_redirects=False)
    assert response.status_code == 303
    run_id = int(response.headers["location"].split("/runs/")[1].split("?")[0])
    for thread in threading.enumerate():
        if thread.name == f"scan-{run_id}":
            thread.join(10)
    assert app.state.db.get_run(run_id)["status"] == "ok"


def test_save_settings(app, client):
    response = client.post("/settings", data=settings_form(app), follow_redirects=False)
    assert response.status_code == 303
    s = load_settings(app.state.db)
    assert s["trackers"]["ram"]["allowed_types"] == ["RDIMM", "LRDIMM"]
    assert s["trackers"]["ram"]["exclude_keywords"] == ["combo", "bundle"]
    assert s["merchants"]["trusted"][1] == {"name": "Newegg", "aliases": ["newegg"], "domains": ["newegg.com"]}
    assert s["trackers"]["ram"]["allow_unknown_speed"] is False


def test_invalid_settings_are_not_saved(app, client):
    response = client.post(
        "/settings", data=settings_form(app, **{"schedule.times": "8am", "trackers.ram.max_price_per_stick": "abc"})
    )
    assert response.status_code == 200
    assert "Use 24-hour HH:MM" in response.text and "Enter a number" in response.text
    assert load_settings(app.state.db)["schedule"]["times"] == ["08:00", "20:00"]


def test_secrets_are_kept_unless_cleared(app, client):
    client.post("/settings", data=settings_form(app, **{"provider.serpapi_api_key": "abc123"}))
    assert load_settings(app.state.db)["provider"]["serpapi_api_key"] == "abc123"
    assert "abc123" not in client.get("/settings").text
    client.post("/settings", data=settings_form(app))
    assert load_settings(app.state.db)["provider"]["serpapi_api_key"] == "abc123"
    client.post("/settings", data=settings_form(app, **{"provider.serpapi_api_key.clear": "on"}))
    assert load_settings(app.state.db)["provider"]["serpapi_api_key"] == ""


def test_env_secret_fallback(app, client, monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "from-env")
    assert "using SERPAPI_API_KEY environment variable" in client.get("/settings").text


def test_trust_and_block_merchants(app, client):
    client.post("/merchants/trust", data={"merchant": "Memory.net", "domain": "memory.net"})
    trusted = load_settings(app.state.db)["merchants"]["trusted"]
    assert trusted[-1] == {"name": "Memory.net", "aliases": ["memory"], "domains": ["memory.net"]}
    client.post("/merchants/block", data={"value": "scam.shop"})
    assert load_settings(app.state.db)["merchants"]["blocked"] == ["scam.shop"]
    client.post("/merchants/unblock", data={"value": "scam.shop"})
    assert load_settings(app.state.db)["merchants"]["blocked"] == []


def test_playground(client):
    response = client.post(
        "/playground",
        data={"tracker": "ram", "title": "Crucial 64GB Kit (2x32GB) DDR4-3200 RDIMM", "merchant": "Newegg.com",
              "price": "$199.99", "shipping": "Free delivery", "link": "", "condition": "", "snippet": ""},
    )  # fmt: skip
    assert response.status_code == 200
    assert "rejected" in response.text and "32GB per stick" in response.text


def test_test_email(app, client, fake_smtp):
    response = client.post("/settings/test-email", follow_redirects=False)
    assert "Cannot+send" in response.headers["location"]
    s = load_settings(app.state.db)
    s["email"].update(smtp_username="me@gmail.com", smtp_password="pw")
    save_settings(app.state.db, s)
    response = client.post("/settings/test-email", follow_redirects=False)
    assert "Test+email+sent" in response.headers["location"]
    assert fake_smtp.sent[0]["Subject"] == "[Price Tracker] Test email"


def test_basic_auth(app, monkeypatch):
    monkeypatch.setattr(config, "UI_PASSWORD", "hunter2")
    with TestClient(app) as c:
        assert c.get("/").status_code == 401
        assert c.get("/healthz").status_code == 200
        token = base64.b64encode(b"admin:hunter2").decode()
        assert c.get("/", headers={"Authorization": f"Basic {token}"}).status_code == 200
        bad = base64.b64encode(b"admin:nope").decode()
        assert c.get("/", headers={"Authorization": f"Basic {bad}"}).status_code == 401


def test_cross_origin_post_rejected(client):
    response = client.post("/scan", headers={"Origin": "https://evil.example"}, follow_redirects=False)
    assert response.status_code == 403


def test_dashboard_keeps_last_good_matches_after_failed_scan(app, client):
    good = start_scan(app.state.db, "manual")
    settings = load_settings(app.state.db)
    settings["provider"]["name"] = "serpapi"  # no API key: the next scan fails
    save_settings(app.state.db, settings)
    start_scan(app.state.db, "manual")
    page = client.get("/").text
    assert f"scan #{good}</a>, the last scan that succeeded" in page
    assert "Not Vendor Locked" in page


def test_unsafe_links_are_dropped(offer):
    listing = offer("x", link="javascript:alert(1)")
    assert listing.link == "" and listing.merchant_url == ""


def test_next_runs_before_scheduler_start(app):
    app.state.scheduler.apply(load_settings(app.state.db))
    assert app.state.scheduler.next_runs() == []
