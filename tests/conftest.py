import copy
import json
import smtplib
import typing

import httpx
import pytest

from pricetracker import notifier
from pricetracker.db import Database
from pricetracker.models import Offer
from pricetracker.settings import DEFAULT_SETTINGS, load_settings, save_settings


@pytest.fixture
def settings():
    return copy.deepcopy(DEFAULT_SETTINGS)


def make_offer(title, merchant="eBay - good_seller", price=150.0, link="https://www.ebay.com/itm/1", **kwargs):
    return Offer(title=title, merchant=merchant, price=price, link=link, merchant_url=link, **kwargs)


@pytest.fixture
def offer():
    return make_offer


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


@pytest.fixture
def configure(db):
    """Update stored settings: configure(lambda s: s['provider'].update(name='demo'))."""

    def apply(change):
        current = load_settings(db)
        change(current)
        save_settings(db, current)
        return current

    return apply


class FakeSMTP:
    sent: typing.ClassVar[list] = []
    fail_with: Exception | None = None

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context=None):
        pass

    def login(self, username, password):
        pass

    def send_message(self, msg):
        if FakeSMTP.fail_with:
            raise FakeSMTP.fail_with
        FakeSMTP.sent.append(msg)


@pytest.fixture
def fake_smtp(monkeypatch):
    FakeSMTP.sent = []
    FakeSMTP.fail_with = None
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    return FakeSMTP


@pytest.fixture
def email_on(configure):
    """Email over SMTP (pair with the fake_smtp fixture)."""

    def change(s):
        s["email"].update(method="smtp", smtp_username="tracker@example.com", smtp_password="app-password")

    return configure(change)


@pytest.fixture
def resend(monkeypatch):
    """Capture Resend API calls; set `resend.status` / `resend.body` to simulate errors."""

    class Resend:
        calls: typing.ClassVar[list] = []
        status = 200
        body: typing.ClassVar[dict] = {"id": "email_123"}

    Resend.calls = []

    def handler(request):
        Resend.calls.append({"headers": dict(request.headers), "json": json.loads(request.content)})
        return httpx.Response(Resend.status, json=Resend.body)

    monkeypatch.setattr(notifier, "http_transport", httpx.MockTransport(handler))
    return Resend


def demo_only(settings: dict) -> None:
    """Use only the demo source (no network)."""
    for cfg in settings["sources"].values():
        cfg["enabled"] = False
    settings["sources"]["demo"]["enabled"] = True
