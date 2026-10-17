import pytest
from conftest import make_offer

from pricetracker.rules import evaluate
from pricetracker.rules.cpu import lock_status

BASE = "AMD Ryzen Threadripper PRO 3975WX 32-Core sWRX8 Processor"


def run(settings, title, price=450.0, **kwargs):
    return evaluate("cpu", make_offer(title, price=price, **kwargs), settings)


def codes(ev):
    return {r.code for r in ev.reasons}


@pytest.mark.parametrize(
    ("title", "lock"),
    [
        (BASE, "not stated"),
        (BASE + " - Not Vendor Locked", "listing says not vendor locked"),
        (BASE + " non-vendor-locked", "listing says not vendor locked"),
        (BASE + " (no vendor lock)", "listing says not vendor locked"),
        (BASE + " not PSB locked", "listing says not vendor locked"),
        (BASE + " vendor unlocked", "listing says not vendor locked"),
        (BASE + " Lenovo P620 pull - not vendor locked", "Lenovo pull, seller says unlocked"),
        ("Brand New Sealed " + BASE, "probably unlocked"),
        (BASE + " unlocked", "probably unlocked"),
        (BASE + " tested working, no bent pins", "not stated"),
        (BASE + " CPU only, no cooler", "not stated"),
        (BASE + " compatible with WRX80 motherboard", "not stated"),
        ("AMD Threadripper PRO 3975 WX", "not stated"),
        ("AMD 100-100000086WOF Threadripper PRO 32-core", "not stated"),
    ],
)
def test_accepted_cpus(settings, title, lock):
    ev = run(settings, title)
    assert ev.accepted, ev.reasons
    assert ev.attributes["vendor_lock"] == lock


@pytest.mark.parametrize(
    ("title", "code"),
    [
        (BASE + " vendor locked", "cpu_vendor_locked"),
        (BASE + " Lenovo Locked", "cpu_vendor_locked"),
        (BASE + " locked to Dell", "cpu_vendor_locked"),
        (BASE + " PSB", "cpu_vendor_locked"),
        (BASE + " locked", "cpu_vendor_locked"),
        (BASE + " from Lenovo P620", "cpu_lenovo_risk"),
        (BASE + " ThinkStation pull", "cpu_lenovo_risk"),
        (BASE + " for parts", "condition_defect"),
        (BASE + " bent pins", "condition_defect"),
        (BASE + " not working", "condition_defect"),
        (BASE + " no POST", "condition_defect"),
        (BASE + " damaged", "condition_defect"),
        (BASE + " untested", "condition_untested"),
        ("AMD Ryzen Threadripper PRO 5975WX", "cpu_model"),
        ("AMD Ryzen Threadripper 3970X", "cpu_model"),
        (BASE + " / 3995WX", "cpu_other_model"),
        ("Noctua NH-U14S TR4-SP3 cooler for 3975WX", "cpu_accessory"),
        ("Empty box for AMD 3975WX", "cpu_accessory"),
        (BASE + " + ASUS Pro WS WRX80E-SAGE motherboard", "cpu_bundle"),
        (BASE + " with Supermicro M12SWA motherboard", "cpu_bundle"),
        (BASE + " combo", "cpu_bundle"),
        (BASE + " with 256GB RAM", "cpu_bundle"),
    ],
)
def test_rejected_cpus(settings, title, code):
    ev = run(settings, title)
    assert code in codes(ev), ev.reasons


def test_lock_mention_in_snippet_rejects(settings):
    ev = run(settings, BASE, snippet="Pulled from a working system. Lenovo locked.")
    assert "cpu_vendor_locked" in codes(ev)


def test_strict_lock_policy(settings):
    settings["trackers"]["cpu"]["lock_policy"] = "require_unlocked"
    assert "cpu_lock_unknown" in codes(run(settings, BASE))
    assert run(settings, BASE + " not vendor locked").accepted
    assert run(settings, "New sealed retail box " + BASE).accepted


def test_condition_field_for_parts(settings):
    assert "condition_defect" in codes(run(settings, BASE, condition="For parts or not working"))


def test_price_limits(settings):
    assert run(settings, BASE, price=499.99).accepted
    assert "price_too_high" in codes(run(settings, BASE, price=500))
    assert "price_too_high" in codes(run(settings, BASE, price=495, shipping=10.0))
    assert "price_suspicious" in codes(run(settings, BASE, price=99))


def test_model_is_configurable(settings):
    settings["trackers"]["cpu"]["model"] = "5975WX"
    settings["trackers"]["cpu"]["model_aliases"] = []
    assert run(settings, "AMD Ryzen Threadripper PRO 5975WX").accepted
    assert "cpu_model" in codes(run(settings, BASE))


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("3975wx not vendor locked", "explicitly_unlocked"),
        ("3975wx isn't locked", "explicitly_unlocked"),
        ("3975wx no psb", "explicitly_unlocked"),
        ("3975wx psb disabled", "explicitly_unlocked"),
        ("3975wx not vendor locked (psb)", "explicitly_unlocked"),
        ("3975wx vendor locked", "locked"),
        ("3975wx clock 3.5ghz", "unknown"),
        ("3975wx lenovo", "lenovo_risk"),
    ],
)
def test_lock_status(text, status):
    assert lock_status(text)[0] == status
