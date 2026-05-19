"""Tests for viam_visuals.registry — in-process resource registry."""

from viam_visuals import registry


def setup_function(_):
    """Clear registry state between tests — module-global dict."""
    for n in registry.names():
        registry.unregister(n)


def test_register_and_lookup():
    obj = object()
    registry.register("vis", obj)
    assert registry.lookup("vis") is obj


def test_lookup_missing_returns_none():
    assert registry.lookup("nope") is None


def test_register_replaces_prior_value():
    first = object()
    second = object()
    registry.register("vis", first)
    registry.register("vis", second)
    assert registry.lookup("vis") is second


def test_unregister_removes_entry():
    registry.register("vis", object())
    registry.unregister("vis")
    assert registry.lookup("vis") is None


def test_unregister_missing_is_no_op():
    # Doesn't raise.
    registry.unregister("never_registered")


def test_names_returns_sorted_list():
    registry.register("zeta", object())
    registry.register("alpha", object())
    registry.register("mu", object())
    assert registry.names() == ["alpha", "mu", "zeta"]
