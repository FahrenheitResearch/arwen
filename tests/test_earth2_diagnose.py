"""gpuwm earth2-diagnose resolves Earth-2 consumers from the table, never by code."""
import json

import pytest

from gpuwm import earth2_diagnose as d


def test_every_diagnostic_row_resolves_and_names_its_class():
    rows = d.consumer_rows()
    for key in ("precip-afno", "climatenet", "tc-tracker", "derived-ws10m"):
        licence = rows[key].get("weights", {}).get("licence")
        if rows[key].get("licence_class") != "permissive":
            with pytest.raises(ValueError, match="--accept-licence"):
                d.resolve(key)
        assert key in d.resolve(key, accept_licence=[licence])
        assert rows[key]["earth2studio_class"]
        assert rows[key]["input_channels"]


def test_unknown_row_is_refused_by_name():
    with pytest.raises(ValueError, match="not a consumers row"):
        d.resolve("no-such-model")


def test_cli_registers():
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    d.register_cli(sub)
    a = p.parse_args(["earth2-diagnose", "x.zarr", "--for", "climatenet", "--out", "o"])
    assert a.consumers == "climatenet" and a.func is d.main


def test_restricted_consumer_cannot_bypass_diagnostic_authority():
    with pytest.raises(ValueError, match="NVIDIA Community Model License"):
        d.resolve("windgust-afno")
