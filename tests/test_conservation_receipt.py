# tests/test_conservation_receipt.py  (WP-5: conservation-closure receipt
# writer, three-term budget, guard inventory, self-consistency tier)
import hashlib
import json
import math

import pytest

from gpuwm.runtime import CONSERVATION_CLOSURE_RECEIPT_NAME
from gpuwm.verify import conservation_closure as cc


def _overhead():
    return {
        "wall_seconds_plain": 12.5,
        "wall_seconds_receipt": 12.9,
        "wall_seconds_per_simulated_minute_delta": 0.004,
        "device_bytes_plain": 1024,
        "device_bytes_receipt": 2048,
    }


def _domain(grid_id=1, *, terms=None, water_value=0.0):
    terms = terms if terms is not None else {
        "lateral_flux_integral": -3.0,
        "lbc_mass_forcing_integral": 1.0,
        "specified_zone_mass_reset": 0.5,
    }
    return cc.build_domain_entry(
        grid_id,
        terms=terms,
        measure_start=100.0, measure_end=98.5,
        mass_residual=cc.residual_entry(
            1.0e-9, tier="observability",
            definition="|dM - sum(terms)| / M0"),
        water_residual=cc.residual_entry(
            water_value, tier="observability",
            definition="|dW - sum(water terms)| / W0"),
        energy_residual=cc.residual_entry(
            2.0e-8, tier="observability",
            definition="|dMSE - sum(MSE terms)| / MSE0"),
        boundary_distance_bands=[{"band": "d0_specified", "cells": 100},
                                 {"band": "interior", "cells": 900}],
        guard_inventory=[
            cc.guard_entry("ysu_nan_guard", "host-side count",
                           counted=True, count=0),
            cc.guard_entry("mynn borrow-from-below", "kernel-side clamp",
                           counted=False,
                           why_not_counted="no host-visible mirror exists"),
        ],
        observer_overhead=_overhead())


def test_receipt_name_is_the_program_wide_one():
    assert CONSERVATION_CLOSURE_RECEIPT_NAME == "conservation-closure.json"


def test_three_terms_are_separate_and_none_defaults_to_zero():
    """A flux-only budget on a forced domain must refuse, not close."""
    full = {"lateral_flux_integral": -3.0,
            "lbc_mass_forcing_integral": 1.0,
            "specified_zone_mass_reset": 0.5}
    residual = cc.mass_budget_residual(100.0, 98.5, full)
    assert residual == pytest.approx(0.0, abs=1e-12)
    for dropped in cc.MASS_BUDGET_TERMS:
        partial = {k: v for k, v in full.items() if k != dropped}
        with pytest.raises(cc.ReceiptError, match="missing"):
            cc.mass_budget_residual(100.0, 98.5, partial)
    with pytest.raises(cc.ReceiptError, match="unknown"):
        cc.mass_budget_residual(100.0, 98.5, dict(full, bogus=1.0))


def test_hardcoded_zero_forcing_is_visible_in_the_residual():
    """The false-closure control: zeroing a real forcing term must move
    the residual, so a receipt cannot pass by structurally omitting it."""
    honest = {"lateral_flux_integral": -3.0,
              "lbc_mass_forcing_integral": 1.0,
              "specified_zone_mass_reset": 0.5}
    faked = dict(honest, lbc_mass_forcing_integral=0.0)
    assert cc.mass_budget_residual(100.0, 98.5, honest) == pytest.approx(0.0)
    assert abs(cc.mass_budget_residual(100.0, 98.5, faked)) == pytest.approx(
        1.0)


def test_relative_residual_refuses_a_zero_measure():
    assert cc.relative_residual(-2.0, 100.0) == pytest.approx(0.02)
    with pytest.raises(cc.ReceiptError):
        cc.relative_residual(1.0, 0.0)


def test_guard_inventory_is_honest_in_both_directions():
    counted = cc.guard_entry("w_damping", "host mirror",
                             counted=True, count=7)
    assert counted["count_or_null"] == 7 and counted["why_not_counted"] is None
    uncounted = cc.guard_entry("pd_renorm", "kernel clamp", counted=False,
                               why_not_counted="not funded this release")
    assert uncounted["count_or_null"] is None
    with pytest.raises(cc.ReceiptError, match="must say why"):
        cc.guard_entry("x", "y", counted=False)
    with pytest.raises(cc.ReceiptError, match="no count"):
        cc.guard_entry("x", "y", counted=True)


def test_residual_entry_will_not_invent_a_bound():
    unpinned = cc.residual_entry(1e-9, tier="observability", definition="d")
    assert unpinned["bound"] is None
    assert unpinned["bound_decision"] is None
    with pytest.raises(cc.ReceiptError, match="gate-tier"):
        cc.residual_entry(1e-9, tier="gate", definition="d")
    with pytest.raises(cc.ReceiptError, match="decision"):
        cc.residual_entry(1e-9, tier="gate", definition="d", bound=1e-8)
    pinned = cc.residual_entry(1e-9, tier="gate", definition="d",
                               bound=1e-8, bound_decision="D-15")
    assert pinned["bound_decision"] == "D-15"
    with pytest.raises(cc.ReceiptError, match="tier"):
        cc.residual_entry(1e-9, tier="probably-fine", definition="d")


def test_domain_entry_requires_every_key_and_the_overhead_measurement():
    entry = _domain()
    assert cc.REQUIRED_DOMAIN_KEYS <= set(entry)
    with pytest.raises(cc.ReceiptError, match="overhead"):
        cc.build_domain_entry(
            1, terms={k: 0.0 for k in cc.MASS_BUDGET_TERMS},
            measure_start=1.0, measure_end=1.0,
            mass_residual={}, water_residual={}, energy_residual={},
            boundary_distance_bands=[], guard_inventory=[],
            observer_overhead={"wall_seconds_plain": 1.0})


def test_receipt_carries_one_entry_per_domain_present_in_the_run():
    payload = cc.build_conservation_receipt(
        [_domain(1), _domain(2), _domain(3), _domain(4)],
        experiment="synthetic-mapped-smoke")
    assert payload["schema"] == cc.SCHEMA_ID
    assert payload["domain_count"] == 4
    assert [d["grid_id"] for d in payload["domains"]] == [1, 2, 3, 4]
    with pytest.raises(cc.ReceiptError, match="at least one domain"):
        cc.build_conservation_receipt([], experiment="empty")
    with pytest.raises(cc.ReceiptError, match="duplicate"):
        cc.build_conservation_receipt([_domain(1), _domain(1)],
                                      experiment="dup")


def test_vs_wrf_tier_is_reserved_with_a_reason_not_omitted():
    payload = cc.build_conservation_receipt([_domain()], experiment="e")
    tier = payload["vs_wrf_tier"]
    assert tier["status"] == "unavailable"
    assert tier["entries"] is None
    assert tier["decision"] == "D-8"
    assert tier["reason"]


def test_self_consistency_section_names_what_it_extends():
    section = cc.build_self_consistency_section(
        lateral_boundary_rows=[
            cc.relaxation_row_entry(0, cells=100, mean_abs_increment=0.0,
                                    max_abs_increment=0.0),
            cc.relaxation_row_entry(1, cells=92, mean_abs_increment=1.5,
                                    max_abs_increment=4.0),
        ],
        nest_edges=[{"child_grid_id": 2, "parent_grid_id": 1,
                     "feedback_applied": True}],
        component_tier_reference="tests/test_lateral_bc.py")
    assert section["component_tier_reference"]
    assert [r["row"] for r in section["lateral_boundary_rows"]] == [0, 1]
    with pytest.raises(cc.ReceiptError, match="contiguous"):
        cc.build_self_consistency_section(
            lateral_boundary_rows=[
                cc.relaxation_row_entry(1, cells=1, mean_abs_increment=0.0,
                                        max_abs_increment=0.0)],
            nest_edges=[], component_tier_reference="x")
    with pytest.raises(cc.ReceiptError, match="component-tier"):
        cc.build_self_consistency_section(
            lateral_boundary_rows=[], nest_edges=[],
            component_tier_reference="")


def test_nan_write_refusal_control(tmp_path):
    """A NaN anywhere in the payload must refuse to encode."""
    good = cc.build_conservation_receipt([_domain()], experiment="e")
    path, digest = cc.write_conservation_receipt(
        tmp_path / CONSERVATION_CLOSURE_RECEIPT_NAME, good)
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    assert json.loads(raw)["schema"] == cc.SCHEMA_ID

    bad = cc.build_conservation_receipt([_domain(water_value=math.nan)],
                                        experiment="e")
    with pytest.raises(cc.ReceiptError, match="non-finite"):
        cc.encode_receipt(bad)
    with pytest.raises(cc.ReceiptError, match="non-finite"):
        cc.write_conservation_receipt(tmp_path / "refused.json", bad)
    assert not (tmp_path / "refused.json").exists()
    assert not list(tmp_path.glob(".refused.json.partial-*"))


def test_write_is_atomic_and_leaves_no_partial(tmp_path):
    payload = cc.build_conservation_receipt([_domain()], experiment="e")
    target = tmp_path / "out" / CONSERVATION_CLOSURE_RECEIPT_NAME
    path, digest = cc.write_conservation_receipt(target, payload)
    assert path == target and path.exists()
    assert not list(target.parent.glob(".*partial*"))
    # rewriting replaces in place and re-digests the exact bytes
    path2, digest2 = cc.write_conservation_receipt(target, payload)
    assert (path2, digest2) == (path, digest)


def test_no_case_token_in_the_receipt_module_or_its_keys():
    import re
    from pathlib import Path
    source = Path(cc.__file__).read_text(encoding="utf-8")
    pattern = re.compile(r"real74|1974|ohio|hrrr", re.IGNORECASE)
    assert not pattern.search(source)
    payload = cc.build_conservation_receipt([_domain()], experiment="e")
    encoded = json.dumps(payload, sort_keys=True)
    assert not pattern.search(encoded)
    # positive control: the grep can see a token when one is present
    assert pattern.search(json.dumps({"case": "real74"}))


# ---- D-29: guard inventory with host-side mirrors and honest gaps ---------

def _countable_sites():
    return [str(entry["site"]) for entry in cc.GUARD_SITES
            if entry["why_not_counted"] is None]


def test_the_guard_inventory_enumerates_every_site_including_uncounted():
    counts = {site: 0 for site in _countable_sites()}
    inventory = cc.default_guard_inventory(counts)
    assert len(inventory) == len(cc.GUARD_SITES)
    counted = [row for row in inventory if row["counted"]]
    uncounted = [row for row in inventory if not row["counted"]]
    assert counted and uncounted, (
        "an inventory with no uncounted rows would be an instrumented "
        "subset presented as the whole clamping surface")
    for row in uncounted:
        assert row["count_or_null"] is None
        assert row["why_not_counted"]


def test_a_countable_site_without_its_count_refuses_rather_than_zero_filling():
    """The named control: silence must not be published as ``0``.

    Reporting an unmeasured guard as zero fires claims the guard never
    fired, which is a strictly stronger statement than "nobody counted".
    """
    partial = {site: 3 for site in _countable_sites()[1:]}
    with pytest.raises(cc.ReceiptError, match="writing zero would claim"):
        cc.default_guard_inventory(partial)


def test_a_count_for_an_uncountable_site_refuses():
    counts = {site: 0 for site in _countable_sites()}
    uncountable = next(str(entry["site"]) for entry in cc.GUARD_SITES
                       if entry["why_not_counted"] is not None)
    counts[uncountable] = 7
    with pytest.raises(cc.ReceiptError, match="does not mark countable"):
        cc.default_guard_inventory(counts)


def test_host_mirrored_sites_name_a_mirror_that_exists():
    """A mirror named in the inventory is imported, not asserted in prose."""
    import importlib

    named = [entry for entry in cc.GUARD_SITES if entry["host_mirror"]]
    assert named, "the mirrored-guard posture must name its mirrors"
    for entry in named:
        module_name, _, attribute = str(entry["host_mirror"]).rpartition(".")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute)), entry["host_mirror"]


def test_a_site_is_either_mirrored_or_says_why_it_is_not_counted():
    for entry in cc.GUARD_SITES:
        if entry["why_not_counted"] is None:
            continue
        assert entry["host_mirror"] is None, (
            "a site with a working host mirror cannot also claim it "
            "cannot be counted")


# ---- D-31: the deferred acoustic/gravity-wave tier ------------------------

def test_the_receipt_marks_the_acoustic_gravity_wave_tier_deferred():
    payload = cc.build_conservation_receipt([_domain()], experiment="e")
    tier = payload["acoustic_gravity_wave_tier"]
    assert tier["status"] == "deferred"
    assert tier["entries"] is None
    assert tier["decision"] == "D-31"
    assert "deferred" in tier["reason"]


def test_deferred_and_unavailable_are_distinct_statuses():
    """Not attempted, and attempted-but-impossible, are different claims."""
    deferred = cc.deferred_tier("not attempted this release", "D-31")
    unavailable = cc.unavailable_tier("the evidence does not exist", "D-8")
    assert deferred["status"] != unavailable["status"]
    assert {deferred["status"], unavailable["status"]} == {
        "deferred", "unavailable"}


def test_a_deferred_tier_without_a_reason_or_decision_refuses():
    with pytest.raises(cc.ReceiptError, match="names its reason"):
        cc.deferred_tier("", "D-31")
    with pytest.raises(cc.ReceiptError, match="names its reason"):
        cc.deferred_tier("some reason", "")


# ---- D-8: the vs-WRF nest tier, reserved by default and measured when fed --

def test_the_vs_wrf_tier_is_reserved_when_no_pair_is_supplied():
    payload = cc.build_conservation_receipt([_domain()], experiment="e")
    tier = payload["vs_wrf_tier"]
    assert tier["status"] == "unavailable"
    assert tier["entries"] is None
    assert tier["decision"] == "D-8"


def test_a_measured_feedback_tier_replaces_the_reserved_one(tmp_path):
    measured = {"schema": "gpuwm.feedback-reference-ab/v1",
                "status": "measured", "decision": "D-8",
                "frames": [], "null_control": {"max_abs_over_all_fields": 0.0}}
    payload = cc.build_conservation_receipt(
        [_domain()], experiment="e", vs_wrf_tier=measured)
    assert payload["vs_wrf_tier"]["status"] == "measured"
    # and it still survives the allow_nan=False encode + atomic write
    path, digest = cc.write_conservation_receipt(
        tmp_path / CONSERVATION_CLOSURE_RECEIPT_NAME, payload)
    reloaded = json.loads(path.read_bytes())
    assert reloaded["vs_wrf_tier"]["status"] == "measured"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
