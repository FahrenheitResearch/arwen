"""CPU-only WDM6 (mp_physics=16) integration and coefficient tests.

The device half is ``tests/test_wdm6_column.py``.  Nothing here compares
against WRF's own ``module_mp_wdm6.F``: no oracle exists for this scheme
yet, and the registry option says so.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.config import RunConfig, load_config, validate_run_config
from gpuwm.core import preflight
from gpuwm.core.kf import KFPhaseMode, kf_phase_mode_for_microphysics
from gpuwm.core.physics import (microphysics_scheme_sr_available,
                                microphysics_scratch_slots)
from gpuwm.core.wdm6_constants import (WDM6_NUMBER_SPECIES, coefficients,
                                       rimed_ice_constants)
from gpuwm.io.restart import MICROPHYSICS_ALGORITHM_IDENTITIES


_TINY = dict(nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
             dt=1.0, run_seconds=10.0)
_KERNEL = Path(__file__).resolve().parents[1] / "gpuwm/core/kernels/wdm6.cu"


def _cfg(**changes):
    return RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=16,
                     **changes)


# --------------------------------------------------------------------------
# Admission and refusal
# --------------------------------------------------------------------------

def test_config_accepts_wdm6_and_validates_its_two_knobs(tmp_path):
    path = tmp_path / "wdm6.toml"
    path.write_text(
        "[grid]\n"
        "nx=8\nny=6\nnz=4\ndx=1000.0\ndy=1000.0\nztop=10000.0\n"
        "[dynamics]\ndt=1.0\nmoist=true\nmoist_cq=true\nmp_physics=16\n"
        "wdm6_hail_opt=1\nwdm6_ccn_conc=5.0e8\n"
        "[run]\nrun_seconds=10.0\n",
        encoding="utf-8")
    cfg = load_config(path)
    assert cfg.mp_physics == 16
    assert cfg.wdm6_hail_opt == 1
    assert cfg.wdm6_ccn_conc == 5.0e8
    with pytest.raises(ValueError, match="wdm6_hail_opt"):
        validate_run_config(replace(cfg, wdm6_hail_opt=2))
    # WDM6's own clamp (module_mp_wdm6.F:585) would discard anything
    # outside [1e8, 2e10] on the first minor loop, so config refuses it.
    with pytest.raises(ValueError, match="1e8, 2e10"):
        validate_run_config(replace(cfg, wdm6_ccn_conc=1.0e6))
    with pytest.raises(ValueError, match="1e8, 2e10"):
        validate_run_config(replace(cfg, wdm6_ccn_conc=1.0e11))


@pytest.mark.parametrize(("value", "name"), [(14, "WDM5"), (26, "WDM7")])
def test_the_out_of_scope_wdm_siblings_are_refused_by_name(value, name):
    """Not swept into the generic tail: 14 and 26 are what a WDM6 user
    types next, and WDM6 cannot stand in for either hydrometeor set."""
    with pytest.raises(ValueError, match=name):
        validate_run_config(RunConfig(**_TINY, moist=True, moist_cq=True,
                                      mp_physics=value))


def test_wdm6_requires_moisture_like_every_other_scheme():
    with pytest.raises(ValueError, match="requires moist=true"):
        validate_run_config(RunConfig(**_TINY, moist=False, mp_physics=16))


@pytest.mark.parametrize(("field", "value"),
                         [("wdm6_hail_opt", 1), ("wdm6_ccn_conc", 5.0e8)])
def test_a_wdm6_knob_under_another_scheme_is_refused_not_ignored(field, value):
    """Nothing outside the mp=16 path reads either field, so a set value
    under another scheme is a request gpuwm would silently drop."""
    cfg = RunConfig(**_TINY, moist=True, moist_cq=True, mp_physics=6,
                    **{field: value})
    with pytest.raises(ValueError, match=f"{field} is a WDM6 selector"):
        validate_run_config(cfg)


def test_the_wdm6_knobs_leave_the_identity_of_every_non_wdm6_run():
    """Adding the pair moved no pre-WDM6 fingerprint, and that is enforced.

    ``restart_identity_payload`` drops a scheme-scoped field from the
    identity of every domain that does not select its scheme -- the
    absent-stays-absent convention ``perturbation`` and per-domain
    ``spawn`` already use.  Two things are checked here, because only the
    pair of them is the property: the keys are ABSENT for a non-WDM6
    domain (so every experiment written before this lane hashes exactly as
    it did), and they are PRESENT AND BOUND for a WDM6 one (so a WDM6
    checkpoint cannot resume under a different CCN initial condition or a
    different rimed-ice mode).
    """
    from gpuwm.core.model import (SCHEME_SCOPED_RUN_FIELDS,
                                  restart_identity_payload)
    from gpuwm.verify.cases.nest_ideal_r1_moist import load_scaffold

    assert SCHEME_SCOPED_RUN_FIELDS[16] == ("wdm6_hail_opt", "wdm6_ccn_conc")
    scaffold = load_scaffold()

    def payload(run):
        exp = replace(scaffold, domains=(
            replace(scaffold.domains[0], run=run),))
        return restart_identity_payload(exp)["domains"][0]["run"]

    wsm6 = payload(RunConfig(**_TINY, moist=True, moist_cq=True,
                             mp_physics=6))
    for name in SCHEME_SCOPED_RUN_FIELDS[16]:
        assert name not in wsm6, name

    wdm6 = payload(_cfg(wdm6_hail_opt=1, wdm6_ccn_conc=5.0e8))
    assert wdm6["wdm6_hail_opt"] == 1
    assert wdm6["wdm6_ccn_conc"] == 5.0e8


# --------------------------------------------------------------------------
# wdm6init coefficients, and the FP32 literals the kernel bakes from them
# --------------------------------------------------------------------------

def test_wdm6init_reuses_wsm6_rimed_ice_rather_than_copying_it():
    """wdm6init's hail_opt arm (:2096-2108) sets the same five constants
    mp_wsm6_init does, so the port SHARES the host function."""
    for hail_opt in (0, 1):
        assert coefficients(hail_opt).rimed == rimed_ice_constants(hail_opt)
    assert coefficients(0).rimed.n0g == 4.0e6
    assert coefficients(0).rimed.deng == 500.0
    assert coefficients(1).rimed.n0g == 4.0e4
    assert coefficients(1).rimed.deng == 700.0


def test_wdm6_double_moment_coefficients_follow_the_fortran():
    c = coefficients(0)
    pi = 4.0 * math.atan(1.0)
    # :2143 pidnr = 4*pi*denr -- the prognostic rain intercept's cube-root
    # denominator, replacing WSM6's fixed-N0r fourth root.
    assert c.pidnr == pytest.approx(4.0 * pi * 1000.0, rel=1e-15)
    # :2116 pidnc = pi*denr/6, the cloud droplet slope.
    assert c.pidnc == pytest.approx(pi * 1000.0 / 6.0, rel=1e-15)
    # :2136/:2137 the mass- and number-weighted rain fall speeds.  They are
    # DIFFERENT numbers; a port that used one for both would sediment rain
    # number at the mass speed and drift the mean drop size every step.
    #
    # The reference is WRF's OWN rgmma, not math.gamma.  rgmma truncates the
    # Weierstrass product at 10000 terms, whose error is about x**2/(2N) --
    # 0.17 per cent at x = 5.8, which is far above any float32 tolerance.
    # Comparing against the true Gamma here would fail, and "fixing" the
    # port to match the true Gamma would be a divergence from WRF, so the
    # truncation is pinned rather than papered over.
    from gpuwm.core.wsm6_constants import _rgmma

    assert c.pvtr != c.pvtrn
    assert c.pvtr == pytest.approx(841.9 * _rgmma(5.8) / 24.0, rel=1e-13)
    assert c.pvtrn == pytest.approx(841.9 * _rgmma(2.8), rel=1e-13)
    assert abs(c.pvtr / (841.9 * math.gamma(5.8) / 24.0) - 1.0) < 3.0e-3
    # :2113/:2114 the maritime and continental autoconversion thresholds
    # differ by exactly the xncr0/xncr1 ratio.
    assert c.qc1 / c.qc0 == pytest.approx(10.0, rel=1e-12)
    # :2140-2141 rain evaporation carries NO n0r factor; nr multiplies at
    # the call site.
    assert c.precr1 == pytest.approx(2.0 * pi * 1.56, rel=1e-15)


@pytest.mark.parametrize(
    ("literal", "value"),
    [
        ("1.25663706e4", lambda c: c.pidnr),
        ("5.23598776e2", lambda c: c.pidnc),
        ("2.99849272e3", lambda c: c.pvtr),
        ("1.41088450e3", lambda c: c.pvtrn),
        ("8.37758041e-5", lambda c: c.qc0),
        ("8.37758041e-4", lambda c: c.qc1),
        ("2.00517852e1", lambda c: c.pvts),
        ("6.28318531e8", lambda c: c.pidn0s),
    ],
)
def test_the_kernel_s_baked_fp32_literals_match_the_float64_wdm6init(
        literal, value):
    """The kernel bakes wdm6init's products as FP32 literals.  This is the
    pin that keeps the two spellings from drifting apart silently."""
    source = _KERNEL.read_text(encoding="utf-8")
    assert literal + "f" in source, f"{literal}f is not in wdm6.cu"
    expected = value(coefficients(0))
    assert float(literal) == pytest.approx(expected, rel=1.0e-8)


def test_the_kernel_source_states_it_has_no_oracle():
    """HONESTY PIN.  If someone runs the oracle campaign and the header
    keeps saying there is none, or removes the sentence without running
    it, this test is the thing that notices."""
    source = _KERNEL.read_text(encoding="utf-8")
    assert "implemented-unverified" in source
    assert re.search(r"no oracle comparison against\s+// the WRF Fortran",
                     source) or "no oracle comparison" in source


# --------------------------------------------------------------------------
# State, preflight and driver seams
# --------------------------------------------------------------------------

def test_wdm6_state_carries_three_number_moments_and_the_ccn_fill(
        monkeypatch):
    import gpuwm.core.state as state_module

    monkeypatch.setattr(state_module, "cp", np)
    cfg = _cfg(wdm6_ccn_conc=3.0e8)
    state = state_module.DomainState(cfg)
    for name in ("qi", "qs", "qg", "qi0", "qs0", "qg0"):
        assert getattr(state, name).shape == (4, 6, 8)
    for name in WDM6_NUMBER_SPECIES:
        assert getattr(state, name).shape == (4, 6, 8)
        assert getattr(state, name + "0").shape == (4, 6, 8)
    # module_mp_wdm6.F:220-227: the CCN reservoir starts at ccn_conc, the
    # droplet and rain numbers at WRF's allocator zero.
    np.testing.assert_array_equal(state.nn, np.float32(3.0e8))
    np.testing.assert_array_equal(state.nc, np.float32(0.0))
    np.testing.assert_array_equal(state.nr, np.float32(0.0))
    # Not Morrison's set, and not Thompson's.
    for name in ("ni", "ns", "ng", "effr", "nwfa", "nifa", "qnn"):
        assert getattr(state, name, None) is None
    np.testing.assert_array_equal(state.effc, np.float32(2.49))
    np.testing.assert_array_equal(state.effi, np.float32(4.99))
    np.testing.assert_array_equal(state.effs, np.float32(9.99))


def test_wdm6_preflight_prices_its_state_scratch_and_kernel():
    cfg = _cfg(km_opt=4, bl_pbl_physics=1)
    shapes = preflight.state_array_shapes(cfg)
    assert {"qi", "qs", "qg", "effc", "effi", "effs"} <= shapes.keys()
    assert set(WDM6_NUMBER_SPECIES) <= shapes.keys()
    assert {n + "0" for n in WDM6_NUMBER_SPECIES} <= shapes.keys()

    scratch = preflight.scratch_slot_registry(cfg)
    for name in ("wdm6_theta", "wdm6_rho", "wdm6_pii", "wdm6_dz",
                 "wdm6_z8w", "mp_rainnc", "mp_snownc", "mp_graupelnc",
                 "mp_sr", "refl_t", "refl_10cm"):
        assert name in scratch, name
    # mp=16 never loads the WSM6 translation unit.
    assert "wsm6_theta" not in scratch
    assert preflight._MICROPHYSICS_KERNEL_MODULES[16] == (
        "wdm6", "microphysics_validation")
    assert 16 in preflight._REFLECTIVITY_MICROPHYSICS


def test_wdm6_reflectivity_is_priced_from_its_own_translation_unit():
    """refl.cu is byte-frozen, so WDM6's reflectivity kernel lives in
    wdm6_refl.cu -- and the rail must reserve THAT frame, not refl.cu's."""
    from gpuwm.experiment import ExperimentConfig  # noqa: F401  (shape doc)

    spec = preflight.LEVEL_SPECIALIZED_KERNEL_FRAMES["wdm6_refl"]
    assert spec.define == "REFL_KMAX"
    assert preflight.KERNEL_MAX_LOCAL_SIZE_BYTES["wdm6_refl"] == 16128
    # Driver-measured rows (RTX 5090); tests/test_wdm6_column.py re-measures
    # them against the real kernel.
    for levels, frame in ((256, 16128), (128, 8064), (64, 4032),
                          (49, 3088), (30, 1904), (10, 640)):
        assert spec.frame_bytes(levels) == frame, levels


def test_the_wdm6_refl_prologue_is_byte_identical_to_refl_cu_s():
    """wdm6_refl.cu copies refl.cu's device prologue because refl.cu may
    not be appended to.  Copied code that nobody checks is code that
    drifts, so the copy is pinned to the original here."""
    root = Path(__file__).resolve().parents[1] / "gpuwm/core/kernels"
    refl = (root / "refl.cu").read_text(encoding="utf-8")
    wdm6_refl = (root / "wdm6_refl.cu").read_text(encoding="utf-8")
    start = refl.index("// Compile-time bound")
    end = refl.index('extern "C" __global__ void refl10cm_morrison_column')
    prologue = refl[start:end].rstrip()
    assert prologue in wdm6_refl, (
        "wdm6_refl.cu's copy of refl.cu's device prologue has drifted; the "
        "two translation units must agree on the Blahak backscatter and "
        "the radar constants")


def test_wdm6_scratch_lifetime_audit_is_closed_world():
    slots = preflight.scratch_slot_registry(
        _cfg(km_opt=4, diff_6th_opt=2, specified=True), n_lbc_intervals=2)
    assert slots
    assert {slot for slot in slots
            if preflight.scratch_slot_lifetime(slot) is None} == set()
    for slot in ("wdm6_theta", "wdm6_rho", "wdm6_pii", "wdm6_dz",
                 "wdm6_z8w"):
        row = preflight.scratch_slot_lifetime(slot)
        assert row is not None and row.kind == "write_before_read"
        assert preflight.scratch_slot_uses_arena(slot)
    for slot in ("mp_rainnc", "mp_sr", "mp_graupelncv"):
        row = preflight.scratch_slot_lifetime(slot)
        assert row is not None and row.kind == "carrying"
        assert not preflight.scratch_slot_uses_arena(slot)


def test_wdm6_driver_seams_treat_it_as_a_full_ice_scheme():
    assert microphysics_scheme_sr_available(16) is True
    assert dict(microphysics_scratch_slots(16)) == {
        "rainnc": "mp_rainnc", "rainncv": "mp_rainncv", "sr": "mp_sr",
        "snownc": "mp_snownc", "snowncv": "mp_snowncv",
        "graupelnc": "mp_graupelnc", "graupelncv": "mp_graupelncv",
    }
    assert kf_phase_mode_for_microphysics(16) == \
        KFPhaseMode.SEPARATE_ICE_SNOW
    assert MICROPHYSICS_ALGORITHM_IDENTITIES[16].startswith(
        "wdm6-double-moment-warm-rain-wrf-v4.6.1-v1")


def test_the_ring_guard_captures_wdm6_s_number_moments():
    """Rain that fully evaporates returns its number to the CCN reservoir
    (:1249-1252), so nn is WRITTEN and an unguarded specified-zone ring
    would keep the change."""
    from gpuwm.core.microphysics import spec_zone_ring_save_slots

    slots = spec_zone_ring_save_slots(_cfg(specified=True, spec_zone=2))
    for name in WDM6_NUMBER_SPECIES:
        assert any(key.startswith(f"mp_ring_save_{name}_")
                   for key in slots), name


def test_wdm6_microphysics_dispatch_is_lazy_and_forwards_diagflag(
        monkeypatch):
    import gpuwm.core.microphysics as dispatcher
    import gpuwm.core.wdm6 as wdm6

    sentinel = object()

    def fake_apply(state, cfg, dt, *, refl_10cm_due=False):
        assert cfg.mp_physics == 16
        assert dt == 7.5
        assert refl_10cm_due is True
        return sentinel

    monkeypatch.setattr(wdm6, "apply", fake_apply)
    state = SimpleNamespace(qv=object())
    assert dispatcher.apply(state, _cfg(), 7.5,
                            refl_10cm_due=True) is sentinel


def test_wdm6_vertical_kernel_tiers_cover_80_without_enlarging_z49():
    from gpuwm.core.wdm6 import _kernel_capacity

    assert _kernel_capacity(49) == 64
    assert _kernel_capacity(64) == 64
    assert _kernel_capacity(65) == 80
    assert _kernel_capacity(80) == 80
    with pytest.raises(ValueError, match="2 <= nz <= 80"):
        _kernel_capacity(81)


def test_the_adapter_refuses_a_state_with_no_land_mask():
    from gpuwm.core.wdm6 import column_land_mask

    for driver in (None, SimpleNamespace(fields={}),
                   SimpleNamespace(fields={"xland": None})):
        with pytest.raises(ValueError, match="XLAND"):
            column_land_mask(SimpleNamespace(physics=driver))


# --------------------------------------------------------------------------
# Reflectivity host products
# --------------------------------------------------------------------------

def test_radar_init_wdm6_moves_only_the_rain_moments_to_mu_1():
    """wdm6init sets xmu_r = 1 and leaves snow/graupel at 0 (:2198-2206)."""
    from gpuwm.core.refl import radar_init, radar_init_wdm6

    base = radar_init(0)
    for hail_opt in (0, 1):
        rc = radar_init_wdm6(hail_opt)
        assert rc.xcre == (4.0, 2.0, 5.0, 8.0)
        assert rc.xcrg[2] == pytest.approx(24.0, rel=1e-13)
        assert rc.xcrg[3] == pytest.approx(5040.0, rel=1e-13)
        assert rc.xorg2 == pytest.approx(1.0, rel=1e-13)
        # Snow and graupel keep the mu = 0 tuples verbatim.
        assert rc.xcse == base.xcse
        assert rc.xcge == base.xcge
        # Rain density is WSM6's 1000, not Morrison's 997.
        assert rc.xam_r == pytest.approx(1000.0 * math.pi / 6.0, rel=1e-13)
        assert rc.xam_g == pytest.approx(
            rimed_ice_constants(hail_opt).deng * math.pi / 6.0, rel=1e-13)


# --------------------------------------------------------------------------
# Namelist importer and registry
# --------------------------------------------------------------------------

def test_a_namelist_naming_wdm6_imports_natively(tmp_path):
    """CROSS-LANE FIXTURE: the WRF namelist pair comes from the importer's
    own suite, so this test cannot pass against a namelist WDM6 invented
    for itself."""
    from test_namelist_import import _generic_hierarchy_pair

    from gpuwm.namelist_import import import_namelists

    wps, inp = _generic_hierarchy_pair(tmp_path, 1)
    text = inp.read_text(encoding="utf-8")
    assert " mp_physics = 6,\n" in text
    text = text.replace(
        " mp_physics = 6,\n",
        " mp_physics = 16,\n hail_opt = 1,\n ccn_conc = 5.0E8,\n")
    inp.write_text(text, encoding="utf-8")

    toml_text, report = import_namelists(wps, inp, name="wdm6-import")
    assert "mp_physics = 16" in toml_text
    assert "wdm6_hail_opt = 1" in toml_text
    assert "wdm6_ccn_conc = 500000000" in toml_text.replace(".0", "")
    # A TRANSLATION, not a substitution: nothing was swapped for WDM6.
    assert not any(sub.key == "mp_physics"
                   for sub in report.substitutions)


def test_the_registry_row_is_honest_about_having_no_oracle():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads(
        (root / "gpuwm/physics_registry_v2.json").read_text(
            encoding="utf-8"))
    option = registry["components"]["microphysics"]["options"]["wdm6-mp16"]
    assert option["implemented"] is True
    assert option["maturity"] == "implemented-unverified"
    assert option["reachability"]["state"] == "component-override"
    assert option["selectors"] == {"mp_physics": 16}
    assert option["scientific_evidence"] == "none"
    first = option["warnings"][0]
    assert first.startswith(
        "NO ORACLE COMPARISON AGAINST THE WRF FORTRAN HAS BEEN RUN")
    assert "What is NOT established" in first
    # The label may not creep without the measurement that would earn it.
    # The sentence naming Shin-Hong's and Grell-Freitas's numbers is where
    # the row says those campaigns are the MODEL for the next stage, so the
    # forbidden words are checked against the claim clause, not the whole
    # warning, which would only teach the next author to delete the model.
    claim, _, _future = first.partition(
        "The oracle campaign that produced")
    for forbidden in ("max_ulp", "bitwise", "model-validated",
                      "validation-candidate", "ULP parity"):
        assert forbidden not in claim, forbidden
    # No template registers mp=16, which is what component-override means.
    for template in registry["templates"].values():
        assert template.get("components", {}).get("microphysics") != \
            "wdm6-mp16"


# --------------------------------------------------------------------------
# The mixed nest edge, and its offline mirror
# --------------------------------------------------------------------------

def test_a_mixed_nest_edge_touching_wdm6_names_wdm6_and_not_thompson():
    """The refusal mp=16 advertises must actually be a refusal.

    Regression: 16 was added to ``UNVALIDATED_MIXED_EDGE_SELECTORS`` and to
    no other table, so ``resolve_microphysics_transition`` looked up a
    missing moments row and raised a bare ``KeyError(16)`` -- while
    ``gpuwm/physics_compat.py`` and the registry warning both advertised the
    named refusal as working.  The message body was also hard-coded
    Thompson-mp28 prose, so the first user to get past the KeyError would
    have been told their scheme is Thompson and pointed at Thompson's
    fallback constants.
    """
    from gpuwm.core import microphysics_transition as mt

    for source, target in ((16, 6), (6, 16), (16, 28), (28, 16)):
        parent = SimpleNamespace(mp_physics=source)
        child = SimpleNamespace(
            mp_physics=target,
            nest_microphysics_transition=mt.EDGE_MATRIX_POLICY)
        with pytest.raises(ValueError) as caught:
            mt.resolve_microphysics_transition(parent, child)
        message = str(caught.value)
        assert f"MP{source}->MP{target} is REFUSED" in message
        # The refusal names the LOWEST refused selector in the pair, and for
        # every pair above that is 16.
        assert "MP16 (WDM6" in message, message
        # WDM6's own moments and WDM6's own Fortran, not Thompson's.
        assert "(nr, nc, nn)" in message
        assert "module_mp_wdm6.F:584" in message
        assert "start_em.F:1750-1774" in message
        assert "module_mp_thompson.F" not in message
        assert "nwfa" not in message and "nifa" not in message

    # And mp=28's own refusal did not move.
    parent = SimpleNamespace(mp_physics=28)
    child = SimpleNamespace(mp_physics=6,
                            nest_microphysics_transition=mt.EDGE_MATRIX_POLICY)
    with pytest.raises(ValueError) as caught:
        mt.resolve_microphysics_transition(parent, child)
    assert "MP28 (Thompson aerosol-aware)" in str(caught.value)
    assert "module_mp_thompson.F:1248-1255" in str(caught.value)


def test_same_scheme_wdm6_nesting_is_unaffected_by_the_refusal():
    """The refusal is about MIXED edges only."""
    from gpuwm.core import microphysics_transition as mt

    cfg = SimpleNamespace(
        mp_physics=16,
        nest_microphysics_transition=mt.SAME_SCHEME_POLICY)
    contract = mt.resolve_microphysics_transition(cfg, cfg)
    assert contract.mixed is False
    assert contract.source_mp_physics == contract.target_mp_physics == 16


def test_a_refused_selector_cannot_join_without_its_own_sentence():
    """Import-time guard, exercised rather than trusted.

    The bare KeyError was reachable because the three tables describing a
    refused selector are independent.  They are now cross-checked at import;
    this drives that check with a selector that has no row.
    """
    from gpuwm.core import microphysics_transition as mt

    source = Path(mt.__file__).read_text(encoding="utf-8")
    patched = source.replace(
        "UNVALIDATED_MIXED_EDGE_SELECTORS = (16, 28)",
        "UNVALIDATED_MIXED_EDGE_SELECTORS = (16, 28, 99)")
    assert patched != source, "the selector tuple moved; re-point this test"
    namespace = {"__name__": "mt_probe", "__file__": mt.__file__}
    with pytest.raises(RuntimeError, match=r"\[99\]"):
        exec(compile(patched, mt.__file__, "exec"), namespace)


def test_the_offline_cross_scheme_refusal_mirrors_the_online_one():
    """The invariant the mirror test protects, from the WDM6 side.

    ``OFFLINE_CHILD_MP_PHYSICS`` already excludes 16, so an mp=16 parent is
    refused EARLIER -- but "unreadable" is a different guarantee from "this
    closure is unmeasured", and it stops being a refusal the day the QNCCN
    field-map row lands.  The set is DERIVED from the online tuple now, so
    the two cannot drift.
    """
    from gpuwm import offline_child as oc
    from gpuwm.core import microphysics_transition as mt

    assert (set(oc._CROSS_SCHEME_REFUSED_MP_PHYSICS)
            == set(mt.UNVALIDATED_MIXED_EDGE_SELECTORS))
    assert 16 in oc._CROSS_SCHEME_REFUSED_MP_PHYSICS
    assert 16 not in oc.OFFLINE_CHILD_MP_PHYSICS
    assert 16 not in oc.PARENT_SCHEME_CONTRACT
    # The offline message names WDM6 too, out of the same table.
    with pytest.raises(oc.OfflineChildContractError) as caught:
        oc.map_microphysics_to_nssl18({}, source_mp_physics=16)
    assert "mp_physics=16 (WDM6" in str(caught.value)
    assert "nr/nc/nn" in str(caught.value)
    assert "Thompson" not in str(caught.value)


# --------------------------------------------------------------------------
# The reflectivity constants have ONE statement
# --------------------------------------------------------------------------

def test_the_wdm6_refl_kernel_takes_its_rain_moments_from_the_host():
    """No second, silent copy of xam_r/xcrg/xorg2 inside the kernel.

    ``radar_init_wdm6`` derived them and the kernel re-spelled them as
    literals, so changing xmu_r on the host would have left the kernel on
    the old exponents with nothing failing.
    """
    from gpuwm.core.refl import radar_init_wdm6

    source = (Path(__file__).resolve().parents[1]
              / "gpuwm/core/kernels/wdm6_refl.cu").read_text(encoding="utf-8")
    for literal in ("const double xcrg3 = 24.0",
                    "const double xam_r = RPI * 1000.0 / 6.0"):
        assert literal not in source, (
            f"{literal!r} is back in the kernel; it must arrive from "
            "gpuwm.core.refl.radar_init_wdm6")
    assert ("const double xam_r, const double xcrg3, const double xcrg4,"
            in source)
    # The two exponents that stay compile-time, and the values behind them.
    rc = radar_init_wdm6(0)
    assert rc.xcre == (4.0, 2.0, 5.0, 8.0)
    assert rc.xcrg == (6.0, 1.0, 24.0, 5040.0)
    assert rc.xorg2 == 1.0
    assert rc.xam_r == 1000.0 * math.pi / 6.0
    assert "lam * lam;   // lam**xcre(2), xcre(2) = 2" in source
    assert "pow(ilamr[k], 8.0)" in source

    # Morrison's rain density (RHOW = 997) must not be reachable here.
    assert "#undef RXAM_R" in source and "#undef RXAM_S" in source


def test_the_wdm6_refl_launcher_refuses_an_xcre_the_kernel_cannot_honour():
    """Show the failing form: the guard is not decoration."""
    import gpuwm.core.refl as refl_module

    original = refl_module.radar_init_wdm6
    try:
        refl_module.radar_init_wdm6 = lambda hail_opt=0: replace(
            original(hail_opt), xcre=(4.0, 3.0, 6.0, 9.0))
        shape = (2, 3, 4)
        arrays = [np.zeros(shape, dtype=np.float32) for _ in range(8)]
        with pytest.raises(ValueError, match="xcre"):
            refl_module.launch_refl10cm_wdm6(*arrays)
    finally:
        refl_module.radar_init_wdm6 = original


def test_the_plm_remap_clamp_divergence_is_documented_at_the_site():
    """The one place the kernel departs from the Fortran, with its citation."""
    source = _KERNEL.read_text(encoding="utf-8")
    index = source.index("kt = (kt > 0) ? kt - 1 : 0;\n        if (kt == kb)")
    preamble = source[max(0, index - 1800):index]
    assert "DELIBERATE, DOCUMENTED DIVERGENCE" in preamble
    for citation in ("nislfv_rain_plmr:2629", "nislfv_rain_plm6:2891"):
        assert citation in preamble, citation


def test_the_oracle_known_deltas_note_exists_and_the_registry_cites_it():
    """The inherited deltas land somewhere the campaign will look."""
    root = Path(__file__).resolve().parents[1]
    note = root / "docs" / "wdm6_oracle_known_deltas.md"
    assert note.exists()
    text = note.read_text(encoding="utf-8")
    for topic in ("_rgmma", "-35 dBZ", "module_mp_wdm6.F:607-614"):
        assert topic in text, topic

    registry = json.loads(
        (root / "gpuwm" / "physics_registry_v2.json").read_text(
            encoding="utf-8"))
    option = registry["components"]["microphysics"]["options"]["wdm6-mp16"]
    assert any("docs/wdm6_oracle_known_deltas.md" in warning
               for warning in option["warnings"])
