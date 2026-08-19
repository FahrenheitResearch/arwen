"""The Windows memory gate prices what a measured card actually does.

The 2026-08-19 GPU walk on a 10 GiB RTX 3080 (Windows 11, WDDM desktop
resident) measured the defect this file pins.  ``gpuwm go`` predicted a
9.91 GiB peak envelope for a 110x88x49 single-domain 12 km forecast --
``footprint 5.66 x 1.75 WDDM floor`` -- against a 7.12 GiB budget and
refused.  Forced through, the machine-wide peak was 5,925 MiB over a
3,227 MiB desktop baseline: the run's own contribution was about
2.6 GiB, UNDER the budget by more than 4 GiB.  Both the footprint
projection and the 1.75 multiplier were fiction on this card, and the
refusal's printed remedy (``gpuwm domain --vram-gib 8``) itself refused
at every size, because 78% of the floored envelope was grid-independent.

The calibration that replaces it: six whole bare-default forecasts on
the 3080 (60x48 to 240x192, rte-rrtmgp and legacy-RRTMG suites),
machine-wide nvidia-smi at 0.25 s beside the runtime's own
GpuPeakMemoryWatcher receipts.  Measured: the itemized estimate tracks
the CuPy pool at 0.95-1.0x on the rte-rrtmgp lane; the machine-wide
peak minus the desktop baseline lands at estimate + the itemized
non-pool residency, within -0.20..+0.95 GiB; the one positive-residual
lane is legacy-RRTMG pool retention, proportional to the estimate
(worst +0.30x).  Receipts:
``docs/public/receipts/wddm/rtx3080-wddm-calibration-20260819.json``
and the walk capture in ``Downloads/ux-walks-replay/gpu-walk-3080.md``.

What ships, and what this file holds:

* ONE envelope.  ``envelope_platform`` no longer switches model by card
  size, so the wizard (which knows the card) and ``gpuwm check`` /
  ``gpuwm go`` (which measure the machine) can never price the same
  bytes with different formulas -- the open task #162 mechanism.
* The Windows envelope is the measured affine form: estimate + itemized
  non-pool + the measured unmodelled constant + a WDDM pool-slack term
  proportional to the estimate.  No 1.75 multiplier, no 5090 pool
  constants in the intercept.
* The walk's exact configuration passes ``gpuwm check`` against the
  walk's measured machine state.
* Refusals are true: the wizard's no-layout-fits refusal ranks lighter
  profiles by the estimator's PRICED envelope (the walk's refusal named
  legacy-RRTMG suites that measure 2.1x heavier), and ``gpuwm go``'s
  refusal stops pointing at a ``--vram-gib`` recursion that cannot
  succeed while keeping its measured free-VRAM honesty.
"""

from __future__ import annotations

import argparse
import math
import types
from datetime import datetime
from pathlib import Path

import pytest

from gpuwm.core import preflight as pf
from gpuwm import domain_wizard as dw

GIB = pf.GIB

#: The walk's measured machine state, verbatim from the capture.
WALK_TOTAL_GIB = 9.99951
WALK_FREE_BYTES = int(8.88 * GIB)
#: The live 3080 profile the check priced non-pool terms against.
WALK_SMS = 68
WALK_THREADS_PER_SM = 1536

PROJECTION = {"map_proj": "lambert", "ref_lat": 39.1, "ref_lon": -94.6,
              "truelat1": 29.1, "truelat2": 49.1, "stand_lon": -94.6}
START = datetime(2026, 8, 19, 0, 0, 0)


def _walk_profile():
    return pf.DeviceLocalMemoryProfile(
        name="NVIDIA GeForce RTX 3080", multiprocessor_count=WALK_SMS,
        max_threads_per_multiprocessor=WALK_THREADS_PER_SM)


def _emit(tmp_path: Path, nx: int, ny: int,
          profile: str = dw.DEFAULT_PHYSICS_PROFILE) -> Path:
    """The wizard's own emission at a pinned size, on disk."""
    area = dw.fetch_area_hint(PROJECTION, nx, ny, source="gfs")
    hints = {"source": "gfs", "cycle": "2026-08-19T00", "hours": 6,
             "area": area, "out": "configs/data/walk", "cadence": 3}
    text = dw.render_config(
        name=f"area_{nx}x{ny}", start_time=START, hours=6,
        projection=PROJECTION, dims=[(nx, ny)], ratios=(),
        fetch_hints=hints, case_data=None, root_dx_m=12000.0,
        profile=profile)
    path = tmp_path / f"walk_{nx}x{ny}.toml"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _walk_machine(monkeypatch, *, free_bytes=WALK_FREE_BYTES,
                  total_gib=WALK_TOTAL_GIB, profile=None):
    """Fake the walk's machine for `gpuwm check`: platform, card, free."""
    # The state module's own `import cupy` must resolve (to the real
    # or absent one) BEFORE the memGetInfo stub lands in sys.modules,
    # or the stub is what the whole model imports.
    import gpuwm.core.state  # noqa: F401
    monkeypatch.setattr(pf.sys, "platform", "win32")
    total = int(total_gib * GIB)
    stub = types.SimpleNamespace(cuda=types.SimpleNamespace(
        runtime=types.SimpleNamespace(
            memGetInfo=lambda: (free_bytes, total))))
    monkeypatch.setitem(pf.sys.modules, "cupy", stub)
    monkeypatch.setattr(
        pf, "live_device_local_memory_profile",
        lambda: (_walk_profile() if profile is None else profile))


def _run_check(argv):
    parser = argparse.ArgumentParser(prog="gpuwm")
    sub = parser.add_subparsers(dest="command", required=True)
    pf.register_cli(sub)
    args = parser.parse_args(argv)
    return args.func(args)


# ---------------------------------------------------------------------------
# The model itself
# ---------------------------------------------------------------------------

def test_envelope_platform_no_longer_switches_by_card_size():
    """One family per platform: the #162 divergence dies structurally.

    The wizard passed the card size and got the experimental small-card
    tier; ``gpuwm check`` measured free VRAM, passed nothing, and got
    the 1.75 WDDM floor -- PASS and exit 4 on the same bytes, seconds
    apart.  The family may depend on the platform only.
    """
    for vram in (None, 8.0, 9.99951, 11.0, 12.0, 16.0, 24.0, 32.0):
        assert pf.envelope_platform("win32", vram) == "windows"
        assert pf.envelope_platform("linux", vram) == "linux"


def test_windows_envelope_is_the_measured_affine_model(monkeypatch,
                                                       tmp_path):
    """estimate + non-pool + unmodelled + measured WDDM slack.  No 1.75.

    Measured on the 3080 (six whole forecasts): machine-wide peak minus
    the desktop baseline = estimate + itemized non-pool, within
    -0.20..+0.95 GiB, where the positive residual is legacy-RRTMG pool
    retention proportional to the estimate (worst +0.30x, so 0.5 GiB
    unmodelled + 0.20x estimate covers it with margin).  The retired
    multiplicative form predicted 9.91 GiB for a run that measured
    2.6 GiB -- 3.8x reality.
    """
    monkeypatch.setattr(pf.sys, "platform", "win32")
    config = _emit(tmp_path, 110, 88)
    exp = dw.experiment_from_text(
        config.read_text(encoding="utf-8"), source=str(config))
    est = pf.estimate_experiment(exp, profile=_walk_profile())
    expected = (est.alloc_estimate_bytes
                + est.non_pool_device_bytes
                + pf.ENVELOPE_UNMODELLED_BYTES
                + math.ceil(pf.WDDM_POOL_SLACK_FRACTION
                            * est.alloc_estimate_bytes))
    assert est.envelope_family == "windows"
    assert est.peak_envelope_bytes == expected
    # The 5090 zero-step probe constant and the pool-retention constant
    # are display projections, never envelope intercept terms.
    assert est.envelope_intercept_bytes == est.non_pool_device_bytes
    # Bounded from above against the walk's measurement, without the
    # 71% slop: the walk measured ~2.6 GiB of own contribution.
    measured = int(2.60 * GIB)
    assert est.peak_envelope_bytes > measured
    assert est.peak_envelope_bytes < int(1.75 * measured), (
        "the envelope must bound the measured peak without the "
        "multiplicative slop class this gate was burned by")
    # And nothing multiplicative survives on the terms line.
    terms = est.peak_envelope_terms()
    assert "1.75" not in terms
    assert "WDDM floor" not in terms
    assert "pool slack" in terms


def test_wizard_and_check_price_one_envelope_for_one_machine(monkeypatch,
                                                             tmp_path):
    """Same experiment, same machine: one envelope, byte for byte.

    The wizard prices with the card it measured or was declared
    (``vram_gib``); check prices with no declaration at all.  Any split
    between those two numbers is the #162 defect resurfacing.
    """
    monkeypatch.setattr(pf.sys, "platform", "win32")
    for vram, (nx, ny) in ((9.99951, (110, 88)), (12.0, (110, 88)),
                           (16.0, (170, 136)), (24.0, (240, 192))):
        config = _emit(tmp_path, nx, ny)
        exp = dw.experiment_from_text(
            config.read_text(encoding="utf-8"), source=str(config))
        profile = pf.card_local_memory_profile(vram)
        wizard = pf.estimate_experiment(exp, vram_gib=vram,
                                        profile=profile)
        check = pf.estimate_experiment(exp, vram_gib=None,
                                       profile=profile)
        assert wizard.peak_envelope_bytes == check.peak_envelope_bytes, (
            vram, nx, ny)
        assert wizard.envelope_family == check.envelope_family


# ---------------------------------------------------------------------------
# The walk's exact machine state, through the real door
# ---------------------------------------------------------------------------

def test_walk_config_passes_check_on_the_walk_machine(monkeypatch,
                                                      tmp_path, capsys):
    """The 110x88 config on a card with 8.88 GiB free: PASS, exit 0.

    This is the walk's step 11 verbatim -- the same emission, the same
    measured free VRAM, the same live-card profile -- which exited 4
    under the multiplicative model while the very same bytes ran to
    completion 1.33 GiB under budget.
    """
    _walk_machine(monkeypatch)
    config = _emit(tmp_path, 110, 88)
    rc = _run_check(["check", str(config)])
    out = capsys.readouterr().out
    assert "WDDM floor" not in out
    assert "x 1.75" not in out and "1.75x" not in out
    assert rc == 0, out


def test_check_still_refuses_what_genuinely_does_not_fit(monkeypatch,
                                                         tmp_path, capsys):
    """The calibrated gate is not a rubber stamp.

    The same machine state with 2 GiB free cannot hold the measured
    2.6 GiB run, and the envelope says so with a nonzero exit.
    """
    _walk_machine(monkeypatch, free_bytes=2 * GIB)
    config = _emit(tmp_path, 110, 88)
    rc = _run_check(["check", str(config)])
    capsys.readouterr()
    assert rc != 0


# ---------------------------------------------------------------------------
# True refusals
# ---------------------------------------------------------------------------

def test_no_layout_refusal_ranks_profiles_by_priced_envelope(monkeypatch):
    """The lever list is measured, not a species heuristic.

    The walk's refusal advised three ``rrtmg-legacy`` suites as lighter
    than the morrison default.  Measured on the 3080 at the same
    110x88 grid: morrison 2.60 GiB, thompson legacy-RRTMG 5.53 GiB --
    the advice named a 2.1x HEAVIER suite.  The estimator prices the
    legacy call-peak envelope; the refusal must rank by that price and
    drop any candidate it cannot price cheaper.
    """
    monkeypatch.setattr(pf.sys, "platform", "win32")
    with pytest.raises(dw.DomainFitError) as caught:
        dw.fit_ladder(
            ladder="12", free_bytes=int(4.5 * GIB), hours=6,
            start_time=START, projection=PROJECTION, source="gfs",
            name="area_test", profile=dw.DEFAULT_PHYSICS_PROFILE,
            vram_gib=WALK_TOTAL_GIB)
    message = str(caught.value)
    assert "does not fit" in message
    assert "grid-independent" in message
    # The two real levers stay named.
    assert "larger card" in message
    # No profile that PRICES heavier than the current one may be
    # offered as lighter.  Every rrtmg-legacy suite does.
    assert "rrtmg-legacy" not in message


def test_go_memory_refusal_names_reachable_remedies():
    """``gpuwm go``'s refusal: no --vram-gib recursion, honesty kept.

    The walk followed the printed remedy (``gpuwm domain --vram-gib 8``)
    and was refused at every grid size -- the remedy pointed back into
    the same broken accounting.  The refusal keeps its measured "the
    card has X GiB free right now" sentence and names remedies that can
    actually succeed: re-size against this card through the bare wizard
    (which measures it), a lighter suite by PRICED envelope, or a
    bigger card.
    """
    from gpuwm.go_cli import memory_refusal_text

    gate = {"verdict": "the forecast is the memory-binding phase at "
                       "9.00 GiB peak envelope; that EXCEEDS the "
                       "7.12 GiB budget by 1.88 GiB",
            "refuse": True, "warn": True,
            "free_bytes": int(8.88 * GIB)}
    text = memory_refusal_text(gate)
    assert "8.88 GiB" in text and "free right now" in text
    assert "--vram-gib" not in text, (
        "the remedy must not recurse into a declared-card sizing that "
        "the measured gate will refuse again")
    assert "gpuwm domain" in text
    assert "--no-memory-gate" in text
    assert "physics-profile" in text or "larger card" in text
