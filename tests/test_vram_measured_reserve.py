"""The fit gate's grid-independent terms, against what they measure as.

Task 206.  Four separate defects, each pinned by its own test and each
named by the breakage it caused:

1. THE NON-POOL TERM WAS CHARGED TWICE.  The wizard compares the affine
   machine-peak ENVELOPE -- which already carries the CUDA context and
   the local-memory backing store -- against a budget that had the very
   same bytes subtracted from free VRAM a second time by the allocation
   reserve.  On a 10 GiB RTX 3080 that spent 2.91 GiB of a 10 GiB card
   twice over, and refused the smallest layout the hrrr ladder has.

2. A PRESENT CARD WAS PRICED ON AN ABSENT ONE'S PROFILE.  The wizard
   MEASURES the local card's capacity and then prices its kernel
   backing store against the 170-SM reference profile.  A 68-SM 3080
   was charged 2.5x its own reservation -- 1.49 GiB -- while
   ``gpuwm check`` on the same machine used the live profile, so the
   two doors disagreed about the same bytes.

3. THE LINUX ENVELOPE CHARGED NO POOL SLACK.  Fifteen instrumented
   whole forecasts on two Linux cards held 1.17-1.19x the itemized
   estimate in the CuPy pool, and the Linux envelope under-predicted
   every one of them by 0.39-0.74 GiB.  The 2026-08-19 WDDM calibration
   had already flagged this and asked for the measurement; this is it.

4. THE CUDA-CONTEXT CONSTANT WAS ONE CARD'S READING.  432 MiB, taken on
   a 5090 in 2026-07, over-charged a 5070 Ti by 48 MiB and UNDER-charged
   a 5090 by 215 MiB at run time.

Every number below comes from
``docs/public/receipts/linux/linux-vram-calibration-20260820.json``.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpuwm.core import preflight as pf

REPO = Path(__file__).resolve().parents[1]
RECEIPT = json.loads(
    (REPO / "docs" / "public" / "receipts" / "linux"
     / "linux-vram-calibration-20260820.json").read_text())
GIB = 1024 ** 3


def _card(name: str) -> dict:
    for row in RECEIPT["cards"]:
        if row["name"] == name:
            return row
    raise KeyError(name)


def _profile(name: str) -> pf.DeviceLocalMemoryProfile:
    row = _card(name)
    return pf.DeviceLocalMemoryProfile(
        name=row["name"],
        multiprocessor_count=row["multiprocessor_count"],
        max_threads_per_multiprocessor=row["max_threads_per_multiprocessor"],
        default_stack_limit_bytes=row["default_stack_limit_bytes"],
        bare_context_bytes=row["bare_cuda_context_bytes"])


def _single_domain(nx: int, ny: int, *, nz: int = 49):
    """The smallest real experiment this estimator will price."""
    from gpuwm.domain_wizard import (LADDER_RATIOS, ROOT_DX_M,
                                     _projection_entries, experiment_from_text,
                                     render_config)

    projection = _projection_entries(39.96, -83.0, "lambert")
    text = render_config(
        name="t206", start_time=datetime(2026, 8, 19, 12, tzinfo=timezone.utc),
        hours=6, projection=projection, dims=[(nx, ny)],
        ratios=LADDER_RATIOS["12"], fetch_hints={}, case_data=None,
        root_dx_m=ROOT_DX_M)
    return experiment_from_text(text, source="<t206>")


# ---------------------------------------------------------------------------
# 1. the non-pool term, charged once
# ---------------------------------------------------------------------------

def test_the_fit_budget_does_not_re_charge_what_the_envelope_carries():
    """The envelope's own terms may not be subtracted from free twice.

    ``machine_peak_envelope_bytes`` is a model of the WHOLE device
    residency this process reaches -- pool, CUDA context, backing store
    and measured residue.  What is left outside it is other processes,
    which is exactly :data:`gpuwm.core.preflight.EXTERNAL_MARGIN_BYTES`.
    Anything else in the budget it is compared against is that process
    paying for itself twice.
    """

    from gpuwm.domain_wizard import sizing_budget_bytes

    exp = _single_domain(60, 48)
    free = int(9.25 * GIB)
    budget = sizing_budget_bytes(
        exp, free_bytes=free, vram_gib=None,
        forcing_interval_seconds=3600.0,
        profile=_profile("NVIDIA GeForce RTX 3080"))
    assert free - budget == pf.EXTERNAL_MARGIN_BYTES


def test_the_fit_gate_is_never_looser_than_the_allocation_gate():
    """A wizard PASS must always be a ``gpuwm check`` PASS.

    The two doors size the same machine and a config that passes the one
    that emits it and fails the one that verifies it is the 2.5.0 walk's
    #162 all over again.  The wizard's comparison is the STRICTER of the
    two by construction: it prices the machine-wide peak where check
    prices the pool.  Held here as an inequality over a span of grids so
    it cannot be broken by tuning either side.
    """

    from gpuwm.domain_wizard import sizing_budget_bytes

    profile = _profile("NVIDIA GeForce RTX 5090")
    free = int(30.0 * GIB)
    for nx, ny in ((60, 48), (170, 136), (300, 300), (466, 374)):
        exp = _single_domain(nx, ny)
        estimate = pf.estimate_experiment(exp, profile=profile)
        fit_budget = sizing_budget_bytes(
            exp, free_bytes=free, vram_gib=None,
            forcing_interval_seconds=3600.0, profile=profile)
        alloc_reserve = pf.ReservePolicy.n0_alloc(
            exp, profile=profile,
            estimate_bytes=estimate.alloc_estimate_bytes)
        alloc_budget = alloc_reserve.budget_bytes(free)
        wizard_fits = estimate.peak_envelope_bytes <= fit_budget
        check_fits = estimate.alloc_estimate_bytes <= alloc_budget
        assert not wizard_fits or check_fits, (
            f"{nx}x{ny}: wizard PASS but check FAIL")


# ---------------------------------------------------------------------------
# 2. a present card is priced on its own profile
# ---------------------------------------------------------------------------

def test_a_measured_local_card_is_priced_on_its_own_shader_count():
    """68 SMs cost 68 SMs, not the reference card's 170.

    The reservation law is a product over the DEVICE's resident-thread
    capacity, so pricing a present 3080 against the 5090 reference
    inflates the one term that cannot be shrunk by a smaller grid.
    """

    exp = _single_domain(60, 48)
    live = _profile("NVIDIA GeForce RTX 3080")
    reference = pf.card_local_memory_profile(10.0)
    assert reference.multiprocessor_count == 170  # the absent-card bound
    on_live = pf.kernel_local_memory_bytes(exp, profile=live)
    on_reference = pf.kernel_local_memory_bytes(exp, profile=reference)
    assert on_live * 2 < on_reference
    ratio = on_reference / on_live
    assert abs(ratio - 170 / 68) < 0.01


def test_the_wizard_prices_the_card_it_measured(monkeypatch):
    """No declaration means the local card -- capacity AND profile.

    The wizard already probes the local card for its capacity.  Reading
    the capacity off the probe and then pricing the kernel set against a
    different card is where the 3080 walk's 1.49 GiB came from.
    """

    from gpuwm import domain_wizard as dw

    live = _profile("NVIDIA GeForce RTX 3080")
    seen = {}

    def fake_probe(*args, **kwargs):
        return {"total_bytes": int(10.0 * GIB), "free_bytes": int(9.0 * GIB),
                "profile": {"name": live.name,
                            "multiprocessor_count": live.multiprocessor_count,
                            "max_threads_per_multiprocessor":
                                live.max_threads_per_multiprocessor,
                            "default_stack_limit_bytes":
                                live.default_stack_limit_bytes,
                            "bare_context_bytes": live.bare_context_bytes}}

    monkeypatch.setattr(dw, "device_memory_probe_subprocess", fake_probe)
    vram_gib, profile, _sentence = dw.resolve_sizing_card(
        card=None, vram_gib=None)
    assert profile is not None, "a measured card must yield a measured profile"
    assert profile.multiprocessor_count == 68
    assert seen == {}


# ---------------------------------------------------------------------------
# 3. the pool-slack term is a pool property, not a driver-model one
# ---------------------------------------------------------------------------

def test_pool_slack_is_charged_on_linux_as_well_as_windows():
    """The CuPy pool holds past the itemization on BOTH driver models."""

    common = dict(alloc_estimate_bytes=4 * GIB, non_pool_bytes=GIB,
                  domains=1)
    linux = pf.machine_peak_envelope_bytes(**common, family="linux")
    windows = pf.machine_peak_envelope_bytes(**common, family="windows")
    bare = (4 * GIB + GIB + pf.ENVELOPE_UNMODELLED_BYTES)
    assert linux > bare, "the Linux envelope charges no pool slack"
    assert linux == windows


def test_pool_slack_is_charged_by_radiation_lane_not_driver_model():
    """It is the legacy engines' retained call-peak workspace.

    Three campaigns split on the same boundary: the rte-rrtmgp lane's
    pool tracks the itemization (0.88-1.00x) and the legacy-RRTMG lane's
    runs past it (1.13-1.47x).  Charging by driver model got both halves
    wrong at once -- Linux legacy paid nothing and Windows rte-rrtmgp
    paid 20% of its estimate for a mechanism it does not have.
    """

    common = dict(alloc_estimate_bytes=4 * GIB, non_pool_bytes=GIB,
                  domains=1)
    for family in ("linux", "windows"):
        legacy = pf.machine_peak_envelope_bytes(
            **common, family=family, legacy_radiation=True)
        modern = pf.machine_peak_envelope_bytes(
            **common, family=family, legacy_radiation=False)
        assert legacy - modern == pytest.approx(
            pf.POOL_SLACK_FRACTION * 4 * GIB, abs=1)


def test_not_saying_which_radiation_lane_charges_the_slack():
    """The default is the conservative one.

    An envelope that guesses optimistically when it has not been told
    is not an envelope.
    """

    common = dict(alloc_estimate_bytes=4 * GIB, non_pool_bytes=GIB,
                  domains=1, family="linux")
    assert (pf.machine_peak_envelope_bytes(**common)
            == pf.machine_peak_envelope_bytes(**common,
                                              legacy_radiation=True))


@pytest.mark.parametrize("run", RECEIPT["runs"], ids=lambda r: (
    f"{r['node'].split()[-1]}-{r['model']}"))
def test_the_envelope_bounds_every_measured_run(run):
    """Fifteen whole forecasts, two Linux cards, sampled at 20 Hz.

    The envelope is re-formed from each run's own itemized estimate and
    its own MEASURED non-pool residency, so this tests the envelope's
    FORM rather than re-testing the estimator.  An envelope that does
    not bound a measured peak is not an envelope.
    """

    estimate = int(run["alloc_estimate_gib"] * GIB)
    measured_non_pool = int(run["measured_non_pool_gib"] * GIB)
    envelope = pf.machine_peak_envelope_bytes(
        alloc_estimate_bytes=estimate,
        non_pool_bytes=measured_non_pool,
        domains=run["domains"], family="linux")
    assert envelope >= int(run["device_peak_gib"] * GIB)


# ---------------------------------------------------------------------------
# 4. the CUDA context follows the card
# ---------------------------------------------------------------------------

def test_the_context_charge_follows_the_measured_card():
    """A 5090's context costs more than a 5070 Ti's, and is measured.

    One constant for every card is wrong in both directions at once:
    the 2026-07 432 MiB reading over-charged a 5070 Ti and under-charged
    the very card it was taken on once that card ran on Linux.
    """

    small = _profile("NVIDIA GeForce RTX 5070 Ti")
    big = _profile("NVIDIA GeForce RTX 5090")
    assert big.cuda_context_bytes > small.cuda_context_bytes
    for profile in (small, big):
        assert profile.cuda_context_bytes > profile.bare_context_bytes, (
            "a run loads kernel modules the bare context has not paid for")


#: The widest per-thread local frame of the AS-BUILT binary the receipt's
#: runs launched: ``ysu_column``'s 9,232 B.  The 2026-08-21 column-workspace
#: cuts (17cf943ef YSU, 48ff6b813 KF, merged by 2bb15b22f and c0f818ef1)
#: retired that frame -- measured 9,232 -> 0 B on NVRTC 13.0/13.3, sm_86 and
#: sm_120 -- so the live pricing legitimately dropped below these readings
#: and the receipt is bounded by the pricing of the binary that PAID it,
#: exactly as tests/test_preflight.py prices the 2026-07-26 runs against
#: KF_AS_BUILT_FRAME.
YSU_AS_BUILT_FRAME_BYTES = 9232


@pytest.mark.parametrize("card_name,lo,hi", [
    ("NVIDIA GeForce RTX 5070 Ti", 1.1958, 1.1994),
    ("NVIDIA GeForce RTX 5090", 2.6402, 2.6447),
])
def test_the_non_pool_charge_covers_the_measured_run_time_residency(
        card_name, lo, hi):
    """Priced non-pool must bound what the runs actually paid.

    The battery's shared 300x300x49 domain, whose widest launched frame
    was the YSU module's, on both Linux cards.  The charge may exceed the
    measurement -- it is a bound -- but it may never fall short, which
    is what the flat context constant did on the 5090.

    Re-pinned 2026-08-30: these residencies were paid by the receipt-era
    binary, before the 2026-08-21 frame cuts moved the YSU and KF column
    arrays into workspaces (see :data:`YSU_AS_BUILT_FRAME_BYTES`), so the
    bound is the as-built price -- context plus the reservation of the
    retired 9,232 B frame -- and the LIVE price is held to the contraction
    direction: never above the as-built one, never below its own context.
    """

    exp = _single_domain(300, 300)
    profile = _profile(card_name)
    as_built = (profile.cuda_context_bytes
                + profile.reservation_bytes(YSU_AS_BUILT_FRAME_BYTES))
    assert as_built >= int(hi * GIB), (
        f"{card_name}: as-built priced {as_built / GIB:.4f} GiB under a "
        f"measured {hi:.4f} GiB")
    # ...and not absurdly over it: a bound nobody can reach is a refusal
    # machine.  1.6x the worst measurement is the stated ceiling.
    assert as_built <= int(hi * 1.6 * GIB)
    # The frame cuts are contractions: the live charge may only sit at or
    # below what the measured binary paid, and it still carries the whole
    # measured per-card context.
    priced = pf.non_pool_device_bytes(exp, profile=profile)
    assert profile.cuda_context_bytes < priced <= as_built


# ---------------------------------------------------------------------------
# the protective direction: this loosens a gate, so prove it still bites
# ---------------------------------------------------------------------------

def test_a_config_that_cannot_fit_is_still_refused():
    """The whole point of the gate.  A 12 GiB card cannot host this."""

    from gpuwm.domain_wizard import sizing_budget_bytes

    profile = _profile("NVIDIA GeForce RTX 5070 Ti")
    exp = _single_domain(900, 900)
    estimate = pf.estimate_experiment(exp, profile=profile)
    free = int(11.0 * GIB)
    budget = sizing_budget_bytes(
        exp, free_bytes=free, vram_gib=None,
        forcing_interval_seconds=3600.0, profile=profile)
    assert estimate.peak_envelope_bytes > budget


def test_free_vram_is_never_more_than_the_card_reports_machine_wide():
    """The instrument that sees other processes is the ceiling.

    ``cudaMemGetInfo`` answers "free if every other process were
    evicted", and under WDDM it can be: measured 2026-08-20 on a loaded
    RTX 3080 desktop, four consecutive samples, memGetInfo said
    9,097 MiB free against NVML's 3,375-3,405 -- 5.7 GiB of a 10 GiB
    card.  Before task 206 that over-statement was masked by a budget
    that also subtracted the non-pool residency twice; with the double
    charge gone it would have been spent.
    """

    source = pf._DEVICE_MEMORY_PROBE_SOURCE
    assert "min(int(free), _nvml_free)" in source
    # ...and both raw readings survive into the payload, so a reader can
    # see which instrument bound the number.
    assert "free_bytes_memgetinfo" in source
    assert "free_bytes_nvml" in source


def test_the_device_probe_source_imports_nothing_from_gpuwm():
    """A bare interpreter must be able to run it.

    The probe exists to answer "is there a card, and is there a runtime
    to read it with" in a process that has neither.  Importing gpuwm to
    borrow a helper turned the no-CuPy exit code into an ImportError
    traceback, which is the confusion :data:`PROBE_EXIT_NO_RUNTIME` was
    added to end.
    """

    # Comments may discuss gpuwm; no statement may import it.
    statements = [line for line in pf._DEVICE_MEMORY_PROBE_SOURCE.splitlines()
                  if not line.lstrip().startswith("#")]
    assert not [line for line in statements
                if line.lstrip().startswith(("import gpuwm", "from gpuwm"))]
    # It is still allowed the standard library it needs.
    assert any(line.startswith("import subprocess") for line in statements)


def test_the_advisory_names_the_row_it_priced():
    """A number a reader cannot trace is a number they cannot check."""

    profile = _profile("NVIDIA GeForce RTX 3080")
    text = pf.non_pool_basis(profile)
    assert profile.name in text
    assert "measured" in text


def test_an_unmeasured_card_says_so_and_is_priced_conservatively():
    """A card with no bare-context reading is modelled, and admits it."""

    unmeasured = pf.DeviceLocalMemoryProfile(
        name="NVIDIA GeForce RTX 4070", multiprocessor_count=46,
        max_threads_per_multiprocessor=1536)
    assert unmeasured.bare_context_bytes is None
    assert "modelled" in pf.non_pool_basis(unmeasured)
    # the modelled rate must bound every card this campaign measured
    for name in ("NVIDIA GeForce RTX 3080", "NVIDIA GeForce RTX 5070 Ti",
                 "NVIDIA GeForce RTX 5090"):
        measured = _profile(name)
        modelled = dataclasses.replace(measured, bare_context_bytes=None)
        assert modelled.cuda_context_bytes >= measured.cuda_context_bytes


# ---------------------------------------------------------------------------
# 5. the ONE line the 206 fix did not reach (task 240)
# ---------------------------------------------------------------------------

def _write_config(tmp_path, nx, ny):
    from datetime import datetime, timezone

    from gpuwm.domain_wizard import (LADDER_RATIOS, ROOT_DX_M,
                                     _projection_entries, render_config)

    text = render_config(
        name="t240", start_time=datetime(2026, 8, 19, 12, tzinfo=timezone.utc),
        hours=6, projection=_projection_entries(39.96, -83.0, "lambert"),
        dims=[(nx, ny)], ratios=LADDER_RATIOS["12"],
        fetch_hints={"source": "gfs"},
        case_data=None, root_dx_m=ROOT_DX_M,
        profile="thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1")
    path = tmp_path / "t240.toml"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _check_report(tmp_path, nx, ny, budget_gib):
    """The real ``gpuwm check`` front door, CPU mode, declared budget."""
    import os
    import subprocess
    import sys

    config = _write_config(tmp_path, nx, ny)
    env = dict(os.environ)
    env["GPUWM_NO_LOCAL_GPU"] = "1"
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run(
        [sys.executable, "-m", "gpuwm.cli", "check", str(config),
         "--budget-gib", str(budget_gib)],
        capture_output=True, text=True, cwd=str(REPO), env=env, timeout=900)


def test_the_binding_phase_line_prices_the_envelope_against_its_own_budget(
        tmp_path):
    """One printed line still charged the non-pool residency twice.

    Task 206 removed that double count from the wizard and from the exit
    code and left it in the sentence a reader actually reads.  Measured
    2026-08-20 on the loaded RTX 3080 with 5.75 GiB free: ``gpuwm check``
    printed "BINDING PHASE: ... 5.40 GiB peak envelope; that EXCEEDS the
    3.79 GiB budget by 1.61 GiB" -- 3.79 being free minus the whole
    ALLOCATION reserve, whose 1.37 GiB of CUDA context and backing store
    the 5.40 GiB envelope already carries -- while the WARNING three
    lines below it, and ``gpuwm go`` seconds later, both used free minus
    the 0.50 GiB other-process margin, and that forecast ran to rc 0 and
    came out byte-identical to the roomy run.

    The breakage is a reader believing the first sentence: it says a
    configuration misses by 1.61 GiB when it misses by 0.15, and it says
    "does not fit" about runs that do.
    """
    import re

    out = _check_report(tmp_path, 60, 48, 4.0)
    text = out.stdout + out.stderr
    binding = re.search(r"BINDING PHASE:.*?the ([\d.]+) GiB budget", text)
    reserve_line = re.search(
        r"free\s+([\d.]+) GiB; budget\s+([\d.]+) GiB", text)
    assert binding and reserve_line, text[-3000:]
    free_gib = float(reserve_line.group(1))
    binding_budget = float(binding.group(1))
    envelope_budget = free_gib - pf.EXTERNAL_MARGIN_BYTES / GIB
    assert abs(binding_budget - envelope_budget) < 0.02, (
        f"BINDING PHASE compared the envelope against "
        f"{binding_budget:.2f} GiB; the envelope's own budget is free "
        f"{free_gib:.2f} minus the "
        f"{pf.EXTERNAL_MARGIN_BYTES / GIB:.2f} GiB other-process margin "
        f"= {envelope_budget:.2f} GiB.  Anything smaller subtracts the "
        "CUDA context and the backing store a second time")


def test_the_binding_phase_verdict_agrees_with_the_exit_code(tmp_path):
    """"It fits" and rc 4 may not come out of the same report.

    The exit code is read off ``envelope_over_budget``; the sentence was
    read off a different budget, so the two could and did disagree.
    """
    out = _check_report(tmp_path, 60, 48, 4.0)
    text = out.stdout + out.stderr
    assert "BINDING PHASE" in text, text[-3000:]
    line = text.split("BINDING PHASE", 1)[1].split("\n", 1)[0]
    fits = "it fits the" in line
    assert fits == (out.returncode != 4), (
        f"rc {out.returncode} against a BINDING PHASE line that says "
        f"{'fits' if fits else 'does not fit'}: {line.strip()}")


# ---------------------------------------------------------------------------
# 6. two meters, one model (task 240)
# ---------------------------------------------------------------------------

TWO_METER = json.loads(
    (REPO / "docs" / "public" / "receipts" / "wddm"
     / "rtx3080-two-meter-20260820.json").read_text())


def test_the_two_meter_receipt_has_points_to_check():
    """Positive evidence of work: a loop over nothing passes every test."""
    assert len(TWO_METER["runs"]) >= 3
    for run in TWO_METER["runs"]:
        assert run["returncode"] == 0, run["name"]
        for key in ("machine_delta_gib", "process_device_peak_gib",
                    "pool_total_peak_gib", "measured_non_pool_gib"):
            assert run[key] > 0, (run["name"], key)


def test_the_envelope_bounds_the_meter_the_gate_compares_against():
    """The gate compares the envelope against MACHINE-WIDE free VRAM.

    So the envelope has to bound the machine-wide addition, and it does
    at every instrumented point.  This is the half task 240 read as
    slack: the margin is 0.03-1.05 GiB and it is not uniform, because
    the meter subtracts a desktop baseline that moves.
    """
    for run in TWO_METER["runs"]:
        envelope = TWO_METER["model"][f"{run['name']}_envelope_gib"]
        assert run["machine_delta_gib"] <= envelope, run["name"]


def test_the_machine_meter_under_reads_the_run_by_the_evicted_desktop():
    """WDDM pages the desktop out, so the two meters cannot be swapped.

    ``machine peak - desktop baseline`` is smaller than the run's own
    cudaMemGetInfo footprint by 0.80-1.13 GiB at these points -- that is
    desktop residency the display driver evicted while the run held the
    card, not memory the run did not use.  Reading the first number as
    the run's cost is what made a model that is SHORT against the run's
    own footprint look 0.76 GiB generous.
    """
    for run in TWO_METER["runs"]:
        evicted = run["process_device_peak_gib"] - run["machine_delta_gib"]
        assert 0.5 < evicted < 1.5, (run["name"], evicted)
        assert abs(evicted - run["evicted_gib"]) < 0.01


def test_the_measured_non_pool_residency_is_grid_independent():
    """1.75 GiB across a 3.4x span of columns, which is what it should be.

    ``device footprint peak - pool-held peak`` is the CUDA context plus
    the launch-time local-memory backing store, and neither scales with
    the grid.  Measured 1.749-1.751 GiB at 60x48, 80x64 and 110x88 on
    this card, where the shipped model charges 1.37 GiB: the 0.38 GiB
    difference is real and is currently absorbed by the unmodelled and
    pool-slack terms rather than named.  Pinned so a future re-fit
    cannot quietly spend it twice.
    """
    values = [run["measured_non_pool_gib"] for run in TWO_METER["runs"]]
    assert max(values) - min(values) < 0.01, values
    assert 1.70 < min(values) < 1.80, values


def test_the_process_meter_repeats_and_the_machine_meter_does_not():
    """Validated in both directions, which is why the verdict is trusted.

    The same configuration, run twice on the same card: the process
    meter repeated to the byte and the machine meter moved 0.24 GiB.  An
    instrument that cannot repeat cannot carry a 0.76 GiB conclusion.
    """
    repeat = TWO_METER["reproducibility"]
    assert (repeat["process_device_peak_bytes_run1"]
            == repeat["process_device_peak_bytes_run2"])
    assert abs(repeat["machine_delta_gib_run1"]
               - repeat["machine_delta_gib_run2"]) > 0.2
    assert repeat["wrfout_frames_byte_identical"] == "7/7"
