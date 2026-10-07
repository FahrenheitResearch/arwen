"""A CPU preparation is admitted against host RAM at every door.

A ``[tiles]`` declaration prepares a GFS forcing on the CPU, so the ingest
phase holds nothing on the card (its device envelope is zero) and its whole
working set is host RAM.  Two figures describe it: the arrays it certainly
holds at once (the floor), which ``gpuwm go`` and ``gpuwm check`` refuse
on, and its estimated peak, which the wizard sizes against
(``tests/test_domain_wizard_resolution.py``) and a verdict reports.  A
machine whose RAM sits between the two is admitted with a warning.  The
domain stays resident under ``mode = "auto"``, so no pinned store is
involved and the preparation is the only host term being weighed.
"""
from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime
from types import SimpleNamespace

import pytest

from gpuwm import cli, domain_wizard as wizard, go_cli
from gpuwm.core import preflight, streaming
from gpuwm.ingest import cpu_backend, real


GIB = 1024 ** 3
MIB = 1024 ** 2
#: How far past each figure a test host sits: well inside the gap between
#: the floor and the estimate of the configuration below.
STEP = 16 * MIB

_RESIDENT_TILED_GFS = """\
[experiment]
name = "prep-host"
start_time = 2026-09-26T06:00:00
run_seconds = 21600.0
restart_interval_s = 0.0

[fetch]
source = "gfs"
cycle = "2026-09-26T06"
hours = 6
cadence = 3

[shared]
nz = 49
ztop = 20000.0
moist = true
moist_cq = true
mp_physics = 10
ra_lw_physics = 4
ra_sw_physics = 4
sf_sfclay_physics = 91
sf_surface_physics = 2
bl_pbl_physics = 1
nwp_diagnostics = 1

[tiles]
mode = "auto"

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 206
ny = 204
time_step = 15
dx = 3000.0
history_interval_s = 3600.0
"""


@pytest.fixture()
def resident_tiled_gfs(tmp_path):
    path = tmp_path / "prep-host.toml"
    path.write_text(textwrap.dedent(_RESIDENT_TILED_GFS), encoding="utf-8")
    return path


@pytest.fixture()
def card(monkeypatch):
    def _probe(*_args, **_kwargs):
        return {"free_bytes": 15 * GIB, "total_bytes": 16 * GIB,
                "name": "test card", "local_memory_bytes_per_thread": 0}

    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", _probe)


def _preparation(config) -> tuple[int, int]:
    """(floor, estimated peak) of this config's CPU preparation."""
    exp = preflight._load_experiment_any(config)
    phases = preflight.estimate_phases(exp, source="gfs",
                                       forcing_interval_seconds=10800.0)
    assert phases.preprocess_backend == "cpu"
    assert phases.ingest_envelope_bytes == 0
    floor = phases.host_preparation_floor_bytes
    need = phases.host_preparation_bytes
    assert floor == phases.ingest.host_preprocess_floor_bytes
    assert need == phases.ingest.host_preprocess_bytes
    # Wide enough apart that a STEP lands strictly between them.
    assert 0 < floor + 2 * STEP < need
    return floor, need


def _host(monkeypatch, total):
    monkeypatch.setattr(streaming, "_host_total_bytes", lambda: int(total))


def test_go_refuses_only_a_preparation_whose_floor_exceeds_host_ram(
        resident_tiled_gfs, card, monkeypatch):
    floor, need = _preparation(resident_tiled_gfs)
    plan = {"config": str(resident_tiled_gfs), "source": "gfs", "cadence": 3}

    _host(monkeypatch, need + STEP)
    admitted = go_cli.memory_gate(plan)
    assert admitted["phases"].streamed is None
    assert admitted["refuse"] is False, admitted["verdict"]
    assert admitted["preparation_warning"] is None
    assert "preparing this configuration on the CPU" not in admitted["verdict"]
    # Admitted, the verdict still says where the preparation's memory is.
    assert (f"ingest 0.00 GiB of card and about {need / GIB:.2f} GiB of host "
            "RAM on the CPU") in admitted["verdict"]

    # Between the floor and the estimate: admitted, and told.
    _host(monkeypatch, floor + STEP)
    warned = go_cli.memory_gate(plan)
    assert warned["refuse"] is False, warned["verdict"]
    assert warned["preparation_refusal"] is None
    assert f"peak at {need / GIB:.2f} GiB of host RAM" in warned["preparation_warning"]

    _host(monkeypatch, floor - STEP)
    refused = go_cli.memory_gate(plan)
    assert refused["phases"].streamed is None
    assert refused["refuse"] is True, refused["verdict"]
    assert refused["card_refuse"] is False
    assert refused["preparation_warning"] is None
    assert "preparing this configuration on the CPU" in refused["verdict"]
    assert f"at least {floor / GIB:.2f} GiB of host RAM" in refused["verdict"]
    text = go_cli.memory_refusal_text(refused)
    assert "does not move host RAM" in text
    assert "free VRAM and re-run" not in text


def test_go_weighs_the_preparation_even_with_no_card_to_read(
        resident_tiled_gfs, monkeypatch):
    floor, _need = _preparation(resident_tiled_gfs)
    monkeypatch.setattr(preflight, "device_memory_probe_subprocess",
                        lambda *a, **k: None)
    plan = {"config": str(resident_tiled_gfs), "source": "gfs", "cadence": 3}
    _host(monkeypatch, floor - STEP)
    refused = go_cli.memory_gate(plan)
    assert refused["refuse"] is True, refused["verdict"]
    assert "host RAM" in refused["verdict"]
    _host(monkeypatch, floor + STEP)
    warned = go_cli.memory_gate(plan)
    assert warned["refuse"] is False, warned["verdict"]
    assert warned["preparation_warning"] is not None


def _check(capsys, config, *flags):
    code = cli.main(["check", str(config), "--free-gib", "15",
                     "--vram-gib", "16", *flags])
    return code, capsys.readouterr()


def test_check_refuses_only_a_preparation_whose_floor_exceeds_host_ram(
        resident_tiled_gfs, capsys, monkeypatch):
    floor, need = _preparation(resident_tiled_gfs)

    _host(monkeypatch, need + STEP)
    code, admitted = _check(capsys, resident_tiled_gfs)
    assert code == 0, admitted.err
    assert "preparing this configuration on the CPU" not in admitted.err

    _host(monkeypatch, floor + STEP)
    code, warned = _check(capsys, resident_tiled_gfs)
    assert code == 0, warned.err
    assert "WARNING: preparing this configuration on the CPU" in warned.err
    assert "REFUSED" not in warned.err

    _host(monkeypatch, floor - STEP)
    code, refused = _check(capsys, resident_tiled_gfs)
    assert code == 5, refused.err
    assert "REFUSED (exit 5)" in refused.err
    assert "preparing this configuration on the CPU" in refused.err

    code, skipped = _check(capsys, resident_tiled_gfs, "--no-host-memory-gate")
    assert code == 0, skipped.err
    assert "SKIPPED by --no-host-memory-gate" in skipped.err


def test_check_json_publishes_the_preparation_host_terms(
        resident_tiled_gfs, capsys, monkeypatch):
    floor, need = _preparation(resident_tiled_gfs)
    host = floor - STEP
    _host(monkeypatch, host)
    code, refused = _check(capsys, resident_tiled_gfs, "--json")
    assert code == 5, refused.err
    document = json.loads(refused.out[refused.out.index("{"):])
    assert document["ingest_host_preparation_bytes"] == need
    assert document["ingest_host_preparation_floor_bytes"] == floor
    assert document["host_ram_bytes"] == host
    assert "host RAM" in document["ingest_host_preparation_refusal"]
    assert document["ingest_host_preparation_warning"] is None


@pytest.fixture()
def nested_tiled_gfs(tmp_path):
    """A real 2-domain ``[tiles]`` GFS tree, the d2-tiled measured layout."""
    text = wizard.render_config(
        name="nested-prep-host", start_time=datetime(2026, 9, 27, 12),
        hours=6, projection=wizard._projection_entries(35.3, -97.5, "auto"),
        dims=[(206, 204), (474, 378)], ratios=(3,),
        fetch_hints={"source": "gfs"}, case_data=None, root_dx_m=9000.0,
        nz=49, tiles="auto")
    path = tmp_path / "nested-prep-host.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_check_reports_nested_cpu_measurement_ranges(
        nested_tiled_gfs, card, capsys, monkeypatch):
    exp = preflight._load_experiment_any(nested_tiled_gfs)
    assert len(exp.domains) == 2
    assert preflight.estimate_phases(
        exp, source="gfs",
        forcing_interval_seconds=10800.0).preprocess_backend == "cpu"
    _host(monkeypatch, 64 * GIB)
    code, reported = _check(capsys, nested_tiled_gfs, "--explain")
    assert code == 0, reported.err
    assert "+ NESTS (1 of them" in reported.out
    assert ("nested trees of 2 and 3 domains the floor came to 0.68 to 0.94 "
            "of the measured peak and the estimate to 1.01 to 1.12 of it"
            in reported.out)


def test_a_visible_forcing_decode_is_weighed_with_the_working_set(
        resident_tiled_gfs):
    exp = preflight._load_experiment_any(resident_tiled_gfs)
    phases = preflight.estimate_phases(
        exp, source="era5", preprocess_backend="cpu",
        forcing_interval_seconds=10800.0, source_grid_points=400_000,
        decoded_valid_times=3, source_fields_per_time=204)
    ingest = phases.ingest
    assert ingest.host_forcing_bytes > 0 and ingest.host_preprocess_bytes > 0
    need = phases.host_preparation_bytes
    floor = phases.host_preparation_floor_bytes
    assert need == ingest.host_forcing_bytes + ingest.host_preprocess_bytes
    assert floor == ingest.host_forcing_bytes + ingest.host_preprocess_floor_bytes
    assert phases.host_preparation_refusal(floor) is None
    refusal = phases.host_preparation_refusal(floor - 1)
    assert "the decoded forcing, the start time's analysis" in refusal

    unseen = preflight.estimate_phases(
        exp, source="era5", preprocess_backend="cpu",
        forcing_interval_seconds=10800.0)
    assert unseen.ingest.host_forcing_bytes is None
    assert unseen.host_preparation_bytes == unseen.ingest.host_preprocess_bytes
    assert (unseen.host_preparation_floor_bytes
            == unseen.ingest.host_preprocess_floor_bytes)
    assert "the forcing decode is held on top of it" in (
        unseen.host_preparation_refusal(unseen.host_preparation_floor_bytes - 1))


def test_the_card_road_holds_no_host_preparation(resident_tiled_gfs):
    exp = preflight._load_experiment_any(resident_tiled_gfs)
    phases = preflight.estimate_phases(
        exp, source="gfs", preprocess_backend="cuda",
        forcing_interval_seconds=10800.0)
    assert phases.ingest_envelope_bytes > 0
    assert phases.host_preparation_bytes == 0
    assert phases.host_preparation_floor_bytes == 0
    assert phases.host_preparation_refusal(1) is None
    assert phases.host_preparation_warning(1) is None


#: Real GFS preparations on the CPU road through the ``gpuwm go`` stages,
#: each a ``[tiles]`` config rendered as below, on the default install
#: with eight preparation threads:
#: (nx, ny, nz, forecast hours, the highest peak resident memory of the
#: preparation process tree seen for it, in bytes).  Repeated runs of one
#: configuration moved by up to 6%.
MEASURED_CPU_PREPARATIONS = (
    (744, 594, 76, 6, 10_829_660_160),
    (744, 594, 49, 6, 7_504_785_408),
    (902, 720, 76, 6, 15_315_107_840),
    (786, 628, 76, 6, 10_973_757_440),
    (474, 380, 49, 72, 5_048_508_416),
    (474, 380, 76, 72, 7_108_345_856),
    (600, 480, 96, 24, 8_943_951_872),
    (600, 480, 76, 72, 10_497_835_008),
)


@pytest.mark.parametrize("nx, ny, nz, hours, peak", MEASURED_CPU_PREPARATIONS)
def test_the_floor_stays_under_and_the_estimate_at_the_measured_peak(
        nx, ny, nz, hours, peak):
    text = wizard.render_config(
        name="measured", start_time=datetime(2026, 9, 26, 12), hours=hours,
        projection=wizard._projection_entries(35.3, -97.5, "auto"),
        dims=[(nx, ny)], ratios=(), fetch_hints={"source": "gfs"},
        case_data=None, root_dx_m=3000.0, nz=nz, tiles="auto")
    exp = wizard.experiment_from_text(text, source="<measured>")
    phases = preflight.estimate_phases(exp, source="gfs",
                                       forcing_interval_seconds=10800.0)
    assert phases.preprocess_backend == "cpu"
    # A refusal on the floor can never fire on a preparation that ran.
    assert phases.host_preparation_floor_bytes < peak
    # Sizing on the estimate never lands under the real peak, and gives
    # away little room above it: the device model carried onto the host
    # came to about 1.5 times the peak at 6 h.
    assert peak <= phases.host_preparation_bytes <= 1.12 * peak


#: Real GFS preparations on the CPU road, 6 h, 3 forcing times, ratio 3.
#: Eight CPU workers; peak is summed RSS of the preparation process tree,
#: sampled every 50 ms. Each row: name, dimensions, nz, tiled,
#: root_dx_m, peak_rss_bytes.
MEASURED_NESTED_CPU_PREPARATIONS = (
    ("d2-resident", ((206, 204), (474, 378)),
     49, False, 9000.0, 3_853_410_304),
    ("d2-tiled", ((206, 204), (474, 378)),
     49, True, 9000.0, 3_862_048_768),
    ("d3-resident", ((206, 204), (474, 378), (600, 480)),
     49, False, 9000.0, 8_415_719_424),
    ("d3-tiled", ((206, 204), (474, 378), (600, 480)),
     49, True, 9000.0, 8_388_251_648),
    ("deep-d2", ((474, 378), (474, 378)),
     76, True, 3000.0, 7_150_358_528),
    ("deep-d3", ((206, 204), (474, 378), (600, 480)),
     76, True, 9000.0, 11_934_109_696),
    ("equal-d2-resident", ((474, 378), (474, 378)),
     49, False, 3000.0, 5_110_476_800),
    ("equal-d2-tiled", ((474, 378), (474, 378)),
     49, True, 3000.0, 5_117_902_848),
    ("equal-d3-resident", ((474, 378), (474, 378), (474, 378)),
     49, False, 3000.0, 7_893_102_592),
    ("equal-d3-tiled", ((474, 378), (474, 378), (474, 378)),
     49, True, 3000.0, 7_956_701_184),
    ("root-wide-d2", ((474, 378), (240, 192)),
     49, True, 3000.0, 3_972_173_824),
    ("root-wide-d3", ((474, 378), (240, 192), (240, 192)),
     49, True, 3000.0, 4_370_173_952),
)


@pytest.mark.parametrize(
    "name, dimensions, nz, tiled, root_dx_m, peak_rss_bytes",
    MEASURED_NESTED_CPU_PREPARATIONS,
    ids=[row[0] for row in MEASURED_NESTED_CPU_PREPARATIONS])
def test_nested_cpu_preparation_bounds_measured_process_tree_peak(
        name, dimensions, nz, tiled, root_dx_m, peak_rss_bytes):
    """Completed CPU preparations must fit between the floor and estimate."""
    text = wizard.render_config(
        name=name, start_time=datetime(2026, 9, 27, 12), hours=6,
        projection=wizard._projection_entries(35.3, -97.5, "auto"),
        dims=list(dimensions), ratios=(3,) * (len(dimensions) - 1),
        fetch_hints={"source": "gfs"}, case_data=None,
        root_dx_m=root_dx_m, nz=nz, tiles="auto" if tiled else None)
    exp = wizard.experiment_from_text(text, source="<measured>")
    phases = preflight.estimate_phases(
        exp, source="gfs", preprocess_backend="cpu",
        forcing_interval_seconds=10800.0)
    peak = peak_rss_bytes
    assert phases.host_preparation_floor_bytes < peak
    assert phases.host_preparation_refusal(peak) is None
    assert peak <= phases.host_preparation_bytes <= 1.12 * peak


#: The thread count every measured row above was prepared with.
CALIBRATED_PREPARATION_WORKERS = 8

#: Real GFS preparations on the CPU road on a 64-vCPU host, 6 h, 3 forcing
#: times, prepared again with ``--preprocess-workers`` forced to each count
#: while all 64 CPUs stayed visible; peak is summed RSS of the preparation
#: process tree sampled every 50 ms, the highest of one to four runs at
#: each count (at eight, runs that named no count are included, since that
#: host then starts eight). Each row: name, dimensions, nz, tiled,
#: root_dx_m, workers, peak_rss_bytes.
MEASURED_CPU_PREPARATION_WORKER_COUNTS = (
    ("single", ((744, 594),), 49, True, 3000.0, 8, 7_623_692_288),
    ("single", ((744, 594),), 49, True, 3000.0, 16, 7_815_073_792),
    ("single", ((744, 594),), 49, True, 3000.0, 32, 8_289_067_008),
    ("single", ((744, 594),), 49, True, 3000.0, 64, 8_365_187_072),
    ("equal-d2-resident", ((474, 378), (474, 378)), 49, False, 3000.0,
     8, 5_292_220_416),
    ("equal-d2-resident", ((474, 378), (474, 378)), 49, False, 3000.0,
     64, 5_544_267_776),
)


def _measured_layout_phases(name, dimensions, nz, tiled, root_dx_m):
    text = wizard.render_config(
        name=name, start_time=datetime(2026, 9, 27, 12), hours=6,
        projection=wizard._projection_entries(35.3, -97.5, "auto"),
        dims=list(dimensions), ratios=(3,) * (len(dimensions) - 1),
        fetch_hints={"source": "gfs"}, case_data=None,
        root_dx_m=root_dx_m, nz=nz, tiles="auto" if tiled else None)
    exp = wizard.experiment_from_text(text, source="<measured>")
    return preflight.estimate_phases(
        exp, source="gfs", preprocess_backend="cpu",
        forcing_interval_seconds=10800.0)


def _host_cpus(monkeypatch, count):
    from gpuwm.ingest import preparation_workers
    monkeypatch.setattr(os, "sched_getaffinity",
                        lambda _pid: set(range(count)), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: count)
    monkeypatch.setattr(preparation_workers, "cgroup_cpu_count", lambda: None)
    monkeypatch.setattr(preparation_workers, "memory_worker_limit", lambda: 10**6)
    monkeypatch.delenv(preparation_workers.PREPARATION_THREADS_ENV, raising=False)
    # ac988f1fd made automatic_workers() reuse its answer for
    # AUTOMATIC_WORKERS_REUSE_SECONDS inside one process. A simulated host is
    # a different machine, so an answer read on the previous one (the gate
    # box's own 24 CPUs, measured) must not carry over into this one.
    monkeypatch.setattr(cpu_backend, "_AUTOMATIC_WORKERS", {})


@pytest.fixture(autouse=True)
def calibrated_worker_machine(monkeypatch):
    """Historical peak rows were measured at eight workers unless named."""
    _host_cpus(monkeypatch, CALIBRATED_PREPARATION_WORKERS)


@pytest.mark.parametrize("host_cpus", (1, 4, 8, 16, 64, 192))
def test_the_cpu_preparation_uses_the_available_worker_budget(
        host_cpus, monkeypatch):
    _host_cpus(monkeypatch, host_cpus)
    automatic = host_cpus
    assert cpu_backend._workers(None, 10 ** 9) == automatic
    assert real._default_column_workers("cpu", None) == automatic
    resolved = SimpleNamespace(name="cpu", workers=None)
    assert real._default_column_workers(resolved, None) == automatic
    # A count the caller names is kept, and the setup columns follow it.
    assert cpu_backend._workers(32, 10 ** 9) == min(32, host_cpus)
    assert real._default_column_workers("cpu", 32) == min(32, host_cpus)
    assert real._default_column_workers(
        SimpleNamespace(name="cpu", workers=32), None) == min(32, host_cpus)
    # The device road's host steps are not priced here and keep every CPU.
    assert real._default_column_workers("cuda", None) == host_cpus


@pytest.mark.parametrize("host_cpus", (1, 8, 16, 64, 192))
@pytest.mark.parametrize(
    "name", sorted({row[0] for row in MEASURED_CPU_PREPARATION_WORKER_COUNTS}))
def test_the_estimate_holds_at_the_worker_count_the_host_would_use(
        name, host_cpus, monkeypatch):
    _host_cpus(monkeypatch, host_cpus)
    workers = cpu_backend._workers(None, 10 ** 9)
    rows = [row for row in MEASURED_CPU_PREPARATION_WORKER_COUNTS
            if row[0] == name]
    peaks = {row[5]: row[6] for row in rows}
    # The peak rises with the thread count, so the nearest measured count
    # at or above the host's bounds what that host holds.
    measured = [count for count in peaks if count >= workers]
    if not measured:
        # This is a scratch-reservation check above the largest measured
        # width, not a claim that a wider real run was measured.
        from gpuwm.ingest.preparation_workers import WORKER_SCRATCH_BYTES
        peak = peaks[max(peaks)] + (workers - max(peaks)) * WORKER_SCRATCH_BYTES
    else:
        peak = peaks[min(measured)]
    phases = _measured_layout_phases(*rows[0][:5])
    assert peak <= phases.host_preparation_bytes


@pytest.mark.parametrize(
    "name, dimensions, nz, tiled, root_dx_m, workers, peak_rss_bytes",
    MEASURED_CPU_PREPARATION_WORKER_COUNTS,
    ids=[f"{row[0]}-{row[5]}" for row in MEASURED_CPU_PREPARATION_WORKER_COUNTS])
def test_the_floor_stays_under_the_peak_at_every_worker_count(
        name, dimensions, nz, tiled, root_dx_m, workers, peak_rss_bytes):
    phases = _measured_layout_phases(name, dimensions, nz, tiled, root_dx_m)
    assert phases.host_preparation_floor_bytes < peak_rss_bytes
    assert phases.host_preparation_refusal(peak_rss_bytes) is None
