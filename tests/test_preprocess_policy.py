"""The memory estimate and executable preparation policy describe one road."""
from dataclasses import asdict, replace
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gpuwm.core import preflight as pf, streaming as st
from gpuwm.preprocess_policy import resolve_preprocess_backend
from test_prepared_tile_memory import experiment, profile
from tilestream.autoplan import GIB, Machine


@pytest.mark.parametrize("source,mode,store,expected", [
    ("gfs", "auto", "host", "cpu"), ("gfs", "on", "host", "cpu"),
    # Unrequested and not host-tiled the road is auto: the door prices its
    # preparation and prepares on the card only when that fits (A65).
    ("gfs", "off", "host", "auto"), ("gfs", "auto", "device", "auto"),
    ("hrrr", "auto", "host", "auto"), ("era5", "auto", "host", "auto"),
])
def test_raw_and_validated_effective_policy_agree(source, mode, store, expected):
    exp = experiment(874, 574, mode=mode, store=store)
    assert resolve_preprocess_backend(source=source, experiment=exp) == expected
    raw_tables = {"tiles": exp.tiles.to_mapping(), "domain": [{}]}
    assert resolve_preprocess_backend(source=source, tables=raw_tables) == expected


@pytest.mark.parametrize("backend", ["cpu", "cuda", "auto"])
def test_explicit_backend_is_preserved(backend):
    assert resolve_preprocess_backend(source="gfs", experiment=experiment(), requested=backend) == backend
    assert resolve_preprocess_backend(source="era5", requested=backend) == backend


def test_per_domain_replacement_semantics_match_the_streaming_owner():
    tables = {"tiles": {"mode": "auto", "store": "host"},
              "domain": [{"grid_id": 1, "tiles": {"mode": "off"}}]}
    assert resolve_preprocess_backend(source="gfs", tables=tables) == "auto"
    tables["domain"].append({"grid_id": 2, "tiles": {"mode": "auto", "store": "device"}})
    assert resolve_preprocess_backend(source="gfs", tables=tables) == "auto"
    tables["domain"].append({"grid_id": 3})
    assert resolve_preprocess_backend(source="gfs", tables=tables) == "cpu"


def test_policy_import_needs_only_the_standard_library():
    root = Path(__file__).resolve().parents[1]
    code = ("import sys; sys.path.insert(0,sys.argv[1]); "
            "from gpuwm.preprocess_policy import resolve_preprocess_backend; "
            "assert resolve_preprocess_backend(source='gfs',tables={'tiles':{'mode':'auto'},'domain':[{}]})=='cpu'; "
            "assert not any(name.split('.')[0] in ('numpy','cupy') for name in sys.modules)")
    done = subprocess.run([sys.executable, "-S", "-c", code, str(root)],
                          capture_output=True, text=True, env=dict(os.environ, GPUWM_NO_LOCAL_GPU="1"))
    assert done.returncode == 0, done.stdout + done.stderr


def test_actual_874_by_574_geometry_keeps_science_and_moves_only_ingest_off_gpu():
    exp = experiment(874, 574)
    original = asdict(exp.root.run)
    machine = Machine(int(6.20 * GIB), 96 * GIB, device_profile=profile())
    cpu = pf.estimate_phases(exp, source="gfs", profile=profile(), machine=machine)
    cuda = pf.estimate_phases(exp, source="gfs", profile=profile(), machine=machine,
                              preprocess_backend="cuda")
    assert cpu.preprocess_backend == cpu.ingest.preprocess_backend == "cpu"
    assert cpu.ingest_envelope_bytes == cpu.ingest.peak_envelope_bytes == 0
    assert cpu.ingest.context_bytes == cpu.ingest.device_overhead_bytes == 0
    # The itemized envelope beneath the Windows probe overhead: 11.55 GiB
    # under the retired 0.65-of-one-time transient (proof/node-reds-276),
    # 8.87 GiB with the setup itemized and measured (A65).  Either way
    # over this 6.2 GiB card, which is what moves the ingest off it.
    assert cuda.ingest_envelope_bytes - cuda.ingest.device_overhead_bytes > 8.8 * GIB
    assert cuda.ingest_envelope_bytes > machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    assert cpu.forecast_envelope_bytes == cuda.forecast_envelope_bytes
    assert cpu.forecast_envelope_bytes <= machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    assert cpu.peak_envelope_bytes == cpu.forecast_envelope_bytes
    assert cpu.ingest.items == cuda.ingest.items
    # Host RAM is priced with the host's own terms: the arrays held at once
    # plus calibrated multiples of the analysis they are built around,
    # never the CUDA pool's transient and allocator headroom.
    floor = cpu.ingest.host_preprocess_floor_bytes
    assert 0 < floor < cpu.ingest.host_preprocess_bytes
    analysis = cpu.ingest.category_bytes("analysis")
    assert cpu.ingest.tree_analysis_bytes == analysis > 0
    assert cpu.ingest.host_preprocess_bytes == floor + math.ceil(
        pf.CPU_PREPARATION_ANALYSIS_MULTIPLE * analysis
        + pf.CPU_PREPARATION_ANALYSIS_MULTIPLE_PER_INTERVAL
        * (cpu.ingest.n_forcing_times - 1) * analysis)
    assert cpu.ingest.host_preprocess_bytes < (
        cpu.ingest.alloc_estimate_bytes + cpu.ingest.boundary_frame_bytes)
    assert cuda.ingest.host_preprocess_bytes == cuda.ingest.host_preprocess_floor_bytes == 0
    # Native GFS decoder retention has no measured row; unknown RAM cannot
    # become zero merely because its interpolation runs on the CPU.
    assert cpu.ingest.host_forcing_bytes is None
    assert cpu.ingest.host_peak_estimate_bytes is None
    assert asdict(exp.root.run) == original


def test_known_decoded_host_bytes_are_preserved_and_cpu_working_set_is_added():
    exp = experiment(80, 64)
    kwargs = dict(source="era5", source_grid_points=10000, decoded_valid_times=3,
                  source_fields_per_time=204, profile=profile())
    cpu = pf.estimate_ingest(exp, preprocess_backend="cpu", **kwargs)
    cuda = pf.estimate_ingest(exp, **kwargs)
    assert cpu.host_forcing_bytes == cuda.host_forcing_bytes == 8 * 10000 * 3 * 204 * 2
    assert cpu.host_peak_estimate_bytes == cpu.host_forcing_bytes + cpu.host_preprocess_bytes
    assert cuda.host_peak_estimate_bytes == cuda.host_forcing_bytes
    assert cpu.peak_envelope_bytes == 0 < cuda.peak_envelope_bytes


def test_explicit_auto_keeps_the_conservative_device_estimate():
    exp = experiment(80, 64)
    auto = pf.estimate_phases(exp, source="gfs", preprocess_backend="auto")
    cuda = pf.estimate_phases(exp, source="gfs", preprocess_backend="cuda")
    assert auto.preprocess_backend == "auto"
    assert auto.ingest_envelope_bytes == cuda.ingest_envelope_bytes > 0


def test_bad_explicit_backend_cannot_zero_a_device_estimate():
    with pytest.raises(ValueError, match="backend"):
        pf.estimate_phases(experiment(), source="gfs", preprocess_backend="automatic")


def test_the_met_em_preparation_road_follows_the_tiles_declaration():
    """A host-store [tiles] declaration prepares met_em on the CPU, like GFS.

    The per-source `if` this replaced short-circuited every source but one
    BEFORE the [tiles] test, so a met_em run that asked to tile was priced
    with its whole initialization resident on the card and refused on it.
    The controls below are what keeps the table a table: another source
    with the same experiment, and the same source with no host store.
    """
    tiled = experiment(874, 574, mode="on", store="host")
    assert resolve_preprocess_backend(source="met_em", experiment=tiled) == "cpu"
    assert resolve_preprocess_backend(source="MET_EM", experiment=tiled) == "cpu"
    from gpuwm.preprocess_policy import CPU_PREPARED_SOURCES

    assert "met_em" in CPU_PREPARED_SOURCES
    # Controls in the same test: nothing became cpu wholesale.
    assert resolve_preprocess_backend(source="era5", experiment=tiled) == "auto"
    assert resolve_preprocess_backend(source="hrrr", experiment=tiled) == "auto"
    assert resolve_preprocess_backend(
        source="met_em", experiment=experiment(874, 574, mode="off")) == "auto"
    assert resolve_preprocess_backend(
        source="met_em",
        experiment=experiment(874, 574, mode="on", store="device")) == "auto"
    # An explicit request still wins on this source, as on every other.
    assert resolve_preprocess_backend(
        source="met_em", experiment=tiled, requested="cuda") == "cuda"


def test_the_met_em_run_door_resolves_its_road_through_the_same_policy():
    """The admission and the run ask one function (no 'cuda' literal)."""
    import inspect

    from gpuwm import metem_forecast

    signature = inspect.signature(metem_forecast.prepare_metem_run)
    assert signature.parameters["preprocess_backend"].default is None
    source = inspect.getsource(metem_forecast.prepare_metem_run)
    assert "gpuwm.preprocess_policy" in source
