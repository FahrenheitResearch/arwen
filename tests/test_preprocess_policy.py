"""The memory estimate and executable preparation policy describe one road."""
from dataclasses import asdict, replace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gpuwm.core import preflight as pf, streaming as st
from gpuwm.preprocess_policy import resolve_preprocess_backend
from tests.test_prepared_tile_memory import experiment, profile
from tilestream.autoplan import GIB, Machine


@pytest.mark.parametrize("source,mode,store,expected", [
    ("gfs", "auto", "host", "cpu"), ("gfs", "on", "host", "cpu"),
    ("gfs", "off", "host", "cuda"), ("gfs", "auto", "device", "cuda"),
    ("hrrr", "auto", "host", "cuda"), ("era5", "auto", "host", "cuda"),
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
    assert resolve_preprocess_backend(source="gfs", tables=tables) == "cuda"
    tables["domain"].append({"grid_id": 2, "tiles": {"mode": "auto", "store": "device"}})
    assert resolve_preprocess_backend(source="gfs", tables=tables) == "cuda"
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
    assert cuda.ingest_envelope_bytes > 12 * GIB
    assert cuda.ingest_envelope_bytes > machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    assert cpu.forecast_envelope_bytes == cuda.forecast_envelope_bytes
    assert cpu.forecast_envelope_bytes <= machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    assert cpu.peak_envelope_bytes == cpu.forecast_envelope_bytes
    assert cpu.ingest.items == cuda.ingest.items
    assert cpu.ingest.host_preprocess_bytes == cpu.ingest.alloc_estimate_bytes + cpu.ingest.boundary_frame_bytes > 0
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
