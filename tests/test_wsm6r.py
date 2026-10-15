"""WSM6R (mp_physics = 106): the host-side facts its exact port rests on.

CPU only.  The GPU gate is tests/test_wsm6r_wrf461_column_exact.py; this
file holds the pieces that gate cannot see from inside a kernel launch:

* the frozen wsm6rinit words are the words WRF's own init computed (the
  committed dump), in the order kernels/wsm6r.cu's enum reads them;
* the oracle column set is the input WRF ran, and it spans the regimes;
* the -O0 reference pulled no glibc vector-math symbol;
* mp_physics = 106 is admitted by TABLE rows (config, importer, registry,
  loader grant) and dispatched to the WSM6 family module, with no adapter
  module of its own.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "oracles" / "wsm6r"
KERNEL = ROOT / "gpuwm" / "core" / "kernels" / "wsm6r.cu"


def _dump() -> dict[str, int]:
    out = {}
    for line in (FIXTURE / "wsm6r-init-f32.txt").read_text().splitlines():
        name, bits, _ = line.split()
        out[name] = int(bits, 16)
    return out


def test_frozen_init_words_are_the_words_wrf_computed():
    """Breakage prevented: a wsm6rinit constant recomputed on the host (a
    float64 Gamma product) instead of copied from WRF's float32 init."""
    from gpuwm.core.wsm6_constants import WSM6R_INIT_F32, WSM6R_RADII_F32

    dump = _dump()
    for name, bits in WSM6R_INIT_F32 + WSM6R_RADII_F32:
        assert dump[name] == bits, name
    assert [n for n, _ in WSM6R_INIT_F32 + WSM6R_RADII_F32] == list(dump)


def test_kernel_enum_reads_the_frozen_words_in_their_order():
    """Breakage prevented: an enum entry added or moved in wsm6r.cu without
    the host table, which shifts every later coefficient by one word."""
    from gpuwm.core.wsm6_constants import WSM6R_INIT_F32

    text = KERNEL.read_text(encoding="utf-8")
    body = re.search(r"enum Wsm6rCoef \{(.*?)\};", text, re.S).group(1)
    enum = re.findall(r"\b(R_[A-Z0-9_]+|WSM6R_NCOEF)\b", body)
    assert enum[-1] == "WSM6R_NCOEF"
    assert [e[2:].lower() for e in enum[:-1]] == [n for n, _ in WSM6R_INIT_F32]


def test_the_column_set_is_the_oracle_input_and_spans_the_regimes():
    """Breakage prevented: a fixture regenerated from a different column set
    than the one WRF ran, or one that lost a regime or its edge columns."""
    meta = json.loads((FIXTURE / "columns.json").read_text())
    digests = json.loads((FIXTURE / "wrf461-step-digests.json").read_text())
    assert hashlib.sha256((FIXTURE / "columns.bin").read_bytes()).hexdigest() \
        == digests["columns_bin_sha256"]
    assert meta["ncol"] >= 64 and meta["nz"] == 45
    regimes = {c["regime"] for c in meta["columns"]}
    for needed in ("deep convection", "stratiform melting layer",
                   "freezing rain", "stable clear night", "radiation fog",
                   "ocean stratocumulus", "winter snow", "high terrain",
                   "cirrus", "supercooled below -40C"):
        assert needed in regimes, needed
    assert sum(r.startswith("edge:") for r in regimes) >= 12
    assert set(digests["runs"]) == {"dt30-n12", "dt60-n6", "dt150-n3",
                                    "dt240-n3", "dt300-n4"}


def test_the_reference_was_built_without_vector_libm():
    report = (FIXTURE / "libmvec-report.txt").read_text()
    o0 = report.split("# -O0 (THE REFERENCE)")[1].split("# -O2")[0]
    assert "_ZGV" not in o0
    assert "_ZGV" in report.split("# positive control:")[1]


def test_minor_loop_count_is_wrf_nint():
    """RAINNCV is graded word for word exactly on single-loop runs, so the
    loop count the grader assumes must be WRF's nint (half away from zero)."""
    from tools.wsm6r_wrf461_column_oracle.run_gpu import minor_loops

    assert [minor_loops(dt) for dt in (30.0, 60.0, 150.0, 180.0, 240.0, 300.0)] \
        == [1, 1, 1, 2, 2, 3]


def test_106_is_admitted_by_table_rows_not_an_adapter_module():
    """Breakage prevented: a scheme wired through a per-scheme adapter file,
    or accepted by the loader and missing from a table a run reads."""
    from gpuwm.config import MP_PHYSICS_ACCEPTED
    from gpuwm.core.kernels import EXTRA_HEADERS, module_options
    from gpuwm.core.wsm6_constants import WSM6R_INIT_F32
    import gpuwm.namelist_import as importer

    assert 106 in MP_PHYSICS_ACCEPTED
    assert importer._MP_MAP[106] == (106, "WSM6R", "WSM6R")
    assert EXTRA_HEADERS["wsm6r"] == ("glibc_flt32.cuh", "glibc_flt64.cuh")
    assert module_options("wsm6r") == ("-std=c++17", "--fmad=false", "--ftz=false")
    assert not (ROOT / "gpuwm" / "core" / "wsm6r.py").exists()
    registry = json.loads((ROOT / "gpuwm" / "physics_registry_v2.json")
                          .read_text(encoding="utf-8"))
    row = registry["components"]["microphysics"]["options"]["wsm6r-mp106"]
    assert row["selectors"] == {"mp_physics": 106}
    assert row["implemented"] is True
    assert row["maturity"] == "implemented-unverified"
    assert row["consumers"]["cloud_optics"]["legacy_declares_radii"] is False
    assert row["consumers"]["ring_guard"]["surface_slots"] == [
        "mp_rainnc", "mp_rainncv", "mp_sr"]
    assert len(WSM6R_INIT_F32) == 60


def test_microphysics_dispatches_106_to_the_wsm6_family_module():
    text = (ROOT / "gpuwm" / "core" / "microphysics.py").read_text(
        encoding="utf-8")
    assert "elif cfg.mp_physics in (6, 106):" in text
    family = (ROOT / "gpuwm" / "core" / "wsm6.py").read_text(encoding="utf-8")
    assert "106: {\"label\": \"WSM6R\"" in family


def test_wsm6r_has_no_scheme_sr_so_the_land_surface_takes_the_proxy():
    """WRF's wsm6r writes no SR; gpuwm must not hand Noah a zero as if it
    were the scheme's frozen fraction."""
    from gpuwm.core.physics import microphysics_scheme_sr_available

    assert microphysics_scheme_sr_available(106) is False

def test_public_namelist_import_preserves_106_and_physics(tmp_path):
    import tomllib
    from gpuwm.namelist_import import import_namelists
    from gpuwm.experiment import load_experiment
    wps = ROOT / "configs/wsm6r_column_forecast.namelist.wps"
    inp = ROOT / "configs/wsm6r_column_forecast.namelist.input"
    text, report = import_namelists(wps, inp, rrtmg_variant="rrtmg_legacy")
    path = tmp_path / "imported.toml"
    path.write_text(text)
    exp = load_experiment(path)
    assert [d.run.mp_physics for d in exp.domains] == [106]
    assert exp.domains[0].run.ra_rrtmg_variant == "rrtmg_legacy"
    assert exp.domains[0].run.nx == 50
    assert exp.domains[0].run.nz == 49
    assert exp.domains[0].run.dt == 18.0


def test_wsm6r_transition_exclusion_is_the_runtime_citation():
    from gpuwm.core.microphysics_transition import (
        MIXED_EDGE_ABSENCES, resolve_microphysics_transition)
    from gpuwm.config import RunConfig
    from dataclasses import replace
    base = RunConfig(nx=16, ny=16, nz=4, dx=3000., dy=3000.,
                     ztop=10000., dt=3., run_seconds=9.,
                     moist=True, mp_physics=106)
    same = resolve_microphysics_transition(base, replace(base, nested=True))
    assert not same.mixed
    assert "106" in MIXED_EDGE_ABSENCES[106]
    with __import__("pytest").raises(ValueError, match="ported selectors"):
        resolve_microphysics_transition(base, replace(base, mp_physics=6))

def test_analyzed_mass_and_source_absent_contract_admit_106():
    from types import SimpleNamespace
    from gpuwm.ingest.real import (
        HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE,
        HRRR_ANALYZED_HYDROMETEORS,
        HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS)
    from gpuwm.ingest.microphysics_cold_start import source_absent_microphysics
    assert 106 in HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS
    assert HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE[106]["retained"] == \
        HRRR_ANALYZED_HYDROMETEORS
    assert source_absent_microphysics(SimpleNamespace(mp_physics=106)) == ((), {})


def test_legacy_radiation_reads_wsm6r_ice_and_snow_membership():
    from gpuwm.core.rrtmg_legacy import (
        legacy_cloud_fraction_flags, legacy_ice_active, _MP_DECLARES_RADII)
    assert legacy_cloud_fraction_flags(106) == (True, True)
    assert legacy_ice_active(106)
    assert _MP_DECLARES_RADII[106] is False


def test_wsm6r_per_call_work_is_rebuilt_for_restart_and_final_digest():
    from gpuwm.config import RunConfig
    from gpuwm.core.preflight import scratch_slot_registry
    from gpuwm.io.restart import classify_scratch_slot
    cfg = RunConfig(nx=50, ny=50, nz=49, dx=3000., dy=3000.,
                    ztop=20000., dt=18., run_seconds=720.,
                    moist=True, mp_physics=106)
    slots = {n for n in scratch_slot_registry(cfg) if n.startswith("wsm6r_")}
    assert slots == {"wsm6r_theta", "wsm6r_rho", "wsm6r_pii",
                     "wsm6r_dz", "wsm6r_z8w", "wsm6r_work"}
    assert all(classify_scratch_slot(n) == "rebuild" for n in slots)
    assert classify_scratch_slot("mp_rainnc") == "serialize"
