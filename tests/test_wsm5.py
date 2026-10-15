"""WSM5 (mp_physics = 4) against WRF v4.6.1's own build, word for word.

THE BREAKAGE THIS PREVENTS
--------------------------
Before this file WOOF had no WSM5: ``import-namelist`` refused
mp_physics = 4 and a WRF namelist that chose it could not run.  The port is
kernels/wsm5.cu, admitted by one table row (gpuwm.core.microphysics.
COLUMN_SCHEMES), and this file holds it to WRF's words so any edit that
moves one word away from them goes red here.

THE ORACLE (tests/data/oracles/wsm5, built by
tools/wsm5_wrf461_column_oracle/build.sh on the sprint CPU box): a column
driver linked against the WRF v4.6.1 build trees' OWN module_mp_wsm5.o and
module_mp_radar.o (gfortran 13.3.0 -O2 -fno-fast-math -ffp-contract=off,
the scalar tree adding -fno-tree-vectorize: the referee), calling WRF's
entry ``module_mp_wsm5::wsm5`` after ``wsm5init`` with WRF's constants
module, REFL_10CM requested and the radiation radii on.  Two column sets:
``main`` (66 columns x 45 levels: warm rain, mixed phase through a melting
layer, deep convection, deep cold, saturated, dry, sedimentation-heavy,
high terrain and ten edge columns) at dt 30, 150, 300, 600 and 1800 s
(1 to 15 minor loops), and ``lev2`` (16 columns x 2 levels) at dt 0.001, 1,
20 and 300 s.

TWO DECLARED DIVERGENCES, both default-on (kernels/wsm5.cu head):
the sedimentation remap's corrected top search (lane 073's water-loss
defect: WRF deletes the condensate of every target cell its search skips)
and psevp, the evaporation of melting snow, which WRF's WSM5 computes from
a psdep that is always zero where it runs.  The oracle's DEFINED arm is
WRF's own preprocessed source with exactly those two edits
(tools/wsm5_wrf461_column_oracle/patch_arms.py), compiled with the scalar
tree's own compile line.

THE CLAIMS
----------
* strict build (GPUWM_WRF_EXACT=1) and default build alike: every word of
  every step of every run equals the defined arm's (SHA-256 per field per
  step, the final step word by word), at both WSM5_KMAX tiers;
* against stock WRF the differing columns are EXACTLY the probe columns of
  the two fixes (value-changing remap fall-throughs, non-zero corrected
  psevp), on every run;
* lane 073's transport-only remap control closes to 1e-6 at 2, 50 and 80
  levels through the production device remap, where stock WRF loses up to
  100 % (2 levels) and 0.2 kg m-2 (50 and 80);
* a planted one-ULP change to one kernel literal fails the gate.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "oracles" / "wsm5"
TOOL = "tools.wsm5_wrf461_column_oracle.run_gpu"


def _dump_words() -> dict:
    out: dict = {}
    for line in (FIXTURE / "wsm5-init-f32.txt").read_text().splitlines():
        parts = line.split()
        out.setdefault(parts[0], []).append(int(parts[1], 16))
    return out


def test_frozen_init_and_radar_words_are_the_words_wrf_computed():
    """The kernel's constants are wsm5init's and radar_init's own words.

    Breakage prevented: a constant recomputed on the host (float64 Gamma,
    an exact 720 for WGAMMA(7.)) instead of copied from WRF's float32 init.
    """
    from gpuwm.core.wsm6_constants import (WSM5_INIT_F32, WSM5_RADAR_F32,
                                           WSM5_RADAR_F64, wsm5_init_words,
                                           wsm5_radar_words)
    dump = _dump_words()
    for name, bits in WSM5_INIT_F32:
        assert dump[name] == [bits], name
    indexed = {"xcre1": ("xcre", 0), "xcre4": ("xcre", 3), "xcrg3": ("xcrg", 2),
               "xcrg4": ("xcrg", 3), "xcse1": ("xcse", 0), "xcse4": ("xcse", 3),
               "xcsg3": ("xcsg", 2), "xcsg4": ("xcsg", 3)}
    for name, bits in WSM5_RADAR_F32:
        key, i = indexed.get(name, (name, 0))
        assert dump[key][i] == bits, name
    scalars = [dump[n][0] for n in ("k_w", "pi5", "lamda4", "m_w_0_re",
                                    "m_w_0_im", "m_i_0_re", "m_i_0_im")]
    bins = dump["xxds"] + dump["xdts"] + dump["simpson"]
    assert [bits for _, bits in WSM5_RADAR_F64] == scalars + bins
    assert len(wsm5_init_words()) == 24
    f32, f64 = wsm5_radar_words()
    assert (len(f32), len(f64)) == (15, 7 + 50 + 50 + 51)
    # pvtr is WRF's float32 rgmma product, not a float64 Gamma value.
    assert float(wsm5_init_words()[7]) == pytest.approx(2500.0637, abs=1e-4)


def test_the_column_sets_are_the_oracle_inputs_and_span_the_families():
    meta = {s: json.loads((FIXTURE / f"{s}.json").read_text()) for s in ("main", "lev2")}
    digests = json.loads((FIXTURE / "wrf461-step-digests.json").read_text())
    for s in ("main", "lev2"):
        assert hashlib.sha256((FIXTURE / f"{s}.bin").read_bytes()).hexdigest() \
            == digests["inputs_sha256"][f"{s}.bin"]
    assert meta["main"]["ncol"] >= 64 and meta["main"]["nz"] == 45
    assert meta["lev2"]["nz"] == 2
    families = {c["family"] for c in meta["main"]["columns"]}
    for needed in ("warm rain", "mixed phase", "deep convection", "deep cold",
                   "saturated", "dry", "sedimentation-heavy", "high terrain"):
        assert needed in families, needed
    assert sum(f.startswith("edge:") for f in families) >= 10
    runs = set(digests["runs"])
    assert "main-dt1800-n1" in runs and "lev2-dt0.001-n3" in runs


def test_the_referee_is_the_scalar_build_without_vector_libm():
    report = (FIXTURE / "libmvec-report.txt").read_text()
    scalar = report.split("# scalar tree (THE REFERENCE)")[1].split("# strict tree")[0]
    assert "_ZGV" not in scalar
    assert "-fno-tree-vectorize" in (FIXTURE / "compile-flags.txt").read_text()


def test_every_difference_between_the_wrf_arms_is_attributed():
    """The Fortran side of the claim, recorded by compare_arms.py.

    Breakage prevented: a defined arm that changed more than its two named
    edits, or a probe that names a different set than the words moved.
    """
    arms = json.loads((FIXTURE / "arms.json").read_text())
    assert arms["failures"] == []
    for tag, row in arms["runs"].items():
        assert row["control_scalar_vs_stock_recompiled"] == 0, tag
        assert row["stock_nonzero_psevp_columns"] == [], tag
        for fix in ("remap", "psevp", "defined"):
            assert row[f"{fix}_vs_stock"]["equal"], (tag, fix)
    stock = {(r["nz"], r["dt"]): r for r in arms["remap_probe"]["scalar"]}
    defined = {(r["nz"], r["dt"]): r for r in arms["remap_probe"]["defined"]}
    for key, row in defined.items():
        assert row["relative_error"] < 1.0e-6, key
        assert stock[key]["lost_kg_m2"] > 0.17, key


def test_the_table_row_admits_wsm5_with_its_species():
    from gpuwm.core.microphysics import COLUMN_SCHEMES
    from gpuwm.core.kernels import UNFLUSHED_OPTIONS, EXTRA_HEADERS

    row = COLUMN_SCHEMES[4]
    assert row.species == ("qv", "qc", "qi", "qr", "qs")
    assert row.unit == "wsm5" and EXTRA_HEADERS["wsm5"] == (
        "glibc_flt32.cuh", "glibc_flt64.cuh")
    assert UNFLUSHED_OPTIONS == ("-std=c++17", "--fmad=false", "--ftz=false")
    assert row.level_tier(2) == 64 and row.level_tier(80) == 80
    with pytest.raises(ValueError):
        row.level_tier(1)


def test_a_bare_namelist_with_mp_physics_4_imports_to_wsm5():
    """mp_physics = 4 is a translation 4 -> 4, not a substitution."""
    from gpuwm.namelist_import import _MP_MAP
    assert _MP_MAP[4][0] == 4


def _run(extra_env: dict, *args) -> dict:
    env = dict(os.environ)
    env.pop("GPUWM_WRF_EXACT", None)
    env.update(extra_env)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "tests"), env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, "-m", TOOL, *args], cwd=str(ROOT),
                          env=env, capture_output=True, text=True, timeout=1800)
    lines = proc.stdout.strip().splitlines()
    assert lines, proc.stderr[-4000:]
    return json.loads(lines[-1])


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("tier", (64, 80))
def test_strict_build_is_the_defined_arm_on_every_word_of_every_step(tier):
    result = _run({"GPUWM_WRF_EXACT": "1"}, "check-fixture", str(FIXTURE),
                  "--tier", str(tier))
    assert result["strict_build"] is True
    assert result["total_mismatch"] == 0, result


@pytest.mark.gpu
@requires_gpu
def test_default_build_is_the_defined_arm_on_every_word_of_every_step():
    """The arithmetic users run: unflushed and uncontracted, so no column
    is excluded (WSM6's default build still flushes subnormals)."""
    result = _run({}, "check-fixture", str(FIXTURE))
    assert result["strict_build"] is False
    assert result["total_mismatch"] == 0, result


@pytest.mark.gpu
@requires_gpu
def test_a_planted_one_ulp_change_is_caught():
    result = _run({}, "plant", str(FIXTURE))
    assert result["total_mismatch"] > 0, result


@pytest.mark.gpu
@requires_gpu
def test_the_remap_transport_control_closes_at_2_50_and_80_levels():
    result = _run({}, "remap-probe")
    assert result["total_mismatch"] == 0, result


def test_planted_literal_changes_exactly_one_bit():
    import numpy as np
    from tools.wsm5_wrf461_column_oracle.run_gpu import plant_literal
    old, new = plant_literal()
    before = int(np.float32(float(old[:-1])).view(np.uint32))
    after = int(np.float32(float(new[:-1])).view(np.uint32))
    assert before ^ after == 1
    assert "effc[id] = re_qc * " + old in (ROOT / "gpuwm/core/kernels/wsm5.cu").read_text()


def test_five_class_state_allocation_transport_and_pricing(monkeypatch):
    import numpy as np
    from gpuwm.config import RunConfig
    from gpuwm.core import state as sm
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.core.moist import moist_species
    from gpuwm.core.preflight import scratch_slot_registry
    monkeypatch.setattr(sm, "cp", np)
    cfg = RunConfig(nx=8, ny=6, nz=4, dx=1000., dy=1000., dt=1.,
                    ztop=10000., run_seconds=10., moist=True, mp_physics=4)
    state = sm.DomainState(cfg)
    assert set(moist_species(state)) == {"qv", "qc", "qr", "qi", "qs"}
    assert getattr(state, "qg", None) is None and getattr(state, "qg0", None) is None
    declared = state_array_shapes(cfg)
    assert all(n in declared for n in ("qi", "qs", "qi0", "qs0", "effc", "effi", "effs"))
    assert "qg" not in declared
    slots = scratch_slot_registry(cfg)
    assert "moist_absent_mass" in slots
    assert all(n in slots for n in ("mpcol_theta", "mpcol_z8w", "mp_rainnc", "mp_snownc"))


def test_registry_and_wrf_input_door_describe_the_same_five_species():
    from gpuwm.physics_registry import physics_registry
    from gpuwm.ingest.wrfinput import SUPPORTED_MICROPHYSICS, UNSUPPORTED_MICROPHYSICS
    from gpuwm.wrf_physics_inventory import stock_wrf_physics_inventory
    row = physics_registry()["components"]["microphysics"]["options"]["wsm5-mp4"]
    assert row["selectors"] == {"mp_physics": 4}
    assert set(row["consumers"]["moments"]["mass_only"]) == {"qv", "qc", "qr", "qi", "qs"}
    assert len(row["extensions"]["declared_divergences"]) == 2
    assert all(d["default_on"] for d in row["extensions"]["declared_divergences"])
    assert SUPPORTED_MICROPHYSICS[4] == "WSM5" and 4 not in UNSUPPORTED_MICROPHYSICS


def test_bare_namelist_import_runs_the_public_cli(tmp_path):
    from test_namelist_import import INPUT_TEXT, _pair
    from gpuwm.cli import main
    from gpuwm.experiment import load_experiment
    inp = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 4, 4")
    wps, namelist = _pair(tmp_path, inp=inp)
    out = tmp_path / "column.toml"
    assert main(["import-namelist", str(wps), str(namelist), "--output", str(out)]) == 0
    exp = load_experiment(out)
    assert all(d.run.mp_physics == 4 for d in exp.domains)


@pytest.mark.gpu
@requires_gpu
def test_unflushed_loader_records_the_image_it_loaded():
    from gpuwm.core.kernels import load_module_unflushed, UNFLUSHED_OPTIONS
    from gpuwm.certify.kernel_manifest import kernel_manifest
    load_module_unflushed("wsm5", (("WSM5_KMAX", 64),))
    row = kernel_manifest()["gpuwm.core.kernels:wsm5[WSM5_KMAX=64]"]
    assert tuple(row["options"]) == UNFLUSHED_OPTIONS
    assert row["compiled_image"]["status"] == "resolved"
    assert len(row["compiled_image"]["sha256"]) == 64


def test_column_species_drive_cumulus_phase_and_cold_start():
    from types import SimpleNamespace
    from gpuwm.core.kf import KFPhaseMode, kf_phase_mode_for_microphysics
    from gpuwm.ingest.microphysics_cold_start import source_absent_microphysics
    from gpuwm.ingest.real import HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE
    assert kf_phase_mode_for_microphysics(4) == KFPhaseMode.SEPARATE_ICE_SNOW
    assert source_absent_microphysics(SimpleNamespace(mp_physics=4)) == ((), {})
    assert HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE[4]["retained"] == ("QC", "QR", "QI", "QS")


def test_column_work_buffers_are_rebuilt_at_a_checkpoint_or_state_digest():
    from gpuwm.io.restart import classify_scratch_slot
    for name in ("theta", "rho", "pii", "dz", "z8w"):
        assert classify_scratch_slot("mpcol_" + name) == "rebuild"
    # Surface totals remain serialized, not covered by the new work prefix.
    for slot in ("mp_rainnc", "mp_snownc"):
        assert classify_scratch_slot(slot) == "serialize"


@requires_gpu
def test_unflushed_kernel_frame_reader_matches_the_real_driver():
    from cupy.cuda import driver
    from gpuwm.core.kernels import get_kernel_unflushed, kernel_local_size_bytes
    function = get_kernel_unflushed("wsm5", "wsm5_column")
    actual = driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                     function.ptr)
    assert actual > 0
    assert kernel_local_size_bytes(function) == actual
