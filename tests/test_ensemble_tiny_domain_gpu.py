"""End-to-end tiny-domain ensemble: two members, one GPU, real runner.

This is the receipt behind the determinism claim.  It runs the shipped
member path -- the same prepare/integrate functions
``gpuwm.runtime.run_experiment`` calls -- twice for the same seed and
asserts the final states are byte-identical, and once more for a
different seed and asserts they are not.

Needs a device and the staged case data the base config cites; skips
cleanly without either.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpuwm.case_data import case_data_root
from gpuwm.ensemble.config import load_ensemble_config
from gpuwm.ensemble.engine import run_ensemble
from gpuwm.ensemble.manifest import (
    ENSEMBLE_MANIFEST_NAME, ENSEMBLE_MANIFEST_SCHEMA, read_manifest,
)

pytest.importorskip("cupy")

REPO = Path(__file__).resolve().parents[1]
ENSEMBLE_CONFIG = REPO / "configs/ensemble/may1999_tiny_2member.toml"

_DATA = case_data_root() / "ERA5_may1999_case_data"
_BUNDLE = case_data_root() / "WRF_1974_MP55_reference_bundle"
requires_case_data = pytest.mark.skipif(
    not (_DATA / "era5_may1999_pl.grib").is_file()
    or not (_BUNDLE / "era5_grib/Vtable.ERA5_CDO").is_file()
    or not (_BUNDLE / "static/WPS_GEOG").is_dir(),
    reason="staged May-1999 ERA5 / reference bundle is absent")
requires_gpu_opt_in = pytest.mark.skipif(
    os.environ.get("GPUWM_NO_LOCAL_GPU", "") not in ("", "0"),
    reason="GPUWM_NO_LOCAL_GPU is set")

#: Two minutes of simulation on a 40x32 grid: long enough that the state
#: has demonstrably moved, short enough to be a smoke test.
RUN_SECONDS = 120.0


def _final_shas(root):
    manifest = read_manifest(Path(root) / ENSEMBLE_MANIFEST_NAME,
                             schema=ENSEMBLE_MANIFEST_SCHEMA)
    assert manifest["status"] == "COMPLETE"
    return [record["final_state_sha256"] for record in manifest["members"]]


@pytest.mark.gpu
@requires_gpu_opt_in
@requires_case_data
@pytest.mark.slow
def test_tiny_two_member_ensemble_is_seed_reproducible(tmp_path):
    cfg = load_ensemble_config(ENSEMBLE_CONFIG)
    first = run_ensemble(cfg, tmp_path / "run-a", run_seconds=RUN_SECONDS)
    second = run_ensemble(cfg, tmp_path / "run-b", run_seconds=RUN_SECONDS)
    assert first.status == "COMPLETE" and second.status == "COMPLETE"
    left = _final_shas(first.ens_root)
    right = _final_shas(second.ens_root)
    assert left == right, "same seeds must give byte-identical final states"
    assert left[0] != left[1], "different seeds must give different members"
    for index in range(cfg.n_members):
        assert list((first.ens_root / f"member_{index:03d}"
                     / "wrfout").rglob("*")) or list(
            (first.ens_root / f"member_{index:03d}").glob("wrfout*")), \
            "each member directory holds a normal run's output"
