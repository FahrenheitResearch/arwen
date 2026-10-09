"""A WPS_GEOG tree shaped the way ``gpuwm fetch-geog`` leaves one.

Directory names and the ``index`` file are the whole of what the
geography check (``gpuwm.doctor.geography_gaps``, asked by ``gpuwm go`` and
since 2.8.8 by its dry run) reads, so nine empty-but-indexed directories
stand in for 16 GB of terrain.  Names come from ``geog_assets``, the module
that stages the real tree, so the stand-in cannot drift from the check.
For dry runs in a SUBPROCESS, which tests/conftest.py's default-root pin
cannot reach, :func:`staged_case_data_env` points the default root at one.
"""
from __future__ import annotations

import os
from pathlib import Path


def staged_geog_tree(geog: Path) -> Path:
    """Stage the stand-in AT ``geog`` (the WPS_GEOG root itself)."""

    from gpuwm.geog_assets import geog_datasets

    geog = Path(geog)
    for name in geog_datasets():
        (geog / name).mkdir(parents=True, exist_ok=True)
        (geog / name / "index").write_text("", encoding="utf-8")
    return geog


def staged_case_data_env(root: Path, base=None) -> dict:
    """``os.environ`` (or ``base``) with the default WPS_GEOG root staged.

    ``GPUWM_CASE_DATA_ROOT`` is ``root``, so ``default_geog_root()`` is
    ``root / "WPS_GEOG"``, which this stages.
    """

    staged_geog_tree(Path(root) / "WPS_GEOG")
    env = dict(os.environ if base is None else base)
    env["GPUWM_CASE_DATA_ROOT"] = str(root)
    return env
