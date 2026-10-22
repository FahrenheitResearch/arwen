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

    from gpuwm.geog_assets import GEOG_ARCHIVES

    geog = Path(geog)
    # A container archive (soilgrids) carries its index in each child, which
    # is where the builder and doctor look (the default soil categories are
    # soilgrids/texture_top and texture_bot).
    for archive in GEOG_ARCHIVES:
        for child in archive.index_subdirs or ("",):
            (geog / archive.dataset / child).mkdir(parents=True, exist_ok=True)
            (geog / archive.dataset / child / "index").write_text(
                "", encoding="utf-8")
    return geog


def staged_wrf_geog_tree(geog: Path) -> Path:
    """Stage exactly the directories the WRF static builder's default opens.

    By directory, as the builder and doctor read them
    (``gpuwm.geog_assets.WRF_DEFAULT_GEOG_DIRS``): the default soil
    categories are two children of the soilgrids container.  For tests that
    then remove one dataset to make the tree incomplete.
    """

    from gpuwm.geog_assets import WRF_DEFAULT_GEOG_DIRS

    geog = Path(geog)
    for directory in WRF_DEFAULT_GEOG_DIRS:
        (geog / directory).mkdir(parents=True, exist_ok=True)
        (geog / directory / "index").write_text(
            "type = continuous\n", encoding="utf-8")
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
