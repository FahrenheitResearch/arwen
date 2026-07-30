"""Every non-Python file under ``gpuwm/`` must be declared as package data.

The whole suite reads its oracle fixtures and parameter tables out of the
source tree, so a file that is missing from ``[tool.setuptools.package-data]``
still passes every test while the built wheel silently omits it.  That is not a
hypothetical: before this gate existed the declaration was a hand-maintained
enumeration that had fallen 52 files behind, including three assets the model
loads at *runtime* rather than only under test --
``data/kf_lutab/kf_lutab.npz`` (``gpuwm.core.kf``), ``data/noahmp/*.TBL``
(``gpuwm.core.noahmp``) and ``data/ruc/oracle/tbq.csv``
(``gpuwm.core.ruc._load_ruc_saturation_table``).  An install of that wheel
raises ``FileNotFoundError`` the first time those schemes are used.

The check runs setuptools' own :meth:`build_py.find_data_files` against the
real ``pyproject.toml`` rather than re-implementing pattern matching, so what
it reports is what the wheel would actually contain.  It is a filesystem walk,
not a ``git ls-files`` walk, so an undeclared file trips it before it is ever
committed.
"""

from __future__ import annotations

import glob as globlib
import os
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_ROOT = REPO_ROOT / "gpuwm"

#: Extensions that ``[tool.setuptools.packages.find]`` already ships as modules,
#: plus their compiled droppings.  Everything else under ``gpuwm/`` is data.
_MODULE_SUFFIXES = (".py", ".pyc", ".pyo")

#: Directories that never belong in a distribution at all.
_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache"})


def _require_source_tree() -> None:
    if not PYPROJECT.is_file() or not PACKAGE_ROOT.is_dir():
        pytest.skip("packaging gate needs the source tree, not an install")


#: The ONLY files that may be deliberately absent from the wheel, in two
#: pinned classes.  Anything else excluded from the wheel is a packaging
#: bug, and either list going stale is one too.
#:
#: License-driven: the four CC-BY-NC-SA-4.0 RFMIP reference-result NetCDFs
#: (gpuwm/data/rrtmgp/PROVENANCE.md) -- in-repo test fixtures whose
#: non-commercial license must not attach to the distributable.  These
#: must stay in the repo.
_LICENSE_EXCLUDED_FROM_WHEEL = frozenset({
    "gpuwm/data/rrtmgp/rfmip-clear-sky-reference-lw-down.nc",
    "gpuwm/data/rrtmgp/rfmip-clear-sky-reference-lw-up.nc",
    "gpuwm/data/rrtmgp/rfmip-clear-sky-reference-sw-down.nc",
    "gpuwm/data/rrtmgp/rfmip-clear-sky-reference-sw-up.nc",
})

#: Size-driven externalized assets: published as GitHub release assets
#: because they exceed distribution-channel limits, staged by
#: ``gpuwm fetch-tables`` under the thompson_contract size+SHA-256 pins.
#: Unlike the license class these MAY be absent from a checkout
#: (freezeH2O.dat is not in the public repository at all); when present
#: they still must not ship in the wheel or the sdist.  Pinned against
#: gpuwm.table_assets and MANIFEST.in so the packaging exclusions and
#: the fetch contract cannot drift apart.
_EXTERNALIZED_FROM_WHEEL = frozenset({
    "gpuwm/data/thompson/tables/freezeH2O.dat",
    "gpuwm/data/thompson/tables/qr_acr_qg_V4.dat",
})


def _package_data_declaration() -> dict[str, list[str]]:
    with PYPROJECT.open("rb") as stream:
        config = tomllib.load(stream)
    return config["tool"]["setuptools"]["package-data"]


def _exclude_package_data_declaration() -> dict[str, list[str]]:
    with PYPROJECT.open("rb") as stream:
        config = tomllib.load(stream)
    return config["tool"]["setuptools"].get("exclude-package-data", {})


def _discover_packages() -> list[str]:
    """Import names of every package under ``gpuwm/`` (an ``__init__.py`` dir)."""

    packages: list[str] = []
    for dirpath, dirnames, filenames in os.walk(PACKAGE_ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        if "__init__.py" not in filenames:
            continue
        relative = Path(dirpath).relative_to(REPO_ROOT)
        packages.append(".".join(relative.parts))
    return packages


def _files_on_disk() -> set[str]:
    """Repo-relative POSIX paths of every data file physically under ``gpuwm/``."""

    present: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(PACKAGE_ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in filenames:
            if name.endswith(_MODULE_SUFFIXES):
                continue
            path = Path(dirpath, name).relative_to(REPO_ROOT)
            present.add(path.as_posix())
    return present


def _files_setuptools_would_ship(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """Ask setuptools itself which files the declaration selects.

    ``build_py.get_data_files_without_manifest`` is the exact code path that
    populates a wheel, so this measures the declaration rather than trusting a
    private re-implementation of glob semantics.
    """

    setuptools = pytest.importorskip("setuptools")
    from setuptools.command.build_py import build_py

    packages = _discover_packages()
    assert "gpuwm" in packages, f"package discovery is broken: {packages}"

    distribution = setuptools.dist.Distribution({
        "name": "gpuwm",
        "packages": packages,
        "package_data": _package_data_declaration(),
        "exclude_package_data": _exclude_package_data_declaration(),
    })
    command = build_py(distribution)
    command.finalize_options()

    # setuptools globs relative to the working directory.
    monkeypatch.chdir(REPO_ROOT)

    shipped: set[str] = set()
    for _package, src_dir, _build_dir, filenames in (
        command.get_data_files_without_manifest()
    ):
        for name in filenames:
            shipped.add(Path(src_dir, name).as_posix())
    return shipped


def test_every_data_file_under_gpuwm_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No data file may reach ``gpuwm/`` without ``package-data`` covering it."""

    _require_source_tree()
    present = _files_on_disk()
    assert present, "found no data files under gpuwm/ -- the walk is broken"

    shipped = _files_setuptools_would_ship(monkeypatch)
    undeclared = sorted(present - shipped - _LICENSE_EXCLUDED_FROM_WHEEL
                        - _EXTERNALIZED_FROM_WHEEL)
    assert not undeclared, (
        f"{len(undeclared)} data file(s) under gpuwm/ would be omitted from a "
        "wheel because [tool.setuptools.package-data] in pyproject.toml does "
        "not cover them. Widen an existing glob (do not append a filename):\n  "
        + "\n  ".join(undeclared)
    )


def test_wheel_exclusions_are_exactly_the_pinned_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the license fixtures + externalized assets are excluded.

    Both directions matter: an exclusion pattern that silently widened
    would strip runtime data from the wheel, and a named fixture leaving
    the repo would break the RFMIP acceptance gates that read it.  The
    externalized assets may legitimately be absent from a checkout (the
    public repository ships them as release assets); when present they
    still must not reach the wheel.
    """

    _require_source_tree()
    present = _files_on_disk()
    shipped = _files_setuptools_would_ship(monkeypatch)
    excluded = present - shipped
    expected = set(_LICENSE_EXCLUDED_FROM_WHEEL) | (
        set(_EXTERNALIZED_FROM_WHEEL) & present)
    assert excluded == expected, (
        "wheel exclusions drifted from the pinned lists:\n"
        f"  unexpectedly excluded: {sorted(excluded - expected)}\n"
        f"  expected but shipped/missing: {sorted(expected - excluded)}"
    )
    for relative in sorted(_LICENSE_EXCLUDED_FROM_WHEEL):
        assert (REPO_ROOT / relative).is_file(), (
            f"license-excluded test fixture vanished from the repo: {relative}"
        )
    # The applicable license text still ships beside the surviving rrtmgp
    # data so PROVENANCE.md stays resolvable inside an installed wheel.
    assert "gpuwm/data/rrtmgp/LICENSE-CC-BY-NC-SA-4.0" in shipped


def test_externalized_assets_match_the_fetch_contract() -> None:
    """The packaging pin and gpuwm.table_assets must name the same files,
    and every one must carry a size+SHA-256 pin in thompson_contract."""

    from gpuwm.core.thompson_contract import CLASSIC_TABLE_ASSETS
    from gpuwm.table_assets import EXTERNALIZED_TABLE_FILENAMES

    pinned_names = {Path(p).name for p in _EXTERNALIZED_FROM_WHEEL}
    assert pinned_names == set(EXTERNALIZED_TABLE_FILENAMES)
    contract_names = {asset.filename for asset in CLASSIC_TABLE_ASSETS}
    assert set(EXTERNALIZED_TABLE_FILENAMES) <= contract_names
    # MANIFEST.in must exclude exactly the same files from the sdist
    # (exclude-package-data governs the wheel only).
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    manifest_excludes = {
        line.split(None, 1)[1].strip()
        for line in manifest.splitlines()
        if line.startswith("exclude ")
    }
    assert manifest_excludes == set(_EXTERNALIZED_FROM_WHEEL)


def test_no_package_data_pattern_is_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pattern that matches nothing is drift -- it names a moved directory."""

    _require_source_tree()
    monkeypatch.chdir(REPO_ROOT)

    declaration = _package_data_declaration()
    dead: list[str] = []
    for package, patterns in declaration.items():
        package_dir = REPO_ROOT.joinpath(*package.split("."))
        for pattern in patterns:
            if not package_dir.is_dir():
                dead.append(f"{package}: <no such package dir> ({pattern})")
                continue
            matches = globlib.glob(
                pattern, root_dir=package_dir, recursive=True
            )
            if not any((package_dir / m).is_file() for m in matches):
                dead.append(f"{package}: {pattern}")

    assert not dead, (
        "package-data pattern(s) match no file on disk; the data they named has "
        "moved or been deleted:\n  " + "\n  ".join(dead)
    )


def test_runtime_data_assets_survive_the_globs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the assets whose absence breaks an *installed* model, not just tests.

    The coverage test above is only as strong as the filesystem it walks. These
    three are loaded by physics code at run time, so an installed wheel without
    them is broken in the field rather than in CI -- worth failing loudly and by
    name if a future rewrite of the globs drops them.
    """

    _require_source_tree()
    shipped = _files_setuptools_would_ship(monkeypatch)
    for relative in (
        "gpuwm/data/kf_lutab/kf_lutab.npz",
        "gpuwm/data/morrison/constants.toml",
        "gpuwm/data/thompson/tables/qr_acr_qsV2.dat",
        "gpuwm/data/thompson/tables/thompson_aux_tables.dat",
        "gpuwm/data/thompson/tables/MANIFEST.sha256",
        "gpuwm/data/noahmp/MPTABLE.TBL",
        "gpuwm/data/noahmp/SOILPARM.TBL",
        "gpuwm/data/noahmp/GENPARM.TBL",
        "gpuwm/data/noah_tables/VEGPARM.TBL",
        "gpuwm/data/ruc/oracle/tbq.csv",
        "gpuwm/data/rrtmgp/rrtmgp-gas-lw-g256.nc",
        "gpuwm/data/wrf_radiation/RRTM_DATA",
    ):
        assert (REPO_ROOT / relative).is_file(), f"fixture vanished: {relative}"
        assert relative in shipped, (
            f"{relative} is loaded at run time but would not ship in a wheel"
        )
    # The externalized tables are loaded at run time too, but deliberately
    # NOT via the wheel: they are release assets staged by
    # `gpuwm fetch-tables`, so the pin here is inverted -- they must never
    # ship even when present in the tree.
    for relative in sorted(_EXTERNALIZED_FROM_WHEEL):
        assert relative not in shipped, (
            f"{relative} is externalized and must not ship in the wheel"
        )
