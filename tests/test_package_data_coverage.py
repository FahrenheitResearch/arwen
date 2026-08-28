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

Why the walk stops at ``gpuwm/``
-------------------------------
``tools`` is a declared package too, but it is not a data directory: it is
mostly scripts, plus two vendored Rust workspaces whose source is *build
input* and has no business in a wheel.  A blanket "everything under ``tools``
must ship" rule would be wrong in 260 MB of ways.

That exemption is exactly how the renderer's map assets disappeared.
``tools/rustwx/assets/basemap`` holds the Natural Earth and US Census
shapefiles ``rw_wrfbatch`` draws coastlines and borders from -- runtime data
living inside a directory this file had decided not to look at -- and the one
``package-data`` entry still written as an enumeration, ``tools =
["prepare_hrrr_*.sh"]``, did not name them.  Nothing failed.  A pip install
rendered a tropical cyclone over a blank rectangle.

So the rule is not "walk more", it is "every *runtime* asset must be
delivered by some declared mechanism, and the test names which".  For the
renderer's assets that mechanism is the bridge bundle rather than the wheel
(the wheel is 74.6 MiB against PyPI's 100 MB per-file cap and the assets
deflate to 20.2 MiB), which
:func:`test_the_renderer_asset_tree_is_delivered_by_a_declared_mechanism`
checks against ``gpuwm.bridge_assets``.
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

#: The companion distribution: one repository, two pyproject.toml files.
#: ``gpuwm-data`` carries the RRTMGP and Thompson table directories since
#: 2.5.0, because the ``gpuwm`` wheel measured 103.62 MiB against PyPI's
#: 100 MiB per-file cap.  This file gates BOTH, and it has to: the moment a
#: directory moved out of ``gpuwm/``, a walk that stops at ``gpuwm/`` would
#: have reported "all declared" over an empty result -- exactly the
#: vacuous-green shape the renderer's map assets went dark in.
COMPANION_ROOT = REPO_ROOT / "gpuwm-data"
COMPANION_PYPROJECT = COMPANION_ROOT / "pyproject.toml"
COMPANION_PACKAGE_ROOT = COMPANION_ROOT / "gpuwm_data"

#: Extensions that ``[tool.setuptools.packages.find]`` already ships as modules,
#: plus their compiled droppings.  Everything else under ``gpuwm/`` is data.
_MODULE_SUFFIXES = (".py", ".pyc", ".pyo")

#: Directories that never belong in a distribution at all.
_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache"})


def _require_source_tree() -> None:
    if not PYPROJECT.is_file() or not PACKAGE_ROOT.is_dir():
        pytest.skip("packaging gate needs the source tree, not an install")


def _require_companion_tree() -> None:
    if not COMPANION_PYPROJECT.is_file() or not COMPANION_PACKAGE_ROOT.is_dir():
        pytest.skip("packaging gate needs the source tree, not an install")


#: The ONLY files that may be deliberately absent from the wheel, in two
#: pinned classes.  Anything else excluded from the wheel is a packaging
#: bug, and either list going stale is one too.
#:
#: License-driven: the four CC-BY-NC-SA-4.0 RFMIP reference-result NetCDFs
#: (gpuwm-data/gpuwm_data/data/rrtmgp/PROVENANCE.md) -- in-repo test
#: fixtures whose non-commercial license must not attach to the
#: distributable.  These must stay in the repo.
#:
#: Repo-relative, and they moved with their directory: both classes below
#: now name files in the COMPANION distribution, because that is where the
#: rrtmgp and thompson/tables directories ship since 2.5.0.  The `gpuwm`
#: wheel excludes nothing at all now, which
#: :func:`test_wheel_exclusions_are_exactly_the_pinned_lists` states
#: directly rather than leaving as an absence.
_LICENSE_EXCLUDED_FROM_WHEEL = frozenset({
    "gpuwm-data/gpuwm_data/data/rrtmgp/rfmip-clear-sky-reference-lw-down.nc",
    "gpuwm-data/gpuwm_data/data/rrtmgp/rfmip-clear-sky-reference-lw-up.nc",
    "gpuwm-data/gpuwm_data/data/rrtmgp/rfmip-clear-sky-reference-sw-down.nc",
    "gpuwm-data/gpuwm_data/data/rrtmgp/rfmip-clear-sky-reference-sw-up.nc",
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
    "gpuwm-data/gpuwm_data/data/thompson/tables/freezeH2O.dat",
    "gpuwm-data/gpuwm_data/data/thompson/tables/qr_acr_qg_V4.dat",
})

#: Third-party data this repository redistributes verbatim from a WRF
#: release, which therefore must SHIP rather than be excluded.
#:
#: ``CCN_ACTIVATE.BIN`` is the aerosol-aware Thompson (mp_physics=28)
#: activation table.  It is not generated by ``thompson_init`` and no
#: recompilation of WRF reproduces it -- ``table_ccnAct``
#: (phys/module_mp_thompson.F:5110-5166) only READS it, and the numbers are
#: offline parcel-model output (WRF's own comment at :5102-5108).  Until
#: 2026-08-01 the port did not redistribute it and ``pyproject.toml`` carried
#: an explicit ``exclude-package-data`` entry to keep a wheel built on a
#: machine that had staged it from publishing it.  That decision was
#: reversed: the committed copy is WRF v4.6.1's ``run/CCN_ACTIVATE.BIN`` bit
#: for bit, WRF's ``LICENSE.txt`` is a public-domain dedication, and the
#: notice ships in ``gpuwm/data/wrf_radiation/LICENSE-WRF.txt``.  The pin
#: below is therefore inverted: these files must reach the wheel, and the
#: exclusion entry must be gone.  See ``gpuwm/data/thompson/PROVENANCE.md``
#: and ``thompson_aerosol_contract.AEROSOL_ASSET_REDISTRIBUTED``.
_REDISTRIBUTED_WRF_DATA = frozenset({
    "gpuwm-data/gpuwm_data/data/thompson/tables/CCN_ACTIVATE.BIN",
})


def _build_system_requires() -> list[str]:
    """What this project declares it needs in order to be built at all."""

    with PYPROJECT.open("rb") as stream:
        config = tomllib.load(stream)
    return list(config.get("build-system", {}).get("requires", []))


def _package_data_declaration(pyproject: Path = PYPROJECT
                              ) -> dict[str, list[str]]:
    with pyproject.open("rb") as stream:
        config = tomllib.load(stream)
    return config["tool"]["setuptools"]["package-data"]


def _exclude_package_data_declaration(pyproject: Path = PYPROJECT
                                      ) -> dict[str, list[str]]:
    with pyproject.open("rb") as stream:
        config = tomllib.load(stream)
    return config["tool"]["setuptools"].get("exclude-package-data", {})


def _discover_packages(package_root: Path = PACKAGE_ROOT,
                       project_root: Path = REPO_ROOT) -> list[str]:
    """Import names of every package under ``package_root``.

    Parameterised over the project because there are two distributions in
    this repository now (``gpuwm`` and the ``gpuwm-data`` companion), and
    the whole value of this file is that it asks setuptools rather than
    restating glob semantics -- which means it has to ask setuptools the
    same question once per project.
    """

    packages: list[str] = []
    for dirpath, dirnames, filenames in os.walk(package_root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        if "__init__.py" not in filenames:
            continue
        relative = Path(dirpath).relative_to(project_root)
        packages.append(".".join(relative.parts))
    return packages


def _files_on_disk(package_root: Path = PACKAGE_ROOT,
                   anchor: Path = REPO_ROOT) -> set[str]:
    """Repo-relative POSIX paths of every data file under ``package_root``."""

    present: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(package_root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in filenames:
            if name.endswith(_MODULE_SUFFIXES):
                continue
            path = Path(dirpath, name).relative_to(anchor)
            present.add(path.as_posix())
    return present


def _files_setuptools_would_ship(monkeypatch: pytest.MonkeyPatch,
                                 *, name: str = "gpuwm",
                                 project_root: Path = REPO_ROOT,
                                 package_root: Path = PACKAGE_ROOT,
                                 pyproject: Path = PYPROJECT,
                                 anchor: Path | None = None) -> set[str]:
    """Ask setuptools itself which files the declaration selects.

    ``build_py.get_data_files_without_manifest`` is the exact code path that
    populates a wheel, so this measures the declaration rather than trusting a
    private re-implementation of glob semantics.

    An absent setuptools is a FAILURE here, not a skip.  It used to be
    ``pytest.importorskip("setuptools")``, and the 2026-08-13 test-estate
    audit measured the consequence: this file is on
    ``tools/battery/stage1_files.txt`` and was **56% dark** on the assembly
    venvs -- 4 passed, 5 skipped -- and the five that skipped are exactly
    the wheel-content assertions the file exists for (every data file
    declared, the exclusion lists, the renderer asset tree, the runtime
    assets, the Thompson table).  That is the same class as the defect
    ``fix(render): the map assets ship with the renderer that reads them``
    closed, and a stage-1 entry cannot be allowed to report green while its
    reason for existing is switched off.

    A skip would be defensible if setuptools were optional.  It is not:
    ``[build-system] requires`` in this project's own pyproject.toml names
    ``setuptools>=77``, so any environment that can build this project has
    it, and one that does not cannot answer the question this file asks.
    The remedy is one line -- ``pip install setuptools`` -- and the message
    below says so.
    """

    try:
        import setuptools
    except ModuleNotFoundError as exc:      # pragma: no cover - env defect
        requires = _build_system_requires()
        assert any(r.replace("_", "-").lower().startswith("setuptools")
                   for r in requires), (
            "setuptools is absent AND [build-system] requires no longer names "
            f"it ({requires}); this file's premise has changed, so decide "
            "deliberately whether the wheel-content assertions still apply")
        raise AssertionError(
            "setuptools is not installed, so the five wheel-content "
            "assertions in this file cannot run -- and this file is on "
            "tools/battery/stage1_files.txt, where a silent skip reports "
            f"green.  pyproject.toml's [build-system] requires {requires}, so "
            "every environment that can build this project has it.  Remedy: "
            "pip install setuptools (or install the project with "
            "'pip install -e .[dev,render]' in a venv built with it)."
        ) from exc

    from setuptools.command.build_py import build_py

    packages = _discover_packages(package_root, project_root)
    assert package_root.name in packages, (
        f"package discovery is broken for {name}: {packages}")

    distribution = setuptools.dist.Distribution({
        "name": name,
        "packages": packages,
        "package_data": _package_data_declaration(pyproject),
        "exclude_package_data": _exclude_package_data_declaration(pyproject),
    })
    command = build_py(distribution)
    command.finalize_options()

    # setuptools globs relative to the working directory.
    monkeypatch.chdir(project_root)

    # Results come back project-relative; re-anchor them on the repository
    # so both distributions' answers are comparable to `_files_on_disk`.
    prefix = (project_root.relative_to(anchor).as_posix() + "/"
              if anchor is not None and project_root != anchor else "")
    shipped: set[str] = set()
    for _package, src_dir, _build_dir, filenames in (
        command.get_data_files_without_manifest()
    ):
        for filename in filenames:
            shipped.add(prefix + Path(src_dir, filename).as_posix())
    return shipped


def test_the_wheel_content_assertions_cannot_go_dark_by_skipping() -> None:
    """The premise behind refusing to skip on an absent setuptools.

    The 2026-08-13 test-estate audit measured this file at 4 passed / 5
    skipped on the assembly venvs, and the five that skipped are the whole
    point of the file.  ``_files_setuptools_would_ship`` now raises instead,
    which is only defensible while setuptools really is a build requirement
    of this project rather than an optional extra.  This is that check: if
    the build backend ever moves, the assertion below fails and whoever
    moves it has to decide what these five assertions become, instead of
    silently inheriting a skip.
    """

    requires = _build_system_requires()
    assert requires, "pyproject.toml declares no [build-system] requires"
    assert any(r.replace("_", "-").lower().startswith("setuptools")
               for r in requires), (
        f"[build-system] requires is {requires} and no longer names "
        "setuptools, so 'every environment that can build this project has "
        "setuptools' has stopped being true.  _files_setuptools_would_ship "
        "raises on an absent setuptools on the strength of that sentence")


def test_every_data_file_under_gpuwm_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No data file may reach ``gpuwm/`` without ``package-data`` covering it."""

    _require_source_tree()
    present = _files_on_disk()
    assert present, "found no data files under gpuwm/ -- the walk is broken"

    shipped = _files_setuptools_would_ship(monkeypatch)
    undeclared = sorted(present - shipped)
    assert not undeclared, (
        f"{len(undeclared)} data file(s) under gpuwm/ would be omitted from a "
        "wheel because [tool.setuptools.package-data] in pyproject.toml does "
        "not cover them. Widen an existing glob (do not append a filename):\n  "
        + "\n  ".join(undeclared)
    )


def _companion_shipped(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    return _files_setuptools_would_ship(
        monkeypatch, name="gpuwm-data", project_root=COMPANION_ROOT,
        package_root=COMPANION_PACKAGE_ROOT, pyproject=COMPANION_PYPROJECT,
        anchor=REPO_ROOT)


def test_every_data_file_in_the_companion_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same rule, for the distribution the big tables moved into.

    Without this twin the split would have created exactly the hole this
    file was written to close: ``rrtmgp/`` and ``thompson/tables/`` left
    ``gpuwm/``, the walk above stopped finding them, and it would have
    reported "all declared" while a mis-declared companion shipped an
    empty wheel and every radiation run died at table load.
    """

    _require_companion_tree()
    present = _files_on_disk(COMPANION_PACKAGE_ROOT, REPO_ROOT)
    assert present, "found no data files in the companion -- walk is broken"

    shipped = _companion_shipped(monkeypatch)
    undeclared = sorted(present - shipped - _LICENSE_EXCLUDED_FROM_WHEEL
                        - _EXTERNALIZED_FROM_WHEEL)
    assert not undeclared, (
        f"{len(undeclared)} data file(s) in gpuwm-data/ would be omitted "
        "from its wheel because [tool.setuptools.package-data] in "
        "gpuwm-data/pyproject.toml does not cover them. Widen an existing "
        "glob (do not append a filename):\n  " + "\n  ".join(undeclared)
    )


def test_the_moved_directories_left_the_gpuwm_wheel_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither directory may ship in BOTH wheels, or in neither.

    The breakage on each side is concrete.  Shipping in both puts the
    64.21 MiB back into the ``gpuwm`` wheel and PyPI rejects the upload at
    100 MiB -- which is the whole reason the companion exists.  Shipping
    in neither is worse and quieter: ``pip install gpuwm`` would succeed
    and then fail at the first radiation table read.

    Driven from ``gpuwm.data_assets.COMPANION_TREES`` rather than a
    literal list, so moving the next directory out needs one entry there
    and no edit here.
    """

    _require_source_tree()
    _require_companion_tree()
    from gpuwm import data_assets

    gpuwm_shipped = _files_setuptools_would_ship(monkeypatch)
    companion_shipped = _companion_shipped(monkeypatch)
    for tree in data_assets.COMPANION_TREES:
        stale = sorted(name for name in gpuwm_shipped
                       if name.startswith(f"gpuwm/data/{tree}/"))
        assert not stale, (
            f"the gpuwm wheel still ships {len(stale)} file(s) under "
            f"gpuwm/data/{tree}/, which gpuwm.data_assets says the "
            f"gpuwm-data companion owns:\n  " + "\n  ".join(stale))
        carried = [name for name in companion_shipped
                   if name.startswith(
                       f"gpuwm-data/gpuwm_data/data/{tree}/")]
        assert carried, (
            f"gpuwm.data_assets.COMPANION_TREES names {tree!r} but the "
            "companion wheel would carry no file under it, so every "
            "consumer of that directory fails at load on a fresh install")


def test_wheel_exclusions_are_exactly_the_pinned_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the license fixtures and the size-externalized assets.

    Both directions matter: an exclusion pattern that silently widened
    would strip runtime data from the wheel, and a named fixture leaving
    the repo would break the RFMIP acceptance gates that read it.  The
    externalized assets may legitimately be absent from a checkout (the
    public repository ships them as release assets); when present they
    still must not reach the wheel.

    There is no third, redistribution-driven class any more.
    ``CCN_ACTIVATE.BIN`` was the only member and it ships now, so it must
    appear on the OTHER side of this comparison -- a re-added exclusion for
    it fails here as an unexpected exclusion.
    """

    _require_source_tree()
    _require_companion_tree()

    # The gpuwm wheel excludes NOTHING now.  Both classes named files under
    # the two directories that moved to the companion in 2.5.0, so an
    # exclusion appearing here again means either a file came back without
    # its reasoning or somebody stripped runtime data from the wheel.
    gpuwm_excluded = sorted(_files_on_disk()
                            - _files_setuptools_would_ship(monkeypatch))
    assert not gpuwm_excluded, (
        "the gpuwm wheel now excludes files, and both pinned exclusion "
        "classes moved to gpuwm-data/pyproject.toml with the directories "
        "they name:\n  " + "\n  ".join(gpuwm_excluded))

    present = _files_on_disk(COMPANION_PACKAGE_ROOT, REPO_ROOT)
    shipped = _companion_shipped(monkeypatch)
    excluded = present - shipped
    expected = set(_LICENSE_EXCLUDED_FROM_WHEEL) | (
        set(_EXTERNALIZED_FROM_WHEEL) & present)
    assert excluded == expected, (
        "companion wheel exclusions drifted from the pinned lists:\n"
        f"  unexpectedly excluded: {sorted(excluded - expected)}\n"
        f"  expected but shipped/missing: {sorted(expected - excluded)}"
    )
    for relative in sorted(_LICENSE_EXCLUDED_FROM_WHEEL):
        assert (REPO_ROOT / relative).is_file(), (
            f"license-excluded test fixture vanished from the repo: {relative}"
        )
    # The applicable license text still ships beside the surviving rrtmgp
    # data so PROVENANCE.md stays resolvable inside an installed wheel.
    assert ("gpuwm-data/gpuwm_data/data/rrtmgp/LICENSE-CC-BY-NC-SA-4.0"
            in shipped)
    # ...and the redistributed WRF table must be on the SHIPPED side.
    for relative in sorted(_REDISTRIBUTED_WRF_DATA):
        assert relative in shipped, relative


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
    # (exclude-package-data governs the wheel only).  The companion's, now:
    # the directory holding both files moved there in 2.5.0, and so did the
    # exclusions.  The root MANIFEST.in must be empty of them for the same
    # reason -- an exclude naming a path that no longer exists is a rule
    # nobody is enforcing.
    companion_manifest = (COMPANION_ROOT / "MANIFEST.in").read_text(
        encoding="utf-8")
    manifest_excludes = {
        "gpuwm-data/" + line.split(None, 1)[1].strip()
        for line in companion_manifest.splitlines()
        if line.startswith("exclude ")
    }
    assert manifest_excludes == set(_EXTERNALIZED_FROM_WHEEL)
    root_manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    rules = [line for line in root_manifest.splitlines()
             if line.startswith(("exclude ", "prune "))]
    # What this test is actually about is the EXCLUDES: every size- and
    # license-driven `exclude` moved to gpuwm-data/MANIFEST.in with the
    # directory it named, and one left behind here would be a rule nobody
    # is enforcing.  So the excludes must be empty.
    assert not [rule for rule in rules if rule.startswith("exclude ")], (
        "the root MANIFEST.in still carries a size- or license-driven "
        "`exclude`.  Every one of them moved to gpuwm-data/MANIFEST.in "
        "with the directory it named in 2.5.0, and an exclude naming a "
        f"path that no longer exists is enforced by nobody; found {rules}")
    # The prunes are a different question, and there are two.  Each one
    # keeps a tree that belongs to a DIFFERENT artifact out of gpuwm's
    # sdist, and neither subsumes the other:
    #
    #   gpuwm-data          -- the companion distribution's source.  A
    #     fragment of it in gpuwm's own sdist is worse than either whole
    #     answer: `gpuwm_data/__init__.py` with none of its data is a
    #     package that says where the tables are and has none.
    #   gpuwm/libexec/bridges -- the staged Rust binaries, which are wheel
    #     payload.  Measured on the 2.5.0 Linux shakeout: an sdist built
    #     after staging swept the staged native ELF executables in -- the
    #     18 declared at that commit -- for +19.67 MB.
    #
    # Pinned as a SET so a third prune has to be justified here, and so
    # that neither can be dropped silently.
    assert set(rules) == {"prune gpuwm-data", "prune gpuwm/libexec/bridges"}, (
        "the root MANIFEST.in's prunes drifted.  Both rules keep another "
        "artifact's tree out of gpuwm's sdist -- the companion "
        "distribution's source, and the staged platform binaries -- and "
        f"a cut has shipped the wrong bytes for each of them; found {rules}")


def test_the_renderer_asset_tree_is_delivered_by_a_declared_mechanism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No runtime asset under ``tools/`` may be delivered by nothing at all.

    The gap this closes: the wheel-coverage walk above deliberately stops at
    ``gpuwm/``, so a data directory under ``tools/`` could -- and did -- reach
    an installed user by no mechanism whatever, silently, and the only symptom
    was a plot with no geography on it.

    Every file under the renderer's asset root must therefore be accounted
    for by one of the two mechanisms that exist: the wheel's package-data, or
    the bridge bundle.  A new asset subdirectory that nobody added to
    ``bridge_assets.REQUIRED_ASSET_SUBDIRS`` is delivered by neither, and
    fails here by name rather than months later in an image.
    """

    _require_source_tree()
    from gpuwm import bridge_assets

    asset_root = REPO_ROOT / "tools" / "rustwx" / "assets"
    if not asset_root.is_dir():
        pytest.skip("no vendored renderer assets in this tree")

    present: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(asset_root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in filenames:
            present.add(
                Path(dirpath, name).relative_to(REPO_ROOT).as_posix())
    assert present, "found no renderer assets -- the walk is broken"

    carried_by_bundle = {
        path for path in present
        if Path(path).relative_to(
            asset_root.relative_to(REPO_ROOT)).parts[0]
        in bridge_assets.REQUIRED_ASSET_SUBDIRS
    }
    shipped_in_wheel = _files_setuptools_would_ship(monkeypatch)
    undelivered = sorted(present - carried_by_bundle - shipped_in_wheel)
    assert not undelivered, (
        f"{len(undelivered)} renderer asset file(s) would reach an installed "
        "user by no mechanism at all -- not the wheel's "
        "[tool.setuptools.package-data], and not the bridge bundle's "
        "gpuwm.bridge_assets.REQUIRED_ASSET_SUBDIRS. A renderer without them "
        "draws plots with no coastlines or borders:\n  "
        + "\n  ".join(undelivered))

    # The declaration must also still name something real: a subdirectory
    # renamed on disk leaves REQUIRED_ASSET_SUBDIRS pointing at nothing, and
    # the bundle would be packed empty.
    for subdir in bridge_assets.REQUIRED_ASSET_SUBDIRS:
        candidate = asset_root / subdir
        assert candidate.is_dir(), (
            f"bridge_assets.REQUIRED_ASSET_SUBDIRS names {subdir!r}, which is "
            f"not a directory at {candidate}")
        assert any(p.is_file() for p in candidate.rglob("*")), (
            f"{candidate} holds no files; the bundle would carry no basemaps")


#: Patterns whose directory is BUILT, not authored, so an empty match in a
#: source checkout is the correct state rather than drift.
#:
#: Exactly one entry, and it is not a loophole: the prebuilt Rust a platform
#: wheel carries is staged into ``gpuwm/libexec/bridges`` by
#: ``tools/stage_wheel_bridges.py`` between ``cargo build`` and
#: ``python -m build``, and the directory is gitignored.  A checkout that has
#: not been through that step legitimately has nothing there -- which is also
#: the state that produces the ``py3-none-any`` fallback wheel.
#:
#: The exemption is only for EMPTINESS.  When the directory does exist it must
#: carry files, so a half-staged or hand-emptied tree still fails here; that a
#: staged tree really produces a wheel carrying all eleven artifacts is proved
#: separately, by ``tests/test_wheel_bridge_staging.py``.
#:
#: The release cut does NOT stage it.  It publishes the sdist and the single
#: universal ``py3-none-any`` wheel this project has shipped 37 versions of,
#: and proves that wheel carries NO staged artifacts; the prebuilt Rust
#: reaches users as the per-platform release-asset bundles ``gpuwm setup``
#: stages.  So this pattern matching nothing in CI is the normal state, not a
#: half-built release.
_STAGED_AT_RELEASE: set[tuple[str, str]] = {("gpuwm", "libexec/bridges/*")}


def test_no_package_data_pattern_is_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pattern that matches nothing is drift -- it names a moved directory."""

    _require_source_tree()
    monkeypatch.chdir(REPO_ROOT)

    declaration = _package_data_declaration()
    dead: list[str] = []
    declared_staged: set[tuple[str, str]] = set()
    for package, patterns in declaration.items():
        package_dir = REPO_ROOT.joinpath(*package.split("."))
        for pattern in patterns:
            staged = (package, pattern) in _STAGED_AT_RELEASE
            if staged:
                declared_staged.add((package, pattern))
                # Absent is fine; present-but-empty is not.
                staged_dir = package_dir / Path(pattern).parent
                if not staged_dir.is_dir():
                    continue
                if not any(child.is_file() for child in staged_dir.iterdir()):
                    dead.append(
                        f"{package}: {pattern} -- the staging directory exists "
                        f"but is empty, which is a half-built wheel rather "
                        f"than an unstaged checkout")
                continue
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
    # The exemption list must not outlive the pattern it excuses.
    stale = _STAGED_AT_RELEASE - declared_staged
    assert not stale, (
        f"_STAGED_AT_RELEASE excuses {sorted(stale)}, which pyproject.toml no "
        f"longer declares; drop the entry rather than leaving a standing "
        f"exemption for a pattern that is gone")


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
    _require_companion_tree()
    # The union of both wheels, which is what `pip install gpuwm` puts on
    # disk: the companion is a hard `==` dependency, so a runtime asset is
    # delivered if EITHER distribution carries it, and the reader of this
    # list should not have to know which.  The prefixes still say which,
    # so a file moving between them is a visible edit here.
    shipped = (_files_setuptools_would_ship(monkeypatch)
               | _companion_shipped(monkeypatch))
    for relative in (
        "gpuwm/data/kf_lutab/kf_lutab.npz",
        "gpuwm/data/morrison/constants.toml",
        "gpuwm-data/gpuwm_data/data/thompson/tables/qr_acr_qsV2.dat",
        "gpuwm-data/gpuwm_data/data/thompson/tables/thompson_aux_tables.dat",
        "gpuwm-data/gpuwm_data/data/thompson/tables/MANIFEST.sha256",
        "gpuwm/data/noahmp/MPTABLE.TBL",
        "gpuwm/data/noahmp/SOILPARM.TBL",
        "gpuwm/data/noahmp/GENPARM.TBL",
        "gpuwm/data/noah_tables/VEGPARM.TBL",
        "gpuwm/data/ruc/oracle/tbq.csv",
        "gpuwm-data/gpuwm_data/data/rrtmgp/rrtmgp-gas-lw-g256.nc",
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


def test_no_shipped_data_file_is_swallowed_by_gitignore() -> None:
    """A shipped data file that .gitignore matches is a sprung trap, not a
    hypothetical.

    The first public-tree CI run (32280494350) paid for it: the release
    snapshot carries this repository's .gitignore, whose repo-wide
    ``*.nc`` rule (written for model OUTPUT) matched the companion's nine
    ``data/rrtmgp/*.nc``.  In the private repository they are tracked --
    force-added once, so nothing here ever noticed -- but any fresh
    ``git add`` of the snapshot silently drops them, and the public
    repository was born without its k-distributions: the companion wheel
    built from that checkout imports, version-checks, and dies at the
    first radiation table load.

    So the rule: every data file the two wheels ship must be invisible
    to .gitignore.  ``--no-index``, because check-ignore consults the
    index by default and a TRACKED-but-ignored file -- the exact trap --
    reports clean without it.  The remedy is a negation beside the rule
    that matches (the crate goldens' ``!*.nc`` is the precedent), never
    untracking the file.
    """

    _require_source_tree()
    _require_companion_tree()
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("no .git here, so there is no ignore machinery to "
                    "spring; the gate runs in every real checkout")
    import subprocess

    candidates = sorted(_files_on_disk()
                        | _files_on_disk(COMPANION_PACKAGE_ROOT, REPO_ROOT))
    assert candidates, "found no data files at all -- the walk is broken"
    # NUL-terminated bytes, deliberately: a text-mode pipe on Windows
    # rewrites "\n" to "\r\n" on the way in, git then finds no path that
    # ends in a carriage return, and the gate reports clean over a probe
    # that measured nothing -- the exact vacuous green it exists to end.
    probe = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        cwd=str(REPO_ROOT),
        input="\0".join(candidates).encode("utf-8"),
        capture_output=True)
    # 0 = some matched, 1 = none matched; anything else is git failing.
    assert probe.returncode in (0, 1), probe.stderr.decode(errors="replace")
    swallowed = [entry for entry in
                 probe.stdout.decode("utf-8").split("\0") if entry]
    assert not swallowed, (
        f"{len(swallowed)} shipped data file(s) are matched by .gitignore, "
        "so a fresh `git add` of a release snapshot drops them and the "
        "published tree ships a companion that dies at table load.  Add a "
        "negation beside the rule that matches (see the crate goldens' "
        "!*.nc for the shape):\n  " + "\n  ".join(swallowed))


def test_the_thompson_aerosol_table_reaches_the_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gpuwm declares it ships this file, so the wheel must actually have it.

    ``CCN_ACTIVATE.BIN`` is the one Thompson coefficient artifact that is not
    a ``thompson_init`` product: it is third-party parcel-model output WRF
    redistributes, and until 2026-08-01 this repository did not.  The pin has
    been inverted rather than deleted, because it is the same failure in the
    other direction: an ``exclude-package-data`` entry left behind, or the
    ``.gitignore`` line reinstated, would ship an mp=28 that fails closed on a
    missing asset for every user while the constant still claimed otherwise.

    The pin is bound to ``AEROSOL_ASSET_REDISTRIBUTED`` so the decision moves
    in one place: flip that constant back and this test tells you the
    packaging entry has to move with it.
    """

    _require_companion_tree()
    from gpuwm.core.thompson_aerosol_contract import (
        AEROSOL_ASSET_REDISTRIBUTED,
        AEROSOL_TABLE_ASSETS,
    )

    assert AEROSOL_ASSET_REDISTRIBUTED is True, (
        "the port declares this asset redistributed; if that changed, the "
        "wheel exclusion and gpuwm/data/thompson/PROVENANCE.md change with it")
    pinned = {asset.filename for asset in AEROSOL_TABLE_ASSETS}
    assert {Path(p).name for p in _REDISTRIBUTED_WRF_DATA} == pinned

    # The COMPANION wheel since 2.5.0 -- the claim is unchanged, the
    # distribution carrying the file is not.
    shipped = _companion_shipped(monkeypatch)
    withheld = sorted(_REDISTRIBUTED_WRF_DATA - shipped)
    assert withheld == [], (
        "these files are declared redistributed by gpuwm yet the wheel would "
        f"not contain them: {withheld}\n"
        "Remove the matching [tool.setuptools.exclude-package-data] entry in "
        "gpuwm-data/pyproject.toml under the 'gpuwm_data' key, or widen "
        "[tool.setuptools.package-data]."
    )

    # ...and the shipped bytes must be the pinned WRF v4.6.1 file, not some
    # other 35 KB table: shipping the wrong one is worse than shipping none.
    for relative in sorted(_REDISTRIBUTED_WRF_DATA):
        path = REPO_ROOT / relative
        assert path.is_file(), (
            f"{relative} is declared redistributed but is not in the tree")
        assert path.stat().st_size == next(
            asset.bytes for asset in AEROSOL_TABLE_ASSETS
            if asset.filename == path.name), (
            f"{relative} is present but is not the pinned WRF v4.6.1 file")


def test_the_redistributed_table_is_not_excluded_in_the_pyproject_declaration(
) -> None:
    """The same pin, asserted without setuptools, so it actually runs.

    Every other test in this module routes through
    :func:`_files_setuptools_would_ship`, which begins with
    ``pytest.importorskip("setuptools")``.  That is the right call for a
    measurement of real wheel contents -- a re-implementation of glob
    semantics would be a second authority -- but it has a consequence worth
    stating plainly: in an environment without setuptools those tests SKIP,
    and a skipped packaging gate is indistinguishable from a passing one in a
    summary line.  The project virtualenv used to run this suite is exactly
    such an environment (cupy, numpy, netCDF4, pytest; no setuptools), so the
    strongest control on this file's packaging was silently inert wherever it
    mattered most.

    This asserts the DECLARATION rather than the measurement, out of
    ``tomllib`` in the standard library, so it runs everywhere.  The two are
    complementary and neither replaces the other: this one cannot tell you
    that the file actually reaches the wheel, and the setuptools one cannot
    tell you anything at all when setuptools is absent.  Together they fail
    closed in both environments.
    """

    _require_companion_tree()

    # Both pyprojects, because a stale exclusion could be left in either:
    # the file's directory moved to the companion in 2.5.0, and an entry
    # kept in the root would be dead text while an entry added to the
    # companion would be live breakage.
    excluded: set[str] = set()
    with COMPANION_PYPROJECT.open("rb") as stream:
        companion = tomllib.load(stream)
    excluded |= {
        "gpuwm-data/gpuwm_data/" + entry
        for entry in companion["tool"]["setuptools"].get(
            "exclude-package-data", {}).get("gpuwm_data", [])}
    with PYPROJECT.open("rb") as stream:
        config = tomllib.load(stream)
    excluded |= {
        "gpuwm/" + entry
        for entry in config["tool"]["setuptools"].get(
            "exclude-package-data", {}).get("gpuwm", [])}

    still_excluded = sorted(set(_REDISTRIBUTED_WRF_DATA) & excluded)
    assert still_excluded == [], (
        "a pyproject.toml excludes a file this repository declares it DOES "
        f"redistribute: {still_excluded}\n"
        "Remove each from [tool.setuptools.exclude-package-data] "
        "(gpuwm-data/pyproject.toml under the 'gpuwm_data' key since 2.5.0). "
        "Leaving the entry in place ships an mp_physics=28 that "
        "fails closed on a missing activation table for every user, while "
        "thompson_aerosol_contract.AEROSOL_ASSET_REDISTRIBUTED claims "
        "otherwise."
    )


def test_license_metadata_is_an_spdx_expression_not_the_pasted_text() -> None:
    """``pip show gpuwm`` must answer ``Apache-2.0``, not 202 pasted lines.

    ``license = { file = "LICENSE" }`` inlines the whole Apache-2.0 text
    into the METADATA ``License`` field -- the first field run of the
    published wheel got a screenful from ``pip show``.  PEP 639 spells
    the fix: ``license`` is the SPDX expression, ``license-files`` ships
    the actual texts in the wheel, and no deprecated ``License ::``
    classifier restates either.  setuptools grew that reading at 77, so
    the build floor must sit there too or an isolated build with an
    older backend ships the old flood again.
    """

    _require_source_tree()

    with PYPROJECT.open("rb") as stream:
        config = tomllib.load(stream)

    project = config["project"]
    assert project["license"] == "Apache-2.0", (
        "project.license must be the bare SPDX expression string; a table "
        "(file= or text=) puts the whole license text into `pip show`")

    shipped = project.get("license-files", [])
    assert shipped == ["LICENSE", "NOTICE"], shipped
    for name in shipped:
        assert (REPO_ROOT / name).is_file(), (
            f"pyproject names a license file that does not exist: {name}")

    # PEP 639 deprecates License classifiers, and setuptools >=77 refuses
    # to combine one with an SPDX expression -- a re-added classifier is a
    # broken release cut, not a cosmetic nit.
    stray = [entry for entry in project.get("classifiers", [])
             if entry.startswith("License ::")]
    assert stray == [], stray

    # The floor that makes all of the above real in an isolated build.
    requires = config["build-system"]["requires"]
    assert any(entry.replace(" ", "").startswith("setuptools>=")
               and float(entry.replace(" ", "").split(">=")[1]) >= 77
               for entry in requires), requires

    # And the legacy spelling must not linger to fight the PEP 639 one.
    assert "license-files" not in config.get("tool", {}).get(
        "setuptools", {}), (
        "[tool.setuptools] license-files duplicates [project] license-files")


# ---------------------------------------------------------------------------
# WHICH PACKAGES a wheel carries, as opposed to which data files
# ---------------------------------------------------------------------------
#
# Everything above measures ``package-data`` -- the files inside packages
# that are already known.  This section measures the step before it:
# ``[tool.setuptools.packages.find]``, which decides what a package IS.
#
# The breakage that earned it, measured on a real
# ``python -m build --wheel``: gpuwm-2.5.0-py3-none-any.whl carried
# ``gpuwm-data/gpuwm_data/__init__.py`` and its top_level.txt named
# ``gpuwm-data``, because ``include = ["gpuwm*"]`` is fnmatch over
# DIRECTORY names and the sibling distribution's project directory is
# called ``gpuwm-data``.  Installing that wheel drops a bare
# ``gpuwm-data/gpuwm_data/`` into site-packages: a gpuwm_data package
# with no ``data/`` beside it, which the resolver's checkout rung then
# finds -- one fragment of another distribution, shipped by this one, in
# a position to answer for it.

#: The top-level names this project's wheel may contain, verbatim from
#: the reasoning in pyproject.toml's ``find`` block.  Written out rather
#: than derived from the include patterns, because a pattern that has
#: started matching something new is exactly what this pins.
_DECLARED_TOP_LEVEL = {"gpuwm", "tools", "tilestream", "configs", "docs"}


def _find_declaration(pyproject: Path = PYPROJECT) -> dict[str, list[str]]:
    with pyproject.open("rb") as stream:
        config = tomllib.load(stream)
    return config["tool"]["setuptools"]["packages"]["find"]


def _packages_setuptools_would_ship(pyproject: Path = PYPROJECT,
                                    project_root: Path = REPO_ROOT
                                    ) -> list[str]:
    """The package list the wheel is built from, from setuptools itself.

    ``find_namespace_packages`` is the finder a pyproject ``find``
    directive runs (``namespaces`` defaults to true there), handed this
    project's own declared ``include``/``exclude``.  Asking it rather
    than re-implementing fnmatch is the same discipline the rest of this
    file follows for ``package-data``: what it reports is what the wheel
    would actually contain, and what ``top_level.txt`` would name.
    """

    from setuptools import find_namespace_packages

    declaration = _find_declaration(pyproject)
    return sorted(find_namespace_packages(
        where=str(project_root),
        include=list(declaration.get("include", ["*"])),
        exclude=list(declaration.get("exclude", []))))


def test_the_gpuwm_wheel_carries_no_member_of_the_companion() -> None:
    """The companion's project directory is not a package of this wheel.

    Concretely: a `pip install gpuwm` must not create
    ``site-packages/gpuwm-data/``.  What landed there was
    ``gpuwm_data/__init__.py`` and not one byte of its data, so
    ``import gpuwm_data`` succeeded, ``gpuwm_data.data_root()`` named a
    directory that did not exist, and the companion's real wheel had a
    fragment of itself sitting in the same interpreter claiming the same
    import name.
    """

    _require_source_tree()
    _require_companion_tree()

    stowaways = [name for name in _packages_setuptools_would_ship()
                 if name.split(".")[0] == COMPANION_ROOT.name]
    assert not stowaways, (
        f"the gpuwm wheel would ship {len(stowaways)} package(s) out of "
        f"the companion's own project directory, which belongs to the "
        f"gpuwm-data distribution:\n  " + "\n  ".join(stowaways))


def test_the_gpuwm_wheel_declares_only_the_top_level_names_it_means_to() -> None:
    """``top_level.txt`` is the wheel's claim on the import namespace.

    setuptools writes it from exactly this package list, so a stray root
    here is a name this distribution asserts ownership of in every
    environment it is installed into -- and uninstalling gpuwm would then
    take that directory with it.
    """

    _require_source_tree()

    roots = {name.split(".")[0]
             for name in _packages_setuptools_would_ship()}
    assert roots == _DECLARED_TOP_LEVEL, (
        f"top_level.txt would name {sorted(roots)}; the declaration in "
        f"pyproject.toml means {sorted(_DECLARED_TOP_LEVEL)}. Unexpected: "
        f"{sorted(roots - _DECLARED_TOP_LEVEL)}; missing: "
        f"{sorted(_DECLARED_TOP_LEVEL - roots)}")


def test_the_companion_wheel_carries_exactly_its_one_package() -> None:
    """The arrow points one way: gpuwm-data ships gpuwm_data and nothing.

    The same finder, on the other pyproject.  Its ``include`` names one
    package explicitly for the reason written beside it -- a glob would
    sweep ``data/`` and its subdirectories in as PEP 420 namespace
    packages -- and this is the measurement that the reason still holds.
    """

    _require_companion_tree()

    found = _packages_setuptools_would_ship(COMPANION_PYPROJECT,
                                            COMPANION_ROOT)
    assert found == [COMPANION_PACKAGE_ROOT.name], found


def test_no_wheel_can_carry_the_marker_that_identifies_a_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``gpuwm.data_assets``'s checkout fallback is guarded on this file.

    That guard is only as strong as the claim "a wheel never contains a
    ``pyproject.toml`` beside the package", so the claim is measured
    rather than assumed, on both distributions and from both directions:

    * nothing is a package of the PROJECT ROOT itself, which is the only
      position from which the root ``pyproject.toml`` could be swept in
      as package data;
    * and no shipped data file is that file.

    If either ever stops holding, an installed wheel could carry the
    marker and the fallback would go live inside site-packages again --
    where a directory named ``gpuwm-data`` beside the package gets read
    as reference data with the version check bypassed.
    """

    _require_source_tree()
    _require_companion_tree()

    for pyproject, project_root in ((PYPROJECT, REPO_ROOT),
                                    (COMPANION_PYPROJECT, COMPANION_ROOT)):
        assert (project_root / "pyproject.toml").is_file()
        packages = _packages_setuptools_would_ship(pyproject, project_root)
        assert packages, f"{pyproject} discovers no packages at all"
        assert "" not in packages and "." not in packages, (
            f"{pyproject} makes the project root itself a package, so its "
            "pyproject.toml is a candidate for package-data and the "
            "checkout marker could ship inside a wheel")

    assert "pyproject.toml" not in _files_setuptools_would_ship(monkeypatch)
    assert "gpuwm-data/pyproject.toml" not in _companion_shipped(monkeypatch)


def _companion_sdist_default_files(monkeypatch: pytest.MonkeyPatch
                                   ) -> set[str]:
    """The companion sdist's file set BEFORE its MANIFEST.in is applied.

    ``sdist._add_defaults_python`` is the step that folds
    ``package-data`` into a source distribution, and the reason this
    measurement exists is that it folds ``exclude-package-data`` in with
    it.  That is not obvious from the names -- ``exclude-package-data``
    is documented as governing the wheel -- and a comment in
    ``gpuwm-data/MANIFEST.in`` asserted the opposite for a whole release
    line.  Same shape as ``tests/test_configs_are_packaged.py``'s sdist
    leg, pointed at the other distribution.
    """

    import setuptools
    from setuptools.command.sdist import sdist
    from distutils.filelist import FileList

    with COMPANION_PYPROJECT.open("rb") as stream:
        config = tomllib.load(stream)["tool"]["setuptools"]
    distribution = setuptools.dist.Distribution({
        "name": "gpuwm-data",
        "version": (COMPANION_PACKAGE_ROOT / "VERSION").read_text(
            encoding="utf-8").strip(),
        "packages": [COMPANION_PACKAGE_ROOT.name],
        "package_data": config["package-data"],
        "exclude_package_data": config.get("exclude-package-data", {}),
    })
    distribution.script_name = "setup.py"
    command = sdist(distribution)
    command.finalize_options()
    command.filelist = FileList()
    monkeypatch.chdir(COMPANION_ROOT)
    command._add_defaults_python()
    return {COMPANION_ROOT.name + "/" + name.replace(os.sep, "/")
            for name in command.filelist.files}


def test_the_companion_sdist_carries_the_same_files_as_its_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One exclusion list, two artifacts -- measured, not assumed.

    ``gpuwm-data/MANIFEST.in`` used to say the four CC-BY-NC-SA-4.0 RFMIP
    reference results were "deliberately NOT excluded here", on the
    reasoning that the sdist is the repository's own form and a source
    build must be able to run the suite that reads them.  The built sdist
    disagreed: 14 of the 18 files under ``data/rrtmgp/`` reached it, the
    same 14 the wheel carries, because ``exclude-package-data`` governs
    the sdist's default file set too.

    The breakage a false line there causes is a workflow built on a
    fixture that is not in the artifact -- someone unpacking the sdist to
    reproduce the RFMIP acceptance gates finds four files missing and no
    statement anywhere that says so.

    Both counts move together or neither does.  Deliberately adding the
    fixtures back to the sdist is a licensing decision about a PUBLISHED
    artifact, so it is allowed to fail here: make the change, re-measure,
    and rewrite the paragraph in gpuwm-data/MANIFEST.in in the same
    commit.
    """

    _require_companion_tree()

    prefix = f"{COMPANION_ROOT.name}/{COMPANION_PACKAGE_ROOT.name}/data/"
    in_sdist = {name for name in _companion_sdist_default_files(monkeypatch)
                if name.startswith(prefix)}
    in_wheel = {name for name in _companion_shipped(monkeypatch)
                if name.startswith(prefix)}
    assert in_sdist == in_wheel, (
        "the companion's sdist and wheel carry different data files, so "
        "gpuwm-data/MANIFEST.in and gpuwm-data/pyproject.toml no longer "
        "describe one exclusion list. Only in the sdist: "
        f"{sorted(in_sdist - in_wheel)}; only in the wheel: "
        f"{sorted(in_wheel - in_sdist)}")
    assert not (in_sdist & _LICENSE_EXCLUDED_FROM_WHEEL), (
        "the CC-BY-NC-SA-4.0 RFMIP reference results now reach the sdist; "
        "that is a licensing decision about a published artifact, so "
        "record it: rewrite the paragraph in gpuwm-data/MANIFEST.in that "
        "says which artifacts carry them and re-measure the counts in it")
