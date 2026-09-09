"""The licence layer, pinned the way every other claim in this repository is.

THE BREAKAGE THIS PREVENTS
--------------------------
The 2.7.0 licence lane found two present breaches of PERMISSIVE licences --
NumPy's BSD-3-Clause and MPAS's -- in a tree that already carried a careful
NOTICE, a licenses/ directory and a 670 KB binary-form notice.  Neither was
found by the licence census, because the census sorted third-party code into
"vendored crate" and "pip dependency" and both of these are neither: they are
FIRST-PARTY CRATES THAT PORT THIRD-PARTY CODE INTO THEMSELVES.
``static-fields`` ports four NumPy SIMD kernels; ``rw-mpas`` is 44,768 lines
of MPAS-Atmosphere v8.4.1.  Both ship compiled, in the release bundles and in
platform wheels, and until 2.7.0 neither appeared in any notice.

Nothing checked any of it.  Before this file, ``grep -rIln 'licenses/' tests/``
returned one unrelated hit: no test asserted that ``licenses/*`` exists, that
it reaches a wheel, that the eighteen per-file notice headers survive an edit,
that the kernel notice's file lists still match the tree, or that the
binary-form notice's two copies still agree.  In a project whose discipline is
"pin it or it drifts", the licence layer was the one thing unpinned.

WHAT THIS FILE ASSERTS
----------------------
1.  Every text in ``licenses/`` is non-empty and is NAMED in the root NOTICE,
    and every ``licenses/...`` path the NOTICE names exists.  A text nothing
    points at, and a pointer at nothing, are both failures.
2.  Both distributions declare ``licenses/*`` in ``license-files``, and the
    kernel-directory notice is named in package-data.  This is the only route
    by which a ``pip install`` user receives any of it.
3.  The binary-form notice's two copies are byte-identical and the bundle copy
    sits under a directory ``build_bridge_bundle`` actually walks.
4.  The eighteen files that carry a per-file notice still carry it.
5.  The Arm scope list in ``gpuwm/core/kernels/LICENSE-third-party.txt`` is
    DERIVED FROM THE TREE, not typed: the fourteen files are exactly the ones
    that reproduce Arm's coefficient tables.  The FDLIBM list is pinned, and a
    kernel that starts defining an FDLIBM routine without joining it fails.
6.  Every first-party crate carrying a third-party licence marker, and every
    upstream Fortran project a first-party crate cites, is named in the NOTICE.
    This is the check that would have caught NumPy and MPAS.
7.  The LGPL gamma transcription deleted at 2.7.0 stays deleted, and so do the
    two glibc-only libm fragments removed with it.

It reads text.  No CUDA, no compiler, no network, no device.
"""

from __future__ import annotations

import pathlib
import hashlib
import re
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTICE = ROOT / "NOTICE"
LICENSES = ROOT / "licenses"
KERNELS = ROOT / "gpuwm" / "core" / "kernels"
KERNEL_NOTICE = KERNELS / "LICENSE-third-party.txt"
COMPANION = ROOT / "gpuwm-data"


def _requires_source_tree() -> None:
    """Skip only when this is genuinely not a source checkout.

    The earlier form skipped whenever ``NOTICE`` or ``licenses/`` was absent,
    which made the whole gate fail open: deleting the entire licence layer
    turned 31 assertions into 31 skips and left the suite green.  That is the
    defect class this branch exists to remove -- an absent measurement scored
    as agreement -- so the two cases are separated here.  A tree carrying
    ``pyproject.toml`` IS a source checkout, and in one the licence layer
    missing is a failure, not an absence of evidence.
    """
    if not (ROOT / "pyproject.toml").is_file():
        pytest.skip("not a source checkout; the licence layer is not present")
    missing = [name for name, ok in (("NOTICE", NOTICE.is_file()),
                                     ("licenses/", LICENSES.is_dir())) if not ok]
    if missing:
        raise AssertionError(
            "source checkout is missing the licence layer: "
            + ", ".join(missing)
            + " -- these ship with the tree and their absence is a failure, "
              "not a reason to skip")


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _kernel_sources() -> list[pathlib.Path]:
    return sorted(list(KERNELS.glob("*.cu")) + list(KERNELS.glob("*.cuh")))


def _decommented(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


# ---------------------------------------------------------------------------
# 1.  the texts exist, are non-empty, and the NOTICE points at each of them
# ---------------------------------------------------------------------------
def test_every_licence_text_is_named_by_the_notice_and_vice_versa() -> None:
    _requires_source_tree()
    notice = _read(NOTICE)

    empty = [p.name for p in sorted(LICENSES.iterdir()) if p.stat().st_size == 0]
    assert empty == [], f"licence texts with no text: {empty}"

    unnamed = [p.name for p in sorted(LICENSES.iterdir())
               if f"licenses/{p.name}" not in notice]
    assert unnamed == [], (
        "the NOTICE names no path for these licence texts, so a reader has no "
        f"way to find them: {unnamed}")

    cited = {m.rstrip(".,;") for m in
             re.findall(r"licenses/[A-Za-z0-9._-]+", notice)}
    cited.discard("licenses/licenses")       # the PEP 639 wheel path, not a file
    dangling = sorted(name for name in cited if not (ROOT / name).exists())
    assert dangling == [], f"the NOTICE points at licence texts that are not here: {dangling}"


def test_the_notice_names_every_grant_that_conditions_reproduction() -> None:
    """A named-grant checklist, so a section cannot quietly disappear.

    Each entry is a grant whose text this distribution is required to
    reproduce, and the token is the shortest string that identifies its
    section.  MPAS and NumPy are on the list because 2.7.0 is when they were
    found missing, and a checklist that only holds what was already right
    would have held in 2.6.5 too.
    """
    _requires_source_tree()
    notice = _read(NOTICE)
    required = (
        "Arm Limited",                    # MIT
        "Sun Microsystems",               # FDLIBM
        "Atmospheric and Environmental Research",   # AER RRTMG, RTE+RRTMGP
        "UChicago Argonne",               # Py-ART
        "Los Alamos National Security",   # MPAS
        "NumPy Developers",               # NumPy
        "Numerical Recipes",
        "SIL Open Font License",
        "public domain",                  # WRF / UCAR
        "WRF Preprocessing System",       # WPS
    )
    missing = [token for token in required if token not in notice]
    assert missing == [], f"the NOTICE no longer names: {missing}"


# ---------------------------------------------------------------------------
# 2.  the texts reach a built distribution
# ---------------------------------------------------------------------------
def test_both_distributions_ship_the_licence_directory() -> None:
    _requires_source_tree()
    for pyproject, licences in ((ROOT / "pyproject.toml", LICENSES),
                                (COMPANION / "pyproject.toml",
                                 COMPANION / "licenses")):
        if not pyproject.is_file():
            pytest.skip(f"{pyproject} is not in this checkout")
        with pyproject.open("rb") as stream:
            config = tomllib.load(stream)
        declared = config["project"].get("license-files", [])
        assert "licenses/*" in declared, (
            f"{pyproject} does not ship licenses/, so a pip install of it "
            f"performs no reproduction condition at all: {declared}")
        assert licences.is_dir() and any(licences.iterdir()), (
            f"{pyproject} globs {licences}, which is missing or empty")


def test_the_kernel_directory_notice_is_declared_as_package_data() -> None:
    """The .cu files are digest-pinned, so their notice sits beside them.

    That only works if it ships.  ``package-data`` is the one thing that
    carries a non-Python file inside a package directory into the wheel.
    """
    _requires_source_tree()
    assert KERNEL_NOTICE.is_file(), f"{KERNEL_NOTICE} is missing"
    with (ROOT / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    entries = config["tool"]["setuptools"]["package-data"]["gpuwm.core.kernels"]
    assert "LICENSE-third-party.txt" in entries, entries


def test_the_binary_form_notice_has_two_byte_identical_copies() -> None:
    """One copy for the wheel, one for the bundle; a drifted pair is a bundle
    shipping a different notice from the wheel built at the same commit."""
    _requires_source_tree()
    wheel_copy = LICENSES / "THIRD-PARTY-LICENSES-bridge-binaries.txt"
    bundle_copy = (ROOT / "tools" / "rustwx" / "assets" / "basemap"
                   / "THIRD-PARTY-LICENSES.txt")
    assert wheel_copy.is_file(), f"{wheel_copy} is missing"
    assert bundle_copy.is_file(), f"{bundle_copy} is missing"
    assert wheel_copy.read_bytes() == bundle_copy.read_bytes(), (
        "the two copies of the binary-form notice have drifted; regenerate "
        "both together")

    # The terminal executable has its own locked dependency tree. A new lock
    # or feature selection must refresh the notice delivered with that binary.
    notice = _read(wheel_copy)
    for name in ("Cargo.toml", "Cargo.lock"):
        path = ROOT / "tools" / "arwen-tui" / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert f"tools/arwen-tui/{name} sha256 {digest}" in notice, (
            "TUI dependency notices are stale; run tools/update_tui_license_notice.py")

    from gpuwm import bridge_assets
    assert bundle_copy.parent.name in bridge_assets.REQUIRED_ASSET_SUBDIRS, (
        "the bundle copy no longer sits under a directory "
        "build_bridge_bundle.collect_assets() walks, so it stops travelling "
        f"with the binaries: {bridge_assets.REQUIRED_ASSET_SUBDIRS}")


def test_the_binary_form_notice_covers_the_first_party_ports() -> None:
    _requires_source_tree()
    text = _read(LICENSES / "THIRD-PARTY-LICENSES-bridge-binaries.txt")
    for token in ("UChicago Argonne", "Los Alamos National Security",
                  "NumPy Developers", "SIL OPEN FONT LICENSE"):
        assert token in text, (
            f"the binary-form notice does not reproduce {token!r}; a compiled "
            "consumer never sees the source tree, so this is the only copy "
            "that reaches them")


# ---------------------------------------------------------------------------
# 3.  the per-file notices survive an edit
# ---------------------------------------------------------------------------
#: Files that open with a THIRD-PARTY NOTICE block because they carry
#: transcribed work.  Measured on the 2.7.0 tree; a file that carries
#: transcription joins the list, and a file that loses its header fails.
NOTICE_CARRIERS: tuple[str, ...] = (
    "gpuwm/core/kernels/glibc_flt32.cuh",
    "gpuwm/core/kernels/thompson_aerosol_common.cuh",
    "gpuwm/core/milbrandt2_constants.py",
    "gpuwm/core/mynn_pbl.py",
    "gpuwm/core/noahmp_libm.py",
    "gpuwm/core/rrtm.py",
    "gpuwm/core/rrtm_lw.py",
    "gpuwm/core/rrtm_tables.py",
    "gpuwm/core/rrtm_taumol.py",
    "gpuwm/core/rrtmg_legacy.py",
    "gpuwm/core/rrtmg_legacy_prep.py",
    "gpuwm/core/rrtmg_lw.py",
    "gpuwm/core/rrtmg_mcica.py",
    "gpuwm/core/rrtmg_sw.py",
    "gpuwm/core/rrtmgp.py",
    "gpuwm/core/ruc.py",
    "gpuwm/core/thompson_aerosol_contract.py",
    "gpuwm/obs/dealias_region.py",
)


@pytest.mark.parametrize("relative", NOTICE_CARRIERS)
def test_every_notice_carrier_still_opens_with_its_notice(relative) -> None:
    _requires_source_tree()
    path = ROOT / relative
    assert path.is_file(), f"{relative} is gone; the NOTICE still cites it"
    head = "".join(_read(path).splitlines(True)[:12])
    assert "THIRD-PARTY NOTICE" in head, (
        f"{relative} no longer opens with its third-party notice")


def test_no_file_carries_a_notice_this_list_does_not_know_about() -> None:
    """The other direction: a new carrier must join the list above."""
    _requires_source_tree()
    found = []
    for path in sorted(list((ROOT / "gpuwm").rglob("*.py"))
                       + list(KERNELS.glob("*.cuh"))):
        head = "".join(_read(path).splitlines(True)[:12])
        if "THIRD-PARTY NOTICE" in head:
            found.append(path.relative_to(ROOT).as_posix())
    assert sorted(found) == sorted(NOTICE_CARRIERS), (
        "NOTICE_CARRIERS is out of step with the tree:\n"
        f"  only in tree: {sorted(set(found) - set(NOTICE_CARRIERS))}\n"
        f"  only in list: {sorted(set(NOTICE_CARRIERS) - set(found))}")


# ---------------------------------------------------------------------------
# 4.  the kernel notice's scope lists match the tree
# ---------------------------------------------------------------------------
#: Any spelling of Arm's logf / exp2f / powf coefficient tables.  A file that
#: reproduces one of these reproduces Arm's work and is inside the MIT grant.
_ARM_TABLE = re.compile(
    r"exp2f_tab|logf_tab|logf_invc|powf_log2_tab|powf_invc", re.I)


def _arm_files() -> list[str]:
    return sorted(p.name for p in _kernel_sources()
                  if _ARM_TABLE.search(_read(p)))


def test_the_arm_scope_list_is_derived_from_the_tree() -> None:
    """The NOTICE says this list is machine-derived.  This is the machine."""
    _requires_source_tree()
    derived = _arm_files()
    assert len(derived) == 14, derived
    listed = _read(KERNEL_NOTICE)
    missing = [name for name in derived if name not in listed]
    assert missing == [], (
        "gpuwm/core/kernels/LICENSE-third-party.txt does not name every file "
        f"that reproduces Arm's tables: {missing}")
    notice = _read(NOTICE)
    missing = [name for name in derived if name not in notice]
    assert missing == [], f"the root NOTICE's Arm list is short of: {missing}"


#: The kernel translation units carrying an FDLIBM-descended reduction
#: (expm1f, tanhf, atanf, log10f).  Pinned rather than derived because each
#: file names the routine differently -- r_atan, ng_atanf, glibc_atanf,
#: mynn_tanhf, nmpe_expm1f, ruc_tanhf_glibc -- and the derivation below is
#: what keeps the pin honest.
FDLIBM_FILES: tuple[str, ...] = (
    "mynn_dmp_sibling.cu", "mynn_pbl.cu", "noahmp_bareflux.cu",
    "noahmp_energy.cu", "noahmp_fluxprep.cu", "noahmp_glacier.cu",
    "noahmp_leaves.cu", "noahmp_vegeflux.cu", "ruc.cu",
)

#: Device routines whose NAME looks like an FDLIBM one but which are not
#: transcriptions: each is a double-precision libm call rounded once, and each
#: file says so at the definition.  Listed with the reason so that adding a
#: real transcription here is a deliberate act.
_NOT_FDLIBM = {
    "p3.cu": ("p3_log10",),                      # (float)log10((double)x)
    "thompson_aerosol_warm.cu": ("thompson_aa_log10f_cr",),
    "ruc.cu": ("ruc_log10f_rn", "ruc_expm1f_glibc"),
}

_FDLIBM_DEF = re.compile(
    r"__device__[^;{]{0,160}?\b(\w*(?:atan|tanh|log10|expm1)\w*)\s*\(")


def test_the_fdlibm_scope_list_matches_the_tree() -> None:
    _requires_source_tree()
    listed = _read(KERNEL_NOTICE)
    missing = [name for name in FDLIBM_FILES if name not in listed]
    assert missing == [], (
        f"the kernel notice's FDLIBM list is short of: {missing}")

    unnotified = {}
    for path in _kernel_sources():
        names = sorted({m.group(1) for m
                        in _FDLIBM_DEF.finditer(_decommented(_read(path)))})
        real = [n for n in names if n not in _NOT_FDLIBM.get(path.name, ())]
        if real and path.name not in FDLIBM_FILES:
            unnotified[path.name] = real
    assert unnotified == {}, (
        "these kernels define an FDLIBM-shaped routine and are not on the "
        "FDLIBM notice list; either they transcribe FDLIBM and the notice "
        "must say so, or they call libm and belong in _NOT_FDLIBM with the "
        f"reason beside them: {unnotified}")


# ---------------------------------------------------------------------------
# 5.  first-party crates that carry third-party expression
# ---------------------------------------------------------------------------
#: Source roots ArWen WROTE.  Not vendor trees and not pip dependencies --
#: the third category, which is where both 2.7.0 breaches were.
_FIRST_PARTY_ROOTS = (
    "tools/rustwx/crates", "tools/rw_wps/crates", "tools/grib1_bridge/src",
    "tools/region_global_dealias/src", "tools/mpas_render_bridge",
)

_LICENCE_MARKER = re.compile(
    r"Copyright\s*\((?:c|C)\)|SPDX-License-Identifier|BSD-3-Clause"
    r"|BSD 3-Clause|MIT License")

#: ``file that carries a marker -> a token the root NOTICE must contain``.
#: A first-party file announcing someone else's copyright and no NOTICE entry
#: is the exact shape of the NumPy breach.
MARKER_FILES: dict[str, str] = {
    "tools/region_global_dealias/src/solver.rs": "UChicago Argonne",
    "tools/rustwx/crates/static-fields/src/projection/npmath.rs":
        "NumPy Developers",
}


def _first_party_rust() -> list[pathlib.Path]:
    out = []
    for relative in _FIRST_PARTY_ROOTS:
        root = ROOT / relative
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.rs")):
            text = path.as_posix()
            if "/vendor/" in text or "/target/" in text:
                continue
            out.append(path)
    return out


def test_first_party_code_announcing_a_foreign_copyright_is_notified() -> None:
    _requires_source_tree()
    sources = _first_party_rust()
    if not sources:
        pytest.skip("the Rust trees are not in this checkout")
    notice = _read(NOTICE)
    found = {}
    for path in sources:
        if _LICENCE_MARKER.search(_read(path)):
            found[path.relative_to(ROOT).as_posix()] = True
    unknown = sorted(set(found) - set(MARKER_FILES))
    assert unknown == [], (
        "these first-party files announce a third-party copyright or licence "
        "and no NOTICE entry is mapped to them.  A first-party crate that "
        "ports third-party code is in neither the vendor census nor the pip "
        f"census, which is how NumPy went unnotified until 2.7.0: {unknown}")
    for relative, token in MARKER_FILES.items():
        assert (ROOT / relative).is_file(), f"{relative} is gone"
        assert token in notice, (
            f"{relative} carries third-party expression and the NOTICE no "
            f"longer names {token!r}")


#: Upstream Fortran projects a first-party crate transcribes, and the token
#: the NOTICE must carry for each.  The classifier below is what says which
#: project a cited file belongs to.
UPSTREAM_TOKENS = {
    "MPAS": "Los Alamos National Security",
    "WPS": "WRF Preprocessing System",
    "WRF": "public domain",
}

#: Cited Fortran sources with NO counterpart in the WRF distribution, so they
#: are WPS's own.  Measured against a WRF v4.6.1 checkout.
_WPS_ONLY = frozenset({
    "rrpr.F", "gribcode.F", "output.F", "rd_grib1.F", "rd_grib2.F",
    "new_storage.F", "process_tile_module.F", "interp_module.F",
    "read_met_module.F90",
})

_FORTRAN_CITATION = re.compile(r"\b([A-Za-z0-9_]+\.(?:F90|F))\b")


def test_every_upstream_a_first_party_crate_transcribes_is_notified() -> None:
    """MPAS was found this way: 44,768 lines of port, no notice, five shipped
    binaries.  Nothing in the tree announced it except the Fortran file names
    in its own doc comments, which is what this reads."""
    _requires_source_tree()
    sources = _first_party_rust()
    if not sources:
        pytest.skip("the Rust trees are not in this checkout")
    cited = set()
    for path in sources:
        cited.update(m.group(1) for m in _FORTRAN_CITATION.finditer(_read(path)))

    projects = set()
    unclassified = []
    for name in sorted(cited):
        if name.startswith("mpas_"):
            projects.add("MPAS")
        elif name in _WPS_ONLY:
            projects.add("WPS")
        elif name.startswith("module_") or name.startswith("mp_"):
            projects.add("WRF")
        else:
            unclassified.append(name)
    assert unclassified == [], (
        "a first-party crate cites Fortran sources this classifier does not "
        "recognise.  Decide which upstream they belong to and give that "
        f"upstream a NOTICE entry before adding them here: {unclassified}")

    notice = _read(NOTICE)
    missing = sorted(project for project in projects
                     if UPSTREAM_TOKENS[project] not in notice)
    assert missing == [], (
        f"first-party crates transcribe {missing} and the NOTICE does not "
        "name them")


# ---------------------------------------------------------------------------
# 6.  what 2.7.0 removed stays removed
# ---------------------------------------------------------------------------
_LGPL_GAMMA_DEFINITION = re.compile(
    r"^\s*(?:__device__|static|float|double)\s+.*"
    r"\b(gfk_gammaf_positive|gfk_gamma_product|gfk_lgamma_pos)\s*\(", re.M)


def test_the_lgpl_gamma_transcription_stays_deleted() -> None:
    """glibc's gammaf_positive / __gamma_productf are FSF-copyright and
    LGPL-2.1-or-later with no permissive upstream.  No notice cures that; only
    deletion does, and only staying deleted keeps it cured."""
    _requires_source_tree()
    back = [p.name for p in _kernel_sources()
            if _LGPL_GAMMA_DEFINITION.search(_read(p))]
    assert back == [], f"the LGPL gamma transcription is back in: {back}"

    fingerprints = ("exp2_adj", "x_adj_mant", "gamma_coeff", "0xBB360B61",
                    "0x3A500D01")
    for path in _kernel_sources():
        text = _decommented(_read(path))
        hit = [token for token in fingerprints if token in text]
        assert hit == [], f"{path.name} carries glibc gamma fingerprints: {hit}"


def test_the_two_glibc_only_libm_fragments_stay_removed() -> None:
    """Both were glibc's own edits to FDLIBM code, both did no work, and both
    were compared against their originals on all 4,294,967,296 float32 bit
    patterns with zero differing outputs before removal."""
    _requires_source_tree()
    tanh_guard = ("gpuwm/core/kernels/noahmp_energy.cu", "if (ix == 0) return x;")
    log10_div = ("gpuwm/core/kernels/noahmp_leaves.cu",
                 "DV(-two25, fabsf(x))")
    py_log10 = ("gpuwm/core/noahmp_libm.py", "-_TWO25 / abs(value)")
    for relative, fragment in (tanh_guard, log10_div, py_log10):
        # Comments are stripped first: the sites that lost these fragments
        # each carry a comment QUOTING what was removed and why, which is the
        # point of the edit, and a naive substring search would read the
        # explanation as the offence.
        raw = _read(ROOT / relative)
        text = (re.sub(r"#[^\n]*", "", raw) if relative.endswith(".py")
                else _decommented(raw))
        assert fragment not in text, (
            f"{relative} has taken glibc's {fragment!r} back; it is "
            "expression this Apache-2.0 distribution removed at 2.7.0")

    python_tanh = re.sub(r"#[^\n]*", "",
                         _read(ROOT / "gpuwm/core/noahmp_libm.py"))
    body = python_tanh.split("def tanhf(", 1)[1].split("\ndef ", 1)[0]
    assert "if ix == 0:" not in body, (
        "gpuwm/core/noahmp_libm.py's tanhf has taken glibc's redundant zero "
        "guard back")
