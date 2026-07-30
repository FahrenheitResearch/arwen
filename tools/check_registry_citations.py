"""Re-resolve every ``file:line`` citation the physics registry publishes.

The registry's warnings and unimplemented reasons are its evidence, and they
cite line numbers.  Line numbers rot.  A user-visible MYNN warning cited
``gpuwm/core/mynn_pbl.py:3601`` as the site where the snow mixing ratio is
zero-substituted; :3601 is ``pmz = np.empty(ncol, ...)`` and the substitution is
at :3637.  Its companion ``mynn_pbl_gpu.py:1260`` was wrong when it was written
and became correct only because a later commit happened to delete twelve lines
above it.  Nothing checked either one.

Existence is not enough to catch that: :3601 exists.  So each repo-local
citation carries an ANCHOR -- a substring that must appear inside the cited
line or range -- and the anchor is the thing the surrounding claim is about.
A citation that drifts by one line loses its anchor and fails here.

Three tables, and all three are exhaustive by construction:

``RESOLVED``
    citation text -> (repo-relative path, anchor).  Every citation whose file
    lives in this worktree must appear, and every row must still appear in the
    registry, so a stale row fails as loudly as a missing one.

``EXTERNAL``
    cited path -> the tree it belongs to.  WRF v4.6.1 is not vendored here, so
    those citations are checked for shape and for being DECLARED external --
    which is what stops a typo in a repo path from being silently downgraded to
    "must be somebody else's file".

``AMBIGUOUS``
    citation text -> the repo-relative path meant, for a bare filename that
    matches more than one file.  ``run_driver.F90`` is both the MYNN and the
    Noah-MP oracle harness; which one a MYNN warning means is a judgement, so
    it is recorded rather than guessed.

Bare continuation line numbers (``module_bl_mynn.F:3872-3873 ... and :4618``)
inherit the last filename named in the same string, which is how they are
written and how they read.

Usage
-----
    python tools/check_registry_citations.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

MODEL = pathlib.Path(__file__).resolve().parents[1]
REGISTRY_PATH = MODEL / "gpuwm" / "physics_registry_v2.json"

#: A path-looking token, optionally followed by ``:line`` or ``:line-line``, or
#: a bare ``:line`` that inherits the previous token's path.  Longer extensions
#: come first: ``F`` before ``F90`` would eat ``run_driver.F`` out of
#: ``run_driver.F90:101`` and then attribute ``:205`` to a file that does not
#: exist, which is exactly what an earlier version of this scanner did.
_TOKEN = re.compile(
    r"(?P<file>(?:[A-Za-z0-9_.\-]+/)*[A-Za-z0-9_\-]+"
    r"\.(?:py|cu|F90|f90|F|json|md|csv))"
    r"(?::(?P<l1>\d+)(?:\s*-\s*(?P<l2>\d+))?)?"
    r"|(?<![\w.])(?P<bare>:(?P<b1>\d+)(?:\s*-\s*(?P<b2>\d+))?)")

_WRF = "WRF v4.6.1 @ d66e442fccc04111067e29274c9f9eaccc3cef28, not vendored here"

#: Citations into another source tree.  Declared, not inferred: an unknown file
#: that happens not to exist here must fail rather than be assumed external.
EXTERNAL: dict[str, str] = {
    "module_bl_mynn.F": _WRF,
    "phys/module_bl_mynn.F": _WRF,
    "module_pbl_driver.F": _WRF,
    "phys/module_pbl_driver.F": _WRF,
    "module_sf_mynn.F": _WRF,
    "phys/module_sf_mynn.F": _WRF,
    "module_sf_ruclsm.F": _WRF,
    "phys/module_sf_ruclsm.F": _WRF,
    "module_surface_driver.F": _WRF,
    "phys/module_surface_driver.F": _WRF,
    "module_soil_pre.F": _WRF,
    "share/module_soil_pre.F": _WRF,
    "module_check_a_mundo.F": _WRF,
    "share/module_check_a_mundo.F": _WRF,
    "module_sf_noahdrv.F": _WRF,
    "phys/module_sf_noahdrv.F": _WRF,
}

#: A bare filename that resolves to more than one file in this worktree, and
#: which one the citing text means.
AMBIGUOUS: dict[str, str] = {
    "run_driver.F90": "tools/mynn_pbl_wrf461_oracle/run_driver.F90",
}

#: citation -> (repo-relative path, anchor that must appear in the cited lines).
#:
#: The anchor is the token the CLAIM is about, never simply whatever the line
#: happens to say -- picking the anchor from the current line content would
#: bless a drifted citation as correct, which is the failure this table exists
#: to prevent.  Twelve of these citations were wrong when the table was first
#: filled in and are marked.
RESOLVED: dict[str, tuple[str, str]] = {
    # --- component warnings -------------------------------------------------
    # REPOINTED from :3601, which is ``pmz = np.empty(ncol, ...)``.
    "gpuwm/core/mynn_pbl.py:3637": (
        "gpuwm/core/mynn_pbl.py", '"qs": kzero'),
    "mynn_pbl_gpu.py:1260": (
        "gpuwm/core/mynn_pbl_gpu.py", '"qs": kzero'),
    "run_driver.F90:101": (
        "tools/mynn_pbl_wrf461_oracle/run_driver.F90", "flag_qs = .false."),
    "run_driver.F90:205": (
        "tools/mynn_pbl_wrf461_oracle/run_driver.F90", "sqs3d(c, k) = 0.0"),
    "tests/test_mynn_pbl_runtime.py:49": (
        "tests/test_mynn_pbl_runtime.py",
        "sf_sfclay_physics=5, sf_surface_physics=2"),

    # --- unimplemented-knob reasons ----------------------------------------
    # REPOINTED from :387 (``"bl_mynn_edmf_mom": 1``): the claim is about the
    # accepted mp_physics set, which is the check in validate_run_config.
    "gpuwm/config.py:862": (
        "gpuwm/config.py", "mp_physics not in (0, 1, 6, 8, 10, 18)"),
    "gpuwm/core/kernels/kf.cu:417-421": (
        "gpuwm/core/kernels/kf.cu", "w_scaled = wlcl * dx / 25000.0f"),
    "kf.cu:594-601": (
        "gpuwm/core/kernels/kf.cu", "shallow_candidate = selected_candidate;"),
    "kf.cu:1211": (
        "gpuwm/core/kernels/kf.cu", "shallow_out[column] = shallow ? 1 : 0;"),
    "gpuwm/core/kernels/morrison.cu:8": (
        "gpuwm/core/kernels/morrison.cu", "fixed 250 cm-3 cloud droplets"),
    "gpuwm/core/kernels/nssl2.cu:217": (
        "gpuwm/core/kernels/nssl2.cu", "alphahl=1 maps the 4e4 m-4 intercept"),
    "gpuwm/core/kernels/nssl2.cu:3297": (
        "gpuwm/core/kernels/nssl2.cu", "velocities (icdx=6)"),
    "gpuwm/core/kernels/nssl2_diagnostics.cu:75": (
        "gpuwm/core/kernels/nssl2_diagnostics.cu",
        "alphah=0, 0.3--20-mm volume bounds"),
    "gpuwm/core/kernels/nssl2_diagnostics.cu:106": (
        "gpuwm/core/kernels/nssl2_diagnostics.cu",
        "alphahl=1, 0.3--40-mm bounds"),
    "gpuwm/core/kernels/nssl2_driver_support.cu:900": (
        "gpuwm/core/kernels/nssl2_driver_support.cu", "velocities (icdx=6)"),
    "gpuwm/core/kernels/nssl2_fused_gs.cu:1923": (
        "gpuwm/core/kernels/nssl2_fused_gs.cu", "alphar=alphah=0"),
    "gpuwm/core/kernels/sfclay.cu:426": (
        "gpuwm/core/kernels/sfclay.cu", "if (isfflx && !land)"),
    "gpuwm/core/kernels/sfclay.cu:441": (
        "gpuwm/core/kernels/sfclay.cu", "if (isfflx)"),
    "gpuwm/core/landuse.py:180": (
        "gpuwm/core/landuse.py", "0.02 if fractional_seaice else 0.5"),
    "gpuwm/core/landuse.py:190-203": (
        "gpuwm/core/landuse.py", "ivgtyp > table.lucats"),
    "gpuwm/core/landuse.py:231": (
        "gpuwm/core/landuse.py", "if fractional_seaice:"),
    "gpuwm/core/landuse.py:239": (
        "gpuwm/core/landuse.py", "np.where(seaice, albbck, albedo)"),
    # REPOINTED from :14 only in company: the docstring scope note is a real
    # citation for "not ported", so it stays.
    "gpuwm/core/noah.py:14": ("gpuwm/core/noah.py", "UA_PHYS"),
    # REPOINTED from :131 (``sbeta: float = 0.0``) to where VEGPARM's category
    # count is actually read.
    "gpuwm/core/noah.py:151": (
        "gpuwm/core/noah.py", "lucats = int(cats_line[0])"),
    # REPOINTED from :178 (``t.cmcmax = float(val())``).
    "gpuwm/core/noah.py:198": (
        "gpuwm/core/noah.py", "t.sltype, t.slcats = sltype, cats"),
    # REPOINTED from :335 (the sh2o_init docstring) to the range check.
    "gpuwm/core/noah.py:355": (
        "gpuwm/core/noah.py", "category > params.slcats"),
    # REPOINTED from :363-374 (the FRH2O loop) to the state declaration that
    # actually shows there is no tile dimension.
    "gpuwm/core/noah.py:381-392": (
        "gpuwm/core/noah.py", '_F3D = ("smois", "tslb", "sh2o", "smcrel")'),
    # REPOINTED from gpuwm/core/noah.py:387-389, a field-name list, to the
    # kernel line that performs the sea-ice skip the claim is about.
    "gpuwm/core/kernels/noah.cu:961-965": (
        "gpuwm/core/kernels/noah.cu", "xice_a[idx] >= xice_threshold"),
    "gpuwm/core/nssl2_contract.py:40": (
        "gpuwm/core/nssl2_contract.py", "SIXTH_MOMENT_FIELDS"),
    "gpuwm/core/nssl2_contract.py:149": (
        "gpuwm/core/nssl2_contract.py", "density = (2 if hail else 1)"),
    "gpuwm/core/nssl2_contract.py:167": (
        "gpuwm/core/nssl2_contract.py", "DEFAULT_MODE = resolve_nssl2_mode()"),
    # REPOINTED from :1103-1109 (a cumulus tendency block) to the call site the
    # claim is about.  The claim itself was also wrong and is corrected: the
    # MYNN arm passes isfflx=1 explicitly at :1332.
    "gpuwm/core/physics.py:1332": ("gpuwm/core/physics.py", "isfflx=1,"),
    "gpuwm/core/physics.py:1335-1341": (
        "gpuwm/core/physics.py",
        "isftcflx=cfg.isftcflx, iz0tlnd=cfg.iz0tlnd)"),
    "gpuwm/core/rrtmgp.py:895-897": (
        "gpuwm/core/rrtmgp.py", "radii require effc, effi, and effs"),
    "gpuwm/core/rrtmgp.py:1733": ("gpuwm/core/rrtmgp.py", "cal_cldfra1"),
    "gpuwm/core/sfclay.py:29": ("gpuwm/core/sfclay.py", '"u10", "v10"'),
    "gpuwm/core/sfclay.py:112": (
        "gpuwm/core/sfclay.py", "isfflx: bool = True"),
    "gpuwm/core/state.py:390": (
        "gpuwm/core/state.py", '"effc", "effi", "effs"'),
    "gpuwm/core/state.py:395-397": (
        "gpuwm/core/state.py", '"qvolg", "qvolh"'),
    "gpuwm/core/state.py:405": (
        "gpuwm/core/state.py", "DTYPE(408163264.0)"),
    "gpuwm/core/state.py:498": ("gpuwm/core/state.py", "self.ht = zeros"),
    "gpuwm/experiment.py:582": (
        "gpuwm/experiment.py", 'feedback = exp.get("feedback", 0)'),
    "gpuwm/experiment.py:590": (
        "gpuwm/experiment.py", 'smooth_option = exp.get("smooth_option", 0)'),
    "gpuwm/ingest/soil.py:277": (
        "gpuwm/ingest/soil.py", "np.where(terrestrial | sea_ice, skin"),
    "gpuwm/ingest/soil.py:358-362": (
        "gpuwm/ingest/soil.py", "a 3 m ice column"),
    "gpuwm/ingest/soil.py:360": (
        "gpuwm/ingest/soil.py", "tmn = np.array(deep"),
    "gpuwm/namelist_compat.py:68": (
        "gpuwm/namelist_compat.py", '"o3input", "aer_opt"'),
    # REPOINTED from :956, :957-958, :963-964 and :971-975.  All four had
    # drifted onto the cumulus and isfflx blocks above them; the drop table the
    # claims quote starts twenty-three lines later.
    "gpuwm/namelist_import.py:979": (
        "gpuwm/namelist_import.py", "Noah snow physics is always active"),
    "gpuwm/namelist_import.py:980-981": (
        "gpuwm/namelist_import.py", "surface fields come from the"),
    "gpuwm/namelist_import.py:986-987": (
        "gpuwm/namelist_import.py", "SST update cycling is not implemented"),
    "gpuwm/namelist_import.py:995-1002": (
        "gpuwm/namelist_import.py", "mosaic land/soil physics is not"),
    # REPOINTED from :488, which is one line above the TMN read.
    "gpuwm/runtime.py:487-489": (
        "gpuwm/runtime.py", 'deep_soil_temperature=static["TMN"]'),
    "gpuwm/runtime.py:533-534": (
        "gpuwm/runtime.py", 'static["SNOALB"] / 100.0'),
    "gpuwm/verify/npref.py:4679": ("gpuwm/verify/npref.py", "UA_PHYS"),
    "gpuwm/wrf_direct_v461_contract.json:89": (
        "gpuwm/wrf_direct_v461_contract.json", '"AERCU_OPT": 0'),
    "gpuwm/wrf_direct_v461_contract.json:90": (
        "gpuwm/wrf_direct_v461_contract.json", '"AERCU_FCT": 1.0'),
    "gpuwm/wrf_direct_v461_contract.json:209": (
        "gpuwm/wrf_direct_v461_contract.json", '"AERCU_OPT"'),
    "gpuwm/wrf_direct_v461_contract.json:214": (
        "gpuwm/wrf_direct_v461_contract.json", '"AERCU_FCT"'),
}


def citations(registry: dict) -> dict[str, list[str]]:
    """Every ``file:line`` citation in the registry -> the paths carrying it."""

    found: dict[str, list[str]] = {}

    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in sorted(node.items()):
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, str):
            last: str | None = None
            for match in _TOKEN.finditer(node):
                if match.group("file"):
                    last = match.group("file")
                    if not match.group("l1"):
                        continue
                    first, second = match.group("l1"), match.group("l2")
                elif last is not None:
                    first, second = match.group("b1"), match.group("b2")
                else:
                    continue
                citation = f"{last}:{first}"
                if second:
                    citation += f"-{second}"
                found.setdefault(citation, [])
                if path not in found[citation]:
                    found[citation].append(path)

    walk(registry, "$")
    return found


def _resolve(cited_path: str) -> list[str]:
    """Repo-relative matches for a cited path, exact or by unique suffix."""

    if (MODEL / cited_path).is_file():
        return [cited_path]
    hits = []
    for candidate in MODEL.rglob(pathlib.Path(cited_path).name):
        relative = candidate.relative_to(MODEL).as_posix()
        if relative.endswith("/" + cited_path) and "__pycache__" not in relative:
            hits.append(relative)
    return sorted(hits)


def check(registry: dict | None = None) -> list[str]:
    """Return one message per unresolved, unanchored or stale citation."""

    if registry is None:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    found = citations(registry)
    failures: list[str] = []

    for citation, where in sorted(found.items()):
        cited_path, _, spec = citation.rpartition(":")
        if cited_path in EXTERNAL:
            if _resolve(cited_path):
                failures.append(
                    f"{citation} (in {where[0]}) is declared EXTERNAL but a "
                    "file of that name exists in this worktree; declare which "
                    "one is meant in AMBIGUOUS/RESOLVED instead")
            continue
        row = RESOLVED.get(citation)
        if row is None:
            hits = _resolve(cited_path)
            if not hits:
                failures.append(
                    f"{citation} (in {where[0]}) names a file that is neither "
                    "in this worktree nor declared in EXTERNAL")
            else:
                failures.append(
                    f"{citation} (in {where[0]}) resolves to {hits} and has no "
                    "RESOLVED row, so nothing checks that the cited line still "
                    "says what the claim says it does")
            continue
        path, anchor = row
        declared = AMBIGUOUS.get(cited_path)
        if declared is not None and declared != path:
            failures.append(
                f"{citation}: AMBIGUOUS says {declared!r}, RESOLVED says "
                f"{path!r}")
            continue
        target = MODEL / path
        if not target.is_file():
            failures.append(f"{citation}: {path} does not exist")
            continue
        lines = target.read_text(encoding="utf-8",
                                 errors="replace").splitlines()
        first, _, second = spec.partition("-")
        low, high = int(first), int(second or first)
        if not 1 <= low <= high <= len(lines):
            failures.append(
                f"{citation}: {path} has {len(lines)} lines, so {low}-{high} "
                "is out of range")
            continue
        window = "\n".join(lines[low - 1:high])
        if anchor not in window:
            failures.append(
                f"{citation}: the anchor {anchor!r} is not in {path} lines "
                f"{low}-{high}; the citation has drifted. Cited text:\n"
                + "\n".join(f"      {n}| {lines[n - 1].rstrip()}"
                            for n in range(low, high + 1)))

    stale = sorted(set(RESOLVED) - set(found))
    for citation in stale:
        failures.append(
            f"{citation} has a RESOLVED row but no longer appears in the "
            "registry; delete the row with the claim")
    unused_external = sorted(
        name for name in EXTERNAL
        if not any(citation.rpartition(":")[0] == name for citation in found))
    unused_ambiguous = sorted(
        name for name in AMBIGUOUS
        if not any(citation.rpartition(":")[0] == name for citation in found))
    if unused_external:
        failures.append(
            "these EXTERNAL declarations are unused; a table that outlives its "
            f"citations stops describing anything: {unused_external}")
    if unused_ambiguous:
        failures.append(
            "these AMBIGUOUS declarations are unused: " f"{unused_ambiguous}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=pathlib.Path, default=REGISTRY_PATH)
    arguments = parser.parse_args(argv)
    registry = json.loads(arguments.registry.read_text(encoding="utf-8"))
    failures = check(registry)
    total = len(citations(registry))
    for message in failures:
        print("FAIL:", message)
    print(f"{total} citations checked, {len(failures)} failing "
          f"({len(RESOLVED)} anchored, {len(EXTERNAL)} declared external)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
