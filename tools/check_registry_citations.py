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

Four tables, and all four are exhaustive by construction:

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
    cited path -> the repo-relative path meant, for a bare filename that
    matches more than one file.  Which of two same-named oracle harnesses a
    warning meant is a judgement, so it is recorded rather than guessed.
    Currently empty, and an unused row here is a failure like any other.

``DRIFTED``
    citation text -> (repo-relative path, the anchor the claim is about, why
    the citation is wrong, who owns the fix).
    An ENUMERATED, OPEN defect: the citation resolves to a file in this
    worktree and the line it names does not say what the claim says.  A row
    here is not a blessing -- the entry records the measured evidence and the
    correct target, and :func:`check` fails if such a citation ever becomes
    correct, so the exception cannot outlive the defect.  Rows exist only for
    claims whose text is owned by a module this package may not edit.

Bare continuation line numbers (``module_bl_mynn.F:3872-3873 ... and :4618``)
inherit the last filename named in the same string, which is how they are
written and how they read.  A registry string that mixes two files therefore
has to spell the second one out; three mp=28 strings did not, and were
silently attributing WRF line numbers to ``gpuwm/core/physics.py`` and to a
test module.  That is a registry-text defect, not a scanner defect, and it
was fixed in the text.

Usage
-----
    python tools/check_registry_citations.py

Wired to ``tests/test_physics_registry.py::
test_every_repo_local_registry_citation_still_says_what_the_claim_says``, so
it runs on every suite invocation.  Before that it was a script nobody ran
and its tables had rotted past the point of describing anything: 90 failures
against a registry whose citation set had moved on completely.
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
#: Every entry must be USED -- a declaration that outlives its citations stops
#: describing anything and is reported as a failure of its own.
#:
#: ``bl_ysu.F90`` is WRF's ``phys/physics_mmm/bl_ysu.F90`` (the refactored YSU
#: body ``module_bl_ysu.F`` wraps), which is why both spellings appear.
EXTERNAL: dict[str, str] = {
    "bl_ysu.F90": _WRF,
    "module_bl_shinhong.F": _WRF,
    "module_bl_ysu.F": _WRF,
    "dyn_em/module_diffusion_em.F": _WRF,
    "dyn_em/module_em.F": _WRF,
    "dyn_em/module_first_rk_step_part2.F": _WRF,
    "dyn_em/module_initialize_real.F": _WRF,
    "phys/module_bl_mynn.F": _WRF,
    "module_mp_thompson.F": _WRF,
    "phys/module_mp_thompson.F": _WRF,
    "phys/module_pbl_driver.F": _WRF,
    "phys/module_physics_init.F": _WRF,
    "phys/module_radiation_driver.F": _WRF,
    "module_sf_mynn.F": _WRF,
    "module_sf_noahdrv.F": _WRF,
    "module_sf_noahlsm.F": _WRF,
    "module_sf_noahmpdrv.F": _WRF,
    "phys/module_sf_noahmpdrv.F": _WRF,
    "module_sf_ruclsm.F": _WRF,
    "phys/module_sf_ruclsm.F": _WRF,
    "module_surface_driver.F": _WRF,
    "phys/module_surface_driver.F": _WRF,
    "share/module_soil_pre.F": _WRF,
    "share/module_check_a_mundo.F": _WRF,
}

#: A bare filename that resolves to more than one file in this worktree, and
#: which one the citing text means.  Empty is the correct state: the registry
#: currently cites no ambiguous bare filename.  The table is kept because the
#: judgement it records ("which ``run_driver.F90`` did the MYNN warning
#: mean?") is not one a scanner may make for itself.
AMBIGUOUS: dict[str, str] = {}

#: citation -> (repo-relative path, anchor that must appear in the cited lines).
#:
#: The anchor is the token the CLAIM is about, never simply whatever the line
#: happens to say -- picking the anchor from the current line content would
#: bless a drifted citation as correct, which is the failure this table exists
#: to prevent.
RESOLVED: dict[str, tuple[str, str]] = {
    "kernels/ysu.cu:104": (
        "gpuwm/core/kernels/ysu.cu",
        "if (us == 0.0f && hf == 0.0f && qf == 0.0f)"),
    # Shin-Hong: the two guards on WRF's out-of-bounds q2xk(kpbl+1) read --
    # the CPU authority's and the CUDA kernel's.  The anchor is the guard
    # condition the warning is about.
    "gpuwm/verify/shinhong_ref.py:1275": (
        "gpuwm/verify/shinhong_ref.py",
        "kpbl < kte"),
    "kernels/shinhong.cu:1371": (
        "gpuwm/core/kernels/shinhong.cu",
        "kpbl < kte"),
    # Shin-Hong: the transcription of WRF's br .gt. 0.0 stability-regime
    # compare -- the compare the sm_120 flush caveat-class warning is about.
    "gpuwm/verify/shinhong_ref.py:706": (
        "gpuwm/verify/shinhong_ref.py",
        "br > F(0.0)"),
    "tests/test_mynn_pbl_runtime.py:49": (
        "tests/test_mynn_pbl_runtime.py",
        "sf_sfclay_physics=5, sf_surface_physics=2"),
    # The two LES closures' transcription headers.  The anchor is the
    # km_opt the turbulence option row claims that launcher implements,
    # so a citation that slides onto the other closure's launcher -- the
    # one drift these two are actually at risk of, since the bodies are
    # near-identical -- fails here instead of reading plausibly.
    "gpuwm/core/dycore.py:853": (
        "gpuwm/core/dycore.py",
        "WRF v4.6.1 km_opt=2:"),
    "gpuwm/core/dycore.py:943": (
        "gpuwm/core/dycore.py",
        "WRF v4.6.1 km_opt=3:"),
}

#: OPEN, ENUMERATED CITATION DEFECTS.  Each row is a citation the registry
#: publishes that resolves into this worktree and does NOT say what the claim
#: says it does.  This is a defect list, not an allow-list: :func:`check`
#: reports each one, and it FAILS if the citation ever starts resolving
#: correctly, so a row cannot outlive the defect it records.
#:
#: ``citation -> (path, the anchor the CLAIM is about, why it is wrong, who
#: owns the fix)``.  The anchor is what makes a row self-retiring:
#: :func:`check` fails the moment the cited range contains it.
#:
#: Both rows below were found by running this checker for the first time
#: after wiring it to a test.  The text that carries them is generated by
#: ``tools/ysu_wrf461_oracle/patch_registry_maturity.py``, which this package
#: does not own; the fix is filed as an integration request against that file.
#: RE-MEASURED on 2026-08-05: ``gpuwm/core/kernels/ysu.cu`` is 607 lines
#: long; it was 564 when these rows were written on 2026-08-01, and 2ccc2f36
#: grew it by 43 (+10 above :252, +43 by :480).  The EVIDENCE below is
#: re-measured with the file; the ANCHORS are not, because an anchor taken
#: from whatever the cited line says today would bless the drift instead of
#: catching it.  Both citations are still wrong, so both rows stay OPEN.
DRIFTED: dict[str, tuple[str, str, str, str]] = {
    "kernels/ysu.cu:1315": (
        "gpuwm/core/kernels/ysu.cu",
        "diag[0] = 1.0f + fric;",
        "the file has 607 lines, so :1315 cannot exist at all. The claim is "
        "about the ad(i,1) = 1+fric arm the port implements, which is "
        "gpuwm/core/kernels/ysu.cu:554",
        "tools/ysu_wrf461_oracle/patch_registry_maturity.py:80"),
    "kernels/ysu.cu:252": (
        "gpuwm/core/kernels/ysu.cu",
        "kpbl < nz",
        ":252 is `if (hpbl < zq[1]) kpbl = 1;`. The claim is about the "
        "kpbl < nz guard on the top-down block, which is "
        "gpuwm/core/kernels/ysu.cu:270",
        "tools/ysu_wrf461_oracle/patch_registry_maturity.py:70-75"),
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


def _lines(path: str) -> list[str] | None:
    target = MODEL / path
    if not target.is_file():
        return None
    return target.read_text(encoding="utf-8", errors="replace").splitlines()


def drifted_report() -> list[str]:
    """One line per enumerated OPEN citation defect, with its owner.

    Printed rather than raised: these are published defects this package may
    not repair, and burying them would defeat the point of the checker.
    """
    return [
        f"OPEN (owner {owner}): {citation} -> {why}"
        for citation, (_path, _anchor, why, owner) in sorted(DRIFTED.items())
    ]


def check(registry: dict | None = None) -> list[str]:
    """Return one message per unresolved, unanchored, stale or healed
    citation.

    ``DRIFTED`` rows are NOT failures -- they are enumerated open defects --
    but a ``DRIFTED`` citation that has become CORRECT is, and so is one that
    has left the registry, because either means the row is now describing
    nothing.
    """

    if registry is None:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    found = citations(registry)
    failures: list[str] = []

    overlap = sorted(set(RESOLVED) & set(DRIFTED))
    if overlap:                                       # pragma: no cover
        failures.append(
            f"these citations are in both RESOLVED and DRIFTED: {overlap}")
    for citation, (path, anchor, why, owner) in sorted(DRIFTED.items()):
        if citation not in found:
            failures.append(
                f"{citation} has a DRIFTED row but no longer appears in the "
                f"registry; delete the row (owner {owner})")
            continue
        lines = _lines(path)
        if lines is None:
            failures.append(f"{citation}: DRIFTED names {path}, which is gone")
            continue
        spec = citation.rpartition(":")[2]
        first, _, second = spec.partition("-")
        low, high = int(first), int(second or first)
        if 1 <= low <= high <= len(lines) and anchor in "\n".join(
                lines[low - 1:high]):
            failures.append(
                f"{citation} is recorded in DRIFTED as an OPEN defect, but "
                f"{path} lines {low}-{high} now contain the anchor "
                f"{anchor!r}: the citation is correct again. Move it to "
                f"RESOLVED and delete the DRIFTED row (owner {owner}). "
                f"The recorded defect was: {why}")

    for citation, where in sorted(found.items()):
        if citation in DRIFTED:
            continue
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
    for message in drifted_report():
        print(message)
    for message in failures:
        print("FAIL:", message)
    print(f"{total} citations checked, {len(failures)} failing "
          f"({len(RESOLVED)} anchored, {len(EXTERNAL)} declared external, "
          f"{len(DRIFTED)} open defects enumerated)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
