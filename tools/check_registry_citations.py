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

``NAMED_ROUTINES``
    ``path::routine`` -> (the definition line, the files citing it).  For
    citations that live in source text rather than in the registry and that
    name a routine instead of a line, because a line number rots and a name
    does not.

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
import functools
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
    "module_cumulus_driver.F": _WRF,
    # WRF's phys/physics_mmm/cu_ntiedtke.F90, the New Tiedtke body that
    # phys/module_cu_ntiedtke.F wraps; tools/ntiedtke_wrf461_oracle/README.md
    # pins its v4.6.1 digest.  The cited :566 is cumastrn's deep-to-shallow
    # demotion (ktype 1 -> 2 where the cloud is thinner than zdnoprc).
    "cu_ntiedtke.F90": _WRF,
    "dyn_em/module_bc_em.F": _WRF,
    "dyn_em/module_diffusion_em.F": _WRF,
    "dyn_em/module_em.F": _WRF,
    "dyn_em/module_first_rk_step_part2.F": _WRF,
    "dyn_em/module_initialize_real.F": _WRF,
    "phys/module_bl_myjpbl.F": _WRF,
    "phys/module_bl_myjurb.F": _WRF,
    "phys/module_bl_mynn.F": _WRF,
    "module_mp_p3.F": _WRF,
    "module_mp_thompson.F": _WRF,
    "phys/module_mp_milbrandt2mom.F": _WRF,
    "phys/module_mp_thompson.F": _WRF,
    "module_mp_wdm6.F": _WRF,
    "phys/module_mp_wdm6.F": _WRF,
    "phys/module_pbl_driver.F": _WRF,
    "phys/module_physics_init.F": _WRF,
    "phys/module_radiation_driver.F": _WRF,
    "phys/module_sf_myjsfc.F": _WRF,
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
    "module_check_a_mundo.F": _WRF,
    "module_mp_nssl_2mom.F": _WRF,
    "share/module_model_constants.F": _WRF,
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
    "kernels/ysu.cu:203": (
        "gpuwm/core/kernels/ysu.cu",
        "if (us == 0.0f && hf == 0.0f && qf == 0.0f)"),
    # YSU: the guard that skips the top-down block where WRF would read
    # thlix(i,k+2) one past an array declared kts:kte.  The anchor is the
    # guard condition the warning names.
    "kernels/ysu.cu:390": (
        "gpuwm/core/kernels/ysu.cu",
        "kpbl < nz"),
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
    "tests/test_mynn_pbl_runtime.py:61": (
        "tests/test_mynn_pbl_runtime.py",
        "sf_sfclay_physics=5, sf_surface_physics=2"),
    # The two LES closures' transcription headers.  The anchor is the
    # km_opt the turbulence option row claims that launcher implements,
    # so a citation that slides onto the other closure's launcher -- the
    # one drift these two are actually at risk of, since the bodies are
    # near-identical -- fails here instead of reading plausibly.
    #
    # RE-PINNED with a reading.  Both had drifted 75 lines and this
    # checker had been failing on them: :910 is now an int32 argument in
    # a kernel call and :1000 a scratch reshape, while the km_opt=2
    # header is at :985 (launch_wrf_tke_km, def at :983) and the km_opt=3
    # header at :1075 (launch_wrf_smag3d_km, def at :1073).  The anchors
    # are unchanged, which is what made the drift visible rather than
    # plausible.
    #
    # RE-PINNED a second time, again with a reading.  The dycore merge
    # that gave each domain its own acoustic substeps and etac moved both
    # 56 lines: :985 became ``def launch_wrf_calc_n2(`` and :1075 a
    # ``suffix = ...`` assignment, while the km_opt=2 header is now at
    # :1041 (launch_wrf_tke_km, def at :1039) and the km_opt=3 header at
    # :1131 (launch_wrf_smag3d_km, def at :1129).  The registry text
    # carrying them is the builder's carried-through input, so the fix is
    # made there and the builder reproduces it.
    "gpuwm/core/dycore.py:1041": (
        "gpuwm/core/dycore.py",
        "WRF v4.6.1 km_opt=2:"),
    "gpuwm/core/dycore.py:1131": (
        "gpuwm/core/dycore.py",
        "WRF v4.6.1 km_opt=3:"),
}

#: Citations written as ``path::routine`` rather than ``path:line``, and
#: every file that carries one.
#:
#: A line number rots the moment anything above it moves.  The vertical
#: limiter's receipt and the two docstrings that quote it cited
#: ``gpuwm/core/dycore.py:2178`` until :2178 had become a batching helper
#: for staggered flux sums, 115 lines above the limiter itself, which is
#: at :2293.  Nothing caught it because nothing was looking: the tables
#: above cover the physics registry, and these citations live in source
#: text.
#:
#: Spelling the citation as the routine's NAME removes the class rather
#: than re-pinning a number that will drift again, and these rows are what
#: keep the name from going stale in its turn.  Each is checked four
#: ways: the target file exists, it still DEFINES the routine, every
#: declared carrier still cites it, and no carrier cites the routine's
#: file by LINE as well.  So a rename fails here, so does a carrier that
#: quietly drops the citation, which is how a table stops describing
#: anything, and so does a line number kept beside the name -- which is
#: not hypothetical: docs/da-nowcast-demo.md carried
#: ``gpuwm/core/dycore.py:2293`` one paragraph above the routine's own
#: name, correct on the day and checked by nothing, because the carrier
#: check is satisfied by the name appearing anywhere in the page.
#:
#: ``citation -> (the definition line that must exist, the files citing it)``
NAMED_ROUTINES: dict[str, tuple[str, tuple[str, ...]]] = {
    "gpuwm/core/dycore.py::apply_w_damping": (
        "def apply_w_damping(",
        ("docs/da-nested-forecast.md",
         "docs/da-nowcast-demo.md",
         "tests/test_eta_ladder_time_step.py",
         "tools/build_stretched_eta_ladder.py",
         "tools/da_cycle_prepared.py"),
    ),
}


def line_citations(text: str, path: str) -> tuple[int, ...]:
    """The line numbers ``text`` cites ``path`` by, as ``path:123``.

    A ``path::routine`` citation is not checked for a line, so a line
    citation of the same file sitting beside it is checked by nothing at
    all and rots the moment anything above the routine moves.  Split out
    so the rule can be exercised on a string rather than on the tree.
    """

    return tuple(int(found.group(1))
                 for found in re.finditer(re.escape(path) + r":(\d+)", text))


def named_routine_failures() -> list[str]:
    """One message per named-routine citation that has stopped being true."""

    failures: list[str] = []
    for citation, (definition, carriers) in sorted(NAMED_ROUTINES.items()):
        path, _, routine = citation.partition("::")
        target = MODEL / path
        if not target.is_file():
            failures.append(f"{citation}: {path} does not exist")
            continue
        body = target.read_text(encoding="utf-8", errors="replace")
        if definition not in body:
            failures.append(
                f"{citation}: {path} no longer contains {definition!r}, so "
                f"the {len(carriers)} place(s) citing {routine} are citing "
                "something that is not there")
        if not carriers:
            failures.append(
                f"{citation} declares no carrier, so nothing checks that "
                "anything still cites it")
        for carrier in carriers:
            source = MODEL / carrier
            if not source.is_file():
                failures.append(
                    f"{citation}: declared carrier {carrier} does not exist")
                continue
            text = source.read_text(encoding="utf-8", errors="replace")
            if citation not in text:
                failures.append(
                    f"{citation} is not cited in {carrier} any more; drop "
                    "the carrier from NAMED_ROUTINES or restore the "
                    "citation")
            beside = line_citations(text, path)
            if beside:
                failures.append(
                    f"{carrier} cites {routine} by name and also cites "
                    f"{path} by line ("
                    + ", ".join(f":{line}" for line in beside)
                    + f"); the name check passes without the number, so the "
                    f"number rots unread. Cite {citation} alone")
    return failures


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
#: EMPTY, and that is the correct state.  The table held two YSU rows,
#: found by running this checker for the first time after wiring it to a
#: test, and both are fixed in the registry text (and in
#: ``tools/ysu_wrf461_oracle/patch_registry_maturity.py``, which carries the
#: same sentences):
#:
#: ``kernels/ysu.cu:252`` cited the ``kpbl < nz`` guard on the top-down
#: block; the file grew around it and :252 became
#: ``if (hpbl < zq[1]) kpbl = 1;``.  The guard is at :390 and is now an
#: anchored RESOLVED row.
#:
#: ``kernels/ysu.cu:1315`` was a bare continuation (``kernels/ysu.cu
#: implements :1315``) that the scanner attributed to the port's kernel,
#: which has 727 lines.  The number was never the kernel's: it is WRF's
#: ``bl_ysu.F90:1315``, the ``ad(i,1) = 1+fric`` arm, which is how
#: ``tools/ysu_wrf461_oracle/run_bl_ysu.F90`` names the contract the kernel
#: was written against, beside the ``bl_ysu.F90:1308`` ctopo arm the same
#: warning cites.  The text now spells the file out, so the citation is a
#: declared EXTERNAL one.  The row's recorded target (the kernel's own
#: ``diag[0] = 1.0f + fric;`` line) read the number as a port line, which
#: the rest of the sentence does not support.
#:
#: A new row needs the measured evidence and an owner, and
#: ``tests/test_physics_registry.py`` pins the set so it cannot change
#: unread.
DRIFTED: dict[str, tuple[str, str, str, str]] = {}


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


@functools.lru_cache(maxsize=1)
def _tree_index() -> dict[str, tuple[str, ...]]:
    """``basename -> every repo-relative path with that basename``, once.

    ``_resolve`` used to run ``MODEL.rglob(name)`` per citation -- a full
    walk of the tree for each of the several hundred citations in the
    registry.  The 2026-08-13 test-estate audit priced the consequence:
    ``test_physics_registry.py`` at 40 s, of which one test is 33 s, and the
    same walk repeated in ``test_build_registry.py`` (51 s) and
    ``test_physics_registry_declarations.py`` (9 s).

    One walk, cached for the process, answers every citation.  This changes
    no answer: ``_resolve``'s result for a given input is identical, and
    ``tests/test_physics_registry.py`` compares the same enumerated defect
    list it always did.
    """

    index: dict[str, list[str]] = {}
    for candidate in MODEL.rglob("*"):
        relative_parts = candidate.relative_to(MODEL).parts
        # Other checkouts live INSIDE the main checkout: nested
        # worktrees and parked clones under the .worktrees and .claude
        # directories.  A citation into THIS worktree must never resolve
        # into some other ref's tree (a rotted citation could come back
        # RESOLVED against a file this tree no longer carries), and those
        # clones contain Linux symlinks Windows cannot stat (WinError
        # 1920), which killed this walk in the one checkout the release
        # battery runs in.
        if relative_parts and relative_parts[0] in (
                ".worktrees", ".claude", ".git"):
            continue
        try:
            if not candidate.is_file():
                continue
        except OSError:
            continue
        relative = candidate.relative_to(MODEL).as_posix()
        if "__pycache__" in relative:
            continue
        index.setdefault(candidate.name, []).append(relative)
    return {name: tuple(sorted(paths)) for name, paths in index.items()}


def _resolve(cited_path: str) -> list[str]:
    """Repo-relative matches for a cited path, exact or by unique suffix."""

    if (MODEL / cited_path).is_file():
        return [cited_path]
    suffix = "/" + cited_path
    return sorted(
        relative for relative in _tree_index().get(
            pathlib.PurePosixPath(cited_path).name, ())
        if relative.endswith(suffix))


@functools.lru_cache(maxsize=None)
def _lines(path: str) -> tuple[str, ...] | None:
    """The cited file's lines.

    Cached for the same reason as ``_tree_index``: a hot file is cited by
    dozens of parameters and was re-read for each of them.  A tuple rather
    than a list so the cache cannot hand out a mutable shared object.
    """

    target = MODEL / path
    if not target.is_file():
        return None
    return tuple(
        target.read_text(encoding="utf-8", errors="replace").splitlines())


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
    failures: list[str] = named_routine_failures()

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
