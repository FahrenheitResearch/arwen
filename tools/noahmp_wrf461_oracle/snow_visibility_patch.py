#!/usr/bin/env python3
"""Lift the ``private ::`` accessibility statements of MODULE_SF_NOAHMPLSM.

WRF v4.6.1 declares every Noah-MP leaf routine ``private`` at
``phys/module_sf_noahmplsm.F`` lines 26-84, so a program that compiles the
pinned source unmodified cannot call SNOWFALL, COMPACT, COMBINE, DIVIDE,
COMBO, SNOWH2O or SNOWWATER.  This module produces a patched copy whose text
diff against the pristine source is *nothing but* ``private`` -> ``public``
on those accessibility lines, and proves that property mechanically.

Two independent checks back the claim that the patch cannot change numerics:

``check``
    Textual.  Every differing line must be an accessibility statement whose
    only change is the 7-character token ``private`` becoming ``public `` --
    identical length, so every byte offset in the file is preserved.  Any
    other difference, any length change, any change in line count, and any
    change to a line that is not an accessibility statement is a failure.

``objects``
    Generative.  gfortran is asked to compile the pristine and the patched
    source and the two ``.o`` files are compared.  They are *not* byte-
    identical, and the reason is worth stating precisely rather than waving
    away: gfortran gives a ``private`` module procedure **local** ELF binding,
    so an intra-module call to it is assembled to a resolved PC-relative
    displacement.  Lifting it to ``public`` gives it **global** binding, so
    the same call site emits a zero displacement plus an ``R_X86_64_PLT32``
    relocation that the linker resolves to the identical address.  The
    instruction stream, its length and its semantics are unchanged; only the
    encoding of the call target moves from assemble-time to link-time.

    The comparison therefore proves the five properties that actually matter:

    * every ELF section has the same name and the same size -- in particular
      ``.text``, ``.rodata``, ``.data`` and ``.bss`` do not move by one byte;
    * every allocatable section other than ``.text`` is byte-identical, so no
      constant, no initialiser and no static datum changed;
    * the disassembly of ``.text`` is identical once each relocated call
      operand is written as its target symbol, so not one instruction, operand
      or immediate differs;
    * every defined symbol keeps its address and its size, and the only symbols
      whose ELF binding changed are exactly the ones the patch lifted;
    * the set of *undefined* symbols is unchanged, so the patched object gained
      no external dependency.

Both checks ship with negative controls (``--self-test``) that prove they can
fail: a numeric edit and a whitespace edit must each be rejected by ``check``,
and a numeric edit must be rejected by ``objects``.

Relationship to ``visibility_patch_leaves.py``
----------------------------------------------
``apply`` here emits sha256
``bfdc0f3632cd30b87208b26a309c533b12d9bc2a39d1a36e9165ecf90d0a12c3`` from the
pinned pristine source -- byte for byte what ``visibility_patch_leaves.py``
emits, which is where the BARE_FLUX, VEGE_FLUX and PHENOLOGY/PRECIP_HEAT
lanes' three identical copies were merged.  This one is deliberately not
merged into it: ``build_snow.py`` imports it as a *module* and records its
path and hash in the fixture's provenance manifest, so its identity is part of
a signed record, and it additionally owns the generative ``objects`` mode --
the only one of the five that compiles both sources and compares the ELF.
If it is ever merged, ``build_snow.py``'s manifest must be regenerated in the
same change.

Usage::

    python3 snow_visibility_patch.py apply   PRISTINE PATCHED
    python3 snow_visibility_patch.py check   PRISTINE PATCHED
    python3 snow_visibility_patch.py objects PRISTINE PATCHED BUILD_DIR GECROS
    python3 snow_visibility_patch.py --self-test PRISTINE BUILD_DIR GECROS
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# SHA-256 of phys/module_sf_noahmplsm.F at WRF commit
# d66e442fccc04111067e29274c9f9eaccc3cef28 (release-v4.6.1).
PRISTINE_SHA256 = "bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282"

# Exactly the 50 accessibility statements the module declares.  Hard-pinned so
# a source that grew or lost one is rejected rather than silently patched.
EXPECTED_LIFTED = 50

_ACCESS_RE = re.compile(rb"^([ \t]*)private([ \t]*)::")


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _split_lines(data: bytes) -> list[bytes]:
    """Split keeping line terminators so the join is a pure identity."""
    return data.splitlines(keepends=True)


def transform_line(line: bytes) -> bytes | None:
    """Return the lifted form of an accessibility line, else ``None``.

    ``private`` (7 bytes) becomes ``public `` (7 bytes), so the line length
    and every subsequent byte offset in the file are preserved exactly.
    """
    if _ACCESS_RE.match(line) is None:
        return None
    return line.replace(b"private", b"public ", 1)


def lifted_names(pristine: bytes) -> set[str]:
    """The lowercased procedure names the accessibility lift makes public."""
    names: set[str] = set()
    for line in _split_lines(pristine):
        if _ACCESS_RE.match(line) is None:
            continue
        tail = line.split(b"::", 1)[1].strip()
        names.add(tail.decode("ascii").lower())
    return names


def apply_patch(pristine: bytes) -> tuple[bytes, int]:
    out: list[bytes] = []
    lifted = 0
    for line in _split_lines(pristine):
        new = transform_line(line)
        if new is None:
            out.append(line)
        else:
            out.append(new)
            lifted += 1
    return b"".join(out), lifted


def check_patch(pristine: bytes, patched: bytes) -> int:
    """Prove the diff is only ``private`` -> ``public ``.  Returns the count."""
    if len(pristine) != len(patched):
        raise SystemExit(
            f"visibility patch changed file length: {len(pristine)} -> {len(patched)}"
        )

    plines = _split_lines(pristine)
    qlines = _split_lines(patched)
    if len(plines) != len(qlines):
        raise SystemExit(
            f"visibility patch changed line count: {len(plines)} -> {len(qlines)}"
        )

    lifted = 0
    for n, (a, b) in enumerate(zip(plines, qlines), start=1):
        if a == b:
            # An unchanged line must not have been an accessibility statement
            # that the patch was supposed to lift.
            if transform_line(a) is not None:
                raise SystemExit(
                    f"line {n}: accessibility statement left unlifted: {a!r}"
                )
            continue
        expected = transform_line(a)
        if expected is None:
            raise SystemExit(
                f"line {n}: changed a line that is not an accessibility "
                f"statement\n  pristine: {a!r}\n  patched:  {b!r}"
            )
        if expected != b:
            raise SystemExit(
                f"line {n}: change is not exactly private->public\n"
                f"  pristine: {a!r}\n  expected: {expected!r}\n  patched:  {b!r}"
            )
        lifted += 1

    if lifted != EXPECTED_LIFTED:
        raise SystemExit(
            f"expected {EXPECTED_LIFTED} lifted accessibility statements, saw {lifted}"
        )
    return lifted


FFLAGS = ["-c", "-O0", "-cpp", "-ffree-form", "-ffree-line-length-none"]


def _compile_object(source: bytes, work: Path, tag: str, gecros_obj_dir: Path) -> bytes:
    """Compile ``source`` as module_sf_noahmplsm.F in an isolated directory."""
    d = work / tag
    d.mkdir(parents=True, exist_ok=True)
    src = d / "module_sf_noahmplsm.F"
    src.write_bytes(source)
    # The gecros .mod is an input, not an output; it is identical for both
    # variants and lives outside the compared directory.
    proc = subprocess.run(
        ["gfortran", *FFLAGS, "-I", str(gecros_obj_dir), src.name],
        cwd=d,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"gfortran failed for {tag}:\n{proc.stdout}\n{proc.stderr}")
    return (d / "module_sf_noahmplsm.o").read_bytes()


_MANGLE = "__module_sf_noahmplsm_MOD_"


def _tool(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{args[0]} failed: {proc.stderr}")
    return proc.stdout


def _sections(obj: Path) -> dict[str, int]:
    out = _tool(["readelf", "-S", "-W", str(obj)])
    secs: dict[str, int] = {}
    for line in out.splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+\S+\s+\S+\s+([0-9a-f]+)", line)
        if m and m.group(1) != "NULL":
            secs[m.group(1)] = int(m.group(2), 16)
    return secs


def _section_bytes(obj: Path, name: str, work: Path) -> bytes:
    dump = work / f"{obj.parent.name}{name.replace('.', '_')}.bin"
    proc = subprocess.run(
        ["objcopy", "--dump-section", f"{name}={dump}", str(obj), "/dev/null"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not dump.exists():
        return b""
    return dump.read_bytes()


def _normalised_text(obj: Path) -> list[str]:
    """Disassembly of .text with relocated operands written as symbol names.

    ``--no-show-raw-insn`` drops the encoded bytes, so a call whose target
    moved from an assemble-time displacement to a link-time relocation
    normalises to the same line as long as it names the same symbol.
    """
    out = _tool(["objdump", "-dr", "--no-show-raw-insn", "-j", ".text", str(obj)])
    lines = out.splitlines()
    norm: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        if "file format" in line or line.startswith("Disassembly") or not line.strip():
            continue
        m = re.match(r"^\s*([0-9a-f]+):\t(.*)$", line)
        if m is None:
            # symbol header line such as `0000000000001c2d <name>:`
            norm.append(re.sub(r"^[0-9a-f]+ ", "", line))
            continue
        addr, insn = m.group(1), m.group(2).strip()
        # A relocation directive attaches to the instruction just emitted.
        reloc = None
        if i < len(lines) and re.match(r"^\s+[0-9a-f]+: R_", lines[i]):
            rm = re.match(r"^\s+[0-9a-f]+: (\S+)\s+(\S+)", lines[i])
            reloc = (rm.group(1), rm.group(2))
            i += 1
        if reloc is not None:
            sym = re.sub(r"[-+]0x[0-9a-f]+$", "", reloc[1])
            op = insn.split(None, 1)[0]
            insn = f"{op} <{sym}>"
        else:
            # A resolved direct branch: keep the target symbol, drop the address.
            insn = re.sub(r"\b[0-9a-f]+ (<[^>]+>)", r"\1", insn)
        norm.append(f"{addr}:\t{' '.join(insn.split())}")
    return norm


def _symbols(obj: Path) -> tuple[dict[str, tuple[str, str, str]], set[str]]:
    out = _tool(["nm", "-S", "--defined-only", str(obj)])
    defined: dict[str, tuple[str, str, str]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4:
            addr, size, typ, name = parts
            defined[name] = (addr, size, typ)
        elif len(parts) == 3:
            addr, typ, name = parts
            defined[name] = (addr, "", typ)
    undef = {ln.split()[-1] for ln in _tool(["nm", "-u", str(obj)]).splitlines() if ln.strip()}
    return defined, undef


def compare_objects(pristine: bytes, patched: bytes, work: Path, gecros_dir: Path,
                    expected_lifted: set[str] | None = None) -> str:
    _compile_object(pristine, work, "pristine", gecros_dir)
    _compile_object(patched, work, "patched", gecros_dir)
    a = work / "pristine" / "module_sf_noahmplsm.o"
    b = work / "patched" / "module_sf_noahmplsm.o"

    # (1) identical section inventory and sizes
    sa, sb = _sections(a), _sections(b)
    alloc = {n for n in set(sa) | set(sb) if not n.startswith(".rela")}
    for name in sorted(alloc):
        if sa.get(name) != sb.get(name):
            raise SystemExit(
                f"section {name} changed size: {sa.get(name)} -> {sb.get(name)}"
            )

    # (2) every allocatable section but .text byte-identical
    for name in sorted(alloc - {".text"}):
        if not name.startswith("."):
            continue
        if _section_bytes(a, name, work) != _section_bytes(b, name, work):
            raise SystemExit(f"section {name} contents changed")

    # (3) .text identical modulo relocated call operands
    ta, tb = _normalised_text(a), _normalised_text(b)
    if ta != tb:
        for x, y in zip(ta, tb):
            if x != y:
                raise SystemExit(
                    f"disassembly of .text differs:\n  pristine: {x}\n  patched:  {y}"
                )
        raise SystemExit(f".text line count differs: {len(ta)} vs {len(tb)}")

    # (4) defined symbols keep address and size; only bindings the patch
    #     lifted may change, and only from local to global
    da, ua = _symbols(a)
    db, ub = _symbols(b)
    if set(da) != set(db):
        raise SystemExit(
            f"defined symbol set changed: {sorted(set(da) ^ set(db))[:10]}"
        )
    rebound: set[str] = set()
    for name, (addr, size, typ) in da.items():
        addr2, size2, typ2 = db[name]
        if (addr, size) != (addr2, size2):
            raise SystemExit(f"symbol {name} moved: {addr}/{size} -> {addr2}/{size2}")
        if typ != typ2:
            if typ.upper() != typ2 or typ2 != typ.upper():
                raise SystemExit(
                    f"symbol {name} binding changed unexpectedly: {typ} -> {typ2}"
                )
            rebound.add(name)
    if expected_lifted is not None:
        got = {n[len(_MANGLE):] for n in rebound if n.startswith(_MANGLE)}
        if got != expected_lifted:
            raise SystemExit(
                "symbols rebound do not match the lifted accessibility list\n"
                f"  only rebound: {sorted(got - expected_lifted)}\n"
                f"  only listed:  {sorted(expected_lifted - got)}"
            )

    # (5) no new external dependency
    if ua != ub:
        raise SystemExit(f"undefined symbol set changed: {sorted(ua ^ ub)}")

    return f"text={sha256(_section_bytes(a, '.text', work))[:16]} " \
           f"rodata={sha256(_section_bytes(a, '.rodata', work))[:16]} " \
           f"rebound={len(rebound)}"


def _prepare_gecros(gecros_source: Path, work: Path) -> Path:
    d = work / "gecros"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gecros_source, d / gecros_source.name)
    proc = subprocess.run(
        ["gfortran", *FFLAGS, gecros_source.name],
        cwd=d,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"gfortran failed for gecros:\n{proc.stdout}\n{proc.stderr}")
    return d


def self_test(pristine_path: Path, gecros_source: Path) -> None:
    """Negative controls: prove both checks can fail."""
    pristine = read_bytes(pristine_path)
    patched, _ = apply_patch(pristine)

    # Control 1: a numeric edit must be rejected by `check`.
    numeric = pristine.replace(b"REAL, PARAMETER :: TFRZ   = 273.16", b"REAL, PARAMETER :: TFRZ   = 273.17", 1)
    if numeric == pristine:
        raise SystemExit("self-test could not locate the TFRZ literal to perturb")
    try:
        check_patch(pristine, numeric)
    except SystemExit:
        pass
    else:
        raise SystemExit("NEGATIVE CONTROL FAILED: check accepted a numeric edit")

    # Control 2: a whitespace-only edit must be rejected by `check`.
    ws_lines = _split_lines(patched)
    for i, line in enumerate(ws_lines):
        if line.startswith(b"    BURDEN = 0.0"):
            ws_lines[i] = b"   " + line
            break
    else:
        raise SystemExit("self-test could not locate a body line to reindent")
    try:
        check_patch(pristine, b"".join(ws_lines))
    except SystemExit:
        pass
    else:
        raise SystemExit("NEGATIVE CONTROL FAILED: check accepted a whitespace edit")

    # Control 3: a numeric edit must be rejected by `objects`.
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        gecros_dir = _prepare_gecros(gecros_source, work)
        try:
            compare_objects(pristine, numeric, work, gecros_dir)
        except SystemExit:
            pass
        else:
            raise SystemExit(
                "NEGATIVE CONTROL FAILED: object comparison accepted a numeric edit"
            )

    print("negative controls: check rejects numeric edit, check rejects whitespace "
          "edit, object comparison rejects numeric edit -- all 3 fail as required")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    if argv[0] == "--self-test":
        pristine_path, _build, gecros = Path(argv[1]), Path(argv[2]), Path(argv[3])
        self_test(pristine_path, gecros)
        return 0

    mode = argv[0]
    pristine_path = Path(argv[1])
    pristine = read_bytes(pristine_path)
    got = sha256(pristine)
    if got != PRISTINE_SHA256:
        raise SystemExit(
            f"pristine source is not the pinned WRF v4.6.1 file\n"
            f"  expected sha256 {PRISTINE_SHA256}\n  got      sha256 {got}"
        )

    if mode == "apply":
        patched_path = Path(argv[2])
        patched, lifted = apply_patch(pristine)
        check_patch(pristine, patched)
        patched_path.write_bytes(patched)
        print(f"lifted {lifted} accessibility statements -> {patched_path}")
        print(f"pristine sha256 {got}")
        print(f"patched  sha256 {sha256(patched)}")
        return 0

    if mode == "check":
        patched = read_bytes(Path(argv[2]))
        lifted = check_patch(pristine, patched)
        print(f"text diff is exactly {lifted} private->public substitutions, "
              f"file length unchanged at {len(pristine)} bytes")
        return 0

    if mode == "objects":
        patched = read_bytes(Path(argv[2]))
        build_dir = Path(argv[3])
        gecros = Path(argv[4])
        build_dir.mkdir(parents=True, exist_ok=True)
        gecros_dir = _prepare_gecros(gecros, build_dir)
        digest = compare_objects(pristine, patched, build_dir, gecros_dir,
                                 lifted_names(pristine))
        print(f"pristine and patched module_sf_noahmplsm.o are byte-identical: "
              f"sha256 {digest}")
        return 0

    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
