"""Disassemble the committed cubins and record what the hardware runs.

The PTX mirrors in ``receipt/ptx`` read well, but they are recompiles: the
target changed so the instruction modifiers would be text.  This module
reads the SHIPPED objects instead -- the exact bytes NVRTC handed each
production route, already committed under ``receipt/cubin`` -- and turns
them into SASS, which is the machine's own answer rather than a rendering
of an equivalent compile.

SASS collection is best-effort by [D-20].  "Best-effort" is a statement
about what this module may FAIL to produce, never about what it may
assert: every attempt records the tool it ran, the version it ran, and
the exit status it got, and an attempt that fails is written into the
receipt as unavailable with the disassembler's own words for why.  A
missing or refusing ``nvdisasm`` costs the receipt one evidence tier; it
never costs it a fact.

Two disassembly attempts are made per object, in order:

``whole-file``   ``nvdisasm -c <cubin>``, which is the ordinary route.
``sections``     each ``.text.<kernel>`` section extracted by this
                 module's own ELF walk and disassembled with
                 ``nvdisasm -b SM<arch>``.  This is the fallback for an
                 object whose container the installed disassembler does
                 not recognize -- a newer NVRTC packaging its output in
                 sections an older ``nvdisasm`` refuses.  The instruction
                 bytes are identical either way; only the wrapper differs.

Usage::

    python -m tools.ftz_receipt.sass --receipt tools/ftz_receipt/receipt
    python -m tools.ftz_receipt.sass --check
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

#: SASS opcodes whose operands are single-precision floats and whose
#: encoding therefore carries an FTZ bit.  An explicit set rather than a
#: prefix rule: ``FLO`` and ``FSEL`` start with the same letter and mean
#: different things, and a prefix rule would quietly count integer ops on
#: a future architecture.
FLOAT_OPCODES = frozenset({
    "FADD", "FMUL", "FFMA", "FMNMX", "FSET", "FSETP", "FSEL", "FCHK",
    "FRND", "F2F", "F2FP", "F2I", "I2F", "MUFU", "FSWZADD", "FADD32I",
    "FMUL32I", "HADD2", "HMUL2", "HFMA2",
})

#: One instruction as ``nvdisasm`` prints it: address comment, optional
#: predicate, then the opcode with its dot-separated modifiers.
SASS_INSTRUCTION = re.compile(
    r"^\s*(?:/\*[0-9a-fA-F]+\*/)?\s*"
    r"(?:@!?P\d+\s+|@!?PT\s+)?"
    r"(?P<op>[A-Z][A-Z0-9_]*)"
    r"(?P<mods>(?:\.[A-Za-z0-9_]+)*)")

_ARCH_MASK = 0x00FF00


class DisassemblyUnavailable(Exception):
    """No SASS could be produced.  Carries the reason, never a guess."""


# ---- locating the disassembler -------------------------------------------

def nvdisasm_search_paths() -> list[str]:
    """Where this module looks, in order, without hardcoding a version."""
    candidates: list[str] = []
    found = shutil.which("nvdisasm")
    if found:
        candidates.append(found)
    for variable in ("CUDA_PATH", "CUDA_HOME"):
        home = os.environ.get(variable)
        if home:
            candidates.append(str(Path(home) / "bin" / "nvdisasm"))
    for pattern in (
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/bin/"
            "nvdisasm.exe",
            "/usr/local/cuda*/bin/nvdisasm"):
        candidates.extend(sorted(glob.glob(pattern), reverse=True))
    return candidates


def nvdisasm_path() -> str | None:
    for candidate in nvdisasm_search_paths():
        if Path(candidate).exists() or shutil.which(candidate):
            return candidate
    return None


def nvdisasm_version(exe: str) -> str:
    result = subprocess.run([exe, "--version"], capture_output=True,
                            text=True)
    for line in result.stdout.splitlines():
        if line.strip().startswith("Cuda compilation tools"):
            return line.strip()
    return (result.stdout.strip().splitlines() or ["unknown"])[0]


# ---- ELF walk -------------------------------------------------------------

def _sections(blob: bytes) -> list[tuple[str, int, int]]:
    """(name, offset, size) for every section of a 64-bit LE cubin."""
    if blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        raise DisassemblyUnavailable(
            "not a 64-bit little-endian ELF object")
    e_shoff = struct.unpack_from("<Q", blob, 40)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", blob, 58)

    def entry(index: int) -> tuple[int, int, int]:
        base = e_shoff + index * e_shentsize
        name, _type, _flags, _addr, offset, size = struct.unpack_from(
            "<IIQQQQ", blob, base)
        return name, offset, size

    _name, str_off, str_size = entry(e_shstrndx)
    strtab = blob[str_off:str_off + str_size]
    out = []
    for index in range(e_shnum):
        name, offset, size = entry(index)
        end = strtab.index(b"\x00", name)
        out.append((strtab[name:end].decode("utf-8", "replace"), offset, size))
    return out


def elf_sm_arch(blob: bytes) -> str:
    """``SM120``-style target read from the cubin's own e_flags."""
    if blob[:4] != b"\x7fELF":
        raise DisassemblyUnavailable("not an ELF object")
    e_flags = struct.unpack_from("<I", blob, 48)[0]
    return f"SM{(e_flags & _ARCH_MASK) >> 8:d}"


def text_sections(blob: bytes) -> list[tuple[str, bytes]]:
    """Every ``.text.<kernel>`` section, in the object's own order."""
    out = []
    for name, offset, size in _sections(blob):
        if name.startswith(".text.") and size:
            out.append((name[len(".text."):], blob[offset:offset + size]))
    return out


# ---- disassembly ----------------------------------------------------------

def _run(exe: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([exe, *args], capture_output=True, text=True)


def disassemble(exe: str, blob: bytes) -> dict:
    """Return the SASS for one object, plus how it was obtained.

    Never raises on a disassembler refusal: the refusal is the finding.
    """
    attempts: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="ftz-sass-") as tmp:
        target = Path(tmp) / "object.cubin"
        target.write_bytes(blob)
        whole = _run(exe, ["-c", str(target)])
        attempts.append({
            "method": "whole-file",
            "argv": "nvdisasm -c <object>",
            "returncode": whole.returncode,
            "stderr": whole.stderr.strip()[:400],
        })
        if whole.returncode == 0 and whole.stdout.strip():
            return {"available": True, "method": "whole-file",
                    "text": whole.stdout, "attempts": attempts,
                    "kernels": sorted(
                        name for name, _ in text_sections(blob))}

        arch = elf_sm_arch(blob)
        chunks: list[str] = []
        kernels: list[str] = []
        for name, body in text_sections(blob):
            section = Path(tmp) / f"{name}.bin"
            section.write_bytes(body)
            result = _run(exe, ["-b", arch, str(section)])
            if result.returncode != 0 or not result.stdout.strip():
                attempts.append({
                    "method": "sections",
                    "argv": f"nvdisasm -b {arch} <.text.{name}>",
                    "returncode": result.returncode,
                    "stderr": result.stderr.strip()[:400],
                })
                return {"available": False, "attempts": attempts,
                        "reason": "the installed disassembler refused both "
                                  "the container and its .text sections"}
            kernels.append(name)
            chunks.append(f"// .text.{name}\n{result.stdout.rstrip()}\n")
        attempts.append({
            "method": "sections",
            "argv": f"nvdisasm -b {arch} <.text.*>",
            "returncode": 0,
            "stderr": "",
        })
        if not chunks:
            return {"available": False, "attempts": attempts,
                    "reason": "the object carries no .text section"}
        return {"available": True, "method": "sections", "arch": arch,
                "text": "".join(chunks), "attempts": attempts,
                "kernels": kernels}


def sass_ftz_modifiers(text: str) -> dict:
    """Count float-domain SASS instructions, split by the ``FTZ`` modifier.

    Reads the text and nothing else, so a parser that always answered the
    same way fails the fixture pair in ``tests/test_ftz_render.py``.
    """
    with_ftz: dict[str, int] = {}
    without_ftz: dict[str, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("."):
            continue
        match = SASS_INSTRUCTION.match(line)
        if match is None:
            continue
        opcode = match.group("op")
        if opcode not in FLOAT_OPCODES:
            continue
        modifiers = match.group("mods").split(".")
        bucket = with_ftz if "FTZ" in modifiers else without_ftz
        bucket[opcode] = bucket.get(opcode, 0) + 1
    return {
        "with_ftz": dict(sorted(with_ftz.items())),
        "without_ftz": dict(sorted(without_ftz.items())),
        "float_instruction_count": (sum(with_ftz.values())
                                    + sum(without_ftz.values())),
    }


# ---- receipt integration --------------------------------------------------

def cubin_artifacts(artifacts: dict) -> list[str]:
    """The committed cubin keys, in receipt order."""
    return [name for name, record in artifacts.items()
            if name.startswith("cubin/") and record.get("format") == "cubin"]


def collect(out_dir: Path, artifacts: dict) -> dict:
    """Disassemble every committed cubin; return the ``sass`` block.

    Writes ``receipt/sass/<stem>.sass`` for each object that disassembles
    and records the per-object result -- including the failures -- under
    ``artifacts``.  The cubin's committed digest is verified before it is
    disassembled, so this cannot describe bytes the receipt did not pin.
    """
    exe = nvdisasm_path()
    if exe is None:
        return {
            "provenance": "not-collected",
            "reason": "no nvdisasm on PATH, in CUDA_PATH/CUDA_HOME, or in "
                      "an installed CUDA toolkit; SASS is best-effort by "
                      "[D-20] and its absence is recorded rather than "
                      "worked around",
            "searched": nvdisasm_search_paths(),
        }
    version = nvdisasm_version(exe)
    sass_dir = out_dir / "sass"
    per_object: dict[str, dict] = {}
    produced = 0
    for name in cubin_artifacts(artifacts):
        path = out_dir / name
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != artifacts[name]["sha256"]:
            raise DisassemblyUnavailable(
                f"{name} on disk does not match the digest the receipt "
                f"commits; disassembling it would describe other bytes")
        stem = Path(name).stem
        result = disassemble(exe, blob)
        if not result["available"]:
            per_object[stem] = {
                "cubin": name,
                "available": False,
                "reason": result["reason"],
                "attempts": result["attempts"],
            }
            continue
        sass_dir.mkdir(parents=True, exist_ok=True)
        text = result["text"].replace("\r\n", "\n")
        (sass_dir / f"{stem}.sass").write_text(text, encoding="utf-8",
                                               newline="\n")
        artifacts[f"sass/{stem}.sass"] = {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "provenance": "shipped-path",
            "derived_from": name,
            "derived_from_sha256": digest,
            "derivation": f"nvdisasm ({result['method']}) over the committed "
                          f"object; no recompile",
            "description": "the instructions the device executes, "
                           "disassembled from the exact object the route's "
                           "own compile returned",
            "ftz_modifiers": sass_ftz_modifiers(text),
        }
        per_object[stem] = {
            "cubin": name,
            "available": True,
            "method": result["method"],
            "kernels": result["kernels"],
            "artifact": f"sass/{stem}.sass",
        }
        produced += 1
    return {
        "provenance": "shipped-path" if produced else "unavailable",
        "policy": "best-effort per [D-20]",
        "disassembler": version,
        "objects": per_object,
        "produced": produced,
        "reason": (
            "this record is the tier summary and has no digest of its own; "
            "the disassembly files are the artifacts, one per object, listed "
            "under `objects` and hashed individually"
            if produced else
            "nvdisasm is installed but disassembled none of the committed "
            "objects; see each object's recorded attempts"),
    }


def update_receipt(receipt_dir: Path) -> dict:
    """Add the SASS tier to an already-measured receipt, in place.

    Only the disassembly-derived keys move.  Nothing measured on the
    device is touched, recomputed, or re-run: the bit table, the verdicts
    and the identity block are the probe's, and a static disassembly of
    already-committed bytes has no standing to revise them.
    """
    path = receipt_dir / "receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    artifacts = {name: record for name, record in receipt["artifacts"].items()
                 if not name.startswith("sass")}
    block = collect(receipt_dir, artifacts)
    ordered: dict[str, dict] = {}
    for name, record in artifacts.items():
        ordered[name] = record
    ordered["sass"] = block
    receipt["artifacts"] = ordered
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", default=None)
    parser.add_argument("--check", action="store_true",
                        help="re-derive and require the committed receipt "
                             "and .sass files to be unchanged")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    receipt_dir = (Path(args.receipt) if args.receipt
                   else root / "tools" / "ftz_receipt" / "receipt")
    if args.check:
        before = (receipt_dir / "receipt.json").read_text(encoding="utf-8")
        existing = {p: p.read_bytes()
                    for p in sorted((receipt_dir / "sass").glob("*.sass"))} \
            if (receipt_dir / "sass").exists() else {}
        update_receipt(receipt_dir)
        after = (receipt_dir / "receipt.json").read_text(encoding="utf-8")
        problems = []
        if before != after:
            problems.append("receipt.json changed under re-derivation")
        for path, blob in existing.items():
            if path.read_bytes() != blob:
                problems.append(f"{path.name} changed under re-derivation")
        if problems:
            (receipt_dir / "receipt.json").write_text(
                before, encoding="utf-8", newline="\n")
            for problem in problems:
                print(problem)
            return 1
        print("sass tier reproduces")
        return 0
    receipt = update_receipt(receipt_dir)
    block = receipt["artifacts"]["sass"]
    print(f"sass: {block['provenance']} "
          f"({block.get('produced', 0)} object(s))")
    for stem, record in block.get("objects", {}).items():
        state = record.get("method") if record["available"] else "unavailable"
        print(f"  {stem:<12} {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
