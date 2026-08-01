#!/usr/bin/env python3
"""Apply only Morrison's measured-parity disclosure to registry JSON bytes.

The registry is compacted onto one physical line and is concurrently edited.
This helper replaces the ``morrison-mp10`` option object byte span and preserves
every byte outside it.  ``--git-revision`` is the safe path for constructing a
HEAD-plus-Morrison blob without staging another agent's working-tree changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path("gpuwm/physics_registry_v2.json")
OPTION_MARKER = b'"morrison-mp10":'
MORRISON_WARNINGS = [
    (
        "MORRISON IS MEASURED AGAINST WRF AND IS NOT BITWISE. "
        "tools/morrison_wrf461_oracle/build.sh calls the public "
        "MP_MORR_TWO_MOMENT wrapper and radar diagnostic from byte-unmodified "
        "WRF v4.6.1 at d66e442fccc04111067e29274c9f9eaccc3cef28, compiled "
        "at -O0 with kind_phys/default REAL verified as FP32. Across 28 "
        "columns (14 atmospheric states in both graupel and hail modes), the "
        "worst production result is 1,709,094,255 FP32 total-order ULP on the "
        "local RTX 5090 and 1,706,351,510 on the rented Linux RTX 5090; only "
        "GRAUPELNC is bit-identical on every fixture column. Theta is at most "
        "154 ULP away, but hydrometeor mass and number fields cross zero and "
        "take different branches; reflectivity reaches 11,072,910 ULP."
    ),
    (
        "ESTABLISHED: gpuwm implements both WRF rimed-ice identities, with "
        "morr_rimed_ice=0 selecting graupel AG=19.3/BG=0.37/RHOG=400 and =1 "
        "selecting WRF-default hail AG=114.5/BG=0.5/RHOG=900 in both the "
        "process kernel and reflectivity diagnostic. The stock-WRF fixture "
        "holds atmospheric inputs fixed between modes and observes 150 "
        "mode-dependent reflectivity lanes. This worktree is not "
        "hardcoded-graupel."
    ),
    (
        "NOT ESTABLISHED: max_ulp 0, or any WRF forecast-trajectory agreement. "
        "The former tests described a float64 transcription mirror as a WRF "
        "oracle; that label was false because the mirror reimplemented the "
        "port's own arithmetic. Open causes measured by the real oracle are "
        "CuPy -ftz=true at subnormal branches, CUDA/glibc transcendental "
        "differences, CUDA tgammaf/cbrtf and algebraic substitutions for "
        "WRF's REAL WGAMMA/powf/statement order, and FMA contraction. "
        "-fmad=false reduces several residuals but does not close the "
        "hydrometeor branch differences. Closing parity requires a systematic "
        "statement-order and REAL-math transcription, not a tolerance change."
    ),
]


def _find_object_span(raw: bytes) -> tuple[int, int]:
    marker = raw.find(OPTION_MARKER)
    if marker < 0 or raw.find(OPTION_MARKER, marker + 1) >= 0:
        raise ValueError("expected exactly one morrison-mp10 option")
    start = raw.find(b"{", marker + len(OPTION_MARKER))
    if start < 0:
        raise ValueError("morrison-mp10 object has no opening brace")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        byte = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte == 0x7B:
            depth += 1
        elif byte == 0x7D:
            depth -= 1
            if depth == 0:
                return start, index + 1
    raise ValueError("morrison-mp10 object has no closing brace")


def patch_bytes(raw: bytes) -> bytes:
    """Return *raw* with only the Morrison option object replaced."""
    start, end = _find_object_span(raw)
    option = json.loads(raw[start:end])
    if option.get("maturity") not in {
        "wrf-matched-run", "implemented-unverified",
    }:
        raise ValueError(f"unexpected Morrison maturity: {option.get('maturity')}")
    if option.get("warnings") not in ([], MORRISON_WARNINGS):
        raise ValueError("refusing to overwrite unknown Morrison warnings")
    option["maturity"] = "implemented-unverified"
    option["warnings"] = MORRISON_WARNINGS
    replacement = json.dumps(
        option, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    result = raw[:start] + replacement + raw[end:]

    # Prove the byte replacement has the intended semantic scope.
    before = json.loads(raw)
    after = json.loads(result)
    before_option = before["components"]["microphysics"]["options"].pop(
        "morrison-mp10")
    after_option = after["components"]["microphysics"]["options"].pop(
        "morrison-mp10")
    assert before == after
    before_option["maturity"] = "implemented-unverified"
    before_option["warnings"] = MORRISON_WARNINGS
    assert before_option == after_option
    return result


def _revision_bytes(revision: str) -> bytes:
    return subprocess.check_output(
        ["git", "cat-file", "blob", f"{revision}:{REGISTRY_PATH.as_posix()}"],
        cwd=REPO_ROOT,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--git-revision")
    source.add_argument("--working-tree", action="store_true")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--stdout", action="store_true")
    destination.add_argument("--write-git-blob", action="store_true")
    destination.add_argument("--update-working-tree", action="store_true")
    args = parser.parse_args()

    path = REPO_ROOT / REGISTRY_PATH
    raw = (
        path.read_bytes()
        if args.working_tree
        else _revision_bytes(args.git_revision)
    )
    result = patch_bytes(raw)
    if args.stdout:
        sys.stdout.buffer.write(result)
    elif args.write_git_blob:
        completed = subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            cwd=REPO_ROOT,
            input=result,
            check=True,
            stdout=subprocess.PIPE,
        )
        sys.stdout.buffer.write(completed.stdout)
    else:
        if not args.working_tree:
            parser.error("--update-working-tree requires --working-tree")
        path.write_bytes(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
