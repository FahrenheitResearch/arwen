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
        "Morrison is the literal transcription of the unmodified WRF v4.6.1 "
        "module_mp_morr_two_moment (driver, MORR_TWO_MOMENT_MICRO, POLYSVP, "
        "GAMMA, refl10cm_hm). Built without the two declared corrections "
        "(MORR_ROW37_OFF) it returns WRF's words bitwise under both strict "
        "and default arithmetic on the 28-column wrapper fixture, six seeded "
        "stress sets (24,576 columns) and 742 weather-regime columns "
        "(tools/morrison_wrf461_oracle). Forecast-trajectory agreement with "
        "WRF remains unestablished."
    ),
    (
        "Both WRF rimed-ice identities are implemented, with "
        "morr_rimed_ice=0 selecting graupel AG=19.3/BG=0.37/RHOG=400 and =1 "
        "selecting WRF-default hail AG=114.5/BG=0.5/RHOG=900 in both the "
        "process kernel and reflectivity diagnostic; the constants are the "
        "words WRF's own MORR_TWO_MOMENT_INIT and radar_init compute."
    ),
    (
        "Declared divergence 1: where WRF's float freezing-rate exponentials "
        "overflow (very cold rain or cloud), WRF returns non-finite state; "
        "WOOF re-evaluates the rates in double and rations the cloud and "
        "rain donors, so its state stays finite. Measured: it fires only in "
        "columns where WRF's own output is not finite."
    ),
    (
        "Declared divergence 2: deposition nucleation caps MNUCCD by the "
        "vapor excess over ice saturation and scales NNUCCD with it. Below "
        "about 159 K WRF's one-sided FUDGEF limiter lets nucleation draw "
        "vapor that is not there (negative vapor); above it, in cold "
        "ice-supersaturated air near the tropopause (measured 178-213 K), "
        "the cap keeps each nucleated crystal at its seed mass where WRF "
        "shrinks the mass but not the number, so ni, qi, qs, qv and "
        "reflectivity differ from WRF in those columns."
    ),
    (
        "Declared divergence 3: WRF's SR = SNOWRT/(PRECRT+1E-12) sums "
        "signed fallout and leaves [0, 1] in degenerate columns (WRF itself "
        "returns SR < 0 in 401 and SR > 1 in 208 of 24,576 stress columns); "
        "the forecast path clips SR to [0, 1] before the land surface reads "
        "it. Only SR moves, only there."
    ),
    (
        "The four effective radii have no WRF expected values in the "
        "wrapper fixture; WRF's wrapper does not return them."
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
