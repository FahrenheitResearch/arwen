"""``NT_STAGE_SIGNATURE`` is re-derived from the kernel source every run.

WHY IT IS A TABLE AT ALL. The assembler binds each stage's arguments by
NAME -- the workspace resolves a name to a buffer and ``NT_SEEDS`` resolves
the cases where the reference gives one array two names. That needs the
ORDER, and a 709-entry table is only safe to have if nothing maintains it
by hand: a kernel that gains, loses or reorders a parameter would otherwise
be launched with its arguments shifted by one, which CUDA accepts happily
and which reads another array's memory as a float.

So the table is generated from ``ntiedtke.cu`` and this file regenerates it
and compares. It is the same shape as the constant-family gate and the
stage-id gate, and the opposite of the four lists that went stale.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from gpuwm.core.ntiedtke import NT_STAGE_SIGNATURE

_CU = (Path(__file__).resolve().parents[1] / "gpuwm" / "core" / "kernels"
       / "ntiedtke.cu")

_TAIL = ("expect_tpb", "expect_nblocks", "geom_report", "order_report",
         "ticket")


def _parse():
    """Every kernel's ordered parameter names, straight from the source."""
    src = _CU.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\(', src):
        i = m.end() - 1
        depth, j = 0, i
        while True:
            depth += 1 if src[j] == "(" else -1 if src[j] == ")" else 0
            if depth == 0 and src[j] == ")":
                break
            j += 1
        params = re.sub(r"/\*.*?\*/", "", src[i + 1:j], flags=re.S)
        params = re.sub(r"//[^\n]*", "", params)
        names = []
        for p in (x.strip() for x in params.split(",")):
            if not p:
                continue
            names.append(re.sub(r"\[\s*\]$", "", p.split()[-1]).lstrip("*"))
        out[m.group(1)] = tuple(names)
    return out


def test_the_parser_sees_the_kernels():
    """A gate that scans nothing passes vacuously."""
    got = _parse()
    assert len(got) >= 21, f"only {len(got)} kernels parsed"
    assert sum(len(v) for v in got.values()) > 600, "argument scan is short"
    assert got["ntiedtke_post_run"][0] == "exner"


def test_the_table_matches_the_kernel_source_exactly():
    got = _parse()
    assert set(got) == set(NT_STAGE_SIGNATURE), (
        f"only in the .cu: {sorted(set(got) - set(NT_STAGE_SIGNATURE))}; "
        f"only in the table: {sorted(set(NT_STAGE_SIGNATURE) - set(got))}")
    for name in sorted(got):
        assert got[name] == NT_STAGE_SIGNATURE[name], (
            f"{name}'s parameter list moved.\n"
            f"  source: {got[name]}\n"
            f"  table : {NT_STAGE_SIGNATURE[name]}\n"
            f"Regenerate NT_STAGE_SIGNATURE; do NOT edit it by hand.")


@pytest.mark.parametrize("stage", sorted(NT_STAGE_SIGNATURE))
def test_every_stage_ends_with_the_geometry_tail(stage):
    """The descriptor arguments are last, on every kernel, in one order.

    ``NtStages.launch`` appends them, so a kernel that took them anywhere
    else would be launched with them in the wrong slots -- and the geometry
    check would then be reading a physical array as its expected tile.
    """
    assert NT_STAGE_SIGNATURE[stage][-len(_TAIL):] == _TAIL, (
        f"{stage} does not end with {_TAIL}")


@pytest.mark.parametrize("stage", sorted(NT_STAGE_SIGNATURE))
def test_no_stage_repeats_a_parameter_name(stage):
    """Two parameters with one name would make binding ambiguous.

    The assembler binds by name, so a duplicate would silently pass the
    same buffer twice -- and aliasing two dummies to one array is exactly
    what NT_SEEDS exists to make explicit rather than accidental.
    """
    names = NT_STAGE_SIGNATURE[stage]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"{stage} repeats {dupes}"
