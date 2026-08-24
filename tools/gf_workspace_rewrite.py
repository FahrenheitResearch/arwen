"""One-shot mechanical rewrite: gf.cu column arrays -> per-thread workspace.

Run once to produce the workspace form of ``gpuwm/core/kernels/gf.cu``.  It
replaces every function-scope ``float NAME[GF_KP];`` / ``int NAME[GF_KP];``
declaration inside the five owning functions with a slice of a caller-provided
global workspace, keeping one source line per original source line so the diff
reads as a declaration swap and nothing else.

Kept in-tree as the audit trail for that rewrite: re-running it on the
pre-rewrite file reproduces the committed file byte for byte, and the slot map
it prints is what the GFWS_SLOT_COUNT_* caps in gf.cu are set from.
"""
from __future__ import annotations

import re
import sys

DECL = re.compile(
    r"^(\s*)(float|int)\s+(.+?);\s*$")
ITEM = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*GF_KP\s*\]$")

# function name -> (workspace base expression used inside it, region)
OWNERS = {
    "gfd_deep_column": ("gfws", "col"),
    "gfd_shallow_column": ("gfws", "col"),
    "gf_deep_stage": ("gfws_own", "drv"),
    "gf_shallow_stage": ("gfws_own", "drv"),
    "gf_gfdrv_stage": ("gfws_own", "drv"),
}


def spans(lines):
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("__device__") or line.startswith('extern "C"'):
            m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
            name = m.group(1) if m else None
            j = i
            while j < len(lines) and not lines[j].rstrip().endswith("{"):
                j += 1
                if j - i > 40:
                    break
            if j >= len(lines) or not lines[j].rstrip().endswith("{"):
                i += 1
                continue
            depth = 1
            k = j
            while depth > 0 and k + 1 < len(lines):
                k += 1
                depth += lines[k].count("{") - lines[k].count("}")
            if name in OWNERS:
                out.append((name, j, k))
            i = k
        i += 1
    return out


def main(path, out_path):
    lines = open(path, encoding="utf-8").read().split("\n")
    slot_map = {}
    for name, body_start, end in spans(lines):
        base, region = OWNERS[name]
        nxt = 0
        used = []
        for n in range(body_start + 1, end):
            m = DECL.match(lines[n])
            if not m:
                continue
            indent, ctype, rest = m.groups()
            items = [s.strip() for s in rest.split(",")]
            parsed = [ITEM.match(s) for s in items]
            if not all(parsed):
                continue
            names = [p.group(1) for p in parsed]
            pieces = []
            for nm in names:
                if ctype == "float":
                    pieces.append(f"float *{nm} = GFWS_AT({base}, {nxt});")
                else:
                    pieces.append(
                        f"int *{nm} = (int *)GFWS_AT({base}, {nxt});")
                used.append((nxt, ctype, nm))
                nxt += 1
            lines[n] = ("\n" + indent).join(pieces)
            lines[n] = indent + lines[n]
        slot_map[name] = (region, used)
    open(out_path, "w", encoding="utf-8", newline="").write("\n".join(lines))
    col = max(len(v[1]) for k, v in slot_map.items() if v[0] == "col")
    drv = max(len(v[1]) for k, v in slot_map.items() if v[0] == "drv")
    for name, (region, used) in slot_map.items():
        print(f"{name}: region={region} slots={len(used)}")
    print(f"GFWS_SLOT_COUNT_COL = {col}")
    print(f"GFWS_SLOT_COUNT_DRV = {drv}")
    print(f"TOTAL SLOTS = {col + drv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
