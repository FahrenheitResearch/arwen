"""Re-locate line-range extracts of a WRF file in another version of that file.

    python3 map_ranges.py OLD_REF NEW_SRC RANGES_IN RANGES_OUT

RANGES_IN lines are ``<rel> <first> <last> <name>`` against OLD_REF (here the
pinned 4.7.1 file).  Rows whose <rel> is not the mapped file pass through
unchanged.  For the mapped file, the two endpoints are carried through a
difflib alignment of the two files' lines; prints, per row, the new range and
whether the extracted text is byte-identical between the versions.
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path


def main(old_path: str, new_path: str, ranges_in: str, ranges_out: str) -> None:
    old = Path(old_path).read_bytes().splitlines()
    new = Path(new_path).read_bytes().splitlines()
    rel_name = "chem/" + Path(old_path).name
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    mapping: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k

    def carry(index: int, direction: int) -> int:
        # nearest aligned line, walking outwards in ``direction``
        probe = index
        while probe not in mapping:
            probe += direction
        return mapping[probe] + (index - probe)

    rows_out = []
    for line in Path(ranges_in).read_text().splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        rel, first, last, name = parts[0], int(parts[1]), int(parts[2]), parts[3]
        if rel != rel_name:
            rows_out.append(line)
            continue
        f0, l0 = first - 1, last - 1
        nf = carry(f0, -1) if f0 not in mapping else mapping[f0]
        nl = carry(l0, +1) if l0 not in mapping else mapping[l0]
        same = old[f0:l0 + 1] == new[nf:nl + 1]
        rows_out.append(f"{rel} {nf + 1} {nl + 1} {name}")
        print(f"{name:24s} 4.7.1 {first}-{last} -> 4.6.1 {nf + 1}-{nl + 1} "
              f"{'IDENTICAL' if same else 'DIFFERS'}")
    Path(ranges_out).write_text("\n".join(rows_out) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:5])
