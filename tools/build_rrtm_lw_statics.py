"""Package WRF v4.6.1 RRTM longwave module DATA arrays as npz.

``phys/module_ra_rrtm.F`` carries its small tables (g-point mapping, Planck
fractions, trace-species cross sections, reference profiles, and the
181-temperature integrated Planck table) as module-level Fortran DATA
statements (module_ra_rrtm.F:189-1727).  The big absorption tables live in
the separate ``RRTM_DATA`` binary and are NOT parsed here --
``gpuwm.core.rrtm.load_rrtm_raw_tables`` owns those.

This tool parses the DATA statements of the provenance-pinned copy at
``gpuwm/data/wrf_radiation/module_ra_rrtm.F`` (SHA-256 verified against
gpuwm.core.rrtm.RRTM_SOURCE_SHA256) and writes
``gpuwm/data/wrf_radiation/rrtm_lw_statics.npz``: one member per Fortran
array, float32/int32, stored flat in Fortran (column-major) element order
exactly as a DATA statement fills it.  ``gpuwm.core.rrtm_lw`` reshapes with
``order="F"`` so indices match the oracle source.

Instrument validation (run on every build): completeness masks prove every
element of every declared array was assigned exactly once; the g-point
bookkeeping identities (sum NGC = 140, NGS = cumsum NGC, sum NGN = 256,
NGB histogram = NGC) are recomputed from the parsed values; and spot values
transcribed independently by hand from the source listing are compared.
Resolution limit: literal -> float32 conversion goes through a Python
double; all literals in this module carry <= 9 significant digits, far
below the ~17 digits where double rounding could differ from a direct
decimal -> float32 conversion.

Usage: python tools/build_rrtm_lw_statics.py [--source PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO / "gpuwm" / "data" / "wrf_radiation" / "module_ra_rrtm.F"
DEFAULT_OUT = REPO / "gpuwm" / "data" / "wrf_radiation" / "rrtm_lw_statics.npz"

#: SHA-256 of the packaged module_ra_rrtm.F (gpuwm/data/wrf_radiation/
#: PROVENANCE.md); must match gpuwm.core.rrtm.RRTM_SOURCE_SHA256.
SOURCE_SHA256 = \
    "07ebd4e83b2a571837a27748a29ca678440c363f96a3694fc9549ae0911335da"

MG = 16
NBANDS = 16
NGPT = 140

#: Declared Fortran shapes (module_ra_rrtm.F:70-111).  Integer arrays are
#: listed in INT_ARRAYS; scalars have shape ().
DECLARATIONS: dict[str, tuple[int, ...]] = {
    "NGC": (NBANDS,), "NGS": (NBANDS,), "NGM": (MG * NBANDS,),
    "NGN": (NGPT,), "NGB": (NGPT,),
    "WT": (MG,),
    "FRACREFA1": (MG,), "FRACREFB1": (MG,), "FORREF1": (MG,),
    "FRACREFA2": (MG, 13), "FRACREFB2": (MG,), "FORREF2": (MG,),
    "FRACREFA3": (MG, 10), "FRACREFB3": (MG, 5),
    "FORREF3": (MG,), "ABSN2OA3": (MG,), "ABSN2OB3": (MG,),
    "FRACREFA4": (MG, 9), "FRACREFB4": (MG, 6),
    "FRACREFA5": (MG, 9), "FRACREFB5": (MG, 5), "CCL45": (MG,),
    "FRACREFA6": (MG,), "ABSCO26": (MG,), "CFC11ADJ6": (MG,),
    "CFC126": (MG,),
    "FRACREFA7": (MG, 9), "FRACREFB7": (MG,), "ABSCO27": (MG,),
    "FRACREFA8": (MG,), "FRACREFB8": (MG,), "ABSCO2A8": (MG,),
    "ABSCO2B8": (MG,), "ABSN2OA8": (MG,), "ABSN2OB8": (MG,),
    "CFC128": (MG,), "CFC22ADJ8": (MG,),
    "FRACREFA9": (MG, 9), "FRACREFB9": (MG,), "ABSN2O9": (3 * MG,),
    "FRACREFA10": (MG,), "FRACREFB10": (MG,),
    "FRACREFA11": (MG,), "FRACREFB11": (MG,),
    "FRACREFA12": (MG, 9),
    "FRACREFA13": (MG, 9),
    "FRACREFA14": (MG,), "FRACREFB14": (MG,),
    "FRACREFA15": (MG, 9),
    "FRACREFA16": (MG, 9),
    "NXMOL": (), "IXINDX": (35,),
    "WAVENUM1": (NBANDS,), "WAVENUM2": (NBANDS,), "DELWAVE": (NBANDS,),
    "NG": (NBANDS,), "NSPA": (NBANDS,), "NSPB": (NBANDS,),
    "HEATFAC": (),
    "PREF": (59,), "PREFLOG": (59,), "TREF": (59,),
    "TOTPLNK": (181, NBANDS), "TOTPLK16": (181,),
}
INT_ARRAYS = frozenset((
    "NGC", "NGS", "NGM", "NGN", "NGB", "NXMOL", "IXINDX",
    "NG", "NSPA", "NSPB"))

#: Subroutine-local DATA arrays the solver also needs, keyed by the
#: subroutine that declares them (several names collide across TAUGBn
#: scopes with different lengths, so scope qualification is mandatory).
#: npz member names are "<SCOPE>__<NAME>".  All are REAL profiles.
SCOPED_DECLARATIONS: dict[str, dict[str, tuple[int, ...]]] = {
    # module_ra_rrtm.F:3576-3600
    "O3DATA": {"O3SUM": (31,), "PPSUM": (31,), "O3WIN": (31,),
               "PPWIN": (31,)},
    # module_ra_rrtm.F:3812-3831 (Cavallo buffer-layer standard atmosphere)
    "MM5ATM": {"PPROF": (60,), "TPROF": (60,)},
    # module_ra_rrtm.F:4616-4619
    "TAUGB2": {"REFPARAM": (13,)},
    # module_ra_rrtm.F:4727-4757
    "TAUGB3": {"ETAREF": (10,), "H2OREF": (59,), "N2OREF": (59,),
               "CO2REF": (59,)},
    # module_ra_rrtm.F:5365-5403
    "TAUGB8": {"H2OREF": (59,), "N2OREF": (59,), "O3REF": (59,)},
    # module_ra_rrtm.F:5507-5522
    "TAUGB9": {"N2OREF": (13,), "H2OREF": (13,), "CH4REF": (13,),
               "ETAREF": (11,)},
}

_NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?"


def _joined_statements(lines: list[str]) -> list[str]:
    """Continuation-joined code statements (comments stripped).

    Comment strip is a bare "!" split: the DATA regions this tool reads
    contain no character literals, and a "!" inside one would be caught
    by the value-list parser refusing the mangled statement.
    """
    joined: list[str] = []
    pending = ""
    for raw in lines:
        code = raw.split("!", 1)[0].rstrip()
        if not code.strip():
            continue
        if pending:
            code = pending + " " + code.strip()
            pending = ""
        if code.rstrip().endswith("&"):
            pending = code.rstrip()[:-1]
            continue
        joined.append(code.strip())
    if pending:
        joined.append(pending.strip())
    return joined


def _module_head_statements(text: str) -> list[str]:
    """Continuation-joined statements of the module head (pre-CONTAINS)."""
    head: list[str] = []
    for raw in text.splitlines():
        if re.match(r"^\s*CONTAINS\s*$", raw, re.IGNORECASE):
            break
        head.append(raw)
    else:
        raise ValueError("no CONTAINS statement found")
    return _joined_statements(head)


def _scoped_statements(text: str) -> dict[str, list[str]]:
    """Continuation-joined statements grouped by containing subroutine."""
    scopes: dict[str, list[str]] = {}
    current: str | None = None
    block: list[str] = []
    for raw in text.splitlines():
        end = re.match(r"^\s*END\s+SUBROUTINE\b", raw, re.IGNORECASE)
        begin = re.match(r"^\s*SUBROUTINE\s+(\w+)", raw, re.IGNORECASE)
        if end and current is not None:
            scopes.setdefault(current, []).extend(_joined_statements(block))
            current, block = None, []
        elif begin:
            current, block = begin.group(1).upper(), []
        elif current is not None:
            block.append(raw)
    return scopes


def _parse_values(clist: str) -> list[str]:
    """Expand a DATA value list, honouring ``n*value`` repeats."""
    values: list[str] = []
    for item in clist.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"empty item in DATA value list: {clist!r}")
        repeat = re.fullmatch(r"(\d+)\s*\*\s*(" + _NUM + r")", item)
        if repeat:
            values.extend([repeat.group(2)] * int(repeat.group(1)))
        else:
            if not re.fullmatch(_NUM, item):
                raise ValueError(f"unparseable DATA value {item!r}")
            values.append(item)
    return values


class _Target:
    """One declared array plus its positional fill mask."""

    def __init__(self, name: str, shape: tuple[int, ...]):
        self.name = name
        self.shape = shape
        count = int(np.prod(shape)) if shape else 1
        self.flat = np.zeros(count, np.float64)
        self.filled = np.zeros(count, bool)

    def assign(self, positions: list[int], values: list[str]) -> None:
        if len(positions) != len(values):
            raise ValueError(
                f"{self.name}: {len(positions)} positions but "
                f"{len(values)} values")
        for pos, value in zip(positions, values):
            if self.filled[pos]:
                raise ValueError(
                    f"{self.name}: element {pos} assigned twice")
            self.flat[pos] = float(value.replace("D", "E").replace("d", "e"))
            self.filled[pos] = True


def _column_major_offset(shape: tuple[int, ...], index: tuple[int, ...]):
    """0-based flat offset of a 1-based Fortran index (column-major)."""
    offset = 0
    stride = 1
    for dim, ix in zip(shape, index):
        if not 1 <= ix <= dim:
            raise ValueError(f"index {index} outside shape {shape}")
        offset += (ix - 1) * stride
        stride *= dim
    return offset


def _parse_data_statements(statements: list[str],
                           declarations: dict[str, tuple[int, ...]],
                           *, ignore_unknown: bool = False
                           ) -> dict[str, _Target]:
    targets = {name: _Target(name, shape)
               for name, shape in declarations.items()}
    seen_data = 0
    for stmt in statements:
        match = re.match(r"^DATA\s+(.*)$", stmt, re.IGNORECASE)
        if not match:
            continue
        body = match.group(1)
        seen_data += 1
        pos = 0
        while pos < len(body):
            # nlist: up to the first unparenthesised '/'
            depth = 0
            for cursor in range(pos, len(body)):
                char = body[cursor]
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif char == "/" and depth == 0:
                    break
            else:
                raise ValueError(f"DATA nlist without value list: {stmt!r}")
            nlist = body[pos:cursor].strip().rstrip(",").strip()
            end = body.index("/", cursor + 1)
            clist = body[cursor + 1:end]
            pos = end + 1
            while pos < len(body) and body[pos] in ", ":
                pos += 1
            values = _parse_values(clist)
            _assign_nlist(targets, nlist, values, stmt,
                          ignore_unknown=ignore_unknown)
    if not seen_data:
        raise ValueError("no DATA statements found in scope")
    return targets


def _nlist_names(nlist: str) -> list[str]:
    """Array names a DATA nlist references (implied-DO or simple items)."""
    implied = re.match(r"^\(\s*(\w+)\s*\(", nlist)
    if implied:
        return [implied.group(1).upper()]
    return [re.match(r"\s*(\w+)", item).group(1).upper()
            for item in re.split(r",(?![^()]*\))", nlist)]


def _assign_nlist(targets: dict[str, _Target], nlist: str,
                  values: list[str], stmt: str,
                  *, ignore_unknown: bool = False) -> None:
    if ignore_unknown:
        names = _nlist_names(nlist)
        known = [name in targets for name in names]
        if not any(known):
            return
        if not all(known):
            raise ValueError(
                f"DATA clause mixes extracted and ignored arrays: {stmt!r}")
    implied = re.fullmatch(
        r"\(\s*(\w+)\s*\(\s*(\w+)\s*(?:,\s*(\d+)\s*)?\)\s*,"
        r"\s*(\w+)\s*=\s*(\d+)\s*,\s*(\d+)\s*\)",
        nlist)
    if implied:
        name = implied.group(1).upper()
        loop_var = implied.group(2).upper()
        fixed = implied.group(3)
        if implied.group(4).upper() != loop_var:
            raise ValueError(f"implied-DO variable mismatch in {stmt!r}")
        lo, hi = int(implied.group(5)), int(implied.group(6))
        target = targets.get(name)
        if target is None:
            raise ValueError(f"DATA for undeclared array {name}")
        if fixed is None:
            positions = [_column_major_offset(target.shape, (i,))
                         for i in range(lo, hi + 1)]
        else:
            positions = [
                _column_major_offset(target.shape, (i, int(fixed)))
                for i in range(lo, hi + 1)]
        target.assign(positions, values)
        return
    # Comma-separated simple items: NAME or NAME(i).  A full-array item
    # consumes size(NAME) values from the front of the list.
    items = []
    depth = 0
    start = 0
    for cursor, char in enumerate(nlist):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(nlist[start:cursor].strip())
            start = cursor + 1
    items.append(nlist[start:].strip())
    consumed = 0
    for item in items:
        sub = re.fullmatch(r"(\w+)\s*\(\s*(\d+)\s*\)", item)
        if sub:
            name = sub.group(1).upper()
            target = targets.get(name)
            if target is None:
                raise ValueError(f"DATA for undeclared array {name}")
            offset = _column_major_offset(target.shape, (int(sub.group(2)),))
            target.assign([offset], values[consumed:consumed + 1])
            consumed += 1
            continue
        if not re.fullmatch(r"\w+", item):
            raise ValueError(f"unparseable DATA nlist item {item!r}")
        name = item.upper()
        target = targets.get(name)
        if target is None:
            raise ValueError(f"DATA for undeclared array {name}")
        count = target.flat.size
        target.assign(list(range(count)), values[consumed:consumed + count])
        consumed += count
    if consumed != len(values):
        raise ValueError(
            f"DATA statement value count mismatch ({consumed} consumed, "
            f"{len(values)} supplied): {stmt!r}")


def _validate(targets: dict[str, _Target]) -> None:
    for target in targets.values():
        if not target.filled.all():
            missing = int((~target.filled).sum())
            raise ValueError(
                f"{target.name}: {missing} elements never assigned")
    ngc = targets["NGC"].flat.astype(np.int64)
    ngs = targets["NGS"].flat.astype(np.int64)
    ngn = targets["NGN"].flat.astype(np.int64)
    ngb = targets["NGB"].flat.astype(np.int64)
    ng = targets["NG"].flat.astype(np.int64)
    if ngc.sum() != NGPT:
        raise ValueError(f"sum(NGC) = {ngc.sum()} != {NGPT}")
    if not np.array_equal(np.cumsum(ngc), ngs):
        raise ValueError("NGS != cumsum(NGC)")
    if ngn.sum() != MG * NBANDS:
        raise ValueError(f"sum(NGN) = {ngn.sum()} != {MG * NBANDS}")
    counts = np.bincount(ngb, minlength=NBANDS + 1)[1:]
    if not np.array_equal(counts, ngc):
        raise ValueError("NGB band histogram != NGC")
    if not np.array_equal(ng, np.full(NBANDS, MG)):
        raise ValueError("NG != 16 everywhere")
    # Hand-transcribed spot values straight from the source listing.
    spots = {
        ("NSPA", 2): 1, ("NSPA", 3): 10, ("NSPB", 4): 6,
        ("WAVENUM1", 1): 10.0, ("WAVENUM2", 16): 3000.0,
        ("DELWAVE", 11): 320.0, ("HEATFAC", 1): 8.4391,
        ("WT", 1): 0.1527534276, ("WT", 16): 0.000075,
        ("FRACREFA1", 1): 0.08452097, ("FORREF1", 16): 0.147405,
        ("NXMOL", 1): 2, ("IXINDX", 2): 2, ("IXINDX", 3): 3,
        ("IXINDX", 35): 0,
    }
    for (name, index), expected in spots.items():
        got = targets[name].flat[index - 1]
        if not np.isclose(got, expected, rtol=0, atol=abs(expected) * 1e-6):
            raise ValueError(f"{name}({index}) = {got} != {expected}")
    pref = targets["PREF"].flat
    preflog = targets["PREFLOG"].flat
    if not np.allclose(np.log(pref), preflog, atol=5e-4):
        raise ValueError("PREFLOG is not ln(PREF) to declared precision")
    totplnk = targets["TOTPLNK"].flat.reshape((181, NBANDS), order="F")
    if not (np.diff(totplnk, axis=0) > 0).all():
        raise ValueError("TOTPLNK not strictly increasing with temperature")
    if not (np.diff(targets["TOTPLK16"].flat) > 0).all():
        raise ValueError("TOTPLK16 not strictly increasing")


def _validate_scoped(members: dict[str, np.ndarray]) -> None:
    """Spot-check scoped profiles against hand-transcribed source values."""
    spots = {
        ("O3DATA__O3SUM", 1): 5.297e-8, ("O3DATA__PPSUM", 31): 0.647,
        ("O3DATA__O3WIN", 31): 6.135e-6, ("O3DATA__PPWIN", 1): 955.747,
        ("MM5ATM__PPROF", 1): 1000.00, ("MM5ATM__PPROF", 60): 0.10,
        ("MM5ATM__TPROF", 1): 279.94, ("MM5ATM__TPROF", 60): 237.53,
        ("TAUGB2__REFPARAM", 1): 0.903661,
        ("TAUGB3__ETAREF", 9): 0.9875, ("TAUGB3__CO2REF", 1): 3.55e-4,
        ("TAUGB3__CO2REF", 59): 3.5079606e-4,
        ("TAUGB8__O3REF", 1): 3.01700e-8,
        ("TAUGB9__ETAREF", 9): 0.96, ("TAUGB9__CH4REF", 13): 1.3573376e-6,
    }
    for (name, index), expected in spots.items():
        got = members[name][index - 1]
        if not np.isclose(got, expected, rtol=0, atol=abs(expected) * 1e-6):
            raise ValueError(f"{name}[{index}] = {got} != {expected}")
    for name in ("TAUGB3__H2OREF", "TAUGB8__H2OREF"):
        if not np.array_equal(members[name],
                              members["TAUGB3__H2OREF"]):
            raise ValueError("59-level H2OREF profiles diverge across bands")
    if not np.array_equal(members["TAUGB3__N2OREF"],
                          members["TAUGB8__N2OREF"]):
        raise ValueError("59-level N2OREF profiles diverge across bands")
    if not np.array_equal(members["TAUGB9__N2OREF"],
                          members["TAUGB3__N2OREF"][:13]):
        raise ValueError("TAUGB9 N2OREF is not the 13-level head")


def build(source: Path, out: Path) -> None:
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(
            f"{source} SHA-256 {digest} != pinned {SOURCE_SHA256}; refusing "
            "to package DATA statements from an unverified source")
    text = data.decode("ascii", "replace")
    targets = _parse_data_statements(
        _module_head_statements(text), DECLARATIONS)
    _validate(targets)
    members: dict[str, np.ndarray] = {}
    for name, target in sorted(targets.items()):
        if name in INT_ARRAYS:
            members[name] = target.flat.astype(np.int32)
        else:
            members[name] = target.flat.astype(np.float32)
    scopes = _scoped_statements(text)
    for scope, declarations in SCOPED_DECLARATIONS.items():
        scoped = _parse_data_statements(
            scopes[scope], declarations, ignore_unknown=True)
        for name, target in sorted(scoped.items()):
            if not target.filled.all():
                raise ValueError(f"{scope}.{name}: incomplete DATA fill")
            members[f"{scope}__{name}"] = target.flat.astype(np.float32)
    _validate_scoped(members)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as stream:
        np.savez(stream, **members)
    total = sum(m.size for m in members.values())
    print(f"wrote {out} ({len(members)} members, {total} values)")
    print("sha256:", hashlib.sha256(out.read_bytes()).hexdigest())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    build(args.source, args.out)


if __name__ == "__main__":
    sys.exit(main())
