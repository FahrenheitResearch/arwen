"""The whole-slab THERMOPROP/TSNOSOI/PHASECHANGE/RADIATION packing, held
byte for byte against the per-column packers it replaces.

:mod:`gpuwm.core.noahmp_thermal_slab` exists to delete a Python loop, not to
change an answer.  So the bar it is held to here is not ``max_ulp 0`` against
an oracle -- the four kernels are already pinned that way by
``tests/test_noahmp_thermal.py``, ``tests/test_noahmp_radiation_cuda.py`` and
``tests/test_noahmp_leaf_batches_cuda.py`` -- but *bitwise identity with the
per-column path*, on both halves of the seam:

1. the flat device rows the two packers build, ``float32`` block and ``int32``
   block, compared as bytes with dtype and shape;
2. every field the two evaluations return, compared as bytes, field by field,
   with the field name in the failure message.

Both halves are needed.  (1) alone would pass a slab that packed correctly and
unpacked into the wrong slice; (2) alone would pass a pair of compensating
defects either side of the kernel.

The columns
-----------
``N_COLUMNS`` columns are drawn, in order, from two fixture sources and from
nothing else:

* every case of the leaf's own pinned oracle CSV -- ``noahmp-leaves.csv`` for
  THERMOPROP, ``noahmp-thermal.csv`` for TSNOSOI and PHASECHANGE,
  ``noahmp-radiation-albedo.csv`` for RADIATION -- read back through the same
  slot layout ``tests/test_noahmp_thermal.py`` and
  ``tests/test_noahmp_radiation_cuda.py`` read it through;
* the physical paused calls ``tests/test_noahmp_leaf_batches_cuda.py``
  collects by running the four unmodified-WRF whole-column fixtures through
  ``sflx_steps``.

RADIATION's SOLAD and SOLAI are not in the ALBEDO fixture, because ALBEDO does
not take them; they are taken from the SURRAD fixture, which is where this
tree's pinned direct/diffuse forcing lives.  No number in this file is
invented.

The distinct columns are then cycled up to ``N_COLUMNS``, which is 96: past 32
as the gate requires, and past 64 so the launch's second, partly-filled block
is exercised as well as its first.  Cycling keeps every neighbouring pair
distinct, which is what makes the one-column roll below observable.
``test_the_columns_reach_the_branches_that_matter`` asserts the heterogeneity
rather than trusting it: ``ISNOW`` spans 0 and a three-layer pack, PHASECHANGE
reaches both melting and freezing layers, and RADIATION carries both night and
daylight columns and both land and lake.

Shown failing first
-------------------
``test_transposing_two_adjacent_slots_is_rejected`` and
``test_rolling_one_field_onto_its_neighbour_is_rejected`` corrupt the slab --
by exchanging two adjacent slots, and by shifting one field one column onto
its neighbour -- and require *both* comparisons above to reject each.  A gate
that has never been seen failing is not evidence.
"""

from __future__ import annotations

import csv
import struct
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu

cp = pytest.importorskip("cupy")

import test_noahmp_leaf_batches_cuda as BATCHES

from gpuwm.core import noahmp_thermal_slab as slab
from gpuwm.core.noahmp_radiation_gpu import (OUTPUT_NAMES, PARAMETER_SLOTS,
                                             STATE_SLOTS,
                                             evaluate_radiation_calls,
                                             pack_radiation_calls)
from gpuwm.core.noahmp_thermal_gpu import (NLAY, NSNOW, NSOIL,
                                           evaluate_phasechange_calls,
                                           evaluate_thermoprop_calls,
                                           evaluate_tsnosoi_calls,
                                           pack_phasechange_calls,
                                           pack_thermoprop_calls,
                                           pack_tsnosoi_calls)

REPO = Path(__file__).resolve().parents[1]
ORACLE = REPO / "gpuwm" / "data" / "noahmp" / "oracle"

#: >= 32 as the gate requires; 96 also puts a second, partly-filled 64-thread
#: block on the grid, so a packer that only ever sees one block is not what is
#: being measured here.
N_COLUMNS = 96

LEAVES = ("thermoprop", "tsnosoi", "phasechange", "radiation")


# ---------------------------------------------------------------------------
# fixture readers -- the same encodings the existing per-leaf tests use
# ---------------------------------------------------------------------------

def _f32_from_bits(bits: str) -> np.float32:
    """``noahmp-thermal.csv`` / ``noahmp-leaves.csv``'s ``bits`` column."""
    return np.frombuffer(struct.pack("<I", int(bits, 16)), dtype=np.float32)[0]


def _f(hexword: str) -> np.float32:
    """``noahmp-radiation-*.csv``'s hex-per-column encoding."""
    return np.float32(struct.unpack("<f", struct.pack("<I",
                                                      int(hexword, 16)))[0])


def _oracle_cases(path: Path, leaf: str):
    """``[(case, {name: value}, {name: int})]`` for one slot-layout leaf.

    A name occupying one slot comes back as a ``np.float32``; a name occupying
    several comes back as a ``float32`` array in slot order, which is the
    Fortran order the per-column packers write into their row.
    """
    grouped: dict[str, dict[str, list]] = defaultdict(
        lambda: defaultdict(list))
    order: list[str] = []
    with path.open(newline="", encoding="ascii") as stream:
        for row in csv.DictReader(stream):
            if row["leaf"] != leaf:
                continue
            if row["case"] not in grouped:
                order.append(row["case"])
            grouped[row["case"]][row["role"]].append(row)

    cases = []
    for case in order:
        floats: dict[str, list] = defaultdict(list)
        for row in sorted(grouped[case]["in"], key=lambda r: int(r["slot"])):
            floats[row["name"]].append(_f32_from_bits(row["bits"]))
        packed = {name: (values[0] if len(values) == 1
                         else np.asarray(values, dtype=np.float32))
                  for name, values in floats.items()}
        ints = {row["name"]: int(float(row["value"]))
                for row in grouped[case]["int"]}
        cases.append((case, packed, ints))
    return cases


def _radiation_rows(leaf: str):
    with (ORACLE / f"noahmp-radiation-{leaf}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


# ---------------------------------------------------------------------------
# the columns
# ---------------------------------------------------------------------------

def _cycled(base: list, n: int) -> list:
    """``n`` columns from ``len(base)`` distinct ones, neighbours distinct."""
    assert len(base) > 1, "a one-column base cannot show a misrouted column"
    return [base[k % len(base)] for k in range(n)]


def _thermoprop_columns():
    base = []
    for _case, floats, ints in _oracle_cases(ORACLE / "noahmp-leaves.csv",
                                             "thermoprop"):
        kwargs = dict(floats)
        kwargs.update(isnow=ints["isnow"], ist=ints["ist"],
                      urban_flag=bool(ints["urban_flag"]))
        base.append(((), kwargs))
    base.extend(BATCHES.COLLECTED["thermoprop"][0])
    return _cycled(base, N_COLUMNS)


def _tsnosoi_columns():
    base = []
    for _case, floats, ints in _oracle_cases(ORACLE / "noahmp-thermal.csv",
                                             "tsnosoi"):
        kwargs = dict(floats)
        kwargs.update(isnow=ints["isnow"])
        base.append(((), kwargs))
    base.extend(BATCHES.COLLECTED["tsnosoi"][0])
    return _cycled(base, N_COLUMNS)


def _phasechange_columns():
    base = []
    for _case, floats, ints in _oracle_cases(ORACLE / "noahmp-thermal.csv",
                                             "phasechange"):
        kwargs = dict(floats)
        kwargs.update(isnow=ints["isnow"], ist=ints["ist"])
        base.append(((), kwargs))
    base.extend(BATCHES.COLLECTED["phasechange"][0])
    return _cycled(base, N_COLUMNS)


#: ``PARAMETER_SLOTS`` names that are two-element vectors on the handle.
_RADIATION_PAIRS = ("albsat", "albdry", "alblak", "rhol", "rhos", "taul",
                    "taus", "omegas")


def _radiation_columns():
    """Ten ALBEDO fixture columns with SURRAD's forcing, then four real calls.

    The ALBEDO oracle carries every ``PARAMETER_SLOTS`` and ``STATE_SLOTS``
    value, SMC, ALBOLD and TAUSS; it does not carry SOLAD/SOLAI because ALBEDO
    does not take them.  Those come from the SURRAD oracle, cycled across the
    ALBEDO rows, so the two forcing vectors are pinned numbers rather than
    chosen ones and the two bands and two streams all differ.
    """
    forcing = _radiation_rows("surrad")
    options = SimpleNamespace(opt_alb=2)
    base = []
    for index, row in enumerate(_radiation_rows("albedo")):
        p = SimpleNamespace()
        for name in PARAMETER_SLOTS:
            if name in _RADIATION_PAIRS:
                setattr(p, name, (_f(row[f"{name}1"]), _f(row[f"{name}2"])))
            else:
                setattr(p, name, _f(row[name]))
        sun = forcing[index % len(forcing)]
        kwargs = {name: _f(row[name]) for name in STATE_SLOTS}
        kwargs.update(
            smc=[_f(row[f"smc{k}"]) for k in range(1, NSOIL + 1)],
            albold=_f(row["albold_in"]), tauss=_f(row["tauss_in"]),
            solad=(_f(sun["solad1"]), _f(sun["solad2"])),
            solai=(_f(sun["solai1"]), _f(sun["solai2"])),
            ist=int(row["ist"]), ice=int(row["ice"]),
            nsoil=int(row["nsoil"]))
        base.append(((p, options), kwargs))
    base.extend(BATCHES.COLLECTED["radiation"][0])
    return _cycled(base, N_COLUMNS)


COLUMNS = {
    "thermoprop": _thermoprop_columns(),
    "tsnosoi": _tsnosoi_columns(),
    "phasechange": _phasechange_columns(),
    "radiation": _radiation_columns(),
}


# ---------------------------------------------------------------------------
# the slabs -- an independent restatement of each leaf's field widths
# ---------------------------------------------------------------------------
# Restated rather than imported from the module under test on purpose: a gate
# that reads its expectations out of the thing it is gating measures nothing.

FLOAT_FIELDS = {
    "thermoprop": {"dzsnso": NLAY, "snice": NSNOW, "snliq": NSNOW,
                   "smc": NSOIL, "sh2o": NSOIL, "stc": NLAY,
                   "snowh": 1, "dt": 1,
                   "smcmax": NSOIL, "csoil": 1, "quartz": NSOIL},
    "tsnosoi": {"zsnso": NLAY, "stc": NLAY, "df": NLAY, "hcpct": NLAY,
                "tbot": 1, "ssoil": 1, "dt": 1, "snowh": 1, "zbot": 1},
    "phasechange": {"fact": NLAY, "dzsnso": NLAY, "stc": NLAY,
                    "snice": NSNOW, "snliq": NSNOW,
                    "smc": NSOIL, "sh2o": NSOIL,
                    "sneqv": 1, "snowh": 1, "dt": 1,
                    "smcmax": NSOIL, "psisat": NSOIL, "bexp": NSOIL},
    "radiation": dict(
        [(name, 2 if name in _RADIATION_PAIRS else 1)
         for name in PARAMETER_SLOTS]
        + [(name, 1) for name in STATE_SLOTS]
        + [("smc", NSOIL), ("albold", 1), ("tauss", 1),
           ("solad", 2), ("solai", 2)]),
}

#: Integer fields, and where each column's value is read from.
INT_FIELDS = {
    "thermoprop": ("isnow", "ist"),
    "tsnosoi": ("isnow",),
    "phasechange": ("isnow", "ist"),
    "radiation": ("ist",),
}

#: Fields the per-column packer reads as a truth value, not as a number.
FLAG_FIELDS = {"thermoprop": ("urban_flag",)}


def _value(call, name, leaf):
    """One column's value for one field, from wherever the leaf keeps it."""
    args, kwargs = call
    if leaf == "radiation" and name in PARAMETER_SLOTS:
        return getattr(args[0], name)
    return kwargs[name]


def _host_fields(leaf) -> dict:
    """The whole-slab ``fields`` mapping, built on the host from the columns.

    The column loop lives *here*, in the fixture, which is the point: the
    module under test has none.
    """
    calls = COLUMNS[leaf]
    fields: dict[str, np.ndarray] = {}
    for name, width in FLOAT_FIELDS[leaf].items():
        if width == 1:
            block = np.array([np.float32(_value(call, name, leaf))
                              for call in calls], dtype=np.float32)
        else:
            block = np.array(
                [np.asarray(_value(call, name, leaf), dtype=np.float32)
                 for call in calls], dtype=np.float32)
        assert block.shape == ((len(calls),) if width == 1
                               else (len(calls), width)), (leaf, name)
        fields[name] = block
    for name in INT_FIELDS[leaf]:
        fields[name] = np.array([int(_value(call, name, leaf))
                                 for call in calls], dtype=np.int32)
    for name in FLAG_FIELDS.get(leaf, ()):
        fields[name] = np.array([bool(_value(call, name, leaf))
                                 for call in calls], dtype=bool)
    return fields


FIELDS = {leaf: _host_fields(leaf) for leaf in LEAVES}


def _device(fields: dict) -> dict:
    """The same mapping as CuPy slabs, which is what the module takes."""
    return {name: cp.asarray(block) for name, block in fields.items()}


# ---------------------------------------------------------------------------
# the two comparisons
# ---------------------------------------------------------------------------

def _differing(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Flat indices whose *bytes* differ -- NaN-safe, unlike ``==``."""
    g = np.ascontiguousarray(got).reshape(-1)
    w = np.ascontiguousarray(want).reshape(-1)
    if g.size == 0:
        return np.empty(0, dtype=np.intp)
    gb = g.view(np.uint8).reshape(g.size, g.itemsize)
    wb = w.view(np.uint8).reshape(w.size, w.itemsize)
    return np.flatnonzero((gb != wb).any(axis=1))


def _assert_bitwise(got, want, what: str) -> None:
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.dtype == want.dtype, (
        f"{what}: dtype {got.dtype} is not {want.dtype}")
    assert got.shape == want.shape, (
        f"{what}: shape {got.shape} is not {want.shape}")
    bad = _differing(got, want)
    assert bad.size == 0, (
        f"{what}: {bad.size} of {got.size} words differ, first at flat index "
        f"{int(bad[0])} -- got {got.reshape(-1)[bad[0]]!r}, want "
        f"{want.reshape(-1)[bad[0]]!r}")


PACK_PER_COLUMN = {
    "thermoprop": pack_thermoprop_calls,
    "tsnosoi": pack_tsnosoi_calls,
    "phasechange": pack_phasechange_calls,
    "radiation": pack_radiation_calls,
}

PACK_SLAB = {
    "thermoprop": slab.pack_thermoprop_slab,
    "tsnosoi": slab.pack_tsnosoi_slab,
    "phasechange": slab.pack_phasechange_slab,
    "radiation": slab.pack_radiation_slab,
}

EVALUATE_SLAB = {
    "thermoprop": slab.evaluate_thermoprop_slab,
    "tsnosoi": slab.evaluate_tsnosoi_slab,
    "phasechange": slab.evaluate_phasechange_slab,
    "radiation": slab.evaluate_radiation_slab,
}

#: How each per-column result becomes named slabs.  Written out rather than
#: introspected so that a leaf which silently reorders its return tuple fails.
_THERMOPROP_RETURN = ("df", "hcpct", "snicev", "snliqv", "epore", "fact")
_PHASECHANGE_RETURN = ("stc", "snice", "snliq", "sneqv", "snowh", "smc",
                       "sh2o", "qmelt", "ponding", "imelt")


def _stack(rows) -> np.ndarray:
    """One per-column output, stacked, keeping the per-column dtype."""
    first = rows[0]
    if isinstance(first, np.ndarray) and first.ndim:
        return np.stack(rows)
    return np.asarray(rows, dtype=np.asarray(first).dtype)


def _reference(leaf, calls) -> dict:
    """The per-column path's answer, as one slab per output name."""
    if leaf == "thermoprop":
        rows = evaluate_thermoprop_calls(calls)
        return {name: _stack([row[k] for row in rows])
                for k, name in enumerate(_THERMOPROP_RETURN)}
    if leaf == "tsnosoi":
        rows = evaluate_tsnosoi_calls(calls)
        return {"stc": _stack([row[0] for row in rows]),
                "eflxb": _stack([row[1] for row in rows])}
    if leaf == "phasechange":
        rows = evaluate_phasechange_calls(calls)
        return {name: _stack([row[k] for row in rows])
                for k, name in enumerate(_PHASECHANGE_RETURN)}
    rows = evaluate_radiation_calls(calls)
    out = {name: _stack([row[name] for row in rows])
           for name in OUTPUT_NAMES + ("bgap", "wgap", "albold", "tauss")}
    for name in ("albsnd", "albsni"):
        out[name] = np.asarray([list(row[name]) for row in rows],
                               dtype=np.float32)
    return out


def _packing_check(leaf: str, fields: dict) -> None:
    """The slab packing is the per-column packing, byte for byte."""
    want = PACK_PER_COLUMN[leaf](COLUMNS[leaf])
    got = PACK_SLAB[leaf](_device(fields), N_COLUMNS)
    assert len(got) == len(want) == 2, leaf
    for block, g, w in zip(("float row", "integer row"), got, want):
        assert g.flags.c_contiguous, f"{leaf} packed {block} is not contiguous"
        _assert_bitwise(cp.asnumpy(g), w, f"{leaf} packed {block}")


def _evaluation_check(leaf: str, fields: dict) -> None:
    """The slab evaluation is the per-column evaluation, field by field."""
    want = _reference(leaf, COLUMNS[leaf])
    got = EVALUATE_SLAB[leaf](_device(fields), N_COLUMNS)
    assert set(got) == set(want), (
        f"{leaf}: returned names differ by {sorted(set(got) ^ set(want))}")
    for name in sorted(want):
        assert got[name].flags.c_contiguous, f"{leaf}.{name} is not contiguous"
        _assert_bitwise(cp.asnumpy(got[name]), want[name], f"{leaf}.{name}")


# ---------------------------------------------------------------------------
# what the columns are
# ---------------------------------------------------------------------------

@requires_gpu
def test_the_columns_reach_the_branches_that_matter():
    """The fixture is heterogeneous, and that is asserted rather than hoped.

    A slab gate driven by 96 copies of one column would pass with the layer
    axis transposed, with ISNOW ignored, and with every branch but one dead.
    """
    for leaf in LEAVES:
        assert len(COLUMNS[leaf]) == N_COLUMNS >= 32, leaf
        assert len({id(call) for call in COLUMNS[leaf]}) >= 10, (
            f"{leaf}: fewer than ten distinct fixture columns")

    for leaf in ("thermoprop", "tsnosoi", "phasechange"):
        isnow = FIELDS[leaf]["isnow"]
        assert (isnow == 0).any(), f"{leaf}: no snow-free column"
        assert (isnow <= -2).any(), f"{leaf}: no multi-layer snow pack"

    # PHASECHANGE's branches: IMELT is 1 where WRF melts a layer and 2 where it
    # freezes one, so both values appearing is the branch coverage claim.
    imelt = _reference("phasechange", COLUMNS["phasechange"])["imelt"]
    assert (imelt == 1).any(), "no PHASECHANGE column melts a layer"
    assert (imelt == 2).any(), "no PHASECHANGE column freezes a layer"
    sh2o = FIELDS["phasechange"]["sh2o"]
    smc = FIELDS["phasechange"]["smc"]
    assert (sh2o < smc).any(), "no PHASECHANGE column carries frozen soil"
    assert (sh2o == smc).any(), "no PHASECHANGE column is unfrozen"

    cosz = FIELDS["radiation"]["cosz"]
    assert (cosz > 0).any() and (cosz <= 0).any(), "RADIATION is all one sky"
    ist = FIELDS["radiation"]["ist"]
    assert set(np.unique(ist)) >= {1, 2}, "RADIATION has no lake column"


# ---------------------------------------------------------------------------
# gate 1: the packing
# ---------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("leaf", LEAVES)
def test_the_slab_packing_is_byte_identical(leaf):
    _packing_check(leaf, FIELDS[leaf])


# ---------------------------------------------------------------------------
# gate 2: the evaluation
# ---------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("leaf", LEAVES)
def test_the_slab_evaluation_is_byte_identical(leaf):
    _evaluation_check(leaf, FIELDS[leaf])


# ---------------------------------------------------------------------------
# the negative controls
# ---------------------------------------------------------------------------
#: Two *adjacent* slots per leaf, of equal width, carrying different numbers
#: and reaching different terms.  SOLAD/SOLAI is the transposition
#: ``tests/test_noahmp_leaf_batches_cuda.py`` already treats as RADIATION's
#: most likely packing defect; the other three are the same shape of mistake.
TRANSPOSE = {
    "thermoprop": ("snowh", "dt"),          # slots 28, 29
    "tsnosoi": ("tbot", "ssoil"),           # slots 35, 36
    "phasechange": ("smcmax", "psisat"),    # slots 45..49, 49..53
    "radiation": ("solad", "solai"),        # slots 52..54, 54..56
}

#: One field per leaf that every column reads and that reaches an output on
#: every column, so a one-column shift cannot hide in a dead branch.
ROLL = {
    "thermoprop": "stc",
    "tsnosoi": "df",
    "phasechange": "stc",
    "radiation": "albold",
}

#: An *unpacking* defect per leaf, overriding the module's own output layout.
#: Both corruptions above are on the way in, and a gate that only ever saw
#: those would pass a slab that read its answer out of the wrong slice.  Two
#: adjacent outputs of equal width are exchanged; TSNOSOI has no such pair, so
#: its scalar is slid one slot back onto STC's bottom layer, which is the
#: off-by-one this seam produces.
#:
#: RADIATION's ``albsnd``/``albsni`` was the first choice here and it does
#: *not* work: under ``opt_alb=2`` SNOWALB_CLASS assigns the same ALB to both
#: bands of both streams, so the four snow albedos are bit-identical on every
#: column and exchanging their slices is not observable.  ``sav``/``sag`` --
#: the canopy-absorbed and ground-absorbed shortwave -- are adjacent and do
#: differ.  The vacuity guard below is what turned that from a passing test
#: into a visible one.
OUTPUT_DEFECT = {
    "thermoprop": ("snicev", "snliqv"),
    "tsnosoi": ("eflxb", NLAY - 1),
    "phasechange": ("snice", "snliq"),
    "radiation": ("sav", "sag"),
}

OUTPUT_LAYOUT = {
    "thermoprop": slab.THERMOPROP_OUT,
    "tsnosoi": slab.TSNOSOI_OUT,
    "phasechange": slab.PHASECHANGE_OUT,
    "radiation": slab.RADIATION_OUT,
}


@requires_gpu
@pytest.mark.parametrize("leaf", LEAVES)
def test_transposing_two_adjacent_slots_is_rejected(leaf):
    """Exchange two adjacent slots and both comparisons must refuse."""
    first, second = TRANSPOSE[leaf]
    fields = dict(FIELDS[leaf])
    assert _differing(fields[first], fields[second]).size, (
        f"{leaf}: {first} and {second} carry the same bytes, so exchanging "
        "them is not a defect and this control proves nothing")
    fields[first], fields[second] = fields[second], fields[first]

    with pytest.raises(AssertionError) as packing:
        _packing_check(leaf, fields)
    assert "packed" in str(packing.value), packing.value

    with pytest.raises(AssertionError) as evaluation:
        _evaluation_check(leaf, fields)
    assert leaf in str(evaluation.value), evaluation.value


@requires_gpu
@pytest.mark.parametrize("leaf", LEAVES)
def test_rolling_one_field_onto_its_neighbour_is_rejected(leaf):
    """Shift one field one column and both comparisons must refuse.

    This is the defect a batched seam actually produces -- a field that is
    right in every column but the wrong column's -- and it is invisible to any
    comparison that only checks shapes, dtypes or aggregate statistics.
    """
    name = ROLL[leaf]
    fields = dict(FIELDS[leaf])
    rolled = np.roll(fields[name], 1, axis=0)
    assert _differing(rolled, fields[name]).size, (
        f"{leaf}: rolling {name} by one column changed no byte, so every "
        "column carries the same value and this control proves nothing")
    fields[name] = rolled

    with pytest.raises(AssertionError) as packing:
        _packing_check(leaf, fields)
    assert "packed" in str(packing.value), packing.value

    with pytest.raises(AssertionError) as evaluation:
        _evaluation_check(leaf, fields)
    assert leaf in str(evaluation.value), evaluation.value


@requires_gpu
@pytest.mark.parametrize("leaf", LEAVES)
def test_reading_an_output_out_of_the_wrong_slice_is_rejected(
        leaf, monkeypatch):
    """Move one output's slice inside the module and the gate must refuse.

    The two controls above are both on the way *in*.  This one is on the way
    out, where a slab's other characteristic defect lives: the row is packed
    perfectly, the kernel is right, and the answer is read out of the
    neighbouring slice.
    """
    layout = OUTPUT_LAYOUT[leaf]
    defect = OUTPUT_DEFECT[leaf]
    want = _reference(leaf, COLUMNS[leaf])
    if isinstance(defect[1], str):
        first, second = defect
        assert _differing(want[first], want[second]).size, (
            f"{leaf}: {first} and {second} answer with identical bytes on "
            "every column, so exchanging their slices is not a defect and "
            "this control proves nothing")
        was_first, was_second = layout[first], layout[second]
        monkeypatch.setitem(layout, first, was_second)
        monkeypatch.setitem(layout, second, was_first)
    else:
        name, start = defect
        # The slid-onto slot is inside STC, which is the only multi-slot
        # output TSNOSOI has; comparing against it is the vacuity guard.
        assert _differing(want["stc"][:, start], want[name]).size, (
            f"{leaf}: sliding {name} onto slot {start} reads bytes it already "
            "answers with, so this control proves nothing")
        monkeypatch.setitem(layout, name, (start, layout[name][1]))

    with pytest.raises(AssertionError) as evaluation:
        _evaluation_check(leaf, FIELDS[leaf])
    assert leaf in str(evaluation.value), evaluation.value


# ---------------------------------------------------------------------------
# the contract around the edges
# ---------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("leaf", LEAVES)
def test_a_wrong_width_is_refused_rather_than_broadcast(leaf):
    """A layer vector of the wrong length must raise, not broadcast.

    NSOIL is 4 and NSNOW is 3, so a soil vector handed to a snow destination
    would broadcast in some spellings and produce a wrong forecast rather than
    an exception.
    """
    name, width = next((n, w) for n, w in FLOAT_FIELDS[leaf].items() if w > 1)
    fields = dict(FIELDS[leaf])
    fields[name] = fields[name][:, : width - 1]
    with pytest.raises(ValueError, match=name):
        PACK_SLAB[leaf](_device(fields), N_COLUMNS)


@requires_gpu
@pytest.mark.parametrize("leaf", LEAVES)
def test_a_missing_field_names_itself(leaf):
    fields = dict(FIELDS[leaf])
    name = next(iter(FLOAT_FIELDS[leaf]))
    del fields[name]
    with pytest.raises(KeyError, match=name):
        PACK_SLAB[leaf](_device(fields), N_COLUMNS)


@requires_gpu
@pytest.mark.parametrize("leaf", LEAVES)
def test_an_empty_slab_launches_nothing_and_returns_empty(leaf):
    """Zero land columns is a real tile, and launches no zero-block grid."""
    empty = {name: block[:0] for name, block in FIELDS[leaf].items()}
    out = EVALUATE_SLAB[leaf](_device(empty), 0)
    assert out, leaf
    for name, block in out.items():
        assert block.shape[0] == 0, (leaf, name, block.shape)
