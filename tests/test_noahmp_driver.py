"""Bitwise gate for the Noah-MP driver cold start.

Replays ``gpuwm/data/noahmp/oracle/noahmp-driver.csv`` through
:mod:`gpuwm.core.noahmp_driver` and requires ``max_ulp 0`` on every emitted
column of every case, then re-runs the fixture's own structural validator so
the CSV cannot drift underneath the port.

Four things beyond the replay:

1. ``SNOW_INIT`` leaves ``ZSNSOXY``'s snow slots unwritten when ``ISNOW == 0``
   and ``NOAHMP_INIT`` leaves ``cropcat`` unwritten on a vegetated column when
   ``iopt_crop == 0``.  Both are ``INTENT(OUT)`` dummies that a live path never
   assigns.  ``test_intent_out_slots_are_pass_through`` proves the port's
   pass-through is the fixture's behaviour rather than a convenience.
2. Fourteen NOAHMP_INIT arguments are inert under the pinned identity and are
   not arguments of the port at all.  ``test_inert_arguments_never_move``
   requires entry to equal exit on every one of them in the fixture, which is
   what makes the omission checkable.
3. The land-use category identity the branches key on is read from the
   byte-pinned ``MPTABLE.TBL`` through :mod:`gpuwm.core.noahmp`, and
   ``test_land_use_identity_matches_the_module`` requires it to agree with the
   ``table_identity`` rows the harness read out of ``NOAHMP_TABLES`` itself.
   Neither side is allowed to be the sole authority.
4. The fixture is byte-pinned as well as validated, because a fixture that can
   be regenerated silently is not a gate.
"""

from __future__ import annotations

import csv
import hashlib
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.noahmp import DEFAULT_VEGETATION_DATASET, load_noahmp_parameters
from gpuwm.core.noahmp_driver import (
    LandUseIdentity,
    noahmp_init_column,
    snow_init_column,
)

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "gpuwm" / "data" / "noahmp" / "oracle" / "noahmp-driver.csv"
VALIDATOR = (REPO / "tools" / "noahmp_wrf461_oracle"
             / "validate_driver_oracle.py")

PINNED = {
    "noahmp-driver.csv":
        "e7f702e90b6df4c0a81e6eb059080bf0f38a2226e929781c7379550ee2323626",
}

NSNOW = 3

# NOAHMP_INIT arguments no live statement writes under the pinned identity.
# The port takes none of them; the fixture must show every one standing still.
INERT = ("xlat", "tmn", "croptype", "irnumsi", "irnummi", "irnumfi",
         "irwatsi", "irwatmi", "irwatfi", "ireloss", "irsivol", "irmivol",
         "irfivol", "irrsplh")


def _bits(x) -> str:
    return struct.pack(">f", np.float32(x)).hex().upper()


def _load():
    table = defaultdict(dict)
    with FIXTURE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["field"], int(row["index"]))
            value = (int(row["value"]) if row["dtype"] == "int"
                     else np.float32(
                         struct.unpack(">f", bytes.fromhex(row["bits"]))[0]))
            table[(row["leaf"], row["case"], row["stage"])][key] = value
    return table


TABLE = _load()
SI_CASES = sorted(c for (leaf, c, stage) in TABLE
                  if leaf == "snow_init" and stage == "input")
NI_CASES = sorted(c for (leaf, c, stage) in TABLE
                  if leaf == "noahmp_init" and stage == "input"
                  and c != "table_identity")


def _stack(slot, field, lo, hi):
    return np.array([slot[(field, k)] for k in range(lo, hi + 1)],
                    dtype=np.float32)


@pytest.fixture(scope="module")
def identity() -> LandUseIdentity:
    bundle = load_noahmp_parameters()
    _, parameters = bundle.vegetation_groups(DEFAULT_VEGETATION_DATASET)
    return LandUseIdentity(
        isice=parameters.scalar("ISICE"),
        isurban=parameters.scalar("ISURBAN"),
        iswater=parameters.scalar("ISWATER"),
        isbarren=parameters.scalar("ISBARREN"),
        natural=parameters.scalar("NATURAL"),
        lcz=tuple(parameters.scalar(f"LCZ_{n}") for n in range(1, 12)),
    )


@pytest.fixture(scope="module")
def sla_table():
    bundle = load_noahmp_parameters()
    _, parameters = bundle.vegetation_groups(DEFAULT_VEGETATION_DATASET)
    return parameters.values["SLA"]


# ---------------------------------------------------------------------------
# Fixture identity
# ---------------------------------------------------------------------------

def test_fixture_is_byte_pinned():
    for name, digest in PINNED.items():
        path = FIXTURE.parent / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


def test_fixture_still_validates():
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--fixture", str(FIXTURE)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_land_use_identity_matches_the_module(identity):
    probe = TABLE[("noahmp_init", "table_identity", "probe")]
    assert identity.isice == probe[("isice_table", 0)]
    assert identity.isurban == probe[("isurban_table", 0)]
    assert identity.iswater == probe[("iswater_table", 0)]
    assert identity.isbarren == probe[("isbarren_table", 0)]
    assert identity.natural == probe[("natural_table", 0)]
    assert identity.lcz[0] == probe[("lcz_1_table", 0)]


# ---------------------------------------------------------------------------
# SNOW_INIT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", SI_CASES)
def test_snow_init_reproduces_the_fixture(case):
    inp = TABLE[("snow_init", case, "input")]
    out = TABLE[("snow_init", case, "output")]
    nsoil = inp[("nsoil", 0)]
    nsnow = inp[("nsnow", 0)]

    got = snow_init_column(
        nsnow, nsoil,
        _stack(inp, "zsoil", 1, nsoil),
        inp[("swe", 0)], inp[("tgxy", 0)], inp[("snodep", 0)],
        _stack(inp, "zsnsoxy", -nsnow + 1, nsoil),
        _stack(inp, "tsnoxy", -nsnow + 1, 0),
        _stack(inp, "snicexy", -nsnow + 1, 0),
        _stack(inp, "snliqxy", -nsnow + 1, 0),
    )

    assert got.isnow == out[("isnowxy", 0)]
    for field, arr, lo in (("zsnsoxy", got.zsnso, -nsnow + 1),
                           ("tsnoxy", got.tsno, -nsnow + 1),
                           ("snicexy", got.snice, -nsnow + 1),
                           ("snliqxy", got.snliq, -nsnow + 1)):
        hi = nsoil if field == "zsnsoxy" else 0
        for k in range(lo, hi + 1):
            want = out[(field, k)]
            mine = arr[k - lo]
            assert _bits(mine) == _bits(want), (
                f"{case} {field}[{k}]: {_bits(mine)} != {_bits(want)}")


def test_snow_init_covers_every_ladder_branch():
    seen = {TABLE[("snow_init", c, "output")][("isnowxy", 0)]
            for c in SI_CASES}
    assert seen == {0, -1, -2, -3}
    depths = sorted({float(TABLE[("snow_init", c, "input")][("snodep", 0)])
                     for c in SI_CASES})
    for edge in (0.025, 0.05, 0.10, 0.25, 0.45):
        assert any(abs(d - np.float32(edge)) == 0.0 for d in depths), edge


def test_snow_init_exercises_more_than_one_soil_depth():
    stacks = {tuple(TABLE[("snow_init", c, "input")][("zsoil", k)]
                    for k in range(1, TABLE[("snow_init", c, "input")]
                                   [("nsoil", 0)] + 1))
              for c in SI_CASES}
    assert len(stacks) >= 2


# ---------------------------------------------------------------------------
# NOAHMP_INIT
# ---------------------------------------------------------------------------

def _run_case(case, identity, sla_table):
    inp = TABLE[("noahmp_init", case, "input")]
    nsoil = inp[("nsoil", 0)]
    vegtyp = inp[("ivgtyp", 0)]
    sla = inp.get(("sla_table", 0))
    return inp, noahmp_init_column(
        nsoil=nsoil,
        dzs=_stack(inp, "dzs", 1, nsoil),
        fndsnowh=inp[("fndsnowh", 0)] == 1,
        identity=identity,
        vegtyp=vegtyp,
        soiltyp=inp[("isltyp", 0)],
        xice=inp[("xice", 0)],
        tsk=inp[("tsk", 0)],
        lai=inp[("lai", 0)],
        bexp=inp[("bexp_table", 0)],
        smcmax=inp[("smcmax_table", 0)],
        psisat=inp[("psisat_table", 0)],
        # SLA_TABLE is only indexed on the vegetated branch (2187); the
        # harness omits the row for a category outside 1..NVEG, which can only
        # be an urban/LCZ column and therefore never reaches 2187.
        sla=sla if sla is not None else np.float32("nan"),
        sla_natural=np.float32(sla_table[identity.natural - 1]),
        snow=inp[("snow", 0)],
        snowh=inp[("snowh", 0)],
        tslb=_stack(inp, "tslb", 1, nsoil),
        smois=_stack(inp, "smois", 1, nsoil),
        zsnsoxy=_stack(inp, "zsnsoxy", -NSNOW + 1, nsoil),
        tsnoxy=_stack(inp, "tsnoxy", -NSNOW + 1, 0),
        snicexy=_stack(inp, "snicexy", -NSNOW + 1, 0),
        snliqxy=_stack(inp, "snliqxy", -NSNOW + 1, 0),
        cropcat=inp[("cropcat", 0)],
        nsnow=inp[("nsnow", 0)],
        iopt_run=inp[("iopt_run", 0)],
        iopt_crop=inp[("iopt_crop", 0)],
        iopt_irr=inp[("iopt_irr", 0)],
        iopt_irrm=inp[("iopt_irrm", 0)],
        sf_urban_physics=inp[("sf_urban_physics", 0)],
    )


_SCALARS = (
    ("snow", "snow"), ("snowh", "snowh"), ("canwat", "canwat"),
    ("tvxy", "tv"), ("tgxy", "tg"), ("canicexy", "canice"),
    ("canliqxy", "canliq"), ("eahxy", "eah"), ("tahxy", "tah"),
    ("cmxy", "cm"), ("chxy", "ch"), ("fwetxy", "fwet"),
    ("sneqvoxy", "sneqvo"), ("alboldxy", "albold"), ("qsnowxy", "qsnow"),
    ("qrainxy", "qrain"), ("wslakexy", "wslake"), ("zwtxy", "zwt"),
    ("waxy", "wa"), ("wtxy", "wt"), ("lai", "lai"), ("xsaixy", "xsai"),
    ("lfmassxy", "lfmass"), ("rtmassxy", "rtmass"), ("stmassxy", "stmass"),
    ("woodxy", "wood"), ("stblcpxy", "stblcp"), ("fastcpxy", "fastcp"),
    ("grainxy", "grain"), ("gddxy", "gdd"), ("t2mvxy", "t2mv"),
    ("t2mbxy", "t2mb"), ("chstarxy", "chstar"), ("qtdrain", "qtdrain"),
)


@pytest.mark.parametrize("case", NI_CASES)
def test_noahmp_init_reproduces_the_fixture(case, identity, sla_table):
    inp, got = _run_case(case, identity, sla_table)
    out = TABLE[("noahmp_init", case, "output")]
    nsoil = inp[("nsoil", 0)]

    assert got.isnow == out[("isnowxy", 0)], case
    assert got.cropcat == out[("cropcat", 0)], case
    for field, attr in _SCALARS:
        mine, want = getattr(got, attr), out[(field, 0)]
        assert _bits(mine) == _bits(want), (
            f"{case} {field}: {_bits(mine)} != {_bits(want)}")
    for field, arr in (("tslb", got.tslb), ("smois", got.smois),
                       ("sh2o", got.sh2o)):
        for k in range(1, nsoil + 1):
            want = out[(field, k)]
            assert _bits(arr[k - 1]) == _bits(want), (
                f"{case} {field}[{k}]: {_bits(arr[k - 1])} != {_bits(want)}")
    for k in range(-NSNOW + 1, nsoil + 1):
        want = out[("zsnsoxy", k)]
        assert _bits(got.zsnso[k + NSNOW - 1]) == _bits(want), (
            f"{case} zsnsoxy[{k}]")
    for field, arr in (("tsnoxy", got.tsno), ("snicexy", got.snice),
                       ("snliqxy", got.snliq)):
        for k in range(-NSNOW + 1, 1):
            want = out[(field, k)]
            assert _bits(arr[k + NSNOW - 1]) == _bits(want), (
                f"{case} {field}[{k}]")


def test_inert_arguments_never_move():
    for case in NI_CASES:
        inp = TABLE[("noahmp_init", case, "input")]
        out = TABLE[("noahmp_init", case, "output")]
        moved = [k for k in inp if k[0] in INERT and out[k] != inp[k]]
        assert not moved, f"{case}: {moved}"


def test_inert_arguments_are_driven_non_zero():
    """An always-zero inert slot proves nothing; require them to vary."""
    values = defaultdict(set)
    for case in NI_CASES:
        for key, value in TABLE[("noahmp_init", case, "input")].items():
            if key[0] in INERT:
                values[key].add(float(value))
    assert values, "no inert slot is emitted at all"
    for key, seen in values.items():
        assert seen != {0.0}, f"{key} is zero in every case"
        assert len(seen) > 1, f"{key} never varies"


def test_intent_out_slots_are_pass_through():
    """The two dummies a live path never assigns must survive the call."""
    snow_slots = 0
    for case in SI_CASES:
        inp = TABLE[("snow_init", case, "input")]
        out = TABLE[("snow_init", case, "output")]
        isnow = out[("isnowxy", 0)]
        nsnow = inp[("nsnow", 0)]
        for k in range(-nsnow + 1, isnow + 1):
            assert out[("zsnsoxy", k)] == inp[("zsnsoxy", k)], (case, k)
            snow_slots += 1
    assert snow_slots > 0, "no case leaves a ZSNSOXY snow slot unwritten"

    vegetated = 0
    for case in NI_CASES:
        inp = TABLE[("noahmp_init", case, "input")]
        out = TABLE[("noahmp_init", case, "output")]
        if out[("cropcat", 0)] != 0:
            assert out[("cropcat", 0)] == inp[("cropcat", 0)], case
            vegetated += 1
    assert vegetated > 0, "no vegetated column leaves cropcat unwritten"


def test_degenerate_soil_guard_is_equivalent_to_the_bexp_test(identity):
    """2092's three-way guard has no STAS category that needs all three.

    ``mutation_study_driver.py`` reports "2092 degenerate-parameter guard drops
    the PSISAT test" as a survivor, and this is why: over the pinned
    ``SOILPARM.TBL`` STAS table there is no category with ``BEXP > 0`` and
    ``SMCMAX > 0`` but ``PSISAT <= 0``, so under ``opt_soil=1`` -- where all
    three come from the table indexed by one ``ISLTYP`` -- the conjunction and
    its ``BEXP``-only weakening select the same branch on every soil type WRF
    can present.  The port keeps the full conjunction because the guard reads
    runtime values, but the fixture provably cannot discriminate it.
    """
    table = load_noahmp_parameters().soil["STAS"]
    bexp = table.column("BB")
    smcmax = table.column("MAXSMC")
    psisat = table.column("SATPSI")
    disagreeing = [i + 1 for i in range(len(bexp))
                   if (bexp[i] > 0.0 and smcmax[i] > 0.0 and psisat[i] > 0.0)
                   != (bexp[i] > 0.0)]
    assert not disagreeing, disagreeing


def test_sla_floor_is_unreachable_on_the_vegetated_branch(identity, sla_table):
    """2187's ``max(SLA_TABLE(VEGTYP), 1.0)`` never selects the 1.0.

    The other survivor the mutation study reports.  Every MODIS category whose
    SLA is below 1.0 is ISICE, ISBARREN or ISWATER, and all three take the
    zeroing branch at 2164 before 2187 is ever reached, so the floor is dead
    under this land-use dataset.  It is kept because the expression is WRF's.
    """
    bare = {identity.isice, identity.isbarren, identity.iswater,
            identity.isurban, *identity.lcz}
    low = [i + 1 for i, v in enumerate(sla_table)
           if v <= 1.0 and (i + 1) not in bare]
    assert not low, low


def test_unsupported_options_are_refused(identity, sla_table):
    inp = TABLE[("noahmp_init", NI_CASES[0], "input")]
    nsoil = inp[("nsoil", 0)]
    kwargs = dict(
        nsoil=nsoil, dzs=_stack(inp, "dzs", 1, nsoil), fndsnowh=True,
        identity=identity, vegtyp=inp[("ivgtyp", 0)],
        soiltyp=inp[("isltyp", 0)], xice=inp[("xice", 0)],
        tsk=inp[("tsk", 0)], lai=inp[("lai", 0)],
        bexp=inp[("bexp_table", 0)], smcmax=inp[("smcmax_table", 0)],
        psisat=inp[("psisat_table", 0)], sla=inp[("sla_table", 0)],
        snow=inp[("snow", 0)], snowh=inp[("snowh", 0)],
        tslb=_stack(inp, "tslb", 1, nsoil),
        smois=_stack(inp, "smois", 1, nsoil),
        zsnsoxy=_stack(inp, "zsnsoxy", -NSNOW + 1, nsoil),
        tsnoxy=_stack(inp, "tsnoxy", -NSNOW + 1, 0),
        snicexy=_stack(inp, "snicexy", -NSNOW + 1, 0),
        snliqxy=_stack(inp, "snliqxy", -NSNOW + 1, 0),
        cropcat=inp[("cropcat", 0)],
    )
    for bad in ({"iopt_run": 5}, {"iopt_crop": 1}, {"iopt_irr": 2},
                {"restart": True}, {"soiltyp": 0}):
        with pytest.raises(ValueError):
            noahmp_init_column(**{**kwargs, **bad})
