"""RUC soil ingest against the unmodified WRF v4.6.1 init_soil_3_real.

The fixture ``gpuwm/data/ruc/oracle/soil_ingest.csv`` is not a mirror of this
transcription: it is the output of ``share/module_soil_pre.F``'s own
``init_soil_depth_3`` and ``init_soil_3_real``, compiled byte-unmodified
through WRF's own ``.F -> .f90`` pipeline by
``tools/ruc_soil_ingest_wrf461_oracle/build.sh`` and driven directly.

The negative controls come FIRST, deliberately.  A max_ulp-0 gate that has
never been observed to fail is not evidence, and this one passed on its first
run; the three mutations below are what make its silence mean something.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from gpuwm.ingest.ruc_soil import (
    RUC_LEVEL_DEPTHS_M,
    remap_soil_to_ruc_levels,
    ruc_soil_depths,
)
import gpuwm.ingest.ruc_soil as ruc_soil


ORACLE = (Path(__file__).resolve().parents[1]
          / "gpuwm" / "data" / "ruc" / "oracle" / "soil_ingest.csv")
NSOIL = 9

#: WRF produces numbers for these rows; gpuwm refuses them.  Each refusal is
#: bound to its oracle evidence by its own test below.
REVERSED_EXPERIMENT = "era5_layers_reversed_adj"
ZERO_MOISTURE_COLUMN = 10


def _rows():
    with ORACLE.open(newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines))


def _call_kwargs(row):
    nlev = int(row["nlev"])
    geometry = "levels" if row["flag_soil_levels"] == "1" else "layers"
    return dict(
        source_temperature=np.array(
            [np.float32(row[f"st_src_{k}"]) for k in range(1, nlev + 1)],
            dtype=np.float32)[:, None],
        source_moisture=np.array(
            [np.float32(row[f"sm_src_{k}"]) for k in range(1, nlev + 1)],
            dtype=np.float32)[:, None],
        source_levels_cm=np.array(
            [int(row[f"lev_cm_{k}"]) for k in range(1, nlev + 1)],
            dtype=np.int64),
        source_geometry=geometry,
        skin_temperature=np.array([np.float32(row["tsk_in"])], dtype=np.float32),
        deep_temperature=np.array([np.float32(row["tmn"])], dtype=np.float32),
        landmask=np.array([np.float32(row["landmask"])], dtype=np.float32),
        sea_surface_temperature=(
            np.array([np.float32(row["sst"])], dtype=np.float32)
            if row["flag_sst"] == "1" else None),
        num_soil_layers=NSOIL,
        moisture_adjustment=(
            row["flag_sm_adj"] == "1" and geometry == "layers"),
    )


def _expected(row):
    return (
        np.array([np.float32(row[f"tslb_{k}"]) for k in range(1, NSOIL + 1)],
                 dtype=np.float32),
        np.array([np.float32(row[f"smois_{k}"]) for k in range(1, NSOIL + 1)],
                 dtype=np.float32),
        np.float32(row["tsk_out"]),
    )


def _ulp(got, want):
    got = np.asarray(got, dtype=np.float32).view(np.int32).astype(np.int64)
    want = np.asarray(want, dtype=np.float32).view(np.int32).astype(np.int64)
    floor = np.int64(np.iinfo(np.int32).min)
    got = np.where(got < 0, floor - got, got)
    want = np.where(want < 0, floor - want, want)
    return np.abs(got - want)


def _scored_rows():
    for row in _rows():
        if row["experiment"] == REVERSED_EXPERIMENT:
            continue
        if int(row["col"]) == ZERO_MOISTURE_COLUMN \
                and row["flag_sm_adj"] == "1" \
                and row["flag_soil_levels"] != "1":
            continue
        yield row


def _worst_ulp():
    worst_t = worst_m = worst_k = 0
    scored = 0
    for row in _scored_rows():
        result = remap_soil_to_ruc_levels(**_call_kwargs(row))
        want_t, want_m, want_k = _expected(row)
        worst_t = max(worst_t, int(_ulp(result.soil_temperature[:, 0], want_t).max()))
        worst_m = max(worst_m, int(_ulp(result.soil_moisture[:, 0], want_m).max()))
        worst_k = max(worst_k, int(_ulp(result.skin_temperature[0], want_k).max()))
        scored += 1
    return scored, worst_t, worst_m, worst_k


# --------------------------------------------------------------------------
# Negative controls.  These run first because the parity gate below passed on
# its first execution, and a gate nobody has seen fail measures nothing.
# --------------------------------------------------------------------------

def test_sampling_the_source_at_layer_bottoms_is_caught():
    """The convention error the gate exists to catch.

    WRF's readers place a layer source at the layer MIDPOINT in whole
    centimetres (``share/module_optional_input.F:1949-1954`` forms
    ``(top+bottom)/2`` in INTEGER arithmetic), so ERA5's 0-7 cm layer is
    sampled at 3 cm.  Sampling it at the layer BOTTOM instead -- which is the
    convention ``gpuwm/ingest/soil.py``'s declarative linear remap uses for
    Noah -- is a plausible reading that produces a plausible soil column.
    """

    worst = 0
    for row in _scored_rows():
        if row["flag_soil_levels"] == "1":
            continue
        kwargs = _call_kwargs(row)
        midpoints = kwargs["source_levels_cm"]
        bottoms = np.array(
            [7, 28, 100, 289] if int(midpoints[0]) == 3 else [10, 40, 100, 200],
            dtype=np.int64)
        kwargs["source_levels_cm"] = bottoms
        result = remap_soil_to_ruc_levels(**kwargs)
        want_t, _want_m, _want_k = _expected(row)
        worst = max(worst, int(_ulp(result.soil_temperature[:, 0], want_t).max()))
    assert worst > 100000, (
        "layer-bottom sampling must be far from WRF's answer; the gate would "
        f"not distinguish the two conventions at {worst} ULP")


def test_treating_dzs_as_a_partition_is_caught(monkeypatch):
    """``init_soil_depth_3``'s ``dzs`` is not a partition of the column.

    Its index shift (:1174-1183) leaves ``dzs(1)`` and ``dzs(2)`` both
    covering [0, 0.005] m and ``sum(dzs) == 3.005``.  ``dzs`` is an input to
    the moisture adjustment, so "fixing" it moves the shipped soil moisture.
    """

    real = ruc_soil.ruc_soil_depths

    def partitioned(num_soil_layers=9):
        zs, dzs = real(num_soil_layers)
        dzs = dzs.copy()
        dzs[1] = np.float32(0.02)   # zs2(2) - zs2(1) with zs2(1) = 0.005
        return zs, dzs

    monkeypatch.setattr(ruc_soil, "ruc_soil_depths", partitioned)
    worst = 0
    for row in _scored_rows():
        if row["flag_sm_adj"] != "1" or row["flag_soil_levels"] == "1":
            continue
        result = remap_soil_to_ruc_levels(**_call_kwargs(row))
        _want_t, want_m, _want_k = _expected(row)
        worst = max(worst, int(_ulp(result.soil_moisture[:, 0], want_m).max()))
    assert worst > 1000, (
        f"a corrected dzs must move the adjusted moisture; measured {worst} ULP")


def test_float64_then_round_once_is_caught(monkeypatch):
    """FP64-then-round-once is a third function, neither FP32 nor exact."""

    real_bracket = ruc_soil._bracket

    def wide(samples, depths, targets, label):
        out = np.empty((targets.size,) + samples.shape[1:], dtype=np.float32)
        wide_samples = samples.astype(np.float64)
        wide_depths = depths.astype(np.float64)
        for want, target in enumerate(targets):
            have = real_bracket(target, depths)
            assert have >= 0, label
            point = np.float64(target)
            out[want] = (
                (wide_samples[have] * (wide_depths[have + 1] - point)
                 + wide_samples[have + 1] * (point - wide_depths[have]))
                / (wide_depths[have + 1] - wide_depths[have])
            ).astype(np.float32)
        return out

    monkeypatch.setattr(ruc_soil, "_interpolate", wide)
    worst = 0
    for row in _scored_rows():
        result = remap_soil_to_ruc_levels(**_call_kwargs(row))
        want_t, want_m, _want_k = _expected(row)
        worst = max(worst, int(_ulp(result.soil_temperature[:, 0], want_t).max()))
        worst = max(worst, int(_ulp(result.soil_moisture[:, 0], want_m).max()))
    assert worst > 0, (
        "float64-then-round-once must be distinguishable from WRF's float32; "
        "if it is not, this gate cannot tell the two apart")


# --------------------------------------------------------------------------
# The gate.
# --------------------------------------------------------------------------

def test_ruc_soil_ingest_is_bit_identical_to_wrf():
    scored, worst_t, worst_m, worst_k = _worst_ulp()
    assert scored == 118, f"expected 118 scored oracle rows, got {scored}"
    assert (worst_t, worst_m, worst_k) == (0, 0, 0), (
        f"TSLB {worst_t} ULP, SMOIS {worst_m} ULP, TSK {worst_k} ULP against "
        "WRF v4.6.1 init_soil_3_real")


def test_every_oracle_experiment_is_exercised():
    """No arm of the fixture is silently skipped."""

    seen = {row["experiment"] for row in _scored_rows()}
    assert seen == {
        "era5_layers_noadj_nosst", "era5_layers_adj_nosst",
        "era5_layers_noadj_sst", "noah_layers_adj_nosst",
        "ruc_levels_noadj_nosst", "ruc_levels_adj_sst",
    }
    assert {row["experiment"] for row in _rows()} - seen == {REVERSED_EXPERIMENT}


def test_level_table_and_depths_match_init_soil_depth_3():
    zs, dzs = ruc_soil_depths(NSOIL)
    assert tuple(float(v) for v in zs) == tuple(
        np.float32(v) for v in RUC_LEVEL_DEPTHS_M[NSOIL])
    # The header line of the oracle carries WRF's own zs/dzs.
    header = ORACLE.read_text().splitlines()[0]
    assert header.startswith("# zs_m=")
    zs_text, dzs_text = header.split(" dzs_m=")
    want_zs = [np.float32(v) for v in zs_text[len("# zs_m="):].split(";") if v]
    want_dzs = [np.float32(v) for v in dzs_text.split(";") if v]
    assert list(zs) == want_zs
    assert list(dzs) == want_dzs
    # The property a caller is most likely to assume, and it is false.
    assert float(dzs.sum()) != 3.0
    ruc_soil_depths(6)                       # WRF's other tabulated length
    with pytest.raises(ValueError, match="init_soil_depth_3 tabulates"):
        ruc_soil_depths(4)


# --------------------------------------------------------------------------
# The two refusals, each bound to the oracle rows that justify it.
# --------------------------------------------------------------------------

def test_unsorted_layer_source_is_refused_and_wrf_corrupts_it():
    reversed_rows = [row for row in _rows()
                     if row["experiment"] == REVERSED_EXPERIMENT]
    assert reversed_rows, "the fixture must carry the reversed-order arm"

    # What WRF does with it: init_soil_3_real's sort permutes st_input(1..n)
    # while a layer source occupies st_input(2..n+1), so uninitialised slots
    # are interpolated into the soil column.  Only land columns show it --
    # :2131-2157 overwrites every water column wholesale, which is exactly
    # why a corruption like this can hide in a plot.
    land_rows = [row for row in reversed_rows if float(row["landmask"]) > 0.5]
    assert len(land_rows) == 18
    corrupted = 0
    for row in land_rows:
        values = [float(row[f"tslb_{k}"]) for k in range(1, NSOIL + 1)]
        if any(value < 170.0 or value > 400.0 for value in values):
            corrupted += 1
    assert corrupted == len(land_rows), (
        "every reversed-order land column must show WRF producing a "
        f"nonphysical soil temperature; {corrupted}/{len(land_rows)} did")

    # What gpuwm does with it.
    for row in reversed_rows[:1]:
        with pytest.raises(ValueError, match="strictly increasing"):
            remap_soil_to_ruc_levels(**_call_kwargs(row))


def test_zero_moisture_land_column_is_refused_and_wrf_divides_by_zero():
    rows = [row for row in _rows()
            if int(row["col"]) == ZERO_MOISTURE_COLUMN
            and row["flag_sm_adj"] == "1"
            and row["flag_soil_levels"] != "1"
            and row["experiment"] != REVERSED_EXPERIMENT]
    assert rows, "the fixture must carry a zero-moisture land column"

    for row in rows:
        assert float(row["landmask"]) > 0.5
        assert all(float(row[f"sm_src_{k}"]) == 0.0
                   for k in range(1, int(row["nlev"]) + 1))
        # 0/0 inside MAX(0.02, .) -- gfortran 13.3.0 returns the 0.02.
        assert all(np.float32(row[f"smois_{k}"]) == np.float32(0.02)
                   for k in range(1, NSOIL))
        with pytest.raises(ValueError, match="divides by zero"):
            remap_soil_to_ruc_levels(**_call_kwargs(row))

    # The same column WITHOUT the adjustment is a normal, reproduced row.
    plain = [row for row in _rows()
             if int(row["col"]) == ZERO_MOISTURE_COLUMN
             and row["flag_sm_adj"] == "0"]
    assert plain
    for row in plain:
        result = remap_soil_to_ruc_levels(**_call_kwargs(row))
        _want_t, want_m, _want_k = _expected(row)
        assert int(_ulp(result.soil_moisture[:, 0], want_m).max()) == 0


def test_a_target_level_outside_the_source_is_an_error_not_a_clamp():
    """WRF leaves such a level UNSET; there is no value to reproduce."""

    shallow = np.array([3, 17], dtype=np.int64)
    with pytest.raises(ValueError, match="outside the source depths"):
        remap_soil_to_ruc_levels(
            source_temperature=np.array([[290.0], [291.0]], dtype=np.float32),
            source_moisture=np.array([[0.2], [0.25]], dtype=np.float32),
            source_levels_cm=shallow,
            source_geometry="levels",
            skin_temperature=np.array([295.0], dtype=np.float32),
            deep_temperature=np.array([288.0], dtype=np.float32),
            landmask=np.array([1.0], dtype=np.float32),
        )


def test_nonphysical_sst_over_water_is_refused_because_ruc_writes_it_to_tsk():
    """The one line by which RUC's arm differs from Noah's, and it is fatal.

    ``init_soil_3_real``:2131-2144 writes ``tsk(i,j) = sst(i,j)`` on every
    water column with no validity test.  ``init_soil_2_real``:1732-1754, the
    Noah arm, writes ``tslb``/``smois``/``sh2o`` over water and never touches
    ``tsk``.  So a gap in the SST analysis kills a RUC run and not a Noah one:
    stock ``real.exe`` aborts at ``module_initialize_real.F:3283`` with
    "grid%tsk unreasonable".

    Measured on this project's own production case: ``met_em.d01`` at
    ``sf_surface_physics=2`` writes ``wrfinput_d01``; the identical file at
    ``sf_surface_physics=3, num_soil_layers=9`` dies at (i,j) = (15,1), the
    first of 1,494 water columns whose SST is 0.
    """

    common = dict(
        source_temperature=np.full((4, 3), 290.0, dtype=np.float32),
        source_moisture=np.full((4, 3), 0.25, dtype=np.float32),
        source_levels_cm=np.array([3, 17, 64, 194], dtype=np.int64),
        source_geometry="layers",
        skin_temperature=np.array([293.0, 291.0, 292.0], dtype=np.float32),
        deep_temperature=np.array([288.0, 288.0, 288.0], dtype=np.float32),
        landmask=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="nonphysical sea_surface_temperature"):
        remap_soil_to_ruc_levels(
            sea_surface_temperature=np.array([290.0, 0.0, 291.0],
                                             dtype=np.float32),
            **common)

    # A land column with a nonsense SST is NOT refused: :2131-2144 never reads
    # SST on land, so refusing there would be a bound this oracle cannot
    # justify.
    result = remap_soil_to_ruc_levels(
        sea_surface_temperature=np.array([0.0, 290.0, 291.0], dtype=np.float32),
        **common)
    assert float(result.soil_temperature[0, 0]) == pytest.approx(293.0)
    assert float(result.skin_temperature[0]) == pytest.approx(293.0)
    assert float(result.skin_temperature[1]) == pytest.approx(290.0)


def test_moisture_adjustment_on_a_node_source_is_refused_not_ignored():
    row = next(row for row in _rows()
               if row["experiment"] == "ruc_levels_adj_sst")
    kwargs = _call_kwargs(row)
    assert kwargs["moisture_adjustment"] is False   # what the fixture scores
    kwargs["moisture_adjustment"] = True
    with pytest.raises(ValueError, match="only to a layer source"):
        remap_soil_to_ruc_levels(**kwargs)
