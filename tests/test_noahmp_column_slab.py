"""The assembled Noah-MP column slab against the scalar authority, bitwise.

:mod:`gpuwm.core.noahmp_column_slab` owns no arithmetic -- every value it
returns is produced by a slab module that carries its own bitwise gate.  What
it owns is the *hand-offs*, and a wrong hand-off (BARE_FLUX reading the entry
QSFC instead of VEGE_FLUX's, PHASECHANGE reading the entry STC instead of
TSNOSOI's, WATER reading ENERGY's DZSNSO instead of the prefix's) produces a
plausible number that every per-leaf oracle in the tree still blesses.  The
only gate that can see it is the whole call: :func:`gpuwm.core.noahmp_sflx
.sflx` driven per column on the host -- the routine the unmodified-WRF
whole-column fixture pins -- against :func:`evaluate_sflx_slab` on the same
columns, **every output field, bitwise**.

The columns are the four ``noahmp-sflx.csv`` cases crossed with forcing
variants, so the slab carries vegetated and bare tiles, day and night, deep
snowpack and none, frozen and unfrozen soil **in one call** -- the mixed
population a real nest hands the orchestration, where a compaction defect
(VEGE_FLUX's gather/scatter) actually has room to misroute.

Two negative controls, because a comparison that has never failed proves
nothing: one input field rolled onto the neighbouring column, and two soil
parameter slabs transposed.  Both must be *visible* -- rejected by the field
comparison or by the balance aborts, either of which is a detection.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

import gpuwm.core.noahmp_sflx as sflx_module
from gpuwm.core.noahmp_column_slab import (OUTPUT_NAMES, evaluate_sflx_slab,
                                           parameter_fields, parameter_row)
from gpuwm.core.noahmp_sflx import NoahmpBalanceError, sflx
from gpuwm.core.noahmp_snow import SnowColumn

from test_noahmp_sflx import (CASES, NSNOW, NSOIL, SF, _column, _forcing,
                              _params, _vec)

_NFULL = NSNOW + NSOIL

#: Forcing variants crossed with the four fixture cases.  Chosen to stay on
#: the admitted identity (no crop, no irrigation, IST=1) while moving real
#: branches: wind moves UR's clamp and the flux leaves, temperature moves the
#: psychrometric split and the balance terms.
_VARIANTS = (
    {},
    {"uu": np.float32(2.5), "vv": np.float32(-1.75)},
    {"sfctmp_offset": np.float32(0.625)},
)


def _cases():
    """(case, overrides) pairs for every column of the slab, in order."""
    columns = []
    for case in CASES:
        for variant in _VARIANTS:
            x = SF[(case, "input")]
            overrides = dict(variant)
            offset = overrides.pop("sfctmp_offset", None)
            if offset is not None:
                overrides["sfctmp"] = np.float32(x[("sfctmp", 0)] + offset)
            columns.append((case, overrides))
    return columns


def _authority(columns):
    """Every column driven through the scalar ``sflx``, leaves on the host.

    Returns per-column dicts carrying the ``SflxResult``, the mutated
    ``SnowColumn``, and ERROR's two residuals, which ``SflxResult`` does not
    carry and which a port could compute wrongly forever without any
    whole-column comparison noticing.
    """
    records = []
    original_error = sflx_module.error
    residuals = []

    def spying_error(**kwargs):
        result = original_error(**kwargs)
        residuals.append((result.errsw, result.erreng))
        return result

    sflx_module.error = spying_error
    try:
        for case, overrides in columns:
            x = SF[(case, "input")]
            p = _params(x)
            col = _column(x)
            result = sflx(p, col, _vec(x, "smc", 1, NSOIL),
                          ficeold=_vec(x, "ficeold", -NSNOW + 1, 0),
                          wa=x[("wa", 0)], wslake=x[("wslake", 0)],
                          **_forcing(x, **overrides))
            errsw, erreng = residuals[-1]
            records.append({"case": case, "result": result, "col": col,
                            "errsw": errsw, "erreng": erreng})
    finally:
        sflx_module.error = original_error
    return records


def _slab_fields(columns):
    """The same inputs as one ``fields`` mapping over the column axis."""
    import cupy as cp

    n = len(columns)
    scalars: dict[str, list] = {}
    vectors: dict[str, list] = {}

    handles = []
    handle_index = []
    handle_key = {}

    for case, overrides in columns:
        x = SF[(case, "input")]
        p = _params(x)
        forcing = _forcing(x, **overrides)
        col = _column(x)

        key = (int(x[("vegtyp", 0)]), int(x[("soiltype", 0)]))
        if key not in handle_key:
            handle_key[key] = len(handles)
            handles.append(p)
        handle_index.append(handle_key[key])

        for name, value in forcing.items():
            if name in ("zsoil",):
                vectors.setdefault(name, []).append(np.asarray(value))
            elif name in ("lai", "sai", "pgs"):
                continue                     # ignored by the slab, as by pre
            else:
                scalars.setdefault(name, []).append(value)
        scalars.setdefault("wa", []).append(x[("wa", 0)])
        scalars.setdefault("wslake", []).append(x[("wslake", 0)])
        scalars.setdefault("acc_ssoil", []).append(np.float32(0.0))
        vectors.setdefault("ficeold", []).append(
            _vec(x, "ficeold", -NSNOW + 1, 0))
        vectors.setdefault("smc", []).append(_vec(x, "smc", 1, NSOIL))
        scalars.setdefault("isnow", []).append(col.isnow)
        scalars.setdefault("snowh", []).append(col.snowh)
        scalars.setdefault("sneqv", []).append(col.sneqv)
        for name in ("snice", "snliq", "stc", "zsnso", "sh2o"):
            vectors.setdefault(name, []).append(np.asarray(getattr(col, name)))

    fields = {}
    for name, values in scalars.items():
        dtype = (np.int32 if isinstance(values[0], (int, np.integer))
                 else np.float32)
        fields[name] = cp.asarray(np.asarray(values, dtype=dtype))
    for name, values in vectors.items():
        fields[name] = cp.asarray(np.asarray(values, dtype=np.float32))

    rows = [parameter_row(p) for p in handles]
    fields.update(parameter_fields(rows, np.asarray(handle_index)))
    assert len(fields["nroot"]) == n
    return fields, n


#: OUTPUT_NAMES split by where the authority's value lives.
_FROM_COLUMN = ("snowh", "sneqv", "isnow", "snice", "snliq", "stc", "zsnso",
                "sh2o")
_RESIDUALS = ("errsw", "erreng")


def _authority_value(record, name):
    if name in _FROM_COLUMN:
        return np.asarray(getattr(record["col"], name))
    if name in _RESIDUALS:
        return np.asarray(record[name])
    return np.asarray(getattr(record["result"], name))


def _differences(records, out):
    """Every (column, field) whose bits differ, with both values."""
    import cupy as cp

    differing = {}
    compared = 0
    for name in OUTPUT_NAMES:
        got_all = cp.asnumpy(out[name])
        for index, record in enumerate(records):
            want = _authority_value(record, name)
            got = got_all[index]
            want32 = np.asarray(want, dtype=got_all.dtype)
            compared += want32.size
            if want32.tobytes() != np.asarray(got).tobytes():
                differing[(index, record["case"], name)] = (got, want32)
    return differing, compared


@requires_gpu
def test_the_assembled_slab_reproduces_the_scalar_column_bitwise():
    columns = _cases()
    records = _authority(columns)
    fields, n = _slab_fields(columns)
    out = evaluate_sflx_slab(fields, n)

    differing, compared = _differences(records, out)
    # Exactly: 63 scalars plus SNICE/SNLIQ (3), STC/ZSNSO/HCPCT (7) and
    # SH2O/SMC (4) per column.  Computed rather than bounded, so a field that
    # stops being compared fails here instead of passing silently.
    layered = {"snice": NSNOW, "snliq": NSNOW, "stc": _NFULL,
               "zsnso": _NFULL, "hcpct": _NFULL, "sh2o": NSOIL, "smc": NSOIL}
    per_column = (len(OUTPUT_NAMES) - len(layered)) + sum(layered.values())
    assert compared == len(records) * per_column, (compared, per_column)
    assert differing == {}, (
        f"{len(differing)} of {compared} words differ between the assembled "
        f"slab and the scalar column: {dict(list(differing.items())[:8])}")


@requires_gpu
def test_the_slab_population_is_heterogeneous():
    """The gate above is only worth its wall time if one call mixes regimes.

    A slab of twelve near-identical columns would pass while exercising no
    compaction at all; this pins that both flux paths, both snow states and
    both times of day are present in the same call.
    """
    columns = _cases()
    records = _authority(columns)
    veg = [bool(r["result"].fveg > 0.0) and float(r["result"].rssun) >= 0.0
           for r in records]
    snow = [float(r["col"].sneqv) for r in records]
    cosz = [float(SF[(case, "input")][("cosz", 0)]) for case, _ in columns]
    fveg = [float(r["result"].fveg) for r in records]
    assert any(f > 0.0 for f in fveg) and any(f == 0.0 for f in fveg), fveg
    assert any(s > 0.0 for s in snow) and any(s == 0.0 for s in snow), snow
    assert any(c > 0.0 for c in cosz) and any(c <= 0.0 for c in cosz), cosz
    assert veg  # the population exists at all


@requires_gpu
def test_rolling_one_forcing_column_is_rejected():
    """One column's SFCTMP handed to its neighbour must be visible.

    Either the field comparison moves or a balance gate aborts; both count,
    and WATER's own ERRWAT abort firing first is the stronger rejection.
    """
    import cupy as cp

    columns = _cases()
    records = _authority(columns)
    fields, n = _slab_fields(columns)
    fields["sfctmp"] = cp.roll(fields["sfctmp"], 1)
    try:
        out = evaluate_sflx_slab(fields, n)
    except (NoahmpBalanceError, ValueError):
        return
    differing, _ = _differences(records, out)
    assert differing, (
        "rolling SFCTMP onto the neighbouring column changed nothing the "
        "whole-column comparison can see")


@requires_gpu
def test_transposing_two_parameter_slabs_is_rejected():
    """SMCMAX and SMCREF swapped must be visible the same way."""
    columns = _cases()
    records = _authority(columns)
    fields, n = _slab_fields(columns)
    fields["smcmax"], fields["smcref"] = fields["smcref"], fields["smcmax"]
    try:
        out = evaluate_sflx_slab(fields, n)
    except (NoahmpBalanceError, ValueError):
        return
    differing, _ = _differences(records, out)
    assert differing, (
        "transposing SMCMAX and SMCREF changed nothing the whole-column "
        "comparison can see")


def _tiled_fields(fields, n_source, n):
    """The slab inputs tiled to ``n`` columns, column ``k`` copying
    ``k % n_source``."""
    import cupy as cp

    reps = -(-n // n_source)
    out = {}
    for name, value in fields.items():
        if value.ndim == 1:
            out[name] = cp.ascontiguousarray(cp.tile(value, reps)[:n])
        else:
            out[name] = cp.ascontiguousarray(
                cp.tile(value, (reps, 1))[:n])
    return out


@requires_gpu
def test_the_chunk_width_is_bitwise_and_its_transient_is_priced():
    """One full SLAB_COLUMN_CHUNK, bitwise, with its VRAM ceiling measured.

    Two claims that belong to the same launch:

    * every column of a 65,536-wide slab reproduces its source column's
      scalar-authority answer bitwise -- the end-to-end gate at the exact
      width the runtime launches, not at 12; and
    * the CuPy pool growth of that launch stays under the
      ``SLAB_TRANSIENT_BYTES_PER_COLUMN`` ceiling preflight prices, and the
      ceiling stays within 2x of the measurement, so it can neither lie low
      nor rot into a number nobody can connect to the machine.
    """
    import cupy as cp

    from gpuwm.core.noahmp_runtime import (SLAB_COLUMN_CHUNK,
                                           SLAB_TRANSIENT_BYTES_PER_COLUMN)

    columns = _cases()
    records = _authority(columns)
    fields, n_source = _slab_fields(columns)

    n = SLAB_COLUMN_CHUNK
    priced = SLAB_COLUMN_CHUNK * SLAB_TRANSIENT_BYTES_PER_COLUMN
    assert priced <= 2**30, (
        "the priced Noah-MP slab transient crossed 1 GiB; against the "
        "29,500 MiB rail with MYNN resident this bound is a correctness "
        "bar, not a preference")

    tiled = _tiled_fields(fields, n_source, n)
    evaluate_sflx_slab(tiled, n)                       # warm: modules, plans

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    cp.cuda.Device().synchronize()
    before = pool.total_bytes()
    out = evaluate_sflx_slab(tiled, n)
    cp.cuda.Device().synchronize()
    grown = pool.total_bytes() - before
    per_column = grown / n
    assert grown <= priced, (
        f"one {n}-column slab call grew the pool by {grown / 2**20:.1f} MiB "
        f"({per_column:.0f} B/column), above the priced "
        f"{priced / 2**20:.1f} MiB; re-measure and move "
        "SLAB_TRANSIENT_BYTES_PER_COLUMN, do not let preflight understate it")
    assert priced <= 2 * grown, (
        f"the ceiling is now {priced / max(grown, 1):.1f}x the measured "
        f"{per_column:.0f} B/column; tighten "
        "SLAB_TRANSIENT_BYTES_PER_COLUMN so preflight stops overstating it")

    differing = 0
    checked = 0
    for name in OUTPUT_NAMES:
        got = cp.asnumpy(out[name])
        want_rows = np.stack([
            np.asarray(_authority_value(records[k % n_source], name),
                       dtype=got.dtype)
            for k in range(n_source)])
        reps = (-(-n // n_source),) + (1,) * (want_rows.ndim - 1)
        want = np.tile(want_rows, reps)[:n]
        checked += want.size
        if got.tobytes() != want.tobytes():
            differing += int(
                (got.reshape(n, -1) != want.reshape(n, -1)).any(axis=1).sum())
    assert checked >= n * len(OUTPUT_NAMES)
    assert differing == 0, (
        f"{differing} of {n} columns differ from their source column's "
        "scalar authority at full chunk width")


@requires_gpu
def test_an_empty_slab_answers_an_empty_bundle():
    import cupy as cp

    columns = _cases()
    fields, _n = _slab_fields(columns)
    empty = {name: value[:0] for name, value in fields.items()}
    out = evaluate_sflx_slab(empty, 0)
    assert set(out) == set(OUTPUT_NAMES)
    for name, value in out.items():
        assert value.shape[0] == 0, (name, value.shape)
    assert out["isnow"].dtype == cp.int32
