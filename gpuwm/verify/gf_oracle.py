"""Read the WRF v4.6.1 Grell-Freitas oracle fixture and measure the port on it.

The fixture is written by ``tools/gf_wrf461_oracle/build.sh`` from the
byte-frozen ``phys/module_cu_gf_*.F``.  It comes in two halves, because GF is
ported stage by stage and a single boundary cannot serve both purposes:

``gf-levels.csv`` / ``gf-surface.csv`` / ``gf-isolation.csv``
    The pinned WRF boundary.  ``run_cu_gf.F90`` drives ``GFDRV`` itself, so
    the mb conversions, the 1.e-8 moisture floors, the ``zo`` half-layer
    stack, ``omeg = -g*rho*w``, the ``mconv`` integral, the ``dhdt`` forcing
    and the ``cuten`` gating are all inside the fixture.  This is what the
    finished port is graded against.
``gf-stage-levels.csv`` / ``gf-stage-surface.csv`` / ``gf-stage-consistency.csv``
    The same cases decomposed.  ``run_cup_gf.F90`` replicates GFDRV's column
    preparation and calls ``CUP_gf_sh``, ``neg_check``, ``cup_gf``,
    ``neg_check`` directly, so ``ierr``, ``ierrc``, ``xmb``, ``kbcon``,
    ``k22``, ``jmin``, ``edt`` and the ten ``forcing`` slots -- all driver
    locals, invisible at the pinned boundary -- become comparable.  This is
    what a *failing* port is diagnosed against.

The decomposition proves itself: ``gf-stage-consistency.csv`` records, per
(case, dx, arm), how many words of GFDRV's output the stage path fails to
reproduce bitwise.  208 of 216 rows are exact.  On the 8 that are not
(:func:`stage_rows_to_distrust`) the GFDRV fixture is authoritative and the
stage fixture is not; see ``tools/gf_wrf461_oracle/README.md`` for why.

Axes: 18 cases x 6 grid spacings x 2 ``ishallow`` arms = 216 columns, each
with 40 levels.  ``dx`` is a fixture axis, not a constant, because GF's
``sig = (1-frh)^2`` factor is the whole point of the scheme.

Nothing in this module knows a tolerance.  It returns measurements; the gate
decides what to do with them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance

__all__ = [
    "GF_ORACLE_DIR",
    "GF_NZ",
    "GF_NCASE",
    "GF_NDX",
    "GfOracleFixture",
    "load_gf_oracle",
    "stage_rows_to_distrust",
    "measure_fields",
]

GF_ORACLE_DIR = Path(__file__).resolve().parents[1] / "data" / "gf" / "oracle"

GF_NZ = 40
GF_NCASE = 18
GF_NDX = 6
GF_NARM = 2


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    return [h.strip() for h in rows[0]], [r for r in rows[1:] if r]


def _f32_columns(header: list[str], rows: list[list[str]]) -> dict[str, np.ndarray]:
    """Every column as float32, except the ones that are plainly integers.

    gfortran writes the reals with ``ES24.16E3`` -- 16 significant digits,
    which round-trips float32 exactly -- so ``np.float32(text)`` recovers the
    word the module wrote rather than an approximation of it.
    """
    out: dict[str, np.ndarray] = {}
    for j, name in enumerate(header):
        raw = [r[j].strip() for r in rows]
        try:
            out[name] = np.array([int(v) for v in raw], dtype=np.int32)
        except ValueError:
            try:
                out[name] = np.array([np.float32(v) for v in raw], dtype=np.float32)
            except ValueError:
                out[name] = np.array(raw, dtype=object)
    return out


@dataclass(frozen=True)
class GfOracleFixture:
    """One loaded fixture.

    Per-level arrays are shaped ``(ncol, nz)`` and per-column arrays
    ``(ncol,)``, where ``ncol`` runs over (dx, arm, case) in the file's own
    order.  ``key`` carries the (case, idx, arm) triple for each column so a
    caller never has to re-derive the ordering.
    """

    levels: dict[str, np.ndarray]
    surface: dict[str, np.ndarray]
    stage_levels: dict[str, np.ndarray]
    stage_surface: dict[str, np.ndarray]
    consistency: dict[str, np.ndarray]
    key: np.ndarray
    shallow_levels: dict[str, np.ndarray]
    shallow_surface: dict[str, np.ndarray]
    shallow_consistency: dict[str, np.ndarray]

    @property
    def ncol(self) -> int:
        return int(self.key.shape[0])


def _column_key(surface: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack(
        [surface["case"], surface["idx"], surface["arm"]], axis=1
    ).astype(np.int32)


def _fold_levels(
    cols: dict[str, np.ndarray], key: np.ndarray, nz: int
) -> dict[str, np.ndarray]:
    """Reshape the flat level table into ``(ncol, nz)`` in surface-file order."""
    lk = np.stack([cols["case"], cols["idx"], cols["arm"]], axis=1).astype(np.int32)
    order = {tuple(int(v) for v in row): i for i, row in enumerate(key)}
    ncol = key.shape[0]
    out: dict[str, np.ndarray] = {}
    slot = np.empty(lk.shape[0], dtype=np.int64)
    kk = cols["k"].astype(np.int64) - 1
    for r in range(lk.shape[0]):
        slot[r] = order[tuple(int(v) for v in lk[r])] * nz + kk[r]
    for name, arr in cols.items():
        if name in ("case", "idx", "arm", "k") or arr.dtype == object:
            continue
        flat = np.zeros(ncol * nz, dtype=arr.dtype)
        flat[slot] = arr
        out[name] = flat.reshape(ncol, nz)
    return out


def _fold_by_case(
    cols: dict[str, np.ndarray], ncase: int, nz: int
) -> dict[str, np.ndarray]:
    """Reshape a (case, k) level table into ``(ncase, nz)``, case 1 at row 0."""
    slot = (cols["case"].astype(np.int64) - 1) * nz + (cols["k"].astype(np.int64) - 1)
    out: dict[str, np.ndarray] = {}
    for name, arr in cols.items():
        if name in ("case", "k") or arr.dtype == object:
            continue
        flat = np.zeros(ncase * nz, dtype=arr.dtype)
        flat[slot] = arr
        out[name] = flat.reshape(ncase, nz)
    return out


def load_gf_oracle(directory: Path | str | None = None) -> GfOracleFixture:
    """Load all six CSVs.  ``directory`` defaults to the packaged fixture."""
    root = Path(directory) if directory is not None else GF_ORACLE_DIR

    sh, sr = _read_csv(root / "gf-surface.csv")
    surface = _f32_columns(sh, sr)
    key = _column_key(surface)

    lh, lr = _read_csv(root / "gf-levels.csv")
    levels = _fold_levels(_f32_columns(lh, lr), key, GF_NZ)

    th, tr = _read_csv(root / "gf-stage-surface.csv")
    stage_surface = _f32_columns(th, tr)

    gh, gr = _read_csv(root / "gf-stage-levels.csv")
    stage_levels = _fold_levels(_f32_columns(gh, gr), key, GF_NZ)

    ch, cr = _read_csv(root / "gf-stage-consistency.csv")
    consistency = _f32_columns(ch, cr)

    # The shallow capture is keyed by CASE alone, because CUP_gf_sh takes no
    # dx and nothing GFDRV hands it depends on one.  That is a measurement,
    # not an assumption: gf-shallow-consistency.csv's ndiff_words_vs_dx1
    # column compares the module's own answer at every dx against its answer
    # at dx = 1000 m, and build.sh fails on any nonzero row.
    sh, sr = _read_csv(root / "gf-shallow-surface.csv")
    shallow_surface = _f32_columns(sh, sr)
    slh, slr = _read_csv(root / "gf-shallow-levels.csv")
    shallow_levels = _fold_by_case(_f32_columns(slh, slr), GF_NCASE, GF_NZ)
    sch, scr = _read_csv(root / "gf-shallow-consistency.csv")
    shallow_consistency = _f32_columns(sch, scr)

    if key.shape[0] != GF_NCASE * GF_NDX * GF_NARM:
        raise ValueError(
            f"expected {GF_NCASE * GF_NDX * GF_NARM} columns, got {key.shape[0]}"
        )
    return GfOracleFixture(
        levels=levels,
        surface=surface,
        stage_levels=stage_levels,
        stage_surface=stage_surface,
        consistency=consistency,
        key=key,
        shallow_levels=shallow_levels,
        shallow_surface=shallow_surface,
        shallow_consistency=shallow_consistency,
    )


def stage_rows_to_distrust(fixture: GfOracleFixture) -> np.ndarray:
    """Boolean mask over columns where the stage decomposition is not exact.

    These are the rows where ``run_cup_gf``'s reconstruction of GFDRV's output
    algebra differs from the driver's own answer.  The cause is measured, not
    guessed: GFDRV evaluates ``omeg``, ``mconv`` and ``dhdt`` with the GFS
    physcons constants, which are ``real(8)`` parameters initialised from
    ``real(4)`` literals, and no float32/float64 spelling of those three
    expressions reproduces the driver on every column.  The residual is a
    uniform 1-2 ULP shift in ``xmb``; no branch flips (``ktop``, ``kbcon`` and
    ``ierr`` agree on all 216 rows).

    A gate comparing a port against the STAGE fixture should skip these
    columns.  A gate comparing against the GFDRV fixture should not: the
    GFDRV fixture is WRF's own answer everywhere.
    """
    ck = np.stack(
        [
            fixture.consistency["case"],
            fixture.consistency["idx"],
            fixture.consistency["arm"],
        ],
        axis=1,
    ).astype(np.int32)
    bad = fixture.consistency["ndiff_words_vs_gfdrv"].astype(np.int64) != 0
    lookup = {tuple(int(v) for v in ck[r]): bool(bad[r]) for r in range(ck.shape[0])}
    return np.array(
        [lookup[tuple(int(v) for v in row)] for row in fixture.key], dtype=bool
    )


@dataclass(frozen=True)
class FieldMeasurement:
    name: str
    max_ulp: int
    n_compared: int
    n_mismatch: int
    worst_col: int
    worst_k: int


def measure_fields(
    got: dict[str, np.ndarray],
    want: dict[str, np.ndarray],
    names: list[str],
    mask: np.ndarray | None = None,
) -> list[FieldMeasurement]:
    """Max FP32 ULP per field, with the location of the worst lane.

    ``mask`` selects columns; ``None`` means all.  Fields may be ``(ncol,)`` or
    ``(ncol, nz)``.
    """
    out: list[FieldMeasurement] = []
    for name in names:
        a = np.asarray(got[name], dtype=np.float32)
        b = np.asarray(want[name], dtype=np.float32)
        if a.shape != b.shape:
            raise ValueError(f"{name}: {a.shape} vs {b.shape}")
        if mask is not None:
            a = a[mask]
            b = b[mask]
        dist = fp32_ulp_distance(a, b)
        finite = dist >= 0
        n_mismatch = int(np.count_nonzero(~finite))
        worst = int(np.argmax(np.where(finite, dist, -1)))
        flat = np.unravel_index(worst, a.shape)
        out.append(
            FieldMeasurement(
                name=name,
                max_ulp=int(dist[finite].max()) if finite.any() else -1,
                n_compared=int(a.size),
                n_mismatch=n_mismatch,
                worst_col=int(flat[0]),
                worst_k=int(flat[1]) + 1 if a.ndim > 1 else 0,
            )
        )
    return out
