"""Read the WRF v4.6.1 Noah LSM oracle fixture and measure the port against it.

The fixture is written by ``tools/noah_wrf461_oracle/run_lsm.F90``, which calls
``lsm`` in the byte-unmodified ``phys/module_sf_noahdrv.F`` -- the same
subroutine ``gpuwm/core/kernels/noah.cu``'s ``noah_column`` transcribes, prep
and post-update included.  Four CSVs live in ``gpuwm/data/noah/oracle/``, one
per combination of the driver switches ``launch_noah`` exposes.

Every column of every row is either a word the harness placed in an input
array before the call or a word ``lsm`` placed in an output array during it.
The two exceptions are ``sfcprs`` and ``zlvl``, which WRF derives from
``p8w3d`` and ``dz8w`` and gpuwm takes as inputs; the harness recomputes them
in the same float32 arithmetic and writes them out, so they are inputs here
too.

Nothing in this module knows a tolerance.  It returns measurements;
``tests/test_noah_wrf461_parity.py`` decides what to do with them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance

__all__ = [
    "NOAH_ORACLE_DIR",
    "NOAH_ORACLE_FILES",
    "NoahOracleFixture",
    "load_noah_oracle",
    "noah_port_outputs",
    "measure_noah_parity",
]

#: Where the packaged fixture lives.
NOAH_ORACLE_DIR = Path(__file__).resolve().parents[1] / "data" / "noah" / "oracle"

#: fixture file -> the driver switches ``run_lsm.F90`` was invoked with.
#: These are the four knobs :func:`gpuwm.core.noah.launch_noah` also takes;
#: ``build.sh`` runs the harness once per row, each in its own process so no
#: pass can inherit the previous pass's mutated state.
NOAH_ORACLE_FILES = {
    "noah-lsm.csv": dict(opt_thcnd=1, frpcpn=False,
                         usemonalb=False, rdlai2d=False),
    "noah-lsm-thcnd2.csv": dict(opt_thcnd=2, frpcpn=False,
                                usemonalb=False, rdlai2d=False),
    "noah-lsm-frpcpn.csv": dict(opt_thcnd=1, frpcpn=True,
                                usemonalb=False, rdlai2d=False),
    "noah-lsm-monalb.csv": dict(opt_thcnd=1, frpcpn=False,
                                usemonalb=True, rdlai2d=True),
}

#: ``run_lsm.F90:42-44``.  The harness fixes these rather than sweeping them,
#: so they are stated here with their source rather than read from the CSV.
FIXTURE_DT = 60.0
FIXTURE_ITIMESTEP = 2
#: ``init_soil_depth_2``'s dzs (``share/module_soil_pre.F:1138``), which is
#: also :data:`gpuwm.core.noah.SOIL_LAYER_THICKNESS_M`.
FIXTURE_DZS = (0.10, 0.30, 0.60, 1.00)
#: ``lsm`` is called with these, matching ``launch_noah``'s defaults.
FIXTURE_ISURBAN = 13
FIXTURE_ISICE = 15
FIXTURE_XICE_THRESHOLD = 0.5

#: launch_noah device-field name -> the fixture column holding its input value.
INPUT_COLUMNS = {
    "psfc": "psfc", "sfcprs": "sfcprs", "sfctmp": "sfctmp", "qv1": "qv1",
    "qgh": "qgh", "dz8w1": "dz8w1", "glw": "glw", "swdown": "swdown",
    "rainbl": "rainbl", "sr": "sr", "chs": "chs_in", "cqs2": "cqs2_in",
    "chs2": "chs2_in", "rib": "rib_in", "vegfra": "vegfra",
    "shdmin": "shdmin", "shdmax": "shdmax", "tmn": "tmn", "xland": "xland",
    "xice": "xice", "snoalb": "snoalb", "embck": "embck",
    "tsk": "tsk_in", "hfx": "hfx_in", "qfx": "qfx_in", "lh": "lh_in",
    "grdflx": "grdflx_in", "qsfc": "qsfc_in", "canwat": "canwat_in",
    "snow": "snow_in", "snowc": "snowc_in", "snowh": "snowh_in",
    "albedo": "albedo_in", "albbck": "albbck_in", "emiss": "emiss_in",
    "znt": "znt_in", "z0": "z0_in", "snotime": "snotime_in", "lai": "lai_in",
    "smstav": "smstav_in", "smstot": "smstot_in",
    "sfcrunoff": "sfcrunoff_in", "udrunoff": "udrunoff_in",
    "acsnow": "acsnow_in", "acsnom": "acsnom_in", "snopcx": "snopcx_in",
    "potevp": "potevp_in",
}

#: launch_noah device-field name -> the fixture column holding WRF's answer.
#: ``chs2`` is written by the driver too (``module_sf_noahdrv.F:1275`` sets it
#: to ``cqs2``) but the harness does not echo it back, so it is not measured
#: here; the kernel's copy of that line is checked by tests/test_noah.py.
OUTPUT_COLUMNS = {
    "tsk": "tsk", "hfx": "hfx", "qfx": "qfx", "lh": "lh",
    "grdflx": "grdflx", "qsfc": "qsfc", "canwat": "canwat", "snow": "snow",
    "snowc": "snowc", "snowh": "snowh", "albedo": "albedo",
    "albbck": "albbck", "emiss": "emiss", "znt": "znt", "z0": "z0",
    "snotime": "snotime", "lai": "lai", "smstav": "smstav",
    "smstot": "smstot", "sfcrunoff": "sfcrunoff", "udrunoff": "udrunoff",
    "acsnom": "acsnom", "acsnow": "acsnow", "snopcx": "snopcx",
    "potevp": "potevp", "noahres": "noahres", "chklowq": "chklowq",
}

#: Soil profiles, compared layer by layer.
SOIL_OUTPUT_COLUMNS = ("smois", "tslb", "sh2o", "smcrel")

#: The one fixture column gpuwm does not compute.  ``ivgtyp == isice`` selects
#: WRF's ``SFLX_GLACIAL`` (``module_sf_noahlsm_glacial_only.F``), which
#: gpuwm/core/noah.py's docstring records as out of scope, so ``noah_column``
#: returns without touching the column.  Excluding it from the parity
#: measurement is a statement of that restriction, not a way to make a number
#: smaller: :func:`glacial_divergence` reports how far WRF moved the column
#: that gpuwm left alone.
GLACIAL_CASE_PREDICATE = "ivgtyp == isice"


@dataclass(frozen=True)
class NoahOracleFixture:
    """One fixture file: ``ncase`` independent land columns."""

    name: str
    switches: dict
    cases: tuple[int, ...]
    ivgtyp: np.ndarray
    isltyp: np.ndarray
    inputs: dict[str, np.ndarray]
    reference: dict[str, np.ndarray]
    soil_reference: dict[str, np.ndarray]

    @property
    def ncase(self) -> int:
        return len(self.cases)

    @property
    def glacial(self) -> np.ndarray:
        """Boolean mask of the columns gpuwm skips by documented restriction."""
        return self.ivgtyp.reshape(-1) == FIXTURE_ISICE


def _column(rows, key) -> np.ndarray:
    """One fixture column as a ``(1, ncase)`` float32 row of land columns."""
    return np.ascontiguousarray(
        np.asarray([np.float32(row[key]) for row in rows],
                   dtype=np.float32).reshape(1, -1))


def load_noah_oracle(name: str = "noah-lsm.csv",
                     directory: Path | None = None) -> NoahOracleFixture:
    """Load one fixture CSV into ``launch_noah``'s ``(1, ncase)`` layout."""
    if name not in NOAH_ORACLE_FILES:
        raise ValueError(f"{name!r} is not a packaged Noah oracle fixture; "
                         f"expected one of {sorted(NOAH_ORACLE_FILES)}")
    directory = Path(directory) if directory is not None else NOAH_ORACLE_DIR
    with (directory / name).open(newline="", encoding="ascii") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{name} carries no fixture rows")

    cases = tuple(int(row["case"]) for row in rows)
    ivgtyp = np.ascontiguousarray(
        np.asarray([int(row["ivgtyp"]) for row in rows],
                   dtype=np.int32).reshape(1, -1))
    isltyp = np.ascontiguousarray(
        np.asarray([int(row["isltyp"]) for row in rows],
                   dtype=np.int32).reshape(1, -1))

    inputs = {field: _column(rows, column)
              for field, column in INPUT_COLUMNS.items()}
    for field in ("smois", "tslb", "sh2o"):
        profile = np.empty((4, 1, len(rows)), dtype=np.float32)
        for layer in range(4):
            profile[layer] = _column(rows, f"{field}{layer + 1}_in")
        inputs[field] = np.ascontiguousarray(profile)

    reference = {field: _column(rows, column)
                 for field, column in OUTPUT_COLUMNS.items()}
    soil_reference = {}
    for field in SOIL_OUTPUT_COLUMNS:
        profile = np.empty((4, 1, len(rows)), dtype=np.float32)
        for layer in range(4):
            profile[layer] = _column(rows, f"{field}{layer + 1}")
        soil_reference[field] = np.ascontiguousarray(profile)

    return NoahOracleFixture(
        name=name, switches=dict(NOAH_ORACLE_FILES[name]), cases=cases,
        ivgtyp=ivgtyp, isltyp=isltyp, inputs=inputs, reference=reference,
        soil_reference=soil_reference,
    )


def noah_port_outputs(fixture: NoahOracleFixture) -> dict[str, np.ndarray]:
    """Run ``launch_noah`` on the fixture and bring the results back to host.

    The kernel updates state in place, so the device arrays start as copies of
    the fixture's inputs -- which is what the WRF driver was handed too.
    """
    import cupy as cp

    from gpuwm.core.noah import launch_noah, load_tables, pack_params

    params = pack_params(load_tables())
    device: dict[str, "cp.ndarray"] = {
        "ivgtyp": cp.asarray(fixture.ivgtyp),
        "isltyp": cp.asarray(fixture.isltyp),
    }
    for field, value in fixture.inputs.items():
        device[field] = cp.asarray(value.copy())
    shape = fixture.ivgtyp.shape
    for field in ("noahres", "reslin", "chklowq"):
        device[field] = cp.zeros(shape, dtype=cp.float32)
    device["smcrel"] = cp.zeros((4, *shape), dtype=cp.float32)
    device["ebal"] = cp.zeros(shape, dtype=cp.int32)

    launch_noah(device, params, FIXTURE_DT, FIXTURE_DZS,
                isurban=FIXTURE_ISURBAN, isice=FIXTURE_ISICE,
                xice_threshold=FIXTURE_XICE_THRESHOLD,
                itimestep=FIXTURE_ITIMESTEP, **fixture.switches)
    return {name: cp.asnumpy(value) for name, value in device.items()
            if name not in ("ivgtyp", "isltyp", "ebal")}


@dataclass(frozen=True)
class FieldParity:
    """The measurement for one output field."""

    field: str
    max_ulp: int
    differing: int
    total: int
    worst_case: int
    worst_layer: int | None
    got: float
    want: float

    def __str__(self) -> str:
        where = (f"case {self.worst_case}"
                 + ("" if self.worst_layer is None
                    else f" layer {self.worst_layer}"))
        return (f"{self.field:<10s} max_ulp={self.max_ulp:<12d} "
                f"differing={self.differing}/{self.total} "
                f"worst at {where}: got {self.got!r} want {self.want!r}")


def _worst(fixture, field, got, want, keep) -> FieldParity:
    got = np.ascontiguousarray(got, np.float32)[..., keep]
    want = np.ascontiguousarray(want, np.float32)[..., keep]
    kept_cases = [c for c, k in zip(fixture.cases, keep) if k]
    distance = fp32_ulp_distance(got, want)
    flat = int(np.argmax(distance)) if distance.size else 0
    index = np.unravel_index(flat, distance.shape) if distance.size else ()
    if got.ndim == 3:
        layer, _, col = index
        worst_layer = int(layer) + 1
    else:
        _, col = index
        worst_layer = None
    return FieldParity(
        field=field,
        max_ulp=int(distance.max()) if distance.size else 0,
        differing=int(np.count_nonzero(
            got.view(np.uint32) != want.view(np.uint32))),
        total=int(got.size),
        worst_case=kept_cases[int(col)] if kept_cases else -1,
        worst_layer=worst_layer,
        got=float(got.ravel()[flat]), want=float(want.ravel()[flat]),
    )


def measure_noah_parity(fixture: NoahOracleFixture,
                        port: dict[str, np.ndarray],
                        *, include_glacial: bool = False
                        ) -> dict[str, FieldParity]:
    """ULP distance from every port output to the word ``lsm`` wrote.

    ``include_glacial`` is False by default because gpuwm does not implement
    ``SFLX_GLACIAL`` and says so; see :func:`glacial_divergence` for the size
    of what is being excluded.
    """
    keep = np.ones(fixture.ncase, dtype=bool)
    if not include_glacial:
        keep &= ~fixture.glacial
    result: dict[str, FieldParity] = {}
    for field in OUTPUT_COLUMNS:
        result[field] = _worst(fixture, field, port[field],
                               fixture.reference[field], keep)
    for field in SOIL_OUTPUT_COLUMNS:
        result[field] = _worst(fixture, field, port[field],
                               fixture.soil_reference[field], keep)
    return result


def glacial_divergence(fixture: NoahOracleFixture,
                       port: dict[str, np.ndarray]) -> dict[str, float]:
    """How far ``SFLX_GLACIAL`` moved the columns gpuwm leaves untouched.

    Reported as a plain difference, not ULP: this is not a rounding gap, it is
    a routine gpuwm does not have.  The number belongs in the registry warning
    so a user can see the size of the restriction rather than only its name.
    """
    mask = fixture.glacial
    out: dict[str, float] = {}
    if not mask.any():
        return out
    for field in OUTPUT_COLUMNS:
        got = np.ascontiguousarray(port[field], np.float32)[..., mask]
        want = fixture.reference[field][..., mask]
        gap = float(np.max(np.abs(got.astype(np.float64)
                                  - want.astype(np.float64))))
        if gap != 0.0:
            out[field] = gap
    return out
