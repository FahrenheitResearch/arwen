"""Read the WRF v4.6.1 revised MM5 surface-layer oracle and run the port on it.

``tools/sfclayrev_wrf461_oracle/run_sfclayrev.F90`` drives the WRF wrapper
``SFCLAYREV`` (``phys/module_sf_sfclayrev.F``) over the byte-unmodified
``phys/physics_mmm/sf_sfclayrev.F90``, built at gfortran ``-O0`` against
glibc's scalar libm.  It reads ``sfclayrev-inputs.hex`` and writes
``sfclayrev-outputs.hex``: six switch arms x three chained steps x every
column, each row the 34 words of
:data:`gpuwm.core.physics_inventory.SFCLAY_OUTPUTS` as float32 hex.

:func:`port_outputs` runs ``kernels/sfclay.cu`` through
:func:`gpuwm.core.sfclay.sfclay` / :func:`gpuwm.core.sfclay.launch_sfclay` --
the production entry points -- on the same words, chaining the inout fields
exactly as WRF does.  Nothing here knows a tolerance; it returns
measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ORACLE_DIR = Path(__file__).resolve().parent / "data" / "oracles" / "sfclayrev"
INPUT_FIELDS = ("u", "v", "t", "qv", "p", "dz8w", "psfc", "tsk", "pblh",
                "mavail", "xland", "lakemask", "dx", "znt", "ust", "ustm",
                "mol", "hfx", "qfx", "qsfc", "zol")
#: (isfflx, isftcflx, iz0tlnd) per arm, the order run_sfclayrev.F90 runs.
ARMS = ((1, 0, 0), (1, 1, 0), (1, 2, 0), (1, 0, 1), (1, 0, 2), (0, 0, 0))
STEPS = 3

#: (arm index, field): words WRF leaves undefined, so there is nothing to
#: compare.  With isfflx=0 sf_sfclayrev_run jumps past every LH assignment
#: (sf_sfclayrev.F90:794 ``goto 410``) and SFCLAYREV copies its never-
#: assigned local ``lh_hv`` into LH (module_sf_sfclayrev.F:226): an
#: uninitialised read in WRF.  The port writes 0.
WRF_UNDEFINED = frozenset({(5, "lh")})


@dataclass(frozen=True)
class Fixture:
    cases: np.ndarray            # (ncol,) case ids
    labels: tuple[str, ...]      # (ncol,)
    inputs: dict[str, np.ndarray]    # field -> (ncol,) float32
    outputs: np.ndarray          # (narm, nstep, ncol, nfield) float32


def _words(tokens) -> np.ndarray:
    return np.array([int(tok, 16) for tok in tokens], dtype=np.uint32).view(np.float32)


def load_fixture(directory: Path = ORACLE_DIR,
                 outputs_name: str = "sfclayrev-outputs.hex") -> Fixture:
    from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS
    rows = [line.split() for line in
            (directory / "sfclayrev-inputs.hex").read_text().splitlines()
            if line and not line.startswith("#")]
    cases = np.array([int(r[0]) for r in rows])
    data = np.stack([_words(r[1:]) for r in rows])
    if data.shape[1] != len(INPUT_FIELDS):
        raise ValueError(f"inputs carry {data.shape[1]} words, want {len(INPUT_FIELDS)}")
    inputs = {name: np.ascontiguousarray(data[:, j]) for j, name in enumerate(INPUT_FIELDS)}
    labels = {}
    for line in (directory / "sfclayrev-cases.csv").read_text().splitlines()[1:]:
        case, label = line.split(",")[:2]
        labels[int(case)] = label
    index = {int(c): i for i, c in enumerate(cases)}
    out = np.full((len(ARMS), STEPS, len(cases), len(SFCLAY_OUTPUTS)), np.nan, np.float32)
    seen = 0
    for line in (directory / outputs_name).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        arm, switches, step, case = int(tok[0]), tuple(map(int, tok[1:4])), int(tok[4]), int(tok[5])
        if switches != ARMS[arm - 1]:
            raise ValueError(f"arm {arm} carries switches {switches}, want {ARMS[arm - 1]}")
        out[arm - 1, step - 1, index[case]] = _words(tok[6:])
        seen += 1
    if seen != out.shape[0] * out.shape[1] * out.shape[2]:
        raise ValueError(f"oracle has {seen} rows, want {out.shape[0] * out.shape[1] * out.shape[2]}")
    return Fixture(cases, tuple(labels[int(c)] for c in cases), inputs, out)


def port_outputs(fixture: Fixture) -> np.ndarray:
    """Run kernels/sfclay.cu (option 1) on every arm/step; same shape as the oracle."""
    import cupy as cp
    from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS
    from gpuwm.core.sfclay import launch_sfclay, sfclay

    inp = fixture.inputs
    out = np.full(fixture.outputs.shape, np.nan, np.float32)
    dxs = np.unique(inp["dx"])
    for a, (isfflx, isftcflx, iz0tlnd) in enumerate(ARMS):
        for dx in dxs:
            sel = np.flatnonzero(inp["dx"] == dx)
            g = {k: cp.asarray(v[sel][None, :]) for k, v in inp.items()}
            atm = (g["u"], g["v"], g["t"], g["qv"], g["p"], g["dz8w"], g["psfc"],
                   g["tsk"])
            result = sfclay(*atm, g["znt"], g["pblh"], g["mavail"], g["xland"],
                            option=1, qsfc=g["qsfc"], zol=g["zol"], ust=g["ust"],
                            ustm=g["ustm"], mol=g["mol"], hfx=g["hfx"],
                            qfx=g["qfx"], lakemask=g["lakemask"], dx=float(dx),
                            isfflx=bool(isfflx), isftcflx=isftcflx,
                            iz0tlnd=iz0tlnd)
            for step in range(STEPS):
                if step:
                    launch_sfclay(*atm, g["pblh"], g["mavail"], g["xland"],
                                  g["lakemask"], result, option=1, dx=float(dx),
                                  isfflx=bool(isfflx), isftcflx=isftcflx,
                                  iz0tlnd=iz0tlnd)
                cp.cuda.Device().synchronize()
                for f, name in enumerate(SFCLAY_OUTPUTS):
                    out[a, step, sel, f] = cp.asnumpy(getattr(result, name))[0]
    return out


def measure(fixture: Fixture, port: np.ndarray) -> dict:
    """Per-field worst ULP and abs diff, and every mismatching lane."""
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS

    want = fixture.outputs
    fields = {}
    lanes = []
    for f, name in enumerate(SFCLAY_OUTPUTS):
        mask = np.ones(want.shape[:3], bool)
        for a in range(len(ARMS)):
            if (a, name) in WRF_UNDEFINED:
                mask[a] = False
        w, p = want[..., f], port[..., f]
        ulp = np.where(mask, fp32_ulp_distance(p, w), 0)
        with np.errstate(invalid="ignore"):
            absd = np.where(mask, np.abs(p.astype(np.float64) - w.astype(np.float64)), 0.0)
        fields[name] = {"max_ulp": int(ulp.max()), "max_abs": float(np.nanmax(absd)),
                        "lanes_differ": int((ulp != 0).sum()),
                        "lanes": int(mask.sum())}
        for a, s, c in zip(*np.nonzero(ulp)):
            lanes.append((name, a, s, int(fixture.cases[c]), fixture.labels[c],
                          int(ulp[a, s, c]), float(w[a, s, c]), float(p[a, s, c])))
    return {"fields": fields, "lanes": lanes,
            "max_ulp": max(v["max_ulp"] for v in fields.values())}
