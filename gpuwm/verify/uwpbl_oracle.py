"""Reader for the WRF v4.7.1 UW PBL (bl_pbl_physics=9) oracle fixtures.

The fixtures are written by ``tools/uwpbl_wrf471_oracle`` (see its README):
each ``<stem>.bin`` is raw little-endian words and ``<stem>.manifest`` names
them, one line per array, ``name dtype count byte_offset``.

Full-driver fixtures (``cases-<grid>``) carry ``const/...`` records (the
physical constants and the saturation table the Fortran used) and, per step
``s<n>/``, the exact float32 inputs camuwpbl received and every output it
wrote.  Every per-column array is stored column by column, so
:func:`step_arrays` hands them back shaped ``(ncol, nlev)``.

Spy fixtures (``cases-<grid>-spy``) are the stage records of the SPY build
(tools/uwpbl_wrf471_oracle/make_spy.py): float64 arguments of trbintd,
caleddy and the compute_vdiff calls, named ``c<col>/s<step>/it<iter>/
<stage>_{in,out}/<arg>``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DTYPES = {"f8": "<f8", "f4": "<f4", "i4": "<i4"}

#: The oracle's mass-level inputs, interface inputs and surface inputs, in
#: the order run_camuwpbl.F90 writes them for every step.
MASS_INPUTS = ("u", "v", "th", "rho", "qv", "qc", "qi", "qnc", "qni", "p",
               "z", "t", "cldfra", "exner", "rthratenlw", "wsedl3d")
FULL_INPUTS = ("p8w", "z_at_w", "kvm3d_in", "kvh3d_in")
SURFACE_INPUTS = ("hfx", "qfx", "ust", "ht", "tauresx2d_in", "tauresy2d_in")
MASS_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                "rqiblten", "rqniblten")
FULL_OUTPUTS = ("kvm3d", "kvh3d", "tke_pbl", "turbtype3d", "smaw3d")
SURFACE_OUTPUTS = ("tauresx2d", "tauresy2d", "tpert2d", "qpert2d",
                   "wpert2d", "pblh2d", "kpbl2d")


@dataclass(frozen=True)
class Fixture:
    """One oracle stream: every named array, read eagerly."""

    path: Path
    arrays: dict

    def __getitem__(self, name):
        return self.arrays[name]

    def __contains__(self, name):
        return name in self.arrays

    def names(self, prefix: str = ""):
        return [n for n in self.arrays if n.startswith(prefix)]


def load(stem) -> Fixture:
    """Read ``<stem>.bin`` through ``<stem>.manifest``."""
    stem = Path(stem)
    if stem.suffix in (".bin", ".manifest"):
        stem = stem.with_suffix("")
    raw = Path(f"{stem}.bin").read_bytes()
    manifest = Path(f"{stem}.manifest")
    arrays = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        name, dtype, count, offset = line.split()
        dt = np.dtype(_DTYPES[dtype])
        arrays[name] = np.frombuffer(raw, dtype=dt, count=int(count),
                                     offset=int(offset)).copy()
    return Fixture(path=stem, arrays=arrays)


def dims(fx: Fixture) -> tuple[int, int, int]:
    """``(ncol, nk, nsteps)`` of a full-driver fixture."""
    return (int(fx["const/ncol"][0]), int(fx["const/nk"][0]),
            int(fx["const/nsteps"][0]))


def step_arrays(fx: Fixture, step: int) -> dict[str, np.ndarray]:
    """Every input and output of one step, shaped per column."""
    ncol, nk, _ = dims(fx)
    out = {}
    for name in MASS_INPUTS + MASS_OUTPUTS:
        out[name] = fx[f"s{step}/{name}"].reshape(ncol, nk)
    for name in FULL_INPUTS + FULL_OUTPUTS:
        out[name] = fx[f"s{step}/{name}"].reshape(ncol, nk + 1)
    for name in SURFACE_INPUTS + SURFACE_OUTPUTS:
        out[name] = fx[f"s{step}/{name}"].reshape(ncol)
    out["itimestep"] = int(fx[f"s{step}/itimestep"][0])
    out["dt"] = np.float32(fx["const/dt"][0])
    return out


def stage(fx: Fixture, col: int, step: int, it: int, name: str) -> dict:
    """The spy records of one stage call, ``{arg: float64 array}``."""
    prefix = f"c{col}/s{step}/it{it}/{name}/"
    return {n[len(prefix):]: fx[n] for n in fx.names(prefix)}
