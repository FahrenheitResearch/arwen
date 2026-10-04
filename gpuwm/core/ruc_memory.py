"""CPU allocation inventory for the full-width RUC runtime.

The generated sfctmp layout includes soil, snow and sea-ice intermediates.
These are allocation shapes, not a second implementation of the physics.
GPU census tests check them against the unchanged runtime's real arrays.
"""

from math import prod

import numpy as np

from gpuwm.core import ruc
from gpuwm.core.ruc_sfctmp_layout import (
    _SFCTMP_ARRAYS, _SFCTMP_CHECKS, _SFCTMP_OUTPUTS, _SFCTMP_SLOTS,
)

DRIVER_COLUMNS = (ruc.RUC_DRIVER_COLUMN_STATE + ruc.RUC_DRIVER_COLUMN_FORCING
            + ruc.RUC_DRIVER_ARW_FORCING)
DRIVER_EXTRAS = ("psfc", "chs2", "cqs2", "cpm", "qgh", "tsk_save", "tsk_sea",
           "flhc_sea", "flqc_sea", "cpm_sea", "cqs2_sea", "chs2_sea",
           "chs_sea", "qsfc_sea", "qgh_sea", "hfx_sea", "qfx_sea", "lh_sea")
DRIVER_LOCALS = ("patm", "conflx", "prcpms", "newsnms", "snowrat", "grauprat",
           "icerat", "curat", "qkms", "tkms", "snwe", "snhei", "canwatr",
           "snowfrac", "rhosnfall", "rhosn", "emissl", "pc", "qwrtz",
           "rhocs", "bclh", "dqm", "ksat", "psis", "qmin", "ref", "wilt",
           "meltfactor", "lmavail", "sat", "cn", "snoh", "snflx", "s",
           "sublim", "evapl", "infiltr", "smelt", "runoff1", "runoff2",
           "t2", "th2", "q2", "scale", "inverse")
DRIVER_PROFILES = ruc.RUC_DRIVER_PROFILE_STATE
DRIVER_WORK_PROFILES = ("soilm1d", "tso1d", "smfrkeep", "keepfr", "soiliqw", "soilice")
DRIVER_SCRATCH_NAMES = DRIVER_COLUMNS + DRIVER_EXTRAS + DRIVER_LOCALS + DRIVER_PROFILES + DRIVER_WORK_PROFILES
DRIVER_INPUT_NAMES = DRIVER_COLUMNS + DRIVER_EXTRAS + DRIVER_PROFILES
SFCTMP_OUTPUT_NAMES = tuple(ruc.RucSurfaceTemperatureStep.__dataclass_fields__)
DRIVER_OUTPUT_NAMES = tuple(name for name in ruc.RucLandSurfaceStep.__dataclass_fields__
                      if name != "ilnb")


DRIVER_SCRATCH_NAMES += ("seaice",)
DRIVER_TARGET_NAMES = (ruc.RUC_DRIVER_COLUMN_STATE + ("albbck", "chs", "flhc", "flqc")
                       + DRIVER_EXTRAS + DRIVER_PROFILES
                       + ("infiltr", "smelt", "runoff1", "runoff2", "t2", "th2", "q2"))
DRIVER_FLAG_SLOTS = 20
SFCTMP_FLAG_WORDS = (len(_SFCTMP_CHECKS) + 63) // 64
SFCTMP_FLAGS_SIZE = SFCTMP_FLAG_WORDS + len(_SFCTMP_CHECKS)


def _dimensions(ncol, num_soil_layers):
    ncol, num_soil_layers = int(ncol), int(num_soil_layers)
    if ncol < 0 or num_soil_layers not in (6, 9):
        raise ValueError("RUC allocation inventory requires nonnegative columns and 6 or 9 soil levels")
    return ncol, num_soil_layers


def sfctmp_scratch_layout(ncol, num_soil_layers):
    """Float32 offsets and total elements of the generated scratch slab."""
    ncol, nzs = _dimensions(ncol, num_soil_layers)
    offsets, units = [], 0
    for size in _SFCTMP_SLOTS:
        offsets.append(units * ncol)
        units += nzs if size == 9 else 1
    return offsets, units * ncol


def driver_workspace_allocations(ncol, num_soil_layers):
    """Resident driver workspace allocations, excluding pinned host mirrors."""
    ncol, nzs = _dimensions(ncol, num_soil_layers)
    profiles = set(DRIVER_PROFILES + DRIVER_WORK_PROFILES)
    rows = sum(nzs if name in profiles else 1 for name in DRIVER_SCRATCH_NAMES)
    return {
        "storage": ((rows, ncol), "float32"),
        "integer": ((5, ncol), "int32"),
        "run": ((ncol,), "bool"),
        "flag_slab": ((DRIVER_FLAG_SLOTS + SFCTMP_FLAGS_SIZE,), "uint64"),
        "sptr": ((1,), "uint64"),
        "iptr": ((len(DRIVER_INPUT_NAMES) + 2,), "uint64"),
        "optr": ((len(SFCTMP_OUTPUT_NAMES),), "uint64"),
        # Constructed at input width and resized at first commit. Price the
        # larger of both allocations; they round to the same pool quantum.
        "cptr": ((max(len(DRIVER_INPUT_NAMES), len(DRIVER_TARGET_NAMES)),), "uint64"),
    }


def sfctmp_workspace_allocations(ncol, num_soil_layers):
    """Resident generated workspace, including snow and sea-ice branches."""
    ncol, nzs = _dimensions(ncol, num_soil_layers)
    _, elements = sfctmp_scratch_layout(ncol, nzs)
    return {
        "scratch": ((elements,), "float32"),
        "pointers": ((len(_SFCTMP_ARRAYS),), "uint64"),
        "private_flags": ((SFCTMP_FLAGS_SIZE,), "uint64"),
        "alive": ((ncol,), "bool"),
    }


def sfctmp_output_allocations(ncol, num_soil_layers):
    """Separate outputs alive alongside both workspaces until driver commit."""
    ncol, nzs = _dimensions(ncol, num_soil_layers)
    return {name: (((nzs, ncol) if _SFCTMP_ARRAYS[index][2] else (ncol,)),
                   _SFCTMP_ARRAYS[index][1])
            for name, index in _SFCTMP_OUTPUTS.items()}


def ruc_table_allocations(num_soil_layers, *, vegetation_rows=30, soil_rows=19):
    """Read-only tables for one selected vegetation/soil parameter bundle."""
    _, nzs = _dimensions(0, num_soil_layers)
    arrays = {name: ((int(vegetation_rows),), "int32" if name == "ifortbl" else "float32")
              for name in ("ifortbl", "z0tbl", "lemitbl", "pctbl", "laitbl", "rstbl", "rgltbl")}
    arrays.update({name: ((int(soil_rows),), "float32") for name in
                   ("bb", "drysmc", "hc", "maxsmc", "refsmc", "satpsi", "satdk", "wltsmc", "qtz")})
    arrays.update(tbq=((5001,), "float32"), zshalf=((nzs,), "float32"))
    return arrays


def allocation_bytes(allocations, *, rounded=True):
    """Sum distinct allocations using CuPy's 512-byte pool quantum."""
    sizes = (prod(shape) * np.dtype(dtype).itemsize
             for shape, dtype in allocations.values())
    return sum(((size + 511) // 512) * 512 if rounded else size for size in sizes)


def ruc_runtime_memory_bytes(ncol, num_soil_layers):
    """Runtime-owned bytes separate from fields already priced by the driver."""
    return {
        "driver_workspace": allocation_bytes(driver_workspace_allocations(ncol, num_soil_layers)),
        "sfctmp_workspace": allocation_bytes(sfctmp_workspace_allocations(ncol, num_soil_layers)),
        "sfctmp_outputs": allocation_bytes(sfctmp_output_allocations(ncol, num_soil_layers)),
        "tables": allocation_bytes(ruc_table_allocations(num_soil_layers)),
    }


def ruc_pinned_host_allocations(ncol, num_soil_layers):
    """Pinned mirrors plus the driver's bounded upload/rebind peak.

    The driver reads its flag slab synchronously before returning. Thus
    one pointer upload can remain pending in an ordinary step, rather than
    an unbounded queue across steps. Include the extra reset copy used by
    the refusal path and both commit-pointer mirrors during first rebind.
    CUDA's internally owned transfer staging is a separate allowance.
    """
    ncol, _ = _dimensions(ncol, num_soil_layers)
    return {
        "driver_input_pointers": ((len(DRIVER_INPUT_NAMES) + 2,), "uint64"),
        "driver_output_pointers": ((len(SFCTMP_OUTPUT_NAMES),), "uint64"),
        "driver_initial_commit_pointers": ((len(DRIVER_INPUT_NAMES),), "uint64"),
        "driver_commit_pointers": ((len(DRIVER_TARGET_NAMES),), "uint64"),
        "driver_psfc": ((ncol,), "float32"),
        "driver_scale": ((ncol,), "float32"),
        "driver_inverse": ((ncol,), "float32"),
        "sfctmp_pointers": ((len(_SFCTMP_ARRAYS),), "uint64"),
        "sfctmp_reset": ((SFCTMP_FLAGS_SIZE,), "uint64"),
        "sfctmp_pointer_upload": ((len(_SFCTMP_ARRAYS),), "uint64"),
        "sfctmp_error_reset": ((SFCTMP_FLAGS_SIZE,), "uint64"),
    }


def ruc_pinned_host_bytes(ncol, num_soil_layers):
    """Known host allocations, rounded to the pinned pool's 512-byte bins."""
    return allocation_bytes(ruc_pinned_host_allocations(ncol, num_soil_layers))
