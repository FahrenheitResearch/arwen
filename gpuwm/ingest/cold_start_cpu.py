"""Native CPU cold-start numbers; Python retains receipt formatting only."""
from __future__ import annotations

import ctypes
from functools import lru_cache

import numpy as np


class _Inputs(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "mass", "number", "inverse_density", "temperature", "theta", "pressure",
        "aerosol", "landmask", "constants", "ice_radii", "droplet_ratio", "cloud_tables")]
    _fields_ += [(name, ctypes.c_size_t) for name in ("length", "aerosol_length", "landmask_length")]
    _fields_ += [("temperature_mode", ctypes.c_uint32), ("species", ctypes.c_uint32),
                 ("workers", ctypes.c_size_t)]


class _Summary(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "invalid_density", "seeded", "repaired", "needs_land", "invalid_aerosol",
        "invalid_cloud_entry", "invalid_result", "supercooled")]
    _fields_ += [("repaired_min", ctypes.c_double), ("repaired_max", ctypes.c_double)]


@lru_cache(maxsize=1)
def _constants():
    from gpuwm.core import constants as c, thompson_entry as te
    from gpuwm.core.host_libm import power
    f = np.float32
    def cube_quotient(a, b):
        return float(power(np.float64(f(a) / f(b)), np.float64(3)))
    pi = f(3.1415926536)
    scalars = np.array([
        f(f(pi * f(1000)) / f(6)), 6., f(pi * f(890)), 9., te.R1, te.R2,
        f(f(1) / f(3)), f(te.CRG2_ORG3),
        cube_quotient(te.MVD_FACTOR, te.RAIN_INITIAL_MVD_M), f(te.AM_R),
        f(te.MVD_FACTOR), f(te.RAIN_MVD_MAX_M), cube_quotient(te.MVD_FACTOR, te.RAIN_MVD_MAX_M),
        f(te.RAIN_MVD_MIN_M), cube_quotient(te.MVD_FACTOR, te.RAIN_MVD_MIN_M),
        f(te.AM_I), te.ICE_NUMBER_CEILING_M3, cube_quotient(te.CIE2, te.ICE_MIN_DIAMETER_M),
        f(te.ICE_MIN_DIAMETER_M), f(te.ICE_MAX_DIAMETER_M), cube_quotient(te.CIE2, te.ICE_MAX_DIAMETER_M),
        f(te.NT_C_MAX), te.NT_C_MAX, f(te.CUH_SCALARS["NC_FLOOR_M3"][1]), f(te.D0C), f(te.D0R),
        c.P0, c.RCP,
    ], dtype=np.float64)
    tables = np.ascontiguousarray(np.stack([te.CCG1, te.CCG2, te.OCG1, te.OCG2, te.CCE2]), dtype=np.float32)
    return scalars, np.ascontiguousarray(te.ICE_RETAB), np.ascontiguousarray(te.DROPLET_G_RATIO), tables


def _merge(target, source):
    for key, value in source.items():
        if key.endswith("_cells"):
            target[key] = target.get(key, 0) + value
        elif key.endswith("_min") or key.endswith("_max"):
            if key not in target or np.isnan(value):
                target[key] = value
            elif not np.isnan(target[key]):
                target[key] = (min if key.endswith("_min") else max)(target[key], value)
        else:
            target[key] = value


def _seed_receipt(name, count, mass, alt, volume, mask, aerosol, land, summary, *, receipt):
    # The diagnostics keep the existing NumPy cbrt contract. Only one
    # bounded receipt block is gathered, after all scientific values are
    # finished and validated by Rust; boundary-only work has no such pass.
    from gpuwm.ingest.closure_device import _seed_receipt as describe
    if not receipt or not count:
        return describe(name, count, None, None, None, None, None, None, build_receipt=False)
    result = {}
    block = 1 << 18
    for start in range(0, mass.size, block):
        stop = min(start + block, mass.size)
        selected = np.flatnonzero(mask.ravel()[start:stop]) + start
        if not selected.size:
            continue
        m = mass.flat[selected]
        a = np.asarray(alt.flat[selected], dtype=np.float32)
        v = volume.flat[selected]
        nw = np.zeros(selected.size, np.float32) if aerosol is None else (
            np.full(selected.size, aerosol.item(), np.float32) if aerosol.size == 1 else aerosol.flat[selected])
        x = np.ones(selected.size, np.float32) if land is None else np.where(
            land.flat[selected % land.size] >= 0.5, 1.0, 2.0).astype(np.float32)
        temp = np.full(selected.size, 273.15, np.float32)
        part = describe(name, selected.size, m, a, v, temp, nw, x)
        _merge(result, part)
    if name == "rain":
        result["supercooled_cells"] = int(summary.supercooled)
    return result


def close_numbers(state, cfg, inverse_density, *, aerosol_number=None, landmask=None,
                  temperature=None, temperature_fields=None, receipt=True):
    """Return the existing receipt, or None for an older/unsupported caller."""
    from gpuwm.core import portable_math as pm, thompson_entry as te
    from gpuwm.boundary_fields import COLD_START_SEEDED_NUMBERS
    from gpuwm.ingest.real import COLD_START_MOMENT_CLOSURE_SCHEMA
    library = pm._load()
    entry = getattr(library, "gpuwm_cold_start_numbers_f32", None)
    copy = getattr(library, "gpuwm_parallel_copy_f32", None)
    if entry is None or copy is None or (callable(temperature) and temperature_fields is None):
        return None
    pairs = COLD_START_SEEDED_NUMBERS[int(cfg.mp_physics)]
    first = getattr(state, pairs[0][0], None)
    if first is None:
        # No state to close: the reference refuses an invalid inverse
        # density before it reads the state, and a caller that hands none
        # (the refusal's own callers) must reach that sentence, not an
        # AttributeError from this lookup.
        return None
    shape = first.shape
    for q, n in pairs:
        if any(not isinstance(value, np.ndarray) or value.dtype != np.float32
               or not value.flags.c_contiguous or not value.flags.aligned or value.shape != shape
               for value in (getattr(state, q), getattr(state, n))):
            return None
        if not getattr(state, n).flags.writeable:
            return None
    alt = np.require(inverse_density, dtype=np.float64, requirements=["C", "A"])
    if alt.shape != shape:
        return None
    aerosol = None if aerosol_number is None else np.require(aerosol_number, dtype=np.float32, requirements=["C", "A"])
    land = None if landmask is None else np.require(landmask, dtype=np.float64, requirements=["C", "A"])
    # Native modulo addressing is exactly a trailing, contiguous broadcast.
    # A valid (ny, 1) land mask instead repeats each row across x, and must
    # retain the reference until a strided native descriptor is supplied.
    for value in (aerosol, land):
        if value is not None:
            try:
                np.broadcast_to(value, shape)
            except ValueError:
                return None
    if aerosol is not None and aerosol.size not in (1, alt.size):
        return None
    if land is not None and (land.size == 0 or alt.size % land.size):
        return None
    if land is not None and land.size != 1:
        suffix = tuple(land.shape)
        while suffix and suffix[0] == 1:
            suffix = suffix[1:]
        if suffix != tuple(shape[-len(suffix):]):
            return None
    theta = pressure = temp = None
    mode = 0
    if temperature_fields is not None:
        theta, pressure = (np.require(value, dtype=np.float64, requirements=["C", "A"]) for value in temperature_fields)
        if theta.shape != shape or pressure.shape != shape:
            return None
        mode = 2
    elif temperature is not None:
        temp = np.require(temperature, dtype=np.float32, requirements=["C", "A"])
        if temp.shape != shape:
            return None
        mode = 1
    entry.argtypes = [ctypes.POINTER(_Inputs), ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_void_p, ctypes.POINTER(_Summary)]
    entry.restype = ctypes.c_int32
    copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    copy.restype = ctypes.c_int32
    constants = _constants()
    workers = pm._workers(None)
    def pointer(value):
        return None if value is None else value.ctypes.data
    seeds, entries, written, total = {}, [], [], 0
    names = {"qc": ("cloud", 2), "qr": ("rain", 0), "qi": ("ice", 1)}
    for mass_field, number_field in pairs:
        name, code = names[mass_field]
        mass, number = getattr(state, mass_field), getattr(state, number_field)
        output = np.empty_like(number)
        volume = np.empty_like(number) if receipt else None
        mask = np.empty(number.shape, np.uint8) if receipt else None
        inputs = _Inputs(pointer(mass), pointer(number), pointer(alt), pointer(temp),
                         pointer(theta), pointer(pressure), pointer(aerosol), pointer(land),
                         *(pointer(value) for value in constants), alt.size,
                         0 if aerosol is None else aerosol.size, 0 if land is None else land.size,
                         mode, code, workers)
        summary = _Summary()
        result = entry(ctypes.byref(inputs), pointer(output), pointer(volume), pointer(mask), ctypes.byref(summary))
        if result:
            raise RuntimeError(f"native cold-start numbers failed with code {result}")
        if summary.invalid_density:
            raise ValueError("the initializer's inverse density is zero or not finite in "
                             f"{summary.invalid_density} cell(s), so the density Thompson's entry block works in cannot be formed there")
        if summary.needs_land:
            raise ValueError("mp_physics=28 cold start: "
                             f"{summary.needs_land} cloudy cell(s) carry no aerosol, and real.exe's make_DropletNumber then "
                             "sizes the droplets by land or water (XLAND), but this initialization was given no target LANDMASK; "
                             "pass landmask=<static LANDMASK> to initialize_real")
        if summary.invalid_aerosol:
            raise ValueError(te.cold_start_aerosol_row_refusal(summary.invalid_aerosol))
        if summary.invalid_cloud_entry:
            raise ValueError(te.cold_start_droplet_row_refusal(summary.invalid_cloud_entry))
        if summary.seeded and name != "cloud" and mode == 0:
            raise ValueError(f"Thompson cold start: {summary.seeded} cell(s) carry {name} mass and no number, "
                             f"and real.exe's make_{name.capitalize()}Number sizes them by temperature, but the closure was given none")
        if summary.seeded and summary.invalid_result:
            raise ValueError(f"the cold-start moment closure produced a non-finite or negative {number_field}; "
                             f"the analyzed {mass_field} it was closed over is not a state the scheme can start from")
        item = {"species": name, "mass_field": mass_field, "number_field": number_field,
                "offending_cells": int(summary.repaired), "repaired_cells": int(summary.repaired)}
        if summary.repaired and receipt:
            item.update(repaired_number_min=summary.repaired_min, repaired_number_max=summary.repaired_max)
        seeds[name] = _seed_receipt(name, int(summary.seeded), mass, alt, volume, mask,
                                    aerosol, land, summary, receipt=receipt)
        if summary.seeded:
            result = copy(pointer(output), pointer(number), number.size, workers)
            if result:
                raise RuntimeError(f"native cold-start publication failed with code {result}")
            written.append(number_field)
        entries.append(item)
        total += int(summary.repaired)
    return {
        "schema": COLD_START_MOMENT_CLOSURE_SCHEMA,
        "droplet_number_seed": seeds.get("cloud"), "rain_number_seed": seeds["rain"],
        "ice_number_seed": seeds["ice"], "repaired": True, "repaired_cells_total": total,
        "authority": te.THOMPSON_ENTRY_AUTHORITY, "mp_physics": int(cfg.mp_physics),
        "q_threshold_kg_kg": float(te.R1), "species": entries,
        "note": ("cells with mass and a number at or below zero first take "
                 "real.exe's make_DropletNumber, make_RainNumber and "
                 "make_IceNumber (droplet_number_seed, rain_number_seed, "
                 "ice_number_seed); cells with mass above the scheme's "
                 "activity threshold and a number moment at or below zero "
                 "were then written by the entry block; every other cell, "
                 "an analysed number above zero included, keeps the value "
                 "real.exe writes"),
        "written_state_fields": written,
        "density": "initializer moist specific volume, rho = 1/alt",
        "entry_block": te.THOMPSON_ENTRY_SOURCE + " through gpuwm.core.thompson_entry",
        "cells_without_mass": "exact FP32 zero, as real.exe writes them",
    }
