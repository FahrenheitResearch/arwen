"""Gathered Thompson cold start; cloud entry retains the host C-library pow.

Temperature uses the Rust libm f64 device twin at seeded cells only.
Receipt diameters remain host diagnostics.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from gpuwm.core import thompson_entry as te
from gpuwm.core.kernels import get_kernel


def make_temperature_provider(theta, pressure):
    """Keep supplied f64 fields on card, lazily, and evaluate only seed indices."""
    import cupy as cp
    from gpuwm.core import constants as c
    fields = []

    def temperature(indices):
        if not fields:
            fields.extend(cp.ascontiguousarray(cp.asarray(value, dtype=cp.float64)).ravel()
                          for value in (theta, pressure))
        idx = cp.ascontiguousarray(indices, dtype=cp.int64)
        output = cp.empty(idx.shape, dtype=cp.float32)
        get_kernel("thompson_cold_start", "cold_start_temperature")(
            ((idx.size + 255) // 256,), (256,),
            (*fields, idx, output, np.int32(idx.size), np.float64(c.P0), np.float64(c.RCP)))
        return output

    return temperature


@lru_cache(maxsize=1)
def _constants():
    """Fold scalar DOUBLE powers with the same host library as the reference."""
    from gpuwm.core.host_libm import power
    f = np.float32

    def cube_quotient(a, b):
        return float(power(np.float64(f(a) / f(b)), np.float64(3.0)))

    pi = f(3.1415926536)
    return np.array([
        f(f(pi * f(1000.0)) / f(6.0)), 6.0, f(pi * f(890.0)), 9.0,
        te.R1, te.R2, f(f(1.0) / f(3.0)), f(te.CRG2_ORG3),
        cube_quotient(te.MVD_FACTOR, te.RAIN_INITIAL_MVD_M), f(te.AM_R),
        f(te.MVD_FACTOR), f(te.RAIN_MVD_MAX_M),
        cube_quotient(te.MVD_FACTOR, te.RAIN_MVD_MAX_M),
        f(te.RAIN_MVD_MIN_M), cube_quotient(te.MVD_FACTOR, te.RAIN_MVD_MIN_M),
        f(te.AM_I), te.ICE_NUMBER_CEILING_M3,
        cube_quotient(te.CIE2, te.ICE_MIN_DIAMETER_M),
        f(te.ICE_MIN_DIAMETER_M), f(te.ICE_MAX_DIAMETER_M),
        cube_quotient(te.CIE2, te.ICE_MAX_DIAMETER_M),
    ], dtype=np.float64)


def gathered_numbers(species, mass, number, alt, temperature, aerosol, xland):
    """Return seeded and closed numbers plus seed per-volume numbers on device."""
    import cupy as cp
    m, n, a, t, w, x = (cp.ascontiguousarray(cp.asarray(v, dtype=cp.float32))
                         for v in (mass, number, alt, temperature, aerosol, xland))
    out = cp.empty_like(m)
    volume = cp.empty_like(m)
    get_kernel("thompson_cold_start", "cold_start_numbers")(
        ((m.size + 255) // 256,), (256,),
        (m, n, a, t, w, x, cp.asarray(te.ICE_RETAB),
         cp.asarray(te.DROPLET_G_RATIO), cp.asarray(_constants()),
         out, volume, np.int32(m.size), np.int32({"rain": 0, "ice": 1, "cloud": 2}[species])))
    if species == "cloud":
        # glibc DOUBLE pow has no qualified device twin. Only offenders cross.
        active = m > cp.float32(te.R1)
        if bool(active.any()):
            from gpuwm.core.thompson_entry import np_thompson_entry_numbers
            rho = 1.0 / a[active].astype(cp.float64)
            closed = np_thompson_entry_numbers(
                "cloud", m[active].get(), out[active].get(), rho.get())
            out[active] = cp.asarray(closed, dtype=cp.float32)
    return out, volume


def _seed_receipt(name, count, mass, alt, volume, temp, nwfa, xland, *, build_receipt=True):
    """Host diagnostics on gathered seed cells, in the reference's association."""
    cloud = name == "cloud"
    rule = {
        "cloud": "qc > 0 and nc <= 0: nc = make_DropletNumber(qc*rho, nwfa*rho, xland) / rho, rho = 1/alt",
        "rain": "qr > 0 and nr <= 0: nr = make_RainNumber(qr*rho, T) / rho",
        "ice": "qi > 0 and ni <= 0: ni = make_IceNumber(qi*rho, T) / rho",
    }[name]
    receipt = {"authority": (te.MAKE_DROPLET_NUMBER_SOURCE if cloud
                              else te.MAKE_RAIN_ICE_NUMBER_SOURCE),
               "rule": rule, "seeded_cells": count}
    if not count or not build_receipt:
        return receipt
    rho = (np.float32(1.0) / alt).astype(np.float32)
    per_mass = (mass * rho).astype(np.float32)
    if cloud:
        surface = nwfa * rho <= np.float32(0.0)
        ocean = xland > 1.5
        diameter = te.droplet_mean_diameter_m(per_mass, volume)
        receipt.update({
            "aerosol_branch_cells": int(np.count_nonzero(~surface)),
            "land_branch_cells": int(np.count_nonzero(surface & ~ocean)),
            "water_branch_cells": int(np.count_nonzero(surface & ocean)),
            "number_per_cm3_min": float(volume.min() * 1.0e-6),
            "number_per_cm3_max": float(volume.max() * 1.0e-6),
            "mean_diameter_um_min": float(diameter.min() * 1.0e6),
            "mean_diameter_um_max": float(diameter.max() * 1.0e6),
        })
    else:
        diameter = (te.rain_median_volume_diameter_m if name == "rain"
                    else te.ice_mean_diameter_m)(per_mass, volume)
        receipt.update({
            "number_per_m3_min": float(volume.min()),
            "number_per_m3_max": float(volume.max()),
            "diameter_um_min": float(diameter.min() * 1.0e6),
            "diameter_um_max": float(diameter.max() * 1.0e6),
            "diameter": "median volume diameter" if name == "rain" else "mean diameter 3/lambda",
        })
        if name == "rain":
            receipt["supercooled_cells"] = int(np.count_nonzero(temp <= np.float32(271.15)))
    return receipt


def _seed_mask(mass, number):
    import cupy as cp
    mb, nb = mass.view(cp.uint32), number.view(cp.uint32)
    na = nb & cp.uint32(0x7fffffff)
    return ((mb > 0) & (mb <= cp.uint32(0x7f800000))
            & ((na == 0) | ((nb >= cp.uint32(0x80000000))
                            & (na <= cp.uint32(0x7f800000)))))


def _rounded_alt(value):
    """Round supplied specific volume on device, retaining FP32 subnormals."""
    import cupy as cp
    array = cp.asarray(value)
    if array.dtype != cp.float64:
        return cp.asarray(array, dtype=cp.float32)
    output = cp.empty(array.shape, dtype=cp.float32)
    get_kernel("thompson_cold_start", "cold_start_alt")(
        ((array.size + 255) // 256,), (256,),
        (cp.ascontiguousarray(array), output, np.int32(array.size)))
    return output


def _closure_chunk(state, state_xp, cfg, inverse_density,
                                     *, aerosol_number=None, landmask=None,
                                     temperature=None, only_pair=None,
                                     defer_result_validation=False, receipt=True):
    """Close device moments; temperature callable accepts device seed indices."""
    import cupy as cp
    from gpuwm.boundary_fields import COLD_START_SEEDED_NUMBERS
    from gpuwm.ingest.real import COLD_START_MOMENT_CLOSURE_SCHEMA

    alt = _rounded_alt(inverse_density)
    alt_bits = alt.view(cp.uint32) & cp.uint32(0x7fffffff)
    invalid = (alt_bits == 0) | (alt_bits > cp.uint32(0x7f800000))
    bad = int(cp.count_nonzero(invalid))
    if bad:
        raise ValueError(
            "the initializer's inverse density is zero or not finite in "
            f"{bad} cell(s), so the density Thompson's entry block works in cannot be formed there")
    entries, written, seeds, total = [], [], {}, 0
    kinds = {"qc": "cloud", "qr": "rain", "qi": "ice"}
    pairs = COLD_START_SEEDED_NUMBERS[int(cfg.mp_physics)] if only_pair is None else (only_pair,)
    for mass_field, number_field in pairs:
        name = kinds[mass_field]
        mass, number = getattr(state, mass_field), getattr(state, number_field)
        # Offenders are a subset of seeds. The host compares the FP32 array to R1.
        # Integer predicates retain positive subnormal masses and numbers.
        nb = number.view(cp.uint32)
        seed = _seed_mask(mass, number)
        idx = cp.flatnonzero(seed.ravel())
        count = int(idx.size)
        m, n, a = mass.ravel()[idx], number.ravel()[idx], alt.ravel()[idx]
        offenders = m > cp.float32(te.R1)
        repairs = int(cp.count_nonzero(offenders))
        entry = {"species": name, "mass_field": mass_field, "number_field": number_field,
                 "offending_cells": repairs, "repaired_cells": 0}
        temp = cp.zeros_like(m)
        nwfa = cp.zeros_like(m)
        xland = cp.ones_like(m)
        if count:
            if name != "cloud":
                if temperature is None:
                    raise ValueError(
                        f"Thompson cold start: {count} cell(s) carry {name} mass "
                        "and no number, and real.exe's make_"
                        f"{name.capitalize()}Number sizes them by temperature, "
                        "but the closure was given none")
                temp = cp.asarray(temperature(idx) if callable(temperature)
                                  else cp.asarray(temperature).ravel()[idx], dtype=cp.float32)
            else:
                nwfa = cp.broadcast_to(cp.asarray(0.0 if aerosol_number is None
                                                 else aerosol_number, dtype=cp.float32), mass.shape).ravel()[idx]
                surface = cp.empty(m.shape, dtype=cp.uint8)
                get_kernel("thompson_cold_start", "cold_start_surface")(
                    ((m.size + 255) // 256,), (256,), (a, nwfa, surface, np.int32(m.size)))
                surface_count = int(cp.count_nonzero(surface == 1))
                if landmask is None and surface_count:
                    raise ValueError(
                        "mp_physics=28 cold start: "
                        f"{surface_count} cloudy cell(s) carry no aerosol, and real.exe's make_DropletNumber then "
                        "sizes the droplets by land or water (XLAND), but this "
                        "initialization was given no target LANDMASK; pass "
                        "landmask=<static LANDMASK> to initialize_real")
                if landmask is not None:
                    land = cp.asarray(landmask, dtype=cp.float64).ravel()
                    xland = cp.where(land[idx % land.size] >= 0.5, 1.0, 2.0).astype(cp.float32)
            fixed, volume = gathered_numbers(name, m, n, a, temp, nwfa, xland)
            # Refuse before publishing. Existing non-seeded numbers are validated too.
            if not defer_result_validation:
                fb = fixed.view(cp.uint32)
                bad_fixed = bool((~cp.isfinite(fixed) | (fb > cp.uint32(0x80000000))).any())
                bad_old = bool(((~cp.isfinite(number) | (nb > cp.uint32(0x80000000))) & ~seed).any())
                if bad_fixed or bad_old:
                    raise ValueError(
                        f"the cold-start moment closure produced a non-finite or negative {number_field}; "
                        f"the analyzed {mass_field} it was closed over is not a state the scheme can start from")
            number.ravel()[idx] = fixed
            written.append(number_field)
            entry["repaired_cells"] = repairs
            if repairs and receipt:
                entry["repaired_number_min"] = float(fixed[offenders].min())
                entry["repaired_number_max"] = float(fixed[offenders].max())
            total += repairs
            args = ([v.get() for v in (m, a, volume, temp, nwfa, xland)]
                    if receipt else [None] * 6)
            seeds[name] = _seed_receipt(name, count, *args, build_receipt=receipt)
        else:
            seeds[name] = _seed_receipt(name, 0, None, None, None, None, None, None,
                                        build_receipt=receipt)
        entries.append(entry)
    return {
        "schema": COLD_START_MOMENT_CLOSURE_SCHEMA,
        "droplet_number_seed": seeds.get("cloud"), "rain_number_seed": seeds.get("rain"),
        "ice_number_seed": seeds.get("ice"), "repaired": True, "repaired_cells_total": total,
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


# Bound scratch independently of the grid. Only candidate number arrays
# span the domain, so a refusal cannot publish half a species.
COLD_START_CHUNK_CELLS = 1048576


def _merge_counts_and_extrema(target, source):
    for key, value in source.items():
        if key.endswith("_cells") or key in ("offending_cells", "repaired_cells"):
            target[key] = target.get(key, 0) + value
        elif key.endswith("_min") or key.endswith("_max"):
            if key not in target:
                target[key] = value
            elif np.isnan(value):
                target[key] = value
            elif np.isnan(target[key]):
                pass
            else:
                target[key] = (min if key.endswith("_min") else max)(target[key], value)
        else:
            target[key] = value


def thompson_cold_start_moment_closure(state, state_xp, cfg, inverse_density,
                                     *, aerosol_number=None, landmask=None,
                                     temperature=None, receipt=True):
    """Close bounded chunks and publish only after the same global validations."""
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.boundary_fields import COLD_START_SEEDED_NUMBERS

    pairs = COLD_START_SEEDED_NUMBERS[int(cfg.mp_physics)]
    size = inverse_density.size
    alt_flat = inverse_density.ravel()
    # Count inverse-density refusals over the whole domain before changing it.
    invalid_count = 0
    for start in range(0, size, COLD_START_CHUNK_CELLS):
        stop = min(size, start + COLD_START_CHUNK_CELLS)
        a = _rounded_alt(alt_flat[start:stop])
        bits = a.view(cp.uint32) & cp.uint32(0x7fffffff)
        invalid_count += int(cp.count_nonzero((bits == 0) | (bits > cp.uint32(0x7f800000))))
    if invalid_count:
        raise ValueError(
            "the initializer's inverse density is zero or not finite in "
            f"{invalid_count} cell(s), so the density Thompson's entry block works in cannot be formed there")
    fields = {q: getattr(state, q).ravel() for q, _ in pairs}
    fields.update({n: getattr(state, n).ravel().copy() for _, n in pairs})
    aggregate = None
    aerosol = None if aerosol_number is None else cp.broadcast_to(cp.asarray(aerosol_number), state.qr.shape).ravel()
    land = None if landmask is None else cp.broadcast_to(cp.asarray(landmask), state.qr.shape)
    if int(cfg.mp_physics) == 28:
        surface_count = nan_count = 0
        for start in range(0, size, COLD_START_CHUNK_CELLS):
            stop = min(size, start + COLD_START_CHUNK_CELLS)
            mass, number = fields["qc"][start:stop], fields["nc"][start:stop]
            seed = _seed_mask(mass, number)
            idx = cp.flatnonzero(seed)
            if not idx.size:
                continue
            a = _rounded_alt(alt_flat[start:stop])[idx]
            w = cp.zeros_like(a) if aerosol is None else aerosol[start:stop][idx].astype(cp.float32)
            surface = cp.empty(a.shape, dtype=cp.uint8)
            get_kernel("thompson_cold_start", "cold_start_surface")(
                ((a.size + 255) // 256,), (256,), (a, w, surface, np.int32(a.size)))
            surface_count += int(cp.count_nonzero(surface == 1))
            nan_count += int(cp.count_nonzero(surface == 2))
        if land is None and surface_count:
            raise ValueError(
                "mp_physics=28 cold start: "
                f"{surface_count} cloudy cell(s) carry no aerosol, and real.exe's make_DropletNumber then "
                "sizes the droplets by land or water (XLAND), but this "
                "initialization was given no target LANDMASK; pass "
                "landmask=<static LANDMASK> to initialize_real")
        if nan_count:
            # The host's NaN NINT becomes INT64_MIN and its table index
            # wraps to INT64_MAX. Preserve that existing refusal verbatim.
            raise IndexError(
                "index 9223372036854775807 is out of bounds for axis 0 with size 15")
    # Complete and validate each species in host order. This preserves which
    # refusal fires when multiple species are invalid in different chunks.
    for q, n in pairs:
        name = {"qc": "cloud", "qr": "rain", "qi": "ice"}[q]
        if temperature is None and name != "cloud":
            count = sum(int(cp.count_nonzero(_seed_mask(
                fields[q][start:start + COLD_START_CHUNK_CELLS],
                fields[n][start:start + COLD_START_CHUNK_CELLS])))
                for start in range(0, size, COLD_START_CHUNK_CELLS))
            if count:
                raise ValueError(
                    f"Thompson cold start: {count} cell(s) carry {name} mass "
                    "and no number, and real.exe's make_"
                    f"{name.capitalize()}Number sizes them by temperature, "
                    "but the closure was given none")
        for start in range(0, size, COLD_START_CHUNK_CELLS):
            stop = min(size, start + COLD_START_CHUNK_CELLS)
            chunk = SimpleNamespace(**{k: v[start:stop] for k, v in fields.items()})
            t = (lambda idx, start=start: temperature(idx + start)) if callable(temperature) else (
                None if temperature is None else temperature.ravel()[start:stop])
            lm = None if land is None else land[cp.unravel_index(
                cp.arange(start, stop, dtype=cp.int64), land.shape)]
            part = _closure_chunk(chunk, cp, cfg, alt_flat[start:stop],
                                  temperature=t, landmask=lm, only_pair=(q, n),
                                  defer_result_validation=True, receipt=receipt,
                                  aerosol_number=None if aerosol is None else aerosol[start:stop])
            if aggregate is None:
                aggregate = part
            else:
                aggregate["repaired_cells_total"] += part["repaired_cells_total"]
                src = part["species"][0]
                matches = [dst for dst in aggregate["species"] if dst["species"] == src["species"]]
                if matches:
                    _merge_counts_and_extrema(matches[0], src)
                else:
                    aggregate["species"].append(src)
                for key in ("droplet_number_seed", "rain_number_seed", "ice_number_seed"):
                    if part[key] is not None:
                        if aggregate[key] is None:
                            aggregate[key] = part[key]
                        else:
                            _merge_counts_and_extrema(aggregate[key], part[key])
                for field in part["written_state_fields"]:
                    if field not in aggregate["written_state_fields"]:
                        aggregate["written_state_fields"].append(field)
        if n in aggregate["written_state_fields"]:
            bad = False
            for start in range(0, size, COLD_START_CHUNK_CELLS):
                value = fields[n][start:start + COLD_START_CHUNK_CELLS]
                bits = value.view(cp.uint32)
                bad |= bool((~cp.isfinite(value) | (bits > cp.uint32(0x80000000))).any())
            if bad:
                raise ValueError(
                    f"the cold-start moment closure produced a non-finite or negative {n}; "
                    f"the analyzed {q} it was closed over is not a state the scheme can start from")
    for n in aggregate["written_state_fields"]:
        getattr(state, n).ravel()[...] = fields[n]
    return aggregate
