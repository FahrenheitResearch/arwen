"""Threaded additive inflation gives the serial version's bytes.

CPU only.  THE BREAKAGE THIS PREVENTS: additive inflation and echo noise
draw every member's every field on the host after each analysis, about
0.5 s per 9 km CONUS draw one after another (two minutes per analysis).
gpuwm.da.perturb.additive_inflation now draws members on threads and
removes the ensemble mean span by span; the noise and the record must be
the serial implementation's, byte for byte.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core import constants as c
from gpuwm.da import perturb
from gpuwm.da.perturb import (SUPPORTED_FIELDS, _clip_draw, _weight_on,
                              boundary_taper, gaussian_random_field)


def serial_reference(priors, cfg, *, seed, leg, scale, weight=None,
                     stream="additive-inflation"):
    """The implementation before threading (ec-da-science d91d4df4b)."""
    members = sorted(int(index) for index in priors)
    noise = {index: {} for index in members}
    fields = []
    for name in cfg.field_names:
        spec = cfg.spec(name)
        attribute = SUPPORTED_FIELDS[name].attribute
        stack = []
        for index in members:
            prior = priors[index]
            target = np.asarray(prior[attribute])
            shape = tuple(int(n) for n in target.shape)
            draw, _ = gaussian_random_field(
                shape, seed=int(seed) + index,
                name=f"{name}/{stream}/leg{int(leg)}",
                dx_km=cfg.dx_km, dy_km=cfg.dy_km,
                length_scale_km=spec.length_scale_km,
                vertical_scale_levels=spec.vertical_scale_levels,
                xp=np, dtype=cfg.compute_dtype, fft_host=True)
            draw = np.asarray(draw, dtype=np.float64)
            taper = boundary_taper(shape[1], shape[2], cfg.rim_width,
                                   kind=cfg.rim_taper, xp=np,
                                   dtype=np.float64)[None]
            if weight is not None:
                taper = taper * _weight_on(weight, shape[1:])[None]
            if spec.mode == "lognormal":
                exponent = (_clip_draw(np, draw, spec.clip_sigmas)
                            * float(spec.amplitude) * scale * taper)
                added = target.astype(np.float64) * np.expm1(exponent)
            else:
                added = draw * float(spec.amplitude) * scale * taper
                if SUPPORTED_FIELDS[name].exner_from_temperature:
                    added = added / (np.asarray(prior["p"], np.float64)
                                     / c.P0) ** c.RCP
            stack.append(added)
        stack = np.stack(stack)
        stack -= stack.mean(axis=0, keepdims=True)
        for slot, index in enumerate(members):
            noise[index][attribute] = stack[slot].astype(
                np.asarray(priors[index][attribute]).dtype)
        fields.append((float(np.sqrt(np.mean(stack ** 2))),
                       float(np.abs(stack).max())))
    return noise, fields


def _case(members=6, nz=5, ny=40, nx=37):
    rng = np.random.default_rng(3)
    priors = {}
    for m in range(members):
        priors[m] = {
            "u": rng.standard_normal((nz, ny, nx + 1)).astype(np.float32),
            "v": rng.standard_normal((nz, ny + 1, nx)).astype(np.float32),
            "thp": rng.standard_normal((nz, ny, nx)).astype(np.float32),
            "qv": (0.01 * rng.random((nz, ny, nx))).astype(np.float32),
            "p": (90000.0 + 100 * rng.random((nz, ny, nx))).astype(np.float32),
        }
    fields = [{"name": n, "amplitude": a, "length_scale_km": 20.0}
              for n, a in (("u", 1.5), ("v", 1.5), ("t", 0.5))]
    fields.append({"name": "qv", "amplitude": 0.05, "length_scale_km": 20.0,
                   "mode": "lognormal", "vertical_scale_levels": 2.0})
    cfg = perturb.PerturbationConfig.from_mapping({
        "dx_km": 3.0, "dy_km": 3.0, "rim_width": 4, "fields": fields})
    return priors, cfg


@pytest.mark.parametrize("weighted", [False, True])
def test_threaded_inflation_is_the_serial_inflation(weighted):
    priors, cfg = _case()
    weight = None
    if weighted:
        weight = np.zeros((40, 37))
        weight[10:25, 5:30] = 1.0
    ref, ref_fields = serial_reference(priors, cfg, seed=7, leg=2, scale=0.25,
                                       weight=weight)
    new, record = perturb.additive_inflation(priors, cfg, seed=7, leg=2,
                                             scale=0.25, weight=weight)
    assert sorted(new) == sorted(ref)
    for index in ref:
        assert sorted(new[index]) == sorted(ref[index])
        for key in ref[index]:
            a, b = ref[index][key], new[index][key]
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    got = [(f["added_rms"], f["added_max_abs"]) for f in record["fields"]]
    assert got == ref_fields
