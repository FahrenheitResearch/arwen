"""Word comparisons and a state adapter for compiled WRF diffusion fixtures."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
from gpuwm.core.fp32_ulp import fp32_ulp_distance

def word_comparison(got, expected):
    """Measure all words, including signed zeros and NaN payloads."""
    a, b = (np.ascontiguousarray(x, dtype=np.float32) for x in (got, expected))
    if a.shape != b.shape:
        raise ValueError(f"Output shape changed: {a.shape} versus {b.shape}")
    unequal = a.view(np.uint32) != b.view(np.uint32)
    ulps = fp32_ulp_distance(a, b)
    return {"words": int(a.size), "different_words": int(unequal.sum()),
            "max_ulp": int(ulps.max(initial=0)),
            "max_absolute": float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)), initial=0)),
            "nonfinite_words": int((~np.isfinite(a) | ~np.isfinite(b)).sum())}

def device_state(arrays, *, cf=(1., 0., 0.)):
    """Present fixture device arrays to the unmodified engine launchers."""
    import cupy as cp
    values = {name: cp.asarray(a, dtype=cp.float32) for name, a in arrays.items()}
    values.setdefault("qv", None)
    values.setdefault("physics", None)
    values.setdefault("mup0", cp.zeros(values["alt"].shape[1:], dtype=cp.float32))
    values.setdefault("p", cp.zeros_like(values["alt"]))
    values.setdefault("thp", cp.zeros_like(values["alt"]))
    values.setdefault("thb", cp.full_like(values["alt"], 300.))
    values.update(dict(zip(("cf1", "cf2", "cf3"), cf)))
    cache = {}
    def scratch(shape, slot):
        key = (tuple(shape), slot)
        if key not in cache:
            cache[key] = cp.zeros(shape, dtype=cp.float32)
        return cache[key]
    values["scratch"] = scratch
    return SimpleNamespace(**values)


def launch_diff6_fixture(settings, values, *, identity=False):
    """Launch the engine's sixth-order filter on compiled-oracle inputs."""
    import cupy as cp
    from gpuwm.core.dycore import launch_diff6

    dev = {key: cp.asarray(value) for key, value in values.items()
           if key not in ("reference", "latitude")}
    if identity:
        for name in ("mapu", "mapv", "mapm"):
            dev[name] = cp.ones_like(dev[name])
    tendency = dev["tendency"].copy()
    launch_diff6(
        dev["field"], tendency, dev["mut"], dev["c1"], dev["c2"],
        settings["factor"], settings["dt"], settings["opt"],
        {"u": "x", "v": "y", "w": "z"}.get(settings["name"], ""),
        phb=dev["phb"], msfu=dev["mapu"], msfv=dev["mapv"], msft=dev["mapm"],
        slopeopt=settings["slopeopt"], thresh=settings["thresh"],
        dx=settings["dx"], dy=settings["dy"],
        bnd_x=bool(settings["boundary_mode"]),
        bnd_y=bool(settings["boundary_mode"]))
    if settings["boundary_mode"]:
        tendency[:, :, :3] = 0
        tendency[:, :, -3:] = 0
        tendency[:, :3, :] = 0
        tendency[:, -3:, :] = 0
    return tendency.get()


def launch_constant_fixture(case, values):
    """Launch the declared constant-K operator on compiled-oracle inputs."""
    import cupy as cp
    from gpuwm.core.diffusion import launch_add_diff2

    field = cp.asarray(values["field"])
    tendency = cp.zeros_like(field)
    launch_add_diff2(field, tendency, case["kh"], case["kv"],
                    case["dx"], case["dy"], values["zf"], case["stagger"])
    return (tendency * cp.asarray(values["coupling_mass"])[None]).get()
