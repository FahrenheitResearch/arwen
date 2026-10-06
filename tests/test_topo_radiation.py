"""WRF v4.7.1 slope_rad / topo_shading: the device port against WRF itself.

The fixture is WRF's own Fortran, cut verbatim from the v4.7.1 sources and
compiled with WRF's gfortran flags (tools/wrf_topo_radiation_v471_oracle).
It was run twice: linked against a correctly rounded float libm
(``cr``), which is how the port evaluates its transcendentals, and against
stock glibc (``glibc``), which is how wrf.exe runs.  The port is held bit
for bit to the first; the second measures the libm seam and pins it, so a
change that widens it is seen.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

FIXTURE = (Path(__file__).parent / "fixtures" / "wrf_topo_radiation_v471"
           / "topo_radiation_v471.npz")


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _names(fx, prefix, depth):
    return sorted({"/".join(k.split("/")[:depth]) for k in fx.files
                   if k.startswith(prefix + "/")})


def _ulp(a, b):
    a = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    return np.abs(a - b)


def test_fixture_covers_every_branch(fx):
    """The cases reach what the port has to get right."""
    slope_names = _names(fx, "slope", 2)
    assert "slope/alps_geogrid" in slope_names
    flat = sum(int((fx[f"{n}/cr/slope"] == 0).sum()) for n in slope_names)
    steep = sum(int((fx[f"{n}/cr/slope"] > 0.3).sum()) for n in slope_names)
    negative_cosa = sum(int((fx[f"{n}/cosa"] < 0).sum())
                        for n in slope_names)
    assert flat and steep and negative_cosa
    shadowed = [int(fx[k].sum()) for k in fx.files
                if k.endswith("/cr/shadowmask")]
    assert sum(1 for s in shadowed if s) >= 10
    raised = [k for k in fx.files if k.endswith("/nested1/cr/ht_loc")]
    assert any((fx[k] != fx[k.split("/sun")[0] + "/ht"]).any()
               for k in raised)
    adjusted = fx["adjust/random/cr/swdown"] != fx["adjust/random/swdown"]
    assert adjusted.any() and (~adjusted).any()


@requires_gpu
def test_slope_geometry_matches_wrf_start_em(fx):
    import cupy as cp

    from gpuwm.core.topo_radiation import slope_geometry

    for name in _names(fx, "slope", 2):
        dx = float(np.float32(1.0) / fx[f"{name}/rdx"])
        slope, azi = slope_geometry(
            fx[f"{name}/ht"], fx[f"{name}/msft"], fx[f"{name}/msft"],
            fx[f"{name}/sina"], fx[f"{name}/cosa"], dx=dx, dy=dx)
        # rdx is what start_em.F reads; the port recomputes it from dx,
        # so the round trip must land on the same single-precision value.
        assert np.float32(1.0) / np.float32(dx) == fx[f"{name}/rdx"]
        np.testing.assert_array_equal(cp.asnumpy(slope),
                                      fx[f"{name}/cr/slope"], err_msg=name)
        np.testing.assert_array_equal(cp.asnumpy(azi),
                                      fx[f"{name}/cr/slp_azi"], err_msg=name)


@requires_gpu
def test_terrain_shadow_matches_wrf_toposhad(fx):
    import cupy as cp

    from gpuwm.core.topo_radiation import terrain_shadow

    for base in _names(fx, "shadow", 2):
        for sun in range(8):
            xtime, gmt, radt, declin, dx, dy, shadlen = (
                float(v) for v in fx[f"{base}/sun{sun}/scalars"])
            for nested in (0, 1):
                key = f"{base}/sun{sun}/nested{nested}"
                mask, shad = terrain_shadow(
                    fx[f"{base}/ht"], fx[f"{base}/xlat"],
                    fx[f"{base}/xlong"], fx[f"{base}/sina"],
                    fx[f"{base}/cosa"], xtime_minutes=xtime, gmt=gmt,
                    radt_minutes=radt, declin=declin, dx=dx, dy=dy,
                    shadlen=shadlen,
                    parent_shadow=(fx[f"{base}/ht_shad_in"] if nested
                                   else None))
                np.testing.assert_array_equal(
                    cp.asnumpy(mask), fx[f"{key}/cr/shadowmask"],
                    err_msg=key)
                np.testing.assert_array_equal(
                    cp.asnumpy(shad), fx[f"{key}/cr/ht_shad"], err_msg=key)


@requires_gpu
def test_diffuse_fraction_matches_wrf_radiation_driver(fx):
    import cupy as cp

    from gpuwm.core.topo_radiation import diffuse_fraction

    for ruiz in (0, 1):
        key = f"diffuse/ruiz{ruiz}"
        swddif = cp.asarray(fx[f"{key}/swddif_in"]).copy()
        frac = diffuse_fraction(
            fx[f"{key}/coszen"], fx[f"{key}/swdown"], fx[f"{key}/ht"],
            swddif, solcon=float(fx[f"{key}/solcon"]),
            scheme_splits_direct_beam=not ruiz)
        np.testing.assert_array_equal(cp.asnumpy(swddif),
                                      fx[f"{key}/cr/swddif"], err_msg=key)
        np.testing.assert_array_equal(cp.asnumpy(frac),
                                      fx[f"{key}/cr/diffuse_frac"],
                                      err_msg=key)


@requires_gpu
def test_surface_adjustment_matches_wrf_topo_rad_adj(fx):
    import cupy as cp

    from gpuwm.core.topo_radiation import (adjust_surface_shortwave,
                                           restore_surface_shortwave)

    key = "adjust/random"
    declin, _solcon = (float(v) for v in fx[f"{key}/scalars"])
    swdown = cp.asarray(fx[f"{key}/swdown"]).copy()
    gsw = cp.asarray(fx[f"{key}/gsw"]).copy()
    save = adjust_surface_shortwave(
        swdown, gsw, xlat=fx[f"{key}/xlat"], coszen=fx[f"{key}/coszen"],
        shadowmask=fx[f"{key}/shadowmask"],
        diffuse_frac=fx[f"{key}/diffuse_frac"], hrang=fx[f"{key}/hrang"],
        slope=fx[f"{key}/slope"], slp_azi=fx[f"{key}/slp_azi"],
        declin=declin)
    np.testing.assert_array_equal(cp.asnumpy(swdown), fx[f"{key}/cr/swdown"])
    np.testing.assert_array_equal(cp.asnumpy(gsw), fx[f"{key}/cr/gsw"])
    np.testing.assert_array_equal(cp.asnumpy(save.swnorm),
                                  fx[f"{key}/cr/swnorm"])
    day = fx[f"{key}/swdown"] > np.float32(1e-3)
    np.testing.assert_array_equal(cp.asnumpy(save.gswsave)[day],
                                  fx[f"{key}/cr/gswsave"][day])
    # After the land surface: the flat fluxes come back exactly, SWNORM
    # holds the slope flux (module_surface_driver.F:4461-4481).
    swnorm = restore_surface_shortwave(swdown, gsw, save)
    np.testing.assert_array_equal(cp.asnumpy(swdown), fx[f"{key}/swdown"])
    np.testing.assert_array_equal(cp.asnumpy(gsw), fx[f"{key}/gsw"])
    np.testing.assert_array_equal(
        cp.asnumpy(swnorm)[day], fx[f"{key}/cr/swdown"][day])


def test_glibc_seam_is_the_libm_alone(fx):
    """What stock glibc does differently, pinned.

    The oracle's two builds share every object but the libm, so any
    difference between them is a libm difference.  The pin states its size,
    so a change to the cases or the port that widens it is visible.
    """
    diffs = {}
    for key in fx.files:
        if "/cr/" not in key:
            continue
        other = key.replace("/cr/", "/glibc/")
        a, b = fx[key], fx[other]
        if a.dtype == np.float32:
            ulp = _ulp(a, b)
            if ulp.any():
                diffs[key] = (int((ulp > 0).sum()), int(ulp.max()))
        elif (a != b).any():
            diffs[key] = (int((a != b).sum()), None)
    assert diffs == GLIBC_SEAM, diffs


#: What stock glibc 2.43 (gfortran 15.2, node-2's x86-64, where glibc
#: dispatches its FMA variants of sinf/cosf/expf/powf) gives differently,
#: as ``array: (values that differ, largest difference in ulp)``: 333 of
#: 189,080 values, no shadow mask flipped.  The largest differences are
#: shadow heights at a grazing sun, where tan(topoelev) - tan(asin(csza))
#: magnifies a one-ulp sine into a few hundred ulp of a height.  glibc's
#: sinf, cosf and expf are not correctly rounded (measured on node-2:
#: 1.4 %, 1.2 % and 0.07 % of random arguments differ by one ulp), and
#: which glibc and which CPU a wrf.exe runs on decides which of them it
#: gets, so the port evaluates every transcendental correctly rounded and
#: states the seam here instead of chasing one machine's libm.
GLIBC_SEAM: dict = {
    "adjust/random/cr/gsw": (25, 5),
    "adjust/random/cr/swdown": (27, 3),
    "diffuse/ruiz1/cr/diffuse_frac": (5, 16),
    "diffuse/ruiz1/cr/swddif": (5, 15),
    "shadow/alps_geogrid/sun7/nested0/cr/ht_shad": (10, 10),
    "shadow/alps_geogrid/sun7/nested1/cr/ht_shad": (8, 10),
    "shadow/synthetic0/sun1/nested0/cr/ht_shad": (2, 8),
    "shadow/synthetic0/sun1/nested1/cr/ht_shad": (2, 8),
    "shadow/synthetic0/sun7/nested0/cr/ht_shad": (116, 818),
    "shadow/synthetic0/sun7/nested1/cr/ht_shad": (126, 818),
    "shadow/synthetic1/sun2/nested0/cr/ht_shad": (1, 72),
    "shadow/synthetic1/sun2/nested1/cr/ht_shad": (1, 72),
    "slope/alps_geogrid/cr/slope": (4, 1),
    "slope/synthetic0/cr/slope": (1, 1),
}


def test_every_field_the_port_publishes_has_a_health_policy():
    """The first real topo_shading forecast refused at initialization.

    TopoShortwave publishes its arrays in the driver's surface fields, which
    the health gate walks; the int32 shadow mask had no integer policy, so
    StateHealthValidator raised before step one.  WRF writes it 0 or 1 only
    (module_radiation_driver.F:4505, :4661, :4694, :4712, :4730, :4748).
    """
    from types import SimpleNamespace

    from gpuwm.core.health import (FieldRule, collect_state_fields,
                                   gpu_integer_policy, validate_fields_cpu)

    shape = (3, 4)
    fields = {name: np.zeros(shape, np.float32)
              for name in ("slope", "slp_azi", "diffuse_frac",
                           "topo_coszen", "hrang", "swnorm", "ht_shad")}
    fields["topo_declin"] = np.zeros((1,), np.float32)
    fields["shadowmask"] = np.zeros(shape, np.int32)
    state = SimpleNamespace(physics=SimpleNamespace(fields=fields))
    collected = {f.name: f for f in collect_state_fields(state)}
    for name in fields:
        field = collected[f"surface.{name}"]
        assert gpu_integer_policy(field.name, field.values.dtype) is not None
    mask = collected["surface.shadowmask"]
    assert mask.rule == FieldRule("surface", 0.0, 1.0)
    fields["shadowmask"][1, 2] = 1
    assert validate_fields_cpu([mask]).ok
    fields["shadowmask"][1, 2] = -1
    assert not validate_fields_cpu([mask]).ok


def test_the_carrier_is_rebuilt_and_its_arrays_ride_the_fields():
    """The first real slope_rad forecast ended at its final state digest:
    PhysicsDriver.topo_shortwave was in no restart class.  It is rebuilt by
    initialize_physics; what it evolves lives in the serialized fields."""
    import inspect

    from gpuwm.core.topo_radiation import TopoShortwave
    from gpuwm.io import restart

    assert "topo_shortwave" in restart.DRIVER_REBUILT_ATTRS
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(TopoShortwave)))
    rebound = []
    for method in tree.body[0].body:
        if not isinstance(method, ast.FunctionDef) or method.name == "__init__":
            continue
        for node in ast.walk(method):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(
                           node, (ast.AugAssign, ast.AnnAssign)) else [])
            for target in targets:
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    rebound.append(f"{method.name}: self.{target.attr}")
    # After construction the carrier writes only into its fields mapping.
    assert rebound == [], rebound


def test_the_checkpoint_echo_carries_the_slope_keys_only_when_on():
    """Off, a checkpoint is byte-identical to one written before the keys
    existed; on, all three bind the resume."""
    from gpuwm.io.restart import _drop_inert_topo_radiation

    off = {"slope_rad": 0, "topo_shading": 0, "shadlen": 25000.0, "dt": 9}
    _drop_inert_topo_radiation(off)
    assert off == {"dt": 9}
    flat = {"slope_rad": 1, "topo_shading": 0, "shadlen": 25000.0}
    _drop_inert_topo_radiation(flat)
    assert flat == {"slope_rad": 1}
    shaded = {"slope_rad": 1, "topo_shading": 1, "shadlen": 30000.0}
    _drop_inert_topo_radiation(shaded)
    assert shaded == {"slope_rad": 1, "topo_shading": 1, "shadlen": 30000.0}
