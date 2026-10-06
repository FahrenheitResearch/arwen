"""Mosaic UCM against byte-unmodified WRF 4.7.1, including shared grid state.

The gate was mutation-tested by restoring lsm's FRC_URB2D < 0.99 test on
the rural T1 input. The first step failed at grid TSK/HFX/QFX/LH/GRDFLX,
tile state and subsequent carried steps. See receipts/ucm-mutation.txt.

The D1/D2 corrected reductions retain the existing mosaic contract and use
independent one-tile WRF increments. Every remaining word is the raw WRF output.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from conftest import requires_gpu
from gpuwm.core.noah import load_tables, pack_params, _F2D, _F3D
from gpuwm.core.noah_mosaic import (
    launch_noah_mosaic, load_mosaic_categories, MOSAIC_TILE_FIELDS,
    MOSAIC_SOIL_FIELDS, MOSAIC_URBAN_TILE_FIELDS, MOSAIC_URBAN_SOIL_FIELDS)
from gpuwm.core.urban_ucm import (
    UCM_STATE_PLANES, UCM_SWITCHES, UCM_TABLE_COLUMNS, UCM_GLOBAL_LAYOUT,
    pack_params as pack_ucm, pack_params_from_rows)
from gpuwm.core.urban_tables import load_urban_params
from gpuwm.verify.noah_mosaic_oracle import (
    load, fixture_device_fields, wrf_to_gpuwm, gpuwm_to_wrf, soil_reduction, ulp_table)

ROOT = Path(__file__).resolve().parents[1] / "tests/data/oracles/noah_mosaic"
D2 = ("sfcrunoff", "udrunoff", "potevp", "acsnom", "snopcx", "acsnow")
GRID = tuple(n for n in _F2D if n != "reslin") + _F3D + ("ivgtyp", "isltyp", "ust", "mosaic_cat_index", "landusef2")
TILES = MOSAIC_TILE_FIELDS + MOSAIC_SOIL_FIELDS + MOSAIC_URBAN_TILE_FIELDS + MOSAIC_URBAN_SOIL_FIELDS
OVERRIDES = ("u10", "v10", "psim", "psih", "gz1oz0", "akhs", "akms")


def _wrf_name(n):
    return "ts_rul_urb2d_mosaic" if n == "ts_rul2d_mosaic" else n


def replay(root=ROOT, families=("ucm", "ucm_lcz"), production_frc=False):
    """Every WRF output word of ``families``.

    ``production_frc`` takes FRC_URB2D from gpuwm's own urban_var_init on
    the fixture's dominant categories (requiring it equal WRF's word for
    word first) instead of the dumped array: the default rule's whole path,
    for the ``*_wrfinit`` families whose fraction is the one WRF's
    initialization leaves.
    """
    import cupy as cp
    from gpuwm.core.urban_ucm import after_surface_diagnostics
    categories = load_mosaic_categories("MODIFIED_IGBP_MODIS_NOAH", isurban=13, iswater=17, isice=15)
    noah = pack_params(load_tables())
    summary, differences = {}, []
    for family in families:
        steps = sorted(load(root / family).values(), key=lambda f: int(f["itimestep"]))
        first = steps[0]
        lcz = int(first["use_wudapt_lcz"])
        table={n:first["ucm_table_"+n] for n in (*UCM_TABLE_COLUMNS,"frc_urb")}
        switches={n:first["ucm_"+n] for n in UCM_SWITCHES}
        switches.update({n:first["ucm_global_"+n] for n,_ in UCM_GLOBAL_LAYOUT})
        params=pack_params_from_rows(table,switches)
        parsed=pack_ucm(load_urban_params(1,lcz))
        assert params.table.tobytes()==parsed.table.tobytes()
        assert params.globals_.tobytes()==parsed.globals_.tobytes()
        assert int(first["ucm_iri_scheme"])==0 and first["ucm_oasis"]==np.float32(1)
        dev = fixture_device_fields(first)
        for n in (*MOSAIC_URBAN_TILE_FIELDS, *MOSAIC_URBAN_SOIL_FIELDS, "ust", *OVERRIDES):
            dev[n] = cp.asarray(wrf_to_gpuwm(first[_wrf_name(n) + "_in"]))
        uf = {n: cp.asarray(wrf_to_gpuwm(first[n + "_in"])) for n, _ in UCM_STATE_PLANES}
        uf["frc_urb2d"] = cp.asarray(wrf_to_gpuwm(first["frc_urb2d_in"]))
        if production_frc:
            from test_noah_mosaic_urban_canopy_rule import _production_init
            frc = _production_init(first)["frc_urb2d"]
            assert frc.view(np.uint32).tobytes() == uf["frc_urb2d"].get().view(np.uint32).tobytes()
            uf["frc_urb2d"] = cp.asarray(frc)
        urban = SimpleNamespace(**uf, fields=uf, params=params,
                                urban_mask=cp.asarray((first["ivgtyp_in"].T == 13) | (first["ivgtyp_in"].T >= 51)))
        for f in steps:
            forcing = fixture_device_fields(f)
            for n in "psfc sfcprs sfctmp qv1 qgh dz8w1 glw swdown rainbl sr chs rib vegfra shdmin shdmax tmn xland xice snoalb".split():
                dev[n][...] = forcing[n]
            solar = SimpleNamespace(hrang=cp.asarray(wrf_to_gpuwm(f["omg_urb2d_in"])), jmonth=1)
            atmosphere = {n: cp.asarray(wrf_to_gpuwm(f[n + "_phy_in"])) for n in ("u", "v")}
            before = {n: gpuwm_to_wrf(dev[n]) for n in D2}
            launch_noah_mosaic(dev, noah, float(f["dt"]), f["dzs_in"], mosaic_cat=3,
                categories=categories, xice_threshold=.5, frpcpn=bool(f["frpcpn"]),
                usemonalb=bool(f["usemonalb"]), rdlai2d=bool(f["rdlai2d"]),
                opt_thcnd=int(f["opt_thcnd"]), itimestep=int(f["itimestep"]),
                urban=urban, atmosphere=atmosphere, solar=solar, use_wudapt_lcz=lcz)
            cp.cuda.Stream.null.synchronize()
            area=f["landusef2_in"][:, :3, :]
            actuals={n: gpuwm_to_wrf(dev[n]) for n in (*GRID,*TILES)}
            actuals.update({n:gpuwm_to_wrf(uf[n]) for n,_ in UCM_STATE_PLANES})
            for n, actual in actuals.items():
                wn=_wrf_name(n)
                if wn+"_out" not in f:
                    assert n in ("psfc", "sfcprs", "sfctmp", "qv1", "dz8w1")
                    expected=gpuwm_to_wrf(forcing[n])
                else:
                    expected=f[wn+"_out"]
                if n in ("mosaic_cat_index", "landusef2"):
                    expected=expected[:,:3,:]
                if n in ("smois","tslb","sh2o"):
                    expected=soil_reduction(f[n+"_mosaic_out"],area)
                if n in D2:
                    inc=f["increment_"+n]
                    total=np.zeros_like(before[n])
                    for t in range(2,-1,-1):
                        value=-inc[:,t,:] if n=="snopcx" else inc[:,t,:]
                        total=total+value*area[:,t,:]
                    expected=before[n]-total if n=="snopcx" else before[n]+total
                    if n=="acsnow":expected=before[n]+inc[:,0,:]
                entry=summary.setdefault(family+"/"+n,dict(max_ulp=0,n_nonzero=0,n=0))
                measured=ulp_table(actual,expected)
                entry["max_ulp"]=max(entry["max_ulp"],measured["max_ulp"])
                entry["n_nonzero"]+=measured["n_nonzero"];entry["n"]+=measured["n"]
                bad=actual.view(np.uint32)!=expected.view(np.uint32)
                if np.any(bad):
                    at=tuple(np.argwhere(bad)[0])
                    differences.append(dict(family=family,step=int(f["itimestep"]),field=n,index=list(map(int,at)),
                        wrf_word=f"0x{int(expected.view(np.uint32)[at]):08x}",
                        cuda_word=f"0x{int(actual.view(np.uint32)[at]):08x}"))
            # Existing production override, still keyed on dominant grid category.
            for n in ("t2","th2","q2"):
                dev.setdefault(n,cp.zeros_like(dev["tsk"]))
            after_surface_diagnostics(urban,lsm=2,fields=dev,atmosphere=atmosphere,cfg=None)
            for n in OVERRIDES:
                actual=gpuwm_to_wrf(dev[n]);expected=f[n+"_override"]
                measured=ulp_table(actual,expected)
                entry=summary.setdefault(family+"/override/"+n,dict(max_ulp=0,n_nonzero=0,n=0))
                entry["max_ulp"]=max(entry["max_ulp"],measured["max_ulp"])
                entry["n_nonzero"]+=measured["n_nonzero"];entry["n"]+=measured["n"]
                if not np.array_equal(actual.view(np.uint32),expected.view(np.uint32)):
                    differences.append(dict(family=family,step=int(f["itimestep"]),field="override/"+n))
    return dict(table=summary,differences=differences)


@requires_gpu
def test_mosaic_ucm_every_output_word():
    result=replay()
    print(json.dumps(result,indent=2))
    assert not result["differences"],result["differences"]
    assert all(v["max_ulp"]==0 and v["n_nonzero"]==0 for v in result["table"].values())


@requires_gpu
def test_wrfs_dominant_urban_rule_every_output_word():
    """The default rule end to end: gpuwm's initialized fraction (the
    canopy at weight 0 in every urban tile of a mostly rural cell) through
    the tile loop, against byte-unmodified WRF with WRF's own fraction."""
    result = replay(families=("ucm_wrfinit", "ucm_lcz_wrfinit"), production_frc=True)
    print(json.dumps(result, indent=2))
    assert not result["differences"], result["differences"]
    assert all(v["max_ulp"] == 0 and v["n_nonzero"] == 0 for v in result["table"].values())
    assert {key.split("/")[0] for key in result["table"]} == {"ucm_wrfinit", "ucm_lcz_wrfinit"}
