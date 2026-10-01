"""WRF mosaic GPU replay. All accepted non-held-out output words must match.

D1's corrected grid soil uses final tile outputs at 4*t+ns. Its WRF control
is time-aware: a defective index into a lower tile reads PRE-step state.
The CPU control asserts that WRF matches this and differs from final-only.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from conftest import requires_gpu
from gpuwm.core.noah import load_tables, pack_params, _F2D, _F3D
from gpuwm.core.noah_mosaic import (
    launch_noah_mosaic, load_mosaic_categories, MOSAIC_TILE_FIELDS, MOSAIC_SOIL_FIELDS,
)
from gpuwm.verify.noah_mosaic_oracle import (
    load, fixture_device_fields, gpuwm_to_wrf, soil_reduction, weighted_twin_increments, ulp_table,
)

ROOT = Path(__file__).resolve().parents[1] / "gpuwm/data/noah_mosaic/oracle"
OUTPUTS = tuple(n for n in _F2D if n not in ("reslin", "psfc", "sfcprs", "sfctmp", "qv1", "dz8w1")) + _F3D + MOSAIC_TILE_FIELDS + MOSAIC_SOIL_FIELDS + ("ivgtyp", "isltyp", "mosaic_cat_index", "landusef2")
# SNOW smallest-subnormal input: CUDA FTZ makes SNOW>0 false in driver prep,
# removing the ice-saturation blend. These 23 fields carry that branch change.
# CHS/CHS2/CQS2 smallest-subnormal inputs: FTZ makes RCH zero, changing the
# Penman/evaporation and soil-update branches. WRF has NaNs in this probe;
# these 23 fields include the resulting finite/NaN and NaN-payload changes.
# Keys are (family, fixture, zero-based column, field); values pin every GPU word.
FTZ_PINNED_SHA256 = {
    ('ftz', 'v11_s1', 17, 'canwat'): '79e8c92581aaabe254f3da1d8082ac2ba2bd5ed3951481be0851e39ad75b8aa5',
    ('ftz', 'v11_s1', 17, 'canwat_mosaic'): 'aed0ea72efd247bf46e16edfd81bddb3061c58f6ac44823f1599b8f403467daf',
    ('ftz', 'v11_s1', 17, 'grdflx'): 'b99bb9a2b309dfc6bf6a2c9fab44a6a5f0630ee12b4678425d21fe00ec3df625',
    ('ftz', 'v11_s1', 17, 'grdflx_mosaic'): '821ec7e2c8c3607a351d81fd009358e3631ff8a40ff8f6a03e887f1d389eb282',
    ('ftz', 'v11_s1', 17, 'hfx'): '4f462477df9466ac8a794c76984e44334d2912267ed4b76ac8ecf2883106f8c3',
    ('ftz', 'v11_s1', 17, 'hfx_mosaic'): '0bbe2c377ba1bc81db358e6f57519234639690121f593626f979dd256f33468f',
    ('ftz', 'v11_s1', 17, 'lh'): 'c823c76b7e2aa36979fc15c50eb50fcd4ede886dab4c67966a40301ea441cbee',
    ('ftz', 'v11_s1', 17, 'lh_mosaic'): '3e0c5009a352b8747697d409d22c152c6da79e05bf5a927a7d6bd76d198a4bf2',
    ('ftz', 'v11_s1', 17, 'noahres'): 'ed47fd8515f2215b4126add847b9bbc8f278e385345fc1dbf4ec4c146e92bc9d',
    ('ftz', 'v11_s1', 17, 'potevp'): '9d0e2aa1d6639b2fc94d664048beda48e35cac9c960153e2ae335a1c74bb8431',
    ('ftz', 'v11_s1', 17, 'qfx'): '115552d579bace6e582c442324f789dfee3848de32a11667335417c680fc59fe',
    ('ftz', 'v11_s1', 17, 'qfx_mosaic'): 'f59e8890e9b59d5cf1e4e96ff4dbf69a9dc2086ab248fc9dfdbcf434e8d6019a',
    ('ftz', 'v11_s1', 17, 'qsfc'): 'd8fedd8072d520a448e0c5cf22b0b391df907bc601800c94d2eeef29b2490563',
    ('ftz', 'v11_s1', 17, 'qsfc_mosaic'): 'f238c0000ab11d1473d7e4724bd8eee89df74e3cc69ec44b66ed822bc45c6d7b',
    ('ftz', 'v11_s1', 17, 'sh2o'): 'a50f53d8d9d9d5232090e4c88c33ea95fcdf180d9679c93110dd525c70ee869d',
    ('ftz', 'v11_s1', 17, 'sh2o_mosaic'): '9d82cf8d8d251e888f1c365ad34a6549855673475d0e67501b2234de0830c503',
    ('ftz', 'v11_s1', 17, 'smcrel'): '654301a6a3e603533b1f4ab460a824101b1cc3784c911dc912e33829b16d2931',
    ('ftz', 'v11_s1', 17, 'smois'): 'a50f53d8d9d9d5232090e4c88c33ea95fcdf180d9679c93110dd525c70ee869d',
    ('ftz', 'v11_s1', 17, 'smois_mosaic'): '9d82cf8d8d251e888f1c365ad34a6549855673475d0e67501b2234de0830c503',
    ('ftz', 'v11_s1', 17, 'tsk'): 'e9183c53873470ae4654bf3403921ecea42668a6cf7869462ca83e2ccce5a579',
    ('ftz', 'v11_s1', 17, 'tsk_mosaic'): '93f1a5bc78cf824a1aa1530e1d74410f42d2288b6289679a8250ada53c45b270',
    ('ftz', 'v11_s1', 17, 'tslb'): '1fe3c109c46f6d6655c69b38087772bde67cb41e9656e99c3401f02c0f82af12',
    ('ftz', 'v11_s1', 17, 'tslb_mosaic'): '447e34e02189c91e1ad1f0a4a5c27b59f9f71cae2c55289a5af1912c031536a2',
    ('ftz', 'v11_s1', 24, 'grdflx'): 'a2c70538651a7e9296b097e8c3dfc1b195a945802ffe45aa471868fba6f1042e',
    ('ftz', 'v11_s1', 24, 'grdflx_mosaic'): 'a9f84d616a0fb6ab1f382591bc3494bd63ce24e552aeef845e74b66efee888f0',
    ('ftz', 'v11_s1', 24, 'hfx'): 'a2c70538651a7e9296b097e8c3dfc1b195a945802ffe45aa471868fba6f1042e',
    ('ftz', 'v11_s1', 24, 'hfx_mosaic'): 'a9f84d616a0fb6ab1f382591bc3494bd63ce24e552aeef845e74b66efee888f0',
    ('ftz', 'v11_s1', 24, 'lh'): 'a2c70538651a7e9296b097e8c3dfc1b195a945802ffe45aa471868fba6f1042e',
    ('ftz', 'v11_s1', 24, 'lh_mosaic'): 'a9f84d616a0fb6ab1f382591bc3494bd63ce24e552aeef845e74b66efee888f0',
    ('ftz', 'v11_s1', 24, 'noahres'): 'a2c70538651a7e9296b097e8c3dfc1b195a945802ffe45aa471868fba6f1042e',
    ('ftz', 'v11_s1', 24, 'potevp'): 'a2c70538651a7e9296b097e8c3dfc1b195a945802ffe45aa471868fba6f1042e',
    ('ftz', 'v11_s1', 24, 'qfx'): 'df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119',
    ('ftz', 'v11_s1', 24, 'qfx_mosaic'): '15ec7bf0b50732b49f8228e07d24365338f9e3ab994b00af08e5a3bffe55fd8b',
    ('ftz', 'v11_s1', 24, 'qsfc'): 'a2c70538651a7e9296b097e8c3dfc1b195a945802ffe45aa471868fba6f1042e',
    ('ftz', 'v11_s1', 24, 'qsfc_mosaic'): 'a9f84d616a0fb6ab1f382591bc3494bd63ce24e552aeef845e74b66efee888f0',
    ('ftz', 'v11_s1', 24, 'sh2o'): 'a5dfb1bab90059ea04bcaff91a0aff013946ce001e7d0b4bbc0332dc844bd7e9',
    ('ftz', 'v11_s1', 24, 'sh2o_mosaic'): '6144926af4fa5e048495ac7ae61f48853860aa28e5b4dfb1753c642161ca3873',
    ('ftz', 'v11_s1', 24, 'smcrel'): 'f6bb1294da2f78cd935b01c7656280df5eaa0439e9d97bc03775825a41a508e4',
    ('ftz', 'v11_s1', 24, 'smois'): 'a5dfb1bab90059ea04bcaff91a0aff013946ce001e7d0b4bbc0332dc844bd7e9',
    ('ftz', 'v11_s1', 24, 'smois_mosaic'): '6144926af4fa5e048495ac7ae61f48853860aa28e5b4dfb1753c642161ca3873',
    ('ftz', 'v11_s1', 24, 'smstav'): 'e00e5eb9444182f352323374ef4e08ebcb784725fdd4fd612d7730540b3e0c8c',
    ('ftz', 'v11_s1', 24, 'smstot'): '08abb60a845468a5d5c7e80336e40a33b34a46f10fed75c8f37419c81d1ce15f',
    ('ftz', 'v11_s1', 24, 'tsk'): 'a2c70538651a7e9296b097e8c3dfc1b195a945802ffe45aa471868fba6f1042e',
    ('ftz', 'v11_s1', 24, 'tsk_mosaic'): 'a9f84d616a0fb6ab1f382591bc3494bd63ce24e552aeef845e74b66efee888f0',
    ('ftz', 'v11_s1', 24, 'tslb'): 'c66856d3622ef1af1acca8762554ca873b9ea946fc38967d2d42caba297dc8d7',
    ('ftz', 'v11_s1', 24, 'tslb_mosaic'): 'ffab7814bcf2dd3549e71a339a47f62ecee7c4e26e40bc6b2f7b344beecf3d96',
}

D2 = ("sfcrunoff", "udrunoff", "potevp", "acsnom", "snopcx", "acsnow")


def replay_nopac(stage="nopac"):
    """First qualification rung: warm unfrozen columns, initial base step."""
    import cupy as cp
    fixture = load(ROOT / "base")["v1_s1"]
    soil = fixture["tslb_mosaic_in"]
    snow = fixture["snow_mosaic_in"]
    categories = fixture["mosaic_cat_index_in"][:, :3, :]
    mask = (fixture["xland_in"][:, 0] < np.float32(1.5)) & (fixture["xice_in"][:, 0] < np.float32(.5))
    warm = np.all(soil[:, :, 0] > np.float32(273.15), axis=1)
    warm &= np.all(snow[:, :, 0] == np.float32(0), axis=1)
    warm &= fixture["tsk_mosaic_in"][:, 0, 0] > np.float32(273.15)
    nonglacial = np.all(categories[:, :, 0] != 15, axis=1)
    if stage == "nopac":
        mask &= warm & nonglacial
    elif stage == "snopac":
        mask &= ~warm & nonglacial
    elif stage == "glacial":
        mask &= ~nonglacial
    else:
        raise ValueError(stage)
    cols = np.flatnonzero(mask)
    dev = fixture_device_fields(fixture, cols)
    cats = load_mosaic_categories("MODIFIED_IGBP_MODIS_NOAH", isurban=13, iswater=17, isice=15)
    params = pack_params(load_tables())
    launch_noah_mosaic(dev, params, float(fixture["dt"]), fixture["dzs_in"],
                        mosaic_cat=3, categories=cats, xice_threshold=.5,
                        frpcpn=False, usemonalb=False, rdlai2d=False, opt_thcnd=1, itimestep=1)
    cp.cuda.Stream.null.synchronize()
    twins = list(load(ROOT / "twins").values())
    table = {}
    differences = []
    for field in OUTPUTS:
        expected = fixture[field + "_out"]
        if field in ("mosaic_cat_index", "landusef2"):
            expected = expected[:, :3, :]
        if field in ("tslb", "smois", "sh2o"):
            expected = soil_reduction(fixture[field + "_mosaic_out"], fixture["landusef2_in"][:, :3, :])
        elif field in D2:
            if field == "acsnow":
                dt = np.float32(fixture["dt"])
                frozen = fixture["t3d_in"][:, 0, :] <= np.float32(273.15)
                expected = fixture[field + "_in"] + np.where(frozen, (fixture["rainbl_in"]/dt)*dt, np.float32(0))
            else:
                twins = [load(ROOT / "twins")[f"v{t+8}_s1"] for t in range(3)]
                expected = fixture[field + "_in"] + weighted_twin_increments(twins, fixture["landusef2_in"][:, :3, :], field)
        actual = gpuwm_to_wrf(dev[field])
        expected = expected[cols]
        table[field] = ulp_table(actual, expected)
        mismatch = actual.view(np.uint32) != expected.view(np.uint32)
        if np.any(mismatch):
            at = tuple(int(v) for v in np.argwhere(mismatch)[0])
            differences.append(dict(field=field, column=int(cols[at[0]]), index=at,
                                    wrf_word=f"0x{int(expected.view(np.uint32)[at]):08x}",
                                    cuda_word=f"0x{int(actual.view(np.uint32)[at]):08x}"))
        assert not np.any(np.isnan(actual) & np.isfinite(expected)), field
    return dict(columns=cols.tolist(), table=table, first_differences=differences)


@requires_gpu
def test_nopac_bitwise():
    result = replay_nopac()
    print(json.dumps(result, indent=2))
    assert all(v["max_ulp"] == 0 for v in result["table"].values()), result["first_differences"]


def replay_all(families=("base", "cats", "lcz", "twins", "ftz")):
    """Replay sequential tile state and independently-oracled D2 increments."""
    import cupy as cp
    cats = load_mosaic_categories("MODIFIED_IGBP_MODIS_NOAH", isurban=13, iswater=17, isice=15)
    params = pack_params(load_tables())
    forcing = ("psfc sfcprs sfctmp qv1 qgh dz8w1 glw swdown rainbl sr chs rib "
               "vegfra shdmin shdmax tmn xland xice snoalb").split()
    summary, differences, fixture_tables, difference_pairs, held_records = {}, [], {}, set(), []
    for family in families:
        fixtures = load(ROOT / family)
        groups = {}
        for name, fixture in fixtures.items():
            groups.setdefault(int(fixture["variant"]), []).append((name, fixture))
        for variant, group in groups.items():
            group.sort(key=lambda pair: int(pair[1]["itimestep"]))
            dev = fixture_device_fields(group[0][1])
            dev["reslin"] = cp.full(dev["tsk"].shape, np.float32(-123.5))
            dev["ebal"] = cp.full(dev["tsk"].shape, -714, dtype=cp.int32)
            for name, fixture in group:
                forcing_dev = fixture_device_fields(fixture)
                for n in forcing:
                    dev[n][...] = forcing_dev[n]
                before = {n: gpuwm_to_wrf(dev[n]) for n in D2}
                mc = int(fixture["mosaic_cat"])
                launch_noah_mosaic(dev, params, float(fixture["dt"]), fixture["dzs_in"],
                                    mosaic_cat=mc, categories=cats, xice_threshold=.5,
                                    frpcpn=bool(fixture["frpcpn"]),
                                    usemonalb=bool(fixture["usemonalb"]),
                                    rdlai2d=bool(fixture["rdlai2d"]),
                                    opt_thcnd=int(fixture["opt_thcnd"]), itimestep=int(fixture["itimestep"]))
                cp.cuda.Stream.null.synchronize()
                assert bool(cp.all(dev["reslin"] == np.float32(-123.5)))
                assert bool(cp.all(dev["ebal"] == -714))
                table = {}
                area = fixture["landusef2_in"][:, :mc, :]
                land = (fixture["xland_in"] < np.float32(1.5)) & (fixture["xice_in"] < np.float32(.5))
                for n in OUTPUTS:
                    actual = gpuwm_to_wrf(dev[n])
                    expected = fixture[n + "_out"].copy()
                    if n in ("mosaic_cat_index", "landusef2"):
                        expected = expected[:, :mc, :]
                    if n in ("tslb", "smois", "sh2o"):
                        reduced = soil_reduction(fixture[n + "_mosaic_out"], area)
                        expected.transpose(1, 0, 2)[:, land] = reduced.transpose(1, 0, 2)[:, land]
                    elif n in D2:
                        inc = fixture["increment_" + n]
                        if n == "acsnow":
                            expected = before[n] + inc[:, 0, :]
                        else:
                            total = np.zeros_like(before[n])
                            for t in range(mc-1, -1, -1):
                                value = -inc[:, t, :] if n == "snopcx" else inc[:, t, :]
                                total = total + value*area[:, t, :]
                            expected = before[n] - total if n == "snopcx" else before[n] + total
                    accepted = np.ones(actual.shape[0], dtype=bool)
                    for held in FTZ_PINNED_SHA256:
                        if held[0] == family and held[1] == name and held[3] == n:
                            accepted[held[2]] = False
                    table[n] = ulp_table(actual[accepted], expected[accepted])
                    entry = summary.setdefault(n, dict(max_ulp=0, n_nonzero=0, n=0))
                    entry["max_ulp"] = max(entry["max_ulp"], table[n]["max_ulp"])
                    entry["n_nonzero"] += table[n]["n_nonzero"]
                    entry["n"] += table[n]["n"]
                    mismatch = actual.view(np.uint32) != expected.view(np.uint32)
                    for column in np.unique(np.argwhere(mismatch)[:, 0]):
                        difference_pairs.add((family, name, int(column), n))
                        held_records.append(dict(family=family, fixture=name, column=int(column), field=n,
                                                 gpu_sha256=hashlib.sha256(actual[column].tobytes(order="F")).hexdigest(),
                                                 wrf_words=[f"0x{int(v):08x}" for v in expected[column].view(np.uint32).ravel(order="F")],
                                                 gpu_words=[f"0x{int(v):08x}" for v in actual[column].view(np.uint32).ravel(order="F")],
                                                 **ulp_table(actual[column], expected[column])))
                    if np.any(mismatch):
                        at = tuple(int(v) for v in np.argwhere(mismatch)[0])
                        differences.append(dict(family=family, fixture=name, field=n, column=at[0], index=at,
                                                wrf_word=f"0x{int(expected.view(np.uint32)[at]):08x}",
                                                cuda_word=f"0x{int(actual.view(np.uint32)[at]):08x}"))
                    assert not np.any(np.isnan(actual) & np.isfinite(expected)), (family, name, n)
                fixture_tables[family + "/" + name] = table
    return dict(table=summary, fixture_tables=fixture_tables, first_differences=differences,
                difference_pairs=sorted(difference_pairs), held_records=held_records)


@requires_gpu
def test_every_family_bitwise():
    result = replay_all()
    assert set(map(tuple, result["difference_pairs"])) == set(FTZ_PINNED_SHA256)
    for record in result["held_records"]:
        key = (record["family"], record["fixture"], record["column"], record["field"])
        assert record["gpu_sha256"] == FTZ_PINNED_SHA256[key], key
    assert all(v["max_ulp"] == 0 for v in result["table"].values())
