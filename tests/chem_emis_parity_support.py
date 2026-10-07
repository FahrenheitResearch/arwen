"""Shared replay of recorded WRF driver inputs, without reference physics."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from gpuwm.verify.chem_oracle import load, ulp_table
from gpuwm.core.chem_dust import pack_dust_rows, launch_dust_gocart, launch_dust_afwa
from gpuwm.core.chem_seasalt import pack_seasalt_rows, launch_seasalt

ROOT = Path(__file__).resolve().parents[1] / "tests/data/oracles/chem/gocart"


def check_pins(process):
    root = ROOT / process
    lines = (root / "oracle-sha256sums.txt").read_text().splitlines()
    assert lines
    for line in lines:
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    actual = {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.suffix == ".bin" or p.name == "MANIFEST.txt"}
    assert actual == {name.removeprefix("./") for _, name in (line.split(maxsplit=1) for line in lines)}


def check_case_coverage(process):
    cases = load(ROOT / process)
    for name, f in cases.items():
        if int(f["step"]) > 1:
            prev = cases[name[:-1] + str(int(f["step"])-1)]
            for prefix in ("chem", "edust", "eseas", "tot_edust"):
                assert np.array_equal(f[prefix+"_before"], prev[prefix+"_after"])
    if process == "afwa":
        # run_gocart_afwa.F90's covering table: (dsr, veg, soils, smois,
        # sf_surface_physics) per variant; tuning is recorded separately.
        switches = {tuple(map(int, f["switches"])) for f in cases.values()}
        assert len(switches) == 13  # variants 1 and 12 differ only in tuning
        step1 = {int(n.split("_")[1]): f for n, f in cases.items() if n.endswith("_step_1")}
        assert sorted(step1) == list(range(14))
        after = {v: f["chem_after"] for v, f in step1.items()}
        # dust_smois changes the answer under the volumetric arms (2, 3, 7)
        for gravimetric, volumetric in ((0, 1), (2, 3), (4, 5)):
            assert np.any(after[gravimetric] != after[volumetric]), volumetric
        # surface 2 and 3 take different drylimit formulas; 3 and 7 share one
        assert np.any(after[1] != after[3])
        assert np.array_equal(after[3], after[5])
        # each switch moved alone from the volumetric Noah base moves the answer
        for variant in (8, 9, 10, 11, 12):
            assert np.any(after[variant] != after[1]), variant
        assert np.any(after[13] != after[1])
    if process in ("afwa", "dust"):
        f = cases["variant_000_step_1"]
        assert set(f["isltyp"].ravel()) == set(range(1,20))
        assert np.any(f["smois"] == 0)
        assert np.any(f["xland"] > 1.5) and np.any(f["xland"] < 1.5)
        assert np.any(f["dz8w"] < 12) and np.any(f["dz8w"] > 12)
    else:
        f = cases["variant_000_step_1"]
        assert {0.,3.,10.,25.}.issubset(set(f["u10"].ravel()))
        assert set(float(f["dt"]) for f in cases.values()) == {30.,60.}
        # Elevated lake and land are unchanged, including emission sentinel.
        for i in (3,4):
            assert np.array_equal(f["chem_before"][i], f["chem_after"][i])
            assert np.array_equal(f["eseas_before"][i], f["eseas_after"][i])


def replay(process):
    import cupy as cp
    results = {}
    cases = load(ROOT / process)
    aggregate = {}
    for name, f in cases.items():
        nrows = 4 if process == "seasalt" else 5
        start = 15 if process == "seasalt" else 10
        species = [cp.asarray(np.ascontiguousarray(f["chem_before"][:,:,:,start+n].transpose(1,2,0))) for n in range(nrows)]
        ek = "eseas" if process == "seasalt" else "edust"
        emissions = [cp.asarray(np.ascontiguousarray(f[ek+"_before"][:,0,:,n+1].T)) for n in range(nrows)]
        nc = species[0].shape[1:]
        def field(key):
            a=f[key]
            if a.ndim == 2: a=a.T
            elif a.ndim == 3:
                if key in ("erod", "erod_dri"): a=a.transpose(2,1,0)
                else: a=a.transpose(1,2,0)
            return cp.asarray(np.ascontiguousarray(a))
        common = {k:field(k) for k in ("xland", "u10", "v10", "dz8w")}
        common["dt"] = float(f["dt"])
        common["g"] = float(f["g"])
        if process == "seasalt":
            params=pack_seasalt_rows([dict(ra=a,rb=b,den_seas=2200.) for a,b in zip((.1,.5,1.5,5.),(.5,1.5,5.,10.))])
            common.update({k:field(k) for k in ("z_at_w", "p8w", "u_phy", "v_phy")})
            # Oracle's arrays include three W levels; species include three mass
            # levels. The driver only reads the lowest two W levels.
            for k in ("z_at_w", "p8w"):
                common[k]=cp.concatenate((common[k],common[k][-1:]),axis=0)
            launch_seasalt(species,emissions,params,dx=float(f["dx"]),**common)
        else:
            common.update({k:field(k) for k in ("isltyp", "erod", "rho_phy")})
            common["smois"]=cp.asarray(np.ascontiguousarray(f["smois"][:,0,:].T))
            if process == "dust":
                params=pack_dust_rows([dict(den_dust=d,reff_dust=r,frac_s=s,ipoint=i,ch_dust=.8e-9) for d,r,s,i in zip((2500.,2650.,2650.,2650.,2650.),(.73e-6,1.4e-6,2.4e-6,4.5e-6,8.e-6),(.1,.25,.25,.25,.25),(3,2,2,2,2))])
                common.update({k:field(k) for k in ("u_phy", "v_phy")})
                launch_dust_gocart(species,emissions,params,**common)
            else:
                params=pack_dust_rows([dict(den_dust=rho,reff_dust=r,distr_dust=d,vis_coefficient=e) for rho,r,d,e in zip((2500.,2650.,2650.,2650.,2650.),(.73e-6,1.4e-6,2.4e-6,4.5e-6,8.e-6),(.1074,.1012,.2078,.4817,.1019),(1.470e-6,7.877e-7,4.623e-7,2.429e-7,1.387e-7))])
                common.update({k:field(k) for k in ("erod_dri", "snowh", "vegfra", "lai_vegmask", "ust", "znt", "clay_wrf", "sand_wrf", "clay_nga", "sand_nga")})
                outputs={k:cp.zeros_like(common["rho_phy"] if k in ("tot_dust", "vis_dust") else common["xland"]) for k in ("afwa_dustloft", "tot_dust", "vis_dust")}
                outputs["tot_edust"]=field("tot_edust_before")
                common.update(outputs)
                common.update(zip(("dust_dsr", "dust_veg", "dust_soils", "dust_smois", "sf_surface_physics"),map(int,f["switches"])))
                common.update({k:float(f[k]) for k in ("alpha", "gamma", "smtune", "ustune")})
                launch_dust_afwa(species,emissions,params,**common)
        pairs={"chem":(np.stack([a.get().transpose(2,0,1) for a in species],axis=3), f["chem_after"][:,:,:,start:start+nrows]),
               ek:(np.stack([a.get().T for a in emissions],axis=2), f[ek+"_after"][:,0,:,1:nrows+1])}
        if process == "afwa":
            for k,a in outputs.items():
                pairs[k]=(a.get().transpose(2,0,1) if a.ndim==3 else a.get().T, f["tot_edust_after" if k=="tot_edust" else k])
        tables={k:ulp_table(*pair) for k,pair in pairs.items()}
        for output, table in tables.items():
            assert table["max_ulp"] == 0, (process, name, output, table)
        for k,t in tables.items():
            s=aggregate.setdefault(k,dict(max_ulp=0,n_nonzero=0,n=0))
            s["max_ulp"]=max(s["max_ulp"],t["max_ulp"]); s["n_nonzero"]+=t["n_nonzero"]; s["n"]+=t["n"]
        results[name]=tables
    print(process, json.dumps(aggregate,sort_keys=True))
    pool_bytes = cp.get_default_memory_pool().total_bytes()
    assert pool_bytes < 2*1024**3
    print(process, "device_pool_bytes", pool_bytes)
    if os.environ.get("CHEM_MEASURE"):
        Path(process+"-ulp.json").write_text(json.dumps(dict(aggregate=aggregate,cases=results),indent=2))
    else:
        expected=json.loads((ROOT/process/"measured-ulp.json").read_text())
        assert aggregate == expected["aggregate"]
        assert results == expected["cases"]
    return cases
