"""Float32 GSL smoke arithmetic, ccpp-physics 3e6660c6.

module_add_emiss_burn.F90:70-159 and rrfs_smoke_wrapper.F90:855-1013.
This is a verification reference, never a production data processing path.
kpbl starts at nz-1 and is capped there: GSL leaves it unwritten when the
PBL is above the column, and kpbl=kte reads wind one mass level out of bounds.
"""
import json
from pathlib import Path
import numpy as np
from gpuwm.core.noahmp_libm import expf, logf, powf, sqrtf

F = np.float32
RULE_PATH = Path(__file__).resolve().parents[1] / "data/chem/fire/fire_type_rules.json"


def rules():
    return json.loads(RULE_PATH.read_text(encoding="utf-8"))


def classify_fire(frac, lat, lon, emission, minimum, data=None):
    """Generic table reader for the rule-edge verification cases."""
    data = rules() if data is None else data
    sums = {}
    for group in data["groups"]:
        value = F(sums.get(group.get("parent"),F(0)))
        for category in group["categories"]:
            value = F(value+F(frac[category-1]))
        sums[group["name"]] = value
    if emission < minimum:
        return 0
    east = F(F(lon) % F(data["longitude_period"]))
    for rule in data["rules"]:
        if "group" in rule:
            match = sums[rule["group"]] > F(rule["gt"])
        elif "longitude_gt" in rule:
            match = east > F(rule["longitude_gt"]) and lat > F(rule["latitude_gt"]) and lat < F(rule["latitude_lt"])
        else:
            match = True
        if match:
            return rule["type"]


def column_mass_change(before, after, dryrho, dz, state_factor=1):
    """Added emission diagnostic, with the CUDA level association."""
    mass=np.zeros(before.shape[1:],np.float32)
    for k in range(min(51,before.shape[0])):
        for ix in np.ndindex(mass.shape):
            x=(k,*ix)
            change=F(F(F(after[x]-before[x])*dryrho[x])*dz[x])
            mass[ix]=F(mass[ix]+F(change/F(state_factor)))
    return mass


def carry_ebu(ebu, coef, restore):
    """GSL scaled-output carry, wrapper.F90:365-370,552-556."""
    out=np.array(ebu,dtype=F,copy=True)
    for k in range(out.shape[0]):
        for ix in np.ndindex(out.shape[1:]):
            x=(k,*ix)
            out[x]=F(out[x]/max(F(1e-4),coef[ix])) if restore else F(out[x]*coef[ix])
    return out


def source_units(kg, msftx, msfty, dx, dy):
    """kg/cell/hour -> ug/m2/s, with the model's map-factor cell area."""
    area=np.empty_like(kg,dtype=F)
    flux=np.empty_like(kg,dtype=F)
    for ix in np.ndindex(kg.shape):
        area[ix]=F(F(F(dx)*F(dy))/F(msftx[ix]*msfty[ix]))
        flux[ix]=F(F(kg[ix]*F(1e9))/F(F(3600)*area[ix]))
    return area,flux


def frp_diagnostic(frp_mw, coef):
    """GSL diagnostic wrapper.F90:620, with MW-to-W at :735."""
    out=np.empty_like(coef,dtype=F)
    for ix in np.ndindex(out.shape):
        out[ix]=F(F(coef[ix]*F(frp_mw[ix]*F(1e6)))*F(1e-6))
    return out


def add_emiss_burn(chem, ebu, rho, dz, coef, hist, hwp, prev, sw, ends,
                  fire_type, dt, time, mode, minimum, scale=1, weight=1, surface=False, state_factor=1):
    out = np.array(chem, dtype=F, copy=True)
    coef, hist = np.array(coef, dtype=F, copy=True), np.array(hist, dtype=F, copy=True)
    data = rules()
    if mode not in (1, 2):
        raise ValueError("ebb_dcycle must be 1 or 2 to prevent reading GSL's unwritten conv")
    if mode == 2:
        for ix in np.ndindex(coef.shape):
            age = max(F(.01), F(F(F(time) / F(3600)) + F(ends[ix] - F(2))))
            curve = next(x for x in data["curves"] if x["type"] == fire_type[ix])
            if curve["kind"] == "lognormal":
                x = F(logf(age) - F(curve["average"]))
                exponent = F(-F(x*x) / F(curve["denominator"]))
                coef[ix] = F(F(F(curve["C"]) / F(F(curve["sigma"]) * age)) * expf(exponent))
            elif curve["kind"] == "hwp":
                for decay in data["night_decay"]:
                    if sw[ix] < F(.1) and age > F(decay["age_gt"]) and hist[ix] > F(decay["cap"]):
                        hist[ix] = F(decay["cap"])
                ratio = min(F(20), max(F(0), F(hwp[ix] / max(F(10), prev[ix]))))
                coef[ix] = F(F(F(scale)*hist[ix])*ratio)
    # GSL kfire_max = 51 (module_add_emiss_burn.F90:50); a shorter column
    # stops at its own top, as the kernel does.
    for k in range(1 if surface else min(51, out.shape[0])):
        for ix in np.ndindex(coef.shape):
            x = (k, *ix)
            emission=F(ebu[x]*F(weight))
            if emission < minimum:
                continue
            conv = F(dt) if mode == 1 else F(coef[ix] * F(dt))
            conv = F(conv / F(rho[x]*dz[x]))
            out[x] = min(F(5000), max(F(0), F(out[x] + F(F(conv*emission)*F(state_factor)))))
    return out, coef, hist


def smoke_prep(t, p, qv, u, v, z, z_at_w, pbl, oro, u10, v10,
               t2m, dpt2m, wetness, totprcp, totprcp_24hrs, swdown,
               snow, hour, method):
    nz = t.shape[0]
    shape = t.shape[1:]
    result = {n: np.zeros(shape, dtype=F) for n in ("uspdavg2d", "windgustpot", "hpbl2d", "hwp")}
    result.update({n: np.zeros(shape, np.int32) for n in ("kpbl", "kpbl_thetav")})
    for ix in np.ndindex(shape):
        kpbl = nz-1
        for k in range(1, nz):
            if z_at_w[(k,*ix)] > pbl[ix]:
                kpbl = min(nz-1, max(2, k+1))
                break
        tv = []
        for k in range(nz):
            x = (k,*ix)
            theta = F(t[x] * powf(F(F(1e5)/p[x]), F(.286)))
            tv.append(F(theta * F(F(1)+F(F(.61)*qv[x]))))
        ktv = 2
        if tv[1] < F(tv[0]+F(.5)):
            for k in range(1,nz):
                ktv = k+1
                if tv[k] > F(tv[0]+F(.5)):
                    break
        sfc = sqrtf(F(F(u10[ix]*u10[ix])+F(v10[ix]*v10[ix])))
        gust, avg = sfc, sfc
        for k in range(1,kpbl+1):
            x = (k,*ix)
            wind = sqrtf(F(F(u[x]*u[x])+F(v[x]*v[x])))
            avg = F(avg+wind)
            d = F(z[x]-oro[ix])
            delta = F(F(wind-sfc)*F(F(1)-min(F(.5),F(d/F(2000)))))
            gust = max(gust,F(sfc+delta))
        avg = F(avg/F(kpbl))
        height = F(z_at_w[(kpbl-1,*ix)]-z_at_w[(0,*ix)])
        pf = F(F(2.5)+F(F(F(hour)*F(2.5))/F(24)))
        deficit = F(t2m[ix]-dpt2m[ix])
        if method in (1,3):
            vent = max(sfc,F(3)) if method == 1 else avg
            a = F(F(F(.022)*max(F(pf-F(F(totprcp[ix]+totprcp_24hrs[ix])*F(1e3))),F(0)))/pf)
            a = F(a*powf(F(F(1)-wetness[ix]),F(.51)))
            a = F(a*powf(F(vent*height),F(.57)))
            a = F(a*powf(min(F(25),max(F(15),deficit)),F(.74)))
            a = F(a*powf(min(F(3),F(F(1)+F(swdown[ix]/F(250)))),F(.18)))
        elif method in (2,4):
            wg = max(gust,F(3)) if method == 2 else max(F(F(1.69)*sfc),F(3))
            se = max(F(F(F(25)-snow[ix])/F(25)),F(0))
            a = F(F(.177)*powf(wg,F(.97)))
            a = F(a*powf(max(deficit,F(15)),F(1.03)))
            a = F(a*powf(F(F(1)-wetness[ix]),F(.4)))
            a = F(a*se)
        else:
            raise ValueError("hwp_method must be 1..4 to prevent silently zero wildfire potential")
        for name,value in (("kpbl",kpbl),("kpbl_thetav",ktv),("uspdavg2d",avg),
                           ("windgustpot",gust),("hpbl2d",height),("hwp",a)):
            result[name][ix] = value
    return result
