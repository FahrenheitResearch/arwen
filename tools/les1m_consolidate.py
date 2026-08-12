"""One readout of every 1 m LES probe artefact that exists."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("/tmp/claude-1000/-home-drew-bowecho-dea/"
           "12456cae-783d-4a37-9cd5-d2db7c7bd8da/scratchpad/out")


def load(name):
    p = OUT / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def hr(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def spectrum_stats(k, p):
    k = np.asarray(k, float)
    p = np.asarray(p, float)
    g = (k > 0) & (p > 0)
    k, p = k[g], p[g]
    if k.size < 8:
        return None
    lo, hi = k.min(), k.max()
    mid = (k >= lo * (hi / lo) ** 0.15) & (k <= lo * (hi / lo) ** 0.55)
    sl = np.polyfit(np.log10(k[mid]), np.log10(p[mid]), 1)[0]
    comp = p * k ** (5.0 / 3.0)
    top = k >= hi / 2.0
    ref = (k >= hi / 16.0) & (k < hi / 4.0)
    pu = float(np.median(comp[top]) / np.median(comp[ref]))
    return {"slope": float(sl), "pileup": pu, "bins": int(k.size),
            "k_lo": float(lo), "k_hi": float(hi)}


# ---------------------------------------------------------------- cost
hr("COST  ns/cell/step, RTX 5090, dry LES rung km_opt=3 unless noted")
rows = []
for f in sorted(OUT.glob("cost_*.json")):
    d = load(f.name)
    if not d or "cost" not in d:
        continue
    for r in d["cost"]:
        r["_file"] = f.name
        rows.append(r)
if rows:
    print(f"{'file':28s} {'km':>3} {'shape':>16} {'Mcell':>8} "
          f"{'ms/step':>9} {'ns/cell':>8} {'spread':>7} {'VRAM GiB':>9} "
          f"{'B/cell':>7} digest")
    for r in rows:
        bpc = r["pool_total_gib"] * 2 ** 30 / r["cells"]
        shape = "{}x{}x{}".format(r["nx"], r["ny"], r["nz"])
        print(f"{r['_file'][:28]:28s} {r['km_opt']:>3} {shape:>16} "
              f"{r['cells'] / 1e6:8.2f} {r['ms_per_step']:9.2f} "
              f"{r['ns_per_cell_step']:8.3f} {r['spread']:7.3f} "
              f"{r['pool_total_gib']:9.2f} {bpc:7.1f} {r['digest']}")

# ------------------------------------------------------------ footprint
hr("FOOTPRINT  resident bytes per cell")
for name in ("footprint_moist.json", "footprint_moist_seeded.json"):
    d = load(name)
    if not d:
        continue
    print(f"-- {name}")
    for r in d["footprint"]:
        print(f"   km_opt={r['km_opt']} carriers={r['n_carriers']:>3} "
              f"carrier B/cell={r['carrier_bytes_per_cell']:7.2f}  "
              f"resident B/cell={r['resident_bytes_per_cell']:7.1f}  "
              f"pool={r['pool_total_gib']:.2f} GiB")

# ----------------------------------------------------------- endurance
hr("ENDURANCE  100 000 steps at dt = 6 ms, dx = 1 m")
for name in ("endurance_km3_100k.json", "endurance_km2_100k.json"):
    d = load(name)
    if not d:
        print(f"-- {name}: not present")
        continue
    b = d["endurance"]
    c = b["config"]
    s = b["samples"]
    print(f"\n-- {name}: km_opt={c['km_opt']} {c['nx']}x{c['ny']}x{c['nz']} "
          f"dx={c['dx']} dt={c['dt']} zi0={c['zi0']}")
    print(f"   steps={b['steps_done']}/{b['steps_requested']}  "
          f"sim={b['sim_seconds']:.0f} s  wall={b['wall_seconds']:.0f} s  "
          f"step-only={b['step_only_ns_per_cell_step']:.2f} ns/cell/step")
    print(f"   digest={b['digest']}  final VRAM pool="
          f"{b.get('pool_total_gib', float('nan')):.2f} GiB")
    f = s[-1]
    print(f"   FINAL: w_max={f['w_max']:.3f}  thp_absmax="
          f"{f['thp_absmax']:.3f}  nan={f['nan']}")
    print(f"          mass_drift_rel={f['mass_drift_rel']:.3e}  "
          f"cfl_sound={f['cfl_sound_horiz']:.4f}  "
          f"cfl_vert={f['cfl_vert_adv']:.5f}")
    print(f"          e_res={f['e_res_ml']:.5g}  e_sgs={f['e_sgs_ml']:.5g}  "
          f"sgs_frac={f['sgs_fraction']:.4f}")
    print(f"          basis: {f['e_sgs_basis']}")
    if f.get("wth_res_max_over_qs") is not None:
        print(f"          wth_res/qs={f['wth_res_max_over_qs']:.3f}  "
              f"wth_sgs/qs={f['wth_sgs_max_over_qs']:.3f}  "
              f"sgs_flux_frac={f['sgs_flux_fraction']:.4f}")
    print("   trajectory (step, w_max, e_res, e_sgs, sgs_frac):")
    for r in s[::max(len(s) // 8, 1)]:
        print(f"     {r['step']:>7} {r['w_max']:7.3f} {r['e_res_ml']:10.5g} "
              f"{r['e_sgs_ml']:10.5g} {r['sgs_fraction']:8.4f}")
    st = spectrum_stats(b["spectrum_k"], b["spectrum_power"])
    if st:
        print(f"   w spectrum at z={b['spectrum_height_m']:.1f} m: "
              f"{st['bins']} bins, slope={st['slope']:.3f} (-5/3=-1.667), "
              f"grid-scale pile-up={st['pileup']:.4f} "
              f"({'PILING UP' if st['pileup'] > 1 else 'rolls off'})")
    er = np.asarray(b["e_res_profile"], float)
    es = np.asarray(b["e_sgs_profile"], float)
    z = np.asarray(b["z_mass"], float)
    fr = es / np.maximum(er + es, 1e-30)
    print("   SGS fraction vs height:")
    for zz in (2, 5, 10, 20, 40, 60, 80, 100):
        i = int(np.argmin(np.abs(z - zz)))
        print(f"     z={z[i]:6.1f} m  e_res={er[i]:9.4g} e_sgs={es[i]:9.4g} "
              f"sgs={fr[i]:6.3f}")

# ------------------------------------------------------------- closure
hr("CLOSURE LADDER  one fixed 128 m box, dx = 2 and 4 m")
d = load("closure_ladder.json")
if d:
    for r in d["closure"]:
        st = spectrum_stats(r["spectrum_k"], r["spectrum_power"])
        print(f"-- km_opt={r['km_opt']} dx={r['dx']} m  {r['nx']}^2x{r['nz']} "
              f"dt={r['dt']:.4f} steps={r['steps']} "
              f"sim={r['sim_seconds']:.0f}s")
        print(f"   e_res={r['e_res_ml']:.5g} e_sgs={r['e_sgs_ml']:.5g} "
              f"sgs_frac={r['sgs_fraction']:.4f} w_max={r['w_max']:.3f} "
              f"km_max={r.get('km_max')}")
        if r.get("sgs_flux_fraction") is not None:
            print(f"   wth_res/qs={r['wth_res_max_over_qs']:.3f} "
                  f"sgs_flux_frac={r['sgs_flux_fraction']:.4f}")
        if st:
            print(f"   spectrum slope={st['slope']:.3f} "
                  f"pile-up={st['pileup']:.4f}")
        print(f"   ns/cell/step={r['ns_per_cell_step']:.2f} "
              f"digest={r['digest']}")

# ------------------------------------------------------------- max dt
hr("MAX dt FROM A SPUN-UP TURBULENT STATE (100 m/s mean wind)")
d = load("maxdt_spunup_u100.json")
if d:
    for r in d["maxdt"]:
        print(f"km_opt={r['km_opt']} ts_sound={r['ts_sound']} "
              f"spinup={r['spinup_steps']} probe={r['probe_steps']}")
        print(f"  spin-up health: {json.dumps(r['spinup_health'])}")
        print(f"  LARGEST STABLE dt = {r['largest_stable_dt']}  "
              f"(smallest failing {r['smallest_failing_dt']})")
        for t in sorted(r["trials"], key=lambda x: x["dt"]):
            print(f"    dt={t['dt'] * 1000:7.3f} ms  cfl_adv="
                  f"{t['cfl_horiz_adv']:6.3f}  ok={str(t['ok']):5s}  "
                  f"{t['note']}")

for name, title in (("cfl_fine.json", "FINE dt LADDER (cold start, 100 m/s)"),
                    ("cfl_control.json", "FORCED vs UNFORCED CONTROL")):
    d = load(name)
    if d:
        hr(title)
        for r in d:
            lbl = r.get("label", "")
            print(f"  {lbl:30s} dt={r['dt'] * 1000:6.2f} ms "
                  f"cfl_adv={r['cfl_adv']:5.3f} -> "
                  f"{'FAIL@' + str(r['failed_at']) if r['failed_at'] else 'OK'}"
                  f"  u_max={r['u_max']:.5g}")

hr("DESIGN-POINT ENDURANCE  50 000 steps at 100 m/s")
for f in sorted(OUT.glob("design_point_u100_dt*.json")):
    d = load(f.name)
    if not d:
        continue
    b = d["endurance"]
    s = b["samples"][-1]
    print(f"-- {f.name}: dt={b['config']['dt']} "
          f"cfl_adv={100.0 * b['config']['dt']:.3f}")
    print(f"   steps={b['steps_done']}/{b['steps_requested']} "
          f"final u_max={s['u_max']:.4f} w_max={s['w_max']:.4f} "
          f"nan={s['nan']} mass_drift={s['mass_drift_rel']:.3e}")
