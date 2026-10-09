"""Generate the WOOF physics/dynamics combo sweep (combos.json).

Reads the integrate/2.8.7 physics registry and the site templates, encodes
the validity rules, and builds a constraint-aware covering array:

  stratum A  every valid pair of option values, plus every valid triple
             of the mp x pbl x sfclay x lsm core; reference: stock WRF v4.6.1
  stratum B  features ported from WRF v4.7.1 (UW PBL, urban, terrain drag,
             implicit vertical advection, Noah mosaic), each non-default value
             paired with every other value; reference: WRF v4.7.1
  stratum C  NOAA WRF 3.9 fork switches, pairwise, on the HRRR suite base
  stratum D  WOOF-own arms with no WRF counterpart (RTE+RRTMGP, analytic
             radiation, SASE), each paired with every core value
  suites     every production suite exactly, plus a WRF-comparable twin
  singles    Grell-Freitas closure members 1..16

Rules learned from stock WRF v4.6.1's own refusals (RECORD.md):

  - mp_physics = 28 needs wif_input_opt = 1: real.exe stops with
    "wif_input_opt=0 but mp_physics=28".  Every mp 28 combo is rendered
    with the whole climatology triple, as the HRRR suite writes it:
    &domains wif_input_opt = 1, num_wif_levels = 30 (the levels of the
    QNWFA_QNIFA_SIGMA_MONTHLY.dat climatology the recordings staged) and
    &physics use_aero_icbc = .true. (WOOF's importer refuses the half
    triple: wif_input_opt = 1 alone allocates the aerosol arrays with
    nothing to fill them).
  - MYNN (bl_pbl_physics = 5) runs with bl_mynn_edmf = 1 here, and
    bl_mynn_edmf > 0 needs ishallow = 0: WRF's namelist check stops a
    Grell-Freitas shallow plume beside the MYNN mass flux.  A constraint.

Usage:
  python gen_combos.py [--seed N]
      rebuilds combos.json from nothing.
  python gen_combos.py [--seed N] --keep-rows-from combos-v1.json
      keeps every generated row of the old list that is still valid, under
      its old id, drops the rows a rule now forbids, and covers what they
      covered with new rows numbered after the old ones.  Suites, twins and
      singles are rebuilt from the registry and templates.  This keeps the
      ids the WRF recordings were made under.
Both are deterministic: the same inputs and seed give the same bytes, in any
process (the covering array no longer depends on Python's string hashing).
Writes combos.json next to this file unless --out says otherwise.
"""
import argparse
import copy
import hashlib
import itertools
import json
import os
import random
import tomllib
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", ".."))
REGISTRY = os.path.join(WT, "gpuwm", "physics_registry_v2.json")
HRRR_RECIPE = os.path.join(WT, "configs", "recipes", "hrrr_v4_gsd41.toml")
SITE = os.environ.get("GPUWM_COMBO_SITE_ROOT")
NA = "na"

# ---------------------------------------------------------------------------
# The option space. First value of every list is the default (ties in the
# greedy fill go to it, so values nobody asked to cover stay at default).
# ---------------------------------------------------------------------------
CORE = OrderedDict([
    ("mp", [8, 1, 6, 9, 10, 16, 18, 28, 50]),
    ("pbl", [1, 0, 2, 5, 11]),
    ("sfclay", [91, 0, 1, 2, 5]),
    ("lsm", [2, 0, 3, 4]),
    ("cu", [0, 1, 3, 16]),
    ("rad", ["rrtmg_legacy", "off", "rrtm_dudhia"]),
    ("turb", ["d2k4", "d1k4", "d1k2", "d2k2", "d2k3", "d2k1"]),
])
DYN = OrderedDict([
    ("v_mom_adv_order", [3, 5]),
    ("v_sca_adv_order", [3, 5]),
    ("damp_opt", [3, 0]),
    ("w_damping", [1, 0]),
    ("hybrid_opt", [2, 0]),
    ("hypsometric_opt", [2, 1]),
    ("epssm", [0.1, 0.5]),
    ("diff_6th_opt", [2, 0]),
    ("top_lid", [False, True]),
    ("mp_zero_out", [0, 1, 2]),
    ("no_mp_heating", [0, 1]),
])
# conditional sub-options: name -> (values, parent keys, active predicate)
SUB = OrderedDict([
    ("diff_6th_slopeopt", ([1, 0], ["diff_6th_opt"], lambda c: c["diff_6th_opt"] != 0)),
    ("moist_mix6_off", ([False, True], ["diff_6th_opt"], lambda c: c["diff_6th_opt"] != 0)),
    ("isftcflx", ([0, 1, 2], ["sfclay"], lambda c: c["sfclay"] in (1, 91))),
    ("iz0tlnd", ([0, 1, 2], ["sfclay"], lambda c: c["sfclay"] in (1, 91))),
    ("ysu_topdown_pblmix", ([1, 0], ["pbl"], lambda c: c["pbl"] == 1)),
    ("bl_mynn_mixlength", ([1, 2], ["pbl"], lambda c: c["pbl"] == 5)),
    ("bl_mynn_mixscalars", ([0, 1], ["pbl"], lambda c: c["pbl"] == 5)),
    ("scalar_pblmix", ([0, 1], ["pbl"], lambda c: c["pbl"] == 5)),
    ("ruc_soil_layers", ([9, 6], ["lsm"], lambda c: c["lsm"] == 3)),
    ("mosaic_lu", ([0, 1], ["lsm"], lambda c: c["lsm"] == 3)),
    ("mosaic_soil", ([0, 1], ["lsm"], lambda c: c["lsm"] == 3)),
    ("fractional_seaice", ([0, 1], ["lsm"], lambda c: c["lsm"] == 3)),
    ("monthly_veg", ([False, True], ["lsm"], lambda c: c["lsm"] in (2, 3))),
    ("rdmaxalb", ([True, False], ["lsm"], lambda c: c["lsm"] == 2)),
    ("opt_thcnd", ([1, 2], ["lsm"], lambda c: c["lsm"] == 2)),
    ("sf_lake_physics", ([0, 1], ["lsm"], lambda c: c["lsm"] != 0)),
    ("ishallow", ([0, 1], ["cu"], lambda c: c["cu"] == 3)),
    ("hail_opt", ([0, 1], ["mp"], lambda c: c["mp"] == 6)),
    ("morr_rimed_ice", ([1, 0], ["mp"], lambda c: c["mp"] == 10)),
    ("nssl_variant", ([18, 17, 22], ["mp"], lambda c: c["mp"] == 18)),
    ("use_mp_re", ([1, 0], ["rad"], lambda c: c["rad"] == "rrtmg_legacy")),
    ("o3input", ([2, 0], ["rad"], lambda c: c["rad"] == "rrtmg_legacy")),
    ("icloud", ([1, 0], ["rad"], lambda c: c["rad"] in ("dudhia", "rrtm_dudhia"))),
    ("swint_opt", ([0, 1], ["rad"], lambda c: c["rad"] in ("rrtmg_legacy", "rte_rrtmgp"))),
    ("slope_rad", ([0, 1], ["rad"], lambda c: c["rad"] not in ("off", "analytic"))),
    ("topo_shading", ([0, 1], ["rad", "slope_rad"],
                      lambda c: c["rad"] not in ("off", "analytic") and c["slope_rad"] == 1)),
])
# stratum B (WRF v4.7.1 ports)
B_EXTRA = OrderedDict([
    ("urban", [0, 1, 2, 3]),
    ("gwd_opt", [0, 1, 3]),
    ("topo_wind", [0, 1, 2]),
    ("zadvect_implicit", [0, 1]),
])
B_SUB = OrderedDict([
    ("w_crit_cfl", ([1.0, 2.0], ["zadvect_implicit"], lambda c: c["zadvect_implicit"] == 1)),
    ("sf_surface_mosaic", ([0, 1], ["lsm"], lambda c: c["lsm"] == 2)),
])
B_NEW_ITEMS = [("pbl", 9), ("urban", 1), ("urban", 2), ("urban", 3), ("gwd_opt", 1),
               ("gwd_opt", 3), ("topo_wind", 1), ("topo_wind", 2),
               ("zadvect_implicit", 1), ("w_crit_cfl", 1.0), ("w_crit_cfl", 2.0),
               ("sf_surface_mosaic", 1)]
D_NEW_ITEMS = [("rad", "rte_rrtmgp"), ("rad", "analytic"), ("rad", "dudhia"), ("pbl", 900),
               ("turb", "k0")]

PBL_SFC = {0: {0, 1, 5, 91}, 1: {1, 91}, 2: {2}, 5: {1, 5, 91}, 11: {1, 91},
           9: {1, 5, 91}, 900: {1, 5, 91}}


def constraints(space):
    """(deps, fn) pairs; fn sees a dict with every dep assigned."""
    cons = [
        (["pbl", "sfclay"], lambda c: c["sfclay"] in PBL_SFC[c["pbl"]]),
        (["sfclay", "lsm"], lambda c: c["sfclay"] != 0 or c["lsm"] == 0),
        (["sfclay", "pbl"], lambda c: c["sfclay"] != 0 or c["pbl"] == 0),
        (["cu", "pbl"], lambda c: c["cu"] != 3 or c["pbl"] != 0),
        (["turb", "pbl"], lambda c: c["turb"] not in ("d2k2", "d2k3") or c["pbl"] == 0),
        (["turb", "pbl"], lambda c: (c["turb"] == "k0") == (c["pbl"] == 900)),
        # WOOF refuses a land surface with both radiation streams off
        (["lsm", "rad"], lambda c: c["lsm"] == 0 or c["rad"] != "off"),
        # MYNN scalar transport: the whole number family or nothing
        (["bl_mynn_mixscalars", "mp"],
         lambda c: c["bl_mynn_mixscalars"] != 1 or c["mp"] in (1, 6, 28)),
        (["scalar_pblmix", "mp"], lambda c: c["scalar_pblmix"] != 1 or c["mp"] == 28),
        (["scalar_pblmix", "bl_mynn_mixscalars"],
         lambda c: c["scalar_pblmix"] != 1 or c["bl_mynn_mixscalars"] != 1),
        # WRF refuses bl_mynn_edmf > 0 (MYNN_IDENTITY) with ishallow = 1
        (["pbl", "ishallow"], lambda c: c["pbl"] != 5 or c["ishallow"] != 1),
    ]
    if "urban" in space:
        cons += [
            (["urban", "lsm"], lambda c: c["urban"] == 0 or c["lsm"] in (2, 4)),
            (["urban", "pbl"], lambda c: c["urban"] not in (2, 3) or c["pbl"] in (1, 2)),
            (["topo_wind", "pbl"], lambda c: c["topo_wind"] == 0 or c["pbl"] == 1),
            (["topo_wind", "urban"], lambda c: c["topo_wind"] == 0 or c["urban"] not in (2, 3)),
            (["gwd_opt", "pbl"], lambda c: c["gwd_opt"] == 0 or c["pbl"] != 0),
            (["sf_surface_mosaic", "urban"],
             lambda c: c["sf_surface_mosaic"] != 1 or c["urban"] not in (2, 3)),
        ]
    for name, spec in space.items():
        if isinstance(spec, tuple):
            values, deps, pred = spec
            cons.append(([name] + deps,
                         (lambda pred, name: lambda c: pred(c) == (c[name] != NA))(pred, name)))
    return cons


class Space:
    def __init__(self, factors):
        self.factors = factors                      # name -> list or (values, deps, pred)
        self.order = list(factors)
        self.domains = {n: (list(s[0]) + [NA] if isinstance(s, tuple) else list(s))
                        for n, s in factors.items()}
        self.cons = constraints(factors)
        self.by_var = {n: [] for n in self.order}
        for deps, fn in self.cons:
            for d in set(deps):
                self.by_var[d].append((deps, fn))

    def values(self, name):
        s = self.factors[name]
        return list(s[0]) if isinstance(s, tuple) else list(s)

    def consistent(self, c, name):
        for deps, fn in self.by_var[name]:
            if all(d in c for d in deps) and not fn(c):
                return False
        return True

    def forward(self, c, doms):
        """Prune domains of unassigned vars by constraints with one free var."""
        changed = True
        while changed:
            changed = False
            for deps, fn in self.cons:
                free = [d for d in set(deps) if d not in c]
                if len(free) != 1:
                    continue
                u = free[0]
                keep = []
                for v in doms[u]:
                    c[u] = v
                    if fn(c):
                        keep.append(v)
                    del c[u]
                if len(keep) != len(doms[u]):
                    if not keep:
                        return False
                    doms[u] = keep
                    changed = True
        return True

    def solve(self, seed):
        c = dict(seed)
        for n in list(c):
            if not self.consistent(c, n):
                return None
        doms = {n: list(self.domains[n]) for n in self.order if n not in c}
        if not self.forward(c, doms):
            return None
        return self._dfs(c, doms)

    def _dfs(self, c, doms):
        free = [n for n in self.order if n not in c]
        if not free:
            return dict(c)
        n = min(free, key=lambda x: len(doms[x]))
        for v in doms[n]:
            c[n] = v
            if self.consistent(c, n):
                d2 = {k: list(vs) for k, vs in doms.items() if k != n}
                if self.forward(c, d2):
                    r = self._dfs(c, d2)
                    if r is not None:
                        return r
            del c[n]
        return None


def tuple_key(t):
    """A total order on required tuples that does not depend on hashing."""
    return tuple(sorted((k, repr(v)) for k, v in t))


def build_cover(space, required, rng, candidates=12, existing=()):
    """Greedy constraint-aware covering (AETG style)."""
    uncovered = set(required)
    for row in existing:
        items = set(row.items())
        uncovered = {t for t in uncovered if not t <= items}
    index = {}
    for t in uncovered:
        for it in t:
            index.setdefault(it, set()).add(t)
    rows = []
    while uncovered:
        best, best_gain = None, -1
        big = [t for t in uncovered if len(t) == 3]
        # sorted: a set of frozensets iterates in string-hash order, which
        # changes from process to process, and rng.choice reads that order
        pool = sorted(big if big else uncovered, key=tuple_key)
        tries = 0
        made = 0
        while made < candidates:
            tries += 1
            if tries > candidates * 20:
                raise RuntimeError("could not build candidates")
            seed = dict(rng.choice(pool))
            if space.solve(seed) is None:
                raise RuntimeError(f"required tuple infeasible: {seed}")
            row = dict(seed)
            doms = {n: list(space.domains[n]) for n in space.order if n not in row}
            if not space.forward(row, doms):
                continue
            order = [n for n in space.order if n not in row]
            rng.shuffle(order)
            dead = False
            for n in order:
                scored = []
                for v in doms[n]:
                    gain = 0
                    for t in index.get((n, v), ()):
                        if t in uncovered and all(row.get(k) == val for k, val in t if k != n):
                            gain += 1
                    pref = 1 if v == space.domains[n][0] else 0
                    scored.append((gain, pref, rng.random(), v))
                scored.sort(reverse=True)
                placed = False
                for gain, pref, _, v in scored:
                    row[n] = v
                    if space.consistent(row, n):
                        d2 = {k: list(vs) for k, vs in doms.items() if k != n and k not in row}
                        if space.forward(row, d2):
                            doms = d2
                            placed = True
                            break
                    del row[n]
                if not placed:
                    dead = True
                    break
            if dead or any(not space.consistent(row, n) for n in space.order):
                continue
            made += 1
            items = set(row.items())
            gain = sum(1 for t in uncovered if t <= items)
            if gain > best_gain:
                best, best_gain = row, gain
        # factor order: the space's, not the order a hashed seed tuple gave
        best = {n: best[n] for n in space.order}
        rows.append(best)
        items = set(best.items())
        uncovered = {t for t in uncovered if not t <= items}
    return rows


def required_pairs(space, items_filter=None):
    names = space.order
    out = set()
    for a, b in itertools.combinations(names, 2):
        for va in space.values(a):
            for vb in space.values(b):
                t = frozenset([(a, va), (b, vb)])
                if items_filter and not items_filter(t):
                    continue
                if space.solve(dict(t)) is not None:
                    out.add(t)
    return out


def required_triples(space, names):
    out = set()
    for trio in itertools.combinations(names, 3):
        for vals in itertools.product(*[space.values(n) for n in trio]):
            t = frozenset(zip(trio, vals))
            if space.solve(dict(t)) is not None:
                out.add(t)
    return out


# ---------------------------------------------------------------------------
# Rendering a factor assignment into the namelist blocks both arms read.
# ---------------------------------------------------------------------------
RAD = {"off": (0, 0), "dudhia": (0, 1), "rrtm_dudhia": (1, 1), "rrtmg_legacy": (4, 4),
       "rte_rrtmgp": (4, 4), "analytic": (90, 90)}
TURB = {"d1k4": (1, 4), "d2k4": (2, 4), "d1k2": (1, 2), "d2k2": (2, 2), "d2k3": (2, 3),
        "d2k1": (2, 1), "k0": (2, 0)}
MYNN_IDENTITY = {"bl_mynn_closure": 2.6, "bl_mynn_cloudpdf": 2, "bl_mynn_edmf": 1,
                 "bl_mynn_edmf_mom": 1, "bl_mynn_edmf_tke": 0, "bl_mynn_cloudmix": 1,
                 "bl_mynn_mixqt": 0, "bl_mynn_output": 0, "bl_mynn_tkeadvect": False,
                 "icloud_bl": 1}
NOAHMP_IDENTITY = {"dveg": 4, "opt_crs": 1, "opt_btr": 1, "opt_run": 3, "opt_sfc": 1,
                   "opt_frz": 1, "opt_inf": 1, "opt_rad": 3, "opt_alb": 2, "opt_snf": 1,
                   "opt_tbot": 2, "opt_stc": 1, "opt_gla": 1, "opt_rsf": 1, "opt_soil": 1,
                   "opt_pedo": 1, "opt_crop": 0, "opt_irr": 0, "opt_irrm": 0,
                   "opt_infdv": 0, "opt_tdrn": 0, "soiltstep": 0.0}
SUITE_RADT_MIN = 1.0   # sweep combos: radiation every minute (fires about every 3 steps)


def g(c, k, default=None):
    v = c.get(k, default)
    return default if v == NA else v


def render(c, *, radt=SUITE_RADT_MIN, cudt=0.0):
    mp = c["mp"]
    if mp == 18 and g(c, "nssl_variant", 18) != 18:
        mp = g(c, "nssl_variant")
    lw, sw = RAD[c["rad"]]
    diff_opt, km_opt = TURB[c["turb"]]
    pbl, sfc, lsm, cu = c["pbl"], c["sfclay"], c["lsm"], c["cu"]
    ph = OrderedDict()
    ph["mp_physics"] = mp
    ph["ra_lw_physics"] = lw
    ph["ra_sw_physics"] = sw
    ph["radt"] = radt
    ph["icloud"] = g(c, "icloud", 1)
    if sw == 1:
        ph["swrad_scat"] = 1.0
    if (lw, sw) == (4, 4):
        ph["use_mp_re"] = g(c, "use_mp_re", 1)
        ph["o3input"] = g(c, "o3input", 2)
        ph["ghg_input"] = 0
        ph["aer_opt"] = c.get("aer_opt", 0)
    if sw in (1, 4):
        ph["swint_opt"] = g(c, "swint_opt", 0)
        ph["slope_rad"] = g(c, "slope_rad", 0)
        ph["topo_shading"] = g(c, "topo_shading", 0)
    ph["sf_sfclay_physics"] = sfc
    if sfc in (1, 91):
        ph["isftcflx"] = g(c, "isftcflx", 0)
        ph["iz0tlnd"] = g(c, "iz0tlnd", 0)
    ph["sf_surface_physics"] = lsm
    ph["num_soil_layers"] = {2: 4, 4: 4, 3: g(c, "ruc_soil_layers", 9), 0: None}[lsm]
    if lsm == 2:
        ph["rdmaxalb"] = g(c, "rdmaxalb", True)
        ph["opt_thcnd"] = g(c, "opt_thcnd", 1)
    if lsm in (2, 3):
        mv = g(c, "monthly_veg", False)
        ph["usemonalb"] = mv
        ph["rdlai2d"] = mv
    if lsm == 3:
        ph["mosaic_lu"] = g(c, "mosaic_lu", 0)
        ph["mosaic_soil"] = g(c, "mosaic_soil", 0)
        ph["fractional_seaice"] = g(c, "fractional_seaice", 0)
        ph["flag_sm_adj"] = 0
    if lsm == 2 and g(c, "sf_surface_mosaic", 0) == 1:
        ph["sf_surface_mosaic"] = 1
        ph["mosaic_cat"] = 3
    ph["sf_urban_physics"] = c.get("urban", 0)
    if ph["sf_urban_physics"]:
        ph["num_urban_hi"] = 15
        ph["use_wudapt_lcz"] = 0
    ph["sf_lake_physics"] = g(c, "sf_lake_physics", 0)
    ph["bl_pbl_physics"] = pbl
    ph["bldt"] = 0.0
    if pbl == 1:
        ph["ysu_topdown_pblmix"] = g(c, "ysu_topdown_pblmix", 1)
    if pbl == 5:
        ph.update(MYNN_IDENTITY)
        ph["bl_mynn_mixlength"] = g(c, "bl_mynn_mixlength", 1)
        ph["bl_mynn_mixscalars"] = g(c, "bl_mynn_mixscalars", 0)
        ph["scalar_pblmix"] = g(c, "scalar_pblmix", 0)
    ph["topo_wind"] = c.get("topo_wind", 0)
    ph["cu_physics"] = cu
    ph["cudt"] = cudt
    if cu == 3:
        ph["ishallow"] = g(c, "ishallow", 0)
        ph["clos_choice"] = c.get("clos_choice", 0)
    ph["cu_rad_feedback"] = False
    ph["isfflx"] = 1
    ph["ifsnow"] = 1
    ph["surface_input_source"] = 1
    ph["num_land_cat"] = 21
    ph["mp_zero_out"] = c["mp_zero_out"]
    ph["mp_zero_out_thresh"] = 1e-8
    ph["no_mp_heating"] = c["no_mp_heating"]
    if mp == 6:
        ph["hail_opt"] = g(c, "hail_opt", 0)
    if mp == 10:
        ph["morr_rimed_ice"] = g(c, "morr_rimed_ice", 1)
    if mp == 28:
        # with wif_input_opt = 1 below: the climatology triple, whole
        ph["use_aero_icbc"] = True
    ph["sst_update"] = 0
    ph["do_radar_ref"] = 1

    dy = OrderedDict()
    dy["hybrid_opt"] = c["hybrid_opt"]
    dy["etac"] = 0.2
    dy["use_theta_m"] = 0
    dy["rk_ord"] = 3
    dy["diff_opt"] = diff_opt
    dy["km_opt"] = km_opt
    # constant K must stay 0 here: WOOF refuses khdif/kvdif > 0 with
    # specified lateral boundaries (the stencils are not boundary aware)
    dy["khdif"] = 0.0
    dy["kvdif"] = 0.0
    dy["c_s"] = 0.25
    dy["c_k"] = 0.15
    if km_opt in (2, 3):
        dy["mix_isotropic"] = 0
        dy["mix_upper_bound"] = 0.1
    if km_opt == 2:
        dy["tke_adv_opt"] = 1
    dy["mix_full_fields"] = diff_opt == 2
    dy["diff_6th_opt"] = c["diff_6th_opt"]
    dy["diff_6th_factor"] = 0.12
    dy["diff_6th_slopeopt"] = g(c, "diff_6th_slopeopt", 0)
    dy["diff_6th_thresh"] = 0.10
    dy["moist_mix6_off"] = g(c, "moist_mix6_off", False)
    dy["damp_opt"] = c["damp_opt"]
    dy["zdamp"] = 5000.0
    dy["dampcoef"] = 0.2
    dy["w_damping"] = c["w_damping"]
    dy["base_temp"] = 290.0
    dy["non_hydrostatic"] = True
    dy["time_step_sound"] = 4
    dy["epssm"] = c["epssm"]
    dy["smdiv"] = 0.1
    dy["emdiv"] = 0.01
    dy["momentum_adv_opt"] = 1
    dy["moist_adv_opt"] = 1
    dy["scalar_adv_opt"] = 1
    dy["h_mom_adv_order"] = 5
    dy["h_sca_adv_order"] = 5
    dy["v_mom_adv_order"] = c["v_mom_adv_order"]
    dy["v_sca_adv_order"] = c["v_sca_adv_order"]
    dy["top_lid"] = c["top_lid"]
    dy["gwd_opt"] = c.get("gwd_opt", 0)
    dy["zadvect_implicit"] = c.get("zadvect_implicit", 0)
    if dy["zadvect_implicit"]:
        dy["w_crit_cfl"] = g(c, "w_crit_cfl", 1.0)

    nl = OrderedDict()
    nl["domains"] = OrderedDict([("hypsometric_opt", c["hypsometric_opt"])])
    if mp == 28:
        # stock real.exe stops "wif_input_opt=0 but mp_physics=28"
        nl["domains"]["wif_input_opt"] = 1
        nl["domains"]["num_wif_levels"] = 30
    nl["physics"] = ph
    if lsm == 4:
        nl["noah_mp"] = OrderedDict(NOAHMP_IDENTITY)
    nl["dynamics"] = dy

    ws = OrderedDict()   # WOOF-only settings (the WRF arm has no such key)
    if c["rad"] == "rrtmg_legacy":
        ws["ra_rrtmg_variant"] = "rrtmg_legacy"
        ws["wrf_rrtmg_compatibility"] = "wrf-rrtmg-4-4-legacy-v1"
    elif c["rad"] == "rte_rrtmgp":
        ws["ra_rrtmg_variant"] = "rte-rrtmgp"
        ws["wrf_rrtmg_compatibility"] = "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"
    elif c["rad"] == "dudhia":
        # stock WRF fatals lw=0 with sw on (module_radiation_driver.F:2245);
        # WOOF runs it only with the constant-longwave acknowledgement
        ws["acknowledgements"] = ["constant-downward-longwave-v1"]
    elif c["rad"] == "analytic":
        ws["ra_physics"] = 0
        ws["ra_lw_physics"] = 90
        ws["ra_sw_physics"] = 90
        ws["wrf_rrtmg_compatibility"] = "none"
    if lsm == 3:
        ws.update(OrderedDict([("ruc_soilprop", "wrf_461"), ("ruc_irrigation", "wrf_461"),
                               ("ruc_snow", "wrf_461"), ("ruc_qvg_cold_start", "wrf"),
                               ("ruc_2m_diagnostic", "flux")]))
    if pbl == 5:
        ws["bl_mynn_version"] = "wrf_461"
    if sfc == 5:
        ws["mynn_sfclay_variant"] = "wrf_461"
    if mp == 28:
        ws["thompson_version"] = "wrf_461"
    if dy["zadvect_implicit"]:
        ws["zadvect_implicit_variant"] = "wrf_471"
    if pbl == 900:
        ws["bl_pbl_physics"] = 900
        ws["km_opt"] = 0
    if ph["sf_urban_physics"] and lsm == 2 and g(c, "sf_surface_mosaic", 0) == 1:
        ws["mosaic_urban_canopy"] = "dominant"
    return nl, ws


def divergences(nl, ws):
    """Declared divergences each combo can show (registry warnings)."""
    ph, dy = nl["physics"], nl["dynamics"]
    out = []
    if ph["cu_physics"] == 3:
        out.append("gf-gamma (docs/gf_gamma_known_delta.md)")
        if ph.get("ishallow") == 1:
            out.append("gf-shallow-k22-section-offset")
        out.append("gf-inversion-loop-clamp (only when kend > ktf-8)")
    if ph["bl_pbl_physics"] == 2 or ph["sf_sfclay_physics"] == 2:
        out.append("myj-eta-ground-relative-interface-heights")
    if ph["mp_physics"] == 9:
        out.append("milbrandt-undefined-branches (only T<173 K or T>323 K and similar)")
    if ph["bl_pbl_physics"] == 11:
        out.append("shinhong-q2xk-past-end (only when kpbl == kte)")
    if ph["bl_pbl_physics"] == 1:
        out.append("ysu-0-over-0 (only when ust**3 underflows)")
    if ph["sf_urban_physics"] in (2, 3) and ph["bl_pbl_physics"] == 1:
        out.append("bep-rural-drag-applied-once (patch the reference)")
    if ph["sf_urban_physics"] == 1 and ph["sf_surface_physics"] == 4:
        out.append("ucm-noahmp-t2-absolute (patch the reference)")
    if dy.get("zadvect_implicit"):
        out.append("ieva-boundary-units A179 (patch the reference)")
    return out


def finish(rows, stratum, space_name):
    out = []
    for c in rows:
        nl, ws = render(c)
        out.append({"stratum": stratum, "factors": {k: v for k, v in c.items() if v != NA},
                    "namelist": nl, "woof_settings": ws})
    return out


# ---------------------------------------------------------------------------
# Production suites
# ---------------------------------------------------------------------------
REG_COMPONENT_TO_FACTOR = {
    "microphysics": ("mp", {"kessler-mp1": 1, "wsm6-mp6": 6, "thompson-mp8": 8,
                            "milbrandt2mom-mp9": 9, "morrison-mp10": 10, "wdm6-mp16": 16,
                            "nssl2-mp18": 18, "thompson-aerosol-mp28": 28, "p3-mp50": 50}),
    "pbl": ("pbl", {"off": 0, "ysu": 1, "myj": 2, "mynn": 5, "uw": 9, "shinhong": 11,
                    "sase": 900}),
    "surface_layer": ("sfclay", {"off": 0, "revised-mm5": 1, "eta-similarity": 2,
                                 "mynn": 5, "classic-mm5": 91}),
    "land_surface": ("lsm", {"off": 0, "noah": 2, "ruc-lsm": 3, "noah-mp": 4}),
    "cumulus": ("cu", {"off": 0, "kain-fritsch": 1, "grell-freitas": 3, "new-tiedtke": 16}),
    "turbulence": ("turb", {"closure-supplied": "k0", "constant-k": "d2k1",
                            "tke-1.5-order": "d2k2", "smagorinsky-3d": "d2k3",
                            "smagorinsky-2d": "d2k4"}),
    "urban": ("urban", {"none": 0, "slucm": 1, "bep": 2, "bep-bem": 3}),
}


def template_factors(tpl):
    comp, par = tpl["components"], tpl["parameters"]
    c = {}
    for k, (f, m) in REG_COMPONENT_TO_FACTOR.items():
        c[f] = m[comp[k]]
    r = comp["radiation"]
    if r == "rte-rrtmgp":
        c["rad"] = "rrtmg_legacy" if par.get("ra_rrtmg_variant") == "rrtmg_legacy" else "rte_rrtmgp"
    else:
        c["rad"] = {"off": "off", "dudhia-shortwave": "dudhia", "wrf-rrtm-dudhia": "rrtm_dudhia",
                    "analytic-clear-sky": "analytic"}[r]
    c.update({"v_mom_adv_order": 3, "v_sca_adv_order": 3, "damp_opt": 3, "w_damping": 1,
              "hybrid_opt": 2, "hypsometric_opt": 2, "epssm": par.get("epssm", 0.1),
              "diff_6th_opt": par.get("diff_6th_opt", 0), "top_lid": par.get("top_lid", False),
              "mp_zero_out": 0, "no_mp_heating": 0})
    if c["diff_6th_opt"]:
        c["diff_6th_slopeopt"] = par.get("diff_6th_slopeopt", 0)
    for k in ("bl_mynn_mixlength", "scalar_pblmix", "morr_rimed_ice"):
        if k in par:
            c[k] = par[k]
    if par.get("usemonalb") or par.get("rdlai2d"):
        c["monthly_veg"] = True
    if "fractional_seaice" in par:
        c["fractional_seaice"] = par["fractional_seaice"]
    if c["lsm"] == 3:
        c["ruc_soil_layers"] = 9
    return c


def suite_from_template(tid, tpl, origin):
    c = template_factors(tpl)
    par = tpl["parameters"]
    nl, ws = render(c, radt=float(par.get("radt", 12.0)),
                    cudt=float(par.get("cudt_minutes", 0.0)))
    nl["dynamics"]["diff_6th_factor"] = par.get("diff_6th_factor", 0.12)
    for k in ("wrf_rrtmg_compatibility", "ra_rrtmg_variant", "bl_mynn_version",
              "mynn_sfclay_variant", "thompson_version", "thompson_fork_snow_fall",
              "bl_mynn_gsd41_unsquared_qtke", "terrain_opt", "alb_sol", "aer_init_opt",
              "wif_input_opt"):
        if k in par:
            ws[k] = par[k]
    if tpl["components"]["radiation"] in ("dudhia-shortwave",) and par.get("radt") == 1.0:
        pass
    return c, nl, ws


def suite_from_site_toml(path):
    with open(path, "rb") as fh:
        t = tomllib.load(fh)
    sh = dict(t.get("shared", {}))
    d0 = dict(t["domain"][0]) if t.get("domain") else {}
    merged = {**sh, **d0}
    lw, sw = merged.get("ra_lw_physics", merged.get("ra_physics", 0)), \
        merged.get("ra_sw_physics", merged.get("ra_physics", 0))
    if (lw, sw) == (4, 4):
        rad = "rrtmg_legacy" if merged.get("ra_rrtmg_variant") == "rrtmg_legacy" else "rte_rrtmgp"
    else:
        rad = {(0, 0): "off", (0, 1): "dudhia", (1, 1): "rrtm_dudhia", (90, 90): "analytic"}[(lw, sw)]
    km = merged.get("km_opt", 4)
    diff = merged.get("diff_opt", 2)
    turb = {(1, 4): "d1k4", (2, 4): "d2k4", (1, 2): "d1k2", (2, 2): "d2k2", (2, 3): "d2k3",
            (2, 1): "d2k1", (2, 0): "k0"}[(diff, km)]
    c = {"mp": merged["mp_physics"], "pbl": merged["bl_pbl_physics"],
         "sfclay": merged["sf_sfclay_physics"], "lsm": merged["sf_surface_physics"],
         "cu": merged.get("cu_physics", 0), "rad": rad, "turb": turb,
         "v_mom_adv_order": merged.get("v_mom_adv_order", 3),
         "v_sca_adv_order": merged.get("v_sca_adv_order", 3),
         "damp_opt": merged.get("damp_opt", 3), "w_damping": merged.get("w_damping", 0),
         "hybrid_opt": merged.get("hybrid_opt", 2),
         "hypsometric_opt": merged.get("hypsometric_opt", 2),
         "epssm": merged.get("epssm", 0.1), "diff_6th_opt": merged.get("diff_6th_opt", 0),
         "top_lid": merged.get("top_lid", False), "mp_zero_out": merged.get("mp_zero_out", 0),
         "no_mp_heating": merged.get("no_mp_heating", 0)}
    for k in ("diff_6th_slopeopt", "isftcflx", "iz0tlnd", "ysu_topdown_pblmix",
              "bl_mynn_mixlength", "bl_mynn_mixscalars", "scalar_pblmix", "mosaic_lu",
              "mosaic_soil", "morr_rimed_ice", "icloud", "use_mp_re", "o3input", "swint_opt",
              "slope_rad", "topo_shading", "sf_lake_physics"):
        if k in merged:
            c[k] = merged[k]
    if c["lsm"] == 3:
        c["ruc_soil_layers"] = merged.get("num_soil_layers", 9)
    if merged.get("usemonalb") or merged.get("rdlai2d"):
        c["monthly_veg"] = True
    if merged.get("sf_urban_physics"):
        c["urban"] = merged["sf_urban_physics"]
    nl, ws = render(c, radt=float(merged.get("radt", 12.0)),
                    cudt=float(merged.get("cudt_minutes", 0.0)))
    nl["dynamics"]["diff_6th_factor"] = merged.get("diff_6th_factor", 0.12)
    nl["dynamics"]["time_step_sound"] = merged.get("time_step_sound", 4)
    for k in ("wrf_rrtmg_compatibility", "ra_rrtmg_variant", "terrain_opt"):
        if k in merged:
            ws[k] = merged[k]
    return c, nl, ws


def hrrr_suite():
    with open(HRRR_RECIPE, "rb") as fh:
        t = tomllib.load(fh)
    sh = dict(t["shared"])
    d0 = dict(t["domain"][0])
    m = {**sh, **d0}
    c = {"mp": 28, "pbl": 5, "sfclay": 5, "lsm": 3, "cu": 0, "rad": "rrtmg_legacy",
         "turb": "d2k4", "v_mom_adv_order": 3, "v_sca_adv_order": 3,
         "damp_opt": m["damp_opt"], "w_damping": m["w_damping"], "hybrid_opt": m["hybrid_opt"],
         "hypsometric_opt": m["hypsometric_opt"], "epssm": m["epssm"],
         "diff_6th_opt": m["diff_6th_opt"], "diff_6th_slopeopt": m["diff_6th_slopeopt"],
         "top_lid": m["top_lid"], "mp_zero_out": 0, "no_mp_heating": 0,
         "bl_mynn_mixlength": m["bl_mynn_mixlength"], "scalar_pblmix": m["scalar_pblmix"],
         "ruc_soil_layers": m["num_soil_layers"], "mosaic_lu": m["mosaic_lu"],
         "mosaic_soil": m["mosaic_soil"], "monthly_veg": True, "swint_opt": m["swint_opt"],
         "sf_lake_physics": m["sf_lake_physics"], "gwd_opt": m["gwd_opt"],
         "topo_wind": m["topo_wind"], "zadvect_implicit": m["zadvect_implicit"],
         "aer_opt": m["aer_opt"]}
    nl, ws = render(c, radt=float(m["radt"]), cudt=0.0)
    ph, dy = nl["physics"], nl["dynamics"]
    ph["use_aero_icbc"] = True
    nl["domains"]["wif_input_opt"] = 1
    nl["domains"]["num_wif_levels"] = 30
    ph["aer_opt"] = 3
    dy["time_step_sound"] = m["time_step_sound"]
    dy["diff_6th_thresh"] = m["diff_6th_thresh"]
    dy["diff_6th_factor"] = m["diff_6th_factor"]
    dy.pop("w_crit_cfl", None)
    ws.clear()
    for k in ("bl_mynn_version", "bl_mynn_gsd41_unsquared_qtke", "bl_mynn_cloud_tendency_form",
              "mynn_sfclay_variant", "thompson_version", "alb_sol", "zadvect_implicit_variant",
              "ruc_soilprop", "ruc_irrigation", "ruc_snow", "ruc_qvg_cold_start",
              "ruc_2m_diagnostic", "ra_rrtmg_variant", "wrf_rrtmg_compatibility",
              "rrtmg_cloud_optics_form", "aer_init_opt", "wif_input_opt", "use_rap_aero_icbc",
              "terrain_opt"):
        if k in m:
            ws[k] = m[k]
    ws["thompson_fork_snow_fall"] = "wrf_39_noaa"
    return c, nl, ws


# ---------------------------------------------------------------------------
# Stratum C: fork switches on the HRRR base
# ---------------------------------------------------------------------------
C_FACTORS = OrderedDict([
    ("thompson_fork_snow_fall", ["wrf_39_noaa", "blend"]),
    ("bl_mynn_mixlength", [2, 1]),
    ("bl_mynn_cloud_tendency_form", ["wrf_461", "gsd_41"]),
    ("bl_mynn_gsd41_unsquared_qtke", ([False, True], ["bl_mynn_mixlength"],
                                      lambda c: c["bl_mynn_mixlength"] == 2)),
    ("scalar_pblmix", [1, 0]),
    # bl_mynn_mixscalars stays 0: WOOF refuses it under gsd_41 (the plume
    # transport carries only the v4.6.1 shallow-cumulus cloud)
    ("aer_opt", [3, 0]),
    ("swint_opt", [1, 0]),
    ("alb_sol", [1, 0]),
    ("diff_6th_form", ["wrf_461", "noaa_wrf39"]),
    ("upper_wind_limiter_form", ["wrf_461", "noaa_wrf39"]),
    ("zadvect_implicit", [1, 0]),
    ("mosaic_lu", [1, 0]),
    ("mosaic_soil", [1, 0]),
    ("fractional_seaice", [0, 1]),
    ("sf_lake_physics", [1, 0]),
    ("gwd_opt", [3, 0]),
    ("mp_zero_out", [0, 2]),
    ("mp_zero_out_all", [0, 1]),
])


class CSpace(Space):
    pass


def c_constraints(space):
    cons = []
    for name, spec in space.items():
        if isinstance(spec, tuple):
            values, deps, pred = spec
            cons.append(([name] + deps,
                         (lambda pred, name: lambda c: pred(c) == (c[name] != NA))(pred, name)))
    return cons


def apply_fork(base_c, base_nl, base_ws, f):
    nl, ws = copy.deepcopy(base_nl), copy.deepcopy(base_ws)
    ph, dy = nl["physics"], nl["dynamics"]
    ws["thompson_fork_snow_fall"] = f["thompson_fork_snow_fall"]
    ph["bl_mynn_mixlength"] = f["bl_mynn_mixlength"]
    ws["bl_mynn_cloud_tendency_form"] = f["bl_mynn_cloud_tendency_form"]
    ws["bl_mynn_gsd41_unsquared_qtke"] = (f["bl_mynn_gsd41_unsquared_qtke"]
                                          if f["bl_mynn_gsd41_unsquared_qtke"] != NA else False)
    ph["scalar_pblmix"] = f["scalar_pblmix"]
    ph["bl_mynn_mixscalars"] = 0
    ph["aer_opt"] = f["aer_opt"]
    ph["swint_opt"] = f["swint_opt"]
    ws["alb_sol"] = f["alb_sol"]
    ws["diff_6th_form"] = f["diff_6th_form"]
    if f["diff_6th_form"] == "noaa_wrf39":
        dy["diff_6th_factor2"] = 0.04
    ws["upper_wind_limiter_form"] = f["upper_wind_limiter_form"]
    dy["zadvect_implicit"] = f["zadvect_implicit"]
    if f["zadvect_implicit"]:
        ws["zadvect_implicit_variant"] = "wrf_legacy"
    else:
        ws.pop("zadvect_implicit_variant", None)
    ph["mosaic_lu"] = f["mosaic_lu"]
    ph["mosaic_soil"] = f["mosaic_soil"]
    ph["fractional_seaice"] = f["fractional_seaice"]
    ph["sf_lake_physics"] = f["sf_lake_physics"]
    dy["gwd_opt"] = f["gwd_opt"]
    ph["mp_zero_out"] = f["mp_zero_out"]
    ws["mp_zero_out_all"] = f["mp_zero_out_all"]
    return nl, ws


# ---------------------------------------------------------------------------
# Regenerating against an older list: keep its still-valid rows and ids.
# ---------------------------------------------------------------------------
def _id_number(cid):
    return int(cid[1:])


def keep_rows(old, stratum, space):
    """The old list's rows of ``stratum`` that are valid in ``space``, as
    [(id, row)], and the ones a rule now forbids, as [(id, reason)]."""
    kept, dropped = [], []
    if old is None:
        return kept, dropped
    for rec in old["combos"]:
        if rec["stratum"] != stratum:
            continue
        factors = rec["factors"]
        row = {n: factors.get(n, NA) for n in space.order}
        if stratum != "C":
            extra = sorted(set(factors) - set(space.order))
            if extra:
                dropped.append((rec["id"], "factors outside this stratum's space: "
                                + ", ".join(extra)))
                continue
        bad = [n for n in space.order if row[n] not in space.domains[n]]
        if bad:
            dropped.append((rec["id"], "values outside this stratum's space: "
                            + ", ".join(f"{n}={row[n]!r}" for n in bad)))
            continue
        broken = []
        for deps, fn in space.cons:
            if not fn(row):
                text = " and ".join(f"{d}={row[d]!r}" for d in dict.fromkeys(deps))
                if text not in broken:
                    broken.append(text)
        if broken:
            dropped.append((rec["id"], "violates a rule: " + "; ".join(broken)))
            continue
        kept.append((rec["id"], row))
    return kept, dropped


def number_rows(stratum, kept, dropped, new_rows):
    """Kept rows keep their ids; new rows are numbered after every old id."""
    used = [_id_number(cid) for cid, _ in kept] + [_id_number(cid) for cid, _ in dropped]
    start = max(used, default=0) + 1
    added = [(f"{stratum}{start + i:03d}", row) for i, row in enumerate(new_rows)]
    return sorted(kept + added, key=lambda ir: _id_number(ir[0])), [cid for cid, _ in added]


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    global HRRR_RECIPE, SITE
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--registry", default=REGISTRY,
                    help="physics_registry_v2.json (default: the integrate/2.8.7 tree)")
    ap.add_argument("--hrrr-recipe", default=HRRR_RECIPE, help="hrrr_v4_gsd41.toml")
    ap.add_argument("--site", default=SITE, required=SITE is None,
                    help="site apps/web root (templates/*/forecast.toml); or set GPUWM_COMBO_SITE_ROOT")
    ap.add_argument("--keep-rows-from", default=None, metavar="JSON",
                    help="an older combos.json whose still-valid generated rows keep their ids")
    ap.add_argument("--out", default=os.path.join(HERE, "combos.json"))
    args = ap.parse_args()
    HRRR_RECIPE, SITE = args.hrrr_recipe, args.site
    rng = random.Random(args.seed)
    reg_bytes = open(args.registry, "rb").read()
    reg = json.loads(reg_bytes)
    old = None
    if args.keep_rows_from:
        with open(args.keep_rows_from, encoding="utf-8") as fh:
            old = json.load(fh)
    old_cases = {rec["id"]: rec["cases"] for rec in (old or {"combos": []})["combos"]}
    regen = OrderedDict([("kept", OrderedDict()), ("dropped", []), ("added", [])])

    def regenerate(stratum, space, required, candidates):
        kept, dropped = keep_rows(old, stratum, space)
        new_rows = build_cover(space, required, rng, candidates=candidates,
                               existing=[row for _, row in kept])
        rows, added = number_rows(stratum, kept, dropped, new_rows)
        if old is not None:
            regen["kept"][stratum] = len(kept)
            regen["dropped"] += [{"id": cid, "reason": why} for cid, why in dropped]
            regen["added"] += added
        return rows

    # ---- stratum A ----
    spaceA_f = OrderedDict(list(CORE.items()) + list(DYN.items()) + list(SUB.items()))
    A = Space(spaceA_f)
    pairsA = required_pairs(A)
    triplesA = required_triples(A, ["mp", "pbl", "sfclay", "lsm"])
    idsA = regenerate("A", A, pairsA | triplesA, 30)
    rowsA = [row for _, row in idsA]

    # ---- stratum B ----
    coreB = OrderedDict(CORE)
    coreB["pbl"] = CORE["pbl"] + [9]
    spaceB_f = OrderedDict(list(coreB.items()) + list(DYN.items()) + list(SUB.items())
                           + list(B_EXTRA.items()) + list(B_SUB.items()))
    B = Space(spaceB_f)
    bitems = set(B_NEW_ITEMS)
    # B items pair with the core, dynamics and B factors only: the sub-options
    # of A-only schemes are already covered against everything in stratum A.
    b_names = set(coreB) | set(DYN) | set(B_EXTRA) | set(B_SUB) | {
        "ysu_topdown_pblmix", "isftcflx", "iz0tlnd", "monthly_veg", "rdmaxalb", "opt_thcnd"}
    pairsB = required_pairs(B, lambda t: bool(set(t) & bitems)
                            and all(k in b_names for k, _ in t))
    idsB = regenerate("B", B, pairsB, 30)
    rowsB = [row for _, row in idsB]

    # ---- stratum D ----
    coreD = OrderedDict(CORE)
    coreD["rad"] = CORE["rad"] + ["rte_rrtmgp", "analytic", "dudhia"]
    coreD["pbl"] = CORE["pbl"] + [900]
    coreD["turb"] = CORE["turb"] + ["k0"]
    spaceD_f = OrderedDict(list(coreD.items()) + list(DYN.items()) + list(SUB.items()))
    D = Space(spaceD_f)
    ditems = set(D_NEW_ITEMS)
    core_names = set(CORE) | set(DYN)
    pairsD = required_pairs(
        D, lambda t: bool(set(t) & ditems) and all(k in core_names for k, _ in t)
        and not ({k for k, _ in t} == {"rad"}))
    # analytic only needs the core physics selectors, not every dynamics value
    pairsD = {t for t in pairsD
              if not (("rad", "analytic") in t and any(k in DYN for k, _ in t))}
    idsD = regenerate("D", D, pairsD, 40)
    rowsD = [row for _, row in idsD]

    combos = []

    seen = {}

    def add(cid, stratum, origin, reference, checks, c, nl, ws, note=None, overrides=None):
        key = json.dumps([nl, ws], sort_keys=True, default=str)
        if stratum in ("suite", "suite-twin") and key in seen:
            prev = seen[key]
            prev.setdefault("also", []).append(f"{cid}: {origin}")
            return
        rec = OrderedDict()
        rec["id"] = cid
        rec["stratum"] = stratum
        rec["origin"] = origin
        rec["reference"] = reference
        rec["checks"] = checks
        rec["declared_divergences"] = divergences(nl, ws)
        rec["cases"] = ["iowa-convective"]
        rec["woof_door"] = ("toml: the namelist importer has no ra_*_physics=90 mapping"
                            if nl["physics"]["ra_lw_physics"] == 90 else "namelist import")
        ov = dict(overrides or {})
        if stratum in ("suite", "suite-twin", "C"):
            # run long enough that radiation (and a timed cumulus) fires twice
            dt = float(ov.get("time_step_s", 15.0))
            ph_ = nl["physics"]
            longest = max(float(ph_.get("radt", 0.0)),
                          float(ph_.get("cudt", 0.0)) if ph_.get("cu_physics") else 0.0)
            need = int(round(longest * 60.0 / dt)) + 4
            if need > 20:
                ov["steps"] = need
        if nl["physics"].get("sf_urban_physics") == 1:
            ov["eta_levels"] = "urban-safe ladder (CASE.md)"
        if ov:
            rec["case_overrides"] = ov
        if note:
            rec["note"] = note
        rec["factors"] = {k: v for k, v in c.items() if v != NA}
        rec["namelist"] = nl
        rec["woof_settings"] = ws
        combos.append(rec)
        if stratum in ("suite", "suite-twin"):
            seen[key] = rec

    REF_A = "stock WRF v4.6.1, strict build"
    REF_B = "stock WRF v4.7.1, strict build, with the declared-divergence patches listed"
    REF_C = "NOAA-EMC WRFV3.9 HRRR fork, strict build"
    CHK_W = ["wrf_0ulp_every_step", "finite"]
    CHK_O = ["completes", "finite", "repeat_run_0ulp"]

    for cid, c in idsA:
        nl, ws = render(c)
        add(cid, "A", "pairwise+core-triples", REF_A, CHK_W, c, nl, ws)
    for cid, c in idsB:
        nl, ws = render(c)
        add(cid, "B", "pairwise (WRF v4.7.1 ports)", REF_B, CHK_W, c, nl, ws)

    # ---- production suites ----
    hc, hnl, hws = hrrr_suite()
    add("S-hrrr-v4-gsd41", "suite", "production suite: HRRR route default "
        "(configs/recipes/hrrr_v4_gsd41.toml, template "
        "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1)", REF_C,
        CHK_W + ["repeat_run_0ulp"], hc, hnl, hws,
        note="run at the case dt and grid; the recipe's 20 s fixed clock, 1797x1057 grid and "
             "RAP boundaries are replaced by the case's")

    # ---- stratum C ----
    Cs = Space.__new__(Space)
    Cs.factors = C_FACTORS
    Cs.order = list(C_FACTORS)
    Cs.domains = {n: (list(s[0]) + [NA] if isinstance(s, tuple) else list(s))
                  for n, s in C_FACTORS.items()}
    Cs.cons = c_constraints(C_FACTORS)
    Cs.by_var = {n: [] for n in Cs.order}
    for deps, fn in Cs.cons:
        for d in set(deps):
            Cs.by_var[d].append((deps, fn))
    pairsC = required_pairs(Cs)
    idsC = regenerate("C", Cs, pairsC, 40)
    rowsC = [row for _, row in idsC]
    for cid, f in idsC:
        nl, ws = apply_fork(hc, hnl, hws, f)
        cc = dict(hc)
        cc.update({k: v for k, v in f.items() if v != NA})
        add(cid, "C", "pairwise fork switches on the HRRR suite", REF_C, CHK_W,
            cc, nl, ws)

    for cid, c in idsD:
        nl, ws = render(c)
        add(cid, "D", "WOOF-own arm paired with every core value", "none (no WRF "
            "counterpart)", CHK_O, c, nl, ws)

    templates = reg["templates"]
    aliases = reg["template_aliases"]
    site_profiles = [
        "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1",
        "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-validation-candidate-v1",
        "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-validation-candidate-v1",
        "thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1",
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1",
        "thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1",
        "p3-mp50-ysu-mm5-noah-rrtmg-legacy-v1",
        "wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1",
        "wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1",
        "thompson-mp8-ysu-mm5-noah-validation-v1",
        "wsm6-ysu-mm5-noah-no-radiation-v1",
        "wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1",
        "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1",
        "wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1",
    ]
    site_resolved = {aliases.get(p, p) for p in site_profiles}
    twins = []

    def suite_reference(c, ws):
        fork = any(ws.get(k) in ("gsd_41", "wrf_39_noaa", "gsl_wrf39") for k in
                   ("bl_mynn_version", "thompson_version", "mynn_sfclay_variant"))
        if fork:
            return REF_C, CHK_W + ["repeat_run_0ulp"]
        if c["rad"] in ("rte_rrtmgp", "analytic", "dudhia") or c["pbl"] == 900:
            return "none (no WRF counterpart for this arm; see its twin)", CHK_O
        if c.get("urban", 0) or c["pbl"] == 9:
            return REF_B, CHK_W + ["repeat_run_0ulp"]
        return REF_A, CHK_W + ["repeat_run_0ulp"]

    for tid in sorted(templates):
        tpl = templates[tid]
        c, nl, ws = suite_from_template(tid, tpl, "registry")
        tag = ("site physics profile and registry template" if tid in site_resolved
               else "registry template")
        if tid == "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1":
            tag = "WOOF default suite for GFS and ERA5 cases; " + tag
        if tid == "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1":
            tag = "product default suite (DEFAULT_SUITE_PHYSICS, conus-3km); " + tag
        ref, chk = suite_reference(c, ws)
        add(f"S-tpl-{tid}", "suite", f"production suite: {tag} {tid} (exact, its own radt)",
            ref, chk, c, nl, ws)
        if c["rad"] in ("rte_rrtmgp", "dudhia") and c["pbl"] != 900:
            twins.append((f"T-tpl-{tid}", c, nl, ws, tid, None))

    default_c = dict(template_factors(templates["thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1"]))
    for name in sorted(os.listdir(os.path.join(SITE, "templates"))):
        p = os.path.join(SITE, "templates", name, "forecast.toml")
        if not os.path.exists(p):
            continue
        c, nl, ws = suite_from_site_toml(p)
        ref, chk = suite_reference(c, ws)
        steps_ok = all(abs((x * 60.0 / 15.0) - round(x * 60.0 / 15.0)) < 1e-9
                       for x in (nl["physics"]["radt"], nl["physics"]["cudt"]))
        add(f"S-site-{name}", "suite", f"production suite: site template {name} "
            "(root domain physics and dynamics exactly, its own radt)", ref, chk, c, nl, ws,
            note="run on the case grid; the template's own grid spacing and nests are out "
                 "of scope here (see EXTRAS.md)",
            overrides=None if steps_ok else {"time_step_s": 7.5, "steps": 40,
                                             "why": "its radt/cudt is not a whole number of "
                                                    "15 s steps; 7.5 s divides it"})
        if c["rad"] in ("rte_rrtmgp", "dudhia"):
            twins.append((f"T-site-{name}", c, nl, ws, name,
                          None if steps_ok else {"time_step_s": 7.5, "steps": 40,
                                                 "why": "as its suite"}))

    for tid, c, nl, ws, src, ov in twins:
        c2 = dict(c)
        nl2, ws2 = copy.deepcopy(nl), copy.deepcopy(ws)
        if c["rad"] == "rte_rrtmgp":
            c2["rad"] = "rrtmg_legacy"
            nl2["physics"]["use_mp_re"] = 1
            nl2["physics"]["o3input"] = 2
            ws2["ra_rrtmg_variant"] = "rrtmg_legacy"
            ws2["wrf_rrtmg_compatibility"] = "wrf-rrtmg-4-4-legacy-v1"
            what = "RTE+RRTMGP replaced by the legacy RRTMG port"
        else:
            c2["rad"] = "rrtm_dudhia"
            nl2["physics"]["ra_lw_physics"] = 1
            ws2.pop("acknowledgements", None)
            what = ("longwave off replaced by WRF RRTM longwave (stock WRF fatals "
                    "longwave off with shortwave on)")
        ref = REF_B if (c.get("urban", 0) or c["pbl"] == 9) else REF_A
        add(tid, "suite-twin", f"WRF-comparable twin of {src}: identical except {what}",
            ref, CHK_W, c2, nl2, ws2, overrides=ov)

    # ---- singles: GF closure members ----
    gf = template_factors(templates["wdm6-mp16-ysu-mm5-noah-grell-freitas-rte-rrtmgp-v1"])
    gf["rad"] = "rrtmg_legacy"
    for k in range(1, 17):
        c = dict(gf)
        c["clos_choice"] = k
        c["ishallow"] = 0
        nl, ws = render(c)
        add(f"G-clos{k:02d}", "single", "Grell-Freitas closure member alone (clos_choice), "
            "WDM6 + YSU + classic MM5 + Noah + legacy RRTMG", REF_A, CHK_W, c, nl, ws)

    # ---- variant cases: pick runs from A covering surface pairs ----
    def pick_variant(rows_ids, factors, case):
        req = set()
        for cid, c in rows_ids:
            for a, b in itertools.combinations(factors, 2):
                if c.get(a, NA) != NA and c.get(b, NA) != NA:
                    req.add(frozenset([(a, c[a]), (b, c[b])]))
        # rows the older list already ran on this case keep it
        chosen = [cid for cid, _ in rows_ids if case in old_cases.get(cid, ())]
        rows = dict(rows_ids)
        left = {t for t in req if not any(t <= set(rows[cid].items()) for cid in chosen)}
        while left:
            best = max(rows_ids, key=lambda rc: sum(1 for t in left if t <= set(rc[1].items())))
            chosen.append(best[0])
            left = {t for t in left if not t <= set(best[1].items())}
        return chosen, len(req)

    a_rows = list(idsA)
    coast_f = ["mp", "pbl", "sfclay", "lsm", "isftcflx", "iz0tlnd", "sf_lake_physics", "rad",
               "cu"]
    snow_f = ["mp", "pbl", "sfclay", "lsm", "ruc_soil_layers", "mosaic_lu", "mosaic_soil",
              "fractional_seaice", "monthly_veg", "rdmaxalb", "opt_thcnd", "rad"]
    coast_ids, coast_req = pick_variant(a_rows, coast_f, "gulf-coast-convective")
    snow_ids, snow_req = pick_variant(a_rows, snow_f, "iowa-snow-ice")
    for rec in combos:
        if rec["id"] in coast_ids:
            rec["cases"].append("gulf-coast-convective")
        if rec["id"] in snow_ids:
            rec["cases"].append("iowa-snow-ice")
        if rec["stratum"] in ("suite", "C"):
            for case in ("gulf-coast-convective", "iowa-snow-ice"):
                if case not in rec["cases"]:
                    rec["cases"].append(case)

    # ---- verify coverage ----
    def coverage(rows, req):
        ok = sum(1 for t in req if any(t <= set(r.items()) for r in rows))
        return ok, len(req)

    cov = OrderedDict()
    cov["A_pairs"] = coverage(rowsA, pairsA)
    cov["A_core_triples"] = coverage(rowsA, triplesA)
    cov["B_pairs"] = coverage(rowsB, pairsB)
    cov["C_pairs"] = coverage(rowsC, pairsC)
    cov["D_pairs"] = coverage(rowsD, pairsD)
    for k, (ok, n) in cov.items():
        assert ok == n, (k, ok, n)

    # validity re-check of every generated row
    for rows, sp in ((rowsA, A), (rowsB, B), (rowsD, D), (rowsC, Cs)):
        for r in rows:
            for n in sp.order:
                assert sp.consistent(r, n), r

    runs = sum(len(r["cases"]) for r in combos)
    repeat = sum(1 for r in combos if "repeat_run_0ulp" in r["checks"])
    by = OrderedDict()
    for r in combos:
        by[r["stratum"]] = by.get(r["stratum"], 0) + 1
    out = OrderedDict()
    out["schema"] = "woof-combo-sweep-v1"
    out["generator"] = "gen_combos.py --seed %d" % args.seed + (
        " --keep-rows-from %s" % os.path.basename(args.keep_rows_from)
        if args.keep_rows_from else "")
    out["registry"] = {"path": "gpuwm/physics_registry_v2.json on integrate/2.8.7",
                       "sha256": hashlib.sha256(reg_bytes).hexdigest(),
                       "registry_version": reg.get("registry_version")}
    out["inputs"] = OrderedDict([
        ("hrrr_recipe_sha256", sha256_file(HRRR_RECIPE)),
        ("site_templates_sha256", OrderedDict(
            (name, sha256_file(os.path.join(SITE, "templates", name, "forecast.toml")))
            for name in sorted(os.listdir(os.path.join(SITE, "templates")))
            if os.path.exists(os.path.join(SITE, "templates", name, "forecast.toml")))),
    ])
    out["wrf_rules"] = [
        "mp_physics 28 is rendered with the climatology triple wif_input_opt = 1, "
        "num_wif_levels = 30, use_aero_icbc = .true. "
        "(stock real.exe: 'wif_input_opt=0 but mp_physics=28')",
        "MYNN (bl_mynn_edmf = 1) never runs beside a Grell-Freitas shallow plume: "
        "ishallow = 0 (stock namelist check: 'bl_mynn_edmf > 0 requires ... ishallow=0')",
    ]
    if old is not None:
        out["regeneration"] = OrderedDict([
            ("keep_rows_from", OrderedDict([
                ("file", os.path.basename(args.keep_rows_from)),
                ("sha256", sha256_file(args.keep_rows_from))])),
            ("kept", regen["kept"]), ("dropped", regen["dropped"]),
            ("added", regen["added"])])
    out["counts"] = OrderedDict([("combos", len(combos)), ("by_stratum", by),
                                 ("runs_including_variant_cases", runs),
                                 ("extra_repeat_runs", repeat)])
    out["coverage"] = OrderedDict((k, {"covered": v[0], "required": v[1]})
                                  for k, v in cov.items())
    out["coverage"]["coast_variant_pairs"] = {"required": coast_req, "runs": len(coast_ids)}
    out["coverage"]["snow_variant_pairs"] = {"required": snow_req, "runs": len(snow_ids)}
    out["fixed_everywhere"] = OrderedDict([
        ("use_theta_m", "0 (WOOF integrates dry theta; 1 is a substitution, not a match)"),
        ("moist_adv_opt / scalar_adv_opt", "1 (the only value WOOF imports)"),
        ("h_mom_adv_order / h_sca_adv_order", "5 (the only value WOOF imports)"),
        ("momentum_adv_opt", "1"), ("rk_ord", "3"), ("non_hydrostatic", ".true."),
        ("isfflx", "1"), ("num_land_cat", "21"), ("bldt", "0 (every step)"),
        ("cudt", "0 in generated combos (every step); suites keep their own"),
        ("radt", "1 minute in generated combos; suites keep their own"),
    ])
    out["factor_legend"] = OrderedDict([
        ("rad", "off=0/0, dudhia=0/1, rrtm_dudhia=1/1, rrtmg_legacy=4/4 with the legacy RRTMG "
                "port, rte_rrtmgp=4/4 with RTE+RRTMGP (no WRF counterpart), analytic=90/90"),
        ("turb", "dNkM = diff_opt N, km_opt M; k0 = km_opt 0 (SASE only)"),
        ("nssl_variant", "mp_physics value written for the NSSL family (17/19/21/22 are the "
                         "deprecated spellings WRF and WOOF both map onto 18 plus flags)"),
        ("monthly_veg", "usemonalb and rdlai2d together"),
        ("ruc_soil_layers", "num_soil_layers under RUC"),
    ])
    out["combos"] = combos
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["counts"], indent=1))
    print(json.dumps(out["coverage"], indent=1))


if __name__ == "__main__":
    main()
