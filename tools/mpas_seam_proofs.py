"""MPAS column-batch seam: parity-vs-ARW proof battery (GPU).

Five proofs against the REAL ARW physics driver on a genuine
mass-coordinate ``DomainState``, run on identical column states.  The
harness plays the MPAS integrator: each step it hands the seam the ARW
state's exact fields (the "same column states"), runs both legs, and
compares.

1. PARITY: an f-plane column set with unit map factors, flat terrain,
   x-uniform winds and a uniform-eta coordinate makes every omitted ARW
   coupling the identity or a pure factor:
     - u/v destagger: 0.5*(a + a) == a exactly (x-uniform winds);
     - fnm/fnp: uniform eta and uniform nominal z both give exactly 0.5;
     - mass coupling: ARW component stacks are fl(chm * raw); the seam's
       identity state (chm == 1) holds raw, so fl(chm_arw * seam_raw)
       must be BIT-IDENTICAL to the ARW stack, component by component;
     - momentum: both drivers retain the raw pre-coupling YSU rates
       (last_ysu), compared bitwise (the a2c face interpolation is the
       omitted coupling; it enters neither compared quantity).
2. CADENCE: 90 steps x 120 s (3 h).  Radiation held rates byte-identical
   between due calls and changing at due calls; buckets monotone and
   bit-equal to the ARW accumulators; counters equal; the full raw
   driver-state manifest (surface, soil, holds, KF trigger history)
   bit-equal at steps 1/45/90.
3. RESTART: export at the step-45 boundary, restore into a fresh seam,
   continue 45 more steps; bit-identical to the uninterrupted seam.
   Run on BOTH the bound arm and the stock seam.
4. PHASE 2: per-step MPAS-transport stand-in perturbation applied to the
   ARW state, marshalled to the caller arrays, then seam WSM6-in-place
   vs the ARW post-RK pair on the same perturbed state.
5. INSTRUMENT VALIDATION: five deliberate breakages (held-tendency drop,
   bucket corruption, silent restart corruption, identity-coupling
   break, cadence corruption) must each turn their proof red.

NAMED RESIDUALS (each measured, each with a mechanism):
  (a) phase-2 marshalling round trips ``alt = fl(1/fl(1/alt))`` and
      ``ph = fl(fl(ph/g)*g)`` -- <= 1 ULP inputs, documented in
      docs/mpas-seam.md.  ARM A binds them bitwise; its zeros prove they
      are the only phase-2 input residual.
  (b) theta carrier accumulation: ARW adds the MP heating increment to
      perturbation theta over a base (``thb + fl(thp + m)``); the seam
      adds it to the caller's total (``fl(T + m)``).  One rounding of
      difference, bounded at <= 2 ULP per step, measured per step.
  (c) exner source: a single-pressure caller that omits ``exner=``
      derives it from the supplied (hydrostatic) pressure where ARW uses
      the EOS pressure.  Quantified one-step in ARM C.

Arms:
  control: the ARW leg run twice must be bit-repeatable, or no bitwise
      comparison below means anything (validate-the-instrument).
  A (bound, resynced): residual (a) bound bitwise, caller state resynced
      from the ARW leg each step.  Everything compared -- component
      tendencies, raw momentum/radiation, surface/soil manifest,
      species, h_diabatic, receipts, buckets, counters, restart -- must
      be BIT-FOR-BIT over the full 3 h; theta carrier shows residual (b)
      only.
  B (contract drift): stock seam, naive marshalling, fully independent
      trajectory.  Reports max-ULP/max-abs drift growth attributed to
      (a)+(b) by Arm A's zeros.
  C (exner source): one step, residual (c) quantified.
  R (red): proof 5.

Run:  python tools/mpas_seam_proofs.py --out receipts/ [--steps 90]
Exit 0 iff every green assertion holds AND every red arm goes red.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time

import numpy as np

PROOF_SCHEMA = "mpas-seam-proofs-v1"

NZ = 40
NCOL = 32
DT = 120.0
ZTOP = 16000.0
DX = 15000.0
START = datetime.datetime(2021, 6, 1, 18, 0)
RESTART_STEP = 45
STEPRA = 5                      # 600 s radiation at dt = 120 s
THETA_CARRIER_ULP_BOUND = 2     # residual (b): one extra rounding

_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")
_PROGNOSTICS = ("theta",) + _SPECIES


# ----------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------

def _bytes(a):
    import cupy as cp
    if hasattr(a, "__cuda_array_interface__"):
        a = cp.asnumpy(a)
    return np.ascontiguousarray(a).tobytes()


def _ordered_int(v: np.ndarray) -> np.ndarray:
    # The one total-order owner, not a local copy: this held the correct
    # INT32_MIN reflection, but tests/test_fp32_ulp.py exists because
    # thirteen such copies carried the sign error, so even a right copy
    # imports the shared map (2026-08-30).
    from gpuwm.core.fp32_ulp import monotone_fp32_key

    return monotone_fp32_key(np.asarray(v, dtype=np.float32))


def ulp_stats(a, b) -> dict:
    """Max ULP distance, max abs delta, differing count for two fp32 sets."""
    import cupy as cp
    if hasattr(a, "__cuda_array_interface__"):
        a = cp.asnumpy(a)
    if hasattr(b, "__cuda_array_interface__"):
        b = cp.asnumpy(b)
    av = np.asarray(a, dtype=np.float32).ravel()
    bv = np.asarray(b, dtype=np.float32).ravel()
    if av.shape != bv.shape:
        raise ValueError(f"shape mismatch {av.shape} vs {bv.shape}")
    diff = (av != bv) & ~(np.isnan(av) & np.isnan(bv))
    n = int(diff.sum())
    if n == 0:
        return {"n_diff": 0, "n_total": int(av.size), "max_ulp": 0,
                "max_abs": 0.0,
                "scale": float(np.abs(bv).max()) if bv.size else 0.0}
    ulp = np.abs(_ordered_int(av) - _ordered_int(bv))
    return {
        "n_diff": n,
        "n_total": int(av.size),
        "max_ulp": int(ulp[diff].max()),
        "max_abs": float(np.abs(av - bv)[diff].max()),
        "scale": float(np.abs(bv).max()),
    }


class ProofFailure(AssertionError):
    pass


class Check:
    """Bitwise comparator that records context-tagged failures."""

    def __init__(self, name):
        self.name = name
        self.failures = []
        self.compared = 0

    def bitwise(self, tag, a, b):
        self.compared += 1
        if _bytes(a) != _bytes(b):
            self.failures.append({"tag": tag, **ulp_stats(a, b)})

    def true(self, tag, condition):
        self.compared += 1
        if not condition:
            self.failures.append({"tag": tag, "condition": False})

    def raise_if_red(self):
        if self.failures:
            raise ProofFailure(
                f"{self.name}: {len(self.failures)} failures, first: "
                f"{self.failures[0]}")


# ----------------------------------------------------------------------
# the ARW leg: a genuine DomainState + PhysicsDriver, driven exactly as
# gpuwm/core/dycore.py:2324-2325 and :2500-2512 drive it.
# ----------------------------------------------------------------------

def surface_setup():
    cols = np.arange(NCOL)
    return {
        "latitude": np.float32(33.0 + 0.05 * cols),
        "longitude": np.float32(-100.0 + 0.06 * cols),
        "terrain": np.zeros(NCOL, dtype=np.float32),
        "landmask": np.ones(NCOL, dtype=np.float32),
        "ivgtyp": np.full(NCOL, 10, dtype=np.int32),
        "isltyp": np.full(NCOL, 6, dtype=np.int32),
        "vegfra": np.float32(40.0 + 20.0 * cols / NCOL),
        "tsk": np.float32(300.0 + 0.25 * np.sin(cols)),
        "tmn": np.full(NCOL, 285.0, dtype=np.float32),
    }


def scheme_cfg_kwargs(cumulus: str) -> dict:
    """The physics/cadence RunConfig kwargs BOTH legs share.

    The MPAS side pinned radiation 600 s and surface/PBL 120 s.  The GF
    arm runs the pinned cumulus 120 s (GF's own cudt=0 law); the KF arm
    runs 600 s to exercise the NCA hold machinery across non-due steps.
    """
    from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY
    kwargs = dict(
        moist=True, mp_physics=6, wsm6_hail_opt=0,
        ra_physics=4, ra_rrtmg_variant=RRTMG_VARIANT_LEGACY,
        sf_sfclay_physics=1, sf_surface_physics=4, bl_pbl_physics=1,
        radt=10.0, radt_minutes=10.0, bldt=2.0)
    if cumulus == "gf":
        kwargs.update(cu_physics=3, cudt_minutes=0.0)
    elif cumulus == "kf":
        kwargs.update(cu_physics=1, cudt_minutes=10.0)
    else:
        raise ValueError(cumulus)
    return kwargs


class ArwLeg:
    """The reference: real mass-coordinate state, real driver, ARW order."""

    def __init__(self, cumulus: str):
        import cupy as cp
        from gpuwm.config import RunConfig
        from gpuwm.core.grid import make_base_state, make_vertical_coord
        from gpuwm.core.moist import init_moist_balanced
        from gpuwm.core.physics import initialize_physics

        self.cp = cp
        self.cfg = RunConfig(
            nx=NCOL, ny=1, nz=NZ, dx=DX, dy=DX, ztop=ZTOP, dt=DT,
            run_seconds=0.0, **scheme_cfg_kwargs(cumulus))
        coord = make_vertical_coord(NZ)          # uniform eta: fnm=fnp=0.5
        base = make_base_state(
            coord, lambda z: 300.0 + 0.003 * np.asarray(z),
            p_surf=self.cfg.p_surf, ztop=ZTOP)
        state = init_moist_balanced(
            self.cfg, coord, base,
            lambda z: 0.012 * np.exp(-np.asarray(z) / 2500.0))

        rng = np.random.default_rng(20260810)
        state.thp[...] += cp.asarray(
            0.3 * rng.standard_normal((NZ, 1, NCOL)), dtype=cp.float32)
        state.qv[...] *= cp.asarray(
            1.0 + 0.02 * rng.standard_normal((NZ, 1, NCOL)),
            dtype=cp.float32)
        state.qc[...] = cp.asarray(
            np.where(np.linspace(0, ZTOP, NZ)[:, None, None] < 4000.0,
                     6.0e-4, 0.0)
            * (1.0 + 0.05 * rng.standard_normal((NZ, 1, NCOL))),
            dtype=cp.float32)
        state.qr[...] = cp.float32(1.0e-5)
        # x-uniform sheared winds: destagger is exact.
        uz = np.float32(5.0 + 10.0 * np.linspace(0, 1, NZ))
        vz = np.float32(-3.0 - 6.0 * np.linspace(0, 1, NZ))
        state.u[...] = cp.asarray(uz)[:, None, None]
        state.v[...] = cp.asarray(vz)[:, None, None]
        state.w[...] = cp.asarray(
            np.float32(0.05 * np.sin(np.linspace(0, np.pi, NZ + 1)))
        )[:, None, None]

        sfc = surface_setup()
        self.driver = initialize_physics(
            state, self.cfg,
            landmask=sfc["landmask"].reshape(1, NCOL),
            tsk=sfc["tsk"].reshape(1, NCOL),
            soil_temperature=285.0, soil_moisture=0.30,
            ivgtyp=sfc["ivgtyp"].reshape(1, NCOL),
            isltyp=sfc["isltyp"].reshape(1, NCOL),
            vegfra=sfc["vegfra"].reshape(1, NCOL),
            tmn=sfc["tmn"].reshape(1, NCOL),
            xice=0.0, snow=0.0, snow_depth=0.0, glw=None,
            radiation_start_time=START,
            radiation_latitude=sfc["latitude"].reshape(1, NCOL),
            radiation_longitude=sfc["longitude"].reshape(1, NCOL))
        self.state = state
        self.phb3 = (state.phb[:, None, None]
                     if state.phb.ndim == 1 else state.phb)

    def chm(self):
        s = self.state
        return (s.c1h[:, None, None] * s.total_mu()[None]
                + s.c2h[:, None, None])

    def pre_physics(self):
        """dycore.py:2324: refresh diagnostics before the physics entry."""
        from gpuwm.core.diagnostics import update_diagnostics
        update_diagnostics(self.state, self.cfg.hypsometric_opt)

    def compute(self):
        """dycore.py:2325 verbatim."""
        return self.driver.compute(self.state, self.cfg)

    def atmosphere(self):
        from gpuwm.core.physics import _prepare_atmosphere
        return _prepare_atmosphere(self.state)

    def pre_microphysics(self):
        """dycore.py:2491: post-RK diagnostics refresh before WSM6."""
        from gpuwm.core.diagnostics import update_diagnostics
        update_diagnostics(self.state, self.cfg.hypsometric_opt)

    def microphysics(self):
        """dycore.py:2507-2512: the post-RK pair plus the clock."""
        from gpuwm.core.microphysics import apply as arw_apply
        result = arw_apply(self.state, self.cfg, self.cfg.dt,
                           refl_10cm_due=False)
        self.driver.accept_microphysics(result)
        self.state.elapsed_seconds += self.cfg.dt
        return result

    def manifest(self):
        from gpuwm.io.restart import _driver_manifest
        return _driver_manifest(self.driver)


# manifest keys that hold chm-COUPLED stacks on the ARW leg (raw on the
# seam's identity state): excluded from the raw bitwise cross-compare and
# proven instead through the fl(chm * raw) component comparisons.
_COUPLED_PREFIXES = ("driver/tendencies/", "driver/pbl_tendencies/",
                     "driver/radiation_tendencies/",
                     "driver/cumulus_tendencies/")


def build_seam(cumulus: str, seam_cls=None):
    from gpuwm.core import mpas_column_batch as mcb
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    coord = make_vertical_coord(NZ)
    base = make_base_state(coord, lambda z: 300.0 + 0.003 * np.asarray(z),
                           p_surf=1.0e5, ztop=ZTOP)
    del coord
    sfc = surface_setup()
    cls = seam_cls or mcb.MpasColumnBatchPhysics
    return cls(
        n_levels=NZ, n_columns=NCOL, dt=DT,
        radiation_seconds=600.0, surface_pbl_seconds=120.0,
        cumulus_seconds=(120.0 if cumulus == "gf" else 600.0),
        cumulus_scheme=cumulus,
        start_time=START,
        latitude_deg=sfc["latitude"], longitude_deg=sfc["longitude"],
        terrain_height_m=sfc["terrain"],
        z_interface_nominal_m=np.linspace(0.0, ZTOP, NZ + 1),
        # DomainState.load_base stores p_top as fp32; the "same column
        # state" is therefore the fp32-rounded value.
        p_top_pa=float(np.float32(base.p_top)), dx_m=DX,
        landmask=sfc["landmask"], ivgtyp=sfc["ivgtyp"],
        isltyp=sfc["isltyp"], vegfra=sfc["vegfra"], tsk=sfc["tsk"],
        tmn=sfc["tmn"], soil_temperature=285.0, soil_moisture=0.30)


# ----------------------------------------------------------------------
# the bound-marshalling seam (ARM A): identical orchestration, with the
# two documented phase-2 round trips bound bitwise to the ARW fields.
# Harness-only diagnostic subclass; the product seam is never modified.
# ----------------------------------------------------------------------

_BOUND = {"alt": None, "ph": None}


def _bound_phase2_apply(state, cfg, dt, *, refl_10cm_due=False):
    from gpuwm.core.microphysics import apply as arw_apply
    state.alt[...] = _BOUND["alt"]
    state.php[...] = _BOUND["ph"]
    return arw_apply(state, cfg, dt, refl_10cm_due=refl_10cm_due)


def make_bound_cls():
    from gpuwm.core import mpas_column_batch as mcb

    class BoundSeam(mcb.MpasColumnBatchPhysics):
        _PHASE2_MICROPHYSICS = staticmethod(_bound_phase2_apply)

    return BoundSeam


# ----------------------------------------------------------------------
# the per-step MPAS RK/transport stand-in (proof 4)
# ----------------------------------------------------------------------

def perturbation(step: int):
    import cupy as cp
    rng = np.random.default_rng(7000 + step)
    return {
        "theta": cp.asarray(
            0.02 * rng.standard_normal((NZ, 1, NCOL)), dtype=cp.float32),
        "qv": cp.asarray(
            2.0e-6 * rng.standard_normal((NZ, 1, NCOL)), dtype=cp.float32),
        "qc": cp.asarray(
            1.0e-7 * rng.standard_normal((NZ, 1, NCOL)), dtype=cp.float32),
    }


def perturb_arw(leg: ArwLeg, pert):
    """Total-theta-space perturbation, positive-definite species."""
    cp = leg.cp
    s = leg.state
    total = s.total_theta() + pert["theta"]        # fl(T + d)
    thb3 = s.thb[:, None, None] if s.thb.ndim == 1 else s.thb
    s.thp[...] = total - thb3                      # exact (Sterbenz)
    s.qv[...] = cp.maximum(s.qv + pert["qv"], cp.float32(0.0))
    s.qc[...] = cp.maximum(s.qc + pert["qc"], cp.float32(0.0))


def perturb_caller(cp, caller, pert):
    """The same perturbation applied to a caller-owned array set."""
    caller["theta"] += pert["theta"].reshape(NZ, NCOL)
    caller["qv"][...] = cp.maximum(
        caller["qv"] + pert["qv"].reshape(NZ, NCOL), cp.float32(0.0))
    caller["qc"][...] = cp.maximum(
        caller["qc"] + pert["qc"].reshape(NZ, NCOL), cp.float32(0.0))


# ----------------------------------------------------------------------
# marshalling: the "same column states", spelled the way the physics
# actually receives them.
# ----------------------------------------------------------------------

def caller_arrays(leg: ArwLeg):
    """Caller-side (MPAS-owned) prognostic arrays, from the ARW state.

    Explicit copies: the caller owns its memory; nothing here may alias
    the ARW state (phase 2 writes these in place).
    """
    cp = leg.cp
    s = leg.state
    out = {"theta": cp.ascontiguousarray(
        s.total_theta().reshape(NZ, NCOL)).copy()}
    for name in _SPECIES:
        out[name] = cp.ascontiguousarray(
            getattr(s, name).reshape(NZ, NCOL)).copy()
    return out


def resync_caller(leg: ArwLeg, caller):
    """Overwrite caller contents (never identities) from the ARW state."""
    caller["theta"][...] = leg.state.total_theta().reshape(NZ, NCOL)
    for name in _SPECIES:
        caller[name][...] = getattr(leg.state, name).reshape(NZ, NCOL)


def phase1_inputs(leg: ArwLeg, caller, *, with_exner=True):
    cp = leg.cp
    atm = leg.atmosphere()

    def flat(a):
        return cp.ascontiguousarray(a.reshape(a.shape[0], NCOL))

    inputs = {
        "u": flat(atm["u"]), "v": flat(atm["v"]),
        "theta": caller["theta"],
        "pressure": flat(atm["pressure"]),
        "pressure_interface": flat(atm["p_interface"]),
        "z_interface": flat(atm["z_interface"]),
        "w": flat(leg.state.w),
        "rho_dry": cp.ascontiguousarray(
            (cp.float32(1.0) / leg.state.alt).reshape(NZ, NCOL)),
        "qv": caller["qv"], "qc": caller["qc"], "qr": caller["qr"],
        "qi": caller["qi"], "qs": caller["qs"], "qg": caller["qg"],
    }
    if with_exner:
        inputs["exner"] = flat(atm["exner"])
    return inputs, atm


def phase2_inputs(leg: ArwLeg, caller):
    cp = leg.cp
    s = leg.state
    from gpuwm.core import constants as c
    ph_total = leg.phb3 + s.php                       # fl(phb + php)
    inputs = {
        "theta": caller["theta"], "qv": caller["qv"], "qc": caller["qc"],
        "qr": caller["qr"], "qi": caller["qi"], "qs": caller["qs"],
        "qg": caller["qg"],
        "pressure": cp.ascontiguousarray(s.p.reshape(NZ, NCOL)),
        "rho_dry": cp.ascontiguousarray(
            (cp.float32(1.0) / s.alt).reshape(NZ, NCOL)),
        "z_interface": cp.ascontiguousarray(
            (ph_total / cp.float32(c.G)).reshape(NZ + 1, NCOL)),
    }
    return inputs, ph_total


def with_caller(inputs, caller):
    """Swap the prognostic entries of an input dict for another caller."""
    merged = dict(inputs)
    for name in _PROGNOSTICS:
        merged[name] = caller[name]
    return merged


# ----------------------------------------------------------------------
# comparisons
# ----------------------------------------------------------------------

_COMPONENTS = ("pbl_tendencies", "radiation_tendencies",
               "cumulus_tendencies")
_STACK_FIELDS = ("rtheta", "rqv", "rqc", "rqr", "rqi", "rqs")


def compare_phase1(check: Check, step, leg: ArwLeg, seam, result):
    cp = leg.cp
    chm = leg.chm()
    sd = seam._driver
    ad = leg.driver
    for comp in _COMPONENTS:
        s_comp = getattr(sd, comp)
        a_comp = getattr(ad, comp)
        if s_comp is None or a_comp is None:
            check.true(f"s{step}/{comp}/presence",
                       (s_comp is None) == (a_comp is None))
            continue
        for field in _STACK_FIELDS:
            sv = getattr(s_comp, field)
            av = getattr(a_comp, field)
            if sv is None or av is None:
                check.true(f"s{step}/{comp}/{field}/presence",
                           (sv is None) == (av is None))
                continue
            check.bitwise(f"s{step}/{comp}/{field}", chm * sv, av)
    # raw retained momentum and radiation rates: bitwise, no factor.
    check.true(f"s{step}/last_ysu/presence",
               (sd.last_ysu is None) == (ad.last_ysu is None))
    if sd.last_ysu is not None and ad.last_ysu is not None:
        for field in ("du", "dv", "dtheta", "dqv", "dqc"):
            check.bitwise(f"s{step}/last_ysu/{field}",
                          sd.last_ysu[field], ad.last_ysu[field])
        check.bitwise(f"s{step}/out_du", result.du,
                      ad.last_ysu["du"].reshape(NZ, NCOL))
        check.bitwise(f"s{step}/out_dv", result.dv,
                      ad.last_ysu["dv"].reshape(NZ, NCOL))
    check.bitwise(f"s{step}/rthratenlw", sd.rthratenlw, ad.rthratenlw)
    check.bitwise(f"s{step}/rthratensw", sd.rthratensw, ad.rthratensw)
    check.bitwise(f"s{step}/h_diabatic", result.h_diabatic,
                  leg.state.h_diabatic.reshape(NZ, NCOL))
    # the seam's composed output IS its driver's composed stack, and on
    # the identity state that stack IS raw (chm == 1): both byte-visible.
    check.bitwise(f"s{step}/out_dtheta", result.dtheta,
                  sd.tendencies.rtheta.reshape(NZ, NCOL))
    check.true(f"s{step}/dqg_zero",
               float(cp.abs(result.dqg).max()) == 0.0)
    check.true(f"s{step}/counters",
               dict(sd.call_counts) == dict(ad.call_counts))


def compare_manifests(check: Check, step, leg: ArwLeg, seam):
    """Raw persistent driver state, cross-leg, bitwise."""
    from gpuwm.io.restart import _driver_manifest
    a_man = leg.manifest()
    s_man = _driver_manifest(seam._driver)
    a_keys = {key for key in a_man if not key.startswith(_COUPLED_PREFIXES)}
    s_keys = {key for key in s_man if not key.startswith(_COUPLED_PREFIXES)}
    check.true(f"s{step}/manifest/keys", a_keys == s_keys)
    for key in sorted(a_keys & s_keys):
        check.bitwise(f"s{step}/manifest/{key}", s_man[key], a_man[key])


def compare_phase2(check: Check, step, leg: ArwLeg, seam, caller, receipt,
                   mp_result):
    """Species/receipts/buckets bitwise; theta carrier <= residual (b)."""
    cp = leg.cp
    s = leg.state
    theta_stats = ulp_stats(caller["theta"],
                            s.total_theta().reshape(NZ, NCOL))
    check.true(f"s{step}/p2/theta_carrier<= {THETA_CARRIER_ULP_BOUND}ulp",
               theta_stats["max_ulp"] <= THETA_CARRIER_ULP_BOUND)
    for name in _SPECIES:
        check.bitwise(f"s{step}/p2/{name}", caller[name],
                      getattr(s, name).reshape(NZ, NCOL))
    check.bitwise(f"s{step}/p2/h_diabatic", seam._state.h_diabatic,
                  s.h_diabatic)
    for name in ("rainncv", "snowncv", "graupelncv", "sr"):
        check.bitwise(f"s{step}/p2/{name}", receipt[name],
                      getattr(mp_result, name).reshape(NCOL))
    for name in ("effc", "effi", "effs"):
        check.bitwise(f"s{step}/p2/{name}", getattr(seam._state, name),
                      getattr(s, name))
    buckets = seam.accumulated_precipitation()
    for name, slot in (("RAINNC", "mp_rainnc"), ("SNOWNC", "mp_snownc"),
                       ("GRAUPELNC", "mp_graupelnc")):
        check.bitwise(f"s{step}/bucket/{name}", buckets[name],
                      s.scratch((1, NCOL), slot).reshape(NCOL))
    rainc = leg.driver.rainc
    if rainc is None:
        check.true(f"s{step}/bucket/RAINC",
                   float(cp.abs(buckets["RAINC"]).max()) == 0.0)
    else:
        check.bitwise(f"s{step}/bucket/RAINC", buckets["RAINC"],
                      rainc.reshape(NCOL))
    return theta_stats


# ----------------------------------------------------------------------
# ARM A: bound bit-for-bit trajectory (proofs 1, 2, 3, 4)
# ----------------------------------------------------------------------

def run_arm_a(cumulus: str, steps: int) -> dict:
    import cupy as cp
    from gpuwm.core.mpas_column_batch import _OUTPUT_BUFFERS
    leg = ArwLeg(cumulus)
    bound_cls = make_bound_cls()
    seam = build_seam(cumulus, bound_cls)

    check = Check(f"armA/{cumulus}")
    # constructional identities
    check.bitwise("fnm", seam._state.fnm, leg.state.fnm)
    check.bitwise("fnp", seam._state.fnp, leg.state.fnp)
    for name in ("effc", "effi", "effs"):
        check.bitwise(f"init/{name}", getattr(seam._state, name),
                      getattr(leg.state, name))
    check.true("p_top", seam._state.p_top == float(leg.state.p_top))
    check.true("chm_identity", float(cp.abs(
        seam._state.c1h[:, None, None] * seam._state.total_mu()[None]
        + seam._state.c2h[:, None, None] - cp.float32(1.0)).max()) == 0.0)
    check.raise_if_red()

    caller = caller_arrays(leg)
    resumed = None
    resumed_caller = None
    payload = None

    held = []           # (lw_bytes, sw_bytes) per step, cadence proof
    flags = []
    bucket_series = []
    theta_carrier = []
    t0 = time.time()
    for step in range(steps):
        leg.pre_physics()
        resync_caller(leg, caller)
        if resumed is not None:
            resync_caller(leg, resumed_caller)
        inputs, _ = phase1_inputs(leg, caller, with_exner=True)
        if step == 0:
            before = {name: _bytes(value)
                      for name, value in inputs.items()}
        result = seam.run_phase1(dt=DT, **inputs)
        if step == 0:
            for name, value in inputs.items():
                check.true(f"s0/read_only/{name}",
                           _bytes(value) == before[name])
        leg.compute()
        compare_phase1(check, step, leg, seam, result)
        if step in (0, RESTART_STEP - 1, steps - 1):
            compare_manifests(check, step, leg, seam)
        held.append((_bytes(seam._driver.rthratenlw),
                     _bytes(seam._driver.rthratensw)))
        flags.append((result.radiation_ran, result.surface_pbl_ran,
                      result.cumulus_ran))

        if resumed is not None:
            res_result = resumed.run_phase1(
                dt=DT, **with_caller(inputs, resumed_caller))
            for name in _OUTPUT_BUFFERS:
                check.bitwise(f"s{step}/restart/{name}",
                              getattr(res_result, name),
                              getattr(result, name))

        pert = perturbation(step)
        perturb_arw(leg, pert)
        leg.pre_microphysics()
        resync_caller(leg, caller)      # the same perturbed state
        if resumed is not None:
            resync_caller(leg, resumed_caller)
        p2, ph_total = phase2_inputs(leg, caller)
        _BOUND["alt"] = leg.state.alt
        _BOUND["ph"] = ph_total
        receipt = seam.run_phase2(**p2)
        receipt_res = None
        if resumed is not None:
            receipt_res = resumed.run_phase2(
                **with_caller(p2, resumed_caller))
        mp_result = leg.microphysics()
        theta_carrier.append(
            compare_phase2(check, step, leg, seam, caller, receipt,
                           mp_result))
        if resumed is not None:
            for name in ("rainncv", "snowncv", "graupelncv", "sr"):
                check.bitwise(f"s{step}/restart/p2/{name}",
                              receipt_res[name], receipt[name])
            for name in _PROGNOSTICS:
                check.bitwise(f"s{step}/restart/state/{name}",
                              resumed_caller[name], caller[name])
        bucket_series.append(
            cp.asnumpy(seam.accumulated_precipitation()["RAINNC"]).copy())

        if step + 1 == RESTART_STEP and steps > RESTART_STEP:
            payload = seam.export_state()          # proof 3: boundary
            resumed = build_seam(cumulus, bound_cls)
            resumed.restore_state(payload)
            resumed_caller = {name: value.copy()
                              for name, value in caller.items()}
        check.raise_if_red()

    # cadence semantics (proof 2) on the seam's own held buffers
    for step, (lw, sw) in enumerate(held):
        due = ((step + 1) % STEPRA) == 1
        if due and step > 0:
            check.true(f"cadence/change@{step}",
                       lw != held[step - 1][0] or sw != held[step - 1][1])
        elif step > 0:
            check.true(f"cadence/hold@{step}",
                       lw == held[step - 1][0] and sw == held[step - 1][1])
        check.true(f"cadence/flag@{step}", flags[step][0] == due)
        check.true(f"cadence/pbl@{step}", flags[step][1])
    for prev, cur in zip(bucket_series, bucket_series[1:]):
        check.true("bucket/monotone", bool((cur >= prev).all()))
    check.true("bucket/nonzero", float(bucket_series[-1].max()) > 0.0)
    if resumed is not None:
        check.true("restart/step_index",
                   resumed.step_index == seam.step_index)
        check.true("restart/counters",
                   resumed.call_counts == seam.call_counts)
    check.raise_if_red()

    worst_carrier = max(theta_carrier, key=lambda st: st["max_ulp"])
    return {
        "arm": "A-bound-bitwise", "cumulus": cumulus, "steps": steps,
        "hours_simulated": steps * DT / 3600.0,
        "comparisons": check.compared, "failures": 0,
        "restart_split_step": RESTART_STEP if payload is not None else None,
        "call_counts": dict(seam.call_counts),
        "final_rainnc_max_mm": float(bucket_series[-1].max()),
        "radiation_due_steps_first10": [s + 1 for s, f in
                                        enumerate(flags[:10]) if f[0]],
        "cumulus_due_steps_first10": [s + 1 for s, f in
                                      enumerate(flags[:10]) if f[2]],
        "theta_carrier_residual_worst_step": worst_carrier,
        "wall_seconds": round(time.time() - t0, 1),
    }


# ----------------------------------------------------------------------
# ARM B: stock seam, naive contract marshalling, independent trajectory
# ----------------------------------------------------------------------

def run_arm_b(cumulus: str, steps: int) -> dict:
    import cupy as cp
    leg = ArwLeg(cumulus)
    seam = build_seam(cumulus)
    caller = caller_arrays(leg)
    restart_check = Check(f"armB/{cumulus}/restart")
    resumed = None
    resumed_caller = None
    from gpuwm.core.mpas_column_batch import _OUTPUT_BUFFERS

    # round-trip input census at step 0: how many elements do the two
    # naive marshalling round trips actually move, and how far?
    from gpuwm.core import constants as c
    leg.pre_physics()
    alt = leg.state.alt
    alt_rt = cp.float32(1.0) / (cp.float32(1.0) / alt)
    ph = leg.phb3 + leg.state.php
    ph_rt = (ph / cp.float32(c.G)) * cp.float32(c.G)
    census = {"alt_round_trip": ulp_stats(alt_rt, alt),
              "ph_round_trip": ulp_stats(ph_rt, ph)}

    drift = []
    for step in range(steps):
        leg.pre_physics()
        inputs, _ = phase1_inputs(leg, caller, with_exner=True)
        result = seam.run_phase1(dt=DT, **inputs)
        leg.compute()
        if resumed is not None:
            res_result = resumed.run_phase1(
                dt=DT, **with_caller(inputs, resumed_caller))
            for name in _OUTPUT_BUFFERS:
                restart_check.bitwise(f"s{step}/{name}",
                                      getattr(res_result, name),
                                      getattr(result, name))
        row = {"step": step + 1}
        chm = leg.chm()
        sd = seam._driver
        row["pbl_rtheta"] = ulp_stats(
            chm * sd.pbl_tendencies.rtheta,
            leg.driver.pbl_tendencies.rtheta)
        if sd.last_ysu is not None and leg.driver.last_ysu is not None:
            row["ysu_du"] = ulp_stats(sd.last_ysu["du"],
                                      leg.driver.last_ysu["du"])
        row["rthratenlw"] = ulp_stats(sd.rthratenlw, leg.driver.rthratenlw)

        pert = perturbation(step)
        perturb_arw(leg, pert)
        perturb_caller(cp, caller, pert)
        if resumed is not None:
            perturb_caller(cp, resumed_caller, pert)
        leg.pre_microphysics()
        p2, _ = phase2_inputs(leg, caller)
        receipt = seam.run_phase2(**p2)
        if resumed is not None:
            resumed.run_phase2(**with_caller(p2, resumed_caller))
            for name in _PROGNOSTICS:
                restart_check.bitwise(f"s{step}/state/{name}",
                                      resumed_caller[name], caller[name])
        mp_result = leg.microphysics()
        row["theta"] = ulp_stats(caller["theta"],
                                 leg.state.total_theta().reshape(NZ, NCOL))
        row["qv"] = ulp_stats(caller["qv"],
                              leg.state.qv.reshape(NZ, NCOL))
        row["rainncv"] = ulp_stats(receipt["rainncv"],
                                   mp_result.rainncv.reshape(NCOL))
        row["RAINNC"] = ulp_stats(
            seam.accumulated_precipitation()["RAINNC"],
            leg.state.scratch((1, NCOL), "mp_rainnc").reshape(NCOL))
        drift.append(row)
        if step + 1 == RESTART_STEP and steps > RESTART_STEP:
            payload = seam.export_state()
            resumed = build_seam(cumulus)
            resumed.restore_state(payload)
            resumed_caller = {name: value.copy()
                              for name, value in caller.items()}
    restart_check.raise_if_red()   # proof 3 on the STOCK seam: bitwise
    milestones = {1, 2, 5, 15, 45, 90, steps}
    return {
        "arm": "B-contract-drift", "cumulus": cumulus, "steps": steps,
        "input_round_trip_census": census,
        "drift_milestones": [row for row in drift
                             if row["step"] in milestones],
        "final_theta_drift": drift[-1]["theta"],
        "final_bucket_drift": drift[-1]["RAINNC"],
        "restart_bitwise_on_stock_seam": True,
        "restart_comparisons": restart_check.compared,
    }


# ----------------------------------------------------------------------
# ARM C: the exner-source residual (single-pressure caller), one step
# ----------------------------------------------------------------------

def run_arm_c(cumulus: str) -> dict:
    import cupy as cp
    from gpuwm.core import constants as c
    leg = ArwLeg(cumulus)
    seam = build_seam(cumulus)
    caller = caller_arrays(leg)
    leg.pre_physics()
    inputs, atm = phase1_inputs(leg, caller, with_exner=False)
    result = seam.run_phase1(dt=DT, **inputs)
    leg.compute()
    exner_delta = ulp_stats(
        (inputs["pressure"] / cp.float32(c.P0)) ** cp.float32(c.RCP),
        atm["exner"].reshape(NZ, NCOL))
    chm = leg.chm()
    sd = seam._driver
    return {
        "arm": "C-exner-source", "cumulus": cumulus,
        "exner_input_delta": exner_delta,
        "pbl_rtheta_delta": ulp_stats(
            chm * sd.pbl_tendencies.rtheta,
            leg.driver.pbl_tendencies.rtheta),
        "radiation_lw_delta": ulp_stats(sd.rthratenlw,
                                        leg.driver.rthratenlw),
        "ysu_du_delta": (ulp_stats(sd.last_ysu["du"],
                                   leg.driver.last_ysu["du"])
                         if sd.last_ysu is not None
                         and leg.driver.last_ysu is not None else None),
        "out_dtheta_delta": ulp_stats(
            chm * result.dtheta.reshape(NZ, 1, NCOL),
            leg.driver.tendencies.rtheta),
    }


# ----------------------------------------------------------------------
# determinism control: the instrument reads zero iff zero is true
# ----------------------------------------------------------------------

def run_control(cumulus: str, steps: int = 5) -> dict:
    manifests = []
    for _ in range(2):
        leg = ArwLeg(cumulus)
        for step in range(steps):
            leg.pre_physics()
            leg.compute()
            perturb_arw(leg, perturbation(step))
            leg.pre_microphysics()
            leg.microphysics()
        manifests.append({key: _bytes(value)
                          for key, value in leg.manifest().items()})
    same = (manifests[0].keys() == manifests[1].keys()
            and all(manifests[0][key] == manifests[1][key]
                    for key in manifests[0]))
    if not same:
        raise ProofFailure(
            "ARW leg is not run-to-run deterministic on this device; "
            "no bitwise comparison below would mean anything")
    return {"arm": "control-determinism", "cumulus": cumulus,
            "steps": steps, "bitwise_repeatable": True,
            "manifest_keys": len(manifests[0])}


# ----------------------------------------------------------------------
# ARM R: instrument validation (proof 5) -- each breakage must go red
# ----------------------------------------------------------------------

def _run_bound_steps(leg, seam, caller, check, first, count,
                     break_hook=None):
    for step in range(first, first + count):
        leg.pre_physics()
        resync_caller(leg, caller)
        inputs, _ = phase1_inputs(leg, caller, with_exner=True)
        result = seam.run_phase1(dt=DT, **inputs)
        leg.compute()
        if break_hook is not None:
            break_hook(step, seam, leg)
        compare_phase1(check, step, leg, seam, result)
        perturb_arw(leg, perturbation(step))
        leg.pre_microphysics()
        resync_caller(leg, caller)
        p2, ph_total = phase2_inputs(leg, caller)
        _BOUND["alt"] = leg.state.alt
        _BOUND["ph"] = ph_total
        receipt = seam.run_phase2(**p2)
        mp_result = leg.microphysics()
        compare_phase2(check, step, leg, seam, caller, receipt, mp_result)


def run_red_arms(cumulus: str) -> dict:
    import cupy as cp
    from gpuwm.core.mpas_column_batch import _OUTPUT_BUFFERS
    bound_cls = make_bound_cls()
    out = {}

    # R1: drop the held radiation tendency on a non-due step.
    leg = ArwLeg(cumulus)
    seam = build_seam(cumulus, bound_cls)
    caller = caller_arrays(leg)
    check = Check("red/held-drop")

    def drop_held(step, seam, leg):
        if step == 2:                      # non-due (due steps are 1, 6)
            seam._driver.radiation_tendencies.rtheta[...] = 0.0

    _run_bound_steps(leg, seam, caller, check, 0, 4, drop_held)
    out["R1_drop_held_radiation"] = {
        "went_red": bool(check.failures),
        "failures": len(check.failures),
        "first_failure": check.failures[0] if check.failures else None}

    # R2: corrupt the precipitation bucket mid-run.
    leg = ArwLeg(cumulus)
    seam = build_seam(cumulus, bound_cls)
    caller = caller_arrays(leg)
    check = Check("red/bucket")
    _run_bound_steps(leg, seam, caller, check, 0, 3)
    pre_failures = len(check.failures)

    def corrupt_bucket(step, seam, leg):
        if step == 3:
            seam._state._scratch["mp_rainnc"][...] *= cp.float32(0.5)

    _run_bound_steps(leg, seam, caller, check, 3, 2, corrupt_bucket)
    bucket_red = [f for f in check.failures if "/bucket/" in f["tag"]]
    out["R2_corrupt_bucket"] = {
        "went_red": bool(bucket_red), "pre_failures": pre_failures,
        "first_failure": bucket_red[0] if bucket_red else None}

    # R3: SILENT restart corruption -- fields/tsk moved by one ULP in an
    # otherwise valid payload; the continuation must diverge byte-visibly.
    leg = ArwLeg(cumulus)
    seam = build_seam(cumulus, bound_cls)
    caller = caller_arrays(leg)
    clean = Check("red/restart-clean")
    _run_bound_steps(leg, seam, caller, clean, 0, 3)
    payload = seam.export_state()
    corrupt = {"identity": dict(payload["identity"]),
               "arrays": dict(payload["arrays"]),
               "scalars": {key: (dict(value) if isinstance(value, dict)
                                 else value)
                           for key, value in payload["scalars"].items()}}
    key = "fields/tsk"
    bumped = np.array(corrupt["arrays"][key], copy=True)
    flat = bumped.reshape(-1)
    flat[0] = np.nextafter(flat[0], np.float32(np.inf))
    corrupt["arrays"][key] = bumped
    resumed = build_seam(cumulus, bound_cls)
    resumed.restore_state(corrupt)
    reference = build_seam(cumulus, bound_cls)
    reference.restore_state(payload)
    ref_caller = {name: value.copy() for name, value in caller.items()}
    res_caller = {name: value.copy() for name, value in caller.items()}
    diverged = Check("red/restart-corrupt")
    for step in range(3, 8):
        leg.pre_physics()
        resync_caller(leg, ref_caller)
        resync_caller(leg, res_caller)
        inputs, _ = phase1_inputs(leg, ref_caller, with_exner=True)
        r_ref = reference.run_phase1(dt=DT, **inputs)
        r_res = resumed.run_phase1(
            dt=DT, **with_caller(inputs, res_caller))
        for name in _OUTPUT_BUFFERS:
            diverged.bitwise(f"s{step}/{name}", getattr(r_res, name),
                             getattr(r_ref, name))
        leg.compute()
        perturb_arw(leg, perturbation(step))
        leg.pre_microphysics()
        resync_caller(leg, ref_caller)
        resync_caller(leg, res_caller)
        p2, ph_total = phase2_inputs(leg, ref_caller)
        _BOUND["alt"] = leg.state.alt
        _BOUND["ph"] = ph_total
        reference.run_phase2(**p2)
        resumed.run_phase2(**with_caller(p2, res_caller))
        leg.microphysics()
    out["R3_silent_restart_corruption"] = {
        "corrupted_key": key, "went_red": bool(diverged.failures),
        "failures": len(diverged.failures),
        "first_failure": diverged.failures[0] if diverged.failures
        else None}

    # R4: break the identity coupling (the chm == 1 construction).
    leg = ArwLeg(cumulus)
    seam = build_seam(cumulus, bound_cls)
    caller = caller_arrays(leg)
    check = Check("red/identity-coupling")
    seam._state.c2h[...] = cp.float32(1.0e-4)
    _run_bound_steps(leg, seam, caller, check, 0, 1)
    out["R4_break_identity_coupling"] = {
        "went_red": bool(check.failures),
        "failures": len(check.failures),
        "first_failure": check.failures[0] if check.failures else None}

    # R5: cadence corruption -- radiation silently rescheduled.
    leg = ArwLeg(cumulus)
    seam = build_seam(cumulus, bound_cls)
    caller = caller_arrays(leg)
    check = Check("red/cadence")
    seam._driver.radt_minutes = 2.0        # 600 s contract -> 120 s
    flags = []
    for step in range(3):
        leg.pre_physics()
        resync_caller(leg, caller)
        inputs, _ = phase1_inputs(leg, caller, with_exner=True)
        result = seam.run_phase1(dt=DT, **inputs)
        leg.compute()
        flags.append(result.radiation_ran)
        perturb_arw(leg, perturbation(step))
        leg.pre_microphysics()
        resync_caller(leg, caller)
        p2, ph_total = phase2_inputs(leg, caller)
        _BOUND["alt"] = leg.state.alt
        _BOUND["ph"] = ph_total
        seam.run_phase2(**p2)
        leg.microphysics()
    schedule_red = any(flag != (((step + 1) % STEPRA) == 1)
                       for step, flag in enumerate(flags))
    counter_red = (seam.call_counts["radiation"]
                   != leg.driver.call_counts["radiation"])
    out["R5_cadence_corruption"] = {
        "went_red": bool(schedule_red or counter_red),
        "flags": flags, "schedule_red": schedule_red,
        "counter_red": counter_red}

    return out


# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--arms", default="control,A,B,C,R")
    args = parser.parse_args()

    import pathlib
    import subprocess
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import cupy as cp
    device = cp.cuda.runtime.getDeviceProperties(0)
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=pathlib.Path(__file__).resolve().parent.parent,
        ).stdout.strip()
    except Exception:
        sha = "unknown"
    receipt = {
        "schema": PROOF_SCHEMA,
        "git_sha": sha,
        "device": device["name"].decode(),
        "cupy": cp.__version__,
        "grid": {"nz": NZ, "ncol": NCOL, "dt_s": DT,
                 "steps": args.steps,
                 "hours": args.steps * DT / 3600.0},
        "cadences_s": {"radiation": 600.0, "surface_pbl": 120.0,
                       "cumulus_gf": 120.0, "cumulus_kf": 600.0},
        "arms": {},
    }
    arms = args.arms.split(",")
    status = 0
    for cumulus in ("kf", "gf"):
        section = {}
        try:
            if "control" in arms:
                section["control"] = run_control(cumulus)
            if "A" in arms:
                section["A"] = run_arm_a(cumulus, args.steps)
            if "B" in arms:
                section["B"] = run_arm_b(cumulus, args.steps)
            if "C" in arms:
                section["C"] = run_arm_c(cumulus)
            if "R" in arms:
                section["R"] = run_red_arms(cumulus)
                for name, red in section["R"].items():
                    if not red["went_red"]:
                        raise ProofFailure(
                            f"instrument validation {name} did NOT go "
                            "red: the detector is blind")
        except ProofFailure as failure:
            section["FAILED"] = str(failure)
            status = 1
        except Exception:                    # noqa: BLE001 -- receipt first
            import traceback
            section["ERROR"] = traceback.format_exc()
            status = 1
        receipt["arms"][cumulus] = section

    receipt["status"] = "GO" if status == 0 else "RED"
    path = out_dir / "mpas_seam_proofs.json"
    path.write_text(json.dumps(receipt, indent=1, default=str))
    print(json.dumps({"status": receipt["status"], "receipt": str(path)}))
    return status


if __name__ == "__main__":
    sys.exit(main())
