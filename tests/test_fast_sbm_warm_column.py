"""The warm FAST_SBM call against WRF's own Fortran: the column oracle gate.

``gpuwm.core.fast_sbm.column.fast_sbm_warm_call`` transcribes WRF v4.8.0's
``FAST_SBM`` (``phys/module_mp_fast_sbm.F`` :4001-5322) on its warm path and
calls the process routines through the ``FsbmProcesses`` protocol.  The
gates here (fast-sbm design 5.5) and what each one catches:

1. Glue gate, runnable before the process items merge.  The processes are
   stand-ins that hand back what WRF's own routines returned inside the
   oracle's KC-1 run (``kc1_probe_o64``), and every call the column makes at
   a probed point must pass the inputs WRF passed (relative 1e-12), in WRF's
   order, with WRF's ``DIFFU``, cloud-base and ``ISYM`` decisions.  The
   column-level phases (sedimentation input, diagnostics, surface
   precipitation, ``th_old``/``qv_old``, the exit conversion and its clip)
   must then reproduce the oracle's whole-tile records.  Catches a wrong
   loop order, a wrong gate, a substep that forces T or q wrongly, a process
   handed the wrong level's fall speed or the wrong cloud-base flag, and an
   exit that clips or converts differently from WRF.
2. Per-call replay of the 16 recorded calls on the whole 4-column tile,
   ``wrf_v480`` preset, relative 1e-9 with a 1e-12 floor of the record
   scale: the column plus the real processes against the promoted build.
3. The native envelope on the native build's pre-call states.
4. The host-driven KC-1 trajectory against the Fortran's own one-ULP
   self-divergence (slow).
5. Budgets: the corrected preset closes total water and liquid static
   energy per call and column; the ``wrf_v480`` residual is exactly D2, D3
   and D4; D1 is exactly the surface pressure factor.
6. Refusals: a cold point and an ice bin raise before anything changes.
7. Negative controls that must fail gate 2 (and gate 1's own controls).

Gates 2-5 and 7 run the real process functions of the other wave-2 items
(derived tables, condensation, nucleation, collision and breakup,
sedimentation), merged beside this one; a missing module fails them.

Run from the worktree root with PYTHONPATH set to it:
``C:/Python313/python -m pytest -p no:cacheprovider
tests/test_fast_sbm_warm_column.py``.
"""

from __future__ import annotations

import dataclasses
import functools
from collections import defaultdict, deque
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.fast_sbm.column import (
    COLD_PROCESSES,
    CoalStep,
    CondensationStep,
    FallStep,
    FsbmTile,
    NucleationStep,
    cloud_base_level_wrf,
    default_processes,
    fast_sbm_warm_call,
)
from gpuwm.core.fast_sbm.constants import (
    AL1,
    COL,
    FsbmBehavior,
    FsbmRefusal,
    NKR,
)
from gpuwm.core.fast_sbm.state import CHEM_SLICES
from gpuwm.core.fast_sbm.tables import load_tables
from tools.fast_sbm_wrf480_oracle import fixtures as fx

WRF = FsbmBehavior.wrf_v480()
CORRECTED = FsbmBehavior.corrected()
DT = 12.0
SPECIES = ("ff1", "ff5", "ff6", "ff8")


# ---------------------------------------------------------------------------
# Oracle records
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def tables():
    return load_tables("float64")


@functools.lru_cache(maxsize=None)
def oracle_derived():
    """FAST_HUCMINIT's derived tables as the O64 build holds them, for the
    stand-ins (the derived-tables item's ``build_derived`` is gated against
    the same record)."""
    dump = fx.load_fixture("tables_o64")
    return SimpleNamespace(**{key.split(".", 1)[1]: value[0]
                              for key, value in dump.items()
                              if key.startswith("derived.")})


@functools.lru_cache(maxsize=None)
def case():
    return next(fx.iter_records("kc1_case", types=["case"]))


@functools.lru_cache(maxsize=None)
def statics(build: str):
    return next(fx.iter_records(f"kc1_summary_{build}", types=["initial"]))


@functools.lru_cache(maxsize=None)
def replay(name: str):
    return {step: (pre, post) for step, pre, post in fx.iter_replay(name)}


@functools.lru_cache(maxsize=None)
def probe_records():
    return tuple(fx.iter_records("kc1_probe_o64"))


@functools.lru_cache(maxsize=None)
def summary(build: str):
    return {int(r["step"]): r for r in fx.iter_records(
        f"kc1_summary_{build}", types=["step"])}


@functools.lru_cache(maxsize=None)
def sensitivity():
    return {int(r["step"]): r for r in fx.iter_records(
        "kc1_sensitivity_o64", types=["step"])}


def f8(a) -> np.ndarray:
    return np.array(a, dtype=np.float64)


def tile_from_pre(pre, build: str = "o64") -> FsbmTile:
    """The tile FAST_SBM received at a replay step: the pre-call record and
    the case's fixed fields, promoted to float64 as the cross-replay did."""
    s = statics(build)
    return FsbmTile.from_chem(
        f8(pre["chem"]), th=f8(pre["th"]), qv=f8(pre["qv"]),
        th_old=f8(pre["th_old"]), qv_old=f8(pre["qv_old"]), p=f8(s["p"]),
        pii=f8(s["pii"]), rho=f8(s["rho"]), dz=f8(s["dz8w"]),
        w=f8(pre["w"]), xland=f8(s["xland"]), rainnc=f8(pre["rainnc"]))


def real_processes():
    """The default processes.  Before the wave 2 merge this skipped while
    the other items' modules were absent; they are merged now, so a missing
    module is a failure, never a silent skip of the end-to-end gates."""
    return default_processes()


def real_derived(behavior: FsbmBehavior = WRF):
    """The derived tables ``build_derived`` makes for ``behavior`` (the
    oracle gates run WRF's; D10 decides ``ECOALMASSM``)."""
    return _real_derived(behavior.fall_speed_lookup)


@functools.lru_cache(maxsize=None)
def _real_derived(fall_speed_lookup: str):
    real_processes()
    from gpuwm.core.fast_sbm.derived import build_derived

    return build_derived(tables(), DT, fall_speed_lookup=fall_speed_lookup)


def rel_err(host, rec, scale) -> float:
    """max |host - rec| / scale, 0 when both are identically zero."""
    host, rec = f8(host), f8(rec)
    diff = float(np.max(np.abs(host - rec))) if host.size else 0.0
    if diff == 0.0:
        return 0.0
    return diff / scale if scale > 0.0 else float("inf")


def within(host, rec, *, rel: float, floor: float, scale: float) -> bool:
    """Elementwise ``|host - rec| <= rel |rec| + floor * scale``."""
    host, rec = f8(host), f8(rec)
    return bool(np.all(np.abs(host - rec) <= rel * np.abs(rec)
                       + floor * scale))


def post_fields(result_tile: FsbmTile, rainncv) -> dict[str, np.ndarray]:
    out = {name: getattr(result_tile, name) for name in SPECIES}
    out.update(th=result_tile.th, qv=result_tile.qv,
               th_old=result_tile.th_old, qv_old=result_tile.qv_old,
               rainncv=np.asarray(rainncv), rainnc=result_tile.rainnc)
    return out


def record_fields(post) -> dict[str, np.ndarray]:
    chem = f8(post["chem"])
    out = {name: chem[:, :, cols] for name, cols in CHEM_SLICES.items()}
    for name in ("th", "qv", "th_old", "qv_old", "rainncv", "rainnc"):
        out[name] = f8(post[name])
    return out


def compare_post(result, post, *, rel: float, floor: float
                 ) -> dict[str, tuple[bool, float]]:
    """Each post-call field against the record: (passes, max error over
    the record scale).  The scale of a bin family is its largest value."""
    host = post_fields(result.tile, result.rainncv)
    rec = record_fields(post)
    report = {}
    for name, r in rec.items():
        scale = float(np.max(np.abs(r))) if r.size else 0.0
        report[name] = (within(host[name], r, rel=rel, floor=floor,
                               scale=scale),
                        rel_err(host[name], r, scale))
    return report


# ---------------------------------------------------------------------------
# Stand-in processes that replay WRF's own calls (gate 1)
# ---------------------------------------------------------------------------

#: Natural scale of each scalar a probe record carries: supersaturations and
#: vapour-pressure ratios are compared on a scale of 1, a vertical velocity
#: in cm/s on 1 cm/s; everything else on its own magnitude.
_SCALAR_FLOORS = {"del1in": 1.0, "del2in": 1.0, "div1": 1.0, "div2": 1.0,
                  "w_stag": 1.0}
GLUE_REL = 1e-12


class ProbeStandIns:
    """FsbmProcesses that return what WRF's routines returned in KC-1.

    At a probed point (the probe levels of a probe step) each call is
    matched, in order, against the next recorded call of that point: its
    inputs are compared with the ``*_before`` record and the ``*_after``
    record is handed back.  At every other point WRF's per-point results
    are not recorded, so each call hands back the point's end state (the
    drop spectrum WRF passed to sedimentation, T and qv from the diagnostics
    record, the aerosol from the post-call record); the column-level
    comparisons then check those points too.  ``trace`` receives the
    column's trace events and compares them with the entry, forcing,
    sedimentation and diagnostics records.  Every mismatch is appended to
    ``failures``.
    """

    def __init__(self, step: int, tile: FsbmTile, post):
        self.step = step
        self.failures: list[str] = []
        recs = [r for r in probe_records()
                if "step" in r.fields and int(r["step"]) == step]
        config = next(r for r in probe_records() if r.type == "config")
        self.levels = {int(v) for v in config["levels"]}
        self.calls: dict[tuple[int, int], deque] = defaultdict(deque)
        self.oracle_order: list[tuple[str, tuple[int, int]]] = []
        self.host_order: list[tuple[str, tuple[int, int]]] = []
        self.oracle_forcing: dict[tuple[int, int], list] = defaultdict(list)
        self.host_forcing: dict[tuple[int, int], list] = defaultdict(list)
        self.entry, self.fall_before, self.fall_after, self.diag = {}, {}, {}, {}
        self.kz_host: dict[int, int] = {}
        self.v_bottom: dict[int, np.ndarray] = {}
        pending = {}
        for r in recs:
            kind, _, phase = r.type.partition("_")
            key = (int(r["i"]), int(r["k"]))
            if kind in ("nucl", "cond1", "coal", "spont") and phase in (
                    "before", "after"):
                if phase == "before":
                    pending[(kind, key)] = r
                    self.oracle_order.append((kind, key))
                else:
                    self.calls[key].append((kind, pending.pop((kind, key)),
                                            r))
            elif r.type == "forcing":
                self.oracle_forcing[key].append(r)
            elif r.type == "entry":
                self.entry[key] = r
            elif r.type == "fall_before":
                self.fall_before[key[0]] = r
            elif r.type == "fall_after":
                self.fall_after[key[0]] = r
            elif r.type == "diag":
                self.diag[key[0]] = r
        assert not pending, f"unpaired probe records at step {step}"
        self.ncol, self.nz = tile.ncol, tile.nz
        rhocgs = tile.rho * 0.001
        chem = f8(post["chem"])
        ff8 = chem[:, :, CHEM_SLICES["ff8"]]
        self.end_f, self.end_t, self.end_q, self.end_fccn = {}, {}, {}, {}
        for i in range(1, self.ncol + 1):
            fb, dg = self.fall_before[i], self.diag[i]
            for k in range(1, self.nz + 1):
                self.end_f[(i, k)] = f8(fb["ffx"])[k - 1] * f8(fb["rhocgs"])[k - 1]
                self.end_t[(i, k)] = float(dg["th"][k - 1]) * tile.pii[i - 1, k - 1]
                self.end_q[(i, k)] = float(dg["qv"][k - 1])
                self.end_fccn[(i, k)] = ff8[i - 1, k - 1] * rhocgs[i - 1, k - 1] / 1000.0

    # -- comparisons ---------------------------------------------------------
    def _cmp(self, what: str, name: str, host, rec) -> None:
        rec_a = np.asarray(rec)
        host_a = np.asarray(host)
        if rec_a.dtype.kind == "i" or isinstance(host, (bool, int, np.integer)):
            if int(host) != int(rec):
                self.failures.append(f"{what}: {name} host {host!r} oracle "
                                     f"{rec!r}")
            return
        if rec_a.ndim == 0:
            scale = max(abs(float(rec)), _SCALAR_FLOORS.get(name, 0.0))
            if abs(float(host) - float(rec)) > GLUE_REL * scale:
                self.failures.append(f"{what}: {name} host {float(host)!r} "
                                     f"oracle {float(rec)!r}")
            return
        scale = float(np.max(np.abs(rec_a))) if rec_a.size else 0.0
        if host_a.shape != rec_a.shape or np.any(
                np.abs(f8(host_a) - f8(rec_a)) > GLUE_REL * scale):
            err = rel_err(host_a, rec_a, scale) if host_a.shape == rec_a.shape \
                else "shape"
            self.failures.append(f"{what}: {name} differs (max error over "
                                 f"the record scale {err})")

    def _next(self, kind: str, point, inputs: dict):
        key = (point.i, point.k)
        if point.k not in self.levels:
            return None
        self.host_order.append((kind, key))
        queue = self.calls.get(key)
        if not queue:
            self.failures.append(f"{kind} at {key}: the host calls it, WRF "
                                 f"did not")
            return None
        rec_kind, before, after = queue.popleft()
        if rec_kind != kind:
            self.failures.append(f"at {key} the host calls {kind} where WRF "
                                 f"called {rec_kind}")
            return None
        for name, value in inputs.items():
            self._cmp(f"{kind} input at {key}", name, value, before[name])
        return after

    def finish(self) -> list[str]:
        for key, queue in self.calls.items():
            for kind, _, _ in queue:
                self.failures.append(f"{kind} at {key}: WRF calls it, the "
                                     f"host did not")
        if self.host_order != self.oracle_order:
            self.failures.append("the host's call order at the probed points "
                                 "differs from WRF's")
        for key in set(self.oracle_forcing) | set(self.host_forcing):
            recs, host = self.oracle_forcing[key], self.host_forcing[key]
            if [int(r["ikl"]) for r in recs] != [v["ikl"] for v in host]:
                self.failures.append(
                    f"forcing substeps at {key}: host "
                    f"{[v['ikl'] for v in host]} oracle "
                    f"{[int(r['ikl']) for r in recs]} (DIFFU)")
                continue
            for r, v in zip(recs, host):
                for name in ("tt", "qq", "del1in", "del2in", "div1", "div2",
                             "dtcond"):
                    self._cmp(f"forcing at {key} ikl {v['ikl']}", name,
                              v[name], r[name])
                t_scale = abs(float(r["tt"]))
                q_scale = abs(float(r["qq"]))
                for name, scale in (("del_t", t_scale), ("del_q", q_scale)):
                    if abs(v[name] - float(r[name])) > GLUE_REL * scale:
                        self.failures.append(
                            f"forcing at {key} ikl {v['ikl']}: {name} host "
                            f"{v[name]!r} oracle {float(r[name])!r}")
        return self.failures

    # -- trace ---------------------------------------------------------------
    def trace(self, event: str, point, values: dict) -> None:
        if event == "entry" and point.k in self.levels:
            rec = self.entry[(point.i, point.k)]
            for name in ("ff1", "ff5", "ff6", "fccn", "t_new", "t_old", "qv",
                         "qv_old", "pcgs", "rhocgs", "zcgs"):
                self._cmp(f"entry at {(point.i, point.k)}", name,
                          values[name], rec[name])
        elif event == "forcing" and point.k in self.levels:
            self.host_forcing[(point.i, point.k)].append(values)
        elif event == "cloud_base":
            self.kz_host[point.i] = values["kz_cloud_base"]
        elif event == "fall":
            rec = self.fall_before[point.i]
            for name, rname in (("ffx", "ffx"), ("vrx", "vrx"),
                                ("rhocgs", "rhocgs"), ("pcgs", "pcgs"),
                                ("zcgs", "zcgs")):
                self._cmp(f"sedimentation input, column {point.i}", name,
                          values[name], rec[rname])
        elif event == "diag":
            rec = self.diag[point.i]
            for name in ("qc", "qr", "qnc", "qnr", "qna", "qv", "th",
                         "rainnc", "rainncv", "vr1_z_level1_after_loop"):
                self._cmp(f"diagnostics, column {point.i}", name,
                          values[name], rec[name])

    # -- the protocol --------------------------------------------------------
    def velocities(self, *, vr1, p_cgs):
        return vr1[None, :] * np.sqrt(1.0e6 / p_cgs)[:, None]

    def initial_aerosol(self, *, derived, rho_cgs, is_land):
        raise AssertionError("no probe step is the first step")

    def nucleation(self, *, point, ff1in, fccn, tt, qq, rho_cgs, p_cgs,
                   del1in, del2in, w_stag, is_cloud_base, kz_cloud_base,
                   tables, derived, behavior):
        rec = self._next("nucl", point, {
            "ff1": ff1in, "fccn": fccn, "tt": tt, "qq": qq,
            "del1in": del1in, "del2in": del2in,
            "is_cloud_base": int(is_cloud_base),
            "kz_cloud_base": int(kz_cloud_base), "w_stag": w_stag,
            "rhocgs": rho_cgs, "pcgs": p_cgs})
        if rec is None:
            key = (point.i, point.k)
            return NucleationStep(self.end_f[key], self.end_fccn[key],
                                  self.end_t[key], self.end_q[key], del1in,
                                  del2in)
        return NucleationStep(f8(rec["ff1"]), f8(rec["fccn"]),
                              float(rec["tt"]), float(rec["qq"]),
                              float(rec["del1in"]), float(rec["del2in"]))

    def onecond1(self, *, point, ff1r, tt, qq, p_cgs, rho_cgs, vr1, del1in,
                 del2in, div1, div2, dtcond, tables):
        rec = self._next("cond1", point, {
            "ff1": ff1r, "tt": tt, "qq": qq, "del1in": del1in,
            "del2in": del2in, "div1": div1, "div2": div2, "dtcond": dtcond,
            "vr1": vr1, "pcgs": p_cgs, "rhocgs": rho_cgs})
        if rec is None:
            key = (point.i, point.k)
            return CondensationStep(self.end_f[key], self.end_t[key],
                                    self.end_q[key], del1in, del2in)
        return CondensationStep(f8(rec["ff1"]), float(rec["tt"]),
                                float(rec["qq"]), float(rec["del1in"]),
                                float(rec["del2in"]))

    def coal(self, *, point, ff1r, tt, qq, del1in, del2in, p_cgs, rho_cgs,
             dt_coll, tables, derived, behavior):
        rec = self._next("coal", point, {
            "ff1": ff1r, "tt": tt, "qq": qq, "del1in": del1in,
            "del2in": del2in, "pp": p_cgs, "rhocgs": rho_cgs,
            "dt_coll": dt_coll})
        if rec is None:
            key = (point.i, point.k)
            return CoalStep(self.end_f[key], self.end_t[key],
                            self.end_q[key], del1in, del2in)
        return CoalStep(f8(rec["ff1"]), float(rec["tt"]), float(rec["qq"]),
                        float(rec["del1in"]), float(rec["del2in"]))

    def spont(self, *, point, ff1r, dt, tables, derived):
        rec = self._next("spont", point, {
            "ff1": ff1r,
            "ikr_spon_break": int(derived.ikr_spon_break)})
        if rec is None:
            return self.end_f[(point.i, point.k)]
        return f8(rec["ff1"])

    def falflux(self, *, point, ffx, v, rho_cgs, zcgs, dt):
        return FallStep(ffx=f8(self.fall_after[point.i]["ffx"]), outflow=None)

    def rainnc_level1(self, *, point, ff_bottom, v_bottom, tables, dt,
                      start):
        self.v_bottom[point.i] = v_bottom
        total = start
        for kr in range(NKR):
            total = total + 10.0 * (3.0 / tables.ro1bl[kr]) * COL * dt \
                * v_bottom[kr] * ff_bottom[kr] * tables.xl[kr] * tables.xl[kr]
        return total

    def rainnc_outflow(self, *, point, outflow, tables, start):
        raise AssertionError("the stand-ins report no outflow")


PROBE_STEPS = (10, 40, 80, 130, 200, 300)


def run_glue(step: int, *, behavior=WRF, tables_used=None,
             cloud_base_rule=None, record_increments=False):
    pre, post = replay("kc1_replay_o64")[step]
    tile = tile_from_pre(pre)
    stand = ProbeStandIns(step, tile, post)
    result = fast_sbm_warm_call(
        tile, step, DT, tables_used or tables(), oracle_derived(),
        behavior=behavior, processes=stand, trace=stand.trace,
        cloud_base_rule=cloud_base_rule,
        record_increments=record_increments)
    failures = stand.finish()
    report = compare_post(result, post, rel=0.0, floor=GLUE_REL)
    for name, (ok, err) in report.items():
        if not ok:
            failures.append(f"post-call {name} differs from the replay record "
                            f"(max error over the record scale {err:.3g})")
    return result, stand, failures


@pytest.mark.parametrize("step", PROBE_STEPS)
def test_glue_gate_every_call_gets_wrfs_inputs_in_wrfs_order(step):
    """Gate 1: processes replaced by WRF's own recorded calls."""
    result, stand, failures = run_glue(step)
    assert not failures, (f"step {step}: {len(failures)} mismatches, first: "
                          + "; ".join(failures[:8]))
    # The probed points really exercised the glue: calls happened.
    assert stand.host_order, f"step {step}: no probed call reached the host"
    # D9: WRF's cloud-base test leaves KZ at the top level in every column
    # at the probe steps (the probe records show it).
    assert set(stand.kz_host.values()) == {40}


def test_glue_gate_covers_every_branch_the_probes_saw():
    """The six probe steps together reach nucleation (both flags), ONECOND1
    in condensation and evaporation, collisional breakup and spontaneous
    breakup through the glue."""
    kinds = set()
    cloud_base_calls = 0
    evaporation = condensation = 0
    for r in probe_records():
        if r.type == "nucl_before":
            kinds.add("nucl")
            cloud_base_calls += int(r["is_cloud_base"])
        elif r.type == "cond1_before":
            kinds.add("cond1")
            if float(r["del1in"]) < 0.0:
                evaporation += 1
            else:
                condensation += 1
        elif r.type == "breakup" and int(r["path"]) > 0:
            kinds.add("breakup")
        elif r.type == "spont_before":
            kinds.add("spont")
    assert kinds == {"nucl", "cond1", "breakup", "spont"}
    assert cloud_base_calls >= 1 and evaporation >= 1 and condensation >= 1


def _first_failure_mentions(failures, *words) -> bool:
    return any(all(w in f for w in words) for f in failures)


def test_glue_gate_fails_when_the_fall_speeds_are_scaled():
    """Negative control: VR1 scaled by 1.001 reaches ONECOND1, the
    sedimentation input and the surface precipitation, and the gate sees
    it."""
    t = tables()
    scaled = dataclasses.replace(t, vr1=np.asarray(t.vr1) * 1.001)
    _, _, failures = run_glue(40, tables_used=scaled)
    assert _first_failure_mentions(failures, "cond1 input", "vr1")
    assert _first_failure_mentions(failures, "sedimentation input", "vrx")
    assert _first_failure_mentions(failures, "vr1_z_level1_after_loop")


def test_glue_gate_fails_on_the_d1_corrected_surface_precipitation():
    """Negative control: each column's own fall-speed factor (D1 corrected)
    is not what WRF did, in every column but the tile's last."""
    d1 = dataclasses.replace(WRF, rainnc_velocity="own_column")
    _, _, failures = run_glue(130, behavior=d1)
    bad = {f.split("column ")[1].split(":")[0] for f in failures
           if f.startswith("diagnostics, column") and "rainnc" in f}
    assert bad == {"1", "2", "3"}


def test_glue_gate_fails_on_the_intended_cloud_base_test():
    """Negative control: the cloud-base test as its comment reads (D9
    corrected) is not the test WRF runs."""
    _, _, failures = run_glue(40, cloud_base_rule="first_level")
    assert _first_failure_mentions(failures, "nucl input", "kz_cloud_base")


def test_the_d1_surface_precipitation_ratio_is_the_pressure_factor():
    """D1 through the glue: switching only ``rainnc_velocity`` scales each
    column's RAINNCV by sqrt(p_last / p_own) at the lowest level, to 1e-12."""
    d1 = dataclasses.replace(WRF, rainnc_velocity="own_column")
    for step in (40, 130, 300):
        wrf, _, _ = run_glue(step)
        own, _, _ = run_glue(step, behavior=d1)
        p1 = f8(statics("o64")["p"])[:, 0] * 10.0
        factor = np.sqrt(1.0e6 / p1)
        expected = factor[-1] / factor
        rain = own.rainncv > 0.0
        assert np.any(rain[:3])
        ratio = wrf.rainncv[rain] / own.rainncv[rain]
        np.testing.assert_allclose(ratio, expected[rain], rtol=1e-12, atol=0)
        assert np.array_equal(wrf.tile.ff1, own.tile.ff1)


def test_increments_account_for_every_change_of_the_call():
    """record_increments: at every point the entry spectrum plus every
    recorded point increment is the spectrum sedimentation received, and
    the column increments then give the post-call spectrum; T and qv close
    the same way from T_old and qv_old."""
    step = 130
    result, _, failures = run_glue(step, record_increments=True)
    assert not failures, failures[:5]
    pre, _ = replay("kc1_replay_o64")[step]
    tile = tile_from_pre(pre)
    t = tables()
    rhocgs = tile.rho * 0.001
    entry_f = f8(tile.ff1) * rhocgs[:, :, None] / COL / t.xl / t.xl / 3.0
    t_old = tile.th_old * tile.pii
    t_new = tile.th * tile.pii
    f = entry_f.copy()
    temp = {}
    vap = {}
    seen = set()
    for inc in result.increments:
        c = inc.point.i - 1
        if inc.point.k is None:
            f[c] = f[c] + inc.d_ff1
            continue
        k = inc.point.k - 1
        if (c, k) not in seen:
            seen.add((c, k))
            temp[(c, k)] = t_old[c, k]
            vap[(c, k)] = tile.qv_old[c, k]
        if inc.d_ff1 is not None:
            f[c, k] = f[c, k] + inc.d_ff1
        temp[(c, k)] += inc.d_t
        vap[(c, k)] += inc.d_qv
    final_f = f8(result.tile.ff1) * rhocgs[:, :, None] / COL / t.xl / t.xl / 3.0
    scale = float(np.max(np.abs(final_f)))
    assert np.max(np.abs(f - final_f)) <= 1e-12 * scale
    for (c, k), value in temp.items():
        assert abs(value - result.tile.th[c, k] * tile.pii[c, k]) <= 1e-9
        assert abs(vap[(c, k)] - result.tile.qv[c, k]) <= 1e-12 * abs(
            result.tile.qv[c, k])
    # Points no process touched keep T_new and qv.
    for c in range(tile.ncol):
        for k in range(tile.nz):
            if (c, k) not in seen:
                assert result.tile.th[c, k] * tile.pii[c, k] == pytest.approx(
                    t_new[c, k], rel=1e-15)
    kinds = {inc.process for inc in result.increments}
    assert {"forcing", "onecond1", "coal_bott", "spont_breakup",
            "sedimentation"} <= kinds


class IdentityProcesses:
    """Processes that change nothing and record what they are handed."""

    def __init__(self):
        self.calls = []

    def velocities(self, *, vr1, p_cgs):
        return vr1[None, :] * np.sqrt(1.0e6 / p_cgs)[:, None]

    def initial_aerosol(self, *, derived, rho_cgs, is_land):
        self.calls.append(("aerosol", is_land))
        src = f8(derived.fccnr_con if is_land else derived.fccnr_mar)
        return (src[None, :] / rho_cgs[0]) * rho_cgs[:, None]

    def nucleation(self, *, point, ff1in, fccn, tt, qq, del1in, del2in, **_):
        self.calls.append(("nucl", point))
        return NucleationStep(ff1in, fccn, tt, qq, del1in, del2in)

    def onecond1(self, *, point, ff1r, tt, qq, del1in, del2in, **_):
        self.calls.append(("cond1", point))
        return CondensationStep(ff1r, tt, qq, del1in, del2in)

    def coal(self, *, point, ff1r, tt, qq, del1in, del2in, **_):
        self.calls.append(("coal", point))
        return CoalStep(ff1r, tt, qq, del1in, del2in)

    def spont(self, *, point, ff1r, **_):
        return ff1r

    def falflux(self, *, point, ffx, **_):
        return FallStep(ffx, np.zeros(NKR))

    def rainnc_level1(self, *, start, **_):
        return start

    def rainnc_outflow(self, *, start, **_):
        return start


def _forcing_dtconds(behavior, dt):
    pre, _ = replay("kc1_replay_o64")[40]
    seen = []
    fast_sbm_warm_call(tile_from_pre(pre), 40, dt, tables(), oracle_derived(),
                       behavior=behavior, processes=IdentityProcesses(),
                       trace=lambda e, p, v: seen.append((p, v["dtcond"]))
                       if e == "forcing" else None)
    return seen


def test_d5_dtcond_carries_across_points_only_under_wrf():
    """D5: ``DTCOND`` is set once per call and shortened by rounding in the
    last substep of every gated point, forced or not (the ``min`` at :4620
    runs whatever ``DIFFU`` is); WRF carries the shortened value to every
    later point, the corrected preset starts each point at DT/NCOND.  At
    dt = 12 s the division is exact and the two agree; at dt = 10 s the
    first substep of every forced point differs by one ULP."""
    for behavior in (WRF, CORRECTED):
        assert {d for _, d in _forcing_dtconds(behavior, DT)} == {4.0}
    third = 10.0 / 3.0
    carried = _forcing_dtconds(WRF, 10.0)
    reset = _forcing_dtconds(CORRECTED, 10.0)
    assert [p for p, _ in carried] == [p for p, _ in reset]
    assert len(carried) > 6
    assert {d for p, d in carried if p.substep == 1} == {
        np.nextafter(third, 0.0)}
    assert {d for p, d in reset if p.substep == 1} == {third}


def test_first_step_initializes_aerosol_and_the_old_state():
    """Step 1: ``th_old``/``qv_old`` come from ``th``/``qv`` before
    ``T_old`` is formed (so whatever the tile held is ignored, :4351-4360),
    the aerosol is the lognormal start of the column's surface type
    (:4281-4296), and with ``T_old = T_new`` no substep forces anything
    (``DIFFU = 0``)."""
    tile = kc1_initial_tile()
    tile.th_old[:] = 0.0
    tile.qv_old[:] = -1.0
    kc1_set_w(tile, 1)
    procs = IdentityProcesses()
    forcing = []
    res = fast_sbm_warm_call(tile, 1, DT, tables(), oracle_derived(),
                             behavior=WRF, processes=procs,
                             trace=lambda e, p, v: forcing.append(p)
                             if e == "forcing" else None)
    assert [c for c in procs.calls if c[0] == "aerosol"] == [
        ("aerosol", True), ("aerosol", False), ("aerosol", False),
        ("aerosol", True)]
    assert not forcing
    assert not [c for c in procs.calls if c[0] in ("nucl", "cond1")]
    assert [c for c in procs.calls if c[0] == "coal"]
    d = oracle_derived()
    rhocgs = tile.rho * 0.001
    for c, land in enumerate((True, False, False, True)):
        src = f8(d.fccnr_con if land else d.fccnr_mar)
        fccn = (src[None, :] / rhocgs[c, 0]) * rhocgs[c][:, None]
        np.testing.assert_allclose(res.tile.ff8[c],
                                   fccn / rhocgs[c][:, None] * 1000.0,
                                   rtol=1e-15)
    assert np.array_equal(res.tile.th_old, res.tile.th)
    assert np.array_equal(res.tile.qv_old, res.tile.qv)


# ---------------------------------------------------------------------------
# D9, the cloud-base rule
# ---------------------------------------------------------------------------

def _synthetic_column(nz=10):
    t = np.full(nz, 290.0)
    p = np.full(nz, 9.0e5)
    es = 2.53e12 * np.exp(-5.42e3 / t)
    sat_q = 0.622 * es / (p - 0.378 * es)
    q = sat_q * 0.9
    q[3:6] = sat_q[3:6] * 1.02          # supersaturated at levels 4-6
    w = np.full(nz + 1, 0.5)            # 50 cm/s everywhere
    z = (np.arange(nz) + 0.5) * 2.0e4   # 200 m layers, all below 3 km
    return t, q, p, w, z


def test_d9_wrf_cloud_base_is_the_highest_level_not_supersaturated_with_updraft():
    t, q, p, w, z = _synthetic_column()
    assert cloud_base_level_wrf(t, q, p, w, z, rule="wrf_v480") == 10
    q2 = q.copy()
    q2[6:] = q2[3]                      # supersaturate the top as well
    assert cloud_base_level_wrf(t, q2, p, w, z, rule="wrf_v480") == 3
    assert cloud_base_level_wrf(t, q, p, w, z, rule="first_level") == 4
    w0 = np.zeros_like(w)
    assert cloud_base_level_wrf(t, q, p, w0, z, rule="wrf_v480") == 10
    assert cloud_base_level_wrf(t, q, p, w0, z, rule="first_level") == 0


def test_d9_matches_every_kz_the_oracle_recorded():
    """The column's own cloud-base transcription gives WRF's recorded
    ``KZ_Cloud_Base`` at every nucleation record of the KC-1 probes."""
    runs = replay("kc1_replay_o64")
    s = statics("o64")
    dz = f8(s["dz8w"])
    zc = np.cumsum(dz * 100.0, axis=1) - 0.5 * dz * 100.0
    checked = 0
    for r in probe_records():
        if r.type != "nucl_before":
            continue
        step, i = int(r["step"]), int(r["i"])
        pre, _ = runs[step]
        kz = cloud_base_level_wrf(f8(pre["th"])[i - 1] * f8(s["pii"])[i - 1],
                                  f8(pre["qv"])[i - 1],
                                  f8(s["p"])[i - 1] * 10.0,
                                  f8(pre["w"])[i - 1], zc[i - 1])
        assert kz == int(r["kz_cloud_base"]), (step, i)
        checked += 1
    assert checked > 0


def _cloud_bases(behavior, step, **kw):
    pre, _ = replay("kc1_replay_o64")[step]
    seen = {}
    fast_sbm_warm_call(tile_from_pre(pre), step, DT, tables(),
                       oracle_derived(), behavior=behavior,
                       processes=IdentityProcesses(),
                       trace=lambda e, p, v: seen.__setitem__(
                           p.i, v["kz_cloud_base"])
                       if e == "cloud_base" else None, **kw)
    return [seen[i] for i in sorted(seen)]


def test_d9_the_behaviour_selects_the_cloud_base_rule():
    """D9 settled at the wave 2 merge: the call takes its cloud-base rule
    from ``behavior.cloud_base_rule``.  On the KC-1 replay states WRF's rule
    gives level 40 in every column; the corrected preset gives the first
    supersaturated updraft level (7 in the two updraft columns at step 40,
    10 in the weak updraft at step 130, none elsewhere), which is what the
    nucleation item's ``cloud_base_level`` gives for the same behaviour.
    The call's ``cloud_base_rule`` argument overrides the behaviour."""

    assert _cloud_bases(WRF, 40) == [40, 40, 40, 40]
    assert _cloud_bases(CORRECTED, 40) == [7, 7, 0, 0]
    assert _cloud_bases(CORRECTED, 130) == [0, 0, 0, 10]
    arm = dataclasses.replace(WRF, rainnc_velocity="own_column")
    assert _cloud_bases(arm, 40) == [40, 40, 40, 40]
    assert _cloud_bases(CORRECTED, 40, cloud_base_rule="wrf_v480") == [
        40, 40, 40, 40]
    assert _cloud_bases(WRF, 40, cloud_base_rule="first_level") == [
        7, 7, 0, 0]
    # the column and the nucleation item evaluate one transcription
    from gpuwm.core.fast_sbm import nucleation as N

    pre, _ = replay("kc1_replay_o64")[40]
    tile = tile_from_pre(pre)
    zc = np.empty_like(tile.dz)
    z_full = np.zeros(tile.ncol)
    for k in range(tile.nz):          # as the call forms zcgs (:4270)
        zc[:, k] = z_full + 0.5 * tile.dz[:, k] * 100.0
        z_full = z_full + tile.dz[:, k] * 100.0
    for behavior in (WRF, CORRECTED):
        for c in range(tile.ncol):
            kz = N.cloud_base_level(tile.th[c] * tile.pii[c], tile.qv[c],
                                    tile.p[c] * 10.0, tile.w[c], zc[c],
                                    behavior=behavior)
            assert (0 if kz is None else kz + 1) == _cloud_bases(
                behavior, 40)[c]


# ---------------------------------------------------------------------------
# Refusals (gate 6)
# ---------------------------------------------------------------------------

class _NoCalls:
    """Processes that must never be reached."""

    def __getattr__(self, name):
        def fail(**kwargs):
            raise AssertionError(f"process {name} was called")
        return fail


def _snapshot(tile: FsbmTile):
    return {name: np.array(getattr(tile, name), copy=True)
            for name in tile.__dataclass_fields__ if name != "ni"}


def _assert_unchanged(tile, snap):
    for name, value in snap.items():
        assert np.array_equal(getattr(tile, name), value, equal_nan=True), name


@pytest.mark.parametrize("field,where", [("th", "T_new"), ("th_old", "T_old")])
def test_a_cold_point_is_refused_before_anything_changes(field, where):
    pre, _ = replay("kc1_replay_o64")[40]
    tile = tile_from_pre(pre)
    getattr(tile, field)[2, 17] = 273.15 / tile.pii[2, 17]
    snap = _snapshot(tile)
    with pytest.raises(FsbmRefusal) as info:
        fast_sbm_warm_call(tile, 40, DT, tables(), oracle_derived(),
                           behavior=WRF, processes=_NoCalls())
    _assert_unchanged(tile, snap)
    text = str(info.value)
    assert where in text and "i=3" in text and "k=18" in text
    for process in COLD_PROCESSES:
        assert process in text


@pytest.mark.parametrize("species", ["ff5", "ff6"])
def test_an_ice_bin_is_refused_before_anything_changes(species):
    pre, _ = replay("kc1_replay_o64")[40]
    tile = tile_from_pre(pre)
    getattr(tile, species)[1, 5, 20] = 1.0e-9
    snap = _snapshot(tile)
    with pytest.raises(FsbmRefusal) as info:
        fast_sbm_warm_call(tile, 40, DT, tables(), oracle_derived(),
                           behavior=CORRECTED, processes=_NoCalls())
    _assert_unchanged(tile, snap)
    assert "bin 21" in str(info.value)
    assert "FREEZ" in str(info.value) and "J_W_MELT" in str(info.value)


@pytest.mark.parametrize("field,index,value", [
    ("th_old", (0, 12), float("nan")),
    ("w", (0, 12), float("nan")),
    ("th", (1, 3), float("nan")),
    ("qv", (2, 30), float("inf")),
    ("qv_old", (3, 0), float("nan")),
    ("xland", (1,), float("nan")),
    ("rainnc", (2,), float("nan")),
    ("ff1", (0, 12, 20), float("nan")),
    ("ff8", (3, 5, 10), float("-inf")),
])
def test_a_non_finite_value_is_refused_before_anything_changes(field, index,
                                                                value):
    """Breakage prevented: a NaN turned into nothing.  A NaN ``th_old`` at
    c1 level 13 of the step-40 replay state passed the cold refusal
    (``NaN <= 273.15`` is false), failed the ``T_old > 193.15`` gate, so the
    point ran no nucleation, condensation or collision, lost its latent
    heating (qc 5.472e-4 against 5.123e-4, theta 305.7750 against 305.6855
    K), and the exit wrote a finite ``th_old`` over the NaN; a NaN ``w`` was
    swallowed the same way by the cloud-base test.  Every tile field is
    checked, before any process runs."""
    pre, _ = replay("kc1_replay_o64")[40]
    tile = tile_from_pre(pre)
    getattr(tile, field)[index] = value
    snap = _snapshot(tile)
    with pytest.raises(FsbmRefusal) as info:
        fast_sbm_warm_call(tile, 40, DT, tables(), real_derived(CORRECTED),
                           behavior=CORRECTED, processes=_NoCalls())
    _assert_unchanged(tile, snap)
    text = str(info.value)
    assert f"tile.{field} is {value!r}" in text
    assert f"i={index[0] + 1}," in text
    if len(index) > 1:
        assert f"level k={index[1] + 1}" in text
    assert "without a refusal" in text


def test_the_first_step_ignores_what_th_old_held_even_a_nan():
    """On step 1 WRF sets ``th_old`` and ``qv_old`` from ``th`` and ``qv``
    before reading them (:4351-4360), so a NaN there is not an input and is
    not refused."""
    tile = kc1_initial_tile()
    tile.th_old[:] = float("nan")
    tile.qv_old[:] = float("nan")
    kc1_set_w(tile, 1)
    res = fast_sbm_warm_call(tile, 1, DT, tables(), oracle_derived(),
                             behavior=WRF, processes=IdentityProcesses())
    assert np.array_equal(res.tile.th_old, res.tile.th)
    assert np.all(np.isfinite(res.tile.qv_old))


def test_a_substep_that_cools_to_freezing_is_refused_and_the_tile_is_untouched():
    """A process that drops T to 273.15 K mid-call: the refusal comes before
    the next process sees the cold temperature, and the caller's tile is
    unchanged because the call works on copies."""
    step = 40
    pre, post = replay("kc1_replay_o64")[step]
    tile = tile_from_pre(pre)
    stand = ProbeStandIns(step, tile, post)
    real_cond = stand.onecond1

    def freezing_cond(**kw):
        out = real_cond(**kw)
        return dataclasses.replace(out, tt=273.0)

    stand.onecond1 = freezing_cond
    snap = _snapshot(tile)
    with pytest.raises(FsbmRefusal, match="at or below 273.15"):
        fast_sbm_warm_call(tile, step, DT, tables(), oracle_derived(),
                           behavior=WRF, processes=stand)
    _assert_unchanged(tile, snap)


def test_the_call_never_writes_to_its_input_tile():
    step = 130
    pre, post = replay("kc1_replay_o64")[step]
    tile = tile_from_pre(pre)
    snap = _snapshot(tile)
    fast_sbm_warm_call(tile, step, DT, tables(), oracle_derived(),
                       behavior=WRF, processes=ProbeStandIns(step, tile, post))
    _assert_unchanged(tile, snap)


def test_float32_tables_and_bad_arguments_are_refused():
    pre, post = replay("kc1_replay_o64")[40]
    tile = tile_from_pre(pre)
    with pytest.raises(FsbmRefusal, match="float64"):
        fast_sbm_warm_call(tile, 40, DT, load_tables("float32"),
                           oracle_derived(), processes=_NoCalls())
    with pytest.raises(FsbmRefusal, match="itimestep"):
        fast_sbm_warm_call(tile, 0, DT, tables(), oracle_derived(),
                           processes=_NoCalls())
    with pytest.raises(FsbmRefusal, match="cloud_base_rule"):
        fast_sbm_warm_call(tile, 40, DT, tables(), oracle_derived(),
                           processes=_NoCalls(), cloud_base_rule="lowest")
    bad = tile.copy()
    bad.w = bad.w[:, :-1]
    with pytest.raises(FsbmRefusal, match="tile.w"):
        fast_sbm_warm_call(bad, 40, DT, tables(), oracle_derived(),
                           processes=_NoCalls())


# ---------------------------------------------------------------------------
# The KC-1 driver, ported from the oracle's run_column.F90 (gate 4)
# ---------------------------------------------------------------------------

def advect(phi: np.ndarray, w: np.ndarray, rho: np.ndarray, dz: np.ndarray,
           dt: float) -> np.ndarray:
    """``advect_column`` of run_column.F90: first-order upwind in flux form
    with the divergence term restored, fixed density, zero flux at the
    bottom and top, evaluated in the Fortran's order.  ``phi`` is
    ``(nz,)`` or ``(nz, m)``; each trailing column is independent."""
    nz = rho.shape[0]
    old = np.array(phi, dtype=np.float64, copy=True)
    tail = old.shape[1:]
    g = np.zeros(nz + 1)
    up = np.zeros((nz + 1,) + tail)
    for k in range(1, nz):
        g[k] = 0.5 * (rho[k - 1] + rho[k]) * w[k]
        up[k] = old[k - 1] if g[k] >= 0.0 else old[k]
    out = np.empty_like(old)
    for k in range(1, nz + 1):
        ten = -((g[k] * (up[k] - old[k - 1]))
                - (g[k - 1] * (up[k - 1] - old[k - 1]))) \
            / (rho[k - 1] * dz[k - 1])
        out[k - 1] = old[k - 1] + dt * ten
    return out


def kc1_initial_tile() -> FsbmTile:
    """``init_state`` of run_column.F90 from the case record."""
    cs = case()
    ncol, nz = f8(cs["theta"]).shape
    ff1 = np.zeros((ncol, nz, NKR))
    ff1[:, :, 14:33] = f8(cs["rain_bins_mass_mixing_ratio"])[:, :, 14:33]
    th = f8(cs["theta"])
    qv = f8(cs["qv"])
    return FsbmTile(
        th=th, qv=qv, th_old=th.copy(), qv_old=qv.copy(),
        p=f8(cs["pressure"]), pii=f8(cs["exner"]), rho=f8(cs["rho"]),
        dz=np.broadcast_to(f8(cs["dz"]), (ncol, nz)).copy(),
        w=np.zeros((ncol, nz + 1)), ff1=ff1, ff5=np.zeros((ncol, nz, NKR)),
        ff6=np.zeros((ncol, nz, NKR)), ff8=np.zeros((ncol, nz, 43)),
        xland=f8(cs["xland"]), rainnc=np.zeros(ncol))


def kc1_set_w(tile: FsbmTile, n: int) -> None:
    cs = case()
    tsec = float(n - 1) * DT
    prof = f8(cs["w_profile_full_levels"])
    on = f8(cs["w_on_seconds"])
    for c in range(tile.ncol):
        tile.w[c] = prof[c] if tsec < on[c] else 0.0


def kc1_advect_all(tile: FsbmTile) -> None:
    for c in range(tile.ncol):
        chem = np.concatenate([tile.ff1[c], tile.ff5[c], tile.ff6[c],
                               tile.ff8[c]], axis=1)
        block = np.concatenate([tile.th[c][:, None], tile.qv[c][:, None],
                                chem], axis=1)
        new = advect(block, tile.w[c], tile.rho[c], tile.dz[c], DT)
        tile.th[c], tile.qv[c] = new[:, 0], new[:, 1]
        chem = new[:, 2:]
        tile.ff1[c] = chem[:, 0:33]
        tile.ff5[c] = chem[:, 33:66]
        tile.ff6[c] = chem[:, 66:99]
        tile.ff8[c] = chem[:, 99:142]


def test_the_ported_driver_starts_where_the_fortran_started():
    """The ported initial state is the O64 summary's initial record, byte
    for byte, and the ported advection reproduces the case's own operator
    check (c1 theta, dry, over the 900 s updraft) exactly."""
    tile = kc1_initial_tile()
    s = statics("o64")
    for name, rec in (("th", "th"), ("qv", "qv"), ("p", "p"), ("pii", "pii"),
                      ("rho", "rho"), ("dz", "dz8w")):
        assert np.array_equal(getattr(tile, name), f8(s[rec])), name
    assert np.array_equal(tile.xland, f8(s["xland"]))
    assert np.array_equal(tile.chem(), f8(s["chem"]))
    check = next(fx.iter_records("kc1_case",
                                 types=["literal_advection_check"]))
    cs = case()
    th = f8(check["th_initial"])
    w = f8(cs["w_profile_full_levels"])[0]
    for _ in range(int(check["steps"])):
        th = advect(th, w, f8(cs["rho"])[0], f8(cs["dz"]), DT)
    assert np.array_equal(th, f8(check["th_case_operator"]))


def test_the_ported_driver_reproduces_a_replay_pre_state_from_the_post_before():
    """Advecting the O64 post-call state of step 2 with step 3's velocity
    gives the recorded pre-call state of step 3 exactly: the ported driver
    is the harness's driver."""
    runs = replay("kc1_replay_o64")
    _, post2 = runs[2]
    pre3, _ = runs[3]
    tile = tile_from_pre(pre3)
    rec = record_fields(post2)
    tile.th, tile.qv = rec["th"].copy(), rec["qv"].copy()
    for name in SPECIES:
        setattr(tile, name, rec[name].copy())
    kc1_set_w(tile, 3)
    assert np.array_equal(tile.w, f8(pre3["w"]))
    kc1_advect_all(tile)
    assert np.array_equal(tile.th, f8(pre3["th"]))
    assert np.array_equal(tile.qv, f8(pre3["qv"]))
    assert np.array_equal(tile.chem(), f8(pre3["chem"]))


# ---------------------------------------------------------------------------
# End-to-end gates with the real processes (2, 3, 4, 5, 7)
# ---------------------------------------------------------------------------

REPLAY_REL = 1e-9
REPLAY_FLOOR = 1e-12
REPLAY_FIELDS = ("ff1", "ff5", "ff6", "ff8", "th", "qv", "th_old", "qv_old",
                 "rainncv")


def replay_report(step, processes, derived, *, tables_used=None,
                  behavior=WRF):
    pre, post = replay("kc1_replay_o64")[step]
    result = fast_sbm_warm_call(tile_from_pre(pre), step, DT,
                                tables_used or tables(), derived,
                                behavior=behavior, processes=processes)
    report = compare_post(result, post, rel=REPLAY_REL, floor=REPLAY_FLOOR)
    return {name: report[name] for name in REPLAY_FIELDS}


def test_replay_all_16_calls_against_the_promoted_build():
    """Gate 2: every recorded call, the whole tile, wrf_v480."""
    processes = real_processes()
    derived = real_derived()
    bad = []
    for step in sorted(replay("kc1_replay_o64")):
        for name, (ok, err) in replay_report(step, processes,
                                             derived).items():
            if not ok:
                bad.append(f"step {step} {name}: max error {err:.3g} of the "
                           f"record scale")
    assert not bad, bad[:10]


def test_native_envelope_on_the_native_builds_states():
    """Gate 3: on the O32 pre-call states the host is no farther from O32
    than the promoted build run on the same states (plus 1e-9 of scale)."""
    processes = real_processes()
    derived = real_derived()
    native = replay("kc1_replay_o32")
    cross = {step: post for step, _, post in
             fx.iter_replay("kc1_crossreplay_o64_on_o32")}
    bad = []
    for step, (pre, post) in sorted(native.items()):
        result = fast_sbm_warm_call(tile_from_pre(pre, "o32"), step, DT,
                                    tables(), derived, behavior=WRF,
                                    processes=processes)
        host = post_fields(result.tile, result.rainncv)
        o32 = record_fields(post)
        o64x = record_fields(cross[step])
        for name in REPLAY_FIELDS:
            scale = float(np.max(np.abs(o32[name])))
            limit = np.abs(o64x[name] - o32[name]) + 1e-9 * scale
            excess = np.abs(host[name] - o32[name]) - limit
            if np.any(excess > 0.0):
                bad.append(f"step {step} {name}: outside the envelope by "
                           f"{float(np.max(excess)):.3g}")
    assert not bad, bad[:10]


def test_first_call_matches_the_o64_summary():
    """Step 1 initializes the aerosol and th_old/qv_old; the first
    trajectory record must follow at the replay tolerance."""
    processes = real_processes()
    derived = real_derived()
    tile = kc1_initial_tile()
    kc1_set_w(tile, 1)
    result = fast_sbm_warm_call(tile, 1, DT, tables(), derived, behavior=WRF,
                                processes=processes)
    rec = summary("o64")[1]
    host = {"th": result.tile.th, "qv": result.tile.qv,
            "qc": result.bulk["qc"], "qr": result.bulk["qr"],
            "qnc": result.bulk["qnc"], "rainnc": result.tile.rainnc}
    for name, value in host.items():
        r = f8(rec[name])
        assert within(value, r, rel=REPLAY_REL, floor=REPLAY_FLOOR,
                      scale=float(np.max(np.abs(r)))), name


TRAJ_FIELDS = ("th", "qv", "qc", "qr", "qnc", "rainnc")


def run_host_trajectory(processes, derived, *, behavior=WRF, nsteps=300,
                        keep_steps=()):
    """The KC-1 run with the host in place of FAST_SBM: per step the
    summary fields, and the full post-call tile at ``keep_steps``."""
    tile = kc1_initial_tile()
    out = {}
    kept = {}
    for n in range(1, nsteps + 1):
        kc1_set_w(tile, n)
        if n > 1:
            kc1_advect_all(tile)
        res = fast_sbm_warm_call(tile, n, DT, tables(), derived,
                                 behavior=behavior, processes=processes)
        tile = res.tile
        out[n] = {"th": tile.th.copy(), "qv": tile.qv.copy(),
                  "qc": res.bulk["qc"].copy(), "qr": res.bulk["qr"].copy(),
                  "qnc": res.bulk["qnc"].copy(),
                  "rainnc": tile.rainnc.copy(),
                  "rainncv": res.rainncv.copy()}
        if n in keep_steps:
            kept[n] = tile.copy()
    return out, kept


def trajectory_ratios(host) -> dict[str, np.ndarray]:
    """Per field and step: the host's distance from O64 over ten times the
    running maximum of the one-ULP self-divergence (plus 1e-12 of the
    field's run scale).  A ratio above 1 fails gate 4."""
    o64, sens = summary("o64"), sensitivity()
    steps = sorted(host)
    ratios = {}
    for name in TRAJ_FIELDS:
        scale = max(float(np.max(np.abs(f8(o64[n][name])))) for n in steps)
        env = 0.0
        vals = []
        for n in steps:
            ref = f8(o64[n][name])
            env = max(env, float(np.max(np.abs(f8(sens[n][name]) - ref))))
            err = float(np.max(np.abs(host[n][name] - ref)))
            vals.append(err / (10.0 * env + 1e-12 * scale))
        ratios[name] = np.array(vals)
    return ratios


@pytest.mark.slow
def test_trajectory_within_ten_times_the_fortrans_self_divergence():
    """Gate 4: the host drives all 300 KC-1 steps."""
    processes = real_processes()
    derived = real_derived()
    host, _ = run_host_trajectory(processes, derived)
    ratios = trajectory_ratios(host)
    worst = {name: float(np.max(r)) for name, r in ratios.items()}
    assert all(v <= 1.0 for v in worst.values()), worst


# -- budgets (gate 5) -------------------------------------------------------

def level_weights(tile: FsbmTile) -> np.ndarray:
    """rho (g/cm3) times the level spacing FALFLUXHUCM_Z conserves (cm):
    the distance to the next level centre, and for the top level the
    distance to the one below (:5364-5373).  On a stretched grid this is
    not the layer depth (D8)."""
    dz = tile.dz * 100.0
    zc = np.cumsum(dz, axis=1) - 0.5 * dz
    spacing = np.empty_like(zc)
    spacing[:, :-1] = zc[:, 1:] - zc[:, :-1]
    spacing[:, -1] = zc[:, -1] - zc[:, -2]
    return tile.rho * 0.001 * spacing


def column_water(tile: FsbmTile, weights) -> np.ndarray:
    liquid = np.sum(tile.ff1, axis=2)
    return np.sum(weights * (tile.qv + liquid), axis=1)


def column_lse(tile: FsbmTile, weights) -> np.ndarray:
    """Liquid static energy over cp at fixed height: T - (L/cp) q_l, with
    WRF's AL1 = 2500 K per unit mixing ratio (ONECOND1 :6830)."""
    liquid = np.sum(tile.ff1, axis=2)
    return np.sum(weights * (tile.th * tile.pii - AL1 * liquid), axis=1)


def process_liquid(result, tile, kinds) -> np.ndarray:
    """Column liquid change (g/cm2) of the given processes' increments."""
    t = tables()
    w = level_weights(tile)
    rhocgs = tile.rho * 0.001
    mass = 3.0 * COL * f8(t.xl) ** 2
    out = np.zeros(tile.ncol)
    for inc in result.increments:
        if inc.process not in kinds or inc.d_ff1 is None:
            continue
        c = inc.point.i - 1
        if inc.point.k is None:
            out[c] += np.sum(w[c] * (inc.d_ff1 @ mass) / rhocgs[c])
        else:
            k = inc.point.k - 1
            out[c] += w[c, k] * float(inc.d_ff1 @ mass) / rhocgs[c, k]
    return out


def budget_terms(step, processes, derived, behavior):
    pre, _ = replay("kc1_replay_o64")[step]
    tile = tile_from_pre(pre)
    res = fast_sbm_warm_call(tile, step, DT, tables(), derived,
                             behavior=behavior, processes=processes,
                             record_increments=True)
    w = level_weights(tile)
    before_w, after_w = column_water(tile, w), column_water(res.tile, w)
    before_e, after_e = column_lse(tile, w), column_lse(res.tile, w)
    return {
        "tile": tile, "result": res, "before_w": before_w,
        "after_w": after_w, "before_e": before_e, "after_e": after_e,
        "rainnc": res.rainncv * 0.1,                  # mm to g/cm2 (RO1BL 1)
        "outflow": -process_liquid(res, tile, {"sedimentation"}),
        "clipped": np.sum(w * res.clipped, axis=1),
        "nucleated": np.sum(w * (res.nucleated + res.nucleation_vapour),
                            axis=1),
        "collision": process_liquid(res, tile, {"coal_bott"}),
        "spont": process_liquid(res, tile, {"spont_breakup"}),
        "zeroing": process_liquid(res, tile, {"negative_zeroing"}),
    }


def test_budgets_close_under_both_presets():
    """Gate 5.  Per call and column, on the level weights sedimentation
    conserves:

    * corrected: total water W and liquid static energy E close exactly
      once the named non-conserving terms are counted: surface
      precipitation (the bottom outflow, D2 corrected), the exit clip (D4)
      and the collision step's own mass change (a property of WRF's Bott
      remap and breakup, measured, not a glue term); activation takes its
      vapour (D3 corrected), so it is not a term.
    * wrf_v480: the residual (W_after + RAINNC) - W_before is the nucleated
      mass (D3) minus the clipped mass (D4) plus RAINNC minus the outflow
      (D2), plus the same collision term.
    """
    processes = real_processes()
    derived = real_derived(WRF)
    seen = {"d2": 0.0, "d3": 0.0, "d4": 0.0, "plain": 0.0}
    for step in sorted(replay("kc1_replay_o64")):
        cor = budget_terms(step, processes, real_derived(CORRECTED),
                           CORRECTED)
        scale = cor["before_w"]
        np.testing.assert_allclose(cor["rainnc"], cor["outflow"],
                                   rtol=1e-12, atol=1e-12 * float(
                                       np.max(scale)))
        # D3 corrected: each activation takes its vapour (to 1e-12 of the
        # point's water; the nucleated mass is a difference of two spectrum
        # totals, so its rounding is relative to the liquid, not to itself).
        res = cor["result"]
        assert np.all(np.abs(res.nucleation_vapour + res.nucleated)
                      <= 1e-12 * cor["tile"].qv), step
        water = (cor["before_w"] - cor["after_w"] - cor["rainnc"]
                 - cor["clipped"] + cor["collision"] + cor["spont"]
                 + cor["zeroing"])
        assert np.all(np.abs(water) <= 1e-12 * scale), (step, water / scale)
        lse = (cor["after_e"] - cor["before_e"]
               - AL1 * (cor["rainnc"] + cor["clipped"] - cor["collision"]
                        - cor["spont"] - cor["zeroing"]))
        assert np.all(np.abs(lse) <= 1e-12 * np.abs(cor["before_e"])), (
            step, lse / cor["before_e"])

        wrf = budget_terms(step, processes, derived, WRF)
        residual = wrf["after_w"] + wrf["rainnc"] - wrf["before_w"]
        expected = (wrf["nucleated"] - wrf["clipped"] + wrf["rainnc"]
                    - wrf["outflow"] + wrf["collision"] + wrf["spont"]
                    + wrf["zeroing"])
        assert np.all(np.abs(residual - expected)
                      <= 1e-10 * wrf["before_w"]), (step, residual, expected)
        w0 = wrf["before_w"]
        seen["d2"] = max(seen["d2"], float(np.max(
            np.abs(wrf["rainnc"] - wrf["outflow"]) / w0)))
        seen["d3"] = max(seen["d3"], float(np.max(wrf["nucleated"] / w0)))
        seen["d4"] = max(seen["d4"], float(np.max(wrf["clipped"] / w0)))
        seen["plain"] = max(seen["plain"], float(np.max(
            np.abs(residual - wrf["collision"]) / w0)))
        # D1 through the real processes: only the velocity factor changes.
        d1 = dataclasses.replace(WRF, rainnc_velocity="own_column")
        own = fast_sbm_warm_call(wrf["tile"], step, DT, tables(), derived,
                                 behavior=d1, processes=processes)
        p1 = wrf["tile"].p[:, 0] * 10.0
        factor = np.sqrt(1.0e6 / p1)
        rain = own.rainncv > 0.0
        np.testing.assert_allclose(wrf["result"].rainncv[rain]
                                   / own.rainncv[rain],
                                   (factor[-1] / factor)[rain], rtol=1e-12)
    # The wrf_v480 terms are real in KC-1, so the residual identity is not
    # trivially 0 = 0, and WRF's water does not close without them.
    assert seen["d2"] > 1e-6 and seen["d3"] > 1e-6 and seen["d4"] > 0.0
    assert seen["plain"] > 1e-10, seen


def test_rainnc_accumulates_this_calls_rain_under_both_presets():
    """Over two consecutive calls from the step-130 replay state, the
    accumulated ``RAINNC`` each call returns is the ``RAINNC`` it was given
    plus its own ``RAINNCV``, under the corrected preset (the bottom
    outflow, D2 corrected) and under ``wrf_v480``.  The two sums differ only
    in rounding, because both accumulate bin by bin (onto the running total
    and onto zero, :5065-5070).

    Breakage prevented: the corrected accumulation reset to this call's
    outflow on every call (``start=0.0`` passed every test of this file
    before): the budget gate reads only ``RAINNCV`` and the trajectory runs
    only ``wrf_v480``."""
    processes = real_processes()
    pre, _ = replay("kc1_replay_o64")[130]
    eps = np.finfo(np.float64).eps
    for behavior in (CORRECTED, WRF):
        tile = tile_from_pre(pre)
        derived = real_derived(behavior)
        raining = 0
        for step in (130, 131):
            before = tile.rainnc.copy()
            res = fast_sbm_warm_call(tile, step, DT, tables(), derived,
                                     behavior=behavior, processes=processes)
            expected = before + res.rainncv
            assert np.all(np.abs(res.tile.rainnc - expected)
                          <= 64.0 * eps * expected), (behavior, step)
            raining += int(np.sum((before > 0.0) & (res.rainncv > 0.0)))
            tile = res.tile
        # The check reads a real accumulation: columns that had rain and
        # rained again in both calls.
        assert raining >= 4, (behavior, raining)


# -- D10, the fall speeds inside the coalescence efficiencies -------------

def test_d10_a_derived_table_of_the_other_lookup_is_refused_before_anything_changes():
    """The derived tables carry the fall-speed lookup their ``ECOALMASSM``
    was built with, and the call refuses a table whose lookup is not its
    behaviour's, in either direction, before any process runs.  The default
    processes also refuse a table that does not say which lookup it holds.

    Breakage prevented: a corrected run on WRF's collision table, which is
    what the milestone 1 evidence published as the corrected preset (0.20
    and 0.10 percent less rain in the two updraft columns over the hour).
    """
    pre, _ = replay("kc1_replay_o64")[40]
    for behavior, other in ((CORRECTED, WRF), (WRF, CORRECTED)):
        tile = tile_from_pre(pre)
        snap = _snapshot(tile)
        with pytest.raises(FsbmRefusal, match="D10") as info:
            fast_sbm_warm_call(tile, 40, DT, tables(), real_derived(other),
                               behavior=behavior, processes=_NoCalls())
        _assert_unchanged(tile, snap)
        assert repr(behavior.fall_speed_lookup) in str(info.value)
    # The oracle's own dump says nothing about its lookup: the stand-ins
    # may use it, the default processes' collision step may not.
    with pytest.raises(FsbmRefusal, match="D10"):
        fast_sbm_warm_call(tile_from_pre(pre), 40, DT, tables(),
                           oracle_derived(), behavior=CORRECTED,
                           processes=real_processes())


def test_d10_the_corrected_default_collides_with_each_bins_own_fall_speed():
    """A bare call (default behaviour, default processes, ``build_derived``
    with no behaviour) runs the corrected preset on the ``own_bin`` table,
    and that table reaches the collision step: on the replay states with
    rain, the corrected preset differs from the same preset on WRF's
    lookup only through the drops that collide.

    Breakage prevented: D10's fix built beside the default and never bound
    to it (the corrected preset silently colliding on WRF's table)."""
    from gpuwm.core.fast_sbm.derived import build_derived

    step = 130
    pre, _ = replay("kc1_replay_o64")[step]
    bare = fast_sbm_warm_call(tile_from_pre(pre), step, DT, tables(),
                              build_derived(tables(), DT))
    fixed = fast_sbm_warm_call(tile_from_pre(pre), step, DT, tables(),
                               real_derived(CORRECTED), behavior=CORRECTED,
                               processes=real_processes())
    for name in ("ff1", "ff8", "th", "qv", "rainnc"):
        assert np.array_equal(getattr(bare.tile, name),
                              getattr(fixed.tile, name)), name
    arm = dataclasses.replace(CORRECTED, fall_speed_lookup="wrf_v480")
    wrf_speeds = fast_sbm_warm_call(tile_from_pre(pre), step, DT, tables(),
                                    real_derived(arm), behavior=arm,
                                    processes=real_processes())
    moved = np.abs(fixed.tile.ff1 - wrf_speeds.tile.ff1)
    assert float(np.max(moved)) > 1e-6 * float(np.max(fixed.tile.ff1))
    # Nothing but the collision table differs between the two runs, so the
    # thermodynamics of this call are untouched.
    assert np.array_equal(fixed.tile.th, wrf_speeds.tile.th)
    assert np.array_equal(fixed.tile.qv, wrf_speeds.tile.qv)


# -- negative controls (gate 7) -------------------------------------------

def _replay_gate_fails(processes, derived, *, tables_used=None) -> bool:
    for step in (40, 130):
        report = replay_report(step, processes, derived,
                               tables_used=tables_used)
        if not all(ok for ok, _ in report.values()):
            return True
    return False


def test_negative_controls_fail_the_replay_gate():
    """Gate 7: cwll scaled by 1.001 and VR1 scaled by 1.001 each fail gate
    2, so gate 2 can fail."""
    base = real_processes()
    derived = real_derived()
    assert not _replay_gate_fails(base, derived)

    class ScaledKernel(type(base)):
        def kernels(self, **kw):
            k = super().kernels(**kw)
            if dataclasses.is_dataclass(k):
                return dataclasses.replace(k, cwll=np.asarray(k.cwll) * 1.001)
            scaled = SimpleNamespace(**vars(k))
            scaled.cwll = np.asarray(k.cwll) * 1.001
            return scaled

    assert _replay_gate_fails(ScaledKernel(), derived)
    t = tables()
    vr1 = dataclasses.replace(t, vr1=np.asarray(t.vr1) * 1.001)
    assert _replay_gate_fails(base, derived, tables_used=vr1)
