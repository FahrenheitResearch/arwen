"""wind7.py (step 7): wind5.py with the never-worse rule.  Where a grid's local-face decision was replaced by
the shipped measured clock's (``gpuwm.terrain_clock_local.never_worse``), the faces checked are the replaced
reading's, every re-derivation (as read, per hour, envelope) applies the same rule against the same shipped decision,
and each hour also counts faces whose limit at the actual wind alone (not the larger of read and actual) is below the
step that ran.  The receipt the preflight wrote is kept in the output.

Check the local-face clock's wind reading against the winds a forecast actually carried, at every output time.

``terrain_clock = "local_face"`` reads each face's crest-level wind as 2.3 x what its window's inputs carry, never more
than the domain-wide reading (``gpuwm.terrain_clock_local.WIND_MARGIN`` and the domain cap).  This tool takes a run's
own history files and, at every output time, reads every face's crest-level wind there exactly as the reading reads
its inputs (same window, same band up to ``band_tops``), then compares it with the wind the clock read for that face.

CPU only.  Two ways to get the clock's own reading:

* ``hrrr PREPARED_ROOT TOML WRFOUT_DIR --out JSON``: the tree runner's own preflight
  (``gpuwm.prepared_domain_tree_forecast.preflight_prepared_tree``), with ``derive_local`` recorded so the exact
  run, step, crest and face fields the clock used are kept;
* ``fixture FACES_NPZ RUN_RECEIPT WRFOUT_DIR --out JSON``: a saved door fixture
  (``tools/terrain_clock_local_check.py door``) re-derived as ``reproduce`` does, checked against the run receipt.

Per output time and domain it reports, over all faces, over the faces whose limit depends on the wind at all
("wind-sensitive"), and over the faces that set or nearly set the step (limit at the read wind equal to the
domain's least, or the governing group where no face limits the step; or within 1.25 x the longest step the domain
ran): the worst ratio of actual to
read wind, how many faces' actual wind passed the read wind, and the worst ratio to the domain-wide cap.  It then
asks what the passing would have changed: per face, whether its limit read at the larger of the read and actual
wind falls below the step the domain ran; per domain, the clock decision re-derived with every face read at that
larger wind (the shipped derivation, ``_derive_measured``), against the decision that ran.  The same is done for
the whole-run envelope (each face's strongest actual wind over every output time).
"""
from __future__ import annotations

import argparse
import datetime as dt_mod
import glob
import json
import math
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("GPUWM_NO_LOCAL_GPU", "1")

import numpy as np  # noqa: E402

NEAR = 1.25


def _capture_hrrr(root: Path, cfg: Path):
    import gpuwm.terrain_clock_local as local
    from gpuwm.fetch import sha256_file
    from gpuwm.go_cli import _hierarchy_document
    from gpuwm.prepared_domain_tree_forecast import preflight_prepared_tree

    rec = {}
    original = local.derive_local

    def recording(grid_id, run, dt, slope, crest, faces, *, label=None):
        result = original(grid_id, run, dt, slope, crest, faces, label=label)
        rec[int(grid_id)] = dict(run=run, dt=Fraction(dt), slope=float(slope), crest=crest, faces=faces,
                                 adaptation=result)
        return result

    local.derive_local = recording
    try:
        doc = _hierarchy_document(root)
        inputs = preflight_prepared_tree(prepared_root=root, preparation_receipt_sha256=sha256_file(doc),
                                         experiment_config=cfg, experiment_config_sha256=sha256_file(cfg))
    finally:
        local.derive_local = original
    return rec, dict(inputs.terrain_clock)


def _capture_fixture(fixture: Path, receipt_path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from terrain_clock_local_check import _Run, load_fixture

    from gpuwm.terrain_clock import CrestWind
    from gpuwm.terrain_clock_local import DomainFaces, candidate_map, derive_local

    fields, meta = load_fixture(fixture)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    clock = receipt.get("terrain_clock", receipt)
    row = next(r for r in clock["domains"] if int(r["grid_id"]) == int(meta["grid_id"]))
    arms = tuple(meta["arms"])
    faces = DomainFaces(fields=fields, table=candidate_map(arms), arms=arms, radius_m=float(meta["radius_m"]),
                        domain_wind=float(meta["domain_wind_m_s"]), dx=float(meta["run"]["dx"]),
                        margin=float(meta["margin"]))
    c = meta["crest"]
    crest = CrestWind(label=f"d{meta['grid_id']:02d}", crest_height_m=c["crest_height_m"],
                      wind_m_s=c["crest_level_wind_m_s"], when=c["when"], source=c["source"], height_m=c["height_m"])
    dt = Fraction(row["configured_dt_s"]["numerator"], row["configured_dt_s"]["denominator"])
    run = _Run({**meta["run"], "terrain_clock": "local_face"})
    adaptation = derive_local(int(meta["grid_id"]), run, dt, float(meta["steepest_slope"]), crest, faces)
    rec = {int(meta["grid_id"]): dict(run=run, dt=dt, slope=float(meta["steepest_slope"]), crest=crest,
                                      faces=faces, adaptation=adaptation)}
    return rec, clock


def _decision(a) -> dict:
    return {"division": int(a.division), "time_step_sound": int(a.time_step_sound),
            "dt_s": float(a.dt), "ceiling_s": None if a.ceiling is None else float(a.ceiling),
            "limit_per_km": a.limit_per_km, "cap_per_km": a.cap_per_km, "cap_source": a.cap_source}


def _run_step_s(a, run) -> float:
    """The longest step the domain ran: the written adaptive ceiling, else the adaptive clock's own longest step,
    else the fixed step."""
    if a.ceiling is not None:
        return float(a.ceiling)
    if a.adaptive:
        from gpuwm.core.adaptive_clock import wrf_default_clamps
        from gpuwm.terrain_clock import _adaptive_upper
        upper = _adaptive_upper(run)
        if upper is None:
            upper = Fraction(wrf_default_clamps(run.dx, run.dy)[1])
        return float(upper)
    return float(a.dt)


class Domain:
    def __init__(self, gid, rec):
        from gpuwm.core.adaptive_clock import least_sound_steps
        from gpuwm.terrain_clock_local import _faces, _keys, band_tops

        self.gid = gid
        self.rec = rec
        a = rec["adaptation"]
        self.a = a
        nw = getattr(a, "never_worse", None)
        self.nw = nw
        # The face-by-face derivation as read (the replaced one where the rule applied).
        la = nw.local if (nw is not None and nw.applied) else a
        self.la = la
        f = rec["faces"]
        self.fields, self.table, self.dx = f.fields, f.table, float(f.dx)
        self.domain = float(f.domain_wind)
        self.margin = float(f.margin)
        self.run = rec["run"]
        self.count = int(la.local.sound_steps) if la.local is not None else int(la.time_step_sound)
        self.counts = [least_sound_steps(self.run)]
        if self.counts[0] < 6:
            self.counts.append(6)
        axis, j, i, slope, crest, wind_in = _faces(self.fields)
        self.axis, self.j, self.i, self.slope, self.crest = axis, j, i, slope, crest
        self.wind_in = wind_in
        local = np.where(np.isfinite(wind_in), wind_in, self.domain)
        from gpuwm.terrain_clock_local import face_wind_read
        self.local_in = local
        self.read = face_wind_read(local, self.domain, self.margin)
        self.tops = band_tops(self.fields.terrain, self.fields.radius)
        self.km = self.dx / 1000.0
        self.run_step_s = _run_step_s(a, self.run)
        self.run_per_km = self.run_step_s / self.km
        # Per wind-independent bin, the effective limit at every wind column (plus one past the last).
        winds = np.asarray(self.table.winds)
        self.winds = winds
        base = _keys(self.table, self.dx, slope, crest, np.zeros_like(slope), self.count)
        uniq, inverse = np.unique(base, return_inverse=True)
        first = np.zeros(uniq.size, dtype=np.int64)
        first[inverse[::-1]] = np.arange(inverse.size)[::-1]
        cols = list(winds) + [float(winds[-1]) + 10.0]
        lim = np.empty((uniq.size, len(cols)))
        held = np.empty((uniq.size, len(cols)))
        for b, k in enumerate(first):
            for c, w in enumerate(cols):
                lim[b, c], held[b, c] = self._effective(float(crest[k]), float(slope[k]), float(w), self.count)
        self.lim, self.held, self.inverse = lim, held, inverse
        self.sensitive = lim[inverse].max(axis=1) > lim[inverse].min(axis=1) + 1e-12
        self.eff_read = self.limit_at(self.read)
        least = float(self.eff_read.min())
        self.least_per_km = least
        g = la.local.governing if la.local is not None else None
        self.governing = None
        if g is not None:
            ax = 0 if g.axis == "x" else 1
            hit = np.flatnonzero((axis == ax) & (j == g.j) & (i == g.i))
            self.governing = int(hit[0]) if hit.size else None
        self.held_read = self.held_at(self.read)
        keys = _keys(self.table, self.dx, slope, crest, self.read, self.count)
        group = (keys == keys[self.governing]) if self.governing is not None else np.zeros(slope.shape, bool)
        sets = (self.eff_read <= least * (1.0 + 1e-9)) if math.isfinite(least) else group
        near = self.eff_read <= NEAR * self.run_per_km + 1e-12
        self.setting = group | sets | near
        self.setting_rule = (f"faces that set the step (limit at the read wind equal to the domain's least, "
                             f"{least * self.km:.4g} s, or where no face limits it the governing group; "
                             f"{int((sets | group).sum())}) and faces that nearly set it (limit at the read wind "
                             f"within {NEAR:.2f} x the longest step run, {self.run_step_s:.4g} s; {int(near.sum())})")
        # The derivation reproduced from the face winds as read: must equal what ran.
        again = self.rederive(self.read)
        self.reproduces = (again["division"], again["time_step_sound"], again["ceiling_s"]) == (
            int(a.division), int(a.time_step_sound), None if a.ceiling is None else float(a.ceiling))

    def _effective(self, crest, slope, wind, count):
        """One face's limit on the domain's longest step, s/km, as the clock takes it: no held step reads the most
        stable pair; a fixed limit; on the adaptive clock the cap ``_adaptive_cap`` gives; inf for none."""
        from gpuwm.terrain_clock import _adaptive_cap, _held_step, read_map
        r = read_map(self.dx, crest, slope, wind, count, self.table)
        if r.per_km is None:
            most = float(r.most_stable_per_km) if r.most_stable_per_km is not None else 0.0
            return most, most
        lim = _held_step(r)
        if self.a.adaptive:
            cap, _src = _adaptive_cap(r, lim, self.run, self.table)
            return (math.inf if cap is None else float(cap)), float(r.per_km)
        return (math.inf if lim is None else float(lim)), float(r.per_km)

    def limit_at(self, wind):
        col = np.searchsorted(self.winds, np.asarray(wind) - 1e-9, side="left")
        return self.lim[self.inverse, col]

    def held_at(self, wind):
        col = np.searchsorted(self.winds, np.asarray(wind) - 1e-9, side="left")
        return self.held[self.inverse, col]

    def rederive(self, wind) -> dict:
        from gpuwm.terrain_clock import _derive_measured, read_map
        from gpuwm.terrain_clock_local import _keys, combine
        combined = {}
        for count in self.counts:
            keys = _keys(self.table, self.dx, self.slope, self.crest, wind, count)
            _u, first = np.unique(keys, return_index=True)
            readings = [read_map(self.dx, float(self.crest[k]), float(self.slope[k]), float(wind[k]), count,
                                 self.table) for k in first]
            combined[count] = combine(readings)[0]
        rec = self.rec
        a = _derive_measured(self.gid, self.run, rec["dt"], rec["slope"], rec["crest"], table=self.table,
                             read=combined.__getitem__)
        local_decision = {**_decision(a), "status": a.status}
        if self.nw is not None:
            from gpuwm.terrain_clock_local import never_worse
            a = never_worse(a, self.nw.shipped, self.run)
        out = {**_decision(a), "status": a.status, "local_face_decision": local_decision,
               "never_worse_applied": None if self.nw is None else bool(a.never_worse.applied),
               "never_worse_reason": None if self.nw is None else a.never_worse.reason}
        return out

    def actual(self, path):
        import netCDF4

        from gpuwm.terrain_clock import StartWinds
        from gpuwm.terrain_clock_local import _window_max, start_column_band
        with netCDF4.Dataset(path) as nc:
            nc.set_auto_mask(False)
            arrays = {"u": nc["U"][0], "v": nc["V"][0], "php": nc["PH"][0], "phb": nc["PHB"][0]}
        col = start_column_band(StartWinds("history", arrays.__getitem__), self.tops)
        end = np.where(np.isfinite(col), col, -np.inf)
        w = _window_max(end, self.fields.radius)
        return np.concatenate([np.maximum(w[:, :-1], w[:, 1:]).ravel(), np.maximum(w[:-1, :], w[1:, :]).ravel()])

    def face(self, k, act=None):
        out = {"axis": "x" if int(self.axis[k]) == 0 else "y", "j": int(self.j[k]), "i": int(self.i[k]),
               "slope": round(float(self.slope[k]), 4), "crest_m": round(float(self.crest[k]), 1),
               "input_wind_m_s": round(float(self.wind_in[k]), 2), "read_m_s": round(float(self.read[k]), 2),
               "limit_at_read_s": round(float(self.eff_read[k]) * self.km, 3),
               "held_at_read_s": round(float(self.held_read[k]) * self.km, 3)}
        if act is not None:
            out["actual_m_s"] = round(float(act[k]), 2)
        return out

    def check(self, act) -> dict:
        ratio = act / self.read
        capr = act / self.domain
        over = act > self.read + 1e-9
        hind = np.maximum(self.read, act)
        eff_h = self.limit_at(hind)
        shorter = eff_h < self.eff_read - 1e-12
        refuse = eff_h < self.run_per_km * (1.0 - 1e-9)
        eff_a = self.limit_at(act)
        refuse_actual = eff_a < self.run_per_km * (1.0 - 1e-9)
        past_held = (self.held_at(hind) < self.run_per_km * (1.0 - 1e-9)) & ~(
            self.held_read < self.run_per_km * (1.0 - 1e-9))
        out = {}
        for name, sel in (("all", np.ones(act.shape, bool)), ("wind_sensitive", self.sensitive),
                          ("step_setting", self.setting)):
            if not sel.any():
                out[name] = {"faces": 0}
                continue
            r = np.where(sel, ratio, -np.inf)
            k = int(np.argmax(r))
            out[name] = {"faces": int(sel.sum()), "worst_ratio": round(float(r[k]), 4),
                         "worst_face": self.face(k, act),
                         "over_read": int((over & sel).sum()),
                         "worst_cap_ratio": round(float(np.where(sel, capr, -np.inf).max()), 4),
                         "over_cap": int(((act > self.domain + 1e-9) & sel).sum()),
                         "limit_shortened": int((shorter & sel).sum()),
                         "would_refuse_step_run": int((refuse & sel).sum()),
                         "would_refuse_at_actual_wind": int((refuse_actual & sel).sum()),
                         "lowest_limit_at_actual_only_s": round(float(eff_a[sel].min()) * self.km, 3),
                         "newly_past_held_step": int((past_held & sel).sum()),
                         "lowest_limit_at_read_s": round(float(self.eff_read[sel].min()) * self.km, 3),
                         "lowest_limit_at_actual_s": round(float(eff_h[sel].min()) * self.km, 3)}
        if refuse.any():
            k = int(np.argmax(np.where(refuse, self.run_per_km - eff_h, -np.inf)))
            out["worst_refusal"] = {**self.face(k, act), "limit_at_actual_s": round(float(eff_h[k]) * self.km, 3)}
        if shorter.any():
            k = int(np.argmin(np.where(shorter, eff_h, np.inf)))
            out["worst_shortened"] = {**self.face(k, act), "limit_at_actual_s": round(float(eff_h[k]) * self.km, 3),
                                      "step_setting": bool(self.setting[k])}
        out["actual_max_m_s"] = round(float(act.max()), 2)
        # Which part of the read the actual wind passed: the cap (actual over the read, within margin x input) or the
        # margin (actual past margin x input); and the input factor k each cap passing needs (actual / input).
        capfail = over & (act <= self.margin * self.local_in + 1e-9)
        marfail = act > self.margin * self.local_in + 1e-9
        for name, sel in (("all", np.ones(act.shape, bool)), ("wind_sensitive", self.sensitive), ("step_setting", self.setting)):
            c = capfail & sel; m = marfail & sel
            kk = np.where(c, act / self.local_in, -np.inf)
            out.setdefault("split", {})[name] = {"cap_passed": int(c.sum()), "margin_passed": int(m.sum()),
                "k_needed_max": (round(float(kk.max()), 4) if c.any() else None),
                "cap_passed_shortened": int((c & shorter).sum()), "margin_passed_shortened": int((m & shorter).sum()),
                "cap_passed_refuse": int((c & refuse).sum()), "margin_passed_refuse": int((m & refuse).sum())}
        d = self.rederive(hind)
        out["hindsight_decision"] = d
        out["hindsight_same"] = (d["division"], d["time_step_sound"], d["ceiling_s"]) == (
            int(self.a.division), int(self.a.time_step_sound),
            None if self.a.ceiling is None else float(self.a.ceiling))
        return out

    def header(self, clock_row) -> dict:
        a = self.a
        g = self.governing
        return {"grid_id": self.gid, "dx_m": self.dx, "faces": int(self.slope.size), "sound_steps_read": self.count,
                "neighbourhood_cells": int(self.fields.radius), "margin": self.margin,
                "domain_cap_m_s": round(self.domain, 3), "as_run": _decision(a),
                "status": clock_row.get("status") if clock_row else None,
                "run_step_s": self.run_step_s, "run_per_km": self.run_per_km,
                "least_limit_at_read_s": self.least_per_km * self.km,
                "reproduces_as_run": self.reproduces, "governing": None if g is None else self.face(g),
                "wind_sensitive_faces": int(self.sensitive.sum()), "step_setting_faces": int(self.setting.sum()),
                "step_setting_rule": self.setting_rule,
                "never_worse": None if self.nw is None else self.nw.receipt(),
                "faces_read_at_cap": int((self.margin * np.where(np.isfinite(self.wind_in), self.wind_in,
                                                                 self.domain) > self.domain).sum())}


_NAME = re.compile(r"wrfout_d(\d\d)_(\d{4})-(\d\d)-(\d\d)_(\d\d)[_:](\d\d)[_:](\d\d)$")


def _files(directory: Path):
    out = {}
    for p in sorted(glob.glob(str(directory / "wrfout_d*"))):
        m = _NAME.search(os.path.basename(p))
        if not m:
            continue
        g = int(m.group(1))
        t = dt_mod.datetime(*map(int, m.groups()[1:]))
        out.setdefault(g, []).append((t, p))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hrrr")
    h.add_argument("root", type=Path); h.add_argument("cfg", type=Path); h.add_argument("wrfout", type=Path)
    f = sub.add_parser("fixture")
    f.add_argument("npz", type=Path); f.add_argument("receipt", type=Path); f.add_argument("wrfout", type=Path)
    for p in (h, f):
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--label", default="")
    ap.add_argument("--input-floor", type=float, default=None)
    a = ap.parse_args(argv)
    import gpuwm.terrain_clock_local as _tl
    if a.input_floor is not None: _tl.INPUT_WIND_FLOOR = a.input_floor
    print(json.dumps({"gpuwm": _tl.__file__, "input_wind_floor": _tl.INPUT_WIND_FLOOR, "margin": _tl.WIND_MARGIN}), flush=True)
    t0 = time.time()
    if a.cmd == "hrrr":
        rec, clock = _capture_hrrr(a.root, a.cfg)
    else:
        rec, clock = _capture_fixture(a.npz, a.receipt)
    rows = {int(r["grid_id"]): r for r in clock.get("domains", ())}
    files = _files(a.wrfout)
    start = min(t for lst in files.values() for t, _ in lst)
    result = {"label": a.label, "cmd": a.cmd, "args": [str(x) for x in (argv or sys.argv[1:])],
              "start": start.isoformat(), "domains": {}}
    for gid in sorted(rec):
        d = Domain(gid, rec[gid])
        row = rows.get(gid, {})
        head = d.header(row)
        # The clock the run carried must be the one re-derived here.
        head["receipt_dt_s"] = (row.get("dt_s") or {}).get("seconds")
        head["receipt_max_time_step_s"] = (row.get("max_time_step_s") or {}).get("seconds")
        head["receipt_time_step_sound"] = row.get("time_step_sound")
        lf = row.get("local_faces") or {}
        head["receipt_governing"] = {k: (lf.get("governing") or {}).get(k) for k in (
            "face", "slope", "crest_m", "input_wind_m_s", "wind_read_m_s", "held_s")}
        print(json.dumps({"domain": gid, **{k: head[k] for k in ("as_run", "run_step_s", "reproduces_as_run",
                                                                  "wind_sensitive_faces", "step_setting_faces")}}),
              flush=True)
        times = []
        envelope = np.full(d.slope.shape, -np.inf)
        for t, p in files.get(gid, []):
            act = d.actual(p)
            envelope = np.maximum(envelope, act)
            c = d.check(act)
            c["valid"] = t.isoformat()
            c["hour"] = round((t - start).total_seconds() / 3600.0, 4)
            times.append(c)
            print(json.dumps({"d": gid, "h": c["hour"], "all": c["all"].get("worst_ratio"),
                              "set": c["step_setting"].get("worst_ratio"),
                              "refuse": c["all"].get("would_refuse_step_run"),
                              "refuse_actual": c["all"].get("would_refuse_at_actual_wind"),
                              "same": c["hindsight_same"]}),
                  flush=True)
        env = d.check(envelope) if times else None
        result["domains"][str(gid)] = {"header": head, "times": times, "envelope": env}
    result["terrain_clock_receipt"] = clock
    result["wall_s"] = round(time.time() - t0, 1)
    a.out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
