"""Gauge-calibration door: fit Noah-OWP, CFE and routing parameters to observed discharge on the GPU.

`gpuwm hydro calibrate REQUEST.json` runs an ensemble of parameter sets as members of ONE
resident realization (the mass-ensemble shape): the forcing and PET stages advance once per hour and
are broadcast to M stacked Noah-OWP copies (when the parameter table has Noah-OWP rows, the
full-column table) and M stacked CFE copies (one CUDA thread per catchment and member), the stacked catchment-to-reach transfer and the stacked t-route Muskingum-Cunge network
(M disjoint copies, the same level-ordered launches, level-pool reservoirs at their levels) route
every member, and the gauge reaches are gathered on the card each hour. Parameters are log
multipliers on the regionalised fields per calibration zone (rw_hydro_calib zones), updated by the
ensemble smoother with multiple data assimilation (ES-MDA; the table cites the method). Scores
(KGE, NSE, volume ratio, peak timing) are computed on the card for the calibration and the separate
validation window. Nothing is read back to the host inside the hour loop.

Python here only binds the engine's existing CUDA components (CFEGPU, PreparedRunoffRouting,
MuskingumCunge, Reservoirs, ReservoirLevel) to stacked device arrays and enqueues launches; the
hydrologic arithmetic is the engine's kernels. Observed discharge and zones come from Rust
(rw_hydro_calib obs / zones); the posterior is written back to the fabric by rw_hydro_calib apply.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib
import json
import math
import time
from pathlib import Path
from types import MappingProxyType

BASE_STAGES = ("network", "reservoirs", "areas", "forcing64", "forcing32", "pet", "noah", "cfe")
# Default: the CFE/routing table, the configuration with the best January 2023 Rhine validation so far
# (Dec+Feb, Kaub/Koeln/Rees NSE 0.60/0.94/0.95). The full-column table (Noah-OWP rows as members) is
# selectable with request["parameters"] = FULL_COLUMN_TABLE; on the Nov-Mar Rhine window it fitted the
# calibration months better but did not validate better (hydro-world-calibration-2, 2026-10-18).
PARAMETER_TABLE = Path(__file__).parents[1] / "data" / "hydro" / "calibration-parameters.v1.json"
FULL_COLUMN_TABLE = Path(__file__).parents[1] / "data" / "hydro" / "calibration-parameters.full-column.v1.json"


def register_cli(commands):
    parser = commands.add_parser("calibrate", help="fit CFE/routing parameters to gauges on the GPU (ES-MDA ensemble)")
    parser.add_argument("request", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.set_defaults(func=lambda args: calibrate(json.loads(args.request.read_text()), device=args.device))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _base_components(settings, device_id):
    """The realization's own factories for the shared stages (one member)."""
    components = {}
    from .assembly import prebuild_kernels
    prebuild_kernels([row for row in settings["stages"] if row["name"] in BASE_STAGES],
                     settings=settings, device_id=device_id)
    for row in settings["stages"]:
        if row["name"] not in BASE_STAGES:
            continue
        module, _, entry = row["factory"].partition(":")
        factory = getattr(importlib.import_module(module), entry)
        components[row["name"]] = factory(settings=settings, options=row.get("options", {}), state=None,
                                          physics=None, device_id=device_id, mode="standalone",
                                          components=MappingProxyType(components))
    missing = set(BASE_STAGES) - set(components)
    if missing:
        raise ValueError(f"realization lacks the stages calibration needs: {sorted(missing)}")
    return components


NOAH_FORCING_KEYS = ("U2D", "V2D", "T2D", "PSFC", "Q2D", "LWDOWN", "SWDOWN", "RAINRATE")


def noah_rows(table):
    """Parameter-table rows that act on the Noah-OWP slab (``noah_fields``)."""
    return [p for p in table["parameters"] if "noah_fields" in p]


def _noah_layout(table):
    """(row, [(offset, count) of lead and companions]) for each Noah-OWP table row, from schema.json."""
    from .noahowp.runtime import schema
    fields = {f["name"]: f for f in schema()["fields"]}
    out = []
    for p in noah_rows(table):
        spans = []
        for name in p["noah_fields"]:
            f = fields.get(name)
            if f is None or f["dtype"] != "f32":
                raise ValueError(f"{p['name']}: {name} is not an f32 field of the Noah-OWP slab schema")
            spans.append((f["offset"], f["count"]))
        out.append((p, spans))
    return out


class StackedEnsemble:
    """M members of one realization's CFE + transfer + MC/level-pool network, one launch each.

    With ``noah_members`` the Noah-OWP land model is stacked too (M copies of its slab, one thread
    per catchment-member), so a member is a full column: Noah-OWP, CFE and routing parameters. The
    forcing and PET stages stay shared and are broadcast."""

    def __init__(self, base, members, routing_dt, noah_members=False):
        import cupy as cp
        from . import cfe as cfe_module
        from .runoff import PreparedRunoffRouting
        from .troute_mc import MuskingumCunge, ReservoirLevel
        from .reservoirs import Reservoirs
        self.M = M = members
        self.base = base
        cfe0, net, res0 = base["cfe"], base["network"], base["reservoirs"]
        a0 = cfe0.arrays
        self.a0 = a0
        self.C = C = cfe0.ncatchments
        tile = lambda x: cp.ascontiguousarray(cp.tile(x, (1, M)) if x.ndim == 2 else cp.tile(x, M))
        # ---- CFE: fields/options tiled; ragged buffers tiled with member offsets shifted. The
        # prepared single-member CFE (never launched here) keeps the initial fields; every run
        # rebuilds the stacked fields from it, so no stacked copy of the initial fields is kept.
        self.fields_base = a0["fields"]
        fields = tile(a0["fields"])
        options = tile(a0["options"])
        offsets = tile(a0["offsets"])
        ragged = {}
        for row, name in enumerate(cfe_module.OFFSET_NAMES):
            buffers = (("smc_profile", "soil_layer_depths", "delta_soil_layer_depths")
                       if name == "soil_profile" else (name,))
            size = a0[buffers[0]].size
            shift = (cp.arange(M, dtype=cp.int64) * size).astype(cp.int32)
            offsets[row] = (offsets[row].reshape(M, C) + shift[:, None]).ravel()
            for b in buffers:
                ragged[b] = cp.ascontiguousarray(cp.tile(a0[b], M))
        noah, pet = base["noah"], base["pet"]
        field = lambda comp, name: (getattr(comp, "fields", {}) or {}).get(name, None) if name in (getattr(comp, "fields", {}) or {}) else comp.arrays[name]
        q = field(noah, "water.qinsur")
        self.qinsur, self.pet = q, field(pet, "potential_et_m_s")
        self.noah = None
        if noah_members:
            # Noah-OWP as members: the prepared slabs (initial state, never advanced here) are the
            # template; M copies run in one launch of the same kernel (its column arithmetic does
            # not depend on the column index, so member m sees exactly the single-member inputs).
            from .noahowp.runtime import PreparedNoahOWP
            if noah._forcing_args is None:
                raise ValueError("Noah-OWP members need the realization's resident forcing binding")
            ident = noah.restart_identity
            self.noah_template = {t: noah.arrays["slab/" + t] for t in ("f32", "i32", "f64")}
            slabs = {t: cp.ascontiguousarray(cp.tile(v, (1, M))) for t, v in self.noah_template.items()}
            self.noah_forcing_src = noah._forcing_args[2:]
            stacked_forcing = {k: cp.empty(M * C, dtype=cp.float32) for k in NOAH_FORCING_KEYS}
            self.noah_forcing_dst = [stacked_forcing[k].reshape(M, C) for k in NOAH_FORCING_KEYS]
            self.noah = PreparedNoahOWP(None, slabs=slabs, forcing=stacked_forcing, numerics=ident["numerics"],
                                        stage=ident["stage"], advance=ident["advance"],
                                        parameter_sha256=ident["parameter_sha256"],
                                        fabric_sha256=ident["fabric_sha256"], time_origin=ident["time_origin"])
            self.noah_slabs = slabs
            self.qinsur = self.noah.arrays["water.qinsur"].reshape(M, C)
        self.rain = cp.zeros(M * C, dtype=q.dtype)
        self.pet_st = cp.zeros(M * C, dtype=cp.float64)
        self.cfe = cfe_module.CFEGPU(
            fields, options, offsets, ragged["giuh_ordinates"], ragged["runoff_queue"],
            ragged["nash_subsurface"], ragged["nash_surface"], ragged["smc_profile"],
            ragged["soil_layer_depths"], ragged["delta_soil_layer_depths"],
            period_seconds=cfe0.period_seconds, rainrate=self.rain, potential_et=self.pet_st,
            initial_step=0, time_step_fraction=1.0, rainrate_units="m/s",
            identity={"stacked_members": M, "base": cfe0.restart_identity["preparation"].get("schema")})
        # Initial CFE state for reset. "fields" is rebuilt by _apply on every run. Arrays that are a
        # plain tile of the prepared single-member array (ragged state, options) are reset by
        # broadcasting from it; anything else (shifted offsets, clocks, status) keeps a stacked copy.
        self.cfe_state0, self.cfe_broadcast = {}, {}
        for k, v in self.cfe.arrays.items():
            if k in ("rainrate_mm_s", "precipitation_m_s", "potential_et_m_s", "fields"):
                continue
            b = a0.get(k) if k != "offsets" else None
            if b is not None and b.size * M == v.size and b.dtype == v.dtype and bool(
                    cp.array_equal(v.reshape((b.shape[0], M, -1) if b.ndim == 2 else (M, -1)),
                                   b[:, None, :] if b.ndim == 2 else b.reshape(1, -1))):
                self.cfe_broadcast[k] = b
            else:
                self.cfe_state0[k] = v.copy()
        # ---- transfer
        na = net.arrays
        N = len(net.node_ids)
        self.N = N
        c2n = na["catchment_to_node"]
        c2n_st = (c2n[None, :] + (cp.arange(M, dtype=cp.int32) * N)[:, None]).ravel().astype(cp.int32)
        self.runoff = PreparedRunoffRouting(
            depth=self.cfe.fields["flux_Qout_m"], area=cp.ascontiguousarray(cp.tile(base["areas"].arrays["areasqkm"], M)),
            catchment_to_node=c2n_st, node_count=M * N, period_seconds=cfe0.period_seconds,
            area_units="km2", identity={"stacked_members": M})
        # ---- MC network: M disjoint copies, same levels
        routing = net.manifest["routing"]
        levels = routing["mc_level_offsets"]
        ordered = na["mc_ordered_nodes"]
        mN = (cp.arange(M, dtype=cp.int32) * N)[:, None]
        parts = [(ordered[a:b][None, :] + mN).ravel() for a, b in zip(levels, levels[1:])]
        ordered_st = cp.ascontiguousarray(cp.concatenate(parts).astype(cp.int32))
        levels_st = [int(x) * M for x in levels]
        uo, ui = na["upstream_offsets"], na["upstream_indices"]
        E = ui.size
        uo_st = cp.concatenate([(uo[:-1][None, :] + (cp.arange(M, dtype=cp.int32) * E)[:, None]).ravel(),
                                cp.asarray([M * E], dtype=cp.int32)]).astype(cp.int32)
        ui_st = cp.ascontiguousarray((ui[None, :] + mN).ravel().astype(cp.int32))
        self.params0 = cp.ascontiguousarray(cp.tile(na["routing_parameters"], (M, 1)))
        self.state0 = cp.ascontiguousarray(cp.tile(na[routing["initial_state_array"]], (M, 1)))
        self.router = MuskingumCunge(
            dt_seconds=float(routing_dt), parameters=self.params0.copy(), state=self.state0.copy(),
            lateral=self.runoff.arrays["routing_lateral"], upstream_offsets=uo_st, upstream_indices=ui_st,
            ordered_nodes=ordered_st, level_offsets=levels_st,
            boundary_previous=cp.zeros(M * N, cp.float32), boundary_current=cp.zeros(M * N, cp.float32),
            identity={"network_sha256": net.restart_identity["source_sha256"], "parameters_sha256": "stacked",
                      "ordering_sha256": "stacked", "time_origin": "stacked"},
            assume_short_ts=routing.get("assume_short_ts", False))
        for name, a in (("CFE fields", self.cfe.arrays["fields"]), ("MC parameters", self.router.arrays["parameters"])):
            if not a.flags.c_contiguous:   # _apply writes members in place through reshaped views
                raise ValueError(f"stacked {name} must be contiguous")
        # ---- level-pool reservoirs: every array tiled per member, selections shifted
        ra = res0.arrays
        nres = ra["height"].size
        if ra["series"].size:
            raise ValueError("stacking reservoirs with observation series is not implemented "
                             "(carried series indices would need member offsets); this fabric's lakes "
                             "must be level-pool only")
        arrays = {}
        for k, v in ra.items():
            if k == "selection":
                continue
            if k == "offsets":
                arrays[k] = cp.zeros(M * nres + 1, dtype=v.dtype)
            elif k in ("series", "times"):
                arrays[k] = v
            else:
                arrays[k] = cp.ascontiguousarray(cp.tile(v, (1, M)) if v.ndim == 2 else cp.tile(v, M))
        declaration = net.manifest["reservoirs"]
        hashes = net.restart_identity["payload_sha256"]
        meta = dict(declaration, network_sha256=net.restart_identity["source_sha256"],
                    parameters_sha256=hashes[declaration["arrays"]["parameters"]],
                    series_sha256=hashes[declaration["arrays"]["series"]])
        self.res0 = {k: v.copy() for k, v in arrays.items()}
        self.reservoirs = Reservoirs(arrays, meta)
        hooks = {}
        ni = na[routing["reservoir_node_indices"]]
        ni_st = cp.ascontiguousarray((ni[None, :] + mN).ravel().astype(cp.int32))
        for sel in routing["reservoir_levels"]:
            rows = na[sel["rows"]]
            rows_st = cp.ascontiguousarray((rows[None, :] + (cp.arange(M, dtype=cp.int32) * nres)[:, None]).ravel().astype(cp.int32))
            hooks.setdefault(sel["level"], []).append(ReservoirLevel(
                self.router, self.reservoirs, rows=rows_st, node_indices=ni_st,
                selection_sha256=f"stacked-{sel['rows']}", mapping_sha256="stacked", validated_mapping=True))
        self.router.bind_level_hooks(hooks)
        self.router_state0 = {k: v.copy() for k, v in self.router._base_arrays.items()
                              if k in ("state", "previous", "diagnostics", "boundary_previous", "boundary_current")}
        # the shared stages carry resident state too (forcing clocks, Noah-OWP slab, PET): every run
        # restarts them from their prepared values; the read-only forcing frames are not copied
        self.shared_state0 = {}
        for name in ("forcing64", "forcing32", "pet") + (() if noah_members else ("noah",)):
            comp = base[name]
            for k, v in comp.arrays.items():
                if k == "frames" or not hasattr(v, "data"):
                    continue
                # views and their slabs are both restored (same instant, so the bytes agree)
                self.shared_state0[(name, k)] = v.copy()
        self.routing_dt = float(routing_dt)
        self.substeps = int(round(cfe0.period_seconds / routing_dt))
        self.network = net

    def reset(self):
        """Every member back to the prepared initial state (device copies only)."""
        import cupy as cp
        M = self.M
        for k, v in self.cfe_state0.items():
            cp.copyto(self.cfe.arrays[k], v)
        for k, b in self.cfe_broadcast.items():
            a = self.cfe.arrays[k]
            if b.ndim == 2:
                cp.copyto(a.reshape(b.shape[0], M, -1), b[:, None, :])
            else:
                cp.copyto(a.reshape(M, -1), b[None, :])
        if self.noah is not None:
            for t, b in self.noah_template.items():
                cp.copyto(self.noah_slabs[t].reshape(b.shape[0], M, -1), b[:, None, :])
            self.noah.arrays["status"].fill(0)
            self.noah.arrays["sticky_status"].fill(0)
        for k, v in self.router_state0.items():
            cp.copyto(self.router.arrays[k], v)
        for k, v in self.res0.items():
            cp.copyto(self.reservoirs.arrays[k], v)
        for k in ("routing_lateral", "routing_discharge_m3_s", "routing_volume_m3", "status"):
            self.runoff.arrays[k].fill(0)
        for (name, k), v in self.shared_state0.items():
            cp.copyto(self.base[name].arrays[k], v)


def _build_multipliers(cp, table, U, zone_of, nz, M, C):
    """Per member and catchment multiplier for each table parameter (device)."""
    out, col = {}, 0
    for p in table["parameters"]:
        if p["scope"] == "zone":
            out[p["name"]] = cp.exp(U[:, col:col + nz])[:, zone_of]          # (M, C)
            col += nz
        else:
            out[p["name"]] = cp.exp(U[:, col])                                 # (M,)
            col += 1
    return out


def _apply(cp, ens, table, mult):
    """Write every member's parameters in place: the stacked CFE fields (rebuilt from the prepared
    single-member fields), the routing columns and, with Noah-OWP members, the Noah-OWP parameter
    rows of the stacked slab (after reset). Each value is the regionalised one times the member's
    multiplier, clamped to the table bounds; companions keep their ratio to the lead field."""
    from .cfe import FIELD_INDEX
    M, C = ens.M, ens.C
    F = len(FIELD_INDEX)
    f0 = ens.fields_base[:, None, :]                      # (F, 1, C) prepared fields, broadcast
    f3 = ens.cfe.arrays["fields"].reshape(F, M, C)
    cp.copyto(f3, f0)
    for p in table["parameters"]:
        m = mult[p["name"]]
        if "cfe_fields" in p:
            lead = FIELD_INDEX[p["cfe_fields"][0]]
            lo, hi = p["bounds"]
            m2 = m if m.ndim == 2 else cp.broadcast_to(m[:, None], (M, C))
            new = cp.clip(f0[lead] * m2, lo, hi)
            ratio = cp.where(f0[lead] != 0, new / cp.where(f0[lead] != 0, f0[lead], 1.0), 1.0)
            for name in p["cfe_fields"]:
                i = FIELD_INDEX[name]
                f3[i] = f0[i] * ratio
            f3[lead] = new
    params = ens.router.arrays["parameters"].reshape(M, ens.N, 8)
    cp.copyto(params, ens.params0.reshape(M, ens.N, 8))
    for p in table["parameters"]:
        if "routing_columns" in p:
            lo, hi = p["bounds"]
            m = mult[p["name"]]
            for c in p["routing_columns"]:
                params[:, :, c] = cp.clip(params[:, :, c] * m[:, None].astype(cp.float32), lo, hi)
    if ens.noah is not None:
        _apply_noah(cp, ens, mult)


def _apply_noah(cp, ens, mult):
    """Noah-OWP rows: lead layers new = clip(f64(value) * m, lo, hi) stored as f32 (an exact zero, e.g.
    no canopy height on bare land, stays zero); each companion
    is multiplied by the lead's ratio (layer by layer when it has the lead's layer count, else by the
    first layer's ratio, e.g. kdt = refkdt * dksat(1) / refdk follows dksat). A multiplier of exactly
    1 leaves every value bit-identical (the control member)."""
    M, C = ens.M, ens.C
    s = ens.noah_slabs["f32"]
    v3 = s.reshape(s.shape[0], M, C)
    for p, spans in ens.noah_layout:
        m = mult[p["name"]]
        m2 = (m if m.ndim == 2 else m[:, None]).astype(cp.float64)
        lo, hi = p["bounds"]
        (off, count), companions = spans[0], spans[1:]
        ratios = []
        for j in range(count):
            cur = v3[off + j].astype(cp.float64)
            new = cp.where(cur != 0, cp.clip(cur * m2, lo, hi), 0.0)     # a structural zero stays zero
            ratios.append(cp.where(cur != 0, new / cp.where(cur != 0, cur, 1.0), 1.0))
            v3[off + j] = new.astype(cp.float32)
        for coff, ccount in companions:
            for j in range(ccount):
                r = ratios[j] if ccount == count else ratios[0]
                v3[coff + j] = (v3[coff + j].astype(cp.float64) * r).astype(cp.float32)


def _run(cp, ens, base, U, table, zone_of, nz, hours, gauge_nodes, stream):
    """One stacked forecast; returns the hourly gauge discharge (hours, M, G) on the card."""
    M, C, N = ens.M, ens.C, ens.N
    # Reset and parameter copies are enqueued on the launch stream itself: the stream is
    # non-blocking, so work left on the default stream would race the first hour's kernels.
    with stream:
        mult = _build_multipliers(cp, table, U, zone_of, nz, M, C)
        ens.reset()
        _apply(cp, ens, table, mult)
    stream.synchronize()
    cp.cuda.get_current_stream().synchronize()
    G = gauge_nodes.size
    idx = (gauge_nodes[None, :] + (cp.arange(M, dtype=cp.int32) * N)[:, None]).ravel()
    series = cp.empty((hours, M * G), dtype=cp.float32)
    published = ens.router.arrays["published"]
    noah = ens.noah
    shared = [base[n] for n in ("forcing64", "forcing32", "pet")] + ([] if noah is not None else [base["noah"]])
    rain2, pet2 = ens.rain.reshape(M, C), ens.pet_st.reshape(M, C)
    with stream:
        for h in range(hours):
            for comp in shared:
                comp.advance(3600.0, stream=stream)
            if noah is not None:
                # shared forcing broadcast to every member's column, then all members' Noah-OWP
                for dst, src in zip(ens.noah_forcing_dst, ens.noah_forcing_src):
                    dst[...] = src[None, :]
                noah.launch(3600.0, stream=stream)
                rain2[...] = ens.qinsur
            else:
                rain2[...] = ens.qinsur[None, :]
            pet2[...] = ens.pet[None, :]
            ens.cfe.launch(3600.0, stream=stream)
            ens.runoff.advance(3600.0, stream=stream)
            for _ in range(ens.substeps):
                ens.router.launch(ens.routing_dt, stream=stream)
            cp.take(published[0], idx, out=series[h])
    return series.reshape(hours, M, G)


def _windows(spec):
    """A window [a, b) or a list of disjoint windows [[a, b), [c, d), ...] (hours) -> list of (a, b)."""
    if len(spec) and isinstance(spec[0], (list, tuple)):
        windows = [(int(w[0]), int(w[1])) for w in spec]
    else:
        windows = [(int(spec[0]), int(spec[1]))]
    for a, b in windows:
        if not 0 <= a < b:
            raise ValueError(f"empty or negative window [{a}, {b})")
    return windows


def _hours(cp, windows):
    """Model hours covered by the windows, in order (index array for the gathered series)."""
    return cp.concatenate([cp.arange(a, b) for a, b in windows])


def _scores(cp, sim, obs, windows):
    """KGE (Gupta et al. 2009), NSE, volume ratio, peak timing (h, model minus observed) per member/gauge.

    Several windows are scored as one sample (their hours concatenated); peak timing is the model
    hour of the largest value minus the observed one, in real model hours."""
    if len(windows) == 1:
        a, b = windows[0]
        s = sim[a:b].astype(cp.float64)                     # (T, M, G)
        o = obs[:, a:b].T.astype(cp.float64)[:, None, :]    # (T, 1, G)
        hour = None
    else:
        hour = _hours(cp, windows)
        s = sim[hour].astype(cp.float64)
        o = obs[:, hour].T.astype(cp.float64)[:, None, :]
    valid = cp.isfinite(o) & cp.isfinite(s)
    n = valid.sum(0)
    z = lambda x: cp.where(valid, x, 0.0)
    so, ss = z(o).sum(0) / n, z(s).sum(0) / n
    do, ds = cp.where(valid, o - so, 0.0), cp.where(valid, s - ss, 0.0)
    vo, vs = (do ** 2).sum(0) / n, (ds ** 2).sum(0) / n
    r = (do * ds).sum(0) / n / cp.sqrt(vo * vs)
    alpha, beta = cp.sqrt(vs / vo), ss / so
    kge = 1 - cp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    nse = 1 - (cp.where(valid, s - o, 0.0) ** 2).sum(0) / (do ** 2).sum(0)
    ks, ko = cp.argmax(cp.where(valid, s, -1.0), 0), cp.argmax(cp.where(valid, o, -1.0), 0)
    if hour is not None:
        ks, ko = hour[ks], hour[ko]
    peak = (ks - ko).astype(cp.float64)
    return {"kge": kge, "nse": nse, "volume_ratio": beta, "r": r, "peak_timing_h": peak}


def _blocks(cp, x, windows, block):
    """Block means over each window [a, b) (hours first axis), natural log, windows concatenated in order."""
    out = []
    for a, b in windows:
        T = (b - a) // block
        y = x[a:a + T * block]
        out.append(cp.log(y.reshape(T, block, *y.shape[1:]).mean(1)))
    return out[0] if len(out) == 1 else cp.concatenate(out)


def calibrate(request, *, device=0):
    import cupy as cp
    import numpy as np
    from gpuwm import numerics
    from .runtime import read_settings, resolve_settings
    t_all = time.perf_counter()
    out = Path(request["out"])
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    table_path = Path(request.get("parameters", PARAMETER_TABLE))
    table = json.loads(table_path.read_text())
    zones = json.loads(Path(request["zones"]).read_text())
    obs_meta = json.loads((Path(request["obs"]) / "obs.json").read_text())
    settings = read_settings(request["realization"])
    settings["enabled"] = True
    settings = resolve_settings(settings)
    numerics.select(settings.get("numerics", "fast"))
    hours = int(request["hours"])
    if hours > obs_meta["hours"]:
        raise ValueError("observations do not cover the requested hours")
    M = int(request.get("members", 64))
    Na = int(request.get("iterations", 4))
    block = int(request.get("block_hours", 6))
    sigma = float(request.get("obs_error_log", 0.10))
    # ES-MDA update localization: "none" (plain ES-MDA) or "correlation" (Luo et al. 2018)
    localization = request.get("localization", "none")
    if localization not in ("none", "correlation"):
        raise ValueError(f"unknown localization {localization!r}")
    # calibration_hours / validation_hours: one window [a, b) or a list of windows; a calibration
    # window that touches a validation hour is refused, so the validation period never informs the
    # posterior (it may sit between two calibration windows, e.g. calibrate Dec + Feb, validate Jan).
    cal = _windows(request["calibration_hours"])
    val = _windows(request["validation_hours"])
    for ca, cb in cal:
        for va, vb in val:
            if not (cb <= va or vb <= ca):
                raise ValueError("calibration and validation windows overlap")
    if max(b for _, b in cal + val) > hours:
        raise ValueError("a calibration or validation window ends after the requested hours")
    budget = float(request.get("budget_seconds", 480))
    stations = [s["station_id"] for s in obs_meta["stations"]]
    cal_st = request.get("calibration_stations", stations)
    cal_cols = [stations.index(s) for s in cal_st]
    nz = len(zones["zones"])
    P = sum(nz if p["scope"] == "zone" else 1 for p in table["parameters"])
    sig = []
    for p in table["parameters"]:
        sig += [p["prior_log_sigma"]] * (nz if p["scope"] == "zone" else 1)
    receipt = {"schema": "arwen.hydro.calibration.receipt.v1", "request": request,
               "parameter_table": {"path": str(table_path), "sha256": _sha(table_path)},
               "realization": {"path": request["realization"], "sha256": _sha(request["realization"])},
               "zones": {"path": request["zones"], "sha256": _sha(request["zones"]),
                         "stations": [z["station_id"] for z in zones["zones"]]},
               "members": M, "parameters_per_member": P, "iterations": Na, "jobs": []}
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        rng = np.random.default_rng(int(request.get("seed", 20230101)))
        U0 = (rng.standard_normal((M, P)) * np.asarray(sig)).tolist()
        state = {"iteration": 0, "U": U0, "history": [], "final": None, "seed": int(request.get("seed", 20230101))}
    job = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "iterations_run": [],
           "gpu": None}
    with cp.cuda.Device(device):
        job["gpu"] = cp.cuda.runtime.getDeviceProperties(device)["name"].decode()
        t0 = time.perf_counter()
        base = _base_components(settings, device)
        routing_dt = next(r for r in settings["stages"] if r["name"] == "routing")["options"].get("dt_seconds", 300)
        # members + 2 diagnostic members: control (regionalised, all multipliers 1) and the ensemble mean
        layout = _noah_layout(table)
        ens = StackedEnsemble(base, M + 2, routing_dt, noah_members=bool(layout))
        ens.noah_layout = layout
        if layout:
            # a regionalised value outside a row's bounds would be clamped even at multiplier 1 and
            # the control member would stop being the regionalised run: refuse instead
            t32 = ens.noah_template["f32"]
            outside = {}
            for p, spans in layout:
                off, count = spans[0]
                v = t32[off:off + count]
                n_out = int(((v != 0) & ((v < p["bounds"][0]) | (v > p["bounds"][1]))).sum())
                if n_out:
                    outside[p["name"]] = n_out
            if outside:
                raise ValueError(f"regionalised Noah-OWP values outside the table bounds (catchment-layers): {outside}")
            receipt["noah_members"] = {"parameters": [p["name"] for p, _ in layout],
                                       "fields": {p["name"]: p["noah_fields"] for p, _ in layout},
                                       "shared_stages": ["forcing64", "forcing32", "pet"],
                                       "stacked_columns": ens.C * (M + 2)}
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        job["device_memory_after_prepare"] = {"pool_used_bytes": int(pool.used_bytes()),
                                              "free_total_bytes": list(cp.cuda.runtime.memGetInfo())}
        zone_of = cp.asarray(zones["catchment_zone"], dtype=cp.int32)
        if zones["catchment_ids"] != list(base["network"].catchment_ids):
            raise ValueError("zones catchment order differs from the realization network")
        gauge_nodes = cp.asarray([s["node_index"] for s in obs_meta["stations"]], dtype=cp.int32)
        G = gauge_nodes.size
        raw = (Path(request["obs"]) / "obs.f32").read_bytes()
        obs = cp.asarray(np.frombuffer(raw, dtype="<f4").reshape(G, obs_meta["hours"])[:, :hours])
        cp.cuda.get_current_stream().synchronize()
        job["prepare_seconds"] = time.perf_counter() - t0
        stream = cp.cuda.Stream(non_blocking=True)
        yobs = _blocks(cp, obs.T, cal, block)[:, cal_cols]                     # (Tb, Gc)
        ok = cp.isfinite(yobs).ravel()
        it_seconds = None
        while state["iteration"] <= Na and (state["final"] is None):
            elapsed = time.perf_counter() - t_all
            if it_seconds is not None and elapsed + it_seconds > budget:
                break
            ti = time.perf_counter()
            U = cp.asarray(state["U"], dtype=cp.float64)
            Ufull = cp.concatenate([U, cp.zeros((1, P)), U.mean(0, keepdims=True)])
            sim = _run(cp, ens, base, Ufull, table, zone_of, nz, hours, gauge_nodes, stream)
            stream.synchronize()
            run_s = time.perf_counter() - ti
            sc_cal = _scores(cp, sim, obs, cal)
            sc_val = _scores(cp, sim, obs, val)
            row = {"iteration": state["iteration"], "run_seconds": run_s,
                   "members_scored": M + 2,
                   "calibration": {k: v.get().tolist() for k, v in sc_cal.items()},
                   "validation": {k: v.get().tolist() for k, v in sc_val.items()},
                   "U_mean": U.mean(0).get().tolist(), "U_std": U.std(0).get().tolist()}
            if state["iteration"] == Na:
                # posterior: the final ensemble's scores; the calibrated set is the ensemble mean
                state["final"] = row
                state["posterior_U_mean"] = row["U_mean"]
                hourly = sim.get()
                np.save(out / "final-gauge-series.npy", hourly)
            else:
                g = _blocks(cp, sim[:, :M, :], cal, block)[:, :, cal_cols]        # (Tb, M, Gc)
                g = g.transpose(1, 0, 2).reshape(M, -1)[:, ok]
                y = yobs.ravel()[ok]
                bad = ~cp.isfinite(g).all(1)
                if bool(bad.any()):
                    g = cp.where(cp.isfinite(g), g, y[None, :])                 # a non-finite member keeps its parameters only through the mean
                alpha = float(Na)
                D = y.size
                Um, gm = U.mean(0), g.mean(0)
                dU, dg = U - Um, g - gm
                Cug = dU.T @ dg / (M - 1)
                Cgg = dg.T @ dg / (M - 1)
                if localization == "correlation":
                    # correlation-based adaptive localization (Luo, Bhakta and Naevdal 2018, SPE J 23(2):
                    # 396-427, for ES-MDA): a parameter-observation covariance is kept only where its
                    # correlation exceeds the noise of an M-member estimate times the universal threshold
                    # sqrt(2 ln(P D)); with many weakly identifiable parameters (the full column) the
                    # spurious correlations otherwise move them and collapse the ensemble.
                    corr = Cug / cp.maximum(cp.outer(dU.std(0, ddof=1), dg.std(0, ddof=1)), 1e-300)
                    threshold = math.sqrt(2.0 * math.log(P * dg.shape[1])) / math.sqrt(M)
                    keep = cp.abs(corr) > threshold
                    Cug = cp.where(keep, Cug, 0.0)
                    row["localization"] = {"method": "correlation (Luo et al. 2018)", "threshold": threshold,
                                           "kept_fraction_per_parameter": keep.mean(1).get().tolist()}
                rng = np.random.default_rng(state["seed"] + 7919 * state["iteration"])
                pert = y[None, :] + math.sqrt(alpha) * sigma * cp.asarray(rng.standard_normal((M, D)))
                K = cp.linalg.solve(Cgg + alpha * sigma ** 2 * cp.eye(D), (pert - g).T)    # (D, M)
                U = U + (Cug @ K).T
                state["U"] = U.get().tolist()
                row["nonfinite_members"] = int(bad.sum())
                row["observations_used"] = int(D)
                if state["iteration"] == 0:
                    # prior sensitivity: mean |corr(u_j, g_d)| over the assimilated observations;
                    # with M members a parameter with no effect still shows about sqrt(2/(pi M))
                    zu = (dU / cp.maximum(dU.std(0), 1e-300))
                    zg = (dg / cp.maximum(dg.std(0), 1e-300))
                    corr = zu.T @ zg / M                                          # (P, D)
                    row["prior_sensitivity_mean_abs_corr"] = cp.abs(corr).mean(1).get().tolist()
                    row["prior_sensitivity_noise_floor"] = math.sqrt(2.0 / (math.pi * M))
            if state["iteration"] == 0:
                hourly0 = sim.get()
                np.save(out / "prior-gauge-series.npy", hourly0)
            state["history"].append(row)
            state["iteration"] += 1
            it_seconds = time.perf_counter() - ti
            job["iterations_run"].append({"iteration": row["iteration"], "seconds": it_seconds, "run_seconds": run_s})
            state_path.write_text(json.dumps(state))
    job["wall_seconds"] = time.perf_counter() - t_all
    job["device_memory_pool_held_bytes_end"] = int(cp.get_default_memory_pool().total_bytes())
    job["parallelism"] = (f"GPU: {M + 2} members stacked in one launch per stage; "
                          + (f"Noah-OWP one thread per catchment-member ({ens.C * (M + 2)} columns, shared forcing broadcast); "
                             if ens.noah is not None else "Noah-OWP once, broadcast; ")
                          + f"CFE one thread per catchment-member "
                          f"({ens.C * (M + 2)} threads); MC {len(ens.router._level_launches)} level launches x "
                          f"{ens.substeps} substeps per hour over {(M + 2)} network copies; ES-MDA update on the card")
    receipt["jobs"] = json.loads((out / "receipt.json").read_text())["jobs"] if (out / "receipt.json").exists() else []
    receipt["jobs"].append(job)
    receipt["done"] = state["final"] is not None
    if receipt["done"]:
        names = []
        for p in table["parameters"]:
            names += ([f"{p['name']}@{z['station_id']}" for z in zones["zones"]] if p["scope"] == "zone" else [p["name"]])
        um = state["posterior_U_mean"]
        posterior, col = [], 0
        for p in table["parameters"]:
            k = nz if p["scope"] == "zone" else 1
            mults = [math.exp(x) for x in um[col:col + k]]
            entry = {"name": p["name"], "column": p["column"], "scope": p["scope"]}
            if "noah_fields" in p:
                entry.update(noah_fields=p["noah_fields"], bounds=p["bounds"])
            entry.update({"multipliers": mults} if p["scope"] == "zone" else {"multiplier": mults[0]})
            posterior.append(entry)
            col += k
        (out / "posterior.json").write_text(json.dumps({"schema": "arwen.hydro.calibration.posterior.v1",
            "parameters": posterior, "zones": [z["station_id"] for z in zones["zones"]],
            "u_names": names, "u_mean": um}, indent=1) + "\n")
        receipt["posterior"] = posterior
        receipt["stations"] = stations
        receipt["member_layout"] = {"0..M-1": "ensemble members", "M": "control: regionalised parameters (all multipliers 1)",
                                    "M+1": "ensemble mean of the parameters (the calibrated set)"}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({"done": receipt["done"], "iteration": state["iteration"], "job": job}))
    return 0 if receipt["done"] else 3
