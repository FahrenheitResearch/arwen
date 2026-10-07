"""GSL fire preparation, cycle and injection, ccpp-physics 3e6660c6.

SourceFrames contract: reference_time is an aware UTC simulation start;
available_hours(source, field) returns posted hourly datetimes; at(source,
field, time) returns host arrays on the model grid. Only at reads data.
Source variable metadata has a fire mapping: frp names the source's MW
field, diagnostic selects the EBU_IN/FRP_MEAN diagnostic, and optional
history_source, history_hwp, history_rain name prior model physics frames.
state_factors maps state units to source-mass conversion factors; ug kg-1
defaults to one, while a gas factor must be supplied as source-row data.

The plume lane publishes k_min/k_max/flam_frac and plume_ran through met,
and ebu_distribute(ctx, emission, out) uses those cached fields. Distribute
runs on a new source hour or a plume call; GSL only rebuilds on plume calls
(rrfs_smoke_wrapper.F90:354-370). Hour-aligned 60-minute calls agree.
"""
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
from pathlib import Path

from gpuwm.core.chem_context import ChemAllocation

KEY = "emission.fire"
LEDGER = "emitted"
REQUIRES = ("dryrho", "dz8w", "msftx", "msfty")
ALLOCATES = (
    ChemAllocation("fire_ebu_in", "2d", output_name="EBU_IN", units="ug m-2 s-1",
                   description="Fire emission flux before the diurnal multiplier"),
    ChemAllocation("fire_frp_mean", "2d", output_name="FRP_MEAN", units="MW",
                   description="Fire radiative power used this hour"),
    ChemAllocation("fire_emitted_mass", "rows_2d", output_name="FIRE_EMITTED", units="ug m-2"),
    ChemAllocation("fire_coef_bb_dc", "rows_2d", output_name="COEF_BB_DC"),
    ChemAllocation("fire_hist", "rows_2d", output_name="FIRE_HIST"),
    ChemAllocation("fire_hwp", "2d", restart="rebuild", output_name="HWP"),
    ChemAllocation("fire_kpbl", "2d", dtype="int32", restart="rebuild"),
    ChemAllocation("fire_kpbl_thetav", "2d", dtype="int32", restart="rebuild"),
    ChemAllocation("fire_uspdavg2d", "2d", restart="rebuild"),
    ChemAllocation("fire_windgustpot", "2d", restart="rebuild"),
    ChemAllocation("fire_hpbl2d", "2d", restart="rebuild"),
    ChemAllocation("fire_type", "2d", dtype="int32", restart="rebuild"),
    ChemAllocation("fire_ebu", "rows_3d"),
    ChemAllocation("fire_cache_hour", "rows_2d", dtype="int32"),
    ChemAllocation("fire_cache_ref", "rows_2d", dtype="int32"),
    # The diurnal coefficient and fire history carried between steps, one
    # plane per row (a row has one fire source field, see step).  They were
    # rows_3d with a plane per possible reference, nz planes of which a smoke
    # row used one: 2 x 108 MB at 896 x 512 x 59 for two 2-D planes.
    ChemAllocation("fire_coef_carry", "rows_2d"),
    ChemAllocation("fire_hist_carry", "rows_2d"),
    *(ChemAllocation(n, "2d", restart="rebuild") for n in
      ("fire_input_mass", "fire_input_frp", "fire_flux", "fire_area", "fire_prev_hwp", "fire_prev_rain", "fire_end")),
)
KERNEL_MODULES = ("chem_fire",)
RULE_PATH = Path(__file__).resolve().parents[1] / "data/chem/fire/fire_type_rules.json"


def rows(table):
    return table.rows_for(KEY)


def refusal(cfg):
    """Why this configuration cannot run emission.fire, or None.

    Checked at configuration, so a run that could never emit stops before
    its preparation instead of at its first chem step:

    * a fire row whose fire sources are all left out of ``chem_sources``
      would emit nothing while the run claims fire smoke (the frames refuse
      a source that is not enabled);
    * ``daily_mean_dcycle`` (RRFS ebb_dcycle = 2) scales the previous day's
      mean by the ratio of today's to yesterday's model wildfire potential,
      and needs yesterday's model HWP and rain as a source row's
      ``history_source`` frames; no enabled fire source names one, and a
      cold start would fabricate them (SRW-SD cycle.py:240-250).
    """
    from gpuwm.chem_table import load
    table = load(cfg)
    if table is None:
        return None
    mode = getattr(cfg, "fire_emission_mode", TRAILING_MODE)
    for row in rows(table):
        fire = [ref for ref in row.emissions
                if table.sources[ref["source"]].variables[ref["field"]].get("fire") is not None]
        enabled = [ref for ref in fire if ref["source"] in table.enabled_sources]
        if fire and not enabled:
            names = sorted({ref["source"] for ref in fire})
            return (f"row {row.name!r} takes its fire emission from {names}, and "
                    f"chem_sources enables none of them, so no fire would be "
                    f"emitted; add {names[0]!r} to chem_sources")
        if mode == "daily_mean_dcycle" and not any(
                table.sources[ref["source"]].variables[ref["field"]]["fire"].get("history_source")
                for ref in enabled):
            return (f"fire_emission_mode = 'daily_mean_dcycle' needs the previous "
                    f"day's model wildfire potential and rain (a fire source's "
                    f"history_source frames), and no enabled fire source of row "
                    f"{row.name!r} names one, so the diurnal ratio would read "
                    f"fabricated history; run the default trailing_24h_dcycle or observed_hourly")
    return None


#: The default fire timing: HRRR-Smoke's method.  HRRRv4 maps the fire
#: hotspots of the previous 24 h onto its grid and emits them "with
#: diurnally varying emissions" (Dowell et al. 2022, Wea. Forecasting 37,
#: doi:10.1175/WAF-D-21-0151.1, section 3g).  Here: each cell's mean hourly
#: emission and FRP over the hours of the trailing 24 h it burned in (the
#: newest hour posted by the start included, so a fire hours old already
#: emits), times a prescribed diurnal curve (:func:`diurnal_factor`).
TRAILING_MODE = "trailing_24h_dcycle"

#: Retired timings, each with the concrete breakage that retired it.
RETIRED_MODES = {
    "persistence_hourly": (
        "fire_emission_mode = 'persistence_hourly' is retired: it read only "
        "the same hour of the previous day, so a fire younger than a day "
        "emitted nothing (measured on the 2025-01-08 12Z Los Angeles case: "
        "no smoke for 7 of 12 hours and the Eaton fire never appeared). "
        "Use the default 'trailing_24h_dcycle' (the previous 24 h of "
        "detections with a diurnal cycle) or 'observed_hourly' for a "
        "hindcast."),
}


def trailing_window(start, available_hours, latency_s):
    """The trailing-24 h window of posted source hours a run started at
    ``start`` can read: every posted hour from 23 hours before the newest
    hour available at the start through that hour.  The newest is the hour
    whose start lies ``latency_s`` (the source row's posting latency) before
    the forecast start, so a hindcast reads only what a real-time run would
    have had.  Missing hours inside the window are gaps, not refusals (a
    fire's mean is over the hours it was seen); an empty window is refused
    by name, because a run with no fire hours would claim smoke it cannot
    emit."""
    if start.tzinfo is None:
        raise ValueError("fire source time needs a timezone to prevent selecting the wrong UTC hourly file")
    if not (latency_s >= 0):
        raise ValueError("the source row's latency_s must be a nonnegative number; a negative one would read hours not yet posted")
    newest = (start.astimezone(timezone.utc) - timedelta(seconds=float(latency_s))).replace(
        minute=0, second=0, microsecond=0)
    first = newest - timedelta(hours=23)
    window = tuple(sorted(h for h in frozenset(available_hours) if first <= h <= newest))
    if not window:
        raise FileNotFoundError(
            f"{TRAILING_MODE}: no hourly source file is posted from {first:%Y%m%d%H} "
            f"through {newest:%Y%m%d%H}; substituting another hour would fabricate fire emissions")
    return window


def diurnal_factor(xp, longitude_deg, when):
    """The prescribed diurnal multiplier at ``when`` (UTC), mean one over a day.

    WRF-Chem v4.7.1's biomass-burning diurnal cycle (chem/
    module_add_emiss_burn.F:53-74, Freitas et al. 2011): a Gaussian of width
    ``cx`` on a small linear base, normalized so its daily integral is one,
    times 86400 s.  WRF-Chem centres it at 18 UTC (South America); here it is
    centred on GSL's longitude-band peak hour for North American fires
    (``peak_bands`` of the fire rule table, rrfs_smoke_wrapper.F90:1048-1060:
    22-23 UTC for the western US), by shifting the curve's clock, which keeps
    the daily mean one.  ``longitude_deg`` is an array (east positive);
    returns float32 on ``xp``.
    """
    data = _data()
    curve = data["trailing_diurnal"]
    lon = xp.mod(xp.asarray(longitude_deg, dtype=xp.float64), float(data["longitude_period"]))
    peak = xp.full(lon.shape, float(data["peak_bands"][-1]["hour"]), dtype=xp.float64)
    for band in reversed(data["peak_bands"][:-1]):
        peak = xp.where(lon < float(band["longitude_lt"]), float(band["hour"]), peak)
    when = when.astimezone(timezone.utc)
    seconds = when.hour * 3600.0 + when.minute * 60.0 + when.second
    bx, cx = float(curve["bx_s"]), float(curve["cx_s"])
    tq = xp.mod(seconds - (peak * 3600.0 - bx), 86400.0)
    r = float(curve["rinti"]) * (float(curve["ax"]) * xp.exp(-((tq - bx) ** 2) / (2.0 * cx * cx))
                                 + float(curve["base"]) - float(curve["slope"]) * tq)
    return (r * 86400.0).astype(xp.float32)


def trailing_planes(ctx, frames, source, field, frp_field, valid):
    """``(mass, frp)`` of the valid hour under :data:`TRAILING_MODE`, on the device.

    The trailing window's per-cell means are formed once per run (cupy, on
    the frames ``frames.at`` returns): a cell's mean over the window hours in
    which its source mass is positive, for mass and for FRP alike, zero where
    it never burned.  Each hour multiplies them by :func:`diurnal_factor` at
    the hour's midpoint.
    """
    import cupy as cp
    chem = ctx.state.chem
    key = (source.name, field, frp_field)
    cache = getattr(chem, "fire_trailing", None)
    if cache is None or cache["key"] != key:
        latency = float((getattr(source, "time", None) or {}).get("latency_s") or 0.0)
        hours = trailing_window(frames.reference_time,
                                frames.available_hours(source.name, field), latency)
        mass_sum = frp_sum = count = None
        for hour in hours:
            mass = cp.asarray(frames.at(source.name, field, hour), dtype=cp.float32)
            power = cp.asarray(frames.at(source.name, frp_field, hour), dtype=cp.float32)
            burning = mass > 0
            if mass_sum is None:
                mass_sum = cp.zeros(mass.shape, cp.float64)
                frp_sum = cp.zeros(mass.shape, cp.float64)
                count = cp.zeros(mass.shape, cp.int32)
            mass_sum += cp.where(burning, mass, 0)
            frp_sum += cp.where(burning, cp.maximum(power, 0), 0)
            count += burning
        seen = cp.maximum(count, 1)
        cache = chem.fire_trailing = {
            "key": key, "hours": hours,
            "mass": (mass_sum / seen).astype(cp.float32),
            "frp": (frp_sum / seen).astype(cp.float32)}
    hour = valid.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    factor = diurnal_factor(cp, ctx.met("xlong"), hour + timedelta(minutes=30))
    return cache["mass"] * factor, cache["frp"] * factor, hour


def select_frames(mode, valid_time, available_hours):
    """Select source hours, with no network or array processing.

    Daily input window follows public SRW-SD preprocessing cycle.py:53-57,
    228-230 at commit 2ad2bc5731819cdc5aef95372a8b27ffe417bf40:
    https://raw.githubusercontent.com/ufs-community/ufs-srweather-app/2ad2bc5731819cdc5aef95372a8b27ffe417bf40/ush/smoke_dust/core/cycle.py
    It starts 25 hours before initialization and ends two hours before it.
    The default :data:`TRAILING_MODE` reads its window through
    :func:`trailing_window`, not here.
    """
    if valid_time.tzinfo is None:
        raise ValueError("fire source time needs a timezone to prevent selecting the wrong UTC hourly file")
    valid_time = valid_time.astimezone(timezone.utc).replace(minute=0,second=0,microsecond=0)
    available = frozenset(available_hours)
    if mode in RETIRED_MODES:
        raise ValueError(RETIRED_MODES[mode])
    if mode == "observed_hourly":
        selected = (valid_time,)
    elif mode == "daily_mean_dcycle":
        selected = tuple(valid_time-timedelta(hours=h) for h in range(25,1,-1))
    else:
        raise ValueError(f"fire emission mode {mode!r} has no defined source clock; an arbitrary hour would corrupt emission timing")
    for hour in selected:
        if hour not in available:
            filename = f"hourly source file {hour:%Y%m%d%H}00_{hour:%Y%m%d%H}59.nc"
            raise FileNotFoundError(f"{mode}: missing {filename}; substituting another hour would fabricate fire emissions")
    return selected


@lru_cache(maxsize=1)
def _data():
    """The fire rule table, read once per process (it is package data)."""
    return json.loads(RULE_PATH.read_text(encoding="utf-8"))


def _tables(module=None):
    import numpy as np
    data = _data()
    cats,groups,rule_values,curves,decay = [],[],[],[],[]
    names = {g["name"]: i for i,g in enumerate(data["groups"])}
    for index,group in enumerate(data["groups"]):
        if group.get("parent") and names[group["parent"]]>=index:
            raise ValueError("fire group parents must precede children; otherwise classification reads unwritten group sums")
        if any(c<1 or int(c)!=c for c in group["categories"]):
            raise ValueError("fire categories must be positive integer table indices; invalid indices would read outside land-use fractions")
        groups.extend((len(cats),len(group["categories"]),names.get(group.get("parent"),-1)))
        cats.extend(c-1 for c in group["categories"])
    for rule in data["rules"]:
        if "group" in rule:
            rule_values.extend((names[rule["group"]],rule["gt"],0,0,rule["type"]))
        elif "longitude_gt" in rule:
            rule_values.extend((-1,rule["longitude_gt"],rule["latitude_gt"],rule["latitude_lt"],rule["type"]))
        else:
            rule_values.extend((-2,0,0,0,rule["type"]))
    for index,curve in enumerate(data["curves"]):
        if curve["type"]!=index:
            raise ValueError("fire curve type rows must be contiguous from zero; a gap would read another type's curve constants")
        kind = {"carry":0,"lognormal":1,"hwp":2}[curve["kind"]]
        curves.extend((kind,curve.get("C",0),curve.get("sigma",0),curve.get("average",0),curve.get("denominator",0)))
    for d in data["night_decay"]:
        decay.extend((d["age_gt"],d["cap"]))
    for key,values,bound in (("categories",cats,64),("groups",groups,48),("rules",rule_values,160),
                             ("curves",curves,160),("decay",decay,64)):
        if len(values)>bound:
            raise ValueError(f"fire {key} table exceeds {bound} constant words; refusing prevents device out-of-bounds rule reads")
        if module is not None:
            host=np.zeros(bound,np.float32)
            host[:len(values)]=values
            pointer=module.get_global(f"fire_{key}")
            pointer.copy_from_host(host.ctypes.data,host.nbytes)
    return data


def _require(ctx, fields, arm):
    for name in fields:
        if not ctx.has(name):
            raise ValueError(f"{arm} needs missing field {name}; a substitute would change GSL wildfire potential or fire classification")


def init(ctx):
    for name in ("add_fire_moist_flux", "add_fire_heat_flux"):
        if getattr(ctx.cfg,name,False):
            raise ValueError(f"{name} is out of v1 (DESIGN 5.6); enabling it would change water or energy without a qualified coupling ledger")
    from gpuwm.core.kernels import load_module
    _tables(load_module("chem_fire"))


def _prep(ctx, hour, kernel, method=None):
    import numpy as np
    method = getattr(ctx.cfg,"hwp_method",2) if method is None else method
    if method not in (0,1,2,3,4):
        raise ValueError("hwp_method must be 1..4 to prevent silently zero wildfire potential")
    fields=("t_phy","p_phy","qv","u_phy","v_phy","z","z_at_w","pblh","oro",
            "u10","v10","t2","dpt2m","wetness","swdown","snow")
    _require(ctx,fields[:11] if method==0 else fields,f"hwp_method={method}")
    if method in (1,3):
        _require(ctx,("totprcp",),f"hwp_method={method}")
    # In snow methods precip is unused, so a valid existing scratch pointer suffices.
    precip=ctx.met("totprcp") if method in (1,3) else ctx.diag["fire_prev_rain"]
    nz,ny,nx=ctx.met("t_phy").shape
    if nz<3:
        raise ValueError("GSL smoke prep needs at least three mass levels; kpbl+1 winds and the second-level virtual temperature would read outside the column")
    kernel(((nx*ny+127)//128,), (128,),
        (*(ctx.met(n) if method or n in fields[:11] else ctx.diag["fire_prev_rain"] for n in fields[:14]),
         precip,ctx.diag["fire_prev_rain"],
         ctx.met("swdown") if method else ctx.diag["fire_prev_rain"],
         ctx.met("snow") if method else ctx.diag["fire_prev_rain"],
         *(ctx.diag["fire_"+n] for n in ("kpbl","kpbl_thetav","uspdavg2d","windgustpot","hpbl2d","hwp")),
         np.int32(nz),np.int32(nx*ny),np.int32(hour),np.int32(method)))


def plume_inputs(ctx, ktau):
    """The plume call's inputs, as GSL's wrapper forms them (rrfs_smoke_wrapper.F90:448-457).

    ``frp_inst`` is the fire source hour's FRP (the hour ``fire_emission_mode``
    selects, MW) in W (conv_frp = 1e6, :735) times the fire's current
    ``coef_bb_dc`` (one under ebb_dcycle 1), times ``sc_factor`` where the
    fire type is 4; the plume launch applies the frp_max cap.  ``kpbl``,
    ``uspdavg2d`` and ``hpbl2d`` come from :func:`prepare_columns`.
    ``ebu_in`` gates the plume to columns that emit.  One fire-power field
    sizes a column's plume: two fire rows naming different FRP fields would
    size one plume twice, and are refused.
    """
    import numpy as np
    import cupy as cp
    from gpuwm.chem_emission_frames import frames_for_context
    frames=frames_for_context(ctx)
    # Plume rise precedes emission.fire. Fresh-strip carry must exist before
    # sizing its first plume, rather than multiplying FRP by an unwritten
    # previous-step diagnostic coefficient.
    start_carry(ctx,ktau)
    mode=getattr(ctx.cfg,"fire_emission_mode",TRAILING_MODE)
    start=frames.reference_time
    valid=start+timedelta(seconds=ctx.clock.curr_secs)
    power=None; emission=None
    for row in rows(ctx.table):
        for ref in row.emissions:
            if ref["vertical"]!="plumerise":
                continue
            source=ctx.table.sources[ref["source"]]
            metadata=source.variables[ref["field"]].get("fire")
            if metadata is None:
                continue
            key=(source.name,metadata["frp"])
            if power is not None and power!=key:
                raise ValueError(f"two fire-power fields {power} and {key} would size one plume twice; one emitting source row per plume")
            power=key
            if mode=="daily_mean_dcycle":
                raise ValueError(f"daily_mean_dcycle's plume power needs the previous-day FRP reduction, which is refused at a cold start (no previous-day model HWP); run the default {TRAILING_MODE} or observed_hourly")
            if mode==TRAILING_MODE:
                flux,frp,_hour=trailing_planes(ctx,frames,source,ref["field"],metadata["frp"],valid)
            else:
                hours=select_frames(mode,valid,frames.available_hours(source.name,ref["field"]))
                frp=cp.asarray(frames.at(source.name,metadata["frp"],hours[0]),dtype=cp.float32)
                flux=cp.asarray(frames.at(source.name,ref["field"],hours[0]),dtype=cp.float32)
            emission=flux if emission is None else cp.maximum(emission,flux)
    if power is None:
        return None
    frp_w=cp.asarray(frp,dtype=cp.float32)*np.float32(1.e6)
    if ktau>1 and "fire_coef_carry" in ctx.diag:
        frp_w=frp_w*ctx.diag["fire_coef_carry"][0]
    prep=prepare_columns(ctx)
    return {"frp_inst":frp_w,"kpbl":prep["kpbl"],"uspdavg2d":prep["uspdavg2d"],
            "hpbl2d":prep["hpbl2d"],"ebu_in":cp.asarray(emission,dtype=cp.float32)[None]}


def prepare_columns(ctx):
    """Plume lane calls this before its plume solve; return its column inputs.

    No HWP-only input is required. GSL index outputs remain 1-based.
    """
    from gpuwm.core.kernels import get_kernel
    _prep(ctx,0,get_kernel("chem_fire","chem_fire_prep"),method=0)
    return {n:ctx.diag["fire_"+n] for n in ("kpbl","kpbl_thetav","uspdavg2d","windgustpot","hpbl2d")}


def start_carry(ctx, ktau):
    """GSL's ktau == 1 fire state, on any state whose fire arrays were never written.

    The diurnal coefficient and fire history start at one and the source
    hour cache at -1 (rrfs_smoke_wrapper.F90's first call).  That is the
    first step, and the fresh strip of a relocated nest. Written overlap
    columns keep their history and coefficient. A written hour cache is
    -1 or an hour stamp, never 0. One device read per chem state.
    Returns whether it started the state.
    """
    chem=getattr(getattr(ctx,"state",None),"chem",None)
    started=False
    if ktau==1 or not getattr(chem,"fire_carry_ready",False):
        empty=ctx.diag["fire_cache_hour"]==0
        if ktau==1:
            empty[...] = True
        if bool(empty.any()):
            ctx.diag["fire_hist_carry"][empty]=1
            ctx.diag["fire_coef_carry"][empty]=1
            ctx.diag["fire_cache_hour"][empty]=-1
            ctx.diag["fire_cache_ref"][empty]=-1
            started=True
        if chem is not None:
            chem.fire_carry_ready=True
    return started


def step(ctx, dt, ktau):
    import numpy as np
    from gpuwm.core.kernels import get_kernel
    active=rows(ctx.table)
    if not active:
        return
    if dt <= 0:
        raise ValueError("emission.fire needs positive dt to prevent reversed emission mass")
    from gpuwm.chem_emission_frames import frames_for_context
    frames=frames_for_context(ctx)
    if not hasattr(frames,"reference_time") or not hasattr(frames,"available_hours"):
        raise ValueError("emission.fire needs SourceFrames.reference_time and available_hours; guessing dates would select unrelated fire files")
    mode=getattr(ctx.cfg,"fire_emission_mode",TRAILING_MODE)
    start=frames.reference_time
    valid=start+timedelta(seconds=ctx.clock.curr_secs)
    nz,ny,nx=ctx.met("dryrho").shape
    nc=nx*ny
    block=((nc+127)//128,); threads=(128,)
    data=_data()
    start_carry(ctx,ktau)
    for r,row in enumerate(active):
        refs=tuple(ref for ref in row.emissions
                   if ctx.table.sources[ref["source"]].variables[ref["field"]].get("fire") is not None)
        if not refs:
            raise ValueError("emission.fire row has no source field with fire metadata; guessing an emitter would inject unrelated inventory mass")
        if len(refs)>1:
            raise ValueError(f"emission.fire row {row.name!r} names {len(refs)} fire source fields; they would share the row's one diurnal carry plane and one injection profile, so each source's fire history would overwrite the other's; give each fire source its own row")
        for e,ref in enumerate(refs):
            source=ctx.table.sources[ref["source"]]
            metadata=source.variables[ref["field"]].get("fire")
            if metadata is None:
                raise ValueError(f"source {source.name} field {ref['field']} lacks fire metadata; guessing the FRP field would drive the wrong plume")
            minimum=np.float32(metadata.get("minimum",data["emission_min"]))
            if not np.isfinite(minimum) or minimum<0:
                raise ValueError("fire minimum must be finite and nonnegative; an invalid gate would accept missing or negative emission flux")
            if metadata.get("frp") not in source.variables:
                raise ValueError(f"source field {ref['field']} needs a fire FRP role naming an existing variable; guessing it would drive the wrong plume radiative power")
            factor=metadata.get("state_factors",{"ug kg-1":1}).get(row.units)
            if factor is None or not np.isfinite(factor) or factor<=0:
                raise ValueError(f"fire source field {ref['field']} needs a positive state_factors entry for {row.units}; mass injection without it would corrupt gas mixing-ratio units")
            if mode==TRAILING_MODE:
                # The valid hour's planes from the trailing window's means and
                # the diurnal curve; uploaded and converted once per hour.
                mass,frp,hour=trailing_planes(ctx,frames,source,ref["field"],metadata["frp"],valid)
                hours=(hour,)
                loaded=(TRAILING_MODE,source.name,ref["field"],metadata["frp"],hour,
                        ctx.diag["fire_input_mass"].data.ptr,ctx.diag["fire_flux"].data.ptr)
                if getattr(ctx.state.chem,"fire_inputs_loaded",None)!=loaded:
                    ctx.diag["fire_input_mass"][...]=mass
                    ctx.diag["fire_input_frp"][...]=frp
                    ctx.diag["fire_prev_hwp"].fill(0);ctx.diag["fire_prev_rain"].fill(0);ctx.diag["fire_end"].fill(0)
                    get_kernel("chem_fire","chem_fire_units")(block,threads,
                        (ctx.diag["fire_input_mass"],ctx.met("msftx"),ctx.met("msfty"),ctx.diag["fire_flux"],ctx.diag["fire_area"],
                         np.float32(ctx.cfg.dx),np.float32(ctx.cfg.dy),np.int32(nc)))
                    ctx.state.chem.fire_inputs_loaded=loaded
            else:
                hours=select_frames(mode,start if mode=="daily_mean_dcycle" else valid,
                                    frames.available_hours(source.name,ref["field"]))
            if mode==TRAILING_MODE:
                pass
            elif mode=="daily_mean_dcycle":
                from gpuwm.chem_fire_reduce import daily_inputs
                history_source=metadata.get("history_source")
                if not history_source:
                    raise ValueError("daily_mean_dcycle cold start lacks previous-day model HWP and rain; fabricating them would change fire intensity (SRW-SD cycle.py:240-250)")
                hh=frames.available_hours(history_source,metadata["history_hwp"])
                history_hours=tuple(h for h in hh if start-timedelta(hours=25)<=h<=start-timedelta(hours=2))
                if not history_hours:
                    raise ValueError("daily_mean_dcycle cold start lacks previous-day model HWP; fabricating it would change the HWP ratio")
                planes=lambda s,f,hs: np.stack([frames.at(s,f,h) for h in hs])
                inputs=daily_inputs(planes(source.name,ref["field"],hours),planes(source.name,metadata["frp"],hours),
                    np.array([(start-h).total_seconds()/3600 for h in hours],np.float32),
                    planes(history_source,metadata["history_hwp"],history_hours),
                    planes(history_source,metadata["history_rain"],history_hours))
                for key,host in zip(("fire_input_mass","fire_input_frp","fire_end","fire_prev_hwp","fire_prev_rain"),inputs):
                    ctx.diag[key].set(np.ascontiguousarray(host,dtype=np.float32))
                ctx.state.chem.fire_inputs_loaded=None
            else:
                # The hour's frames are uploaded and converted once, not every
                # step: the inputs, the zeroed history planes and the units
                # kernel's flux and area depend on nothing but the source hour
                # (and the domain's fixed map factors).  The key names the
                # buffers it filled, so a rebound array reads again.
                loaded=(source.name,ref["field"],metadata["frp"],hours[0],
                        ctx.diag["fire_input_mass"].data.ptr,ctx.diag["fire_flux"].data.ptr)
                if getattr(ctx.state.chem,"fire_inputs_loaded",None)!=loaded:
                    for key,field in (("fire_input_mass",ref["field"]),("fire_input_frp",metadata["frp"])):
                        ctx.diag[key].set(np.ascontiguousarray(frames.at(source.name,field,hours[0]),dtype=np.float32))
                    ctx.diag["fire_prev_hwp"].fill(0);ctx.diag["fire_prev_rain"].fill(0);ctx.diag["fire_end"].fill(0)
                    get_kernel("chem_fire","chem_fire_units")(block,threads,
                        (ctx.diag["fire_input_mass"],ctx.met("msftx"),ctx.met("msfty"),ctx.diag["fire_flux"],ctx.diag["fire_area"],
                         np.float32(ctx.cfg.dx),np.float32(ctx.cfg.dy),np.int32(nc)))
                    ctx.state.chem.fire_inputs_loaded=loaded
            if mode=="daily_mean_dcycle":
                get_kernel("chem_fire","chem_fire_units")(block,threads,
                    (ctx.diag["fire_input_mass"],ctx.met("msftx"),ctx.met("msfty"),ctx.diag["fire_flux"],ctx.diag["fire_area"],
                     np.float32(ctx.cfg.dx),np.float32(ctx.cfg.dy),np.int32(nc)))
            if metadata.get("diagnostic",False):
                ctx.diag["fire_ebu_in"][...]=ctx.diag["fire_flux"]
                ctx.diag["fire_frp_mean"][...]=ctx.diag["fire_input_frp"]
            cycle=2 if mode=="daily_mean_dcycle" else 1
            if cycle==2:
                _require(ctx,("vegtype_frac","xlat","xlong","landuse_dataset"),mode)
                if ctx.met("landuse_dataset")!=data["dataset"]:
                    raise ValueError(f"fire classification needs {data['dataset']} fractions; reusing category numbers from another dataset would misclassify fires")
                if ctx.met("vegtype_frac").shape[0]<max(c for g in data["groups"] for c in g["categories"]):
                    raise ValueError("fire land-use fractions lack a category named by the rule table; classification would read beyond the supplied category planes")
                _prep(ctx,int(ctx.clock.curr_secs//3600),get_kernel("chem_fire","chem_fire_prep"))
                get_kernel("chem_fire","chem_fire_classify")(block,threads,
                    (ctx.met("vegtype_frac"),ctx.met("xlat"),ctx.met("xlong"),ctx.diag["fire_flux"],
                     ctx.diag["fire_type"],
                     np.int32(nc),np.int32(len(data["groups"])),np.int32(len(data["rules"])),
                     np.float32(data["longitude_period"]),minimum))
            coef=ctx.diag["fire_coef_carry"][r];hist=ctx.diag["fire_hist_carry"][r]
            stamp=int(hours[-1].timestamp()//3600)
            changed_columns=((ctx.diag["fire_cache_hour"][r]!=stamp)
                              |(ctx.diag["fire_cache_ref"][r]!=e))
            out=ctx.diag["fire_ebu"][r]
            vertical=ref["vertical"]
            if vertical not in ("surface","plumerise"):
                raise ValueError(f"fire vertical {vertical!r} has no injection rule; guessing levels would change emitted column mass")
            if vertical=="plumerise" and "plume_k_min" not in ctx.diag:
                raise ValueError("a plumerise emission needs the plumerise.freitas process on its row; without it the injection bounds are never computed and every fire would be put in the first level")
            # The plume ran THIS step (it runs before emission.fire in
            # CHEM_STEP_ORDER) and left new bounds in its cache.
            plume_ran=(vertical=="plumerise"
                       and getattr(ctx.state.chem,"plume_ran_ktau",None)==ktau)
            if plume_ran:
                plume_columns=getattr(ctx.state.chem,"plume_ran_columns",None)
                if plume_columns is None:
                    changed_columns[...] = True
                else:
                    changed_columns |= plume_columns
            changed=bool(changed_columns.any())
            all_changed=changed and bool(changed_columns.all())
            if not all_changed:
                get_kernel("chem_fire","chem_fire_carry")(block,threads,
                    (out,coef,np.int32(nc),np.int32(nz),np.int32(1)))
            get_kernel("chem_fire","chem_fire_cycle")(block,threads,
                (coef,hist,ctx.diag["fire_hwp"],ctx.diag["fire_prev_hwp"],ctx.met("swdown") if cycle==2 else ctx.diag["fire_prev_rain"],
                 ctx.diag["fire_end"],ctx.diag["fire_type"],
                 np.int32(nc),np.int32(len(data["night_decay"])),np.float32(ctx.clock.curr_secs),
                 np.float32(getattr(ctx.cfg,"sc_factor",1)),np.int32(cycle)))
            ctx.diag["fire_coef_bb_dc"][r]=coef;ctx.diag["fire_hist"][r]=hist
            if metadata.get("diagnostic",False):
                get_kernel("chem_fire","chem_fire_frp_diag")(block,threads,
                    (ctx.diag["fire_input_frp"],coef,ctx.diag["fire_frp_mean"],np.int32(nc)))
            if changed:
                if all_changed:
                    out.fill(0)
                    columns=None
                else:
                    import cupy as cp

                    columns=cp.flatnonzero(changed_columns.ravel()).astype(cp.int32)
                    out.reshape(nz,nc)[:,columns]=0
                if vertical=="surface":
                    if all_changed:
                        out[0]=ctx.diag["fire_flux"]
                    else:
                        out[0,changed_columns]=ctx.diag["fire_flux"][changed_columns]
                else:
                    # GSL ebu_driver's distribution (module_plumerise.F90:
                    # 158-166) over the plume's cached bounds, on the columns
                    # that emit this hour; a column whose fire ended keeps
                    # nothing from the hour before.
                    from gpuwm.core.chem_plumerise import apply_cached
                    apply_cached(ctx,ctx.diag["fire_flux"][None],out[None],columns)
                ctx.diag["fire_cache_hour"][r,changed_columns]=stamp
                ctx.diag["fire_cache_ref"][r,changed_columns]=e
            # ug/kg-dry state requires a dry-air denominator for source mass
            # identity. GSL's supplied rho=p/(Rd*T), wrapper.F90:889, is not
            # WRF-Chem's moist rho. The injection arithmetic itself is unchanged.
            get_kernel("chem_fire","chem_fire_inject")(block,threads,
                (ctx.field(row),out,ctx.met("dryrho"),ctx.met("dryrho"),ctx.met("dz8w"),coef,
                 ctx.diag["fire_emitted_mass"][r],np.int32(nc),np.int32(nz),np.int32(cycle),
                 np.float32(dt),minimum,np.float32(ref["weight"]),np.int32(vertical=="surface"),np.float32(factor)))
            get_kernel("chem_fire","chem_fire_carry")(block,threads,
                (out,coef,np.int32(nc),np.int32(nz),np.int32(0)))
    # Every emission row has consumed the current plume selection. It is
    # transient 2-D device scratch, never scientific or restart state.
    if getattr(ctx.state.chem,"plume_ran_ktau",None)==ktau:
        ctx.state.chem.plume_ran_columns=None
