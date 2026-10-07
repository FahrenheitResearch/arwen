"""Persistent GPU SFIRE state and WRF fire-model pass orchestration.

``advance`` follows ``module_fr_fire_model.F`` passes 3 through 6: supply
winds and moisture, propagate, update ignition times, diagnose flame length,
reinitialize, copy the level set, apply prescribed ignitions, then consume
fuel and release sensible and latent heat. Static preprocessing and exchange
with the atmosphere belong to their separate Rust and CUDA drivers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Mapping

import numpy as np

from gpuwm.core import sfire_core as core
from gpuwm.core.sfire_phys import FuelTable, set_fire_params, heat_fluxes, weighted_moisture
from gpuwm.core.sfire_moisture import MoistureState, advance_moisture

STAGE_FIELDS = ("lfn_0","lfn_1","lfn_2","lfn_s0","lfn_s1","lfn_s2","lfn_s3")
DIAGNOSTIC_FIELDS = ("lfn_out","ros","ros_front","flame_length","flineint",
                     "burnt_area_dt","fgrnhfx","fgrnqfx","fcanhfx","fcanqfx",
                     "dry_burnt_mass")
STATIC_FIELDS = ("nfuel_cat","zsf","dzdxf","dzdyf","coord_xf","coord_yf")
PARAM_FIELDS = ("fgip","ischap","betafl","bbb","fuel_time","phiwc","r_0","iboros","fmc_g")


def registry_core_options():
    """User defaults from ``Registry/registry.fire`` rather than util DATA."""
    return core.CoreOptions(advection=1,viscosity_ngp=2,boundary_guard=8)


@dataclass(frozen=True)
class FireOptions:
    """Options used by the native fire driver, plus optional subcycling."""

    core: core.CoreOptions = field(default_factory=registry_core_options)
    lsm_reinit: bool = True
    const_time: float = -1.0
    const_grnhfx: float = 0.0
    const_grnqfx: float = 0.0
    fire_is_real_perim: bool = False
    fire_fmc_read: int = 1
    nfuel_cat0: int = 1
    fmoist_run: bool = False
    fmoist_interp: bool = False
    fmoist_only: bool = False
    fmoist_freq: int = 0
    fmoist_dt: float = 600.0
    fmep_decay_tlag: float = 999999.0
    max_fire_dt: float | None = None

    def __post_init__(self):
        if self.fmoist_run and self.fire_fmc_read != 0:
            raise ValueError("fmoist_run requires fire_fmc_read=0 so modeled moisture enters fuel parameters")
        if self.fire_fmc_read not in (0,1,2):
            raise ValueError("fire_fmc_read must select provided, prescribed or ideal-file moisture")
        if self.fmoist_freq < 0 or self.fmoist_dt <= 0:
            raise ValueError("SFIRE moisture timing needs nonnegative frequency and positive interval")
        if self.max_fire_dt is not None and (not math.isfinite(self.max_fire_dt) or self.max_fire_dt <= 0):
            raise ValueError("SFIRE substep duration must be finite and positive")


@dataclass
class FireState:
    """Every evolving and derived field needed to continue a fire exactly."""

    data: dict
    dx: float
    dy: float
    domain: core.Bounds
    tiles: tuple[core.Bounds,...]
    options: FireOptions
    table: FuelTable
    ignitions: tuple[core.IgnitionLine,...] = ()
    time_seconds: float = 0.0
    step_count: int = 0
    unit_x: float = 1.0
    unit_y: float = 1.0
    moisture: MoistureState | None = None
    moisture_initialized: bool = False
    moisture_lasttime: float = 0.0
    moisture_nexttime: float = 0.0
    perimeter_applied: bool = False
    last_cfl_bound: float = 0.0
    last_cfl_exceeded: bool = False
    last_ignited_counts: tuple[int,...] = ()

    @classmethod
    def from_static(cls, nfuel_cat, zsf, dzdxf, dzdyf, dx, dy, *, halo=3,
                    domain=None, tiles=None, coord_xf=None, coord_yf=None,
                    unit_x=1.0, unit_y=1.0, time_start=0.0, options=None,
                    table=None, ignitions=(), fmc_g=None, lfn_hist=None,
                    historical_tign=None, moisture_shape=None,
                    moisture_state=None):
        """Initialize passes 1 and 2 using preprocessed fine-grid static data.

        Inputs include their physical halos. Coordinates default to WRF's
        ideal metric coordinates. Geographic coordinates require their
        meter-per-coordinate-unit factors from the domain geometry.
        """
        cp = core._cupy()
        if not all(math.isfinite(float(v)) and v>0 for v in (dx,dy,unit_x,unit_y)):
            raise ValueError("SFIRE grid spacing and coordinate units must be finite and positive")
        opt = options or FireOptions()
        table = table or FuelTable()
        cat = core._field(nfuel_cat)
        b = core._bounds(cat.shape,halo,domain)
        ts = core._tiles(b,tiles)
        data = core.init_no_fire(cat.shape,dx,dy,time_start,domain=b)
        data.update(nfuel_cat=cat,zsf=core._field(zsf,cat.shape),
                    dzdxf=core._field(dzdxf,cat.shape),dzdyf=core._field(dzdyf,cat.shape))
        core.continue_at_boundary(data["zsf"],bias=0.0,domain=b)
        if (coord_xf is None) != (coord_yf is None):
            raise ValueError("SFIRE ignition coordinates require both coordinate fields")
        if coord_xf is None:
            x,y = cp.empty_like(cat),cp.empty_like(cat)
            core._launch("sfire_ideal_coords",cat.size,(x,y,*core._ints(cat.shape[1],cat.shape[0],b[0],b[2]),
                         *core._floats(dx,dy)))
        else:
            x,y = core._field(coord_xf,cat.shape),core._field(coord_yf,cat.shape)
        data.update(coord_xf=x,coord_yf=y)
        data.update({name:cp.zeros_like(cat) for name in (*STAGE_FIELDS,*DIAGNOSTIC_FIELDS,"vx","vy")})
        data.update(set_fire_params(cat,fmc_g,table,fire_fmc_read=opt.fire_fmc_read,nfuel_cat0=opt.nfuel_cat0))
        # Registry fire_sfire always retains the input/history plane. Only
        # the explicit perimeter option enables its assimilation.
        data["lfn_hist"] = cp.zeros_like(cat) if lfn_hist is None else core._field(lfn_hist,cat.shape)
        if historical_tign is not None:
            data["historical_tign"] = core._field(historical_tign,cat.shape)
        if opt.fire_is_real_perim and lfn_hist is None:
            raise ValueError("Observed-perimeter ignition requires a fine-grid signed-distance field")
        if moisture_state is None and (opt.fmoist_run or opt.fmoist_interp):
            moisture_state = MoistureState.zeros(moisture_shape or cat.shape,table)
        lines = tuple(line if isinstance(line,core.IgnitionLine) else core.IgnitionLine(**line) for line in ignitions)
        if opt.fire_is_real_perim and not lines:
            raise ValueError("Observed-perimeter ignition requires an ignition start time")
        return cls(data,float(dx),float(dy),b,ts,opt,table,lines,float(time_start),
                   unit_x=float(unit_x),unit_y=float(unit_y),moisture=moisture_state,
                   moisture_lasttime=float(time_start),moisture_nexttime=float(time_start))

    create = from_static

    @property
    def shape(self):
        return self.data["lfn"].shape

    @property
    def interior(self):
        a,b,c,d = self.domain
        return (slice(c,d+1),slice(a,b+1))

    def arrays(self):
        """Restart-ready named device arrays, including every RK stage."""
        result = dict(self.data)
        if self.moisture is not None:
            result.update({f"moisture.{key}":value for key,value in self.moisture.arrays().items()})
        return result

    def metadata(self):
        """JSON-compatible clocks and controls required with the state arrays."""
        return dict(version=1,dx=self.dx,dy=self.dy,domain=list(self.domain),
                    tiles=[list(t) for t in self.tiles],options=asdict(self.options),
                    table=dict(scalars=self.table.scalars,categories=self.table.categories,
                               moisture=self.table.moisture),
                    ignitions=[asdict(line) for line in self.ignitions],
                    time_seconds=self.time_seconds,step_count=self.step_count,
                    unit_x=self.unit_x,unit_y=self.unit_y,
                    moisture_initialized=self.moisture_initialized,
                    moisture_lasttime=self.moisture_lasttime,moisture_nexttime=self.moisture_nexttime,
                    perimeter_applied=self.perimeter_applied,last_cfl_bound=self.last_cfl_bound,
                    last_cfl_exceeded=self.last_cfl_exceeded,
                    last_ignited_counts=list(self.last_ignited_counts))

    @classmethod
    def from_restart(cls, arrays: Mapping, metadata: Mapping):
        """Restore buffers without recomputing any fuel or fire state word."""
        cp = core._cupy()
        if metadata.get("version") != 1:
            raise ValueError("Unsupported SFIRE restart version would misinterpret persisted fire fields")
        restored = {}
        for key,value in arrays.items():
            a = cp.asarray(value)
            if a.dtype != cp.float32:
                raise ValueError(f"SFIRE restart field {key} must be float32 to preserve its output words")
            restored[key] = cp.ascontiguousarray(a)
        keys = (*STATIC_FIELDS,*PARAM_FIELDS,*STAGE_FIELDS,*DIAGNOSTIC_FIELDS,
                "lfn","tign","fuel_frac","fire_area","vx","vy")
        missing = set(keys)-set(restored)
        if missing:
            raise ValueError(f"SFIRE restart is missing continuation fields: {sorted(missing)}")
        data = {key:value for key,value in restored.items() if not key.startswith("moisture.")}
        shape = data["lfn"].shape
        if any(a.shape != shape for a in data.values()):
            raise ValueError("SFIRE restart fire fields must share the same fine-grid shape")
        opts = dict(metadata["options"])
        opts["core"] = core.CoreOptions(**opts["core"])
        table = FuelTable(**metadata["table"])
        table.validate()
        moisture_arrays = {key.split(".",1)[1]:value for key,value in restored.items() if key.startswith("moisture.")}
        moisture = MoistureState(**moisture_arrays) if moisture_arrays else None
        b = core._bounds(shape,domain=metadata["domain"])
        ts = core._tiles(b,metadata["tiles"])
        result = cls(data,metadata["dx"],metadata["dy"],b,ts,FireOptions(**opts),table,
                     tuple(core.IgnitionLine(**v) for v in metadata["ignitions"]),
                     moisture=moisture)
        for name in ("time_seconds","step_count","unit_x","unit_y","moisture_initialized",
                     "moisture_lasttime","moisture_nexttime","perimeter_applied",
                     "last_cfl_bound","last_cfl_exceeded"):
            setattr(result,name,metadata[name])
        result.last_ignited_counts = tuple(metadata["last_ignited_counts"])
        return result

    def _update_moisture(self, surface, interpolate_moisture):
        opt = self.options
        if not (opt.fmoist_run or opt.fmoist_interp):
            return
        initialize = not self.moisture_initialized
        moisture_time = np.float32(self.time_seconds)
        due = (opt.fmoist_freq>0 and (self.step_count+1)%opt.fmoist_freq==0) or (
            opt.fmoist_freq==0 and not moisture_time<np.float32(self.moisture_nexttime))
        if opt.fmoist_run and due:
            if surface is None:
                raise ValueError("Fuel-moisture evolution requires RAINC, RAINNC, T2, Q2 and PSFC surface fields")
            # Initialize on the first actual scheduled advance. A positive
            # frequency can defer that call beyond atmosphere step one.
            dt = np.float32(moisture_time-np.float32(self.moisture_lasttime))
            advance_moisture(self.moisture,dt,*[surface[key] for key in ("rainc","rainnc","t2","q2","psfc")],
                    initialize=initialize,table=self.table,fmep_decay_tlag=opt.fmep_decay_tlag)
            self.moisture_lasttime = float(moisture_time)
            if opt.fmoist_freq == 0:
                self.moisture_nexttime = float(np.float32(moisture_time+np.float32(opt.fmoist_dt)))
            self.moisture_initialized = True
        refresh = not opt.fmoist_only and opt.fmoist_interp and (
            (opt.fmoist_run and due) or (not opt.fmoist_run and self.step_count==0))
        if refresh:
            classes = self.moisture.fmc_gc
            if interpolate_moisture is not None:
                classes = interpolate_moisture(classes)
            if classes.shape[1:] != self.shape:
                raise ValueError("Coarse moisture classes require atmosphere-to-fire interpolation before fuel weighting")
            self.data["fmc_g"] = weighted_moisture(classes,self.data["nfuel_cat"],self.table,nfuel_cat0=opt.nfuel_cat0)
            params = set_fire_params(self.data["nfuel_cat"],self.data["fmc_g"],self.table,
                    fire_fmc_read=0,nfuel_cat0=opt.nfuel_cat0)
            self.data.update(params)

    def frozen(self, time_start=None):
        """Freeze after the configured time since first ignition.

        This corrects the native model's inverted ``time_start<const_time``
        test and its absolute rather than ignition-relative threshold.
        Fuel continues burning after spread freezes.
        """
        ts = self.time_seconds if time_start is None else float(time_start)
        ignition = min((line.start_time for line in self.ignitions),default=0.0)
        return self.options.const_time>0 and ts>=ignition+self.options.const_time

    def _apply_ignitions(self, ts, dt):
        cp = core._cupy()
        d,opt = self.data,self.options
        dated_perimeter = False
        if opt.fire_is_real_perim:
            start = self.ignitions[0].start_time
            if not self.perimeter_applied and ts>=start and ts<start+dt:
                count = cp.zeros(1,dtype=cp.int32)
                history_tign = d.get("historical_tign",d["tign"])
                for tile in self.tiles:
                    core._launch("sfire_assimilate_perimeter",core._size(tile),
                        (d["lfn"],d["tign"],d["lfn_hist"],history_tign,
                         np.int32("historical_tign" in d),count,*core._ints(self.shape[1],*tile),np.float32(ts)))
                self.last_ignited_counts = (int(count.item()),)
                if hasattr(self, "owned_domain"):
                    owned_count = cp.zeros(1, dtype=cp.int32)
                    core._launch("sfire_assimilate_perimeter", core._size(self.owned_domain),
                        (d["lfn"], d["tign"], d["lfn_hist"], history_tign,
                         np.int32("historical_tign" in d), owned_count,
                         *core._ints(self.shape[1], *self.owned_domain), np.float32(ts)))
                    self.last_ignited_counts = (int(owned_count.item()),)
                self.perimeter_applied = True
                dated_perimeter = "historical_tign" in d
            else:
                self.last_ignited_counts = (0,)
        else:
            self.last_ignited_counts = tuple(core.ignite_fire(d["lfn"],d["tign"],d["coord_xf"],d["coord_yf"],
                    line,ts,np.float32(ts)+np.float32(dt),unit_x=self.unit_x,unit_y=self.unit_y,
                    domain=self.domain,tiles=self.tiles) for line in self.ignitions)
            if hasattr(self, "owned_domain"):
                # Reapplying a prescribed ignition is idempotent. The same
                # native routine counts only disjoint owned nodes, so halo
                # overlaps do not inflate the domain diagnostic.
                self.last_ignited_counts = tuple(core.ignite_fire(
                    d["lfn"], d["tign"], d["coord_xf"], d["coord_yf"], line,
                    ts, np.float32(ts)+np.float32(dt), unit_x=self.unit_x, unit_y=self.unit_y,
                    domain=self.owned_domain) for line in self.ignitions)
        return dated_perimeter

    def _step(self, dt, exchange):
        cp = core._cupy()
        d,opt,ts = self.data,self.options,np.float32(self.time_seconds)
        frozen = self.frozen(ts)
        if not frozen:
            out,ros,bound = core.prop_ls_rk3(d["lfn"],d,self.dx,self.dy,ts,dt,options=opt.core,
                    domain=self.domain,tiles=self.tiles,exchange=exchange,stages=d,
                    reduction_domain=getattr(self, "owned_domain", None))
            core.tign_update(d["lfn"],out,d["tign"],ts,dt,boundary_guard=opt.core.boundary_guard,
                    domain=self.domain,tiles=self.tiles,
                    guard_domain=getattr(self, "global_guard_domain", None))
            d["ros"] = ros
            d["flame_length"],d["ros_front"],d["flineint"] = core.calc_flame_length(
                    ros,d["iboros"],d["fire_area"],domain=self.domain,tiles=self.tiles)
            if opt.lsm_reinit:
                out = core.reinit_ls_rk3(d["lfn"],out,self.dx,self.dy,options=opt.core,
                        domain=self.domain,tiles=self.tiles,exchange=exchange,stages=d)
            cp.copyto(d["lfn_out"],out)
            cp.copyto(d["lfn"],out)
            self.last_cfl_bound = float(bound.item())
            self.last_cfl_exceeded = bool(np.float32(dt)>np.float32(self.last_cfl_bound))
        dated_perimeter = self._apply_ignitions(ts,dt)
        if exchange is not None:
            exchange(d["lfn"])
            exchange(d["tign"])
        core.continue_at_boundary(d["lfn"],bias=opt.core.lfn_ext_up,domain=self.domain)
        core.continue_at_boundary(d["tign"],bias=0.0,domain=self.domain)
        if dated_perimeter:
            # A dated initial perimeter contains fuel already consumed before
            # this forecast. Release only the loss during the current interval.
            d["fuel_frac"],_ = core.fuel_left(d["lfn"],d["tign"],d["fuel_time"],ts,
                    domain=self.domain,tiles=self.tiles,method=opt.core.fuel_left_method)
        fuel,area = core.fuel_left(d["lfn"],d["tign"],d["fuel_time"],np.float32(ts)+np.float32(dt),
                domain=self.domain,tiles=self.tiles,method=opt.core.fuel_left_method)
        d["fire_area"] = area
        d["burnt_area_dt"].fill(0)
        for tile in self.tiles:
            core._launch("sfire_driver_burn",core._size(tile),(d["fuel_frac"],fuel,d["burnt_area_dt"],
                        *core._ints(self.shape[1],*tile)))
        d["fgrnhfx"],d["fgrnqfx"],d["dry_burnt_mass"] = heat_fluxes(dt,d["fgip"],d["burnt_area_dt"],d["fmc_g"],
                self.table,return_dry_mass=True)
        if frozen and opt.const_grnhfx>=0 and opt.const_grnqfx>=0:
            d["fgrnhfx"][self.interior] = np.float32(opt.const_grnhfx)
            d["fgrnqfx"][self.interior] = np.float32(opt.const_grnqfx)
        self.time_seconds += float(dt)

    def advance(self, dt, vx=None, vy=None, *, time_start=None, fmc_g=None,
                surface=None, interpolate_moisture=None, exchange=None):
        """Advance one atmosphere interval and return named fire-grid outputs.

        With ``max_fire_dt`` set, native fire steps divide that interval and
        the returned heat fluxes are weighted by substep duration. Burnt mass
        and fuel fractions sum across substeps. Subcycling is an explicit
        driver extension; the WRF fire model itself runs one native step.
        """
        cp = core._cupy()
        if not math.isfinite(float(dt)) or dt<=0:
            raise ValueError("SFIRE advance duration must be finite and positive to define release rates")
        if time_start is not None and np.float32(time_start)!=np.float32(self.time_seconds):
            raise ValueError("SFIRE start time must match its restart clock to preserve ignition and fuel ages")
        if (vx is None)!=(vy is None):
            raise ValueError("SFIRE wind update requires both horizontal components")
        if vx is not None:
            self.data["vx"],self.data["vy"] = core._field(vx,self.shape),core._field(vy,self.shape)
        if fmc_g is not None:
            self.data.update(set_fire_params(self.data["nfuel_cat"],fmc_g,self.table,
                             fire_fmc_read=0,nfuel_cat0=self.options.nfuel_cat0))
        self._update_moisture(surface,interpolate_moisture)
        target = self.time_seconds+float(dt)
        if self.options.fmoist_only:
            self.time_seconds = target
            self.step_count += 1
            return self.outputs()
        count = 1 if self.options.max_fire_dt is None else int(math.ceil(float(dt)/self.options.max_fire_dt))
        if count == 1:
            self._step(np.float32(dt),exchange)
        else:
            sums = [cp.zeros_like(self.data["lfn"]) for _ in range(4)]
            sub_dt = float(dt)/count
            for k in range(count):
                length = np.float32(target-self.time_seconds if k==count-1 else sub_dt)
                self._step(length,exchange)
                d = self.data
                core._launch("sfire_flux_accumulate",d["lfn"].size,
                    (d["fgrnhfx"],d["fgrnqfx"],d["dry_burnt_mass"],d["burnt_area_dt"],
                     *sums,np.int32(d["lfn"].size),length))
            core._launch("sfire_flux_average",sums[0].size,(sums[0],sums[1],np.int32(sums[0].size),np.float32(dt)))
            self.data.update(fgrnhfx=sums[0],fgrnqfx=sums[1],dry_burnt_mass=sums[2],burnt_area_dt=sums[3])
        # Atmosphere time is authoritative, including an adaptive outer step.
        self.time_seconds = target
        self.step_count += 1
        return self.outputs()

    def outputs(self):
        """All native fire outputs; the perimeter is the zero level contour."""
        d = self.data
        names = ("lfn","tign","fuel_frac","fire_area",*DIAGNOSTIC_FIELDS)
        return {name:d[name] for name in names}


__all__ = ["FireOptions","FireState","registry_core_options","STAGE_FIELDS",
           "DIAGNOSTIC_FIELDS","STATIC_FIELDS","PARAM_FIELDS"]
