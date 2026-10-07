"""WRF SFIRE exchange at the first RK atmosphere-physics call.

The refined fire grid owns its level set and combustion clock. Atmospheric
wind is interpolated before spread; completed heat and water fluxes are
averaged in native summation order and inserted as mass-coupled tendencies.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math

import numpy as np

from gpuwm.core.sfire import FireOptions, FireState
from gpuwm.core.sfire_core import CoreOptions, IgnitionLine
from gpuwm.core.sfire_phys import FuelTable
from gpuwm.core.sfire_moisture import MoistureState
from gpuwm.core import sfire_atm, sfire_wind

MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
MODULE_KEY = "gpuwm.core.sfire_coupler:sfire_coupling"


def module_source(kernel_dir=None):
    from gpuwm.core.kernels import module_source as compose
    root = Path(__file__).with_name("kernels") if kernel_dir is None else Path(kernel_dir)
    return compose("sfire_coupling", kernel_dir=root)


@lru_cache(maxsize=None)
def _module(device):
    import cupy as cp
    from cupy.cuda import compiler
    from gpuwm.certify.kernel_manifest import record_module
    from gpuwm.kernel_compile_notice import observe_module_compile
    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(source, MODULE_OPTIONS, None, "sfire_coupling.cu")
        result = cp.cuda.function.Module()
        result.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS, module=None)
    return result


def _launch(name, n, args):
    import cupy as cp
    _module(cp.cuda.Device().id).get_function(name)(((n + 127) // 128,), (128,), args)


def geographic_ignition_units(latitude):
    """Native REAL metre-per-degree units from the WRF fire driver.

    WRF's reciprocal Earth radius is valid. Its REAL constants, expression
    ordering and scalar cosine are retained without double host arithmetic.
    """
    import cupy as cp
    latitude = cp.ascontiguousarray(cp.asarray(latitude, dtype=cp.float32))
    if not bool(cp.all(cp.isfinite(latitude))):
        raise ValueError("Geographic SFIRE ignition latitude must be finite to define coordinate distances")
    out = cp.empty((2,) + latitude.shape, dtype=cp.float32)
    _launch("sfire_geographic_ignition_units", latitude.size,
            (latitude, out, np.int32(latitude.size)))
    return out[0], out[1]


def _pad(value, shape, *, linear=False):
    import cupy as cp
    src = cp.ascontiguousarray(cp.asarray(value, dtype=cp.float32))
    if src.shape != shape:
        raise ValueError(f"SFIRE plane shape {src.shape} differs from {shape}; grids cannot be aligned")
    ny, nx = shape
    out = cp.empty((ny + 2, nx + 2), dtype=cp.float32)
    _launch("sfire_pad_plane", out.size, (src, out, np.int32(nx), np.int32(ny), np.int32(linear)))
    return out


def fire_options(cfg):
    core_names = dict(upwinding="fire_upwinding", upwinding_reinit="fire_upwinding_reinit",
        upwind_split="fire_upwind_split", grows_only="fire_grows_only", advection="fire_advection",
        slope_factor="fire_slope_factor", lsm_band_ngp="fire_lsm_band_ngp", viscosity="fire_viscosity",
        viscosity_bg="fire_viscosity_bg", viscosity_band="fire_viscosity_band",
        viscosity_ngp="fire_viscosity_ngp", lfn_ext_up="fire_lfn_ext_up",
        lsm_reinit_iter="fire_lsm_reinit_iter", boundary_guard="fire_boundary_guard",
        fuel_left_method="fire_fuel_left_method")
    core = CoreOptions(**{name: getattr(cfg, key) for name, key in core_names.items()})
    return FireOptions(core=core, lsm_reinit=cfg.fire_lsm_reinit, const_time=cfg.fire_const_time,
        const_grnhfx=cfg.fire_const_grnhfx, const_grnqfx=cfg.fire_const_grnqfx,
        fire_is_real_perim=cfg.fire_is_real_perim, fire_fmc_read=cfg.fire_fmc_read,
        nfuel_cat0=cfg.fire_fuel_cat, fmoist_run=cfg.fmoist_run, fmoist_interp=cfg.fmoist_interp,
        fmoist_only=cfg.fmoist_only, fmoist_freq=cfg.fmoist_freq, fmoist_dt=cfg.fmoist_dt,
        fmep_decay_tlag=cfg.fmep_decay_tlag)


def ignition_lines(cfg):
    """Expand the five registry ignition slots exactly as the WRF driver."""
    ideal = bool(cfg.fire_ignition_start_x1 or cfg.fire_ignition_start_y1)
    geo = bool(cfg.fire_ignition_start_lon1 or cfg.fire_ignition_start_lat1)
    if ideal and geo:
        raise ValueError("SFIRE ignition coordinates cannot mix metric and geographic axes")
    count = max((i for i in range(1, cfg.fire_num_ignitions + 1)
                 if getattr(cfg, f"fire_ignition_radius{i}") > 0), default=0)
    if cfg.fire_is_real_perim and cfg.fire_num_ignitions > 0:
        count = max(1, count)
    lines = []
    for i in range(1, count + 1):
        ax, ay = ("lon", "lat") if geo else ("x", "y")
        sx, sy = getattr(cfg, f"fire_ignition_start_{ax}{i}"), getattr(cfg, f"fire_ignition_start_{ay}{i}")
        ex, ey = getattr(cfg, f"fire_ignition_end_{ax}{i}"), getattr(cfg, f"fire_ignition_end_{ay}{i}")
        ts, te = getattr(cfg, f"fire_ignition_start_time{i}"), getattr(cfg, f"fire_ignition_end_time{i}")
        lines.append(IgnitionLine(sx, sy, ex or sx, ey or sy, ts, te or ts,
                                 getattr(cfg, f"fire_ignition_radius{i}"), getattr(cfg, f"fire_ignition_ros{i}")))
    return tuple(lines), geo


class FireCoupler:
    """Per-domain fire state, coarse exchange fields and restart identity."""

    def __init__(self, state, cfg, surface, static=None, *, fire_tiles=None):
        from gpuwm.sfire_config import validate_fuel_subcell_geometry
        validate_fuel_subcell_geometry(cfg)
        import cupy as cp
        self.coarse_shape = tuple(state.p.shape[1:])
        self.sr_x, self.sr_y = int(cfg.sr_x), int(cfg.sr_y)
        ny, nx = self.coarse_shape
        self.fine_shape = (ny * self.sr_y, nx * self.sr_x)
        self.geometry = dict(coarse_shape=list(self.coarse_shape), sr_x=self.sr_x, sr_y=self.sr_y)
        static = {str(k).upper(): v for k, v in (static or {}).items()}
        self.spotting = static.get("_SPOTTING_OWNER")
        self._tile_spotting_owner = self.spotting is not None
        if cfg.fs_firebrand_gen_lim > 0:
            from gpuwm.core.sfire_spotting import SpottingState, SpottingOptions
            if self.spotting is None:
                self.spotting = SpottingState(self.coarse_shape,self.sr_x,self.sr_y,SpottingOptions.from_config(cfg))
        from gpuwm.sfire_coordinates import fire_coordinate_mode
        coordinate_mode = fire_coordinate_mode(static)
        bundle = static.get("_METADATA")
        if bundle is not None:
            if (bundle.get("sr_x") != self.sr_x or bundle.get("sr_y") != self.sr_y
                    or tuple(bundle.get("fine_shape", ())) != self.fine_shape):
                raise ValueError("SFIRE static bundle grid/refinement differs from the atmospheric domain")
        table = FuelTable.from_namelist(cfg.fire_fuel_namelist) if cfg.fire_fuel_namelist else FuelTable()
        if (cfg.fmoist_run or cfg.fmoist_interp) and int(table.moisture["moisture_classes"]) > cfg.nfmc:
            raise ValueError("nfmc must include every namelist.fire moisture class; active classes cannot be truncated")
        # Source spacing is REAL atmospheric spacing divided by INTEGER refinement.
        dx, dy = np.float32(cfg.dx) / np.float32(self.sr_x), np.float32(cfg.dy) / np.float32(self.sr_y)
        if bundle is not None and (np.float32(bundle.get("fine_dx_m", 0)) != dx
                                   or np.float32(bundle.get("fine_dy_m", 0)) != dy):
            raise ValueError("SFIRE static spacing differs from the native atmosphere-to-fire refinement")

        def fine(name, default=None):
            value = static.get(name, default)
            if value is None:
                return None
            value = cp.asarray(value, dtype=cp.float32)
            # Native history carries a terminal refinement extension beyond physical cells.
            if value.ndim == 2 and value.shape == (self.fine_shape[0] + self.sr_y, self.fine_shape[1] + self.sr_x):
                value = value[:self.fine_shape[0], :self.fine_shape[1]]
            if value.shape == (self.fine_shape[0] + 2, self.fine_shape[1] + 2):
                return cp.ascontiguousarray(value)
            if value.ndim == 0:
                value = cp.full(self.fine_shape, value, dtype=cp.float32)
            return _pad(value, self.fine_shape)

        zsf = fine("ZSF")
        if zsf is None:
            if not cfg.fire_topo_from_atm:
                raise ValueError("SFIRE fire_topo_from_atm=0 needs fine-grid ZSF from Rust static preparation")
            zsf = self.interpolate(state.ht)
        gx, gy = fine("DZDXF"), fine("DZDYF")
        if gx is None or gy is None:
            gx, gy = cp.empty_like(zsf), cp.empty_like(zsf)
            _launch("sfire_static_gradient", zsf.size,
                    (zsf, gx, gy, np.int32(zsf.shape[1]), np.int32(zsf.shape[0]), dx, dy))
        if cfg.fire_fuel_read not in (-1, 0, 1, 2):
            raise ValueError("fire_fuel_read must select supplied, constant, altitude or ideal-file fuel")
        cat = fine("NFUEL_CAT", cfg.fire_fuel_cat if cfg.fire_fuel_read in (0, 1) else None)
        if cat is None:
            raise ValueError("SFIRE supplied-fuel mode needs NFUEL_CAT from Rust static preparation")
        _launch("sfire_set_nfuel", cat.size, (cat, zsf, np.int32(cat.size),
                                            np.int32(cfg.fire_fuel_read), np.int32(cfg.fire_fuel_cat)))
        lines, geographic = ignition_lines(cfg)
        coord_x, coord_y = fine("FXLONG"), fine("FXLAT")
        unit_x = unit_y = 1.0
        if geographic:
            if coordinate_mode != "geographic":
                raise ValueError("Geographic SFIRE ignitions require geographic fine coordinates")
            if coord_x is None or coord_y is None:
                raise ValueError("Geographic SFIRE ignitions need fine-grid FXLONG and FXLAT")
            latitude = static.get("CEN_LAT", coord_y[self.fine_shape[0] // 2, self.fine_shape[1] // 2])
            ux, uy = geographic_ignition_units(latitude)
            unit_x, unit_y = float(ux.item()), float(uy.item())
        else:
            coord_x = coord_y = None
        self.grid = FireState.from_static(cat, zsf, gx, gy, dx, dy, halo=1, tiles=fire_tiles,
            options=fire_options(cfg), table=table, ignitions=lines, time_start=state.elapsed_seconds,
            coord_xf=coord_x, coord_yf=coord_y, unit_x=unit_x, unit_y=unit_y,
            fmc_g=fine("FMC_G"), lfn_hist=fine("LFN_HIST"), historical_tign=fine("HISTORICAL_TIGN"),
            # Registry fire_sfire allocates these input/history fields even
            # when their evolution and interpolation switches are off.
            moisture_state=MoistureState.zeros(self.coarse_shape, table, class_count=cfg.nfmc,
                                               require_active=cfg.fmoist_run or cfg.fmoist_interp))
        self.geometry["fire_tiles"] = [list(tile) for tile in self.grid.tiles]
        if self.grid.moisture is not None:
            for name, expected in self.grid.moisture.arrays().items():
                native = "FMC_TEND" if name == "fmc_lag" else name.upper()
                if native in static:
                    value = cp.asarray(static[native], dtype=cp.float32)
                    if value.shape != expected.shape:
                        raise ValueError(f"Supplied {native} shape differs from the native moisture grid")
                    setattr(self.grid.moisture, name, cp.ascontiguousarray(value))
            if cfg.fmoist_interp and not cfg.fmoist_run and "FMC_GC" not in static:
                raise ValueError("fmoist_interp without fmoist_run needs supplied FMC_GC class fields")
        self.data = {name: cp.zeros(self.coarse_shape, dtype=cp.float32) for name in
            ("grnhfx", "grnqfx", "grnhfx_fu", "grnqfx_fu", "canhfx", "canqfx", "avg_fuel_frac")}
        self.data.update(uah=cp.zeros((ny, nx + 1), cp.float32), vah=cp.zeros((ny + 1, nx), cp.float32),
                         rthfrten=cp.zeros_like(state.p), rqvfrten=cp.zeros_like(state.p))
        self.data["lfn_time"] = cp.ascontiguousarray(cp.asarray(static.get("LFN_TIME", cp.zeros((1,), cp.float32)), dtype=cp.float32))
        if self.data["lfn_time"].shape != (1,):
            raise ValueError("LFN_TIME must retain the native single i_lfn_history entry")
        if self.spotting is None:
            from gpuwm.core.sfire_spotting import COARSE_REAL_FIELDS, COARSE_INTEGER_FIELDS
            self.data.update({name:cp.zeros(self.coarse_shape,cp.float32) for name in COARSE_REAL_FIELDS})
            self.data.update({name:cp.zeros(self.coarse_shape,cp.int32) for name in COARSE_INTEGER_FIELDS})
        self.data["fz0"] = fine("FZ0")
        if self.data["fz0"] is None:
            self.data["fz0"] = self.interpolate(surface["znt"], floor=0.001)
        supplied_lat, supplied_lon = fine("FXLAT"), fine("FXLONG")
        if (supplied_lat is None) != (supplied_lon is None):
            raise ValueError("SFIRE fine geography must contain both FXLAT and FXLONG")
        self.data["fxlat"] = self.grid.data["coord_yf"] if supplied_lat is None else supplied_lat
        self.data["fxlong"] = self.grid.data["coord_xf"] if supplied_lon is None else supplied_lon
        # Metric ignition coordinates and geographic output coordinates
        # are independent. Do not discard real static geography for a line
        # specified in meters or an observed-perimeter initialization.
        self.geometry["coordinate_mode"] = coordinate_mode
        statics = {name: hashlib.sha256(cp.asnumpy(self.grid.data[name]).tobytes()).hexdigest()
                   for name in ("nfuel_cat", "zsf", "dzdxf", "dzdyf", "coord_xf", "coord_yf", "fmc_g")}
        statics["fz0"] = hashlib.sha256(cp.asnumpy(self.data["fz0"]).tobytes()).hexdigest()
        statics["lfn_time"] = hashlib.sha256(cp.asnumpy(self.data["lfn_time"]).tobytes()).hexdigest()
        for name in ("fxlat", "fxlong"):
            statics[name] = hashlib.sha256(cp.asnumpy(self.data[name]).tobytes()).hexdigest()
        for name in ("lfn_hist", "historical_tign"):
            if name in self.grid.data:
                statics[name] = hashlib.sha256(cp.asnumpy(self.grid.data[name]).tobytes()).hexdigest()
        if self.grid.moisture is not None:
            for name, value in self.grid.moisture.arrays().items():
                native = "FMC_TEND" if name == "fmc_lag" else name.upper()
                if native in static:
                    statics[native] = hashlib.sha256(cp.asnumpy(value).tobytes()).hexdigest()
        self._setup = dict(geometry=self.geometry, dx=float(dx), dy=float(dy),
            options=asdict(self.grid.options), table=self.grid.metadata()["table"],
            ignitions=[asdict(line) for line in lines], static_sha256=statics,
            exchange={name: getattr(cfg, name) for name in ("fire_wind_height", "fire_lsm_zcoupling",
                "fire_lsm_zcoupling_ref", "fire_atm_feedback", "fire_ext_grnd", "fire_ext_crwn",
                "fire_crwn_hgt", "fire_sfc_flx", "fire_heat_peak", "fire_tg_ub")})
        if cfg.fmoist_run and cfg.sf_sfclay_physics == 0 and cfg.sf_surface_physics == 0:
            self._setup["moisture_surface_source"] = "no-exchange near-ground atmospheric proxy"
        if bundle is not None:
            self._setup["static_grid_spec"] = bundle["grid_spec"]
        if self.spotting is not None:
            self._setup["spotting"] = self.spotting.metadata()["options"]

    def interpolate(self, field, *, floor=None):
        """Native mass-center interpolation with linear one-cell continuation."""
        coarse = _pad(field, self.coarse_shape, linear=True)
        if hasattr(self, "_tile_atmos_domain"):
            from gpuwm.core.sfire_core import continue_at_boundary
            continue_at_boundary(coarse, bias=0.0, domain=self._tile_atmos_domain)
        result = sfire_atm.interpolate_2d(coarse,
            (self.fine_shape[0] + 2, self.fine_shape[1] + 2), self.sr_x, self.sr_y,
            coarse_origin=(1.0, 1.0),
            fine_origin=(1.0 + (self.sr_x - 1) * 0.5, 1.0 + (self.sr_y - 1) * 0.5))
        if floor is not None:
            _launch("sfire_plane_floor", result.size, (result, np.int32(result.size), np.float32(floor)))
        return result

    def advance(self, state, cfg, atmosphere, time_seconds, dt, surface):
        if cfg.fire_test_steps > 0:
            from gpuwm.sfire_debug import run_test_steps
            run_test_steps(self, state, cfg, atmosphere, time_seconds, dt, surface)
        return self._advance_once(state, cfg, atmosphere, time_seconds, dt, surface)

    def _advance_once(self, state, cfg, atmosphere, time_seconds, dt, surface, *, feedback=True, report=True):
        import cupy as cp
        def classes(values):
            return cp.stack([self.interpolate(value) for value in values])
        if self.grid.options.fmoist_only:
            self.grid.advance(dt, time_start=time_seconds, surface=surface, interpolate_moisture=classes)
            if report:
                self._report_step(cfg)
            return self.data["rthfrten"], self.data["rqvfrten"]
        wind = sfire_wind.interpolate_native_atm2fire(state.u, state.v, state.php, state.phb,
            surface["znt"], state.ht, self.data["fz0"], self.sr_x, self.sr_y,
            fine_domain=self.grid.domain, return_staggered_diagnostics=True,
            coarse_domain=getattr(self, "_tile_atmos_domain", None),
            fire_wind_height=cfg.fire_wind_height, fire_lsm_zcoupling=cfg.fire_lsm_zcoupling,
            fire_lsm_zcoupling_ref=cfg.fire_lsm_zcoupling_ref)
        self.data.update(uah=wind["uah"], vah=wind["vah"])
        self.grid.advance(dt, vx=wind["uf"], vy=wind["vf"], time_start=time_seconds,
                          surface=surface, interpolate_moisture=classes)
        d, g = self.data, self.grid.data
        # A tile may have a smaller physical fire domain inside its atmosphere
        # halo. Its exchange planes still have the complete compute-window
        # shape; only owned physical columns are scattered into the domain.
        interior = ((slice(1, -1), slice(1, -1)) if hasattr(self, "_tile_spec")
                    else self.grid.interior)
        sums = [sfire_atm.sum_fire_cells(g[name][interior], self.sr_x, self.sr_y)
                for name in ("fgrnhfx", "fgrnqfx", "fuel_frac")]
        _launch("sfire_feedback", d["grnhfx"].size,
            (*sums, d["grnhfx"], d["grnqfx"], d["grnhfx_fu"], d["grnqfx_fu"], d["avg_fuel_frac"],
             np.int32(d["grnhfx"].size), np.int32(self.sr_x * self.sr_y), np.float32(cfg.fire_atm_feedback)))
        ny, nx = self.coarse_shape
        # fire_driver_wrf passes ide-1/jde-1; the exchange excludes both boundary rows.
        if feedback:
            d["rthfrten"], d["rqvfrten"] = sfire_atm.fire_tendency(d["grnhfx"], d["grnqfx"],
                d["canhfx"], d["canqfx"], terrain=state.ht, z_at_w=atmosphere["z_interface"],
                dz8w=atmosphere["dz"], mu=state.total_mu(), c1h=state.c1h, c2h=state.c2h,
                rho=atmosphere["rho"], fire_ext_grnd=cfg.fire_ext_grnd,
                fire_ext_crwn=cfg.fire_ext_crwn, crown_height=cfg.fire_crwn_hgt,
                fire_sfc_flx=cfg.fire_sfc_flx, fire_heat_peak=cfg.fire_heat_peak,
                fire_tg_ub=cfg.fire_tg_ub,
                domain=getattr(self, "_tile_exchange_domain", (1, nx - 2, 1, ny - 2)))
        if report:
            self._report_step(cfg)
        return d["rthfrten"], d["rqvfrten"]

    def _report_step(self, cfg):
        if not hasattr(self, "_tile_spec") and (cfg.fire_print_msg or cfg.fire_print_file):
            from gpuwm.sfire_debug import report_step
            report_step(self, cfg)

    def setup_identity(self):
        return json.loads(json.dumps(self._setup))

    def output_attributes(self):
        attrs = dict(FIRE_COORDINATE_MODE=self.geometry["coordinate_mode"],
                     IFIRE=2, SR_X=self.sr_x, SR_Y=self.sr_y)
        if "moisture_surface_source" in self._setup:
            attrs["FIRE_MOISTURE_SURFACE_SOURCE"] = self._setup["moisture_surface_source"]
        return attrs

    def arrays(self):
        result = {**{f"grid.{name}": value for name, value in self.grid.arrays().items()}, **self.data}
        if self.spotting is not None:
            result.update({f"spotting.{name}":value for name,value in self.spotting.arrays().items()})
        return result

    def metadata(self):
        result = dict(version=1, geometry=self.geometry, grid=self.grid.metadata())
        if self.spotting is not None:
            result["spotting"] = self.spotting.metadata()
        return result

    def validate_restart(self, arrays, metadata):
        if metadata.get("version") != 1 or metadata.get("geometry") != self.geometry:
            raise ValueError("SFIRE restart geometry or version differs from the attached fire grid")
        if set(arrays) != set(self.arrays()):
            raise ValueError("SFIRE restart must contain every persisted fine and coarse field")
        for name, expected in self.arrays().items():
            value = arrays[name]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"SFIRE restart field {name} has incompatible shape or dtype")
        g = metadata["grid"]
        initial = self.grid.metadata()
        for name in ("version", "dx", "dy", "domain", "tiles", "options", "table", "ignitions", "unit_x", "unit_y"):
            if g.get(name) != initial[name]:
                raise ValueError(f"SFIRE restart {name} differs from immutable fire setup")
        for name in ("time_seconds", "moisture_lasttime", "moisture_nexttime", "last_cfl_bound"):
            if isinstance(g.get(name), bool) or not isinstance(g.get(name), (int, float)) or not math.isfinite(g[name]):
                raise ValueError(f"SFIRE restart {name} must be a finite continuation clock")
        if not isinstance(g.get("step_count"), int) or isinstance(g["step_count"], bool) or g["step_count"] < 0:
            raise ValueError("SFIRE restart step_count must be a nonnegative integer")
        for name in ("moisture_initialized", "perimeter_applied", "last_cfl_exceeded"):
            if not isinstance(g.get(name), bool):
                raise ValueError(f"SFIRE restart {name} must be boolean")
        if (g["moisture_lasttime"] > g["time_seconds"] or
                (self.grid.options.fmoist_freq == 0 and g["moisture_nexttime"] < g["moisture_lasttime"])):
            raise ValueError("SFIRE restart moisture clocks cannot exceed or reverse the fire clock")
        if not isinstance(g.get("last_ignited_counts"), list) or any(
                not isinstance(v, int) or v < 0 for v in g["last_ignited_counts"]):
            raise ValueError("SFIRE restart ignition counts must be nonnegative integers")
        if self.spotting is None:
            if metadata.get("spotting") is not None:
                raise ValueError("SFIRE restart carries firebrands but generation is disabled")
        else:
            self.spotting.validate_restart({name[9:]:value for name,value in arrays.items()
                if name.startswith("spotting.")},metadata["spotting"])

    def restore(self, arrays, metadata):
        self.validate_restart(arrays, metadata)
        self.grid = FireState.from_restart({name[5:]: value for name, value in arrays.items()
                                           if name.startswith("grid.")}, metadata["grid"])
        self.data = {name: value for name, value in arrays.items()
                     if not name.startswith(("grid.","spotting."))}
        if self.spotting is not None:
            self.spotting.restore({name[9:]:value for name,value in arrays.items()
                if name.startswith("spotting.")},metadata["spotting"])

    def output_fields(self):
        from gpuwm.io.sfire_schema import SFIRE_GRID_FIELD_MAP, SFIRE_COARSE_FIELD_NAMES, SFIRE_MOISTURE_FIELD_MAP
        fine_names = dict(SFIRE_GRID_FIELD_MAP)
        if "lfn_hist" in self.grid.data:
            fine_names["LFN_HIST"] = "lfn_hist"
        result = {name: self.grid.data[key][self.grid.interior] for name, key in fine_names.items()}
        result.update({name: self.data[name.lower()] for name in SFIRE_COARSE_FIELD_NAMES})
        result["LFN_TIME"] = self.data["lfn_time"]
        result.update({name.upper(): self.data[name][self.grid.interior] for name in ("fz0", "fxlat", "fxlong")})
        if self.grid.moisture is not None:
            result.update({name: getattr(self.grid.moisture, key) for name, key in SFIRE_MOISTURE_FIELD_MAP.items()})
            import cupy as cp
            result["FMOIST_LASTTIME"] = cp.asarray(np.float32(self.grid.moisture_lasttime))
            result["FMOIST_NEXTTIME"] = cp.asarray(np.float32(self.grid.moisture_nexttime))
        if self.spotting is not None:
            result.update(self.spotting.output_fields())
        else:
            from gpuwm.core.sfire_spotting import COARSE_REAL_FIELDS, COARSE_INTEGER_FIELDS
            result.update({name.upper():self.data[name] for name in (*COARSE_REAL_FIELDS,*COARSE_INTEGER_FIELDS)})
        return result
