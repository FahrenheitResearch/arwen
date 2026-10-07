"""WRF 4.7.1 SFIRE mathematical core on the CUDA device.

The pinned authority is ``phys/module_fr_fire_core.F`` in WRF v4.7.1.
All field arithmetic runs in CUDA float32 kernels. Python validates geometry,
allocates device buffers and orders the Runge-Kutta stage barriers. Arrays use
C order ``(ny, nx)``; domain and tile bounds are inclusive ``(xlo,xhi,ylo,yhi)``
indices into those arrays. A physical domain needs one surrounding halo row.
WENO uses up to three neighboring points, but WRF uses ENO1 within ten points
of physical boundaries, so one physical halo is sufficient.

Corrected source defects are explicit: missing normal components for schemes
0 and 4; the negative-x split-advection derivative; the undefined ignition
count on an inactive ignition; and boundary coordinates relative to domain
start. WRF's unimplemented fuel method 2 and split option 2 remain refusals.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

NVRTC_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
_KDIR = Path(__file__).with_name("kernels")
_THREADS = 128
Bounds = tuple[int, int, int, int]
_ROS_FIELDS = ("vx", "vy", "dzdxf", "dzdyf", "bbb", "betafl", "phiwc", "r_0", "ischap")


@dataclass(frozen=True)
class CoreOptions:
    """Defaults from WRF's ``module_fr_fire_util.F:24-51``."""

    upwinding: int = 9
    upwinding_reinit: int = 4
    upwind_split: int = 0
    grows_only: int = 1
    advection: int = 0
    slope_factor: float = 1.0
    lsm_band_ngp: int = 4
    viscosity: float = 0.4
    viscosity_bg: float = 0.4
    viscosity_band: float = 0.5
    viscosity_ngp: int = 4
    lfn_ext_up: float = 1.0
    lsm_reinit_iter: int = 1
    boundary_guard: int = -1
    fuel_left_method: int = 1

    def __post_init__(self):
        if self.upwinding not in range(10):
            raise ValueError("SFIRE upwinding must be 0 through 9; other values omit the spread gradient")
        if self.upwind_split not in (0, 1):
            raise ValueError("SFIRE split option 2 has no implemented Lax-Friedrichs equation in WRF 4.7.1")
        if self.fuel_left_method != 1:
            raise ValueError("SFIRE fuel method 2 is unimplemented in the published WRF 4.7.1 build")
        if self.lsm_reinit_iter < 0:
            raise ValueError("SFIRE reinitialization iterations cannot be negative")
        if self.lsm_band_ngp < 0 or self.viscosity_ngp < 0:
            raise ValueError("SFIRE level-set bands cannot be negative")


@dataclass(frozen=True)
class IgnitionLine:
    """WRF ignition_line_type fields; distances use coordinate units."""

    start_x: float
    start_y: float
    end_x: float
    end_y: float
    start_time: float
    end_time: float
    radius: float
    ros: float
    stop_time: float = 0.0
    wind_red: float = 1.0
    wrdist: float = 0.0
    wrupwind: float = 0.0

    def __post_init__(self):
        if self.ros <= 0:
            raise ValueError("SFIRE ignition ROS must be positive to define ignition propagation time")
        if self.radius < 0:
            raise ValueError("SFIRE ignition radius cannot be negative")
        if self.end_time < self.start_time:
            raise ValueError("SFIRE ignition line end time cannot precede its start time")


class FireBoundaryReached(RuntimeError):
    """Fire reached the configured boundary guard region."""


def _cupy():
    from gpuwm.local_gpu import no_local_gpu
    if no_local_gpu():
        raise RuntimeError("SFIRE device execution is disabled by the local CUDA device opt-out")
    import cupy as cp
    return cp


def source(kernel_dir=None) -> str:
    """Return the exact compiled source without opening a CUDA device."""
    from gpuwm.core.kernels import _preamble
    root = _KDIR if kernel_dir is None else Path(kernel_dir)
    return (_preamble(root)
            + (root / "glibc_flt32.cuh").read_text(encoding="utf-8")
            + "\n" + (root / "sfire_phys.cuh").read_text(encoding="utf-8")
            + "\n" + (root / "sfire_core.cu").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _module(device: int):
    cp = _cupy()
    from cupy.cuda import compiler
    from gpuwm.certify.kernel_manifest import record_module
    from gpuwm.core.kernels import MODULE_KEY_ROOT
    from gpuwm.kernel_compile_notice import observe_module_compile
    text = source()
    key = f"{MODULE_KEY_ROOT}:sfire_core"
    # RawModule's helper appends --ftz=true in the supported CuPy build.
    # Compile these oracle routines directly to preserve WRF subnormals.
    with cp.cuda.Device(device), observe_module_compile(key):
        ptx,_ = compiler.compile_using_nvrtc(text,NVRTC_OPTIONS,None,"sfire_core.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx,str) else ptx)
    record_module(key, source=text, options=NVRTC_OPTIONS, module=None)
    return module


def _kernel(name):
    cp = _cupy()
    return _module(cp.cuda.Device().id).get_function(name)


def _launch(name, n: int, args):
    if n:
        _kernel(name)(((int(n) + _THREADS - 1) // _THREADS,), (_THREADS,), args)


def _bounds(shape, halo=1, domain=None) -> Bounds:
    if len(shape) != 2:
        raise ValueError("SFIRE fields need a two-dimensional (ny,nx) shape")
    ny, nx = map(int, shape)
    if domain is None:
        if int(halo) != halo or halo < 1:
            raise ValueError("SFIRE needs at least one physical halo row for boundary derivatives")
        domain = (int(halo), nx-int(halo)-1, int(halo), ny-int(halo)-1)
    b = tuple(map(int, domain))
    if len(b) != 4 or any(int(v) != v for v in domain):
        raise ValueError("SFIRE domain bounds must be four integer array indices")
    xlo, xhi, ylo, yhi = b
    if not (1 <= xlo < xhi < nx-1 and 1 <= ylo < yhi < ny-1):
        raise ValueError("SFIRE domain requires two nodes per axis and one surrounding halo row")
    return b


def _tiles(domain: Bounds, tiles: Sequence[Bounds] | None) -> tuple[Bounds, ...]:
    if tiles is None:
        return (domain,)
    result = tuple(tuple(map(int, tile)) for tile in tiles)
    xlo, xhi, ylo, yhi = domain
    total = 0
    for k, b in enumerate(result):
        if len(b) != 4:
            raise ValueError("SFIRE tile bounds must be four integer array indices")
        a, c, d, e = b
        if not (xlo <= a <= c <= xhi and ylo <= d <= e <= yhi):
            raise ValueError("SFIRE tiles must lie inside the physical domain")
        total += (c-a+1)*(e-d+1)
        for p in result[:k]:
            if max(a,p[0]) <= min(c,p[1]) and max(d,p[2]) <= min(e,p[3]):
                raise ValueError("Overlapping SFIRE tiles would update the same Runge-Kutta nodes twice")
    if total != _size(domain):
        raise ValueError("SFIRE tiles must cover every domain node to avoid stale stage values")
    return result


def _size(b: Bounds):
    return (b[1]-b[0]+1)*(b[3]-b[2]+1)


def _ints(*values):
    return tuple(np.int32(v) for v in values)


def _floats(*values):
    return tuple(np.float32(v) for v in values)


def _field(value, shape=None):
    cp = _cupy()
    array = cp.asarray(value, dtype=cp.float32, order="C")
    if shape is not None and array.shape != tuple(shape):
        if array.ndim == 0:
            array = cp.full(shape, array, dtype=cp.float32)
        else:
            raise ValueError(f"SFIRE field shape {array.shape} differs from {tuple(shape)}")
    return array


def continue_at_boundary(lfn, *, bias=1.0, halo=1, domain=None):
    """Extrapolate one physical halo, including diagonal corner values."""
    a = _field(lfn)
    b = _bounds(a.shape, halo, domain)
    _launch("sfire_boundary", max(b[1]-b[0]+1,b[3]-b[2]+1),
            (a, *_ints(a.shape[1],*b), *_floats(bias)))
    return a


def init_no_fire(shape, dx, dy, time_now=0.0, *, halo=1, domain=None):
    """Initialize WRF no-fire state; physical halos are also initialized."""
    cp = _cupy()
    b = _bounds(shape,halo,domain)
    arrays = {key: cp.zeros(shape,dtype=cp.float32)
              for key in ("fuel_frac","fire_area","lfn","tign")}
    _launch("sfire_no_fire",_size(b),(*arrays.values(),*_ints(shape[1],*b),*_floats(dx,dy,time_now)))
    for a in arrays.values():
        continue_at_boundary(a,bias=0.0,domain=b)
    return arrays


initialize_no_fire = init_no_fire


def nearest(ax, ay, line: IgnitionLine, *, unit_x=1.0, unit_y=1.0):
    """WRF midpoint-based nearest point and interpolated ignition time."""
    cp = _cupy()
    ax = _field(ax)
    ay = _field(ay,ax.shape)
    d,t = cp.empty_like(ax),cp.empty_like(ax)
    _launch("sfire_nearest",ax.size,(ax,ay,d,t,np.int32(ax.size),
            *_floats(line.start_x,line.start_y,line.start_time,
                     line.end_x,line.end_y,line.end_time,unit_x*unit_x,unit_y*unit_y)))
    return d,t


def ignite_fire(lfn, tign, coord_x, coord_y, line: IgnitionLine, start_ts, end_ts,
                *, unit_x=1.0, unit_y=1.0, halo=1, domain=None, tiles=None):
    """Mutate ignition state and return WRF's count of nodes in the ignition."""
    cp = _cupy()
    lfn,tign = _field(lfn),_field(tign,lfn.shape)
    coord_x,coord_y = _field(coord_x,lfn.shape),_field(coord_y,lfn.shape)
    b = _bounds(lfn.shape,halo,domain)
    count = cp.zeros(1,dtype=cp.int32)
    for tile in _tiles(b,tiles):
        _launch("sfire_ignite",_size(tile),(lfn,tign,coord_x,coord_y,count,
                *_ints(lfn.shape[1],*tile),
                *_floats(start_ts,end_ts,line.start_x,line.start_y,line.start_time,
                         line.end_x,line.end_y,line.end_time,line.radius,line.ros,unit_x,unit_y)))
    return int(count.item())


def fuel_left_cell_1(lfn_corners, tign_corners, time_now, fuel_time):
    """WRF cell fuel integral, corner order 00,01,10,11."""
    cp = _cupy()
    l = _field(lfn_corners)
    if l.ndim < 1 or l.shape[-1] != 4:
        raise ValueError("SFIRE fuel cell needs four corners ordered 00,01,10,11")
    t = _field(tign_corners,l.shape)
    shape = l.shape[:-1]
    tau = _field(fuel_time,shape)
    fuel,area = cp.empty(shape,dtype=cp.float32),cp.empty(shape,dtype=cp.float32)
    _launch("sfire_fuel_cell",fuel.size,(l,t,tau,np.float32(time_now),fuel,area,np.int32(fuel.size)))
    return fuel,area


def fuel_left(lfn, tign, fuel_time, time_now, *, halo=1, domain=None, tiles=None,
              method=1):
    """WRF four-subcell fuel consumption and burning area, with one input halo."""
    if method != 1:
        raise ValueError("SFIRE fuel method 2 has no implementation in the published WRF 4.7.1 build")
    cp = _cupy()
    lfn = _field(lfn)
    tign,fuel_time = _field(tign,lfn.shape),_field(fuel_time,lfn.shape)
    b = _bounds(lfn.shape,halo,domain)
    fuel,area = cp.ones_like(lfn),cp.zeros_like(lfn)
    for tile in _tiles(b,tiles):
        _launch("sfire_fuel_left",_size(tile),(lfn,tign,fuel_time,np.float32(time_now),fuel,area,
                *_ints(lfn.shape[1],*tile)))
    return fuel,area


def select_derivative(stencil, dx, uf=1.0, *, method=7):
    """Apply native derivative selectors to batches of seven-point stencils.

    ``method`` uses 1=upwind,2=Godunov,3=ENO1,5=second,6=WENO3,7=WENO5,
    10=fourth. Stencil order is i-3 through i+3.
    """
    if method not in (1,2,3,5,6,7,10):
        raise ValueError("SFIRE derivative selector must be 1,2,3,5,6,7 or 10")
    cp = _cupy()
    s = _field(stencil)
    if s.ndim < 1 or s.shape[-1] != 7:
        raise ValueError("SFIRE derivative selector requires seven points i-3 through i+3")
    shape = s.shape[:-1]
    dx,uf = _field(dx,shape),_field(uf,shape)
    out = cp.empty(shape,dtype=cp.float32)
    _launch("sfire_select",out.size,(s,dx,uf,out,np.int32(out.size),np.int32(method)))
    return out


def select_sided(left, right, *, method=3):
    """Apply WRF's first-order selectors directly to one-sided differences."""
    if method not in (1,2,3):
        raise ValueError("SFIRE one-sided selector must be upwind, Godunov or ENO1")
    cp = _cupy()
    left = _field(left)
    right = _field(right,left.shape)
    out = cp.empty_like(left)
    _launch("sfire_select_sided",left.size,(left,right,out,np.int32(left.size),np.int32(method)))
    return out


def tend_ls(lfn, fp: Mapping, dx, dy, *, options=None, halo=1, domain=None,
            tiles=None, output=None, extend=True, reduction_domain=None):
    """Compute WRF level-set tendency, ROS and CFL bound on the device."""
    cp = _cupy()
    opt = options or CoreOptions()
    lfn = _field(lfn)
    b = _bounds(lfn.shape,halo,domain)
    ts = _tiles(b,tiles)
    fields = tuple(_field(fp[key],lfn.shape) for key in _ROS_FIELDS)
    if extend:
        continue_at_boundary(lfn,bias=opt.lfn_ext_up,domain=b)
    if output is None:
        tend,ros,cfl = (cp.zeros_like(lfn) for _ in range(3))
    else:
        tend,ros,cfl = output
        cfl.fill(0)
    for tile in ts:
        _launch("sfire_tend",_size(tile),(lfn,*fields,tend,ros,cfl,
                *_ints(lfn.shape[1],*b,*tile),*_floats(dx,dy),
                *_ints(opt.upwinding,opt.upwind_split,opt.grows_only,opt.advection),
                *_floats(opt.slope_factor,opt.lsm_band_ngp,opt.viscosity,opt.viscosity_bg,
                         opt.viscosity_band,opt.viscosity_ngp)))
    # MAX is order-independent for these finite nonnegative CFL rates.
    reduced = b if reduction_domain is None else _bounds(lfn.shape, domain=reduction_domain)
    maximum = cp.maximum(cfl[reduced[2]:reduced[3]+1,reduced[0]:reduced[1]+1].max(),np.float32(0.0))
    # Keep the final reciprocal on device with a correctly-rounded division.
    bound = _cfl_reciprocal(maximum)
    return tend,ros,bound


@lru_cache(maxsize=None)
def _reciprocal_kernel(device):
    return _module(device).get_function("sfire_cfl_reciprocal")


def _cfl_reciprocal(value):
    cp = _cupy()
    output = cp.empty((),dtype=cp.float32)
    _reciprocal_kernel(cp.cuda.Device().id)((1,),(1,),(value,output))
    return output


def prop_ls_rk3(lfn, fp: Mapping, dx, dy, ts, dt, *, options=None, halo=1,
                domain=None, tiles=None, exchange: Callable | None=None, stages=None,
                reduction_domain=None):
    """WRF three-stage RK propagation, with a barrier between all tile stages.

    ``exchange(stage_array)`` may update process halos on the owning device
    before a tendency call. Physical halos are then extrapolated as in WRF.
    Returns ``(lfn_out, ros, tbound)``; ``tign_update`` runs separately, before
    reinitialization, matching ``module_fr_fire_model.F:327-340``.
    """
    cp = _cupy()
    lfn = _field(lfn)
    b = _bounds(lfn.shape,halo,domain)
    ts_tiles = _tiles(b,tiles)
    initial = lfn.copy()
    stage = initial.copy()
    tend,ros,cfl = (cp.zeros_like(lfn) for _ in range(3))
    bounds = []
    for number in (1,2,3):
        if exchange is not None:
            exchange(stage)
        tend,ros,bound = tend_ls(stage,fp,dx,dy,options=options,domain=b,
                                tiles=ts_tiles,output=(tend,ros,cfl),reduction_domain=reduction_domain)
        if stages is not None:
            cp.copyto(stages[f"lfn_{number-1}"],stage)
        bounds.append(bound)
        next_stage = initial.copy()
        for tile in ts_tiles:
            _launch("sfire_rk_stage",_size(tile),(initial,tend,next_stage,
                    *_ints(lfn.shape[1],*tile),np.float32(dt),np.int32(number)))
        stage = next_stage
    if stages is not None:
        # WRF overwrites the interior of lfn_2 with the final propagation.
        xlo,xhi,ylo,yhi = b
        stages["lfn_2"][ylo:yhi+1,xlo:xhi+1] = stage[ylo:yhi+1,xlo:xhi+1]
    return stage,ros,cp.minimum(cp.minimum(bounds[0],bounds[1]),bounds[2])


def advance_ls_reinit(sign, initial, current, dx, dy, dt_s, rk_coeff,
                      *, method=4, threshold=None, bdy_eno1=10, halo=1,
                      domain=None, tiles=None):
    """One original WRF reinitialization PDE stage."""
    cp = _cupy()
    current = _field(current)
    sign,initial = _field(sign,current.shape),_field(initial,current.shape)
    b = _bounds(current.shape,halo,domain)
    if bdy_eno1 < 3:
        raise ValueError("SFIRE WENO5 reinitialization needs a three-point stencil inside its ENO boundary")
    if threshold is None:
        threshold = 4*np.float32(dx)
    out = current.copy()
    for tile in _tiles(b,tiles):
        _launch("sfire_advance_reinit",_size(tile),(sign,initial,current,out,
                *_ints(current.shape[1],*b,*tile),*_floats(dx,dy,dt_s,threshold),
                *_ints(method,bdy_eno1),np.float32(rk_coeff)))
    return out


def reinit_ls_rk3(lfn_in, lfn_out, dx, dy, *, options=None, halo=1, domain=None,
                  tiles=None, exchange: Callable | None=None, stages=None):
    """Original sign smoothing, pseudo-time RK3 and nonshrinking fire area."""
    cp = _cupy()
    opt = options or CoreOptions()
    before = _field(lfn_in)
    current = _field(lfn_out,before.shape).copy()
    b = _bounds(before.shape,halo,domain)
    ts_tiles = _tiles(b,tiles)
    sign = cp.zeros_like(current)
    for tile in ts_tiles:
        _launch("sfire_reinit_sign",_size(tile),(current,sign,*_ints(current.shape[1],*tile),np.float32(dx)))
    if stages is not None:
        cp.copyto(stages["lfn_s0"],sign)
    if exchange is not None:
        exchange(current)
    continue_at_boundary(current,bias=opt.lfn_ext_up,domain=b)
    dt_s = np.float32(0.01)*np.float32(dx)
    threshold = np.float32(opt.lsm_band_ngp)*np.float32(dx)
    for _ in range(opt.lsm_reinit_iter):
        initial = current.copy()
        for number,coeff in enumerate((np.float32(1.0)/np.float32(3.0),np.float32(0.5),np.float32(1.0)),1):
            current = advance_ls_reinit(sign,initial,current,dx,dy,dt_s,coeff,
                    method=opt.upwinding_reinit,threshold=threshold,domain=b,tiles=ts_tiles)
            if exchange is not None:
                exchange(current)
            continue_at_boundary(current,bias=opt.lfn_ext_up,domain=b)
            if stages is not None:
                cp.copyto(stages[f"lfn_s{number}"],current)
    if stages is not None and opt.lsm_reinit_iter == 0:
        cp.copyto(stages["lfn_s3"],current)
    for tile in ts_tiles:
        _launch("sfire_growth_limit",_size(tile),(before,current,*_ints(current.shape[1],*tile)))
    return current


def tign_update(lfn_in, lfn_out, tign, ts, dt, *, boundary_guard=-1, halo=1,
                domain=None, tiles=None, raise_boundary=True, guard_domain=None):
    """Interpolate new ignition times and detect a reached boundary guard."""
    cp = _cupy()
    before = _field(lfn_in)
    after,tign = _field(lfn_out,before.shape),_field(tign,before.shape)
    b = _bounds(before.shape,halo,domain)
    guard_bounds = b if guard_domain is None else tuple(map(int, guard_domain))
    hit = cp.zeros(1,dtype=cp.int32)
    for tile in _tiles(b,tiles):
        _launch("sfire_tign_update",_size(tile),(before,after,tign,hit,
                *_ints(before.shape[1],*guard_bounds,*tile),*_floats(ts,dt),np.int32(boundary_guard)))
    reached = bool(hit.item())
    if reached and raise_boundary:
        raise FireBoundaryReached("Fire reached the configured domain boundary guard")
    return reached


def calc_flame_length(ros, iboros, fire_area, *, halo=1, domain=None, tiles=None):
    """WRF flame length, front ROS and fire-line intensity."""
    cp = _cupy()
    ros = _field(ros)
    iboros,area = _field(iboros,ros.shape),_field(fire_area,ros.shape)
    b = _bounds(ros.shape,halo,domain)
    length,front,intensity = (cp.zeros_like(ros) for _ in range(3))
    for tile in _tiles(b,tiles):
        _launch("sfire_flame",_size(tile),(ros,iboros,area,length,front,intensity,
                *_ints(ros.shape[1],*tile)))
    return length,front,intensity


__all__ = ["CoreOptions","IgnitionLine","FireBoundaryReached","init_no_fire",
           "initialize_no_fire","nearest","ignite_fire","fuel_left_cell_1",
           "fuel_left","select_derivative","select_sided","continue_at_boundary","tend_ls",
           "prop_ls_rk3","advance_ls_reinit","reinit_ls_rk3","tign_update",
           "calc_flame_length"]
