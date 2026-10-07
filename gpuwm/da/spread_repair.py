"""Observation-aware spread maintenance for replay and cycled LETKF.

The operational array path is CuPy. NumPy is a CPU numerical reference for
focused tests. This module does not read weather files or generate maps.
New controls are selectable until the S1/S4/S5/C4 qualification is recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
from typing import Mapping

import numpy as np

SCHEMA = "gpuwm.da.spread-repair/v1"


def _xp(array):
    if type(array).__module__.split(".")[0] == "cupy":
        import cupy
        return cupy
    return np


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _as_backend(array, xp):
    """Explicit host/device transfer; CuPy forbids implicit NumPy copies."""
    source = _xp(array)
    if xp is np and source is not np:
        array = source.asnumpy(array)
    return xp.asarray(array)


@dataclass(frozen=True)
class SpreadRepairConfig:
    """Named replay policy. Amplitudes are physical standard deviations.

    ``blanket`` is an explicitly labelled comparison arm. The normal
    ``innovation`` policy never modifies uncovered or observed-clear cells.
    The inverse-gamma update uses the enhanced finite-ensemble likelihood,
    numerical quadrature and moment projection, rather than claiming byte
    parity with the DART implementation.
    """
    policy: str = "innovation"
    observed_threshold_dbz: float = 25.0
    innovation_threshold_dbz: float = 10.0
    targeted_threshold_dbz: float = 15.0
    dry_member_threshold_dbz: float = 1.0
    targeted_variance_factor: float = 1.5
    temperature_std_k: float = 0.5
    wind_std_ms: float = 0.5
    vapour_fraction_std: float = 0.05
    horizontal_scale_cells: float = 2.5
    vertical_scale_levels: float = 2.0
    top_taper_levels: int = 3
    adaptive: bool = True
    inflation_initial_mode: float = 1.05
    inflation_initial_sd: float = 0.2
    inflation_min: float = 1.0
    inflation_max: float = 3.0
    inflation_damping: float = 0.9
    quadrature_points: int = 129
    quadrature_chunk_cells: int = 16384

    def __post_init__(self):
        if self.policy not in ("off", "innovation", "blanket"):
            raise ValueError("spread_repair policy must be off, innovation or blanket")
        for key, value in vars(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if not 0 <= self.targeted_threshold_dbz < self.observed_threshold_dbz:
            raise ValueError("targeted echo threshold must be below the strong-echo threshold")
        if self.innovation_threshold_dbz < 0:
            raise ValueError("negative innovation threshold would perturb overpredicted storms")
        if self.targeted_variance_factor < 1:
            raise ValueError("targeted_variance_factor must be at least one")
        for key in ("temperature_std_k", "wind_std_ms", "vapour_fraction_std"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if self.horizontal_scale_cells < 2 or self.vertical_scale_levels < 0:
            raise ValueError("noise must resolve at least two horizontal cells")
        if int(self.top_taper_levels) != self.top_taper_levels or self.top_taper_levels < 0:
            raise ValueError("top taper must be a nonnegative integer")
        if not 0 < self.inflation_min <= self.inflation_initial_mode <= self.inflation_max:
            raise ValueError("initial inflation must lie inside positive inflation bounds")
        if self.inflation_initial_sd <= 0 or not 0 <= self.inflation_damping <= 1:
            raise ValueError("inflation uncertainty must be positive and damping in [0,1]")
        if int(self.quadrature_points) != self.quadrature_points or self.quadrature_points < 33:
            raise ValueError("quadrature needs at least 33 points to resolve the posterior")
        if int(self.quadrature_chunk_cells) != self.quadrature_chunk_cells or self.quadrature_chunk_cells < 1:
            raise ValueError("quadrature chunk must contain at least one cell")


def echo_gates(observed, simulated, coverage, config):
    """Cellwise gates, with missing coverage distinct from clear air."""
    xp = _xp(simulated)
    z = xp.asarray(observed, dtype=xp.float64)
    hx = xp.asarray(simulated, dtype=xp.float64)
    mask = xp.asarray(coverage, dtype=bool)
    if hx.ndim != z.ndim + 1 or hx.shape[1:] != z.shape or mask.shape != z.shape:
        raise ValueError("reflectivity members and coverage must sit on the same grid")
    if hx.shape[0] < 2:
        raise ValueError("spread repair needs at least two members to preserve an ensemble mean")
    valid = mask & xp.isfinite(z) & xp.all(xp.isfinite(hx), axis=0)
    innovation = z - hx.mean(axis=0)
    echo = valid & (z > config.targeted_threshold_dbz)
    targeted = echo & (hx.max(axis=0) < config.dry_member_threshold_dbz)
    additive = valid & (z > config.observed_threshold_dbz) & (innovation > config.innovation_threshold_dbz)
    if config.policy == "blanket":
        additive = xp.ones_like(valid)
    elif config.policy == "off":
        additive = xp.zeros_like(valid)
        targeted = xp.zeros_like(valid)
        echo = xp.zeros_like(valid)
    return {"echo": echo, "additive": additive, "targeted": targeted,
            "innovation": innovation, "valid": valid}


def _ig_from_mode_variance(mode, variance):
    # beta=mode*(alpha+1); var=beta^2/((alpha-1)^2*(alpha-2)).
    lo, hi = 2.0 + 1e-8, 1e9
    ratio = variance / (mode * mode)
    for _ in range(80):
        a = 0.5 * (lo + hi)
        r = (a + 1) ** 2 / ((a - 1) ** 2 * (a - 2))
        if r > ratio:
            lo = a
        else:
            hi = a
    a = 0.5 * (lo + hi)
    return a, mode * (a + 1)


@dataclass
class AdaptiveInflationState:
    """Persistent spatial inverse-gamma shape and scale, not one scalar."""
    shape: object
    scale: object
    cycles: int = 0

    @classmethod
    def initial(cls, grid_shape, config, xp=np):
        a, b = _ig_from_mode_variance(config.inflation_initial_mode,
                                     config.inflation_initial_sd ** 2)
        return cls(xp.full(grid_shape, a, dtype=xp.float64),
                   xp.full(grid_shape, b, dtype=xp.float64))

    def validate(self, grid_shape):
        xp = _xp(self.shape)
        if self.shape.shape != tuple(grid_shape) or self.scale.shape != tuple(grid_shape):
            raise ValueError("inflation restart grid mismatch would move uncertainty to wrong cells")
        _require(xp.all(xp.isfinite(self.shape) & (self.shape > 2)),
                 "inverse-gamma shape must exceed two for finite uncertainty")
        _require(xp.all(xp.isfinite(self.scale) & (self.scale > 0)),
                 "inverse-gamma scale must be finite and positive")
        if self.cycles < 0:
            raise ValueError("inflation cycle counter must be nonnegative")

    def mode(self, config):
        xp = _xp(self.shape)
        return xp.clip(self.scale / (self.shape + 1),
                       config.inflation_min, config.inflation_max)

    def checkpoint(self):
        """Arrays stay on their backend; the caller's restart writer owns I/O."""
        return {"schema": SCHEMA, "shape": self.shape.copy(),
                "scale": self.scale.copy(), "cycles": self.cycles}

    @classmethod
    def restore(cls, record, grid_shape, xp=np):
        if record.get("schema") != SCHEMA:
            raise ValueError("unknown inflation restart schema cannot safely resume this policy")
        state = cls(xp.asarray(record["shape"]).copy(),
                    xp.asarray(record["scale"]).copy(), int(record["cycles"]))
        state.validate(grid_shape)
        return state


def update_adaptive_inflation(state, observed, simulated, errors, gate,
                              config, *, localization_weight=1.0):
    """Enhanced IG likelihood on *uninflated* H(x), for the next cycle.

    V=R+(([1+gamma*(sqrt(lambda)-1)]**2)-1/N)*P. Observation
    error R and finite-member correction are retained. Independent cells
    use bounded-memory log-lambda quadrature, followed by IG moment fitting.
    No-information cells keep their prior. This is an enhanced-IG numerical
    variant, not an assertion of DART implementation identity.
    """
    xp = _xp(simulated)
    hx = xp.asarray(simulated, dtype=xp.float64)
    z = xp.asarray(observed, dtype=xp.float64)
    state.validate(z.shape)
    n = hx.shape[0]
    if n < 2 or hx.shape[1:] != z.shape:
        raise ValueError("adaptive inflation requires aligned ensemble H(x)")
    r = xp.broadcast_to(xp.asarray(errors, dtype=xp.float64), z.shape) ** 2
    gamma = xp.broadcast_to(xp.asarray(localization_weight, dtype=xp.float64), z.shape)
    eligible = xp.asarray(gate, dtype=bool)
    if eligible.shape != z.shape:
        raise ValueError("adaptive inflation gate shape differs from the radar grid")
    _require(xp.all(~eligible | (xp.isfinite(r) & (r > 0))),
             "positive observation error prevents a singular innovation likelihood")
    _require(xp.all(xp.isfinite(gamma) & (gamma >= 0) & (gamma <= 1)),
             "localization weights must lie in [0,1]")
    p = hx.var(axis=0, ddof=1)
    d = z - hx.mean(axis=0)
    active = eligible & xp.isfinite(d) & xp.isfinite(p) & (p > 1e-12) & (gamma > 0)
    # Integration is numerical and finite. Extend it using the prior tail
    # scale and innovation, independently of the applied-factor bounds.
    side_points = config.quadrature_points//4
    core_points = config.quadrature_points-2*side_points
    a_out, b_out = state.shape.copy(), state.scale.copy()
    arrays = [x.reshape(-1) for x in (state.shape, state.scale, p, d, r, gamma, active)]
    for start in range(0, z.size, config.quadrature_chunk_cells):
        sl = slice(start, start + config.quadrature_chunk_cells)
        a, b, pp, dd, rr, gg, use = [x[sl] for x in arrays]
        pp = xp.where(use, pp, 0.)
        dd = xp.where(use, dd, 0.)
        rr = xp.where(use, rr, 1.)
        lower = xp.minimum(.05, b/(a+1)/64.)
        upper = xp.maximum(max(20., 8*config.inflation_max),
                           128*b/(a-2))
        upper = xp.maximum(upper, 4*dd**2/xp.maximum(pp, 1e-12))
        # Find the posterior peak, then resolve its curvature. A fixed
        # wide grid would snap a precise restart prior onto one distant node.
        def log_density(t):
            ll = xp.exp(t)
            vv = rr + ((1+gg*(xp.sqrt(ll)-1))**2-1/n)*pp
            safe = xp.maximum(vv,1e-300)
            return xp.where(vv > 0,-a*t-b/ll-.5*xp.log(safe)-dd**2/(2*safe),-xp.inf)
        support_root = xp.maximum(0.,(xp.sqrt(xp.maximum(1/n-rr/xp.maximum(pp,1e-300),0.))-1+gg)/xp.maximum(gg,1e-300))**2
        lower = xp.maximum(lower,support_root*(1+1e-10))
        left,right = xp.log(lower),xp.log(upper)
        phi = (math.sqrt(5)-1)/2
        for _ in range(64):
            t1 = right-phi*(right-left)
            t2 = left+phi*(right-left)
            choose_left = log_density(t1) > log_density(t2)
            right = xp.where(choose_left,t2,right)
            left = xp.where(choose_left,left,t1)
        peak = .5*(left+right)
        ll = xp.exp(peak)
        root = xp.sqrt(ll)
        ss = 1+gg*(root-1)
        vv = rr+(ss**2-1/n)*pp
        vp = pp*gg*root*ss
        vpp = .5*pp*(gg*root*ss+gg**2*ll)
        curvature = -b/ll+.5*vpp*(dd**2/vv**2-1/vv)+.5*vp**2*(1/vv**2-2*dd**2/vv**3)
        width = xp.sqrt(1/xp.maximum(-curvature,1e-12))
        core_low = xp.maximum(xp.log(lower),peak-10*width)
        core_high = xp.minimum(xp.log(upper),peak+10*width)
        def interval(lo,hi,count):
            unit = xp.linspace(0.,1.,count,dtype=xp.float64)[:,None]
            return lo[None]+unit*(hi-lo)[None]
        log_nodes = xp.concatenate((interval(xp.log(lower),core_low,side_points),
                       interval(core_low,core_high,core_points),
                       interval(core_high,xp.log(upper),side_points)),axis=0)
        trapezoid = xp.empty_like(log_nodes)
        trapezoid[0] = .5*(log_nodes[1]-log_nodes[0])
        trapezoid[-1] = .5*(log_nodes[-1]-log_nodes[-2])
        trapezoid[1:-1] = .5*(log_nodes[2:]-log_nodes[:-2])
        lam = xp.exp(log_nodes)
        v = rr[None] + ((1 + gg[None]*(xp.sqrt(lam)-1))**2 - 1/n)*pp[None]
        valid_variance = v > 0
        safe_v = xp.where(valid_variance, v, 1.)
        # IG density times d(lambda)/d(log lambda). Normalizing constants
        # cancel inside each cell, and a log maximum prevents underflow.
        logw = -a[None]*log_nodes - b[None]/lam - .5*xp.log(safe_v) - dd[None]**2/(2*safe_v)
        logw = xp.where(valid_variance, logw, -xp.inf)
        weights = xp.exp(logw - logw.max(axis=0, keepdims=True))*trapezoid
        total = weights.sum(axis=0)
        mean = (weights*lam).sum(axis=0)/total
        variance = (weights*(lam-mean[None])**2).sum(axis=0)/total
        new_a = 2 + mean**2/xp.maximum(variance, 1e-10)
        new_b = mean*(new_a-1)
        a_out.reshape(-1)[sl] = xp.where(use, new_a, a)
        b_out.reshape(-1)[sl] = xp.where(use, new_b, b)
    return AdaptiveInflationState(a_out, b_out, state.cycles + 1)


def _on_field(mass, shape, *, gate=False):
    """Mass-grid quantity on staggered faces. Gates use intersection."""
    xp = _xp(mass)
    if tuple(mass.shape) == tuple(shape):
        return mass
    if mass.shape[:-1] == tuple(shape[:-1]) and mass.shape[-1]+1 == shape[-1]:
        padded = xp.pad(mass, ((0,0), (0,0), (1,1)), mode="edge")
        left, right = padded[..., :-1], padded[..., 1:]
    elif mass.shape[0] == shape[0] and mass.shape[1]+1 == shape[1] and mass.shape[2] == shape[2]:
        padded = xp.pad(mass, ((0,0), (1,1), (0,0)), mode="edge")
        left, right = padded[:, :-1], padded[:, 1:]
    else:
        raise ValueError("spread grid does not align with the field's mass or staggered geometry")
    return (left & right) if gate else .5*(left+right)


def scale_deviations(ensemble, variance_factor, *, gate, lower=None, upper=None,
                     return_factor=False):
    """Mean-preserving common rescale, with bounds and no wetward clipping."""
    xp = _xp(ensemble)
    x = xp.asarray(ensemble)
    mean = x.mean(axis=0, dtype=xp.float64)
    anomaly = x.astype(xp.float64) - mean
    factor = xp.sqrt(xp.asarray(variance_factor, dtype=xp.float64))
    _require(xp.all(xp.isfinite(factor) & (factor >= 0)), "spread factor must be finite and nonnegative")
    factor = xp.broadcast_to(factor, mean.shape).copy()
    if lower is not None:
        bound = xp.broadcast_to(xp.asarray(lower), mean.shape)
        _require(xp.all(mean >= bound), "mean below lower bound cannot be preserved by a spread limiter")
        limit = xp.where(anomaly < 0, (mean-bound)/xp.maximum(-anomaly, 1e-30), xp.inf).min(axis=0)
        factor = xp.minimum(factor, limit)
    if upper is not None:
        bound = xp.broadcast_to(xp.asarray(upper), mean.shape)
        _require(xp.all(mean <= bound), "mean above upper bound cannot be preserved by a spread limiter")
        limit = xp.where(anomaly > 0, (bound-mean)/xp.maximum(anomaly, 1e-30), xp.inf).min(axis=0)
        factor = xp.minimum(factor, limit)
    out = (mean + anomaly*factor).astype(x.dtype)
    output = xp.where(xp.asarray(gate)[None], out, x)
    return (output, factor) if return_factor else output


def _smooth_noise(shape, seed, config, xp):
    """Counter-based Gaussian noise, nonperiodic in every direction.

    All arithmetic runs on the selected array backend. Counter hashing
    makes streams independent of chunks; CUDA transcendental byte identity
    still needs Blackwell certification. Normalize before gates and taper.
    """
    counter = xp.arange(math.prod(shape), dtype=xp.uint64).reshape(shape)
    with np.errstate(over="ignore"):
        def uniform(salt):
            q = counter + xp.uint64((int(seed) + salt) % (1 << 64))
            q = (q ^ (q >> xp.uint64(30)))*xp.uint64(0xbf58476d1ce4e5b9)
            q = (q ^ (q >> xp.uint64(27)))*xp.uint64(0x94d049bb133111eb)
            q = q ^ (q >> xp.uint64(31))
            return ((q >> xp.uint64(11)).astype(xp.float64)+.5)/(1 << 53)
        noise = xp.sqrt(-2*xp.log(uniform(0x9e3779b97f4a7c15)))*xp.cos(2*math.pi*uniform(0x243f6a8885a308d3))
    for axis, sigma in enumerate((config.vertical_scale_levels,
                                  config.horizontal_scale_cells,
                                  config.horizontal_scale_cells)):
        if sigma <= 0:
            continue
        radius = int(math.ceil(4*sigma))
        weights = [math.exp(-.5*(i/sigma)**2) for i in range(-radius, radius+1)]
        norm = math.sqrt(sum(w*w for w in weights))
        pad = [(0,0)]*3
        pad[axis] = (radius, radius)
        padded = xp.pad(noise, pad, mode="constant")
        filtered = xp.zeros_like(noise)
        for i, weight in enumerate(weights):
            slices = [slice(None)]*3
            slices[axis] = slice(i, i+shape[axis])
            filtered += (weight/norm)*padded[tuple(slices)]
        noise = filtered
    return noise


def additive_repair(prior: Mapping[str, object], gates, config, *, seed,
                    cycle, pressure=None, vapour_upper=None):
    """Perturb wind, temperature and vapour, never hydrometeor mass.

    The dry-echo gate receives separate thermodynamic noise because a
    multiplicative factor cannot restore exactly zero anomalies. Creating
    hydrometeor/H(x) covariance still requires a subsequent forecast leg.
    """
    if config.policy == "off":
        return {name: value.copy() for name, value in prior.items()}
    from gpuwm.da.perturb import vertical_taper
    support = gates["additive"] | gates["targeted"]
    out = {name: value.copy() for name, value in prior.items()}
    # Table work defines a field's treatment. No model-specific code paths.
    rules = (("u", config.wind_std_ms, "absolute"),
             ("v", config.wind_std_ms, "absolute"),
             ("thp", config.temperature_std_k, "temperature"),
             ("qv", config.vapour_fraction_std, "fractional"))
    for stream, (field, amplitude, kind) in enumerate(rules):
        if field not in prior or amplitude == 0:
            continue
        x = prior[field]
        xp = _xp(x)
        if x.ndim != 4 or x.shape[0] < 2:
            raise ValueError("spread fields need (members, levels, rows, columns)")
        gate = _on_field(support, x.shape[1:], gate=True)
        taper = vertical_taper(x.shape[1], top_width_levels=config.top_taper_levels, xp=xp)[:,None,None]
        noise = xp.stack([_smooth_noise(x.shape[1:],
                          seed + 1000003*cycle + 104729*stream + 7919*member,
                          config, xp) for member in range(x.shape[0])])
        noise -= noise.mean(axis=0, keepdims=True)
        noise *= amplitude*taper[None]*gate[None]
        if kind == "fractional":
            noise *= x.mean(axis=0, dtype=xp.float64)[None]
        elif kind == "temperature":
            if pressure is None:
                raise ValueError("temperature noise needs pressure to convert kelvin to potential temperature")
            from gpuwm.core import constants
            p = (xp.stack([xp.asarray(value) for value in pressure])
                 if isinstance(pressure, (tuple, list)) else xp.asarray(pressure))
            if p.shape == x.shape[1:]:
                p = xp.broadcast_to(p, x.shape)
            if p.shape != x.shape:
                raise ValueError("pressure must align with temperature members")
            _require(xp.all(xp.isfinite(p) & (p > 0)), "positive pressure prevents invalid Exner conversion")
            noise /= (p/constants.P0)**constants.RCP
            noise -= noise.mean(axis=0, keepdims=True)
        if kind == "fractional":
            # Limit each cell's entire noise vector, preserving its mean.
            _require(xp.all(x >= 0), "negative prior vapour cannot be preserved by an additive spread limiter")
            limit = xp.where(noise < 0, x/xp.maximum(-noise, 1e-30), xp.inf).min(axis=0)
            if vapour_upper is not None:
                upper = xp.broadcast_to(xp.asarray(vapour_upper), x.shape)
                _require(xp.all(x >= 0) & xp.all(x <= upper), "prior vapour outside bounds cannot be repaired without changing its mean")
                limit = xp.minimum(limit, xp.where(noise > 0,
                    (upper-x)/xp.maximum(noise, 1e-30), xp.inf).min(axis=0))
            noise *= xp.minimum(1., xp.maximum(0., limit))[None]
        out[field] = xp.where(gate[None], (x+noise).astype(x.dtype), x)
    return out


class SpreadRepairController:
    """Persistent cycle API with explicit H(x) rebuild after prior changes."""
    def __init__(self, config=SpreadRepairConfig(), *, seed=0, state=None):
        self.config, self.seed, self.state = config, int(seed), state
        self.last_receipt = None
        self._analysis_valid_time = None
        self._noise_phase = None

    def set_analysis_time(self, valid_time):
        """Use validated radar time for matching cycle and replay noise draws.

        Without time metadata the existing analysis counter remains the
        noise phase. UTC normalization gives the same phase to equivalent
        timestamp spellings, independently of process or replay ordering.
        """
        if valid_time is None:
            self._analysis_valid_time = self._noise_phase = None
            return
        try:
            stamp = (valid_time if isinstance(valid_time,datetime)
                     else datetime.fromisoformat(str(valid_time).replace("Z","+00:00")))
        except (ValueError,TypeError) as error:
            raise ValueError("analysis valid time must identify an actual UTC instant for reproducible spread noise") from error
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("analysis valid time needs an explicit timezone to avoid selecting a different noise phase")
        stamp = stamp.astimezone(timezone.utc)
        elapsed = stamp-datetime(1970,1,1,tzinfo=timezone.utc)
        self._noise_phase = ((elapsed.days*86400+elapsed.seconds)*1000000
                             +elapsed.microseconds)
        self._analysis_valid_time = stamp.isoformat(timespec="microseconds")

    def _noise_cycle(self):
        return (self._noise_phase if self._noise_phase is not None
                else self.state.cycles if self.state is not None else 0)

    def analyze(self, prior, observations, grid, filter_config, *, radar,
                rebuild_observations, pressure=None, vapour_upper=None,
                 solve_namespace=None, diagnostics=None, progress=None,
                 maintain_spread=True, inflation_radar=None):
        """Run repair, rebuild nonlinear H(x), LETKF, then additive maintenance.

        ``radar`` is a full-grid GriddedObs precipitation batch. A mandatory
        callback prevents stale H(x) being paired with inflated model states.
        ``"linearized"`` instead rescales model and H(x) anomalies together,
        the usual multiplicative ensemble-covariance inflation approximation.
        It is stamped in the receipt and does not claim nonlinear reevaluation.
        The filter's result is converted back to increments from the caller's
        original prior, so existing restart/application tools can consume it.
        Native replay passes ``maintain_spread=False`` and performs additive
        maintenance once after its configured posterior positivity stage.
        """
        from gpuwm.da.letkf import analyze
        cfg = self.config
        if cfg.policy == "off":
            result = analyze(prior, observations, grid, filter_config,
                             diagnostics, solve_namespace=solve_namespace,
                             progress=progress)
            self.last_receipt = {"schema":SCHEMA,"policy":"off","adaptive":False}
            return result
        if rebuild_observations is None:
            raise ValueError("spread repair needs H(x) rebuilding to prevent state/operator covariance mismatch")
        if getattr(radar, "window", None) is not None:
            raise ValueError("spread repair radar must cover the full mass grid to prevent gate displacement")
        if filter_config.prior_inflation != 1.0:
            raise ValueError("spread repair owns spatial prior inflation; a second scalar inflation would double inflate covariance")
        fields = tuple(filter_config.analysis_fields)
        xp = solve_namespace if solve_namespace is not None else _xp(radar.simulated)
        if solve_namespace is None:
            # A native resident solve has device priors but its captured raw
            # gate may still be on the host. That gate must not switch the
            # model computation to NumPy and attempt an implicit device copy.
            for field in fields:
                source = _xp(prior[field])
                if source is not np:
                    xp = source
                    break
        radar = materialize_radar_batch(radar, xp)
        gates = echo_gates(radar.values, radar.simulated, radar.mask, cfg)
        if cfg.adaptive:
            # The raw operator defines echo gates. The final, uninflated
            # solver batch defines the coherent class/floor/tempered R
            # likelihood and which observations were actually retained.
            likelihood = materialize_radar_batch(
                radar if inflation_radar is None else inflation_radar, xp)
            if likelihood.values.shape != radar.values.shape:
                raise ValueError("adaptive likelihood and raw echo gates must share the same mass grid")
        if self.state is None:
            self.state = AdaptiveInflationState.initial(gates["echo"].shape, cfg, xp=xp)
        elif _xp(self.state.shape) is not xp:
            self.state = AdaptiveInflationState(
                _as_backend(self.state.shape,xp), _as_backend(self.state.scale,xp),
                self.state.cycles)
        self.state.validate(gates["echo"].shape)
        factor = xp.ones(gates["echo"].shape, dtype=xp.float64)
        if cfg.adaptive:
            factor = (1 + cfg.inflation_damping*(xp.sqrt(self.state.mode(cfg))-1))**2
        factor = xp.where(gates["targeted"], xp.maximum(factor, cfg.targeted_variance_factor), factor)
        factor = xp.where(gates["echo"], factor, 1.)
        # A shared effective covariance factor keeps linearized H(x) and
        # every bounded model variable coherent. Per-field clipping followed
        # by nominal H(x) inflation would silently use the wrong covariance.
        for field in fields:
            if field not in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"):
                continue
            x = xp.asarray(prior[field])
            _, effective = scale_deviations(x, factor, gate=gates["echo"],
                lower=0., upper=vapour_upper if field == "qv" else None,
                return_factor=True)
            factor = xp.minimum(factor, effective**2)
        repaired = dict(prior)
        for field in fields:
            x = prior[field]
            original_backend = _xp(x)
            x = xp.asarray(x)
            f = _on_field(factor, x.shape[1:])
            gate = _on_field(gates["echo"], x.shape[1:], gate=True)
            value = scale_deviations(x, f, gate=gate,
                lower=0. if field in ("qv", "qc", "qr", "qi", "qs", "qg", "qh") else None,
                upper=vapour_upper if field == "qv" else None)
            repaired[field] = xp.asnumpy(value) if original_backend is np and xp is not np else value
        if rebuild_observations == "linearized":
            batches = inflate_observation_anomalies(observations, factor, xp)
        else:
            batches = rebuild_observations(repaired)
        increments = analyze(repaired, batches, grid, filter_config,
                             diagnostics, solve_namespace=solve_namespace,
                             progress=progress)
        result = {}
        noise_cycle = self._noise_cycle()
        # One field on the device at a time: staged 32-member backgrounds
        # need not become a whole-state resident allocation for spread repair.
        for field in fields:
            value = repaired[field]
            base = xp.asarray(value)
            posterior = base + xp.asarray(increments[field])
            perturbed = (additive_repair({field: posterior}, gates, cfg, seed=self.seed,
                cycle=noise_cycle, pressure=pressure, vapour_upper=vapour_upper)[field]
                if maintain_spread else posterior)
            increment = perturbed - xp.asarray(prior[field])
            result[field] = xp.asnumpy(increment) if _xp(prior[field]) is np and xp is not np else increment
        new_state = (update_adaptive_inflation(self.state, likelihood.values,
                      likelihood.simulated, likelihood.errors,
                      gates["echo"] & likelihood.mask, cfg)
                     if cfg.adaptive else AdaptiveInflationState(
                         self.state.shape.copy(), self.state.scale.copy(), self.state.cycles+1))
        # Commit state only after a successful complete solve and noise step.
        self.state = new_state
        self.last_receipt = {"schema": SCHEMA, "policy": cfg.policy,
            "operator_update": "linearized-covariance" if rebuild_observations == "linearized" else "reevaluated",
            "adaptive": cfg.adaptive, "cycles": new_state.cycles,
            "strong_innovation_cells": int(gates["additive"].sum()),
            "missed_echo_cells": int(gates["targeted"].sum()),
            "observed_echo_cells": int(gates["echo"].sum()),
            "inflation_min": float(factor.min()), "inflation_max": float(factor.max()),
            "clear_air_noise": cfg.policy == "blanket",
            "maintenance_applied": bool(maintain_spread),
            "noise_seed": self.seed, "noise_phase": noise_cycle,
            "noise_phase_source": "analysis-valid-time" if self._noise_phase is not None else "analysis-counter",
            "analysis_valid_time": self._analysis_valid_time,
            "compute_backend": "numpy-reference" if xp is np else "cupy"}
        return result

    def analysis_runner(self, *, pressure=None, vapour_upper=None, gate_radar=None):
        """Native replay: solve, bounded posterior, then one maintenance step.

        ``gate_radar`` preserves raw precipitation H(x) when the observations
        consumed by LETKF have been conditioned to an echo floor. The caller
        uses ``subsolve`` for cropped field-rule solves and calls ``finish``
        once after posterior positivity on the full mass grid.
        """
        from gpuwm.da.obs_radar import REFLECTIVITY_NAME
        from gpuwm.da.letkf import analyze
        pending = None
        def run(prior, observations, grid, config, diagnostics=None,
                *, solve_namespace=None, progress=None):
            nonlocal pending
            pending = None
            radar = next((b for b in observations if b.name == REFLECTIVITY_NAME), None)
            if radar is None:
                self.last_receipt = {"schema":SCHEMA,"policy":self.config.policy,
                                     "applied":False,"reason":"this pass has no precipitation reflectivity batch"}
                return analyze(prior,observations,grid,config,diagnostics,
                               solve_namespace=solve_namespace,progress=progress)
            raw_radar = run.gate_radar if run.gate_radar is not None else radar
            cycle = self._noise_cycle()
            previous_state = self.state
            try:
                result = self.analyze(prior, observations, grid, config, radar=raw_radar,
                    rebuild_observations="linearized", pressure=pressure,
                    vapour_upper=vapour_upper, diagnostics=diagnostics,
                    solve_namespace=solve_namespace, progress=progress,
                    maintain_spread=False, inflation_radar=radar)
            except BaseException:
                if self.config.adaptive:
                    self.state = previous_state
                raise
            next_state = self.state
            if self.config.adaptive:
                # Positivity and capacity retries still follow this solve.
                # Hold the candidate until finish succeeds exactly once.
                self.state = previous_state
            if self.config.policy != "off":
                xp = solve_namespace if solve_namespace is not None else _xp(
                    next(iter(result.values())))
                raw = materialize_radar_batch(raw_radar, xp)
                pending = {"gates":echo_gates(raw.values,raw.simulated,raw.mask,self.config),
                           "cycle":cycle,"fields":tuple(config.analysis_fields)}
                if self.config.adaptive:
                    pending["next_state"] = next_state
            return result

        def finish(prior, increments, diagnostics=None, solve_namespace=None):
            nonlocal pending
            if pending is None:
                return increments
            work = pending
            xp = solve_namespace if solve_namespace is not None else _xp(
                next(iter(increments.values())))
            gates = {name:_as_backend(value,xp) for name,value in work["gates"].items()}
            updated = dict(increments)
            for field in work["fields"]:
                if field not in increments:
                    continue
                if field not in ("u","v","thp","qv"):
                    continue
                posterior = _as_backend(prior[field],xp)+_as_backend(increments[field],xp)
                maintained = additive_repair({field:posterior}, gates, self.config,
                    seed=self.seed, cycle=work["cycle"], pressure=pressure,
                    vapour_upper=vapour_upper)[field]
                increment = maintained-_as_backend(prior[field],xp)
                updated[field] = _as_backend(increment,_xp(increments[field]))
            # Consume only after all fields succeed. Failed maintenance can
            # be retried, while a repeated finish never injects another draw.
            if self.config.adaptive:
                self.state = work["next_state"]
                self.last_receipt["state_commit"] = "after-posterior-positivity"
            pending = None
            self.last_receipt["maintenance_applied"] = True
            self.last_receipt["maintenance_cycle"] = work["cycle"]
            self.last_receipt["maintenance_stage"] = "after-posterior-positivity"
            self.last_receipt["gate_reflectivity"] = "raw" if run.gate_radar is not None else "analysis-batch"
            return updated

        # radar_assimilation uses this capability to retry an allocation
        # failure with host inputs and a bounded CUDA transform. Without it,
        # wrapping LETKF removes that recovery route before repair runs.
        run.supports_host_staging = True
        run.finish = finish
        run.subsolve = analyze
        run.gate_radar = gate_radar
        run.spread_controller = self
        return run

    def analyze_time_shifted(self, snapshots, grid, filter_config, *, analysis_seconds,
                             observation_builder, pressure=None, vapour_upper=None,
                             mode="vtsm", solve_namespace=None, diagnostics=None):
        """Analyse 3K covariance samples and return increments for K members.

        The caller supplies an observation builder for the pooled model
        states. Every observation batch must have all 3K simulated values;
        the same builder is called again after prior inflation. Retained
        snapshot metadata enforces the observation cutoff before any solve.
        """
        from gpuwm.da.obs_radar import REFLECTIVITY_NAME
        pooled, receipt = time_shifted_covariance(snapshots,
                            analysis_seconds=analysis_seconds, mode=mode)
        if pressure is None and "p" in pooled:
            pressure = pooled["p"]
        if pressure is not None:
            expected = next(iter(pooled.values())).shape
            actual = ((len(pressure),)+pressure[0].shape
                      if isinstance(pressure,(tuple,list)) else pressure.shape)
            if actual not in (expected,expected[1:]):
                raise ValueError("time-shift pressure must cover all 3K covariance samples or the mass grid")
        if vapour_upper is not None and hasattr(vapour_upper,"shape"):
            expected = next(iter(pooled.values())).shape
            if vapour_upper.shape not in ((),expected,expected[1:]):
                raise ValueError("time-shift vapour bounds must cover all 3K covariance samples or the mass grid")
        observations = observation_builder(pooled)
        radar = next((b for b in observations if b.name == REFLECTIVITY_NAME), None)
        if radar is None:
            raise ValueError("time-shift spread repair needs the precipitation batch to define observed echo")
        increments = self.analyze(pooled, observations, grid, filter_config,
            radar=radar, rebuild_observations=observation_builder,
            pressure=pressure, vapour_upper=vapour_upper,
            solve_namespace=solve_namespace, diagnostics=diagnostics)
        central = central_time_posterior(pooled, increments, receipt)
        original = next(s.states for s in snapshots if s.offset_seconds == 0)
        result = {field: central[field]-original[field] for field in increments}
        self.last_receipt["valid_time_shifting"] = receipt
        return result


def materialize_radar_batch(batch, xp):
    """Dense full-grid radar gate, from the existing dense or point contract."""
    if getattr(batch, "window", None) is not None:
        raise ValueError("spread radar window cannot define a full-domain clear-air gate")
    points = getattr(batch, "points", None)
    if points is None:
        return replace(batch, simulated=xp.asarray(batch.simulated),
                       values=xp.asarray(batch.values), mask=xp.asarray(batch.mask),
                       errors=xp.asarray(batch.errors))
    shape = batch.values.shape
    hx = xp.zeros((points.simulated.shape[0],)+shape, dtype=xp.float64)
    hx.reshape(hx.shape[0], -1)[:, xp.asarray(points.flat_index)] = xp.asarray(points.simulated)
    return replace(batch, simulated=hx, points=None, values=xp.asarray(batch.values),
                   mask=xp.asarray(batch.mask), errors=xp.asarray(batch.errors))


def inflate_observation_anomalies(observations, factor, xp):
    """Coherent linearized covariance inflation, including sparse H(x)."""
    from gpuwm.da.letkf import PointSet, PointSimulated
    output = []
    for batch in observations:
        points = getattr(batch, "points", None)
        original_backend = _xp(points.simulated if points is not None else batch.simulated)
        f = factor
        if batch.window is not None:
            j0,j1,i0,i1 = batch.window
            f = f[:,j0:j1+1,i0:i1+1]
        if points is not None:
            local_shape = f.shape
            f = f.reshape(-1)[xp.asarray(points.flat_index)]
            hx = xp.asarray(points.simulated)
        else:
            hx = xp.asarray(batch.simulated)
        mean = hx.mean(axis=0, keepdims=True)
        inflated = mean + (hx-mean)*xp.sqrt(f)[None]
        if original_backend is np and xp is not np:
            inflated = xp.asnumpy(inflated)
        if points is not None:
            changed = PointSet(points.flat_index, points.values, points.errors, inflated)
            output.append(replace(batch, points=changed,
                                  simulated=PointSimulated(changed, local_shape)))
        else:
            output.append(replace(batch, simulated=inflated))
    return output


@dataclass(frozen=True)
class TimeShiftSnapshot:
    offset_seconds: int
    states: Mapping[str, object]
    forecast_origin_seconds: int
    observation_cutoff_seconds: int


class TimeShiftBuffer:
    """Owned retained forecast slots, with bounded age and restart records."""
    def __init__(self, *, retention_seconds=2700):
        if retention_seconds < 1800:
            raise ValueError("time-shift retention must cover both adjacent 15-minute slots")
        self.retention_seconds = int(retention_seconds)
        self._slots = {}

    def retain(self, valid_seconds, states, *, forecast_origin_seconds,
               observation_cutoff_seconds):
        valid = int(valid_seconds)
        if valid in self._slots:
            raise ValueError("replacing a retained time slot could mix forecasts from different cycle origins")
        if forecast_origin_seconds > valid or observation_cutoff_seconds > valid:
            raise ValueError("retained forecast must not contain observations after its valid time")
        if not states:
            raise ValueError("a retained forecast slot must contain model fields")
        owned = {name:value.copy() for name,value in states.items()}
        self._slots[valid] = (owned, int(forecast_origin_seconds), int(observation_cutoff_seconds))
        newest = max(self._slots)
        for old in list(self._slots):
            if old < newest-self.retention_seconds:
                del self._slots[old]

    def snapshots(self, analysis_seconds):
        output = []
        for offset in (-900,0,900):
            valid = int(analysis_seconds)+offset
            if valid not in self._slots:
                raise ValueError(f"missing retained forecast slot at offset {offset}: time-shift covariance would be incomplete")
            states, origin, cutoff = self._slots[valid]
            output.append(TimeShiftSnapshot(offset,
                          {name:value.copy() for name,value in states.items()}, origin, cutoff))
        # Validate cutoffs and roster before returning a usable buffer view.
        time_shifted_covariance(output, analysis_seconds=analysis_seconds)
        return output

    def checkpoint(self):
        return {"schema": SCHEMA, "retention_seconds": self.retention_seconds,
            "slots": [{"valid_seconds":valid, "forecast_origin_seconds":origin,
                       "observation_cutoff_seconds":cutoff,
                       "states":{name:value.copy() for name,value in states.items()}}
                      for valid,(states,origin,cutoff) in sorted(self._slots.items())]}

    @classmethod
    def restore(cls, record, xp=np):
        if record.get("schema") != SCHEMA:
            raise ValueError("unknown time-shift buffer restart cannot safely resume covariance samples")
        buffer = cls(retention_seconds=int(record["retention_seconds"]))
        for slot in record["slots"]:
            buffer.retain(slot["valid_seconds"],
                {name:xp.asarray(value) for name,value in slot["states"].items()},
                forecast_origin_seconds=slot["forecast_origin_seconds"],
                observation_cutoff_seconds=slot["observation_cutoff_seconds"])
        return buffer


def time_shifted_covariance(snapshots, *, analysis_seconds, mode="vtsm"):
    """Retain -15/0/+15 minute forecasts and form explicit 3K samples.

    VTSM retains between-time phase covariance. VTSP recenters each time
    slab on the central mean. Neither is 3K independent forecast members.
    Return central-member indices so a caller keeps K trajectories, plus a
    receipt that makes the future-state observation cutoff reviewable.
    """
    if mode not in ("vtsm", "vtsp"):
        raise ValueError("valid-time-shift mode must be vtsm or vtsp")
    ordered = sorted(snapshots, key=lambda s: s.offset_seconds)
    if [s.offset_seconds for s in ordered] != [-900, 0, 900]:
        raise ValueError("valid-time shifting needs exactly minus 15, zero and plus 15 minute states")
    fields = set(ordered[1].states)
    if not fields or any(set(s.states) != fields for s in ordered):
        raise ValueError("time slots need the same fields or covariance would silently drop variables")
    first = next(iter(ordered[1].states.values()))
    count = first.shape[0]
    if count < 2:
        raise ValueError("time-shifted covariance needs at least two central trajectories")
    for slot in ordered:
        valid = analysis_seconds + slot.offset_seconds
        if slot.forecast_origin_seconds > valid or slot.observation_cutoff_seconds > min(valid, analysis_seconds):
            raise ValueError("time-shift snapshot uses observations or a forecast origin after its permitted cutoff")
    if len({s.forecast_origin_seconds for s in ordered}) != 1 or len({s.observation_cutoff_seconds for s in ordered}) != 1:
        raise ValueError("time-shift slots need a common forecast origin and observation cutoff to avoid mixing cycle ensembles")
    pooled = {}
    for field in sorted(fields):
        arrays = [s.states[field] for s in ordered]
        if any(a.shape != arrays[1].shape or a.shape[0] != count for a in arrays):
            raise ValueError("time slots must preserve member roster and grid geometry")
        xp = _xp(arrays[1])
        if mode == "vtsp":
            mean = arrays[1].mean(axis=0)
            arrays = [a-a.mean(axis=0)+mean for a in arrays]
        pooled[field] = xp.concatenate(arrays, axis=0)
    receipt = {"schema": SCHEMA, "mode": mode, "trajectory_members": count,
               "covariance_samples": 3*count, "analysis_seconds": analysis_seconds,
               "slots": [{"offset_seconds": s.offset_seconds,
                          "forecast_origin_seconds": s.forecast_origin_seconds,
                          "observation_cutoff_seconds": s.observation_cutoff_seconds}
                         for s in ordered],
               "central_indices": list(range(count, 2*count))}
    return pooled, receipt


def central_time_posterior(pooled_prior, pooled_increments, receipt):
    """Return K analysed trajectories, recentred on the full posterior mean."""
    count = int(receipt["trajectory_members"])
    output = {}
    for field, prior in pooled_prior.items():
        if field not in pooled_increments:
            output[field] = prior[count:2*count].copy()
            continue
        posterior = prior + pooled_increments[field]
        central = posterior[count:2*count]
        shifted = central-central.mean(axis=0)+posterior.mean(axis=0)
        if field in ("qv","qc","qr","qi","qs","qg","qh","nr","ni","nc","ns","ng","nh"):
            xp = _xp(shifted)
            shifted = scale_deviations(shifted,1.,gate=xp.ones(shifted.shape[1:],dtype=bool),lower=0.)
        output[field] = shifted
    return output
