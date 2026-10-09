"""Per-term WOOF dumps inside _add_slow_tendencies (wraps the launches dycore calls by module name)."""
import cupy as cp, numpy as np
from types import SimpleNamespace
import gpuwm.core.dycore as D
import woof_hook as H
from gpuwm.core.state import DomainState
def find_state(args, kw):
    for x in list(args) + list(kw.values()):
        if isinstance(x, DomainState):
            return x
for name, tag in (("launch_flux_div_scalar", "k4_advt"), ("launch_flux_div_u", "k1_advu"), ("launch_flux_div_v", "k2_advv"),
                  ("launch_flux_div_w", "k3_advw"), ("_launch_slow_pgf", "k6_hpg"), ("_launch_slow_buoyancy", "k7_buoy"),
                  ("_launch_slow_geopotential", "k5_rhsph"), ("add_coriolis_curvature", "kA_corcurv")):
    orig = getattr(D, name)
    def make(orig=orig, tag=tag):
        def w(*a, **k):
            r = orig(*a, **k)
            st = find_state(a, k)
            if st is not None and getattr(st, "_loc_dt", None) is not None and H.active(st, SimpleNamespace(dt=st._loc_dt)):
                H.dump(st, tag)
            return r
        return w
    setattr(D, name, make())
