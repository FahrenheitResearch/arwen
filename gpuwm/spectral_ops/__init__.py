"""Level-2 regional spectral numerical operators for Arwen."""

from .adaptive import BandObservation, fit_hyperdiffusion
from .config import (ScalarTarget, SpectralNumericsConfig, WindControl,
                     from_mapping)
from .coupling import (blend_parent_child, nudge_large_scales, split_scales)
from .elliptic import solve_helmholtz, solve_poisson
from .hook import SpectralLargeStepHook
from .pins import PINS, PINS_SHA256, SCHEMA, registration
from .runtime import SpectralBudgetExceeded, apply_spectral_step
from .scalar import apply_transfer, hyperdiffuse
from .transfer import Hyperdiffusion, RaisedCosineLowPass
from .vector import (damp_c_grid_divergence, damp_divergence,
                     helmholtz_decompose)

__all__ = [
    "BandObservation", "Hyperdiffusion", "PINS", "PINS_SHA256",
    "RaisedCosineLowPass", "SCHEMA", "ScalarTarget",
    "SpectralLargeStepHook",
    "SpectralBudgetExceeded", "SpectralNumericsConfig", "WindControl",
    "apply_spectral_step", "apply_transfer", "blend_parent_child",
    "damp_c_grid_divergence",
    "damp_divergence", "fit_hyperdiffusion", "from_mapping",
    "helmholtz_decompose", "hyperdiffuse", "nudge_large_scales",
    "registration", "split_scales",
    "solve_helmholtz", "solve_poisson",
]
