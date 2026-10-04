"""Held member scalar mixing on the original metric Smagorinsky/diff6 graph.

The parent fixed-tendency helper computes K from time-t fields, then calls
these scalar hooks at the original dependency boundaries. Scalar tendencies
retain their map factors; only dry momentum/theta carries the later divide.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np

from gpuwm.core.device_inventory import state_array_shapes
from gpuwm.ensemble.batch_state import BatchedDomainState, BatchStateUnsupported, _exact_key


def _planned_species(cfg):
    """Read the already priced transported scalar rows without importing CUDA."""
    from gpuwm.core.preflight import scratch_slot_registry
    fields = state_array_shapes(cfg)
    registry = scratch_slot_registry(cfg)
    return tuple(name for name in fields if name + '0' in fields and 'smag_r' + name in registry
                 and name not in ('u', 'v', 'w', 'thp', 'php', 'mup', 'tke'))


def workspace_specs(cfg, *, has_msf=True):
    """Scalar hooks borrow the combined fixed-mixing mass and flux workspace."""
    from gpuwm.ensemble.batch_mixing import workspace_specs as combined
    return combined(cfg, has_msf=has_msf)


def required_scratch_slots(cfg):
    """Extend the authoritative dry inventory with every scalar held buffer."""
    from gpuwm.core.preflight import scratch_slot_registry
    from gpuwm.ensemble.batch_mixing import required_scratch_slots as dry
    result = dict(dry(cfg))
    if cfg.moist and (cfg.km_opt == 4 or cfg.diff_6th_opt > 0):
        registry = scratch_slot_registry(cfg)
        for name in _planned_species(cfg):
            slot = 'smag_r' + name
            if registry[slot] != (cfg.nz, cfg.ny, cfg.nx):
                raise BatchStateUnsupported('scalar held-tendency registry has a different mass-grid shape')
            result[slot] = np.dtype('float32')
    return result


class _ScalarFixed:
    def __init__(self, *, state, check, held, horizontal, diffusion, entries, mass):
        self.state, self.check, self.held = state, check, held
        self._horizontal, self._diffusion = horizontal, diffusion
        self.numerical_entries = entries
        self.tendencies = MappingProxyType({name: array for name, array in held})
        self.mass = mass

    def clear(self):
        self.check()
        for _, array in self.held:
            array.fill(0)

    def horizontal(self):
        """Consume Km/Kh produced by the parent's preceding metric closure."""
        self.check()
        for array, flux, divergence in self._horizontal:
            array.fill(0)
            flux()
            divergence()
            from gpuwm.ensemble.batch_mixing import _zero_strips
            _zero_strips(array, self.state.cfg, 1)

    def diff6(self):
        """Add scalar-strength diff6 after the parent's dry diff6 calls."""
        self.check()
        for launch in self._diffusion:
            launch()

    def map_divide(self):
        """Scalar forward tendencies retain the original raw map coupling."""
        self.check()

    def __call__(self, *, coefficients_ready=False):
        """Run scalar rows alone only when the caller confirms current K."""
        if self.state.cfg.km_opt == 4 and coefficients_ready is not True:
            raise BatchStateUnsupported('standalone scalar mixing must follow the time-t Km/Kh closure')
        import cupy as cp
        from gpuwm.ensemble.batch_mixing import _array
        self.clear()
        cp.add(_array(self.state, 'mub2d'), _array(self.state, 'mup0'), out=self.mass)
        self.horizontal()
        self.diff6()


def prepare_scalar_fixed_tendencies(state, *, mass='mixing_mut'):
    """Bind scalar hooks; do not calculate or silently substitute the K fields."""
    if not isinstance(state, BatchedDomainState) or not state.cfg.moist:
        raise TypeError('scalar fixed tendencies require an admitted moist member state')
    from gpuwm.ensemble import batch_mixing as mixing
    from gpuwm.core.dycore import _clock_scaled_diff6_factor, diff6_exempt_slots
    from gpuwm.core.moist import moist_species
    cfg, _ = mixing._context(state)
    if cfg.km_opt != 4 and cfg.diff_6th_opt <= 0:
        return None
    names = moist_species(state)
    if set(names) != set(_planned_species(cfg)):
        raise BatchStateUnsupported('transported scalar and priced held-tendency inventories differ')
    held = tuple((name, mixing._array(state, 'scratch:smag_r' + name, output=True)) for name in names)
    total_mass = mixing._array(state, mass, output=True)
    if state.storage.specs[mass].shape != state.storage.specs['mup'].shape:
        raise ValueError('scalar diff6 mass carrier has a different column shape')
    horizontal = []
    entries = []
    if cfg.km_opt == 4:
        common, dims = mixing._common(state)
        common_fields = mixing._common_fields(state)
        kh = mixing._array(state, 'scratch:smag_kh', output=True)
        fx, fy = (mixing._array(state, 'scratch:' + slot, output=True) for slot in ('diff6_x', 'diff6_y'))
        for name, target in held:
            field = name + '0'
            slot = 'scratch:smag_r' + name
            flux = mixing._raw(state, 'smag2d', 'wrf_smag_flux_s', common_fields +
                (('f', field), ('kh', 'scratch:smag_kh'), ('thb', 'thb'),
                 ('fx', 'scratch:diff6_x'), ('fy', 'scratch:diff6_y')),
                common + (mixing._array(state, field), kh, mixing._array(state, 'thb'), np.int32(0),
                          np.int32(len(state.storage.specs['thb'].shape) == 3), fx, fy) + dims,
                ((cfg.nx + 1 + 127) // 128, cfg.ny + 1, cfg.nz))
            divergence = mixing._raw(state, 'smag2d', 'wrf_smag_hd_s', common_fields +
                (('fx', 'scratch:diff6_x'), ('fy', 'scratch:diff6_y'), ('tend', slot)),
                common + (fx, fy, target) + dims, mixing._grid(state.storage.specs['p'].shape))
            horizontal.append((target, flux, divergence))
            entries.extend(flux.numerical_entries + divergence.numerical_entries)
    diffusion = []
    if cfg.diff_6th_opt > 0:
        factor = _clock_scaled_diff6_factor(cfg)
        exempt = diff6_exempt_slots(cfg)
        for name, _ in held:
            slot = 'smag_r' + name
            if slot in exempt:
                continue
            launch = mixing._diff6_launch(state, (name + '0', slot, 'diff6_m', '', 'c1h', 'c2h'), factor, mass)
            diffusion.append(launch)
            entries.extend(launch.numerical_entries)
    from cupy.cuda import runtime
    device = int(runtime.getDevice())
    bound = tuple((name, id(array)) for name, array in state.storage.arrays.items())
    configuration = _exact_key(cfg)

    def check():
        if int(runtime.getDevice()) != device:
            raise ValueError('prepared scalar mixing belongs to another CUDA device')
        if bound != tuple((name, id(array)) for name, array in state.storage.arrays.items()):
            raise BatchStateUnsupported('scalar mixing backings changed; rebind before submission')
        if _exact_key(state.cfg) != configuration:
            raise BatchStateUnsupported('scalar mixing configuration changed; rebind its clock and boundary conversions')
    return _ScalarFixed(state=state, check=check, held=held, horizontal=tuple(horizontal),
                        diffusion=tuple(diffusion), entries=tuple(entries), mass=total_mass)
