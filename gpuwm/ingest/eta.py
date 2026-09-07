"""Metadata binding for WRF's shared Rust automatic eta generator."""
from __future__ import annotations

import math

# WRF 4.6.1 Registry.EM_COMMON, domains generator scalars. These are
# algorithm defaults, independent of the source model or target physics.
WRF_ETA_DEFAULTS = {'auto_levels_opt': 2, 'max_dz': 1000.0, 'dzbot': 50.0,
                    'dzstretch_s': 1.3, 'dzstretch_u': 1.1}


def wrf_automatic_eta_requested(levels):
    """Recognize WRF's automatic-grid marker after its REAL conversion.

    WRF4.6.1 module_initialize_real.F:7590 selects explicit levels only
    when ABS(eta_levels(1)+1.) exceeds 1e-7. Near -1 the subtraction is
    exact; round the input to binary32 before applying that test.
    """
    if not levels or not (-2.0 < levels[0] < 0.0):
        return False
    import struct
    first = struct.unpack('<f', struct.pack('<f', levels[0]))[0]
    return abs(first + 1.0) <= 1e-7


def wrf_eta_options(domains):
    """Resolve scalar namelist values without losing explicit selections."""
    options = {}
    for name, default in WRF_ETA_DEFAULTS.items():
        raw = domains.get(name, [default])
        if len(raw) != 1 or isinstance(raw[0], bool) or not isinstance(raw[0], (int, float)):
            raise ValueError(f'{name} requires one numeric value')
        value = raw[0]
        if not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
        if name == 'auto_levels_opt':
            if not isinstance(value, int) or value not in (1, 2):
                raise ValueError('auto_levels_opt must be integer 1 or 2')
        options[name] = value
    return options


def generate_wrf_eta(e_vert, *, domains, p_top, base_temp, cpu_bridge=None):
    """Return the native full-level array and its resolved control receipt."""
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    options = wrf_eta_options(domains)
    import hashlib
    backend = CpuPreprocessBackend(cpu_bridge)
    eta = backend.generate_wrf_eta(e_vert, p_top=p_top, base_temp=base_temp, **options)
    with backend.path.open('rb') as stream:
        bridge_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt = {'algorithm': 'WRF-4.6.1-compute_eta-portable-exp-v1', 'e_vert': e_vert,
               'native_bridge_sha256': bridge_sha, 'native_bridge_abi': backend.abi_version,
               'eta_f32_sha256': hashlib.sha256(eta.tobytes()).hexdigest(),
               'p_top_requested': p_top, 'base_temp': base_temp,
               'controls': options,
               'explicit_controls': [key for key in options if key in domains]}
    return eta, receipt
