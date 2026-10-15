"""CuPy-free snow admission shared by MYNN and registry generation."""
from gpuwm.microphysics_schemes import NAMED_SCHEMES as _NAMED_MP_SCHEMES

MYNN_SNOW_MICROPHYSICS = frozenset((2, 4, 6, 8, 9, 10, 16, 18, 24, 28, 30, 106)) | frozenset(
    _s.mp_id for _s in _NAMED_MP_SCHEMES.values()
    if "qs" in _s.ice_mass_species)
