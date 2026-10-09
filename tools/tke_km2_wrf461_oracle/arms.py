"""Namelist arms of the km_opt=2 column oracle (plain data, no imports)."""

#: Namelist arms run on every case: (name, mix_isotropic, isfflx, c_k,
#: tke_drag_coefficient, tke_heat_flux).  "seed" is isfflx=0 with both
#: prescribed fluxes off, the only arm where tke_km's 1e-6 seed is live.
ARMS = (
    ("iso0_sfx0", 0, 0, 0.10, 0.0013, 0.24),
    ("iso1_sfx1", 1, 1, 0.15, 0.0, 0.0),
    ("iso0_sfx2", 0, 2, 0.10, 0.0, -0.02),
    ("iso1_seed", 1, 0, 0.10, 0.0, 0.0),
)
