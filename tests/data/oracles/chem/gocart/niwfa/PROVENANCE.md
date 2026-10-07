# get_niwfa oracle fixture

NOAA GSL's aerosol-aware Thompson coupling, `get_niwfa`, from
ufs-community/ccpp-physics `3e6660c6df54e95a0871e990c2294dd397ae3860`,
`physics/MP/Thompson/mp_thompson.F90` lines 1025-1068 (file sha256
`35b531fac3fa84bd26e5f0d9c436278954123f79f5c30a22944737a09db9d06d`,
Apache-2.0), lifted by exact line range into a harness module with
`kind_phys = 8` and compiled at `-O0` by

    bash tools/chem_wrf471_oracle/gocart/build_niwfa.sh <ccpp-physics> <build> <fixtures>

on Linux (GNU Fortran 15.2.0, glibc 2.43), 2026-09-30.  The driver is
`tools/chem_wrf471_oracle/gocart/niwfa_driver.F90`: six columns (clean,
continental, polluted, two dust-storm magnitudes, sulfate and OC only) of
five levels, aerosol mass in kg/kg rounded to the float32 values the port is
handed.  Outputs are recorded as the float32 the engine stores and as the
raw float64 bit pattern (`*_f64_bits`, two int32 words per value, low word
first).  `oracle-sha256sums.txt` pins every file.
