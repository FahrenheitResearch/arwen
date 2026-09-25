# real_em_461_theta_seam_west_strip.npz

A west-boundary strip cut from two stock WRF 4.6.1 `real.exe` runs of one
`met_em` set that differ only in `use_theta_m`. It is the file-backed proof
of what `real.exe` writes under each setting and the fixture for the
wrfinput/wrfbdy temperature seam tests in `tests/test_wrfinput_theta_records.py`.

## Producer

- WRF 4.6.1, `main/real.exe` from a stock build (git d66e442fccc04111067e29274c9f9eaccc3cef28,
  `ifx` with Intel MPI 2021.18 as `configure.wrf` and `ldd real.exe` state,
  `mpirun -np 4`), run on 2026-09-19.
- Inputs: `met_em.d01.1974-04-03_12_00_00.nc` (md5 recorded in the provenance
  receipt beside the run) and `met_em.d01.1974-04-03_18_00_00.nc`, a 12 km
  251 x 201 domain, Lambert conformal, 50 eta levels (49 mass levels),
  p_top 10000 Pa, hybrid_opt 2, etac 0.2, mp_physics 10, sf_surface_physics 2.
- Two arms with identical namelists except `&dynamics use_theta_m = 0` (dry)
  and `= 1` (moist). Both reported `SUCCESS COMPLETE REAL_EM INIT`.
- The extraction asserted, before writing, that the two arms' `wrfinput_d01`
  files are equal in every variable the strip carries (`T`, `QVAPOR`, `MU`,
  `MUB`, `MAPFAC_M`, `MAPFAC_U`, `MAPFAC_V`, `C1H`, `C2H`, `C1F`, `C2F`) and
  that the dry arm's `THM` equals its `T`, and that the two `wrfbdy_d01` files
  are equal in `QVAPOR_BXS`, `QVAPOR_BTXS`, `MU_BXS` and `MU_BTXS` at record 0.
  Variables outside that set were not compared by the extraction.

## What is in the strip

`west_east` 0..4 (the five boundary columns), `south_north` rows 90..105,
all 49 mass levels, `Time` 0, float32 as written.

| key | shape | source |
| --- | --- | --- |
| `T`, `QVAPOR` | (49, 16, 5) | wrfinput, both arms identical |
| `THM_moist` | (49, 16, 5) | wrfinput of the moist arm (`THM` of the dry arm equals `T`) |
| `MU`, `MUB`, `MAPFAC_M` | (16, 5) | wrfinput |
| `MAPFAC_U` (16, 6), `MAPFAC_V` (17, 5) | | wrfinput |
| `C1H`, `C2H` (49), `C1F`, `C2F` (50) | | wrfinput |
| `T_BXS_dry`, `T_BTXS_dry`, `T_BXS_moist`, `T_BTXS_moist` | (5, 49, 16) | wrfbdy record 0, WRF order (width, z, y) |
| `QVAPOR_BXS`, `QVAPOR_BTXS` | (5, 49, 16) | wrfbdy record 0, both arms identical |
| `MU_BXS`, `MU_BTXS` | (5, 16) | wrfbdy record 0, both arms identical |

Global attributes of the moist arm's wrfinput: `TITLE = OUTPUT FROM REAL_EM
V4.6.1 PREPROCESSOR`, `USE_THETA_M = 1`, `HYBRID_OPT = 2`, `ETAC = 0.2`.

## What the files show

- `T` is dry theta-300 under both settings (identical arrays in both arms).
- `THM_moist` equals `(T + 300) * (1 + Rv/Rd * QVAPOR) - 300` bit for bit
  (`Rv/Rd` as the FP32 quotient 461.6/287.0).
- `T_BXS_moist` equals `THM_moist * (C1H*MU + (C1H*MUB + C2H))` bit for bit;
  `T_BXS_dry` equals the same coupling of `T`.

This is WRF's own definition (Registry.EM_COMMON:209-211: `th_phy_m_t0` is
written as `T`, the prognostic `t` as `THM`; module_initialize_real.F, WRF
4.6.1 lines 4888-4909: `T` saved dry, `t_2` converted to moist theta when
`use_theta_m = 1`; main/real_em.F:872: `T_BXS` couples `t_2`).

Full-file md5 sums, the namelist and the extraction script live in the lane
proof folder (`real_em_461_theta_seam_west_strip.provenance.json`).
