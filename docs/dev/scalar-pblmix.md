# Scalar diffusion after MYNN

`scalar_pblmix=1` selects WRF's local implicit diffusion after the PBL
scheme. It is different from `bl_mynn_mixscalars=1`, which includes MYNN's
EDMF number fluxes. In WRF v4.6.1, `phys/module_pbl_driver.F:2251-2260`
calls `diff4d`; the number flux routines in `module_bl_mynn.F` do not
implement this option.

The port transcribes `diff4d`, `diff`, and `invert` from
`phys/module_pbl_driver.F:2598-2844`. The source SHA-256 is
`90336e30296991fb397ffde87649a4bd20eaa2b7dc6e90639b043810c8420b56`.
WRF's public-domain notice is in
`licenses/LICENSE-WRF-public-domain.txt` and the repository `NOTICE`.

For aerosol-aware Thompson, the active fields are cloud number, ice
number, water-friendly aerosol number and ice-friendly aerosol number.
WRF excludes precipitating number scalars, total-water scalar, graupel
volume and advected TKE from this diffusion. The port forwards `nc`, `ni`,
`nwfa`, and `nifa`; rain number `nr` receives no PBL diffusion tendency.

The solve consumes MYNN's final `exch_h` in m2/s at the lower interface of
each mass layer, layer depth in m, air density, number mixing ratio and
the PBL time step in s. Number fields do not undergo the water-species
specific-humidity conversion. The bottom flux is zero and the top-layer
value is prescribed. Rates are differences divided by the PBL time step,
then coupled to dry column mass by the physics driver. There is no added
positivity clamp. Elimination runs from the top downward before the
bottom-up substitution, preserving WRF `invert` operation order.

WRF's `share/module_check_a_mundo.F:2497-2511` disables `scalar_pblmix`
when MYNN's `bl_mynn_mixscalars=1` is selected. The engine rejects that
contradictory pair rather than silently changing either requested value.

The fixture builder extracts the three unmodified Fortran routines from
the pinned source and compiles them with `gfortran -O0 -fno-fast-math
-ffp-contract=off`. It uses eight 50-layer columns with zero, weak and
strong diffusion, surface and upper-level inversions, and 0.5, 15 and
300 s steps. Four active scalars produce 1,600 reference rates; seven
excluded fields retain 2,800 sentinel values. Build it with:

```sh
python tools/mynn_pbl_wrf461_oracle/build_scalar_pblmix.py \
  /path/to/WRF/phys/module_pbl_driver.F /path/to/owned/build
```

`tests/test_scalar_pblmix.py` compares every active CPU tendency bitwise
to the Fortran fixture. `tests/test_scalar_pblmix_gpu.py` applies the same
gate at several batch widths and checks the actual MYNN physics call
through three RK3 steps. The fixture receipt records source, harness and
output hashes, compiler identity and flags. These are implementation
checks, not observation-based forecast skill.
