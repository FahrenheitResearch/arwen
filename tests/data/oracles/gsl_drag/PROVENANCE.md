# GSL drag suite (gwd_opt = 3), WRF v4.6.1 column oracle

`gsl_drag_wrf461.npz` holds the inputs and every output of WRF v4.6.1's own
`phys/module_bl_gwdo_gsl.F` (tag v4.6.1, commit
d66e442fccc04111067e29274c9f9eaccc3cef28, byte-unmodified; its sha256 is the
one the WRF v4.7.1 terrain-drag oracle pins, the file did not change between
the two releases).

Built by `tools/gsl_drag_wrf461_oracle/build.sh` on the W1 verification box
(TOOLCHAIN.txt: GNU Fortran 13.3.0, glibc 2.39, x86_64) at -O0 with scalar
libm (libm-symbols.txt), then folded by `pack.py`.  SOURCES.sha256 pins the
WRF files, HARNESS.sha256 the harness, RAW.sha256 every raw word file the
oracle wrote, FIXTURES.sha256 the packed file.

Keys are `"<grid>/<case>/<name>"`, Fortran index order kept ((ncol, nz) for
level fields):

* `A/inputs`: 192 columns on an HRRR-style 50-level eta grid (p_top 2000 Pa).
  Columns 1-40 are edge cases (calm, sub-1 m/s wind, winds on the axes and on
  the eight direction-sector boundaries, flat land, the variance caps exactly,
  2*VAR above the model top, KPBL = 1 and = nz, PBLH = 0, OA = -1 / OL = 0,
  subnormal form drag, BR = 0, reversal between the two lowest levels, frmax
  and efmax clamps, water with statistics, superadiabatic layers, U or V
  exactly zero at the small-scale scheme's kpbl2, negative-zero winds, a PBL
  above kpblmax); then nine weather families: convective day, stable night,
  near-neutral windy, blocking, ocean, high terrain (to 4.6 km), snow/ice
  arctic inversion, tropical moist, jet with a critical level.
* `A/dx<dx>`: 750, 1000, 1009, 1010, 3000, 3637, 3638, 5000, 9000, 12000,
  13000, 25000 m (every taper branch and both 0.01 thresholds), plus
  `A/dx3000_spp`, `A/dx5000_spp`, `A/dx13000_spp` with spp_pbl = 1.
* `B/inputs`: 64 family columns on a 40-level grid (p_top 5000 Pa);
  `B/dx3000`, `B/dx9000`, `B/dx15000`, `B/dx30000`, `B/dx9000_spp`.

Each case records RUBLTEN, RVBLTEN, RTHBLTEN after the call (the inputs carry
the PBL tendencies before it), the eight DTAUX3D/DTAUY3D components, the
eight column stresses, and `diags_agree` (the tendencies of a gwd_diags = 0
call on the same inputs are the same words).

VARIANTS.json measures two attribution builds against this oracle: WRF's own
gfortran flags (-O2 -ftree-vectorize -funroll-loops; the form-drag loop is
vectorized to libmvec expf/powf) and a correctly rounded atan2f.
