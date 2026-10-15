# WSM3 (mp_physics = 3): where gpuwm differs from WRF v4.6.1, and why

The CUDA kernel `gpuwm/core/kernels/wsm3.cu` is a word-for-word port of
WRF v4.6.1 `phys/module_mp_wsm3.F`. It is graded against that file compiled
by gfortran 13.3.0 at -O0 (`tools/wsm3_wrf461_column_oracle`,
`tests/test_wsm3_wrf461_column_exact.py`). Everything not listed here is
WRF's own arithmetic, word for word.

## Declared divergences in default arithmetic

**D1. Cloud ice falling out of a warm lowest level is counted as rain.**
WRF adds the cloud-ice fallout `fallc(i,1)` to the surface only when the
lowest level is below 0 C (`module_mp_wsm3.F:604-607`). Ice has no fall
speed at a level at or above 0 C, but the semi-Lagrangian scheme's CFL
limiter (`nislfv_rain_plm`, the `decfl > con1` branch) moves the bottom
interface speed `wi(1)` off zero whenever ice falls into a warm lowest level
from a cold one above. That ice leaves the column and WRF counts it nowhere,
so water is lost. gpuwm adds `fallc(i,1)` to the rain total whatever the
lowest level's temperature; the snow total keeps WRF's cold-only arm, so a
cold lowest level gives WRF's words exactly. GPUWM_WRF_EXACT=1 retains
stock WRF rain accounting, including its warm-level loss, so strict grading
compares every field against the unmodified reference.

Measured on the oracle's 77 columns: WRF and a WRF build carrying only this
one moved line differ in 6 columns (one natural stratiform column with its
melting level near the ground, five edge columns), and only in RAINNC,
RAINNCV and SR. The default build is graded against both references. Per-field arithmetic
differences are recorded in the delivery receipts.

**D2. A melting level at the model top with rising air freezes at the top
level.** WRF sets the freezing level one layer above the melting level when
`w > 0` there (`:572-574`); when the melting level is the top level that is
`kte+1`, and the freezing step then reads `t(i,kte+1)` past the end of its
`(its:ite, kts:kte)` temperature array (`:582-592`), which is undefined
behaviour. gpuwm freezes at the top level instead. No oracle column reaches
it (every column's top level is below 0 C), so it moves no graded word.

**D3. Upper-cell sedimentation search.** The delivered WSM6/WDM6 repair is
used by WSM3's default arm. The search includes the last arrival interface,
uses the existing zero-density extension on fall-through and enforces
kt >= kb. GPUWM_WRF_EXACT=1 retains the stock search to compare against
unmodified WRF. The pure transport tests cover 2, 50 and 80 levels at
0.001, 1 and 20 s, including a planted predecessor search.

## gpuwm's own diagnostic

**REFL_10CM.** WRF computes none for WSM3 (the driver passes no
`refl_10cm`; `module_diag_nwp.F:549` "nothing to do here"). gpuwm's REFL_10CM
for mp=3 is WRF's `refl10cm_wsm6` evaluated on WSM3's own phase split of
`qr` (rain at or above 0 C, snow below, no graupel), whose rain intercept,
temperature-dependent snow intercept and snow density are WSM3's constants
(`gpuwm.core.refl.wsm3_precipitation_phases`).

## Coupling facts other components key on

* Registry package `wsm3scheme` (`Registry.EM_COMMON:3017`):
  `moist:qv,qc,qr; state:re_cloud,re_ice,re_snow`. F_QI, F_QS and F_QG are
  false; `module_physics_init.F:4482` leaves `warm_rain` false.
* Radiation: WSM3SCHEME is in the `use_mp_re` list
  (`module_physics_init.F:1008`), so RRTMG receives the scheme's three radii.
  WRF's radiation wrappers move cold `qc` into `qi` and cold `qr` into `qs`
  for a scheme without F_QI and without `warm_rain` before reading them
  (`module_ra_sw.F:193-203`). gpuwm publishes WSM3's flags; the Dudhia, RRTM
  and RTE+RRTMGP conversion keyed on them belongs to the radiation side and
  lands with the cold-species-conversion lane.
* Nest edges: same-scheme WSM3 nests resolve; a mixed edge with another
  scheme is refused by name (no phase-split closure exists for WSM3's two
  condensate classes).
