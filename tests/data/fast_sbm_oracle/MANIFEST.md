# Fast SBM Fortran oracle fixtures

These files are the reference outputs of WRF v4.8.0's fast spectral bin microphysics (mp_physics 30), compiled from the pinned WRF source and run on the build node. The float64 host port in `gpuwm/core/fast_sbm/` is gated against them. Read them only through `tools/fast_sbm_wrf480_oracle/fixtures.py`, which checks each file's SHA-256 before it returns anything. `MANIFEST.json` holds every number quoted here.

## How they were made

Compiler: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0 on weather-node-1 (Intel(R) Core(TM) Ultra 7 270K Plus), 2026-09-24T20:45:34Z, CPU only. The whole harness ran in 186 seconds.

Two builds are packed. O64 promotes every default real to float64 (`-cpp -DBUILD_SBM_FAST=1 -DIWORDSIZE=4 -ffree-form -ffree-line-length-none -O0 -ffp-contract=off -freal-4-real-8`) and is the tight gate for the float64 host. O32 is WRF as it runs (`-cpp -DBUILD_SBM_FAST=1 -DIWORDSIZE=4 -ffree-form -ffree-line-length-none -O0 -ffp-contract=off`) and is compared through an envelope. A third build at WRF's production optimisation (`-O2 -ftree-vectorize -funroll-loops`) is evidence only and is never gated.

The WRF sources were LF-normalized and refused unless their SHA-256 matched the pins: `phys/module_mp_fast_sbm.F` fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e and `phys/module_mp_SBM_polar_radar.F` dffddfbbca15b575aa2948482f34ce996896657b2b0809d3e3746c1c5a1bdf40. The table archive SBM_input_33.tgz (1339105 bytes, 25ed63ab3550aed110ae24df229faece20eafedfdf21cde01d8973826a7294bf) and its 18 members were checked the same way.

Two source transforms let the runners reach private routines and record inside FAST_SBM. The access transform deletes exactly one line, line 3862 (` PRIVATE` of module module_mp_fast_sbm). The probe transform adds 58 CALL lines and nothing else. The controls prove neither changes the answer: over the whole KC-1 run the pristine, access-transformed and probe builds give byte-identical FAST_SBM outputs in each precision, two runs of the same build are byte-identical, and each build replayed on its own recorded pre-call states in a fresh process gives back its own post-call states byte for byte, so FAST_SBM carries no state between calls that the replay records miss.

## What is in them

Every file is one record stream: each record is an int32 type tag followed by that type's fields, raw little-endian, arrays in Fortran order. O64 files hold only float64 and int32 fields; O32 files hold float32 where WRF declares default real and float64 where WRF declares REAL(KIND=8).

* `tables_*`: every table as read from `SBM_input_33` and the derived tables after FAST_HUCMINIT with a 12 s step.
* `kernels_*`: the collision kernels after Kernals_KS at ten pressures from 1000 to 250 hPa.
* `sweep_<routine>_*`: one family per process routine, inputs exactly as passed and the outputs the routine returned.
* `refusals`: six inputs that stop WRF, the input records, the message and the exit status, for both builds.
* `kc1_case`: the warm kinematic column case, defined in float64.
* `kc1_summary_*`: the state after every call; `kc1_replay_*`: full pre-call and post-call states at 16 steps; `kc1_probe_*`: records inside FAST_SBM and COAL_BOTT_NEW at 6 steps and 8 levels per column.
* `kc1_crossreplay_o64_on_o32`: the O64 build run once on each O32 pre-call state.
* `kc1_sensitivity_o64`: KC-1 again with every initial theta one ULP higher.

## The warm column KC-1

Four columns are called together, so the cross-column behaviour of FAST_SBM is inside the fixture. c1 is a land column with a 2 m/s updraft for 15 minutes, c2 the same over water, c3 still air with Marshall-Palmer rain above 1.2 km, and c4 a weak 0.3 m/s updraft at 850 hPa, last in the tile. The step is 12 s and the run is 300 steps (one hour). Between calls theta, qv and every bin are advected by a first-order upwind scheme.

Two choices differ from the design text and are recorded. The advection is the flux form with its divergence term restored, because the plain flux form with a fixed density empties the lowest level (it cools c1's lowest level to 55.8 K in 15 minutes). And c4 carries a moist cap, 99 percent relative humidity above 2.8 km, because of the cloud-base finding below: without it the cloud-base activation branch is never reached.

Acceptance, both builds (columns c1 to c4):

* O64: lowest temperature 281.93 K, 282.06 K, 284.82 K, 277.75 K; surface rain 5.22 mm, 5.63 mm, 2.62 mm, 4.38e-14 mm; cloud-base activations [0, 0, 0, 1], in-cloud activations [3953, 3911, 0, 2386]; ONECOND1 evaporation calls [674, 1053, 0, 1651]; collisional breakup [6969, 3957, 2171, 0]; spontaneous breakup [10331, 10762, 11967, 0].
* O32: lowest temperature 281.93 K, 282.06 K, 284.82 K, 277.75 K; surface rain 5.22 mm, 5.63 mm, 2.62 mm, 3.6e-14 mm; cloud-base activations [0, 0, 0, 1], in-cloud activations [3819, 3908, 0, 2117]; ONECOND1 evaporation calls [674, 1053, 0, 1653]; collisional breakup [6971, 3957, 2171, 0]; spontaneous breakup [7707, 6695, 11055, 0].

## Findings from the oracle

* D9_cloud_base_eqv_precedence: FAST_SBM's cloud-base test at :4387 joins its terms with .and. around 'K_found .eqv. .FALSE.'; .EQV. binds more loosely than .AND., so the test is (SUP_WATER > 0 .and. w > 10 cm/s .and. K_found) .eqv. (.FALSE. .and. ...), whose right side is always false. KZ_Cloud_Base therefore ends as the highest level that is not (supersaturated with w above 10 cm/s), or 1, instead of the lowest level above the second that is. With the design's sounding the top level never saturates, so KZ is level 40 in every column at every call (the probe records show it) and the cloud-base branch never ran in 300 steps; with the c4 moist cap it runs once, at step 40, level 40, column c4, in both builds.
* still_air_has_no_diffusional_growth: Condensation and evaporation are driven only by the change of T and qv since the previous call (DIFFU = 0 when DIV1 equals DIV3, :4615). In a column with no advection since the last call (c3, still air) no drop grows or evaporates, even rain falling into 60 percent humidity: ONECOND1 is called 0 times in c3.
* spontaneous_breakup_bin: Spontanous_Init sets ikr_spon_break to the first bin with radius at least 0.3 cm, which on the 33-bin grid is bin 33 (3.25 mm). Spont_Rain_BreakUp does work only when bin 33 holds any positive value; in the float64 build that is almost every rainy call, in the float32 build noticeably fewer (the counts differ by build).
* falflux_bare_stop: FALFLUXHUCM_Z stops the run with a bare STOP when every fall speed is zero (:5360): the process exits with status 0 and no message (refusal case 6).

## Evidence only: WRF's production flags

The native build at WRF's production optimisation is not byte-identical to the -O0 native build (first difference at step 1). Surface rain after one hour: 5.2194, 5.6314, 2.6152, 0.0000 mm against 5.2194, 5.6314, 2.6152, 0.0000 mm. This is recorded, never gated.

## Milestone 2: the cold half

The same harness adds the ice processes. Twelve cold sweep families run in the probe build of each precision: FREEZ, J_W_MELT, ICE_NUCL (reached through JERNUCL01_KS, since ICE_NUCL is private to its module; the probe build records its own call), JERNUCL01_KS with ice, the ice branches of JERRATE_KS, JERTIMESC_KS and JERDFUN_KS, ONECOND2, ONECOND3 (with the JERSUPSAT_KS call inside each read back), coll_xyx_lwf, coll_xyy_lwf and coll_xyz_lwf, modkrn_KS CASE 11 (with INTERPOL_SE), and cold COAL_BOTT_NEW, which also runs in the graupel build. The 143 live cases of the 09-13/14 port carry over with their grids: FREEZ, J_W_MELT, ICE_NUCL, JERNUCL01_KS and modkrn_KS on the real tables, and its 24 collision cases, with their random non-symmetric kernels, three ways: on their synthetic grid with y = 1.25 x (the orientation cases, because WRF's five mass grids are equal and a swapped x and y cannot show on them), on the synthetic y = x, and on the real tables (x = y = xl_mg with the module's chucm and ima). Its draws came from numpy; here a Park-Miller generator started at the old seeds draws them, and every draw is in the records. In the cold COAL_BOTT_NEW records colleff is WRF's value only where colleff_defined is 1; elsewhere WRF returns an uninitialised value (D6 and -40 C) and the record holds 0.

The graupel variant is built from a third transform that changes only line 3886, `hail_opt = 1` to `hail_opt = 0`, applied to the pristine source and to the probe source; each is refused unless that line is its whole diff. The probe transform now adds 58 CALL lines: milestone 1's and the cold anchors in FAST_SBM, JERNUCL01_KS, ONECOND2, ONECOND3 and COAL_BOTT_NEW.

Every milestone 1 stream came out of the extended harness byte for byte as packed before (37 streams compared).

### The cold refusals

* Case 1, JERNUCL01_KS/ICE_NUCL: -15 C with a habit-3 grid whose first mass is the smallest subnormal of the build's real: 1/DX overflows and the concentration check sees an infinite change. O64: bare STOP 099 at :3463; O32: returned. FAST_SBM cannot reach this input.
* Case 2, JERDFUN_KS/JERNEWF_KS: a negative snow bin and no growth. O64: wrf_error_fatal at :2750; O32: wrf_error_fatal. FAST_SBM cannot reach this input.
* Case 3, ONECOND2: no vapour (DEL1N = DEL2N = -1) with snow: RW and PW are NaN. O64: wrf_error_fatal at :7441; O32: wrf_error_fatal. FAST_SBM cannot reach this input.
* Case 4, ONECOND2: a negative substep at 10 percent ice supersaturation (KCOND 21). O64: wrf_error_fatal at :7452; O32: wrf_error_fatal. FAST_SBM cannot reach this input.
* Case 5, ONECOND2: 200 percent ice supersaturation over dense snow for 3600 s: a deposition heating above 5 K. O64: wrf_error_fatal at :7694; O32: wrf_error_fatal. FAST_SBM cannot reach this input.
* Case 6, ONECOND3: no vapour with drops and snow: RW and PW are NaN. O64: wrf_error_fatal at :7993; O32: wrf_error_fatal. FAST_SBM cannot reach this input.
* Case 7, ONECOND3: a negative substep. O64: wrf_error_fatal at :8049; O32: wrf_error_fatal. FAST_SBM cannot reach this input.
* Case 8, ONECOND3: 200 percent water supersaturation over a dense cloud and snow for 3600 s: a heating above 5 K. O64: wrf_error_fatal at :8239; O32: wrf_error_fatal. FAST_SBM cannot reach this input.
* Case 9, COAL_BOTT_NEW: a flat, very dilute drop spectrum over a 1e4 s step at 263 K: the gmin floor of coll_xxx_lwf adds more than 5 percent to the drops. O64: wrf_error_fatal at :8599; O32: wrf_error_fatal. FAST_SBM cannot reach this input.
* Case 10, COAL_BOTT_NEW: a dense cloud with drizzle rimed by much hail over 600 s at 400 hPa: a riming heat above 5 K. O64: wrf_error_fatal at :8636; O32: wrf_error_fatal. FAST_SBM cannot reach this input.
* Case 11, COAL_BOTT_NEW: a negative snow bin at 288 K: the snow check at the end of the step. O64: wrf_error_fatal at :8614; O32: wrf_error_fatal. FAST_SBM cannot reach this input.
* Case 12, ONECOND2: a negative substep at 10 percent below ice saturation (KCOND 20). O64: wrf_error_fatal at :7535; O32: wrf_error_fatal. FAST_SBM cannot reach this input.
* Case 13, JERNUCL01_KS/ICE_NUCL: -15 C at 60 percent ice supersaturation with a finite mixed-sign habit-3 spectrum whose concentration terms are about 7.5, +1e17 and -1.00001e17 per cm3: rounding in the real(8) sum turns the 1.106 per cm3 of nucleation into a change of 16. O64: bare STOP 099 at :3463; O32: bare STOP 099. FAST_SBM cannot reach this input.

### The cold column KC-2

A 3 x 3 tile, i and j 2..4, in the corner of a 4 x 4 domain (ids..ide and jds..jde 1..5), memory 0..6, called together in one FAST_SBM call; the pre-pass reads the halo row j = 1 and column i = 1 and takes its IDE-1 and JDE-1 edge branches at i or j = 4. Four tile columns carry the cold physics:

* A (2,2): deep mixed-phase updraft: surface 300 K, 7 K/km, w1 = 4 m/s sin(pi z / H) for the whole hour; RH 92 percent below 1.5 km, 85 percent to 0 C, ice-saturated to 11 km, 70 percent of ice saturation above.
* B (3,3): Hallett-Mossop updraft: 298 K, 6.5 K/km, w1 = 1.5 m/s for 1800 s; a water-saturated cloud layer 3.5-6.5 km holding a supercooled cloud (250 cm-3, 9 um) with a 2 mm/h Marshall-Palmer tail from 3.8 to 6 km, dense ice (bins 12-26) from 4.2 to 5.5 km and snow (bins 8-28) from 5 to 6.5 km.
* C (4,4): melting stratiform column: 275.5 K (0 C near 390 m), 6 K/km, w1 = 0.2 m/s; snow over every bin between 1.5 and 5 km (ramped over the lowest kilometre), ice-saturated to 5.5 km; the last tile column.
* D (4,2): hail shaft: 299 K, 7 K/km, w1 = -0.8 m/s (a downdraft); large dense ice (bins 18-33) between 5 and 8 km, a supercooled cloud (150 cm-3, 7 um) between 4 and 6.5 km and light snow between 6 and 8 km.
* environment: the other five tile columns and the seven halo columns: still, no hydrometeors, surface 299 K falling 0.8 K per column in x and 0.5 K in y, 6.8 K/km, RH 75/60/45 percent below 1.5 km, to 4 km and above, falling 4 percent per column in y.

50 levels, lowest layer 134 m, each layer 2.5 percent deeper than the one below, top 13.06 km; w on the 51 full levels, zero at the bottom and top. The step is 12 s and the run 300 steps. The winds: u = 6 + 1.5 (i-1) + 0.5 (j-1) + 0.1 k m/s, v = -2 + 0.5 (i-1) - (j-1) + 0.05 k m/s, prescribed and constant; they enter only the ice pre-pass. Replay steps 2, 12, 30, 60, 120, 200, 300; probe steps 10, 60, 150, 300 on levels 5, 12, 18, 23, 27, 31, 36, 42 of the four cold columns. The graupel build runs the same case with a replay at steps 2, 150 and 300.

Acceptance, asserted in both precisions (sums over the tile):

* O64: temperatures from 198.2 K to 299.5 K; ICE_NUCL called in the six bands 7085, 6978, 7003, 6877, 5783, 13129 times; homogeneous freezing 349; ONECOND1, 2 and 3 54545, 37807 and 31214 calls; KCOND 30, 31 and 32 4244, 11616 and 15354; Hallett-Mossop splinters 1864; FREEZ to plates 14963 and to hail 7571; D11 4668 levels; surface snow up to 0.857 mm and hail up to 0.704 mm. Graupel build: surface graupel up to 0.0664 mm, Hallett-Mossop 1968.
* O32: temperatures from 198.2 K to 299.5 K; ICE_NUCL called in the six bands 7088, 6979, 7003, 6876, 5784, 13128 times; homogeneous freezing 409; ONECOND1, 2 and 3 54537, 37865 and 31212 calls; KCOND 30, 31 and 32 4244, 11613 and 15355; Hallett-Mossop splinters 1821; FREEZ to plates 13389 and to hail 5212; D11 4213 levels; surface snow up to 0.854 mm and hail up to 0.704 mm. Graupel build: surface graupel up to 0.0664 mm, Hallett-Mossop 1923.

The low-edge run: the KC-2 domain (ids..ide and jds..jde 1..5, memory 0..6) with the tile moved to i, j = 1..3, called together in one FAST_SBM call, so the pre-pass takes its I = 1 and J = 1 branches: DERIVT_X and DERIVT_Y (:4422-4423, :4429-4430) and the DSUPICE_XYZ differences (:4460-4462, :4468-4469, :4483-4485, :4491-4492); the halo is the column i = 4 and the row j = 4. KC-2's columns (A at (2,2) and B at (3,3) in the tile; D at (4,2) and C at (4,4) in the halo) plus a second Hallett-Mossop column, type B, in the corner (1,1): the DSUPICE_XYZ branch (:4441) needs 2 percent ice supersaturation below -5 C at the point itself, which KC-2's environment columns never reach; every other column as in KC-2. 6 calls of 12 s from the KC-2 initial state, advected between calls as in KC-2; replay records (the static fields, then pre and post) at steps 2, 4 and 6, and the pre-pass arrays of the tile at the same steps from the probe build (no per-column probe records). In both precisions the corner's pre-pass values are nonzero on the I = 1 and J = 1 branches (DSUPICE_XYZ at 35 (O64), 35 (O32) recorded points of the column i = 1) and ICE_NUCL runs in the corner.

Every live cold collision call ran in both precisions: calls 1, 3, 4, 10, 11, 12 in the hail build and 2, 3, 5, 6, 7, 12 in the graupel build (numbered in source order, :8452 to :8563); the others never ran.

### Findings from the cold oracle

* ice_nucl_stop_099_unreachable: ICE_NUCL stops with a bare STOP 099 (:3463) when a habit's crystal concentration, the running real(8) sum of 3 COL PSI2 xi over all 33 bins (:3375-3381, :3446-3452), changes by more than 10 per cm3 in one call. Nucleation adds DELTAF, at most HELEK1 = C1 exp(A1 + B1 DELSUPICE_MAX) = 1.106 per cm3, to bin 1 of one habit, and only when that habit's concentration is at most HELEK1 (:3435). With nonnegative bins that bounds every term, so the change stays at most 1.106. Finite bins of mixed sign reach the stop through rounding in the sum: a term of +1e17 (where the sum's spacing is 16) and a slightly larger negative term keep the concentration negative, and a bin-1 term just under 8 is absorbed before nucleation and adds 16 after it (refusal case 13 stops in both builds). A non-finite change reaches it too: a subnormal first grid mass makes 1/DX overflow (refusal case 1, the promoted build only; from a float32 grid the product cannot underflow in real(8), so the native build returns). FAST_SBM cannot reach the stop because it passes JERNUCL01_KS no crystals: FF2R is zeroed at every point (:4548-4550) and again after every nucleation call (:4667-4674), so FF2IN is zero on every call. The only crystals ICE_NUCL can then see are drops that JERNUCL01_KS's homogeneous freezing (:3101-3110) put into habit 2 in the same call, which happens only when its latent heat lifts the point from -38 C or below to -37 C or above; the nucleating habit is then habit 1 (below -22.4 C), which is zero, and habit 2 does not change.
* modkrn_unset_at_minus_40: modkrn_KS CASE 11 (:1481-1498) sets its output through INTERPOL_SE (:1362-1393), which assigns nothing when H equals its last node (-40 C): every test is strict and the last node is the smallest. At exactly -40 C the output keeps whatever the caller held (-1 in the sweep), as above 0 C, where modkrn_KS returns first (:1416-1417, finding D6). FAST_SBM calls COAL_BOTT_NEW for TT >= 233.15 K, so a point at exactly -40 C collides snow with snow using an undefined PRDKRN, and COAL_BOTT_NEW returns CollEff = PRDKRN (:8322) undefined there too.
* jerdfun_negative_bin_with_growth: JERNEWF_KS's Kovetz-Olund remap (:2537-2573) moves only bins with FI > 0, so when any bin grows a negative bin is dropped without a refusal (the jerdfun_ice sweep's set 3). Its negative check (:2735-2751) fires when nothing grows (refusal case 2), and also when a bin's new mass passes the top of the remap's grid, RRS(NRX+1) = 1024 RRS(NRX) (:2515): the bracketing search then stops at the top bin, GMAT = (RRP - RN)/(RRP - RRTMP) is negative (:2564-2565) and the top bin can go negative. That needs a bin to grow by more than a factor of 1024 in mass in one call, which is extreme but not excluded. JERDFUN_KS's own check (:2426-2429) tests the same values after that check and so can never fire.
* deldrop_reached_only_through_the_gmin_floor: In COAL_BOTT_NEW the drops lose mass in every cold collision, and the breakup step keeps a gain of at most 0.1 percent; the only other source of drop mass is coll_xxx_lwf's floor, which raises a bin left below gmin = 1e-13 to gmin. The deldrop check (:8598) fires only when those floors add more than 5 percent, which needs a very dilute spectrum and a very long step (refusal case 9).
* snow_breakup_guard_always_true: The snow-breakup guard (:4836) sums the slice FF3R(KR_SNOW_MIN:NKR), empty on the 33-bin grid (KR_SNOW_MIN = 34), so it is 0 > (33 - 34) 1e-30 and true at every point; BreakUp_Snow is then skipped because KR_SNOW_MAX = 35 > NKR. KC-2's counts show the guard true at all 15000 points of every tile column and no call.
* onecond2_sign_flip_checks_unreachable: ONECOND2's two sign checks (:7522-7525 and :7608-7611) cannot fire: its drop rates are zero, so JERSUPSAT_KS takes the ice-only branch (or the no-rate branch), where DEL2N = DEL2 exp(-PI DT) keeps DEL2's sign, the dynamic terms being zero on every WRF call.
* supsat_small_ratio_branches_unreachable_from_onecond3: JERSUPSAT_KS's branches 8 and 9 (finding S1) need RI/RW at least 1e12 (or at most 1e-12) while the only-ice and only-water tests pass. In ONECOND3 RI/RW and PI/PW are the same ratio of ice to drop relaxation rates up to factors between 1 and 1.2, so for nonnegative spectra a ratio of 1e12 is sent to the only-ice (or only-water) branch first. Over ONECOND2 and ONECOND3, KC-2 reaches branches 0, 2 and 4 to 7 in both builds and precisions and never 8 or 9 (kc2.catalogue_counts); S2 (branch 7) is reached by real ONECOND3 calls.
* graupel_build_dead_calls: In the graupel build FF5R is zero whenever COAL_BOTT_NEW runs (FREEZ's hail is merged into graupel after the collisions), so the drop-hail calls (:8524, :8534) never run; the two drop-graupel-to-hail calls (:8501, :8512) need ALWC of at least ALCR_G = 100 g/m3. In the hail build the graupel calls never run. KC-2 and KC-2g count each call site.
* anti_bergeron_regime: ONECOND3's KCOND 50 (:8006-8016) prints 'Anti-Bregeron Regime, No DIFFU' and returns without growth. It needs DEL1N > 0 > DEL2N or an exact zero. FAST_SBM's saturation fits (2.53e12 exp(-5420/T) over water, 3.41e13 exp(-6130/T) over ice) cross at 272.96 K, about -0.186 C, and below that the water saturation pressure is the larger, so DEL1N > 0 > DEL2N is excluded wherever ONECOND3 is dispatched (below -0.187 C, :7908); between the two it is not. The sweep reaches KCOND 50 with stated inputs, KC-2 never does.

Evidence only: the native build at WRF's production flags does not give the -O0 native build's KC-2 state byte for byte; its per-call surface precipitation and minimum temperatures first differ at step 161, and after one hour its surface rain differs by at most 0.00079 mm. This is recorded, never gated.

## Reproduction on a second compiler

The same harness was run under WSL Ubuntu 24.04 on the desktop (CPU only) with GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0. 134 of 134 raw streams, including the production-flag evidence build, came out byte-identical to the node-1 streams. This is recorded, not gated.

Total packed size: 18149071 bytes (limit 20971520).

WRF is public domain software from NCAR, operated by UCAR. The scheme is Alexander Khain's Hebrew University Cloud Model bin microphysics; FSBM-2 in WRF was coded and is maintained by Jacob Shpund (Shpund et al., 2019). The tables are NCAR's SBM_input_33 distribution.
