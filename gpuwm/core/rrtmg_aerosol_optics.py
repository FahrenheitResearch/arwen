"""WRF ``aer_opt = 3``: Thompson aerosol optics for the legacy RRTMG shortwave.

Port authority: NOAA-EMC/HRRR tag v4.1.21 (the WRF fork operational HRRR v4
runs, ``sorc/hrrr_wrfarw.fd/WRFV3.9/phys/``, commit 40ee6058c):

* ``module_radiation_driver.F`` (sha256 7464639e...) builds the aerosol
  on a radiation step when ``F_QNWFA`` (the Thompson aerosol-aware scheme,
  mp_physics = 28) and ``aer_opt == 3`` and the shortwave is RRTMG
  (:950-1000): ``gt_aod`` (:4645-4832) turns the water-friendly and
  ice-friendly aerosol number mixing ratios (QNWFA, QNIFA; ``nwfa``/``nifa``
  here) into a layer optical depth at 550 nm, ``taod5503d``, from an
  RH-and-temperature lookup of per-particle extinction; then (:1883-1896)
  ``calc_aerosol_rrtmg_sw`` with the PARAMETERs ``taer_type = 1`` (rural),
  ``taer_aod550_opt = 2``, ``taer_angexp_opt = taer_ssa_opt =
  taer_asy_opt = 3`` (:769-773) spreads that over the 14 RRTMG shortwave
  bands and adds the single-scattering albedo and asymmetry;
* ``module_ra_aerosol.F`` (sha256 9931482a...): ``calc_aerosol_rrtmg_sw``
  (:409-802) with ``calc_relative_humidity`` (:1473-1503, Bolton's
  saturation pressure, a DIFFERENT RH from gt_aod's RSLF one),
  ``calc_spectral_aod_rrtmg_sw`` (:1057-1202, its 3-D branch),
  ``calc_spectral_ssa_rrtmg_sw`` and ``calc_spectral_asy_rrtmg_sw``
  (:1204-1418): Lagrange interpolation over the eight RH nodes of the
  Shettle-Fenn rural tables;
* ``module_mp_thompson.F`` (sha256 4d600111...) ``RSLF``;
* ``module_ra_rrtmg_sw.F`` (sha256 04e97b7e...) ``RRTMG_SWRAD``
  (:10930-10947): tauaer/ssaaer/asmaer start at 0/1/0 on every layer
  kts..kte+1, the model layers kts..kte take the driver's values, the
  layer above the model top keeps 0/1/0, and ``aer_opt = 3`` runs
  ``iaer = 10`` (:9183), so ``spcvmc_sw`` sees them unchanged.

The LONGWAVE is not touched by aer_opt = 3 in this fork: RRTMG_LWRAD fills
its tauaer from the chemistry arrays only (WRF_CHEM and aer_ra_feedback,
module_ra_rrtmg_lw.F:12538-12560), otherwise zero.

The diagnostics ``calc_aerosol_rrtmg_sw`` also writes (angexp2d, aerssa2d,
aerasy2d) feed no radiation and are not formed; ``taod5502d`` (the column
AOD at 550 nm) is returned beside the optics.

Prescribed smoke can supply the fork's extra layer AOD. The explicit
smoke_feedback path adds MIN(3, AOD3D_SMOKE) before the existing rural
spectral optics (radiation driver :984). Dry smoke mixing ratios use the
source 4.5 m2/g extinction and the receiving model's dry density and layer
thickness. The source's active conversion has no humidity enhancement.
This optical coupling does not implement smoke transport, emissions or
deposition. Missing selected input is refused, never replaced by clean air.

Range.  The shortwave engine forms ``pasya * pomga * ptaua / zomcc`` on
the hardware division (module_ra_rrtmg_sw.F:8448; the SW-audit invariant in
gpuwm/core/kernels/rrtmg_sw.cu requires every operand of that division to
be zero or a normal float32).  These optics meet it by construction:
gt_aod floors the two numbers at 1 and 0.01 per kg and its smallest table
entries are 5.73936e-15 and 2.63577e-12 m^2, so the layer AOD is at least
3.2e-14 times the layer mass ``dz8w * rhoa`` (kg m^-2); the Lagrange
factors stay above 0.058 (optical depth), 0.55 (single-scattering albedo)
and 0.61 (asymmetry) over the clamped RH range 0 to 99.  The numerator is
therefore at least 6.3e-16 times the layer mass: normal for any layer
heavier than 1.9e-23 kg m^-2, which every model layer exceeds by more than
twenty orders of magnitude, and exactly zero on the layer above the model
top.  tests/test_rrtmg_aerosol_optics.py measures the three factor floors
and the bound.

Aerosol type.  The HRRR fork fixes rural type 1.  The NOAA MPAS fork RRFS
v2 runs (RRFSx/MPAS-Model 1d6e370e) uses the same gt_aod, Bolton RH and
Lagrange bodies (physics_wrf/module_mp_thompson_aerosols.F gt_aod,
physics_wrf/module_ra_rrtmg_sw_aerosols.F calc_aerosol_rrtmg_sw, RSLF from
mpas_atmphys_functions.F) with the same PARAMETERs (mpas_atmphys_vars.F
:840-854), but its driver sets the type per column from the land/water
flag: 1 over land, 3 (maritime) over water.  ``aer_type`` selects that;
``rrtmg_aer3_type_source = "landmask"`` (gpuwm.config) reaches it through
the legacy RRTMG adapter.  tests/test_rrtmg_aer3_landmask_type.py holds the
twin and kernel to that fork's Fortran on a mixed land/water deck
(tools/rrfs_mpas_aer3_oracle).  The maritime rows meet the range argument
below too: their smallest Lagrange factors are measured by that test.

Two implementations, one statement order: the NumPy float32 functions here
are the oracle twin of ``gpuwm/core/kernels/rrtmg_aer3.cu``; both use glibc's
own ``expf`` (gpuwm.core.noahmp_libm / glibc_flt32.cuh, the function
gfortran calls for EXP on REAL(4)), and tests/test_rrtmg_aerosol_optics.py
holds both to the fork's Fortran word for word
(tools/hrrr_radiation_driver_oracle).
"""
from __future__ import annotations

import hashlib

import numpy as np

F = np.float32
NBNDSW = 14

# ---------------------------------------------------------------------------
# Tables, as the Fortran declares them (float32 rounding of the literals).
# ---------------------------------------------------------------------------

#: gt_aod rh_arr (module_radiation_driver.F:4662-4663).
GT_RH_ARR = np.array([10., 60., 70., 80., 85., 90., 95., 99.8], F)

#: gt_aod lookup_tabl(RH, temperature index, 1 = water-friendly /
#: 2 = ice-friendly), m^2 per particle (:4666-4736), stored [rh][t][species].
GT_LOOKUP = np.array([
    [[5.73936E-15, 2.63577E-12], [5.73936E-15, 2.63577E-12],
     [5.73936E-15, 2.63577E-12], [5.73936E-15, 2.63577E-12]],
    [[6.93515E-15, 2.72095E-12], [6.93168E-15, 2.72092E-12],
     [6.92570E-15, 2.72091E-12], [6.91833E-15, 2.72087E-12]],
    [[7.24707E-15, 2.77219E-12], [7.23809E-15, 2.77222E-12],
     [7.23108E-15, 2.77201E-12], [7.21800E-15, 2.77111E-12]],
    [[8.95130E-15, 2.87263E-12], [9.01582E-15, 2.87252E-12],
     [9.13216E-15, 2.87241E-12], [9.16219E-15, 2.87211E-12]],
    [[1.06695E-14, 2.96752E-12], [1.06370E-14, 2.96726E-12],
     [1.05999E-14, 2.96702E-12], [1.05443E-14, 2.96603E-12]],
    [[1.37908E-14, 3.15081E-12], [1.37172E-14, 3.15020E-12],
     [1.36362E-14, 3.14927E-12], [1.35287E-14, 3.14817E-12]],
    [[2.26019E-14, 3.66798E-12], [2.24435E-14, 3.66540E-12],
     [2.23254E-14, 3.66173E-12], [2.20496E-14, 3.65796E-12]],
    [[4.41983E-13, 7.50091E-11], [3.93335E-13, 6.79097E-11],
     [3.45569E-13, 6.07845E-11], [2.96971E-13, 5.36085E-11]],
], F)

#: The RH nodes of the spectral Lagrange interpolation (module_ra_aerosol.F
#: :1094, :1229, :1338).
SPEC_RHS = np.array([0., 50., 70., 80., 90., 95., 98., 99.], F)

#: aer_type = 1 (rural, Shettle and Fenn 1979) rows of raod_lut, ssa_lut and
#: asy_lut, [rh node][band] (module_ra_aerosol.F:1097-1112, :1232-1247,
#: :1341-1356).  The fork fixes taer_type = 1 (module_radiation_driver.F:769).
RAOD_RURAL = np.array([
    [0.0735, 0.0997, 0.1281, 0.1529, 0.1882, 0.2512, 0.3010, 0.4550, 0.7159,
     1.0357, 1.3582, 1.6760, 2.2523, 0.0582],
    [0.0741, 0.1004, 0.1289, 0.1537, 0.1891, 0.2522, 0.3021, 0.4560, 0.7166,
     1.0351, 1.3547, 1.6687, 2.2371, 0.0587],
    [0.0752, 0.1017, 0.1304, 0.1554, 0.1909, 0.2542, 0.3042, 0.4580, 0.7179,
     1.0342, 1.3485, 1.6559, 2.2102, 0.0596],
    [0.0766, 0.1034, 0.1323, 0.1575, 0.1932, 0.2567, 0.3068, 0.4605, 0.7196,
     1.0332, 1.3411, 1.6407, 2.1785, 0.0608],
    [0.0807, 0.1083, 0.1379, 0.1635, 0.1998, 0.2639, 0.3143, 0.4677, 0.7244,
     1.0305, 1.3227, 1.6031, 2.1006, 0.0644],
    [0.0884, 0.1174, 0.1482, 0.1746, 0.2118, 0.2769, 0.3277, 0.4805, 0.7328,
     1.0272, 1.2977, 1.5525, 1.9976, 0.0712],
    [0.1072, 0.1391, 0.1724, 0.2006, 0.2396, 0.3066, 0.3581, 0.5087, 0.7510,
     1.0231, 1.2622, 1.4818, 1.8565, 0.0878],
    [0.1286, 0.1635, 0.1991, 0.2288, 0.2693, 0.3377, 0.3895, 0.5372, 0.7686,
     1.0213, 1.2407, 1.4394, 1.7739, 0.1072],
], F)
SSA_RURAL = np.array([
    [0.8730, 0.6695, 0.8530, 0.8601, 0.8365, 0.7949, 0.8113, 0.8810, 0.9305,
     0.9436, 0.9532, 0.9395, 0.8007, 0.8634],
    [0.8428, 0.6395, 0.8571, 0.8645, 0.8408, 0.8007, 0.8167, 0.8845, 0.9326,
     0.9454, 0.9545, 0.9416, 0.8070, 0.8589],
    [0.8000, 0.6025, 0.8668, 0.8740, 0.8503, 0.8140, 0.8309, 0.8943, 0.9370,
     0.9489, 0.9577, 0.9451, 0.8146, 0.8548],
    [0.7298, 0.5666, 0.9030, 0.9049, 0.8863, 0.8591, 0.8701, 0.9178, 0.9524,
     0.9612, 0.9677, 0.9576, 0.8476, 0.8578],
    [0.7010, 0.5606, 0.9312, 0.9288, 0.9183, 0.9031, 0.9112, 0.9439, 0.9677,
     0.9733, 0.9772, 0.9699, 0.8829, 0.8590],
    [0.6933, 0.5620, 0.9465, 0.9393, 0.9346, 0.9290, 0.9332, 0.9549, 0.9738,
     0.9782, 0.9813, 0.9750, 0.8980, 0.8594],
    [0.6842, 0.5843, 0.9597, 0.9488, 0.9462, 0.9470, 0.9518, 0.9679, 0.9808,
     0.9839, 0.9864, 0.9794, 0.9113, 0.8648],
    [0.6786, 0.5897, 0.9658, 0.9522, 0.9530, 0.9610, 0.9651, 0.9757, 0.9852,
     0.9871, 0.9883, 0.9835, 0.9236, 0.8618],
], F)
ASY_RURAL = np.array([
    [0.7444, 0.7711, 0.7306, 0.7103, 0.6693, 0.6267, 0.6169, 0.6207, 0.6341,
     0.6497, 0.6630, 0.6748, 0.7208, 0.7419],
    [0.7444, 0.7747, 0.7314, 0.7110, 0.6711, 0.6301, 0.6210, 0.6251, 0.6392,
     0.6551, 0.6680, 0.6799, 0.7244, 0.7436],
    [0.7438, 0.7845, 0.7341, 0.7137, 0.6760, 0.6381, 0.6298, 0.6350, 0.6497,
     0.6657, 0.6790, 0.6896, 0.7300, 0.7477],
    [0.7336, 0.7934, 0.7425, 0.7217, 0.6925, 0.6665, 0.6616, 0.6693, 0.6857,
     0.7016, 0.7139, 0.7218, 0.7495, 0.7574],
    [0.7111, 0.7865, 0.7384, 0.7198, 0.6995, 0.6864, 0.6864, 0.6987, 0.7176,
     0.7326, 0.7427, 0.7489, 0.7644, 0.7547],
    [0.7009, 0.7828, 0.7366, 0.7196, 0.7034, 0.6958, 0.6979, 0.7118, 0.7310,
     0.7452, 0.7542, 0.7593, 0.7692, 0.7522],
    [0.7226, 0.8127, 0.7621, 0.7434, 0.7271, 0.7231, 0.7248, 0.7351, 0.7506,
     0.7622, 0.7688, 0.7719, 0.7756, 0.7706],
    [0.7296, 0.8219, 0.7651, 0.7513, 0.7404, 0.7369, 0.7386, 0.7485, 0.7626,
     0.7724, 0.7771, 0.7789, 0.7790, 0.7760],
], F)

#: aer_type = 3 (maritime, Shettle and Fenn 1979) rows of raod_lut, ssa_lut
#: and asy_lut, [rh node][band], as the NOAA MPAS fork declares them
#: (RRFSx/MPAS-Model 1d6e370e, src/core_atmosphere/physics/physics_wrf/
#: module_ra_rrtmg_sw_aerosols.F:530-546, :677-693 and :792-808).  NOAA's
#: radiation driver selects them over water (mpas_atmphys_driver_radiation_sw.F
#: :583-586 and :637-640: xland == 2 gives taer_type 3, xland == 1 gives 1);
#: see aer_type_from_xland.  The rural rows above are the same words in
#: that file (:494-511, :641-658, :756-773).
RAOD_MARITIME = np.array([
    [0.3053, 0.3507, 0.3932, 0.4261, 0.4681, 0.5334, 0.5797, 0.6962, 0.8583,
     1.0187, 1.1705, 1.3049, 1.5205, 0.2748],
    [0.3566, 0.4023, 0.4443, 0.4765, 0.5170, 0.5792, 0.6227, 0.7298, 0.8756,
     1.0162, 1.1472, 1.2614, 1.4415, 0.3256],
    [0.4359, 0.4803, 0.5203, 0.5505, 0.5879, 0.6441, 0.6828, 0.7756, 0.8985,
     1.0135, 1.1198, 1.2109, 1.3518, 0.4051],
    [0.5128, 0.5544, 0.5913, 0.6187, 0.6523, 0.7020, 0.7358, 0.8149, 0.9174,
     1.0115, 1.0995, 1.1740, 1.2875, 0.4835],
    [0.6479, 0.6816, 0.7108, 0.7320, 0.7575, 0.7946, 0.8193, 0.8752, 0.9455,
     1.0092, 1.0728, 1.1263, 1.2061, 0.6236],
    [0.7582, 0.7831, 0.8043, 0.8196, 0.8377, 0.8636, 0.8806, 0.9184, 0.9649,
     1.0080, 1.0564, 1.0973, 1.1576, 0.7399],
    [0.8482, 0.8647, 0.8785, 0.8884, 0.9000, 0.9164, 0.9272, 0.9506, 0.9789,
     1.0072, 1.0454, 1.0780, 1.1256, 0.8360],
    [0.8836, 0.8965, 0.9073, 0.9149, 0.9239, 0.9365, 0.9448, 0.9626, 0.9841,
     1.0069, 1.0415, 1.0712, 1.1145, 0.8741],
], F)
SSA_MARITIME = np.array([
    [0.9697, 0.9183, 0.9749, 0.9820, 0.9780, 0.9712, 0.9708, 0.9778, 0.9831,
     0.9827, 0.9826, 0.9723, 0.8763, 0.9716],
    [0.9070, 0.8491, 0.9730, 0.9816, 0.9804, 0.9742, 0.9738, 0.9802, 0.9847,
     0.9841, 0.9838, 0.9744, 0.8836, 0.9546],
    [0.8378, 0.7761, 0.9797, 0.9827, 0.9829, 0.9814, 0.9812, 0.9852, 0.9882,
     0.9875, 0.9871, 0.9791, 0.9006, 0.9348],
    [0.7866, 0.7249, 0.9890, 0.9822, 0.9856, 0.9917, 0.9924, 0.9932, 0.9943,
     0.9938, 0.9933, 0.9887, 0.9393, 0.9204],
    [0.7761, 0.7164, 0.9959, 0.9822, 0.9834, 0.9941, 0.9955, 0.9952, 0.9960,
     0.9956, 0.9951, 0.9922, 0.9538, 0.9152],
    [0.7671, 0.7114, 0.9902, 0.9786, 0.9838, 0.9954, 0.9970, 0.9965, 0.9971,
     0.9968, 0.9964, 0.9943, 0.9644, 0.9158],
    [0.7551, 0.7060, 0.9890, 0.9743, 0.9807, 0.9966, 0.9989, 0.9978, 0.9982,
     0.9980, 0.9978, 0.9964, 0.9757, 0.9122],
    [0.7439, 0.7000, 0.9870, 0.9695, 0.9769, 0.9970, 1.0000, 0.9984, 0.9988,
     0.9986, 0.9984, 0.9975, 0.9825, 0.9076],
], F)
ASY_MARITIME = np.array([
    [0.6620, 0.7011, 0.7111, 0.7068, 0.6990, 0.6918, 0.6883, 0.6827, 0.6768,
     0.6773, 0.6863, 0.6940, 0.7245, 0.6719],
    [0.6880, 0.7394, 0.7297, 0.7240, 0.7162, 0.7083, 0.7038, 0.6957, 0.6908,
     0.6917, 0.6952, 0.7035, 0.7356, 0.6977],
    [0.7266, 0.7970, 0.7666, 0.7593, 0.7505, 0.7427, 0.7391, 0.7293, 0.7214,
     0.7210, 0.7212, 0.7265, 0.7519, 0.7340],
    [0.7683, 0.8608, 0.8120, 0.8030, 0.7826, 0.7679, 0.7713, 0.7760, 0.7723,
     0.7716, 0.7726, 0.7767, 0.7884, 0.7768],
    [0.7776, 0.8727, 0.8182, 0.8083, 0.7985, 0.7939, 0.7953, 0.7913, 0.7846,
     0.7870, 0.7899, 0.7918, 0.7969, 0.7870],
    [0.7878, 0.8839, 0.8231, 0.8130, 0.8050, 0.7977, 0.7945, 0.7932, 0.7955,
     0.7992, 0.8025, 0.8035, 0.8055, 0.7956],
    [0.8005, 0.8957, 0.8273, 0.8179, 0.8105, 0.8035, 0.8010, 0.8030, 0.8081,
     0.8108, 0.8143, 0.8174, 0.8174, 0.8042],
    [0.8104, 0.9034, 0.8294, 0.8212, 0.8144, 0.8087, 0.8077, 0.8118, 0.8175,
     0.8202, 0.8239, 0.8265, 0.8246, 0.8095],
], F)

#: RSLF's polynomial (module_mp_thompson.F RSLF): C0..C8.
RSLF_C = np.array([.611583699E03, .444606896E02, .143177157E01,
                   .264224321E-1, .299291081E-3, .203154182E-5,
                   .702620698E-8, .379534310E-11, -.321582393E-13], F)


def tables_sha256() -> str:
    """SHA-256 over every table above, float32 little-endian, in a fixed
    order.  The CUDA unit declares the same literals; the test recomputes
    this and the restart identity records it."""
    h = hashlib.sha256()
    for table in (GT_RH_ARR, GT_LOOKUP, SPEC_RHS, RAOD_RURAL, SSA_RURAL,
                  ASY_RURAL, RSLF_C):
        h.update(np.ascontiguousarray(table, "<f4").tobytes())
    return h.hexdigest()


#: Pinned at transcription; tests/test_rrtmg_aerosol_optics.py recomputes it.
AER3_TABLES_SHA256 = (
    "9962257443d94d4e71341d29165dbb40ece68b61f21b92c8b83c520cf03ae2b8")


def maritime_tables_sha256() -> str:
    """SHA-256 over the three maritime tables, float32 little-endian.

    Kept apart from :func:`tables_sha256` so a run on the rural optics
    (every WRF and HRRR configuration) keeps the identity it always had;
    only a run that selects the land/water type records this one."""
    h = hashlib.sha256()
    for table in (RAOD_MARITIME, SSA_MARITIME, ASY_MARITIME):
        h.update(np.ascontiguousarray(table, "<f4").tobytes())
    return h.hexdigest()


#: Pinned at transcription; tests/test_rrtmg_aer3_landmask_type.py
#: recomputes it and checks the words against the NOAA source text.
AER3_MARITIME_TABLES_SHA256 = (
    "16ef3c5062969b054350ac5299691f9b5504031af4c9c8d86a4b3b484d18b1d6")

#: Where the per-column aerosol type comes from.  ``rural``: type 1
#: everywhere, the HRRR fork's PARAMETER taer_type = 1
#: (module_radiation_driver.F:769), the default.  ``landmask``: the NOAA
#: MPAS fork's driver, type 1 where xland == 1 and type 3 (maritime)
#: where xland == 2 (mpas_atmphys_driver_radiation_sw.F:583-586).
AER3_TYPE_SOURCES = ("rural", "landmask")


def aer_type_from_xland(xland):
    """NOAA MPAS taer_type from the land/water flag: 1 over land (xland 1),
    3 over water (xland 2), int32.

    The NOAA driver assigns nothing for any other xland value, so the type
    would be whatever the array held from an earlier call (uninitialized on
    the first): a value outside {1, 2} is refused rather than given a type
    the source never assigns."""
    xland = np.asarray(xland, np.float32)
    land = xland == F(1.0)
    water = xland == F(2.0)
    if not np.all(land | water):
        bad = np.unique(xland[~(land | water)])[:4]
        raise ValueError(
            "rrtmg_aer3_type_source='landmask' needs xland 1 (land) or 2 "
            f"(water) in every column, got {bad.tolist()}: NOAA's driver "
            "assigns no aerosol type to any other value "
            "(mpas_atmphys_driver_radiation_sw.F:583-586)")
    return np.where(water, 3, 1).astype(np.int32)


def _type_array(aer_type, shape):
    """Validate a per-column aerosol type against the column shape."""
    aer_type = np.asarray(aer_type)
    if aer_type.shape != tuple(shape) or not np.issubdtype(aer_type.dtype, np.integer):
        raise ValueError(
            f"aer_type must be an integer array of shape {tuple(shape)}, got "
            f"{aer_type.dtype} {aer_type.shape}")
    if not np.all((aer_type == 1) | (aer_type == 3)):
        raise ValueError("aer_type admits 1 (rural) and 3 (maritime) only, "
                         "the two types NOAA's driver assigns")
    return aer_type


# ---------------------------------------------------------------------------
# NumPy float32 twin.
# ---------------------------------------------------------------------------

def _nint(x):
    """Fortran NINT on REAL(4): round half away from zero."""
    x = np.asarray(x, np.float64)
    return np.trunc(x + np.copysign(0.5, x)).astype(np.int64)


def _expf(x):
    from gpuwm.core.noahmp_libm import expf
    x = np.asarray(x, np.float32)
    out = np.empty(x.shape, np.float32)
    flat = out.reshape(-1)
    for i, v in enumerate(x.reshape(-1)):
        flat[i] = expf(F(v))
    return out


def rslf(p, t):
    """Thompson ``RSLF(P, T)`` (float32 arrays)."""
    p = np.asarray(p, np.float32)
    t = np.asarray(t, np.float32)
    c = RSLF_C
    x = np.maximum(F(-80.0), t - F(273.16))
    esl = c[7] + x * c[8]
    for coef in c[6::-1]:
        esl = coef + x * esl
    esl = np.minimum(esl, p * F(0.15))
    return ((F(0.622) * esl) / (p - esl)).astype(np.float32)


def gt_aod(p, dz8w, t, qv, nwfa, nifa):
    """``gt_aod``: the layer AOD at 550 nm, float32, same shape as ``p``."""
    p, dz8w, t, qv, nwfa, nifa = (np.asarray(a, np.float32) for a in
                                  (p, dz8w, t, qv, nwfa, nifa))
    rhoa = p / (F(287.0) * t)
    t_idx = np.maximum(1, np.minimum(_nint(F(10.999) - F(0.0333) * t), 4))
    qvsat = rslf(p, t)
    rh = np.minimum(F(98.0), np.maximum(F(10.1), (qv / qvsat) * F(100.0)))
    rind = 8
    arr = GT_RH_ARR
    idx1 = np.ones(rh.shape, np.int64)
    idx2 = np.full(rh.shape, 2, np.int64)
    # RH < 60: (1, 2); 60 <= RH < 80 and RH >= 80 below.
    mid = (rh >= F(60.0)) & (rh < F(80.0))
    high = ~(rh < F(60.0)) & ~mid
    # 60 <= RH < 80: RH_idx = nint(0.1*RH - 4)
    idx_mid = _nint(F(0.1) * rh + F(-4.0))
    # RH >= 80: RH_idx = MIN(rind, nint(0.2*RH - 12))
    idx_high = np.minimum(rind, _nint(F(0.2) * rh + F(-12.0)))
    for sel, idx in ((mid, idx_mid), (high, idx_high)):
        safe = np.clip(idx, 1, rind)
        rh_d = rh - arr[safe - 1]
        below = rh_d < F(0.0)
        i1 = np.where(below, idx - 1, idx)
        i2 = np.where(below, idx, idx + 1)
        over = (~below) & (i2 > rind)
        i2 = np.where(over, rind, i2)
        i1 = np.where(over, rind - 1, i1)
        idx1 = np.where(sel, i1, idx1)
        idx2 = np.where(sel, i2, idx2)
    a1 = arr[idx1 - 1]
    a2 = arr[idx2 - 1]
    b1 = a1 / (F(100.0) - a1)
    rh_f = np.maximum(F(0.0), np.minimum(F(1.0), (rh / (F(100.0) - rh) - b1)
                                         / (a2 / (F(100.0) - a2) - b1)))
    ti = t_idx - 1
    l1w = GT_LOOKUP[idx1 - 1, ti, 0]
    l2w = GT_LOOKUP[idx2 - 1, ti, 0]
    l1i = GT_LOOKUP[idx1 - 1, ti, 1]
    l2i = GT_LOOKUP[idx2 - 1, ti, 1]
    unit_bext1 = l1w + (l2w - l1w) * rh_f
    unit_bext3 = l1i + (l2i - l1i) * rh_f
    ntemp = np.maximum(F(1.0), np.minimum(F(99999.E6), nwfa))
    aod_wfa = ((unit_bext1 * ntemp) * dz8w) * rhoa
    ntemp = np.maximum(F(0.01), np.minimum(F(9999.E6), nifa))
    aod_ifa = ((unit_bext3 * ntemp) * dz8w) * rhoa
    return (aod_wfa + aod_ifa).astype(np.float32)


def relative_humidity(p, t, qv):
    """``calc_relative_humidity`` (Bolton), percent, float32."""
    p, t, qv = (np.asarray(a, np.float32) for a in (p, t, qv))
    tc = t - F(273.15)
    rv = np.maximum(F(0.0), qv)
    es = F(6.112) * _expf((F(17.6) * tc) / (tc + F(243.5)))
    e = ((F(0.01) * rv) * p) / (rv + F(0.62197))
    return np.minimum(F(99.0), np.maximum(F(0.0), (F(100.0) * e) / es)
                      ).astype(np.float32)


def _lagrange(rh, table, scale=None):
    """The 4-point-wide Lagrange sum of calc_spectral_*_rrtmg_sw over the
    eight RH nodes, per element of ``rh``, all 14 bands; ``scale`` (the
    AOD) multiplies each term as the AOD routine does."""
    rh = np.asarray(rh, np.float32)
    rhs = SPEC_RHS
    n_rh = rhs.size
    ii = 1 + np.sum(rh[..., None] > rhs, axis=-1)
    imin = np.maximum(1, ii - 2 - 1)
    imax = np.minimum(n_rh, ii + 2)
    out = np.zeros(rh.shape + (NBNDSW,), np.float32)
    for a in range(6):
        jj = imin + a
        ok_j = jj <= imax
        jjc = np.clip(jj, 1, n_rh)
        lj = np.ones(rh.shape, np.float32)
        for b in range(6):
            k = imin + b
            ok_k = (k <= imax) & (k != jj)
            kc = np.clip(k, 1, n_rh)
            with np.errstate(divide="ignore", invalid="ignore"):
                cand = (lj * (rh - rhs[kc - 1])) / (rhs[jjc - 1] - rhs[kc - 1])
            lj = np.where(ok_k, cand, lj).astype(np.float32)
        term = lj[..., None] * table[jjc - 1]
        if scale is not None:
            term = term * np.asarray(scale, np.float32)[..., None]
        out = np.where(ok_j[..., None], out + term, out).astype(np.float32)
    return out


def _smoke_array_numpy(name, value, shape, *, positive=False, upper=None):
    """Validate a prescribed optical input without inventing missing values."""
    if value is None:
        raise ValueError(f"smoke feedback requires {name}; missing smoke cannot be replaced by zero")
    value = np.asarray(value)
    if value.dtype != np.float32 or value.shape != shape:
        raise ValueError(f"{name} must be a float32 array of shape {shape}, got {value.dtype} {value.shape}")
    invalid = ~np.isfinite(value) | (value <= F(0) if positive else value < F(0))
    if upper is not None:
        invalid |= value >= F(upper)
    if invalid.any():
        raise ValueError(f"{name} has non-finite or physically invalid values; prescribed smoke is refused")
    return value


def smoke_aod_from_dry_mixing_ratio(smoke_ugkg, rho_dry, dz8w):
    """Oracle twin of the fork's layer smoke AOD, dimensionless at 550 nm.

    Source: smoke/module_add_emiss_burn.F:78,184,194 of v4.1.21.
    The smoke mixing ratio is ug/kg-dryair, rho_dry is kg-dryair/m3,
    and dz8w is m. No humidity enhancement is present in that source.
    Invalid external input is refused instead of invoking the source's
    tracer reset to 1.e-16 (:190-191). This is validation/oracle code;
    production smoke conversion runs in the device twin below.
    """
    shape = np.asarray(smoke_ugkg).shape
    smoke = _smoke_array_numpy("smoke_ugkg", smoke_ugkg, shape, upper=11000)
    rho = _smoke_array_numpy("rho_dry", rho_dry, shape, positive=True)
    dz = _smoke_array_numpy("dz8w", dz8w, shape, positive=True)
    ext2 = F(F(4.0) + F(0.5))
    aod = (((F(1.e-6) * ext2) * smoke) * rho) * dz
    return _smoke_array_numpy("smoke_aod", aod.astype(np.float32), shape)


def smoke_dry_mixing_ratio_from_posted_density(pm_kgm3, donor_p, donor_t):
    """Oracle inverse of the fork's posted PMTF concentration in kg/m3.

    UPP MDLFLD.f:2249 writes ((1/RD)*(P/T)*smoke_ugkg)*1.e-9;
    params.F:57 defines RD=287.04. Recover the dry-air mixing ratio
    using the donor's pressure and temperature, not the receiving model's.
    GRIB packing makes this recovery approximate. Production preparation
    must use the Rust decoder and the same declared units and donor grid.
    """
    shape = np.asarray(pm_kgm3).shape
    pm = _smoke_array_numpy("PMTF_kgm3", pm_kgm3, shape)
    p = _smoke_array_numpy("donor_pressure_Pa", donor_p, shape, positive=True)
    t = _smoke_array_numpy("donor_temperature_K", donor_t, shape, positive=True)
    density = F(F(1.0) / F(287.04)) * (p / t)
    density = _smoke_array_numpy("donor_density_coefficient", density, shape, positive=True)
    smoke = (pm / F(1.e-9)) / density
    return _smoke_array_numpy("smoke_ugkg", smoke.astype(np.float32), shape, upper=11000)


def _smoke_selection_numpy(shape, *, smoke_aod, smoke_feedback):
    if type(smoke_feedback) is not bool:
        raise ValueError("smoke_feedback must be an explicit boolean")
    if not smoke_feedback:
        if smoke_aod is not None:
            raise ValueError("smoke_aod was supplied while smoke feedback is disabled; refusing to discard it")
        return None
    return _smoke_array_numpy("smoke_aod", smoke_aod, shape)


def aer3_sw_optics(p, t, qv, dz8w, nwfa, nifa, *, smoke_aod=None,
                  smoke_feedback=False, aer_type=None):
    """The fork's aer_opt = 3 shortwave optics on (..., nz) float32 arrays.

    Returns ``(tauaer, ssaaer, asyaer, taod5503d)``: the first three
    (..., nz, 14) as RRTMG_SWRAD hands them to rrtmg_sw for the model
    layers, the last the layer AOD at 550 nm gt_aod produced.

    ``aer_type`` (None, or an integer array of the column shape
    ``p.shape[:-1]`` holding 1 or 3) selects the Shettle-Fenn rows per
    column as the NOAA MPAS fork does; None is rural type 1 everywhere,
    the HRRR fork.  Every element's arithmetic reads only its own table,
    so a type-1 column is the same words either way.
    """
    shape = np.asarray(p).shape
    types = None if aer_type is None else _type_array(aer_type, shape[:-1])
    if types is not None and smoke_feedback:
        raise ValueError(
            "prescribed smoke follows the HRRR fork's rural optics; the NOAA "
            "MPAS land/water aerosol type has no smoke coupling in its driver "
            "(mpas_atmphys_driver_radiation_sw.F:573-683), so the pair has "
            "no source behavior to reproduce")
    smoke = _smoke_selection_numpy(shape, smoke_aod=smoke_aod,
                                   smoke_feedback=smoke_feedback)
    taod = gt_aod(p, dz8w, t, qv, nwfa, nifa)
    if smoke is not None:
        # radiation driver :984: cap EACH layer at 3 before the column
        # sum and the existing rural spectral optics, no RH multiplier.
        taod = (taod + np.minimum(F(3.0), smoke)).astype(np.float32)
    rh = relative_humidity(p, t, qv)
    tau = _lagrange(rh, RAOD_RURAL, scale=taod)
    ssa = _lagrange(rh, SSA_RURAL)
    asy = _lagrange(rh, ASY_RURAL)
    if types is not None and np.any(types == 3):
        water = (types == 3)[..., None, None]
        tau = np.where(water, _lagrange(rh, RAOD_MARITIME, scale=taod), tau)
        ssa = np.where(water, _lagrange(rh, SSA_MARITIME), ssa)
        asy = np.where(water, _lagrange(rh, ASY_MARITIME), asy)
    return tau, ssa, asy, taod


def sw_engine_layout(tau, ssa, asy, nlayers):
    """(ncol, nz, 14) optics -> the batched SW engine's (ncol, 14, nlayers)
    ztaua/zasya/zomga, the layers above the model top at 0/0/1
    (RRTMG_SWRAD's initial values, module_ra_rrtmg_sw.F:10930-10936)."""
    tau = np.asarray(tau, np.float32)
    ncol, nz, nb = tau.shape
    ztaua = np.zeros((ncol, nb, nlayers), np.float32)
    zasya = np.zeros((ncol, nb, nlayers), np.float32)
    zomga = np.ones((ncol, nb, nlayers), np.float32)
    ztaua[:, :, :nz] = tau.transpose(0, 2, 1)
    zasya[:, :, :nz] = np.asarray(asy, np.float32).transpose(0, 2, 1)
    zomga[:, :, :nz] = np.asarray(ssa, np.float32).transpose(0, 2, 1)
    return ztaua, zasya, zomga


# ---------------------------------------------------------------------------
# Device entry.
# ---------------------------------------------------------------------------

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        from gpuwm.core.kernels import load_module
        _MODULE = load_module("rrtmg_aer3")
    return _MODULE


def _smoke_array_device(name, value, shape, *, positive=False, upper=None):
    import cupy as cp
    if value is None:
        raise ValueError(f"smoke feedback requires {name}; missing smoke cannot be replaced by zero")
    if not isinstance(value, cp.ndarray) or value.dtype != cp.float32 or value.shape != shape:
        raise ValueError(f"{name} must be a float32 device array of shape {shape}")
    invalid = ~cp.isfinite(value) | (value <= F(0) if positive else value < F(0))
    if upper is not None:
        invalid |= value >= F(upper)
    if bool(cp.any(invalid).item()):
        raise ValueError(f"{name} has non-finite or physically invalid values; prescribed smoke is refused")
    return cp.ascontiguousarray(value)


def smoke_aod_from_dry_mixing_ratio_device(smoke_ugkg, rho_dry, dz8w):
    """Device source arithmetic for prescribed ug/kg-dryair smoke, no transport.

    This converts an already time/grid/units-validated prescribed profile.
    It does not provide smoke emission, advection, deposition or chemistry.
    """
    import cupy as cp
    shape = getattr(smoke_ugkg, "shape", None)
    if shape is None:
        raise ValueError("smoke feedback requires smoke_ugkg; missing smoke cannot be replaced by zero")
    arrays = [_smoke_array_device("smoke_ugkg", smoke_ugkg, shape, upper=11000),
              _smoke_array_device("rho_dry", rho_dry, shape, positive=True),
              _smoke_array_device("dz8w", dz8w, shape, positive=True)]
    result = cp.empty(shape, dtype=cp.float32)
    if result.size:
        _module().get_function("rrtmg_smoke_aod")(
            ((result.size + 127) // 128,), (128,),
            (np.int64(result.size), *arrays, result))
    return _smoke_array_device("smoke_aod", result, shape)


def smoke_dry_air_density_device(alt):
    """Source dry density 1/ALT with an explicit single-rounded division."""
    import cupy as cp
    shape = getattr(alt, "shape", None)
    if shape is None:
        raise ValueError("smoke feedback requires ALT_dry_specific_volume")
    value = _smoke_array_device("ALT_dry_specific_volume", alt, shape, positive=True)
    result = cp.empty(shape, cp.float32)
    if result.size:
        _module().get_function("rrtmg_smoke_dry_density")(
            ((result.size + 127) // 128,), (128,),
            (np.int64(result.size), value, result))
    return _smoke_array_device("rho_dry", result, shape, positive=True)


def smoke_dry_mixing_ratio_from_posted_density_device(pm_kgm3, donor_p, donor_t):
    """Device inverse of the posted PMTF coefficient, retaining donor p/T.

    Inputs are kg/m3, Pa and K on one time/grid-validated donor profile.
    The output is ug/kg-dryair. Source RD is 287.04, not the receiving
    model's constant. Convert this profile to AOD with model rho_dry/dz.
    """
    import cupy as cp
    shape = getattr(pm_kgm3, "shape", None)
    if shape is None:
        raise ValueError("smoke feedback requires PMTF_kgm3; missing smoke cannot be replaced by zero")
    arrays = [_smoke_array_device("PMTF_kgm3", pm_kgm3, shape),
              _smoke_array_device("donor_pressure_Pa", donor_p, shape, positive=True),
              _smoke_array_device("donor_temperature_K", donor_t, shape, positive=True)]
    result = cp.empty(shape, dtype=cp.float32)
    if result.size:
        _module().get_function("rrtmg_smoke_posted_inverse")(
            ((result.size + 127) // 128,), (128,),
            (np.int64(result.size), *arrays, result))
    return _smoke_array_device("smoke_ugkg", result, shape, upper=11000)


def aer_type_from_xland_device(xland):
    """Device twin of :func:`aer_type_from_xland`: int32 (nc,), 1 where
    xland == 1, 3 where xland == 2, any other value refused by name."""
    import cupy as cp
    if not isinstance(xland, cp.ndarray) or xland.ndim != 1:
        raise ValueError("aer_type_from_xland_device needs a 1-D device xland")
    x = xland.astype(cp.float32, copy=False)
    land = x == F(1.0)
    water = x == F(2.0)
    if not bool(cp.all(land | water).item()):
        bad = cp.unique(x[~(land | water)])[:4].get().tolist()
        raise ValueError(
            "rrtmg_aer3_type_source='landmask' needs xland 1 (land) or 2 "
            f"(water) in every column, got {bad}: NOAA's driver assigns no "
            "aerosol type to any other value "
            "(mpas_atmphys_driver_radiation_sw.F:583-586)")
    return cp.where(water, cp.int32(3), cp.int32(1)).astype(cp.int32)


def aer3_sw_optics_device(p3d, t3d, qv3d, dz8w, nwfa, nifa, nlayers,
                         *, smoke_aod=None, smoke_feedback=False,
                         aer_type=None):
    """The batched SW engine's aerosol slabs on the device.

    Inputs are (nc, nz) float32 device arrays of one adapter chunk (the
    radiation wrapper's p, t, qv and dz8w and the Thompson numbers).
    Returns ``(ztaua, zasya, zomga, taod)``: the first three
    (nc, 14, nlayers) in rsw_spcvmc_gpt_b's per-column band-major layout,
    the layers from nz up at 0/0/1, and the (nc, nz) layer AOD at 550 nm.

    ``aer_type`` (None or an int32 (nc,) device array of 1 and 3) runs the
    NOAA MPAS per-column type through ``rrtmg_aer3_typed_sw_optics``; None
    launches the rural kernel exactly as before.
    """
    import cupy as cp
    nc, nz = (int(s) for s in p3d.shape)
    nlayers = int(nlayers)
    if nlayers < nz:
        raise ValueError(f"nlayers={nlayers} below the {nz} model layers")
    arrays = []
    for name, a in (("p3d", p3d), ("t3d", t3d), ("qv3d", qv3d),
                    ("dz8w", dz8w), ("nwfa", nwfa), ("nifa", nifa)):
        if not isinstance(a, cp.ndarray) or a.dtype != cp.float32 \
                or a.shape != (nc, nz):
            raise ValueError(
                f"aer_opt=3 optics: {name} must be a float32 device array "
                f"of shape {(nc, nz)}, got {getattr(a, 'dtype', None)} "
                f"{getattr(a, 'shape', None)}")
        arrays.append(cp.ascontiguousarray(a))
    if type(smoke_feedback) is not bool:
        raise ValueError("smoke_feedback must be an explicit boolean")
    if aer_type is not None:
        if smoke_feedback or smoke_aod is not None:
            raise ValueError(
                "prescribed smoke follows the HRRR fork's rural optics; the "
                "NOAA MPAS land/water aerosol type has no smoke coupling in "
                "its driver (mpas_atmphys_driver_radiation_sw.F:573-683), so "
                "the pair has no source behavior to reproduce")
        if not isinstance(aer_type, cp.ndarray) or aer_type.dtype != cp.int32                 or aer_type.shape != (nc,):
            raise ValueError(
                f"aer_type must be an int32 device array of shape {(nc,)}, "
                f"got {getattr(aer_type, 'dtype', None)} "
                f"{getattr(aer_type, 'shape', None)}")
        if not bool(cp.all((aer_type == 1) | (aer_type == 3)).item()):
            raise ValueError("aer_type admits 1 (rural) and 3 (maritime) "
                             "only, the two types NOAA's driver assigns")
        ztaua = cp.empty((nc, NBNDSW, nlayers), dtype=cp.float32)
        zasya = cp.empty((nc, NBNDSW, nlayers), dtype=cp.float32)
        zomga = cp.empty((nc, NBNDSW, nlayers), dtype=cp.float32)
        taod = cp.empty((nc, nz), dtype=cp.float32)
        total = nc * nlayers
        if total:
            _module().get_function("rrtmg_aer3_typed_sw_optics")(
                ((total + 127) // 128,), (128,),
                (np.int32(nc), np.int32(nz), np.int32(nlayers), *arrays,
                 cp.ascontiguousarray(aer_type), ztaua, zasya, zomga, taod))
        return ztaua, zasya, zomga, taod
    if smoke_feedback:
        smoke = _smoke_array_device("smoke_aod", smoke_aod, (nc, nz))
    elif smoke_aod is not None:
        raise ValueError("smoke_aod was supplied while smoke feedback is disabled; refusing to discard it")
    else:
        smoke = None
    ztaua = cp.empty((nc, NBNDSW, nlayers), dtype=cp.float32)
    zasya = cp.empty((nc, NBNDSW, nlayers), dtype=cp.float32)
    zomga = cp.empty((nc, NBNDSW, nlayers), dtype=cp.float32)
    taod = cp.empty((nc, nz), dtype=cp.float32)
    total = nc * nlayers
    if total:
        kernel_name = "rrtmg_aer3_sw_optics" if smoke is None else "rrtmg_aer3_smoke_sw_optics"
        _module().get_function(kernel_name)(
            ((total + 127) // 128,), (128,),
            (np.int32(nc), np.int32(nz), np.int32(nlayers), *arrays,
             *((smoke,) if smoke is not None else ()),
             ztaua, zasya, zomga, taod))
    return ztaua, zasya, zomga, taod


# WRF 4.6.1 Tegen six-type climatology. These entry points only orchestrate
# device arrays. The interpolation and band sums run in rrtmg_aer1.cu.
def _aer1_launch(name, count, args):
    from gpuwm.core.kernels import load_module
    if count:
        load_module("rrtmg_aer1").get_function(name)(
            ((count + 127) // 128,), (128,), args)


def _aer1_array(name, value, shape):
    import cupy as cp
    if not isinstance(value, cp.ndarray) or value.dtype != cp.float32 or value.shape != shape:
        raise ValueError(f"aer_opt=1 {name} needs float32 device shape {shape}")
    if bool(cp.any(~cp.isfinite(value)).item()):
        raise ValueError(f"aer_opt=1 {name} contains non-finite values")
    return cp.ascontiguousarray(value)


def aer1_climatology_device(latitude, longitude, tables):
    import cupy as cp
    nc = int(latitude.size)
    lat = _aer1_array("latitude", latitude, (nc,))
    lon = _aer1_array("longitude", longitude, (nc,))
    if bool(cp.any((lat < -90) | (lat > 90) | (lon < -180) | (lon > 180)).item()):
        raise ValueError("aer_opt=1 coordinates must use latitude [-90,90] and longitude [-180,180]")
    out = cp.empty((nc, 12, 12, 6), cp.float32)
    _aer1_launch("rrtmg_aer1_monthly", out.size,
                 (np.int32(nc), lat, lon, cp.asarray(tables["latitude"]),
                  cp.asarray(tables["longitude"]), cp.asarray(tables["climatology"]), out))
    return out


def aer1_time_pressure_device(monthly, julian, pressure, pressure_full, tables):
    import cupy as cp
    nc, nz = map(int, pressure.shape)
    monthly = _aer1_array("aerodm", monthly, (nc, 12, 12, 6))
    p = _aer1_array("pressure", pressure, (nc, nz))
    pf = _aer1_array("interface pressure", pressure_full, (nc, nz+1))
    if not np.isfinite(julian) or float(julian) < 0:
        raise ValueError("aer_opt=1 julian must be finite and nonnegative")
    if bool(cp.any((p <= 0)).item()) or bool(cp.any(p[:,1:] >= p[:,:-1]).item()) or bool(cp.any(pf[:,1:] >= pf[:,:-1]).item()):
        raise ValueError("aer_opt=1 requires positive pressure decreasing from bottom to top")
    timed = cp.empty((nc, 12, 6), cp.float32)
    aerod = cp.empty((nc, nz, 6), cp.float32)
    total = cp.empty(nc, cp.float32)
    _aer1_launch("rrtmg_aer1_time", timed.size, (np.int32(nc), F(julian), monthly,
                  cp.asarray(tables["month_days"]), timed))
    _aer1_launch("rrtmg_aer1_pressure", aerod.size, (np.int32(nc), np.int32(nz), p, pf,
                  cp.asarray(tables["pressure"]), timed, aerod))
    _aer1_launch("rrtmg_aer1_total", nc, (np.int32(nc), np.int32(nz), aerod, total))
    return aerod, total, timed


def aer1_sw_optics_device(aerod, nlayers, sw_tables):
    import cupy as cp
    nc, nz, ns = map(int, aerod.shape)
    if ns != 6 or type(nlayers) is not int or nlayers != nz+1:
        raise ValueError("aer_opt=1 needs six species and one zero aerosol layer above the model top")
    aerod = _aer1_array("aerod", aerod, (nc, nz, 6))
    if bool(cp.any(aerod < 0).item()):
        raise ValueError("aer_opt=1 layer optical depth must be nonnegative")
    out = tuple(cp.empty((nc, 14, nlayers), cp.float32) for _ in range(3))
    from gpuwm.core.rrtmg_sw import validate_aer1_optics_tables
    validate_aer1_optics_tables(sw_tables)
    coeffs = tuple(cp.asarray(np.ascontiguousarray(getattr(sw_tables, name), dtype=F))
                   for name in ("rsrtaua", "rsrpiza", "rsrasya"))
    _aer1_launch("rrtmg_aer1_optics", out[0].size,
                 (np.int32(nc), np.int32(nz), np.int32(nlayers), aerod, *coeffs, *out))
    return out
