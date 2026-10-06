"""EXPERIMENTAL: the radar half of the GSD cloud analysis, on the device.

After a filter solve, an analysis still holds whatever rain, snow and graupel
the ensemble carried where the radar saw none, and nothing where the radar
saw echo no member built.  NOAA's operational answer for the HRRR is the
precipitation part of the GSD cloud analysis, which runs after the last
outer loop of the variational solve.  This module is that part, for the
research DA line: every numeric step is a CUDA kernel compiled here, and
there is no host implementation (:func:`require_device` names the refusal).

The specification and the oracle are NOAA's own source at NOAA-EMC/HRRR tag
``v4.1.21`` (paths below are relative to that tag):

* ``sorc/hrrr_gsi.fd/libsrc/GSD/gsdcloud/hydro_mxr_thompson.f90`` -- the
  Thompson retrieval: reflectivity, temperature and pressure to rain mass,
  rain number and snow mass.  Snow is capped at 28 dBZ and rain at 55 dBZ
  (:55-58), the rain fraction is linear from 0 to 5 C (:126-132), and it
  makes no graupel.  It runs on levels 2 to nz-1 only (:117).
* ``sorc/hrrr_gsi.fd/src/gsi/gsdcloudanalysis.F90`` :872-1049 -- the final
  precipitation analysis in its three modes, and :1053-1066, the clamp.

Three modes, named for what they are:

``clear``
    :910-927, what HRRRDAS applies to every member
    (``parm/hrrrdas/hrrrdas_gsiparm.anl:169-170``): where the radar observed
    no echo, precipitating mass and its number go to zero.  Scheme-neutral:
    it clears whatever fields it is handed.
``trim-build``
    :928-1049, the rule NOAA runs on the deterministic HRRR analysis and
    never on an HRRRDAS member (members take ``clear`` only,
    ``l_precip_clear_only``).  A column with echo whose
    first level is colder than 5 C adds snow and takes it out of rain and
    graupel; a warmer column scales the background rain and snow down when
    its largest rain plus snow exceeds the retrieval at the level of
    strongest echo, and otherwise inserts the retrieval at that level only
    (3 g/kg cap); light precipitation from 15 to 28 dBZ is kept up to
    1 g/kg; a column with no echo is cleared per level, or whole when
    satellite cloud-top pressure says clear.
``retrieve-all``
    :872-909: every cell with echo takes the retrieval (15 g/kg cap).

``trim-build`` and ``retrieve-all`` write Thompson's rain mass, rain number and
snow, so they take exactly the fields ``qr``, ``nr``, ``qs``, ``qg``.

Reflectivity arrives in NOAA's sentinel convention (:884-906): a value above
0 dBZ is echo, a value at or below 0 and above -100 is observed no echo, and
anything else is no coverage.  :func:`radar_grid_reflectivity` builds that
field on the device from a ``gpuwm-obs.radar-grid`` document.

Precision is NOAA's: the retrieval is double precision on float32
temperature and pressure and stores float32, the analysis block mixes the
two exactly as the Fortran does, and ``pow`` is glibc's
(``gpuwm/core/kernels/glibc_flt64.cuh``, measured equal to glibc 2.43's
words), so the kernels can be compared bit for bit with NOAA's routines
compiled by gfortran (``tools/gsd_precip_oracle``).  Every double division
is ``__ddiv_rn`` and every product and sum is an explicit round-to-nearest
intrinsic, so neither a reciprocal rewrite nor a fused multiply-add can
move a result between cards.

Three properties of NOAA's rule a caller should know, because the port
reproduces them:

* The retrieval is missing on the first and the last level.  ``trim-build``
  therefore reads a retrieval of zero when the strongest echo of a column
  is on the first level, and scales that column's background rain and snow
  to zero; ``retrieve-all`` zeroes rain and snow in an echo cell on either
  level.  The receipt counts those columns.
* NOAA's trim takes its ratio from the largest background rain plus snow
  on ANY level of the column, scales every level, and its clamp at zero
  acts on every cell.  The port does exactly that under ``scope="noaa"``
  (the oracle runs it).  The default, ``scope="covered"``, is the rule
  restricted to what the radar sampled: the background maximum the trim
  compares with the retrieval is taken over sampled levels only, and every
  cell the radar did not sample is kept bit for bit (the receipt counts
  the cells NOAA's rule would have moved there).  Taking the maximum over
  every level while keeping the unsampled levels would cut the sampled
  levels by a ratio set by rain below the beam and leave that rain in
  place; the receipt counts the columns where the largest background sat
  on an unsampled level.  The analysis seam runs ``covered``, so a cell
  with no coverage gets an increment of exactly zero from this stage.
* The water removed is not put anywhere.  It is a declared sink, as it is
  in the HRRR, and the receipt carries the amount per field.

It is a pure function of its inputs and never writes a state:
:mod:`gpuwm.ensemble.increments` stays the one writer.  Nothing here is
wired into a default route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

import numpy as np

EXPERIMENTAL = True

#: Receipt schema tag.
SCHEMA = "gpuwm-da.hydrometeor-analysis.v1"

#: The three modes, named for what they do, in NOAA's order of use:
#: ensemble members (HRRRDAS), the deterministic analysis (HRRR), RTMA.
MODES = ("clear", "trim-build", "retrieve-all")

#: The fields ``trim-build`` and ``retrieve-all`` read and write: Thompson's rain
#: mass, rain number, snow and graupel (ges_qr, ges_qnr, ges_qs, ges_qg).
THOMPSON_FIELDS = ("qr", "nr", "qs", "qg")

#: Which cells the analysis may move.  ``noaa`` is NOAA's footprint
#: exactly: the ``trim-build`` trim compares the retrieval with the largest
#: background on any level and scales every level of a trimmed column, and
#: the clamp of gsdcloudanalysis.F90:1053-1066 acts on every cell.
#: ``covered`` takes that largest background over sampled levels only, keeps
#: every cell the radar did not sample (reflectivity at or below -100)
#: exactly as given, and counts the cells NOAA's rule would have moved
#: there: a radar that did not sample a cell says nothing about it.
SCOPES = ("covered", "noaa")

#: NOAA's sentinels.  ``-99`` is the mosaic's observed-no-echo value
#: (gsdcloud/vinterp_radar_ref.f90:113) and ``-99999`` its missing value
#: (gsdcloudanalysis.F90:190).
NO_ECHO_DBZ = -99.0
NO_COVERAGE_DBZ = -99999.0

#: The precipitating masses the engine's radar operators already name
#: (gpuwm.da.obsop.PRECIPITATING_SPECIES): rain, snow, graupel, hail.
PRECIPITATING_MASSES = ("qr", "qs", "qg", "qh")

#: Every setting, with the value the HRRR runs and where NOAA sets it.
NOAA_SETTINGS = {
    "opt_hydrometeor_retri": {
        "value": 3, "meaning": "Thompson retrieval",
        "source": "sorc/hrrr_gsi.fd/src/gsi/gsdcloudanalysis.F90:295"},
    "r_cleanSnow_WarmTs_threshold": {
        "value": 5.0, "meaning": "first-level temperature (C) below which "
                                 "a column with echo takes the snow rule",
        "source": "parm/conus/hrrr_gsiparm.anl.sh:171"},
    "i_lightpcp": {
        "value": 1, "meaning": "keep light precipitation from 15 to 28 dBZ",
        "source": "parm/conus/hrrr_gsiparm.anl.sh:186"},
    "iclean_hydro_withRef": {
        "value": 1, "meaning": "clear where radar observes no echo",
        "source": "sorc/hrrr_gsi.fd/src/gsi/rapidrefresh_cldsurf_mod.f90:332"},
    "iclean_hydro_withRef_allcol": {
        "value": 1, "meaning": "clear the whole column when satellite "
                               "cloud-top pressure says clear",
        "source": "parm/conus/hrrr_gsiparm.anl.sh:181"},
    "l_precip_clear_only": {
        "value": True, "meaning": "HRRRDAS members take the clear step only",
        "source": "parm/hrrrdas/hrrrdas_gsiparm.anl:170"},
    "limits_kg_kg": {
        "value": {"trim_build_insert": 3.0e-3, "light_precipitation": 1.0e-3,
                  "retrieve_all": 15.0e-3},
        "meaning": "caps on inserted rain and snow",
        "source": "sorc/hrrr_gsi.fd/src/gsi/gsdcloudanalysis.F90:873, "
                  ":929-930"},
}

#: Per-cell action flags the column kernel writes (one byte per cell).
ACTION_CLEAR = 1
ACTION_TRIM = 2
ACTION_BUILD = 4
ACTION_LIGHT = 8
ACTION_COLD_SNOW = 16
ACTION_RETRIEVE = 32
ACTION_KEPT_WITHOUT_COVERAGE = 64

#: Column classes of the ``trim-build`` rule (low four bits), plus 16 when
#: the strongest echo sits on the first level, plus 32 when the column's
#: largest background rain plus snow sits on a level the radar did not
#: sample (a warm column only; under ``covered`` the trim ignores it).
COLUMN_CLASSES = {
    0: "no_coverage",
    1: "no_echo_cleared_per_level",
    2: "satellite_clear_whole_column",
    3: "cold_surface_snow",
    4: "warm_trimmed",
    5: "warm_built_at_strongest_echo",
    6: "strongest_echo_on_top_level",
}
COLUMN_FIRST_LEVEL_FLAG = 16
COLUMN_UNSAMPLED_MAXIMUM_FLAG = 32

__all__ = [
    "EXPERIMENTAL", "SCHEMA", "MODES", "THOMPSON_FIELDS", "SCOPES",
    "NO_ECHO_DBZ", "NO_COVERAGE_DBZ", "NOAA_SETTINGS",
    "HydrometeorAnalysisConfig", "HydrometeorAnalysisError",
    "HydrometeorAnalysisResult", "hydrometeor_analysis",
    "precipitating_fields", "radar_grid_reflectivity", "require_device",
    "require_thompson_rain_number", "temperature_from_density",
    "temperature_from_theta", "thompson_retrieval",
]


class HydrometeorAnalysisError(ValueError):
    """The stage cannot run as configured.  Never a warning."""


# ---------------------------------------------------------------------------
# CUDA source
# ---------------------------------------------------------------------------

_SOURCE = r"""
// Reflectivity classes, gsdcloudanalysis.F90:884-906.
__device__ __forceinline__ bool ha_echo(const double ref) { return ref > 0.0; }
__device__ __forceinline__ bool ha_no_echo(const double ref) {
    return ref <= 0.0 && ref > -100.0;
}
__device__ __forceinline__ float ha_f32(const double x) { return __double2float_rn(x); }
__device__ __forceinline__ double ha_min(const double a, const double b) { return (b < a) ? b : a; }
__device__ __forceinline__ double ha_max(const double a, const double b) { return (b > a) ? b : a; }

// hydro_mxr_thompson.f90:103-183, one point.  Outputs in NOAA's units
// (g/kg, /kg, g/kg) with its -999 sentinel below min_ref.
__device__ __forceinline__ void ha_thompson(
        const float t32, const float p32, const double ref,
        float* qr_gkg, float* qnr, float* qs_gkg) {
    if (!(ref >= 0.0)) {                                   // :121, :177-181
        *qr_gkg = -999.0f; *qnr = -999.0f; *qs_gkg = -999.0f;
        return;
    }
    const double n0r_mp = 8.0e6, am_s = 0.069, pi = 3.1415926536;   // :60-64
    const double rho_i = 890.0, rho_w = 1000.0, a_min = 1.0e-5;     // :65-66, :79
    // :92-95: default-real DATA constants stored in double arrays.
    const double sa1 = (double)5.065339f, sa2 = (double)-0.062659f,
                 sa3 = (double)-3.032362f, sa4 = (double)0.029469f,
                 sa5 = (double)-0.000285f, sa6 = (double)0.31255f,
                 sa7 = (double)0.000204f, sa8 = (double)0.003199f,
                 sa9 = (double)0.0f, sa10 = (double)-0.015952f;
    const double sb1 = (double)0.476221f, sb2 = (double)-0.015896f,
                 sb3 = (double)0.165977f, sb4 = (double)0.007468f,
                 sb5 = (double)-0.000141f, sb6 = (double)0.060366f,
                 sb7 = (double)0.000079f, sb8 = (double)0.000594f,
                 sb9 = (double)0.0f, sb10 = (double)-0.003577f;
    // :105, left to right as written.
    const double six_pi = __ddiv_rn(6.0, pi);
    const double ams_rhoi = __ddiv_rn(am_s, rho_i);
    const double f = __dmul_rn(__dmul_rn(__dmul_rn(__dmul_rn(
        __ddiv_rn(0.176, 0.93), six_pi), six_pi), ams_rhoi), ams_rhoi);
    const double cse3 = __dmul_rn(2.0, 2.0);                 // :108
    const double oams = __ddiv_rn(1.0, am_s);                // :109
    const double crg3 = 24.0, crg4 = 5040.0;                 // :113-114
    const double am_r = __ddiv_rn(__dmul_rn(pi, rho_w), 6.0);   // :115

    const double t = (double)t32;
    const double rho = __ddiv_rn((double)p32, __dmul_rn(287.0, t));   // :123
    const double tc = __dsub_rn(t, 273.15);                           // :124
    double rfract;                                                    // :126-132
    if (tc <= 0.0) rfract = 0.0;
    else if (tc >= 5.0) rfract = 1.0;
    else rfract = __dmul_rn(0.20, tc);
    const double zes = __dmul_rn(__dmul_rn(                           // :134-136
        glibc_pow(10.0, __dmul_rn(0.1, ha_min(ref, 28.0))),
        __dsub_rn(1.0, rfract)), 1.0e-18);
    const double zer = __dmul_rn(__dmul_rn(                           // :138-140
        glibc_pow(10.0, __dmul_rn(0.1, ha_min(ref, 55.0))), rfract), 1.0e-18);
    // :142 MIN(-0.1, tc): the literal is default real, promoted to double.
    const double tc0 = ha_min((double)(-0.1f), tc);

    double loga = sa1;                                                // :149-153
    loga = __dadd_rn(loga, __dmul_rn(sa2, tc0));
    loga = __dadd_rn(loga, __dmul_rn(sa3, cse3));
    loga = __dadd_rn(loga, __dmul_rn(__dmul_rn(sa4, tc0), cse3));
    loga = __dadd_rn(loga, __dmul_rn(__dmul_rn(sa5, tc0), tc0));
    loga = __dadd_rn(loga, __dmul_rn(__dmul_rn(sa6, cse3), cse3));
    loga = __dadd_rn(loga, __dmul_rn(__dmul_rn(__dmul_rn(sa7, tc0), tc0), cse3));
    loga = __dadd_rn(loga, __dmul_rn(__dmul_rn(__dmul_rn(sa8, tc0), cse3), cse3));
    loga = __dadd_rn(loga, __dmul_rn(__dmul_rn(__dmul_rn(sa9, tc0), tc0), tc0));
    loga = __dadd_rn(loga, __dmul_rn(__dmul_rn(__dmul_rn(sa10, cse3), cse3), cse3));
    const double a_ = ha_max(glibc_pow(10.0, loga), a_min);          // :154
    double b_ = sb1;                                                  // :155-158
    b_ = __dadd_rn(b_, __dmul_rn(sb2, tc0));
    b_ = __dadd_rn(b_, __dmul_rn(sb3, cse3));
    b_ = __dadd_rn(b_, __dmul_rn(__dmul_rn(sb4, tc0), cse3));
    b_ = __dadd_rn(b_, __dmul_rn(__dmul_rn(sb5, tc0), tc0));
    b_ = __dadd_rn(b_, __dmul_rn(__dmul_rn(sb6, cse3), cse3));
    b_ = __dadd_rn(b_, __dmul_rn(__dmul_rn(__dmul_rn(sb7, tc0), tc0), cse3));
    b_ = __dadd_rn(b_, __dmul_rn(__dmul_rn(__dmul_rn(sb8, tc0), cse3), cse3));
    b_ = __dadd_rn(b_, __dmul_rn(__dmul_rn(__dmul_rn(sb9, tc0), tc0), tc0));
    b_ = __dadd_rn(b_, __dmul_rn(__dmul_rn(__dmul_rn(sb10, cse3), cse3), cse3));

    const double qs = __ddiv_rn(                                      // :160
        glibc_pow(__ddiv_rn(zes, __dmul_rn(f, a_)), __ddiv_rn(1.0, b_)),
        __dmul_rn(rho, oams));
    *qs_gkg = ha_f32(__dmul_rn(1000.0, qs));                          // :161
    const double qr = __dmul_rn(                                      // :163
        __ddiv_rn(__dmul_rn(__dmul_rn(n0r_mp, am_r), crg3), rho),
        glibc_pow(__ddiv_rn(zer, __dmul_rn(n0r_mp, crg4)), __ddiv_rn(4.0, 7.0)));
    const float qn = ha_f32(__dmul_rn(                                // :164-165
        glibc_pow(__ddiv_rn(n0r_mp, rho), __ddiv_rn(3.0, 4.0)),
        glibc_pow(__ddiv_rn(qr, __dmul_rn(am_r, crg3)), __ddiv_rn(1.0, 4.0))));
    *qnr = ha_f32(ha_max(1.0, (double)qn));                           // :167
    *qr_gkg = ha_f32(__dmul_rn(1000.0, qr));                          // :168
}

// The retrieval as gsdcloudanalysis.F90 holds it after PrecipMxR_radar:
// hydro_mxr_thompson.f90:117 loops k = 2, nz-1, so the first and the last
// level keep the -99999 that :613-615 gave them.
__device__ __forceinline__ void ha_retrieved(
        const float* t, const float* p, const double* ref,
        const int k, const int nz, const long long ncol, const long long c,
        float* qr_gkg, float* qnr, float* qs_gkg) {
    if (k == 0 || k == nz - 1) {
        *qr_gkg = -99999.0f; *qnr = -99999.0f; *qs_gkg = -99999.0f;
        return;
    }
    const long long n = (long long)k * ncol + c;
    ha_thompson(t[n], p[n], ref[n], qr_gkg, qnr, qs_gkg);
}

// gsdcloudanalysis.F90:940-941: g/kg to kg/kg, floored at zero, stored single.
__device__ __forceinline__ float ha_kgkg(const float gkg) {
    return ha_f32(ha_max(__dmul_rn((double)gkg, 0.001), 0.0));
}

extern "C" __global__ void ha_thompson_retrieval(
        const float* __restrict__ t, const float* __restrict__ p,
        const double* __restrict__ ref,
        float* __restrict__ qr_gkg, float* __restrict__ qnr,
        float* __restrict__ qs_gkg,
        const int nz, const long long ncol) {
    const long long total = (long long)nz * ncol;
    for (long long n = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         n < total; n += (long long)gridDim.x * blockDim.x) {
        const int k = (int)(n / ncol);
        const long long c = n - (long long)k * ncol;
        ha_retrieved(t, p, ref, k, nz, ncol, c, qr_gkg + n, qnr + n, qs_gkg + n);
    }
}

// max(0.0_r_single, v) as gfortran evaluates it: a negative value becomes
// +0 and a zero keeps its sign (NOAA's clamp leaves -0.0 where a negative
// number moment was scaled by zero).
__device__ __forceinline__ float ha_clamp(const float v) { return (v < 0.0f) ? 0.0f : v; }
__device__ __forceinline__ bool ha_same(const float a, const float b) {
    return __float_as_int(a) == __float_as_int(b);
}

// The clamp of gsdcloudanalysis.F90:1053-1066, then the coverage scope:
// with scope_noaa = 0 a cell the radar did not sample (ref <= -100) keeps
// its input, and its action records 64 when NOAA's rule would have moved it.
__device__ __forceinline__ void ha_store(
        const long long n, const double r, const int scope_noaa,
        float rain, float nrain, float snow, float grau, signed char act,
        const float* qr, const float* qnr, const float* qs, const float* qg,
        float* o_qr, float* o_qnr, float* o_qs, float* o_qg,
        signed char* action) {
    rain = ha_clamp(rain); nrain = ha_clamp(nrain);
    snow = ha_clamp(snow); grau = ha_clamp(grau);
    if (!scope_noaa && !(r > -100.0)) {
        const bool moved = !ha_same(rain, qr[n]) || !ha_same(nrain, qnr[n])
                        || !ha_same(snow, qs[n]) || !ha_same(grau, qg[n]);
        rain = qr[n]; nrain = qnr[n]; snow = qs[n]; grau = qg[n];
        act = moved ? (signed char)64 : (signed char)0;
    }
    o_qr[n] = rain; o_qnr[n] = nrain; o_qs[n] = snow; o_qg[n] = grau;
    action[n] = act;
}

// gsdcloudanalysis.F90:910-927 for one field, with the clamp of :1053-1066
// and the coverage scope of ha_store.
extern "C" __global__ void ha_clear_field(
        const float* __restrict__ field, const double* __restrict__ ref,
        float* __restrict__ out, const long long total, const int scope_noaa) {
    for (long long n = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         n < total; n += (long long)gridDim.x * blockDim.x) {
        const double r = ref[n];
        float v = field[n];
        if (ha_no_echo(r)) v = 0.0f;
        v = ha_clamp(v);
        if (!scope_noaa && !(r > -100.0)) v = field[n];
        out[n] = v;
    }
}

// gsdcloudanalysis.F90:872-909 (mode 2) and :928-1049 (mode 1), one column
// per thread, then the clamp of :1053-1066.
extern "C" __global__ void ha_precip_column(
        const float* __restrict__ qr, const float* __restrict__ qnr,
        const float* __restrict__ qs, const float* __restrict__ qg,
        const float* __restrict__ t, const float* __restrict__ p,
        const double* __restrict__ ref,
        const float* __restrict__ ctp, const int has_ctp,
        float* __restrict__ o_qr, float* __restrict__ o_qnr,
        float* __restrict__ o_qs, float* __restrict__ o_qg,
        signed char* __restrict__ action, signed char* __restrict__ column_class,
        const int nz, const long long ncol, const int mode,
        const int i_lightpcp, const int iclean_withref, const int iclean_allcol,
        const double cold_threshold_c, const int scope_noaa) {
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         c < ncol; c += (long long)gridDim.x * blockDim.x) {
        signed char cls = 0;
        if (mode == 2) {
            const double qrlimit = __dmul_rn(15.0, 0.001);                // :873
            for (int k = 0; k < nz; ++k) {
                const long long n = (long long)k * ncol + c;
                float rr, rn, rs;
                ha_retrieved(t, p, ref, k, nz, ncol, c, &rr, &rn, &rs);
                float rain = qr[n], nrain = qnr[n], snow = qs[n], grau = qg[n];
                signed char act = 0;
                const double r = ref[n];
                if (ha_echo(r)) {                                         // :884-894
                    snow = ha_f32(ha_min(__dmul_rn(ha_max((double)rs, 0.0), 0.001), qrlimit));
                    const double raintemp = __dmul_rn(ha_max((double)rr, 0.0), 0.001);
                    if (raintemp <= qrlimit) {
                        rain = ha_f32(raintemp);
                        nrain = rn;
                    } else {
                        rain = ha_f32(qrlimit);
                        nrain = ha_f32(__dmul_rn((double)rn, __ddiv_rn(qrlimit, raintemp)));
                    }
                    act = 32;
                } else if (ha_no_echo(r)) {                               // :895-900
                    rain = 0.0f; nrain = 0.0f; snow = 0.0f; grau = 0.0f;
                    act = 1;
                }
                ha_store(n, r, scope_noaa, rain, nrain, snow, grau, act,
                         qr, qnr, qs, qg, o_qr, o_qnr, o_qs, o_qg, action);
            }
            column_class[c] = 0;
            continue;
        }

        const double qrlimit = __dmul_rn(3.0, 0.001);                     // :929
        const double qrlimit_light = __dmul_rn(1.0, 0.001);               // :930
        double refmax = -999.0;                                           // :933-939
        int imax = -1;
        for (int k = 0; k < nz; ++k) {
            const double r = ref[(long long)k * ncol + c];
            if (r > refmax) { imax = k; refmax = r; }
        }
        // :947  imaxlvl_ref > 0 .and. imaxlvl_ref < nsig, one-based.
        if (refmax > 0.0 && imax >= 0 && imax < nz - 1) {
            // :948  first-level temperature in Celsius.
            const double tsfc = __dsub_rn((double)t[c], 273.15);
            if (tsfc < cold_threshold_c) {                                // :949-972
                cls = 3;
                for (int k = 0; k < nz; ++k) {
                    const long long n = (long long)k * ncol + c;
                    float rr, rn, rs;
                    ha_retrieved(t, p, ref, k, nz, ncol, c, &rr, &rn, &rs);
                    double snowtemp = (double)ha_kgkg(rs);
                    float rain = qr[n], nrain = qnr[n], snow = qs[n], grau = qg[n];
                    signed char act = 0;
                    if (ha_echo(ref[n])) {
                        snowtemp = ha_min(ha_max(snowtemp, (double)qs[n]), qrlimit);
                        const double snowadd = ha_max(__dsub_rn(snowtemp, (double)snow), 0.0);
                        snow = ha_f32(snowtemp);
                        const double raintemp = (double)__fadd_rn(rain, grau);
                        if (raintemp > snowadd) {
                            if (raintemp > 1.0e-6) {
                                const double ratio2 = __dsub_rn(1.0, __ddiv_rn(snowadd, raintemp));
                                rain = ha_f32(__dmul_rn((double)rain, ratio2));
                                grau = ha_f32(__dmul_rn((double)grau, ratio2));
                            }
                        } else {
                            rain = 0.0f;
                            grau = 0.0f;
                        }
                        act = 16;
                    }
                    ha_store(n, ref[n], scope_noaa, rain, nrain, snow, grau, act,
                             qr, qnr, qs, qg, o_qr, o_qnr, o_qs, o_qg, action);
                }
            } else {                                                      // :973-1020
                float mr, mn, ms;
                ha_retrieved(t, p, ref, imax, nz, ncol, c, &mr, &mn, &ms);
                const float rain_m = ha_kgkg(mr), snow_m = ha_kgkg(ms);
                const double max_retrieved = (double)__fadd_rn(snow_m, rain_m);   // :974
                // :975-980 take the largest background over every level.
                // Under the covered scope only sampled levels count: the
                // unsampled ones are kept as given, so a ratio set by them
                // would cut the sampled levels and remove nothing there.
                double max_sampled = -999.0, max_unsampled = -999.0;
                for (int k = 0; k < nz; ++k) {
                    const long long n = (long long)k * ncol + c;
                    const double s = __dadd_rn((double)qr[n], (double)qs[n]);
                    if (ref[n] > -100.0) {
                        if (s > max_sampled) max_sampled = s;
                    } else if (s > max_unsampled) {
                        max_unsampled = s;
                    }
                }
                const double max_bk = scope_noaa
                    ? ha_max(max_sampled, max_unsampled) : max_sampled;
                const bool unsampled_max = max_unsampled > 0.0
                    && max_unsampled > max_sampled;
                const bool trim = max_bk > max_retrieved;                         // :981
                double ratio = 0.0;
                if (trim) ratio = ha_max(ha_min(__ddiv_rn(max_retrieved, max_bk), 1.0), 0.0);
                cls = trim ? 4 : 5;
                if (unsampled_max) cls |= 32;
                for (int k = 0; k < nz; ++k) {
                    const long long n = (long long)k * ncol + c;
                    float rain = qr[n], nrain = qnr[n], snow = qs[n], grau = qg[n];
                    signed char act = 0;
                    if (trim) {                                                   // :983-994
                        if ((double)qr[n] > 0.0) {
                            rain = ha_f32(__dmul_rn((double)qr[n], ratio));
                            nrain = ha_f32(__dmul_rn((double)qnr[n], ratio));
                            act |= 2;
                        }
                        if ((double)qs[n] > 0.0) {
                            snow = ha_f32(__dmul_rn((double)qs[n], ratio));
                            act |= 2;
                        }
                    } else if (k == imax) {                                       // :996-1007
                        snow = ha_f32(ha_min((double)snow_m, qrlimit));
                        rain = ha_f32(ha_min((double)rain_m, qrlimit));
                        nrain = mn;
                        act |= 4;
                    }
                    const double r = ref[n];
                    if (i_lightpcp == 1 && r >= 15.0 && r <= 28.0) {              // :1009-1019
                        float lr, ln, ls;
                        ha_retrieved(t, p, ref, k, nz, ncol, c, &lr, &ln, &ls);
                        rain = ha_f32(ha_max(ha_min((double)ha_kgkg(lr), qrlimit_light), (double)rain));
                        snow = ha_f32(ha_max(ha_min((double)ha_kgkg(ls), qrlimit_light), (double)snow));
                        nrain = (ln > nrain) ? ln : nrain;
                        act |= 8;
                    }
                    ha_store(n, r, scope_noaa, rain, nrain, snow, grau, act,
                             qr, qnr, qs, qg, o_qr, o_qnr, o_qs, o_qg, action);
                }
            }
            if (imax == 0) cls |= 16;
        } else {                                                          // :1021-1046
            const bool satellite_clear = iclean_allcol == 1
                && (refmax <= 0.0 && refmax >= -100.0)
                && has_ctp && ((double)ctp[c] >= 1010.0 && (double)ctp[c] < 1050.0);
            if (satellite_clear && iclean_withref == 1) cls = 2;
            else if (refmax > 0.0) cls = 6;
            else if (refmax > -100.0) cls = 1;
            else cls = 0;
            for (int k = 0; k < nz; ++k) {
                const long long n = (long long)k * ncol + c;
                float rain = qr[n], nrain = qnr[n], snow = qs[n], grau = qg[n];
                signed char act = 0;
                const double r = ref[n];
                if (iclean_withref == 1) {
                    if (satellite_clear || ha_no_echo(r)) {
                        rain = 0.0f; nrain = 0.0f; snow = 0.0f; grau = 0.0f;
                        act = 1;
                    }
                }
                ha_store(n, r, scope_noaa, rain, nrain, snow, grau, act,
                         qr, qnr, qs, qg, o_qr, o_qnr, o_qs, o_qg, action);
            }
        }
        column_class[c] = cls;
    }
}

// A gpuwm-obs.radar-grid document in NOAA's sentinel convention: echo keeps
// its dBZ, an observed-clear cell is -99, anything else is -99999.
extern "C" __global__ void ha_radar_grid_reflectivity(
        const float* __restrict__ z, const signed char* __restrict__ z_mask,
        const signed char* __restrict__ z0_mask, double* __restrict__ ref,
        const long long total) {
    for (long long n = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         n < total; n += (long long)gridDim.x * blockDim.x) {
        double v = -99999.0;
        if (z_mask[n]) {
            const float zv = z[n];
            const bool finite = (__float_as_int(zv) & 0x7f800000) != 0x7f800000;
            if (finite && zv > -100.0f) v = (double)zv;
        } else if (z0_mask[n]) {
            v = -99.0;
        }
        ref[n] = v;
    }
}

// T = (thb + thp) * (p / p0) ** (Rd / cp), double inside, float32 stored.
extern "C" __global__ void ha_temperature_from_theta(
        const float* __restrict__ thp, const float* __restrict__ thb,
        const int thb_is_column, const float* __restrict__ p,
        float* __restrict__ t, const long long total, const long long ncol,
        const double p0, const double rcp) {
    for (long long n = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         n < total; n += (long long)gridDim.x * blockDim.x) {
        const double base = (double)(thb_is_column ? thb[n / ncol] : thb[n]);
        const double theta = __dadd_rn(base, (double)thp[n]);
        t[n] = ha_f32(__dmul_rn(theta, glibc_pow(__ddiv_rn((double)p[n], p0), rcp)));
    }
}

// The equation of state: T = p * alt / (Rd * (1 + (Rv/Rd) qv)).
extern "C" __global__ void ha_temperature_from_density(
        const float* __restrict__ p, const float* __restrict__ alt,
        const float* __restrict__ qv, float* __restrict__ t,
        const long long total, const double rd, const double rvovrd) {
    for (long long n = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         n < total; n += (long long)gridDim.x * blockDim.x) {
        const double moist = __dadd_rn(1.0, __dmul_rn(rvovrd, (double)qv[n]));
        t[n] = ha_f32(__ddiv_rn(__dmul_rn((double)p[n], (double)alt[n]),
                                __dmul_rn(rd, moist)));
    }
}
"""

#: The device libm the retrieval's ``pow`` comes from: glibc 2.43's binary64
#: exp, log and pow, transcribed and measured equal to glibc's words.  Read,
#: never edited, from the kernel directory that owns it.
_LIBM_HEADER = "glibc_flt64.cuh"

#: Threads per block; the grid is a function of the problem size alone and
#: every output element is one thread's own arithmetic.
_THREADS = 256
_MAX_BLOCKS = 65535


@lru_cache(maxsize=None)
def _module():
    import cupy as cp

    from gpuwm.core import kernels as kernel_loader

    header = (Path(kernel_loader.__file__).parent / _LIBM_HEADER).read_text(
        encoding="utf-8")
    return cp.RawModule(code=header + _SOURCE, options=("-std=c++17",))


def _kernel(name: str):
    return _module().get_function(name)


def _blocks(total: int) -> int:
    return max(1, min(_MAX_BLOCKS, (int(total) + _THREADS - 1) // _THREADS))


# ---------------------------------------------------------------------------
# the device, and nothing else
# ---------------------------------------------------------------------------


def require_device():
    """CuPy, or a refusal that names what is missing.

    The breakage this prevents: a cycle configured to clear false rain
    carrying it into the next leg while its receipt reads as if the
    precipitation analysis ran.  There is no NumPy implementation to fall
    back to, on purpose: a second copy of NOAA's rule on the host would be a
    second authority for one analysis.
    """
    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # noqa: BLE001 - any failure is the same refusal
        raise HydrometeorAnalysisError(
            "the radar hydrometeor analysis (gpuwm.da.hydrometeor_analysis) "
            "runs on a CUDA device and this process cannot reach one "
            f"({type(exc).__name__}: {exc}). It has no host implementation, "
            "so without a device the analysis would keep the precipitation "
            "the radar observed as absent while the cycle's receipt named "
            "the stage. Run on a host with a CUDA device, or set the "
            "precipitation analysis off") from exc
    if count < 1:
        raise HydrometeorAnalysisError(
            "the radar hydrometeor analysis (gpuwm.da.hydrometeor_analysis) "
            "runs on a CUDA device and this process sees none. It has no "
            "host implementation; run on a host with a CUDA device, or set "
            "the precipitation analysis off")
    return cp


def _device_array(cp, value, dtype, name: str):
    """``value`` as a C-contiguous device array of ``dtype``.

    A host array is refused rather than uploaded: the function is a device
    stage, and a caller that hands it NumPy has usually built its inputs
    on the host by mistake.
    """
    if not isinstance(value, cp.ndarray):
        raise HydrometeorAnalysisError(
            f"{name} must be a CuPy array on the device, got "
            f"{type(value).__name__}; this stage does no host arithmetic "
            "and does not move data for its caller")
    if value.dtype != dtype:
        raise HydrometeorAnalysisError(
            f"{name} must be {np.dtype(dtype).name}, got {value.dtype}; the "
            "analysis reproduces NOAA's precision and does not cast for "
            "its caller")
    return cp.ascontiguousarray(value)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HydrometeorAnalysisConfig:
    """One mode and NOAA's settings for it.

    Every default is the value the HRRR runs; :data:`NOAA_SETTINGS` carries
    the file and line of each.  ``mode`` has no default: the three modes do
    different things and the caller names one.
    """

    mode: str
    #: r_cleanSnow_WarmTs_threshold (parm/conus/hrrr_gsiparm.anl.sh:171).
    cold_surface_threshold_c: float = 5.0
    #: i_lightpcp = 1 (parm/conus/hrrr_gsiparm.anl.sh:186).
    light_precipitation: bool = True
    #: iclean_hydro_withRef = 1 (rapidrefresh_cldsurf_mod.f90:332).
    clear_with_reflectivity: bool = True
    #: iclean_hydro_withRef_allcol = 1 (parm/conus/hrrr_gsiparm.anl.sh:181).
    clear_column_when_satellite_clear: bool = True
    #: Which cells may move; see :data:`SCOPES`.  Not a NOAA setting: the
    #: oracle runs ``noaa``, and the analysis seam runs ``covered``.
    scope: str = "covered"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise HydrometeorAnalysisError(
                f"mode must be one of {MODES}, got {self.mode!r}")
        if self.scope not in SCOPES:
            raise HydrometeorAnalysisError(
                f"scope must be one of {SCOPES}, got {self.scope!r}")
        threshold = float(self.cold_surface_threshold_c)
        if not np.isfinite(threshold):
            raise HydrometeorAnalysisError(
                "cold_surface_threshold_c must be finite, got "
                f"{self.cold_surface_threshold_c!r}")


class HydrometeorAnalysisResult(NamedTuple):
    """``(increments, receipt, analysed)``.

    ``analysed`` holds the fields after the analysis (float32, on the
    device); ``increments`` is ``analysed`` minus the input, exactly zero in
    every cell the stage did not move and exactly minus the input in every
    cell it cleared; ``receipt`` is JSON-serialisable.
    """

    increments: dict
    receipt: dict
    analysed: dict


# ---------------------------------------------------------------------------
# which fields, for which scheme
# ---------------------------------------------------------------------------


def precipitating_fields(available: Sequence[str], *,
                         mp_physics: int | None = None) -> tuple[str, ...]:
    """Every precipitating mass a state carries, with its paired moments.

    The fields ``clear`` zeroes, derived from :mod:`gpuwm.da.moments` so a
    mass is never cleared without the number (and volume) moment the scheme
    pairs with it.  Rain, snow, graupel and hail are the engine's own list
    of precipitating species (:data:`gpuwm.da.obsop.PRECIPITATING_SPECIES`).
    """
    from gpuwm.da.moments import pairs_present

    have = set(available)
    pairs = {pair.mass: pair
             for pair in pairs_present(tuple(available), mp_physics=mp_physics)}
    names: list[str] = []
    for mass in PRECIPITATING_MASSES:
        if mass not in have:
            continue
        names.append(mass)
        pair = pairs.get(mass)
        if pair is not None:
            names.append(pair.number)
            if pair.volume is not None:
                names.append(pair.volume)
    return tuple(names)


def require_thompson_rain_number(mp_physics, *, mode: str) -> None:
    """Refuse ``trim-build`` and ``retrieve-all`` for a scheme they do not fit.

    Both write the Thompson retrieval's rain mass, rain number and snow.
    The breakage this prevents: on a scheme that pairs snow or graupel with
    a number moment (Morrison, NSSL, Milbrandt-Yau) the rule would move the
    mass and leave that number at the background's value, and on a scheme
    with no rain number there is nowhere to put the retrieved one.
    """
    from gpuwm.da.moments import (THOMPSON_REPAIR_AUTHORITY,
                                  MomentPolicyError, scheme_moments)

    if mp_physics is None:
        raise HydrometeorAnalysisError(
            f"the {mode!r} precipitation analysis writes Thompson's rain "
            "mass, rain number and snow and needs the scheme stated "
            "(mp_physics); with none stated there is nothing to check the "
            "number moments against. State mp_physics, or use 'clear', "
            "which fits any scheme")
    try:
        scheme = scheme_moments(int(mp_physics))
    except MomentPolicyError as exc:
        raise HydrometeorAnalysisError(str(exc)) from exc
    rain = scheme.pair_for_mass("qr")
    fits = (scheme.repair_authority == THOMPSON_REPAIR_AUTHORITY
            and rain is not None and rain.number == "nr"
            and "qs" in scheme.mass_only and "qg" in scheme.mass_only)
    if not fits:
        raise HydrometeorAnalysisError(
            f"the {mode!r} precipitation analysis writes Thompson's rain "
            "mass, rain number and snow (NOAA's hydro_mxr_thompson), and "
            f"mp_physics={int(mp_physics)} ({scheme.name}) does not carry "
            "that moment set: its snow, graupel or rain number moments "
            "would be left at the background's values beside masses this "
            "rule moved. Use 'clear', which zeroes each precipitating mass "
            "together with its paired number for any scheme")


# ---------------------------------------------------------------------------
# inputs built on the device
# ---------------------------------------------------------------------------


def radar_grid_reflectivity(document: Mapping, *, z_source: str = "z_obs"):
    """``(reflectivity, provenance)`` in NOAA's sentinel convention.

    ``document`` is a read ``gpuwm-obs.radar-grid`` document.  A cell with
    ``z_mask`` set carries its dBZ, a cell with the clear-air mask
    ``z0_mask`` set is observed no echo (-99), and a cell with neither is
    no coverage (-99999).  The mapping runs on the device; the result is a
    float64 CuPy array ``(level, south_north, west_east)``.

    Refusals follow :mod:`gpuwm.da.obs_radar`'s for the clear-air batch,
    for the same reasons: a file with no clear-air assessment cannot tell
    observed no echo from a cell the beam never reached, and a clear-air
    source outside the allow-list may be built from range-folded gates.
    """
    from gpuwm.da.obs_radar import CLEAR_AIR_SOURCES, Z_SOURCES

    cp = require_device()
    if z_source not in Z_SOURCES:
        raise HydrometeorAnalysisError(
            f"z_source must be one of {Z_SOURCES}, got {z_source!r}")
    variables = document["variables"]
    absent = [name for name in (z_source, "z_mask", "z0_mask")
              if name not in variables]
    if absent:
        raise HydrometeorAnalysisError(
            f"the radar file carries no {absent}. The precipitation "
            "analysis needs echo (z_mask) AND a clear-air assessment "
            "(z0_mask): a false z_mask means only that no echo was "
            "recorded, which covers every cell the beam never reached, and "
            "reading those as observed no echo would clear precipitation "
            "over most of the domain. Rebuild the observation file with a "
            "build that establishes clear air")
    source = document.get("clear_air_source")
    if source not in CLEAR_AIR_SOURCES:
        raise HydrometeorAnalysisError(
            f"the radar file declares clear_air_source {source!r}; the "
            f"precipitation analysis reads {sorted(CLEAR_AIR_SOURCES)}. A "
            "source outside that list may count range-folded gates, which "
            "can be storms, as clear air, and this stage would then remove "
            "the rain under them")
    z = cp.asarray(np.ascontiguousarray(variables[z_source], dtype=np.float32))
    z_mask = cp.asarray(np.ascontiguousarray(
        np.asarray(variables["z_mask"]) != 0, dtype=np.int8))
    z0_mask = cp.asarray(np.ascontiguousarray(
        np.asarray(variables["z0_mask"]) != 0, dtype=np.int8))
    if z.shape != z_mask.shape or z.shape != z0_mask.shape or z.ndim != 3:
        raise HydrometeorAnalysisError(
            f"{z_source} {z.shape}, z_mask {z_mask.shape} and z0_mask "
            f"{z0_mask.shape} must be one (level, south_north, west_east) "
            "grid")
    overlap = int(cp.count_nonzero((z_mask != 0) & (z0_mask != 0)))
    if overlap:
        raise HydrometeorAnalysisError(
            f"{overlap} cell(s) are marked both as echo and as clear air; "
            "the writer vetoes clear air wherever any radar found echo, so "
            "the file is internally inconsistent and this stage would "
            "have to guess which half to believe")
    ref = cp.empty(z.shape, dtype=cp.float64)
    total = int(z.size)
    _kernel("ha_radar_grid_reflectivity")(
        (_blocks(total),), (_THREADS,),
        (z, z_mask, z0_mask, ref, np.int64(total)))
    provenance = {
        "schema": SCHEMA,
        "source_variable": z_source,
        "clear_air_source": source,
        "convention": ("echo keeps its dBZ; observed no echo (z0_mask) is "
                       f"{NO_ECHO_DBZ:g}; no coverage is "
                       f"{NO_COVERAGE_DBZ:g}; an echo cell at or below 0 "
                       "dBZ reads as observed no echo, as in NOAA's rule"),
        **_coverage(cp, ref),
    }
    return ref, provenance


def temperature_from_theta(thp, base_theta, pressure):
    """Air temperature (K, float32) from perturbation and base theta.

    ``T = (thb + thp) * (p / P0) ** (Rd / cp)``, the form the reflectivity
    operator diagnoses, evaluated in double on the device.  ``base_theta``
    is ``(nz,)`` or the full grid.
    """
    from gpuwm.core import constants as c

    cp = require_device()
    thp = _device_array(cp, thp, np.float32, "thp")
    pressure = _device_array(cp, pressure, np.float32, "pressure")
    base = _device_array(cp, base_theta, np.float32, "base_theta")
    if thp.ndim != 3 or pressure.shape != thp.shape:
        raise HydrometeorAnalysisError(
            f"thp {thp.shape} and pressure {pressure.shape} must be one "
            "(nz, ny, nx) grid")
    if base.shape not in ((thp.shape[0],), thp.shape):
        raise HydrometeorAnalysisError(
            f"base_theta must be (nz,) or {thp.shape}, got {base.shape}")
    out = cp.empty(thp.shape, dtype=cp.float32)
    total = int(thp.size)
    ncol = int(thp.shape[1] * thp.shape[2])
    _kernel("ha_temperature_from_theta")(
        (_blocks(total),), (_THREADS,),
        (thp, base, np.int32(base.ndim == 1), pressure, out,
         np.int64(total), np.int64(ncol), np.float64(c.P0),
         np.float64(c.RCP)))
    return out


def temperature_from_density(pressure, inverse_density, vapour):
    """Air temperature (K, float32) from the equation of state.

    ``T = p * alt / (Rd * (1 + (Rv/Rd) qv))``: what a checkpoint that does
    not carry the base state can still give
    (:func:`gpuwm.ensemble.increments.saturation_limit` reads the same
    form), evaluated in double on the device.
    """
    from gpuwm.core import constants as c

    cp = require_device()
    pressure = _device_array(cp, pressure, np.float32, "pressure")
    inverse_density = _device_array(cp, inverse_density, np.float32,
                                    "inverse_density")
    vapour = _device_array(cp, vapour, np.float32, "vapour")
    if not (pressure.shape == inverse_density.shape == vapour.shape):
        raise HydrometeorAnalysisError(
            f"pressure {pressure.shape}, inverse_density "
            f"{inverse_density.shape} and vapour {vapour.shape} must share "
            "one shape")
    out = cp.empty(pressure.shape, dtype=cp.float32)
    total = int(pressure.size)
    _kernel("ha_temperature_from_density")(
        (_blocks(total),), (_THREADS,),
        (pressure, inverse_density, vapour, out, np.int64(total),
         np.float64(c.RD), np.float64(c.RVOVRD)))
    return out


def thompson_retrieval(temperature, pressure, reflectivity):
    """NOAA's Thompson retrieval on a ``(nz, ny, nx)`` grid.

    Returns ``(qr, qnr, qs)`` in NOAA's units (g/kg, /kg, g/kg), float32 on
    the device, with -999 where the reflectivity is below 0 dBZ and -99999
    on the first and the last level, which ``hydro_mxr_thompson.f90:117``
    does not visit.
    """
    cp = require_device()
    temperature = _device_array(cp, temperature, np.float32, "temperature")
    pressure = _device_array(cp, pressure, np.float32, "pressure")
    reflectivity = _device_array(cp, reflectivity, np.float64, "reflectivity")
    shape = temperature.shape
    if len(shape) != 3 or pressure.shape != shape or reflectivity.shape != shape:
        raise HydrometeorAnalysisError(
            f"temperature {shape}, pressure {pressure.shape} and "
            f"reflectivity {reflectivity.shape} must be one (nz, ny, nx) "
            "grid")
    qr = cp.empty(shape, dtype=cp.float32)
    qnr = cp.empty(shape, dtype=cp.float32)
    qs = cp.empty(shape, dtype=cp.float32)
    total = int(temperature.size)
    _kernel("ha_thompson_retrieval")(
        (_blocks(total),), (_THREADS,),
        (temperature, pressure, reflectivity, qr, qnr, qs,
         np.int32(shape[0]), np.int64(shape[1] * shape[2])))
    return qr, qnr, qs


# ---------------------------------------------------------------------------
# the analysis
# ---------------------------------------------------------------------------


def _coverage(cp, reflectivity) -> dict:
    echo = int(cp.count_nonzero(reflectivity > 0.0))
    no_echo = int(cp.count_nonzero((reflectivity <= 0.0)
                                   & (reflectivity > -100.0)))
    return {"echo_cells": echo, "no_echo_cells": no_echo,
            "no_coverage_cells": int(reflectivity.size) - echo - no_echo}


def _field_receipt(cp, before, after, *, units: str) -> dict:
    change = after.astype(cp.float64) - before.astype(cp.float64)
    removed = float(cp.sum(cp.where(change < 0.0, -change, 0.0)))
    added = float(cp.sum(cp.where(change > 0.0, change, 0.0)))
    return {
        "units": units,
        "cells_decreased": int(cp.count_nonzero(change < 0.0)),
        "cells_increased": int(cp.count_nonzero(change > 0.0)),
        "removed_sum": removed,
        "added_sum": added,
        "min_after": float(cp.min(after)) if int(after.size) else 0.0,
    }


def _units(name: str) -> str:
    """The unit a field's removed and added sums are in.  A sum over cells
    of a mixing ratio is not a mass; it is the quantity this module can
    state without the cell air masses, and it is labelled as what it is."""
    if name.startswith("qvol"):
        return "m3 kg-1, summed over cells"
    if name.startswith("n") or name.startswith("qn"):
        return "kg-1, summed over cells"
    return "kg kg-1, summed over cells"


def hydrometeor_analysis(fields: Mapping, reflectivity,
                         cfg: HydrometeorAnalysisConfig, *,
                         temperature=None, pressure=None,
                         cloud_top_pressure=None) -> HydrometeorAnalysisResult:
    """NOAA's radar precipitation analysis for one member.

    ``fields`` maps state field names to the member's analysed values,
    float32 CuPy arrays ``(nz, ny, nx)``.  ``clear`` clears every field it
    is handed (:func:`precipitating_fields` builds the list for a scheme);
    ``trim-build`` and ``retrieve-all`` take exactly :data:`THOMPSON_FIELDS`, and
    need ``temperature`` (K) and ``pressure`` (Pa), float32 on the same grid.

    ``reflectivity`` is the observed reflectivity on that grid, float64, in
    NOAA's sentinel convention (:func:`radar_grid_reflectivity`).

    ``cloud_top_pressure`` (hPa, float32 ``(ny, nx)``) is optional.  It
    feeds one branch of ``trim-build`` only: a column with observed no echo and a
    cloud-top pressure from 1010 up to 1050 hPa is cleared at every level
    (gsdcloudanalysis.F90:1028-1034).  Without it only the per-level clear
    of :1036-1042 is reachable, and the receipt says so.

    Returns :class:`HydrometeorAnalysisResult`.  Equal inputs give equal
    bytes: no random numbers, no clock, no dependence on the launch grid.
    """
    if not isinstance(cfg, HydrometeorAnalysisConfig):
        raise TypeError(
            f"cfg must be a HydrometeorAnalysisConfig, got {type(cfg).__name__}")
    cp = require_device()
    if not fields:
        raise HydrometeorAnalysisError(
            "no field was given; an analysis that reads nothing would "
            "return an empty increment that looks like a clear sky")
    reflectivity = _device_array(cp, reflectivity, np.float64, "reflectivity")
    shape = tuple(int(n) for n in reflectivity.shape)
    if len(shape) != 3:
        raise HydrometeorAnalysisError(
            f"reflectivity must be (nz, ny, nx), got {shape}")
    names = tuple(fields)
    before = {}
    for name in names:
        value = _device_array(cp, fields[name], np.float32, f"field {name!r}")
        if tuple(value.shape) != shape:
            raise HydrometeorAnalysisError(
                f"field {name!r} is {tuple(value.shape)}, the reflectivity "
                f"grid is {shape}")
        before[name] = value
    nz, ny, nx = shape
    ncol = ny * nx
    total = nz * ncol
    scope_noaa = np.int32(cfg.scope == "noaa")

    after: dict = {}
    action = None
    columns = None
    if cfg.mode == "clear":
        for name in names:
            out = cp.empty(shape, dtype=cp.float32)
            _kernel("ha_clear_field")(
                (_blocks(total),), (_THREADS,),
                (before[name], reflectivity, out, np.int64(total), scope_noaa))
            after[name] = out
    else:
        if set(names) != set(THOMPSON_FIELDS):
            raise HydrometeorAnalysisError(
                f"the {cfg.mode!r} mode reads and writes exactly "
                f"{THOMPSON_FIELDS} (Thompson's rain mass, rain number, "
                f"snow and graupel), got {sorted(names)}")
        if temperature is None or pressure is None:
            raise HydrometeorAnalysisError(
                f"the {cfg.mode!r} mode runs NOAA's Thompson retrieval, "
                "which needs temperature (K) and pressure (Pa); "
                "temperature_from_theta and temperature_from_density build "
                "the temperature on the device")
        temperature = _device_array(cp, temperature, np.float32, "temperature")
        pressure = _device_array(cp, pressure, np.float32, "pressure")
        if tuple(temperature.shape) != shape or tuple(pressure.shape) != shape:
            raise HydrometeorAnalysisError(
                f"temperature {tuple(temperature.shape)} and pressure "
                f"{tuple(pressure.shape)} must match the grid {shape}")
        has_ctp = cloud_top_pressure is not None
        if has_ctp:
            ctp = _device_array(cp, cloud_top_pressure, np.float32,
                                "cloud_top_pressure")
            if tuple(ctp.shape) != (ny, nx):
                raise HydrometeorAnalysisError(
                    f"cloud_top_pressure must be {(ny, nx)}, got "
                    f"{tuple(ctp.shape)}")
        else:
            ctp = cp.zeros(1, dtype=cp.float32)
        outs = {name: cp.empty(shape, dtype=cp.float32)
                for name in THOMPSON_FIELDS}
        action = cp.empty(shape, dtype=cp.int8)
        columns = cp.empty((ny, nx), dtype=cp.int8)
        _kernel("ha_precip_column")(
            (_blocks(ncol),), (_THREADS,),
            (before["qr"], before["nr"], before["qs"], before["qg"],
             temperature, pressure, reflectivity, ctp, np.int32(has_ctp),
             outs["qr"], outs["nr"], outs["qs"], outs["qg"],
             action, columns, np.int32(nz), np.int64(ncol),
             np.int32(1 if cfg.mode == "trim-build" else 2),
             np.int32(bool(cfg.light_precipitation)),
             np.int32(bool(cfg.clear_with_reflectivity)),
             np.int32(bool(cfg.clear_column_when_satellite_clear)),
             np.float64(cfg.cold_surface_threshold_c), scope_noaa))
        after = {name: outs[name] for name in names}

    increments = {name: after[name] - before[name] for name in names}

    # -- the receipt, counted on the device ---------------------------------
    changed = None
    for name in names:
        moved = after[name] != before[name]
        changed = moved if changed is None else (changed | moved)
    no_echo = (reflectivity <= 0.0) & (reflectivity > -100.0)
    covered = reflectivity > -100.0
    if action is None:
        cells = {"cleared": int(cp.count_nonzero(no_echo & changed))}
    else:
        def _count(flag):
            return int(cp.count_nonzero(((action & flag) != 0) & changed))
        cells = {
            "cleared": _count(ACTION_CLEAR),
            "trimmed": _count(ACTION_TRIM),
            "built": _count(ACTION_BUILD),
            "light_precipitation": _count(ACTION_LIGHT),
            "cold_surface_snow": _count(ACTION_COLD_SNOW),
            "retrieved": _count(ACTION_RETRIEVE),
        }
        cells["kept_without_coverage"] = int(cp.count_nonzero(
            (action & ACTION_KEPT_WITHOUT_COVERAGE) != 0))
    cells["changed"] = int(cp.count_nonzero(changed))
    cells["changed_without_coverage"] = int(
        cp.count_nonzero(changed & ~covered))
    receipt = {
        "schema": SCHEMA,
        "stability": "experimental",
        "mode": cfg.mode,
        "device": "cuda",
        "noaa_source": {
            "repository": "NOAA-EMC/HRRR", "tag": "v4.1.21",
            "retrieval": ("sorc/hrrr_gsi.fd/libsrc/GSD/gsdcloud/"
                          "hydro_mxr_thompson.f90"),
            "analysis": {
                "clear": "sorc/hrrr_gsi.fd/src/gsi/gsdcloudanalysis.F90:910-927",
                "trim-build":
                    "sorc/hrrr_gsi.fd/src/gsi/gsdcloudanalysis.F90:928-1049",
                "retrieve-all":
                    "sorc/hrrr_gsi.fd/src/gsi/gsdcloudanalysis.F90:872-909",
            }[cfg.mode],
            "clamp": "sorc/hrrr_gsi.fd/src/gsi/gsdcloudanalysis.F90:1053-1066",
        },
        "config": asdict(cfg),
        "settings_source": {name: row["source"]
                            for name, row in NOAA_SETTINGS.items()},
        "grid_shape": [nz, ny, nx],
        "fields": list(names),
        "coverage": _coverage(cp, reflectivity),
        "clear_rule_cells": int(cp.count_nonzero(no_echo)),
        "cells": cells,
        "per_field": {name: _field_receipt(cp, before[name], after[name],
                                           units=_units(name))
                      for name in names},
        "removed_water": ("removed, not moved to another field: a declared "
                          "sink, as in the HRRR's cloud analysis"),
        "deterministic": True,
    }
    if cfg.mode == "trim-build":
        base = columns & 15
        receipt["columns"] = {
            label: int(cp.count_nonzero(base == code))
            for code, label in COLUMN_CLASSES.items()}
        receipt["columns"]["strongest_echo_on_first_level"] = int(
            cp.count_nonzero((columns & COLUMN_FIRST_LEVEL_FLAG) != 0))
        receipt["columns"]["background_maximum_on_unsampled_level"] = int(
            cp.count_nonzero((columns & COLUMN_UNSAMPLED_MAXIMUM_FLAG) != 0))
        receipt["trim_background_maximum"] = (
            "over every level of the column, as NOAA's rule takes it"
            if cfg.scope == "noaa" else
            "over the levels the radar sampled; under NOAA's every-level "
            "maximum the columns counted in "
            "background_maximum_on_unsampled_level would have had their "
            "sampled levels cut by a ratio set by unsampled rain")
        receipt["satellite_cloud_top_pressure"] = {
            "supplied": cloud_top_pressure is not None,
            "note": ("the whole-column clear of gsdcloudanalysis.F90:"
                     "1028-1034 needs satellite cloud-top pressure"
                     + ("" if cloud_top_pressure is not None else
                        "; none was given, so only the per-level clear of "
                        ":1036-1042 was reachable")),
        }
    return HydrometeorAnalysisResult(increments=increments, receipt=receipt,
                                     analysed=after)
