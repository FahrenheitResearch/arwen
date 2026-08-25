"""WRF v4.6.1 legacy RRTMG longwave (ra_lw_physics=4), NumPy FP32 reference.

Port authority: phys/module_ra_rrtmg_lw.F from the campaign reference
bundle (WRF_1974_MP55_reference_bundle, WRF_source_v4.6.1_group)
  sha256 cefe75fa749205a0451dceb873caa86651e999057b51e342acdb531d132d20f9
compiled with kind_rb = kind(1.0) (FP32, RWORDSIZE=4), EM_CORE=1, WRF_CHEM=0.

Every routine here is gated at max_ulp 0 against per-routine fixtures dumped
from the UNMODIFIED Fortran by tools/rrtmg_wrf461_oracle/lw_extract.F90.

FP32 discipline (the rules every function in this file follows):

* One Fortran statement maps to the same sequence of float32 operations,
  associating exactly as gfortran -O0 on x86-64 SSE2 does: left-to-right
  within equal precedence, parentheses honored, no FMA contraction, no
  reassociation.  numpy float32 elementwise ops are IEEE-754 correctly
  rounded, so a chain of single ops reproduces the Fortran chain bitwise.
* Vectorising over layers or g-points is allowed ONLY across independent
  iterations of a Fortran DO loop (each element sees the identical op
  sequence).  Sequential reductions keep their loop form.
* ``F(x)`` is a float32 constructor for literals; Fortran default-real
  literals (``1.``, ``0.5_rb``) become ``F("1.0")`` etc.  NumPy >= 2 weak
  promotion keeps float32 * python-scalar in float32.
* ``int()`` on a float32 truncates toward zero, matching Fortran INT().
* Transcendentals are NOT numpy's.  gfortran on x86-64 calls glibc, and
  glibc's FP32 logf/expf/powf are not correctly rounded.  ``logf``/``expf``/
  ``powf`` below adapt the single audited transcription of glibc 2.39
  sysdeps/ieee754/flt-32 in ``gpuwm.core.noahmp_libm`` (see Section 0).
  ``x**4`` with an integer literal
  exponent is NOT powf: gfortran expands it by squaring ((x*x)*(x*x)).
* Fortran 1-based indices (jp, jt, ind0, inds, ...) are carried 1-based in
  the state dict; every table access subtracts 1 at the access site.

Layout note: all coefficient tables keep their Fortran shapes; a Fortran
array A(i,j) appears here as numpy array A with A[i-1, j-1] the same
element (order='F' reshaping in the fixture reader preserves this).

The frozen integration contract for coefficients is
``gpuwm.ingest.rrtmg_coeffs.load_rrtmg_lw_coefficients(path)`` (sibling
lane) returning {module_name: {var_name: np.ndarray}} with the RAW
RRTMG_LW_DATA arrays (kao/kbo/selfrefo/...).  ``build_lw_coefficients``
below consumes exactly that structure; tests feed it the identical
structure read from the oracle's lw_coeffs.bin instead.
"""

from __future__ import annotations

import struct

import numpy as np
# numpy >= 2 is load-bearing for the FP32 max_ulp-0 discipline in this
# module: NEP-50 weak promotion keeps float32 op python-scalar in
# float32, while numpy 1.x would silently widen those chains to float64
# and break bitwise parity with the WRF oracle.  Fail closed at import.
if tuple(int(part) for part in np.__version__.split(".")[:2]) < (2, 0):
    raise ImportError(
        f"{__name__} requires numpy >= 2 (NEP-50 weak promotion is part "
        f"of its bitwise FP32 contract); found numpy {np.__version__}")


from gpuwm.core import noahmp_libm as _libm

F = np.float32
_U32 = 0xFFFFFFFF
MAX_RADIATION_LAYERS = 128

# ---------------------------------------------------------------------------
# Section 0 -- glibc 2.39 FP32 libm (logf, expf, powf).
#
# The single audited transcription of glibc 2.39's FP32 entry points lives
# in gpuwm.core.noahmp_libm (float32-in/float32-out on a Python-float
# carrier; each value is exactly a float32, so the F() re-wrap below is a
# type change, not a rounding).  The adapters only fix the carrier type to
# np.float32 for this module's scalar chains; they add no arithmetic.  The
# lane-local generic-C transcriptions that sat here until the integration
# merge were audited bitwise against noahmp_libm over 1.6e6 domain-relevant
# probes (0 mismatches, including glibc's two known multiarch expf residual
# arguments) before removal; the 179-case fixture deck re-gates the chain
# end-to-end on every run.  noahmp_libm raises NotImplementedError on
# argument classes outside its audited domain (negative/subnormal powf
# bases, logf of a non-positive) instead of returning an unvouched value;
# no RRTMG LW call site can produce such arguments (every powf base sits
# behind a ratX > const guard, every logf argument is a positive pressure).
# ---------------------------------------------------------------------------

GLIBC_VERSION = _libm.GLIBC_VERSION


def _asfloat(bits: int) -> np.float32:
    return F(struct.unpack("<f", struct.pack("<I", bits & _U32))[0])


def logf(x) -> np.float32:
    """glibc 2.39 ``logf`` via the audited shared transcription."""
    return F(_libm.logf(float(x)))


def expf(x) -> np.float32:
    """glibc 2.39 ``expf`` via the audited shared transcription."""
    return F(_libm.expf(float(x)))


def powf(x, y) -> np.float32:
    """glibc 2.39 ``powf`` via the audited shared transcription."""
    return F(_libm.powf(float(x), float(y)))


def logf_v(arr):
    """Elementwise logf over a float32 array (scalar transcription applied
    per element; independent lanes, so bitwise identical to a Fortran loop)."""
    flat = np.asarray(arr, dtype=np.float32).ravel()
    out = np.empty_like(flat)
    for i, v in enumerate(flat):
        out[i] = logf(v)
    return out.reshape(np.shape(arr))


def expf_v(arr):
    """Elementwise expf over a float32 array."""
    flat = np.asarray(arr, dtype=np.float32).ravel()
    out = np.empty_like(flat)
    for i, v in enumerate(flat):
        out[i] = expf(v)
    return out.reshape(np.shape(arr))


def powf_v(arr, y):
    """Elementwise powf(x, y) with scalar y over a float32 array."""
    flat = np.asarray(arr, dtype=np.float32).ravel()
    out = np.empty_like(flat)
    for i, v in enumerate(flat):
        out[i] = powf(v, y)
    return out.reshape(np.shape(arr))


def pow4(x):
    """Fortran x**4 with an INTEGER literal exponent: gfortran expands by
    squaring, (x*x)*(x*x), which differs bitwise from powf(x, 4.0)."""
    x = np.asarray(x, dtype=np.float32)
    x2 = (x * x).astype(np.float32)
    return (x2 * x2).astype(np.float32)


def trunc_int(x):
    """Fortran INT(): truncation toward zero.  Works on arrays/scalars."""
    return np.trunc(np.asarray(x, dtype=np.float32)).astype(np.int32)


# ---------------------------------------------------------------------------
# Section 1 -- structural constants (parrrtm + lwdatinit + lwcmbdat).
# Integer structure is hand-transcribed from the Fortran (verified against
# the oracle's post-init module dump by tests); float DATA blocks live in
# the loader (raw file) or the generated blob at the bottom of this file.
# ---------------------------------------------------------------------------

NBNDLW = 16
NGPTLW = 140
MXMOL = 38
MAXXSEC = 4
MG = 16
NTBL = 10000
TBLINT = F(10000.0)

# lwdatinit
NG = np.full(16, 16, dtype=np.int32)
NSPA = np.array([1, 1, 9, 9, 9, 1, 9, 1, 9, 1, 1, 9, 9, 1, 9, 9],
                dtype=np.int32)
NSPB = np.array([1, 1, 5, 5, 5, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
                dtype=np.int32)
WAVENUM1 = np.array([10., 350., 500., 630., 700., 820., 980., 1080.,
                     1180., 1390., 1480., 1800., 2080., 2250., 2380.,
                     2600.], dtype=np.float32)
WAVENUM2 = np.array([350., 500., 630., 700., 820., 980., 1080., 1180.,
                     1390., 1480., 1800., 2080., 2250., 2380., 2600.,
                     3250.], dtype=np.float32)
DELWAVE = np.array([340., 150., 130., 70., 120., 160., 100., 100., 210.,
                    90., 320., 280., 170., 130., 220., 650.],
                   dtype=np.float32)
NXMOL = 4
IXINDX = np.array([1, 2, 3, 4] + [0] * (MAXXSEC + 34 - 4), dtype=np.int32)

# lwcmbdat
NGC = np.array([10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2],
               dtype=np.int32)
NGS = np.array([10, 22, 38, 52, 68, 76, 88, 96, 108, 114, 122, 130, 134,
                136, 138, 140], dtype=np.int32)
NGM = np.array([
    1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10,          # band 1
    1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12,     # band 2
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,    # band 3
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14,    # band 4
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,    # band 5
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,           # band 6
    1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12,      # band 7
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,           # band 8
    1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12,     # band 9
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,           # band 10
    1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8,           # band 11
    1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,           # band 12
    1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,           # band 13
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,           # band 14
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,           # band 15
    1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,           # band 16
], dtype=np.int32)
NGN = np.array([
    1, 1, 2, 2, 2, 2, 2, 2, 1, 1,                             # band 1
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,                       # band 2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,           # band 3
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3,                 # band 4
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,           # band 5
    2, 2, 2, 2, 2, 2, 2, 2,                                   # band 6
    2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,                       # band 7
    2, 2, 2, 2, 2, 2, 2, 2,                                   # band 8
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,                       # band 9
    2, 2, 2, 2, 4, 4,                                         # band 10
    1, 1, 2, 2, 2, 2, 3, 3,                                   # band 11
    1, 1, 1, 1, 2, 2, 4, 4,                                   # band 12
    3, 3, 4, 6,                                               # band 13
    8, 8,                                                     # band 14
    8, 8,                                                     # band 15
    4, 12,                                                    # band 16
], dtype=np.int32)
NGB = np.repeat(np.arange(1, 17, dtype=np.int32), NGC)
WT = np.array(["0.1527534276", "0.1491729617", "0.1420961469",
               "0.1316886544", "0.1181945205", "0.1019300893",
               "0.0832767040", "0.0626720116", "0.0424925000",
               "0.0046269894", "0.0038279891", "0.0030260086",
               "0.0022199750", "0.0014140010", "0.0005330000",
               "0.0000750000"], dtype=np.float32)

# rtrnmc module data
WTDIFF = F("0.5")
REC_6 = F("0.166667")
A0 = np.array(["1.66", "1.55", "1.58", "1.66", "1.54", "1.454", "1.89",
               "1.33", "1.668", "1.66", "1.66", "1.66", "1.66", "1.66",
               "1.66", "1.66"], dtype=np.float32)
A1 = np.array(["0.00", "0.25", "0.22", "0.00", "0.13", "0.446", "-0.10",
               "0.40", "-0.006", "0.00", "0.00", "0.00", "0.00", "0.00",
               "0.00", "0.00"], dtype=np.float32)
A2 = np.array(["0.00", "-12.0", "-11.7", "0.00", "-0.72", "-0.243",
               "0.19", "-0.062", "0.414", "0.00", "0.00", "0.00", "0.00",
               "0.00", "0.00", "0.00"], dtype=np.float32)


def lwdatinit_scalars(cpdair):
    """The computed scalars of lwdatinit (FP32, exact statement order)."""
    grav = F("9.8066")
    avogad = F("6.02214199e+23")
    secdy = F("8.6400e4")
    oneminus = F(F("1.0") - F("1.e-6"))
    # pi = 2 * asin(1.); glibc asinf(1.0f) is the correctly rounded pi/2.
    pi = F(F("2.0") * _asfloat(0x3FC90FDB))
    fluxfac = F(pi * F("2.e4"))
    heatfac = F(F(grav * secdy) / F(F(cpdair) * F("1.e2")))
    return {
        "grav": grav, "avogad": avogad, "secdy": secdy,
        "oneminus": oneminus, "pi": pi, "fluxfac": fluxfac,
        "heatfac": heatfac,
    }


# ---------------------------------------------------------------------------
# Section 2 -- init chain: rwgt, lookup tables, cmbgb1..16 g-point reduction.
# build_lw_coefficients(raw, cpdair) consumes the RAW module dict from the
# frozen loader contract and returns every derived array the compute chain
# needs, bitwise-equal to the Fortran post-init module state.
# ---------------------------------------------------------------------------


def _compute_rwgt():
    """rrtmg_lw_ini's relative weights for the g-point reduction (FP32)."""
    rwgt = np.zeros(NBNDLW * MG, dtype=np.float32)
    igcsm = 0
    for ibnd in range(1, NBNDLW + 1):
        iprsm = 0
        if NGC[ibnd - 1] < MG:
            wtsm = np.zeros(MG, dtype=np.float32)
            for igc in range(1, NGC[ibnd - 1] + 1):
                igcsm += 1
                wtsum = F(0.0)
                for _ in range(int(NGN[igcsm - 1])):
                    iprsm += 1
                    wtsum = F(wtsum + WT[iprsm - 1])
                wtsm[igc - 1] = wtsum
            for ig in range(1, int(NG[ibnd - 1]) + 1):
                ind = (ibnd - 1) * MG + ig
                rwgt[ind - 1] = F(WT[ig - 1] / wtsm[NGM[ind - 1] - 1])
        else:
            for ig in range(1, int(NG[ibnd - 1]) + 1):
                igcsm += 1
                ind = (ibnd - 1) * MG + ig
                rwgt[ind - 1] = F(1.0)
    return rwgt


def _compute_tables():
    """The 10001-entry tau/transmittance/transition tables (rrtmg_lw_ini)."""
    tau_tbl = np.zeros(NTBL + 1, dtype=np.float32)
    exp_tbl = np.zeros(NTBL + 1, dtype=np.float32)
    tfn_tbl = np.zeros(NTBL + 1, dtype=np.float32)
    expeps = F("1.e-20")
    pade = F("0.278")
    bpade = F(F("1.0") / pade)
    tau_tbl[0] = F(0.0)
    tau_tbl[NTBL] = F("1.e10")
    exp_tbl[0] = F(1.0)
    exp_tbl[NTBL] = expeps
    tfn_tbl[0] = F(0.0)
    tfn_tbl[NTBL] = F(1.0)
    one = F(1.0)
    for itr in range(1, NTBL):
        tfn = F(F(itr) / F(NTBL))
        tau_tbl[itr] = F(F(bpade * tfn) / F(one - tfn))
        e = expf(F(-tau_tbl[itr]))
        if e <= expeps:
            e = expeps
        exp_tbl[itr] = e
        if tau_tbl[itr] < F("0.06"):
            tfn_tbl[itr] = F(tau_tbl[itr] / F("6.0"))
        else:
            t1 = F(one / tau_tbl[itr])
            t2 = F(exp_tbl[itr] / F(one - exp_tbl[itr]))
            tfn_tbl[itr] = F(one - F(F("2.0") * F(t1 - t2)))
    return tau_tbl, exp_tbl, tfn_tbl, bpade


def _reduce_g(raw_arr, band, rwgt, weighted=True):
    """One cmbgb weighted reduction along the LAST axis (the g-point axis).

    Mirrors the sumk loops: for each new g-point igc of `band`, sum the
    consecutive original g-points (ngn of them), each times rwgt (offset by
    (band-1)*16), sequentially in FP32.  With weighted=False mirrors the
    fracref plain sums.
    """
    ngc_b = int(NGC[band - 1])
    ngn_b = _ngn_for_band(band)
    out_shape = raw_arr.shape[:-1] + (ngc_b,)
    out = np.zeros(out_shape, dtype=np.float32)
    iprsm = 0
    for igc in range(ngc_b):
        n = ngn_b[igc]
        acc = np.zeros(raw_arr.shape[:-1], dtype=np.float32)
        for _ in range(n):
            iprsm += 1
            term = raw_arr[..., iprsm - 1]
            if weighted:
                term = (term * rwgt[(band - 1) * MG + iprsm - 1]) \
                    .astype(np.float32)
            acc = (acc + term).astype(np.float32)
        out[..., igc] = acc
    return out


def _ngn_for_band(band):
    start = int(NGC[:band - 1].sum())
    return [int(v) for v in NGN[start:start + int(NGC[band - 1])]]


def build_lw_coefficients(raw, cpdair):
    """Raw loader dict -> derived coefficient dict (post-init module state).

    ``raw``: {module: {var: ndarray}} per the frozen loader contract
    (rrlw_kg01..rrlw_kg16 with kao/kbo/selfrefo/forrefo/fracrefao/... plus
    any minor-gas arrays), FP32, Fortran shapes.

    Returns a flat dict keyed 'kgNN/name' for the reduced band arrays plus
    'tbl/...', 'con/...' entries.  NOTE: the full per-band reduction set
    (cmbgb1..cmbgb16) is completed by _CMBGB_IMPLS below; bands not yet
    ported raise KeyError at access time rather than returning wrong data.
    """
    rwgt = _compute_rwgt()
    tau_tbl, exp_tbl, tfn_tbl, bpade = _compute_tables()
    con = lwdatinit_scalars(cpdair)
    out = {
        "tbl/tau_tbl": tau_tbl, "tbl/exp_tbl": exp_tbl,
        "tbl/tfn_tbl": tfn_tbl, "tbl/bpade": bpade,
        "wvn/rwgt": rwgt,
    }
    for name, val in con.items():
        out[f"con/{name}"] = val
    for band in range(1, 17):
        impl = _CMBGB_IMPLS.get(band)
        if impl is None:
            continue
        mod = raw[f"rrlw_kg{band:02d}"]
        impl(mod, rwgt, out)
    return out


def _cmbgb1(mod, rwgt, out):
    """cmbgb1: ka/kb over (5,13|47,g), self/for, ka_mn2/kb_mn2, fracref."""
    out["kg01/ka"] = _reduce_g(mod["kao"], 1, rwgt)
    out["kg01/kb"] = _reduce_g(mod["kbo"], 1, rwgt)
    out["kg01/selfref"] = _reduce_g(mod["selfrefo"], 1, rwgt)
    out["kg01/forref"] = _reduce_g(mod["forrefo"], 1, rwgt)
    out["kg01/ka_mn2"] = _reduce_g(mod["kao_mn2"], 1, rwgt)
    out["kg01/kb_mn2"] = _reduce_g(mod["kbo_mn2"], 1, rwgt)
    out["kg01/fracrefa"] = _reduce_g(mod["fracrefao"], 1, rwgt,
                                     weighted=False)
    out["kg01/fracrefb"] = _reduce_g(mod["fracrefbo"], 1, rwgt,
                                     weighted=False)


#: band -> reduction implementation; filled in as bands are ported.
_CMBGB_IMPLS = {1: _cmbgb1}


# ---------------------------------------------------------------------------
# Section 3 -- inatm (rrtmg_lw_rad::inatm, module lines 11186-11522).
# ---------------------------------------------------------------------------

NMOL = 7


def inatm(iplon, nlay, icld, iaer, play, plev, tlay, tlev, tsfc, h2ovmr,
          o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, cfc11vmr, cfc12vmr,
          cfc22vmr, ccl4vmr, emis, inflglw, iceflglw, liqflglw,
          cldfmcl, taucmcl, ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl,
          resnmcl, tauaer, C):
    """Port of inatm.  2-D inputs are (ncol, nlay) exactly as in the
    Fortran (iplon selects the column, 1-based); mcica inputs are
    (ngptlw, ncol, nlay).  Returns the Fortran output arguments as a dict;
    pz/tz are length nlayers+1 with [0] = Fortran (0) element.

    WRF-undefined behaviour note: with icld == 0 the Fortran leaves
    inflag/iceflag/liqflag undefined (intent(out), never assigned) and
    cldprmc then reads inflag.  WRF's wrapper always passes
    icld = cldovrlp >= 1, so the read is unreachable in WRF; this port
    fails closed instead of reproducing an undefined read.
    """
    if icld < 1:
        raise ValueError(
            "icld < 1: WRF never drives this path and the Fortran would "
            "read undefined inflag/iceflag/liqflag (documented divergence)")
    grav = C["con/grav"]
    avogad = C["con/avogad"]
    amd = F("28.9660")
    amw = F("18.0160")

    nlayers = int(nlay)
    ip = int(iplon) - 1
    wkl = np.zeros((MXMOL, nlayers), dtype=np.float32)
    wx = np.zeros((MAXXSEC, nlayers), dtype=np.float32)
    pavel = np.zeros(nlayers, dtype=np.float32)
    tavel = np.zeros(nlayers, dtype=np.float32)
    pz = np.zeros(nlayers + 1, dtype=np.float32)
    tz = np.zeros(nlayers + 1, dtype=np.float32)
    coldry = np.zeros(nlayers, dtype=np.float32)
    wbrodl = np.zeros(nlayers, dtype=np.float32)
    semiss = np.zeros(NBNDLW, dtype=np.float32)
    cldfmc = np.zeros((NGPTLW, nlayers), dtype=np.float32)
    taucmc = np.zeros((NGPTLW, nlayers), dtype=np.float32)
    ciwpmc = np.zeros((NGPTLW, nlayers), dtype=np.float32)
    clwpmc = np.zeros((NGPTLW, nlayers), dtype=np.float32)
    cswpmc = np.zeros((NGPTLW, nlayers), dtype=np.float32)
    reicmc = np.zeros(nlayers, dtype=np.float32)
    relqmc = np.zeros(nlayers, dtype=np.float32)
    resnmc = np.zeros(nlayers, dtype=np.float32)
    taua = np.zeros((nlayers, NBNDLW), dtype=np.float32)

    amttl = F(0.0)
    wvttl = F(0.0)
    tbound = F(tsfc[ip])
    pz[0] = plev[ip, 0]
    tz[0] = tlev[ip, 0]
    for lidx in range(nlayers):
        pavel[lidx] = play[ip, lidx]
        tavel[lidx] = tlay[ip, lidx]
        pz[lidx + 1] = plev[ip, lidx + 1]
        tz[lidx + 1] = tlev[ip, lidx + 1]
        wkl[0, lidx] = h2ovmr[ip, lidx]
        wkl[1, lidx] = co2vmr[ip, lidx]
        wkl[2, lidx] = o3vmr[ip, lidx]
        wkl[3, lidx] = n2ovmr[ip, lidx]
        wkl[5, lidx] = ch4vmr[ip, lidx]
        wkl[6, lidx] = o2vmr[ip, lidx]
        amm = F(F(F(F("1.0") - wkl[0, lidx]) * amd)
                + F(wkl[0, lidx] * amw))
        num = F(F(F(pz[lidx] - pz[lidx + 1]) * F("1.e3")) * avogad)
        den = F(F(F(F("1.e2") * grav) * amm)
                * F(F("1.0") + wkl[0, lidx]))
        coldry[lidx] = F(num / den)

    for lidx in range(nlayers):
        wx[0, lidx] = ccl4vmr[ip, lidx]
        wx[1, lidx] = cfc11vmr[ip, lidx]
        wx[2, lidx] = cfc12vmr[ip, lidx]
        wx[3, lidx] = cfc22vmr[ip, lidx]

    for lidx in range(nlayers):
        summol = F(0.0)
        for imol in range(2, NMOL + 1):
            summol = F(summol + wkl[imol - 1, lidx])
        wbrodl[lidx] = F(coldry[lidx] * F(F("1.0") - summol))
        for imol in range(1, NMOL + 1):
            wkl[imol - 1, lidx] = F(coldry[lidx] * wkl[imol - 1, lidx])
        amttl = F(F(amttl + coldry[lidx]) + wkl[0, lidx])
        wvttl = F(wvttl + wkl[0, lidx])
        for ix in range(1, MAXXSEC + 1):
            if IXINDX[ix - 1] != 0:
                wx[IXINDX[ix - 1] - 1, lidx] = F(
                    F(coldry[lidx] * wx[ix - 1, lidx]) * F("1.e-20"))

    wvsh = F(F(amw * wvttl) / F(amd * amttl))
    pwvcm = F(F(wvsh * F(F("1.e3") * pz[0])) / F(F("1.e2") * grav))

    for n in range(NBNDLW):
        semiss[n] = emis[ip, n]

    if iaer >= 1:
        for lidx in range(nlayers):
            for ib in range(NBNDLW):
                taua[lidx, ib] = tauaer[ip, lidx, ib]

    inflag = int(inflglw)
    iceflag = int(iceflglw)
    liqflag = int(liqflglw)
    cldfmc[:, :] = cldfmcl[:, ip, :]
    taucmc[:, :] = taucmcl[:, ip, :]
    ciwpmc[:, :] = ciwpmcl[:, ip, :]
    clwpmc[:, :] = clwpmcl[:, ip, :]
    cswpmc[:, :] = cswpmcl[:, ip, :]
    reicmc[:] = reicmcl[ip, :]
    relqmc[:] = relqmcl[ip, :]
    resnmc[:] = resnmcl[ip, :]

    return {
        "nlayers": nlayers, "pavel": pavel, "pz": pz, "tavel": tavel,
        "tz": tz, "tbound": tbound, "semiss": semiss, "coldry": coldry,
        "wkl": wkl, "wbrodl": wbrodl, "wx": wx, "pwvcm": pwvcm,
        "inflag": inflag, "iceflag": iceflag, "liqflag": liqflag,
        "cldfmc": cldfmc, "taucmc": taucmc, "ciwpmc": ciwpmc,
        "clwpmc": clwpmc, "cswpmc": cswpmc, "reicmc": reicmc,
        "relqmc": relqmc, "resnmc": resnmc, "taua": taua,
    }


# ---------------------------------------------------------------------------
# Section 4 -- cldprmc (rrtmg_lw_cldprmc, module lines 2764-3025).
# ---------------------------------------------------------------------------

CLDMIN = F("1.e-20")


def cldprmc(nlayers, inflag, iceflag, liqflag, cldfmc, ciwpmc, clwpmc,
            cswpmc, reicmc, relqmc, resnmc, taucmc, C):
    """Port of cldprmc.  Mutates and returns (ncbands, taucmc).

    WRF reaches only inflag >= 2 with iceflag in {3,4,5} and liqflag = 1
    (see the wrapper); the other branches are ported for completeness
    except inflag 0/1, which return/stop in the Fortran:
    inflag == 0 returns immediately (taucmc already set); inflag == 1
    STOPs in the Fortran under McICA and raises here.
    """
    absice0 = C["cld/absice0"]
    absice1 = C["cld/absice1"]
    absice2 = C["cld/absice2"]
    absice3 = C["cld/absice3"]
    absliq0 = C["cld/absliq0"]
    absliq1 = C["cld/absliq1"]
    ngb = C["wvn/ngb"]
    icb = np.array([1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5],
                   dtype=np.int32)

    nl = int(nlayers)
    ncbands = 1
    taucmc = np.array(taucmc, dtype=np.float32, copy=True)

    for lay in range(1, nl + 1):
        radice = F(reicmc[lay - 1])
        for ig in range(1, NGPTLW + 1):
            cwp = F(F(ciwpmc[ig - 1, lay - 1] + clwpmc[ig - 1, lay - 1])
                    + cswpmc[ig - 1, lay - 1])
            if not (cldfmc[ig - 1, lay - 1] >= CLDMIN
                    and (cwp >= CLDMIN
                         or taucmc[ig - 1, lay - 1] >= CLDMIN)):
                continue
            if inflag == 0:
                return ncbands, taucmc
            if inflag == 1:
                raise ValueError(
                    "INFLAG = 1 OPTION NOT AVAILABLE WITH MCICA")
            # inflag >= 2
            ice_plus_snow = F(ciwpmc[ig - 1, lay - 1]
                              + cswpmc[ig - 1, lay - 1])
            if ice_plus_snow == F(0.0):
                abscoice = F(0.0)
                abscosno = F(0.0)
            elif iceflag == 0:
                if radice < F("10.0"):
                    raise ValueError("ICE RADIUS TOO SMALL")
                abscoice = F(absice0[0] + F(absice0[1] / radice))
                abscosno = F(0.0)
            elif iceflag == 1:
                if radice < F("13.0") or radice > F("130.0"):
                    raise ValueError("ICE RADIUS OUT OF BOUNDS")
                ncbands = 5
                ib = int(icb[int(ngb[ig - 1]) - 1])
                abscoice = F(absice1[0, ib - 1]
                             + F(absice1[1, ib - 1] / radice))
                abscosno = F(0.0)
            elif iceflag == 2:
                if radice < F("5.0") or radice > F("131.0"):
                    raise ValueError("ICE RADIUS OUT OF BOUNDS")
                ncbands = 16
                factor = F(F(radice - F("2.0")) / F("3.0"))
                index = int(factor)
                if index == 43:
                    index = 42
                fint = F(factor - F(index))
                ib = int(ngb[ig - 1])
                abscoice = F(absice2[index - 1, ib - 1]
                             + F(fint * F(absice2[index, ib - 1]
                                          - absice2[index - 1, ib - 1])))
                abscosno = F(0.0)
            else:  # iceflag >= 3
                if radice < F("5.0") or radice > F("140.0"):
                    raise ValueError(
                        "ICE GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS")
                ncbands = 16
                factor = F(F(radice - F("2.0")) / F("3.0"))
                index = int(factor)
                if index == 46:
                    index = 45
                fint = F(factor - F(index))
                ib = int(ngb[ig - 1])
                abscoice = F(absice3[index - 1, ib - 1]
                             + F(fint * F(absice3[index, ib - 1]
                                          - absice3[index - 1, ib - 1])))
                abscosno = F(0.0)

            if cswpmc[ig - 1, lay - 1] > F(0.0) and iceflag == 5:
                radsno = F(resnmc[lay - 1])
                if radsno < F("5.0") or radsno > F("140.0"):
                    raise ValueError(
                        "SNOW GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS")
                ncbands = 16
                factor = F(F(radsno - F("2.0")) / F("3.0"))
                index = int(factor)
                if index == 46:
                    index = 45
                fint = F(factor - F(index))
                ib = int(ngb[ig - 1])
                abscosno = F(absice3[index - 1, ib - 1]
                             + F(fint * F(absice3[index, ib - 1]
                                          - absice3[index - 1, ib - 1])))

            if clwpmc[ig - 1, lay - 1] == F(0.0):
                abscoliq = F(0.0)
            elif liqflag == 0:
                abscoliq = F(absliq0)
            else:  # liqflag == 1
                radliq = F(relqmc[lay - 1])
                if radliq < F("2.5") or radliq > F("60.0"):
                    raise ValueError(
                        "LIQUID EFFECTIVE RADIUS OUT OF BOUNDS")
                index = int(F(radliq - F("1.5")))
                if index == 0:
                    index = 1
                if index == 58:
                    index = 57
                fint = F(F(radliq - F("1.5")) - F(index))
                ib = int(ngb[ig - 1])
                abscoliq = F(absliq1[index - 1, ib - 1]
                             + F(fint * F(absliq1[index, ib - 1]
                                          - absliq1[index - 1, ib - 1])))

            taucmc[ig - 1, lay - 1] = F(
                F(F(ciwpmc[ig - 1, lay - 1] * abscoice)
                  + F(clwpmc[ig - 1, lay - 1] * abscoliq))
                + F(cswpmc[ig - 1, lay - 1] * abscosno))

    return ncbands, taucmc


# ---------------------------------------------------------------------------
# Section 5 -- setcoef.
# ---------------------------------------------------------------------------


def setcoef(nlayers, istart, pavel, tavel, tz, tbound, semiss, coldry,
            wkl, wbroad, C):
    """Port of rrtmg_lw_setcoef::setcoef (module lines 3556-3921).

    Scalar per-layer loop, FP32 statement for statement.  ``tz`` has
    Fortran extent 0:nlayers passed as a python array of length nlayers+1
    (tz[0] is Fortran tz(0)).  ``C`` must provide 'wvn/totplnk' (181,16),
    'wvn/totplk16' (181), 'ref/pref-log' etc under keys
    'ref/preflog' (59), 'ref/tref' (59), 'ref/chi_mls' (7,59).

    Returns a dict of the Fortran output arguments; index arrays keep
    their 1-based Fortran values.  Arrays that setcoef leaves undefined in
    a region (rat_h2oo3/h2on2o/h2och4/n2oco2 above laytrop; rat_o3co2,
    indself/selffrac below-only) are zero-filled there -- callers must not
    read them there, exactly as in the Fortran.
    """
    totplnk = C["wvn/totplnk"]
    totplk16 = C["wvn/totplk16"]
    preflog = C["ref/preflog"]
    tref = C["ref/tref"]
    chi_mls = C["ref/chi_mls"]

    nl = int(nlayers)
    jp = np.zeros(nl, dtype=np.int32)
    jt = np.zeros(nl, dtype=np.int32)
    jt1 = np.zeros(nl, dtype=np.int32)
    planklay = np.zeros((nl, NBNDLW), dtype=np.float32)
    planklev = np.zeros((nl + 1, NBNDLW), dtype=np.float32)
    plankbnd = np.zeros(NBNDLW, dtype=np.float32)
    colh2o = np.zeros(nl, dtype=np.float32)
    colco2 = np.zeros(nl, dtype=np.float32)
    colo3 = np.zeros(nl, dtype=np.float32)
    coln2o = np.zeros(nl, dtype=np.float32)
    colco = np.zeros(nl, dtype=np.float32)
    colch4 = np.zeros(nl, dtype=np.float32)
    colo2 = np.zeros(nl, dtype=np.float32)
    colbrd = np.zeros(nl, dtype=np.float32)
    fac00 = np.zeros(nl, dtype=np.float32)
    fac01 = np.zeros(nl, dtype=np.float32)
    fac10 = np.zeros(nl, dtype=np.float32)
    fac11 = np.zeros(nl, dtype=np.float32)
    rat = {name: np.zeros(nl, dtype=np.float32) for name in (
        "rat_h2oco2", "rat_h2oco2_1", "rat_h2oo3", "rat_h2oo3_1",
        "rat_h2on2o", "rat_h2on2o_1", "rat_h2och4", "rat_h2och4_1",
        "rat_n2oco2", "rat_n2oco2_1", "rat_o3co2", "rat_o3co2_1")}
    selffac = np.zeros(nl, dtype=np.float32)
    selffrac = np.zeros(nl, dtype=np.float32)
    indself = np.zeros(nl, dtype=np.int32)
    forfac = np.zeros(nl, dtype=np.float32)
    forfrac = np.zeros(nl, dtype=np.float32)
    indfor = np.zeros(nl, dtype=np.int32)
    minorfrac = np.zeros(nl, dtype=np.float32)
    scaleminor = np.zeros(nl, dtype=np.float32)
    scaleminorn2 = np.zeros(nl, dtype=np.float32)
    indminor = np.zeros(nl, dtype=np.int32)

    stpfac = F(F("296.0") / F("1013.0"))
    tbound = F(tbound)

    indbound = int(F(tbound - F("159.0")))
    if indbound < 1:
        indbound = 1
    elif indbound > 180:
        indbound = 180
    tbndfrac = F(F(tbound - F("159.0")) - F(indbound))
    tz0 = F(tz[0])
    indlev0 = int(F(tz0 - F("159.0")))
    if indlev0 < 1:
        indlev0 = 1
    elif indlev0 > 180:
        indlev0 = 180
    t0frac = F(F(tz0 - F("159.0")) - F(indlev0))
    laytrop = 0

    for lay in range(1, nl + 1):
        tav = F(tavel[lay - 1])
        indlay = int(F(tav - F("159.0")))
        if indlay < 1:
            indlay = 1
        elif indlay > 180:
            indlay = 180
        tlayfrac = F(F(tav - F("159.0")) - F(indlay))
        tzl = F(tz[lay])
        indlev = int(F(tzl - F("159.0")))
        if indlev < 1:
            indlev = 1
        elif indlev > 180:
            indlev = 180
        tlevfrac = F(F(tzl - F("159.0")) - F(indlev))

        for iband in range(1, 16):
            if lay == 1:
                dbdtlev = F(totplnk[indbound, iband - 1]
                            - totplnk[indbound - 1, iband - 1])
                plankbnd[iband - 1] = F(
                    F(semiss[iband - 1])
                    * F(totplnk[indbound - 1, iband - 1]
                        + F(tbndfrac * dbdtlev)))
                dbdtlev = F(totplnk[indlev0, iband - 1]
                            - totplnk[indlev0 - 1, iband - 1])
                planklev[0, iband - 1] = F(
                    totplnk[indlev0 - 1, iband - 1] + F(t0frac * dbdtlev))
            dbdtlev = F(totplnk[indlev, iband - 1]
                        - totplnk[indlev - 1, iband - 1])
            dbdtlay = F(totplnk[indlay, iband - 1]
                        - totplnk[indlay - 1, iband - 1])
            planklay[lay - 1, iband - 1] = F(
                totplnk[indlay - 1, iband - 1] + F(tlayfrac * dbdtlay))
            planklev[lay, iband - 1] = F(
                totplnk[indlev - 1, iband - 1] + F(tlevfrac * dbdtlev))

        iband = 16
        if istart == 16:
            if lay == 1:
                dbdtlev = F(totplk16[indbound] - totplk16[indbound - 1])
                plankbnd[iband - 1] = F(
                    F(semiss[iband - 1])
                    * F(totplk16[indbound - 1] + F(tbndfrac * dbdtlev)))
                dbdtlev = F(totplnk[indlev0, iband - 1]
                            - totplnk[indlev0 - 1, iband - 1])
                planklev[0, iband - 1] = F(
                    totplk16[indlev0 - 1] + F(t0frac * dbdtlev))
            dbdtlev = F(totplk16[indlev] - totplk16[indlev - 1])
            dbdtlay = F(totplk16[indlay] - totplk16[indlay - 1])
            planklay[lay - 1, iband - 1] = F(
                totplk16[indlay - 1] + F(tlayfrac * dbdtlay))
            planklev[lay, iband - 1] = F(
                totplk16[indlev - 1] + F(tlevfrac * dbdtlev))
        else:
            if lay == 1:
                dbdtlev = F(totplnk[indbound, iband - 1]
                            - totplnk[indbound - 1, iband - 1])
                plankbnd[iband - 1] = F(
                    F(semiss[iband - 1])
                    * F(totplnk[indbound - 1, iband - 1]
                        + F(tbndfrac * dbdtlev)))
                dbdtlev = F(totplnk[indlev0, iband - 1]
                            - totplnk[indlev0 - 1, iband - 1])
                planklev[0, iband - 1] = F(
                    totplnk[indlev0 - 1, iband - 1] + F(t0frac * dbdtlev))
            dbdtlev = F(totplnk[indlev, iband - 1]
                        - totplnk[indlev - 1, iband - 1])
            dbdtlay = F(totplnk[indlay, iband - 1]
                        - totplnk[indlay - 1, iband - 1])
            planklay[lay - 1, iband - 1] = F(
                totplnk[indlay - 1, iband - 1] + F(tlayfrac * dbdtlay))
            planklev[lay, iband - 1] = F(
                totplnk[indlev - 1, iband - 1] + F(tlevfrac * dbdtlev))

        plog = logf(F(pavel[lay - 1]))
        jp_l = int(F(F("36.0") - F(F(5.0) * F(plog + F("0.04")))))
        if jp_l < 1:
            jp_l = 1
        elif jp_l > 58:
            jp_l = 58
        jp[lay - 1] = jp_l
        jp1 = jp_l + 1
        fp = F(F("5.0") * F(preflog[jp_l - 1] - plog))

        jt_l = int(F(F("3.0") + F(F(tav - tref[jp_l - 1]) / F("15.0"))))
        if jt_l < 1:
            jt_l = 1
        elif jt_l > 4:
            jt_l = 4
        jt[lay - 1] = jt_l
        ft = F(F(F(tav - tref[jp_l - 1]) / F("15.0")) - F(jt_l - 3))
        jt1_l = int(F(F("3.0") + F(F(tav - tref[jp1 - 1]) / F("15.0"))))
        if jt1_l < 1:
            jt1_l = 1
        elif jt1_l > 4:
            jt1_l = 4
        jt1[lay - 1] = jt1_l
        ft1 = F(F(F(tav - tref[jp1 - 1]) / F("15.0")) - F(jt1_l - 3))
        water = F(wkl[0, lay - 1] / coldry[lay - 1])
        scalefac = F(F(F(pavel[lay - 1]) * stpfac) / tav)

        plog_le = bool(plog <= F("4.56"))
        if not plog_le:
            laytrop += 1

            forfac[lay - 1] = F(scalefac / F(F(1.0) + water))
            factor = F(F(F("332.0") - tav) / F("36.0"))
            indfor[lay - 1] = min(2, max(1, int(factor)))
            forfrac[lay - 1] = F(factor - F(int(indfor[lay - 1])))

            selffac[lay - 1] = F(water * forfac[lay - 1])
            factor = F(F(tav - F("188.0")) / F("7.2"))
            indself[lay - 1] = min(9, max(1, int(factor) - 7))
            selffrac[lay - 1] = F(factor - F(int(indself[lay - 1]) + 7))

            scaleminor[lay - 1] = F(F(pavel[lay - 1]) / tav)
            scaleminorn2[lay - 1] = F(
                F(F(pavel[lay - 1]) / tav)
                * F(wbroad[lay - 1]
                    / F(coldry[lay - 1] + wkl[0, lay - 1])))
            factor = F(F(tav - F("180.8")) / F("7.2"))
            indminor[lay - 1] = min(18, max(1, int(factor)))
            minorfrac[lay - 1] = F(factor - F(int(indminor[lay - 1])))

            jpm1 = jp_l - 1
            rat["rat_h2oco2"][lay - 1] = F(chi_mls[0, jpm1]
                                           / chi_mls[1, jpm1])
            rat["rat_h2oco2_1"][lay - 1] = F(chi_mls[0, jpm1 + 1]
                                             / chi_mls[1, jpm1 + 1])
            rat["rat_h2oo3"][lay - 1] = F(chi_mls[0, jpm1]
                                          / chi_mls[2, jpm1])
            rat["rat_h2oo3_1"][lay - 1] = F(chi_mls[0, jpm1 + 1]
                                            / chi_mls[2, jpm1 + 1])
            rat["rat_h2on2o"][lay - 1] = F(chi_mls[0, jpm1]
                                           / chi_mls[3, jpm1])
            rat["rat_h2on2o_1"][lay - 1] = F(chi_mls[0, jpm1 + 1]
                                             / chi_mls[3, jpm1 + 1])
            rat["rat_h2och4"][lay - 1] = F(chi_mls[0, jpm1]
                                           / chi_mls[5, jpm1])
            rat["rat_h2och4_1"][lay - 1] = F(chi_mls[0, jpm1 + 1]
                                             / chi_mls[5, jpm1 + 1])
            rat["rat_n2oco2"][lay - 1] = F(chi_mls[3, jpm1]
                                           / chi_mls[1, jpm1])
            rat["rat_n2oco2_1"][lay - 1] = F(chi_mls[3, jpm1 + 1]
                                             / chi_mls[1, jpm1 + 1])
        else:
            forfac[lay - 1] = F(scalefac / F(F(1.0) + water))
            factor = F(F(tav - F("188.0")) / F("36.0"))
            indfor[lay - 1] = 3
            forfrac[lay - 1] = F(factor - F("1.0"))

            selffac[lay - 1] = F(water * forfac[lay - 1])

            scaleminor[lay - 1] = F(F(pavel[lay - 1]) / tav)
            scaleminorn2[lay - 1] = F(
                F(F(pavel[lay - 1]) / tav)
                * F(wbroad[lay - 1]
                    / F(coldry[lay - 1] + wkl[0, lay - 1])))
            factor = F(F(tav - F("180.8")) / F("7.2"))
            indminor[lay - 1] = min(18, max(1, int(factor)))
            minorfrac[lay - 1] = F(factor - F(int(indminor[lay - 1])))

            jpm1 = jp_l - 1
            rat["rat_h2oco2"][lay - 1] = F(chi_mls[0, jpm1]
                                           / chi_mls[1, jpm1])
            rat["rat_h2oco2_1"][lay - 1] = F(chi_mls[0, jpm1 + 1]
                                             / chi_mls[1, jpm1 + 1])
            rat["rat_o3co2"][lay - 1] = F(chi_mls[2, jpm1]
                                          / chi_mls[1, jpm1])
            rat["rat_o3co2_1"][lay - 1] = F(chi_mls[2, jpm1 + 1]
                                            / chi_mls[1, jpm1 + 1])

        colh2o[lay - 1] = F(F("1.e-20") * wkl[0, lay - 1])
        colco2[lay - 1] = F(F("1.e-20") * wkl[1, lay - 1])
        colo3[lay - 1] = F(F("1.e-20") * wkl[2, lay - 1])
        coln2o[lay - 1] = F(F("1.e-20") * wkl[3, lay - 1])
        colco[lay - 1] = F(F("1.e-20") * wkl[4, lay - 1])
        colch4[lay - 1] = F(F("1.e-20") * wkl[5, lay - 1])
        colo2[lay - 1] = F(F("1.e-20") * wkl[6, lay - 1])
        if colco2[lay - 1] == F(0.0):
            colco2[lay - 1] = F(F("1.e-32") * coldry[lay - 1])
        if colo3[lay - 1] == F(0.0):
            colo3[lay - 1] = F(F("1.e-32") * coldry[lay - 1])
        if coln2o[lay - 1] == F(0.0):
            coln2o[lay - 1] = F(F("1.e-32") * coldry[lay - 1])
        if colco[lay - 1] == F(0.0):
            colco[lay - 1] = F(F("1.e-32") * coldry[lay - 1])
        if colch4[lay - 1] == F(0.0):
            colch4[lay - 1] = F(F("1.e-32") * coldry[lay - 1])
        colbrd[lay - 1] = F(F("1.e-20") * wbroad[lay - 1])

        compfp = F(F(1.0) - fp)
        fac10[lay - 1] = F(compfp * ft)
        fac00[lay - 1] = F(compfp * F(F("1.0") - ft))
        fac11[lay - 1] = F(fp * ft1)
        fac01[lay - 1] = F(fp * F(F("1.0") - ft1))

        selffac[lay - 1] = F(colh2o[lay - 1] * selffac[lay - 1])
        forfac[lay - 1] = F(colh2o[lay - 1] * forfac[lay - 1])

    out = {
        "laytrop": laytrop, "jp": jp, "jt": jt, "jt1": jt1,
        "planklay": planklay, "planklev": planklev, "plankbnd": plankbnd,
        "colh2o": colh2o, "colco2": colco2, "colo3": colo3,
        "coln2o": coln2o, "colco": colco, "colch4": colch4,
        "colo2": colo2, "colbrd": colbrd,
        "fac00": fac00, "fac01": fac01, "fac10": fac10, "fac11": fac11,
        "selffac": selffac, "selffrac": selffrac, "indself": indself,
        "forfac": forfac, "forfrac": forfrac, "indfor": indfor,
        "minorfrac": minorfrac, "scaleminor": scaleminor,
        "scaleminorn2": scaleminorn2, "indminor": indminor,
    }
    out.update(rat)
    return out


# ---------------------------------------------------------------------------
# Section 6 -- taumol band routines.  Each _taugbN fills taug/fracs columns
# for its band's g-points.  st: state dict (setcoef outputs + pavel, wx,
# coldry, col*), C: coefficient dict from build_lw_coefficients (+ ref data).
# Vectorised over layers within each region; g-points via slices.
# ---------------------------------------------------------------------------


def _taugb1(st, C, taug, fracs):
    """Band 1: 10-350 cm-1 (key h2o low+high, minor n2 low+high).

    Fortran: module lines 5073-5166.
    """
    ka = C["kg01/ka"]          # (5,13,ng1) -- 'absa' equivalence: absa[(jp*5+jt), ig] == ka
    kb = C["kg01/kb"]          # (5,47,ng1), jp index 13..59
    selfref = C["kg01/selfref"]
    forref = C["kg01/forref"]
    ka_mn2 = C["kg01/ka_mn2"]
    kb_mn2 = C["kg01/kb_mn2"]
    fracrefa = C["kg01/fracrefa"]
    fracrefb = C["kg01/fracrefb"]
    absa = ka.reshape(65, -1, order="F")
    absb = kb.reshape(235, -1, order="F")

    nl = len(st["pavel"])
    laytrop = st["laytrop"]
    ng1 = int(NGC[0])
    gs = 0  # g-point offset of band 1

    lo = slice(0, laytrop)
    jp = st["jp"][lo].astype(np.int64)
    jt = st["jt"][lo].astype(np.int64)
    jt1 = st["jt1"][lo].astype(np.int64)
    ind0 = ((jp - 1) * 5 + (jt - 1)) * int(NSPA[0]) + 1
    ind1 = (jp * 5 + (jt1 - 1)) * int(NSPA[0]) + 1
    inds = st["indself"][lo].astype(np.int64)
    indf = st["indfor"][lo].astype(np.int64)
    indm = st["indminor"][lo].astype(np.int64)
    pp = st["pavel"][lo]
    corradj = np.where(
        pp < F("250.0"),
        (F("1.0") - (F("0.15") * (F("250.0") - pp).astype(np.float32)
                     / F("154.4")).astype(np.float32)).astype(np.float32),
        F("1.0")).astype(np.float32)

    scalen2 = (st["colbrd"][lo] * st["scaleminorn2"][lo]).astype(np.float32)
    selffac = st["selffac"][lo][:, None]
    selffrac = st["selffrac"][lo][:, None]
    forfac = st["forfac"][lo][:, None]
    forfrac = st["forfrac"][lo][:, None]
    minorfrac = st["minorfrac"][lo][:, None]
    colh2o = st["colh2o"][lo][:, None]

    tauself = (selffac * (selfref[inds - 1, :]
               + (selffrac * (selfref[inds, :] - selfref[inds - 1, :]
                              ).astype(np.float32)).astype(np.float32)
               ).astype(np.float32)).astype(np.float32)
    taufor = (forfac * (forref[indf - 1, :]
              + (forfrac * (forref[indf, :] - forref[indf - 1, :]
                            ).astype(np.float32)).astype(np.float32)
              ).astype(np.float32)).astype(np.float32)
    taun2 = (scalen2[:, None] * (ka_mn2[indm - 1, :]
             + (minorfrac * (ka_mn2[indm, :] - ka_mn2[indm - 1, :]
                             ).astype(np.float32)).astype(np.float32)
             ).astype(np.float32)).astype(np.float32)

    fac00 = st["fac00"][lo][:, None]
    fac10 = st["fac10"][lo][:, None]
    fac01 = st["fac01"][lo][:, None]
    fac11 = st["fac11"][lo][:, None]
    tmaj = (fac00 * absa[ind0 - 1, :]).astype(np.float32)
    tmaj = (tmaj + (fac10 * absa[ind0, :]).astype(np.float32)) \
        .astype(np.float32)
    tmaj = (tmaj + (fac01 * absa[ind1 - 1, :]).astype(np.float32)) \
        .astype(np.float32)
    tmaj = (tmaj + (fac11 * absa[ind1, :]).astype(np.float32)) \
        .astype(np.float32)
    t = (colh2o * tmaj).astype(np.float32)
    t = (t + tauself).astype(np.float32)
    t = (t + taufor).astype(np.float32)
    t = (t + taun2).astype(np.float32)
    taug[lo, gs:gs + ng1] = (corradj[:, None] * t).astype(np.float32)
    fracs[lo, gs:gs + ng1] = fracrefa[None, :]

    hi = slice(laytrop, nl)
    jp = st["jp"][hi].astype(np.int64)
    jt = st["jt"][hi].astype(np.int64)
    jt1 = st["jt1"][hi].astype(np.int64)
    ind0 = ((jp - 13) * 5 + (jt - 1)) * int(NSPB[0]) + 1
    ind1 = ((jp - 12) * 5 + (jt1 - 1)) * int(NSPB[0]) + 1
    indf = st["indfor"][hi].astype(np.int64)
    indm = st["indminor"][hi].astype(np.int64)
    pp = st["pavel"][hi]
    corradj = (F("1.0") - (F("0.15") * (pp / F("95.6")).astype(np.float32)
                           ).astype(np.float32)).astype(np.float32)

    scalen2 = (st["colbrd"][hi] * st["scaleminorn2"][hi]).astype(np.float32)
    forfac = st["forfac"][hi][:, None]
    forfrac = st["forfrac"][hi][:, None]
    minorfrac = st["minorfrac"][hi][:, None]
    colh2o = st["colh2o"][hi][:, None]

    taufor = (forfac * (forref[indf - 1, :]
              + (forfrac * (forref[indf, :] - forref[indf - 1, :]
                            ).astype(np.float32)).astype(np.float32)
              ).astype(np.float32)).astype(np.float32)
    taun2 = (scalen2[:, None] * (kb_mn2[indm - 1, :]
             + (minorfrac * (kb_mn2[indm, :] - kb_mn2[indm - 1, :]
                             ).astype(np.float32)).astype(np.float32)
             ).astype(np.float32)).astype(np.float32)

    fac00 = st["fac00"][hi][:, None]
    fac10 = st["fac10"][hi][:, None]
    fac01 = st["fac01"][hi][:, None]
    fac11 = st["fac11"][hi][:, None]
    tmaj = (fac00 * absb[ind0 - 1, :]).astype(np.float32)
    tmaj = (tmaj + (fac10 * absb[ind0, :]).astype(np.float32)) \
        .astype(np.float32)
    tmaj = (tmaj + (fac01 * absb[ind1 - 1, :]).astype(np.float32)) \
        .astype(np.float32)
    tmaj = (tmaj + (fac11 * absb[ind1, :]).astype(np.float32)) \
        .astype(np.float32)
    t = (colh2o * tmaj).astype(np.float32)
    t = (t + taufor).astype(np.float32)
    t = (t + taun2).astype(np.float32)
    taug[hi, gs:gs + ng1] = (corradj[:, None] * t).astype(np.float32)
    fracs[hi, gs:gs + ng1] = fracrefb[None, :]


#: band number -> implementation; bands land here as they are ported.
TAUGB_IMPLS = {1: _taugb1}


# ---------------------------------------------------------------------------
# Section 7 -- rtrnmc (rrtmg_lw_rtrnmc, module lines 3085-3522).
#
# Vectorised over each band's g-points (independent lanes); the layer
# marches stay sequential.  The drad/clrdrad (and urad/clrurad)
# accumulations replicate the Fortran's per-g-point ordering exactly:
# Fortran adds one g-point's full profile at a time, so at every level the
# accumulator sums lanes in lane order; the clear-sky accumulators are
# OVERWRITTEN with the total-sky running sum while a lane is still
# cloud-free (iclddn = 0), which this port reproduces lane by lane.
# ---------------------------------------------------------------------------


def rtrnmc(nlayers, istart, iend, iout, pz, semiss, ncbands, cldfmc,
           taucmc, planklay, planklev, plankbnd, pwvcm, fracs, taut, C):
    """Port of rtrnmc.  pz/planklev have the Fortran 0: lower bound folded
    to python index 0.  Returns dict with totuflux/totdflux/fnet/htr and
    the clear-sky quartet, each of length nlayers+1 (index 0 = surface).
    """
    if iout != 0:
        raise NotImplementedError("iout /= 0 band-by-band output not ported")
    tau_tbl = C["tbl/tau_tbl"]
    exp_tbl = C["tbl/exp_tbl"]
    tfn_tbl = C["tbl/tfn_tbl"]
    bpade = F(C["tbl/bpade"])
    fluxfac = F(C["con/fluxfac"])
    heatfac = F(C["con/heatfac"])
    ngb = C["wvn/ngb"]
    delwave = C["wvn/delwave"]
    ngs = NGS

    nl = int(nlayers)
    f32 = np.float32

    secdiff = np.zeros(NBNDLW, dtype=f32)
    for ibnd in range(1, NBNDLW + 1):
        if ibnd == 1 or ibnd == 4 or ibnd >= 10:
            secdiff[ibnd - 1] = F("1.66")
        else:
            sd = F(A0[ibnd - 1] + F(A1[ibnd - 1]
                                    * expf(F(A2[ibnd - 1] * F(pwvcm)))))
            if sd > F("1.80"):
                sd = F("1.80")
            if sd < F("1.50"):
                sd = F("1.50")
            secdiff[ibnd - 1] = sd

    # Prologue: cloud optical depth / absorptivity per (lay, ig).
    odcld = np.zeros((nl, NGPTLW), dtype=f32)
    abscld = np.zeros((nl, NGPTLW), dtype=f32)
    efclfrac = np.zeros((nl, NGPTLW), dtype=f32)
    icldlyr = np.zeros(nl, dtype=np.int32)
    for lay in range(nl):
        for ig in range(NGPTLW):
            if cldfmc[ig, lay] == F(1.0):
                ib = int(ngb[ig])
                od = F(secdiff[ib - 1] * taucmc[ig, lay])
                odcld[lay, ig] = od
                transcld = expf(F(-od))
                abscld[lay, ig] = F(F("1.0") - transcld)
                efclfrac[lay, ig] = F(abscld[lay, ig] * cldfmc[ig, lay])
                icldlyr[lay] = 1

    totuflux = np.zeros(nl + 1, dtype=f32)
    totdflux = np.zeros(nl + 1, dtype=f32)
    totuclfl = np.zeros(nl + 1, dtype=f32)
    totdclfl = np.zeros(nl + 1, dtype=f32)
    urad = np.zeros(nl + 1, dtype=f32)
    drad = np.zeros(nl + 1, dtype=f32)
    clrurad = np.zeros(nl + 1, dtype=f32)
    clrdrad = np.zeros(nl + 1, dtype=f32)

    c0_06 = F("0.06")
    half = F("0.5")
    one = F("1.0")

    for iband in range(int(istart), int(iend) + 1):
        g_lo = 0 if iband == 1 else int(ngs[iband - 2])
        g_hi = int(ngs[iband - 1])
        lanes = np.arange(g_lo, g_hi)
        nlane = len(lanes)
        sd = secdiff[iband - 1]

        # Per-lane marching state and stored profiles.
        radld = np.zeros(nlane, dtype=f32)
        radclrd = np.zeros(nlane, dtype=f32)
        iclddn = np.zeros(nlane, dtype=bool)
        atrans_p = np.zeros((nl, nlane), dtype=f32)
        atot_p = np.zeros((nl, nlane), dtype=f32)
        bbugas_p = np.zeros((nl, nlane), dtype=f32)
        bbutot_p = np.zeros((nl, nlane), dtype=f32)
        radld_p = np.zeros((nl, nlane), dtype=f32)
        radclrd_p = np.zeros((nl, nlane), dtype=f32)
        iclddn_p = np.zeros((nl, nlane), dtype=bool)

        with np.errstate(all="ignore"):
            for lev in range(nl, 0, -1):
                li = lev - 1
                plfrac = fracs[li, lanes]
                blay = planklay[li, iband - 1]
                dplankup = f32(planklev[lev, iband - 1] - blay)
                dplankdn = f32(planklev[li, iband - 1] - blay)
                odepth = (sd * taut[li, lanes]).astype(f32)
                odepth = np.where(odepth < F(0.0), F(0.0), odepth)

                if icldlyr[li] == 1:
                    iclddn[:] = True
                    oc = odcld[li, lanes]
                    odtot = (odepth + oc).astype(f32)

                    # Branch A: odtot < 0.06
                    atransA = (odepth - (half * (odepth * odepth
                               ).astype(f32)).astype(f32)).astype(f32)
                    od_recA = (REC_6 * odepth).astype(f32)
                    gassrcA = ((plfrac * (blay + (dplankdn * od_recA
                               ).astype(f32)).astype(f32)).astype(f32)
                               * atransA).astype(f32)
                    atotA = (odtot - (half * (odtot * odtot).astype(f32)
                             ).astype(f32)).astype(f32)
                    odtot_recA = (REC_6 * odtot).astype(f32)
                    bbdtotA = (plfrac * (blay + (dplankdn * odtot_recA
                               ).astype(f32)).astype(f32)).astype(f32)
                    bbdA = (plfrac * (blay + (dplankdn * od_recA
                            ).astype(f32)).astype(f32)).astype(f32)
                    bbugasA = (plfrac * (blay + (dplankup * od_recA
                               ).astype(f32)).astype(f32)).astype(f32)
                    bbutotA = (plfrac * (blay + (dplankup * odtot_recA
                               ).astype(f32)).astype(f32)).astype(f32)

                    # Branch B: odtot >= 0.06 and odepth <= 0.06
                    tblindB = (odtot / (bpade + odtot).astype(f32)
                               ).astype(f32)
                    ittotB = trunc_int((TBLINT * tblindB).astype(f32)
                                       + half)
                    tfactotB = tfn_tbl[ittotB]
                    bbdtotB = (plfrac * (blay + (tfactotB * dplankdn
                               ).astype(f32)).astype(f32)).astype(f32)
                    atotB = (one - exp_tbl[ittotB]).astype(f32)
                    bbutotB = (plfrac * (blay + (tfactotB * dplankup
                               ).astype(f32)).astype(f32)).astype(f32)
                    # atrans/gassrc/bbd/bbugas shared with branch A forms

                    # Branch C: odepth > 0.06
                    tblindC = (odepth / (bpade + odepth).astype(f32)
                               ).astype(f32)
                    itgasC = trunc_int((TBLINT * tblindC).astype(f32)
                                       + half)
                    odepthC = tau_tbl[itgasC]
                    atransC = (one - exp_tbl[itgasC]).astype(f32)
                    tfacgasC = tfn_tbl[itgasC]
                    gassrcC = ((atransC * plfrac).astype(f32)
                               * (blay + (tfacgasC * dplankdn).astype(f32)
                                  ).astype(f32)).astype(f32)
                    odtotC = (odepthC + oc).astype(f32)
                    tblindC2 = (odtotC / (bpade + odtotC).astype(f32)
                                ).astype(f32)
                    ittotC = trunc_int((TBLINT * tblindC2).astype(f32)
                                       + half)
                    tfactotC = tfn_tbl[ittotC]
                    bbdtotC = (plfrac * (blay + (tfactotC * dplankdn
                               ).astype(f32)).astype(f32)).astype(f32)
                    bbdC = (plfrac * (blay + (tfacgasC * dplankdn
                            ).astype(f32)).astype(f32)).astype(f32)
                    atotC = (one - exp_tbl[ittotC]).astype(f32)
                    bbugasC = (plfrac * (blay + (tfacgasC * dplankup
                               ).astype(f32)).astype(f32)).astype(f32)
                    bbutotC = (plfrac * (blay + (tfactotC * dplankup
                               ).astype(f32)).astype(f32)).astype(f32)

                    is_a = odtot < c0_06
                    is_b = ~is_a & (odepth <= c0_06)
                    atrans = np.where(is_a | is_b, atransA, atransC)
                    gassrc = np.where(is_a | is_b, gassrcA, gassrcC)
                    atot = np.where(is_a, atotA,
                                    np.where(is_b, atotB, atotC))
                    bbdtot = np.where(is_a, bbdtotA,
                                      np.where(is_b, bbdtotB, bbdtotC))
                    bbd = np.where(is_a | is_b, bbdA, bbdC)
                    bbugas = np.where(is_a | is_b, bbugasA, bbugasC)
                    bbutot = np.where(is_a, bbutotA,
                                      np.where(is_b, bbutotB, bbutotC))

                    cf = cldfmc[lanes, li]
                    ef = efclfrac[li, lanes]
                    radld = (radld
                             - (radld * (atrans + (ef * (one - atrans
                                ).astype(f32)).astype(f32)).astype(f32)
                                ).astype(f32)).astype(f32)
                    radld = (radld + gassrc).astype(f32)
                    radld = (radld + (cf * ((bbdtot * atot).astype(f32)
                             - gassrc).astype(f32)).astype(f32)
                             ).astype(f32)
                    atrans_p[li] = atrans
                    atot_p[li] = atot
                    bbugas_p[li] = bbugas
                    bbutot_p[li] = bbutot
                else:
                    # Clear layer
                    atransD = (odepth - (half * (odepth * odepth
                               ).astype(f32)).astype(f32)).astype(f32)
                    od_recD = (REC_6 * odepth).astype(f32)
                    bbdD = (plfrac * (blay + (dplankdn * od_recD
                            ).astype(f32)).astype(f32)).astype(f32)
                    bbugasD = (plfrac * (blay + (dplankup * od_recD
                               ).astype(f32)).astype(f32)).astype(f32)

                    tblindE = (odepth / (bpade + odepth).astype(f32)
                               ).astype(f32)
                    itrE = trunc_int((TBLINT * tblindE).astype(f32) + half)
                    transcE = exp_tbl[itrE]
                    atransE = (one - transcE).astype(f32)
                    tausfacE = tfn_tbl[itrE]
                    bbdE = (plfrac * (blay + (tausfacE * dplankdn
                            ).astype(f32)).astype(f32)).astype(f32)
                    bbugasE = (plfrac * (blay + (tausfacE * dplankup
                               ).astype(f32)).astype(f32)).astype(f32)

                    is_d = odepth <= c0_06
                    atrans = np.where(is_d, atransD, atransE)
                    bbd = np.where(is_d, bbdD, bbdE)
                    bbugas = np.where(is_d, bbugasD, bbugasE)
                    radld = (radld + ((bbd - radld).astype(f32) * atrans
                             ).astype(f32)).astype(f32)
                    atrans_p[li] = atrans
                    bbugas_p[li] = bbugas

                radld_p[li] = radld
                radclrd = np.where(
                    iclddn,
                    (radclrd + ((bbd - radclrd).astype(f32) * atrans
                     ).astype(f32)).astype(f32),
                    radld)
                radclrd_p[li] = radclrd
                iclddn_p[li] = iclddn

        # Ordered accumulation into drad/clrdrad (lane order = igc order).
        for lane in range(nlane):
            drad[:nl] = (drad[:nl] + radld_p[:, lane]).astype(f32)
            clrdrad[:nl] = np.where(
                iclddn_p[:, lane],
                (clrdrad[:nl] + radclrd_p[:, lane]).astype(f32),
                drad[:nl])

        # Surface reflection, upward march.
        rad0 = (fracs[0, lanes] * plankbnd[iband - 1]).astype(f32)
        reflect = F(one - semiss[iband - 1])
        radlu = (rad0 + (reflect * radld).astype(f32)).astype(f32)
        radclru = (rad0 + (reflect * radclrd).astype(f32)).astype(f32)

        radlu_p = np.zeros((nl, nlane), dtype=f32)
        radclru_p = np.zeros((nl, nlane), dtype=f32)
        with np.errstate(all="ignore"):
            for lev in range(1, nl + 1):
                li = lev - 1
                if icldlyr[li] == 1:
                    gassrc = (bbugas_p[li] * atrans_p[li]).astype(f32)
                    ef = efclfrac[li, lanes]
                    cf = cldfmc[lanes, li]
                    radlu = (radlu
                             - (radlu * (atrans_p[li]
                                + (ef * (one - atrans_p[li]).astype(f32)
                                   ).astype(f32)).astype(f32)
                                ).astype(f32)).astype(f32)
                    radlu = (radlu + gassrc).astype(f32)
                    radlu = (radlu + (cf * ((bbutot_p[li] * atot_p[li]
                             ).astype(f32) - gassrc).astype(f32)
                             ).astype(f32)).astype(f32)
                else:
                    radlu = (radlu + ((bbugas_p[li] - radlu).astype(f32)
                             * atrans_p[li]).astype(f32)).astype(f32)
                radlu_p[li] = radlu
                radclru = np.where(
                    iclddn,
                    (radclru + ((bbugas_p[li] - radclru).astype(f32)
                     * atrans_p[li]).astype(f32)).astype(f32),
                    radlu)
                radclru_p[li] = radclru

        # urad(0)/clrurad(0): Fortran adds the initial (surface) radlu /
        # radclru once per g-point before that g-point's upward march.
        radlu_sfc = (rad0 + (reflect * radld).astype(f32)).astype(f32)
        radclru_sfc = (rad0 + (reflect * radclrd).astype(f32)).astype(f32)
        for lane in range(nlane):
            urad[0] = F(urad[0] + radlu_sfc[lane])
            clrurad[0] = F(clrurad[0] + radclru_sfc[lane])
        for lane in range(nlane):
            urad[1:nl + 1] = (urad[1:nl + 1] + radlu_p[:, lane]
                              ).astype(f32)
            if bool(iclddn[lane]):
                clrurad[1:nl + 1] = (clrurad[1:nl + 1]
                                     + radclru_p[:, lane]).astype(f32)
            else:
                clrurad[1:nl + 1] = urad[1:nl + 1]

        # Band flux processing.
        uflux = (urad * WTDIFF).astype(f32)
        dflux = (drad * WTDIFF).astype(f32)
        uclfl = (clrurad * WTDIFF).astype(f32)
        dclfl = (clrdrad * WTDIFF).astype(f32)
        urad[:] = F(0.0)
        drad[:] = F(0.0)
        clrurad[:] = F(0.0)
        clrdrad[:] = F(0.0)
        dw = delwave[iband - 1]
        totuflux = (totuflux + (uflux * dw).astype(f32)).astype(f32)
        totdflux = (totdflux + (dflux * dw).astype(f32)).astype(f32)
        totuclfl = (totuclfl + (uclfl * dw).astype(f32)).astype(f32)
        totdclfl = (totdclfl + (dclfl * dw).astype(f32)).astype(f32)

    fnet = np.zeros(nl + 1, dtype=f32)
    fnetc = np.zeros(nl + 1, dtype=f32)
    htr = np.zeros(nl + 1, dtype=f32)
    htrc = np.zeros(nl + 1, dtype=f32)
    totuflux = (totuflux * fluxfac).astype(f32)
    totdflux = (totdflux * fluxfac).astype(f32)
    totuclfl = (totuclfl * fluxfac).astype(f32)
    totdclfl = (totdclfl * fluxfac).astype(f32)
    fnet = (totuflux - totdflux).astype(f32)
    fnetc = (totuclfl - totdclfl).astype(f32)
    for lev in range(1, nl + 1):
        l0 = lev - 1
        htr[l0] = F(F(heatfac * F(fnet[l0] - fnet[lev]))
                    / F(pz[l0] - pz[lev]))
        htrc[l0] = F(F(heatfac * F(fnetc[l0] - fnetc[lev]))
                     / F(pz[l0] - pz[lev]))
    htr[nl] = F(0.0)
    htrc[nl] = F(0.0)

    return {
        "totuflux": totuflux, "totdflux": totdflux, "fnet": fnet,
        "htr": htr, "totuclfl": totuclfl, "totdclfl": totdclfl,
        "fnetc": fnetc, "htrc": htrc,
    }


def taumol(nlayers, st, C):
    """Port of rrtmg_lw_taumol::taumol -- dispatches all 16 band routines.

    Raises NotImplementedError if any band is missing so an incomplete
    port can never silently produce wrong optical depths.
    """
    nl = int(nlayers)
    taug = np.zeros((nl, NGPTLW), dtype=np.float32)
    fracs = np.zeros((nl, NGPTLW), dtype=np.float32)
    missing = [b for b in range(1, 17) if b not in TAUGB_IMPLS]
    if missing:
        raise NotImplementedError(f"taugb not ported for bands {missing}")
    for band in range(1, 17):
        TAUGB_IMPLS[band](st, C, taug, fracs)
    return fracs, taug



# ---------------------------------------------------------------------------
# Section 6b -- taugb/cmbgb implementations merged from the gated band
# fragments (each held at max_ulp 0 / bitwise before merging; the module-
# level gates re-verify after the merge).
# ---------------------------------------------------------------------------

# ---- merged fragment: bands 2/10/11/12 ----

def _f32(x):
    return x.astype(np.float32)


def _cont(fac, frac, ref, ind):
    """fac(lay) * (ref(ind,ig) + frac(lay) * (ref(ind+1,ig) - ref(ind,ig))).

    fac/frac are (nlay,1) float32 columns; ind is 1-based (nlay,) int64.
    """
    lo_ = ref[ind - 1, :]
    hi_ = ref[ind, :]
    return _f32(fac * _f32(lo_ + _f32(frac * _f32(hi_ - lo_))))


def _major4(fac00, fac10, fac01, fac11, tab, ind0, ind1):
    """fac00*tab(ind0) + fac10*tab(ind0+1) + fac01*tab(ind1) + fac11*tab(ind1+1),
    accumulated left to right (the 1-key-species major term)."""
    t = _f32(fac00 * tab[ind0 - 1, :])
    t = _f32(t + _f32(fac10 * tab[ind0, :]))
    t = _f32(t + _f32(fac01 * tab[ind1 - 1, :]))
    t = _f32(t + _f32(fac11 * tab[ind1, :]))
    return t


def _taugb2(st, C, taug, fracs):
    """Band 2: 350-500 cm-1 (low key - h2o; high key - h2o).

    Fortran: module lines 5169-5238.  Lower region has
    corradj = 1 - .05*(pp - 100)/900; upper region has no corradj.
    """
    absa = C["kg02/absa"]
    absb = C["kg02/absb"]
    selfref = C["kg02/selfref"]
    forref = C["kg02/forref"]
    fracrefa = C["kg02/fracrefa"]
    fracrefb = C["kg02/fracrefb"]

    nl = len(st["pavel"])
    laytrop = st["laytrop"]
    ng2 = int(NGC[1])
    gs = int(NGS[0])

    lo = slice(0, laytrop)
    jp = st["jp"][lo].astype(np.int64)
    jt = st["jt"][lo].astype(np.int64)
    jt1 = st["jt1"][lo].astype(np.int64)
    ind0 = ((jp - 1) * 5 + (jt - 1)) * int(NSPA[1]) + 1
    ind1 = (jp * 5 + (jt1 - 1)) * int(NSPA[1]) + 1
    inds = st["indself"][lo].astype(np.int64)
    indf = st["indfor"][lo].astype(np.int64)
    pp = st["pavel"][lo]
    # corradj = 1._rb - .05_rb * (pp - 100._rb) / 900._rb
    corradj = _f32(F("1.0")
                   - _f32(_f32(F("0.05") * _f32(pp - F("100.0")))
                          / F("900.0")))

    selffac = st["selffac"][lo][:, None]
    selffrac = st["selffrac"][lo][:, None]
    forfac = st["forfac"][lo][:, None]
    forfrac = st["forfrac"][lo][:, None]
    colh2o = st["colh2o"][lo][:, None]

    tauself = _cont(selffac, selffrac, selfref, inds)
    taufor = _cont(forfac, forfrac, forref, indf)
    tmaj = _major4(st["fac00"][lo][:, None], st["fac10"][lo][:, None],
                   st["fac01"][lo][:, None], st["fac11"][lo][:, None],
                   absa, ind0, ind1)
    t = _f32(colh2o * tmaj)
    t = _f32(t + tauself)
    t = _f32(t + taufor)
    taug[lo, gs:gs + ng2] = _f32(corradj[:, None] * t)
    fracs[lo, gs:gs + ng2] = fracrefa[None, :]

    hi = slice(laytrop, nl)
    jp = st["jp"][hi].astype(np.int64)
    jt = st["jt"][hi].astype(np.int64)
    jt1 = st["jt1"][hi].astype(np.int64)
    ind0 = ((jp - 13) * 5 + (jt - 1)) * int(NSPB[1]) + 1
    ind1 = ((jp - 12) * 5 + (jt1 - 1)) * int(NSPB[1]) + 1
    indf = st["indfor"][hi].astype(np.int64)

    forfac = st["forfac"][hi][:, None]
    forfrac = st["forfrac"][hi][:, None]
    colh2o = st["colh2o"][hi][:, None]

    taufor = _cont(forfac, forfrac, forref, indf)
    tmaj = _major4(st["fac00"][hi][:, None], st["fac10"][hi][:, None],
                   st["fac01"][hi][:, None], st["fac11"][hi][:, None],
                   absb, ind0, ind1)
    t = _f32(colh2o * tmaj)
    t = _f32(t + taufor)
    taug[hi, gs:gs + ng2] = t
    fracs[hi, gs:gs + ng2] = fracrefb[None, :]


def _taugb10(st, C, taug, fracs):
    """Band 10: 1390-1480 cm-1 (low key - h2o; high key - h2o).

    Fortran: module lines 6838-6902.  No corradj, no minor species.
    """
    absa = C["kg10/absa"]
    absb = C["kg10/absb"]
    selfref = C["kg10/selfref"]
    forref = C["kg10/forref"]
    fracrefa = C["kg10/fracrefa"]
    fracrefb = C["kg10/fracrefb"]

    nl = len(st["pavel"])
    laytrop = st["laytrop"]
    ng10 = int(NGC[9])
    gs = int(NGS[8])

    lo = slice(0, laytrop)
    jp = st["jp"][lo].astype(np.int64)
    jt = st["jt"][lo].astype(np.int64)
    jt1 = st["jt1"][lo].astype(np.int64)
    ind0 = ((jp - 1) * 5 + (jt - 1)) * int(NSPA[9]) + 1
    ind1 = (jp * 5 + (jt1 - 1)) * int(NSPA[9]) + 1
    inds = st["indself"][lo].astype(np.int64)
    indf = st["indfor"][lo].astype(np.int64)

    selffac = st["selffac"][lo][:, None]
    selffrac = st["selffrac"][lo][:, None]
    forfac = st["forfac"][lo][:, None]
    forfrac = st["forfrac"][lo][:, None]
    colh2o = st["colh2o"][lo][:, None]

    tauself = _cont(selffac, selffrac, selfref, inds)
    taufor = _cont(forfac, forfrac, forref, indf)
    tmaj = _major4(st["fac00"][lo][:, None], st["fac10"][lo][:, None],
                   st["fac01"][lo][:, None], st["fac11"][lo][:, None],
                   absa, ind0, ind1)
    t = _f32(colh2o * tmaj)
    t = _f32(t + tauself)
    t = _f32(t + taufor)
    taug[lo, gs:gs + ng10] = t
    fracs[lo, gs:gs + ng10] = fracrefa[None, :]

    hi = slice(laytrop, nl)
    jp = st["jp"][hi].astype(np.int64)
    jt = st["jt"][hi].astype(np.int64)
    jt1 = st["jt1"][hi].astype(np.int64)
    ind0 = ((jp - 13) * 5 + (jt - 1)) * int(NSPB[9]) + 1
    ind1 = ((jp - 12) * 5 + (jt1 - 1)) * int(NSPB[9]) + 1
    indf = st["indfor"][hi].astype(np.int64)

    forfac = st["forfac"][hi][:, None]
    forfrac = st["forfrac"][hi][:, None]
    colh2o = st["colh2o"][hi][:, None]

    taufor = _cont(forfac, forfrac, forref, indf)
    tmaj = _major4(st["fac00"][hi][:, None], st["fac10"][hi][:, None],
                   st["fac01"][hi][:, None], st["fac11"][hi][:, None],
                   absb, ind0, ind1)
    t = _f32(colh2o * tmaj)
    t = _f32(t + taufor)
    taug[hi, gs:gs + ng10] = t
    fracs[hi, gs:gs + ng10] = fracrefb[None, :]


def _taugb11(st, C, taug, fracs):
    """Band 11: 1480-1800 cm-1 (low - h2o, minor o2; high - h2o, minor o2).

    Fortran: module lines 6905-6982.  tauo2 = colo2*scaleminor scaled
    interpolation of ka_mo2/kb_mo2 in BOTH regions.
    """
    absa = C["kg11/absa"]
    absb = C["kg11/absb"]
    ka_mo2 = C["kg11/ka_mo2"]
    kb_mo2 = C["kg11/kb_mo2"]
    selfref = C["kg11/selfref"]
    forref = C["kg11/forref"]
    fracrefa = C["kg11/fracrefa"]
    fracrefb = C["kg11/fracrefb"]

    nl = len(st["pavel"])
    laytrop = st["laytrop"]
    ng11 = int(NGC[10])
    gs = int(NGS[9])

    lo = slice(0, laytrop)
    jp = st["jp"][lo].astype(np.int64)
    jt = st["jt"][lo].astype(np.int64)
    jt1 = st["jt1"][lo].astype(np.int64)
    ind0 = ((jp - 1) * 5 + (jt - 1)) * int(NSPA[10]) + 1
    ind1 = (jp * 5 + (jt1 - 1)) * int(NSPA[10]) + 1
    inds = st["indself"][lo].astype(np.int64)
    indf = st["indfor"][lo].astype(np.int64)
    indm = st["indminor"][lo].astype(np.int64)
    scaleo2 = _f32(st["colo2"][lo] * st["scaleminor"][lo])

    selffac = st["selffac"][lo][:, None]
    selffrac = st["selffrac"][lo][:, None]
    forfac = st["forfac"][lo][:, None]
    forfrac = st["forfrac"][lo][:, None]
    minorfrac = st["minorfrac"][lo][:, None]
    colh2o = st["colh2o"][lo][:, None]

    tauself = _cont(selffac, selffrac, selfref, inds)
    taufor = _cont(forfac, forfrac, forref, indf)
    tauo2 = _cont(scaleo2[:, None], minorfrac, ka_mo2, indm)
    tmaj = _major4(st["fac00"][lo][:, None], st["fac10"][lo][:, None],
                   st["fac01"][lo][:, None], st["fac11"][lo][:, None],
                   absa, ind0, ind1)
    t = _f32(colh2o * tmaj)
    t = _f32(t + tauself)
    t = _f32(t + taufor)
    t = _f32(t + tauo2)
    taug[lo, gs:gs + ng11] = t
    fracs[lo, gs:gs + ng11] = fracrefa[None, :]

    hi = slice(laytrop, nl)
    jp = st["jp"][hi].astype(np.int64)
    jt = st["jt"][hi].astype(np.int64)
    jt1 = st["jt1"][hi].astype(np.int64)
    ind0 = ((jp - 13) * 5 + (jt - 1)) * int(NSPB[10]) + 1
    ind1 = ((jp - 12) * 5 + (jt1 - 1)) * int(NSPB[10]) + 1
    indf = st["indfor"][hi].astype(np.int64)
    indm = st["indminor"][hi].astype(np.int64)
    scaleo2 = _f32(st["colo2"][hi] * st["scaleminor"][hi])

    forfac = st["forfac"][hi][:, None]
    forfrac = st["forfrac"][hi][:, None]
    minorfrac = st["minorfrac"][hi][:, None]
    colh2o = st["colh2o"][hi][:, None]

    taufor = _cont(forfac, forfrac, forref, indf)
    tauo2 = _cont(scaleo2[:, None], minorfrac, kb_mo2, indm)
    tmaj = _major4(st["fac00"][hi][:, None], st["fac10"][hi][:, None],
                   st["fac01"][hi][:, None], st["fac11"][hi][:, None],
                   absb, ind0, ind1)
    t = _f32(colh2o * tmaj)
    t = _f32(t + taufor)
    t = _f32(t + tauo2)
    taug[hi, gs:gs + ng11] = t
    fracs[hi, gs:gs + ng11] = fracrefb[None, :]


def _taugb12(st, C, taug, fracs):
    """Band 12: 1800-2080 cm-1 (low - h2o,co2; high - NOTHING).

    Fortran: module lines 6985-7186.  Lower region: 9-species speccomb
    machinery with the specparm branch triplet (.lt. 0.125 / .gt. 0.875 /
    else).  Upper region: taug(lay,ig) = 0.0 and fracs(lay,ig) = 0.0,
    transcribed literally.

    Vectorized branch selection: all three branch values are computed on
    all lanes and chosen with np.where; discarded-lane row indices are
    clipped to the table (garbage values on discarded lanes are irrelevant,
    but numpy must never index past the end).
    """
    absa = C["kg12/absa"]          # (585, ng12), Fortran equivalence of ka
    selfref = C["kg12/selfref"]
    forref = C["kg12/forref"]
    fracrefa = C["kg12/fracrefa"]  # (ng12, 9): fracrefa(ig, jpl)
    chi = C["ref/chi_mls"]
    oneminus = F(C["con/oneminus"])

    # refrat_planck_a = chi_mls(1,10)/chi_mls(2,10)   (P = 174.164 mb)
    refrat_planck_a = F(chi[0, 9] / chi[1, 9])

    nl = len(st["pavel"])
    laytrop = st["laytrop"]
    ng12 = int(NGC[11])
    gs = int(NGS[10])
    nrows = absa.shape[0]

    lo = slice(0, laytrop)
    colh2o = st["colh2o"][lo]
    colco2 = st["colco2"][lo]

    def _spec(rat):
        # speccomb = colh2o + rat*colco2 ; specparm = colh2o/speccomb
        # if (specparm >= oneminus) specparm = oneminus
        # specmult = 8*specparm ; j = 1 + int(specmult) ; f = mod(specmult,1)
        sc = _f32(colh2o + _f32(rat * colco2))
        sp = _f32(colh2o / sc)
        sp = np.where(sp >= oneminus, oneminus, sp).astype(np.float32)
        sm = _f32(F("8.0") * sp)
        j = 1 + trunc_int(sm).astype(np.int64)
        f = np.fmod(sm, F("1.0")).astype(np.float32)
        return sc, sp, j, f

    speccomb, specparm, js, fs = _spec(st["rat_h2oco2"][lo])
    speccomb1, specparm1, js1, fs1 = _spec(st["rat_h2oco2_1"][lo])
    _, _, jpl, fpl = _spec(refrat_planck_a)

    jp = st["jp"][lo].astype(np.int64)
    jt = st["jt"][lo].astype(np.int64)
    jt1 = st["jt1"][lo].astype(np.int64)
    ind0 = ((jp - 1) * 5 + (jt - 1)) * int(NSPA[11]) + js
    ind1 = (jp * 5 + (jt1 - 1)) * int(NSPA[11]) + js1
    inds = st["indself"][lo].astype(np.int64)
    indf = st["indfor"][lo].astype(np.int64)

    tauself = _cont(st["selffac"][lo][:, None], st["selffrac"][lo][:, None],
                    selfref, inds)
    taufor = _cont(st["forfac"][lo][:, None], st["forfrac"][lo][:, None],
                   forref, indf)

    fac00 = st["fac00"][lo]
    fac10 = st["fac10"][lo]
    fac01 = st["fac01"][lo]
    fac11 = st["fac11"][lo]

    def _edge_facs(p, facx, facy):
        # p4 = p**4 (integer exponent -> square-and-multiply)
        # fk0 = p4 ; fk1 = 1 - p - 2.0*p4 ; fk2 = p + p4
        p4 = pow4(p)
        fk0 = p4
        fk1 = _f32(_f32(F("1.0") - p) - _f32(F("2.0") * p4))
        fk2 = _f32(p + p4)
        return (_f32(fk0 * facx), _f32(fk1 * facx), _f32(fk2 * facx),
                _f32(fk0 * facy), _f32(fk1 * facy), _f32(fk2 * facy))

    def _rows(r):
        # 0-based rows, clipped so discarded lanes cannot index past the
        # table end (negative wrap on discarded lanes is harmless).
        return np.minimum(r, nrows - 1)

    def _tmaj_lt(sc, ind, f000, f100, f200, f010, f110, f210):
        r = ind - 1
        t = _f32(f000[:, None] * absa[_rows(r), :])
        t = _f32(t + _f32(f100[:, None] * absa[_rows(r + 1), :]))
        t = _f32(t + _f32(f200[:, None] * absa[_rows(r + 2), :]))
        t = _f32(t + _f32(f010[:, None] * absa[_rows(r + 9), :]))
        t = _f32(t + _f32(f110[:, None] * absa[_rows(r + 10), :]))
        t = _f32(t + _f32(f210[:, None] * absa[_rows(r + 11), :]))
        return _f32(sc[:, None] * t)

    def _tmaj_gt(sc, ind, f000, f100, f200, f010, f110, f210):
        r = ind - 1
        t = _f32(f200[:, None] * absa[_rows(r - 1), :])
        t = _f32(t + _f32(f100[:, None] * absa[_rows(r), :]))
        t = _f32(t + _f32(f000[:, None] * absa[_rows(r + 1), :]))
        t = _f32(t + _f32(f210[:, None] * absa[_rows(r + 8), :]))
        t = _f32(t + _f32(f110[:, None] * absa[_rows(r + 9), :]))
        t = _f32(t + _f32(f010[:, None] * absa[_rows(r + 10), :]))
        return _f32(sc[:, None] * t)

    def _tmaj_mid(sc, ind, f000, f100, f010, f110):
        r = ind - 1
        t = _f32(f000[:, None] * absa[_rows(r), :])
        t = _f32(t + _f32(f100[:, None] * absa[_rows(r + 1), :]))
        t = _f32(t + _f32(f010[:, None] * absa[_rows(r + 9), :]))
        t = _f32(t + _f32(f110[:, None] * absa[_rows(r + 10), :]))
        return _f32(sc[:, None] * t)

    def _tau_major(sc, sp, ind, f, facx, facy):
        lt = sp < F("0.125")
        gt = sp > F("0.875")
        p_lt = _f32(f - F("1.0"))
        e_lt = _edge_facs(p_lt, facx, facy)
        p_gt = -f
        e_gt = _edge_facs(p_gt, facx, facy)
        one_m_f = _f32(F("1.0") - f)
        m000 = _f32(one_m_f * facx)
        m010 = _f32(one_m_f * facy)
        m100 = _f32(f * facx)
        m110 = _f32(f * facy)
        tm_lt = _tmaj_lt(sc, ind, *e_lt)
        tm_gt = _tmaj_gt(sc, ind, *e_gt)
        tm_mid = _tmaj_mid(sc, ind, m000, m100, m010, m110)
        return np.where(lt[:, None], tm_lt,
                        np.where(gt[:, None], tm_gt, tm_mid)
                        ).astype(np.float32)

    tau_major = _tau_major(speccomb, specparm, ind0, fs, fac00, fac10)
    tau_major1 = _tau_major(speccomb1, specparm1, ind1, fs1, fac01, fac11)

    # taug = tau_major + tau_major1 + tauself + taufor
    t = _f32(tau_major + tau_major1)
    t = _f32(t + tauself)
    t = _f32(t + taufor)
    taug[lo, gs:gs + ng12] = t

    # fracs = fracrefa(ig,jpl) + fpl*(fracrefa(ig,jpl+1) - fracrefa(ig,jpl))
    fr0 = fracrefa[:, jpl - 1].T
    fr1 = fracrefa[:, jpl].T
    fracs[lo, gs:gs + ng12] = _f32(
        fr0 + _f32(fpl[:, None] * _f32(fr1 - fr0)))

    # Upper atmosphere loop: taug(lay,ngs11+ig) = 0.0 ; fracs = 0.0
    hi = slice(laytrop, nl)
    taug[hi, gs:gs + ng12] = F("0.0")
    fracs[hi, gs:gs + ng12] = F("0.0")

# ---- merged fragment: bands 6-9 ----

def _nine_major(specparm, fs, facp, fact, absa, ind, speccomb):
    """The specparm branch triplet + tau_major sum shared by the 9-species
    bands (7 and 9).  facp/fact are fac00/fac10 (ind0 half) or fac01/fac11
    (ind1 half).  Returns the tau_major (length-ng float32 vector).

    Fortran: e.g. lines 6278-6307 (facs) and 6350-6372 (tau_major).
    """
    if specparm < F("0.125"):
        p = fs - F("1.0")
        p2 = p * p
        p4 = p2 * p2          # p**4, gfortran square-and-multiply
        fk0 = p4
        fk1 = F("1.0") - p - F("2.0") * p4
        fk2 = p + p4
        fac0p = fk0 * facp
        fac1p = fk1 * facp
        fac2p = fk2 * facp
        fac0t = fk0 * fact
        fac1t = fk1 * fact
        fac2t = fk2 * fact
        return speccomb * (fac0p * absa[ind - 1, :]
                           + fac1p * absa[ind, :]
                           + fac2p * absa[ind + 1, :]
                           + fac0t * absa[ind + 8, :]
                           + fac1t * absa[ind + 9, :]
                           + fac2t * absa[ind + 10, :])
    elif specparm > F("0.875"):
        p = -fs
        p2 = p * p
        p4 = p2 * p2
        fk0 = p4
        fk1 = F("1.0") - p - F("2.0") * p4
        fk2 = p + p4
        fac0p = fk0 * facp
        fac1p = fk1 * facp
        fac2p = fk2 * facp
        fac0t = fk0 * fact
        fac1t = fk1 * fact
        fac2t = fk2 * fact
        return speccomb * (fac2p * absa[ind - 2, :]
                           + fac1p * absa[ind - 1, :]
                           + fac0p * absa[ind, :]
                           + fac2t * absa[ind + 7, :]
                           + fac1t * absa[ind + 8, :]
                           + fac0t * absa[ind + 9, :])
    else:
        fac0p = (F("1.0") - fs) * facp
        fac0t = (F("1.0") - fs) * fact
        fac1p = fs * facp
        fac1t = fs * fact
        return speccomb * (fac0p * absa[ind - 1, :]
                           + fac1p * absa[ind, :]
                           + fac0t * absa[ind + 8, :]
                           + fac1t * absa[ind + 9, :])


def _taugb6(st, C, taug, fracs):
    """Band 6: 820-980 cm-1 (low key h2o; low minor co2; upper NOTHING but
    the cfc11adj/cfc12 cross-section terms).  Fortran lines 6090-6175."""
    chi_mls = C["ref/chi_mls"]
    absa = C["kg06/absa"]
    selfref = C["kg06/selfref"]
    forref = C["kg06/forref"]
    ka_mco2 = C["kg06/ka_mco2"]
    fracrefa = C["kg06/fracrefa"]
    cfc11adj = C["kg06/cfc11adj"]
    cfc12 = C["kg06/cfc12"]

    nl = len(st["pavel"])
    laytrop = int(st["laytrop"])
    gs = int(NGS[4])
    ng6 = int(NGC[5])
    sl = slice(gs, gs + ng6)
    nspa6 = int(NSPA[5])
    wx = st["wx"]

    for lay in range(1, laytrop + 1):
        l = lay - 1
        jpv = int(st["jp"][l])
        chi_co2 = st["colco2"][l] / st["coldry"][l]
        ratco2 = F("1.e20") * chi_co2 / chi_mls[1, jpv]
        if ratco2 > F("3.0"):
            adjfac = F("2.0") + powf(ratco2 - F("2.0"), F("0.77"))
            adjcolco2 = adjfac * chi_mls[1, jpv] * st["coldry"][l] \
                * F("1.e-20")
        else:
            adjcolco2 = st["colco2"][l]

        ind0 = ((jpv - 1) * 5 + (int(st["jt"][l]) - 1)) * nspa6 + 1
        ind1 = (jpv * 5 + (int(st["jt1"][l]) - 1)) * nspa6 + 1
        inds = int(st["indself"][l])
        indf = int(st["indfor"][l])
        indm = int(st["indminor"][l])

        tauself = st["selffac"][l] * (
            selfref[inds - 1, :] + st["selffrac"][l]
            * (selfref[inds, :] - selfref[inds - 1, :]))
        taufor = st["forfac"][l] * (
            forref[indf - 1, :] + st["forfrac"][l]
            * (forref[indf, :] - forref[indf - 1, :]))
        absco2 = (ka_mco2[indm - 1, :] + st["minorfrac"][l]
                  * (ka_mco2[indm, :] - ka_mco2[indm - 1, :]))
        taug[l, sl] = (st["colh2o"][l]
                       * (st["fac00"][l] * absa[ind0 - 1, :]
                          + st["fac10"][l] * absa[ind0, :]
                          + st["fac01"][l] * absa[ind1 - 1, :]
                          + st["fac11"][l] * absa[ind1, :])
                       + tauself + taufor
                       + adjcolco2 * absco2
                       + wx[1, l] * cfc11adj
                       + wx[2, l] * cfc12)
        fracs[l, sl] = fracrefa

    # Upper atmosphere: taug gets ONLY the cfc cross-section terms.
    for lay in range(laytrop + 1, nl + 1):
        l = lay - 1
        taug[l, sl] = (F("0.0")
                       + wx[1, l] * cfc11adj
                       + wx[2, l] * cfc12)
        fracs[l, sl] = fracrefa


def _taugb7(st, C, taug, fracs):
    """Band 7: 980-1080 cm-1 (low key h2o,o3 + co2 minor; high key o3 +
    co2 minor + empirical g-point rescale).  Fortran lines 6178-6449."""
    chi_mls = C["ref/chi_mls"]
    oneminus = F(C["con/oneminus"])
    absa = C["kg07/absa"]
    absb = C["kg07/absb"]
    selfref = C["kg07/selfref"]
    forref = C["kg07/forref"]
    ka_mco2 = C["kg07/ka_mco2"]
    kb_mco2 = C["kg07/kb_mco2"]
    fracrefa = C["kg07/fracrefa"]
    fracrefb = C["kg07/fracrefb"]

    # P = 706.2620 mb / 706.2720 mb (identical expressions in this source)
    refrat_planck_a = chi_mls[0, 2] / chi_mls[2, 2]
    refrat_m_a = chi_mls[0, 2] / chi_mls[2, 2]

    nl = len(st["pavel"])
    laytrop = int(st["laytrop"])
    gs = int(NGS[5])
    ng7 = int(NGC[6])
    sl = slice(gs, gs + ng7)
    nspa7 = int(NSPA[6])
    nspb7 = int(NSPB[6])

    for lay in range(1, laytrop + 1):
        l = lay - 1
        colh2o = st["colh2o"][l]
        colo3 = st["colo3"][l]
        jpv = int(st["jp"][l])

        speccomb = colh2o + st["rat_h2oo3"][l] * colo3
        specparm = colh2o / speccomb
        if specparm >= oneminus:
            specparm = oneminus
        specmult = F("8.0") * specparm
        js = 1 + int(specmult)
        fs = np.fmod(specmult, F("1.0"))

        speccomb1 = colh2o + st["rat_h2oo3_1"][l] * colo3
        specparm1 = colh2o / speccomb1
        if specparm1 >= oneminus:
            specparm1 = oneminus
        specmult1 = F("8.0") * specparm1
        js1 = 1 + int(specmult1)
        fs1 = np.fmod(specmult1, F("1.0"))

        speccomb_mco2 = colh2o + refrat_m_a * colo3
        specparm_mco2 = colh2o / speccomb_mco2
        if specparm_mco2 >= oneminus:
            specparm_mco2 = oneminus
        specmult_mco2 = F("8.0") * specparm_mco2
        jmco2 = 1 + int(specmult_mco2)
        fmco2 = np.fmod(specmult_mco2, F("1.0"))

        chi_co2 = st["colco2"][l] / st["coldry"][l]
        ratco2 = F("1.e20") * chi_co2 / chi_mls[1, jpv]
        if ratco2 > F("3.0"):
            adjfac = F("3.0") + powf(ratco2 - F("3.0"), F("0.79"))
            adjcolco2 = adjfac * chi_mls[1, jpv] * st["coldry"][l] \
                * F("1.e-20")
        else:
            adjcolco2 = st["colco2"][l]

        speccomb_planck = colh2o + refrat_planck_a * colo3
        specparm_planck = colh2o / speccomb_planck
        if specparm_planck >= oneminus:
            specparm_planck = oneminus
        specmult_planck = F("8.0") * specparm_planck
        jpl = 1 + int(specmult_planck)
        fpl = np.fmod(specmult_planck, F("1.0"))

        ind0 = ((jpv - 1) * 5 + (int(st["jt"][l]) - 1)) * nspa7 + js
        ind1 = (jpv * 5 + (int(st["jt1"][l]) - 1)) * nspa7 + js1
        inds = int(st["indself"][l])
        indf = int(st["indfor"][l])
        indm = int(st["indminor"][l])

        tauself = st["selffac"][l] * (
            selfref[inds - 1, :] + st["selffrac"][l]
            * (selfref[inds, :] - selfref[inds - 1, :]))
        taufor = st["forfac"][l] * (
            forref[indf - 1, :] + st["forfrac"][l]
            * (forref[indf, :] - forref[indf - 1, :]))
        co2m1 = (ka_mco2[jmco2 - 1, indm - 1, :] + fmco2
                 * (ka_mco2[jmco2, indm - 1, :]
                    - ka_mco2[jmco2 - 1, indm - 1, :]))
        co2m2 = (ka_mco2[jmco2 - 1, indm, :] + fmco2
                 * (ka_mco2[jmco2, indm, :]
                    - ka_mco2[jmco2 - 1, indm, :]))
        absco2 = co2m1 + st["minorfrac"][l] * (co2m2 - co2m1)

        tau_major = _nine_major(specparm, fs, st["fac00"][l],
                                st["fac10"][l], absa, ind0, speccomb)
        tau_major1 = _nine_major(specparm1, fs1, st["fac01"][l],
                                 st["fac11"][l], absa, ind1, speccomb1)

        taug[l, sl] = (tau_major + tau_major1
                       + tauself + taufor
                       + adjcolco2 * absco2)
        fracs[l, sl] = (fracrefa[:, jpl - 1] + fpl
                        * (fracrefa[:, jpl] - fracrefa[:, jpl - 1]))

    for lay in range(laytrop + 1, nl + 1):
        l = lay - 1
        jpv = int(st["jp"][l])
        chi_co2 = st["colco2"][l] / st["coldry"][l]
        ratco2 = F("1.e20") * chi_co2 / chi_mls[1, jpv]
        if ratco2 > F("3.0"):
            adjfac = F("2.0") + powf(ratco2 - F("2.0"), F("0.79"))
            adjcolco2 = adjfac * chi_mls[1, jpv] * st["coldry"][l] \
                * F("1.e-20")
        else:
            adjcolco2 = st["colco2"][l]

        ind0 = ((jpv - 13) * 5 + (int(st["jt"][l]) - 1)) * nspb7 + 1
        ind1 = ((jpv - 12) * 5 + (int(st["jt1"][l]) - 1)) * nspb7 + 1
        indm = int(st["indminor"][l])

        absco2 = (kb_mco2[indm - 1, :] + st["minorfrac"][l]
                  * (kb_mco2[indm, :] - kb_mco2[indm - 1, :]))
        taug[l, sl] = (st["colo3"][l]
                       * (st["fac00"][l] * absb[ind0 - 1, :]
                          + st["fac10"][l] * absb[ind0, :]
                          + st["fac01"][l] * absb[ind1 - 1, :]
                          + st["fac11"][l] * absb[ind1, :])
                       + adjcolco2 * absco2)
        fracs[l, sl] = fracrefb

        # Empirical stratospheric-cooling modification (reduced g-points
        # 6..11 of this band).
        taug[l, gs + 5] = taug[l, gs + 5] * F("0.92")
        taug[l, gs + 6] = taug[l, gs + 6] * F("0.88")
        taug[l, gs + 7] = taug[l, gs + 7] * F("1.07")
        taug[l, gs + 8] = taug[l, gs + 8] * F("1.1")
        taug[l, gs + 9] = taug[l, gs + 9] * F("0.99")
        taug[l, gs + 10] = taug[l, gs + 10] * F("0.855")


def _taugb8(st, C, taug, fracs):
    """Band 8: 1080-1180 cm-1 (low key h2o + co2/o3/n2o minors + cfc12/
    cfc22adj; high key o3 + co2/n2o minors + cfcs).  Fortran 6452-6572."""
    chi_mls = C["ref/chi_mls"]
    absa = C["kg08/absa"]
    absb = C["kg08/absb"]
    selfref = C["kg08/selfref"]
    forref = C["kg08/forref"]
    ka_mco2 = C["kg08/ka_mco2"]
    ka_mo3 = C["kg08/ka_mo3"]
    ka_mn2o = C["kg08/ka_mn2o"]
    kb_mco2 = C["kg08/kb_mco2"]
    kb_mn2o = C["kg08/kb_mn2o"]
    fracrefa = C["kg08/fracrefa"]
    fracrefb = C["kg08/fracrefb"]
    cfc12 = C["kg08/cfc12"]
    cfc22adj = C["kg08/cfc22adj"]

    nl = len(st["pavel"])
    laytrop = int(st["laytrop"])
    gs = int(NGS[6])
    ng8 = int(NGC[7])
    sl = slice(gs, gs + ng8)
    nspa8 = int(NSPA[7])
    nspb8 = int(NSPB[7])
    wx = st["wx"]

    for lay in range(1, laytrop + 1):
        l = lay - 1
        jpv = int(st["jp"][l])
        chi_co2 = st["colco2"][l] / st["coldry"][l]
        ratco2 = F("1.e20") * chi_co2 / chi_mls[1, jpv]
        if ratco2 > F("3.0"):
            adjfac = F("2.0") + powf(ratco2 - F("2.0"), F("0.65"))
            adjcolco2 = adjfac * chi_mls[1, jpv] * st["coldry"][l] \
                * F("1.e-20")
        else:
            adjcolco2 = st["colco2"][l]

        ind0 = ((jpv - 1) * 5 + (int(st["jt"][l]) - 1)) * nspa8 + 1
        ind1 = (jpv * 5 + (int(st["jt1"][l]) - 1)) * nspa8 + 1
        inds = int(st["indself"][l])
        indf = int(st["indfor"][l])
        indm = int(st["indminor"][l])

        tauself = st["selffac"][l] * (
            selfref[inds - 1, :] + st["selffrac"][l]
            * (selfref[inds, :] - selfref[inds - 1, :]))
        taufor = st["forfac"][l] * (
            forref[indf - 1, :] + st["forfrac"][l]
            * (forref[indf, :] - forref[indf - 1, :]))
        absco2 = (ka_mco2[indm - 1, :] + st["minorfrac"][l]
                  * (ka_mco2[indm, :] - ka_mco2[indm - 1, :]))
        abso3 = (ka_mo3[indm - 1, :] + st["minorfrac"][l]
                 * (ka_mo3[indm, :] - ka_mo3[indm - 1, :]))
        absn2o = (ka_mn2o[indm - 1, :] + st["minorfrac"][l]
                  * (ka_mn2o[indm, :] - ka_mn2o[indm - 1, :]))
        taug[l, sl] = (st["colh2o"][l]
                       * (st["fac00"][l] * absa[ind0 - 1, :]
                          + st["fac10"][l] * absa[ind0, :]
                          + st["fac01"][l] * absa[ind1 - 1, :]
                          + st["fac11"][l] * absa[ind1, :])
                       + tauself + taufor
                       + adjcolco2 * absco2
                       + st["colo3"][l] * abso3
                       + st["coln2o"][l] * absn2o
                       + wx[2, l] * cfc12
                       + wx[3, l] * cfc22adj)
        fracs[l, sl] = fracrefa

    for lay in range(laytrop + 1, nl + 1):
        l = lay - 1
        jpv = int(st["jp"][l])
        chi_co2 = st["colco2"][l] / st["coldry"][l]
        ratco2 = F("1.e20") * chi_co2 / chi_mls[1, jpv]
        if ratco2 > F("3.0"):
            adjfac = F("2.0") + powf(ratco2 - F("2.0"), F("0.65"))
            adjcolco2 = adjfac * chi_mls[1, jpv] * st["coldry"][l] \
                * F("1.e-20")
        else:
            adjcolco2 = st["colco2"][l]

        ind0 = ((jpv - 13) * 5 + (int(st["jt"][l]) - 1)) * nspb8 + 1
        ind1 = ((jpv - 12) * 5 + (int(st["jt1"][l]) - 1)) * nspb8 + 1
        indm = int(st["indminor"][l])

        absco2 = (kb_mco2[indm - 1, :] + st["minorfrac"][l]
                  * (kb_mco2[indm, :] - kb_mco2[indm - 1, :]))
        absn2o = (kb_mn2o[indm - 1, :] + st["minorfrac"][l]
                  * (kb_mn2o[indm, :] - kb_mn2o[indm - 1, :]))
        taug[l, sl] = (st["colo3"][l]
                       * (st["fac00"][l] * absb[ind0 - 1, :]
                          + st["fac10"][l] * absb[ind0, :]
                          + st["fac01"][l] * absb[ind1 - 1, :]
                          + st["fac11"][l] * absb[ind1, :])
                       + adjcolco2 * absco2
                       + st["coln2o"][l] * absn2o
                       + wx[2, l] * cfc12
                       + wx[3, l] * cfc22adj)
        fracs[l, sl] = fracrefb


def _taugb9(st, C, taug, fracs):
    """Band 9: 1180-1390 cm-1 (low key h2o,ch4 + n2o minor; high key ch4 +
    n2o minor).  Fortran lines 6575-6835."""
    chi_mls = C["ref/chi_mls"]
    oneminus = F(C["con/oneminus"])
    absa = C["kg09/absa"]
    absb = C["kg09/absb"]
    selfref = C["kg09/selfref"]
    forref = C["kg09/forref"]
    ka_mn2o = C["kg09/ka_mn2o"]
    kb_mn2o = C["kg09/kb_mn2o"]
    fracrefa = C["kg09/fracrefa"]
    fracrefb = C["kg09/fracrefb"]

    # P = 212 mb
    refrat_planck_a = chi_mls[0, 8] / chi_mls[5, 8]
    # P = 706.272 mb
    refrat_m_a = chi_mls[0, 2] / chi_mls[5, 2]

    nl = len(st["pavel"])
    laytrop = int(st["laytrop"])
    gs = int(NGS[7])
    ng9 = int(NGC[8])
    sl = slice(gs, gs + ng9)
    nspa9 = int(NSPA[8])
    nspb9 = int(NSPB[8])

    for lay in range(1, laytrop + 1):
        l = lay - 1
        colh2o = st["colh2o"][l]
        colch4 = st["colch4"][l]
        jpv = int(st["jp"][l])

        speccomb = colh2o + st["rat_h2och4"][l] * colch4
        specparm = colh2o / speccomb
        if specparm >= oneminus:
            specparm = oneminus
        specmult = F("8.0") * specparm
        js = 1 + int(specmult)
        fs = np.fmod(specmult, F("1.0"))

        speccomb1 = colh2o + st["rat_h2och4_1"][l] * colch4
        specparm1 = colh2o / speccomb1
        if specparm1 >= oneminus:
            specparm1 = oneminus
        specmult1 = F("8.0") * specparm1
        js1 = 1 + int(specmult1)
        fs1 = np.fmod(specmult1, F("1.0"))

        speccomb_mn2o = colh2o + refrat_m_a * colch4
        specparm_mn2o = colh2o / speccomb_mn2o
        if specparm_mn2o >= oneminus:
            specparm_mn2o = oneminus
        specmult_mn2o = F("8.0") * specparm_mn2o
        jmn2o = 1 + int(specmult_mn2o)
        fmn2o = np.fmod(specmult_mn2o, F("1.0"))

        chi_n2o = st["coln2o"][l] / st["coldry"][l]
        ratn2o = F("1.e20") * chi_n2o / chi_mls[3, jpv]
        if ratn2o > F("1.5"):
            adjfac = F("0.5") + powf(ratn2o - F("0.5"), F("0.65"))
            adjcoln2o = adjfac * chi_mls[3, jpv] * st["coldry"][l] \
                * F("1.e-20")
        else:
            adjcoln2o = st["coln2o"][l]

        speccomb_planck = colh2o + refrat_planck_a * colch4
        specparm_planck = colh2o / speccomb_planck
        if specparm_planck >= oneminus:
            specparm_planck = oneminus
        specmult_planck = F("8.0") * specparm_planck
        jpl = 1 + int(specmult_planck)
        fpl = np.fmod(specmult_planck, F("1.0"))

        ind0 = ((jpv - 1) * 5 + (int(st["jt"][l]) - 1)) * nspa9 + js
        ind1 = (jpv * 5 + (int(st["jt1"][l]) - 1)) * nspa9 + js1
        inds = int(st["indself"][l])
        indf = int(st["indfor"][l])
        indm = int(st["indminor"][l])

        tauself = st["selffac"][l] * (
            selfref[inds - 1, :] + st["selffrac"][l]
            * (selfref[inds, :] - selfref[inds - 1, :]))
        taufor = st["forfac"][l] * (
            forref[indf - 1, :] + st["forfrac"][l]
            * (forref[indf, :] - forref[indf - 1, :]))
        n2om1 = (ka_mn2o[jmn2o - 1, indm - 1, :] + fmn2o
                 * (ka_mn2o[jmn2o, indm - 1, :]
                    - ka_mn2o[jmn2o - 1, indm - 1, :]))
        n2om2 = (ka_mn2o[jmn2o - 1, indm, :] + fmn2o
                 * (ka_mn2o[jmn2o, indm, :]
                    - ka_mn2o[jmn2o - 1, indm, :]))
        absn2o = n2om1 + st["minorfrac"][l] * (n2om2 - n2om1)

        tau_major = _nine_major(specparm, fs, st["fac00"][l],
                                st["fac10"][l], absa, ind0, speccomb)
        tau_major1 = _nine_major(specparm1, fs1, st["fac01"][l],
                                 st["fac11"][l], absa, ind1, speccomb1)

        taug[l, sl] = (tau_major + tau_major1
                       + tauself + taufor
                       + adjcoln2o * absn2o)
        fracs[l, sl] = (fracrefa[:, jpl - 1] + fpl
                        * (fracrefa[:, jpl] - fracrefa[:, jpl - 1]))

    for lay in range(laytrop + 1, nl + 1):
        l = lay - 1
        jpv = int(st["jp"][l])
        chi_n2o = st["coln2o"][l] / st["coldry"][l]
        ratn2o = F("1.e20") * chi_n2o / chi_mls[3, jpv]
        if ratn2o > F("1.5"):
            adjfac = F("0.5") + powf(ratn2o - F("0.5"), F("0.65"))
            adjcoln2o = adjfac * chi_mls[3, jpv] * st["coldry"][l] \
                * F("1.e-20")
        else:
            adjcoln2o = st["coln2o"][l]

        ind0 = ((jpv - 13) * 5 + (int(st["jt"][l]) - 1)) * nspb9 + 1
        ind1 = ((jpv - 12) * 5 + (int(st["jt1"][l]) - 1)) * nspb9 + 1
        indm = int(st["indminor"][l])

        absn2o = (kb_mn2o[indm - 1, :] + st["minorfrac"][l]
                  * (kb_mn2o[indm, :] - kb_mn2o[indm - 1, :]))
        taug[l, sl] = (st["colch4"][l]
                       * (st["fac00"][l] * absb[ind0 - 1, :]
                          + st["fac10"][l] * absb[ind0, :]
                          + st["fac01"][l] * absb[ind1 - 1, :]
                          + st["fac11"][l] * absb[ind1, :])
                       + adjcoln2o * absn2o)
        fracs[l, sl] = fracrefb

# ---- merged fragment: cmbgb 2-16 ----

def _reduce_first(raw_arr, band):
    """Plain-sum reduction of fracref*(iprsm, jp) over the FIRST axis.

    Mirrors `sumf = sumf + fracrefao(iprsm,jp)` / `fracrefa(igc,jp) =
    sumf`: for each new g-point igc, sum the ngn consecutive original
    g-points sequentially in FP32, independently for every jp column.
    """
    ngc_b = int(NGC[band - 1])
    ngn_b = _ngn_for_band(band)
    out = np.zeros((ngc_b,) + raw_arr.shape[1:], dtype=np.float32)
    iprsm = 0
    for igc in range(ngc_b):
        acc = np.zeros(raw_arr.shape[1:], dtype=np.float32)
        for _ in range(ngn_b[igc]):
            iprsm += 1
            acc = (acc + raw_arr[iprsm - 1]).astype(np.float32)
        out[igc] = acc
    return out


def _cmbgb2(mod, rwgt, out):
    """cmbgb2 (module lines 8432-8512): ka/kb/self/for, 1-D fracrefs."""
    out["kg02/ka"] = _reduce_g(mod["kao"], 2, rwgt)
    out["kg02/kb"] = _reduce_g(mod["kbo"], 2, rwgt)
    out["kg02/selfref"] = _reduce_g(mod["selfrefo"], 2, rwgt)
    out["kg02/forref"] = _reduce_g(mod["forrefo"], 2, rwgt)
    out["kg02/fracrefa"] = _reduce_g(mod["fracrefao"], 2, rwgt,
                                     weighted=False)
    out["kg02/fracrefb"] = _reduce_g(mod["fracrefbo"], 2, rwgt,
                                     weighted=False)


def _cmbgb3(mod, rwgt, out):
    """cmbgb3 (8515-8642): 9-species; fracrefs reduce over axis 0."""
    out["kg03/ka"] = _reduce_g(mod["kao"], 3, rwgt)
    out["kg03/kb"] = _reduce_g(mod["kbo"], 3, rwgt)
    out["kg03/ka_mn2o"] = _reduce_g(mod["kao_mn2o"], 3, rwgt)
    out["kg03/kb_mn2o"] = _reduce_g(mod["kbo_mn2o"], 3, rwgt)
    out["kg03/selfref"] = _reduce_g(mod["selfrefo"], 3, rwgt)
    out["kg03/forref"] = _reduce_g(mod["forrefo"], 3, rwgt)
    out["kg03/fracrefa"] = _reduce_first(mod["fracrefao"], 3)
    out["kg03/fracrefb"] = _reduce_first(mod["fracrefbo"], 3)


def _cmbgb4(mod, rwgt, out):
    """cmbgb4 (8645-8741): 9-species; fracrefa jp=1..9, fracrefb jp=1..5."""
    out["kg04/ka"] = _reduce_g(mod["kao"], 4, rwgt)
    out["kg04/kb"] = _reduce_g(mod["kbo"], 4, rwgt)
    out["kg04/selfref"] = _reduce_g(mod["selfrefo"], 4, rwgt)
    out["kg04/forref"] = _reduce_g(mod["forrefo"], 4, rwgt)
    out["kg04/fracrefa"] = _reduce_first(mod["fracrefao"], 4)
    out["kg04/fracrefb"] = _reduce_first(mod["fracrefbo"], 4)


def _cmbgb5(mod, rwgt, out):
    """cmbgb5 (8744-8867): 9-species + ka_mo3 + rwgt-weighted ccl4 tail."""
    out["kg05/ka"] = _reduce_g(mod["kao"], 5, rwgt)
    out["kg05/kb"] = _reduce_g(mod["kbo"], 5, rwgt)
    out["kg05/ka_mo3"] = _reduce_g(mod["kao_mo3"], 5, rwgt)
    out["kg05/selfref"] = _reduce_g(mod["selfrefo"], 5, rwgt)
    out["kg05/forref"] = _reduce_g(mod["forrefo"], 5, rwgt)
    out["kg05/fracrefa"] = _reduce_first(mod["fracrefao"], 5)
    out["kg05/fracrefb"] = _reduce_first(mod["fracrefbo"], 5)
    out["kg05/ccl4"] = _reduce_g(mod["ccl4o"], 5, rwgt)


def _cmbgb6(mod, rwgt, out):
    """cmbgb6 (8870-8958): ka/ka_mco2/self/for; fracrefa plain,
    cfc11adj/cfc12 rwgt-weighted (interleaved in one Fortran loop)."""
    out["kg06/ka"] = _reduce_g(mod["kao"], 6, rwgt)
    out["kg06/ka_mco2"] = _reduce_g(mod["kao_mco2"], 6, rwgt)
    out["kg06/selfref"] = _reduce_g(mod["selfrefo"], 6, rwgt)
    out["kg06/forref"] = _reduce_g(mod["forrefo"], 6, rwgt)
    out["kg06/fracrefa"] = _reduce_g(mod["fracrefao"], 6, rwgt,
                                     weighted=False)
    out["kg06/cfc11adj"] = _reduce_g(mod["cfc11adjo"], 6, rwgt)
    out["kg06/cfc12"] = _reduce_g(mod["cfc12o"], 6, rwgt)


def _cmbgb7(mod, rwgt, out):
    """cmbgb7 (8961-9082): 9-species low key, 1-species high key;
    fracrefa over axis 0, fracrefb 1-D plain."""
    out["kg07/ka"] = _reduce_g(mod["kao"], 7, rwgt)
    out["kg07/kb"] = _reduce_g(mod["kbo"], 7, rwgt)
    out["kg07/ka_mco2"] = _reduce_g(mod["kao_mco2"], 7, rwgt)
    out["kg07/kb_mco2"] = _reduce_g(mod["kbo_mco2"], 7, rwgt)
    out["kg07/selfref"] = _reduce_g(mod["selfrefo"], 7, rwgt)
    out["kg07/forref"] = _reduce_g(mod["forrefo"], 7, rwgt)
    out["kg07/fracrefa"] = _reduce_first(mod["fracrefao"], 7)
    out["kg07/fracrefb"] = _reduce_g(mod["fracrefbo"], 7, rwgt,
                                     weighted=False)


def _cmbgb8(mod, rwgt, out):
    """cmbgb8 (9085-9201): five 2-D minor-gas arrays (interleaved
    sumk1..sumk5 in the Fortran), plain fracrefs, weighted cfc tails."""
    out["kg08/ka"] = _reduce_g(mod["kao"], 8, rwgt)
    out["kg08/kb"] = _reduce_g(mod["kbo"], 8, rwgt)
    out["kg08/selfref"] = _reduce_g(mod["selfrefo"], 8, rwgt)
    out["kg08/forref"] = _reduce_g(mod["forrefo"], 8, rwgt)
    out["kg08/ka_mco2"] = _reduce_g(mod["kao_mco2"], 8, rwgt)
    out["kg08/kb_mco2"] = _reduce_g(mod["kbo_mco2"], 8, rwgt)
    out["kg08/ka_mo3"] = _reduce_g(mod["kao_mo3"], 8, rwgt)
    out["kg08/ka_mn2o"] = _reduce_g(mod["kao_mn2o"], 8, rwgt)
    out["kg08/kb_mn2o"] = _reduce_g(mod["kbo_mn2o"], 8, rwgt)
    out["kg08/fracrefa"] = _reduce_g(mod["fracrefao"], 8, rwgt,
                                     weighted=False)
    out["kg08/fracrefb"] = _reduce_g(mod["fracrefbo"], 8, rwgt,
                                     weighted=False)
    out["kg08/cfc12"] = _reduce_g(mod["cfc12o"], 8, rwgt)
    out["kg08/cfc22adj"] = _reduce_g(mod["cfc22adjo"], 8, rwgt)


def _cmbgb9(mod, rwgt, out):
    """cmbgb9 (9204-9326): fracrefa over axis 0, fracrefb 1-D plain."""
    out["kg09/ka"] = _reduce_g(mod["kao"], 9, rwgt)
    out["kg09/kb"] = _reduce_g(mod["kbo"], 9, rwgt)
    out["kg09/ka_mn2o"] = _reduce_g(mod["kao_mn2o"], 9, rwgt)
    out["kg09/kb_mn2o"] = _reduce_g(mod["kbo_mn2o"], 9, rwgt)
    out["kg09/selfref"] = _reduce_g(mod["selfrefo"], 9, rwgt)
    out["kg09/forref"] = _reduce_g(mod["forrefo"], 9, rwgt)
    out["kg09/fracrefa"] = _reduce_first(mod["fracrefao"], 9)
    out["kg09/fracrefb"] = _reduce_g(mod["fracrefbo"], 9, rwgt,
                                     weighted=False)


def _cmbgb10(mod, rwgt, out):
    """cmbgb10 (9329-9413): ka/kb/self/for, 1-D fracrefs."""
    out["kg10/ka"] = _reduce_g(mod["kao"], 10, rwgt)
    out["kg10/kb"] = _reduce_g(mod["kbo"], 10, rwgt)
    out["kg10/selfref"] = _reduce_g(mod["selfrefo"], 10, rwgt)
    out["kg10/forref"] = _reduce_g(mod["forrefo"], 10, rwgt)
    out["kg10/fracrefa"] = _reduce_g(mod["fracrefao"], 10, rwgt,
                                     weighted=False)
    out["kg10/fracrefb"] = _reduce_g(mod["fracrefbo"], 10, rwgt,
                                     weighted=False)


def _cmbgb11(mod, rwgt, out):
    """cmbgb11 (9416-9516): ka_mo2/kb_mo2 interleaved in the Fortran."""
    out["kg11/ka"] = _reduce_g(mod["kao"], 11, rwgt)
    out["kg11/kb"] = _reduce_g(mod["kbo"], 11, rwgt)
    out["kg11/ka_mo2"] = _reduce_g(mod["kao_mo2"], 11, rwgt)
    out["kg11/kb_mo2"] = _reduce_g(mod["kbo_mo2"], 11, rwgt)
    out["kg11/selfref"] = _reduce_g(mod["selfrefo"], 11, rwgt)
    out["kg11/forref"] = _reduce_g(mod["forrefo"], 11, rwgt)
    out["kg11/fracrefa"] = _reduce_g(mod["fracrefao"], 11, rwgt,
                                     weighted=False)
    out["kg11/fracrefb"] = _reduce_g(mod["fracrefbo"], 11, rwgt,
                                     weighted=False)


def _cmbgb12(mod, rwgt, out):
    """cmbgb12 (9519-9588): low key only; fracrefa over axis 0."""
    out["kg12/ka"] = _reduce_g(mod["kao"], 12, rwgt)
    out["kg12/selfref"] = _reduce_g(mod["selfrefo"], 12, rwgt)
    out["kg12/forref"] = _reduce_g(mod["forrefo"], 12, rwgt)
    out["kg12/fracrefa"] = _reduce_first(mod["fracrefao"], 12)


def _cmbgb13(mod, rwgt, out):
    """cmbgb13 (9591-9701): ka_mco2/ka_mco interleaved; kb_mo3 only in
    the high region; fracrefb (1-D plain) precedes fracrefa in the
    Fortran -- order is bitwise-irrelevant, both transcribed."""
    out["kg13/ka"] = _reduce_g(mod["kao"], 13, rwgt)
    out["kg13/ka_mco2"] = _reduce_g(mod["kao_mco2"], 13, rwgt)
    out["kg13/ka_mco"] = _reduce_g(mod["kao_mco"], 13, rwgt)
    out["kg13/kb_mo3"] = _reduce_g(mod["kbo_mo3"], 13, rwgt)
    out["kg13/selfref"] = _reduce_g(mod["selfrefo"], 13, rwgt)
    out["kg13/forref"] = _reduce_g(mod["forrefo"], 13, rwgt)
    out["kg13/fracrefb"] = _reduce_g(mod["fracrefbo"], 13, rwgt,
                                     weighted=False)
    out["kg13/fracrefa"] = _reduce_first(mod["fracrefao"], 13)


def _cmbgb14(mod, rwgt, out):
    """cmbgb14 (9704-9788): ka/kb/self/for, 1-D fracrefs."""
    out["kg14/ka"] = _reduce_g(mod["kao"], 14, rwgt)
    out["kg14/kb"] = _reduce_g(mod["kbo"], 14, rwgt)
    out["kg14/selfref"] = _reduce_g(mod["selfrefo"], 14, rwgt)
    out["kg14/forref"] = _reduce_g(mod["forrefo"], 14, rwgt)
    out["kg14/fracrefa"] = _reduce_g(mod["fracrefao"], 14, rwgt,
                                     weighted=False)
    out["kg14/fracrefb"] = _reduce_g(mod["fracrefbo"], 14, rwgt,
                                     weighted=False)


def _cmbgb15(mod, rwgt, out):
    """cmbgb15 (9791-9875): low key only + ka_mn2; fracrefa over axis 0."""
    out["kg15/ka"] = _reduce_g(mod["kao"], 15, rwgt)
    out["kg15/ka_mn2"] = _reduce_g(mod["kao_mn2"], 15, rwgt)
    out["kg15/selfref"] = _reduce_g(mod["selfrefo"], 15, rwgt)
    out["kg15/forref"] = _reduce_g(mod["forrefo"], 15, rwgt)
    out["kg15/fracrefa"] = _reduce_first(mod["fracrefao"], 15)


def _cmbgb16(mod, rwgt, out):
    """cmbgb16 (9878-9971): fracrefb (1-D plain) precedes fracrefa
    (axis-0) in the Fortran."""
    out["kg16/ka"] = _reduce_g(mod["kao"], 16, rwgt)
    out["kg16/kb"] = _reduce_g(mod["kbo"], 16, rwgt)
    out["kg16/selfref"] = _reduce_g(mod["selfrefo"], 16, rwgt)
    out["kg16/forref"] = _reduce_g(mod["forrefo"], 16, rwgt)
    out["kg16/fracrefb"] = _reduce_g(mod["fracrefbo"], 16, rwgt,
                                     weighted=False)
    out["kg16/fracrefa"] = _reduce_first(mod["fracrefao"], 16)


TAUGB_IMPLS.update({
    2: _taugb2, 6: _taugb6, 7: _taugb7, 8: _taugb8, 9: _taugb9,
    10: _taugb10, 11: _taugb11, 12: _taugb12,
})
_CMBGB_IMPLS.update({
    2: _cmbgb2, 3: _cmbgb3, 4: _cmbgb4, 5: _cmbgb5, 6: _cmbgb6,
    7: _cmbgb7, 8: _cmbgb8, 9: _cmbgb9, 10: _cmbgb10, 11: _cmbgb11,
    12: _cmbgb12, 13: _cmbgb13, 14: _cmbgb14, 15: _cmbgb15, 16: _cmbgb16,
})


# ---- merged fragment: bands 3-5 ----

def _spec_major(specparm, fs, facp, fact, absa, ind, speccomb):
    """The specparm branch triplet + tau_major sum shared by the two-major-
    species bands (3, 4, 5).  facp/fact are fac00/fac10 (ind0 half) or
    fac01/fac11 (ind1 half).  Returns tau_major (length-ng f32 vector).

    Fortran: e.g. taugb3 lines 5343-5402 (facs) and 5415-5461 (tau_major).
    The facs are computed once per layer in the Fortran, before the ig
    loop; they are per-layer scalars, so computing them here per call is
    the identical op sequence.
    """
    if specparm < F("0.125"):
        p = fs - F("1.0")
        p4 = pow4(p)              # p**4, gfortran square-and-multiply
        fk0 = p4
        fk1 = F("1.0") - p - F("2.0") * p4
        fk2 = p + p4
        fac0p = fk0 * facp
        fac1p = fk1 * facp
        fac2p = fk2 * facp
        fac0t = fk0 * fact
        fac1t = fk1 * fact
        fac2t = fk2 * fact
        return speccomb * (fac0p * absa[ind - 1, :]
                           + fac1p * absa[ind, :]
                           + fac2p * absa[ind + 1, :]
                           + fac0t * absa[ind + 8, :]
                           + fac1t * absa[ind + 9, :]
                           + fac2t * absa[ind + 10, :])
    elif specparm > F("0.875"):
        p = -fs
        p4 = pow4(p)
        fk0 = p4
        fk1 = F("1.0") - p - F("2.0") * p4
        fk2 = p + p4
        fac0p = fk0 * facp
        fac1p = fk1 * facp
        fac2p = fk2 * facp
        fac0t = fk0 * fact
        fac1t = fk1 * fact
        fac2t = fk2 * fact
        return speccomb * (fac2p * absa[ind - 2, :]
                           + fac1p * absa[ind - 1, :]
                           + fac0p * absa[ind, :]
                           + fac2t * absa[ind + 7, :]
                           + fac1t * absa[ind + 8, :]
                           + fac0t * absa[ind + 9, :])
    else:
        fac0p = (F("1.0") - fs) * facp
        fac0t = (F("1.0") - fs) * fact
        fac1p = fs * facp
        fac1t = fs * fact
        return speccomb * (fac0p * absa[ind - 1, :]
                           + fac1p * absa[ind, :]
                           + fac0t * absa[ind + 8, :]
                           + fac1t * absa[ind + 9, :])


def _taugb3(st, C, taug, fracs):
    """Band 3: 500-630 cm-1 (low key h2o,co2 + n2o minor; high key h2o,co2
    + n2o minor).  Fortran lines 5241-5553."""
    chi_mls = C["ref/chi_mls"]
    oneminus = F(C["con/oneminus"])
    absa = C["kg03/absa"]
    absb = C["kg03/absb"]
    selfref = C["kg03/selfref"]
    forref = C["kg03/forref"]
    ka_mn2o = C["kg03/ka_mn2o"]
    kb_mn2o = C["kg03/kb_mn2o"]
    fracrefa = C["kg03/fracrefa"]
    fracrefb = C["kg03/fracrefb"]

    # P = 212.725 mb
    refrat_planck_a = chi_mls[0, 8] / chi_mls[1, 8]
    # P = 95.58 mb
    refrat_planck_b = chi_mls[0, 12] / chi_mls[1, 12]
    # P = 706.270 mb
    refrat_m_a = chi_mls[0, 2] / chi_mls[1, 2]
    # P = 95.58 mb
    refrat_m_b = chi_mls[0, 12] / chi_mls[1, 12]

    nl = len(st["pavel"])
    laytrop = int(st["laytrop"])
    gs = int(NGS[1])
    ng3 = int(NGC[2])
    sl = slice(gs, gs + ng3)
    nspa3 = int(NSPA[2])
    nspb3 = int(NSPB[2])

    for lay in range(1, laytrop + 1):
        l = lay - 1
        colh2o = st["colh2o"][l]
        colco2 = st["colco2"][l]
        jpv = int(st["jp"][l])

        speccomb = colh2o + st["rat_h2oco2"][l] * colco2
        specparm = colh2o / speccomb
        if specparm >= oneminus:
            specparm = oneminus
        specmult = F("8.0") * specparm
        js = 1 + int(specmult)
        fs = np.fmod(specmult, F("1.0"))

        speccomb1 = colh2o + st["rat_h2oco2_1"][l] * colco2
        specparm1 = colh2o / speccomb1
        if specparm1 >= oneminus:
            specparm1 = oneminus
        specmult1 = F("8.0") * specparm1
        js1 = 1 + int(specmult1)
        fs1 = np.fmod(specmult1, F("1.0"))

        speccomb_mn2o = colh2o + refrat_m_a * colco2
        specparm_mn2o = colh2o / speccomb_mn2o
        if specparm_mn2o >= oneminus:
            specparm_mn2o = oneminus
        specmult_mn2o = F("8.0") * specparm_mn2o
        jmn2o = 1 + int(specmult_mn2o)
        fmn2o = np.fmod(specmult_mn2o, F("1.0"))
        # Fortran computes fmn2omf = minorfrac(lay)*fmn2o here; it is never
        # read anywhere in taugb3 (dead statement), so it is omitted.

        chi_n2o = st["coln2o"][l] / st["coldry"][l]
        ratn2o = F("1.e20") * chi_n2o / chi_mls[3, jpv]   # chi_mls(4,jp+1)
        if ratn2o > F("1.5"):
            adjfac = F("0.5") + powf(ratn2o - F("0.5"), F("0.65"))
            adjcoln2o = adjfac * chi_mls[3, jpv] * st["coldry"][l] \
                * F("1.e-20")
        else:
            adjcoln2o = st["coln2o"][l]

        speccomb_planck = colh2o + refrat_planck_a * colco2
        specparm_planck = colh2o / speccomb_planck
        if specparm_planck >= oneminus:
            specparm_planck = oneminus
        specmult_planck = F("8.0") * specparm_planck
        jpl = 1 + int(specmult_planck)
        fpl = np.fmod(specmult_planck, F("1.0"))

        ind0 = ((jpv - 1) * 5 + (int(st["jt"][l]) - 1)) * nspa3 + js
        ind1 = (jpv * 5 + (int(st["jt1"][l]) - 1)) * nspa3 + js1
        inds = int(st["indself"][l])
        indf = int(st["indfor"][l])
        indm = int(st["indminor"][l])

        tauself = st["selffac"][l] * (
            selfref[inds - 1, :] + st["selffrac"][l]
            * (selfref[inds, :] - selfref[inds - 1, :]))
        taufor = st["forfac"][l] * (
            forref[indf - 1, :] + st["forfrac"][l]
            * (forref[indf, :] - forref[indf - 1, :]))
        n2om1 = (ka_mn2o[jmn2o - 1, indm - 1, :] + fmn2o
                 * (ka_mn2o[jmn2o, indm - 1, :]
                    - ka_mn2o[jmn2o - 1, indm - 1, :]))
        n2om2 = (ka_mn2o[jmn2o - 1, indm, :] + fmn2o
                 * (ka_mn2o[jmn2o, indm, :]
                    - ka_mn2o[jmn2o - 1, indm, :]))
        absn2o = n2om1 + st["minorfrac"][l] * (n2om2 - n2om1)

        tau_major = _spec_major(specparm, fs, st["fac00"][l],
                                st["fac10"][l], absa, ind0, speccomb)
        tau_major1 = _spec_major(specparm1, fs1, st["fac01"][l],
                                 st["fac11"][l], absa, ind1, speccomb1)

        taug[l, sl] = (tau_major + tau_major1
                       + tauself + taufor
                       + adjcoln2o * absn2o)
        fracs[l, sl] = (fracrefa[:, jpl - 1] + fpl
                        * (fracrefa[:, jpl] - fracrefa[:, jpl - 1]))

    for lay in range(laytrop + 1, nl + 1):
        l = lay - 1
        colh2o = st["colh2o"][l]
        colco2 = st["colco2"][l]
        jpv = int(st["jp"][l])

        speccomb = colh2o + st["rat_h2oco2"][l] * colco2
        specparm = colh2o / speccomb
        if specparm >= oneminus:
            specparm = oneminus
        specmult = F("4.0") * specparm
        js = 1 + int(specmult)
        fs = np.fmod(specmult, F("1.0"))

        speccomb1 = colh2o + st["rat_h2oco2_1"][l] * colco2
        specparm1 = colh2o / speccomb1
        if specparm1 >= oneminus:
            specparm1 = oneminus
        specmult1 = F("4.0") * specparm1
        js1 = 1 + int(specmult1)
        fs1 = np.fmod(specmult1, F("1.0"))

        fac000 = (F("1.0") - fs) * st["fac00"][l]
        fac010 = (F("1.0") - fs) * st["fac10"][l]
        fac100 = fs * st["fac00"][l]
        fac110 = fs * st["fac10"][l]
        fac001 = (F("1.0") - fs1) * st["fac01"][l]
        fac011 = (F("1.0") - fs1) * st["fac11"][l]
        fac101 = fs1 * st["fac01"][l]
        fac111 = fs1 * st["fac11"][l]

        speccomb_mn2o = colh2o + refrat_m_b * colco2
        specparm_mn2o = colh2o / speccomb_mn2o
        if specparm_mn2o >= oneminus:
            specparm_mn2o = oneminus
        specmult_mn2o = F("4.0") * specparm_mn2o
        jmn2o = 1 + int(specmult_mn2o)
        fmn2o = np.fmod(specmult_mn2o, F("1.0"))
        # fmn2omf: dead statement in the Fortran here too; omitted.

        chi_n2o = st["coln2o"][l] / st["coldry"][l]
        # Fortran line 5508 writes the literal as 1.e20 (default real, same
        # f32 value as 1.e20_rb in the lower loop).
        ratn2o = F("1.e20") * chi_n2o / chi_mls[3, jpv]
        if ratn2o > F("1.5"):
            adjfac = F("0.5") + powf(ratn2o - F("0.5"), F("0.65"))
            adjcoln2o = adjfac * chi_mls[3, jpv] * st["coldry"][l] \
                * F("1.e-20")
        else:
            adjcoln2o = st["coln2o"][l]

        speccomb_planck = colh2o + refrat_planck_b * colco2
        specparm_planck = colh2o / speccomb_planck
        if specparm_planck >= oneminus:
            specparm_planck = oneminus
        specmult_planck = F("4.0") * specparm_planck
        jpl = 1 + int(specmult_planck)
        fpl = np.fmod(specmult_planck, F("1.0"))

        ind0 = ((jpv - 13) * 5 + (int(st["jt"][l]) - 1)) * nspb3 + js
        ind1 = ((jpv - 12) * 5 + (int(st["jt1"][l]) - 1)) * nspb3 + js1
        indf = int(st["indfor"][l])
        indm = int(st["indminor"][l])

        taufor = st["forfac"][l] * (
            forref[indf - 1, :] + st["forfrac"][l]
            * (forref[indf, :] - forref[indf - 1, :]))
        n2om1 = (kb_mn2o[jmn2o - 1, indm - 1, :] + fmn2o
                 * (kb_mn2o[jmn2o, indm - 1, :]
                    - kb_mn2o[jmn2o - 1, indm - 1, :]))
        n2om2 = (kb_mn2o[jmn2o - 1, indm, :] + fmn2o
                 * (kb_mn2o[jmn2o, indm, :]
                    - kb_mn2o[jmn2o - 1, indm, :]))
        absn2o = n2om1 + st["minorfrac"][l] * (n2om2 - n2om1)

        taug[l, sl] = (speccomb
                       * (fac000 * absb[ind0 - 1, :]
                          + fac100 * absb[ind0, :]
                          + fac010 * absb[ind0 + 4, :]
                          + fac110 * absb[ind0 + 5, :])
                       + speccomb1
                       * (fac001 * absb[ind1 - 1, :]
                          + fac101 * absb[ind1, :]
                          + fac011 * absb[ind1 + 4, :]
                          + fac111 * absb[ind1 + 5, :])
                       + taufor
                       + adjcoln2o * absn2o)
        fracs[l, sl] = (fracrefb[:, jpl - 1] + fpl
                        * (fracrefb[:, jpl] - fracrefb[:, jpl - 1]))


def _taugb4(st, C, taug, fracs):
    """Band 4: 630-700 cm-1 (low key h2o,co2; high key o3,co2 + empirical
    stratospheric-cooling g-point rescale).  Fortran lines 5556-5812."""
    chi_mls = C["ref/chi_mls"]
    oneminus = F(C["con/oneminus"])
    absa = C["kg04/absa"]
    absb = C["kg04/absb"]
    selfref = C["kg04/selfref"]
    forref = C["kg04/forref"]
    fracrefa = C["kg04/fracrefa"]
    fracrefb = C["kg04/fracrefb"]

    # P = 142.5940 mb
    refrat_planck_a = chi_mls[0, 10] / chi_mls[1, 10]
    # P = 95.58350 mb
    refrat_planck_b = chi_mls[2, 12] / chi_mls[1, 12]

    nl = len(st["pavel"])
    laytrop = int(st["laytrop"])
    gs = int(NGS[2])
    ng4 = int(NGC[3])
    sl = slice(gs, gs + ng4)
    nspa4 = int(NSPA[3])
    nspb4 = int(NSPB[3])

    for lay in range(1, laytrop + 1):
        l = lay - 1
        colh2o = st["colh2o"][l]
        colco2 = st["colco2"][l]
        jpv = int(st["jp"][l])

        speccomb = colh2o + st["rat_h2oco2"][l] * colco2
        specparm = colh2o / speccomb
        if specparm >= oneminus:
            specparm = oneminus
        specmult = F("8.0") * specparm
        js = 1 + int(specmult)
        fs = np.fmod(specmult, F("1.0"))

        speccomb1 = colh2o + st["rat_h2oco2_1"][l] * colco2
        specparm1 = colh2o / speccomb1
        if specparm1 >= oneminus:
            specparm1 = oneminus
        specmult1 = F("8.0") * specparm1
        js1 = 1 + int(specmult1)
        fs1 = np.fmod(specmult1, F("1.0"))

        speccomb_planck = colh2o + refrat_planck_a * colco2
        specparm_planck = colh2o / speccomb_planck
        if specparm_planck >= oneminus:
            specparm_planck = oneminus
        specmult_planck = F("8.0") * specparm_planck
        jpl = 1 + int(specmult_planck)
        fpl = np.fmod(specmult_planck, F("1.0"))

        ind0 = ((jpv - 1) * 5 + (int(st["jt"][l]) - 1)) * nspa4 + js
        ind1 = (jpv * 5 + (int(st["jt1"][l]) - 1)) * nspa4 + js1
        inds = int(st["indself"][l])
        indf = int(st["indfor"][l])

        tauself = st["selffac"][l] * (
            selfref[inds - 1, :] + st["selffrac"][l]
            * (selfref[inds, :] - selfref[inds - 1, :]))
        taufor = st["forfac"][l] * (
            forref[indf - 1, :] + st["forfrac"][l]
            * (forref[indf, :] - forref[indf - 1, :]))

        tau_major = _spec_major(specparm, fs, st["fac00"][l],
                                st["fac10"][l], absa, ind0, speccomb)
        tau_major1 = _spec_major(specparm1, fs1, st["fac01"][l],
                                 st["fac11"][l], absa, ind1, speccomb1)

        taug[l, sl] = (tau_major + tau_major1
                       + tauself + taufor)
        fracs[l, sl] = (fracrefa[:, jpl - 1] + fpl
                        * (fracrefa[:, jpl] - fracrefa[:, jpl - 1]))

    for lay in range(laytrop + 1, nl + 1):
        l = lay - 1
        colo3 = st["colo3"][l]
        colco2 = st["colco2"][l]

        speccomb = colo3 + st["rat_o3co2"][l] * colco2
        specparm = colo3 / speccomb
        if specparm >= oneminus:
            specparm = oneminus
        specmult = F("4.0") * specparm
        js = 1 + int(specmult)
        fs = np.fmod(specmult, F("1.0"))

        speccomb1 = colo3 + st["rat_o3co2_1"][l] * colco2
        specparm1 = colo3 / speccomb1
        if specparm1 >= oneminus:
            specparm1 = oneminus
        specmult1 = F("4.0") * specparm1
        js1 = 1 + int(specmult1)
        fs1 = np.fmod(specmult1, F("1.0"))

        fac000 = (F("1.0") - fs) * st["fac00"][l]
        fac010 = (F("1.0") - fs) * st["fac10"][l]
        fac100 = fs * st["fac00"][l]
        fac110 = fs * st["fac10"][l]
        fac001 = (F("1.0") - fs1) * st["fac01"][l]
        fac011 = (F("1.0") - fs1) * st["fac11"][l]
        fac101 = fs1 * st["fac01"][l]
        fac111 = fs1 * st["fac11"][l]

        speccomb_planck = colo3 + refrat_planck_b * colco2
        specparm_planck = colo3 / speccomb_planck
        if specparm_planck >= oneminus:
            specparm_planck = oneminus
        specmult_planck = F("4.0") * specparm_planck
        jpl = 1 + int(specmult_planck)
        fpl = np.fmod(specmult_planck, F("1.0"))

        ind0 = ((int(st["jp"][l]) - 13) * 5
                + (int(st["jt"][l]) - 1)) * nspb4 + js
        ind1 = ((int(st["jp"][l]) - 12) * 5
                + (int(st["jt1"][l]) - 1)) * nspb4 + js1

        taug[l, sl] = (speccomb
                       * (fac000 * absb[ind0 - 1, :]
                          + fac100 * absb[ind0, :]
                          + fac010 * absb[ind0 + 4, :]
                          + fac110 * absb[ind0 + 5, :])
                       + speccomb1
                       * (fac001 * absb[ind1 - 1, :]
                          + fac101 * absb[ind1, :]
                          + fac011 * absb[ind1 + 4, :]
                          + fac111 * absb[ind1 + 5, :]))
        fracs[l, sl] = (fracrefb[:, jpl - 1] + fpl
                        * (fracrefb[:, jpl] - fracrefb[:, jpl - 1]))

        # Empirical modification to improve stratospheric cooling rates
        # for co2 (reduced g-points 8..14 of this band); literals are
        # default real in the Fortran (lines 5802-5808).
        taug[l, gs + 7] = taug[l, gs + 7] * F("0.92")
        taug[l, gs + 8] = taug[l, gs + 8] * F("0.88")
        taug[l, gs + 9] = taug[l, gs + 9] * F("1.07")
        taug[l, gs + 10] = taug[l, gs + 10] * F("1.1")
        taug[l, gs + 11] = taug[l, gs + 11] * F("0.99")
        taug[l, gs + 12] = taug[l, gs + 12] * F("0.88")
        taug[l, gs + 13] = taug[l, gs + 13] * F("0.943")


def _taugb5(st, C, taug, fracs):
    """Band 5: 700-820 cm-1 (low key h2o,co2 + o3 minor + ccl4 xsec;
    high key o3,co2 + ccl4 xsec).  Fortran lines 5815-6087."""
    chi_mls = C["ref/chi_mls"]
    oneminus = F(C["con/oneminus"])
    absa = C["kg05/absa"]
    absb = C["kg05/absb"]
    selfref = C["kg05/selfref"]
    forref = C["kg05/forref"]
    ka_mo3 = C["kg05/ka_mo3"]
    fracrefa = C["kg05/fracrefa"]
    fracrefb = C["kg05/fracrefb"]
    ccl4 = C["kg05/ccl4"]

    # P = 473.420 mb
    refrat_planck_a = chi_mls[0, 4] / chi_mls[1, 4]
    # P = 0.2369 mb
    refrat_planck_b = chi_mls[2, 42] / chi_mls[1, 42]
    # P = 317.3480 mb
    refrat_m_a = chi_mls[0, 6] / chi_mls[1, 6]

    nl = len(st["pavel"])
    laytrop = int(st["laytrop"])
    gs = int(NGS[3])
    ng5 = int(NGC[4])
    sl = slice(gs, gs + ng5)
    nspa5 = int(NSPA[4])
    nspb5 = int(NSPB[4])
    wx = st["wx"]

    for lay in range(1, laytrop + 1):
        l = lay - 1
        colh2o = st["colh2o"][l]
        colco2 = st["colco2"][l]

        speccomb = colh2o + st["rat_h2oco2"][l] * colco2
        specparm = colh2o / speccomb
        if specparm >= oneminus:
            specparm = oneminus
        specmult = F("8.0") * specparm
        js = 1 + int(specmult)
        fs = np.fmod(specmult, F("1.0"))

        speccomb1 = colh2o + st["rat_h2oco2_1"][l] * colco2
        specparm1 = colh2o / speccomb1
        if specparm1 >= oneminus:
            specparm1 = oneminus
        specmult1 = F("8.0") * specparm1
        js1 = 1 + int(specmult1)
        fs1 = np.fmod(specmult1, F("1.0"))

        speccomb_mo3 = colh2o + refrat_m_a * colco2
        specparm_mo3 = colh2o / speccomb_mo3
        if specparm_mo3 >= oneminus:
            specparm_mo3 = oneminus
        specmult_mo3 = F("8.0") * specparm_mo3
        jmo3 = 1 + int(specmult_mo3)
        fmo3 = np.fmod(specmult_mo3, F("1.0"))

        speccomb_planck = colh2o + refrat_planck_a * colco2
        specparm_planck = colh2o / speccomb_planck
        if specparm_planck >= oneminus:
            specparm_planck = oneminus
        specmult_planck = F("8.0") * specparm_planck
        jpl = 1 + int(specmult_planck)
        fpl = np.fmod(specmult_planck, F("1.0"))

        ind0 = ((int(st["jp"][l]) - 1) * 5
                + (int(st["jt"][l]) - 1)) * nspa5 + js
        ind1 = (int(st["jp"][l]) * 5
                + (int(st["jt1"][l]) - 1)) * nspa5 + js1
        inds = int(st["indself"][l])
        indf = int(st["indfor"][l])
        indm = int(st["indminor"][l])

        tauself = st["selffac"][l] * (
            selfref[inds - 1, :] + st["selffrac"][l]
            * (selfref[inds, :] - selfref[inds - 1, :]))
        taufor = st["forfac"][l] * (
            forref[indf - 1, :] + st["forfrac"][l]
            * (forref[indf, :] - forref[indf - 1, :]))
        o3m1 = (ka_mo3[jmo3 - 1, indm - 1, :] + fmo3
                * (ka_mo3[jmo3, indm - 1, :]
                   - ka_mo3[jmo3 - 1, indm - 1, :]))
        o3m2 = (ka_mo3[jmo3 - 1, indm, :] + fmo3
                * (ka_mo3[jmo3, indm, :]
                   - ka_mo3[jmo3 - 1, indm, :]))
        abso3 = o3m1 + st["minorfrac"][l] * (o3m2 - o3m1)

        tau_major = _spec_major(specparm, fs, st["fac00"][l],
                                st["fac10"][l], absa, ind0, speccomb)
        tau_major1 = _spec_major(specparm1, fs1, st["fac01"][l],
                                 st["fac11"][l], absa, ind1, speccomb1)

        taug[l, sl] = (tau_major + tau_major1
                       + tauself + taufor
                       + abso3 * st["colo3"][l]
                       + wx[0, l] * ccl4)
        fracs[l, sl] = (fracrefa[:, jpl - 1] + fpl
                        * (fracrefa[:, jpl] - fracrefa[:, jpl - 1]))

    for lay in range(laytrop + 1, nl + 1):
        l = lay - 1
        colo3 = st["colo3"][l]
        colco2 = st["colco2"][l]

        speccomb = colo3 + st["rat_o3co2"][l] * colco2
        specparm = colo3 / speccomb
        if specparm >= oneminus:
            specparm = oneminus
        specmult = F("4.0") * specparm
        js = 1 + int(specmult)
        fs = np.fmod(specmult, F("1.0"))

        speccomb1 = colo3 + st["rat_o3co2_1"][l] * colco2
        specparm1 = colo3 / speccomb1
        if specparm1 >= oneminus:
            specparm1 = oneminus
        specmult1 = F("4.0") * specparm1
        js1 = 1 + int(specmult1)
        fs1 = np.fmod(specmult1, F("1.0"))

        fac000 = (F("1.0") - fs) * st["fac00"][l]
        fac010 = (F("1.0") - fs) * st["fac10"][l]
        fac100 = fs * st["fac00"][l]
        fac110 = fs * st["fac10"][l]
        fac001 = (F("1.0") - fs1) * st["fac01"][l]
        fac011 = (F("1.0") - fs1) * st["fac11"][l]
        fac101 = fs1 * st["fac01"][l]
        fac111 = fs1 * st["fac11"][l]

        speccomb_planck = colo3 + refrat_planck_b * colco2
        specparm_planck = colo3 / speccomb_planck
        if specparm_planck >= oneminus:
            specparm_planck = oneminus
        specmult_planck = F("4.0") * specparm_planck
        jpl = 1 + int(specmult_planck)
        fpl = np.fmod(specmult_planck, F("1.0"))

        ind0 = ((int(st["jp"][l]) - 13) * 5
                + (int(st["jt"][l]) - 1)) * nspb5 + js
        ind1 = ((int(st["jp"][l]) - 12) * 5
                + (int(st["jt1"][l]) - 1)) * nspb5 + js1

        taug[l, sl] = (speccomb
                       * (fac000 * absb[ind0 - 1, :]
                          + fac100 * absb[ind0, :]
                          + fac010 * absb[ind0 + 4, :]
                          + fac110 * absb[ind0 + 5, :])
                       + speccomb1
                       * (fac001 * absb[ind1 - 1, :]
                          + fac101 * absb[ind1, :]
                          + fac011 * absb[ind1 + 4, :]
                          + fac111 * absb[ind1 + 5, :])
                       + wx[0, l] * ccl4)
        fracs[l, sl] = (fracrefb[:, jpl - 1] + fpl
                        * (fracrefb[:, jpl] - fracrefb[:, jpl - 1]))


# ---- merged fragment: bands 13-16 (shares _nine_major with 6-9) ----

def _taugb13(st, C, taug, fracs):
    """Band 13: 2080-2250 cm-1 (low key h2o,n2o; low minors co2 and co;
    high minor o3 only).  Fortran lines 7189-7445."""
    chi_mls = C["ref/chi_mls"]
    oneminus = F(C["con/oneminus"])
    absa = C["kg13/absa"]
    selfref = C["kg13/selfref"]
    forref = C["kg13/forref"]
    ka_mco2 = C["kg13/ka_mco2"]
    ka_mco = C["kg13/ka_mco"]
    kb_mo3 = C["kg13/kb_mo3"]
    fracrefa = C["kg13/fracrefa"]
    fracrefb = C["kg13/fracrefb"]

    # P = 473.420 mb (Level 5)
    refrat_planck_a = chi_mls[0, 4] / chi_mls[3, 4]
    # P = 1053. (Level 1)
    refrat_m_a = chi_mls[0, 0] / chi_mls[3, 0]
    # P = 706. (Level 3)
    refrat_m_a3 = chi_mls[0, 2] / chi_mls[3, 2]

    nl = len(st["pavel"])
    laytrop = int(st["laytrop"])
    gs = int(NGS[11])
    ng13 = int(NGC[12])
    sl = slice(gs, gs + ng13)
    nspa13 = int(NSPA[12])

    for lay in range(1, laytrop + 1):
        l = lay - 1
        colh2o = st["colh2o"][l]
        coln2o = st["coln2o"][l]

        speccomb = colh2o + st["rat_h2on2o"][l] * coln2o
        specparm = colh2o / speccomb
        if specparm >= oneminus:
            specparm = oneminus
        specmult = F("8.0") * specparm
        js = 1 + int(specmult)
        fs = np.fmod(specmult, F("1.0"))

        speccomb1 = colh2o + st["rat_h2on2o_1"][l] * coln2o
        specparm1 = colh2o / speccomb1
        if specparm1 >= oneminus:
            specparm1 = oneminus
        specmult1 = F("8.0") * specparm1
        js1 = 1 + int(specmult1)
        fs1 = np.fmod(specmult1, F("1.0"))

        speccomb_mco2 = colh2o + refrat_m_a * coln2o
        specparm_mco2 = colh2o / speccomb_mco2
        if specparm_mco2 >= oneminus:
            specparm_mco2 = oneminus
        specmult_mco2 = F("8.0") * specparm_mco2
        jmco2 = 1 + int(specmult_mco2)
        fmco2 = np.fmod(specmult_mco2, F("1.0"))

        # CO2-too-major empirical adjustment (fixed 3.55e-4 reference).
        chi_co2 = st["colco2"][l] / st["coldry"][l]
        ratco2 = F("1.e20") * chi_co2 / F("3.55e-4")
        if ratco2 > F("3.0"):
            adjfac = F("2.0") + powf(ratco2 - F("2.0"), F("0.68"))
            adjcolco2 = adjfac * F("3.55e-4") * st["coldry"][l] * F("1.e-20")
        else:
            adjcolco2 = st["colco2"][l]

        speccomb_mco = colh2o + refrat_m_a3 * coln2o
        specparm_mco = colh2o / speccomb_mco
        if specparm_mco >= oneminus:
            specparm_mco = oneminus
        specmult_mco = F("8.0") * specparm_mco
        jmco = 1 + int(specmult_mco)
        fmco = np.fmod(specmult_mco, F("1.0"))

        speccomb_planck = colh2o + refrat_planck_a * coln2o
        specparm_planck = colh2o / speccomb_planck
        if specparm_planck >= oneminus:
            specparm_planck = oneminus
        specmult_planck = F("8.0") * specparm_planck
        jpl = 1 + int(specmult_planck)
        fpl = np.fmod(specmult_planck, F("1.0"))

        ind0 = ((int(st["jp"][l]) - 1) * 5
                + (int(st["jt"][l]) - 1)) * nspa13 + js
        ind1 = (int(st["jp"][l]) * 5
                + (int(st["jt1"][l]) - 1)) * nspa13 + js1
        inds = int(st["indself"][l])
        indf = int(st["indfor"][l])
        indm = int(st["indminor"][l])

        tauself = st["selffac"][l] * (
            selfref[inds - 1, :] + st["selffrac"][l]
            * (selfref[inds, :] - selfref[inds - 1, :]))
        taufor = st["forfac"][l] * (
            forref[indf - 1, :] + st["forfrac"][l]
            * (forref[indf, :] - forref[indf - 1, :]))
        co2m1 = (ka_mco2[jmco2 - 1, indm - 1, :] + fmco2
                 * (ka_mco2[jmco2, indm - 1, :]
                    - ka_mco2[jmco2 - 1, indm - 1, :]))
        co2m2 = (ka_mco2[jmco2 - 1, indm, :] + fmco2
                 * (ka_mco2[jmco2, indm, :]
                    - ka_mco2[jmco2 - 1, indm, :]))
        absco2 = co2m1 + st["minorfrac"][l] * (co2m2 - co2m1)
        com1 = (ka_mco[jmco - 1, indm - 1, :] + fmco
                * (ka_mco[jmco, indm - 1, :]
                   - ka_mco[jmco - 1, indm - 1, :]))
        com2 = (ka_mco[jmco - 1, indm, :] + fmco
                * (ka_mco[jmco, indm, :]
                   - ka_mco[jmco - 1, indm, :]))
        absco = com1 + st["minorfrac"][l] * (com2 - com1)

        tau_major = _nine_major(specparm, fs, st["fac00"][l],
                                st["fac10"][l], absa, ind0, speccomb)
        tau_major1 = _nine_major(specparm1, fs1, st["fac01"][l],
                                 st["fac11"][l], absa, ind1, speccomb1)

        taug[l, sl] = (tau_major + tau_major1
                       + tauself + taufor
                       + adjcolco2 * absco2
                       + st["colco"][l] * absco)
        fracs[l, sl] = (fracrefa[:, jpl - 1] + fpl
                        * (fracrefa[:, jpl] - fracrefa[:, jpl - 1]))

    # Upper atmosphere: o3 minor ONLY (no foreign continuum in this copy).
    for lay in range(laytrop + 1, nl + 1):
        l = lay - 1
        indm = int(st["indminor"][l])
        abso3 = (kb_mo3[indm - 1, :] + st["minorfrac"][l]
                 * (kb_mo3[indm, :] - kb_mo3[indm - 1, :]))
        taug[l, sl] = st["colo3"][l] * abso3
        fracs[l, sl] = fracrefb


def _taugb14(st, C, taug, fracs):
    """Band 14: 2250-2380 cm-1 (low - co2; high - co2).  Fortran lines
    7448-7506."""
    absa = C["kg14/absa"]
    absb = C["kg14/absb"]
    selfref = C["kg14/selfref"]
    forref = C["kg14/forref"]
    fracrefa = C["kg14/fracrefa"]
    fracrefb = C["kg14/fracrefb"]

    nl = len(st["pavel"])
    laytrop = int(st["laytrop"])
    gs = int(NGS[12])
    ng14 = int(NGC[13])
    sl = slice(gs, gs + ng14)
    nspa14 = int(NSPA[13])
    nspb14 = int(NSPB[13])

    for lay in range(1, laytrop + 1):
        l = lay - 1
        ind0 = ((int(st["jp"][l]) - 1) * 5
                + (int(st["jt"][l]) - 1)) * nspa14 + 1
        ind1 = (int(st["jp"][l]) * 5
                + (int(st["jt1"][l]) - 1)) * nspa14 + 1
        inds = int(st["indself"][l])
        indf = int(st["indfor"][l])
        tauself = st["selffac"][l] * (
            selfref[inds - 1, :] + st["selffrac"][l]
            * (selfref[inds, :] - selfref[inds - 1, :]))
        taufor = st["forfac"][l] * (
            forref[indf - 1, :] + st["forfrac"][l]
            * (forref[indf, :] - forref[indf - 1, :]))
        taug[l, sl] = (st["colco2"][l]
                       * (st["fac00"][l] * absa[ind0 - 1, :]
                          + st["fac10"][l] * absa[ind0, :]
                          + st["fac01"][l] * absa[ind1 - 1, :]
                          + st["fac11"][l] * absa[ind1, :])
                       + tauself + taufor)
        fracs[l, sl] = fracrefa

    for lay in range(laytrop + 1, nl + 1):
        l = lay - 1
        ind0 = ((int(st["jp"][l]) - 13) * 5
                + (int(st["jt"][l]) - 1)) * nspb14 + 1
        ind1 = ((int(st["jp"][l]) - 12) * 5
                + (int(st["jt1"][l]) - 1)) * nspb14 + 1
        taug[l, sl] = (st["colco2"][l]
                       * (st["fac00"][l] * absb[ind0 - 1, :]
                          + st["fac10"][l] * absb[ind0, :]
                          + st["fac01"][l] * absb[ind1 - 1, :]
                          + st["fac11"][l] * absb[ind1, :]))
        fracs[l, sl] = fracrefb


def _taugb15(st, C, taug, fracs):
    """Band 15: 2380-2600 cm-1 (low key n2o,co2 + n2 minor via
    colbrd*scaleminor; high - NOTHING: taug = fracs = 0).  Fortran lines
    7509-7731."""
    chi_mls = C["ref/chi_mls"]
    oneminus = F(C["con/oneminus"])
    absa = C["kg15/absa"]
    selfref = C["kg15/selfref"]
    forref = C["kg15/forref"]
    ka_mn2 = C["kg15/ka_mn2"]
    fracrefa = C["kg15/fracrefa"]

    # P = 1053. mb (Level 1) -- Planck and minor use the same ratio.
    refrat_planck_a = chi_mls[3, 0] / chi_mls[1, 0]
    refrat_m_a = chi_mls[3, 0] / chi_mls[1, 0]

    nl = len(st["pavel"])
    laytrop = int(st["laytrop"])
    gs = int(NGS[13])
    ng15 = int(NGC[14])
    sl = slice(gs, gs + ng15)
    nspa15 = int(NSPA[14])

    for lay in range(1, laytrop + 1):
        l = lay - 1
        coln2o = st["coln2o"][l]
        colco2 = st["colco2"][l]

        speccomb = coln2o + st["rat_n2oco2"][l] * colco2
        specparm = coln2o / speccomb
        if specparm >= oneminus:
            specparm = oneminus
        specmult = F("8.0") * specparm
        js = 1 + int(specmult)
        fs = np.fmod(specmult, F("1.0"))

        speccomb1 = coln2o + st["rat_n2oco2_1"][l] * colco2
        specparm1 = coln2o / speccomb1
        if specparm1 >= oneminus:
            specparm1 = oneminus
        specmult1 = F("8.0") * specparm1
        js1 = 1 + int(specmult1)
        fs1 = np.fmod(specmult1, F("1.0"))

        speccomb_mn2 = coln2o + refrat_m_a * colco2
        specparm_mn2 = coln2o / speccomb_mn2
        if specparm_mn2 >= oneminus:
            specparm_mn2 = oneminus
        specmult_mn2 = F("8.0") * specparm_mn2
        jmn2 = 1 + int(specmult_mn2)
        fmn2 = np.fmod(specmult_mn2, F("1.0"))

        speccomb_planck = coln2o + refrat_planck_a * colco2
        specparm_planck = coln2o / speccomb_planck
        if specparm_planck >= oneminus:
            specparm_planck = oneminus
        specmult_planck = F("8.0") * specparm_planck
        jpl = 1 + int(specmult_planck)
        fpl = np.fmod(specmult_planck, F("1.0"))

        ind0 = ((int(st["jp"][l]) - 1) * 5
                + (int(st["jt"][l]) - 1)) * nspa15 + js
        ind1 = (int(st["jp"][l]) * 5
                + (int(st["jt1"][l]) - 1)) * nspa15 + js1
        inds = int(st["indself"][l])
        indf = int(st["indfor"][l])
        indm = int(st["indminor"][l])

        scalen2 = st["colbrd"][l] * st["scaleminor"][l]

        tauself = st["selffac"][l] * (
            selfref[inds - 1, :] + st["selffrac"][l]
            * (selfref[inds, :] - selfref[inds - 1, :]))
        taufor = st["forfac"][l] * (
            forref[indf - 1, :] + st["forfrac"][l]
            * (forref[indf, :] - forref[indf - 1, :]))
        n2m1 = (ka_mn2[jmn2 - 1, indm - 1, :] + fmn2
                * (ka_mn2[jmn2, indm - 1, :]
                   - ka_mn2[jmn2 - 1, indm - 1, :]))
        n2m2 = (ka_mn2[jmn2 - 1, indm, :] + fmn2
                * (ka_mn2[jmn2, indm, :]
                   - ka_mn2[jmn2 - 1, indm, :]))
        taun2 = scalen2 * (n2m1 + st["minorfrac"][l] * (n2m2 - n2m1))

        tau_major = _nine_major(specparm, fs, st["fac00"][l],
                                st["fac10"][l], absa, ind0, speccomb)
        tau_major1 = _nine_major(specparm1, fs1, st["fac01"][l],
                                 st["fac11"][l], absa, ind1, speccomb1)

        taug[l, sl] = (tau_major + tau_major1
                       + tauself + taufor
                       + taun2)
        fracs[l, sl] = (fracrefa[:, jpl - 1] + fpl
                        * (fracrefa[:, jpl] - fracrefa[:, jpl - 1]))

    for lay in range(laytrop + 1, nl + 1):
        l = lay - 1
        taug[l, sl] = F("0.0")
        fracs[l, sl] = F("0.0")


def _taugb16(st, C, taug, fracs):
    """Band 16: 2600-3250 cm-1 (low key h2o,ch4; high key ch4).  Fortran
    lines 7734-7940.  NOTE nspb(16)=0 in this copy, so the upper-loop
    ind0/ind1 are always 1 (transcribed as-is)."""
    chi_mls = C["ref/chi_mls"]
    oneminus = F(C["con/oneminus"])
    absa = C["kg16/absa"]
    absb = C["kg16/absb"]
    selfref = C["kg16/selfref"]
    forref = C["kg16/forref"]
    fracrefa = C["kg16/fracrefa"]
    fracrefb = C["kg16/fracrefb"]

    # P = 387. mb (Level 6)
    refrat_planck_a = chi_mls[0, 5] / chi_mls[5, 5]

    nl = len(st["pavel"])
    laytrop = int(st["laytrop"])
    gs = int(NGS[14])
    ng16 = int(NGC[15])
    sl = slice(gs, gs + ng16)
    nspa16 = int(NSPA[15])
    nspb16 = int(NSPB[15])

    for lay in range(1, laytrop + 1):
        l = lay - 1
        colh2o = st["colh2o"][l]
        colch4 = st["colch4"][l]

        speccomb = colh2o + st["rat_h2och4"][l] * colch4
        specparm = colh2o / speccomb
        if specparm >= oneminus:
            specparm = oneminus
        specmult = F("8.0") * specparm
        js = 1 + int(specmult)
        fs = np.fmod(specmult, F("1.0"))

        speccomb1 = colh2o + st["rat_h2och4_1"][l] * colch4
        specparm1 = colh2o / speccomb1
        if specparm1 >= oneminus:
            specparm1 = oneminus
        specmult1 = F("8.0") * specparm1
        js1 = 1 + int(specmult1)
        fs1 = np.fmod(specmult1, F("1.0"))

        speccomb_planck = colh2o + refrat_planck_a * colch4
        specparm_planck = colh2o / speccomb_planck
        if specparm_planck >= oneminus:
            specparm_planck = oneminus
        specmult_planck = F("8.0") * specparm_planck
        jpl = 1 + int(specmult_planck)
        fpl = np.fmod(specmult_planck, F("1.0"))

        ind0 = ((int(st["jp"][l]) - 1) * 5
                + (int(st["jt"][l]) - 1)) * nspa16 + js
        ind1 = (int(st["jp"][l]) * 5
                + (int(st["jt1"][l]) - 1)) * nspa16 + js1
        inds = int(st["indself"][l])
        indf = int(st["indfor"][l])

        tauself = st["selffac"][l] * (
            selfref[inds - 1, :] + st["selffrac"][l]
            * (selfref[inds, :] - selfref[inds - 1, :]))
        taufor = st["forfac"][l] * (
            forref[indf - 1, :] + st["forfrac"][l]
            * (forref[indf, :] - forref[indf - 1, :]))

        tau_major = _nine_major(specparm, fs, st["fac00"][l],
                                st["fac10"][l], absa, ind0, speccomb)
        tau_major1 = _nine_major(specparm1, fs1, st["fac01"][l],
                                 st["fac11"][l], absa, ind1, speccomb1)

        taug[l, sl] = (tau_major + tau_major1
                       + tauself + taufor)
        fracs[l, sl] = (fracrefa[:, jpl - 1] + fpl
                        * (fracrefa[:, jpl] - fracrefa[:, jpl - 1]))

    for lay in range(laytrop + 1, nl + 1):
        l = lay - 1
        ind0 = ((int(st["jp"][l]) - 13) * 5
                + (int(st["jt"][l]) - 1)) * nspb16 + 1
        ind1 = ((int(st["jp"][l]) - 12) * 5
                + (int(st["jt1"][l]) - 1)) * nspb16 + 1
        taug[l, sl] = (st["colch4"][l]
                       * (st["fac00"][l] * absb[ind0 - 1, :]
                          + st["fac10"][l] * absb[ind0, :]
                          + st["fac01"][l] * absb[ind1 - 1, :]
                          + st["fac11"][l] * absb[ind1, :]))
        fracs[l, sl] = fracrefb


TAUGB_IMPLS.update({
    3: _taugb3, 4: _taugb4, 5: _taugb5,
    13: _taugb13, 14: _taugb14, 15: _taugb15, 16: _taugb16,
})

# ---------------------------------------------------------------------------
# Section 8 -- rrtmg_lw composition (rrtmg_lw_rad::rrtmg_lw, lines
# 10694-11183).  Same inputs/outputs as the Fortran subroutine the WRF
# driver calls; the driver-side prep (RRTMG_LWRAD) belongs to the
# integration wave and is NOT reproduced here.
# ---------------------------------------------------------------------------


def rrtmg_lw(ncol, nlay, icld, play, plev, tlay, tlev, tsfc, h2ovmr,
             o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, cfc11vmr, cfc12vmr,
             cfc22vmr, ccl4vmr, emis, inflglw, iceflglw, liqflglw,
             cldfmcl, taucmcl, ciwpmcl, clwpmcl, cswpmcl, reicmcl,
             relqmcl, resnmcl, tauaer, C):
    """Full LW compute chain for ncol columns; arguments mirror the
    Fortran dummies (2-D (ncol, nlay); mcica 3-D (ngptlw, ncol, nlay)).

    Returns dict of uflx/dflx/hr/uflxc/dflxc/hrc (+ zeroed
    uflxcln/dflxcln, the WRF_CHEM=0 path).  ``C`` from
    build_lw_coefficients (production) or the oracle dump (tests).

    Driver-side interface, field by field -- exactly what WRF v4.6.1's
    RRTMG_LWRAD builds before calling rrtmg_lw (authority: the wrapper
    transcription in tools/rrtmg_wrf461_oracle/lw_extract.F90, proven
    bitwise-equal to the untouched RRTMG_LWRAD on every fixture).  nlay
    is the module-level nlayers = kme + nint(p_top[Pa]*0.01/4mb) - 1,
    i.e. model layers plus 4-mb buffer layers to ~0 mb (Cavallo):

      play    (ncol,nlay)   hPa   p3d/100 below kte; buffer midpoints above
      plev    (ncol,nlay+1) hPa   p8w/100; plev(nlay+1) = 0.0 exactly
      tlay    (ncol,nlay)   K     t3d below kte; std-profile match above
      tlev    (ncol,nlay+1) K     t8w below; shifted MLS-blend table above
      tsfc    (ncol)        K     TSK
      h2ovmr  (ncol,nlay)         max(qv,1e-12)*1.607793 (mmr->vmr)
      o3vmr   (ncol,nlay)         o33d (vmr) below kte; shifted INIRAD
                                  climatology above (o3input=2 path)
      co2vmr/ch4vmr/n2ovmr        scalar GHG values broadcast; co2 from
                                  (280+90*exp(0.02*(yr-2000)))*1e-6
      o2vmr                       0.209488; cfc11 0.251e-9, cfc12
                                  0.538e-9, cfc22 0.169e-9, ccl4 0.093e-9
      emis    (ncol,16)           EMISS broadcast to all 16 bands
      inflglw/iceflglw/liqflglw   2/3/1 base; has_reqc->3, has_reqi->4/4,
                                  has_reqs->5/5, P3 special-case->5/5
      cldfmcl..cswpmcl (140,ncol,nlay)  mcica_subcol_lw outputs
                                  (kissvec irng=0, permuteseed=150,
                                  icld=cldovrlp); water paths g/m2 from
                                  the CAM block (pdel*100/g*1000, in-cloud
                                  by max(0.01,cldfrac))
      reicmcl (ncol,nlay)   um    per iceflglw (iceflglw=3, Fu: *1.0315
                                  THEN capped at 140 um -- min(140, x),
                                  module_ra_rrtmg_lw.F:12598-12605; an
                                  upper cap, NOT a floor)
      relqmcl (ncol,nlay)   um    relcalc or re_cloud path
      resnmcl (ncol,nlay)   um    max(10,re_snow*1e6), wrapper-clamped 130
      tauaer  (ncol,nlay,16)      zeros with aer_ra_feedback=0
      icld                        cldovrlp namelist value (>=1; this port
                                  fails closed at 0, see inatm)
    """
    istart = 1
    iend = 16
    iout = 0
    iaer = 10
    nl = int(nlay)
    ncol = int(ncol)
    ngb = C["wvn/ngb"]

    uflx = np.zeros((ncol, nl + 1), dtype=np.float32)
    dflx = np.zeros((ncol, nl + 1), dtype=np.float32)
    uflxc = np.zeros((ncol, nl + 1), dtype=np.float32)
    dflxc = np.zeros((ncol, nl + 1), dtype=np.float32)
    uflxcln = np.zeros((ncol, nl + 1), dtype=np.float32)
    dflxcln = np.zeros((ncol, nl + 1), dtype=np.float32)
    hr = np.zeros((ncol, nl), dtype=np.float32)
    hrc = np.zeros((ncol, nl), dtype=np.float32)

    for iplon in range(1, ncol + 1):
        a = inatm(iplon, nl, icld, iaer, play, plev, tlay, tlev, tsfc,
                  h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,
                  cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, emis, inflglw,
                  iceflglw, liqflglw, cldfmcl, taucmcl, ciwpmcl,
                  clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl, tauaer, C)

        ncbands, taucmc = cldprmc(
            a["nlayers"], a["inflag"], a["iceflag"], a["liqflag"],
            a["cldfmc"], a["ciwpmc"], a["clwpmc"], a["cswpmc"],
            a["reicmc"], a["relqmc"], a["resnmc"], a["taucmc"], C)

        sc = setcoef(a["nlayers"], istart, a["pavel"], a["tavel"],
                     a["tz"], a["tbound"], a["semiss"], a["coldry"],
                     a["wkl"], a["wbrodl"], C)

        st = dict(sc)
        st.update({"pavel": a["pavel"], "wx": a["wx"],
                   "coldry": a["coldry"], "wbroad": a["wbrodl"]})
        fracs, taug = taumol(a["nlayers"], st, C)

        taut = np.zeros_like(taug)
        for ig in range(NGPTLW):
            taut[:, ig] = (taug[:, ig]
                           + a["taua"][:, int(ngb[ig]) - 1]
                           ).astype(np.float32)

        rt = rtrnmc(a["nlayers"], istart, iend, iout, a["pz"],
                    a["semiss"], ncbands, a["cldfmc"], taucmc,
                    sc["planklay"], sc["planklev"], sc["plankbnd"],
                    a["pwvcm"], fracs, taut, C)

        uflx[iplon - 1, :] = rt["totuflux"]
        dflx[iplon - 1, :] = rt["totdflux"]
        uflxc[iplon - 1, :] = rt["totuclfl"]
        dflxc[iplon - 1, :] = rt["totdclfl"]
        hr[iplon - 1, :] = rt["htr"][:nl]
        hrc[iplon - 1, :] = rt["htrc"][:nl]

    return {
        "uflx": uflx, "dflx": dflx, "hr": hr,
        "uflxc": uflxc, "dflxc": dflxc, "hrc": hrc,
        "uflxcln": uflxcln, "dflxcln": dflxcln,
    }

# ---------------------------------------------------------------------------
# Section 10 -- CUDA host layer.  Compiles gpuwm/core/kernels/rrtmg_lw*.cu
# as one translation unit (rrtmg_lw.cu first, band files after, sorted) and
# exposes per-routine launchers mirroring the NumPy reference.  cupy is
# imported lazily; nothing above this section needs a GPU.
# ---------------------------------------------------------------------------

#: taumol float-state slot order -- FROZEN, mirrored by rrtmg_lw.cu.
GPU_FSLOTS = (
    "pavel", "coldry", "colh2o", "colco2", "colo3", "coln2o", "colco",
    "colch4", "colo2", "colbrd", "fac00", "fac01", "fac10", "fac11",
    "rat_h2oco2", "rat_h2oco2_1", "rat_h2oo3", "rat_h2oo3_1",
    "rat_h2on2o", "rat_h2on2o_1", "rat_h2och4", "rat_h2och4_1",
    "rat_n2oco2", "rat_n2oco2_1", "rat_o3co2", "rat_o3co2_1",
    "selffac", "selffrac", "forfac", "forfrac", "minorfrac",
    "scaleminor", "scaleminorn2")
GPU_ISLOTS = ("jp", "jt", "jt1", "indself", "indfor", "indminor")

#: Per-band coefficient table order for the universal band-kernel
#: signature's tabs pointer array -- FROZEN, mirrored by each
#: rrtmg_lw_taugb*.cu file.
GPU_BAND_TABS = {
    1: ["absa", "absb", "selfref", "forref", "ka_mn2", "kb_mn2",
        "fracrefa", "fracrefb"],
    2: ["absa", "absb", "selfref", "forref", "fracrefa", "fracrefb"],
    3: ["absa", "absb", "selfref", "forref", "ka_mn2o", "kb_mn2o",
        "fracrefa", "fracrefb"],
    4: ["absa", "absb", "selfref", "forref", "fracrefa", "fracrefb"],
    5: ["absa", "absb", "selfref", "forref", "ka_mo3", "ccl4",
        "fracrefa", "fracrefb"],
    6: ["absa", "selfref", "forref", "ka_mco2", "cfc11adj", "cfc12",
        "fracrefa"],
    7: ["absa", "absb", "selfref", "forref", "ka_mco2", "kb_mco2",
        "fracrefa", "fracrefb"],
    8: ["absa", "absb", "selfref", "forref", "ka_mco2", "kb_mco2",
        "ka_mn2o", "kb_mn2o", "ka_mo3", "cfc12", "cfc22adj",
        "fracrefa", "fracrefb"],
    9: ["absa", "absb", "selfref", "forref", "ka_mn2o", "kb_mn2o",
        "fracrefa", "fracrefb"],
    10: ["absa", "absb", "selfref", "forref", "fracrefa", "fracrefb"],
    11: ["absa", "absb", "selfref", "forref", "ka_mo2", "kb_mo2",
         "fracrefa", "fracrefb"],
    12: ["absa", "selfref", "forref", "fracrefa"],
    13: ["absa", "selfref", "forref", "ka_mco2", "ka_mco", "kb_mo3",
         "fracrefa", "fracrefb"],
    14: ["absa", "absb", "selfref", "forref", "fracrefa", "fracrefb"],
    15: ["absa", "selfref", "forref", "ka_mn2", "fracrefa"],
    16: ["absa", "absb", "selfref", "forref", "fracrefa", "fracrefb"],
}

_GPU_MODULE = None
_GPU_KERNELS = {}
_GPU_PREFLIGHTED = False


def _gpu_source():
    import glob as _glob
    import os as _os
    kdir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "kernels")
    parts = [open(_os.path.join(kdir, "rrtmg_lw.cu"),
                  encoding="utf-8").read()]
    for path in sorted(_glob.glob(_os.path.join(kdir, "rrtmg_lw_*.cu"))):
        parts.append(open(path, encoding="utf-8").read())
    return "\n".join(parts)


def _gpu_module():
    """Compile the translation unit via NVRTC DIRECTLY and load the PTX.

    cupy.RawModule cannot be used here: CuPy's _compile_with_cache_cuda
    appends '-ftz=true' AFTER user options unconditionally (verified in
    cupy 14.0.1 source and by a live probe on the RTX 5090), and NVRTC
    honours the last occurrence, so subnormal float32 results would be
    flushed.  compile_using_nvrtc + cuda.function.Module bypasses the
    injection; gpu_preflight() proves subnormal survival on every run.

    ``-arch`` is NOT passed.  cupy appends its own
    (``cuda/compiler.py:_compile``) and NVRTC 13.0 rejects a repeated
    option outright -- ``nvrtc: error: --gpu-architecture (-arch) defined
    more than once`` -- where NVRTC 12 quietly honoured the last one.
    MEASURED on cupy 14.1.1 / CUDA 13.0.2 / sm_120: with the explicit
    ``-arch=compute_120`` this raises, without it the same source
    compiles and ``--ftz=false`` still survives (cupy injects no -ftz on
    this path).  The arch cupy derives is ``_cc._get_arch()``, which is
    the value the explicit option carried, so nothing about the compile
    target changes.
    """
    global _GPU_MODULE
    if _GPU_MODULE is None:
        import cupy as cp
        from cupy.cuda import compiler as _cc
        ptx, _mapping = _cc.compile_using_nvrtc(
            _gpu_source(),
            ("-std=c++17", "--ftz=false"),
            None, "rrtmg_lw.cu")
        mod = cp.cuda.function.Module()
        mod.load(ptx.encode() if isinstance(ptx, str) else ptx)
        _GPU_MODULE = mod
    return _GPU_MODULE


def _gpu_kernel(name):
    if name not in _GPU_KERNELS:
        _GPU_KERNELS[name] = _gpu_module().get_function(name)
    return _GPU_KERNELS[name]


def gpu_preflight(force=False):
    """Prove toolchain behaviour on the live device: subnormal survival
    through __fmul_rn under our compile options, and the device libm
    transcriptions equal the host ones bitwise on probe values."""
    global _GPU_PREFLIGHTED
    if _GPU_PREFLIGHTED and not force:
        return
    import cupy as cp
    x = cp.asarray(np.array(
        [1.0e-30, 1.0e-10, -100.0, 3.14159, 1.5, 0.65], dtype=np.float32))
    o = cp.zeros(4, dtype=cp.float32)
    _gpu_kernel("rlw_probe")((1,), (32,), (x, o, np.int32(4)))
    got = cp.asnumpy(o)
    want = np.array([
        np.float32(1.0e-30) * np.float32(1.0e-10),
        expf(np.float32(-100.0)),
        logf(np.float32(3.14159)),
        powf(np.float32(1.5), np.float32(0.65)),
    ], dtype=np.float32)
    if got.view(np.uint32).tolist() != want.view(np.uint32).tolist():
        raise RuntimeError(
            "rrtmg_lw GPU preflight failed: got %r want %r (subnormal "
            "flush or libm divergence on this toolchain)" % (got, want))
    _GPU_PREFLIGHTED = True


def gpu_pack_taumol_state(st, ncol=1):
    """Pack a (single-column) taumol state dict into the device slabs the
    band kernels consume.  Returns (laytrop, fs, isv, wx) device arrays."""
    import cupy as cp
    nl = len(st["pavel"])
    fs = np.zeros((len(GPU_FSLOTS), ncol, nl), dtype=np.float32)
    isv = np.zeros((len(GPU_ISLOTS), ncol, nl), dtype=np.int32)
    for i, name in enumerate(GPU_FSLOTS):
        fs[i, 0, :] = st[name]
    for i, name in enumerate(GPU_ISLOTS):
        isv[i, 0, :] = st[name]
    wx = np.zeros((ncol, MAXXSEC, nl), dtype=np.float32)
    wx[0] = st["wx"][:MAXXSEC, :]
    laytrop = np.full(ncol, int(st["laytrop"]), dtype=np.int32)
    return (cp.asarray(laytrop), cp.asarray(np.ascontiguousarray(fs)),
            cp.asarray(np.ascontiguousarray(isv)),
            cp.asarray(np.ascontiguousarray(wx)))


def gpu_band_tabs(band, C):
    """Device coefficient arrays + device pointer table for one band.
    Keep the returned arrays alive until the kernel has synchronised."""
    import cupy as cp
    arrays = []
    for name in GPU_BAND_TABS[band]:
        host = np.asarray(C["kg%02d/%s" % (band, name)], dtype=np.float32)
        arrays.append(cp.asarray(np.ascontiguousarray(
            host.ravel(order="F"))))
    ptrs = cp.asarray(np.array([a.data.ptr for a in arrays],
                               dtype=np.uint64))
    return arrays, ptrs


def gpu_taugb(band, st, C, taug_d=None, fracs_d=None, ncol=1):
    """Run one band kernel; returns device taug/fracs (ncol, nl, 140)."""
    import cupy as cp
    gpu_preflight()
    nl = len(st["pavel"])
    laytrop, fs, isv, wx = gpu_pack_taumol_state(st, ncol)
    chi = cp.asarray(np.asarray(C["ref/chi_mls"], dtype=np.float32
                                ).ravel(order="F"))
    oneminus = np.float32(C["con/oneminus"])
    arrays, ptrs = gpu_band_tabs(band, C)
    if taug_d is None:
        taug_d = cp.zeros((ncol, nl, NGPTLW), dtype=cp.float32)
    if fracs_d is None:
        fracs_d = cp.zeros((ncol, nl, NGPTLW), dtype=cp.float32)
    total = ncol * nl
    threads = 128
    blocks = (total + threads - 1) // threads
    _gpu_kernel("rlw_taugb%d" % band)(
        (blocks,), (threads,),
        (np.int32(ncol), np.int32(nl), laytrop, fs, isv, wx, chi,
         oneminus, ptrs, taug_d, fracs_d))
    cp.cuda.runtime.deviceSynchronize()
    del arrays
    return taug_d, fracs_d


def gpu_setcoef(nlayers, istart, pavel, tavel, tz, tbound, semiss, coldry,
                wkl, wbroad, C, ncol=1):
    """Device setcoef for one column; returns the NumPy-setcoef dict."""
    import cupy as cp
    gpu_preflight()
    nl = int(nlayers)

    def dev(a, dt=np.float32):
        return cp.asarray(np.ascontiguousarray(np.asarray(a, dtype=dt)))

    fkeys = ("colh2o", "colco2", "colo3", "coln2o", "colco", "colch4",
             "colo2", "colbrd", "fac00", "fac01", "fac10", "fac11",
             "rat_h2oco2", "rat_h2oco2_1", "rat_h2oo3", "rat_h2oo3_1",
             "rat_h2on2o", "rat_h2on2o_1", "rat_h2och4", "rat_h2och4_1",
             "rat_n2oco2", "rat_n2oco2_1", "rat_o3co2", "rat_o3co2_1",
             "selffac", "selffrac", "forfac", "forfrac", "minorfrac",
             "scaleminor", "scaleminorn2")
    outs_f = {k: cp.zeros((ncol, nl), dtype=cp.float32) for k in fkeys}
    outs_i = {k: cp.zeros((ncol, nl), dtype=cp.int32) for k in
              ("jp", "jt", "jt1", "indself", "indfor", "indminor")}
    laytrop = cp.zeros(ncol, dtype=cp.int32)
    planklay = cp.zeros((ncol, nl, NBNDLW), dtype=cp.float32)
    planklev = cp.zeros((ncol, nl + 1, NBNDLW), dtype=cp.float32)
    plankbnd = cp.zeros((ncol, NBNDLW), dtype=cp.float32)

    args = (
        np.int32(ncol), np.int32(nl), np.int32(istart),
        dev(pavel).reshape(ncol, nl), dev(tavel).reshape(ncol, nl),
        dev(tz).reshape(ncol, nl + 1),
        dev(np.atleast_1d(np.float32(tbound))),
        dev(semiss).reshape(ncol, NBNDLW),
        dev(coldry).reshape(ncol, nl),
        dev(np.asarray(wkl, dtype=np.float32)).reshape(ncol, MXMOL, nl),
        dev(wbroad).reshape(ncol, nl),
        dev(np.asarray(C["wvn/totplnk"], dtype=np.float32
                       ).ravel(order="F")),
        dev(C["wvn/totplk16"]),
        dev(C["ref/preflog"]), dev(C["ref/tref"]),
        dev(np.asarray(C["ref/chi_mls"], dtype=np.float32
                       ).ravel(order="F")),
        laytrop,
        outs_i["jp"], outs_i["jt"], outs_i["jt1"],
        planklay, planklev, plankbnd,
        outs_f["colh2o"], outs_f["colco2"], outs_f["colo3"],
        outs_f["coln2o"], outs_f["colco"], outs_f["colch4"],
        outs_f["colo2"], outs_f["colbrd"],
        outs_f["fac00"], outs_f["fac01"], outs_f["fac10"],
        outs_f["fac11"],
        outs_f["rat_h2oco2"], outs_f["rat_h2oco2_1"],
        outs_f["rat_h2oo3"], outs_f["rat_h2oo3_1"],
        outs_f["rat_h2on2o"], outs_f["rat_h2on2o_1"],
        outs_f["rat_h2och4"], outs_f["rat_h2och4_1"],
        outs_f["rat_n2oco2"], outs_f["rat_n2oco2_1"],
        outs_f["rat_o3co2"], outs_f["rat_o3co2_1"],
        outs_f["selffac"], outs_f["selffrac"], outs_i["indself"],
        outs_f["forfac"], outs_f["forfrac"], outs_i["indfor"],
        outs_f["minorfrac"], outs_f["scaleminor"],
        outs_f["scaleminorn2"], outs_i["indminor"],
    )
    _gpu_kernel("rlw_setcoef")((1,), (64,), args)
    cp.cuda.runtime.deviceSynchronize()

    out = {"laytrop": int(cp.asnumpy(laytrop)[0]),
           "planklay": cp.asnumpy(planklay)[0],
           "planklev": cp.asnumpy(planklev)[0],
           "plankbnd": cp.asnumpy(plankbnd)[0]}
    for k, v in outs_f.items():
        out[k] = cp.asnumpy(v)[0]
    for k, v in outs_i.items():
        out[k] = cp.asnumpy(v)[0]
    return out


def gpu_cldprmc(nlayers, inflag, iceflag, liqflag, cldfmc, ciwpmc, clwpmc,
                cswpmc, reicmc, relqmc, resnmc, taucmc, C, ncol=1):
    """Device cldprmc; returns (ncbands, taucmc) like the NumPy port."""
    import cupy as cp
    gpu_preflight()
    nl = int(nlayers)

    def dev(a, dt=np.float32):
        return cp.asarray(np.ascontiguousarray(np.asarray(a, dtype=dt)))

    taucmc_d = dev(taucmc)
    ncb_flag = cp.zeros(ncol, dtype=cp.int32)
    err_flag = cp.zeros(1, dtype=cp.int32)
    total = ncol * NGPTLW
    threads = 128
    blocks = (total + threads - 1) // threads
    _gpu_kernel("rlw_cldprmc")(
        (blocks,), (threads,),
        (np.int32(ncol), np.int32(nl), np.int32(inflag),
         np.int32(iceflag), np.int32(liqflag),
         dev(cldfmc), dev(ciwpmc), dev(clwpmc), dev(cswpmc),
         dev(reicmc), dev(relqmc), dev(resnmc),
         dev(np.asarray(C["cld/absice1"], np.float32).ravel(order="F")),
         dev(np.asarray(C["cld/absice2"], np.float32).ravel(order="F")),
         dev(np.asarray(C["cld/absice3"], np.float32).ravel(order="F")),
         dev(C["cld/absice0"]),
         dev(np.asarray(C["cld/absliq1"], np.float32).ravel(order="F")),
         np.float32(C["cld/absliq0"]),
         dev(C["wvn/ngb"], np.int32),
         taucmc_d, ncb_flag, err_flag))
    cp.cuda.runtime.deviceSynchronize()
    err = int(cp.asnumpy(err_flag)[0])
    if err:
        raise ValueError(f"rlw_cldprmc device abort, code {err} "
                         "(bounds violation, mirrors the Fortran stop)")
    hit = int(cp.asnumpy(ncb_flag)[0])
    if hit:
        ncbands = 5 if iceflag == 1 else 16
    else:
        ncbands = 1
    return ncbands, cp.asnumpy(taucmc_d)


def gpu_rtrnmc(nlayers, istart, iend, iout, pz, semiss, ncbands, cldfmc,
               taucmc, planklay, planklev, plankbnd, pwvcm, fracs, taut,
               C, ncol=1):
    """Device rtrnmc pipeline; returns the NumPy-rtrnmc dict."""
    import cupy as cp
    gpu_preflight()
    nl = int(nlayers)
    assert nl <= MAX_RADIATION_LAYERS, "rlw_rtrn_march RLW_MAXLAY"
    if (istart, iend, iout) != (1, 16, 0):
        raise NotImplementedError("istart/iend/iout fixed to 1/16/0")

    def dev(a, dt=np.float32):
        return cp.asarray(np.ascontiguousarray(np.asarray(a, dtype=dt)))

    cldfmc_d = dev(cldfmc)
    taucmc_d = dev(taucmc)
    secdiff = cp.zeros((ncol, NBNDLW), dtype=cp.float32)
    _gpu_kernel("rlw_rtrn_secdiff")(
        (1,), (64,),
        (np.int32(ncol), dev(np.atleast_1d(np.float32(pwvcm))),
         dev(A0), dev(A1), dev(A2), secdiff))

    odcld = cp.zeros((ncol, NGPTLW, nl), dtype=cp.float32)
    efclfrac = cp.zeros((ncol, NGPTLW, nl), dtype=cp.float32)
    icldlyr = cp.zeros((ncol, nl), dtype=cp.int32)
    total = ncol * nl
    _gpu_kernel("rlw_rtrn_prol")(
        ((total + 127) // 128,), (128,),
        (np.int32(ncol), np.int32(nl), cldfmc_d, taucmc_d, secdiff,
         dev(C["wvn/ngb"], np.int32), odcld, efclfrac, icldlyr))

    prof = lambda: cp.zeros((ncol, NGPTLW, nl), dtype=cp.float32)
    radld_p, radclrd_p = prof(), prof()
    radlu_p, radclru_p = prof(), prof()
    iclddn_p = cp.zeros((ncol, NGPTLW, nl), dtype=cp.uint8)
    radlu_sfc = cp.zeros((ncol, NGPTLW), dtype=cp.float32)
    radclru_sfc = cp.zeros((ncol, NGPTLW), dtype=cp.float32)
    total = ncol * NGPTLW
    _gpu_kernel("rlw_rtrn_march")(
        ((total + 127) // 128,), (128,),
        (np.int32(ncol), np.int32(nl), cldfmc_d, odcld, efclfrac,
         icldlyr, secdiff, dev(semiss).reshape(ncol, NBNDLW),
         dev(planklay), dev(planklev), dev(plankbnd),
         dev(fracs), dev(taut),
         dev(C["tbl/tau_tbl"]), dev(C["tbl/exp_tbl"]),
         dev(C["tbl/tfn_tbl"]), np.float32(C["tbl/bpade"]),
         dev(C["wvn/ngb"], np.int32),
         radld_p, radclrd_p, radlu_p, radclru_p, iclddn_p,
         radlu_sfc, radclru_sfc))

    totuflux = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    totdflux = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    totuclfl = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    totdclfl = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    total = ncol * (nl + 1)
    _gpu_kernel("rlw_rtrn_accum")(
        ((total + 127) // 128,), (128,),
        (np.int32(ncol), np.int32(nl), radld_p, radclrd_p, radlu_p,
         radclru_p, iclddn_p, radlu_sfc, radclru_sfc,
         dev(NGS, np.int32), dev(C["wvn/delwave"]), WTDIFF,
         totuflux, totdflux, totuclfl, totdclfl))

    fnet = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    fnetc = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    htr = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    htrc = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    _gpu_kernel("rlw_rtrn_final")(
        (1,), (64,),
        (np.int32(ncol), np.int32(nl), dev(pz).reshape(ncol, nl + 1),
         np.float32(C["con/fluxfac"]), np.float32(C["con/heatfac"]),
         totuflux, totdflux, totuclfl, totdclfl, fnet, fnetc, htr, htrc))
    cp.cuda.runtime.deviceSynchronize()

    return {
        "totuflux": cp.asnumpy(totuflux)[0],
        "totdflux": cp.asnumpy(totdflux)[0],
        "fnet": cp.asnumpy(fnet)[0], "htr": cp.asnumpy(htr)[0],
        "totuclfl": cp.asnumpy(totuclfl)[0],
        "totdclfl": cp.asnumpy(totdclfl)[0],
        "fnetc": cp.asnumpy(fnetc)[0], "htrc": cp.asnumpy(htrc)[0],
    }


def gpu_inatm(iplon, nlay, icld, iaer, play, plev, tlay, tlev, tsfc,
              h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, cfc11vmr,
              cfc12vmr, cfc22vmr, ccl4vmr, emis, inflglw, iceflglw,
              liqflglw, cldfmcl, taucmcl, ciwpmcl, clwpmcl, cswpmcl,
              reicmcl, relqmcl, resnmcl, tauaer, C):
    """inatm with the derived quantities (coldry/wbrodl/wkl/wx/pwvcm)
    computed by the rlw_inatm device kernel; the pure data movement
    (slot installs, mcica copies) carries no arithmetic and stays host-
    side.  Same outputs as the NumPy inatm."""
    import cupy as cp
    gpu_preflight()
    if icld < 1:
        raise ValueError(
            "icld < 1: WRF never drives this path and the Fortran would "
            "read undefined inflag/iceflag/liqflag (documented divergence)")
    nlayers = int(nlay)
    ip = int(iplon) - 1

    wkl = np.zeros((MXMOL, nlayers), dtype=np.float32)
    wx = np.zeros((MAXXSEC, nlayers), dtype=np.float32)
    wkl[0, :] = h2ovmr[ip, :]
    wkl[1, :] = co2vmr[ip, :]
    wkl[2, :] = o3vmr[ip, :]
    wkl[3, :] = n2ovmr[ip, :]
    wkl[5, :] = ch4vmr[ip, :]
    wkl[6, :] = o2vmr[ip, :]
    wx[0, :] = ccl4vmr[ip, :]
    wx[1, :] = cfc11vmr[ip, :]
    wx[2, :] = cfc12vmr[ip, :]
    wx[3, :] = cfc22vmr[ip, :]

    wkl_d = cp.asarray(np.ascontiguousarray(wkl)).reshape(1, MXMOL,
                                                          nlayers)
    wx_d = cp.asarray(np.ascontiguousarray(wx)).reshape(1, MAXXSEC,
                                                        nlayers)
    plev_d = cp.asarray(np.ascontiguousarray(
        np.asarray(plev[ip, :nlayers + 1], dtype=np.float32))).reshape(
            1, nlayers + 1)
    coldry_d = cp.zeros((1, nlayers), dtype=cp.float32)
    wbrodl_d = cp.zeros((1, nlayers), dtype=cp.float32)
    pwvcm_d = cp.zeros(1, dtype=cp.float32)
    _gpu_kernel("rlw_inatm")(
        (1,), (32,),
        (np.int32(1), np.int32(nlayers), plev_d,
         np.float32(C["con/grav"]), np.float32(C["con/avogad"]),
         wkl_d, wx_d, coldry_d, wbrodl_d, pwvcm_d))
    cp.cuda.runtime.deviceSynchronize()

    taua = np.zeros((nlayers, NBNDLW), dtype=np.float32)
    if iaer >= 1:
        taua[:, :] = tauaer[ip, :nlayers, :]
    out = {
        "nlayers": nlayers,
        "pavel": np.asarray(play[ip, :nlayers], dtype=np.float32),
        "pz": np.asarray(plev[ip, :nlayers + 1], dtype=np.float32),
        "tavel": np.asarray(tlay[ip, :nlayers], dtype=np.float32),
        "tz": np.asarray(tlev[ip, :nlayers + 1], dtype=np.float32),
        "tbound": np.float32(tsfc[ip]),
        "semiss": np.asarray(emis[ip, :], dtype=np.float32),
        "coldry": cp.asnumpy(coldry_d)[0],
        "wkl": cp.asnumpy(wkl_d)[0], "wbrodl": cp.asnumpy(wbrodl_d)[0],
        "wx": cp.asnumpy(wx_d)[0],
        "pwvcm": np.float32(cp.asnumpy(pwvcm_d)[0]),
        "inflag": int(inflglw), "iceflag": int(iceflglw),
        "liqflag": int(liqflglw),
        "cldfmc": np.ascontiguousarray(cldfmcl[:, ip, :]),
        "taucmc": np.ascontiguousarray(taucmcl[:, ip, :]),
        "ciwpmc": np.ascontiguousarray(ciwpmcl[:, ip, :]),
        "clwpmc": np.ascontiguousarray(clwpmcl[:, ip, :]),
        "cswpmc": np.ascontiguousarray(cswpmcl[:, ip, :]),
        "reicmc": np.asarray(reicmcl[ip, :], dtype=np.float32),
        "relqmc": np.asarray(relqmcl[ip, :], dtype=np.float32),
        "resnmc": np.asarray(resnmcl[ip, :], dtype=np.float32),
        "taua": taua,
    }
    return out


def gpu_rrtmg_lw(ncol, nlay, icld, play, plev, tlay, tlev, tsfc, h2ovmr,
                 o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, cfc11vmr,
                 cfc12vmr, cfc22vmr, ccl4vmr, emis, inflglw, iceflglw,
                 liqflglw, cldfmcl, taucmcl, ciwpmcl, clwpmcl, cswpmcl,
                 reicmcl, relqmcl, resnmcl, tauaer, C):
    """Full LW chain with every arithmetic stage on the device
    (inatm-derived, cldprmc, setcoef, all 16 band kernels, taut,
    rtrnmc pipeline).  Same outputs as the NumPy rrtmg_lw."""
    import cupy as cp
    gpu_preflight()
    nl = int(nlay)
    ncol = int(ncol)
    uflx = np.zeros((ncol, nl + 1), dtype=np.float32)
    dflx = np.zeros((ncol, nl + 1), dtype=np.float32)
    uflxc = np.zeros((ncol, nl + 1), dtype=np.float32)
    dflxc = np.zeros((ncol, nl + 1), dtype=np.float32)
    hr = np.zeros((ncol, nl), dtype=np.float32)
    hrc = np.zeros((ncol, nl), dtype=np.float32)

    for iplon in range(1, ncol + 1):
        a = gpu_inatm(iplon, nl, icld, 10, play, plev, tlay, tlev, tsfc,
                      h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,
                      cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, emis,
                      inflglw, iceflglw, liqflglw, cldfmcl, taucmcl,
                      ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl,
                      resnmcl, tauaer, C)
        ncbands, taucmc = gpu_cldprmc(
            a["nlayers"], a["inflag"], a["iceflag"], a["liqflag"],
            a["cldfmc"], a["ciwpmc"], a["clwpmc"], a["cswpmc"],
            a["reicmc"], a["relqmc"], a["resnmc"], a["taucmc"], C)
        sc = gpu_setcoef(a["nlayers"], 1, a["pavel"], a["tavel"],
                         a["tz"], a["tbound"], a["semiss"], a["coldry"],
                         a["wkl"], a["wbrodl"], C)
        st = dict(sc)
        st.update({"pavel": a["pavel"], "wx": a["wx"],
                   "coldry": a["coldry"], "wbroad": a["wbrodl"]})

        taug_d = cp.zeros((1, nl, NGPTLW), dtype=cp.float32)
        fracs_d = cp.zeros((1, nl, NGPTLW), dtype=cp.float32)
        for band in range(1, 17):
            gpu_taugb(band, st, C, taug_d=taug_d, fracs_d=fracs_d)

        taua_d = cp.asarray(np.ascontiguousarray(a["taua"])).reshape(
            1, nl, NBNDLW)
        taut_d = cp.zeros((1, nl, NGPTLW), dtype=cp.float32)
        total = nl * NGPTLW
        _gpu_kernel("rlw_taut")(
            ((total + 127) // 128,), (128,),
            (np.int32(1), np.int32(nl), taug_d, taua_d,
             cp.asarray(np.asarray(C["wvn/ngb"], np.int32)), taut_d))
        cp.cuda.runtime.deviceSynchronize()

        rt = gpu_rtrnmc(a["nlayers"], 1, 16, 0, a["pz"], a["semiss"],
                        ncbands, a["cldfmc"], taucmc,
                        sc["planklay"], sc["planklev"], sc["plankbnd"],
                        a["pwvcm"], cp.asnumpy(fracs_d)[0],
                        cp.asnumpy(taut_d)[0], C)
        uflx[iplon - 1] = rt["totuflux"]
        dflx[iplon - 1] = rt["totdflux"]
        uflxc[iplon - 1] = rt["totuclfl"]
        dflxc[iplon - 1] = rt["totdclfl"]
        hr[iplon - 1] = rt["htr"][:nl]
        hrc[iplon - 1] = rt["htrc"][:nl]

    return {
        "uflx": uflx, "dflx": dflx, "hr": hr,
        "uflxc": uflxc, "dflxc": dflxc, "hrc": hrc,
        "uflxcln": np.zeros((ncol, nl + 1), dtype=np.float32),
        "dflxcln": np.zeros((ncol, nl + 1), dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Section 11 -- batched multi-column CUDA path.
#
# Every kernel above already takes ncol and indexes (col, ...) slabs, so
# batching is HOST PLUMBING AND GRID SIZING ONLY: the same kernels, the
# same compile options (Section 10's NVRTC path), the same per-thread
# statement order.  No kernel was changed for this section.  Each thread's
# arithmetic depends only on its own column's data (there are no
# cross-column reductions anywhere in the LW chain), so per-column results
# are bitwise identical at any batch width and any chunk size; the gates
# in tests/test_rrtmg_lw_cuda.py prove that over the full fixture deck.
#
# Batched input layout (= gpu_rrtmg_lw's argument set, which is already
# ncol-major, processed all-at-once instead of one iplon at a time):
#   play (ncol, nlay), plev (ncol, nlay+1), tlay/tlev likewise,
#   tsfc (ncol,), gas vmr arrays (ncol, nlay), emis (ncol, 16),
#   mcica arrays (NGPTLW, ncol, nlay), reicmcl/relqmcl/resnmcl
#   (ncol, nlay), tauaer (ncol, nlay, 16);
#   icld/inflglw/iceflglw/liqflglw are python ints SHARED by the batch
#   (callers with mixed flags must split the batch by flag tuple).
# Arrays may be numpy or cupy; float32 is the contract dtype.  Non-f32
# inputs are cast with numpy semantics on the host (exactly what the
# per-column wrappers do via np.asarray(..., float32)); a float32->float32
# device copy/transpose is pure data movement and bit-preserving, so cupy
# f32 inputs never leave the device.
# ---------------------------------------------------------------------------

#: Ceiling of the auto-sized columns-per-chunk (#310).  4096 was the
#: hardwired default, sized when the reference card was a 170 SM part: at
#: nlay=74 the peak transient of one chunk is ~1.5 GiB (see
#: lw_batched_vram_bytes), and the rtrn march grid (4096*140 = 573,440
#: threads) is ~2.2x even that part's resident-thread capacity (170 SMs
#: x 1536 = 261,120), so occupancy was already saturated and the excess
#: bought nothing but VRAM.  The default width is now
#: ``batch_column_chunk(NGPTLW, LW_BATCH_COLUMN_CHUNK_CEILING)`` -- the
#: smallest quantum multiple that saturates THIS device, never above
#: this ceiling -- resolved lazily through the module attribute
#: ``LW_BATCH_COLUMN_CHUNK`` so CPU-only imports never touch CUDA.
LW_BATCH_COLUMN_CHUNK_CEILING = 4096

#: The auto width is rounded UP to this quantum and never sinks below it.
BATCH_CHUNK_QUANTUM = 256

#: Cache of the device resident-thread capacity, keyed by device id.
_RESIDENT_THREADS_CACHE: dict = {}


def _device_resident_threads():
    """Resident-thread capacity of the current CUDA device, or ``None``.

    ``None`` (no cupy, no driver, no device) makes the auto width fall
    back to the ceiling -- the exact pre-#310 behaviour -- so CPU-only
    environments and host-side pricing stay deterministic and price the
    conservative (widest) workspace.
    """
    try:
        import cupy as cp
        dev_id = int(cp.cuda.runtime.getDevice())
        cached = _RESIDENT_THREADS_CACHE.get(dev_id)
        if cached is not None:
            return cached
        attrs = cp.cuda.Device(dev_id).attributes
        capacity = (int(attrs["MultiProcessorCount"])
                    * int(attrs["MaxThreadsPerMultiProcessor"]))
    except Exception:
        return None
    if capacity <= 0:
        return None
    _RESIDENT_THREADS_CACHE[dev_id] = capacity
    return capacity


def batch_column_chunk(threads_per_column, ceiling, *,
                       quantum=BATCH_CHUNK_QUANTUM, resident_threads=None):
    """Columns per chunk that saturate the device (#310 narrowing).

    The batched RRTMG engines launch ``threads_per_column`` threads per
    column (LW rtrn march: NGPTLW=140; SW spcvmc: NGPTSW=112) and their
    per-chunk workspace scales linearly with the chunk width, so any
    width beyond the device's resident-thread capacity buys VRAM and no
    occupancy.  Returns the smallest multiple of ``quantum`` whose
    launch covers ``resident_threads`` (the current device's capacity
    when not given), clamped to ``[quantum, ceiling]``.  With no usable
    device the ceiling is returned unchanged.  Chunk width is workspace
    shape only: per-column results are bitwise identical at any width
    (both translation units' contract, proved over the fixture decks).
    """
    ceiling = int(ceiling)
    if resident_threads is None:
        resident_threads = _device_resident_threads()
    if not resident_threads or int(resident_threads) <= 0:
        return ceiling
    quantum = int(quantum)
    need = -(-int(resident_threads) // int(threads_per_column))
    need = -(-need // quantum) * quantum
    return max(quantum, min(ceiling, need))


def __getattr__(name):
    if name == "LW_BATCH_COLUMN_CHUNK":
        return batch_column_chunk(NGPTLW, LW_BATCH_COLUMN_CHUNK_CEILING)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")

#: Every kernel in the LW translation unit (local-frame audit surface).
LW_GPU_KERNEL_NAMES = (
    "rlw_probe", "rlw_inatm", "rlw_cldprmc", "rlw_setcoef", "rlw_taut",
    "rlw_rtrn_secdiff", "rlw_rtrn_prol", "rlw_rtrn_march",
    "rlw_rtrn_accum", "rlw_rtrn_final",
) + tuple("rlw_taugb%d" % b for b in range(1, 17))


def gpu_local_frame_bytes():
    """CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES for every LW kernel.

    A per-thread local frame of F bytes reserves roughly
    F x (max resident threads) = F x 1536 x 170 bytes machine-wide on the
    RTX 5090 at first launch (the KF_KMAX lesson), so frames are audited
    and bounded by tests/test_rrtmg_lw_cuda.py.
    """
    from cupy.cuda import driver
    out = {}
    for name in LW_GPU_KERNEL_NAMES:
        fn = _gpu_kernel(name)
        out[name] = int(driver.funcGetAttribute(
            driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn.ptr))
    return out


def _r512(nbytes):
    """Round up to CuPy's 512-byte pool allocation quantum."""
    return (int(nbytes) + 511) & ~511


def lw_batched_const_bytes(C):
    """Device bytes of the chunk-size-independent constant uploads of one
    gpu_rrtmg_lw_batched call (coefficient tables, lookup tables, band
    pointer tables).  Additive to lw_batched_vram_bytes; ~2-3 MiB."""
    total = 0
    for band in range(1, 17):
        for name in GPU_BAND_TABS[band]:
            total += _r512(
                np.asarray(C["kg%02d/%s" % (band, name)]).size * 4)
        total += _r512(len(GPU_BAND_TABS[band]) * 8)   # device ptr table
    for key in ("ref/chi_mls", "wvn/totplnk", "wvn/totplk16",
                "ref/preflog", "ref/tref", "tbl/tau_tbl", "tbl/exp_tbl",
                "tbl/tfn_tbl", "cld/absice1", "cld/absice2",
                "cld/absice3", "cld/absice0", "cld/absliq1",
                "wvn/delwave"):
        total += _r512(np.asarray(C[key]).size * 4)
    total += _r512(NGPTLW * 4) + _r512(NBNDLW * 4)     # ngb, ngs
    total += 3 * _r512(NBNDLW * 4)                     # a0, a1, a2
    return total


def lw_batched_vram_bytes(ncol_chunk, nlay, ncol_total=None):
    """Peak transient device bytes of ONE chunk of the batched LW chain.

    Derived from exactly the shapes gpu_rrtmg_lw_batched_device allocates,
    as the max over its three allocation high-water stages (upload/inatm/
    cldprmc; setcoef+band kernels; rtrn march), plus the batch-level
    output slabs (priced at ncol_total, default = ncol_chunk).  Every term
    is rounded to CuPy's 512-byte pool quantum, so the estimate tracks
    mempool.used_bytes() tightly (the honesty test in
    tests/test_rrtmg_lw_cuda.py requires estimate >= measured >=
    0.5*estimate).  The per-call constant tables are NOT included: add
    lw_batched_const_bytes(C) (~2-3 MiB) for the full preflight term.
    """
    nc = int(ncol_chunk)
    nl = int(nlay)
    nt = nc if ncol_total is None else int(ncol_total)
    f = 4
    s140 = _r512(nc * NGPTLW * nl * f)       # one (nc, 140, nl) f32 slab
    s_nl = _r512(nc * nl * f)
    s_nl1 = _r512(nc * (nl + 1) * f)
    s_col = _r512(nc * f)
    s_b = _r512(nc * NBNDLW * f)
    s_lnb = _r512(nc * nl * NBNDLW * f)      # planklay / taua
    s_lnb1 = _r512(nc * (nl + 1) * NBNDLW * f)  # planklev
    # batch-level outputs, alive for the whole call
    out_b = 4 * _r512(nt * (nl + 1) * f) + 2 * _r512(nt * nl * f)
    # tiny per-column vectors kept alive across the whole chunk
    tiny = 5 * s_col + _r512(4)              # tsfc pwvcm laytrop ncb + err
    # stage U: all uploads + inatm outputs + cldprmc flags
    stage_u = (5 * s140                      # cldfmc taucmc ciwp clwp cswp
               + _r512(nc * MXMOL * nl * f)  # wkl
               + _r512(nc * MAXXSEC * nl * f)  # wx
               + 2 * s_nl                    # play tlay
               + 2 * s_nl1                   # plev tlev
               + s_b                         # emis
               + 3 * s_nl                    # reicmc relqmc resnmc
               + s_lnb                       # taua
               + 2 * s_nl                    # coldry wbrodl
               + tiny + out_b)
    # stage B: band kernels (ice/liq/snow slabs freed; fs/isv/planks/
    # taug/fracs live; play/tlay/tlev/wkl/wbrodl/coldry freed)
    stage_b = (2 * s140                      # cldfmc taucmc
               + 2 * s140                    # taug fracs
               + _r512(len(GPU_FSLOTS) * nc * nl * f)   # fs
               + _r512(len(GPU_ISLOTS) * nc * nl * f)   # isv
               + _r512(nc * MAXXSEC * nl * f)           # wx
               + s_lnb + s_lnb1 + s_b        # planklay planklev plankbnd
               + s_lnb                       # taua
               + s_nl1 + s_b                 # plev emis
               + tiny + out_b)
    # stage M: rtrn march (the peak): cldfmc + fracs/taut + odcld/efclfrac
    # + 4 radiance profiles + iclddn (u8) + sfc lanes + planks + secdiff
    stage_m = (s140                          # cldfmc
               + 2 * s140                    # fracs taut
               + 2 * s140                    # odcld efclfrac
               + 4 * s140                    # radld radclrd radlu radclru
               + _r512(nc * NGPTLW * nl)     # iclddn_p (uint8)
               + 2 * _r512(nc * NGPTLW * f)  # radlu_sfc radclru_sfc
               + s_lnb + s_lnb1 + s_b        # planklay planklev plankbnd
               + s_b                         # secdiff
               + _r512(nc * nl * f)          # icldlyr
               + s_nl1 + s_b                 # plev emis
               + tiny + out_b)
    return max(stage_u, stage_b, stage_m)


def _lw_dev_chunk(cp, a, rows, cols=None):
    """Contiguous float32 device copy of a leading-axis chunk of ``a``.

    numpy in: host slice -> float32 cast (numpy semantics, same as the
    per-column wrappers) -> upload.  cupy f32 in: device slice + copy
    (pure data movement, bit-preserving).  cupy non-f32: host round-trip
    cast so the conversion uses numpy semantics, never a CuPy ufunc.
    """
    sub = a[rows] if cols is None else a[(rows,) + cols]
    if isinstance(sub, cp.ndarray):
        if sub.dtype == cp.float32:
            return cp.ascontiguousarray(sub)
        sub = cp.asnumpy(sub)
    return cp.asarray(np.ascontiguousarray(np.asarray(sub,
                                                      dtype=np.float32)))


def _lw_dev_mcica(cp, a, c0, c1, nl):
    """(NGPTLW, ncol, nlay) input -> (nc, NGPTLW, nl) contiguous device
    chunk (transpose is data movement only; same bits per column as the
    per-column wrappers' np.ascontiguousarray(a[:, ip, :]))."""
    sub = a[:, c0:c1, :nl]
    if isinstance(sub, cp.ndarray):
        if sub.dtype == cp.float32:
            return cp.ascontiguousarray(sub.transpose(1, 0, 2))
        sub = cp.asnumpy(sub)
    sub = np.asarray(sub, dtype=np.float32)
    return cp.asarray(np.ascontiguousarray(sub.transpose(1, 0, 2)))


def _lw_dev_consts(cp, C):
    """Upload the chunk-independent constants once per batched call.
    Sizes priced by lw_batched_const_bytes -- keep the two in step."""
    def up(a, dt=np.float32, forder=False):
        h = np.asarray(a, dtype=dt)
        if forder:
            h = h.ravel(order="F")
        return cp.asarray(np.ascontiguousarray(h))

    K = {"band": {}}
    for band in range(1, 17):
        K["band"][band] = gpu_band_tabs(band, C)   # (arrays, ptr table)
    K["chi"] = up(C["ref/chi_mls"], forder=True)
    K["totplnk"] = up(C["wvn/totplnk"], forder=True)
    K["totplk16"] = up(C["wvn/totplk16"])
    K["preflog"] = up(C["ref/preflog"])
    K["tref"] = up(C["ref/tref"])
    K["tau_tbl"] = up(C["tbl/tau_tbl"])
    K["exp_tbl"] = up(C["tbl/exp_tbl"])
    K["tfn_tbl"] = up(C["tbl/tfn_tbl"])
    K["absice1"] = up(C["cld/absice1"], forder=True)
    K["absice2"] = up(C["cld/absice2"], forder=True)
    K["absice3"] = up(C["cld/absice3"], forder=True)
    K["absice0"] = up(C["cld/absice0"])
    K["absliq1"] = up(C["cld/absliq1"], forder=True)
    K["ngb"] = up(C["wvn/ngb"], dt=np.int32)
    K["ngs"] = up(NGS, dt=np.int32)
    K["delwave"] = up(C["wvn/delwave"])
    K["a0"], K["a1"], K["a2"] = up(A0), up(A1), up(A2)
    K["oneminus"] = np.float32(C["con/oneminus"])
    K["bpade"] = np.float32(C["tbl/bpade"])
    K["absliq0"] = np.float32(C["cld/absliq0"])
    K["grav"] = np.float32(C["con/grav"])
    K["avogad"] = np.float32(C["con/avogad"])
    K["fluxfac"] = np.float32(C["con/fluxfac"])
    K["heatfac"] = np.float32(C["con/heatfac"])
    return K


def gpu_rrtmg_lw_batched_device(ncol, nlay, icld, play, plev, tlay, tlev,
                                tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr,
                                n2ovmr, o2vmr, cfc11vmr, cfc12vmr,
                                cfc22vmr, ccl4vmr, emis, inflglw,
                                iceflglw, liqflglw, cldfmcl, taucmcl,
                                ciwpmcl, clwpmcl, cswpmcl, reicmcl,
                                relqmcl, resnmcl, tauaer, C,
                                column_chunk=None, _stage_probe=None):
    """Batched full LW chain, device-resident: same kernels, same compile
    path, same launch-side numerics as gpu_rrtmg_lw -- the only changes
    are grid sizes (ncol > 1) and host plumbing.  Inputs numpy or cupy
    (see Section 11 header); returns a dict of cupy float32 arrays
    (uflx/dflx/hr/uflxc/dflxc/hrc + zeroed uflxcln/dflxcln), fetch host
    copies via lw_batched_to_host.  ``column_chunk`` bounds the transient
    VRAM of the internal pipeline (lw_batched_vram_bytes prices it); the
    only mid-chunk host sync is the 4-byte cldprmc error-flag check.
    ``_stage_probe`` is test instrumentation (called at the allocation
    high-water stages); it must not affect results.
    """
    import cupy as cp
    gpu_preflight()
    if icld < 1:
        raise ValueError(
            "icld < 1: WRF never drives this path and the Fortran would "
            "read undefined inflag/iceflag/liqflag (documented divergence)")
    nl = int(nlay)
    ncol = int(ncol)
    assert nl <= MAX_RADIATION_LAYERS, "rlw_rtrn_march RLW_MAXLAY"
    chunk = (int(column_chunk) if column_chunk
             else batch_column_chunk(NGPTLW, LW_BATCH_COLUMN_CHUNK_CEILING))
    if chunk < 1:
        raise ValueError("column_chunk must be >= 1")

    K = _lw_dev_consts(cp, C)
    i32, f32 = np.int32, np.float32

    uflx = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    dflx = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    uflxc = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    dflxc = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    hr = cp.zeros((ncol, nl), dtype=cp.float32)
    hrc = cp.zeros((ncol, nl), dtype=cp.float32)

    fslot = {name: i for i, name in enumerate(GPU_FSLOTS)}
    islot = {name: i for i, name in enumerate(GPU_ISLOTS)}

    for c0 in range(0, ncol, chunk):
        c1 = min(c0 + chunk, ncol)
        nc = c1 - c0
        rows = slice(c0, c1)

        # ---- uploads --------------------------------------------------
        play_d = _lw_dev_chunk(cp, play, rows, (slice(0, nl),))
        plev_d = _lw_dev_chunk(cp, plev, rows, (slice(0, nl + 1),))
        tlay_d = _lw_dev_chunk(cp, tlay, rows, (slice(0, nl),))
        tlev_d = _lw_dev_chunk(cp, tlev, rows, (slice(0, nl + 1),))
        tsfc_d = _lw_dev_chunk(cp, tsfc, rows)
        emis_d = _lw_dev_chunk(cp, emis, rows, (slice(0, NBNDLW),))
        # wkl/wx slot installs are the same pure data movement the
        # per-column gpu_inatm does host-side (unfilled slots stay 0).
        wkl_d = cp.zeros((nc, MXMOL, nl), dtype=cp.float32)
        for slot, arr in ((1, h2ovmr), (2, co2vmr), (3, o3vmr),
                          (4, n2ovmr), (6, ch4vmr), (7, o2vmr)):
            wkl_d[:, slot - 1, :] = _lw_dev_chunk(cp, arr, rows,
                                                  (slice(0, nl),))
        wx_d = cp.zeros((nc, MAXXSEC, nl), dtype=cp.float32)
        for slot, arr in ((1, ccl4vmr), (2, cfc11vmr), (3, cfc12vmr),
                          (4, cfc22vmr)):
            wx_d[:, slot - 1, :] = _lw_dev_chunk(cp, arr, rows,
                                                 (slice(0, nl),))
        cldfmc_d = _lw_dev_mcica(cp, cldfmcl, c0, c1, nl)
        taucmc_d = _lw_dev_mcica(cp, taucmcl, c0, c1, nl)
        ciwpmc_d = _lw_dev_mcica(cp, ciwpmcl, c0, c1, nl)
        clwpmc_d = _lw_dev_mcica(cp, clwpmcl, c0, c1, nl)
        cswpmc_d = _lw_dev_mcica(cp, cswpmcl, c0, c1, nl)
        reicmc_d = _lw_dev_chunk(cp, reicmcl, rows, (slice(0, nl),))
        relqmc_d = _lw_dev_chunk(cp, relqmcl, rows, (slice(0, nl),))
        resnmc_d = _lw_dev_chunk(cp, resnmcl, rows, (slice(0, nl),))
        taua_d = _lw_dev_chunk(cp, tauaer, rows,
                               (slice(0, nl), slice(0, NBNDLW)))

        # ---- inatm derived quantities ----------------------------------
        coldry_d = cp.zeros((nc, nl), dtype=cp.float32)
        wbrodl_d = cp.zeros((nc, nl), dtype=cp.float32)
        pwvcm_d = cp.zeros(nc, dtype=cp.float32)
        _gpu_kernel("rlw_inatm")(
            ((nc + 31) // 32,), (32,),
            (i32(nc), i32(nl), plev_d, K["grav"], K["avogad"],
             wkl_d, wx_d, coldry_d, wbrodl_d, pwvcm_d))

        # ---- cldprmc ----------------------------------------------------
        ncb_flag = cp.zeros(nc, dtype=cp.int32)
        err_flag = cp.zeros(1, dtype=cp.int32)
        total = nc * NGPTLW
        _gpu_kernel("rlw_cldprmc")(
            ((total + 127) // 128,), (128,),
            (i32(nc), i32(nl), i32(inflglw), i32(iceflglw), i32(liqflglw),
             cldfmc_d, ciwpmc_d, clwpmc_d, cswpmc_d,
             reicmc_d, relqmc_d, resnmc_d,
             K["absice1"], K["absice2"], K["absice3"], K["absice0"],
             K["absliq1"], K["absliq0"], K["ngb"],
             taucmc_d, ncb_flag, err_flag))
        if _stage_probe is not None:
            _stage_probe("upload")
        err = int(cp.asnumpy(err_flag)[0])
        if err:
            raise ValueError(f"rlw_cldprmc device abort, code {err} "
                             "(bounds violation, mirrors the Fortran stop)")
        del ciwpmc_d, clwpmc_d, cswpmc_d, reicmc_d, relqmc_d, resnmc_d

        # ---- setcoef, writing straight into the frozen taumol slabs ----
        fs_d = cp.zeros((len(GPU_FSLOTS), nc, nl), dtype=cp.float32)
        isv_d = cp.zeros((len(GPU_ISLOTS), nc, nl), dtype=cp.int32)
        fs_d[fslot["pavel"]] = play_d
        fs_d[fslot["coldry"]] = coldry_d
        laytrop_d = cp.zeros(nc, dtype=cp.int32)
        planklay = cp.zeros((nc, nl, NBNDLW), dtype=cp.float32)
        planklev = cp.zeros((nc, nl + 1, NBNDLW), dtype=cp.float32)
        plankbnd = cp.zeros((nc, NBNDLW), dtype=cp.float32)

        def FV(name):
            return fs_d[fslot[name]]

        def IV(name):
            return isv_d[islot[name]]

        _gpu_kernel("rlw_setcoef")(
            ((nc + 63) // 64,), (64,),
            (i32(nc), i32(nl), i32(1),
             FV("pavel"), tlay_d, tlev_d, tsfc_d, emis_d,
             FV("coldry"), wkl_d, wbrodl_d,
             K["totplnk"], K["totplk16"], K["preflog"], K["tref"],
             K["chi"], laytrop_d,
             IV("jp"), IV("jt"), IV("jt1"),
             planklay, planklev, plankbnd,
             FV("colh2o"), FV("colco2"), FV("colo3"), FV("coln2o"),
             FV("colco"), FV("colch4"), FV("colo2"), FV("colbrd"),
             FV("fac00"), FV("fac01"), FV("fac10"), FV("fac11"),
             FV("rat_h2oco2"), FV("rat_h2oco2_1"),
             FV("rat_h2oo3"), FV("rat_h2oo3_1"),
             FV("rat_h2on2o"), FV("rat_h2on2o_1"),
             FV("rat_h2och4"), FV("rat_h2och4_1"),
             FV("rat_n2oco2"), FV("rat_n2oco2_1"),
             FV("rat_o3co2"), FV("rat_o3co2_1"),
             FV("selffac"), FV("selffrac"), IV("indself"),
             FV("forfac"), FV("forfrac"), IV("indfor"),
             FV("minorfrac"), FV("scaleminor"), FV("scaleminorn2"),
             IV("indminor")))
        del play_d, tlay_d, tlev_d, wkl_d, wbrodl_d, coldry_d

        # ---- taumol: the 16 band kernels --------------------------------
        taug_d = cp.zeros((nc, nl, NGPTLW), dtype=cp.float32)
        fracs_d = cp.zeros((nc, nl, NGPTLW), dtype=cp.float32)
        total = nc * nl
        blocks = (total + 127) // 128
        for band in range(1, 17):
            _gpu_kernel("rlw_taugb%d" % band)(
                (blocks,), (128,),
                (i32(nc), i32(nl), laytrop_d, fs_d, isv_d, wx_d,
                 K["chi"], K["oneminus"], K["band"][band][1],
                 taug_d, fracs_d))
        if _stage_probe is not None:
            _stage_probe("bands")
        del fs_d, isv_d, wx_d

        # ---- taut --------------------------------------------------------
        taut_d = cp.zeros((nc, nl, NGPTLW), dtype=cp.float32)
        total = nc * nl * NGPTLW
        _gpu_kernel("rlw_taut")(
            ((total + 127) // 128,), (128,),
            (i32(nc), i32(nl), taug_d, taua_d, K["ngb"], taut_d))
        del taug_d, taua_d

        # ---- rtrnmc pipeline ----------------------------------------------
        secdiff = cp.zeros((nc, NBNDLW), dtype=cp.float32)
        _gpu_kernel("rlw_rtrn_secdiff")(
            ((nc + 63) // 64,), (64,),
            (i32(nc), pwvcm_d, K["a0"], K["a1"], K["a2"], secdiff))

        odcld = cp.zeros((nc, NGPTLW, nl), dtype=cp.float32)
        efclfrac = cp.zeros((nc, NGPTLW, nl), dtype=cp.float32)
        icldlyr = cp.zeros((nc, nl), dtype=cp.int32)
        total = nc * nl
        _gpu_kernel("rlw_rtrn_prol")(
            ((total + 127) // 128,), (128,),
            (i32(nc), i32(nl), cldfmc_d, taucmc_d, secdiff, K["ngb"],
             odcld, efclfrac, icldlyr))
        del taucmc_d

        radld_p = cp.zeros((nc, NGPTLW, nl), dtype=cp.float32)
        radclrd_p = cp.zeros((nc, NGPTLW, nl), dtype=cp.float32)
        radlu_p = cp.zeros((nc, NGPTLW, nl), dtype=cp.float32)
        radclru_p = cp.zeros((nc, NGPTLW, nl), dtype=cp.float32)
        iclddn_p = cp.zeros((nc, NGPTLW, nl), dtype=cp.uint8)
        radlu_sfc = cp.zeros((nc, NGPTLW), dtype=cp.float32)
        radclru_sfc = cp.zeros((nc, NGPTLW), dtype=cp.float32)
        total = nc * NGPTLW
        _gpu_kernel("rlw_rtrn_march")(
            ((total + 127) // 128,), (128,),
            (i32(nc), i32(nl), cldfmc_d, odcld, efclfrac, icldlyr,
             secdiff, emis_d, planklay, planklev, plankbnd,
             fracs_d, taut_d, K["tau_tbl"], K["exp_tbl"], K["tfn_tbl"],
             K["bpade"], K["ngb"],
             radld_p, radclrd_p, radlu_p, radclru_p, iclddn_p,
             radlu_sfc, radclru_sfc))
        if _stage_probe is not None:
            _stage_probe("march")
        del (cldfmc_d, odcld, efclfrac, fracs_d, taut_d, planklay,
             planklev, plankbnd, secdiff, icldlyr)

        totuflux = cp.zeros((nc, nl + 1), dtype=cp.float32)
        totdflux = cp.zeros((nc, nl + 1), dtype=cp.float32)
        totuclfl = cp.zeros((nc, nl + 1), dtype=cp.float32)
        totdclfl = cp.zeros((nc, nl + 1), dtype=cp.float32)
        total = nc * (nl + 1)
        _gpu_kernel("rlw_rtrn_accum")(
            ((total + 127) // 128,), (128,),
            (i32(nc), i32(nl), radld_p, radclrd_p, radlu_p, radclru_p,
             iclddn_p, radlu_sfc, radclru_sfc,
             K["ngs"], K["delwave"], WTDIFF,
             totuflux, totdflux, totuclfl, totdclfl))
        del (radld_p, radclrd_p, radlu_p, radclru_p, iclddn_p,
             radlu_sfc, radclru_sfc)

        fnet = cp.zeros((nc, nl + 1), dtype=cp.float32)
        fnetc = cp.zeros((nc, nl + 1), dtype=cp.float32)
        htr = cp.zeros((nc, nl + 1), dtype=cp.float32)
        htrc = cp.zeros((nc, nl + 1), dtype=cp.float32)
        _gpu_kernel("rlw_rtrn_final")(
            ((nc + 63) // 64,), (64,),
            (i32(nc), i32(nl), plev_d, K["fluxfac"], K["heatfac"],
             totuflux, totdflux, totuclfl, totdclfl,
             fnet, fnetc, htr, htrc))

        uflx[rows] = totuflux
        dflx[rows] = totdflux
        uflxc[rows] = totuclfl
        dflxc[rows] = totdclfl
        hr[rows] = htr[:, :nl]
        hrc[rows] = htrc[:, :nl]
        # Free every per-chunk transient before the next iteration so no
        # stale slab inflates the next chunk's high-water (keeps
        # lw_batched_vram_bytes an upper bound; data movement only).
        del (totuflux, totdflux, totuclfl, totdclfl, fnet, fnetc, htr,
             htrc)

    cp.cuda.runtime.deviceSynchronize()
    return {
        "uflx": uflx, "dflx": dflx, "hr": hr,
        "uflxc": uflxc, "dflxc": dflxc, "hrc": hrc,
        "uflxcln": cp.zeros((ncol, nl + 1), dtype=cp.float32),
        "dflxcln": cp.zeros((ncol, nl + 1), dtype=cp.float32),
    }


def lw_batched_to_host(out):
    """Host (numpy) copies of a gpu_rrtmg_lw_batched_device result."""
    import cupy as cp
    return {k: cp.asnumpy(v) for k, v in out.items()}


def gpu_rrtmg_lw_batched(*args, **kwargs):
    """gpu_rrtmg_lw_batched_device + host fetch: batched twin of
    gpu_rrtmg_lw (same argument set, ncol processed in device chunks,
    numpy outputs).  See gpu_rrtmg_lw_batched_device."""
    return lw_batched_to_host(gpu_rrtmg_lw_batched_device(*args, **kwargs))
