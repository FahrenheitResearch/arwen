"""WRF v4.6.1 RRTM longwave radiation (``ra_lw_physics=1``).

Transcription of ``phys/module_ra_rrtm.F``'s forecast path: ``O3DATA``
(:3556-3667), ``MM5ATM`` (:3670-4180), ``SETCOEF`` (:4183-4366),
``GASABS``'s lookup-index tail (:6746-6754), ``RTRN`` (:6195-6600) and
the ``RRTM``/``RRTMLWRAD`` drivers (:1731-2298).  The band optical depths
live in :mod:`gpuwm.core.rrtm_taumol`; the coefficient tables in
:mod:`gpuwm.core.rrtm_tables`.

Column layout follows Dudhia's adapter: gpuwm state is bottom-to-top,
WRF's 1-D radiation arrays are top-to-bottom, and the adapter owns that
single packing boundary.  Inside, RRTM's own bottom-to-top ``1..NLAYERS``
indexing is used, extended above the model top by the Cavallo buffer
layers (``NLAYERS = kme + NINT(p_top_hPa/4) - 1``, ``rrtminit`` F:6783).

Documented divergences from a literal transliteration, each of which
guards a Fortran out-of-bounds read rather than changing a defined
result:

* ``RTRN``'s Planck-table subscripts ``INDBOUND``/``INDLEV``/``INDLAY``
  (F:6334, 6342, 6349) index ``TOTPLNK(181,...)`` directly from
  ``T - 159.``  Any level colder than 160 K or warmer than 340 K reads
  outside the table.  WRF guards only the surface value (``TBOUND`` is
  clamped to 339.99, F:3949-3953); level and layer temperatures are
  unguarded.  Here every one of the three is clamped into the 1-based
  range ``[1, 180]`` before :func:`_planck_index` converts it to the
  zero-based row, so the interpolation pair ``(row, row + 1)`` always
  lands inside the 181 rows for any finite temperature.
* ``SETCOEF``'s round-off repair ``IF (JP(LAYSWTCH+1) .LE. 6)``
  (F:4363) reads one past the top layer when ``LAYSWTCH == NLAYERS``.
  The index is clamped to the last layer.
* ``RTRN``'s cloudy-layer scratch (``BBUTOT``, ``ODCLR``, ``ODCLD``,
  ``ABSCLD``, ``EFCLFRAC``) is declared ``kts:ktep1-1`` but written at
  ``LEV`` up to ``NLAYERS`` (F:6323-6324, 6371).  It is unreachable
  because MM5ATM zeroes ``CLDFRAC`` above the model top, so the top
  layer is never cloudy; the arrays here are simply full length.

Nothing else is smoothed over.  In particular the port keeps WRF's
second ozone interpolation off by one layer relative to the first: the
buffer-zone call reverses ``PZ`` such that ``PRLEVH(K) == PZ(K)``
(F:4106-4110) where the model-column call gives ``PRLEVH(K) == PZ(K-1)``.
That is what WRF computes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np

from gpuwm.core.rrtm import RRTM_DELTAP_HPA, rrtm_default_trace_gases
from gpuwm.core.rrtm_tables import (RRTM_BPADE, RRTM_FLUXFAC, RRTM_NBANDS,
                                    RRTM_SECANG, RRTM_WTNUM,
                                    load_rrtm_lw_tables)
from gpuwm.core.rrtm_taumol import RRTM_NGPT, rrtm_taumol


#: MM5ATM's molecular-weight ratios and cloud absorption coefficients,
#: module_ra_rrtm.F:3838-3890.
_AMD = 28.9644
_AMW = 18.0154
_AMDW = 1.607758
_AMDO = 0.603461
_AMDC = 1.805423
_AMDN = 0.658090
_AVGDRO = 6.022e23
_H2OVMR = 5.00e-6
_ABCW = 0.144
_ABICE = 0.0735
_ABRN = 0.330e-3
_ABSN = 2.34e-3
#: RRTM's own upper bound on the Planck lookup, F:3949-3953.
_TBOUND_MAX = 339.99
#: TOTPLNK's first dimension (module_ra_rrtm.F:110).
_TOTPLNK_ROWS = 181
#: Floor for the auto-sized solver chunk (and the fallback when free
#: device memory cannot be queried).  The transfer holds several
#: (ncol, nlayers, 140) arrays at once, so the chunk bounds peak memory --
#: and the choice is a real trade, measured on an RTX 5090 at 53 layers:
#: 4096 columns run in 85 ms with a 1.67 GiB device pool peak, while 256
#: run in 93 ms with about a sixteenth of that.  The work is
#: host-dispatch bound: every chunk re-dispatches the whole ~8k-op CuPy
#: graph, so on a 500x400 nest a fixed 512-column chunk was 391 graph
#: replays and ~3.2M kernel launches per radiation step (~93% of the
#: step wall was host dispatch gap; task #185 profile, node-7 RTX 3090).
#: The adapters therefore auto-size the chunk from free device memory by
#: default (``column_chunk=None``); an explicit ``column_chunk`` pins it.
#: Chunking is per-column arithmetic only, so the answer is byte
#: identical across chunk sizes
#: (tests/test_rrtm_longwave.py::test_column_chunking_does_not_change_the_answer).
#: NOTE these transients are NOT priced by gpuwm.core.preflight, which
#: counts persistent arrays only.
DEFAULT_COLUMN_CHUNK = 512
#: Fraction of the currently reusable device memory (runtime free plus
#: the CuPy pool's free blocks) the auto-sized transfer may claim.
_TRANSIENT_MEMORY_FRACTION = 0.5
#: Measured transient cost of the chunk solve: the 1.67 GiB pool peak at
#: 4096 columns and 53 layers above is 8261 bytes per column-layer.
_TRANSIENT_BYTES_PER_COLUMN_LAYER = 8261


def _resolve_column_chunk(configured, ncol: int, nlayers: int, xp) -> int:
    """Columns per solver pass: the configured pin, or the memory fit.

    ``None`` sizes the chunk so the solve's transient arrays fit inside
    ``_TRANSIENT_MEMORY_FRACTION`` of what the device can hand back
    without growing its footprint, floored at ``DEFAULT_COLUMN_CHUNK``
    and capped at the block's own column count.  Host (NumPy) blocks
    have no device pool to protect and run in one pass.

    NEVER CALL THIS INSIDE CUDA GRAPH CAPTURE: ``cudaMemGetInfo`` is not
    a capturable operation and the runtime refuses it mid-capture.  The
    adapters resolve the auto-size at their first EAGER solve and cache
    it (:meth:`RRTMLongwaveRadiation._column_chunk_for`); a capture that
    arrives cold falls back to the floor, which is byte identical by the
    chunk-invariance guarantee.
    """
    if configured is not None:
        return int(configured)
    cuda = getattr(xp, "cuda", None)
    if cuda is None:
        return max(int(ncol), 1)
    free_device, _total = cuda.runtime.memGetInfo()
    reusable = int(free_device) + int(xp.get_default_memory_pool().free_bytes())
    per_column = _TRANSIENT_BYTES_PER_COLUMN_LAYER * max(int(nlayers), 1)
    fit = int(reusable * _TRANSIENT_MEMORY_FRACTION) // per_column
    return min(int(ncol), max(DEFAULT_COLUMN_CHUNK, int(fit)))


def _stream_is_capturing(cp) -> bool:
    """True when the current stream is inside CUDA graph capture.

    Defensive on purpose: an ``is_capturing`` probe that itself raised
    (an exotic stream object, an old CuPy) must degrade to the eager
    answer rather than take the radiation step down, so any failure here
    reads as "not capturing" and the eager path runs exactly as before.
    """
    try:
        return bool(cp.cuda.get_current_stream().is_capturing())
    except Exception:                                   # pragma: no cover
        return False


#: Constant coefficient tables moved onto the array namespace once per
#: process, keyed by (table id, builder name, namespace, dtype).  Before
#: this cache every solver chunk re-uploaded its tables from pageable
#: host memory -- hundreds of synchronous H2D copies per radiation step.
_DEVICE_CONSTANT_CACHE: dict[tuple, object] = {}


def _device_constant(xp, dtype, key, build):
    """Return ``xp.asarray(build(), dtype)`` cached per process."""
    cache_key = (*key, xp.__name__, np.dtype(dtype).str)
    value = _DEVICE_CONSTANT_CACHE.get(cache_key)
    if value is None:
        value = xp.asarray(build(), dtype=dtype)
        _DEVICE_CONSTANT_CACHE[cache_key] = value
    return value


def _array_namespace(*values):
    """Return CuPy only when at least one input is a CUDA array."""
    if any(hasattr(value, "__cuda_array_interface__") for value in values):
        import cupy as cp
        return cp
    return np


def rrtm_layer_count(nz: int, p_top_pa: float) -> int:
    """``NLAYERS`` for a column of ``nz`` mass levels (``rrtminit`` F:6783).

    WRF computes ``kme + NINT(p_top*0.01/deltap) - 1`` with ``kme = kte+1``
    and ``kte = nz``, i.e. the model layers plus the Cavallo buffer.
    """
    if nz < 3:
        raise ValueError("RRTM longwave needs at least three model layers")
    if not np.isfinite(p_top_pa) or p_top_pa <= 0.0:
        raise ValueError("p_top_pa must be finite and positive")
    buffers = int(np.floor(p_top_pa * 0.01 / RRTM_DELTAP_HPA + 0.5))
    return int(nz) + buffers


@lru_cache(maxsize=1)
def _o3_climatology() -> tuple[np.ndarray, np.ndarray]:
    """``O3WRK``/``PPWRKH`` from O3DATA's summer/winter blend (F:3634-3660)."""
    from gpuwm.core.rrtm_tables import load_rrtm_lw_statics

    statics = load_rrtm_lw_statics()
    o3sum = statics["O3DATA__O3SUM"].astype(np.float32)
    o3win = statics["O3DATA__O3WIN"].astype(np.float32)
    ppsum = statics["O3DATA__PPSUM"].astype(np.float32)
    ppwin = statics["O3DATA__PPWIN"].astype(np.float32)
    ppann = ppsum.copy()
    o3ann = np.empty(31, dtype=np.float32)
    o3ann[0] = np.float32(0.5) * (o3sum[0] + o3win[0])
    for k in range(1, 31):
        o3ann[k] = (o3win[k - 1]
                    + (o3win[k] - o3win[k - 1]) / (ppwin[k] - ppwin[k - 1])
                    * (ppsum[k] - ppwin[k - 1]))
    for k in range(1, 31):
        o3ann[k] = np.float32(0.5) * (o3ann[k] + o3sum[k])
    ppwrkh = np.empty(32, dtype=np.float32)
    ppwrkh[0] = np.float32(1100.0)
    for k in range(1, 31):
        ppwrkh[k] = (ppann[k] + ppann[k - 1]) / np.float32(2.0)
    ppwrkh[31] = np.float32(0.0)
    o3ann.setflags(write=False)
    ppwrkh.setflags(write=False)
    return o3ann, ppwrkh


def o3data_columns(xp, prlevh, nlay: int):
    """``O3DATA``'s pressure-weighted layer ozone (F:3646-3665).

    ``prlevh`` is ``(ncol, nlay+1)`` bottom-to-top level pressure in mb,
    matching the Fortran's reversed ``PRLEVH``.  Returns ``(ncol, nlay)``
    mass mixing ratio.
    """
    dtype = prlevh.dtype
    o3wrk = _device_constant(
        xp, dtype, ("o3data", "o3wrk"), lambda: _o3_climatology()[0])
    ppwrkh = _device_constant(
        xp, dtype, ("o3data", "ppwrkh"), lambda: _o3_climatology()[1])
    bottom = prlevh[:, :nlay, None]
    top = prlevh[:, 1:nlay + 1, None]
    zero = dtype.type(0.0)
    # PB1/PB2/PT1/PT2: the Fortran's "IF (-(P-PH)) >= 0 THEN 0 ELSE P-PH"
    # is exactly a positive part.
    pb1 = xp.maximum(bottom - ppwrkh[None, None, :31], zero)
    pb2 = xp.maximum(bottom - ppwrkh[None, None, 1:32], zero)
    pt1 = xp.maximum(top - ppwrkh[None, None, :31], zero)
    pt2 = xp.maximum(top - ppwrkh[None, None, 1:32], zero)
    weighted = ((pb2 - pb1 - pt2 + pt1) * o3wrk[None, None, :]).sum(axis=2)
    return weighted / (prlevh[:, :nlay] - prlevh[:, 1:nlay + 1])


def _standard_atmosphere_temperature(xp, statics, pz):
    """Interpolate MM5ATM's mean ``TPROF`` to ``PZ`` (F:4069-4093)."""
    dtype = pz.dtype
    pprof = _device_constant(
        xp, dtype, (id(statics), "MM5ATM__PPROF"),
        lambda: statics["MM5ATM__PPROF"])
    tprof = _device_constant(
        xp, dtype, (id(statics), "MM5ATM__TPROF"),
        lambda: statics["MM5ATM__TPROF"])
    nprof = int(pprof.shape[0])
    # klev = LL-1 for the first LL>=2 with PPROF(LL) < PZ; PPROF is
    # monotonically decreasing, so that is the count of entries at or
    # above PZ, floored at 1.  The outer guard sends PZ <= PPROF(last)
    # straight to the last level.
    above = (pprof[None, None, :] >= pz[..., None]).sum(axis=-1)
    klev = xp.clip(above, 1, nprof)
    klev = xp.where(pz > pprof[nprof - 1], klev, nprof).astype(xp.int32)
    last = klev == nprof
    k0 = xp.clip(klev - 1, 0, nprof - 1)
    k1 = xp.clip(klev, 0, nprof - 1)
    vark = tprof[k0]
    vark1 = xp.where(last, vark, tprof[k1])
    # klev == nprof collapses k0 and k1 onto the same table entry, so the
    # Fortran's ELSE branch sets WGHT = 0 without dividing.  Neutralise
    # the denominator rather than divide and discard.
    span = pprof[k1] - pprof[k0]
    span = xp.where(last, dtype.type(1.0), span)
    wght = xp.where(last, dtype.type(0.0), (pz - pprof[k0]) / span)
    return wght * (vark1 - vark) + vark


def mm5atm_columns(xp, tables, *, t, p_mb, dz, qv, qc, qr, qi, qs, qg,
                   cldfra, tw, pw_mb, tsfc, emiss, r, g,
                   co2vmr, n2ovmr, ch4vmr, nlayers):
    """``MM5ATM``: build RRTM's profile arrays (F:3670-4180).

    Layer inputs are ``(ncol, nz)`` TOP-DOWN; level inputs ``(ncol, nz+1)``
    TOP-DOWN; ``tsfc``/``emiss`` are ``(ncol,)``.  Everything returned is
    RRTM's bottom-to-top ordering: layer arrays ``(ncol, nlayers)`` and
    level arrays ``(ncol, nlayers+1)`` indexed ``PZ(0..NLAYERS)``.
    """
    dtype = t.dtype
    real = dtype.type
    ncol, nz = t.shape
    gravit = real(g) * real(100.0)

    #  MID-LAYER VALUES, F:3891-3915.
    qv = xp.maximum(qv, real(1.0e-12))
    ro = p_mb / (real(r) * t) * real(100.0)
    clwp = ro * qc * dz * real(1000.0)
    ciwp = ro * qi * dz * real(1000.0)
    plwp = (ro * qr) ** real(0.75) * dz * real(1000.0)
    piwp = (ro * qs) ** real(0.75) * dz * real(1000.0)

    #  TBOUND, F:3946-3953.  WRF resets a hotter surface to 339.99 and
    #  prints; here it is a silent clamp because there is no per-column
    #  message channel.
    tbound = xp.where(tsfc > real(340.0), real(_TBOUND_MAX), tsfc)

    wkl = xp.zeros((ncol, nlayers, 35), dtype=dtype)
    wx = xp.zeros((ncol, nlayers, 4), dtype=dtype)
    pavel = xp.zeros((ncol, nlayers), dtype=dtype)
    tavel = xp.zeros((ncol, nlayers), dtype=dtype)
    coldry = xp.zeros((ncol, nlayers), dtype=dtype)
    pz = xp.zeros((ncol, nlayers + 1), dtype=dtype)
    tz = xp.zeros((ncol, nlayers + 1), dtype=dtype)

    # F:3956-3979.  PAVEL(L)=p(kte+1-L) etc: a straight reversal.
    pavel[:, :nz] = p_mb[:, ::-1]
    tavel[:, :nz] = t[:, ::-1]
    pz[:, :nz + 1] = pw_mb[:, ::-1]
    tz[:, :nz + 1] = tw[:, ::-1]

    #  Ozone on the model column.  INIRAD/O3DATA take PRLEVH(K)=PZ(K-1),
    #  so the model-layer call sees each layer's own bounding levels.
    o3prof = o3data_columns(xp, pz[:, :nz + 1], nz)

    wkl[:, :nz, 0] = qv[:, ::-1] * real(_AMDW)
    wkl[:, :nz, 1] = real(co2vmr)
    # F:3971-3972: WKL(3,L)=o3(kte+1-L) is overwritten on the next line
    # by WKL(3,L)=o3(L)*amdo.  The first assignment is dead in WRF; the
    # ozone profile is already bottom-to-top.
    wkl[:, :nz, 2] = o3prof * real(_AMDO)
    # N2O/CH4 are filled as constants by F:3917-3918 (n2ovmr/amdn and
    # ch4vmr/amdc) and multiplied straight back by amdn/amdc here.
    wkl[:, :nz, 3] = real(n2ovmr / _AMDN) * real(_AMDN)
    wkl[:, :nz, 5] = real(ch4vmr / _AMDC) * real(_AMDC)
    amm = ((real(1.0) - wkl[:, :nz, 0]) * real(_AMD)
           + wkl[:, :nz, 0] * real(_AMW))
    coldry[:, :nz] = ((pz[:, :nz] - pz[:, 1:nz + 1]) * real(1.0e3)
                      * real(_AVGDRO)
                      / (gravit * amm * (real(1.0) + wkl[:, :nz, 0])))

    #  Cavallo buffer levels, F:4030-4041.
    deltap = real(RRTM_DELTAP_HPA)
    for level in range(nz + 1, nlayers):
        pz[:, level] = pz[:, level - 1] - deltap
        pavel[:, level - 1] = real(0.5) * (pz[:, level] + pz[:, level - 1])
    pz[:, nlayers] = real(0.0)
    pavel[:, nlayers - 1] = real(0.5) * (pz[:, nlayers] + pz[:, nlayers - 1])

    #  Buffer temperatures from the mean standard atmosphere, F:4043-4073.
    varint = _standard_atmosphere_temperature(
        xp, tables.statics, pz[:, 1:nlayers + 1])
    if nlayers > nz:
        offset = (tz[:, nz] - varint[:, nz - 1])[:, None]
        tz[:, nz + 1:nlayers + 1] = varint[:, nz:nlayers] + offset
        for level in range(nz + 1, nlayers + 1):
            tavel[:, level - 1] = real(0.5) * (tz[:, level] + tz[:, level - 1])

    #  Buffer-zone ozone, F:4075-4103.  PZR reverses PZ(1..NLAYERS), so
    #  O3DATA's PRLEVH(K) lands on PZ(K) -- one layer up from the model
    #  column's PZ(K-1).  Reproduced as WRF computes it.
    o3prof2 = xp.empty((ncol, nlayers), dtype=dtype)
    o3prof2[:, :nlayers - 1] = o3data_columns(
        xp, pz[:, 1:nlayers + 1], nlayers - 1)
    o3prof2[:, nlayers - 1] = real(6.135e-6)
    if nz != nlayers:
        wkl[:, :, 2] = o3prof2 * real(_AMDO)
        wkl[:, nz:, 0] = real(_H2OVMR)
        wkl[:, nz:, 1] = real(co2vmr)
        wkl[:, nz:, 3] = wkl[:, nz - 1:nz, 3]
        wkl[:, nz:, 5] = wkl[:, nz - 1:nz, 5]
        amm_buf = ((real(1.0) - wkl[:, nz:, 0]) * real(_AMD)
                   + wkl[:, nz:, 0] * real(_AMW))
        coldry[:, nz:] = ((pz[:, nz:nlayers] - pz[:, nz + 1:nlayers + 1])
                          * real(1.0e3) * real(_AVGDRO)
                          / (gravit * amm_buf
                             * (real(1.0) + wkl[:, nz:, 0])))
        wx[:, nz:, 1] = wx[:, nz - 1:nz, 1]
        wx[:, nz:, 2] = wx[:, nz - 1:nz, 2]

    #  vmr -> molec/cm2, F:4105-4114.  NMOL=6 and the IXINDX cross-section
    #  remap (IXINDX = 0,2,3,0) both come from the packaged statics.
    wkl[:, :, :6] = wkl[:, :, :6] * coldry[:, :, None]
    ixindx = tables.statics["IXINDX"]
    for ix in range(4):
        target = int(ixindx[ix])
        if target != 0:
            wx[:, :, target - 1] = (coldry * wx[:, :, ix]
                                    * real(1.0e-20))

    #  Spectral surface emissivity, F:4121-4123: one land-use value for
    #  every band.
    semiss = xp.broadcast_to(emiss[:, None], (ncol, RRTM_NBANDS))

    #  Cloud optical depth and fraction, F:4133-4145.
    taucloud = xp.zeros((ncol, nlayers), dtype=dtype)
    tau_model = (real(_ABCW) * clwp[:, ::-1] + real(_ABICE) * ciwp[:, ::-1]
                 + real(_ABRN) * plwp[:, ::-1] + real(_ABSN) * piwp[:, ::-1])
    taucloud[:, :nz] = tau_model
    cldfrac = xp.zeros((ncol, nlayers), dtype=dtype)
    cldfrac[:, :nz] = xp.where(tau_model > real(0.01), real(1.0),
                               cldfra[:, ::-1])
    return {
        "pavel": pavel, "tavel": tavel, "pz": pz, "tz": tz,
        "cldfrac": cldfrac, "taucloud": taucloud, "coldry": coldry,
        "wkl": wkl, "wx": wx, "tbound": tbound, "semiss": semiss,
    }


def setcoef_columns(xp, tables, *, pavel, tavel, coldry, wkl):
    """``SETCOEF``: interpolation indices and column amounts (F:4183-4366)."""
    dtype = pavel.dtype
    real = dtype.type
    ncol, nlayers = pavel.shape
    preflog = _device_constant(
        xp, dtype, (id(tables), "PREFLOG"),
        lambda: tables.statics["PREFLOG"])
    tref = _device_constant(
        xp, dtype, (id(tables), "TREF"), lambda: tables.statics["TREF"])
    stpfac = real(296.0 / 1013.0)

    plog = xp.log(pavel)
    jp = xp.clip(xp.trunc(real(36.0) - real(5.0) * (plog + real(0.04))
                          ).astype(xp.int32), 1, 58)
    fp = real(5.0) * (preflog[jp - 1] - plog)
    jt = xp.clip(xp.trunc(real(3.0) + (tavel - tref[jp - 1]) / real(15.0)
                          ).astype(xp.int32), 1, 4)
    ft = ((tavel - tref[jp - 1]) / real(15.0)) - (jt - 3).astype(dtype)
    jt1 = xp.clip(xp.trunc(real(3.0) + (tavel - tref[jp]) / real(15.0)
                           ).astype(xp.int32), 1, 4)
    ft1 = ((tavel - tref[jp]) / real(15.0)) - (jt1 - 3).astype(dtype)

    water = wkl[:, :, 0] / coldry
    scalefac = pavel * stpfac / tavel

    #  LAYTROP/LAYSWTCH/LAYLOW are COUNTS over the whole column
    #  (F:4291-4295), which the branch bounds "DO LAY = 1, LAYTROP" then
    #  treat as a contiguous split.
    troposphere = plog > real(4.56)
    laytrop = troposphere.sum(axis=1).astype(xp.int32)
    layswtch = (troposphere & (plog > real(5.76))).sum(axis=1).astype(xp.int32)
    laylow = (troposphere & (plog >= real(6.62))).sum(axis=1).astype(xp.int32)

    forfac = scalefac / (real(1.0) + water)
    selffac = water * forfac
    factor = (tavel - real(188.0)) / real(7.2)
    indself = xp.clip(xp.trunc(factor).astype(xp.int32) - 7, 1, 9)
    selffrac = factor - (indself + 7).astype(dtype)
    #  Above LAYTROP the self-continuum terms are not evaluated at all
    #  (F:4327-4344 omits them), so their values there are irrelevant;
    #  TAUMOL's masks discard them.

    scale = real(1.0e-20)
    colh2o = scale * wkl[:, :, 0]
    colco2 = scale * wkl[:, :, 1]
    colo3 = scale * wkl[:, :, 2]
    coln2o = scale * wkl[:, :, 3]
    colch4 = scale * wkl[:, :, 5]
    colo2 = scale * wkl[:, :, 6]
    floor = real(1.0e-32) * coldry
    colco2 = xp.where(colco2 == real(0.0), floor, colco2)
    coln2o = xp.where(coln2o == real(0.0), floor, coln2o)
    colch4 = xp.where(colch4 == real(0.0), floor, colch4)
    co2reg = real(3.55e-24) * coldry
    co2mult = ((colco2 - co2reg) * real(272.63)
               * xp.exp(real(-1919.4) / tavel)
               / (real(8.7604e-4) * tavel))

    compfp = real(1.0) - fp
    fac10 = compfp * ft
    fac00 = compfp * (real(1.0) - ft)
    fac11 = fp * ft1
    fac01 = fp * (real(1.0) - ft1)

    #  F:4361-4365.  LAYLOW=0 becomes 1, and a round-off repair bumps
    #  LAYSWTCH when the next layer is still below the ~300 mb switch.
    #  JP(LAYSWTCH+1) reads one past the top when LAYSWTCH == NLAYERS;
    #  the index is clamped (see the module docstring).
    laylow = xp.where(laylow == 0, 1, laylow)
    probe = xp.clip(layswtch, 0, nlayers - 1)
    jp_next = xp.take_along_axis(jp, probe[:, None], axis=1)[:, 0]
    layswtch = xp.where(jp_next <= 6, layswtch + 1, layswtch)

    return {
        "colh2o": colh2o, "colco2": colco2, "colo3": colo3,
        "coln2o": coln2o, "colch4": colch4, "colo2": colo2,
        "co2mult": co2mult, "fac00": fac00, "fac01": fac01,
        "fac10": fac10, "fac11": fac11, "forfac": forfac,
        "selffac": selffac, "selffrac": selffrac, "jp": jp, "jt": jt,
        "jt1": jt1, "indself": indself, "laytrop": laytrop,
        "layswtch": layswtch, "laylow": laylow,
    }


def _planck_index(xp, temperature, dtype):
    """``INT(T - 159.)`` and its fraction, as a ZERO-BASED TOTPLNK row.

    Fortran's ``INDBOUND = INT(T - 159.)`` (F:6334, 6342, 6349) is a
    1-based subscript into ``TOTPLNK(181,16)``, whose first row is 160 K,
    so ``TOTPLNK(i)`` is the row for ``159 + i`` kelvin.  The zero-based
    row for the same temperature is ``i - 1``, and that subtraction is
    the whole difference between this function and the Fortran: without
    it every Planck evaluation reads one kelvin too warm (about +1.3% on
    GLW at 288 K) and a clamped 339.99 K surface runs off the end of the
    array.  The sibling ``TAU``/``TF``/``TRANS`` tables are declared
    ``0:5000`` in the Fortran and are indexed here without any shift;
    the two conventions really do sit side by side in the source.

    The clamp is applied to the 1-based value so the returned row pair
    ``(row, row + 1)`` stays inside ``[0, 180]``.  See the module
    docstring for why the clamp exists at all.
    """
    index = xp.trunc(temperature - dtype.type(159.0)).astype(xp.int32)
    frac = temperature - xp.trunc(temperature)
    return xp.clip(index, 1, _TOTPLNK_ROWS - 1) - 1, frac


def _planck_bands(totplnk, delwave, row, frac):
    """``DELWAVE*(TOTPLNK(row) + frac*(TOTPLNK(row+1)-TOTPLNK(row)))``.

    F:6371-6395, with ``row`` the zero-based row from
    :func:`_planck_index`.  Hoisted out of :func:`rtrn_columns` so the
    Stefan-Boltzmann table check in the test suite exercises the shipped
    arithmetic rather than a copy of it.
    """
    low = totplnk[row]
    high = totplnk[row + 1]
    return delwave * (low + frac[..., None] * (high - low))


def rtrn_columns(xp, tables, *, tavel, pz, tz, cldfrac, taucloud, itr,
                 pfrac, tbound, semiss):
    """``RTRN``: the two-stream diffusivity-angle transfer (F:6195-6600).

    Returns ``(totdflux, totuflux, htr, htrc)`` with level arrays
    ``(ncol, nlayers+1)`` and heating rates in K/day.
    """
    dtype = tavel.dtype
    real = dtype.type
    ncol, nlayers = tavel.shape
    totplnk = _device_constant(
        xp, dtype, (id(tables), "TOTPLNK"),
        lambda: tables.statics["TOTPLNK"].reshape(
            (_TOTPLNK_ROWS, RRTM_NBANDS), order="F"))
    delwave = _device_constant(
        xp, dtype, (id(tables), "DELWAVE"),
        lambda: tables.statics["DELWAVE"])
    heatfac = real(tables.statics["HEATFAC"][0])
    ngb0 = _device_constant(
        xp, np.int32, (id(tables), "NGB0"),
        lambda: np.asarray(tables.statics["NGB"], dtype=np.int32) - 1)
    tau_tab = _device_constant(
        xp, dtype, (id(tables), "TAU"), lambda: tables.tau)
    tf_tab = _device_constant(
        xp, dtype, (id(tables), "TF"), lambda: tables.tf)
    trans_tab = _device_constant(
        xp, dtype, (id(tables), "TRANS"), lambda: tables.trans)

    indbound, tbndfrac = _planck_index(xp, tbound, dtype)
    indlev, tlevfrac = _planck_index(xp, tz, dtype)
    indlay, tlayfrac = _planck_index(xp, tavel, dtype)

    #  Integrated Planck functions, F:6371-6395.
    def planck(row, frac):
        return _planck_bands(totplnk, delwave, row, frac)

    plankbnd = planck(indbound, tbndfrac)          # (ncol, nbands)
    plvl = planck(indlev, tlevfrac)                # (ncol, nlayers+1, nbands)
    play = planck(indlay, tlayfrac)                # (ncol, nlayers, nbands)
    plnkemit = semiss * plankbnd

    #  Cloud optics, F:6350-6360.
    odcld = RRTM_SECANG.astype(dtype) * taucloud
    abscld = real(1.0) - xp.exp(-odcld)
    efclfrac = abscld * cldfrac
    icldlyr = cldfrac > real(0.0)

    odclr = tau_tab[itr]
    tauf = tf_tab[itr]
    abss = real(1.0) - trans_tab[itr]

    semis = semiss[:, ngb0]
    raduemit = pfrac[:, 0, :] * plnkemit[:, ngb0]
    bglev = pfrac[:, nlayers - 1, :] * plvl[:, nlayers, :][:, ngb0]

    radld = xp.zeros((ncol, RRTM_NGPT), dtype=dtype)
    radclrd = xp.zeros((ncol, RRTM_NGPT), dtype=dtype)
    totdflux = xp.zeros((ncol, nlayers + 1), dtype=dtype)
    totdclfl = xp.zeros((ncol, nlayers + 1), dtype=dtype)
    bbu = xp.zeros((ncol, nlayers, RRTM_NGPT), dtype=dtype)
    bbutot = xp.zeros((ncol, nlayers, RRTM_NGPT), dtype=dtype)
    atot = xp.zeros((ncol, nlayers, RRTM_NGPT), dtype=dtype)
    iclddn = xp.zeros((ncol,), dtype=bool)

    #  DOWNWARD RADIATIVE TRANSFER, F:6416-6499.
    for lev in range(nlayers, 0, -1):
        k = lev - 1
        cloudy = icldlyr[:, k]
        iclddn = iclddn | cloudy
        cloudy3 = cloudy[:, None]
        bglay = pfrac[:, k, :] * play[:, k, :][:, ngb0]
        delbgup = bglev - bglay
        layer_tauf = tauf[:, k, :]
        bbu[:, k, :] = bglay + layer_tauf * delbgup
        bglev = pfrac[:, k, :] * plvl[:, lev - 1, :][:, ngb0]
        delbgdn = bglev - bglay
        bbd = bglay + layer_tauf * delbgdn
        layer_abss = abss[:, k, :]

        odsm = odclr[:, k, :] + odcld[:, k, None]
        factot = odsm / (RRTM_BPADE.astype(dtype) + odsm)
        bbutot[:, k, :] = bglay + factot * delbgup
        bbdlevd = bglay + factot * delbgdn
        layer_atot = (layer_abss + abscld[:, k, None]
                      - layer_abss * abscld[:, k, None])
        atot[:, k, :] = layer_atot
        gassrc = bbd * layer_abss

        radld_cloudy = (radld - radld * (layer_abss + efclfrac[:, k, None]
                                         * (real(1.0) - layer_abss))
                        + gassrc + cldfrac[:, k, None]
                        * (bbdlevd * layer_atot - gassrc))
        radld_clear = radld + (bbd - radld) * layer_abss
        radclrd_step = radclrd + (bbd - radclrd) * layer_abss
        radld = xp.where(cloudy3, radld_cloudy, radld_clear)
        #  In a clear layer the clear stream only separates once a cloud
        #  has been crossed; until then it tracks the total stream
        #  (F:6485-6492).
        radclrd = xp.where(cloudy3 | iclddn[:, None], radclrd_step, radld)
        totdflux[:, lev - 1] = radld.sum(axis=1) * RRTM_WTNUM.astype(dtype)
        totdclfl[:, lev - 1] = radclrd.sum(axis=1) * RRTM_WTNUM.astype(dtype)

    #  Surface emission and reflection, F:6506-6520.
    radlu = raduemit + (real(1.0) - semis) * radld
    radclru = raduemit + (real(1.0) - semis) * radclrd
    totuflux = xp.zeros((ncol, nlayers + 1), dtype=dtype)
    totuclfl = xp.zeros((ncol, nlayers + 1), dtype=dtype)
    totuflux[:, 0] = radlu.sum(axis=1) * RRTM_WTNUM.astype(dtype)
    totuclfl[:, 0] = radclru.sum(axis=1) * RRTM_WTNUM.astype(dtype)

    #  UPWARD RADIATIVE TRANSFER, F:6527-6574.
    for lev in range(1, nlayers + 1):
        k = lev - 1
        cloudy3 = icldlyr[:, k][:, None]
        layer_abss = abss[:, k, :]
        gassrc = bbu[:, k, :] * layer_abss
        radlu_cloudy = (radlu - radlu * (layer_abss + efclfrac[:, k, None]
                                         * (real(1.0) - layer_abss))
                        + gassrc + cldfrac[:, k, None]
                        * (bbutot[:, k, :] * atot[:, k, :] - gassrc))
        radlu_clear = radlu + (bbu[:, k, :] - radlu) * layer_abss
        radlu = xp.where(cloudy3, radlu_cloudy, radlu_clear)
        radclru = radclru + (bbu[:, k, :] - radclru) * layer_abss
        totuflux[:, lev] = radlu.sum(axis=1) * RRTM_WTNUM.astype(dtype)
        totuclfl[:, lev] = radclru.sum(axis=1) * RRTM_WTNUM.astype(dtype)

    #  Radiances to fluxes and heating rates, F:6580-6596.
    fluxfac = RRTM_FLUXFAC.astype(dtype)
    totuflux = totuflux * fluxfac
    totdflux = totdflux * fluxfac
    totuclfl = totuclfl * fluxfac
    totdclfl = totdclfl * fluxfac
    fnet = totuflux - totdflux
    fnetc = totuclfl - totdclfl
    dp = pz[:, :nlayers] - pz[:, 1:nlayers + 1]
    htr = xp.zeros((ncol, nlayers + 1), dtype=dtype)
    htrc = xp.zeros((ncol, nlayers + 1), dtype=dtype)
    htr[:, :nlayers] = heatfac * (fnet[:, :nlayers]
                                  - fnet[:, 1:nlayers + 1]) / dp
    htrc[:, :nlayers] = heatfac * (fnetc[:, :nlayers]
                                   - fnetc[:, 1:nlayers + 1]) / dp
    return totdflux, totuflux, htr, htrc


def rrtm_longwave_columns(*, t, p_mb, dz, qv, qc, qr, qi, qs, qg, cldfra,
                          tw, pw_mb, tsfc, emiss, r, g, co2vmr, n2ovmr,
                          ch4vmr, nlayers, tables=None):
    """``RRTM``'s per-column driver (F:2059-2298) for a block of columns.

    Layer inputs are ``(ncol, nz)`` TOP-DOWN, level inputs
    ``(ncol, nz+1)`` TOP-DOWN, pressures in mb.  Returns a dict with
    ``tten``/``ttenc`` (``(ncol, nz)`` BOTTOM-UP temperature tendency,
    K s-1), ``glw`` and ``olr`` (``(ncol,)``, W m-2).
    """
    if tables is None:
        tables = load_rrtm_lw_tables()
    xp = _array_namespace(t, p_mb, qv)
    dtype = t.dtype
    if dtype.kind != "f":
        raise TypeError("RRTM longwave inputs must be floating point")
    ncol, nz = t.shape
    profile = mm5atm_columns(
        xp, tables, t=t, p_mb=p_mb, dz=dz, qv=qv, qc=qc, qr=qr, qi=qi,
        qs=qs, qg=qg, cldfra=cldfra, tw=tw, pw_mb=pw_mb, tsfc=tsfc,
        emiss=emiss, r=r, g=g, co2vmr=co2vmr, n2ovmr=n2ovmr,
        ch4vmr=ch4vmr, nlayers=nlayers)
    coef = setcoef_columns(
        xp, tables, pavel=profile["pavel"], tavel=profile["tavel"],
        coldry=profile["coldry"], wkl=profile["wkl"])
    taug, pfrac = rrtm_taumol(
        coldry=profile["coldry"], wx=profile["wx"], tables=tables, **coef)

    #  GASABS's Pade lookup index, F:6746-6754.
    real = dtype.type
    odepth = RRTM_SECANG.astype(dtype) * taug
    tff = odepth / (RRTM_BPADE.astype(dtype) + odepth)
    tff = xp.where(odepth <= real(0.0), real(0.0), tff)
    itr = xp.clip(xp.trunc(real(5.0e3) * tff + real(0.5)).astype(xp.int32),
                  0, 5000)

    totdflux, totuflux, htr, htrc = rtrn_columns(
        xp, tables, tavel=profile["tavel"], pz=profile["pz"],
        tz=profile["tz"], cldfrac=profile["cldfrac"],
        taucloud=profile["taucloud"], itr=itr, pfrac=pfrac,
        tbound=profile["tbound"], semiss=profile["semiss"])

    #  F:2288-2297.  TTEN(K)=HTR(kte-K)/86400 in the top-down 1-D frame is
    #  HTR(k-1) for bottom-up level k, so the model-column heating is just
    #  the first nz entries of HTR.  GLW is the surface downward flux and
    #  OLR is TOTUFLUX at the model top -- not at NLAYERS.
    day = real(86400.0)
    return {
        "tten": htr[:, :nz] / day,
        "ttenc": htrc[:, :nz] / day,
        "glw": totdflux[:, 0],
        "olr": totuflux[:, nz],
        # Whole-column diagnostics, buffer layers included.  HTR is
        # K/day exactly as RTRN leaves it, so a caller can check the
        # flux-divergence identity the scheme is built on.
        "htr": htr,
        "totdflux": totdflux,
        "totuflux": totuflux,
        "pz": profile["pz"],
    }


def _t8w_columns(xp, t3d, z_at_w, fnm, fnp):
    """Device twin of ``rrtmg_legacy._t8w_columns`` (phy_prep's ``t8w``).

    Same transcription, same source seam; this one stays on whatever
    array namespace the caller is using instead of forcing a host copy.
    """
    dtype = t3d.dtype
    real = dtype.type
    ncol, nz = t3d.shape
    z = real(0.5) * (z_at_w[:, :-1] + z_at_w[:, 1:])
    t8w = xp.empty((ncol, nz + 1), dtype=dtype)
    t8w[:, 1:nz] = (fnm[None, 1:nz] * t3d[:, 1:nz]
                    + fnp[None, 1:nz] * t3d[:, 0:nz - 1])
    w1 = (z_at_w[:, 0] - z[:, 1]) / (z[:, 0] - z[:, 1])
    t8w[:, 0] = w1 * t3d[:, 0] + (real(1.0) - w1) * t3d[:, 1]
    w1 = ((z_at_w[:, nz] - z[:, nz - 2]) / (z[:, nz - 1] - z[:, nz - 2]))
    t8w[:, nz] = w1 * t3d[:, nz - 1] + (real(1.0) - w1) * t3d[:, nz - 2]
    return t8w


@dataclass
class RRTMLongwaveRadiation:
    """GPU-resident adapter for WRF ``ra_lw_physics=1``.

    This adapter owns the longwave half only: it publishes RTHRATENLW,
    GLW and OLR and returns zero shortwave.  It is therefore composable
    with any shortwave adapter -- see :class:`RRTMDudhiaRadiation` for the
    1/1 pairing WRF's ``lwrad_select CASE (1)`` / ``swrad_select CASE (1)``
    describes.

    GLW provenance is ``"scheme"``: RRTM computes the surface downward
    longwave from its own transfer and writes it, never reading whatever
    the field carried before.
    """

    start_time: datetime
    latitude_deg: object
    longitude_deg: object
    p_top: float | None = None
    icloud: int = 1
    #: ``None`` auto-sizes the solver chunk from free device memory at
    #: the first EAGER solve of each ``(ncol, nlayers)`` geometry and
    #: caches it (:meth:`_column_chunk_for`); an integer pins it.  The
    #: answer is byte identical either way.
    column_chunk: int | None = None
    update_count: int = field(default=0, init=False)
    #: ``(ncol, nlayers) -> resolved chunk``, filled by eager solves only.
    #: Plain ints, so it never enters the restart classifier's array walk.
    _auto_chunk: dict = field(default_factory=dict, init=False, repr=False)

    publishes_olr = True
    #: The GLW buffer is produced by this scheme, not carried in.
    glw_provenance = "scheme"

    def __post_init__(self) -> None:
        import cupy as cp

        if not isinstance(self.start_time, datetime):
            raise TypeError("radiation_start_time must be a datetime")
        # Coerce to a Python float exactly as the legacy-RRTMG sibling does
        # (gpuwm/core/rrtmg_legacy.py): initialize_physics hands over
        # ``state.p_top``, a NumPy float32 SCALAR, which the restart
        # classifier's ``_is_array_like`` treats as an array -- an
        # unclassified one, so every RRTM run's end-of-run canonical
        # digest (and any restart write) died on a perfect integration
        # (1.9.1 D3).
        self.p_top = None if self.p_top is None else float(self.p_top)
        self.latitude_deg = cp.ascontiguousarray(
            cp.asarray(self.latitude_deg, dtype=cp.float32))
        self.longitude_deg = cp.ascontiguousarray(
            cp.asarray(self.longitude_deg, dtype=cp.float32))
        if self.latitude_deg.shape != self.longitude_deg.shape:
            raise ValueError("radiation latitude/longitude shapes must match")
        if self.icloud not in (0, 1):
            raise ValueError("icloud must be 0 or 1")
        if self.column_chunk is not None and self.column_chunk < 1:
            raise ValueError("column_chunk must be positive")
        # Fail closed at construction if the packaged tables are missing
        # or corrupt, rather than mid-forecast.
        self._tables = load_rrtm_lw_tables()

    @staticmethod
    def _top_down(array):
        import cupy as cp
        nz, ny, nx = array.shape
        return cp.ascontiguousarray(
            array.transpose(1, 2, 0).reshape(ny * nx, nz)[:, ::-1])

    @staticmethod
    def _state_field(state, name, fallback):
        value = getattr(state, name, None)
        return fallback if value is None else value

    def _resolve_p_top(self, atmosphere, state) -> float:
        if self.p_top is not None:
            return float(self.p_top)
        declared = getattr(state, "p_top", None)
        if declared is not None:
            return float(declared)
        return float(atmosphere["p_interface"][-1].max())

    def _column_chunk_for(self, ncol: int, nlayers: int, cp) -> int:
        """The solver chunk: the pin, the cached auto-size, or the floor.

        The auto-size reads ``cudaMemGetInfo``, which is ILLEGAL inside
        CUDA graph capture -- the 8d94a2ff7 perf fix called it on every
        solve and took both graph-replay rungs of the tiles gate down
        with ``GraphCaptureError``.  So the memory query runs on EAGER
        solves only, once per ``(ncol, nlayers)`` geometry, and its
        answer is cached; a captured solve reuses the cached value.  A
        capture that arrives COLD (no eager solve ran first) uses the
        ``DEFAULT_COLUMN_CHUNK`` floor without caching it, so a later
        eager call still auto-sizes.  Every branch returns the same
        bytes: chunking is per-column arithmetic only
        (tests/test_rrtm_longwave.py::
        test_column_chunking_does_not_change_the_answer), so this choice
        is throughput, never trajectory.  An explicit ``column_chunk``
        pins all of it, exactly as before.
        """
        if self.column_chunk is not None:
            return int(self.column_chunk)
        key = (int(ncol), int(nlayers))
        cached = self._auto_chunk.get(key)
        if cached is not None:
            return cached
        if _stream_is_capturing(cp):
            return min(int(ncol), DEFAULT_COLUMN_CHUNK)
        resolved = _resolve_column_chunk(None, ncol, nlayers, cp)
        self._auto_chunk[key] = resolved
        return resolved

    def longwave(self, *, atmosphere, fields, state, cfg):
        """Run RRTM LW and return ``(rthratenlw, glw, olr)`` on device."""
        import cupy as cp
        from gpuwm.core import constants as c
        from gpuwm.core.mynn_radiation import (merge_mynn_bl_clouds,
                                               mynn_bl_cloud_active,
                                               wrf_itimestep)
        from gpuwm.core.rrtmgp import cal_cldfra1

        temperature = atmosphere["temperature"]
        nz, ny, nx = temperature.shape
        if self.latitude_deg.shape != (ny, nx):
            raise ValueError(
                "radiation latitude/longitude must match state grid")
        ncol = ny * nx
        nlayers = rrtm_layer_count(nz, self._resolve_p_top(atmosphere, state))

        zero = cp.zeros_like(temperature)
        qv = self._top_down(cp.maximum(atmosphere["qv"], cp.float32(0.0)))
        qc = self._top_down(cp.maximum(
            self._state_field(state, "qc", atmosphere["qc"]),
            cp.float32(0.0)))
        qr = self._top_down(cp.maximum(
            self._state_field(state, "qr", zero), cp.float32(0.0)))
        qi = self._top_down(cp.maximum(
            self._state_field(state, "qi", atmosphere["qi"]),
            cp.float32(0.0)))
        qs = self._top_down(cp.maximum(
            self._state_field(state, "qs", zero), cp.float32(0.0)))
        qg = self._top_down(cp.maximum(
            self._state_field(state, "qg", zero), cp.float32(0.0)))
        t_cols = self._top_down(temperature)
        p_cols = self._top_down(atmosphere["pressure"])
        dz_cols = self._top_down(atmosphere["dz"])

        if self.icloud:
            # cal_cldfra1 is the driver's icloud=1 branch and takes
            # (ncol, nlay) blocks; it is elementwise, so the top-down
            # packing carries through unchanged.
            cldfra = cal_cldfra1(qv, qc, qi, qs, t_cols, p_cols,
                                 f_qc=True, f_qi=True, f_qs=True)
            active_bl = mynn_bl_cloud_active(
                getattr(cfg, "bl_pbl_physics", 0),
                getattr(cfg, "icloud_bl", 0))
            qc, qi, cldfra = merge_mynn_bl_clouds(
                qc, qi, cldfra,
                qc_bl=self._top_down(fields["qc_bl"]) if active_bl else None,
                qi_bl=self._top_down(fields["qi_bl"]) if active_bl else None,
                cldfra_bl=(self._top_down(fields["cldfra_bl"])
                           if active_bl else None),
                bl_pbl_physics=getattr(cfg, "bl_pbl_physics", 0),
                icloud_bl=getattr(cfg, "icloud_bl", 0),
                itimestep=(wrf_itimestep(state.elapsed_seconds, cfg.dt)
                           if active_bl else 1))
            #  RRTMLWRAD's F_QI-absent branch (F:1929-1943), which moves
            #  QC into QI and QR into QS below 273.15 K, is deliberately
            #  not reproduced: it exists for schemes that do not carry
            #  ice, and gpuwm's state always presents QI/QS -- zero-filled
            #  for a warm-rain scheme, which is the same result WRF
            #  reaches through its warm_rain guard.
        else:
            #  RRTMLWRAD's "IF (ICLOUD .ne. 0)" block never runs, leaving
            #  every hydrometeor and the cloud fraction at zero
            #  (F:1888-1971).
            cldfra = cp.zeros_like(t_cols)
            qc = cp.zeros_like(t_cols)
            qr = cp.zeros_like(t_cols)
            qi = cp.zeros_like(t_cols)
            qs = cp.zeros_like(t_cols)
            qg = cp.zeros_like(t_cols)

        p8w = self._top_down(atmosphere["p_interface"])
        z_at_w = self._top_down(atmosphere["z_interface"])
        #  _top_down reverses to top-first; t8w's phy_prep recurrence wants
        #  bottom-up columns, so reverse back for the build and again for
        #  the RRTM frame.
        t8w = _t8w_columns(
            cp, t_cols[:, ::-1], z_at_w[:, ::-1],
            cp.asarray(state.fnm, dtype=cp.float32),
            cp.asarray(state.fnp, dtype=cp.float32))
        tw_cols = cp.ascontiguousarray(t8w[:, ::-1])

        tsk = cp.asarray(fields["tsk"], dtype=cp.float32).reshape(-1)
        emiss = cp.asarray(fields["emiss"], dtype=cp.float32).reshape(-1)

        valid_time = (self.start_time
                      + timedelta(seconds=float(state.elapsed_seconds)))
        gases = rrtm_default_trace_gases(valid_time.year)

        tten = cp.empty((ncol, nz), dtype=cp.float32)
        glw = cp.empty((ncol,), dtype=cp.float32)
        olr = cp.empty((ncol,), dtype=cp.float32)
        hpa = cp.float32(0.01)
        chunk = self._column_chunk_for(ncol, nlayers, cp)
        for start in range(0, ncol, chunk):
            stop = min(start + chunk, ncol)
            block = slice(start, stop)
            result = rrtm_longwave_columns(
                t=t_cols[block], p_mb=p_cols[block] * hpa,
                dz=dz_cols[block], qv=qv[block], qc=qc[block],
                qr=qr[block], qi=qi[block], qs=qs[block], qg=qg[block],
                cldfra=cldfra[block], tw=tw_cols[block],
                pw_mb=p8w[block] * hpa, tsfc=tsk[block],
                emiss=emiss[block], r=c.RD, g=c.G,
                co2vmr=gases["co2"], n2ovmr=gases["n2o"],
                ch4vmr=gases["ch4"], nlayers=nlayers, tables=self._tables)
            tten[block] = result["tten"]
            glw[block] = result["glw"]
            olr[block] = result["olr"]

        #  RRTMLWRAD F:2010-2013: rthraten += tten/pi3d, bottom-up.
        heating = cp.ascontiguousarray(
            tten.reshape(ny, nx, nz).transpose(2, 0, 1))
        rthratenlw = heating / atmosphere["exner"]
        self.update_count += 1
        return rthratenlw, glw.reshape(ny, nx), olr.reshape(ny, nx)

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp
        from gpuwm.core.physics import RadiationResult

        rthratenlw, glw, olr = self.longwave(
            atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
        return RadiationResult(
            rthratenlw=rthratenlw,
            rthratensw=cp.zeros_like(rthratenlw),
            swdown=cp.zeros(glw.shape, dtype=cp.float32),
            glw=glw,
            gsw=cp.zeros(glw.shape, dtype=cp.float32),
            coszen=cp.zeros(glw.shape, dtype=cp.float32),
            olr=olr)

    @property
    def restart_identity(self) -> dict[str, object]:
        """Trajectory-changing setup bound into tree restart identity."""
        return {
            "algorithm": "wrf-v4.6.1-rrtm-longwave",
            "wrf_source": "phys/module_ra_rrtm.F:RRTMLWRAD/RRTM",
            "icloud": int(self.icloud),
            "ghg_input": 0,
            "supported_path": "no-chem,no-cam-gases,o3-climatology",
        }


@dataclass
class RRTMDudhiaRadiation:
    """WRF's classic ``ra_lw_physics=1`` + ``ra_sw_physics=1`` pairing.

    The two WRF schemes are independent: ``module_radiation_driver.F``
    dispatches ``lwrad_select`` and ``swrad_select`` separately and each
    accumulates into its own tendency.  gpuwm's other radiation options
    are coupled adapters that own both streams, which is why no
    longwave-on/Dudhia-shortwave combination existed before.  This class
    is that composition and nothing more: it runs
    :class:`RRTMLongwaveRadiation` and
    :class:`~gpuwm.core.dudhia.DudhiaShortwaveRadiation` and merges their
    results, leaving both single-stream adapters untouched.

    GLW comes from RRTM (provenance ``"scheme"``), so the composed result
    never depends on what the GLW field held on entry -- which is the
    property that makes the pairing safe whichever way the surrounding
    GLW-defaulting work lands.
    """

    start_time: datetime
    latitude_deg: object
    longitude_deg: object
    p_top: float | None = None
    icloud: int = 1
    swrad_scat: float = 1.0
    #: Forwarded to :class:`RRTMLongwaveRadiation`: ``None`` auto-sizes.
    column_chunk: int | None = None

    publishes_olr = True
    glw_provenance = "scheme"

    def __post_init__(self) -> None:
        from gpuwm.core.dudhia import DudhiaShortwaveRadiation

        # Same coercion as the longwave adapter below performs for its
        # own copy: this dataclass retains ``p_top`` as a direct
        # attribute, and a NumPy float32 scalar here is an unclassified
        # "array" to the restart classifier (1.9.1 D3).
        self.p_top = None if self.p_top is None else float(self.p_top)
        self.longwave_adapter = RRTMLongwaveRadiation(
            self.start_time, self.latitude_deg, self.longitude_deg,
            p_top=self.p_top, icloud=self.icloud,
            column_chunk=self.column_chunk)
        self.shortwave_adapter = DudhiaShortwaveRadiation(
            self.start_time, self.latitude_deg, self.longitude_deg,
            swrad_scat=self.swrad_scat, icloud=self.icloud)

    @property
    def update_count(self) -> int:
        return self.longwave_adapter.update_count

    def __call__(self, *, atmosphere, fields, state, cfg):
        from gpuwm.core.physics import RadiationResult

        rthratenlw, glw, olr = self.longwave_adapter.longwave(
            atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
        shortwave = self.shortwave_adapter(
            atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
        return RadiationResult(
            rthratenlw=rthratenlw,
            rthratensw=shortwave.rthratensw,
            swdown=shortwave.swdown,
            glw=glw,
            gsw=shortwave.gsw,
            coszen=shortwave.coszen,
            olr=olr)

    @property
    def restart_identity(self) -> dict[str, object]:
        return {
            "algorithm": "wrf-v4.6.1-rrtm-lw+dudhia-sw",
            "longwave": self.longwave_adapter.restart_identity,
            "shortwave": self.shortwave_adapter.restart_identity,
        }


__all__ = ["DEFAULT_COLUMN_CHUNK", "RRTMDudhiaRadiation",
           "RRTMLongwaveRadiation", "mm5atm_columns", "o3data_columns",
           "rrtm_layer_count", "rrtm_longwave_columns", "rtrn_columns",
           "setcoef_columns"]
