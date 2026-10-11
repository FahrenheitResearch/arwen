"""GPU TEMPO runtime against TEMPO's own Fortran (tools/tempo_ref).

Needs the reference artifacts from tools/tempo_ref/tempo_ref_driver.F90:
TEMPO_BLOB (tempo_blob.bin, TEMPO's tempo_init state) and TEMPO_REF_COLUMNS
(tempo_ref_columns.bin, scenarios run through TEMPO's tempo_run).  Skips
when they are not provided.

Breakage this prevents: a TEMPO kernel or pack/unpack change that moves the
GPU result off TEMPO 9adf9ef by even one bit, or makes it depend on column
position or on the run (non-reproducible).
"""
import os

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu

BLOB = os.environ.get("TEMPO_BLOB")
REF = os.environ.get("TEMPO_REF_COLUMNS")
NFLD = 17


def _scenarios():
    raw = np.fromfile(REF, dtype="<f8")
    i, out = 0, []
    while i < raw.size:
        hdr = raw[i:i + 5].view("<i8")
        sid, nz, nsteps = int(hdr[0]), int(hdr[1]), int(hdr[2])
        dt, semi = float(raw[i + 3]), float(raw[i + 4])
        i += 5
        init = raw[i:i + NFLD * nz].reshape(NFLD, nz)
        i += NFLD * nz
        step = NFLD * nz + 7 + 4 * nz
        steps = raw[i:i + step * nsteps].reshape(nsteps, step)
        i += step * nsteps
        out.append(dict(sid=sid, nz=nz, nsteps=nsteps, dt=dt, semi=semi, init=init, steps=steps))
    return out


def _fields(init, ny, nx, dtype):
    from gpuwm.core.tempo_runtime import STATE_FIELDS
    nz = init.shape[1]
    return {name: cp.ascontiguousarray(
        cp.broadcast_to(cp.asarray(init[j], dtype=dtype)[:, None, None], (nz, ny, nx)))
        for j, name in enumerate(STATE_FIELDS)}


@pytest.mark.skipif(not (BLOB and REF), reason="TEMPO_BLOB / TEMPO_REF_COLUMNS not set")
def test_tempo_runtime_matches_fortran_bitwise_and_is_reproducible():
    """Every updated field of every step of all six scenarios, semi-Lagrangian
    sedimentation (S3) included, is TEMPO's Fortran word for word.

    This was a 1e-6 relative tolerance with per-field floors while the
    kernels called CUDA's libm and an inexact NINT; both are fixed
    (tests/test_tempo_device_libm.py), so the tolerance is retired.
    """
    from gpuwm.core.tempo_runtime import STATE_FIELDS, TempoConfig, TempoRuntime

    scen = _scenarios()
    finals = []
    for attempt in range(2):
        rt = TempoRuntime(BLOB, nz_max=64, max_work_bytes=2 << 20)
        assert rt.batch < 7 * 9  # forces several column batches
        misses = []
        run_final = []
        for s in scen:
            f = _fields(s["init"], 7, 9, np.float64)
            nz = s["nz"]
            cfg = TempoConfig(semi_sedi=s["semi"] > 0.5)
            for it in range(s["nsteps"]):
                rt.step(f, s["dt"], cfg)
                ref = s["steps"][it]
                for name in ("t", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa",
                             "nifa", "ng", "qb"):
                    j = STATE_FIELDS.index(name)
                    want = np.ascontiguousarray(ref[j * nz:(j + 1) * nz])
                    got = f[name][:, 0, 0].get()
                    bad = int((got.view(np.int64) != want.view(np.int64)).sum())
                    if bad:
                        misses.append((s["sid"], it + 1, name, bad))
            for name in STATE_FIELDS:
                a = f[name]
                assert bool(cp.all(a == a[:, :1, :1])), f"{name}: columns differ"
            run_final.append({n: f[n].get() for n in STATE_FIELDS})
        assert not misses, misses[:20]
        finals.append(run_final)
    for a, b in zip(*finals):
        for n in a:
            assert np.array_equal(a[n].view(np.int64), b[n].view(np.int64)), n


@pytest.mark.skipif(not (BLOB and REF), reason="TEMPO_BLOB / TEMPO_REF_COLUMNS not set")
def test_tempo_runtime_float32_storage_runs_finite():
    from gpuwm.core.tempo_runtime import TempoConfig, TempoRuntime

    s = _scenarios()[1]
    rt = TempoRuntime(BLOB, nz_max=64)
    f = _fields(s["init"], 4, 4, np.float32)
    for _ in range(30):
        out = rt.step(f, s["dt"], TempoConfig())
    for a in f.values():
        assert bool(cp.all(cp.isfinite(a)))
    assert float(out["surface"]["rain_precip"].max()) >= 0.0


def test_composed_source_is_the_include_graph_once_each():
    """``composed_source`` is the include graph, each header exactly once.

    Breakage this prevents: the device-libm test and the libm edge probe
    compile TEMPO through ``composed_source`` (no include path).  A composition
    that dropped a header, or pasted a ``#pragma once`` header twice, would
    have them measure a kernel the runtime never runs.  (These two tests were
    carried in without the helper they import; they failed on ImportError
    until 2.8.8's RRFS follow-up added it.)
    """
    from gpuwm.core.tempo_runtime import (
        _QUOTED_INCLUDE, KERNEL_DIR, composed_source)

    code = composed_source()
    assert '#include "' not in code
    # the TEMPO headers plus WOOF's float64 libm header they include
    headers = sorted(KERNEL_DIR.glob("*.cuh")) + [KERNEL_DIR.parent / "glibc_flt64.cuh"]
    assert code.count("#pragma once") == len(headers)
    for header in headers:
        # A header's own text, between its include lines, lands once.
        for piece in _QUOTED_INCLUDE.split(header.read_text(encoding="utf-8"))[::2]:
            if piece.strip():
                assert code.count(piece) == 1, header.name


def test_composed_source_compiles_to_the_include_path_ptx():
    """Composing the headers by hand is exactly what ``-I`` compiled.

    Breakage this prevents: a composed probe whose PTX differs from the
    runtime's ``-I`` build would certify different arithmetic.
    """
    from cupy.cuda import compiler

    from gpuwm.core.tempo_runtime import _KERNEL, KERNEL_DIR, composed_source

    common = ("-std=c++17", "-DTEMPO_NZ_MAX=100", "--fmad=false")
    by_path = compiler.compile_using_nvrtc(
        _KERNEL, options=common + (f"-I{KERNEL_DIR}",))
    composed = compiler.compile_using_nvrtc(composed_source(), options=common)
    assert composed == by_path
