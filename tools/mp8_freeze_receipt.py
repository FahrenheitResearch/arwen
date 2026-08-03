#!/usr/bin/env python3
"""mp_physics=8 (classic Thompson) freeze receipt.

This module is the *instrument* half of WP-00 of the mp_physics=28
(Thompson aerosol-aware) port.  It computes, from the live tree, an
auditable JSON receipt describing every surface the mp=8 numerics depend
on.  ``tests/test_mp8_frozen.py`` is the *gate* half: it asserts each
section of this receipt against literals pinned from the pristine tree at
commit ``789f61181fb0b198ace10775f3ea184eb5e786a3``.

Why this exists
---------------
mp=28 is a second, near-complete port of Thompson that reuses none of
``gpuwm/core/kernels/thompson.cu``.  The whole mp=8 non-regression
argument is a *mechanism*, not a measurement: ``load_module`` compiles one
``cupy.RawModule`` from a single source string, so if that string is
byte-identical the PTX, the register allocation, the FP contraction and
therefore every mp=8 result are identical by construction.  The receipt
pins that string's digest (not merely the ``.cu`` file's), so a change to
``_preamble()``, to ``common.cuh``, to ``CUDA_DEFINES``, or to the loader
itself is caught with the same force as an edit to ``thompson.cu``.

Around that mechanism sit six receipts, each a real failure mode:

R1  ``thompson.cu`` / ``thompson.py`` source digests, the assembled
    compile string digest, and ``thompson.__all__``.
R2  ``thompson_contract.CLASSIC_TABLE_ASSETS`` (filename/bytes/sha256),
    ``TABLE_SET_ID``, ``MP_PHYSICS``, ``NUMBER_SPECIES``.  Also asserts
    ``CCN_ACTIVATE.BIN`` never appears there.
R3  ``moist.extra_moist_species`` on synthetic mp=8 and mp=10 states --
    Morrison's deliberate ``nc`` exclusion must survive the aerosol
    transport work.
R4  ``preflight.state_array_shapes`` / ``scratch_slot_registry`` /
    ``nest_field_kinds`` for an mp=8 config: the scratch arena layout and
    every aliasing decision.
R5  ``microphysics_transition._EDGE_FIELD_CODES`` for the 20 pre-existing
    names (28 must be APPENDED to ``PORTED_MP_PHYSICS``, never inserted).
R6  ``acoustic.prepare_moist_cq`` n_mass selection for mp=8, plus a
    call-recording double proving ``microphysics._apply_thompson`` still
    issues the identical ordered launcher argument tuples.

Plus two fixture receipts:

F1  the 92 committed mp=8 oracle CSVs under
    ``gpuwm/data/thompson/oracle/`` (aggregate + per-file digests).
F2  the *documented* four-file exception to a clean oracle rebuild
    (see :data:`ORACLE_REBUILD_EXCEPTIONS`).

Usage
-----
    python tools/mp8_freeze_receipt.py                  # JSON to stdout
    python tools/mp8_freeze_receipt.py -o receipt.json
    python tools/mp8_freeze_receipt.py --rebuild-dir DIR

``--rebuild-dir`` points at the output directory of a completed
``tools/thompson_wrf461_oracle/build.sh`` run and adds an ``oracle_rebuild``
section comparing the fresh CSVs against the committed ones.  It is opt-in
because the rebuild needs gfortran, the pristine WRF tree and ~380 MB of
regenerated coefficient tables.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                            # direct-script use
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

KERNEL_DIR = ROOT / "gpuwm" / "core" / "kernels"
ORACLE_DIR = ROOT / "gpuwm" / "data" / "thompson" / "oracle"


def present_kernel_modules() -> tuple[str, ...]:
    """Every ``.cu`` translation unit currently in the kernels directory.

    The receipt covers them all; the gate pins the ones that existed on the
    frozen tree and ignores additions, so WP-02..WP-08 can land new mp=28
    modules without the freeze gate going red for the wrong reason.
    """
    return tuple(sorted(p.stem for p in KERNEL_DIR.glob("*.cu")))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str)


def _digest(obj) -> str:
    return _sha256_bytes(_canonical_json(obj).encode("utf-8"))


@contextlib.contextmanager
def _patched(target, name: str, value):
    """Temporarily rebind ``target.name``; restore even on exception."""
    missing = object()
    previous = getattr(target, name, missing)
    setattr(target, name, value)
    try:
        yield
    finally:
        if previous is missing:                          # pragma: no cover
            delattr(target, name)
        else:
            setattr(target, name, previous)


# --------------------------------------------------------------------------
# The compile-string capture (the mechanism the whole port rests on)
# --------------------------------------------------------------------------

def capture_compiled_source(name: str) -> tuple[str, str]:
    """Return ``(source_string, method)`` cupy would compile for ``name``.

    Primary method ``loader-capture``: replace ``cupy.RawModule`` with a
    recorder and drive the REAL
    :func:`gpuwm.core.kernels.load_module`, so any future change to the
    loader (for example WP-02's ``_EXTRA_HEADERS`` allow-list) is observed
    rather than re-implemented here.  If cupy is unavailable the fallback
    ``reconstructed`` method rebuilds ``_preamble() + <name>.cu``; that
    path can only be trusted when the loader is known-unchanged, so the
    receipt records which one was used.
    """
    from gpuwm.core import kernels

    try:
        import cupy as cp
    except Exception:                                    # pragma: no cover
        src = kernels._preamble() + (
            kernels._KDIR / f"{name}.cu").read_text(encoding="utf-8")
        return src, "reconstructed"

    captured: dict[str, str] = {}

    class _RecordingRawModule:
        def __init__(self, *, code, **_kwargs):
            captured["code"] = code

        def compile(self):
            return None

        def get_function(self, _func):                   # pragma: no cover
            raise RuntimeError("freeze receipt module is not executable")

    kernels.load_module.cache_clear()
    try:
        with _patched(cp, "RawModule", _RecordingRawModule):
            kernels.load_module(name)
    finally:
        kernels.load_module.cache_clear()                # never cache the fake
    return captured["code"], "loader-capture"


# --------------------------------------------------------------------------
# R1 -- source identity
# --------------------------------------------------------------------------

def receipt_r1_sources() -> dict:
    from gpuwm.core import kernels, thompson
    from gpuwm.core.constants import CUDA_DEFINES

    kdir = KERNEL_DIR
    preamble = kernels._preamble()

    modules = {}
    for name in present_kernel_modules():
        path = kdir / f"{name}.cu"
        source, method = capture_compiled_source(name)
        reconstructed = preamble + path.read_text(encoding="utf-8")
        modules[name] = {
            "file_sha256": _sha256_file(path),
            "file_bytes": path.stat().st_size,
            "compiled_source_sha256": _sha256_bytes(source.encode("utf-8")),
            "compiled_source_len": len(source),
            "capture_method": method,
            "loader_matches_preamble_plus_file":
                source == reconstructed,
        }

    launch_symbols = tuple(sorted(
        n for n in dir(thompson) if n.startswith("launch_")))
    return {
        "kernels_init_sha256":
            _sha256_file(kdir / "__init__.py"),
        "common_cuh_sha256": _sha256_file(kdir / "common.cuh"),
        "preamble_sha256": _sha256_bytes(preamble.encode("utf-8")),
        "preamble_len": len(preamble),
        "cuda_defines": {k: float(v) for k, v in CUDA_DEFINES.items()},
        "modules": modules,
        "thompson_py_sha256":
            _sha256_file(ROOT / "gpuwm" / "core" / "thompson.py"),
        "thompson_py_all": tuple(thompson.__all__),
        "thompson_py_launch_symbols": launch_symbols,
        "thompson_cu_literal_sites": _thompson_cu_literal_sites(kdir),
    }


def _thompson_cu_literal_sites(kdir: Path) -> dict:
    """1-based line numbers of the constant-Nt_c literals mp=28 must replace.

    Informational, not a gate on its own (the source digest is the gate),
    but it is the concrete inventory a mp=28 reviewer needs: every one of
    these lines encodes ``Nt_c = 100e6`` or a gamma ratio frozen at
    ``nu_c = 12``, and every one becomes a live per-gridpoint value in
    mp=28.  MEASURED on the frozen tree, not copied from the port spec.
    """
    text = (kdir / "thompson.cu").read_text(encoding="utf-8").splitlines()
    sites: dict[str, list[int]] = {
        "100.0e6f": [], "2730.0f": [], "272.0f": [],
        "cloud_number_bin = 65": [],
    }
    for lineno, line in enumerate(text, start=1):
        for token in sites:
            if token in line:
                sites[token].append(lineno)
    return sites


# --------------------------------------------------------------------------
# R2 -- classic table contract
# --------------------------------------------------------------------------

def receipt_r2_tables() -> dict:
    from gpuwm.core import thompson_contract as tc

    assets = tuple(
        {"filename": a.filename, "bytes": a.bytes, "sha256": a.sha256}
        for a in tc.CLASSIC_TABLE_ASSETS)
    return {
        "classic_table_assets": assets,
        "table_set_id": tc.TABLE_SET_ID,
        "mp_physics": tc.MP_PHYSICS,
        "number_species": tuple(tc.NUMBER_SPECIES),
        "mass_species": tuple(tc.MASS_SPECIES),
        "transported_species": tuple(tc.TRANSPORTED_SPECIES),
        "wrf_reference_version": tc.WRF_REFERENCE_VERSION,
        "wrf_reference_commit": tc.WRF_REFERENCE_COMMIT,
        "auxiliary_table_file": tc.AUXILIARY_TABLE_FILE,
        "aerosol_blob_in_classic_assets": any(
            a["filename"] == "CCN_ACTIVATE.BIN" for a in assets),
    }


# --------------------------------------------------------------------------
# R3 -- transported species
# --------------------------------------------------------------------------

class _SpeciesProbeState:
    """Attribute-only stand-in for ``DomainState``.

    ``extra_moist_species`` is presence-based (``getattr(state, name, None)
    is not None``), so a namespace carrying the right attribute names is a
    faithful probe and needs no GPU.
    """

    def __init__(self, **fields):
        for key, value in fields.items():
            setattr(self, key, value)


def _mp8_probe_state():
    marker = object()
    return _SpeciesProbeState(
        qv=marker, qc=marker, qr=marker,
        qi=marker, qs=marker, qg=marker, nr=marker, ni=marker)


def _mp10_probe_state():
    marker = object()
    # Morrison allocates nc (state.py:393) and deliberately does not
    # transport it -- the probe carries it so its exclusion is proved,
    # not merely unexercised.
    return _SpeciesProbeState(
        qv=marker, qc=marker, qr=marker,
        qi=marker, qs=marker, qg=marker,
        nr=marker, ni=marker, ns=marker, ng=marker, nc=marker)


def receipt_r3_species() -> dict:
    from gpuwm.core import moist

    return {
        "extra_moist_species_mp8":
            tuple(moist.extra_moist_species(_mp8_probe_state())),
        "extra_moist_species_mp10":
            tuple(moist.extra_moist_species(_mp10_probe_state())),
        "moist_species_mp8":
            tuple(moist.moist_species(_mp8_probe_state())),
        "transported_number_species":
            tuple(moist.TRANSPORTED_NUMBER_SPECIES),
        "ice_mass_species": tuple(moist.ICE_MASS_SPECIES),
        "species": tuple(moist.SPECIES),
    }


# --------------------------------------------------------------------------
# R4 -- preflight allocation surface
# --------------------------------------------------------------------------

#: The mp=8 shape probe.  Deliberately tiny and deliberately fixed: the
#: pinned literals in the gate are for exactly these dimensions.
PREFLIGHT_PROBE_CONFIG = dict(
    nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
    dt=1.0, run_seconds=10.0, moist=True, mp_physics=8)


def _preflight_probe_config():
    from gpuwm.config import RunConfig
    return RunConfig(**PREFLIGHT_PROBE_CONFIG)


def receipt_r4_preflight() -> dict:
    from gpuwm.core import preflight as pf

    cfg = _preflight_probe_config()
    shapes = {k: tuple(v) for k, v in pf.state_array_shapes(cfg).items()}
    slots = {k: tuple(v) for k, v in pf.scratch_slot_registry(cfg).items()}
    return {
        "probe_config": dict(PREFLIGHT_PROBE_CONFIG),
        "state_array_shapes": shapes,
        "state_array_shapes_digest": _digest(shapes),
        "state_array_count": len(shapes),
        "scratch_slot_registry": slots,
        "scratch_slot_registry_digest": _digest(slots),
        "scratch_slot_count": len(slots),
        "nest_field_kinds": tuple(pf.nest_field_kinds(cfg)),
    }


# --------------------------------------------------------------------------
# R5 -- nest edge field codes
# --------------------------------------------------------------------------

def receipt_r5_edge_codes() -> dict:
    from gpuwm.core import microphysics_transition as mt

    return {
        "ported_mp_physics": tuple(mt.PORTED_MP_PHYSICS),
        "edge_field_codes": dict(mt._EDGE_FIELD_CODES),
        "all_edge_fields": tuple(mt._ALL_EDGE_FIELDS),
        "mass_fields": {str(k): tuple(v)
                        for k, v in mt._MASS_FIELDS.items()},
        "moment_fields": {str(k): tuple(v)
                          for k, v in mt._MOMENT_FIELDS.items()},
        "mp18_field_codes": dict(mt._FIELD_CODES),
    }


# --------------------------------------------------------------------------
# R6 -- acoustic cq selection and the adapter call graph
# --------------------------------------------------------------------------

class _HostAdapterState:
    """NumPy-backed ``DomainState`` stand-in for a launcher-only call.

    Same shape as ``tests/test_thompson_adapter_composition.py``'s probe:
    every launcher is replaced by a recorder, so no CUDA code runs and no
    coefficient table is read.  Only the ORDER and the ARGUMENT IDENTITY
    of the launcher calls are under test here.
    """

    NZ, NY, NX = 3, 1, 1

    def __init__(self) -> None:
        shape = (self.NZ, self.NY, self.NX)
        interface = (self.NZ + 1, self.NY, self.NX)
        self.p = np.full(shape, 80000.0, dtype=np.float32)
        self.thb = np.full((self.NZ,), 300.0, dtype=np.float32)
        self.thp = np.zeros(shape, dtype=np.float32)
        self.phb = np.asarray([0.0, 9810.0, 19620.0, 29430.0],
                              dtype=np.float32)
        self.php = np.zeros(interface, dtype=np.float32)
        self.w = np.zeros(interface, dtype=np.float32)
        self.qv = np.full(shape, 0.005, dtype=np.float32)
        self.qc = np.full(shape, 2.0e-4, dtype=np.float32)
        self.qr = np.full(shape, 1.0e-4, dtype=np.float32)
        self.nr = np.full(shape, 2.0e5, dtype=np.float32)
        self.qi = np.full(shape, 1.0e-6, dtype=np.float32)
        self.ni = np.full(shape, 1.0e4, dtype=np.float32)
        self.qs = np.full(shape, 2.0e-5, dtype=np.float32)
        self.qg = np.full(shape, 3.0e-5, dtype=np.float32)
        self.effc = np.zeros(shape, dtype=np.float32)
        self.effi = np.zeros(shape, dtype=np.float32)
        self.effs = np.zeros(shape, dtype=np.float32)
        self.h_diabatic = np.zeros(shape, dtype=np.float32)
        self._scratch: dict[str, np.ndarray] = {}

    def scratch(self, shape, name):
        value = self._scratch.get(name)
        if value is None:
            value = np.zeros(shape, dtype=np.float32)
            self._scratch[name] = value
        else:
            assert value.shape == tuple(shape)
        return value


#: Every launcher ``_apply_thompson`` imports from ``gpuwm.core.thompson``.
ADAPTER_LAUNCHER_NAMES = (
    "launch_cloud_sedimentation",
    "launch_cloud_saturation_adjust",
    "launch_classic_graupel_number_finalize",
    "launch_classic_graupel_number_init",
    "launch_effective_radius",
    "launch_final_phase_cleanup",
    "launch_frozen_vapor_network_from_owner",
    "launch_graupel_fallout_column_mask",
    "launch_graupel_sedimentation",
    "launch_hydrometeor_column_mask",
    "launch_ice_sedimentation",
    "launch_rain_evaporation",
    "launch_rain_sedimentation",
    "launch_snow_sedimentation",
    "launch_warm_frozen_source_network_from_owner",
)

_TABLE_OWNER_SENTINEL = "<classic-table-owner>"


def _label_map(state: _HostAdapterState) -> dict[int, str]:
    labels: dict[int, str] = {}
    for name, value in vars(state).items():
        if isinstance(value, np.ndarray):
            labels[id(value)] = f"state.{name}"
    for name, value in state._scratch.items():
        labels[id(value)] = f"scratch[{name}]"
    return labels


def _label(value, labels: dict[int, str]) -> str:
    if value is _TABLE_OWNER_SENTINEL:
        return _TABLE_OWNER_SENTINEL
    if isinstance(value, np.ndarray):
        known = labels.get(id(value))
        if known is not None:
            return known
        base = value.base
        if base is not None and id(base) in labels:
            # e.g. state.w[:-1] -- the lower full-level slice.
            return f"{labels[id(base)]}[view:{tuple(value.shape)}]"
        return f"<array shape={tuple(value.shape)} dtype={value.dtype}>"
    if isinstance(value, (bool, int, float, str)) or value is None:
        return repr(value)
    return f"<{type(value).__name__}>"


def record_adapter_calls(*, refl_10cm_due: bool) -> list[dict]:
    """Drive the real ``_apply_thompson`` with every launcher spied on.

    Returns one entry per call: the launcher name plus its positional and
    keyword arguments rendered as stable *identity labels* (which state
    field or which named scratch slot), which is what actually encodes the
    aliasing contract the mp=8 trajectory depends on.
    """
    import os

    from gpuwm.core import microphysics
    from gpuwm.core import refl as refl_mod
    from gpuwm.core import thompson
    from gpuwm.core import thompson_runtime

    calls: list[tuple[str, tuple, dict]] = []

    def spy(name):
        def launch(*args, **kwargs):
            calls.append((name, args, kwargs))
        return launch

    state = _HostAdapterState()
    previous_root = os.environ.get("GPUWM_THOMPSON_TABLE_ROOT")
    os.environ["GPUWM_THOMPSON_TABLE_ROOT"] = "mp8-freeze-receipt-fixture"
    try:
        with contextlib.ExitStack() as stack:
            for name in ADAPTER_LAUNCHER_NAMES:
                stack.enter_context(_patched(thompson, name, spy(name)))
            stack.enter_context(_patched(
                thompson_runtime, "load_classic_device_tables",
                lambda _root: _TABLE_OWNER_SENTINEL))
            stack.enter_context(_patched(microphysics, "cp", np))
            stack.enter_context(_patched(
                microphysics, "save_pre_mp_theta", spy("save_pre_mp_theta")))
            stack.enter_context(_patched(
                microphysics, "moist_physics_finish",
                spy("moist_physics_finish")))
            stack.enter_context(_patched(
                refl_mod, "compute_and_stash_refl_10cm", spy("reflectivity")))
            # The adapter's SR reduction divides by an all-zero RAINNCV in
            # this launcher-only probe; the VALUES are not under test here,
            # only the call order and argument identity.
            with np.errstate(invalid="ignore", divide="ignore"):
                microphysics._apply_thompson(
                    state, SimpleNamespace(), 10.0,
                    refl_10cm_due=refl_10cm_due)
    finally:
        if previous_root is None:
            os.environ.pop("GPUWM_THOMPSON_TABLE_ROOT", None)
        else:
            os.environ["GPUWM_THOMPSON_TABLE_ROOT"] = previous_root

    labels = _label_map(state)
    return [
        {
            "launcher": name,
            "args": [_label(a, labels) for a in args],
            "kwargs": {k: _label(v, labels) for k, v in sorted(kw.items())},
        }
        for name, args, kw in calls
    ]


def _record_acoustic_n_mass(mp_physics: int) -> int:
    """Return the ``n_mass`` ``prepare_moist_cq`` selects, via a kernel spy."""
    from gpuwm.core import acoustic

    captured: dict[str, int] = {}

    def fake_get_kernel(module, func):
        assert (module, func) == ("acoustic", "calc_cq"), (module, func)

        def launch(_grid, _block, args):
            captured["n_mass"] = int(args[10])
        return launch

    cfg = SimpleNamespace(
        nx=_HostAdapterState.NX, ny=_HostAdapterState.NY,
        nz=_HostAdapterState.NZ, mp_physics=mp_physics, moist_cq=True)
    state = _HostAdapterState()
    state.qh = state.qv
    with _patched(acoustic, "get_kernel", fake_get_kernel):
        cqu, cqv, cqw, use = acoustic.prepare_moist_cq(state, cfg)
    assert use is True
    assert (cqu.shape, cqv.shape, cqw.shape) == (
        (3, 1, 2), (3, 2, 1), (4, 1, 1))
    return captured["n_mass"]


def receipt_r6_call_graph() -> dict:
    return {
        "acoustic_n_mass": {
            "mp1": _record_acoustic_n_mass(1),
            "mp6": _record_acoustic_n_mass(6),
            "mp8": _record_acoustic_n_mass(8),
            "mp10": _record_acoustic_n_mass(10),
            "mp18": _record_acoustic_n_mass(18),
        },
        "adapter_calls_no_refl": record_adapter_calls(refl_10cm_due=False),
        "adapter_calls_with_refl": record_adapter_calls(refl_10cm_due=True),
    }


# --------------------------------------------------------------------------
# F1 -- committed mp=8 oracle fixture freeze
# --------------------------------------------------------------------------

def receipt_f1_oracle_fixtures() -> dict:
    files = sorted(p.name for p in ORACLE_DIR.glob("*.csv"))
    per_file = {name: _sha256_file(ORACLE_DIR / name) for name in files}
    return {
        "directory": str(ORACLE_DIR.relative_to(ROOT)),
        "count": len(files),
        "per_file_sha256": per_file,
        "aggregate_sha256": _digest(per_file),
    }


# --------------------------------------------------------------------------
# F2 -- the documented clean-rebuild exception (port-spec blocking unknown #2)
# --------------------------------------------------------------------------

#: The FOUR committed mp=8 oracle CSVs that a clean rebuild of the tracked
#: harness (``tools/thompson_wrf461_oracle/build.sh`` against
#: ``wrf461-pristine``) does NOT reproduce byte-for-byte.
#: Everything else is exact: 4/4 ``.dat`` SHA-256 pins and 88/92 CSVs are
#: reproduced byte-for-byte.  MEASURED, on gfortran 13.3.0 / glibc 2.39.
#:
#: CAUSE -- and this CORRECTS the port spec's blocking unknown #2, which
#: described the drift as a float32-vs-float64 evaluation of the vapour seed
#: with "p_pa, pii and theta byte-identical".  That is not what the bytes
#: say.  Measured facts, each independently reproducible:
#:
#:   (a) ``p_pa`` and ``pii`` in the committed warm/mixed/ice columns differ
#:       from a fresh build at 13 of the 24 levels, by exactly one float32
#:       ulp (max 1.50e-07 relative).  ``theta_k``/``temp_k`` follow.
#:   (b) ``p_pa`` is a pure function of the harness's own z profile
#:       (``p = p0*exp(-z/8000)``, run_column.F90:219) and does not depend
#:       on the scenario at all.  Grouping the 46 committed column CSVs by
#:       their ``before``-phase p_pa profile yields exactly TWO groups: 43
#:       files in one, and {warm, mixed, ice} in the other.  This check is
#:       hermetic -- it compares committed repository bytes only -- and it is
#:       what :func:`fixture_provenance_witness` asserts.
#:   (c) Neither the committed nor a rebuilt ``qv`` seed column is the clean
#:       float64 or float32 evaluation of ``max(1.0e-5, 0.014*exp(-z/2500))``
#:       at more than roughly half the levels.  The k=1 coincidence the port
#:       spec relied on (committed 1.2667723931372166E-002 == REAL(4) of the
#:       float64 expression) does not generalise to k=2..24.
#:   (d) No compiler-flag variation reproduces the committed bytes: -O0, -O1,
#:       -O2, -O3, -ffast-math, -march=native and -ffloat-store all give the
#:       same 26 differing p_pa rows.  Editing the seed to double precision
#:       reproduces ``mixed-surface.csv`` exactly but none of the columns.
#:
#: So this is a BUILD-PROVENANCE difference (a different libm ``expf``,
#: i.e. a different machine or glibc), affecting exactly the three ORIGINAL
#: fixtures -- ``gpuwm/data/thompson/PROVENANCE.md`` records warm, mixed and
#: ice as the first three columns generated -- and propagating through the
#: nonlinear scheme into their ``after`` rows and into mixed-surface rainnc.
#: It is NOT a physics difference and NOT an ArWen-side defect.
#:
#: It is recorded rather than repaired, because regenerating the four files
#: would move a model-validated baseline: the GPU tests consume the committed
#: ``before`` rows, so the mp=8 comparison is self-consistent as it stands.
#: Every OTHER fixture is required to be byte-exact.  No tolerance anywhere
#: in this gate is loosened to accommodate this.
#:
#: CONSEQUENCE FOR THE mp=28 PORT: the aerosol fixtures (WP-03) must be
#: generated in one pass on one recorded toolchain and their build
#: environment written down, so this class of drift cannot recur.
ORACLE_REBUILD_EXCEPTIONS = {
    "warm-column.csv": {
        "scenario": "warm (id 1)",
        "phases_affected": ["before", "after"],
        "max_relative_overall": 4.058782771046963e-06,
        "worst_field": "qc",
        "note": "1-ulp p_pa/pii/qv seed drift, amplified by the scheme",
    },
    "ice-column.csv": {
        "scenario": "ice (id 3)",
        "phases_affected": ["before", "after"],
        "max_relative_overall": 7.263e-06,
        "worst_field": "qs",
        "note": "worst field is a 1.2e-10 kg/kg trace snow mass",
    },
    "mixed-column.csv": {
        "scenario": "mixed (id 2)",
        "phases_affected": ["before", "after"],
        "max_relative_overall": 1.304e-02,
        "worst_field": "qs",
        "note": ("worst field is a 6.5e-12 kg/kg trace snow mass; the "
                 "largest non-trace deviation is qc at 4.6e-04"),
    },
    "mixed-surface.csv": {
        "scenario": "mixed (id 2)",
        "phases_affected": [],
        "max_relative_overall": 1.128e-07,
        "worst_field": "rainnc_mm",
        "note": "surface accumulation propagated from the column drift",
    },
}

#: Cross-fixture witness: the two distinct ``before``-phase p_pa profiles
#: present in the committed oracle directory, and the file group carrying the
#: minority one.  Purely a statement about committed bytes.
ORACLE_PROVENANCE_WITNESS = {
    "distinct_p_pa_profiles": 2,
    "majority_group_size": 43,
    "minority_group": ("ice-column.csv", "mixed-column.csv",
                       "warm-column.csv"),
    "p_pa_expression": "p0 * exp(-z(k) / 8000.0)   (run_column.F90:219)",
    "p_pa_levels_differing": 13,
    "p_pa_levels": (2, 3, 5, 6, 7, 8, 10, 14, 16, 17, 18, 19, 22),
    "p_pa_max_relative": 1.49741856270429e-07,
    "rebuild_toolchain": "gfortran 13.3.0, glibc 2.39, Ubuntu 24.04",
}


def _before_column(path: Path, field: str) -> tuple[str, ...]:
    """Verbatim ``before``-phase text of one CSV field (no float parsing)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = [h.strip() for h in lines[0].split(",")]
    index = header.index(field)
    out = []
    for line in lines[1:]:
        fields = [f.strip() for f in line.split(",")]
        if fields[0] == "before":
            out.append(fields[index])
    return tuple(out)


def fixture_provenance_witness() -> dict:
    """Group the committed column fixtures by their ``before`` p_pa profile.

    HERMETIC: reads only committed repository bytes -- no gfortran, no WRF
    tree, no rebuild.  ``p_pa`` is a pure function of the harness's z grid,
    so every scenario must print the same 24 values.  Any file that does not
    was produced by a different build of the harness, which is precisely the
    claim :data:`ORACLE_REBUILD_EXCEPTIONS` makes.
    """
    groups: dict[tuple[str, ...], list[str]] = {}
    for path in sorted(ORACLE_DIR.glob("*-column.csv")):
        groups.setdefault(_before_column(path, "p_pa"), []).append(path.name)
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    majority = ordered[0][1] if ordered else []
    minority = [name for _profile, names in ordered[1:] for name in names]

    detail = {}
    if len(ordered) == 2:
        ref, odd = ordered[0][0], ordered[1][0]
        differing = [i + 1 for i, (a, b) in enumerate(zip(ref, odd)) if a != b]
        worst = 0.0
        for i in differing:
            a, b = float(ref[i - 1]), float(odd[i - 1])
            worst = max(worst, abs(b - a) / abs(a))
        detail = {"levels_differing": differing,
                  "n_levels": len(ref),
                  "max_relative": worst}
    return {
        "distinct_p_pa_profiles": len(ordered),
        "group_sizes": [len(names) for _p, names in ordered],
        "majority_group_size": len(majority),
        "minority_group": sorted(minority),
        "p_pa_detail": detail,
        "seed_k1_note": _seed_k1_note(),
    }


def _seed_k1_note() -> dict:
    """Record the k=1 vapour-seed coincidence the port spec generalised from.

    Kept because it is true AT k=1 and a reader will look for it, and
    labelled because it is NOT the cause: see (c) in the
    :data:`ORACLE_REBUILD_EXCEPTIONS` docstring.
    """
    z = 250.0
    f64_then_real4 = float(np.float32(
        max(1.0e-5, 0.014 * np.exp(-z / 2500.0))))
    f32_throughout = float(np.float32(max(
        np.float32(1.0e-5),
        np.float32(0.014) * np.exp(np.float32(-z) / np.float32(2500.0)))))
    committed = None
    path = ORACLE_DIR / "warm-column.csv"
    if path.exists():
        committed = float(_before_column(path, "qv")[0])
    return {
        "z_m": z,
        "float64_then_real4": f64_then_real4,
        "float32_throughout": f32_throughout,
        "committed_warm_column_k1_qv": committed,
        "committed_matches_float64_path": (
            committed is not None
            and float(np.float32(committed)) == f64_then_real4),
        "generalises_to_other_levels": False,
    }


def compare_rebuilt_oracle(build_dir: Path) -> dict:
    """Diff a completed ``build.sh`` output against the committed tree.

    ``build_dir`` is the second argument that was given to ``build.sh``.
    Returns the .dat digest comparison and the per-CSV byte comparison.
    """
    from gpuwm.core import thompson_contract as tc

    build_dir = Path(build_dir)
    dats = {}
    for asset in tc.CLASSIC_TABLE_ASSETS:
        path = build_dir / asset.filename
        dats[asset.filename] = {
            "present": path.exists(),
            "expected_sha256": asset.sha256,
            "expected_bytes": asset.bytes,
            "rebuilt_sha256": _sha256_file(path) if path.exists() else None,
            "rebuilt_bytes": path.stat().st_size if path.exists() else None,
            "matches": (path.exists()
                        and _sha256_file(path) == asset.sha256
                        and path.stat().st_size == asset.bytes),
        }

    fresh_dir = build_dir / "column-oracle"
    identical, differing, missing = [], {}, []
    for name in sorted(p.name for p in ORACLE_DIR.glob("*.csv")):
        fresh = fresh_dir / name
        if not fresh.exists():
            missing.append(name)
            continue
        committed_bytes = (ORACLE_DIR / name).read_bytes()
        if fresh.read_bytes() == committed_bytes:
            identical.append(name)
        else:
            differing[name] = _diff_csv(ORACLE_DIR / name, fresh)
    return {
        "build_dir": str(build_dir),
        "dat_assets": dats,
        "dat_all_match": all(v["matches"] for v in dats.values()),
        "csv_total": len(identical) + len(differing) + len(missing),
        "csv_identical": len(identical),
        "csv_differing": sorted(differing),
        "csv_missing": missing,
        "csv_differences": differing,
        "differing_set_equals_documented_exceptions":
            sorted(differing) == sorted(ORACLE_REBUILD_EXCEPTIONS),
    }


def _diff_csv(committed: Path, fresh: Path) -> dict:
    """Per-column max relative deviation between two oracle CSVs."""
    old = committed.read_text(encoding="utf-8").splitlines()
    new = fresh.read_text(encoding="utf-8").splitlines()
    if len(old) != len(new):
        return {"shape_mismatch": [len(old), len(new)]}
    header = [h.strip() for h in old[0].split(",")]
    if old[0] != new[0]:
        return {"header_mismatch": [old[0], new[0]]}
    worst: dict[str, dict] = {}
    phases: set[str] = set()
    for lineno, (a, b) in enumerate(zip(old[1:], new[1:]), start=2):
        if a == b:
            continue
        fa = [f.strip() for f in a.split(",")]
        fb = [f.strip() for f in b.split(",")]
        for col, (x, y) in enumerate(zip(fa, fb)):
            if x == y:
                continue
            name = header[col] if col < len(header) else f"col{col}"
            try:
                xv, yv = float(x), float(y)
            except ValueError:
                worst.setdefault(name, {})["non_numeric"] = [x, y]
                continue
            rel = abs(yv - xv) / abs(xv) if xv else float("inf")
            prev = worst.get(name, {}).get("max_relative", -1.0)
            if rel > prev:
                worst[name] = {
                    "max_relative": rel,
                    "committed": xv,
                    "rebuilt": yv,
                    "line": lineno,
                }
            if fa and fa[0] in ("before", "after"):
                phases.add(fa[0])
    return {
        "fields": worst,
        "phases_affected": sorted(phases),
        "max_relative_overall": max(
            (v.get("max_relative", 0.0) for v in worst.values()),
            default=0.0),
    }


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def build_receipt(*, rebuild_dir: Path | None = None) -> dict:
    receipt = {
        "schema": "gpuwm.mp8-freeze-receipt.v1",
        "frozen_at_commit": "789f61181fb0b198ace10775f3ea184eb5e786a3",
        "r1_sources": receipt_r1_sources(),
        "r2_tables": receipt_r2_tables(),
        "r3_species": receipt_r3_species(),
        "r4_preflight": receipt_r4_preflight(),
        "r5_edge_codes": receipt_r5_edge_codes(),
        "r6_call_graph": receipt_r6_call_graph(),
        "f1_oracle_fixtures": receipt_f1_oracle_fixtures(),
        "f2_rebuild_exceptions": {
            "files": {k: dict(v)
                      for k, v in ORACLE_REBUILD_EXCEPTIONS.items()},
            "pinned_witness": dict(ORACLE_PROVENANCE_WITNESS),
            "measured_witness": fixture_provenance_witness(),
        },
    }
    if rebuild_dir is not None:
        receipt["oracle_rebuild"] = compare_rebuilt_oracle(rebuild_dir)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", type=Path,
                        help="write the JSON receipt here (default stdout)")
    parser.add_argument("--rebuild-dir", type=Path,
                        help="build.sh output directory to compare against")
    parser.add_argument("--section", action="append",
                        help="emit only these top-level sections")
    args = parser.parse_args(argv)

    receipt = build_receipt(rebuild_dir=args.rebuild_dir)
    if args.section:
        receipt = {k: v for k, v in receipt.items()
                   if k in set(args.section) | {"schema"}}
    text = json.dumps(receipt, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8", newline="\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":                               # pragma: no cover
    raise SystemExit(main())
