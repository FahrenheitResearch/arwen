"""Phase-5 Task-15 full-state corruption gates (CPU + GPU twins)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.health import (
    GPU_INTEGER_EXCLUSIONS, MAX_HEALTH_FIELDS, HealthCheckError,
    StateHealthValidator,
    collect_state_fields, cuda_source, validate_fields_cpu,
    validate_state_cpu,
)


# Every uncovered class named by scout-robust-report.md.  Each fixture has
# exactly one failing value so field attribution is unambiguous.
CORRUPTIONS = (
    ("v", 501.0),
    ("php", np.nan),
    ("mup", 0.0),
    ("p", 0.0),
    ("al", np.inf),
    ("alt", 0.0),
    *((name, -1.0e-6) for name in ("qv", "qc", "qr", "qi", "qs", "qg")),
    *((name, -1.0) for name in ("nc", "nr", "ni", "ns", "ng")),
    *((name, -1.0e-12) for name in ("qvolg", "qvolh")),
    *((name, -1.0) for name in ("effc", "effr", "effi", "effs")),
    ("held.pbl.ru", np.nan),
    ("lbc.d01.u.west.value", np.nan),
    ("nest.d02.theta.west.value", np.nan),
    ("surface.tsk", 401.0),
    ("surface.smois", 1.01),
    ("surface.sh2o", -0.01),
    ("surface.tslb", 149.0),
)


@pytest.mark.parametrize(("field", "corruption"), CORRUPTIONS)
def test_cpu_mirror_attributes_every_uncovered_corruption(field, corruption):
    baseline = 300.0 if field in ("surface.tsk", "surface.tslb") else 1.0
    values = np.full((2, 3), baseline, dtype=np.float32)
    values.flat[4] = corruption
    report = validate_fields_cpu({field: values}, phase="injected")
    assert not report.ok
    assert report.first_bad_field == field
    assert report.first_bad_index == (1, 1)
    assert report.first_bad_flat_index == 4
    assert report.status_bits != 0
    assert report.reason


def test_cpu_mirror_ors_classes_but_keeps_deterministic_first_field():
    report = validate_fields_cpu({
        "v": np.array([0.0, 900.0], dtype=np.float32),
        "p": np.array([100_000.0, np.nan], dtype=np.float32),
        "surface.smois": np.array([1.2], dtype=np.float32),
    })
    assert not report.ok
    assert report.first_bad_field == "v"
    assert set(report.failing_classes) == {
        "wind", "pressure", "soil_moisture"}


def test_cpu_derived_theta_and_coupled_mass_denominators():
    from gpuwm.core.health import HealthField, rule_for_field

    thp = np.array([[[0.0]], [[-250.0]]], dtype=np.float32)
    thb = np.array([300.0, 300.0], dtype=np.float32)
    mup = np.array([[-101.0]], dtype=np.float32)
    mub = np.array([[100.0]], dtype=np.float32)
    fields = (
        HealthField("thp", thp, rule_for_field("thp"), thb, "level", 1),
        HealthField("mup", mup, rule_for_field("mup"), mub, "direct", 0),
    )
    report = validate_fields_cpu(fields)
    assert report.first_bad_field == "thp"
    assert report.first_bad_index == (1, 0, 0)
    assert set(report.failing_classes) == {"theta", "coupled_mass"}


def test_theta_bound_admits_legal_wk82_class_upper_state():
    legal = validate_fields_cpu({
        "thp": np.array([505.0], dtype=np.float32),
    })
    runaway = validate_fields_cpu({
        "thp": np.array([601.0], dtype=np.float32),
    })
    assert legal.ok
    assert not runaway.ok


def test_health_error_is_terminal_and_carries_report():
    report = validate_fields_cpu({"p": np.array([0.0], dtype=np.float32)},
                                 phase="post-sync")
    error = HealthCheckError(report)
    assert error.report is report
    assert "post-sync" in str(error)
    assert "p" in str(error)


def test_cpu_state_gate_includes_restart_persistent_scratch():
    state = SimpleNamespace(
        physics=None, lateral_boundaries=None,
        _scratch={"mp_rainnc": np.array([np.nan], dtype=np.float32)})
    report = validate_state_cpu(state)
    assert not report.ok
    assert report.first_bad_field == "surface.microphysics.scratch.mp_rainnc"


def test_nssl2_post_step_descriptor_high_watermark_fits_cpu_inventory():
    """Pin the 527-field high-water mark observed after a real NSSL step."""
    one = np.ones((1,), dtype=np.float32)
    state = SimpleNamespace(
        qvolg=one, physics=None, lateral_boundaries=None, _scratch={})
    extra = {f"post_step_{index:03d}": one for index in range(526)}
    fields = collect_state_fields(state, backend="cpu", extra_tables=extra)
    assert len(fields) == 527
    assert MAX_HEALTH_FIELDS == 1024


def test_health_cuda_source_compiles_offline_with_nvrtc():
    """Host-only NVRTC gate: compilation, no module load or device launch."""
    from cupy.cuda.compiler import compile_using_nvrtc

    ptx, _ = compile_using_nvrtc(
        cuda_source(), options=("-std=c++17",), arch="80",
        filename="health.cu")
    assert ptx


_REAL_DRIVER_INVENTORY = tuple("""
u v w thp php mup p al alt qv qc qr qi qs qg nc nr ni ns ng effc effr effi
effs h_diabatic held.pbl.ru held.pbl.rv held.pbl.rtheta held.pbl.rqv
held.pbl.rqc held.pbl.rqr held.pbl.rqi held.pbl.rqs
held.radiation.ru held.radiation.rv held.radiation.rtheta
held.radiation.rqv held.radiation.rqc held.cumulus.ru held.cumulus.rv
held.cumulus.rtheta held.cumulus.rqv held.cumulus.rqc held.cumulus.rqr
held.cumulus.rqi held.cumulus.rqs held.rthratenlw
held.rthratensw held.cu_nca held.cu_pratec held.cu_raincv
held.cu_rates.rqccuten held.cu_rates.rqicuten held.cu_rates.rqrcuten
held.cu_rates.rqscuten held.cu_rates.rqvcuten held.cu_rates.rthcuten
held._pending_rainbl surface.acsnom surface.acsnow
surface.albbck surface.albedo surface.br surface.canwat surface.cd surface.cda
surface.chklowq surface.chs surface.chs2 surface.ck surface.cka surface.cpm
surface.cqs2 surface.dz8w1 surface.ebal surface.embck surface.emiss
surface.exch_h surface.exch_m surface.fh surface.flhc surface.flqc surface.fm
surface.glw surface.grdflx surface.gz1oz0 surface.hfx surface.isltyp
surface.ivgtyp surface.kpbl surface.lai surface.lakemask surface.landmask
surface.lh surface.mavail surface.mol surface.noahres surface.pblh
surface.potevp surface.psfc surface.psih surface.psim surface.q2 surface.qfx
surface.qgh surface.qsfc surface.qv1 surface.rainbl surface.regime
surface.reslin surface.rib surface.rmol surface.sfcprs surface.sfcrunoff
surface.sfctmp surface.sh2o surface.shdmax surface.shdmin surface.smcrel
surface.smois surface.smstav surface.smstot surface.snoalb surface.snopcx
surface.snotime surface.snow surface.snowc surface.snowh surface.sr
surface.swdown surface.t2 surface.th2 surface.tmn surface.tsk surface.tslb
surface.u10 surface.udrunoff surface.ust surface.v10 surface.vegfra
surface.wspd surface.xice surface.xland surface.z0 surface.znt surface.zol
  surface.microphysics.rainnc surface.microphysics.rainncv
  surface.microphysics.sr surface.microphysics.snownc
  surface.microphysics.snowncv surface.microphysics.graupelnc
  surface.microphysics.graupelncv held.scratch.cu_nca held.scratch.cu_pratec
  held.scratch.cu_rainc held.scratch.cu_raincv held.scratch.cu_rqccuten
  held.scratch.cu_rqicuten held.scratch.cu_rqrcuten held.scratch.cu_rqscuten
  held.scratch.cu_rqvcuten held.scratch.cu_rthcuten
  surface.microphysics.scratch.mp_graupelnc
  surface.microphysics.scratch.mp_graupelncv
  surface.microphysics.scratch.mp_rainnc
  surface.microphysics.scratch.mp_rainncv
  surface.microphysics.scratch.mp_snownc
  surface.microphysics.scratch.mp_snowncv surface.microphysics.scratch.mp_sr
  lbc.forcing_tables lbc.lbc_weights_0
""".split())


def _inventory_config():
    from gpuwm.config import RunConfig

    return RunConfig(
        nx=3, ny=2, nz=8, dx=2000.0, dy=2000.0, ztop=8000.0,
        dt=10.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=10, ra_physics=4, cu_physics=1,
        sf_sfclay_physics=1, sf_surface_physics=2, bl_pbl_physics=1)


def _attach_real_driver(state, cfg, physics_mod, xp):
    physics_mod.initialize_physics(
        state, cfg, noah_params=object(), radiation=lambda **kwargs: None,
        cumulus=lambda **kwargs: None)
    state._scratch["lbc_forcing_tables"] = xp.zeros((5,), xp.float32)
    state._scratch["lbc_weights_0"] = xp.zeros((2,), xp.float32)


def test_real_domainstate_physics_inventory_names_and_dtypes_are_pinned(
        monkeypatch):
    """Production constructors under a CPU CuPy stub; no synthetic state."""
    import gpuwm.core.physics as physics_mod
    import gpuwm.core.state as state_mod

    monkeypatch.setattr(state_mod, "cp", np)
    monkeypatch.setattr(state_mod, "DTYPE", np.float32)
    monkeypatch.setattr(physics_mod, "cp", np)
    monkeypatch.setattr(physics_mod, "DTYPE", np.float32)
    cfg = _inventory_config()
    state = state_mod.DomainState(cfg)
    _attach_real_driver(state, cfg, physics_mod, np)
    fields = collect_state_fields(state, backend="gpu")

    assert tuple(field.name for field in fields) == _REAL_DRIVER_INVENTORY
    integer_fields = {
        field.name for field in fields
        if np.dtype(field.values.dtype) == np.dtype(np.int32)
    }
    assert integer_fields == {
        "surface.ebal", "surface.isltyp", "surface.ivgtyp", "surface.kpbl"}
    assert GPU_INTEGER_EXCLUSIONS == {
        "surface.isltyp", "surface.ivgtyp"} | {
        f"nest.scratch.nest_sint_{name}_{stag}"
        for name in ("ci", "ip", "cj", "jp") for stag in ("m", "x", "y")}
    assert all(np.dtype(field.values.dtype) in {
        np.dtype(np.float32), np.dtype(np.int32)} for field in fields)


@pytest.mark.gpu
def test_real_gpu_inventory_descriptor_build_handles_integer_fields():
    """Controller twin: the real constructor inventory must reach _refresh."""
    import cupy as cp
    import gpuwm.core.physics as physics_mod
    from gpuwm.core.state import DomainState

    cfg = _inventory_config()
    state = DomainState(cfg)
    _attach_real_driver(state, cfg, physics_mod, cp)
    validator = StateHealthValidator(state)
    validator._refresh()
    assert {field.name for field in validator.excluded_integer_fields} == {
        "surface.ivgtyp", "surface.isltyp"}
    assert {field.name for field in validator.fields
            if field.values.dtype == cp.int32} == {
                "surface.ebal", "surface.kpbl"}


class _GPUState:
    """Minimal state/scratch owner for controller-run GPU corruption twins."""

    def __init__(self, cp, field, corruption):
        self._cp = cp
        self._scratch = {}
        self.physics = None
        self.lateral_boundaries = None
        self._lateral_boundary_device = None
        self._extra = None
        baseline = 300.0 if field in ("surface.tsk", "surface.tslb") else 1.0
        value = cp.full((2, 3), cp.float32(baseline), dtype=cp.float32)
        value.reshape(-1)[4] = cp.float32(corruption)
        if field.startswith("held."):
            component = field.rsplit(".", 1)[-1]
            tendency = SimpleNamespace(**{component: value})
            self.physics = SimpleNamespace(
                pbl_tendencies=tendency, radiation_tendencies=None,
                cumulus_tendencies=None, fields={}, microphysics=None,
                rthratenlw=None, rthratensw=None, cu_nca=None,
                cu_pratec=None, cu_raincv=None, cu_rates=None,
                _pending_rainbl=None)
        elif field.startswith("lbc."):
            # Exercise the production packed-table branch.  Prepared real
            # states always own this resident allocation.
            self._scratch["lbc_forcing_tables"] = value
        elif field.startswith("nest."):
            self._extra = {"bad": value}
        elif field.startswith("surface."):
            leaf = field.rsplit(".", 1)[-1]
            self.physics = SimpleNamespace(
                pbl_tendencies=None, radiation_tendencies=None,
                cumulus_tendencies=None, fields={leaf: value},
                microphysics=None, rthratenlw=None, rthratensw=None,
                cu_nca=None, cu_pratec=None, cu_raincv=None, cu_rates=None,
                _pending_rainbl=None)
        else:
            setattr(self, field, value)

    def scratch(self, shape, slot):
        if slot not in self._scratch:
            self._scratch[slot] = self._cp.zeros(shape, dtype=self._cp.float32)
        return self._scratch[slot]


@pytest.mark.gpu
def test_gpu_validator_handles_nssl2_post_step_descriptor_high_watermark():
    import cupy as cp

    state = _GPUState(cp, "qvolg", 0.0)
    # Distinct allocations: the device build keeps one descriptor per
    # allocation, so aliasing every table onto one buffer would exercise the
    # de-duplication rather than the 527-descriptor high-water mark.
    extra = {f"post_step_{index:03d}": cp.ones((1,), dtype=cp.float32)
             for index in range(526)}
    validator = StateHealthValidator(state, extra_tables=extra)
    report = validator.validate(phase="nssl2-post-step-high-watermark")
    assert len(validator.fields) == 527
    assert report.ok


@pytest.mark.gpu
@pytest.mark.parametrize(("field", "corruption"), CORRUPTIONS)
def test_gpu_fused_gate_twins_attribute_every_uncovered_corruption(
        field, corruption):
    import cupy as cp

    state = _GPUState(cp, field, corruption)
    validator = StateHealthValidator(state, extra_tables=state._extra)
    report = validator.validate(phase="gpu-injected")
    expected = {
        "held.pbl.ru": "held.pbl.ru",
        "lbc.d01.u.west.value": "lbc.forcing_tables",
        "nest.d02.theta.west.value": "nest.bad",
    }.get(field, field)
    assert not report.ok
    assert report.first_bad_field == expected
    assert report.first_bad_flat_index == 4


class _ChunkedGPUState:
    """Device state whose scanned fields span several gate chunks each.

    The fused gate spreads one descriptor over ``ceil(size / chunk)`` blocks,
    so a fault that only ever lands in the first chunk would not exercise the
    geometry at all.  16,384 elements is four chunks at this inventory size.
    """

    def __init__(self, cp):
        self._cp = cp
        self._scratch = {}
        self.physics = None
        self.lateral_boundaries = None
        self._lateral_boundary_device = None
        self.u = cp.zeros((4, 64, 64), dtype=cp.float32)
        self.w = cp.zeros((4, 64, 64), dtype=cp.float32)

    def scratch(self, shape, slot):
        if slot not in self._scratch:
            self._scratch[slot] = self._cp.zeros(shape, dtype=self._cp.float32)
        return self._scratch[slot]


@pytest.mark.gpu
@pytest.mark.parametrize(("corruption", "flat"), (
    (np.nan, 9000),
    (np.inf, 13000),
    (-np.inf, 4096),        # the first element of a chunk
    (-300.0, 12287),        # finite, but outside the |w| <= 200 bound
    (250.0, 16383),         # the very last element of the last chunk
))
def test_gpu_chunked_gate_catches_a_fault_outside_the_first_chunk(
        corruption, flat):
    import cupy as cp

    state = _ChunkedGPUState(cp)
    state.w.reshape(-1)[flat] = cp.float32(corruption)
    validator = StateHealthValidator(state)
    report = validator.validate(phase="gpu-chunked")
    assert validator._chunk_count > 1
    assert flat >= validator._chunk
    assert not report.ok
    assert report.first_bad_field == "w"
    assert report.first_bad_flat_index == flat
    assert report.status_bits != 0
    assert report.reason


@pytest.mark.gpu
def test_gpu_chunked_gate_keeps_index_first_then_field_first_attribution():
    """Chunking must not disturb which fault is reported.

    The lowest flat index within a field wins even when a later chunk finds
    its own fault first, and the lowest field id still wins overall -- both
    come from the same atomicMin over (field << 48) | index.
    """
    import cupy as cp

    state = _ChunkedGPUState(cp)
    for flat in (15000, 9000, 4100, 12000):
        state.w.reshape(-1)[flat] = cp.float32(np.nan)
    validator = StateHealthValidator(state)
    report = validator.validate(phase="gpu-chunked-many")
    mirror = validate_fields_cpu({"u": cp.asnumpy(state.u),
                                  "w": cp.asnumpy(state.w)})
    assert not report.ok
    assert report.first_bad_field == "w"
    assert report.first_bad_flat_index == 4100
    assert (report.first_bad_field, report.first_bad_flat_index) == (
        mirror.first_bad_field, mirror.first_bad_flat_index)

    state.u.reshape(-1)[15000] = cp.float32(np.inf)
    report = validator.validate(phase="gpu-chunked-field-first")
    assert report.first_bad_field == "u"
    assert report.first_bad_flat_index == 15000


class _AliasedGPUState:
    """Device state registering a cumulus tendency under both spellings.

    ``collect_state_fields`` reaches ``driver.cu_rates`` and the restart-
    persistent ``cu_rthcuten`` scratch slot separately; on a real prepared
    state they are one allocation.
    """

    def __init__(self, cp, held, scratch):
        self._cp = cp
        self._scratch = {"cu_rthcuten": scratch}
        self.lateral_boundaries = None
        self._lateral_boundary_device = None
        self.physics = SimpleNamespace(
            pbl_tendencies=None, radiation_tendencies=None,
            cumulus_tendencies=None, fields={}, microphysics=None,
            rthratenlw=None, rthratensw=None, cu_nca=None, cu_pratec=None,
            cu_raincv=None, cu_rates=SimpleNamespace(rthcuten=held),
            _pending_rainbl=None)

    def scratch(self, shape, slot):
        if slot not in self._scratch:
            self._scratch[slot] = self._cp.zeros(shape, dtype=self._cp.float32)
        return self._scratch[slot]


@pytest.mark.gpu
@pytest.mark.parametrize("aliased", (True, False))
def test_gpu_gate_scans_an_aliased_buffer_once_and_still_attributes_it(
        aliased):
    import cupy as cp

    held = cp.zeros((2, 3), dtype=cp.float32)
    scratch = held if aliased else cp.zeros((2, 3), dtype=cp.float32)
    held.reshape(-1)[4] = cp.float32(np.nan)
    validator = StateHealthValidator(_AliasedGPUState(cp, held, scratch))
    report = validator.validate(phase="gpu-aliased")
    assert [field.name for field in validator.fields] == (
        ["held.cu_rates.rthcuten"] if aliased
        else ["held.cu_rates.rthcuten", "held.scratch.cu_rthcuten"])
    assert not report.ok
    assert report.first_bad_field == "held.cu_rates.rthcuten"
    assert report.first_bad_flat_index == 4
