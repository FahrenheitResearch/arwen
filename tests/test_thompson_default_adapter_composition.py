"""Composition gates for the guarded classic-Thompson adapter.

The older Thompson GPU tests deliberately admit individual CUDA slices.  The
tests in this module protect the production call graph: warm and cold source
networks must be disjoint, classic graupel number must exist on every call,
and the same number scratch must flow through source, fallout, final cleanup,
and (when requested) reflectivity.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from test_thompson_adapter_composition import _device_adapter_state


class _HostAdapterState:
    """Small NumPy-backed state for a launcher-only adapter call-graph gate."""

    def __init__(self) -> None:
        shape = (3, 1, 1)
        interface_shape = (4, 1, 1)
        self.p = np.full(shape, 80000.0, dtype=np.float32)
        self.thb = np.full((3,), 300.0, dtype=np.float32)
        self.thp = np.zeros(shape, dtype=np.float32)
        self.phb = np.asarray(
            [0.0, 9810.0, 19620.0, 29430.0], dtype=np.float32)
        self.php = np.zeros(interface_shape, dtype=np.float32)
        self.w = np.zeros(interface_shape, dtype=np.float32)
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


def _record_adapter_call(monkeypatch, *, refl_due: bool):
    """Run the real adapter while replacing every GPU launcher with a spy.

    No real kernel may be reached from this call: every gpuwm binding of the
    kernel loader is replaced by one that fails the gate by name.  The
    breakage that prevents: the per-column microphysics flag launch
    (``launch_microphysics_columns``, the no-microphysics column exit) joined
    the adapter without joining the spy list, so on the host backend, which
    accepts NumPy arrays, the real kernel ran and the gate stayed green
    without recording the launch it exists to see, and on a card the same
    call raised cupy's TypeError (THOMPSON-CARD 4.1, RTX 4090, b2e0bc80).
    """
    import gpuwm.core.kernels as kernel_loader
    import gpuwm.core.microphysics as microphysics
    import gpuwm.core.refl as refl
    import gpuwm.core.thompson as thompson
    import gpuwm.core.thompson_runtime as runtime

    calls: list[tuple[str, tuple, dict]] = []

    def spy(name):
        def launch(*args, **kwargs):
            calls.append((name, args, kwargs))
        return launch

    launcher_names = (
        "launch_adapter_entry",
        "launch_adapter_finish",
        "launch_adapter_masks",
        "launch_adapter_prepare",
        "launch_cloud_sedimentation",
        "launch_cloud_saturation_adjust",
        "launch_classic_graupel_number_finalize",
        "launch_effective_radius",
        "launch_final_phase_cleanup",
        "launch_frozen_vapor_network_from_owner",
        "launch_graupel_sedimentation",
        "launch_hydrometeor_column_mask",
        "launch_ice_sedimentation",
        "launch_rain_evaporation",
        "launch_rain_sedimentation",
        "launch_snow_sedimentation",
        "launch_warm_frozen_source_network_from_owner",
    )
    for name in launcher_names:
        monkeypatch.setattr(thompson, name, spy(name))

    def unspied(loader_name):
        def refuse(module_name, func, *args, **kwargs):
            raise AssertionError(
                f"_apply_thompson reached the real kernel {module_name}:"
                f"{func} through {loader_name}, from a launcher this gate "
                "does not spy; add the launcher to launcher_names and pin "
                "its place in the call graph")
        return refuse

    for loader_name in ("get_kernel", "get_kernel_int_defines"):
        loader = getattr(kernel_loader, loader_name)
        for module in list(sys.modules.values()):
            if (getattr(module, "__name__", "").startswith("gpuwm.")
                    and getattr(module, loader_name, None) is loader):
                monkeypatch.setattr(module, loader_name, unspied(loader_name))

    table_owner = object()
    monkeypatch.setattr(
        runtime, "load_classic_device_tables", lambda _root: table_owner)
    monkeypatch.setattr(microphysics, "cp", np)
    monkeypatch.setattr(
        microphysics, "save_pre_mp_theta", spy("save_pre_mp_theta"))
    monkeypatch.setattr(
        microphysics, "moist_physics_finish", spy("moist_physics_finish"))
    monkeypatch.setattr(
        refl, "compute_and_stash_refl_10cm", spy("reflectivity"))
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", "host-call-graph-fixture")

    state = _HostAdapterState()
    diagnostics = microphysics._apply_thompson_default(
        state, SimpleNamespace(), 10.0, refl_10cm_due=refl_due)
    assert diagnostics.rainnc is state._scratch["mp_rainnc"]
    return calls, state, table_owner


def _named_call(calls, name):
    matches = [call for call in calls if call[0] == name]
    assert len(matches) == 1, (name, [call[0] for call in calls])
    return matches[0]


def test_adapter_composes_all_thompson_phases_independently_of_output_cadence(
        monkeypatch):
    calls_off, state_off, owner_off = _record_adapter_call(
        monkeypatch, refl_due=False)
    calls_on, state_on, owner_on = _record_adapter_call(
        monkeypatch, refl_due=True)
    names_off = [name for name, _args, _kwargs in calls_off]
    names_on = [name for name, _args, _kwargs in calls_on]

    # Asking for an output may add only the diagnostic itself.  In particular,
    # it cannot decide whether classic ng exists or which source/fallout path
    # drives the forecast trajectory.
    assert names_off == [name for name in names_on if name != "reflectivity"]
    assert "reflectivity" not in names_off

    ordered_phases = (
        "launch_adapter_prepare",
        "launch_adapter_entry",
        "launch_warm_frozen_source_network_from_owner",
        "launch_frozen_vapor_network_from_owner",
        "launch_adapter_masks",
        "launch_cloud_saturation_adjust",
        "launch_rain_evaporation",
        "launch_cloud_sedimentation",
        "launch_ice_sedimentation",
        "launch_snow_sedimentation",
        "launch_graupel_sedimentation",
        "launch_rain_sedimentation",
        "launch_final_phase_cleanup",
        "launch_classic_graupel_number_finalize",
        "launch_effective_radius",
        "launch_adapter_finish",
    )
    for name in ordered_phases:
        assert names_off.count(name) == 1, (name, names_off)

    position = {name: names_off.index(name) for name in ordered_phases}
    # The entry rewrite, then WRF's column exit (:1646, :1827-1990, :2020)
    # and the private graupel number on the rewritten entry state, all before
    # any source process runs.
    assert position["launch_adapter_prepare"] < position[
        "launch_adapter_entry"]
    assert position["launch_adapter_entry"] < min(
        position["launch_warm_frozen_source_network_from_owner"],
        position["launch_frozen_vapor_network_from_owner"],
    )
    # The post-source rain and graupel fallout masks read the sources' state.
    assert max(
        position["launch_warm_frozen_source_network_from_owner"],
        position["launch_frozen_vapor_network_from_owner"],
    ) < position["launch_adapter_masks"]
    assert position["launch_adapter_masks"] < position[
        "launch_cloud_saturation_adjust"]
    assert position["launch_cloud_saturation_adjust"] < position[
        "launch_rain_evaporation"]
    assert position["launch_rain_evaporation"] < min(
        position["launch_cloud_sedimentation"],
        position["launch_ice_sedimentation"],
        position["launch_snow_sedimentation"],
        position["launch_graupel_sedimentation"],
        position["launch_rain_sedimentation"],
    )
    assert max(
        position["launch_cloud_sedimentation"],
        position["launch_ice_sedimentation"],
        position["launch_snow_sedimentation"],
        position["launch_graupel_sedimentation"],
        position["launch_rain_sedimentation"],
    ) < position["launch_final_phase_cleanup"]
    assert position["launch_final_phase_cleanup"] < position[
        "launch_classic_graupel_number_finalize"]
    assert position["launch_classic_graupel_number_finalize"] < position[
        "launch_effective_radius"]
    assert position["launch_effective_radius"] < position[
        "launch_adapter_finish"]

    # One caller-owned classic ng scratch must flow through every phase.  The
    # cold and fallout kernels intentionally receive it as a tracked shadow,
    # not as the explicit mass-velocity number branch.
    for calls, state, owner in (
            (calls_off, state_off, owner_off),
            (calls_on, state_on, owner_on)):
        shadow = state._scratch["mp_thompson_graupel_number_shadow"]
        _, prepare_args, prepare_kwargs = _named_call(
            calls, "launch_adapter_prepare")
        _, entry_args, entry_kwargs = _named_call(
            calls, "launch_adapter_entry")
        _, masks_args, _ = _named_call(calls, "launch_adapter_masks")
        _, warm_args, _ = _named_call(
            calls, "launch_warm_frozen_source_network_from_owner")
        _, cold_args, cold_kwargs = _named_call(
            calls, "launch_frozen_vapor_network_from_owner")
        _, _evap_args, evap_kwargs = _named_call(
            calls, "launch_rain_evaporation")
        _, _snow_args, snow_kwargs = _named_call(
            calls, "launch_snow_sedimentation")
        _, fallout_args, fallout_kwargs = _named_call(
            calls, "launch_graupel_sedimentation")
        _, final_args, _ = _named_call(
            calls, "launch_classic_graupel_number_finalize")
        # The flag is written into the one (ny, nx) scratch slot the phase
        # cleanup reads, which carries WRF's terminal vapour floor only in
        # the columns that had microphysics (:3974); the entry launch clears
        # it and the next launch sets it on the rewritten entry state.
        _, _cleanup_args, cleanup_kwargs = _named_call(
            calls, "launch_final_phase_cleanup")
        micro_columns = state._scratch["mp_thompson_micro_columns"]
        assert micro_columns.shape == state.p.shape[1:]
        assert not prepare_kwargs and not entry_kwargs
        assert len(prepare_args) == 20, len(prepare_args)
        assert prepare_args[5] is state._scratch["mp_pii"]
        assert prepare_args[8] is state.h_diabatic
        assert prepare_args[16] is state._scratch[
            "mp_thompson_frozen_reference_temperature"]
        assert prepare_args[17] is state._scratch[
            "mp_thompson_graupel_melt_marker"]
        assert prepare_args[19] is micro_columns
        assert len(entry_args) == 10, len(entry_args)
        assert entry_args[0] is state.qc
        assert entry_args[4] is state.qg
        assert entry_args[7] is state.qv
        assert entry_args[8] is shadow
        assert entry_args[9] is micro_columns
        assert cleanup_kwargs["micro_columns"] is micro_columns
        # The graupel fallout guard is the entry graupel marker's column
        # mask, carried in SR until the graupel fallout reads it.
        assert masks_args[2] is prepare_args[16]
        assert masks_args[3] is state._scratch["mp_rainncv"]
        assert masks_args[4] is state._scratch["mp_sr"]
        assert fallout_kwargs["active_columns"] is masks_args[4]
        assert warm_args[5] is shadow
        assert warm_args[6] is state._scratch[
            "mp_thompson_graupel_melt_marker"]
        assert warm_args[7] is state._scratch[
            "mp_thompson_snow_melt_marker"]
        assert warm_args[11] is owner
        assert evap_kwargs["graupel_melt_marker"] is warm_args[6]
        assert snow_kwargs["snow_melt_marker"] is warm_args[7]
        assert evap_kwargs["reference_density"] is state._scratch[
            "mp_thompson_rain_reference_density"]
        assert evap_kwargs["source_density"] is state._scratch[
            "mp_thompson_frozen_reference_density"]
        _, _, adjust_kwargs = _named_call(calls, "launch_cloud_saturation_adjust")
        assert adjust_kwargs["condensation_marker"] is state._scratch["mp_th"]
        assert evap_kwargs["condensation_marker"] is adjust_kwargs["condensation_marker"]
        assert evap_kwargs["reference_density"] is not warm_args[6]
        assert cold_args[9] is owner
        assert cold_kwargs["graupel_number_shadow"] is shadow
        assert fallout_kwargs["graupel_number_shadow"] is shadow
        assert fallout_kwargs.get("graupel_number") is None
        assert final_args[4] is shadow
        if "reflectivity" in [call[0] for call in calls]:
            _, _refl_args, refl_kwargs = _named_call(calls, "reflectivity")
            assert refl_kwargs["thompson_graupel_number"] is shadow


@requires_gpu
@pytest.mark.gpu
def test_apply_thompson_publishes_kernel_micron_radii_exactly_once(
        monkeypatch):
    """Composed end-to-end radius gate: exactly ONE metre->micron conversion.

    The effective-radius kernel writer (kernels/thompson.cu) applies WRF's
    own driver-side metre->micron conversion after its clamps, so the state
    contract fields ``effc``/``effi``/``effs`` must hold the kernel's device
    output with not one changed bit.  Two earlier gates each proved half of
    this seam in isolation -- the kernel writes microns; a standalone helper
    scaled metres -- and no test composed them, which is exactly how a second
    adapter-side multiply (a ~1e6-fold magnification of every published
    radius) escaped.  This gate drives the real
    ``_apply_thompson`` path and pins both halves together:

    (a) ``state.effc/effi/effs`` are BIT-IDENTICAL to the kernel's micron
        output, captured on device at the launch seam (no tolerance); and
    (b) every published radius sits inside the kernel's own final clamp
        bands converted by its own fl(1e6) factor -- fl(2.49e-6)*fl(1e6) ..
        fl(50.0e-6)*fl(1e6) um cloud, fl(4.99e-6)*fl(1e6) ..
        fl(125.0e-6)*fl(1e6) um ice, fl(9.99e-6)*fl(1e6) ..
        fl(999.0e-6)*fl(1e6) um snow (thompson.cu final writer clamps).
    """
    import cupy as cp

    import gpuwm.core.microphysics as microphysics
    import gpuwm.core.thompson as thompson

    table_root = os.environ.get("GPUWM_THOMPSON_TABLE_ROOT")
    if not table_root or not Path(table_root).is_dir():
        pytest.skip("GPUWM_THOMPSON_TABLE_ROOT with the validated classic "
                    "tables is required for the composed mp8 radius gate")

    state, cfg = _device_adapter_state()

    real_launch = thompson.launch_effective_radius
    kernel_output: dict[str, object] = {}

    def capture_effective_radius(temperature, pressure, qv, qc, qi, ni, qs,
                                 effc, effi, effs):
        real_launch(temperature, pressure, qv, qc, qi, ni, qs,
                    effc, effi, effs)
        cp.cuda.Stream.null.synchronize()
        kernel_output["effc"] = effc.copy()
        kernel_output["effi"] = effi.copy()
        kernel_output["effs"] = effs.copy()

    monkeypatch.setattr(
        thompson, "launch_effective_radius", capture_effective_radius)

    microphysics._apply_thompson_default(state, cfg, float(cfg.dt))
    cp.cuda.Stream.null.synchronize()
    assert set(kernel_output) == {"effc", "effi", "effs"}

    def micron_band(low_metres, high_metres):
        scale = np.float32(1.0e6)
        return (np.float32(np.float32(low_metres) * scale),
                np.float32(np.float32(high_metres) * scale))

    bands = {
        "effc": micron_band(2.49e-6, 50.0e-6),
        "effi": micron_band(4.99e-6, 125.0e-6),
        "effs": micron_band(9.99e-6, 999.0e-6),
    }
    for name in ("effc", "effi", "effs"):
        published = cp.asnumpy(getattr(state, name))
        from_kernel = cp.asnumpy(kernel_output[name])
        # (a) Exactly one conversion: the published state field is the
        # kernel's micron output, bit for bit.
        np.testing.assert_array_equal(
            published, from_kernel,
            err_msg=f"state.{name} is not the kernel's own micron output "
                    "(a second conversion ran between kernel and state)")
        assert published.tobytes() == from_kernel.tobytes(), name
        # (b) Physical magnitude: inside the kernel's own clamp band, in
        # microns (order 1e0..1e3, never 1e6-fold beyond it).
        low, high = bands[name]
        assert float(published.min()) >= float(low), (
            f"state.{name} min {float(published.min())!r} um below the "
            f"kernel clamp floor {float(low)!r} um")
        assert float(published.max()) <= float(high), (
            f"state.{name} max {float(published.max())!r} um above the "
            f"kernel clamp ceiling {float(high)!r} um")
        # The mixed column carries active cloud/ice/snow, so the composed
        # path must publish at least one diagnosed (off-background) radius.
        assert bool((published != low).any()), (
            f"state.{name} never left the clear-sky background -- the "
            "composed gate did not exercise a diagnosed radius")
