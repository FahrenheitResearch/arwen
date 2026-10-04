"""Labelled processes on original real preparation, forecasts and checkpoints."""
from dataclasses import replace
import gc
import json

import pytest

from conftest import requires_gpu
from test_ensemble_runtime_forecast_gpu import runtime_case, _RuntimeWords, _RuntimeProducts, _products
from test_ensemble_spp_runtime_gpu import _observe, _initialize
from test_ensemble_production_forecast_gpu import _collector

pytestmark = [pytest.mark.gpu, requires_gpu]


def _fixed(base):
    cfg = replace(base.root.run, dt=12., clock_dt=0., run_seconds=36., use_adaptive_time_step=False)
    node = replace(base.root, run=cfg, time_step=12, time_step_fract_num=0,
        time_step_fract_den=1, history_interval_s=12., history_begin_s=0., history_end_s=None)
    return replace(base, domains=(node,), run_seconds=36., restart_interval_s=12.)


class _Words(_RuntimeWords):
    def snapshot(self, phase, *, member, grid_id):
        super().snapshot(phase, member=member, grid_id=grid_id)
        state, cfg, clock = self.owners[(member, grid_id)]
        from gpuwm.io.restart import state_manifest, _scratch_manifest, _driver_manifest
        banks = dict(state_manifest(state))
        banks.update(_scratch_manifest(state))
        banks.update(_driver_manifest(state.physics))
        key = f"member-{member:04d}/d{grid_id:02d}/{phase}"
        self._arrays(key, "checkpoint_arrays", banks)
        binding = state._ensemble_stochastic
        assert binding.member_id == member
        hook = binding.hook
        processes = {"sppt": hook.sppt}
        if hook.skebs is not None:
            processes.update(skebs_psi=hook.skebs.psi, skebs_theta=hook.skebs.theta)
        assert hook.wrf_seed_labels and not hook.spp
        self._arrays(key, "spectra", {name: process.spectrum for name, process in processes.items()})
        row = {"member_id": binding.member_id, "applied_steps": binding.applied_steps,
            "completed_step": hook.completed_step, "wrf_seed_labels": hook.snapshot()["wrf_seed_labels"],
            "processes": {name: process.metadata() for name, process in processes.items()}}
        if self.reference is not None:
            assert row == self.reference.records[key]["stochastic"], (key, "labelled stochastic metadata")
        self.records[key]["stochastic"] = row

    def progress(self, **event):
        elapsed = event.get("member_model_elapsed_seconds", event.get("model_elapsed_seconds"))
        if event.get("phase") != "post-d01-sync" or elapsed != 36.:
            return
        member = int(event.get("member_id", self._member()))
        for owner_member, grid_id in tuple(self.owners):
            if owner_member == member:
                self.snapshot("final", member=member, grid_id=grid_id)
                del self.owners[(member, grid_id)]


@pytest.mark.parametrize("skebs", [False, True])
def test_labelled_real_members_products_and_complete_restart_match(runtime_case, tmp_path, monkeypatch, skebs):
    from gpuwm import runtime
    from gpuwm.ensemble.production import PreparedEnsembleSession
    from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope, ensemble_scope
    from gpuwm.ensemble.runtime_preparation import RuntimeMemberInputs
    from gpuwm.ensemble.seeds import member_seed
    from gpuwm.ensemble.stochastic_model import StochasticModelProvider
    from gpuwm.ensemble.stochastic_seeds import process_seed
    from gpuwm.io.restart import read_restart_header
    from test_ensemble_batch_product_spool_gpu import _renderer
    exp, data = _fixed(runtime_case[0]), runtime_case[1]
    renderer, members, base_seed = _renderer(), 2, 271828
    controls = {"sppt": True, "skebs": skebs,
        "wrf_seed_labels": {"nens": 19, "iseed_sppt": -117, "iseed_skebs": 913}}
    provider = StochasticModelProvider.from_mapping(controls)
    inputs = RuntimeMemberInputs(exp, data)
    reference = _Words(tmp_path / "independent-words")
    current = [reference]
    _observe(monkeypatch, current)
    independent_collector = _collector(tmp_path / "independent-products", inputs, members, renderer)
    independent = _RuntimeProducts(independent_collector, reference)
    for member in range(members):
        reference.pure_member = member
        with member_output_scope(MemberOutputCapture(independent.submit, member,
                initialize_callback=_initialize(provider, member, member_seed(base_seed, member)))):
            result = runtime.run_experiment(exp, data, tmp_path / "ordinary" / f"member-{member:04d}",
                progress_callback=reference.progress)
        assert result.completed_seconds == 36. and result.nan_free
        gc.collect()
    expected = _products(independent_collector, 1)
    reference.flush()
    observed = _Words(tmp_path / "ensemble-words", reference=reference)
    current[0] = observed
    collector = _collector(tmp_path / "ensemble", inputs, members, renderer)
    # Bound to the session, not carried by the request: a request may not
    # carry active stochastic controls while they are uncalibrated.
    session = PreparedEnsembleSession({"members": members, "base_seed": base_seed},
        output_directory=collector.root, collector=_RuntimeProducts(collector, observed),
        stochastic_provider=StochasticModelProvider.from_mapping(controls))
    with ensemble_scope(session):
        result = runtime.run_experiment(exp, data, collector.root, progress_callback=observed.progress)
    assert result.completed_seconds == 36. and result.nan_free
    assert set(observed.records) == set(reference.records)
    assert _products(collector, 1) == expected
    observed.flush()
    manifest = json.loads(result.ensemble_manifest.read_text())
    assert len(manifest["runtime_stochastic_authorities"]) == members
    assert all(row["wrf_seed_labels"]["labels"] == provider.wrf_seed_labels
               for row in manifest["runtime_stochastic_authorities"])
    restarts = sorted((tmp_path / "ordinary" / "member-0000").glob("**/gpuwmrst_*.npz"))
    assert restarts, "the original writer must persist complete labelled processes"
    header = read_restart_header(restarts[-1])
    assert header["ensemble_stochastic"]["hook"]["wrf_seed_labels"]["labels"] == provider.wrf_seed_labels
    key = "member-0000/d01/final"
    for kind, metadata in reference.records[key]["stochastic"]["processes"].items():
        assert metadata["member_seed"] == process_seed(member_seed(base_seed, 0), kind, provider.wrf_seed_labels)
    restored = _Words(tmp_path / "restored-words")
    current[0] = restored
    with member_output_scope(MemberOutputCapture(lambda **kwargs: None, 0,
            initialize_callback=_initialize(provider, 0, member_seed(base_seed, 0)))):
        result = runtime.run_experiment(exp, data, tmp_path / "restored", restart=restarts[-1],
            progress_callback=restored.progress)
    assert result.completed_seconds == 36. and result.nan_free
    if restored.owners:
        restored.snapshot("final", member=0, grid_id=1)
        restored.owners.clear()
    for group in ("checkpoint_arrays", "spectra", "stochastic", "controls"):
        assert restored.records[key][group] == reference.records[key][group], group
    restored.flush()
    changed = StochasticModelProvider.from_mapping(dict(controls,
        wrf_seed_labels=dict(controls["wrf_seed_labels"], iseed_rand_pert=4)))
    current[0] = _Words(tmp_path / "rejected-words")
    with member_output_scope(MemberOutputCapture(lambda **kwargs: None, 0,
            initialize_callback=_initialize(changed, 0, member_seed(base_seed, 0)))):
        with pytest.raises((ValueError, RuntimeError), match="stochastic|seed labels|restart"):
            runtime.run_experiment(exp, data, tmp_path / "rejected", restart=restarts[-1],
                progress_callback=current[0].progress)
