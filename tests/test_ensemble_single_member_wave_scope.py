from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace

from gpuwm.config import RunConfig
from gpuwm.ensemble.admission import EnsembleMemoryModel, MemoryComponent
from gpuwm.ensemble.packing import CardBudget
from gpuwm.ensemble.production import PreparedEnsembleSession
from gpuwm.ensemble.runtime_context import current_capture


def test_one_member_waves_still_receive_independent_cuda_owners(tmp_path, monkeypatch):
    from gpuwm.ensemble import member_stream
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=2250., dy=2250., ztop=20000., dt=30.,
        run_seconds=60., moist=True, mp_physics=8,
        bl_pbl_physics=5, sf_sfclay_physics=5, sf_surface_physics=3,
        num_soil_layers=6, ra_lw_physics=4, ra_sw_physics=4,
        ra_rrtmg_variant="rte-rrtmgp", use_adaptive_time_step=True)
    domain = SimpleNamespace(run=cfg, grid_id=1)
    exp = SimpleNamespace(root=domain, domains=(domain,), devices=None,
        run_seconds=60, start_time=datetime(2024, 1, 1, tzinfo=timezone.utc))
    prepared = SimpleNamespace(experiment=exp, boundary_interval_seconds=3600)
    scopes = []
    @contextmanager
    def owned(**kwargs):
        scopes.append(kwargs["member_id"])
        yield SimpleNamespace(receipt=lambda: {"member_id": kwargs["member_id"]})
    monkeypatch.setattr(member_stream, "member_cuda_scope", owned)
    pool = SimpleNamespace(free_all_blocks=lambda: None)
    xp = SimpleNamespace(cuda=SimpleNamespace(Stream=lambda: None,
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)),
        get_default_memory_pool=lambda: pool)
    collector = SimpleNamespace(submit=lambda **kwargs: None,
        finish_run=lambda: {}, require_complete=lambda: None)
    owner = PreparedEnsembleSession(2, output_directory=tmp_path, collector=collector,
        input_provider=lambda **kwargs: prepared, array_module=xp,
        cards=(CardBudget(0, 100),), device_scope=lambda device: nullcontext(),
        memory_model=EnsembleMemoryModel((MemoryComponent("complete original", "ordinary", fixed_bytes=100),)))
    def runner(*args, **kwargs):
        return {"status": "PASS", "member": current_capture().member_id}
    result = owner.run_prepared(runner, prepared)
    assert scopes == [0, 1]
    assert result["packing"]["waves"] == 2
    assert [row["result"]["cuda_scope"]["member_id"] for row in result["member_results"]] == [0, 1]
