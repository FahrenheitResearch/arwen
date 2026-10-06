"""GPU surface callback attribution and a pristine ordinary control arm."""
from types import SimpleNamespace
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_roster_callbacks_record_member_name_land_seed_and_unity_control_bytes(tmp_path):
    cp = pytest.importorskip("cupy")
    from gpuwm.config import RunConfig
    from gpuwm.ensemble.production import PreparedEnsembleSession
    from gpuwm.ensemble.seeds import member_seed
    variants = [
        {"name": "ruc-control"},
        {"name": "ruc-dry", "surface": {"soil_moisture_scale": 0.8}},
        {"name": "noah-warm", "physics": {"sf_surface_physics": 2, "num_soil_layers": 4},
         "surface": {"sst_offset_k": 1}},
    ]
    session = PreparedEnsembleSession({"base_seed": 73, "member_variants": variants},
        output_directory=tmp_path, input_provider=lambda **unused: None, array_module=cp)
    def model(scheme, layers):
        cfg = RunConfig(nx=5, ny=4, nz=4, dx=3000, dy=3000, ztop=12000, dt=12,
                        sf_surface_physics=scheme, num_soil_layers=layers)
        fields = {"landmask": cp.ones((4, 5), cp.float32),
                  "lakemask": cp.zeros((4, 5), cp.float32), "xice": cp.zeros((4, 5), cp.float32),
                  "smois": cp.full((layers, 4, 5), 0.4, cp.float32),
                  "sh2o": cp.full((layers, 4, 5), 0.4, cp.float32),
                  "smcrel": cp.zeros((layers, 4, 5), cp.float32),
                  "tsk": cp.full((4, 5), 288, cp.float32)}
        fields["landmask"][0] = 0
        if scheme == 3:
            fields["smfr3d"] = cp.zeros((layers, 4, 5), cp.float32)
            fields["tsk_save"] = fields["tsk"].copy()
            fields["tsk_sea"] = fields["tsk"].copy()
        state = SimpleNamespace(physics=SimpleNamespace(fields=fields), elapsed_seconds=0)
        domain = SimpleNamespace(grid_id=1, run=cfg)
        node = SimpleNamespace(cfg=domain, state=state)
        return SimpleNamespace(walk_parent_first=lambda: iter((node,))), node
    models = [model(3, 6), model(3, 6), model(2, 4)]
    session._remember_member_land_layouts({index: SimpleNamespace(experiment=SimpleNamespace(domains=(node.cfg,)))
        for index, (_model, node) in enumerate(models)})
    control_model, control = models[0]
    before = {name: array.get().tobytes() for name, array in control.state.physics.fields.items()}
    assert session._initialization_callback(0) is None
    assert not hasattr(control.state, "_ensemble_surface_state")
    for name, raw in before.items():
        assert control.state.physics.fields[name].get().tobytes() == raw
    assert session._variant_receipt(0)["name"] == "ruc-control"
    assert session._variant_receipt(0)["resolved_land"][0]["num_soil_layers"] == 6
    session._initialization_callback(1)(model=models[1][0])
    session._initialization_callback(2, seed=905)(model=models[2][0])
    dry, warm = session._surface_receipts[1], session._surface_receipts[2]
    assert dry["seed"] == member_seed(73, 1)
    assert dry["member_variant"]["name"] == "ruc-dry"
    assert dry["member_variant"]["resolved_land"][0] == {
        "grid_id": 1, "sf_surface_physics": 3, "num_soil_layers": 6}
    assert warm["seed"] == 905 and warm["member_variant"]["name"] == "noah-warm"
    assert warm["member_variant"]["physics"] == {"sf_surface_physics": 2, "num_soil_layers": 4}
    assert warm["domains"][0]["realized"] == {"soil_moisture_scale": 1, "sst_offset_k": 1}
    np.testing.assert_array_equal(models[2][1].state.physics.fields["smois"].get(),
                                  np.full((4, 4, 5), 0.4, np.float32))
    assert float(models[2][1].state.physics.fields["tsk"][0, 0].get()) == 289
