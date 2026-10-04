"""Fixed per-rank MYNN widths preserve the resident forecast's exact words."""
from dataclasses import replace

import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("widths", [(8192, 8192), (32768, 32768), (8192, 32768)])
def test_rank_widths_match_resident_carriers_for_six_steps(widths):
    import cupy as cp

    from gpuwm.core import dycore, streaming
    from gpuwm.core.devices import DeviceOptions
    from gpuwm.core.mynn_pbl_scratch import mynn_pbl_scratch_shapes
    from tilestream.ranks_gate import config, fixture

    cfg = replace(config(256, 192, 12, rung="mynn"),
                  mp_physics=6, ra_physics=0, sf_surface_physics=0,
                  num_soil_layers=4)
    resident, bundle = fixture(cfg)
    options = DeviceOptions(count=2, ids=(0, 0))
    run = streaming.ranked_domain_builder(
        bundle, clock=None, options=options, mynn_column_chunks=widths)(
            None, cfg, streaming.ranked_decision(cfg, options)).tiled_run
    inventory = streaming.streamed_store_inventory()
    try:
        for _ in range(6):
            dycore.step(resident, cfg)
            run.sweep(1)
            actual = run.store
            for name, array in inventory(resident).items():
                assert actual[name].tobytes() == cp.asnumpy(array).tobytes(), name
        for rank, tile in enumerate(run.tiles):
            width = min(widths[rank], run.sub_cfgs[rank].nx * run.sub_cfgs[rank].ny)
            assert tile._mynn_rank_column_chunk == width
            assert run.devices_report()["mynn_column_chunks"][rank] == width
            for slot, shape in mynn_pbl_scratch_shapes(width, cfg.nz).items():
                assert tile._scratch[slot].shape == shape, slot
    finally:
        run.close()
