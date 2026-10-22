"""Rank admission includes chemistry carriers and peak process buffers."""
from dataclasses import replace
from pathlib import Path
from gpuwm.core import preflight as pf
from gpuwm.core.devices import DeviceOptions
from gpuwm.core.devices_memory import inventory_shapes, estimate_devices, devices_gate
from gpuwm.core.mynn_pbl_scratch import mynn_pricing_memory
from gpuwm.experiment import load_experiment

RECIPE = Path(__file__).resolve().parents[1] / 'gpuwm/data/recipes/hrrr_v4_gsd41.toml'


def _smoke_recipe(tmp_path):
    """The recipe with --smoke's settings (2.9.4 ships HRRR-Smoke off)."""
    import tomllib
    from gpuwm.smoke_execution import apply_smoke_settings
    from gpuwm.toml_document import emit_experiment_toml
    out = tmp_path / RECIPE.name
    out.write_text(emit_experiment_toml(apply_smoke_settings(
        tomllib.loads(RECIPE.read_text(encoding='utf-8')))), encoding='utf-8')
    return load_experiment(out)


def test_full_conus_one_96_gib_card_names_oversized_smoke(tmp_path):
    from gpuwm.smoke_execution import capacity_reason
    exp = _smoke_recipe(tmp_path)
    reason = capacity_reason(exp, probe={'free_bytes':96*pf.GIB}, source='rap-native')
    assert 'resident smoke envelope' in reason and "card's 96.00 GiB free" in reason
    assert exp.root.run.chem_sets == 'smoke'


def test_full_conus_two_96_gib_cards_admit_all_smoke_buffers(tmp_path):
    exp = replace(_smoke_recipe(tmp_path), devices=DeviceOptions(count=2))
    with mynn_pricing_memory(total_bytes=96*pf.GIB, free_bytes=96*pf.GIB):
        price = estimate_devices(exp, vram_gib=96, forcing_intervals=1,
            source='rap-native', budgets={0:96*pf.GIB, 1:96*pf.GIB})
    assert all(card['required_bytes'] < 96*pf.GIB for card in price['cards'])
    assert not devices_gate(price, budgets={0:96*pf.GIB, 1:96*pf.GIB})['refuse']
    shapes = inventory_shapes(exp.root.run)
    assert shapes['state/chem_smoke'] == (50, 1059, 1799)
    assert shapes['state/chemdiag_fire_ebu'] == (1, 50, 1059, 1799)
    from gpuwm.core.device_inventory import chem_process_memory_shapes
    held, work = chem_process_memory_shapes(exp.root.run)
    assert held['chem/emission.fire/trailing_means'] == (2,1,1059,1799)
    assert work['chem/plumerise.freitas/workspace'] == (2048,131*202)
