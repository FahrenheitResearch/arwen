"""The product/go request contract and durable native output contract."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from gpuwm import hubwind, rustwx


def test_configured_heights_and_row_driven_curve_reach_native_request(tmp_path):
    path = tmp_path / 'forecast.toml'
    path.write_text('[hubwind]\nheights_m=[92.5,137,180]\nturbine="new_curve"\nrated_mw=4.2\n')
    config = hubwind.configuration(path)
    request = hubwind._request(config, [tmp_path / 'wrfout_d02'], tmp_path,
                              ['hub_capacity_factor'], 'd02', (1200, 900))
    assert request['options']['heights_m'] == [92.5, 137, 180]
    assert request['options']['turbine'] == 'new_curve'
    assert request['inputs'] == [str(tmp_path / 'wrfout_d02')]
    assert request['products'] == ['hub_capacity_factor']


def test_go_forwards_hubwind_forecast_configuration(tmp_path):
    from gpuwm.go_cli import render_command
    plan = {'run': tmp_path, 'render': tmp_path / 'pictures',
            'config': tmp_path / 'forecast.toml',
            'render_products': 'hub_speed,hub_capacity_factor'}
    command = render_command(plan, [tmp_path / 'wrfout_d02'])
    index = command.index('--hubwind-config')
    assert command[index + 1] == str(plan['config'])
    assert command[command.index('--products') + 1] == plan['render_products']


def test_native_success_without_receipt_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(rustwx.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout='', stderr=''))
    with pytest.raises(RuntimeError, match='receipt'):
        rustwx.run_hubwind_renderer(Path('/native/renderer'), tmp_path / 'request.json')


def test_native_missing_picture_is_failure(tmp_path, monkeypatch):
    receipt = tmp_path / 'receipt.json'
    receipt.write_text('{}')
    monkeypatch.setattr(rustwx.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(returncode=0, stderr='',
                            stdout=f'RENDERED\t{tmp_path / "missing.png"}\nRECEIPT\t{receipt}\n'))
    with pytest.raises(RuntimeError, match='pictures'):
        rustwx.run_hubwind_renderer(Path('/native/renderer'), tmp_path / 'request.json')


def test_native_rows_are_packaged_and_licensed():
    rows = hubwind.catalog()
    assert {'hub_speed', 'hub_direction', 'hub_shear', 'hub_ti',
            'hub_power', 'hub_capacity_factor', 'hub_ramp_up_events'} <= hubwind.product_names()
    assert {r['license'] for r in rows['turbines']} == {'BSD-3-Clause'}
    assert all(r['url'].startswith('https://') and r['license_url'] for r in rows['observations'])
    assert {r['height_m'] for r in rows['observations']} >= {100, 300}


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + path.name.encode())
    return path


def test_a_blank_hub_map_is_counted_undrawn_never_drawn(tmp_path, monkeypatch):
    """Breakage it prevents (2.9.4 rolling pass, 2026-10-21 17:10Z): every hub
    TI map of a YSU run was a blank "NO VALID TI" picture while the render
    summary reported 0 undrawn families and counted nothing missing."""
    from gpuwm.render_receipts import read_summary

    out = tmp_path / "png"
    drawn = _png(out / "d01-12km" / "hub_speed_100m" / "20240507" / "20240507T060000Z.png")
    blank = _png(out / "d01-12km" / "hub_ti_100m" / "20240507" / "20240507T060000Z.png")
    reason = "NO VALID TI: PBL TKE absent, invalid, or calm wind"
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    monkeypatch.setattr(rustwx.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=0, stderr='',
        stdout=f'RENDERED\t{drawn}\nUNDRAWN\t{blank}\t{reason}\nRECEIPT\t{receipt}\n'))
    result = rustwx.run_hubwind_renderer(Path('/native/renderer'), tmp_path / 'request.json')
    assert result['maps'] == [str(drawn)]
    assert result['undrawn'] == [(str(blank), reason)]

    hubwind._publish_hubwind_receipt(out, SimpleNamespace(products="all"), [], result)
    summary = read_summary(out)
    assert summary["rendered_png_count"] == 1
    assert [row["name"] for row in summary["rendered_families"]] == ["hub_speed_100m"]
    undrawn = {row["name"]: row for row in summary["undrawn_families"]}
    assert set(undrawn) == {"hub_ti_100m"}
    assert summary["undrawn_family_count"] == 1
    assert any("NO VALID TI" in text for text in undrawn["hub_ti_100m"]["reasons"])
