"""The actual 2020 GFS inventory which the modern fallback misdeclared."""
from pathlib import Path
from types import SimpleNamespace

from gpuwm import bridges, fetch
from tools.download_gfs_native_subset import levels_for_top


def test_archived_header_inventory_excludes_incomplete_40_hpa(monkeypatch, tmp_path):
    import subprocess
    inventory = (Path(__file__).parent / 'fixtures/gfs-2020080300-inventory.tsv').read_text()
    monkeypatch.setattr(bridges, 'find_bridge', lambda name: Path('grib2_inventory'))
    def inspect(command, **kwargs):
        assert command == ['grib2_inventory', str(tmp_path / 'archive.grib2')]
        return SimpleNamespace(stdout=inventory)
    monkeypatch.setattr(subprocess, 'run', inspect)
    available, receipt = fetch._gfs_file_levels(tmp_path / 'archive.grib2', 'gfs')
    assert 40 not in available
    assert levels_for_top(5000, available=available)[:4] == (30., 50., 70., 100.)
    assert receipt['method'] == 'native-file-inventory'
