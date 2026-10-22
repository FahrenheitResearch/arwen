"""The Earth-2 source is table data: its packaged documents are what the table produces."""
from datetime import datetime, timedelta
from types import SimpleNamespace
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _author():
    spec = importlib.util.spec_from_file_location(
        "author_earth2_profile", ROOT / "tools" / "author_earth2_profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shipped_earth2_documents_are_what_the_model_table_produces():
    for name, data in _author().documents().items():
        assert (ROOT / "gpuwm" / "authorities" / name).read_bytes() == data, name


def test_earth2_mapping_declares_the_earth2studio_layout():
    from gpuwm.mapped_source import load_mapping
    mapping = load_mapping(ROOT / "gpuwm/authorities/rw-wps-earth2-pl13-q-netcdf.mapping.json")
    assert mapping["coordinates"]["time"]["lead"]["units"] == "hours"
    temperature = mapping["fields"]["air_temperature"]
    assert temperature["selector_stack_axis"] == "vertical"
    assert [s["name"] for s in temperature["selectors"]][-1] == ["t1000"]
    assert mapping["producer_attributes"]["z500"] == {"units": "m2 s-2"}


def test_a_file_holding_the_whole_forecast_prepares_only_the_run_window():
    from gpuwm.mapped_direct import _within_run_window
    start = datetime(2015, 9, 29)
    leads = tuple(SimpleNamespace(valid_time=datetime(2015, 9, 27) + timedelta(hours=6 * k))
                  for k in range(17))
    kept = _within_run_window(leads, start, 48 * 3600)
    assert [s.valid_time for s in kept] == [start + timedelta(hours=6 * k) for k in range(9)]
    short = _within_run_window(leads, start, 3 * 3600)
    assert [s.valid_time for s in short] == [start, start + timedelta(hours=6)]


def test_short_forecast_keeps_next_boundary_without_loading_lazy_fields():
    from gpuwm.mapped_direct import _within_run_window
    start = datetime(2024, 9, 24, 12)
    class Lazy:
        valid_times = tuple(start + timedelta(hours=6 * k) for k in range(13))
        def select(self, indices):
            return indices
        def __getitem__(self, index):
            raise AssertionError("select boundary times without loading weather arrays")
    assert _within_run_window(Lazy(), start, 3600) == (0, 1)
    assert _within_run_window(Lazy(), start, 21600) == (0, 1)
    assert _within_run_window(Lazy(), start + timedelta(hours=6), 3600) == (1, 2)
