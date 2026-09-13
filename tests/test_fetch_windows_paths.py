"""Managed cache depth must not break native acquisition on Windows."""
from datetime import datetime
from pathlib import Path

import pytest

from gpuwm import fetch_routes
from test_fetch_routes import _fake_downloader


@pytest.mark.parametrize('source', ['ecmwf-open-data', 'icon-eu', 'aigefs'])
def test_deep_cache_acquisition_and_handoff_keep_the_complete_input_paths(tmp_path, source):
    out = tmp_path / ('cache-' + 'x' * max(1, 194 - len(str(tmp_path)) - 7))
    plan = fetch_routes.resolve_request(source, cycle=datetime(2026, 9, 12), hours=0, out=out)
    fetched = []
    fetch_routes.run_plan(plan, out=out, downloader=_fake_downloader(fetched),
                         probe=lambda _: False, progress=lambda _: None)
    assert len(fetched) == len(plan.objects)
    inputs, _ = fetch_routes.write_handoff(plan, out)
    assert all(Path(line).is_file() for line in inputs.read_text(encoding='utf-8').splitlines())
    reused = []
    result = fetch_routes.run_plan(plan, out=out, downloader=_fake_downloader(reused),
                                  probe=lambda _: False, progress=lambda _: None)
    assert reused == []
    assert all(row['reused'] for row in result['files'])
