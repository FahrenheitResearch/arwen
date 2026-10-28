"""A resumed HRRR fetch lands every hour of a product in one byte mode.

Breakage this prevents (updraft factory, 2026-10-27, 2023-08-07 and
2019-06-09): ``gpuwm fetch --mode idx-subset --analyzed-aerosol`` resuming
a folder an interrupted full-file fetch had started kept the whole files
and fetched only the missing hours as subsets.  A whole wrfnat carries
PBLH/LCDC/MCDC/HCDC, which hrrr_grib2_bridge selects wherever they are
published, so the preparation refused the series 80-146 s in ("f16
selected inventory does not exactly equal the first series frame").
The run's own fetch (default full-file) then relabelled the folder
"full-file".
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from gpuwm import fetch
from tools import download_hrrr_native_subset as hrrr_transport

#: Census of a whole object, well above any subset selection.
WHOLE = {"atmosphere": 1136, "soil": 743}
CYCLE = datetime(2023, 8, 7, 12)


def _grib2_stream(messages: int) -> bytes:
    one = (b"GRIB" + b"\x00\x00" + b"\x00" + b"\x02"
           + (20).to_bytes(8, "big") + b"7777")
    return one * messages


def _install_backbone(monkeypatch, calls: list[tuple[int, str, str]]):
    """A backbone that lands what the mode asks: whole or subset."""

    def fake(*, cycle, hour, kind, mode, out, analyzed_aerosol=False,
             **_ignored):
        calls.append((hour, kind, mode))
        if mode == "full-file":
            records = WHOLE[kind]
        elif kind == "atmosphere":
            records = hrrr_transport.atmosphere_record_count(
                analyzed_aerosol=analyzed_aerosol)
        else:
            records = hrrr_transport.SOIL_RECORD_COUNT
        name = (f"hrrr.t{cycle:%H}z.wrfnatf{hour:02d}.grib2"
                if kind == "atmosphere"
                else f"hrrr.t{cycle:%H}z.wrfprsf{hour:02d}.grib2")
        (out / name).write_bytes(_grib2_stream(records))
        return {
            "name": name, "idx_name": None,
            "mode": "full-file" if mode == "full-file" else "idx-subset",
            "mode_reason": "fixture", "bytes": records * 20,
            "wall_seconds": 0.1, "source": "aws",
            "grib_url": f"https://example.invalid/{name}",
            "selected_record_count": records,
            "probe": {"idx_covers_object": False,
                      "idx_record_count": records},
        }

    monkeypatch.setattr(fetch, "_rw_fetch_hrrr", fake)


def _fetch(out: Path, hours, mode: str, **options) -> dict:
    fetch.fetch_hrrr(cycle=CYCLE, hours=tuple(hours), area=None, out=out,
                     transport="s3", engine="rust",
                     engine_bin=Path("rw_fetch"), mode=mode,
                     progress=lambda _line: None, **options)
    return json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())


def test_idx_subset_resume_over_whole_files_lands_one_series(
        tmp_path, monkeypatch):
    from test_fetch import _install_runtime_surface_inputs
    _install_runtime_surface_inputs(monkeypatch)
    calls: list[tuple[int, str, str]] = []
    _install_backbone(monkeypatch, calls)
    out = tmp_path / "hrrr"

    # The interrupted full-file fetch: f00 and f01 whole.
    _fetch(out, (0, 1), "full-file")

    # The resume that mixed them: now every hour is a subset.
    calls.clear()
    manifest = _fetch(out, (0, 1, 2), "idx-subset", analyzed_aerosol=True)
    payload = [f for f in manifest["files"]
               if f["role"] in ("atmosphere", "soil")]
    assert {f["mode"] for f in payload} == {"idx-subset"}
    assert manifest["mode"] == "idx-subset"
    subset = hrrr_transport.atmosphere_record_count(analyzed_aerosol=True)
    assert {f["records"] for f in payload if f["role"] == "atmosphere"} \
        == {subset}
    assert sorted(calls) == [(hour, kind, "idx-subset") for hour in (0, 1, 2)
                             for kind in ("atmosphere", "soil")]
    # The whole files were moved aside, not deleted.
    assert len(list(out.glob("*.rejected-*"))) == 4

    # The run's own fetch (default full-file) reuses the uniform series
    # as it is and says what the files are.
    calls.clear()
    manifest = _fetch(out, (0, 1, 2), "full-file", analyzed_aerosol=True)
    assert calls == []
    assert manifest["mode"] == "idx-subset"
    assert manifest["requested_mode"] == "full-file"
