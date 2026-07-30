from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import threading
import time

import pytest

from tools import download_hrrr_native_subset as download


def test_download_source_hours_keep_absolute_identity_and_cycle_horizon():
    assert download._hours(
        "12,13,14,15,16,17,18",
        cycle=datetime(2026, 7, 18, 5)) == tuple(range(12, 19))
    assert download._hours(
        "40,41,42,43,44,45,46",
        cycle=datetime(2026, 7, 18, 18)) == tuple(range(40, 47))
    with pytest.raises(ValueError, match="horizon f18"):
        download._hours("18,19", cycle=datetime(2026, 7, 18, 5))


def test_file_workers_overlap_products_but_preserve_receipt_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_product(
        request: download.ProductRequest, *, workers: int, retries: int,
    ) -> dict[str, object]:
        nonlocal active, maximum_active
        assert workers == 2
        assert retries == 3
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            # Vary completion order; Executor.map must still retain request
            # order in the sealed receipt.
            time.sleep(0.01 if request.kind == "soil" else 0.03)
            payload = request.destination.name.encode("ascii")
            request.destination.write_bytes(payload)
            request.index_path.write_text("fixture\n", encoding="ascii")
            return {
                "kind": request.kind,
                "subset_path": request.destination.name,
                "subset_bytes": len(payload),
                "subset_sha256": hashlib.sha256(payload).hexdigest(),
                "source_bytes": len(payload),
            }
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(download, "_download_product", fake_product)
    output = tmp_path / "download"
    assert download.main([
        "--cycle", "2026-07-18_00:00:00",
        "--forecast-hours", "0,1",
        "--output-root", str(output),
        "--workers", "2",
        "--file-workers", "4",
        "--retries", "3",
    ]) == 0

    assert maximum_active == 4
    receipt = json.loads((output / "download-receipt.json").read_text())
    assert receipt["file_workers"] == 4
    assert receipt["workers"] == 2
    assert receipt["source_forecast_hours"] == [0, 1]
    assert [item["subset_path"] for item in receipt["products"]] == [
        "hrrr.t00z.wrfnatf00.grib2",
        "hrrr.t00z.soilf00.grib2",
        "hrrr.t00z.wrfnatf01.grib2",
        "hrrr.t00z.soilf01.grib2",
    ]


@pytest.mark.parametrize(
    ("workers", "file_workers", "message"),
    ((2, 17, "file-workers must be 1..16"),
     (16, 8, "workers \\* file-workers may not exceed 64")),
)
def test_file_worker_resource_bounds_fail_before_download(
    tmp_path: Path, workers: int, file_workers: int, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        download.main([
            "--cycle", "2026-07-18_00:00:00",
            "--forecast-hours", "0,1",
            "--output-root", str(tmp_path / "download"),
            "--workers", str(workers),
            "--file-workers", str(file_workers),
        ])
