"""The Zarr reader's HTTP store gives a chunk time to arrive.

The breakage: zarrs_http builds reqwest's blocking client with its 30 second
whole-request timeout.  One ERA5 chunk on Google's public copy is about 80 MB,
so on a link slower than about 2.7 MB/s every chunk was cut off mid-body and an
ERA5 run failed in its download stage with "error decoding response body"
(measured 2026-09-25 on a node whose link fell to 1.3 MB/s).
"""

from __future__ import annotations

from pathlib import Path
import re

BRIDGE = Path(__file__).resolve().parents[1] / "tools" / "zarr_bridge" / "src"


def test_the_reader_fetches_through_its_own_store_with_a_chunk_sized_timeout():
    main = (BRIDGE / "main.rs").read_text(encoding="utf-8")
    store = (BRIDGE / "http_store.rs").read_text(encoding="utf-8")
    assert "http_store::HttpStore::new" in main and "zarrs_http::HTTPStore::new" not in main
    seconds = int(re.search(r"REQUEST_TIMEOUT: Duration = Duration::from_secs\((\d+)\)", store).group(1))
    # An 80 MB chunk at 150 kB/s takes about 530 s.
    assert seconds >= 530
    assert ".timeout(REQUEST_TIMEOUT)" in store and "TRIES" in store
