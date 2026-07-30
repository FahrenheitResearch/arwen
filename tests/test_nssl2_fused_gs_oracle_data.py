from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter
from pathlib import Path


DATA = Path(__file__).parents[1] / "gpuwm" / "data" / "nssl2" / "fused-gs-oracle"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fused_gs_official_oracle_schema_hash_and_repetitions() -> None:
    path = DATA / "fused-gs.csv"
    assert _sha256(path) == "fc27cd1c1a9a1ddefcd086551d0a0ea53f731800bf398445de684d53fcf15971"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames is not None
    assert len(reader.fieldnames) == 50
    assert len(rows) == 240
    assert len({row["case"] for row in rows}) == 30
    assert {round(float(row["dt_s"]), 6) for row in rows} == {0.1, 1.0, 10.0, 60.0, 300.0}

    by_case_level: dict[tuple[str, int], dict[int, dict[str, str]]] = {}
    for row in rows:
        numeric = list(row.values())[4:]
        assert all(math.isfinite(float(value)) for value in numeric)
        key = (row["case"], int(row["repetition"]))
        by_case_level.setdefault(key, {})[int(row["k"])] = row

    for case in {row["case"] for row in rows}:
        assert sorted(by_case_level[(case, 0)]) == [1, 2, 3, 4]
        assert sorted(by_case_level[(case, 1)]) == [1, 2, 3, 4]
        for level in range(1, 5):
            left = dict(by_case_level[(case, 0)][level])
            right = dict(by_case_level[(case, 1)][level])
            left.pop("repetition")
            right.pop("repetition")
            assert left == right


def test_fused_gs_diagnostics_schema_hash_and_required_records() -> None:
    path = DATA / "fused-gs-diagnostics.csv"
    assert _sha256(path) == "f2281b6fdfed3daa6cd66b6dfb9e9bf949a8a4a232bdc02488635f6b3dae0d69"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames == [
        "schema_version",
        "case",
        "case_family",
        "repetition",
        "k",
        "record",
        "field",
        "value",
    ]
    assert len(rows) == 27_410
    counts = Counter(row["record"] for row in rows)
    assert {
        "BIGG_ACTIVE",
        "FROZEN_DEP_PRE",
        "FROZEN_DEP_APPLY",
        "FROZEN_SUB_PRE",
        "FROZEN_SUB_APPLY",
        "RAIN_HEAT_PRE_COLD",
        "RAIN_HEAT_FACTOR",
        "RAIN_HEAT_POST",
        "LIMIT",
    } == set(counts)
    assert all(math.isfinite(float(row["value"])) for row in rows)

    heat_factors = [
        float(row["value"])
        for row in rows
        if row["record"] == "RAIN_HEAT_FACTOR" and row["field"] == "shared_factor"
    ]
    assert min(heat_factors) == 0.0
    assert max(heat_factors) == 1.0

    bigg_diameters = [
        float(row["value"])
        for row in rows
        if row["record"] == "BIGG_ACTIVE" and row["field"] == "bigg_diameter_m"
    ]
    assert min(bigg_diameters) < 0.0003
    assert max(bigg_diameters) > 0.008
