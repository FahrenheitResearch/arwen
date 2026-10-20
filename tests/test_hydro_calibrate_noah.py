"""Full-column calibration table: Noah-OWP rows resolve against the slab schema (CPU only)."""
import json

from gpuwm.hydro import calibrate


def test_full_column_table_resolves():
    table = json.loads(calibrate.FULL_COLUMN_TABLE.read_text())
    assert calibrate.PARAMETER_TABLE.name == "calibration-parameters.v1.json"   # default: CFE/routing rows
    assert not calibrate.noah_rows(json.loads(calibrate.PARAMETER_TABLE.read_text()))
    rows = calibrate.noah_rows(table)
    assert len(rows) >= 10
    layout = calibrate._noah_layout(table)
    assert [p["name"] for p, _ in layout] == [p["name"] for p in rows]
    for p, spans in layout:
        lo, hi = p["bounds"]
        assert lo < hi and p["prior_log_sigma"] > 0 and p["column"].startswith("noah:")
        lead_count = spans[0][1]
        for _, count in spans[1:]:
            assert count in (1, lead_count)        # per-layer ratio or the first layer's
    # the CFE/routing rows are the CFE-only table's rows, unchanged
    cfe_only = json.loads(calibrate.PARAMETER_TABLE.read_text())
    assert [p for p in table["parameters"] if "noah_fields" not in p] == cfe_only["parameters"]


def test_unknown_noah_field_is_refused():
    table = {"parameters": [{"name": "x", "noah_fields": ["parameters.not_a_field"], "bounds": [0, 1]}]}
    try:
        calibrate._noah_layout(table)
    except ValueError as exc:
        assert "not an f32 field" in str(exc)
    else:
        raise AssertionError("unknown field accepted")
