"""Public visibility thresholds and the resident diagnostic allocation contract."""
from gpuwm.ensemble.batch_products import DEFAULT_THRESHOLDS, default_product_requests
from gpuwm.ensemble.batch_product_output import headline_diagnostic_memory_plan
from gpuwm.da.enprod import expand_requests
from gpuwm.verify.obs.contracts import SURFACE_UNITS, SEAM_BOUNDS


def test_visibility_defaults_are_strict_below_both_aviation_thresholds():
    requests, unavailable = default_product_requests(DEFAULT_THRESHOLDS)
    vis, = (r for r in requests if r.field == "visibility")
    assert vis.units == "m"
    assert vis.thresholds == (1000.0, 400.0)
    assert vis.comparison == "lt"
    assert "visibility" not in unavailable
    for name in ("visibility", "fog_prob"):
        rows = expand_requests((name,), ("prob",), (), (0.0,))
        assert [r.threshold for r in rows] == [1000.0, 400.0]
    assert SURFACE_UNITS["visibility"] == "m"
    assert SEAM_BOUNDS["visibility"] == (0.0, 90000.0)


def test_visibility_prices_lowest_three_level_uploads_and_both_surfaces():
    plan = headline_diagnostic_memory_plan((120, 160), host_inputs=True)
    rows = {row["name"]: row for row in plan.inventory(1)}
    for name in ("visibility", "fog_flag"):
        assert rows[f"headline:{name}"]["shape"] == (120, 160)
    carriers = [row for name, row in rows.items() if name.startswith("headline:visibility:")]
    assert len(carriers) == 10
    assert all(row["shape"] == (3, 120, 160) for row in carriers)
    assert sum(row["payload_bytes"] for row in carriers) == 10 * 3 * 120 * 160 * 4
