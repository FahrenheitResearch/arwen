"""Public below-freezing option regression and native default semantics."""
import numpy as np
import pytest
from gpuwm.ensemble.request import EnsembleRequest
from gpuwm.ensemble.batch_products import DEFAULT_THRESHOLDS, default_product_requests
from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector
from datetime import datetime

def test_public_comparison_survives_normalization_and_collector(tmp_path):
    request = EnsembleRequest.from_mapping({"members": 36, "comparison": "lt",
        "thresholds": {"temperature2": [273.15]}})
    restored = EnsembleRequest.from_mapping(request.receipt())
    owner = HeadlineDiagnosticCollector(tmp_path, members=36, renderer="rw_wrfbatch",
        start_time=datetime(2023, 1, 31, 12), thresholds=restored.thresholds,
        comparison=restored.comparison, array_module=np, available_bytes=0)
    product = next(r for r in owner.requests if r.field == "temperature2")
    assert product.comparison == "lt"
    assert product.thresholds == (float(np.float32(273.15)),)
    owner.product_consumer.close()

def test_native_defaults_are_below_zero_and_strict_radial_exceedance():
    requests, _ = default_product_requests(DEFAULT_THRESHOLDS)
    rows = {r.field: r for r in requests}
    assert rows["temperature2_below_freezing"].comparison == "lt"
    assert rows["ice_accretion"].comparison == "gt"
    assert rows["ice_accretion"].units == "mm"
    assert rows["ice_accretion"].thresholds == tuple(float(np.float32(x)) for x in (2.54, 6.35, 12.7))
    assert rows["temperature2"].comparison == "ge"

@pytest.mark.parametrize("comparison", ["below", {"missing": "lt"}, {"temperature2": "below"}])
def test_invalid_comparison_names_concrete_breakage(comparison):
    with pytest.raises(ValueError, match="comparison"):
        EnsembleRequest.from_mapping({"members": 2, "comparison": comparison})

def test_field_comparisons_detach_and_do_not_reverse_unrelated_fields():
    spec = {"temperature2": "lt"}
    request = EnsembleRequest.from_mapping({"members": 4, "comparison": spec})
    spec["temperature2"] = "ge"
    rows = {r.field: r for r in default_product_requests(DEFAULT_THRESHOLDS,
        comparison=request.comparison)[0]}
    assert rows["temperature2"].comparison == "lt"
    assert rows["rain_total"].comparison == "ge"
    assert rows["ice_accretion"].comparison == "gt"
