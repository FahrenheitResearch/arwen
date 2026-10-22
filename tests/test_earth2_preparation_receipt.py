"""Receipt generations keep their own sealed source/profile bindings."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from gpuwm import earth2_run


@pytest.fixture
def legacy():
    # Metadata from the completed public 2024-09-24T12 shared trajectory.
    return {
        "schema": "gpuwm-earth2-fetch-v1", "model": "fcn3",
        "model_row": {"earth2studio_class": "FCN3",
                      "profile": "earth2-pl13-q-netcdf-v1", "step_hours": 6},
        "profile": "earth2-pl13-q-netcdf-v1",
    }


def select(tmp_path, report, source="earth2"):
    receipt = tmp_path / earth2_run.RECEIPT
    receipt.write_text(json.dumps(report))
    original = receipt.read_bytes()
    args = SimpleNamespace(mapped_inputs=[tmp_path / "forecast.nc"])
    adapter = SimpleNamespace(source_id=source, packaged_profile="unused")
    try:
        return earth2_run.preparation_profile(args, adapter)
    finally:
        assert receipt.read_bytes() == original


@pytest.mark.parametrize('source', ['earth2-fcn3'])
def test_completed_v1_receipt_needs_no_v2_fields(tmp_path, legacy, source):
    assert select(tmp_path, legacy, source) == legacy["profile"]


@pytest.mark.parametrize('field, value', [
    pytest.param('model', None, id='model-None'),
    pytest.param('profile', 'earth2-pl13-q-netcdf-v1-gfs', id='profile-earth2-pl13-q-netcdf-v1-gfs'),
    pytest.param('model_row.step_hours', 1, id='model_row.step_hours-1'),
])
def test_v1_identity_and_base_profile_cannot_drift(tmp_path, legacy, field, value):
    report = deepcopy(legacy)
    if "." in field:
        parent, key = field.split(".")
        report[parent][key] = value
    else:
        report[field] = value
    with pytest.raises(ValueError, match="legacy preparation"):
        select(tmp_path, report)


def test_v1_cannot_enter_another_source_door(tmp_path, legacy):
    with pytest.raises(ValueError, match="different generated source"):
        select(tmp_path, legacy, "earth2-aifs-ens")


def test_v2_keeps_donor_variant_binding(tmp_path):
    model_id, row = earth2_run.resolve_model("fcn3")
    variants = earth2_run.model_table()["donor_policy"]["profile_variants"][row["profile"]]
    donor_id, profile = next(iter(variants.items()))
    report = {"schema": "gpuwm-earth2-fetch-v2", "source": f"earth2-{model_id}",
              "donor": {"id": donor_id}, "profile": profile}
    assert select(tmp_path, report) == profile
    report["profile"] = "unbound-profile"
    with pytest.raises(ValueError, match="donor/profile pair"):
        select(tmp_path, report)
