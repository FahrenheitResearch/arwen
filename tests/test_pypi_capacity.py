"""Recorded JSON fixtures and exact local bytes exercise project capacity without network."""
import hashlib
import json
from pathlib import Path

import pytest

from tools.release import pypi_capacity as capacity


def row(name, size=260_000_000, sha="a" * 64):
    return {"filename": name, "bytes": size, "sha256": sha}


def public(project, size, *, name="old.whl", yanked=False):
    return {"info": {"name": project}, "releases": {"2.8.8": [
        {"filename": name, "size": size, "yanked": yanked, "digests": {"sha256": "a" * 64}}]}}


@pytest.fixture
def recorded():
    # Recorded reduced shape of PyPI's project JSON, with the historical totals
    # from the intake. No test reaches a network, even on an error path.
    return json.loads(Path(__file__).with_name("pypi_capacity_recorded.json").read_text())


def test_historical_over_limit_refusal_names_project_and_every_number(recorded):
    receipt = capacity.check_plan({"gpuwm": [row("gpuwm-new.whl")],
        "gpuwm-data": [row("gpuwm_data-new.whl", 190_000_000)]}, fetch=recorded.__getitem__)
    assert receipt["status"] == "REFUSED"
    assert receipt["projects"][1]["status"] == "PASS"
    assert capacity.refusal(receipt["projects"][0]) == (
        "REFUSED: gpuwm project storage: current=10690000000 bytes + new=260000000 bytes = "
        "10950000000 bytes; limit=10000000000 bytes; exceeds by 950000000 bytes")


def test_companion_can_refuse_independently_and_limit_increase_is_configurable(recorded):
    recorded["gpuwm-data"] = public("gpuwm_data", 9_900_000_000)
    plan = {"gpuwm": [row("gpuwm-new.whl")], "gpuwm-data": [row("gpuwm_data-new.whl", 190_000_000)]}
    receipt = capacity.check_plan(plan, fetch=recorded.__getitem__)
    assert [r["status"] for r in receipt["projects"]] == ["REFUSED", "REFUSED"]
    raised = capacity.check_plan(plan, fetch=recorded.__getitem__, limits=dict.fromkeys(capacity.PROJECTS, 20_000_000_000))
    assert raised["status"] == "PASS"


def test_exact_limit_is_allowed_yanked_files_count_and_retry_does_not_double_count():
    document = public("gpuwm", 100, yanked=True)
    document["urls"] = document["releases"]["2.8.8"]  # do not count latest urls twice
    assert capacity.evaluate("gpuwm", document, [row("new.whl", 10)], 110)["status"] == "PASS"
    assert capacity.evaluate("gpuwm", document, [row("old.whl", 100)], 100)["new_bytes"] == 0
    with pytest.raises(capacity.CapacityError, match="differs"):
        capacity.evaluate("gpuwm", document, [row("old.whl", 100, "b" * 64)], 1000)


@pytest.mark.parametrize("document", [{}, {"info": {"name": "gpuwm"}}, public("wrong", 10), public("gpuwm", -1)])
def test_incomplete_or_wrong_public_inventory_refuses(document):
    with pytest.raises(capacity.CapacityError):
        capacity.evaluate("gpuwm", document, [], 1000)


def test_packet_hashes_are_verified_and_a_refusal_receipt_survives(tmp_path, recorded):
    artifacts = []
    for name in ("gpuwm-9.0.whl", "gpuwm_data-9.0.whl"):
        payload = b"sealed bytes"
        (tmp_path / name).write_bytes(payload)
        artifacts.append(row(name, len(payload), hashlib.sha256(payload).hexdigest()))
    (tmp_path / "PUBLICATION-ASSETS.json").write_text(json.dumps({"pypi": {"artifacts": artifacts}}))
    out = tmp_path / "capacity.json"
    with pytest.raises(capacity.CapacityError, match="gpuwm project storage"):
        capacity.check_packet(tmp_path, fetch=recorded.__getitem__, out=out)
    assert json.loads(out.read_text())["status"] == "REFUSED"
    (tmp_path / artifacts[0]["filename"]).write_bytes(b"changed")
    with pytest.raises(capacity.CapacityError, match="packet file differs"):
        capacity.check_packet(tmp_path, fetch=lambda p: pytest.fail("network was attempted"))


def test_duplicates_conflicting_rows_and_invalid_limits_refuse():
    document = public("gpuwm", 10)
    with pytest.raises(capacity.CapacityError, match="duplicate"):
        capacity.evaluate("gpuwm", document, [row("new", 1), row("new", 1)], 100)
    document["releases"]["retry"] = [dict(document["releases"]["2.8.8"][0], size=11)]
    with pytest.raises(capacity.CapacityError, match="conflicting"):
        capacity.inventory(document, "gpuwm")
    for limit in (True, -1, 0, "10 GB"):
        with pytest.raises(capacity.CapacityError):
            capacity.evaluate("gpuwm", public("gpuwm", 10), [], limit)
