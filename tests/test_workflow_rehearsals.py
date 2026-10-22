import json
from pathlib import Path

import pytest
import yaml

from tools import workflow_rehearsals as routes

ROOT = Path(__file__).resolve().parents[1]


def clone_workflows(tmp_path):
    target = tmp_path / ".github/workflows"
    target.mkdir(parents=True)
    for source in (ROOT / ".github/workflows").glob("*.yml"):
        (target / source.name).write_bytes(source.read_bytes())
    return tmp_path


def test_every_workflow_job_has_a_rehearsal_and_every_contract_file_exists():
    rows = routes.census()
    assert len(rows) == 17
    assert {row["workflow"] for row in rows} == {"ci.yml", "publish.yml"}
    assert all((ROOT / path).is_file() for path in routes.CONTRACT_FILES)
    by_job = {(row["workflow"], row["job"]): row for row in rows}
    assert by_job["publish.yml", "controller"]["routes"] == ["R"]
    assert by_job["publish.yml", "test"]["routes"] == ["P"]
    assert by_job["ci.yml", "publication_contracts"]["routes"] == ["R"]


def test_new_job_and_new_workflow_cannot_bypass_rehearsal(tmp_path):
    root = clone_workflows(tmp_path)
    path = root / ".github/workflows/publish.yml"
    document = yaml.safe_load(path.read_text())
    document["jobs"]["forgotten"] = {"runs-on": "ubuntu-24.04", "steps": []}
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="forgotten"):
        routes.census(root)
    path.write_bytes((ROOT / ".github/workflows/publish.yml").read_bytes())
    (path.parent / "new.yml").write_text("jobs:\n  forgotten:\n    steps: []\n")
    with pytest.raises(ValueError, match="new.yml"):
        routes.census(root)


def test_partial_or_failed_runs_never_qualify_and_receipts_bind_workflow_bytes(tmp_path):
    root = clone_workflows(tmp_path)
    results = dict.fromkeys("S I A B C D Q P R".split(), 0)
    for key in ("P", "R", "I"):
        with pytest.raises(ValueError, match=key):
            routes.record(root, "a" * 40, {**results, key: 1})
    receipt = routes.record(root, "a" * 40, results)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    routes.require_receipt(path, root, "a" * 40)
    for workflow in (root / ".github/workflows").glob("*.yml"):
        workflow.write_bytes(workflow.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    routes.require_receipt(path, root, "a" * 40)
    with pytest.raises(ValueError, match="bind"):
        routes.require_receipt(path, root, "b" * 40)
    workflow = root / ".github/workflows/publish.yml"
    workflow.write_text(workflow.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="bind"):
        routes.require_receipt(path, root, "a" * 40)
