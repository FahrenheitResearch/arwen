"""Raw transport protocol fixtures only; no weather fields are fabricated."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from gpuwm import remote_input_transfer as transfer, remote_cli, remote_plan


def request(workspace, payload):
    return {"schema": "gpuwm.remote.request.v1", "action": "put-input", "workspace": str(workspace),
            "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_receive_publishes_only_verified_raw_bytes_and_reuses_exact_object(tmp_path):
    raw = b"raw protocol fixture" * 4000
    value = request(tmp_path, raw)
    first = transfer.receive(value, tmp_path, io.BytesIO(raw))
    path = tmp_path / ".arwen-input-cache" / value["sha256"]
    assert path.read_bytes() == raw and first["reused"] is False
    assert transfer.receive(value, tmp_path, io.BytesIO(raw))["reused"] is True
    state = transfer.status({"schema": value["schema"], "action": "input-status", "workspace": str(tmp_path),
                             "inputs": [{key: value[key] for key in ("size", "sha256")}]}, tmp_path)
    assert state["inputs"] == [{"size": len(raw), "sha256": value["sha256"], "ready": True}]


@pytest.mark.parametrize("raw", [b"short", b"A" * 30, b"B" * 20])
def test_partial_excess_or_changed_stream_never_publishes(tmp_path, raw):
    value = request(tmp_path, b"A" * 20)
    with pytest.raises(ValueError):
        transfer.receive(value, tmp_path, io.BytesIO(raw))
    assert not (tmp_path / ".arwen-input-cache" / value["sha256"]).exists()
    assert not list((tmp_path / ".arwen-input-cache").glob("*.part"))


@pytest.mark.parametrize("change", [{"path": "/outside"}, {"size": True}, {"size": transfer.MAX_BLOB_BYTES + 1}, {"sha256": "bad"}])
def test_stream_identity_refuses_arbitrary_paths_and_unbounded_sizes(tmp_path, change):
    value = {**request(tmp_path, b"raw"), **change}
    with pytest.raises(ValueError):
        transfer.receive(value, tmp_path, io.BytesIO(b"raw"))


def test_changed_local_source_is_refused_before_launch_transport(tmp_path):
    source = tmp_path / "selected.grib"
    source.write_bytes(b"original raw fixture")
    record = transfer.describe(source)
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema": "arwen.companion-remote-review.v1", "remote_review": {"source_blobs": [record]}}))
    expected = hashlib.sha256(transfer._encoded([record])).hexdigest()
    assert transfer.verify_review_file(review) == expected
    source.write_bytes(b"different raw fixture")
    with pytest.raises(ValueError, match="changed"):
        transfer.verify_review_file(review)


def test_real_binary_pipe_preserves_inputs_and_repeat_transfer_uses_cache(tmp_path, monkeypatch):
    source = tmp_path / "selected.grib"
    raw = b"no meteorology; exact binary input transport" * 5000
    source.write_bytes(raw)
    node = tmp_path / "node"
    node.mkdir()
    value = request(node, raw)
    program = "import json,sys;from pathlib import Path;from gpuwm.remote_input_transfer import receive;v=json.loads(sys.stdin.buffer.readline());r=receive(v,Path(v['workspace']),sys.stdin.buffer);print(json.dumps(dict(schema='gpuwm.remote.result.v1',ok=True,action='put-input',**r)))"
    assert transfer.upload([sys.executable, "-c", program], value, source, timeout=10)["sha256"] == value["sha256"]
    bundle = {"workspace": str(node), "blobs": [{**transfer.describe(source), "name": "selected.grib", "role": "forcing", "placement": "inputs"}]}
    monkeypatch.setattr(remote_cli, "_transport", lambda _command, query, **_: {"ok": True, **transfer.status(query, node)})
    monkeypatch.setattr(transfer, "upload", lambda *_args, **_kwargs: pytest.fail("unchanged raw object must not transfer again"))
    assert transfer.transfer_bundle_inputs(bundle, [], []) == 0
    assert (node / ".arwen-input-cache" / value["sha256"]).read_bytes() == raw


def test_source_manifest_limits_and_duplicate_paths_are_checked(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"fixture")
    record = transfer.describe(path)
    with pytest.raises(ValueError, match="unique"):
        transfer.verify_sources([record, record])
    invalid = dict(record, size=transfer.MAX_BLOB_BYTES + 1)
    with pytest.raises(ValueError, match="bounded"):
        transfer.verify_sources([invalid])


def test_binary_command_has_only_fixed_words_and_never_local_paths(monkeypatch):
    import shlex
    monkeypatch.setattr(remote_cli.shutil, "which", lambda _: "ssh")
    args = SimpleNamespace(host="node", python="/runtime with space/python", workspace="/owned/work",
                           port=None, identity=None, ssh_config=None)
    command = remote_cli.ssh_command(args, input_stream=True)
    assert shlex.split(command[-1]) == [args.python, "-I", "-m", "gpuwm.remote_worker", "--input-stream"]
    assert args.workspace not in command[-1]
