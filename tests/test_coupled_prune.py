import json
import pytest
from tools.coupled_prune import prune


def setup(root):
    root.joinpath("MANIFEST.txt").write_text(str(root) + "\n")
    shard = root / "replay/shard-1"
    result = shard / "results/1of48.jsonl"
    result.parent.mkdir(parents=True)
    raw = shard / "woof/case/combo/step0001.bin"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"measured array")
    return raw, result


def test_only_a_run_with_a_verdict_can_be_pruned(tmp_path):
    raw, result = setup(tmp_path)
    result.write_text('{"case":')
    assert prune(tmp_path) == []
    assert raw.exists()
    result.write_text(json.dumps(dict(case="case", combo="combo", outcome="FAIL")) + "\n")
    rows = prune(tmp_path)
    assert rows == [dict(path=str(raw), bytes=14)]
    assert not raw.exists()
    assert str(raw) in (tmp_path / "MANIFEST.txt").read_text()


def test_root_requires_owned_manifest(tmp_path):
    with pytest.raises(ValueError, match="owned lane root"):
        prune(tmp_path)
