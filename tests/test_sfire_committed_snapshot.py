"""Qualification archives must identify the actual committed code they execute."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def test_committed_snapshot_excludes_later_workspace_edits(tmp_path):
    root = tmp_path / "checkout"
    (root / "gpuwm").mkdir(parents=True)
    (root / "tools/sfire_coupled_ideal").mkdir(parents=True)
    source = root / "gpuwm/physics.py"
    original = b"qualified = True\n"
    source.write_bytes(original)
    (root / "tools/sfire_coupled_ideal/run.py").write_bytes(b"print('fixture')\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "core.autocrlf", "false"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "gpuwm", "tools"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Snapshot Test",
                    "-c", "user.email=snapshot@example.invalid", "commit", "-qm", "fixture"], check=True)
    tip = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    source.write_bytes(b"unqualified = True\n")
    (root / "gpuwm/later.py").write_bytes(b"uncommitted = True\n")
    archive_path = tmp_path / "committed.tar"
    script = Path(__file__).resolve().parents[1] / "tools/sfire_coupled_ideal/package_snapshot.py"
    subprocess.run([sys.executable, str(script), str(root), str(archive_path), "--committed-tree"], check=True)
    with tarfile.open(archive_path) as archive:
        assert archive.extractfile("gpuwm/physics.py").read() == original
        assert "gpuwm/later.py" not in archive.getnames()
        receipt = json.load(archive.extractfile("SFIRE_SOURCE_PROVENANCE.json"))
    assert receipt["git_tip"] == tip
    assert receipt["dirty_selected_paths"] == []
    assert receipt["workspace_dirty_selected_paths"]
    assert receipt["engine_sources_sha256"] == {
        "gpuwm/physics.py": hashlib.sha256(original).hexdigest()}
