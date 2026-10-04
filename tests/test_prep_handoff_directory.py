"""Native acquisition handoffs retain their published artifact authority."""
import hashlib
import json

import pytest

from gpuwm.prep_handoff import preparation_arguments_from_directory


def test_old_bound_command_is_parsed_without_shell_execution(tmp_path):
    command = tmp_path/"prep-command.txt"
    command.write_text("# verified native inputs\ngpuwm prep \\\n  --source gfs \\\n  --gfs-series '/fixture with spaces/gfs-series.tsv' \\\n  --cycle 2024-01-25_00:00:00\n")
    manifest = {"source":"gfs", "files":[{"role":"prep-command", "name":command.name,
                 "sha256":hashlib.sha256(command.read_bytes()).hexdigest()}]}
    (tmp_path/"fetch-manifest.json").write_text(json.dumps(manifest))
    assert preparation_arguments_from_directory(tmp_path)==[
        "--source","gfs","--gfs-series","/fixture with spaces/gfs-series.tsv",
        "--cycle","2024-01-25_00:00:00"]
    command.write_text(command.read_text().replace("gfs-series.tsv","changed.tsv"))
    with pytest.raises(ValueError,match="differs from its acquisition"):
        preparation_arguments_from_directory(tmp_path)


def test_structured_member_handoff_keeps_native_verification(tmp_path,monkeypatch):
    from gpuwm import prep_handoff
    document={"argv":["--source","gefs"],"member_verification":{"set":"test","member":"p01"}}
    (tmp_path/"prep-arguments.json").write_text(json.dumps(document))
    def refuse(*args):raise ValueError("native member mismatch")
    monkeypatch.setattr(prep_handoff,"_verify_selected_inputs",refuse)
    with pytest.raises(ValueError,match="native member mismatch"):
        preparation_arguments_from_directory(tmp_path)
