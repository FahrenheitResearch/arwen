"""Remote control never interpolates request data into a shell or trusts prose."""
from argparse import Namespace
import json
import os
from pathlib import Path
import shlex
import sys

import pytest

from gpuwm import remote_cli as rc


def args(**overrides):
    return Namespace(**{**dict(host="weather-node", python="/opt/ArWen's runtime/bin/python",
        workspace="/srv/weather space", port=None, identity=None, ssh_config=None,
        remote_action="probe", json=True), **overrides})


@pytest.mark.parametrize("host", ["-oProxyCommand=x", "node;echo", "node\nnext", "$(secret)", "node x", "", "node`x`"])
def test_host_cannot_add_ssh_options_or_shell_words(host):
    with pytest.raises(ValueError, match="--host"):
        rc.ssh_command(args(host=host))


def test_only_quoted_python_and_fixed_protocol_words_enter_remote_shell(monkeypatch, tmp_path):
    config = tmp_path / "a config"
    config.write_text("Host weather-node\n")
    key = tmp_path / "identity"
    key.write_bytes(b"fixture placeholder, not read by this test")
    monkeypatch.setattr(rc.shutil, "which", lambda name: "ssh-fixture")
    options = args(ssh_config=str(config), identity=str(key), port=2222)
    command = rc.ssh_command(options)
    assert command[-3:-1] == ["--", "weather-node"]
    assert shlex.split(command[-1]) == [options.python, "-I", "-m", "gpuwm.remote_worker", "--rpc"]
    assert options.workspace not in command[-1]
    assert "BatchMode=yes" in command and "StrictHostKeyChecking=yes" in command
    assert command[command.index("-F") + 1] == str(config.resolve())
    assert command[command.index("-i") + 1] == str(key.resolve())
    assert command[command.index("-p") + 1] == "2222"


def test_windows_prefers_the_system_openssh_client_and_the_record_names_it(monkeypatch, tmp_path):
    system = tmp_path / "Windows"
    openssh = system / "System32" / "OpenSSH" / "ssh.exe"
    openssh.parent.mkdir(parents=True)
    openssh.write_bytes(b"")
    git_bin = tmp_path / "Git" / "usr" / "bin"
    git_bin.mkdir(parents=True)
    (git_bin / "ssh.EXE").write_bytes(b"")
    environ = {"SystemRoot": str(system), "PATH": str(git_bin), "PATHEXT": ".EXE;.BAT"}
    assert rc.ssh_executable(environ=environ, windows=True) == str(openssh)
    # Without the system client, PATH order decides, as before.
    openssh.unlink()
    assert Path(rc.ssh_executable(environ=environ, windows=True)) == git_bin / "ssh.EXE"
    # Linux never consults SystemRoot.
    assert Path(rc.ssh_executable(environ=environ, windows=False)) == git_bin / "ssh.EXE"
    # The resolved client rides on every record, including a refusal.
    monkeypatch.setattr(rc, "ssh_executable", lambda: str(openssh))
    command = rc.ssh_command(args())
    assert command[0] == str(openssh)


def test_records_and_ssh_level_failures_name_the_client_that_ran(monkeypatch, capsys, tmp_path):
    transport = rc._transport
    monkeypatch.setattr(rc, "ssh_executable", lambda: "C:/Windows/System32/OpenSSH/ssh.exe")
    monkeypatch.setattr(rc, "_transport", lambda command, request, **kwargs: rc.result("probe", runtime={}, capabilities={}))
    assert rc.remote_main(args()) == 0
    reply = json.loads(capsys.readouterr().out.splitlines()[0])
    assert reply["ok"] is True and reply["ssh_client"] == "C:/Windows/System32/OpenSSH/ssh.exe"
    # An authentication failure is an SSH-level failure: the node never
    # answered, and the message says which client asked.
    monkeypatch.setattr(rc, "_transport", transport)
    program = _program(tmp_path, "sys.stderr.write('Permission denied (publickey).\\n')\nraise SystemExit(255)\n")
    with pytest.raises(ValueError, match=r"Permission denied \(publickey\).*\(ssh client: ") as failure:
        rc._transport(program, {"action": "probe"})
    assert str(failure.value).endswith(f"(ssh client: {program[0]})")
    monkeypatch.setattr(rc, "ssh_command", lambda options, **kwargs: program)
    assert rc.remote_main(args()) == 2
    reply = json.loads(capsys.readouterr().out.splitlines()[0])
    assert reply["ok"] is False and reply["ssh_client"] == program[0]
    assert "Permission denied (publickey)" in reply["error"]["message"] and program[0] in reply["error"]["message"]


@pytest.mark.parametrize("field", ["ssh_config", "identity"])
def test_relative_ssh_config_and_identity_paths_are_refused(monkeypatch, tmp_path, field):
    (tmp_path / "sshconf").write_text("Host *\n  ProxyCommand planted\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rc.shutil, "which", lambda name: "ssh-fixture")
    with pytest.raises(ValueError, match="absolute local path"):
        rc.ssh_command(args(**{field: "sshconf"}))
    command = rc.ssh_command(args(**{field: str(tmp_path / "sshconf")}))
    assert command[command.index("-F" if field == "ssh_config" else "-i") + 1] == str((tmp_path / "sshconf").resolve())


@pytest.mark.parametrize("python,workspace,port", [("python", "/srv/x", None),
    ("/bin/python\nnext", "/srv/x", None), ("/bin/python", "relative", None),
    ("/bin/python", "/srv/x", 0), ("/bin/python", "/srv/x", 65536)])
def test_transport_profile_validation(monkeypatch, python, workspace, port):
    monkeypatch.setattr(rc.shutil, "which", lambda name: "ssh")
    with pytest.raises(ValueError):
        rc.ssh_command(args(python=python, workspace=workspace, port=port))


def _program(tmp_path, body):
    script = tmp_path / "ssh_fixture.py"
    script.write_text("import sys,json,time\nrequest=json.loads(sys.stdin.buffer.read())\n" + body, encoding="utf-8")
    return [sys.executable, "-u", str(script)]


def test_transport_preserves_request_and_parses_only_one_envelope(tmp_path):
    request = {"action": "start", "config": "/tmp/' $(unchanged) 日本語.toml"}
    command = _program(tmp_path, "print(json.dumps({'schema':'gpuwm.remote.result.v1','ok':True,'action':request['action'],'request':request}))\n")
    assert rc._transport(command, request)["request"] == request


@pytest.mark.parametrize("body,match", [
    ("print('plain prose')", "Expecting value"),
    ("print('{}\\n{}')", "one ArWen response"),
    ("print(json.dumps({'schema':'wrong','ok':True,'action':'probe'}))", "incompatible"),
    ("print(json.dumps({'schema':'gpuwm.remote.result.v1','ok':True,'action':'start'}))", "incompatible"),
    ("print(json.dumps({'schema':'gpuwm.remote.result.v1','ok':True,'action':'probe'}));sys.exit(7)", "disagrees"),
    ("sys.stderr.write('Permission denied (publickey).');sys.exit(255)", "Permission denied"),
    ("sys.stdout.write('x'*200000)", "bounded protocol"),
    ("sys.stderr.write('x'*20000)", "bounded protocol"),
])
def test_transport_refuses_protocol_and_ssh_errors(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        rc._transport(_program(tmp_path, body), {"action": "probe"})


def test_transport_timeout_does_not_claim_start_did_not_happen(tmp_path):
    with pytest.raises(ValueError, match="start may already exist"):
        rc._transport(_program(tmp_path, "time.sleep(10)"), {"action": "start"}, timeout=.1)


def test_refusal_is_one_json_line_with_exit_two(monkeypatch, capsys):
    assert rc.remote_main(args(host="-oops")) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    reply = json.loads(lines[0])
    assert reply["schema"] == rc.SCHEMA and reply["action"] == "probe" and reply["ok"] is False


def test_cli_forwards_binding_and_paths_only_as_request_data(monkeypatch, capsys):
    monkeypatch.setattr(rc, "ssh_command", lambda options: ["ssh-fixture"])
    seen = []
    monkeypatch.setattr(rc, "_transport", lambda command, request: seen.append((command, request)) or rc.result("start", dry_run=True))
    options = args(remote_action="start", config="/srv/a path.toml", outdir="/srv/new output",
                   geog_root="/srv/geography", products="t2,wind10", dry_run=True,
                   expected_input_sha256="1" * 64)
    assert rc.remote_main(options) == 0
    assert seen[0][0] == ["ssh-fixture"]
    assert seen[0][1]["expected_input_sha256"] == "1" * 64
    assert seen[0][1]["config"] == options.config
    assert seen[0][1]["products"] == options.products
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_public_parser_has_review_and_reconnect_options():
    from gpuwm.cli import build_parser
    parser = build_parser()
    options = parser.parse_args(["remote", "resume", "--host", "node", "--python", "/opt/python",
        "--workspace", "/work", "--job", "valid_123", "--outdir", "/new-output", "--from", "latest",
        "--dry-run", "--expected-input-sha256", "a" * 64, "--json"])
    assert options.func is rc.remote_main
    assert options.from_checkpoint == "latest" and options.dry_run and options.json


def test_artifact_parser_and_fixed_binary_stream_keep_selectors_off_the_shell(monkeypatch):
    from gpuwm.cli import build_parser
    options = build_parser().parse_args(["remote", "sync-artifacts", "--host", "node", "--python", "/opt/python",
        "--workspace", "/owned/work", "--job", "job-fixture", "--domain", "2", "--cache-root", "C:/owned/cache", "--json"])
    assert options.domain == 2 and options.cache_root == "C:/owned/cache"
    monkeypatch.setattr(rc.shutil, "which", lambda _: "ssh-fixture")
    command = rc.ssh_command(options, artifact_stream=True)
    assert shlex.split(command[-1]) == [options.python, "-I", "-m", "gpuwm.remote_worker", "--artifact-stream"]
    assert options.job not in command[-1] and options.cache_root not in command[-1]
