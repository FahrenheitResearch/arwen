import pytest
from _release_export import private_input
from tools.battery.wheel_size import MAX_BYTES

@private_input("tools/release/cut/packages.py")
def test_cut_uses_the_same_limit_and_refusal(tmp_path):
    from tools.release.cut.packages import refuse_oversize, size_limit
    assert size_limit() == MAX_BYTES
    with pytest.raises(SystemExit, match="oversize.whl is 100000001 bytes"):
        refuse_oversize([{"path": str(tmp_path / "oversize.whl"), "bytes": MAX_BYTES + 1}], MAX_BYTES)


@private_input("tools/release/cut/wheel_size_gate.py")
@pytest.mark.parametrize("code", [0, 2])
def test_early_cut_hook_returns_before_expensive_gates(monkeypatch, code):
    from types import SimpleNamespace
    from tools.release.cut import wheel_size_gate
    calls = []
    def ssh(host, command, **kwargs):
        calls.append((host, command, kwargs))
        return SimpleNamespace(returncode=code, stdout=b"oversize.whl is 100000001 bytes", stderr=b"")
    monkeypatch.setattr(wheel_size_gate.remote, "ssh", ssh)
    result = wheel_size_gate.check("host", "python", "/owned/source", "/owned/size")
    assert len(calls) == 1
    assert "--build --build-natives" in calls[0][1]
    assert (result is None) == (code == 0)
    if code:
        assert "oversize.whl is 100000001 bytes" in result["red"][0]
