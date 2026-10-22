from pathlib import Path

import pytest

from tools.battery.wheel_size import MAX_BYTES, main, measure


@pytest.mark.parametrize('size', [
    pytest.param(MAX_BYTES, id='100000000'),
])
def test_size_boundary_names_file_and_measured_bytes(tmp_path, size):
    wheel = tmp_path / "gpuwm-platform.whl"
    with wheel.open("wb") as stream:
        stream.truncate(size)
    if size > MAX_BYTES:
        with pytest.raises(ValueError, match=f"gpuwm-platform.whl is {size} bytes"):
            measure([wheel])
    else:
        assert measure([wheel]) == [{"path": str(wheel), "bytes": size}]


def test_no_wheel_is_not_a_pass():
    with pytest.raises(ValueError, match="no wheels"):
        measure([])


def test_cli_failure_happens_before_later_files(tmp_path, capsys):
    wheel = tmp_path / "oversize.whl"
    with wheel.open("wb") as stream:
        stream.truncate(MAX_BYTES + 1)
    assert main([str(wheel), str(tmp_path / "missing.whl")]) == 2
    assert f"oversize.whl is {MAX_BYTES + 1} bytes" in capsys.readouterr().err
