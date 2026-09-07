"""Output scratch cannot prevent a production streamed checkpoint."""
import numpy as np
import pytest

from gpuwm.io import restart
from tilestream import physics_inventory, restart_stream


class ReachedValidatedHeader(Exception):
    pass


@pytest.mark.parametrize("mode", ["complete", "output_scratch", "missing_state", "unknown_scratch"])
def test_streamed_checkpoint_classifies_extra_output_without_dropping_state(monkeypatch, tmp_path, mode):
    value = np.zeros((2, 2), dtype=np.float32)
    arrays = {"state/thp": value}
    if mode == "output_scratch":
        arrays["scratch/refl_10cm"] = value.copy()
    elif mode == "missing_state":
        arrays = {"scratch/refl_10cm": value.copy()}
    elif mode == "unknown_scratch":
        arrays["scratch/unclassified_future_carrier"] = value.copy()
    monkeypatch.setattr(physics_inventory, "carrier_manifest", lambda state: {"state/thp": value})

    def inspect_header(setup, template, stored, elapsed):
        assert set(stored) == {"state/thp"}
        raise ReachedValidatedHeader

    monkeypatch.setattr(restart_stream, "domain_header_view", inspect_header)
    expected = (restart_stream.RestartRefused if mode == "missing_state"
                else restart.RestartManifestError if mode == "unknown_scratch"
                else ReachedValidatedHeader)
    with pytest.raises(expected):
        restart_stream.write_streamed_restart(
            tmp_path / "restart.npz", arrays, object(), setup=object(),
            template_state=object(), scalars={"elapsed_seconds": 0.0}, check_pinned=False)
    assert not (tmp_path / "restart.npz").exists()
