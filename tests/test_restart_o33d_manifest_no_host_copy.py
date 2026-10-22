"""The restart/stream manifest never materializes the legacy o33d host copy.

Named breakage (box G, 2026-10-05, 1 km HRRR physics, 2 cards): with the
legacy adapter keeping o33d on the device, every streamed sweep's inventory
read the adapter's ``_o33d_grid`` property, which copies the whole field to
the host, and then dropped the slot as restart-only.  Under host memory
pressure one of those copies held step 551 for 27 s with both cards idle.
"""

from types import SimpleNamespace

import numpy as np

from gpuwm.io.restart import radiation_o33d_carrier


class _DeviceKeepingAdapter:
    def __init__(self, device):
        self._device = device
        self.host_reads = 0

    def o33d_device(self):
        return self._device

    @property
    def _o33d_grid(self):
        self.host_reads += 1
        return np.asarray(self._device)


def test_manifest_takes_the_device_field_and_never_the_host_property():
    field = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    adapter = _DeviceKeepingAdapter(field)
    driver = SimpleNamespace(o3rad=None, radiation_callable=adapter)
    assert radiation_o33d_carrier(driver) is field
    assert adapter.host_reads == 0


def test_tree_owned_o3rad_wins_and_host_only_adapters_keep_their_field():
    o3rad = np.zeros((2, 3, 4), np.float32)
    adapter = _DeviceKeepingAdapter(np.ones((2, 3, 4), np.float32))
    assert radiation_o33d_carrier(
        SimpleNamespace(o3rad=o3rad, radiation_callable=adapter)) is o3rad
    host = np.full((2, 3, 4), 7.0, np.float32)
    legacy = SimpleNamespace(_o33d_grid=host)
    assert radiation_o33d_carrier(
        SimpleNamespace(o3rad=None, radiation_callable=legacy)) is host
    # No device field yet (e.g. just restored from a restart): the adapter's
    # own host field is the answer, read once.
    empty = _DeviceKeepingAdapter(None)
    radiation_o33d_carrier(SimpleNamespace(o3rad=None, radiation_callable=empty))
    assert empty.host_reads == 1
