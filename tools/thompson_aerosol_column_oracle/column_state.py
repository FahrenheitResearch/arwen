"""Array-backed DomainState shared by the Thompson column oracles."""

from types import SimpleNamespace

import numpy as np


class ColumnState:
    """Keep oracle layouts and scratch slots consistent on NumPy and CuPy.

    Host parity runs copy their inputs, including single-column views.
    Classic oracles allocate zeros on the host before uploading them.
    """

    _scratch_error = " reused with a new shape or dtype"

    def __init__(self, cols, xp, species, *, copy_inputs=False,
                 host_zeros=False):
        ncol, nz = cols["p"].shape
        self._allocate = (lambda shape, dtype: xp.asarray(np.zeros(shape, dtype))
                          if host_zeros else xp.zeros(shape, dtype))

        def vol(a):
            if copy_inputs:
                a = np.array(a.T.reshape(a.shape[1], 1, ncol),
                             dtype=np.float32, order="C", copy=True)
            else:
                a = np.asarray(a, np.float32)
                a = np.ascontiguousarray(a.T.reshape(a.shape[1], 1, ncol))
            return xp.asarray(a)

        self.p = vol(cols["p"])
        self.thb = self._allocate((nz,), np.float32)
        self.thp = vol(cols["th"])
        self.phb = self._allocate((nz + 1,), np.float32)
        self.php = vol(cols["geop"])
        self.w = vol(cols["w"])
        for name in species:
            setattr(self, name, vol(cols[name]))
        for name in ("effc", "effi", "effs", "h_diabatic"):
            setattr(self, name, self._allocate((nz, 1, ncol), np.float32))
        if "nwfa" in species:
            for name in ("nwfa2d", "nifa2d"):
                a = cols[name].reshape(1, ncol)
                a = (np.array(a, dtype=np.float32, copy=True) if copy_inputs
                     else np.asarray(a, np.float32))
                setattr(self, name, xp.asarray(a))
        self.physics = SimpleNamespace(refl_10cm=None, state=self)
        self._scratch = {}

    def scratch(self, shape, slot, dtype=None):
        shape = tuple(shape)
        want = np.dtype(np.float32 if dtype is None else dtype)
        value = self._scratch.get(slot)
        if value is None:
            value = self._allocate(shape, want)
            self._scratch[slot] = value
        elif value.shape != shape or value.dtype != want:
            raise ValueError(f"scratch slot {slot!r}" + self._scratch_error)
        return value

    def existing_scratch(self, slot):
        return self._scratch.get(slot)
