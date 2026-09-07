"""Canonical, bounded device operands for streamed nest transactions.

The live store owns prognostics, and the geography inventory owns horizontal
setup. A slab template is only a source for vertical arrays and scalars.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class NestWindowSource:
    """Drain once, then read exact windows from a domain's canonical arrays."""

    def __init__(self, state):
        self.state = state
        self.owner = getattr(state, "_streamed_domain", None)
        self.store = self.owner.store if self.owner is not None else None
        self.geography = (getattr(self.owner, "_geography", None) or {})
        self.template = (getattr(self.owner, "template_state", None) or state)
        self.host_to_device_bytes = 0
        self.device_to_host_bytes = 0
        self.max_operand_bytes = 0
        self.max_host_scratch_bytes = 0

    def array(self, name):
        if self.store is not None:
            key = name if "/" in name else f"state/{name}"
            if key in self.store:
                return self.store[key]
            key = f"setup/{name}"
            if key in self.geography:
                return self.geography[key]
            value = getattr(self.template, name, None)
            if getattr(value, "ndim", 0) >= 2:
                raise RuntimeError(
                    f"canonical nest operand {name!r} is missing; a slab "
                    "template cannot supply horizontal domain data")
            return value
        return getattr(self.state, name, None)

    def device(self, name, window=None):
        return self.device_array(self.array(name), window)

    def device_array(self, value, window=None):
        """Pack one bounded view, including a separately inventoried carrier."""
        if value is None or not hasattr(value, "ndim") or np.isscalar(value):
            return value
        import cupy as cp
        if window is not None and value.ndim >= 2:
            value = value[(...,) + window]
        if isinstance(value, np.ndarray):
            self.host_to_device_bytes += int(value.nbytes)
            result = cp.asarray(np.ascontiguousarray(value))
        else:
            result = cp.ascontiguousarray(value)
        self.max_operand_bytes = max(self.max_operand_bytes, int(result.nbytes))
        return result

    def write(self, name, window, value):
        import cupy as cp

        target = self.array(name)[(...,) + window]
        if isinstance(target, np.ndarray):
            # cp.asnumpy packs only this bounded result, never a full field.
            target[...] = cp.asnumpy(value)
            self.device_to_host_bytes += int(value.nbytes)
        else:
            target[...] = value

    def coupled(self, kind, window):
        """Run the unchanged coupling kernel on a one-cell mass halo.

        The extra mass row/column supplies face averages at artificial
        window edges. Only true physical edges use the kernel's clamp.
        """
        import cupy as cp
        from gpuwm.ingest.lateral_bc import couple_nest_field

        ny, nx = self.array("mup").shape
        y, x = window
        j0, j1 = max(0, y.start - 1), min(ny, y.stop + 1)
        i0, i1 = max(0, x.start - 1), min(nx, x.stop + 1)
        local = SimpleNamespace()
        for name in ("mup", "mub2d", "thb", "c1h", "c2h", "c1f", "c2f",
                     "msft", "msfu", "msfv", "has_msf"):
            extra_y, extra_x = (int(name == "msfv"), int(name == "msfu"))
            setattr(local, name, self.device(name, (
                slice(j0, j1 + extra_y), slice(i0, i1 + extra_x))))
        attr = {"t": "thp", "ph": "php", "mu": "mup"}.get(kind, kind)
        if kind != "mu":
            setattr(local, attr, self.device(attr, (
                slice(j0, j1 + int(kind == "v")),
                slice(i0, i1 + int(kind == "u")))))
        field = local.mup[None] if kind == "mu" else getattr(local, attr)
        out = cp.empty(field.shape, dtype=cp.float32)
        couple_nest_field(local, kind, out=out)
        return cp.ascontiguousarray(out[
            :, y.start - j0:y.stop - j0, x.start - i0:x.stop - i0])

    def raw(self, kind, window):
        """Read WRF's uncoupled feedback spelling, including theta - 300."""
        attr = {"t": "thp", "ph": "php", "mu": "mup"}.get(kind, kind)
        value = self.device(attr, window)
        if kind == "t":
            value = value.copy()
            thb = self.device("thb", window)
            value += thb if thb.ndim == 3 else thb[:, None, None]
            value -= np.float32(300.0)
        return value


def streamed_chunk_shape(state):
    """Use the admitted tile dimensions, never an unpriced full field."""
    owner = getattr(state, "_streamed_domain", None)
    decision = getattr(owner, "decision", None)
    ny, nx = (getattr(decision, "tile_ny", None),
              getattr(decision, "tile_nx", None))
    if ny is None or nx is None or int(ny) < 1 or int(nx) < 1:
        raise RuntimeError("bounded nest coupling needs admitted tile dimensions")
    return int(ny), int(nx)


def boundary_windows(reg, width, chunk_shape):
    """Yield side, exact child rectangle and rolling-table destination."""
    sy, sx = chunk_shape
    for side in ("west", "east"):
        x = slice(0, width) if side == "west" else slice(reg.nxc-width, reg.nxc)
        for j in range(0, reg.nyc, sy):
            y = slice(j, min(j + sy, reg.nyc))
            yield side, (y, x), (slice(None), y, slice(None))
    for side in ("south", "north"):
        y = slice(0, width) if side == "south" else slice(reg.nyc-width, reg.nyc)
        for i in range(0, reg.nxc, sx):
            x = slice(i, min(i + sx, reg.nxc))
            yield side, (y, x), (slice(None), slice(None), x)


def rectangle_windows(window, chunk_shape):
    y, x = window
    sy, sx = chunk_shape
    for j in range(y.start, y.stop, sy):
        for i in range(x.start, x.stop, sx):
            yield (slice(j, min(j+sy, y.stop)), slice(i, min(i+sx, x.stop)))


def smooth_canonical_parent(source, kind, reg, *, smooth_option, chunk_shape):
    """Use the original two-stage CUDA smoother with a host J-pass owner.

    Every J output exists before any I output overwrites the source. The
    single host scratch rectangle retains the original one-cell outer ring.
    This is an explicit additional host allocation, reported on ``source``;
    device operands remain at most one tile plus that ring.
    """
    import cupy as cp
    from gpuwm.core.nest_interp import (
        SMDSM_XNU, _kernel, _launch, smoother_parent_window)

    i0, j0, niw, njw = smoother_parent_window(reg)
    if smooth_option == 0 or niw <= 0 or njw <= 0:
        return
    if smooth_option not in (1, 2):
        raise ValueError("smooth_option must be 0, 1 or 2")
    attr = {"t": "thp", "ph": "php", "mu": "mup"}.get(kind, kind)
    field = source.array(attr)
    nz = field.shape[0] if field.ndim == 3 else 1
    snapshot = np.empty((nz, njw+2, niw+2), dtype=np.float32)
    source.max_host_scratch_bytes = max(source.max_host_scratch_bytes, snapshot.nbytes)
    output = (slice(j0, j0+njw), slice(i0, i0+niw))
    ring = (slice(j0-1, j0+njw+1), slice(i0-1, i0+niw+1))
    mode = np.int32(smooth_option == 2)
    passes = SMDSM_XNU if smooth_option == 2 else (np.float32(0.0),)
    for xnu in passes:
        # Preserve the exact global ring. A device store is staged in the
        # same bounded chunks; a host store needs no device roundtrip.
        if isinstance(field, np.ndarray):
            snapshot[...] = field[(...,)+ring]
        else:
            for win in rectangle_windows(ring, chunk_shape):
                y, x = win
                value = source.device(attr, win)
                snapshot[:, y.start-j0+1:y.stop-j0+1,
                         x.start-i0+1:x.stop-i0+1] = cp.asnumpy(value)
                source.device_to_host_bytes += value.nbytes
        for win in rectangle_windows(output, chunk_shape):
            y, x = win
            ny, nx = y.stop-y.start, x.stop-x.start
            halo = (slice(y.start-1, y.stop+1), slice(x.start-1, x.stop+1))
            values = source.device(attr, halo)
            work = cp.empty((nz, ny+2, nx+2), dtype=cp.float32)
            dims = tuple(np.int32(v) for v in (1, 1, nx, ny, nz, ny+2, nx+2))
            _launch(_kernel("nest_smooth_j"), work.size,
                    (values, work, mode, xnu, *dims))
            block = cp.ascontiguousarray(work[:, 1:-1, 1:-1])
            snapshot[:, y.start-j0+1:y.stop-j0+1,
                     x.start-i0+1:x.stop-i0+1] = cp.asnumpy(block)
            source.device_to_host_bytes += block.nbytes
        for win in rectangle_windows(output, chunk_shape):
            y, x = win
            ny, nx = y.stop-y.start, x.stop-x.start
            values = source.device_array(snapshot[:, y.start-j0:y.stop-j0+2,
                                                   x.start-i0:x.stop-i0+2])
            work = cp.empty(values.shape, dtype=cp.float32)
            dims = tuple(np.int32(v) for v in (1, 1, nx, ny, nz, ny+2, nx+2))
            _launch(_kernel("nest_smooth_i"), nz*ny*nx,
                    (work, values, mode, xnu, *dims))
            result = work[:, 1:-1, 1:-1]
            source.write(attr, win, result[0] if field.ndim == 2 else result)


def diagnose_canonical_parent(source, window, *, hypsometric_opt, chunk_shape):
    """Recompute changed columns using the unchanged local diagnostics kernel."""
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics

    names = ("thp", "php", "mup", "qv", "thb", "phb", "dphb_resid", "alb",
             "rdnw", "c1h", "c2h", "c3h", "c4h", "c3f", "c4f",
             "dc3f", "dc4f", "mub2d", "p_top")
    nz = source.array("p").shape[0]
    for win in rectangle_windows(window, chunk_shape):
        y, x = win
        local = SimpleNamespace(**{name: source.device(name, win) for name in names})
        for name in ("p", "al", "alt"):
            setattr(local, name, cp.empty((nz, y.stop-y.start, x.stop-x.start),
                                         dtype=cp.float32))
        update_diagnostics(local, hypsometric_opt)
        for name in ("p", "al", "alt"):
            source.write(name, win, getattr(local, name))
