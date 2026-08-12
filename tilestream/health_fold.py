"""The run loop's stability report, folded per tile inside the sweep.

WHY THIS EXISTS
---------------
``gpuwm.runtime.integrate_prepared_case`` guards every dynamics substep with
``stability_report(state, ...)`` (runtime.py:2234) and raises when it reports a
non-finite state.  Under ``[tiles]`` with a host store the domain's arrays
are NOT on that state: :func:`gpuwm.core.streaming.attach` copies the prepared
state's carriers into pinned host memory and every model step is a sweep of
tiles through the card.  The state is never written again, so the report is
taken on a corpse -- healthy at t=0 and healthy forever.  ``nan_free`` stays
``True`` through a run that went entirely non-finite in the store, ``w_max``
stays at its t=0 value, and the run writes a checkpoint and calls itself a
forecast.  MEASURED: a 128 x 128 x 49 streamed run whose store held 819 196
non-finite ``w`` cells out of 819 200 completed 20 of 20 steps reporting
``nan_free=True`` and a constant ``w_max``.

It is also a per-substep GPU reduction over the whole resident domain that
nobody is using -- so the defect is simultaneously a wrong answer and a tax.

WHY FOLDING IS ALLOWED
----------------------
Everything in the report is a MAX fold (``u_max``, ``w_max``, ``th_max``, the
boundary and free-interior ``w`` maxima, and the co-located ``|w|/dz`` rate) or
an OR fold (the per-field NaN flags).  Both are associative, commutative and
idempotent, and float max is EXACT -- it selects an operand and never rounds --
so the per-tile fold is bit-identical to the whole-domain reduction rather than
close to it.  The blocking is irrelevant for the same reason, including the
argmax: ties are broken by lowest DOMAIN index, which is a property of the
data, not of the launch geometry.

The defect was which memory the reduction read, not the mathematics.

WHAT IS EASY TO GET WRONG, AND IS THEREFORE A PARAMETER HERE
------------------------------------------------------------
* **Interiors, never halos.**  The window is :meth:`TileSpec.interior_in_tile`
  -- the exact set the scatter writes -- so the union over a plan covers the
  domain once.  A tile's halo holds a neighbour's cells it did not integrate,
  and at a domain edge it holds seam fill; on a ``poison`` seam, folding the
  halo would report NaN on a perfectly healthy forecast.
* **Domain indices.**  ``w_argmax`` and the boundary/interior split are
  computed from ``(spec.j0 + jj, spec.i0 + ii)``, not from tile-local indices.
* **The closing staggered face.**  ``interior_in_tile("u")`` already adds the
  ``nx`` slot to whichever tile owns it, so the fold asks ``spec`` instead of
  re-deriving ownership and dropping a domain-edge face.
"""

from __future__ import annotations

import math

import numpy as np

#: Threads per block in ``health.cu``; the kernel's shared arrays are sized
#: with it, so it is not tunable from here.
HEALTH_THREADS = 256
#: Partial-record width in ``health.cu``: six maxima, two index words, a mask.
HEALTH_FIELDS = 9
#: Blocks per tile.  Fixed, so a sweep's partial records form one contiguous
#: array that ``health_final`` reduces in a single launch.  The VALUE does not
#: affect the answer -- see the module docstring on blocking.
BLOCKS_PER_TILE = 256


class TileHealthFold:
    """One sweep's stability report, accumulated tile by tile.

    Usage mirrors the sweep::

        fold.begin()                       # once per sweep
        for each tile:
            fold.tile(tile_state, spec, itile, stream)   # after its step
        report = fold.finish()             # after the streams join

    ``finish`` returns exactly the dict
    :func:`gpuwm.core.dycore.stability_report` returns for the same domain, so
    the run loop's accounting does not have to know which one it got.
    """

    #: Deliberate breakages, for the negative controls.  Each disables
    #: exactly one of the three properties the module docstring calls
    #: load-bearing, and each MUST make the fold differ from the resident
    #: reduction -- a control that cannot fire proves nothing.
    #:
    #:   ``"halo"``       fold the whole compute window, halo included
    #:   ``"tileindex"``  compare and classify by TILE-local indices
    #:   ``"dropface"``   fold u over the mass window, dropping the closing
    #:                    face the east-edge tile owns
    CONTROLS = ("halo", "tileindex", "dropface")

    def __init__(self, cfg, ntiles: int, *, boundary_width: int | None = None,
                 control: str | None = None):
        import cupy as cp

        if control is not None and control not in self.CONTROLS:
            raise ValueError(f"unknown control {control!r}; expected one of "
                             f"{list(self.CONTROLS)}")
        self.control = control
        #: Turned off only to PRICE the fold: the same run, same tiles, same
        #: everything, with the reduction not launched.  A forecast never
        #: sets this -- an ungated streamed run is the defect this module
        #: exists to remove.
        self.enabled = True
        self.cfg = cfg
        self.ntiles = int(ntiles)
        self.width = 0 if boundary_width is None else int(boundary_width)
        if boundary_width is not None and self.width <= 0:
            raise ValueError("boundary_width must be positive")
        self._partial = cp.zeros(
            (self.ntiles * BLOCKS_PER_TILE, HEALTH_FIELDS), dtype=cp.float32)
        self._result = cp.zeros((8,), dtype=cp.float32)
        self._swdown = cp.zeros((self.ntiles,), dtype=cp.float32)
        self._have_swdown = False
        self._folded = 0

    # -- per sweep ---------------------------------------------------------

    def begin(self) -> None:
        """Reset the accumulators for a new sweep.

        The partial buffer is zeroed rather than left dirty: a block that a
        ragged tile leaves unwritten would otherwise carry the PREVIOUS
        sweep's maximum into this one, which is exactly the kind of defect
        that looks like a plausible forecast.
        """
        self._partial.fill(0)
        self._swdown.fill(0)
        self._have_swdown = False
        self._folded = 0

    def tile(self, state, spec, itile: int, stream=None) -> None:
        """Fold one tile's INTERIOR into this sweep's partial records."""
        from gpuwm.core.kernels import get_kernel
        import gpuwm.core.constants as c

        ys, xs = spec.interior_in_tile("mass")
        uys, uxs = spec.interior_in_tile("u")
        tny, tnx = (int(v) for v in state.thp.shape[1:])
        nz = int(state.thp.shape[0])
        dj0, di0 = int(spec.j0), int(spec.i0)
        if self.control == "halo":
            ys = xs = None
            ys, xs = slice(0, tny), slice(0, tnx)
            uys, uxs = slice(0, tny), slice(0, tnx + 1)
            dj0, di0 = int(spec.cj0), int(spec.ci0)
        elif self.control == "tileindex":
            dj0 = di0 = 0
        elif self.control == "dropface":
            uys, uxs = ys, xs
        phb_full = int(state.phb.ndim == 3)
        kernel = get_kernel("health", "health_partial_tile")
        view = self._partial[itile * BLOCKS_PER_TILE:
                             (itile + 1) * BLOCKS_PER_TILE]
        args = (state.u, state.w, state.thp, state.php, state.phb, view,
                np.int32(tny), np.int32(tnx),
                np.int32(ys.start), np.int32(ys.stop - ys.start),
                np.int32(xs.start), np.int32(xs.stop - xs.start),
                np.int32(uys.start), np.int32(uys.stop - uys.start),
                np.int32(uxs.start), np.int32(uxs.stop - uxs.start),
                np.int32(nz), np.int32(dj0), np.int32(di0),
                np.int32(spec.ny), np.int32(spec.nx), np.int32(self.width),
                np.int32(phb_full), np.int32(1), np.float32(c.G))
        if stream is not None:
            with stream:
                kernel((BLOCKS_PER_TILE,), (HEALTH_THREADS,), args)
                self._fold_swdown(state, ys, xs, itile)
        else:
            kernel((BLOCKS_PER_TILE,), (HEALTH_THREADS,), args)
            self._fold_swdown(state, ys, xs, itile)
        self._folded += 1

    def _fold_swdown(self, state, ys, xs, itile: int) -> None:
        """``runtime.py:2257``'s ``cp.max(physics.fields['swdown'])``, per tile.

        Written into a device slot rather than read back, so a sweep costs one
        readback in total rather than one per tile.
        """
        physics = getattr(state, "physics", None)
        field = None if physics is None else physics.fields.get("swdown")
        if field is None:
            return
        self._swdown[itile] = field[ys, xs].max()
        self._have_swdown = True

    def finish(self) -> dict:
        """Reduce the sweep's tiles and return ``stability_report``'s dict."""
        import cupy as cp

        from gpuwm.core.kernels import get_kernel

        if self._folded != self.ntiles:
            raise AssertionError(
                f"the sweep folded {self._folded} tiles of {self.ntiles}; a "
                "partial fold would under-report the domain's maxima, which "
                "is a disarmed gate wearing the fixed gate's name")
        kernel = get_kernel("health", "health_final")
        kernel((1,), (HEALTH_THREADS,),
               (self._partial, self._result,
                np.int32(self.ntiles * BLOCKS_PER_TILE)))
        host = cp.asnumpy(self._result)          # the sweep's sole readback
        return self._report(host)

    # -- the report --------------------------------------------------------

    def _report(self, host) -> dict:
        """``stability_report``'s arithmetic, on the folded record.

        Deliberately identical, including the detail that ``nan`` is decided
        by the finiteness of the three MAXIMA rather than by the kernel's mask
        -- ``health_final`` already turns a masked field into a NaN maximum,
        and reproducing the same expression here keeps the two paths from
        drifting apart.
        """
        cfg = self.cfg
        u_max, w_max, th_max = (float(v) for v in host[:3])
        nan = not (math.isfinite(u_max) and math.isfinite(w_max)
                   and math.isfinite(th_max))
        cfl = horizontal_cfl = vertical_cfl = None
        if cfg is not None and not nan:
            horizontal_cfl = cfg.dt * u_max / cfg.dx
            vertical_cfl = cfg.dt * float(host[5])
            cfl = max(horizontal_cfl, vertical_cfl)
        report = {"u_max": u_max, "w_max": w_max, "th_max": th_max,
                  "cfl": cfl, "horizontal_cfl": horizontal_cfl,
                  "vertical_cfl": vertical_cfl, "nan": nan}
        if self.width > 0:
            index_words = host[6:8].view(np.uint32)
            w_argmax = int(index_words[0]) | (int(index_words[1]) << 32)
            report.update(boundary_w_max=float(host[3]),
                          interior_w_max=float(host[4]),
                          w_argmax=w_argmax)
        if self._have_swdown:
            import cupy as cp

            report["swdown_max"] = float(cp.asnumpy(self._swdown).max())
        return report


def fold_coverage(specs, *, nx: int, ny: int, periodic: bool = False) -> dict:
    """How many times the fold reads each domain CELL and each u FACE.

    Deterministic, and 2-D: a 4 x 4 tiling covers every mass ROW with four
    tiles, so counting rows and columns separately says "4" and means
    nothing.  What has to be true is that the (j, i) SETS the per-tile
    windows read partition the domain exactly once.

    This exists because the stochastic control for the closing staggered face
    is weak: dropping ``u``'s ``nx`` slot only changes the answer on a step
    where the domain's largest ``|u|`` happens to sit on that single column,
    MEASURED at a 100% miss rate over 10 substeps at 128 x 128.  So the
    property is checked directly rather than hoped for.

    Under ``periodic=True`` u's slot ``nx`` is a duplicate of slot 0 by
    definition and no tile owns it, so its expected count is 0 -- folding it
    would be double-counting, not coverage.
    """
    mass = np.zeros((ny, nx), dtype=np.int32)
    faces = np.zeros((ny, nx + 1), dtype=np.int32)
    for spec in specs:
        ys, xs = spec.interior_in_tile("mass")
        uys, uxs = spec.interior_in_tile("u")
        mass[spec.j0:spec.j0 + (ys.stop - ys.start),
             spec.i0:spec.i0 + (xs.stop - xs.start)] += 1
        faces[spec.j0:spec.j0 + (uys.stop - uys.start),
              spec.i0:spec.i0 + (uxs.stop - uxs.start)] += 1
    want_faces = np.ones((ny, nx + 1), dtype=np.int32)
    if periodic:
        want_faces[:, nx] = 0
    return {
        "mass_ok": bool((mass == 1).all()),
        "faces_ok": bool((faces == want_faces).all()),
        "mass_counts": sorted(int(v) for v in np.unique(mass)),
        "face_counts": sorted(int(v) for v in np.unique(faces)),
        "closing_face_counts": sorted(
            int(v) for v in np.unique(faces[:, nx])),
        "cells": int(mass.size), "faces": int(faces.size),
    }
