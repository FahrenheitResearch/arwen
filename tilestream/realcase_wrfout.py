"""wrfout history for a streamed REAL case, so ``gpuwm render`` can render it.

``tilestream/RENDERING.md`` is unambiguous: ``gpuwm render --engine rust`` is
the product tier, it reads **wrfout NetCDF**, and nothing else should be
hand-rolled.  :mod:`tilestream.output` already writes a wrfout frame straight
out of the pinned host store with no device traffic.  What that module cannot
know is a REAL case's identity -- the Lambert projection globals, the
curvilinear XLAT/XLONG the renderer takes its extent from, the land mask, and
``REFL_10CM``, which is a driver diagnostic rather than a carrier and so is
not in the store at all.

This module supplies exactly those four things and nothing else:

``globals``
    :func:`gpuwm.runtime._global_wrf_attrs` verbatim -- the same call
    ``runtime.write_case_output`` makes for a resident run, off the same
    ``prepared.grid`` / ``geog_selection`` / vertical coordinate.  A streamed
    file and a resident one therefore carry the same projection block, which
    is what makes the renderer treat them identically.

``metadata``
    :func:`gpuwm.runtime._metadata_frame` verbatim: XLAT/XLONG and their U/V
    stagger, map factors, Coriolis, SINALPHA/COSALPHA (without which the
    renderer degrades the 10 m wind panel to grid-relative), HGT, LANDMASK,
    LU_INDEX.  All of it is grid geometry, computed once.

``REFL_10CM``
    :func:`tilestream.realcase.reflectivity_3d` -- ``gpuwm.core.refl``'s own
    diagnostic, run over the host store by row slabs because reflectivity is
    column-local.  It is a MODEL field, not radar.

``the cadence discipline``
    :meth:`tilestream.output.StoreFrame.fields` must be called at a sweep
    boundary or the frame mixes generations.  :meth:`History.write` is only
    ever called between ``run_tiled`` calls, and it says so.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np


#: The volumetric fields a frame keeps.  A FULL frame at this rung is 272
#: B/mass cell -- 12.3 GiB at 960x960x49 -- so writing every carrier every
#: half hour is 300 GiB of disk for one 12 h forecast, and the box has 149.
#: These are the ones ``gpuwm render``'s products are actually computed from:
#: ``wrf.getvar`` builds pressure from ``P + PB``, height from ``PH + PHB``,
#: temperature from ``T``, moisture from ``QVAPOR``, the wind suite from
#: ``U``/``V``/``W``, the hydrometeor and cloud panels from the mixing
#: ratios, and composite reflectivity is the column maximum of ``REFL_10CM``.
#: Everything smaller than :data:`SMALL_FIELD_BYTES` is kept regardless --
#: that is every 2-D surface field and the soil column -- so the surface and
#: severe suites lose nothing.  What IS lost is named in the run log by
#: :meth:`History.write`, because a silently thinned file would let a reader
#: conclude a product is unsupported by the model rather than absent here.
VOLUME_FIELDS = (
    "U", "V", "W", "PH", "PHB", "T", "P", "PB", "QVAPOR",
    "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "REFL_10CM",
)

#: Any field at or under this many bytes is kept whatever it is.
SMALL_FIELD_BYTES = 64 * 2 ** 20


def slim(fields: dict) -> tuple[dict, list[str]]:
    """``(kept, dropped)`` under the rule above."""
    kept, dropped = {}, []
    for name, value in fields.items():
        array = np.asarray(value)
        if name in VOLUME_FIELDS or array.nbytes <= SMALL_FIELD_BYTES:
            kept[name] = value
        else:
            dropped.append(name)
    return kept, dropped


class History:
    """A wrfout writer for one streamed real case."""

    def __init__(self, prep, exp, out_dir, *, title: str,
                 slab_rows: int = 64, verbose=print):
        from gpuwm.io.wrfout import wrfout_filename
        from gpuwm.runtime import _global_wrf_attrs, _metadata_frame

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = prep.cfg
        self.title = title
        self.slab_rows = int(slab_rows)
        self.verbose = verbose
        self.domain_id = int(getattr(prep.cfg, "grid_id", 1))
        self._filename = wrfout_filename
        self.start_time = exp.start_time
        self.written: list[Path] = []

        self.attrs = _global_wrf_attrs(
            prep.grid, exp.start_time,
            getattr(prep, "geog_selection", None),
            domain=prep.cfg,
            coord=getattr(prep.initial_result, "coord", None))
        self.metadata = {k: np.ascontiguousarray(np.asarray(v))
                         for k, v in _metadata_frame(
                             prep.grid, prep.static_fields).items()}
        self.plan = None
        self.setup = None
        self._frame = None

    # -- the capture that has to happen while a resident state exists ----

    def capture(self, state, cfg) -> None:
        """``realcase.build_stores(on_state=...)`` calls this."""
        from tilestream import checkpoint as tsck, output as tsout

        self.plan = tsout.frame_plan(state)
        self.setup = tsck.DomainSetup.capture(state, cfg)
        # DomainSetup carries three fields through UNCONVERTED, for the
        # restart identity: lateral_boundaries, _lateral_boundary_device and
        # the nest classification.  On a real case the first two are DEVICE
        # data at domain shape, and this object outlives the prepared case by
        # the whole forecast -- so keeping them would pin domain-sized arrays
        # on the card that the streaming design says nothing may pin.
        # output_statics reads only STATE_SETUP_ARRAYS, which are host copies.
        for name in ("lateral_boundaries", "_lateral_boundary_device"):
            if getattr(self.setup, name, None) is not None:
                setattr(self.setup, name, None)
        if self.verbose:
            n3 = sum(1 for n in self.plan.order
                     if len(self.plan.shape[n]) == 3)
            self.verbose(f"    wrfout plan: {len(self.plan.order)} fields "
                         f"({n3} three-dimensional), setup "
                         f"{self.setup.nbytes / 2**20:.0f} MiB")

    def bind(self, case) -> None:
        """Attach the pinned store the frames are read from."""
        from tilestream import output as tsout

        if self.plan is None:
            raise RuntimeError(
                "History.capture was never called -- the frame plan can only "
                "be read off the resident state build_stores releases")
        self.case = case
        # require_complete=False, DECLARED.  This harness takes its plan off
        # the resident state before build_stores releases it and its store
        # is the carrier set, so the output-only driver diagnostics were
        # never in it.  The PRODUCT route carries them
        # (gpuwm.core.streaming.diagnostic_inventory) and StoreFrame refuses
        # a short frame there by default; a harness that publishes one says
        # so here rather than being quietly exempt.
        self._frame = tsout.StoreFrame(self.plan, case.store, self.setup,
                                       self.cfg, require_complete=False)

    # -- one frame ------------------------------------------------------

    def write(self, elapsed_s: float) -> Path:
        """One published wrfout.  MUST be called at a sweep boundary."""
        from tilestream import output as tsout
        from tilestream import realcase

        valid = self.start_time + timedelta(seconds=float(elapsed_s))
        fields = self._frame.fields()
        fields.update(self.metadata)
        fields["REFL_10CM"] = realcase.reflectivity_3d(
            self.case, slab_rows=self.slab_rows)
        fields, dropped = slim(fields)
        if self.verbose and not self.written:
            self.verbose(f"    wrfout keeps {len(fields)} fields; drops "
                         f"{len(dropped)} volumetric carriers no product "
                         f"reads: {', '.join(sorted(dropped))}")
        path = self.out_dir / self._filename(valid, self.domain_id)
        tsout.write_frame(path, fields, self.cfg,
                          valid.strftime("%Y-%m-%d_%H:%M:%S"),
                          title=self.title, global_attrs=self.attrs)
        self.written.append(path)
        return path
