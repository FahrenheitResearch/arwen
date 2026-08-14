"""The streamed-child coupling corridor: rolling nest tables, per tile.

A RESIDENT child consumes FORCE output through one device attachment on its
own full-size state (``lateral_bc.attach_nest_boundaries``): full-perimeter
rolling value/tendency tables in F4/F16 manifest slots, refreshed in place
by ``bdy_interp1`` every parent step, applied with clock-recurrent dtbc.  A
STREAMED child steps as tile buffers, and a tile buffer's only boundary
door is the per-buffer attachment its tile hook installs.  This module is
the bridge: the domain keeps its rolling tables exactly as the resident
path builds them, and every tile buffer carries a PACKED, WINDOWED copy of
them -- owned sides sliced tangentially to the tile's compute window, seam
sides inert zeros -- refreshed at kernel-launch time whenever the domain's
rolling generation moves.

Three decisions carry the correctness weight:

**Copies, not views.**  The boundary kernels are raw CUDA kernels indexing
their operands linearly, so a tangential SLICE of a rolling table -- which
is strided -- would be read as the wrong bytes with no error anywhere.
Each buffer therefore owns one packed contiguous slot, the same accounting
(``lbc_forcing_tables`` scratch, ``device_nbytes``) as the specified
streaming attachment.

**Refresh at launch time, not hook time.**  ``TiledRun`` fires the tile
hook only when a buffer CHANGES tiles; with ``nbuffers >= ntiles`` that is
once per run, so a hook-time refresh would serve the first FORCE's forcing
forever -- finite, plausible and silent, the failure mode of this whole
subsystem.  Instead ``attach_nest_boundaries`` bumps a monotonic
``rolling_generation`` on the domain attachment at every FORCE, and
``_active_device_interval`` runs this module's ``nested_reload`` before
serving the nested interval: generation moved -> re-copy the owned sides
(device-to-device, O(tile boundary) bytes, owned sides only).  A buffer is
structurally unable to apply interval N's forcing to a substep of interval
N+1.

**Zeros on the seams.**  An interior side's tables are deliberately not
the domain's data, so a seam that reached a tile interior would show; the
bit-identity gate (``tilestream/test_streamed_child.py``) is the referee,
exactly as it was for the specified streamed path that proved the seam
mechanics.

The FORCE traffic half lives in :mod:`gpuwm.core.nest`: a streamed child's
coupled-field pull is windowed to the boundary FRAME
(:func:`child_frame_windows`), because ``bdy_interp1`` reads the child
only inside its boundary zone.  The feedback pull deliberately stays
whole-field -- the restriction reads the whole child interior, and that is
the honest cost of two-way feedback, not a windowing miss.
"""

from __future__ import annotations

import math

import numpy as np

#: How far past ``spec_bdy_width`` the streamed child's FORCE frame pull
#: reaches, in CHILD cells.  ``bdy_interp1`` reads the child only inside
#: ``bdy_width(spec_zone, relax_zone, spec_bdy_width)`` of each edge;
#: ``couple_nest_field``'s u/v face averages of ``mup`` add one, the
#: staggered face adds one, and 8 bounds all of it at every admitted
#: config -- deliberately the same number and the same argument as
#: ``nest.NEST_FORCE_HALO_PARENT_CELLS``: a superset is free (the reader
#: ignores what it does not need), a subset is a stale cell inside the
#: zone the reader DOES use.  The twin gate's frame-starved negative
#: control overrides this to prove the window is really narrowing the
#: read.
NEST_FORCE_FRAME_HALO_CHILD_CELLS = 8

#: Field-name staggering for the rolling table layout, the same map the
#: specified windowing carries (``streaming._LBC_STAGGER``): one extra
#: face on the field's own axis.
_NEST_STAGGER: dict[str, tuple[int, int]] = {"u": (0, 1), "v": (1, 0)}

_STATE_ATTR = {"theta": "thp", "phi": "php"}


def child_frame_windows(run_cfg) -> tuple:
    """The four frame strips a streamed child's FORCE pull moves.

    ``streaming.frame_windows`` over the child's mass extents at width
    ``spec_bdy_width + NEST_FORCE_FRAME_HALO_CHILD_CELLS``, clamped by the
    one ``window_slices`` rule at use.  A child small enough that the
    strips overlap into full coverage degenerates to the whole-field pull,
    which is the superset rule doing exactly what it says.
    """
    from gpuwm.core.streaming import frame_windows

    width = int(run_cfg.spec_bdy_width) + int(NEST_FORCE_FRAME_HALO_CHILD_CELLS)
    return frame_windows(int(run_cfg.ny), int(run_cfg.nx),
                         min(width, int(run_cfg.ny), int(run_cfg.nx)))


def _side_shape(field_name, side_name, nzf, width, cny, cnx):
    """One packed tile table's shape, in the ``window_interval`` convention."""
    ey, ex = _NEST_STAGGER.get(field_name, (0, 0))
    if side_name in ("west", "east"):
        return (int(nzf), int(cny) + ey, int(width))
    return (int(nzf), int(width), int(cnx) + ex)


def _tangential_slice(field_name, side_name, tspec):
    """The domain-table slice that lands in one tile table."""
    ey, ex = _NEST_STAGGER.get(field_name, (0, 0))
    if side_name in ("west", "east"):
        return (slice(None), slice(tspec.cj0, tspec.cj0 + tspec.cny + ey),
                slice(None))
    return (slice(None), slice(None),
            slice(tspec.ci0, tspec.ci0 + tspec.cnx + ex))


def _rolling_source(domain_state):
    """The domain's rolling attachment, or a loud refusal.

    The executor guarantees FORCE precedes the first child STEP (a child
    STEP with INVALID tables is an assertion there), so an absent or
    invalid source here is an ordering bug worth naming, not a case to
    paper over with zeros.
    """
    source = getattr(domain_state, "_lateral_boundary_device", None)
    if source is None or not source.rolling:
        raise RuntimeError(
            "a streamed child's tile buffer asked for nest boundary tables "
            "before NestCoupler.force attached any to the domain state; "
            "ordinary parent STEP -> FORCE must precede the first child "
            "tile pass")
    if not source.valid:
        raise RuntimeError(
            "the domain's rolling nest tables are INVALID (restored or "
            "relocated without a FORCE since); a tile buffer must not "
            "serve them")
    return source


def _copy_owned_sides(specs, source) -> int:
    """Fill the owned-side packed views from the domain tables.  Returns
    the number of table pairs copied.  Module-level on purpose: the twin
    gate's stale-tables negative control monkeypatches this to a no-op and
    the child MUST diverge, which is what proves the launch-time reload is
    load-bearing rather than decorative.
    """
    fields = source.intervals[0].fields
    copied = 0
    for field_name, side_name, sl, value_view, tendency_view in specs:
        side = getattr(fields[field_name], side_name)
        src_value = side.value[sl]
        src_tendency = side.tendency[sl]
        if tuple(src_value.shape) != tuple(value_view.shape):
            raise RuntimeError(
                f"nest tile table {field_name}/{side_name} windows to "
                f"{tuple(src_value.shape)} but the packed slot holds "
                f"{tuple(value_view.shape)}; the rolling table layout "
                "moved under the buffer")
        value_view[...] = src_value
        tendency_view[...] = src_tendency
        copied += 1
    return copied


def attach_streaming_nest_boundaries(tile_state, domain_state, tspec,
                                     clock) -> None:
    """Give one tile buffer its windowed rolling nest attachment.

    ``tile_state`` is the buffer (its arrays are compute-window shaped),
    ``domain_state`` is the streamed child's own ``DomainState`` -- the
    one ``NestCoupler.force`` attaches the full-perimeter rolling tables
    to -- and ``tspec`` is the buffer's current tile.  ``clock`` is the
    child's ``DomainClock``, shared by every buffer, exactly as the
    resident attachment shares it: dtbc is host metadata read at launch.

    Idempotent per (buffer, tile) and cheap to re-target: the packed slot
    is reused (one ``tile_cfg`` for every buffer means one layout), the
    Davies-weight cache survives, and ``rolling_generation`` resets to 0
    so the first launch after a re-target re-copies unconditionally.
    """
    from gpuwm.core.streaming import owned_edges
    from gpuwm.ingest.lateral_bc import (_DeviceBoundaryInterval,
                                         _DeviceFieldBoundary,
                                         _DeviceLateralBoundaries,
                                         _DeviceSideBoundary, _lbc_scratch,
                                         _release_resident_scratch,
                                         RollingNestBoundaries)
    from types import MappingProxyType

    source = _rolling_source(domain_state)
    rolling = domain_state.lateral_boundaries
    if not isinstance(rolling, RollingNestBoundaries):
        raise RuntimeError(
            "the domain state's lateral_boundaries is "
            f"{type(rolling).__name__}, not the RollingNestBoundaries the "
            "FORCE attach installs; refusing to window tables whose "
            "application semantics are unknown")
    fields = source.intervals[0].fields
    owns = owned_edges(tspec)
    cny, cnx = int(tspec.cny), int(tspec.cnx)

    # One layout pass: shapes, then the packed slot, then the views.
    layout = []          # (field, side, shape, owned)
    total = 0
    for field_name, boundary in fields.items():
        nzf = int(boundary.west.value.shape[0])
        width = int(boundary.west.value.shape[-1])
        for side_name in ("west", "east", "south", "north"):
            shape = _side_shape(field_name, side_name, nzf, width, cny, cnx)
            layout.append((field_name, side_name, shape))
            total += 2 * math.prod(shape)

    existing = getattr(tile_state, "_lateral_boundary_device", None)
    mine = (existing is not None and existing.rolling
            and existing.nested_reload is not None)
    if mine:
        weights = existing.weights
        scratch_slots = set(existing.scratch_slots)
        next_weight_slot = existing.next_weight_slot
    else:
        _release_resident_scratch(tile_state)
        weights, scratch_slots, next_weight_slot = {}, set(), 0

    packed = _lbc_scratch(tile_state, (int(total),), "lbc_forcing_tables")
    scratch_slots.add("lbc_forcing_tables")
    xp = np if isinstance(packed, np.ndarray) else None
    if xp is None:
        import cupy as cp

        xp = cp

    offset = 0
    converted = {}
    reload_specs = []
    for field_name, side_name, shape in layout:
        views = []
        for _table in range(2):
            count = math.prod(shape)
            view = packed[offset:offset + count].reshape(shape)
            offset += count
            views.append(view)
        value_view, tendency_view = views
        if owns[side_name]:
            reload_specs.append(
                (field_name, side_name,
                 _tangential_slice(field_name, side_name, tspec),
                 value_view, tendency_view))
        else:
            value_view[...] = 0
            tendency_view[...] = 0
        converted.setdefault(field_name, {})[side_name] = \
            _DeviceSideBoundary(value_view, tendency_view)

    # Validate the windowed layout against the buffer it will be applied
    # to, the same both-shapes question ``attach_nest_boundaries`` asks of
    # the full state: a mismatch here is a wrong-tiling bug, and the raw
    # kernels it would feed cannot see it.
    for field_name, sides in converted.items():
        nzf, ny_t, width = sides["west"].value.shape
        nx_t = sides["south"].value.shape[2]
        target_name = _STATE_ATTR.get(field_name, field_name)
        target = (tile_state.mup[None] if field_name == "mu"
                  else getattr(tile_state, target_name, None))
        if target is None or tuple(target.shape) != (nzf, ny_t, nx_t):
            raise RuntimeError(
                f"nest tile table {field_name} implies buffer shape "
                f"{(nzf, ny_t, nx_t)} but the buffer holds "
                f"{None if target is None else tuple(target.shape)}")

    device_intervals = (_DeviceBoundaryInterval(
        MappingProxyType({name: _DeviceFieldBoundary(**sides)
                          for name, sides in converted.items()})),)

    def nested_reload(state) -> None:
        attachment = state._lateral_boundary_device
        src = _rolling_source(domain_state)
        if attachment.rolling_generation == src.rolling_generation:
            return
        # Through the module namespace, so the stale-tables negative
        # control can disarm exactly this copy and nothing else.
        _copy_owned_sides(reload_specs, src)
        attachment.rolling_generation = src.rolling_generation
        attachment.external_reload_count += 1

    tile_state._lateral_boundary_device = _DeviceLateralBoundaries(
        device_intervals, MappingProxyType({}), weights, scratch_slots,
        int(packed.nbytes), next_weight_slot=next_weight_slot,
        rolling=True, clock=clock, valid=True,
        # 0 != any real generation (attach_nest_boundaries starts at 1),
        # so the first launch after attach or re-target always copies.
        rolling_generation=0, nested_reload=nested_reload)
    tile_state.lateral_boundaries = RollingNestBoundaries(
        int(rolling.spec_bdy_width), int(rolling.spec_zone),
        int(rolling.relax_zone))


def make_nest_tile_hook(node):
    """A ``TiledRun`` ``tile_hook`` for one streamed CHILD domain.

    The counterpart of :func:`gpuwm.core.streaming.make_tile_hook` for a
    domain whose forcing is rebuilt per parent step instead of tabulated
    per run.  The hook re-targets the buffer's windowed attachment when
    the buffer changes tiles; staying CURRENT between hook fires is the
    launch-time reload's job, not the hook's (see the module docstring for
    why hook-time refresh cannot work).
    """
    def hook(tile_state, tspec, itile, stream):
        attach_streaming_nest_boundaries(
            tile_state, node.state, tspec, node.clock)

    return hook


def corridor_claim_bytes(node, decision=None) -> int:
    """The coupling corridor's device claim for one CHILD node, in bytes.

    Priced for the tree budget walk (``streaming.steppers_for_tree``):
    every persistent ``nest_*`` slot the coupler registers -- the rolling
    tables, the two full-field coupling slots, the SINT geometry -- plus,
    when the child itself STREAMS, the per-buffer packed table windows.
    A superset where exact is unknowable here (the packed bound uses the
    tile window's upper extent): over-claiming refuses early with names,
    under-claiming is the OOM mid-run the walk exists to remove.
    """
    from gpuwm.core.preflight import nest_slot_dtypes, nest_slot_shapes

    dc = node.cfg
    parent = node.parent.cfg if node.parent is not None else None
    width = int(dc.run.spec_bdy_width)
    shapes = nest_slot_shapes(dc, width, parent)
    dtypes = nest_slot_dtypes(dc, width, parent)
    # A RESIDENT child's two full-field coupling slots alias dead RK
    # backings inside its own arena (preflight's shared-arena aliases:
    # nest_parent_field -> rk_ru, nest_child_field -> rk_ww), so they are
    # already inside the child's resident price and counting them again
    # would double-claim two full fields.  A STREAMED child has no
    # resident arena to alias into, so for it they are real claims.
    aliased_when_resident = () if (decision is not None and decision.stream) \
        else ("nest_parent_field", "nest_child_field")
    total = sum(
        math.prod(shape) * np.dtype(dtypes[slot]).itemsize
        for slot, shape in shapes.items()
        if slot not in aliased_when_resident)
    if decision is not None and decision.stream and decision.tile_nx:
        halo = int(decision.halo or 0)
        span_x = min(int(dc.run.nx), int(decision.tile_nx) + 2 * halo + 1)
        span_y = min(int(dc.run.ny), int(decision.tile_ny) + 2 * halo + 1)
        per_buffer = sum(
            math.prod(shape) * np.dtype(dtypes[slot]).itemsize
            # Rolling table slots only: nest_{kind}_b*/bt* families.
            * (1.0 if "_b" in slot else 0.0)
            # Scale the tangential extent from the domain to the tile.
            * ((span_y / int(dc.run.ny)) if slot.endswith(("xs", "xe"))
               else (span_x / int(dc.run.nx)))
            for slot, shape in shapes.items()
            if slot.startswith("nest_") and ("_b" in slot))
        total += int(per_buffer) * int(decision.nbuffers or 1)
    return int(total)


__all__ = [
    "NEST_FORCE_FRAME_HALO_CHILD_CELLS",
    "attach_streaming_nest_boundaries", "child_frame_windows",
    "corridor_claim_bytes", "make_nest_tile_hook",
]
