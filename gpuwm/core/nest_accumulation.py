"""Nest-only precipitation on a fixed native-grid lattice.

The archive grows only at relocation. It never fills unvisited ground with
parent precipitation. A return to a former placement restores that nest's
own totals; periods outside the nest remain unobserved, not estimated rain.
Arrays stay on their owner's backend. The origin is relative to the first
placement, so ancestor moves use the same physical shift as prognostics.
"""
PREFIX = 'nest_accumulation/'
META = PREFIX + 'origin'


def is_archive_slot(slot):
    from gpuwm.core.physics_continuation import STRIP_ACCUMULATORS
    return slot == META or slot in {PREFIX + s for s in STRIP_ACCUMULATORS}


def capture(state, store=None):
    """Retain immutable archive references before the outgoing owner releases."""
    # Streamed runs retain this domain-level archive on their template;
    # it is deliberately absent from the per-tile transport inventory.
    return {key: value for key, value in getattr(state, '_scratch', {}).items()
            if is_archive_slot(key)}


def advance(archive, values, plan, ring=0):
    """Book the outgoing valid interior, then seed the new footprint.

    The specified boundary ring is not integrated by the nest physics. It
    cannot overwrite a previously integrated interior cell in the archive.
    Unvisited strips stay zero, with exposure supplied separately by products.
    """
    from gpuwm.core.physics_continuation import STRIP_ACCUMULATORS, _array_module
    values = {s: values[s] for s in STRIP_ACCUMULATORS if s in values}
    if not values:
        return {}, {}, dict(rule='nest-only fixed-frame totals', seeded=[])
    first = next(iter(values.values()))
    xp = _array_module(first)
    ny, nx = first.shape
    if any(v.shape != (ny, nx) for v in values.values()):
        raise ValueError('nest precipitation counters must share one mass grid')
    if archive:
        meta = archive[META]
        if meta.shape != (4,) or meta.dtype != xp.dtype('int64'):
            raise ValueError('nest accumulation origin must contain four int64 coordinates')
        if hasattr(meta, 'get'):
            meta = meta.get()  # four integers, only on a relocation boundary
        x, y, ox, oy = map(int, meta)
    else:
        x = y = ox = oy = 0
    tx, ty = x + int(plan.shift_i), y + int(plan.shift_j)
    old_plane = next((v for k, v in archive.items() if k != META), None)
    ah, aw = (ny, nx) if old_plane is None else old_plane.shape
    if archive:
        if (old_plane is None or x < ox or y < oy
                or x + nx > ox + aw or y + ny > oy + ah
                or any(v.shape != (ah, aw) for k, v in archive.items() if k != META)):
            raise ValueError('nest accumulation origin or plane lies outside its archive')
    left, bottom = min(ox, x, tx), min(oy, y, ty)
    right, top = max(ox + aw, x + nx, tx + nx), max(oy + ah, y + ny, ty + ny)
    r = int(ring)
    if r < 0 or 2*r >= min(ny, nx):
        raise ValueError('specified ring leaves no nest precipitation interior')
    interior = (slice(r, ny-r), slice(r, nx-r))
    result, seeds = {}, {}
    for slot, value in values.items():
        key = PREFIX + slot
        plane = xp.zeros((top-bottom, right-left), dtype=value.dtype)
        old = archive.get(key)
        if old is not None:
            if xp.__name__ == 'numpy' and hasattr(old, 'get'):
                old = old.get()
            plane[oy-bottom:oy-bottom+ah, ox-left:ox-left+aw] = old
        plane[y-bottom+r:y-bottom+ny-r, x-left+r:x-left+nx-r] = value[interior]
        seeds[slot] = plane[ty-bottom:ty-bottom+ny, tx-left:tx-left+nx].copy()
        result[key] = plane
    result[META] = xp.asarray([tx, ty, left, bottom], dtype=xp.int64)
    return result, seeds, dict(rule='nest-only fixed-frame totals; unvisited strips zero',
        seeded=sorted(seeds), parent_precipitation=False, old_ring_cells=r,
        archive_shape=[top-bottom, right-left],
        archive_bytes=sum(int(a.nbytes) for a in result.values()))


def install(state, archive):
    # The new owner has no previous archive. Do not ask the shape-locked
    # scratch allocator to resize a slot from an earlier placement.
    for key in list(state._scratch):
        if is_archive_slot(key):
            del state._scratch[key]
    state._scratch.update(archive)


def relocate(archive, values, plan, ring, parent_node, child_dc):
    """One policy for resident and streamed moves; explicit legacy control."""
    if not getattr(child_dc.run, 'nest_accumulation_archive', True):
        from gpuwm.core.physics_continuation import parent_strip_accumulations
        seed, receipt = parent_strip_accumulations(parent_node, child_dc, values)
        receipt['old_ring_cells'] = ring
        return {}, seed, receipt
    return advance(archive, values, plan, ring)
