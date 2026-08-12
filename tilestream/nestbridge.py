"""What a domain tree needs from a STREAMED parent, and the one piece missing.

A streamed domain's arrays live in its host STORE, not on its
``DomainState``.  That is the whole premise -- the state's device arrays are
where the tile buffers came from, not where the domain is -- and every
consumer inside :mod:`tilestream` respects it.  ``gpuwm.core.nest`` does not,
because it predates the mode and had no reason to:

    def _coupled_parent_field(self, kind):          # nest.py:227
        parent_state = self.child_node.parent.state
        ...
        couple_nest_field(parent_state, kind, out=out)

``NestCoupler.force`` reads the PARENT'S DEVICE STATE for every forcing field
-- u, v, w, thp, php, mup and every active moist species.  With the parent
streamed those arrays hold whatever was in them when the store was filled,
so the child is forced from the parent's INITIAL CONDITION for the whole
forecast.  Nothing raises, nothing is NaN, and the child keeps producing
weather: it is exactly the silent-and-plausible failure this project keeps
being caught by.  The same sentence in reverse describes two-way feedback --
``feedback_commit`` MUTATES ``parent.state``, and those writes land on arrays
the streamed parent never reads again.

THE GAP IS UNIMPLEMENTED, NOT STRUCTURAL, AND THE ARITHMETIC SAYS SO
--------------------------------------------------------------------
The reflex is to call this structural: forcing a nest needs the WHOLE parent
and a tile cannot see the whole parent.  The first half is true and the
second does not follow, because the coupler already works ONE FULL FIELD AT A
TIME -- ``_coupled_parent_field`` borrows a single full-parent-shaped prefix
out of the F16 arena, fills it, hands it to ``bdy_interp1``, and reuses the
same backing for the next kind.  So a streamed parent has to materialise one
field, not one state.

One fp32 field of an ``N x N x 49`` domain is ``196 * N^2`` bytes.  Against
20 GiB of usable VRAM that is ``N = 10,100``; against the 44.14 GiB pinned
ceiling a single 32.26 B/cell dry store already tops out at ``N = 5,476``, and
the full carrier set tops out far below that.  The one-field working set is
therefore never the binding constraint in any domain the mode can reach.
What is missing is the copy, and the copy is this module.

WHAT THIS MODULE IS AND IS NOT
------------------------------
:func:`refresh_nest_fields` puts the store's forcing set back onto the state,
field by field, so ``NestCoupler.force`` reads live data;
:func:`commit_nest_fields` sends the same set the other way, so a two-way
feedback the coupler wrote onto the state reaches the store.  Together they
make the coupler's INPUT and OUTPUT correct, and
:func:`tilestream.test_mgphys.nest_probe` proves it bit-for-bit against a
resident run with the no-refresh case as its negative control.

They do NOT make a nested streamed forecast work, and reporting them as if
they did would be the sixth false result.  The executor
(``gpuwm.core.model.execute_experiment``) still calls neither, no route wires
them, and until it does the honest behaviour for a tree whose parent streams
is REFUSAL -- ``steppers_for_tree`` currently returns a streamed stepper for a
parent with children without comment, which is how a user gets the stale nest
above.  Cost, for the record: one refresh is a whole-domain H2D of the
forcing set (16 fields at ``full(real74)+KF``), i.e. roughly a third of one
sweep's gather traffic, once per parent step.
"""

from __future__ import annotations

import numpy as np

#: ``kind -> DomainState attribute``.  ``gpuwm.core.nest._field_shape`` and
#: ``lateral_bc.couple_nest_field`` both apply exactly this map; three kinds
#: are named differently on the state and the rest are their own names.
NEST_STATE_ATTR: dict[str, str] = {"t": "thp", "ph": "php", "mu": "mup"}


class NestBridgeError(RuntimeError):
    """A nest forcing set that cannot be moved between a state and a store."""


def nest_carrier_keys(cfg) -> dict[str, str]:
    """``{kind: carrier key}`` for every field a nest is forced from.

    The keys are :func:`tilestream.physics_inventory.carrier_inventory`'s --
    restart member names -- so they address the host store directly.  The set
    is ``cfg``-dependent: ``nest_field_kinds`` adds the moist species the
    configured microphysics actually transports, which is why this takes a
    config rather than being a constant.
    """
    from gpuwm.core.preflight import nest_field_kinds

    return {kind: f"state/{NEST_STATE_ATTR.get(kind, kind)}"
            for kind in nest_field_kinds(cfg)}


def _store_arrays(store):
    inner = getattr(store, "arrays", None)
    return inner if isinstance(inner, dict) else store


def refresh_nest_fields(state, store, cfg) -> int:
    """Host store -> device state, for the nest forcing set only.  Bytes moved.

    Call this on a STREAMED parent immediately before ``NestCoupler.force``.
    One field at a time by construction (the loop never holds two), which is
    the property that keeps it usable on a domain far larger than VRAM -- see
    the module docstring's arithmetic.

    Deliberately NOT the whole carrier set.  Forcing reads u/v/w/thp/php/mup
    and the transported species and nothing else, and a state refreshed
    field-by-field from a store it does not otherwise own should be refreshed
    with exactly what the next reader needs, so that anything else read off
    it is obviously stale rather than plausibly fresh.
    """
    arrays = _store_arrays(store)
    moved = 0
    for kind, key in nest_carrier_keys(cfg).items():
        target = getattr(state, NEST_STATE_ATTR.get(kind, kind), None)
        if target is None:
            continue                      # species the state does not carry
        if key not in arrays:
            raise NestBridgeError(
                f"the store has no {key!r}; a nest cannot be forced from a "
                f"parent whose {kind!r} field is not streamed")
        src = arrays[key]
        if tuple(src.shape) != tuple(target.shape):
            raise NestBridgeError(
                f"{key!r} is {tuple(src.shape)} in the store and "
                f"{tuple(target.shape)} on the state")
        # ``arr[...] = host_array`` routes through cupy's fill and raises
        # "non-scalar numpy.ndarray cannot be used for fill", so the host
        # side goes through ndarray.set -- the same reason
        # physics_inventory.load_carriers does.
        target.set(np.ascontiguousarray(src))
        moved += int(target.nbytes)
    return moved


def commit_nest_fields(state, store, cfg) -> int:
    """Device state -> host store, for the nest forcing set only.  Bytes moved.

    The feedback direction.  ``NestCoupler.feedback_commit`` writes the
    child's average back into the parent's arrays over the child's footprint;
    on a streamed parent those arrays are not the domain, so without this the
    feedback is silently discarded at the next sweep.

    Whole fields rather than the child's window on purpose: the window is an
    optimisation and the correctness argument does not need it, while a
    windowed scatter would have to agree with ``nest_interp``'s own
    parent-cell mapping -- a second place to get the geometry wrong for a
    saving of at most one field's worth of traffic per parent step.
    """
    arrays = _store_arrays(store)
    moved = 0
    for kind, key in nest_carrier_keys(cfg).items():
        source = getattr(state, NEST_STATE_ATTR.get(kind, kind), None)
        if source is None:
            continue
        if key not in arrays:
            raise NestBridgeError(f"the store has no {key!r}")
        dst = arrays[key]
        if tuple(dst.shape) != tuple(source.shape):
            raise NestBridgeError(
                f"{key!r} is {tuple(dst.shape)} in the store and "
                f"{tuple(source.shape)} on the state")
        dst[...] = source.get() if hasattr(source, "get") else np.asarray(
            source)
        moved += int(dst.nbytes)
    return moved


def coupled_parent_digests(state, cfg) -> dict[str, str]:
    """``{kind: sha256}`` of every field ``NestCoupler.force`` would read.

    Runs ``lateral_bc.couple_nest_field`` -- the coupler's own routine, not a
    re-implementation of it -- into a scratch buffer per kind, so the digest
    is of the bytes ``bdy_interp1`` would actually be handed.  That is what
    makes "the nest is forced from stale data" a measurement rather than a
    reading of the source.
    """
    import hashlib

    import cupy as cp

    from gpuwm.ingest.lateral_bc import couple_nest_field

    out: dict[str, str] = {}
    for kind in nest_carrier_keys(cfg):
        attr = NEST_STATE_ATTR.get(kind, kind)
        field = getattr(state, attr, None)
        if field is None:
            continue
        shape = ((1, *state.mup.shape) if kind == "mu"
                 else tuple(int(n) for n in field.shape))
        buf = cp.empty(shape, dtype=cp.float32)
        couple_nest_field(state, kind, out=buf)
        out[kind] = hashlib.sha256(
            np.ascontiguousarray(cp.asnumpy(buf)).tobytes()).hexdigest()
        del buf
    return out
