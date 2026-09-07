"""Bounded allocation lifecycle for the common child-relocation primitive.

This supplies construction/ownership operations. Placement admissibility,
alignment, overlap transplant, relocation records and subtree ordering remain
owned by nest_relocation.relocate_child.
"""
from copy import copy
from types import SimpleNamespace
import gc

import numpy as np


def mark_reconstruction_nodes(nodes, exp):
    """Give reporting and execution the same moving-subtree capacity scope."""
    if exp is None:
        return
    relocation = getattr(exp, "relocation", None)
    roots = {int(dc.grid_id) for dc in exp.domains if getattr(dc, "follow", None) is not None}
    if relocation is not None and relocation.enabled and (relocation.moves or relocation.follow is not None):
        roots.add(int(relocation.grid_id))
        if relocation.containment is not None:
            roots.add(int(relocation.containment.grid_id))
    for node in nodes:
        ancestor = node
        while ancestor is not None:
            if int(ancestor.cfg.grid_id) in roots:
                node._streamed_reconstruction_required = True
                node._reconstruction_p_top = float(exp.vertical.p_top)
                break
            ancestor = getattr(ancestor, "parent", None)


def reconstruction_claim(node, decision, options):
    """The tile owner plus its one retained reconstruction slab, in bytes."""
    from gpuwm.core.streaming import radiation_footprint
    cfg = node.cfg.run
    cells = (int(decision.tile_nx)+2*int(decision.halo)) * (int(decision.tile_ny)+2*int(decision.halo)) * int(cfg.nz)
    fp = radiation_footprint(cfg, options)
    from gpuwm.config import radiation_scheme_ids
    from gpuwm.physics_compat import rrtmg_variant, RRTMG_VARIANT_LEGACY
    lw, sw = radiation_scheme_ids(cfg)
    transient = int(fp.radiation_transient_bytes)
    if 4 in (lw, sw) and rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
        from gpuwm.core.rrtmg_legacy import legacy_radiation_vram_bytes
        transient = legacy_radiation_vram_bytes(
            ncol=cells//int(cfg.nz), nz=cfg.nz, p_top=node._reconstruction_p_top,
            column_chunk=None, longwave=lw == 4, shortwave=sw == 4)
    # A slab contains no more horizontal cells than one compute window.
    # Its full inventory remains as metadata beside the live tile owner.
    cap = int(np.ceil((fp.marginal_bytes(cells, int(decision.nbuffers)+1)+transient)/512))*512
    decision.detail["reconstruction_default_allocator_bytes"] = cap
    decision.detail["reconstruction_radiation_call_bytes"] = transient
    return cap


def wire_reconstruction_runner(runner):
    """Use the allocation reservation admitted with each live moving store."""
    def factory(node, *, initializer, preparer):
        from gpuwm.core.streaming import StreamingRefused
        stream = getattr(node.state, "_streamed_domain", None)
        reservation = getattr(stream, "_reconstruction_reservation", None)
        if reservation is None:
            raise StreamingRefused("moving host store has no admitted reconstruction reservation")
        return StreamedChildReconstruction(
            stream, initializer=initializer, preparer=preparer,
            device_budget_bytes=reservation.budget_bytes,
            host_budget_bytes=int(stream._reconstruction_host_budget_bytes))
    runner.streamed_reconstruction_factory = factory
    reground = getattr(runner, "reground_descendant", None)
    if reground is not None:
        reground.streamed_reconstruction_factory = factory
    return runner


class StreamedChildReconstruction:
    def __init__(self, stream, *, initializer, preparer, device_budget_bytes,
                 host_budget_bytes, rows_per_slab=None):
        from gpuwm.core.streaming import StreamingRefused
        if not callable(getattr(initializer, "prepare_footprint", None)):
            raise StreamingRefused("streamed relocation requires footprint-aware real reconstruction")
        if not callable(getattr(preparer, "prepare_windows", None)):
            raise StreamingRefused("streamed relocation requires global continuation window preparation")
        if stream.decision.store != "host":
            raise StreamingRefused("bounded relocation requires a canonical host store")
        self.stream, self.initializer, self.preparer = stream, initializer, preparer
        self.device_budget_bytes = int(device_budget_bytes)
        self.host_budget_bytes = int(host_budget_bytes)
        self.rows_per_slab = rows_per_slab
        self.bundle = None
        self.reservation = getattr(stream, "_reconstruction_reservation", None)
        if self.reservation is not None and self.reservation.budget_bytes != self.device_budget_bytes:
            raise StreamingRefused("repeated relocation must reuse its admitted device reservation")
        self.rk_seeds = ()

    @staticmethod
    def state_digest(state):
        from gpuwm.ensemble.state_sha import hash_state_arrays, live_state_sha256, serialized_state_attrs
        stream = getattr(state, "_streamed_domain", None)
        if stream is None:
            return live_state_sha256(state)
        store = stream.store
        return hash_state_arrays((name, store[f"state/{name}"])
                                 for name in serialized_state_attrs() if f"state/{name}" in store)

    @staticmethod
    def _facade(template, cfg, store, geography, scalars, *, scratch_allocator=None):
        from gpuwm.core.streamed_state import CanonicalStoreState
        from gpuwm.core.streaming import streamed_store_inventory
        from tilestream.driver import geography_inventory
        return CanonicalStoreState(template, cfg, store=store, geography=geography,
            scalars=scalars, inventory=streamed_store_inventory()(template),
            geography_inventory=geography_inventory(template), scratch_allocator=scratch_allocator)

    def capture_source(self, node):
        from gpuwm.core.nest_relocation import HostStateSnapshot, relocatable_attrs, _DONOR_ALIGNMENT_FIELDS
        template = self.stream.template if self.stream.template is not None else node.state
        facade = self._facade(template, node.cfg.run, self.stream.store,
                              self.stream._geography, self.stream.scalars)
        fields = {name: getattr(facade, name) for name in tuple(relocatable_attrs())+_DONOR_ALIGNMENT_FIELDS
                  if getattr(facade, name, None) is not None}
        # Borrow the immutable outgoing host arrays. This object outlives the
        # closed tile owner, and no operation writes its carrier store.
        return HostStateSnapshot(fields, pinned=True)

    def release_outgoing(self, node):
        from gpuwm.core.nest_relocation import release_state_arrays
        from gpuwm.core.streamed_state import CanonicalStoreState
        self.stream.tiled_run.close()
        if isinstance(node.state, CanonicalStoreState):
            node.state._scratch.clear()
            node.state._view_cache.clear()
            node.state._template_metadata = None
            node.state._scratch_allocator = None
            released = {"canonical_host_state": True}
        else:
            released = release_state_arrays(node.state)
            node.state._scratch.clear()
        for name in ("_lateral_boundary_device", "_lateral_boundary_binding"):
            if name in vars(node.state):
                setattr(node.state, name, None)
        self.stream._template = None
        self.stream._frame = self.stream._setup = self.stream._statics_setup = None
        # Retire donor-device bindings as well as rolling tables. Geometry is
        # rebuilt by the common commit after the new placement is installed.
        node.coupler.registrations = node.coupler._build_registrations()
        node.coupler._geometry_bound = False
        node.coupler._last_tables = None
        node.coupler._prepared_feedback = None
        gc.collect()
        return {"tile_owner_closed": True, "state": released}

    def initialize(self, new_dc, parent_node):
        import cupy as cp
        from gpuwm.ingest.reconstruction_store import ReconstructionReservation, store_from_reconstruction
        parent_stream = getattr(parent_node.state, "_streamed_domain", None)
        if parent_stream is not None:
            parent = copy(parent_node)
            template = parent_stream.template if parent_stream.template is not None else parent_node.state
            parent.state = self._facade(template, parent_node.cfg.run, parent_stream.store,
                                        parent_stream._geography, parent_stream.scalars)
        else:
            parent = parent_node
        footprint = self.initializer.prepare_footprint(new_dc, parent)
        prepare = self.preparer.prepare_windows(new_dc, parent, footprint)
        if self.reservation is None:
            self.reservation = ReconstructionReservation(self.device_budget_bytes)
        decision = self.stream.decision
        cells = int(decision.tile_nx+2*decision.halo)*int(decision.tile_ny+2*decision.halo)
        rows = (max(1, min(int(new_dc.run.ny), cells//int(new_dc.run.nx)))
                if self.rows_per_slab is None else int(self.rows_per_slab))
        self.bundle = store_from_reconstruction(
            lambda dc, node, **kw: self.initializer(dc, node, footprint=footprint, **kw),
            new_dc, parent, rows_per_slab=rows,
            device_budget_bytes=self.device_budget_bytes, host_budget_bytes=self.host_budget_bytes,
            prepare=prepare, reservation=self.reservation)
        reservation = self.reservation
        def scratch(shape, slot, dtype=None):
            with reservation.activate():
                return cp.zeros(shape, dtype=np.float32 if dtype is None else dtype)
        state = self._facade(self.bundle.template, new_dc.run, self.bundle.store,
                             self.bundle.geography, self.bundle.scalars, scratch_allocator=scratch)
        self.new_dc = new_dc
        return SimpleNamespace(state=state, grid=footprint.grid, static_fields=footprint.static_fields,
                               preprocess_receipt=dict(self.bundle.receipt))

    def post_transplant(self, *, source_state, target_state, plan):
        from gpuwm.ingest.reconstruction_store import rederive_reconstructed_store
        from gpuwm.ingest.relocation_init import relocation_base_changed_cells
        changed = relocation_base_changed_cells(source_state, target_state, plan)
        cfg, decision = self.new_dc.run, self.stream.decision
        receipt = rederive_reconstructed_store(self.bundle, cfg, reservation=self.reservation,
            tile_nx=min(int(decision.tile_nx), cfg.nx-2),
            tile_ny=min(int(decision.tile_ny), cfg.ny-2))
        self.rk_seeds = tuple(receipt["rk_seeded"])
        target_state._relocation_held_physics_recoupled = True
        return {**receipt, "base_changed_cells": changed, "diagnostics_rederived": True,
                "perturbation_carry": "bitwise; totals resplit against the rebuilt base"}

    def commit(self, node):
        from gpuwm.core.streaming import store_domain_builder
        with self.reservation.activate():
            replacement = store_domain_builder(self.bundle, node=node, clock=node.clock)(
                None, node.cfg.run, self.stream.decision)
        self.stream.rebind_after_reconstruction(replacement, state=node.state)
        self.stream._reconstruction_reservation = self.reservation
        self.bundle = None
