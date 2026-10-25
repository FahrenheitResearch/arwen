# SPDX-License-Identifier: Apache-2.0
"""A door initialization whose member is built into a card-wide shared workspace.

The prepared tree runner (:func:`gpuwm.prepared_domain_tree_forecast.run_prepared_tree`)
hands every initialization the step-scratch arena and the restart-rebuilt state workspace
its tree shares, and ``None`` for both on a single domain, which then keeps its own.
:class:`WorkspaceInitialization` wraps any door initialization (the WRF-input door's, its
warm start, or the analysis door's) and substitutes one workspace shared by every resident
member on the card (:class:`gpuwm.ensemble.sequential_workspace.SequentialMemberWorkspace`),
exactly as a nest takes its tree's.  Everything else is the wrapped initialization's own.

WHY A WRAPPER AND NOT A RUNNER ARGUMENT.  ``prepared_domain_tree_forecast.py`` is hashed into
every tree checkpoint's restart identity (``runtime_source_identity``) and
``wrfinput_forecast.py`` into every WRF-input preparation receipt (``adapter_code``).  A
parameter added to either refuses every checkpoint written before it, including the cycle's
own 23:00Z member checkpoints (measured: ``RestartMismatchError ... preparation_receipt_sha256,
runtime_source_identity differ(s)``).  The initialization seam already carries the arena, so
the workspace enters there and both files stay byte for byte as the files path runs them.

The tree runner records ``None`` as the state's and the model's arena on a single domain,
after the restore; this initialization binds the workspace again before the physics is
initialized (the restored domain's ``initialize_physics``) and on the model in
:meth:`after_restore`, before the executor is built, so the executor takes the rebuilt-state
workspace for every STEP and FORCE (the member's turn must be an executor turn).
"""
from __future__ import annotations

from gpuwm.forecast_initialization import DomainInitialization


class WorkspaceInitializationRefused(ValueError):
    """A member this workspace cannot build, refused before it allocates."""


class WorkspaceInitialization:
    """``base`` with its domain built into ``workspace`` (``scratch``, ``rebuilt``, ``admits``, ``member``)."""

    def __init__(self, base, workspace):
        for name in ("scratch", "rebuilt", "admits"):
            if getattr(workspace, name, None) is None:
                raise TypeError(f"workspace has no {name!r}: not a resident member workspace")
        self._base = base
        self.workspace = workspace
        self.bound = []

    def __getattr__(self, name):
        # Every other door attribute (lateral_boundaries, warm_start_clocks, step_observer,
        # run_report_blocks, plan, ...) is the wrapped initialization's.
        return getattr(self.__dict__["_base"], name)

    def verify_inputs(self, inputs):
        if len(inputs.experiment.domains) != 1:
            raise WorkspaceInitializationRefused(
                "a member workspace is one domain's shared step scratch; this tree has "
                f"{len(inputs.experiment.domains)} domains, which share their own arena, so a second one "
                "would hand one step's scratch to another domain")
        return self._base.verify_inputs(inputs)

    def domain_content_sha256(self, bundle):
        return self._base.domain_content_sha256(bundle)

    def domain_metadata(self, bundle):
        return self._base.domain_metadata(bundle)

    def restore_domain(self, domain, grid, bundle, *, start_time, scratch_arena, dycore_state_workspace):
        if scratch_arena is not None or dycore_state_workspace is not None:
            raise WorkspaceInitializationRefused(
                "the runner handed this domain its tree's shared arena: a member workspace serves a "
                "single-domain member only, and two arenas would hand one step's scratch to another domain")
        if getattr(self.workspace, "member", None) is None:
            raise WorkspaceInitializationRefused(
                "member construction outside its workspace turn: another member's step could overwrite "
                "this member's rebuilt arrays while it restores (take the member's executor turn around "
                "the construction)")
        self.workspace.admits(domain)
        restored = self._base.restore_domain(
            domain, grid, bundle, start_time=start_time,
            scratch_arena=self.workspace.scratch, dycore_state_workspace=self.workspace.rebuilt)
        if not isinstance(restored, DomainInitialization):
            raise TypeError("input adapter must return DomainInitialization")
        state = restored.state
        initialize = restored.initialize_physics
        workspace = self.workspace

        def initialize_physics():
            # The runner set the state's arena to its own None after this restore returned.
            state._scratch_arena = workspace.scratch
            return initialize()

        return DomainInitialization(restored.initial_result, initialize_physics)

    def after_restore(self, *, model, nodes, start_time, run_seconds, restart=None):
        for node in nodes.values():
            node.state._scratch_arena = self.workspace.scratch
        model._scratch_arena = self.workspace.scratch
        model._dycore_state_workspace = self.workspace.rebuilt
        self.bound.append(id(model))
        after = getattr(self._base, "after_restore", None)
        if callable(after):
            after(model=model, nodes=nodes, start_time=start_time, run_seconds=run_seconds,
                  restart=restart)


__all__ = ["WorkspaceInitialization", "WorkspaceInitializationRefused"]
