"""One audited workspace for serial turns of resident ensemble members.

Carried prognostics, held physics tendencies and accumulators remain private.
Only the engine's existing write-before-read arena and restart-rebuilt state
inventory are shared. Every construction, force and step must hold ``turn``.
The fixed CUDA stream makes queued writes ordered even after the CPU lock exits.

Two kinds of turn share one member lock.  A direct turn (the default) also
holds the rebuilt-state workspace, for callers that step ``dycore.step``
themselves.  An executor turn (``executor=True``) leaves that workspace to the
model executor, which takes it for every STEP and FORCE
(:func:`gpuwm.core.model.execute_experiment`); holding it here as well would
refuse the executor's first step.  A door run constructs a member inside an
executor turn by passing the workspace as ``member_workspace`` to
:func:`gpuwm.prepared_domain_tree_forecast.run_prepared_tree`.
"""
from contextlib import contextmanager
import threading

import cupy as cp

from gpuwm.core.state import (build_shared_scratch_arena,
                              build_shared_dycore_state_workspace)


class WorkspaceRefused(ValueError):
    """A member the shared workspace cannot serve, refused before it allocates."""


class SequentialMemberWorkspace:
    def __init__(self, domain):
        domains = (domain,) if hasattr(domain, 'run') else tuple(domain)
        if not domains:
            raise ValueError('resident workspace needs at least one domain configuration')
        self.configs = tuple(dc.run for dc in domains)
        self.domains = domains
        self.device = cp.cuda.runtime.getDevice()
        self.stream = cp.cuda.get_current_stream().ptr
        self.scratch = build_shared_scratch_arena(domains)
        self.rebuilt = build_shared_dycore_state_workspace(domains)
        from gpuwm.core.preflight import shared_scratch_arena_aliases
        self.aliases = dict(shared_scratch_arena_aliases(domains))
        self._turn_lock = threading.Lock()
        self.member = None
        self.turns = 0
        self.turn_members = []

    @contextmanager
    def turn(self, member_id, *, executor=False):
        if cp.cuda.runtime.getDevice() != self.device:
            raise RuntimeError('resident workspace used on another CUDA device')
        if cp.cuda.get_current_stream().ptr != self.stream:
            raise RuntimeError('resident workspace requires its original CUDA stream')
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError(
                f'member {member_id!r} cannot take the resident workspace while '
                f'member {self.member!r} holds it: two members on one shared '
                'arena at once would overwrite each other\'s step scratch and '
                'rebuilt arrays')
        self.member = member_id
        try:
            if executor:
                yield self
            else:
                with self.rebuilt.acquire(member_id):
                    yield self
        finally:
            self.member = None
            self.turns += 1
            if len(self.turn_members) < 4096:
                self.turn_members.append(member_id)
            self._turn_lock.release()

    def restore(self, restored, cfg):
        if self.rebuilt.owner is None:
            raise RuntimeError('restore requires an active member turn')
        if cfg not in self.configs:
            raise ValueError('resident workspace configuration mismatch')
        from gpuwm.ingest.wrfinput import restore_domain_state
        return restore_domain_state(restored, cfg, scratch_arena=self.scratch,
                                    dycore_state_workspace=self.rebuilt)

    def admits(self, domain):
        """Refuse, by name, a member domain this shared workspace cannot serve.

        A configuration the workspace was built from was audited with the rest
        at construction. Any other is admitted only when joining it would not
        change the lifetime aliases (an alias the member's own audit keeps
        separate would overwrite one of its live buffers with another) and
        every scratch slot and rebuilt symbol it requests fits the shared
        backing (a short backing is a refused view at its first step).
        """
        if domain.run in self.configs:
            return
        from gpuwm.core.preflight import (shared_dycore_state_workspace_shapes,
                                          shared_scratch_arena_aliases,
                                          shared_scratch_arena_shapes)
        joined = dict(shared_scratch_arena_aliases(self.domains + (domain,)))
        if joined != self.aliases:
            changed = sorted(set(joined.items()) ^ set(self.aliases.items()))
            raise WorkspaceRefused(
                'this member changes the shared arena\'s lifetime aliases '
                f'({changed}): its own audit keeps buffers separate that the '
                'workspace shares, so stepping it there would overwrite a live '
                'buffer; build the workspace with this member\'s configuration')
        for slot, shape in shared_scratch_arena_shapes((domain,)).items():
            try:
                self.scratch.view(shape, slot)
            except (KeyError, ValueError) as error:
                raise WorkspaceRefused(
                    f'scratch slot {slot!r} {tuple(shape)} does not fit the '
                    f'shared arena ({error}); the member\'s first step would '
                    'be refused at that view') from error
        for symbol, shape in shared_dycore_state_workspace_shapes(
                (domain,)).items():
            try:
                self.rebuilt.view(symbol, shape)
            except (KeyError, ValueError) as error:
                raise WorkspaceRefused(
                    f'rebuilt state symbol {symbol!r} {tuple(shape)} does not '
                    f'fit the shared workspace ({error}); the member\'s state '
                    'construction would be refused at that view') from error

    @property
    def nbytes(self):
        return int(self.scratch.nbytes) + int(self.rebuilt.nbytes)

    def receipt(self):
        return dict(device=self.device, stream=self.stream,
                    scratch_bytes=self.scratch.nbytes,
                    rebuilt_bytes=self.rebuilt.nbytes,
                    rebuilt_symbols=sorted(self.rebuilt.symbols),
                    scratch_slots=sorted(self.scratch.slot_shapes),
                    scratch_aliases=dict(sorted(self.aliases.items())),
                    configurations=len(self.configs),
                    turns=self.turns,
                    execution='serial member turns on one CUDA stream')


def workspace_slot_bytes(domains):
    """Bytes a workspace for ``domains`` allocates, from the preflight registry.

    The same functions the builder calls, so the planner can price the arena
    before anything is allocated.
    """
    from gpuwm.core.preflight import (shared_dycore_state_workspace_bytes,
                                      shared_scratch_arena_bytes)
    domains = tuple(domains)
    return dict(scratch=int(shared_scratch_arena_bytes(domains)),
                rebuilt=int(shared_dycore_state_workspace_bytes(domains)))
