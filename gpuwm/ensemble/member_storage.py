# SPDX-License-Identifier: Apache-2.0
"""Resident members on one card: a shared workspace, private parkable pages.

Composes :class:`gpuwm.ensemble.sequential_workspace.SequentialMemberWorkspace`
(the step scratch arena and restart-rebuilt arrays every member's step uses in
turn) with one :class:`gpuwm.ensemble.managed_storage.ManagedStorage` owner per
member (that member's state, physics, boundary tables and any attached analysis
forcing, allocated as managed pages).  The workspace stays ordinary device
memory: it is allocated before, and outside, every member's allocation scope.

A member's turn is one workspace turn: construction, forcing and stepping of
that member only, member-major.  A residency plan
(:func:`gpuwm.ensemble.member_budget.plan_residency`) says which members stay
on the card; the others are parked to host memory after each turn and resumed
for the next.  :meth:`ResidentMemberStorage.storage_owners` hands out
owner adapters with the ``allocate``/``park``/``resume``/``phase`` surface a
retained-member scheduler takes: entering ``allocate`` is the member's
workspace turn, and ``park`` keeps a pinned member on the card (it only
returns its cached transient blocks).

Allocation policy (``allocation``):

``turn``
    Every allocation inside a member's turn is its own (managed), including
    step transients; its pool is trimmed after each turn.  Nothing the member
    allocates can stay behind on the card when it parks.
``construction``
    Only construction and explicit :meth:`ResidentMemberStorage.allocating`
    scopes (an analysis application) allocate managed pages; step transients
    come from the ordinary device pool.  An array a step creates lazily stays
    on the card when its member parks: the receipt reports it as residue.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from time import perf_counter

ALLOCATION_POLICIES = ("turn", "construction")
PARKING = ("managed", "none")


class MemberStorageOwner:
    """One member's storage, as a retained-member scheduler sees it."""

    def __init__(self, storage, member_id):
        self._storage = storage
        self.member_id = member_id
        self.managed = None
        self.turns = 0
        self.parks = 0
        self.resumes = 0
        self.park_seconds = 0.0
        self.resume_seconds = 0.0
        self.last_inventory = None
        self._parked = False

    @property
    def phase(self):
        return "host" if self._parked else "device"

    def _managed(self):
        if self._storage.parking == "none":
            return None
        if self.managed is None:
            # Created on the first allocating thread: a managed owner belongs
            # to its creating control thread.
            self.managed = self._storage._owner_factory()
        return self.managed

    @contextmanager
    def allocate(self):
        """The member's workspace turn, allocating into its own pages.

        The first turn is the member's construction: with a residency plan the door's
        single-forecast envelope gate is left to the plan (see
        :meth:`ResidentMemberStorage.construct`).
        """
        from gpuwm.core.resident_admission import memory_gate_override
        storage = self._storage
        managed = self._managed()
        if self._parked:
            raise RuntimeError(f"member {self.member_id!r} is parked; resume it before its turn")
        allocating = managed is not None and (storage.allocation == "turn" or self.turns == 0)
        with memory_gate_override(storage.plan is not None and self.turns == 0):
            with storage.workspace.turn(self.member_id, executor=True):
                with managed.allocate() if allocating else nullcontext():
                    yield self
        self.turns += 1

    def trim(self):
        if self.managed is not None and self.managed.phase == "device":
            self.managed.pool.free_all_blocks()

    def park(self):
        """Park unless the plan keeps this member resident; return the migration row."""
        storage = self._storage
        if self.managed is None or storage.keeps_resident(self.member_id):
            self.trim()
            self._record_inventory()
            return dict(destination="device", bytes=0, seconds=0.0, kept_resident=True)
        started = perf_counter()
        row = self.managed.park()
        self.park_seconds += perf_counter() - started
        self.parks += 1
        self._parked = True
        self._record_inventory()
        storage._log(self.member_id, "park", row)
        return row

    def resume(self):
        if self.managed is None or not self._parked:
            return dict(destination="device", bytes=0, seconds=0.0, kept_resident=True)
        started = perf_counter()
        row = self.managed.resume()
        self.resume_seconds += perf_counter() - started
        self.resumes += 1
        self._parked = False
        self._storage._log(self.member_id, "resume", row)
        return row

    def inventory(self):
        """The managed owner's inventory (what a session's storage receipt reads)."""
        if self.managed is None:
            return dict(phase=self.phase, storage="device", kept_resident=True)
        return dict(self.managed.inventory(),
                    kept_resident=self._storage.keeps_resident(self.member_id))

    def _record_inventory(self):
        if self.managed is not None:
            self.last_inventory = self.managed.inventory()

    def receipt(self):
        return dict(member_id=self.member_id, phase=self.phase, turns=self.turns, parks=self.parks,
                    resumes=self.resumes, park_seconds=self.park_seconds,
                    resume_seconds=self.resume_seconds,
                    kept_resident=self._storage.keeps_resident(self.member_id),
                    managed=self.last_inventory)


class MemberWorkspace:
    """A member's view of the shared workspace; :meth:`initialization` wraps its door initialization."""

    def __init__(self, storage, member_id):
        self._storage = storage
        self.member_id = member_id

    @property
    def scratch(self):
        return self._storage.workspace.scratch

    @property
    def rebuilt(self):
        return self._storage.workspace.rebuilt

    @property
    def member(self):
        """This member while it holds the workspace turn, else ``None``."""
        held = self._storage.workspace.member
        return held if held == self.member_id else None

    def admits(self, domain):
        return self._storage.workspace.admits(domain)

    def initialization(self, base):
        """``base`` (any door initialization) building this member into the shared workspace."""
        from gpuwm.ensemble.workspace_initialization import WorkspaceInitialization
        return WorkspaceInitialization(base, self)


class ResidentMemberStorage:
    def __init__(self, workspace, *, plan=None, parking="managed", allocation="turn",
                 owner_factory=None):
        if parking not in PARKING:
            raise ValueError(f"parking {parking!r} is not one of {PARKING}")
        if allocation not in ALLOCATION_POLICIES:
            raise ValueError(f"allocation {allocation!r} is not one of {ALLOCATION_POLICIES}")
        if parking == "none" and plan is not None and plan.parks:
            raise ValueError("the residency plan parks members and parking is off: the members over the "
                             "resident count would stop in a CUDA out-of-memory")
        self.workspace = workspace
        self.plan = plan
        self.parking = parking
        self.allocation = allocation
        if owner_factory is None:
            from gpuwm.ensemble.managed_storage import ManagedStorage
            owner_factory = ManagedStorage
        self._owner_factory = owner_factory
        self._owners = {}
        self.migrations = []

    def keeps_resident(self, member_id):
        return self.plan is None or self.plan.keeps_resident(member_id)

    def owner(self, member_id):
        owner = self._owners.get(member_id)
        if owner is None:
            owner = self._owners[member_id] = MemberStorageOwner(self, member_id)
        return owner

    def storage_owners(self, member_ids):
        """Owner adapters for ``RetainedMembers(storage_owners=...)``; ``{}`` when nothing parks."""
        if self.parking == "none":
            return {}
        return {member: self.owner(member) for member in member_ids}

    def member(self, member_id):
        return MemberWorkspace(self, member_id)

    @contextmanager
    def turn(self, member_id):
        """Resume if parked, take the workspace turn, then park or trim per the plan."""
        owner = self.owner(member_id)
        owner.resume()
        with owner.allocate():
            yield owner
        # Only after a turn that finished: a failed turn leaves its pages
        # where they are for the caller's failure handling.
        owner.park()

    @contextmanager
    def construct(self, member_id):
        """The member's construction turn: :meth:`turn`, with the door's envelope gate left to the plan.

        The door prices one standalone forecast (its state, physics, boundary tables, workspaces,
        transients, CUDA context and allocator margin) against the card's free memory.  In a
        planned workspace the workspace, the context and the other members are already on the card
        and the residency plan priced this member's own bytes, so the single-forecast envelope
        would refuse members the plan placed.  With a plan, the envelope gate is skipped for the
        construction (``memory_gate_override``); the constructors' exact floors still refuse a
        member that cannot be allocated.  Without a plan the door's gate stands.
        """
        if self.owner(member_id).turns:
            raise RuntimeError(f"member {member_id!r} is already constructed")
        with self.turn(member_id) as owner:
            yield owner

    @contextmanager
    def allocating(self, member_id):
        """Inside a turn under ``allocation='construction'``: route new arrays to the member's pages.

        For allocations a member makes after construction that it keeps (an
        analysis applied at a cycle instant attaches its forcing here).
        """
        owner = self.owner(member_id)
        managed = owner.managed
        if managed is None or self.allocation == "turn":
            yield owner
            return
        if self.workspace.member != member_id:
            raise RuntimeError("an allocation scope belongs inside the member's own turn")
        with managed.allocate():
            yield owner

    def park(self, member_id, *, force=False):
        """Park one member now (an analysis phase); ``force`` parks a pinned member too."""
        owner = self.owner(member_id)
        if owner.managed is None or owner.phase == "host":
            return None
        if not force and self.keeps_resident(member_id):
            return None
        started = perf_counter()
        row = owner.managed.park()
        owner.park_seconds += perf_counter() - started
        owner.parks += 1
        owner._parked = True
        self._log(member_id, "park", row)
        return row

    def resume(self, member_id):
        return self.owner(member_id).resume()

    def _log(self, member_id, kind, row):
        if len(self.migrations) < 100000:
            self.migrations.append(dict(member_id=member_id, kind=kind,
                                        bytes=row.get("bytes"), seconds=row.get("seconds")))

    def receipt(self):
        owners = [owner.receipt() for _, owner in sorted(self._owners.items(), key=lambda kv: str(kv[0]))]
        parks = [row for row in self.migrations if row["kind"] == "park"]
        resumes = [row for row in self.migrations if row["kind"] == "resume"]
        return dict(schema="gpuwm-resident-member-storage/1", parking=self.parking,
                    allocation=self.allocation, workspace=self.workspace.receipt(),
                    plan=None if self.plan is None else self.plan.receipt(),
                    members=owners, parks=len(parks), resumes=len(resumes),
                    park_seconds=sum(row["seconds"] or 0.0 for row in parks),
                    resume_seconds=sum(row["seconds"] or 0.0 for row in resumes),
                    park_bytes=sum(row["bytes"] or 0 for row in parks))


__all__ = ["ALLOCATION_POLICIES", "MemberStorageOwner", "MemberWorkspace", "PARKING",
           "ResidentMemberStorage"]
