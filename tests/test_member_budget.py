"""CPU checks of the resident-member planner and the shared-workspace storage composition."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.member_budget import (CardBudget, ESTIMATED, Figure, MemberBudgetRefused,
                                          MemberBytes, plan_residency)
from gpuwm.ensemble.member_storage import ResidentMemberStorage

GB = 10 ** 9


def member(gb, iau=0.0, residue=0.0, source="measured"):
    return MemberBytes(state=Figure(int(gb * 0.6 * GB), source), physics=Figure(int(gb * 0.3 * GB), source),
                       forcing=Figure(int(gb * 0.1 * GB), source), iau=Figure(int(iau * GB), source),
                       residue=Figure(int(residue * GB), source))


def card(total=24, arena=1, transient=1, context=1, reserve=1, analysis=0, export=0, host=None, source="measured"):
    return CardBudget(total=Figure(total * GB, source), context=Figure(context * GB), arena=Figure(arena * GB),
                      transient=Figure(transient * GB), reserve=Figure(reserve * GB),
                      analysis=Figure(analysis * GB), export=Figure(export * GB),
                      host=None if host is None else Figure(host * GB), label="test card")


def test_everyone_resident_when_the_roster_fits():
    plan = plan_residency({m: member(3) for m in range(1, 7)}, card(total=24))
    assert plan.resident_count == 6 and not plan.parks and plan.rotating == ()
    assert plan.peaks["forecast"] == 4 * GB + sum(m.parkable for m in plan.members.values())
    assert plan.receipt()["source"] == "measured"


def test_pinned_set_and_one_rotating_slot_when_over_budget():
    plan = plan_residency({m: member(3) for m in range(1, 7)}, card(total=16))
    # fixed 3 + transient 1 = 4; four pinned (12) would leave no slot: 3 pinned + slot = 16
    assert len(plan.pinned) == 3 and plan.resident_count == 4 and len(plan.rotating) == 3
    assert plan.peaks["forecast"] <= 16 * GB
    assert all(plan.keeps_resident(m) for m in plan.pinned)
    assert not any(plan.keeps_resident(m) for m in plan.rotating)


def test_iau_forcing_counts_inside_the_window():
    plain = plan_residency({m: member(3) for m in range(1, 5)}, card(total=16))
    forced = plan_residency({m: member(3, iau=0.25) for m in range(1, 5)}, card(total=16))
    assert forced.peaks["iau"] > forced.peaks["forecast"]
    assert forced.resident_count <= plain.resident_count


def test_one_member_that_cannot_step_is_refused_by_its_breakage():
    with pytest.raises(MemberBudgetRefused, match="first radiation step would stop in a CUDA out-of-memory"):
        plan_residency({1: member(10)}, card(total=12, transient=2))


def test_parking_off_refuses_a_roster_over_the_resident_count():
    with pytest.raises(MemberBudgetRefused, match="with parking off"):
        plan_residency({m: member(3) for m in range(1, 7)}, card(total=16), parking="none")


def test_parked_pages_over_the_host_budget_are_refused():
    with pytest.raises(MemberBudgetRefused, match="exceed the host memory"):
        plan_residency({m: member(3) for m in range(1, 37)}, card(total=16, host=20))


def test_analysis_peak_parks_pinned_members():
    plan = plan_residency({m: member(3) for m in range(1, 7)}, card(total=24, analysis=8, export=2))
    assert not plan.parks and plan.park_during_analysis
    assert plan.peaks["analysis"] <= 24 * GB
    with pytest.raises(MemberBudgetRefused, match="even with every member parked"):
        plan_residency({1: member(3)}, card(total=12, analysis=10))


def test_an_estimated_figure_labels_the_plan():
    plan = plan_residency({1: member(3, source=ESTIMATED), 2: member(3)}, card(total=24))
    assert plan.receipt()["source"] == ESTIMATED
    assert any("estimate" in note for note in plan.notes)


class FakeWorkspace:
    def __init__(self):
        self.member = None
        self.events = []
        self.scratch = object()
        self.rebuilt = object()

    @contextmanager
    def turn(self, member_id, *, executor=False):
        assert executor and self.member is None
        self.member = member_id
        self.events.append(("turn", member_id))
        try:
            yield self
        finally:
            self.member = None

    def admits(self, domain):
        return None

    def receipt(self):
        return {}


class FakeManaged:
    log = []

    def __init__(self):
        self.phase = "device"
        self.pool = SimpleNamespace(free_all_blocks=lambda: FakeManaged.log.append(("trim", id(self))))

    @contextmanager
    def allocate(self):
        FakeManaged.log.append(("allocate", id(self)))
        yield self

    def park(self):
        self.phase = "host"
        return dict(destination="host", bytes=10, seconds=0.5)

    def resume(self):
        self.phase = "device"
        return dict(destination="device", bytes=10, seconds=0.25)

    def inventory(self):
        return dict(phase=self.phase, backing_bytes=10)


def test_storage_parks_only_rotating_members_and_holds_the_workspace_turn():
    plan = plan_residency({m: member(3) for m in range(1, 7)}, card(total=16))
    storage = ResidentMemberStorage(FakeWorkspace(), plan=plan, owner_factory=FakeManaged)
    for cycle in range(2):
        for m in range(1, 7):
            with storage.turn(m) as owner:
                assert storage.workspace.member == m
                assert storage.member(m).member == m and storage.member(m + 100).member is None
    receipt = storage.receipt()
    rotating = len(plan.rotating)
    assert receipt["parks"] == 2 * rotating and receipt["resumes"] == rotating
    for m in plan.pinned:
        assert storage.owner(m).phase == "device" and storage.owner(m).parks == 0
    for m in plan.rotating:
        assert storage.owner(m).phase == "host"


def test_owner_adapters_have_the_retained_member_surface_and_construction_policy():
    FakeManaged.log.clear()
    storage = ResidentMemberStorage(FakeWorkspace(), plan=None, allocation="construction",
                                    owner_factory=FakeManaged)
    owners = storage.storage_owners([1, 2])
    assert set(owners) == {1, 2}
    for owner in owners.values():
        assert all(callable(getattr(owner, name)) for name in ("allocate", "park", "resume"))
        assert owner.phase == "device"
    with owners[1].allocate():
        pass
    with owners[1].allocate():
        with storage.allocating(1):
            pass
    allocations = [event for event in FakeManaged.log if event[0] == "allocate"]
    # construction allocates; the plain second turn does not; the explicit scope does
    assert len(allocations) == 2
    assert owners[1].park()["kept_resident"]
    assert ResidentMemberStorage(FakeWorkspace(), parking="none").storage_owners([1]) == {}


def test_workspace_initialization_substitutes_the_shared_arena_and_binds_the_model():
    from gpuwm.ensemble.workspace_initialization import (WorkspaceInitialization,
                                                         WorkspaceInitializationRefused)
    from gpuwm.forecast_initialization import DomainInitialization

    calls = []

    class Base:
        lateral_boundaries = "tables"

        def restore_domain(self, domain, grid, bundle, *, start_time, scratch_arena, dycore_state_workspace):
            calls.append((scratch_arena, dycore_state_workspace))
            state = SimpleNamespace(_scratch_arena=scratch_arena)
            return DomainInitialization(SimpleNamespace(state=state), lambda: calls.append(("physics", state._scratch_arena)))

        def after_restore(self, **kw):
            calls.append(("after", kw["model"]._dycore_state_workspace))

        def verify_inputs(self, inputs):
            calls.append("verified")

    storage = ResidentMemberStorage(FakeWorkspace(), parking="none")
    ws = storage.workspace
    init = storage.member(7).initialization(Base())
    assert init.lateral_boundaries == "tables"
    one = SimpleNamespace(experiment=SimpleNamespace(domains=(1,)))
    init.verify_inputs(one)
    with pytest.raises(WorkspaceInitializationRefused, match="share their own arena"):
        init.verify_inputs(SimpleNamespace(experiment=SimpleNamespace(domains=(1, 2))))
    domain = SimpleNamespace(grid_id=1, run=None)
    with pytest.raises(WorkspaceInitializationRefused, match="outside its workspace turn"):
        init.restore_domain(domain, None, None, start_time=0, scratch_arena=None, dycore_state_workspace=None)
    with storage.turn(7):
        with pytest.raises(WorkspaceInitializationRefused, match="tree's shared arena"):
            init.restore_domain(domain, None, None, start_time=0, scratch_arena=object(),
                                dycore_state_workspace=None)
        restored = init.restore_domain(domain, None, None, start_time=0, scratch_arena=None,
                                       dycore_state_workspace=None)
        restored.state._scratch_arena = None      # what the runner records on a single domain
        restored.initialize_physics()
        model = SimpleNamespace(_scratch_arena=None, _dycore_state_workspace=None)
        init.after_restore(model=model, nodes={1: SimpleNamespace(state=restored.state)}, start_time=0,
                           run_seconds=1.0, restart=None)
    assert calls[1] == (ws.scratch, ws.rebuilt)
    assert calls[2] == ("physics", ws.scratch)
    assert model._scratch_arena is ws.scratch and model._dycore_state_workspace is ws.rebuilt
    assert calls[3] == ("after", ws.rebuilt)


def test_construction_leaves_the_envelope_gate_to_a_plan():
    import os
    from gpuwm.core.resident_admission import MEMORY_GATE_OVERRIDE_ENV
    plan = plan_residency({m: member(3) for m in range(1, 4)}, card(total=24))
    storage = ResidentMemberStorage(FakeWorkspace(), plan=plan, parking="none")
    with storage.construct(1):
        assert os.environ.get(MEMORY_GATE_OVERRIDE_ENV) == "1"
    assert os.environ.get(MEMORY_GATE_OVERRIDE_ENV) in (None, "")
    unplanned = ResidentMemberStorage(FakeWorkspace(), parking="none")
    with unplanned.construct(1):
        assert os.environ.get(MEMORY_GATE_OVERRIDE_ENV) in (None, "")
