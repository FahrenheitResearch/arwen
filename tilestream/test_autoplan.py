"""Tests for :mod:`tilestream.autoplan`, negative controls included.

A planner is an unusually easy thing to write a vacuous test for.  "It
returned a plan" proves nothing; "it refused" proves nothing either, because
a planner that refuses everything passes every refusal test ever written.  So
every capability here is checked twice -- once for the answer, and once for a
CONTROL that fails if the feature is switched off or quietly broken:

===========================  ===============================================
capability                   the control that would catch it being fake
===========================  ===============================================
fits resident                a domain one notch larger must NOT fit resident
needs tiling                 the tile must shrink when the card does
needs host RAM               the store must exceed VRAM, and be pinnable
cannot run                   a domain one notch SMALLER must plan fine
halo from ``halo_radius``    it must change with ``time_step_sound``
tile divides the domain      a prime extent must be reported as ragged
the VRAM model itself        allocate the plan and compare (GPU only)
===========================  ===============================================

The last row is the one that matters most.  Everything above it tests that
the planner is self-consistent; only that one tests that the planner is
RIGHT, and it is the only test here that touches a GPU.  Run it with::

    python -m tilestream.test_autoplan            # arithmetic only
    python -m tilestream.test_autoplan --gpu      # plus the allocation check
"""

from __future__ import annotations

import sys

from tilestream import autoplan as A
from tilestream import harness as H

GIB = A.GIB
MIB = A.MIB


def _machine(vram_gib: float, host_gib: float, name: str = "test") -> A.Machine:
    return A.Machine(int(vram_gib * GIB), int(host_gib * GIB), name=name)


M5090 = _machine(32, 125, "RTX 5090")
M4090 = _machine(24, 128, "RTX 4090")
M5070 = _machine(12, 64, "RTX 5070")


# --------------------------------------------------------------------------
# rung classification
# --------------------------------------------------------------------------

def test_rung_classification_reads_the_authority_not_the_selector():
    """``ra_sw_physics = -1`` is "unset", not "RRTMGP is on".

    The dry harness config leaves both radiation selectors at ``-1``.  A
    truthiness test on them classifies every dry domain as ``full`` and books
    3.2 GiB of radiation footprint that does not exist -- which is what this
    test caught the first time it ran.
    """
    for rung in ("dry", "moist", "full", "full+mynn+noahmp"):
        cfg = A._config_for_rung(256, 256, 49, rung)
        assert A.rung_of(cfg) == rung, (rung, A.rung_of(cfg))
    dry = A._config_for_rung(256, 256, 49, "dry")
    assert int(dry.ra_sw_physics) == -1, (
        "this test is only meaningful while the dry default really is -1; "
        "it has changed, so re-derive the classification")


def test_footprints_are_ordered_by_rung():
    """A dearer rung must cost more per cell, or the table is scrambled."""
    order = ["dry", "moist", "full", "full+mynn+noahmp"]
    per_cell = [A.FOOTPRINTS[r].bytes_per_cell for r in order]
    assert per_cell == sorted(per_cell), per_cell
    store = [A.FOOTPRINTS[r].store_bytes_per_cell for r in order]
    assert store == sorted(store), store


def test_measured_dry_store_reproduces_the_projects_own_number():
    """32.3 B/cell here against 32.26 derived independently by the inventory
    survey.  Two different measurements of the same quantity."""
    assert abs(A.FOOTPRINTS["dry"].store_bytes_per_cell - 32.26) < 0.1


# --------------------------------------------------------------------------
# fits resident, and the control
# --------------------------------------------------------------------------

def test_small_domain_fits_resident():
    cfg = A._config_for_rung(256, 256, 49, "dry")
    p = A.plan(cfg, M5090)
    assert p.mode == "resident", p.explain()
    assert p.run_kwargs == {}, "a resident plan must not ask for a tiling"
    assert p.store_bytes == 0 and p.arena_bytes == 0


def test_resident_boundary_is_not_vacuous():
    """CONTROL: the resident answer must FLIP at a size, and the size must be
    where the model says it is.

    A planner that always answers "resident" passes the test above.  This one
    walks the domain up until the answer changes and then checks that the
    change happened within one step of the modelled capacity, so both the
    decision and the arithmetic behind it are under test.
    """
    fp = A.FOOTPRINTS["dry"]
    budget = M5090.vram_budget_bytes
    n = 16
    while A.plan(A._config_for_rung(n, n, 49, "dry"), M5090).mode == "resident":
        n *= 2
        assert n < 1 << 16, "nothing ever stopped fitting resident"
    smaller = n // 2
    assert fp.resident_bytes(smaller * smaller * 49) <= budget
    assert fp.resident_bytes(n * n * 49) > budget
    assert A.plan(A._config_for_rung(smaller, smaller, 49, "dry"),
                  M5090).mode == "resident"


def test_a_tiny_card_does_not_fit_what_a_big_one_does():
    """CONTROL: the same domain, two machines, two different decisions."""
    cfg = A._config_for_rung(512, 512, 49, "full+mynn+noahmp")
    assert A.plan(cfg, M5090).mode == "resident"
    assert A.plan(cfg, M5070).mode == "tiled"


# --------------------------------------------------------------------------
# needs tiling
# --------------------------------------------------------------------------

def test_large_domain_is_tiled_and_the_tile_divides_it():
    cfg = A._config_for_rung(4096, 4096, 49, "dry")
    p = A.plan(cfg, M5090)
    assert p.mode == "tiled"
    assert 4096 % p.tile_nx == 0 and 4096 % p.tile_ny == 0, p.explain()
    assert not p.ragged
    assert p.ntiles_x * p.tile_nx == 4096
    assert p.nbuffers >= 2, "overlap is worth 1.32x and it fits here"


def test_the_tile_shrinks_when_the_card_does():
    """CONTROL: if the VRAM budget were ignored the tile would not move.

    Compared on window AREA rather than on ``tile_nx``, because the two axes
    are searched independently and a smaller card can answer with a long thin
    tile: 4096^2 dry comes out 1024x1024 on 32 GiB and 1024x512 on 12 GiB.
    """
    cfg = A._config_for_rung(4096, 4096, 49, "dry")
    plans = [A.plan(cfg, m) for m in (M5090, M4090, M5070)]
    areas = [p.window_nx * p.window_ny for p in plans]
    assert areas[0] >= areas[1] >= areas[2], areas
    assert areas[0] > areas[2], (
        f"the 32 GiB and the 12 GiB card planned the same window {areas}; "
        "the VRAM budget is not reaching the search")
    reds = [p.redundancy for p in plans]
    assert reds[0] <= reds[2], reds


def test_no_affordable_tiling_beats_the_planned_one():
    """The objective, checked by brute force over the whole candidate space.

    "Largest tile that fits" is a heuristic for the real objective, and where
    the two disagree the objective is right: a 4128 x 325 window is larger
    than a 1056 x 1056 one and fits the same card, and it is a much worse
    plan (1.90x redundancy against 1.06x).  So what is asserted is that no
    affordable candidate has a lower redundancy, not that none is bigger.
    """
    cfg = A._config_for_rung(4096, 4096, 49, "dry")
    p = A.plan(cfg, M4090)
    fp = p.footprint
    for tx in A.tile_candidates(4096):
        for ty in A.tile_candidates(4096):
            window = (tx + 2 * p.halo) * (ty + 2 * p.halo) * 49
            if fp.vram_bytes(window, p.nbuffers) > p.vram_budget_bytes:
                continue
            red = A.redundancy(4096, 4096, tx, ty, p.halo)
            assert red >= p.redundancy * (1 - 1e-9), (
                f"tile {tx}x{ty} fits and does {red:.4f}x of the work "
                f"against the planned {p.tile_nx}x{p.tile_ny} at "
                f"{p.redundancy:.4f}x")


def test_plan_fits_its_own_budget():
    for machine in (M5090, M4090, M5070):
        for rung in ("dry", "moist", "full", "full+mynn+noahmp"):
            for n in (512, 1024, 2048):
                cfg = A._config_for_rung(n, n, 49, rung)
                try:
                    p = A.plan(cfg, machine)
                except A.CannotPlan:
                    continue
                # Compared against the MACHINE, not against the plan's own
                # record of its budget: the latter is self-referential and a
                # planner that ignored the budget entirely would still pass.
                assert p.vram_bytes <= machine.vram_bytes, p.explain()
                assert p.host_bytes <= machine.host_bytes, p.explain()
                assert p.vram_bytes <= p.vram_budget_bytes, p.explain()
                assert p.host_bytes <= p.host_budget_bytes, p.explain()


def test_run_kwargs_are_what_run_tiled_takes():
    """The plan must be spendable without retyping, and ``plan_tiles`` must
    accept the geometry -- which is where a non-periodic domain with a window
    wider than itself would blow up."""
    from tilestream import spec as tspec

    for periodic in (True, False):
        overrides = {} if periodic else dict(periodic=False, specified=True)
        cfg = A._config_for_rung(2048, 1536, 49, "dry", **overrides)
        p = A.plan(cfg, M4090)
        kw = p.run_kwargs
        assert set(kw) == {"tile_nx", "tile_ny", "halo", "nbuffers",
                           "write_mode", "periodic", "periodic_x",
                           "periodic_y"}
        assert kw["periodic"] is periodic
        assert kw["periodic_x"] is periodic and kw["periodic_y"] is periodic
        specs = tspec.plan_tiles(2048, 1536, kw["tile_nx"], kw["tile_ny"],
                                 kw["halo"], kw["periodic"],
                                 periodic_x=kw["periodic_x"],
                                 periodic_y=kw["periodic_y"])
        tspec.validate_plan(specs, 1536, 2048)
        assert len(specs) == p.ntiles


def test_non_periodic_window_never_exceeds_the_domain():
    """CONTROL: ``plan_tiles`` REFUSES a non-periodic window wider than the
    domain, so a planner that ignores the boundary flags raises instead of
    running slowly."""
    cfg = A._config_for_rung(320, 320, 49, "dry", periodic=False,
                             specified=True)
    p = A.plan(cfg, _machine(4, 64))
    assert p.window_nx <= 320 and p.window_ny <= 320, p.explain()
    assert not A.is_periodic(cfg)


# --------------------------------------------------------------------------
# the halo
# --------------------------------------------------------------------------

def test_halo_comes_from_halo_radius_and_moves_with_sound_steps():
    """CONTROL: a planner that hardcodes 16 passes at ns=4 and fails here."""
    for ns, want in ((4, 16), (6, 19), (8, 22)):
        cfg = A._config_for_rung(2048, 2048, 49, "dry", time_step_sound=ns)
        p = A.plan(cfg, M4090)
        assert p.halo == want == H.halo_radius(cfg), (ns, p.halo)
    small = A.plan(A._config_for_rung(2048, 2048, 49, "dry",
                                      time_step_sound=4), M4090)
    big = A.plan(A._config_for_rung(2048, 2048, 49, "dry",
                                    time_step_sound=8), M4090)
    assert big.redundancy > small.redundancy, (
        "a wider halo must cost more redundancy at the same tile; it did not, "
        "so the halo is not reaching the geometry")


def test_plan_takes_no_halo_argument():
    """The halo is not a knob.  If it ever becomes one, this fails."""
    import inspect

    assert "halo" not in inspect.signature(A.plan).parameters


# --------------------------------------------------------------------------
# needs host RAM
# --------------------------------------------------------------------------

def test_needs_host_ram_and_the_store_is_larger_than_the_card():
    cfg = A._config_for_rung(5120, 5120, 49, "dry")
    p = A.plan(cfg, M5090)
    assert p.mode == "tiled"
    assert p.store_bytes > M5090.vram_bytes, (
        "this case is supposed to be one that CANNOT live in VRAM")
    assert p.host_bytes <= p.host_budget_bytes
    assert 0.02 <= p.arena_bytes / p.store_bytes <= 0.10, (
        f"ring arena {p.arena_bytes / p.store_bytes:.1%}; an exact tiling "
        "measured 2-6% and anything outside that is a geometry bug")


def test_shadow_costs_a_whole_second_store():
    cfg = A._config_for_rung(4096, 4096, 49, "dry")
    ring = A.plan(cfg, M5090, write_mode="ring")
    shadow = A.plan(cfg, M5090, write_mode="shadow")
    assert abs(shadow.arena_bytes - shadow.store_bytes) < 1
    assert shadow.host_bytes > 1.8 * ring.host_bytes


def test_host_ram_is_the_binding_resource_when_it_is():
    """A domain that a big card could tile all day, on a box with no RAM."""
    cfg = A._config_for_rung(8192, 8192, 49, "dry")
    try:
        A.plan(cfg, _machine(32, 16))
    except A.CannotPlan as exc:
        assert exc.resource == "host", exc.resource
        assert "pinned" in str(exc)
    else:
        raise AssertionError("a 106 GiB store fitted in 16 GiB of host RAM")


def test_host_budget_never_comes_from_meminfo_in_a_container(monkeypatch=None):
    """CONTROL for the ``/proc/meminfo`` lie.

    Simulated rather than mocked: the container's cgroup limit and the host's
    MemTotal are handed to the same arithmetic and the smaller must win, which
    is the whole of the rule.  On the box these constants were measured on the
    two numbers were 241.7 GiB and 503 GiB.
    """
    cgroup, memtotal = int(241.7 * GIB), int(503 * GIB)
    chosen = min(cgroup, memtotal)
    assert chosen == cgroup
    m = A.Machine(24 * GIB, chosen, host_source="cgroup limit")
    assert m.host_budget_bytes < int(A.PINNED_FRACTION * memtotal), (
        "sizing from MemTotal would have pinned "
        f"{A.PINNED_FRACTION * memtotal / GIB:.0f} GiB inside a "
        f"{cgroup / GIB:.0f} GiB container")


# --------------------------------------------------------------------------
# cannot run
# --------------------------------------------------------------------------

def test_cannot_run_names_the_binding_resource():
    huge = A._config_for_rung(16384, 16384, 49, "full+mynn+noahmp")
    try:
        A.plan(huge, M5070)
    except A.CannotPlan as exc:
        assert exc.resource in ("host", "vram"), exc.resource
    else:
        raise AssertionError("a 3.4 TiB store was planned onto a 64 GiB box")


def test_refusal_is_not_vacuous():
    """CONTROL: one notch smaller must PLAN.

    A planner that refuses everything passes every refusal test.  This walks
    the domain down from the refused size until it is accepted and asserts
    that the boundary exists at all -- and that it is a boundary in the domain
    size, not a flat "no".
    """
    machine = M5070
    n = 16384
    cfg = A._config_for_rung(n, n, 49, "full+mynn+noahmp")
    try:
        A.plan(cfg, machine)
    except A.CannotPlan:
        pass
    else:
        raise AssertionError("expected a refusal to start from")
    while n > 16:
        n //= 2
        try:
            p = A.plan(A._config_for_rung(n, n, 49, "full+mynn+noahmp"),
                       machine)
        except A.CannotPlan:
            continue
        assert p.mode in ("tiled", "resident")
        return
    raise AssertionError("every size was refused on a 12 GiB / 64 GiB box")


def test_vram_refusal_when_the_process_fixed_cost_alone_is_too_big():
    """The refusal that is NOT about the domain at all.

    ``full+mynn+noahmp`` costs 2.5 GiB per process before a single tile
    exists.  On a 2 GiB card no tile size whatsoever helps, and the message
    has to say so rather than suggesting a smaller tile.
    """
    cfg = A._config_for_rung(2048, 2048, 49, "full+mynn+noahmp")
    try:
        A.plan(cfg, _machine(2, 256))
    except A.CannotPlan as exc:
        assert exc.resource == "vram", exc.resource
        assert "per-process fixed cost" in str(exc), str(exc)
    else:
        raise AssertionError("a 2 GiB card planned a full-physics tile")


def test_redundancy_limit_refuses_a_tile_that_is_almost_all_halo():
    """A run that would technically proceed while spending the card on halo."""
    # A card sized, from the model itself, to admit exactly a 56-cell window:
    # interior 24 inside a 32-cell halo on each side, so 5.5x of the work is
    # halo.  Building the machine from the footprint rather than picking a
    # round number keeps the test pinned to the regime it is about.
    fp = A.FOOTPRINTS["dry"]
    budget = fp.vram_bytes(56 * 56 * 49, 1)
    small = A.Machine(int(budget / (1 - A.VRAM_HEADROOM)), 1024 * GIB,
                      name="tiny")
    cfg = A._config_for_rung(4096, 4096, 49, "dry")
    try:
        A.plan(cfg, small)
    except A.CannotPlan as exc:
        assert exc.resource == "vram", exc.resource
        assert "halo cells" in str(exc), str(exc)
    else:
        raise AssertionError("a card with room for a 56-cell window planned "
                             "4096^2 without complaint")
    # CONTROL: the same case must SUCCEED with the limit lifted, or the
    # refusal is coming from somewhere else entirely.
    p = A.plan(cfg, small, max_redundancy=None)
    assert p.redundancy > 4.0, p.redundancy
    assert p.window_nx <= 64, p.window_nx     # 25 + 2*16, one cell of rounding


def test_a_domain_too_small_to_tile_names_geometry_not_vram():
    """The scaling lane's finding, reproduced and pinned: a SMALLER forced-
    tiled arm died CannotPlan while every larger arm planned fine.

    At 32^2 non-periodic and halo 16 the transport itself is the constraint:
    ``spec.plan_tiles`` refuses ``tile + 2*halo > nx`` on a clamped axis, so
    tile_nx <= 32 - 32 = 0 and NO tile is legal at ANY budget.  The refusal
    used to say "no tile fits in 13.26 GiB of VRAM ... the smallest legal
    compute window at halo 16 is 33^2" -- a resource a bigger card would fix
    and a window the transport would refuse, both false.  A refusal names
    the concrete breakage or it does not exist; this one's breakage is the
    domain's own geometry.
    """
    small = A._config_for_rung(32, 32, 49, "dry", periodic=False,
                               specified=True)
    try:
        A.plan(small, M5090, prefer_resident=False)
    except A.CannotPlan as exc:
        assert exc.resource == "geometry", (exc.resource, str(exc))
        assert "non-periodic" in str(exc), str(exc)
        assert "resident" in str(exc), (
            f"the remedy for a domain this small is to run it resident, and "
            f"the refusal has to say so: {exc}")
        assert "GiB of VRAM" not in str(exc), (
            f"the refusal still blames the card for a geometry bound: {exc}")
    else:
        raise AssertionError("a 32^2 non-periodic domain planned a tiling "
                             "the transport cannot serve at halo 16")
    # CONTROL 1: the same domain is not cursed -- resident preference plans.
    p = A.plan(small, M5090)
    assert p.mode == "resident", p.explain()
    # CONTROL 2, the inversion itself: a LARGER domain, same machine, same
    # forced tiling, plans.  The boundary is the domain size, not the card.
    bigger = A._config_for_rung(64, 64, 49, "dry", periodic=False,
                                specified=True)
    p2 = A.plan(bigger, M5090, prefer_resident=False)
    assert p2.mode == "tiled", p2.explain()


def test_redundancy_refusal_names_geometry_when_the_domain_caps_the_tile():
    """The same misattribution one size up, where tiles EXIST but are capped.

    At 48^2 non-periodic and halo 16 the transport admits tiles up to
    48 - 32 = 16, so the best legal tiling does 9x the necessary work AT ANY
    BUDGET -- yet the refusal said "This is a VRAM problem".  Its sibling
    (``test_redundancy_limit_refuses_a_tile_that_is_almost_all_halo`` above)
    is the case where VRAM really is the cap, and it must KEEP saying vram --
    the two tests together pin that the resource label is derived, not
    hardcoded either way.
    """
    cfg = A._config_for_rung(48, 48, 49, "dry", periodic=False,
                             specified=True)
    try:
        A.plan(cfg, M5090, prefer_resident=False)
    except A.CannotPlan as exc:
        assert exc.resource == "geometry", (exc.resource, str(exc))
        assert "max_redundancy=None" in str(exc), (
            f"the run-it-anyway remedy was lost from the refusal: {exc}")
        assert "VRAM problem" not in str(exc), str(exc)
    else:
        raise AssertionError("a 9x-redundant tiling was planned without "
                             "complaint on a 48^2 clamped domain")
    # CONTROL: the refusal is the limit's, not a blanket one -- lifted, the
    # same case plans, at the redundancy the message quoted.
    p = A.plan(cfg, M5090, prefer_resident=False, max_redundancy=None)
    assert p.mode == "tiled" and p.redundancy > 4.0, p.explain()


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def test_ring_arena_fraction_matches_every_measured_plan():
    """The nine arenas ``rings.py`` has measured, in bytes, against this model
    in cells.  Within 0.6 percentage points and always low."""
    cases = [
        (1950, 650, 0.0497), (3276, 546, 0.0584), (4608, 768, 0.0417),
        (5120, 512, 0.0619), (5395, 1079, 0.0298), (5412, 1353, 0.0239),
        (192, 64, 0.4429), (3276, 512, 0.2446), (5182, 512, 0.2273),
    ]
    for n, tile, measured in cases:
        got = A.ring_arena_fraction(n, n, tile, tile, 16)
        assert got <= measured + 1e-9, (n, tile, got, measured)
        assert measured - got < 0.006, (n, tile, got, measured)


def test_ring_model_needs_the_ragged_overhang_term():
    """CONTROL: without the overhang the two ragged rows collapse onto the
    exact-tiling answer, which is 4x too cheap."""
    exact = A.ring_arena_fraction(3276, 3276, 546, 546, 16)
    ragged = A.ring_arena_fraction(3276, 3276, 512, 512, 16)
    assert ragged > 4.0 * exact, (exact, ragged)


def test_a_ragged_tiling_is_labelled_and_priced():
    """A prime extent has no usable divisor, so the planner goes ragged --
    and must say so, and must put a number on it."""
    cfg = A._config_for_rung(4099, 4099, 49, "dry")     # 4099 is prime
    p = A.plan(cfg, M4090)
    assert p.ragged, p.explain()
    assert p.ntiles_x * p.tile_nx >= 4099
    assert any("does NOT divide" in line for line in p.notes + p.warnings)
    assert any("ring arena" in line for line in p.notes + p.warnings)


def test_a_badly_ragged_tiling_is_warned_about_and_a_fix_suggested():
    """The measured 22-25% arena case, not the two-cells-short one.

    3276^2 at tile 512 leaves 13 ragged tiles and costs 24.46% of the store
    (measured); 546 divides it exactly and costs 5.84%.  The warning has to
    fire on the first and not on the second, or it is noise.
    """
    frac = A.ring_arena_fraction(3276, 3276, 512, 512, 16)
    assert frac > A.EXPENSIVE_ARENA
    assert A.ring_arena_fraction(3276, 3276, 546, 546, 16) < A.EXPENSIVE_ARENA
    suggestions = A.suggest_friendly_domains(3276, 512)
    assert suggestions and all(n % t == 0 for n, t in suggestions), suggestions


def test_forbidding_ragged_tiles_changes_the_answer():
    """CONTROL: if the raggedness handling were a no-op, ``allow_ragged=False``
    would return the same plan everywhere.  On a prime extent there is no
    exact tiling at all and the refusal must name GEOMETRY, not VRAM -- a
    bigger card would not help."""
    cfg = A._config_for_rung(4099, 4099, 49, "dry")
    assert A.plan(cfg, M4090, allow_ragged=True).ragged
    try:
        A.plan(cfg, M4090, allow_ragged=False)
    except A.CannotPlan as exc:
        assert exc.resource == "geometry", exc.resource
        assert "divides by" in str(exc), str(exc)
    else:
        raise AssertionError("4099 is prime; no tile can divide it")


def test_arena_tie_break_beats_the_plain_divisor_rule():
    """The case where "prefer a tile that divides the domain" is wrong.

    nx = 4098 = 2*3*683, so the exact ladder jumps 683 -> 1366.  A 1025 tile
    leaves a 1023-cell trailing tile -- ragged by two cells -- and has BOTH
    less redundancy and less arena than the exact 683.  A planner that
    preferred division for its own sake would take the worse plan.
    """
    exact = (A.redundancy(4098, 4098, 683, 683, 16),
             A.ring_arena_fraction(4098, 4098, 683, 683, 16))
    nearly = (A.redundancy(4098, 4098, 1025, 1025, 16),
              A.ring_arena_fraction(4098, 4098, 1025, 1025, 16))
    assert nearly[0] < exact[0] and nearly[1] < exact[1], (exact, nearly)
    cfg = A._config_for_rung(4098, 4098, 49, "dry")
    default = A.plan(cfg, M4090)
    relaxed = A.plan(cfg, M4090, prefer_exact=False)
    assert not default.ragged, default.explain()
    assert relaxed.ragged and relaxed.redundancy < default.redundancy, (
        default.redundancy, relaxed.redundancy)


def test_redundancy_counts_the_full_window_of_a_ragged_tile():
    """``plan_tiles`` gives every tile the same window, so a ragged tiling
    pays for a full window and gets a part-tile out of it."""
    from tilestream import spec as tspec

    for nx, ny, tx, ty in ((4096, 4096, 512, 512), (4099, 4099, 456, 456),
                           (2048, 1536, 256, 384)):
        specs = tspec.plan_tiles(nx, ny, tx, ty, 16, True)
        want = sum(s.cnx * s.cny for s in specs) / (nx * ny)
        assert abs(A.redundancy(nx, ny, tx, ty, 16) - want) < 1e-9


def test_tile_candidates_contain_every_reachable_divisor():
    """``ceil(n/k)`` over k enumerates every divisor whose tile count is
    within the search bound, which is what makes an exact tiling reachable at
    all."""
    for n in (4096, 3276, 1950, 5395):
        candidates = set(A.tile_candidates(n))
        missing = {d for d in A._divisors(n)
                   if -(-n // d) <= 4096} - candidates
        assert not missing, (n, sorted(missing)[:10])


# --------------------------------------------------------------------------
# buffers
# --------------------------------------------------------------------------

def test_two_buffers_are_taken_whenever_they_fit():
    """The per-BUFFER fixed cost is small (measured), so overlap is affordable
    even on the 12 GiB card -- which is the finding the planner rests on."""
    cfg = A._config_for_rung(2048, 2048, 49, "dry")
    for machine in (M5090, M4090, M5070):
        p = A.plan(cfg, machine)
        assert p.nbuffers >= 2, (machine.name, p.explain())


def test_the_fixed_cost_is_not_charged_per_buffer():
    """The measurement this module exists to have made, stated as behaviour.

    On a 12 GiB card at full physics, a second buffer costs its own cells
    plus a 1 GiB per-buffer fixed -- NOT another 2.5 GiB of process fixed.
    So the window a second buffer costs is 42% of the single-buffer window,
    not 23%.  Under the assumption this replaced (fixed is per buffer) the
    ratio drops below a third, which is the discriminating number.
    """
    fp = A.FOOTPRINTS["full+mynn+noahmp"]
    budget = _machine(12, 64).vram_budget_bytes
    one = A._max_window_cells(fp, 1, budget)
    two = A._max_window_cells(fp, 2, budget)
    assert 0 < two < one, (one, two)
    assert two / one > 0.35, (
        f"a second buffer costs all but {two / one:.1%} of the window, which "
        "is what a per-BUFFER fixed cost would do; the measurement says the "
        "fixed part is per process")
    assert fp.buffer_fixed_bytes < fp.process_fixed_bytes


def test_the_fixed_cost_is_not_charged_per_domain():
    """The same measurement, restated for DOMAINS -- a tree is one process.

    ``vram_bytes`` prices ONE domain in ONE process, context included, so a
    caller that walks a tree and subtracts it per domain charges the CUDA
    context and the k-distribution tables once per GRID.  At the ``full``
    rung that is 3.760 GiB of phantom bytes for every domain after the
    first, and 7.519 GiB on a three-domain tree -- which is what refused a
    tree priced at 12.4 GiB on a card with 15.2 GiB free.

    The discriminating number is the DIFFERENCE between the whole price and
    the marginal one: it must be the process overhead exactly, and it must
    not move with the window or the buffer count, because none of what is
    in it is per window or per buffer.
    """
    fp = A.FOOTPRINTS["full"]
    overhead = fp.process_overhead_bytes
    assert abs(overhead / GIB - 3.760) < 0.005, overhead / GIB
    for window, nbuffers in ((331 * 331 * 49, 1), (64 * 64 * 49, 3),
                             (1024 * 512 * 49, 2)):
        whole = fp.vram_bytes(window, nbuffers)
        marginal = fp.marginal_bytes(window, nbuffers)
        assert abs((whole - marginal) - overhead) < 1.0, (window, nbuffers)
        assert marginal > 0
    cells = 331 * 331 * 49
    correct = overhead + 3 * fp.marginal_resident_bytes(cells)
    per_domain = 3 * fp.resident_bytes(cells)
    assert abs((per_domain - correct) / GIB - 7.519) < 0.01, (
        f"the phantom on a three-domain tree is "
        f"{(per_domain - correct) / GIB:.3f} GiB, not 7.519")


def test_every_rung_pays_its_overhead_once_and_only_once():
    """CONTROL for the above: the identity must hold at EVERY rung.

    A ``marginal_bytes`` that happened to be right at one rung and wrong at
    another would pass the test above and still mis-price a mixed tree,
    which is the shape a nest ladder with radiation on one rung actually is.
    """
    for name, fp in A.FOOTPRINTS.items():
        overhead = fp.process_overhead_bytes
        assert overhead >= A.CUDA_CONTEXT_BYTES, name
        assert abs(fp.resident_bytes(1 << 20)
                   - fp.marginal_resident_bytes(1 << 20)
                   - overhead) < 1.0, name


# --------------------------------------------------------------------------
# the radiation transient: reserved, never claimed
# --------------------------------------------------------------------------

def test_the_radiation_transient_is_reserved_and_replaces_the_headroom():
    """MEASURED, both sides, on the run in the module docstring.

    A three-domain 9/3/1 km forecast on a 15.92 GiB card with 15.245 GiB
    free peaked at 15.46 GiB against a 12.72 GiB steady state: +2.74 GiB of
    RRTMGP per-call transient that this model did not price.  The reservation
    is the LARGER of the percentage headroom and that measurement, never
    their sum -- stacking them plans a tiling 40% smaller than the one that
    ran, and charging neither plans the tiling that would have died.
    """
    free = int(15.245 * GIB)
    machine = A.Machine(vram_bytes=free, host_bytes=int(123.25 * GIB))
    full = A.FOOTPRINTS["full"]
    headroom = free - machine.vram_budget_bytes

    assert full.radiation_transient_bytes > headroom, (
        "this test is only meaningful while the measured transient is the "
        "bigger of the two on this card; re-derive it")
    budget = A.budget_for(machine, full)
    assert abs(budget / GIB - 12.505) < 0.01, budget / GIB
    # NOT the sum: that is 11.285 GiB and it buys a smaller tile than the
    # tiling that actually ran.
    stacked = machine.vram_budget_bytes - full.radiation_transient_bytes
    assert budget > stacked
    # and NOT ignored: that is the 14.025 GiB budget which selects the
    # tiling the measured run's own analysis says would have been fatal.
    assert budget < machine.vram_budget_bytes


def test_a_rung_without_radiation_is_left_exactly_where_it_was():
    """CONTROL: the reservation must be invisible where nothing was measured.

    Every dry and moist plan in this project predates the constant, so if
    the reservation leaked into them their tile sizes would move and the
    measured capacity table above would stop describing the planner.
    """
    machine = _machine(12, 64)
    for name in ("dry", "moist"):
        fp = A.FOOTPRINTS[name]
        assert fp.radiation_transient_bytes == 0, name
        assert A.budget_for(machine, fp) == machine.vram_budget_bytes, name


def test_a_card_big_enough_makes_the_percentage_the_binding_reservation():
    """CONTROL: the ``max`` really is a max, in both directions.

    Above ~34 GiB the 8% headroom is the larger of the two and the measured
    transient stops binding.  Without this, ``budget_for`` could be
    subtracting the transient unconditionally and every test above would
    still pass.
    """
    big = A.Machine(vram_bytes=80 * GIB, host_bytes=256 * GIB)
    fp = A.FOOTPRINTS["full"]
    assert 0.08 * big.vram_bytes > fp.radiation_transient_bytes
    assert A.budget_for(big, fp) == big.vram_budget_bytes


def test_buffers_never_exceed_the_tile_count():
    cfg = A._config_for_rung(1024, 1024, 49, "full+mynn+noahmp")
    p = A.plan(cfg, M5090)
    assert p.nbuffers <= p.ntiles, p.explain()


# --------------------------------------------------------------------------
# capacity
# --------------------------------------------------------------------------

def test_largest_runnable_domain_is_a_real_boundary():
    """CONTROL: the returned size must plan and the next one up must not."""
    for machine in (M5090, M4090, M5070):
        for rung in ("dry", "full+mynn+noahmp"):
            n, p = A.largest_runnable_domain(machine, rung=rung)
            assert p is not None and n > 0
            try:
                A.plan(A._config_for_rung(n + 1, n + 1, 49, rung), machine,
                       rung=rung)
            except A.CannotPlan:
                pass
            else:
                raise AssertionError(
                    f"{machine.name}/{rung}: {n}^2 was called the largest and "
                    f"{n + 1}^2 planned fine")


def test_a_bigger_box_runs_a_bigger_domain():
    small, _ = A.largest_runnable_domain(_machine(12, 64), rung="dry")
    big, _ = A.largest_runnable_domain(_machine(32, 256), rung="dry")
    assert big > small, (small, big)


# --------------------------------------------------------------------------
# the model itself, against the card
# --------------------------------------------------------------------------

def test_vram_model_against_a_real_allocation(rung: str = "dry",
                                              n: int = 640) -> None:
    """THE control that could actually falsify this module.

    Everything else here checks that the planner is self-consistent.  This
    builds the buffers the plan asks for and compares what the driver reports
    against what the model predicted.  It must be an UPPER bound (a planner
    that under-predicts hands out a tile that OOMs) and it must not be a
    vacuous one -- a model that returned the whole card would pass an
    upper-bound test and is useless, so the prediction is also required to be
    within 35% of the truth.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv

    warm = cp.zeros(1024, dtype=cp.float64)
    warm += 1.0
    cp.cuda.runtime.deviceSynchronize()
    del warm
    cp.get_default_memory_pool().free_all_blocks()

    def used() -> int:
        free, total = cp.cuda.runtime.memGetInfo()
        return total - free

    base = used()
    fp = A.FOOTPRINTS[rung]
    cells = n * n * 49
    cfg = A._config_for_rung(n, n, 49, rung)
    keep = []
    for k in (1, 2):
        state, _drv = physinv.default_builder(cfg, 4242)
        H.run_steps(state, cfg, 1)
        cp.cuda.runtime.deviceSynchronize()
        keep.append(state)
        measured = used() - base + A.CUDA_CONTEXT_BYTES
        predicted = fp.vram_bytes(cells, k)
        ratio = predicted / measured
        print(f"     {rung} {n}^2 x49, {k} buffer(s): "
              f"measured {measured / GIB:.3f} GiB, "
              f"model {predicted / GIB:.3f} GiB, {ratio:.3f}x")
        assert ratio >= 1.0, (
            f"the model UNDER-predicted at {k} buffers ({ratio:.3f}x); a "
            "plan built on it would hand out a tile that does not fit")
        assert ratio <= 1.35, (
            f"the model over-predicted by {ratio:.3f}x at {k} buffers, which "
            "would refuse domains that run")
    del keep
    cp.get_default_memory_pool().free_all_blocks()


def test_planned_run_is_bit_exact(n: int = 192, machine=None) -> None:
    """A plan, spent on ``run_tiled``, must reproduce a monolithic run.

    Small and dry on purpose: the bit-exact gate owns the physics matrix, and
    what is under test here is only that the PLANNER emits a parameter set
    that is correct -- in particular a halo that is wide enough, which is the
    one mistake in this module that would be silent.  The card is faked down
    to a size that forces tiling, because a real 5090 would run 192^2
    resident and prove nothing.
    """
    import cupy as cp

    from tilestream import driver, gather

    cfg = A._config_for_rung(n, n, 49, "dry")
    fp = A.FOOTPRINTS["dry"]
    # A budget that admits a 96-cell interior tile and not the whole domain.
    window = (96 + 2 * H.halo_radius(cfg)) ** 2 * 49
    budget = fp.vram_bytes(window, 2)
    machine = machine or A.Machine(int(budget / (1 - A.VRAM_HEADROOM)),
                                   64 * GIB, name="faked-small")
    p = A.plan(cfg, machine, write_mode="shadow")
    assert p.mode == "tiled", p.explain()
    print(f"     plan: tile {p.tile_nx}x{p.tile_ny}, halo {p.halo}, "
          f"{p.nbuffers} buffers, {p.ntiles} tiles")

    state = H.make_state(cfg)
    H.run_steps(state, cfg, 3)
    reference = H.hash_state(state)

    fresh = H.make_state(cfg)
    store = {k: gather.pinned_copy(cp.asnumpy(v))
             for k, v in H.state_arrays(fresh).items()}
    driver.run_tiled(store, cfg, nsteps=3, **p.run_kwargs)
    for name, array in H.state_arrays(fresh).items():
        array[...] = cp.asarray(store[name])
    assert H.hash_state(fresh) == reference, (
        "a run using the planner's own parameters did not reproduce the "
        "monolithic answer")
    print(f"     digest {reference[:16]} matches")


GPU_TESTS = ("test_vram_model_against_a_real_allocation",
             "test_planned_run_is_bit_exact")


# --------------------------------------------------------------------------
# the controls, controlled
# --------------------------------------------------------------------------

def _hardcode_halo():
    """The most dangerous single mistake available in this module."""
    original = A._harness.halo_radius
    A._harness.halo_radius = lambda cfg: 16
    return lambda: setattr(A._harness, "halo_radius", original)


def _ignore_the_vram_budget():
    original = A.Machine.vram_budget_bytes
    A.Machine.vram_budget_bytes = property(lambda self: 1 << 60)
    return lambda: setattr(A.Machine, "vram_budget_bytes", original)


def _fixed_cost_per_buffer():
    """The hypothesis this module was written to test: 2.5 GiB per BUFFER.

    Under it a second buffer costs 2.5 GiB of fixed on top of its cells, and
    a 12 GiB card at any physics rung can no longer afford one.
    """
    saved = dict(A.FOOTPRINTS)
    A.FOOTPRINTS.update({
        name: A.Footprint(fp.rung, 0, fp.process_fixed_bytes
                          + fp.buffer_fixed_bytes, fp.bytes_per_cell,
                          fp.store_bytes_per_cell)
        for name, fp in saved.items()})
    return lambda: (A.FOOTPRINTS.clear(), A.FOOTPRINTS.update(saved))


def _fixed_cost_per_domain():
    """The defect this lane closed: the process overhead charged per GRID.

    ``marginal_bytes`` answering the WHOLE price is exactly what the tree
    walk did before -- every domain paying for its own CUDA context and its
    own copy of the k-distribution tables.
    """
    original = A.Footprint.marginal_bytes
    A.Footprint.marginal_bytes = lambda self, w, n: self.vram_bytes(w, n)
    return lambda: setattr(A.Footprint, "marginal_bytes", original)


def _stack_the_transient_on_the_headroom():
    """The tempting wrong arm: reserve the percentage AND the measurement.

    It looks safer and it is not: on the card this was measured on it plans
    a 175x250 tiling where a 175x375 one ran, so it makes every radiation
    forecast slower to protect against bytes already withheld.
    """
    original = A.budget_for
    A.budget_for = lambda machine, fp: max(
        0, machine.vram_budget_bytes - fp.radiation_transient_bytes)
    return lambda: setattr(A, "budget_for", original)


def _ignore_the_radiation_transient():
    """The other wrong arm: price the radiation call's transient at zero.

    The state before this constant existed.  It hands the freed budget to
    the tile search, which spends it on a tile whose steady footprint plus
    the transient is past the card -- dead at the first radiation call.
    """
    saved = dict(A.RADIATION_TRANSIENT_BYTES)
    A.RADIATION_TRANSIENT_BYTES.update({k: 0 for k in saved})
    return lambda: (A.RADIATION_TRANSIENT_BYTES.clear(),
                    A.RADIATION_TRANSIENT_BYTES.update(saved))


def _drop_the_ragged_overhang():
    original = A.ring_arena_fraction

    def naive(nx, ny, tile_nx, tile_ny, halo):
        fx, fy = halo / tile_nx, halo / tile_ny
        return 1.0 - (1.0 - fx) * (1.0 - fy)

    A.ring_arena_fraction = naive
    return lambda: setattr(A, "ring_arena_fraction", original)


def _under_predict_vram():
    saved = dict(A.FOOTPRINTS)
    A.FOOTPRINTS.update({
        name: A.Footprint(fp.rung, fp.process_fixed_bytes // 2,
                          fp.buffer_fixed_bytes // 2,
                          fp.bytes_per_cell * 0.5, fp.store_bytes_per_cell)
        for name, fp in saved.items()})
    return lambda: (A.FOOTPRINTS.clear(), A.FOOTPRINTS.update(saved))


def _never_refuse():
    original = A.CannotPlan.__init__

    def swallow(self, message, resource, detail=None):
        raise AssertionError("this planner does not refuse")

    A.CannotPlan.__init__ = swallow
    return lambda: setattr(A.CannotPlan, "__init__", original)


#: ``(what is broken, how to break it, which tests must then FAIL)``.  A gate
#: that has never failed is not a gate, so this breaks each capability in turn
#: and requires its control to notice.  Anything here that keeps passing while
#: broken is a test that was proving nothing.
BREAKAGES = (
    ("the halo is hardcoded to 16", _hardcode_halo,
     ("test_halo_comes_from_halo_radius_and_moves_with_sound_steps",)),
    ("the VRAM budget is ignored", _ignore_the_vram_budget,
     ("test_the_tile_shrinks_when_the_card_does",
      "test_plan_fits_its_own_budget")),
    ("the fixed cost is charged per BUFFER, as was assumed",
     _fixed_cost_per_buffer,
     ("test_the_fixed_cost_is_not_charged_per_buffer",)),
    ("the fixed cost is charged per DOMAIN, as the tree walk did",
     _fixed_cost_per_domain,
     ("test_the_fixed_cost_is_not_charged_per_domain",
      "test_every_rung_pays_its_overhead_once_and_only_once")),
    # NOT ``test_a_rung_without_radiation_is_left_exactly_where_it_was``:
    # the stacked form subtracts zero at a rung with no transient, so that
    # test cannot see this breakage and listing it would be a control that
    # never fires.  It is the control for the reservation LEAKING, which is
    # a different mistake.
    ("the radiation transient is stacked on the percentage headroom",
     _stack_the_transient_on_the_headroom,
     ("test_the_radiation_transient_is_reserved_and_replaces_the_headroom",
      "test_a_card_big_enough_makes_the_percentage_the_binding_reservation")),
    ("the radiation transient is not priced at all",
     _ignore_the_radiation_transient,
     ("test_the_radiation_transient_is_reserved_and_replaces_the_headroom",)),
    ("the ring model drops the ragged overhang", _drop_the_ragged_overhang,
     ("test_ring_arena_fraction_matches_every_measured_plan",
      "test_ring_model_needs_the_ragged_overhang_term")),
    ("the VRAM model under-predicts by 2x", _under_predict_vram,
     ("test_vram_model_against_a_real_allocation",)),
    ("nothing is ever refused", _never_refuse,
     ("test_cannot_run_names_the_binding_resource",
      "test_host_ram_is_the_binding_resource_when_it_is")),
)


def run_controls(gpu: bool = False) -> int:
    """Break each capability and require its control to catch it."""
    bad = 0
    for label, breaker, expect in BREAKAGES:
        names = [n for n in expect if gpu or n not in GPU_TESTS]
        if not names:
            print(f"skip {label}  (its control needs a GPU)")
            continue
        restore = breaker()
        try:
            for name in names:
                try:
                    globals()[name]()
                except Exception:
                    print(f"ok   {label}  -> {name} caught it")
                else:
                    bad += 1
                    print(f"FAIL {label}  -> {name} STILL PASSED; that test "
                          f"is proving nothing")
        finally:
            restore()
    print(f"\n{len(BREAKAGES)} breakages, {bad} went unnoticed")
    return 1 if bad else 0


def _run_all(gpu: bool = False) -> int:
    names = [n for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)
             and (gpu or n not in GPU_TESTS)]
    failed = 0
    for name in names:
        try:
            globals()[name]()
        except Exception as exc:                          # pragma: no cover
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(names) - failed} passed, {failed} failed"
          + ("" if gpu else f"  ({len(GPU_TESTS)} GPU tests skipped; --gpu)"))
    return 1 if failed else 0


if __name__ == "__main__":
    _gpu = "--gpu" in sys.argv
    _rc = _run_all(_gpu)
    if "--no-controls" not in sys.argv:
        print("\n--- controls: each capability broken on purpose ---")
        _rc |= run_controls(_gpu)
    raise SystemExit(_rc)
