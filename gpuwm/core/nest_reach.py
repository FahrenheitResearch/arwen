"""How far a moving nest can get from where it starts: its REACH.

A moving nest moves in whole cells of its parent, at cadence
opportunities, by a bounded shift per move.  Its settings therefore bound
how far it can get from its declared placement by the end of the run, and
so do the rows of an explicit ``[[relocation.move]]`` itinerary when those
drive it.  The reach is that bound: per grid axis, in whole cells of the
mover's parent, the lowest and highest displacement the run can produce.

WHAT IT SIZES.  The statics corridor (:mod:`gpuwm.static.corridor`) holds
child-resolution statics for every footprint a relocation can crop.  Sized
to the whole parent it made a storm-following 500 m tier need a 64 to
128 GiB host: a 6 h run over a 2,700 x 3,000 km parent sealed a 25 GB
corridor for the 500 m nest and peaked at 62 GB while preparing.  Sized to
the reach it covers the ground the nest can get to and nothing else.

WHY A SPEED BOUND.  A tracker's settings bound its reach, but loosely: a
nest allowed 8 parent cells every 30 minutes can cross 17,000 km of a
9 km parent in five days, so over a long run the settings alone reach the
whole parent.  ``reach_speed_m_s`` is the named bound for that case: at
model time ``t`` the nest may be at most ``reach_speed_m_s * t`` plus one
move from its declared placement along each grid axis.  The runner
enforces it the way it enforces ``max_move_parent_cells``: a move that
would pass it is clamped, and the receipt names ``reach_speed_m_s`` in
``clamped_by``.  Because it grows with ``t`` rather than being fixed by
the run length, a run extended on restart moves exactly as the longer run
would have.

THE DEFAULT, :data:`DEFAULT_REACH_SPEED_M_S`, is 40 m/s: WRF's own default
for the same quantity (``max_vortex_speed``, the fastest vortex its
vortex-following nest searches for), and above the forward speed of any
tropical cyclone or supercell on record (about 30 m/s at the extreme).
It is the smallest bound that clamps no real storm; a configuration that
follows something faster names a larger value.

Nothing here reads a field or a GPU array: it is arithmetic on the
validated experiment, shared by the corridor emission, the run-plan
estimate, the terrain survey, the corridor loader and the runner's clamp,
so all five agree on one number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: The named setting, spelled once: ``[relocation]`` and a per-domain
#: ``follow`` table both accept it.
REACH_SPEED_KEY = "reach_speed_m_s"

#: The bound a follower gets when it names none.  See the module
#: docstring for why 40 m/s.
DEFAULT_REACH_SPEED_M_S = 40.0

#: Float slack for "is this quotient a whole number": cadences and
#: cooldowns are decimal seconds, and 1800 / 900 must count as exactly 2.
_WHOLE_TOL = 1.0e-9


def validate_reach_speed(value, label: str) -> float | None:
    """``None`` (take the default) or a finite, positive speed in m/s."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{label} {REACH_SPEED_KEY} = {value!r} must be a number of "
            "metres per second")
    speed = float(value)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError(
            f"{label} {REACH_SPEED_KEY} = {value!r} must be a finite, "
            "positive speed in m/s: it is how fast, on average since the "
            "start, the nest may travel, and the statics corridor is sized "
            "to it")
    return speed


def speed_cells(speed_m_s: float, seconds: float, parent_dx_m: float) -> int:
    """Whole parent cells a speed covers in ``seconds``, rounded up."""
    if seconds <= 0.0:
        return 0
    return int(math.ceil(float(speed_m_s) * float(seconds)
                         / float(parent_dx_m) - _WHOLE_TOL))


def cadence_opportunities(run_seconds: float, cadence_seconds: float) -> int:
    """Cadence boundaries strictly inside the run: ``k * cadence < T``.

    ``t = run_seconds`` is never an opportunity (the runner fires at the
    START of each period), and ``t = 0`` is initial placement, not a move.
    """
    quotient = float(run_seconds) / float(cadence_seconds)
    return max(0, int(math.ceil(quotient - _WHOLE_TOL)) - 1)


def most_moves(opportunities: int, cadence_seconds: float,
               cooldown_seconds: float) -> int:
    """The largest number of moves a tracker can execute.

    After an executed move the tracker holds for ``cooldown_seconds``, so
    two moves are at least that far apart, on the cadence lattice.
    """
    if opportunities <= 0:
        return 0
    spacing = 1
    if cooldown_seconds > 0.0:
        spacing = max(1, int(math.ceil(float(cooldown_seconds)
                                       / float(cadence_seconds)
                                       - _WHOLE_TOL)))
    return 1 + (int(opportunities) - 1) // spacing


@dataclass(frozen=True)
class MoverReach:
    """One mover's displacement range over the whole run.

    ``lo_i``/``hi_i``/``lo_j``/``hi_j`` are whole cells of the mover's
    parent (``lo <= 0 <= hi``), ``None`` when nothing bounds them.
    ``ground`` marks a mover whose displacement is counted over the
    ground rather than inside its parent: the tracked mover under
    ``[relocation.containment]``, which its sliding parent carries
    earth-fixed.  ``basis`` says what bounded it, with the numbers.
    """

    grid_id: int
    lo_i: int | None
    hi_i: int | None
    lo_j: int | None
    hi_j: int | None
    ground: bool = False
    basis: dict = field(default_factory=dict)

    @property
    def bounded(self) -> bool:
        return None not in (self.lo_i, self.hi_i, self.lo_j, self.hi_j)


def _by_id(exp) -> dict:
    return {int(dc.grid_id): dc for dc in exp.domains}


def _root_period_seconds(exp) -> float:
    """One complete root step: the runner's opportunity when no cadence is
    configured."""
    root = exp.root
    dt_exact = getattr(exp, "dt_exact", None)
    if callable(dt_exact):
        return float(dt_exact(int(root.grid_id)))
    return float(root.run.dt)


def _adaptive(exp) -> bool:
    return bool(getattr(exp.root.run, "use_adaptive_time_step", False))


def _dormant(dc) -> bool:
    """A nest whose placement is chosen when it fires, not declared."""
    return getattr(dc, "spawn", None) is not None


def follow_source(exp, grid_id: int) -> dict | None:
    """What moves ``grid_id``, read off the configuration.

    A per-domain ``follow`` table wins over ``[relocation]``: the runner
    for a per-domain follower is built from a view whose ``[relocation]``
    restates the same follower, and both readings must give one answer.
    """
    by_id = _by_id(exp)
    dc = by_id.get(int(grid_id))
    if dc is None:
        return None
    follow = getattr(dc, "follow", None)
    if follow is not None:
        return {"kind": "follow", "table": "[[domain]].follow",
                "tracker": follow.tracker,
                "cadence_seconds": float(follow.cadence_seconds),
                "max_move_parent_cells": follow.max_move_parent_cells,
                "speed": getattr(follow, "reach_speed_m_s", None)}
    relocation = getattr(exp, "relocation", None)
    if relocation is None or not getattr(relocation, "enabled", False):
        return None
    if (relocation.grid_id is not None
            and int(relocation.grid_id) == int(grid_id)):
        if relocation.moves:
            return {"kind": "itinerary", "table": "[[relocation.move]]",
                    "moves": tuple(relocation.moves),
                    "max_move_parent_cells": relocation.max_move_parent_cells}
        if relocation.follow is not None:
            return {"kind": "follow", "table": "[relocation]",
                    "tracker": relocation.follow,
                    "cadence_seconds": relocation.cadence_seconds,
                    "max_move_parent_cells": relocation.max_move_parent_cells,
                    "speed": getattr(relocation, "reach_speed_m_s", None)}
        return None
    containment = getattr(relocation, "containment", None)
    if (containment is not None
            and int(containment.grid_id) == int(grid_id)
            and (relocation.follow is not None or relocation.moves)):
        return {"kind": "containment", "table": "[relocation.containment]",
                "containment": containment,
                "cadence_seconds": (containment.cadence_seconds
                                    if containment.cadence_seconds is not None
                                    else relocation.cadence_seconds)}
    return None


def _contained_mover(exp) -> int | None:
    """The tracked mover a ``[relocation.containment]`` ancestor carries."""
    relocation = getattr(exp, "relocation", None)
    if (relocation is None or not getattr(relocation, "enabled", False)
            or getattr(relocation, "containment", None) is None
            or relocation.grid_id is None):
        return None
    return int(relocation.grid_id)


def _clip_to_parent(exp, dc, lo: int, hi: int, axis: str
                    ) -> tuple[int, int]:
    """A placement never leaves its parent, so neither does the range."""
    parent = _by_id(exp)[int(dc.parent_id)]
    start = int(dc.i_parent_start if axis == "i" else dc.j_parent_start)
    extent = int(parent.run.nx if axis == "i" else parent.run.ny)
    return max(lo, 1 - start), min(hi, extent - start)


def mover_reach(exp, grid_id: int) -> MoverReach | None:
    """The reach of one mover, or ``None`` when ``grid_id`` never moves."""
    source = follow_source(exp, grid_id)
    if source is None:
        return None
    by_id = _by_id(exp)
    dc = by_id[int(grid_id)]
    gid = int(grid_id)
    ground = _contained_mover(exp) == gid
    run_seconds = float(exp.run_seconds)
    parent_dx = float(by_id[int(dc.parent_id)].run.dx)
    if _dormant(dc):
        return MoverReach(gid, None, None, None, None, ground, {
            "bounded_by": None,
            "reason": (f"d{gid:02d} is dormant: its placement is chosen "
                       "when it fires, so there is no start to measure "
                       "a reach from")})
    kind = source["kind"]
    if kind == "itinerary":
        limit = source["max_move_parent_cells"]
        cap = (lambda d: abs(int(d)) if limit is None
               else min(abs(int(d)), int(limit)))
        rows = [move for move in source["moves"]
                if float(move.at_seconds) < run_seconds]
        lo_i = -sum(cap(m.di_parent_cells) for m in rows
                    if int(m.di_parent_cells) < 0)
        hi_i = sum(cap(m.di_parent_cells) for m in rows
                   if int(m.di_parent_cells) > 0)
        lo_j = -sum(cap(m.dj_parent_cells) for m in rows
                    if int(m.dj_parent_cells) < 0)
        hi_j = sum(cap(m.dj_parent_cells) for m in rows
                   if int(m.dj_parent_cells) > 0)
        basis = {"bounded_by": "itinerary", "table": source["table"],
                 "moves": len(rows),
                 "max_move_parent_cells": limit}
    elif kind == "containment":
        containment = source["containment"]
        per = containment.max_move_parent_cells
        cadence = source["cadence_seconds"]
        cadence = (_root_period_seconds(exp) if cadence is None
                   else float(cadence))
        if per is None or _adaptive(exp):
            return MoverReach(gid, None, None, None, None, False, {
                "bounded_by": None, "table": source["table"],
                "reason": ("containment.max_move_parent_cells is unset, so "
                           "one slide is unbounded" if per is None else
                           "the root steps adaptively, so the slide "
                           "opportunities are not a fixed count")})
        slides = cadence_opportunities(run_seconds, cadence)
        reach = (slides + 1) * int(per)
        lo_i, hi_i = _clip_to_parent(exp, dc, -reach, reach, "i")
        lo_j, hi_j = _clip_to_parent(exp, dc, -reach, reach, "j")
        return MoverReach(gid, lo_i, hi_i, lo_j, hi_j, False, {
            "bounded_by": "containment settings", "table": source["table"],
            "cadence_seconds": cadence, "slides": slides,
            "max_move_parent_cells": int(per),
            "reach_parent_cells": reach})
    else:
        tracker = source["tracker"]
        per = int(tracker.max_shift_cells)
        if source["max_move_parent_cells"] is not None:
            per = min(per, int(source["max_move_parent_cells"]))
        cadence = source["cadence_seconds"]
        cadence = (_root_period_seconds(exp) if cadence is None
                   else float(cadence))
        cooldown = float(tracker.cooldown_seconds)
        speed = source["speed"]
        speed_source = "configured" if speed is not None else "default"
        speed = DEFAULT_REACH_SPEED_M_S if speed is None else float(speed)
        by_speed = speed_cells(speed, run_seconds, parent_dx) + per
        if _adaptive(exp):
            moves = None
            by_settings = None
        else:
            opportunities = cadence_opportunities(run_seconds, cadence)
            moves = most_moves(opportunities, cadence, cooldown)
            # One move of margin: a resume that could not restore the
            # tracker's cooldown anchor may move once early.
            by_settings = (moves + 1) * per
        reach = by_speed if by_settings is None else min(by_settings,
                                                        by_speed)
        bounded_by = ("follow settings" if by_settings is not None
                      and by_settings <= by_speed else REACH_SPEED_KEY)
        basis = {"bounded_by": bounded_by, "table": source["table"],
                 "cadence_seconds": cadence,
                 "cooldown_seconds": cooldown,
                 "move_parent_cells": per,
                 "moves": moves,
                 "settings_parent_cells": by_settings,
                 REACH_SPEED_KEY: speed,
                 "speed_source": speed_source,
                 "speed_parent_cells": by_speed,
                 "parent_dx_m": parent_dx,
                 "run_seconds": run_seconds,
                 "reach_parent_cells": reach}
        lo_i, hi_i, lo_j, hi_j = -reach, reach, -reach, reach
    if not ground:
        lo_i, hi_i = _clip_to_parent(exp, dc, lo_i, hi_i, "i")
        lo_j, hi_j = _clip_to_parent(exp, dc, lo_j, hi_j, "j")
    return MoverReach(gid, lo_i, hi_i, lo_j, hi_j, ground, basis)


@dataclass(frozen=True)
class ReachClamp:
    """The runner's side of ``reach_speed_m_s``: one tracked mover.

    ``displacement`` is the mover's own, in whole cells of its parent:
    placement change from the declared one, plus -- for the tracked mover
    under ``[relocation.containment]`` -- the sliding parent's placement
    change in the same cells, because the slide carries the mover
    earth-fixed and the compensation must not count as travel.
    """

    grid_id: int
    speed_m_s: float
    speed_source: str
    parent_dx_m: float
    margin_cells: int
    declared_i: int
    declared_j: int
    ancestor_grid_id: int | None = None
    ancestor_ratio: int = 1
    ancestor_declared_i: int = 0
    ancestor_declared_j: int = 0

    def allowance(self, elapsed_seconds: float) -> int:
        """Parent cells the mover may be from its start at this time."""
        return (speed_cells(self.speed_m_s, max(0.0, float(elapsed_seconds)),
                            self.parent_dx_m) + int(self.margin_cells))

    def displacement(self, node) -> tuple[int, int]:
        di = int(node.cfg.i_parent_start) - int(self.declared_i)
        dj = int(node.cfg.j_parent_start) - int(self.declared_j)
        if self.ancestor_grid_id is not None:
            ancestor = node.parent
            if int(ancestor.cfg.grid_id) != int(self.ancestor_grid_id):
                raise ValueError(
                    f"d{self.grid_id:02d} is contained by "
                    f"d{self.ancestor_grid_id:02d}, but its live parent is "
                    f"d{int(ancestor.cfg.grid_id):02d}")
            ratio = int(self.ancestor_ratio)
            di += (int(ancestor.cfg.i_parent_start)
                   - int(self.ancestor_declared_i)) * ratio
            dj += (int(ancestor.cfg.j_parent_start)
                   - int(self.ancestor_declared_j)) * ratio
        return di, dj

    def bound(self, node, shift, elapsed_seconds: float
              ) -> tuple[int, int, bool]:
        """Clip a proposed shift so the mover stays inside its allowance.

        Toward zero only: a mover already past its allowance (a restart
        under a smaller setting) is held, never dragged back.
        """
        allow = self.allowance(elapsed_seconds)
        cur_i, cur_j = self.displacement(node)
        di, dj = int(shift[0]), int(shift[1])
        out_i = _clip_toward_zero(di, -allow - cur_i, allow - cur_i)
        out_j = _clip_toward_zero(dj, -allow - cur_j, allow - cur_j)
        return out_i, out_j, (out_i, out_j) != (di, dj)

    def receipt(self) -> dict:
        return {REACH_SPEED_KEY: float(self.speed_m_s),
                "speed_source": self.speed_source,
                "parent_dx_m": float(self.parent_dx_m),
                "margin_parent_cells": int(self.margin_cells)}


def _clip_toward_zero(step: int, lo: int, hi: int) -> int:
    lo = min(lo, 0)
    hi = max(hi, 0)
    return max(lo, min(hi, step))


def reach_clamp_for(exp, grid_id: int) -> ReachClamp | None:
    """The speed clamp for a tracked mover, or ``None``.

    ``None`` for an itinerary (its rows ARE the track, and a scripted move
    is never second-guessed), for a containment slide (bounded by its own
    settings), and for a dormant nest (its placement is chosen when it
    fires, so there is no declared start to measure from).
    """
    source = follow_source(exp, grid_id)
    if source is None or source["kind"] != "follow":
        return None
    by_id = _by_id(exp)
    dc = by_id[int(grid_id)]
    if _dormant(dc):
        return None
    tracker = source["tracker"]
    per = int(tracker.max_shift_cells)
    if source["max_move_parent_cells"] is not None:
        per = min(per, int(source["max_move_parent_cells"]))
    speed = source["speed"]
    kwargs = {}
    if _contained_mover(exp) == int(grid_id):
        ancestor = by_id[int(dc.parent_id)]
        kwargs = {"ancestor_grid_id": int(ancestor.grid_id),
                  "ancestor_ratio": int(ancestor.parent_grid_ratio),
                  "ancestor_declared_i": int(ancestor.i_parent_start),
                  "ancestor_declared_j": int(ancestor.j_parent_start)}
    return ReachClamp(
        grid_id=int(grid_id),
        speed_m_s=(DEFAULT_REACH_SPEED_M_S if speed is None
                   else float(speed)),
        speed_source="configured" if speed is not None else "default",
        parent_dx_m=float(by_id[int(dc.parent_id)].run.dx),
        margin_cells=per,
        declared_i=int(dc.i_parent_start),
        declared_j=int(dc.j_parent_start),
        **kwargs)


__all__ = [
    "DEFAULT_REACH_SPEED_M_S", "MoverReach", "REACH_SPEED_KEY",
    "ReachClamp", "cadence_opportunities", "follow_source", "most_moves",
    "mover_reach", "reach_clamp_for", "speed_cells", "validate_reach_speed",
]
