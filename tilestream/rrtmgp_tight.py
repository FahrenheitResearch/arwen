"""Tighten the RRTMGP workspace: the RTE phases carry cubes they never read.

WHAT THE ATTRIBUTION POINTS AT
------------------------------
``SharedRRTMGPChunkWorkspace`` allocates the MAXIMUM over its four solver
phases, and at ``nz = 49, p_top = 10 kPa, chunk = 3125`` that maximum is
``lw_rte`` at 978.2 MiB against ``lw_optics`` 569.0, ``sw_optics`` 738.5 and
``sw_rte`` 712.9.  One phase sets the whole allocation, so one phase is where
a reclamation has to land.

``lw_rte``'s declared live set is 978.2 MiB because it keeps every
``lw_optics`` slot at its original offset and then APPENDS its own outputs
after them.  But three of those carried slots are not read by anything in the
RTE phase:

  ==============  ========  =====================================
  carried slot       MiB     last reader
  ==============  ========  =====================================
  ``gas_tau``       225.8    ``_finalize_cloud_optics`` (optics)
  ``cld_tau``        14.1    ``_finalize_cloud_optics`` (optics)
  ``cld_ssa``        14.1    ``_finalize_cloud_optics`` (optics)
  ``cld_asy``        14.1    ``_finalize_cloud_optics`` (optics)
  ``col_dry``         0.9    ``_gas_optics`` / Planck takes ``vmr``
  ==============  ========  =====================================

``_lw_rte`` is handed ``optics_lw.tau``, the Planck sources and the
emissivity; ``_planck_sources`` is handed ``play/plev/tlay/tlev/tsfc`` and
``vmr``.  Neither ever sees ``gas_tau`` or the band cloud optics.  The
shipped layout is not wrong -- it is a covering superset, and a superset is
the safe direction -- but it holds a quarter of the workspace for slots that
are dead.

WHAT THIS MODULE DOES, AND WHY REORDERING IS THE WHOLE TRICK
------------------------------------------------------------
``phase()`` assigns offsets by walking its layout in order, so a carried slot
moves the instant anything before it changes size.  That is why the shipped
layout cannot simply drop the dead slots: ``optics_tau`` has to stay at the
byte offset ``lw_optics`` wrote it to.

Dropping them in place and padding the holes recovers only 180.3 MiB,
because ``lev_source`` (228.9 MiB) does not fit the hole ``gas_tau``
(225.8 MiB) leaves -- three megabytes short -- and the padding then wastes
most of what the other holes freed.  So this module first REORDERS each
optics phase to put the carried slots at the FRONT.  Reordering inside a
phase is free (every slot of an optics phase is written before it is read in
that same phase, so where it sits is invisible), and it turns the scattered
holes into one contiguous tail.  A tail needs no padding at all: the RTE
phase lays its outputs over it and simply stops earlier.

That is the difference between 180.3 MiB and 239.7 MiB, and it is why there
is not a single ``_pad`` entry in the result.

WHAT IT IS WORTH -- IT DEPENDS ON THE MODEL TOP, SO BOTH ARE STATED
--------------------------------------------------------------------
The LW column is ``nz`` plus a synthetic above-model cap whose depth is a
function of ``p_top``, and the saving lives in the LW phases:

  ==================================  ==========  =========  =========
  configuration                        shipped      tight      saved
  ==================================  ==========  =========  =========
  p_top 10 kPa (real74, LW 74 layers)  978.2 MiB  738.5 MiB  239.7 MiB
  p_top 5.7 kPa (WK82 20 km, LW 63)    834.6 MiB  738.5 MiB   96.1 MiB
  ==================================  ==========  =========  =========

Both land on 738.5 MiB because after the tightening the binding phase is no
longer ``lw_rte`` but ``sw_optics``, whose five g-point cubes really are
simultaneously live inside ``_finalize_cloud_optics``.  738.5 MiB at chunk
3125 is therefore the floor for this reclamation; going below it means
changing what the solver holds, not where it holds it.

WHY THE POISON CONTROL IS THE POINT
-----------------------------------
"Nothing reads gas_tau during lw_rte" is a lifetime claim, and this project
has produced seven false results from claims that looked at least this
obvious.  So the claim is not argued, it is EXECUTED: :func:`build` with
``poison="dead"`` NaN-fills the entire region this module calls dead, at the
exact instant the RTE phase begins, and the gate requires the answer to stay
bit-exact.  If any of it is really read, NaN propagates into the fluxes and
the digest moves by a mile.

And the control has its own control: ``poison="live"`` fills the carried
prefix instead, which the RTE phase demonstrably does read.  That one must
NOT survive -- and it does not: it dies in
``PhysicsDriver._run_radiation``'s finite check with "radiation rthratenlw
contains a non-finite value".  A poison that cannot break the run proves
nothing by failing to break it.
"""

from __future__ import annotations

import math

MIB = 1 << 20

#: Slots each RTE phase inherits from its optics phase but never reads.
#: Keyed by phase, and stated as the exhaustive complement of what the phase
#: DOES read, so a new consumer added to the RTE path fails the poison gate
#: rather than silently reading poison.
DEAD_IN_RTE = {
    "lw_rte": ("gas_tau", "cld_tau", "cld_ssa", "cld_asy", "col_dry"),
    "sw_rte": ("gas_tau", "gas_ssa", "vmr", "cld_tau", "cld_ssa", "cld_asy",
               "col_dry"),
}

#: Slots each RTE phase genuinely reads back from its optics phase.  Used by
#: the control-on-the-control: poisoning one of these MUST change the answer.
LIVE_IN_RTE = {
    "lw_rte": ("optics_tau", "vmr"),
    "sw_rte": ("optics_tau", "optics_ssa", "optics_g"),
}


def _phase_bytes(items) -> int:
    return sum(math.prod(shape) * size for shape, size in items.values())


def tight_phases(nz: int, column_chunk: int, p_top: float = 10000.0):
    """The shipped phase layouts with the RTE outputs packed into the holes.

    Returns ``(layouts, report)``.  ``layouts`` is in exactly the form
    ``SharedRRTMGPChunkWorkspace(_phase_layouts_input=...)`` accepts;
    ``report`` records, per phase, the shipped and tightened byte totals and
    which slot each RTE output was placed into, so the result is auditable
    without re-deriving it.
    """
    from gpuwm.core.preflight import rrtmgp_workspace_phases

    shipped = rrtmgp_workspace_phases(nz, column_chunk, p_top)
    layouts: dict[str, dict] = {}
    report: dict[str, dict] = {}

    for rte, optics in (("lw_rte", "lw_optics"), ("sw_rte", "sw_optics")):
        opt_items = shipped[optics]
        rte_items = shipped[rte]
        live = [name for name in LIVE_IN_RTE[rte] if name in opt_items]
        missing = set(LIVE_IN_RTE[rte]) - set(opt_items)
        if missing:
            raise RuntimeError(
                f"{rte} claims to carry {sorted(missing)}, which {optics} "
                "does not declare; the phase layouts have changed shape and "
                "this reclamation's lifetime claim no longer applies")

        # STEP 1 -- reorder the OPTICS phase so every slot the RTE phase
        # carries sits at the FRONT.  Reordering within a phase is free:
        # every slot of an optics phase is written before it is read inside
        # that same phase, so where it sits is invisible.  What it buys is
        # that all the dead storage then forms ONE contiguous tail, and a
        # tail needs no padding -- the RTE phase simply stops earlier.  The
        # shipped layout scatters the dead slots between the live ones, so
        # every hole has to be padded to keep the slot after it in place,
        # and the padding is most of what a naive tightening fails to save.
        rest = [name for name in opt_items if name not in live]
        # 1-byte slots last, so no 4-byte slot can start on an odd offset.
        rest.sort(key=lambda name: opt_items[name][1] == 1)
        opt_tight = {name: opt_items[name] for name in [*live, *rest]}
        layouts[optics] = opt_tight

        # STEP 2 -- the RTE phase is the carried prefix, then its own
        # outputs packed straight into the tail the dead slots vacated.
        own = {name: spec for name, spec in rte_items.items()
               if name not in opt_items}
        tight = {name: opt_items[name] for name in live}
        placement = {}
        carried_bytes = sum(math.prod(opt_items[n][0]) * opt_items[n][1]
                            for n in live)
        tail = _phase_bytes(opt_items) - carried_bytes
        used = 0
        for name, spec in sorted(
                own.items(),
                key=lambda kv: (-math.prod(kv[1][0]) * kv[1][1], kv[0])):
            size = math.prod(spec[0]) * spec[1]
            tight[name] = spec
            placement[name] = ("reuses dead tail" if used + size <= tail
                               else "past the tail")
            used += size

        layouts[rte] = tight
        report[rte] = {
            "shipped_bytes": _phase_bytes(rte_items),
            "tight_bytes": _phase_bytes(tight),
            "placement": placement,
            "carried": live,
            "dead": sorted(set(DEAD_IN_RTE[rte]) & set(opt_items)),
            "dead_tail_bytes": tail,
            "own_output_bytes": used,
        }

    for phase in ("lw_optics", "sw_optics"):
        report[phase] = {"shipped_bytes": _phase_bytes(shipped[phase]),
                         "tight_bytes": _phase_bytes(layouts[phase]),
                         "placement": {}, "dead": []}
    report["_total"] = {
        "shipped_bytes": max(_phase_bytes(v) for v in shipped.values()),
        "tight_bytes": max(_phase_bytes(v) for v in layouts.values()),
    }
    return layouts, report


def describe(nz: int, column_chunk: int, p_top: float = 10000.0) -> str:
    layouts, report = tight_phases(nz, column_chunk, p_top)
    lines = [f"RRTMGP workspace phases   nz={nz}  chunk={column_chunk}  "
             f"p_top={p_top:.0f}",
             "",
             f"  {'phase':<12}{'shipped MiB':>14}{'tight MiB':>12}"
             f"{'saved':>10}   dead slots reused"]
    lines.append("  " + "-" * 76)
    for phase in ("lw_optics", "lw_rte", "sw_optics", "sw_rte"):
        r = report[phase]
        lines.append(
            f"  {phase:<12}{r['shipped_bytes'] / MIB:>14.2f}"
            f"{r['tight_bytes'] / MIB:>12.2f}"
            f"{(r['shipped_bytes'] - r['tight_bytes']) / MIB:>10.2f}"
            f"   {', '.join(r['dead']) or '-'}")
    t = report["_total"]
    lines.append("  " + "-" * 76)
    lines.append(
        f"  {'ALLOCATION':<12}{t['shipped_bytes'] / MIB:>14.2f}"
        f"{t['tight_bytes'] / MIB:>12.2f}"
        f"{(t['shipped_bytes'] - t['tight_bytes']) / MIB:>10.2f}"
        f"   ({100.0 * (t['shipped_bytes'] - t['tight_bytes'])
              / t['shipped_bytes']:.1f}% of the workspace)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# building one, and poisoning it
# --------------------------------------------------------------------------

#: Byte fill used to poison a region.  0xFF in every byte of a float32 is a
#: quiet NaN, so any read of a poisoned slot propagates NaN into the fluxes
#: and moves the digest by a mile rather than by an ulp.  A poison that could
#: be mistaken for rounding would be no poison at all.
POISON_BYTE = 0xFF

POISON_TARGETS = ("dead", "live", None)


def build(nz: int, column_chunk: int, p_top: float, *, tight: bool = True,
          lazy: bool = False, poison: str | None = None):
    """Build a chunk workspace: shipped or tightened, eager or lazy.

    ``poison`` is the control lever and takes ``"dead"`` or ``"live"``.
    ``"dead"`` NaN-fills the region this module claims no RTE phase reads,
    at the instant the RTE phase begins -- the run must stay bit-exact.
    ``"live"`` NaN-fills the carried prefix instead, which the RTE phase
    demonstrably does read -- the run must NOT stay bit-exact.  The second is
    the control on the first: a poison that cannot break the answer proves
    nothing by failing to break it.
    """
    if poison not in POISON_TARGETS:
        raise ValueError(f"poison must be one of {POISON_TARGETS}")
    layouts = None
    if tight:
        layouts, _ = tight_phases(nz, column_chunk, p_top)
    base = _lazy_class() if lazy else _shared_class()
    kwargs = dict(nz=nz, column_chunk=column_chunk, p_top=p_top)
    if layouts is not None:
        kwargs["_phase_layouts_input"] = layouts
    if poison is None:
        return base(**kwargs)
    return _poisoned(base)(poison=poison, **kwargs)


def _shared_class():
    from gpuwm.core.model import SharedRRTMGPChunkWorkspace

    return SharedRRTMGPChunkWorkspace


def _lazy_class():
    from tilestream.rrtmgp_lazy import LazyRRTMGPChunkWorkspace

    return LazyRRTMGPChunkWorkspace


def carried_prefix_bytes(layouts, rte: str) -> int:
    """Bytes of the RTE phase's carried prefix under ``layouts``.

    Everything at or after this offset is the dead tail the RTE phase reuses;
    everything before it is a value the optics phase produced and the RTE
    phase reads back.  The split is read off the layout rather than
    hardcoded, so a layout change moves the poison with it.
    """
    optics = "lw_optics" if rte == "lw_rte" else "sw_optics"
    live = [n for n in LIVE_IN_RTE[rte] if n in layouts[optics]]
    total = 0
    for name in layouts[optics]:
        if name not in live:
            break
        shape, itemsize = layouts[optics][name]
        total += math.prod(shape) * itemsize
    return total


def _poisoned(base):
    """Return a ``base`` subclass that NaN-fills a region at the RTE seam."""

    class _Poisoned(base):                                 # type: ignore[misc]
        def __init__(self, *args, poison: str = "dead", **kwargs):
            self._poison = poison
            self._poison_firings = 0
            super().__init__(*args, **kwargs)

        @property
        def poison_firings(self) -> int:
            """How many times the poison actually ran.

            Reported, not assumed: a control that never executed is not a
            control that passed, and three of this project's false results
            came from measurement windows where the thing being measured
            never happened.
            """
            return int(self._poison_firings)

        def phase(self, name, ncol):
            views = super().phase(name, ncol)
            if name in ("lw_rte", "sw_rte"):
                cut = carried_prefix_bytes(dict(self._phase_layouts), name)
                storage = self._storage
                if self._poison == "dead":
                    storage[cut:] = POISON_BYTE
                else:
                    storage[:cut] = POISON_BYTE
                self._poison_firings += 1
            return views

    _Poisoned.__name__ = f"Poisoned{base.__name__}"
    return _Poisoned
