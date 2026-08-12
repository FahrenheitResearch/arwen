"""Gate for the RRTMGP VRAM reclamations, and the controls that police them.

WHAT IS CLAIMED
    1. LAZY.  A workspace freed after one firing and re-allocated for the
       next gives bit-identical answers, because every slot is written before
       it is read.
    2. TIGHT.  The two RTE phases carry cubes nothing in them reads, so those
       bytes can hold the RTE phase's own outputs instead.

Both are LIFETIME claims, and a lifetime claim that is only argued is how
this project produced seven false results.  So each is executed.

THE CONTROL, AND WHY IT IS THE POISON AND NOT THE RACE
------------------------------------------------------
The obvious control is "free the arena mid-call and let another allocation
take the bytes".  It is written here as ``release_between_phases`` and its
miss rate is reported -- but it is the WEAK control, and the reason is
mechanical: the views the optics phase was handed are live CuPy arrays that
hold a reference to the block, so ``MemoryPool.free_all_blocks()`` declines
to release it and the squatter is handed different memory.  The hazard is
largely inert BY CONSTRUCTION, which is a real (and reassuring) property of
the design -- and exactly why a control built on it would be a control that
cannot fail.  Reporting it as a clean checkmark would be the eighth false
result.

The STRONG control writes the bytes directly, which is what a real aliasing
defect does anyway:

  ``poison="dead"``   NaN-fill everything the tightening reuses, at the
                      instant the RTE phase begins.  The answer MUST NOT
                      move.  This is the claim under test.
  ``poison="live"``   NaN-fill the carried prefix instead -- storage the RTE
                      phase demonstrably reads back.  The answer MUST move,
                      or the instrument is blind and the first result means
                      nothing.

Both are deterministic: no allocator, no scheduler, no race.  Both are run
many times anyway, and both firing rates are printed, because "the control
did not fire" and "the control passed" are different sentences.
"""

from __future__ import annotations

import argparse
import json
import sys

from tilestream.rrtmgp_bench import run_trial_subprocess

MIB = 1 << 20

#: 1,920 columns -- three orders of magnitude above the "never quote a
#: window under ~500 cells" floor, and more than one chunk at 256 so the
#: chunk loop really loops.
CTRL_NX, CTRL_NY, CTRL_NZ = 48, 40, 49

#: The finite check the live-poison control is expected to trip.  Matched on
#: text so a crash from some unrelated cause is not counted as a catch.
EXPECTED_CATCH = "contains a non-finite value"


def _trial(**kw):
    kw.setdefault("rung", "full")
    kw.setdefault("nx", CTRL_NX)
    kw.setdefault("ny", CTRL_NY)
    kw.setdefault("nz", CTRL_NZ)
    kw.setdefault("fire", True)
    kw.setdefault("steps", 3)
    kw.setdefault("cycles", 1)
    return run_trial_subprocess(**kw)


def _require(res, label):
    if res.get("error"):
        raise RuntimeError(
            f"{label} failed rc={res['returncode']}\n{res['stderr'][-2000:]}")
    if not res["radiation_firings"]:
        raise RuntimeError(
            f"{label} never fired radiation -- not a radiation measurement")
    return res


# --------------------------------------------------------------------------
# 1. equivalence
# --------------------------------------------------------------------------

def equivalence(chunks=(3125, 512, 256)) -> list[str]:
    """Every mode, every chunk, one digest."""
    lines, reference = [], None
    plan = [("none", None)]
    for chunk in chunks:
        plan += [("persistent", chunk), ("lazy", chunk),
                 ("tight", chunk), ("tight+lazy", chunk)]
    for mode, chunk in plan:
        res = _require(_trial(mode=mode, column_chunk=chunk),
                       f"{mode}/{chunk}")
        tag = (f"{mode:<11} chunk={str(chunk):<5} "
               f"W={res['workspace_bytes'] / MIB:>7.1f} MiB "
               f"fires={res['radiation_firings']}")
        if reference is None:
            reference = res["digest"]
            lines.append(f"reference  {tag}  {reference[:16]}")
            continue
        if res["digest"] != reference:
            raise AssertionError(
                f"{mode} at chunk {chunk} changed the answer: "
                f"{res['digest']} != {reference}")
        lines.append(f"PASS       {tag}  {res['digest'][:16]}")
    return lines


# --------------------------------------------------------------------------
# 2. residency
# --------------------------------------------------------------------------

def residency(chunk: int = 3125) -> list[str]:
    res = _require(_trial(mode="lazy", column_chunk=chunk, cycles=3),
                   "lazy residency")
    lines = []
    held = res["workspace_resident_between_firings"]
    if held != 0:
        raise AssertionError(
            f"lazy workspace still holds {held} bytes between firings")
    lines.append(f"PASS       released to 0 B between firings "
                 f"(W = {res['workspace_bytes'] / MIB:.1f} MiB)")
    if res["workspace_allocations"] < 2 or res["workspace_releases"] < 2:
        raise AssertionError(
            f"lazy workspace did not cycle: {res['workspace_allocations']} "
            f"allocations / {res['workspace_releases']} releases -- a "
            "workspace allocated once and never released is the shipped "
            "behaviour wearing a new name")
    lines.append(f"PASS       cycled {res['workspace_allocations']} "
                 f"allocations / {res['workspace_releases']} releases")
    per = _require(_trial(mode="persistent", column_chunk=chunk),
                   "persistent residency")
    if per["workspace_resident_between_firings"] is not None:
        raise AssertionError(
            "the shipped workspace grew a residency counter; this control "
            "can no longer tell the two objects apart")
    lines.append("PASS       shipped workspace has no release path "
                 "(the two objects are genuinely different)")
    return lines


# --------------------------------------------------------------------------
# 3. the poison controls
# --------------------------------------------------------------------------

def poison_controls(runs: int = 20, chunk: int = 256) -> dict:
    """Run both poisons ``runs`` times each and report both firing rates."""
    ref = _require(_trial(mode="tight", column_chunk=chunk), "tight baseline")
    reference = ref["digest"]

    dead = {"runs": runs, "survived": 0, "fired": 0, "moved": 0, "crashed": 0}
    for _ in range(runs):
        res = _trial(mode="tight", column_chunk=chunk, poison="dead")
        if res.get("error"):
            dead["crashed"] += 1
            continue
        dead["fired"] += int(res["poison_firings"] > 0)
        if res["digest"] == reference:
            dead["survived"] += 1
        else:
            dead["moved"] += 1

    live = {"runs": runs, "caught": 0, "fired": 0, "missed": 0,
            "caught_by_digest": 0, "caught_by_finite_check": 0}
    for _ in range(runs):
        res = _trial(mode="tight", column_chunk=chunk, poison="live")
        if res.get("error"):
            if EXPECTED_CATCH in (res.get("stderr") or ""):
                live["caught"] += 1
                live["caught_by_finite_check"] += 1
                live["fired"] += 1
            else:
                live["missed"] += 1
            continue
        live["fired"] += int(res["poison_firings"] > 0)
        if res["digest"] != reference:
            live["caught"] += 1
            live["caught_by_digest"] += 1
        else:
            live["missed"] += 1

    return {"chunk": chunk, "reference": reference,
            "dead": dead, "live": live,
            "dead_miss_rate": dead["moved"] / runs if runs else None,
            "live_miss_rate": live["missed"] / runs if runs else None}


# --------------------------------------------------------------------------
# 4. the release race, reported for what it is
# --------------------------------------------------------------------------

def release_race(runs: int = 40, chunk: int = 256) -> dict:
    """The weak control: free the backing mid-call and let a squatter in."""
    ref = _require(_trial(mode="lazy", column_chunk=chunk), "lazy baseline")
    reference = ref["digest"]
    misses, fired, crashes, moved = 0, 0, 0, 0
    for _ in range(runs):
        res = _trial(mode="lazy", column_chunk=chunk,
                     hazard="release_between_phases")
        if res.get("error"):
            crashes += 1
            continue
        fired += int(res["hazard_firings"] > 0)
        if res["digest"] == reference:
            misses += 1
        else:
            moved += 1
    return {"runs": runs, "chunk": chunk, "hazard_fired_runs": fired,
            "crashes": crashes, "answer_moved": moved, "misses": misses,
            "miss_rate": misses / runs if runs else None}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poison-runs", type=int, default=20)
    p.add_argument("--race-runs", type=int, default=40)
    p.add_argument("--control-chunk", type=int, default=256)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    failures: list[str] = []
    report: dict = {}

    for title, fn in (
            ("EQUIVALENCE  (no reclamation may change a single bit)",
             equivalence),
            ("RESIDENCY  (the bytes must actually go back)", residency)):
        print("=" * 78)
        print(title)
        print("=" * 78)
        try:
            for line in fn():
                print("  " + line)
        except Exception as exc:                               # noqa: BLE001
            failures.append(f"{title.split()[0].lower()}: {exc}")
            print(f"  FAIL  {exc}")
        print()

    print("=" * 78)
    print(f"POISON CONTROLS  (the strong pair, {args.poison_runs} runs each)")
    print("=" * 78)
    try:
        stats = poison_controls(runs=args.poison_runs,
                                chunk=args.control_chunk)
        report["poison"] = stats
        d, l = stats["dead"], stats["live"]
        print(f"  chunk {stats['chunk']}, {CTRL_NX}x{CTRL_NY}x{CTRL_NZ}")
        print()
        print("  poison=dead  -- NaN over every byte the tightening reuses")
        print(f"    poison FIRED in      : {d['fired']}/{d['runs']} runs")
        print(f"    answer unchanged     : {d['survived']}")
        print(f"    answer MOVED (bad)   : {d['moved']}")
        print(f"    crashed              : {d['crashed']}")
        if d["fired"] < d["runs"]:
            failures.append("the dead-poison did not fire in every run")
        if d["moved"] or d["crashed"]:
            failures.append(
                f"dead-poison changed the answer in {d['moved']} runs and "
                f"crashed {d['crashed']}: a slot this reclamation calls dead "
                "is read after all")
        else:
            print("    VERDICT              : the reused storage is "
                  "genuinely dead")
        print()
        print("  poison=live  -- NaN over the carried prefix (MUST break)")
        print(f"    poison FIRED in      : {l['fired']}/{l['runs']} runs")
        print(f"    CAUGHT               : {l['caught']} "
              f"({l['caught_by_finite_check']} by the finite check, "
              f"{l['caught_by_digest']} by the digest)")
        print(f"    MISSED               : {l['missed']}")
        print(f"    MISS RATE            : "
              f"{100.0 * stats['live_miss_rate']:.1f}%")
        if l["missed"]:
            failures.append(
                f"live-poison MISSED {l['missed']}/{l['runs']} "
                f"({100.0 * stats['live_miss_rate']:.1f}%): the instrument "
                "cannot reliably see a lifetime violation, so the "
                "dead-poison result above proves nothing")
        else:
            print("    VERDICT              : the control FIRES and is "
                  "caught every time, 0 misses")
    except Exception as exc:                                   # noqa: BLE001
        failures.append(f"poison: {exc}")
        print(f"  FAIL  {exc}")
    print()

    print("=" * 78)
    print(f"RELEASE RACE  (the weak control, reported as such, "
          f"{args.race_runs} runs)")
    print("=" * 78)
    try:
        race = release_race(runs=args.race_runs, chunk=args.control_chunk)
        report["race"] = race
        print(f"  hazard fired in : {race['hazard_fired_runs']}/"
              f"{race['runs']} runs")
        print(f"  answer moved    : {race['answer_moved']}")
        print(f"  crashed         : {race['crashes']}")
        print(f"  MISSES          : {race['misses']}")
        print(f"  MISS RATE       : {100.0 * race['miss_rate']:.1f}%")
        print()
        print("  A high miss rate here is NOT a pass and NOT a failure of the")
        print("  reclamation.  CuPy will not free a block that a live view")
        print("  still references, so the squatter is handed other memory and")
        print("  the hazard is inert by construction.  That is why the poison")
        print("  controls above, not this one, are what the gate rests on.")
    except Exception as exc:                                   # noqa: BLE001
        failures.append(f"race: {exc}")
        print(f"  FAIL  {exc}")
    print()

    print("=" * 78)
    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
    else:
        print("PASSED")
    print("=" * 78)
    if args.json:
        report["failures"] = failures
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
