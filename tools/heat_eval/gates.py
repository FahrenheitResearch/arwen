"""The box run's gates: G0 (the heating's GPU tests), G3 (convection grows), G1 (cards per arm).

G2 (the StormScope member count) lives in ``case_steps.sh nowcast``, which
halves the batch on a failed run.  The three here:

* **G0**, once, on four cards through the mutex, as soon as the first case
  has its door case (``case_steps.sh g0-door``): work package 2's GPU tests,
  with ``GPUWM_HEAT_BOX_DOOR`` naming that case so the real door runs on 2
  and 4 cards and every history variable is compared.  Passes only when
  tests ran and none failed or errored, and the multi-card heating file ran
  with no skip: a GPU test that skipped proved nothing, and a vacuous pass
  would put card time on unproven heating.  Only heated arms wait for it.
* **G3**, per case, before its preparations: the MRMS composite's 35 dBZ
  area must grow at least 1.5 times from t0 to t0 + 3 h.  Prevents spending
  card time on a case with no new storms to place, which cannot answer the
  question.  The composites are fetched and decoded by ``rw_mrms``; this
  only counts cells.  A decision case that fails is replaced by the
  reserve.  The anchor's G3 is recorded and never gates: rule 6 asks only
  that the anchor is not made worse, which needs no new storms, and the
  anchor is this package's acceptance case.  A G3 that cannot be measured
  (MRMS unreachable) fails the case rather than leaving the chain waiting.
* **G1**, once, on the first case that can run: a probe of arm A on two
  cards, stopped after its first forecast hour, records peak memory per
  card and seconds per forecast hour, and sets the cards per arm: 2, or 4
  when a card passed 85 GiB or the probe did not reach f01.  The heating's
  own memory is added from the design (DESIGN.md 5c: about 2.3 GB of slab
  per card at 2 cards, and about 3 GB more on the card that builds the
  slots), so the probe needs neither gate G0 nor any window, and arm A can
  start on every case while G0 runs.

    python -m tools.heat_eval.gates g0 [--root /work/nh] [--engine DIR]
    python -m tools.heat_eval.gates g3 --case ID [--root /work/nh]
    python -m tools.heat_eval.gates g1 --probe DIR [--heated 0|1] [--root /work/nh]

``--heated 0`` (the default) adds the heating's memory to the probe's peak.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from tools.heat_eval import plan as planmod
from tools.heat_eval.score_case import find_frames, seam

G1_LIMIT_GIB = 85.0
#: The heating's per-card slab at two cards plus the transient on the card
#: that builds the slots (DESIGN.md section 5c: about 2.3 GB and about 3 GB),
#: added to an unheated probe's peak.  Both on every card: conservative.
FORCING_GIB_PER_CARD = 2.3 + 3.0
#: Work package 2's box list (WP2-ADAPTER-HOOK-NOTES.md), test_radar_tten_forcing.py
#: included: an early desktop run of it touched a card and failed inside CuPy.
G0_TESTS = ("tests/test_radar_tten_lane6.py", "tests/test_radar_tten_grid.py",
            "tests/test_forecast_heating_gpu.py", "tests/test_radar_tten_forcing.py")
#: The file whose tests are the multi-card heating proofs; a skip there is a fail.
G0_REQUIRED_FILE = "test_forecast_heating_gpu"


def g0_verdict(junit: Path, returncode: int) -> dict[str, object]:
    """Pass only when tests ran, none failed or errored, and the GPU file ran unskipped."""
    cases = list(ET.parse(junit).getroot().iter("testcase")) if junit.is_file() else []
    failures = sum(1 for case in cases if case.find("failure") is not None)
    errors = sum(1 for case in cases if case.find("error") is not None)
    skipped = [f"{case.get('classname', '')}::{case.get('name', '')}"
               for case in cases if case.find("skipped") is not None]
    required_ran = any(G0_REQUIRED_FILE in (case.get("classname") or "") for case in cases)
    required_skips = [name for name in skipped if G0_REQUIRED_FILE in name]
    passed = (returncode == 0 and bool(cases) and failures == 0 and errors == 0
              and required_ran and not required_skips)
    return {"gate": "G0", "pass": passed, "pytest_rc": returncode, "tests": len(cases),
            "failures": failures, "errors": errors, "skipped": skipped,
            "required_file_ran": required_ran, "required_file_skips": required_skips,
            "prevents": "card time on multi-card heating that no test exercised"}


def g0(root: Path, engine: Path) -> dict[str, object]:
    coord = root / "coord"
    coord.mkdir(parents=True, exist_ok=True)
    junit = coord / "G0.junit.xml"
    junit.unlink(missing_ok=True)
    env = dict(os.environ)
    door = coord / "g0-door.json"
    if door.is_file():
        env["GPUWM_HEAT_BOX_DOOR"] = str(door)
    with open(coord / "G0.pytest.log", "w", encoding="utf-8") as log:
        done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", f"--junitxml={junit}",
                               *G0_TESTS], cwd=engine, stdout=log, stderr=subprocess.STDOUT,
                              env=env)
    record = g0_verdict(junit, done.returncode)
    (coord / "G0.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def g3_growth(cells_t0: int, cells_later: int) -> float:
    """Area growth.  No echo at t0 counts as growth only if echo appears later."""
    if cells_t0:
        return cells_later / cells_t0
    return float("inf") if cells_later else 0.0


def g3_record(case: planmod.Case, areas: dict[int, int] | None, error: str = "") -> dict[str, object]:
    """The G3 verdict: ``measured_pass`` is the number, ``open`` is whether the arms may run."""
    growth = None if areas is None else g3_growth(areas[0], areas[3])
    measured = growth is not None and growth >= case.g3_growth
    gates = case.role != "anchor"
    return {"gate": "G3", "case": case.id, "role": case.role,
            "threshold_dbz": case.g3_threshold_dbz,
            "cells_t0": None if areas is None else areas[0],
            "cells_t0_plus_3h": None if areas is None else areas[3],
            "growth": growth, "needed": case.g3_growth, "measured_pass": measured,
            "gating": gates, "open": measured or not gates, "error": error,
            "prevents": "card time on a case with no new storms to place"}


def g3(root: Path, case: planmod.Case) -> dict[str, object]:
    areas: dict[int, int] | None = {}
    error = ""
    try:
        from gpuwm.obs.mrms_fetch import MrmsCompositeCache
        from gpuwm.obs.sources import MrmsCompositeSource
        cache = MrmsCompositeCache(planmod.case_dir(root, case) / "mrms-g3", window_seconds=120)
        for hours in (0, 3):
            valid = seam(case.t0 + timedelta(hours=hours))
            cache.ensure(valid)
            source = MrmsCompositeSource(cache.pack_paths(), cache.geometry_path, match_seconds=120)
            field = source.field(valid)
            areas[hours] = int(np.count_nonzero(field.valid & (field.values >= case.g3_threshold_dbz)))
    except Exception as exc:  # noqa: BLE001 - a gate that cannot measure records why; it never hangs
        areas, error = None, f"{type(exc).__name__}: {exc}"
    record = g3_record(case, areas, error)
    out = planmod.case_dir(root, case)
    out.mkdir(parents=True, exist_ok=True)
    (out / "G3.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for name in ("G3.pass", "G3.fail"):
        (out / name).unlink(missing_ok=True)
    note = "" if record["measured_pass"] or not record["open"] else " (anchor: recorded, not gating)"
    (out / ("G3.pass" if record["open"] else "G3.fail")).write_text(
        f"{record['growth']}{note}\n", encoding="utf-8")
    return record


def g1(probe: Path, heated: bool = False) -> dict[str, object]:
    """Peak memory per card and seconds per forecast hour of a probe run."""
    cards = [c for c in (probe / "cards.txt").read_text(encoding="utf-8").split("cards=")[1].split()[0].split(",") if c]
    peak: dict[str, float] = {}
    with open(probe / "gpu-samples.csv", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            index, used = row[1].strip(), row[2].strip()
            if index in cards and used.replace(".", "", 1).isdigit():
                peak[index] = max(peak.get(index, 0.0), float(used) / 1024.0)
    frames = find_frames(probe / "out")
    times = sorted(frames)
    hour_seconds = None
    if len(times) >= 2:
        hour_seconds = frames[times[1]].stat().st_mtime - frames[times[0]].stat().st_mtime
    done = {}
    if (probe / "done.json").is_file():
        done = json.loads((probe / "done.json").read_text(encoding="utf-8"))
    reached = bool(done.get("reached_f01")) and len(times) >= 2
    worst = max(peak.values()) if peak else None
    if worst is not None and not heated:
        worst += FORCING_GIB_PER_CARD
    if not reached:
        note = "the probe did not reach f01 on two cards: four cards per arm"
    elif worst is None:
        note = "no memory samples: four cards per arm"
    else:
        note = ""
    cards_per_arm = 4 if (not reached or worst is None or worst > G1_LIMIT_GIB) else 2
    record = {"gate": "G1", "cards": cards, "peak_gib_per_card": peak, "limit_gib": G1_LIMIT_GIB,
              "heated_probe": heated, "worst_gib_with_forcing": worst,
              "reached_f01": reached, "probe_rc": done.get("rc"),
              "seconds_first_forecast_hour": hour_seconds, "cards_per_arm": cards_per_arm,
              "note": note}
    (probe / "G1.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.heat_eval.gates")
    sub = parser.add_subparsers(dest="gate", required=True)
    zero = sub.add_parser("g0")
    zero.add_argument("--root", type=Path, default=planmod.DEFAULT_ROOT)
    zero.add_argument("--engine", type=Path, default=Path(os.environ.get("E", ".")))
    three = sub.add_parser("g3")
    three.add_argument("--case", required=True)
    three.add_argument("--root", type=Path, default=planmod.DEFAULT_ROOT)
    one = sub.add_parser("g1")
    one.add_argument("--probe", type=Path, required=True)
    one.add_argument("--heated", type=int, default=0, choices=(0, 1))
    one.add_argument("--root", type=Path, default=planmod.DEFAULT_ROOT)
    args = parser.parse_args(argv)
    if args.gate == "g0":
        record = g0(args.root, args.engine)
        print(json.dumps(record))
        return 0 if record["pass"] else 1
    if args.gate == "g3":
        record = g3(args.root, planmod.case_by_id(args.case))
        print(json.dumps(record))
        return 0 if record["open"] else 1
    record = g1(args.probe, heated=bool(args.heated))
    (args.root / "coord").mkdir(parents=True, exist_ok=True)
    (args.root / "coord" / "cards-per-arm").write_text(f"{record['cards_per_arm']}\n", encoding="utf-8")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
