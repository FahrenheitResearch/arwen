"""Case and family sheets, with all weather fields drawn by Rust renderers.

Sheets come first: a person judges them before any number is read.  Each
sheet is redrawn from whatever has landed, so it can be called after every
arm finishes and the columns fill in while the runs go on.

1. ``refc-arms-f01-f03``  rows f01-f03, columns MRMS | A | B | C | E | HRRR
2. ``refc-arms-f06-f18``  rows f06, f12, f18, the same columns
3. ``refc-optional-f01-f03``  MRMS | C | C60 | E | E3D | HRRR (full plan)
4. ``nowcast-vs-mrms``    the nowcast's frames at +30, +60, +90 and +120 min
                          beside MRMS (``frames:`` panels)
5. ``heating-windows``    the C and E heating windows ending 18:30, 19:00,
                          19:30 and 20:00 beside MRMS (``ttenref:`` panels)
6. ``surface-t2m`` / ``surface-td2m`` / ``surface-wspd10``  rows f01, f03,
                          f06, columns A | C | HRRR, station error dots

A column is drawn only when that arm holds a frame for every row of the
sheet (arm B's core run stops at f06, so it is left off the f06-f18 sheet).
An arm whose heating read an oracle window is refused anywhere but an
oracle column, here as in the decision.

    python -m tools.heat_eval.sheets --case ID [--root /work/nh] [--plan core]
        [--only NAME,...] [--gallery DIR]

Family sheets use the existing rw_wrfbatch probability/postage product table:
``python -m tools.heat_eval.sheets --receipt RECEIPT --renderer BINARY --out DIR``.
Python composes the completed native PNGs, without reading weather arrays.

``--gallery`` copies the sheets and their plain captions to a folder
(the evidence gallery, once copied back from the box).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
import json
import os
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Sequence

from gpuwm.rustwx import run_ensemble_product_renderer
from tools.heat_eval import plan as planmod
from tools.heat_eval.score_case import find_frames, seam

CORE_COLUMNS = ("A", "B", "C", "E")
OPTIONAL_COLUMNS = ("C", "C60", "E", "E3D")


class SheetRefused(ValueError):
    """A sheet that would put a run where it does not belong."""


def renderer() -> str:
    path = os.environ.get("GPUWM_RW_COMPARE") or shutil.which("rw_compare")
    if not path:
        raise SystemExit("rw_compare not found (set GPUWM_RW_COMPARE)")
    return path


def _arm_frames(root: Path, case: planmod.Case, arm: planmod.Arm) -> dict[str, Path]:
    folder = planmod.arm_dir(root, case, arm)
    if not (folder / "READY").is_file() and not (folder / "out").is_dir():
        return {}
    return find_frames(folder / "out")


def _classes(root: Path, case: planmod.Case, arm: planmod.Arm) -> list[str]:
    path = planmod.arm_dir(root, case, arm) / "heating-classes.json"
    if not arm.heated or not path.is_file():
        return []
    return list(json.loads(path.read_text(encoding="utf-8"))["classes"])


def check_column(arm: planmod.Arm, classes: Sequence[str], oracle_columns: Sequence[str]) -> None:
    if "oracle" in classes and arm.column not in oracle_columns:
        raise SheetRefused(
            f"arm {arm.id} read oracle windows; it can fill only an oracle column "
            f"({', '.join(oracle_columns)}), never {arm.column!r}")


def arm_panels(root: Path, case: planmod.Case, arms: Sequence[planmod.Arm],
               valid_hours: Sequence[int], oracle_columns: Sequence[str]) -> tuple[list[list[str]], list[str], list[str]]:
    """Per row, the ``--panel`` values of every arm with a frame in every row."""
    drawn, skipped = [], []
    frames = {}
    for arm in arms:
        frames[arm.id] = _arm_frames(root, case, arm)
        check_column(arm, _classes(root, case, arm), oracle_columns)
    for arm in arms:
        if all(seam(case.t0 + timedelta(hours=h)) in frames[arm.id] for h in valid_hours):
            drawn.append(arm)
        else:
            skipped.append(arm.id)
    rows = []
    for hours in valid_hours:
        valid = seam(case.t0 + timedelta(hours=hours))
        rows.append([f"{arm.label}={frames[arm.id][valid]}" for arm in drawn])
    return rows, [arm.id for arm in drawn], skipped


def _base(root: Path, case: planmod.Case, name: str, products: str = "refc") -> list[str]:
    case_root = planmod.case_dir(root, case)
    return [renderer(), "--store-root", str(case_root / "sheets" / "store" / name),
            "--out-dir", str(case_root / "sheets"), "--layout", "flat", "--products", products,
            "--sheet-name", name, "--reference-dir", str(case_root / "reference"),
            "--reference-cache", str(case_root / "sheets" / "reference-cache"),
            "--cycle", case.t0.strftime("%Y%m%d%H"), "--source-label", "WOOF"]


def _with_rows(command: list[str], rows: Sequence[Sequence[str]]) -> list[str]:
    for index, row in enumerate(rows):
        if index:
            command.append("--row")
        for panel in row:
            command += ["--panel", panel]
    return command


def plan_sheets(root: Path, case: planmod.Case, plan: str) -> list[dict[str, object]]:
    """Every sheet this case can draw now: name, caption, command, columns."""
    table = planmod.load_arms()
    arms = {arm.id: arm for arm in table.for_case(case, plan)}
    oracle = table.oracle_columns
    sheets: list[dict[str, object]] = []

    def arm_sheet(name, ids, hours, caption, products="refc", references="mrms,hrrr",
                  stations=None):
        chosen = [arms[i] for i in ids if i in arms]
        rows, drawn, skipped = arm_panels(root, case, chosen, hours, oracle)
        if "A" in ids and "A" not in drawn:
            return
        if not drawn:
            return
        command = _base(root, case, name, products) + ["--reference", references]
        refs = references.split(",")
        order = [r for r in refs if r == "mrms"] + [str(i + 1) for i in range(len(drawn))] + \
                [r for r in refs if r != "mrms"]
        command += ["--order", ",".join(order)]
        if stations:
            command += ["--stations", str(stations), "--station-mode", "error"]
        sheets.append({"name": name, "caption": caption, "columns": order,
                       "arms": drawn, "skipped_arms": skipped,
                       "command": _with_rows(command, rows)})

    arm_sheet("refc-arms-f01-f03", CORE_COLUMNS, (1, 2, 3),
              "Composite reflectivity one to three hours after the start. Columns: observed "
              "radar (MRMS), then WOOF with no heating (A), with observed radar heating in the "
              "hour before (B), with the nowcast's heating (C), with real future radar through "
              "the same path (E, the ceiling, not a forecast), then HRRR. Grey: no radar "
              "coverage.")
    arm_sheet("refc-arms-f06-f18", CORE_COLUMNS, (6, 12, 18),
              "The same columns six, twelve and eighteen hours after the start: whether a "
              "gain survives and whether the heating harms the longer forecast.")
    if plan == "full":
        arm_sheet("refc-optional-f01-f03", OPTIONAL_COLUMNS, (1, 2, 3),
                  "The nowcast arm with two hours (C) and one hour (C60) of heating, and the "
                  "two ceilings: real radar as a 2D composite (E) and as real 3D volumes (E3D).")
    stations = planmod.case_dir(root, case) / "scores" / "stations" / "obs-sheets.json"
    for product, words in (("t2m", "2 m temperature"), ("td2m", "2 m dewpoint"),
                           ("wspd10", "10 m wind speed")):
        arm_sheet(f"surface-{product}", ("A", "C"), (1, 3, 6),
                  f"{words.capitalize()} one, three and six hours after the start: no heating "
                  f"(A), nowcast heating (C) and HRRR, with station errors (observed minus "
                  f"forecast) as dots, to show cold pools and outflow.",
                  products=product, references="hrrr",
                  stations=stations if stations.is_file() else None)

    grid_arm = arms.get("A")
    grid_frames = _arm_frames(root, case, grid_arm) if grid_arm else {}
    if grid_frames:
        grid_file = grid_frames[sorted(grid_frames)[0]]
        nowcast = planmod.case_dir(root, case) / "nowcast" / "stormscope"
        if (nowcast / "nowcast.json").is_file():
            rows = [[f"StormScope nowcast=frames:{nowcast}@{planmod.iso(case.t0 + timedelta(minutes=m))}"]
                    for m in (30, 60, 90, 120)]
            command = _base(root, case, "nowcast-vs-mrms") + [
                "--reference", "mrms", "--order", "1,mrms", "--panel-grid", str(grid_file)]
            sheets.append({"name": "nowcast-vs-mrms", "columns": ["nowcast", "mrms"],
                           "arms": [], "skipped_arms": [],
                           "caption": "The nowcast's composite reflectivity (probability-matched "
                                      "mean of its members) 30, 60, 90 and 120 minutes after the "
                                      "start, beside what the radar then saw.",
                           "command": _with_rows(command, rows)})
        windows = planmod.case_dir(root, case) / "windows"
        pairs = [("C heating windows", "stormscope"), ("E heating windows (oracle)", "mrms2d")]
        present = [(label, name) for label, name in pairs if (windows / name).is_dir()]
        ends = [case.t0 + timedelta(minutes=m) for m in (30, 60, 90, 120)]
        present = [(label, name) for label, name in present
                   if all((windows / name / planmod.stamp(end) / "ref.json").is_file() for end in ends)]
        if present:
            rows = [[f"{label}=ttenref:{windows / name / planmod.stamp(end)}" for label, name in present]
                    for end in ends]
            order = ["mrms"] + [str(i + 1) for i in range(len(present))]
            command = _base(root, case, "heating-windows") + [
                "--reference", "mrms", "--order", ",".join(order), "--panel-grid", str(grid_file)]
            sheets.append({"name": "heating-windows", "columns": order, "arms": [],
                           "skipped_arms": [],
                           "caption": "The heating each arm was given: the column maximum of "
                                      "the radar windows the nowcast arm (C) and the oracle arm "
                                      "(E) read, beside what the radar saw at the same time.",
                           "command": _with_rows(command, rows)})
    return sheets


def draw(root: Path, case: planmod.Case, plan: str, only: Sequence[str] | None,
         gallery: Path | None) -> list[dict[str, object]]:
    out = planmod.case_dir(root, case) / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for sheet in plan_sheets(root, case, plan):
        if only and sheet["name"] not in only:
            continue
        log = out / f"{sheet['name']}.log"
        with open(log, "w", encoding="utf-8") as handle:
            done = subprocess.run(sheet["command"], stdout=handle, stderr=subprocess.STDOUT)
        png = out / f"{sheet['name']}.png"
        record = {key: sheet[key] for key in ("name", "caption", "columns", "arms", "skipped_arms")}
        record.update({"rc": done.returncode, "png": str(png) if png.is_file() else None,
                       "log": str(log)})
        results.append(record)
        if gallery and png.is_file():
            gallery.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, gallery / f"{case.id}-{png.name}")
            (gallery / f"{case.id}-{sheet['name']}.txt").write_text(
                f"{case.id} ({case.date}, {case.region}): {sheet['caption']}\n", encoding="utf-8")
    (out / "sheets.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


LABELS = {"observed": "MRMS observed composite", "ai": "StormScope",
          "physics": "DA forecast"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_family_panels(receipt, *, renderer, out):
    if receipt.get("schema") != "gpuwm-ensemble-grid-match.v1" or receipt.get("status") != "complete":
        raise ValueError("panels require a complete native gridded match receipt")
    out.mkdir(parents=True, exist_ok=False)
    # The shared resolver otherwise prefers an installed rw_ensbatch. This
    # CLI's explicit renderer path is the artifact whose bytes we pin.

    def draw(product):
        source = Path(product["path"])
        if digest(source) != product["sha256"]:
            raise ValueError(f"native product changed: {source}")
        family = product["family"]
        start = time.monotonic()
        files, stdout = run_ensemble_product_renderer(renderer, source, out_dir=out / family,
            fields=("refl",), domain="d01-3km", width=960, height=720,
            source_label=LABELS[family], products=tuple(product["products"]))
        (out / f"{family}.renderer.log").write_text(stdout)
        return {"family": family, "files": [{"path": str(path), "sha256": digest(path)} for path in files],
                "wall_seconds": time.monotonic() - start, "parallelism": "native Rust renderer, Rayon pool"}

    previous = os.environ.get("GPUWM_ENSEMBLE_RENDERER")
    os.environ["GPUWM_ENSEMBLE_RENDERER"] = str(renderer)
    try:
        with ThreadPoolExecutor(max_workers=len(receipt["products"])) as pool:
            rows = list(pool.map(draw, receipt["products"]))
    finally:
        if previous is None:
            os.environ.pop("GPUWM_ENSEMBLE_RENDERER", None)
        else:
            os.environ["GPUWM_ENSEMBLE_RENDERER"] = previous
    # Composition only: all map pixels, labels, scales and geographic overlays
    # were rendered by rw_wrfbatch. Pillow does not see weather values.
    from PIL import Image, ImageDraw
    sheet = Image.new("RGB", (3 * 960, 760), "#eef0f3")
    pen = ImageDraw.Draw(sheet)
    for index, family in enumerate(("observed", "ai", "physics")):
        row = next((row for row in rows if row["family"] == family), None)
        match = None if row is None else next((Path(item["path"]) for item in row["files"]
            if ("ens_mean_refl" if family == "observed" else "ens_prob_refl_ge_35") in item["path"]), None)
        pen.text((index * 960 + 20, 12), LABELS[family], fill="black")
        if match is None:
            pen.text((index * 960 + 40, 330), "No family product in this scored frame", fill="black")
        else:
            with Image.open(match) as picture:
                sheet.paste(picture.convert("RGB"), (index * 960, 40))
    comparison = out / "mrms-stormscope-da.png"
    sheet.save(comparison)
    result = {"schema": "gpuwm-ensemble-reflectivity-panels.v1", "valid_time": receipt["valid_time"],
              "renderer": {"path": str(renderer), "sha256": digest(renderer)}, "families": rows,
              "comparison": {"path": str(comparison), "sha256": digest(comparison)},
              "scope": "MRMS dBZ, family probability >=35 dBZ, and native member postage; missing family labelled"}
    (out / "panels.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.heat_eval.sheets")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case")
    mode.add_argument("--receipt", type=Path, help="native ensemble family match receipt")
    parser.add_argument("--renderer", type=Path, help="explicit rw_wrfbatch binary for family sheets")
    parser.add_argument("--out", type=Path, help="new family sheet directory")
    parser.add_argument("--root", type=Path, default=planmod.DEFAULT_ROOT)
    parser.add_argument("--plan", default="core", choices=planmod.PLANS)
    parser.add_argument("--only", default="")
    parser.add_argument("--gallery", type=Path)
    args = parser.parse_args(argv)
    if args.receipt:
        if args.renderer is None or args.out is None:
            parser.error("--receipt needs --renderer and --out")
        draw_family_panels(json.loads(args.receipt.read_text()),
                           renderer=args.renderer.resolve(), out=args.out.resolve())
        return 0
    try:
        results = draw(args.root, planmod.case_by_id(args.case), args.plan,
                       [n for n in args.only.split(",") if n] or None, args.gallery)
    except SheetRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    for row in results:
        print(f"{row['name']}\trc={row['rc']}\t{row['png']}")
    return 0 if all(row["rc"] == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
