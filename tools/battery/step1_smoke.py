"""GPU step-1 smoke matrix: every shipped config and physics profile steps.

THE BREAKAGE IT PREVENTS: 2.8.8's candidate crashed every real forecast at
model step 1 (``NameError: name 'WRF_DIFFUSION' is not defined`` in
``gpuwm/core/dycore.py``).  CPU tests cannot reach the GPU step, and nobody
had run a forecast before the intake handoff.  This tool is the forecast
nobody ran, cut down to what finds that class: for every configuration the
wheel ships (``configs/**/*.toml``, the recipes included) and every shipped
single-domain physics profile, it builds a small at-rest state carrying
exactly that row's physics and advances a few model steps on a card.

What a row is
-------------
* ``config:<path>#d<NN>`` -- that domain's resolved ``RunConfig`` from
  ``gpuwm.experiment.load_experiment`` (``#flat`` for a legacy
  ``[grid]``/``[dynamics]``/``[run]`` file, read by
  ``gpuwm.config.load_config``), with its physics, diffusion, damping,
  vertical and time-step settings kept and only the horizontal extent and
  lateral boundaries replaced: ``--nxy`` square, open (radiating) on all
  four sides, flat, no map projection.  A real forecast's specified
  lateral boundary, nest and preparation paths are therefore NOT
  exercised here; everything the dynamics and the columns do on the card
  is.
* ``profile:<id>`` -- the profile's own switch table
  (``gpuwm.physics_compat.single_domain_runtime_switches``) on the
  harness's idealized moist grid (3 km, ``--nz`` levels, 20 km top).

The state is ``tilestream.physics_inventory.default_builder``: the WK82
sounding with seeded noise, moisture from the sounding, and the full
``initialize_physics`` driver, i.e. the builder the tile gates step, plus
the declarations a real road makes from its static catalogue and the
builder cannot: a leaf area of :data:`DECLARED_LEAF_AREA` where RUC's
``rdlai2d`` left LAI unseeded, and, under RUC's mosaic switches, category
fractions wholly in the builder's own dominant land-use and soil
categories (:func:`_declared_mosaic_fractions`), and, under ``topo_wind``
or ``gwd_opt``, the sub-grid orographic statistics of flat ground, all
zero (:func:`_declared_terrain_drag_static`).  Rows
whose shrunk ``RunConfig`` is identical in every field are one computation
and run once; every config they stand for is listed in ``covers``.

A row PASSES when every step returns, the device synchronizes, and every
floating array of the persisted inventory is finite.  A row FAILS with the
phase it died in (``load``, ``shrink``, ``build``, ``step``, ``finite``,
``timeout``, ``process``) and the exception.  A TOML carrying no model
table at all (a verification spec, an ensemble overlay, a scaffold) is a
SKIP row naming the tables it does carry.

Running it
----------
One process per row (a crash, a CUDA fault or a hang costs that row only),
``--per-card`` rows at once on each of ``--cards`` cards::

    python tools/battery/step1_smoke.py --cards 8 --out step1.json
    python tools/battery/step1_smoke.py --list            # rows, no GPU
    python tools/battery/step1_smoke.py --only profile:   # a subset

Exit status: 0 when no row fails, 1 when any fails, 2 on a usage error.
The receipt (``--out``) carries every row with its exception and traceback
tail, the commit and the card names.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The settings a row replaces.  Everything else in the RunConfig -- every
#: physics switch, diffusion, damping, the vertical coordinate, dt and the
#: acoustic step -- is the config's own.
#: Open rather than periodic: a real forecast never runs periodic, and the
#: run-time diagnostics (``nwp_diagnostics``) refuse a periodic domain.
OPEN_FLAT = dict(open_x=True, open_y=True, specified=False,
                 nested=False, map_proj=0, terrain_opt=0, hill_height=0.0)

#: A TOML carrying none of these tables is not a model configuration.
MODEL_TABLES = frozenset({"grid", "dynamics", "run", "domains", "domain",
                          "shared", "experiment", "physics", "projection"})

#: Fields printed beside each row so a failure names its physics.
SUMMARY_FIELDS = ("mp_physics", "bl_pbl_physics", "sf_sfclay_physics",
                  "sf_surface_physics", "ra_lw_physics", "ra_sw_physics",
                  "cu_physics", "km_opt", "diff_6th_opt", "nz", "dt")

_UNSET_VARIABLE = re.compile(r"references \$\{(\w+)\}, which is not set")


@dataclasses.dataclass
class Row:
    row_id: str
    kind: str                       # "config" | "profile"
    covers: list[str]
    cfg: object | None = None       # the RunConfig to step, or None
    phase: str | None = None        # set when enumeration settled the row
    status: str = "FAIL"            # of a row enumeration settled
    error: str | None = None
    detail: str | None = None


class NotAModelConfig(ValueError):
    """The TOML carries no model table; it configures something else."""


def _error_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0] if text else ''}"[:400]


def _summary(cfg) -> dict:
    return {name: getattr(cfg, name, None) for name in SUMMARY_FIELDS}


def _signature(cfg) -> str:
    payload = repr(sorted(dataclasses.asdict(cfg).items()))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def shipped_configs(root: Path = ROOT) -> list[Path]:
    """Every TOML the wheel ships under ``configs/`` (recipes included)."""
    return sorted(p for p in (root / "configs").rglob("*.toml") if p.is_file())


def _load_domains(path: Path) -> list[tuple[str, object]]:
    """``[(label, RunConfig)]`` for every domain a shipped TOML runs.

    The experiment loader first.  A case-data path spelled through an
    environment variable (``${NAME}``) is pointed at the temp directory
    for the load only: the loader names the variable, and the smoke reads
    no case input.  A legacy ``[grid]``/``[dynamics]``/``[run]`` file is
    read by :func:`gpuwm.config.load_config`, the loader its doors use.
    Anything else is re-raised.
    """
    from gpuwm.config import load_config
    from gpuwm.experiment import load_experiment

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if not MODEL_TABLES & set(raw):
        raise NotAModelConfig(
            f"not a model configuration (tables: {', '.join(sorted(raw))})")
    placed: list[str] = []
    try:
        for _ in range(16):
            try:
                exp = load_experiment(path)
            except ValueError as exc:
                match = _UNSET_VARIABLE.search(str(exc))
                if match and match.group(1) not in os.environ:
                    os.environ[match.group(1)] = tempfile.gettempdir()
                    placed.append(match.group(1))
                    continue
                if {"grid", "dynamics", "run"} <= set(raw):
                    return [("flat", load_config(path))]
                raise
            return [(f"d{int(dc.grid_id):02d}", dc.run) for dc in exp.domains]
        raise RuntimeError(f"more than 16 unset variables: {placed}")
    finally:
        for name in placed:
            os.environ.pop(name, None)


def _config_rows(path: Path, nxy: int) -> list[Row]:
    rel = path.relative_to(ROOT).as_posix()
    try:
        domains = _load_domains(path)
    except NotAModelConfig as exc:
        return [Row(f"config:{rel}", "config", [rel], phase="load",
                    status="SKIP", error=str(exc))]
    except BaseException as exc:  # noqa: BLE001 - every failure is a row
        return [Row(f"config:{rel}", "config", [rel], phase="load",
                    error=_error_line(exc),
                    detail=traceback.format_exc(limit=6))]
    rows = []
    for label, run in domains:
        row_id = f"config:{rel}#{label}"
        try:
            cfg = dataclasses.replace(run, nx=nxy, ny=nxy, **OPEN_FLAT)
        except BaseException as exc:  # noqa: BLE001
            rows.append(Row(row_id, "config", [row_id], phase="shrink",
                            error=_error_line(exc),
                            detail=traceback.format_exc(limit=6)))
            continue
        rows.append(Row(row_id, "config", [row_id], cfg=cfg))
    return rows


def _profile_rows(nxy: int, nz: int) -> list[Row]:
    from gpuwm.physics_compat import (SINGLE_DOMAIN_PHYSICS_PROFILES,
                                      single_domain_runtime_switches)
    from tilestream.harness import make_config

    rows = []
    for profile in SINGLE_DOMAIN_PHYSICS_PROFILES:
        row_id = f"profile:{profile}"
        try:
            switches = single_domain_runtime_switches(profile)
            cfg = make_config(nxy, nxy, nz, dx=3000.0, dy=3000.0,
                              ztop=20000.0, dt=12.0, moist=True)
            # A profile pins real-data switches (terrain_opt among them);
            # the row's grid stays the flat open one.
            cfg = dataclasses.replace(cfg, **{**switches, **OPEN_FLAT})
        except BaseException as exc:  # noqa: BLE001
            rows.append(Row(row_id, "profile", [row_id], phase="shrink",
                            error=_error_line(exc),
                            detail=traceback.format_exc(limit=6)))
            continue
        rows.append(Row(row_id, "profile", [row_id], cfg=cfg))
    return rows


def enumerate_rows(*, nxy: int = 32, nz: int = 40, only: str | None = None,
                   dedupe: bool = True) -> list[Row]:
    """Every row, CPU only: configs then profiles, identical configs merged.

    Loader advisories are muted: they describe the full-size files, and
    the rows step something else.
    """
    from gpuwm.explain import muted_warnings

    rows: list[Row] = []
    with muted_warnings():
        for path in shipped_configs():
            rows.extend(_config_rows(path, nxy))
        rows.extend(_profile_rows(nxy, nz))
    if only:
        rows = [row for row in rows
                if any(only in name for name in (row.row_id, *row.covers))]
    if not dedupe:
        return rows
    merged: list[Row] = []
    seen: dict[str, Row] = {}
    for row in rows:
        if row.cfg is None:
            merged.append(row)
            continue
        key = _signature(row.cfg)
        if key in seen:
            seen[key].covers.extend(row.covers)
            continue
        seen[key] = row
        merged.append(row)
    return merged


# --------------------------------------------------------------------------
# one row, in its own process, on the card CUDA_VISIBLE_DEVICES names
# --------------------------------------------------------------------------

#: The leaf area a row declares where a real road would interpolate the
#: static LAI12M field (``gpuwm.core.landuse.surface_leaf_area``): RUC
#: under ``rdlai2d`` starts LAI not-a-number and refuses a road that
#: seeded none, and the at-rest builder has no static catalogue.
DECLARED_LEAF_AREA = 2.0


def _declared_mosaic_fractions(cfg, ny: int, nx: int) -> dict:
    """``landusef``/``soilctop`` for a RUC mosaic row, or nothing.

    ``initialize_physics`` refuses RUC's mosaic without the static
    category fractions a WPS road carries.  The at-rest builder runs one
    land-use category (``ivgtyp``) and one soil category (``isltyp``)
    everywhere, so the fractions that describe its surface are one in
    those categories and zero in every other row of the RUC tables the run
    loads (all rows present, so irrigation finds its crop and natural
    categories).
    """
    import inspect

    import numpy as np

    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.ruc import load_ruc_parameters

    if int(getattr(cfg, "sf_surface_physics", 0)) != 3:
        return {}
    defaults = inspect.signature(initialize_physics).parameters
    bundle = load_ruc_parameters()
    vegetation = bundle.vegetation_for(defaults["landuse_dataset"].default)
    out = {}
    for switch, name, category, rows in (
            ("mosaic_lu", "landusef", "ivgtyp", len(vegetation.rows)),
            ("mosaic_soil", "soilctop", "isltyp", len(bundle.soil.rows))):
        if not int(getattr(cfg, switch, 0) or 0):
            continue
        fractions = np.zeros((rows, ny, nx), dtype=np.float32)
        fractions[int(defaults[category].default) - 1] = 1.0
        out[name] = fractions
    return out


def _declared_terrain_drag_static(cfg, ny: int, nx: int) -> dict:
    """``terrain_drag_static`` for a ``topo_wind``/``gwd_opt`` row, or nothing.

    The terrain drag reads sub-grid orographic statistics a prepared
    forecast carries from WPS_GEOG and refuses a route that hands none.
    The row's ground is flat, so every statistic is zero, beside the flat
    ``HGT_M`` the drag checks the columns against.
    """
    import numpy as np

    from gpuwm.core.terrain_drag import required_static_fields

    topo_wind = int(getattr(cfg, "topo_wind", 0) or 0)
    gwd_opt = int(getattr(cfg, "gwd_opt", 0) or 0)
    if not (topo_wind or gwd_opt):
        return {}
    static = {name: np.zeros((ny, nx), dtype=np.float32)
              for name in ("HGT_M",
                           *required_static_fields(topo_wind, gwd_opt))}
    return {"terrain_drag_static": static}


def _seed_static_leaf_area(driver) -> None:
    fields = getattr(driver, "fields", None) or {}
    lai = fields.get("lai") if hasattr(fields, "get") else None
    if lai is None:
        return
    import cupy as cp
    if not bool(cp.isfinite(lai).all()):
        lai[...] = DECLARED_LEAF_AREA


def run_one(cfg, steps: int, seed: int) -> dict:
    """Build, step and check one config.  Raises nothing; returns a result."""
    started = time.perf_counter()
    result = {"status": "FAIL", "phase": "build", "error": None,
              "detail": None, "steps_done": 0}
    try:
        import cupy as cp

        from tilestream import harness, physics_inventory

        from gpuwm.core import physics as physics_module

        declared = {
            **_declared_mosaic_fractions(cfg, int(cfg.ny), int(cfg.nx)),
            **_declared_terrain_drag_static(cfg, int(cfg.ny), int(cfg.nx))}
        if declared:
            original = physics_module.initialize_physics

            def initialize_with_fractions(*a, **kw):
                return original(*a, **{**declared, **kw})
            physics_module.initialize_physics = initialize_with_fractions
        state, driver = physics_inventory.default_builder(cfg, seed)
        _seed_static_leaf_area(driver)
        result["phase"] = "step"
        for _ in range(int(steps)):
            harness.run_steps(state, cfg, 1)
            result["steps_done"] += 1
        result["phase"] = "finite"
        bad = sorted(
            name for name, array in harness.state_arrays(state).items()
            if getattr(getattr(array, "dtype", None), "kind", None) == "f"
            and not bool(cp.isfinite(array).all()))
        if bad:
            raise FloatingPointError(
                f"non-finite after {steps} steps: {', '.join(bad)}")
        result.update(status="PASS", phase=None)
    except BaseException as exc:  # noqa: BLE001 - the row's verdict
        result["error"] = _error_line(exc)
        result["detail"] = "".join(traceback.format_exception(exc))[-4000:]
    result["seconds"] = round(time.perf_counter() - started, 2)
    return result


def _worker(args) -> int:
    with open(args.row_file, "rb") as handle:
        cfg = pickle.load(handle)
    result = run_one(cfg, args.steps, args.seed)
    Path(args.result_file).write_text(json.dumps(result), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------

def _card_names() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _card_device(card: int) -> str:
    """The CUDA_VISIBLE_DEVICES value for launch slot ``card``.

    Slot i is the i-th device this process was granted (a card queue exports
    the grant as CUDA_VISIBLE_DEVICES, often as UUIDs); only an unset or empty
    grant falls back to the bare index.  Breakage it prevents: on a shared box
    every smoke worker landed on physical card 0 whatever cards the queue had
    granted, which left the granted cards idle and starved other jobs.
    """
    granted = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
               if d.strip()]
    if not granted:
        return str(card)
    if card >= len(granted):
        raise SystemExit(f"step1_smoke: --cards asks for slot {card}, but only "
                         f"{len(granted)} device(s) were granted "
                         f"(CUDA_VISIBLE_DEVICES={','.join(granted)})")
    return granted[card]


def _launch(row: Row, card: int, args, work: Path, index: int) -> dict:
    # Keyed by launch index, never by config signature: under --no-dedupe
    # identical configs run at once, and a twin's result file would hand a
    # crashed or timed-out row its twin's verdict (a crash reported PASS).
    row_file = work / f"row-{index:04d}.pkl"
    result_file = work / f"row-{index:04d}.json"
    with open(row_file, "wb") as handle:
        pickle.dump(row.cfg, handle)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_card_device(card))
    env.pop("GPUWM_NO_LOCAL_GPU", None)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--row-file", str(row_file), "--result-file", str(result_file),
               "--steps", str(args.steps), "--seed", str(args.seed)]
    started = time.perf_counter()
    try:
        done = subprocess.run(command, env=env, capture_output=True,
                              text=True, timeout=args.timeout, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "phase": "timeout",
                "error": f"TimeoutExpired: no verdict in {args.timeout} s",
                "detail": None, "seconds": args.timeout}
    if result_file.is_file():
        return json.loads(result_file.read_text(encoding="utf-8"))
    tail = (done.stderr or done.stdout or "")[-4000:]
    last = [line for line in tail.splitlines() if line.strip()]
    return {"status": "FAIL", "phase": "process",
            "error": f"worker exited {done.returncode}: "
                     f"{last[-1] if last else 'no output'}"[:400],
            "detail": tail, "seconds": round(time.perf_counter() - started, 2)}


def run_matrix(rows: list[Row], args) -> list[dict]:
    """Every row; settled rows recorded as they are, the rest launched."""
    work = Path(tempfile.mkdtemp(prefix="step1-smoke-"))
    free = [card for card in range(args.cards) for _ in range(args.per_card)]
    slots = len(free)
    lock = threading.Lock()
    records: list[dict] = []

    def record(row: Row, result: dict, card) -> dict:
        entry = {"row": row.row_id, "kind": row.kind, "covers": row.covers,
                 "card": card,
                 "physics": None if row.cfg is None else _summary(row.cfg),
                 **result}
        line = f"{entry['status']:4s}  {row.row_id}"
        if entry["status"] != "PASS":
            line += f"  [{entry['phase']}] {entry['error']}"
        with lock:
            print(line, flush=True)
        return entry

    def task(indexed: tuple[int, Row]) -> dict:
        index, row = indexed
        with lock:
            card = free.pop(0)
        try:
            result = _launch(row, card, args, work, index)
        finally:
            with lock:
                free.append(card)
        return record(row, result, card)

    for row in rows:
        if row.cfg is None:
            records.append(record(row, {"status": row.status,
                                        "phase": row.phase,
                                        "error": row.error,
                                        "detail": row.detail,
                                        "seconds": 0.0}, None))
    try:
        with ThreadPoolExecutor(max_workers=slots) as pool:
            records.extend(pool.map(
                task, list(enumerate(row for row in rows
                                     if row.cfg is not None))))
    finally:
        for path in work.iterdir():
            path.unlink()
        work.rmdir()
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="step1_smoke",
        description="Advance every shipped config and physics profile a few "
                    "model steps on the GPU, one process per row.")
    parser.add_argument("--cards", type=int, default=1,
                        help="cards to spread rows over (0..N-1)")
    parser.add_argument("--per-card", type=int, default=1,
                        help="rows running at once on each card")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--nxy", type=int, default=32,
                        help="horizontal extent of every row's grid")
    parser.add_argument("--nz", type=int, default=40,
                        help="levels of the profile rows' grid")
    parser.add_argument("--seed", type=int, default=20_261_008)
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="seconds one row may take, kernel compiles "
                             "included")
    parser.add_argument("--only", help="keep rows whose id contains this")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="run identical shrunk configs separately")
    parser.add_argument("--list", action="store_true",
                        help="print the rows and exit (no GPU)")
    parser.add_argument("--out", type=Path, help="JSON receipt path")
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--row-file", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker:
        return _worker(args)
    if args.cards < 1 or args.per_card < 1 or args.steps < 1:
        parser.error("--cards, --per-card and --steps must be at least 1")
    rows = enumerate_rows(nxy=args.nxy, nz=args.nz, only=args.only,
                          dedupe=not args.no_dedupe)
    if args.list:
        for row in rows:
            state = ("" if row.cfg is not None
                     else f"  {row.status} [{row.phase}] {row.error}")
            extra = (f"  (+{len(row.covers) - 1} identical)"
                     if len(row.covers) > 1 else "")
            print(f"{row.row_id}{extra}{state}")
        print(f"{len(rows)} rows", file=sys.stderr)
        return 0
    started = time.perf_counter()
    records = run_matrix(rows, args)
    failed = [entry for entry in records if entry["status"] == "FAIL"]
    skipped = [entry for entry in records if entry["status"] == "SKIP"]
    receipt = {"schema": "gpuwm.step1-smoke.v1", "commit": _commit(),
               "cards": _card_names()[:args.cards], "steps": args.steps,
               "nxy": args.nxy, "rows": len(records),
               "passed": len(records) - len(failed) - len(skipped),
               "failed": len(failed), "skipped": len(skipped),
               "wall_seconds": round(time.perf_counter() - started, 1),
               "results": records}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, default=str) + "\n",
                            encoding="utf-8")
    print(f"\n{receipt['passed']} passed, {receipt['failed']} failed, "
          f"{receipt['skipped']} not model configurations, of "
          f"{receipt['rows']} rows in {receipt['wall_seconds']} s")
    for entry in failed:
        print(f"  FAIL {entry['row']} [{entry['phase']}] {entry['error']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
