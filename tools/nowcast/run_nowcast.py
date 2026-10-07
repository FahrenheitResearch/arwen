#!/usr/bin/env python3
"""Run a 2D reflectivity nowcast and write ``gpuwm-obs.nowcast-frames.v1``.

    <venv-e2s>/bin/python tools/nowcast/run_nowcast.py --source stormscope-3km-10min \\
        --issue 2026-10-01T18:00Z --minutes 120 --members 8 --seed 1 \\
        --device cuda:0 --out ROOT

This is orchestration and CUDA driver code over a third-party model
package (earth2studio).  It runs in its own virtual environment
(``tools/nowcast/setup_venv.sh``); the engine never imports it.  The
package fetches and decodes its own satellite, radar and lightning inputs;
this runner adds no decoder, no regrid and no reshape of WOOF data.  Its
output is a raw dump of the model's output tensor plus a receipt.

Order of work, so that refusals cost no GPU time:

1. the source row, the weights pin, the GOES-East satellite and the slot
   plan are checked (``--plan-only`` stops here and prints the plan);
2. every history slot is fetched on the CPU with each data source tapped,
   so every object read is logged with its stamp, and the ledger is
   checked (causality, duplicates, off-nominal slots, satellite);
3. the pinned weights are loaded on the CPU and checked against the row;
4. only then does the card see work: one 10-minute frame per step, the
   ensemble being the batch dimension.

Exit codes: 0 READY, 3 refused (``nowcast.json`` says REFUSED and why),
anything else failed (``nowcast.json`` says FAILED).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.nowcast import frames as F  # noqa: E402

DEFAULT_SOURCES = Path(__file__).resolve().with_name("sources.toml")
EXIT_REFUSED = 3


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True, help="row id in sources.toml")
    ap.add_argument("--issue", required=True, help="issue time, e.g. 2026-10-01T18:00Z")
    ap.add_argument("--minutes", type=int, default=120, help="nowcast length (a multiple of the step)")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--batch-members", type=int, default=None,
                    help="members per rollout when all of them do not fit on one card")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, help="frames root")
    ap.add_argument("--sources", default=str(DEFAULT_SOURCES))
    ap.add_argument("--revision", default=None,
                    help="weights revision; anything but the row's pin is refused")
    ap.add_argument("--satellite", default=None,
                    help="GOES satellite; anything but GOES-East at the issue time is refused")
    ap.add_argument("--plan-only", action="store_true",
                    help="check the row, pin, satellite and slot plan, print it, and stop")
    return ap.parse_args(argv)


def make_plan(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    """Everything decidable without the model package.  Raises refusals."""
    issue = F.parse_utc(args.issue)
    step = int(row["step_minutes"])
    revision = F.check_revision(args.revision, row["revision"])
    satellite = F.check_satellite(args.satellite, issue, row["goes_east"])
    if args.minutes <= 0 or args.minutes % step:
        raise F.NowcastRefusal(
            f"--minutes {args.minutes} is not a positive multiple of the {step} min step",
            "the last window would end between frames and be read as unforced time",
        )
    if args.members < 1:
        raise ValueError("--members must be at least 1")
    batch = args.batch_members or args.members
    if not 1 <= batch <= args.members:
        raise ValueError("--batch-members must be between 1 and --members")
    F.frame_dirname(issue)  # whole-minute check
    slots = F.history_slots(issue, step, int(row["history_frames"]))
    groups: dict[str, float] = {}
    for s in row["streams"]:
        off = float(s.get("request_offset_seconds", 0))
        if groups.setdefault(s["fetch"], off) != off:
            raise ValueError(f"streams of fetch group {s['fetch']!r} disagree on request_offset_seconds")
    request_times = {group: [n + timedelta(seconds=off) for n in slots]
                     for group, off in groups.items()}
    F.check_request_window(issue, [t for ts in request_times.values() for t in ts], satellite,
                           row["goes_east"], row["archive_from"], int(row["issue_align_minutes"]))
    requests = {group: [F.fmt_utc(t) for t in ts] for group, ts in request_times.items()}
    return {
        "source": row["id"],
        "issue": issue,
        "revision": revision,
        "satellite": satellite,
        "steps": args.minutes // step,
        "step_minutes": step,
        "members": args.members,
        "batch_members": batch,
        "slots": slots,
        "request_offsets": groups,
        "requests": requests,
    }


def plan_json(plan: dict[str, Any]) -> dict[str, Any]:
    out = dict(plan)
    out["issue"] = F.fmt_utc(plan["issue"])
    out["slots"] = [F.fmt_utc(s) for s in plan["slots"]]
    out["valid_times"] = [F.fmt_utc(plan["issue"] + timedelta(minutes=plan["step_minutes"] * k))
                          for k in range(1, plan["steps"] + 1)]
    return out


def source_block(row: dict[str, Any], revision: str, code: dict[str, str]) -> dict[str, Any]:
    return {"kind": row["kind"], "id": row["id"], "driver": row["driver"],
            "package": row["package"], "revision": revision, "variant": row["variant"],
            "code": code}


def _code_versions() -> dict[str, str]:
    from importlib import metadata

    out = {"python": platform.python_version()}
    for dist in ("earth2studio", "torch", "nvidia-physicsnemo", "natten", "numpy"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = "absent"
    return out


def _naive(dt: datetime):
    import numpy as np

    return np.datetime64(dt.astimezone(timezone.utc).replace(tzinfo=None), "s")


def _class(path: str):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def run(args: argparse.Namespace, row: dict[str, Any], plan: dict[str, Any],
        out: Path) -> dict[str, Any]:
    """Steps 2 to 4: fetch with a ledger, check, load, roll out, write."""
    import numpy as np
    import torch
    from earth2studio.data import GOES, MRMS, GOESGLMGrid, fetch_data
    from earth2studio.models.auto import Package

    issue: datetime = plan["issue"]
    slots: list[datetime] = plan["slots"]
    step = plan["step_minutes"]
    sat = plan["satellite"]
    device = torch.device(args.device)
    timing: dict[str, Any] = {}
    code = _code_versions()

    # 2. Fetch every history slot with every read on the ledger
    ledger = F.InputLedger(row["streams"])
    goes_src = GOES(satellite=sat, scan_mode="C")
    F.tap_reads(goes_src, "_fetch_remote_file", ledger)
    mrms_src = MRMS()
    F.tap_reads(mrms_src, "_download_and_decompress_async", ledger)
    # The source ranks files nearest first, after the slot as well as before;
    # keep only those at or before the slot being fetched.
    nearest_first = mrms_src._resolve_s3_time_candidates

    async def causal_nearest(*a: Any, **k: Any) -> list:
        return F.causal_candidates(await nearest_first(*a, **k), ledger.current_slot)

    mrms_src._resolve_s3_time_candidates = causal_nearest
    # The lightning platform is pinned to the ABI satellite, so the two can
    # never disagree; the period table already refused the handover gap.
    glm_src = GOESGLMGrid(satellite=f"G{sat[-2:]}")
    F.tap_reads(glm_src._events, "_fetch_remote_file", ledger,
                local_of=lambda obj, a, k, r: obj._cache_path(a[0]))

    input_times = np.arange(-(len(slots) - 1), 1) * np.timedelta64(step, "m")
    zero = np.array([np.timedelta64(0, "m")])

    def fetch_slots(source, variables, group):
        offset = timedelta(seconds=plan["request_offsets"][group])
        parts, coords = [], None
        for nominal in slots:
            with ledger.slot(nominal):
                x, c = fetch_data(source, time=np.array([_naive(nominal + offset)]),
                                  variable=np.array(variables), lead_time=zero, device="cpu")
            parts.append(x)
            coords = coords if coords is not None else c
        x = torch.cat(parts, dim=1)
        coords = coords.copy()
        coords["time"] = np.array([_naive(issue)])
        coords["lead_time"] = input_times
        return x, coords

    t0 = time.perf_counter()
    x_goes, c_goes = fetch_slots(goes_src, row["satellite_variables"], "satellite")
    x_radar, c_radar = fetch_slots(mrms_src, row["radar_variables"], "radar")
    x_glm, c_glm = fetch_slots(glm_src, row["lightning_variables"], "lightning")
    timing["fetch_seconds"] = time.perf_counter() - t0

    resolution = F.check_history(ledger, issue, slots, sat, float(row["history_tolerance_seconds"]))
    ledger_json = ledger.to_json(issue, resolution)
    ledger_json["source"] = row["id"]
    inputs_sha = F.write_json_atomic(out / "inputs.json", ledger_json)
    if not ledger_json["causal"]:
        raise F.NowcastRefusal("the input ledger is not causal", "a hindcast would be scored as a forecast")
    # [time=1, slot, variable, ...]: a field with no finite value is a fetch
    # that failed quietly; refuse it here rather than on the card.
    for x, names, what in ((x_goes, row["satellite_variables"], "satellite"),
                           (x_radar, row["radar_variables"], "radar")):
        F.check_history_fields(torch.isfinite(x[0]).flatten(2).any(-1).tolist(), names, slots, what)

    # 3. Load the pinned weights (on the CPU first), then check the models
    #    against the row before anything touches the card
    t1 = time.perf_counter()
    goes_cls = _class(row["satellite_model"])
    mrms_cls = _class(row["radar_model"])
    package = Package(f"{row['package']}@{plan['revision']}",
                      cache_options={"cache_storage": Package.default_cache("stormscope")})
    F.check_revision(str(package.root), row["revision"])
    goes_m = goes_cls.load_model(package=package, model_name=row["variant"],
                                 conditioning_data_source=None,
                                 amp=bool(row["amp"]), compile=bool(row["compile"]))
    # Observed lightning comes only from the tapped fetch of the starting
    # frames above; with no lightning source the coupled rollout carries the
    # model's own predicted lightning after that.
    mrms_m = mrms_cls.load_model(package=package, model_name=row["variant"],
                                 conditioning_data_source=None, glm_data_source=None,
                                 amp=bool(row["amp"]), compile=bool(row["compile"]))
    timing["load_seconds"] = time.perf_counter() - t1
    for m in (goes_m, mrms_m):
        m.sampler_args = dict(row["sampler"])
        if not np.array_equal(np.asarray(m.input_times), input_times):
            raise F.NowcastRefusal(
                f"{type(m).__name__} input times {m.input_times} are not the row's "
                f"{len(slots)} frames at {step} min",
                "the history frames would be fed to the model under the wrong lead times",
            )
    for got, want, what in (
        (list(goes_m.input_coords()["variable"]), list(row["satellite_variables"]), "satellite"),
        (list(mrms_m.variables), list(row["radar_variables"]) + list(row["lightning_variables"]), "radar"),
    ):
        if got != want:
            raise F.NowcastRefusal(
                f"{what} model variables {got} are not the row's {want}",
                "state channels would be stacked in the wrong order",
            )

    # The lattice comes from the model's own coordinates, still on the CPU,
    # so a lattice refusal costs no card time.
    lattice = F.lattice_from_latlon(mrms_m.latitudes.detach().cpu().numpy(),
                                    mrms_m.longitudes.detach().cpu().numpy(), row["lattice"])

    # 4. Take the card (its engine lock, so no other run shares it), build
    #    interpolators, assemble the starting state, roll out
    with contextlib.ExitStack() as card:
        if device.type == "cuda":
            gpu_uuid = F.nvidia_smi_uuid(torch.cuda.get_device_properties(device).uuid)
            card.enter_context(F.card_lock(gpu_uuid, f"nowcast-{row['id']}-{F.fmt_utc(issue)}"))
            timing["gpu_uuid"] = gpu_uuid
        return _roll_out(args, row, plan, out, SimpleNamespace(
            device=device, timing=timing, code=code, ledger_json=ledger_json,
            inputs_sha=inputs_sha, goes_m=goes_m, mrms_m=mrms_m, lattice=lattice, sat=sat,
            x_goes=x_goes, c_goes=c_goes, x_radar=x_radar, c_radar=c_radar,
            x_glm=x_glm, c_glm=c_glm, t0=t0))


def _roll_out(args: argparse.Namespace, row: dict[str, Any], plan: dict[str, Any],
              out: Path, p: SimpleNamespace) -> dict[str, Any]:
    """Step 4, on the card the caller holds the lock for."""
    import numpy as np
    import torch
    from earth2studio.data import GOES

    issue: datetime = plan["issue"]
    slots: list[datetime] = plan["slots"]
    step = plan["step_minutes"]
    device, timing, sat, lattice = p.device, p.timing, p.sat, p.lattice
    goes_m, mrms_m = p.goes_m, p.mrms_m
    x_goes, c_goes, x_radar, c_radar, x_glm, c_glm = (
        p.x_goes, p.c_goes, p.x_radar, p.c_radar, p.x_glm, p.c_glm)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    goes_m = goes_m.to(device).eval()
    mrms_m = mrms_m.to(device).eval()
    goes_lat, goes_lon = GOES.grid(satellite=sat, scan_mode="C")
    goes_m.build_input_interpolator(goes_lat, goes_lon)
    mrms_m.build_input_interpolator(c_radar["lat"], c_radar["lon"])
    mrms_m.build_conditioning_interpolator(goes_lat, goes_lon)
    mrms_m.build_glm_interpolator(c_glm["lat"], c_glm["lon"])
    x_radar = mrms_m.input_interp(x_radar.to(device))
    # Cells where any history slot of the observed composite had no radar
    # coverage: the model fills them with clear air, so its frames there are
    # written as NaN (no coverage), never as a clear-air nowcast.
    primary = list(row["radar_variables"]).index(row["primary_variable"])
    keep, unobserved = F.unobserved_cells(x_radar[0, :, primary], mrms_m.valid_mask,
                                          float(row["no_coverage_ceiling_dbz"]))
    x_glm = mrms_m.interpolate_glm(x_glm.to(device))
    x_mrms = torch.cat([x_radar, x_glm], dim=2).to(dtype=torch.float32)
    c_mrms = c_radar.copy()
    c_mrms["variable"] = np.array(mrms_m.variables)
    del c_mrms["lat"], c_mrms["lon"]
    c_mrms["y"] = mrms_m.y
    c_mrms["x"] = mrms_m.x
    x_goes = x_goes.to(device=device, dtype=torch.float32)

    ny, nx = int(lattice["ny"]), int(lattice["nx"])
    outputs = list(row["output_variables"])
    var_index = {v: list(mrms_m.variables).index(v) for v in outputs}
    writer = F.FrameWriter(out, issue, step, plan["members"], ny, nx, outputs, row["primary_variable"])

    step_seconds: list[list[float]] = []
    nan = torch.tensor(float("nan"), device=device)
    for b0 in range(0, plan["members"], plan["batch_members"]):
        nb = min(plan["batch_members"], plan["members"] - b0)
        torch.manual_seed(args.seed + b0)
        y = x_goes.unsqueeze(0).repeat(nb, 1, 1, 1, 1, 1)
        yc = c_goes.copy()
        yc["batch"] = np.arange(nb)
        yc.move_to_end("batch", last=False)
        ym = x_mrms.unsqueeze(0).repeat(nb, 1, 1, 1, 1, 1)
        ymc = c_mrms.copy()
        ymc["batch"] = np.arange(nb)
        ymc.move_to_end("batch", last=False)
        this_batch: list[float] = []
        for k in range(1, plan["steps"] + 1):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ts = time.perf_counter()
            g_pred, g_pc = goes_m(y, yc)
            r_pred, r_pc = mrms_m.call_with_conditioning(ym, ymc, conditioning=y, conditioning_coords=yc)
            y, yc = goes_m.next_input(g_pred, g_pc, y, yc)
            ym, ymc = mrms_m.next_input(r_pred, r_pc, ym, ymc)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            this_batch.append(time.perf_counter() - ts)
            lead = k * step
            got = np.asarray(r_pc["lead_time"])[-1]
            if got != np.timedelta64(lead, "m"):
                raise F.NowcastRefusal(
                    f"step {k} came back with lead {got}, expected {lead} min",
                    "a frame would be labelled with the wrong valid time",
                )
            masked = torch.where(keep, r_pred, nan)[:, 0, 0]
            for v in outputs:
                writer.write(lead, v, b0, masked[:, var_index[v]].float().cpu().numpy())
        step_seconds.append(this_batch)
        print(f"members {b0}..{b0 + nb - 1}: {plan['steps']} steps, "
              f"{sum(this_batch):.1f} s", flush=True)

    frames = writer.finish()
    flat = [s for batch in step_seconds for s in batch]
    steady = [s for batch in step_seconds for s in batch[1:]]
    timing.update({
        "step_seconds": step_seconds,
        "seconds_per_step_first": flat[0],
        "seconds_per_step_steady": (sum(steady) / len(steady)) if steady else None,
        "first_step_includes_compile": bool(row["compile"]),
        "total_seconds": time.perf_counter() - p.t0,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "cuda": torch.version.cuda,
    })
    peak = None
    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_reserved(device))
        timing["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return F.build_receipt(
        status=F.STATUS_READY,
        source=source_block(row, plan["revision"], p.code),
        issue=issue, ledger_json=p.ledger_json, members=plan["members"],
        variable=row["primary_variable"], units=row["units"], lattice=lattice, frames=frames,
        sampler=dict(row["sampler"]), seed=args.seed, timing=timing, peak_device_bytes=peak,
        inputs_sha256=p.inputs_sha,
        history={
            "policy": "newest object complete at or before each slot",
            "slots": [F.fmt_utc(s) for s in slots],
            "no_coverage": {
                "rule": f"NaN wherever the model's valid mask is false or any slot of the "
                        f"observed {row['primary_variable']} is NaN or <= "
                        f"{float(row['no_coverage_ceiling_dbz'])} dBZ",
                "unobserved_valid_cells_per_slot": unobserved,
                "valid_cells": int(mrms_m.valid_mask.sum()),
                "written_cells": int(keep.sum()),
            },
            "lightning": "observed in the starting frames only, then the model's own",
            "satellite": sat,
            "seed_rule": "torch.manual_seed(seed + index of the first member in the batch)",
            "batch_members": plan["batch_members"],
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    row = F.source_row(F.load_sources(args.sources), args.source)
    out = Path(args.out)
    receipt_path = out / "nowcast.json"
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text(encoding="utf-8")).get("status")
        if prior == F.STATUS_READY:
            print(f"refused: {receipt_path} is already READY. Breakage prevented: "
                  "frames would change under a receipt something may already have read.",
                  file=sys.stderr)
            return EXIT_REFUSED
    issue_text = args.issue
    plan: dict[str, Any] | None = None
    try:
        plan = make_plan(args, row)
        if args.plan_only:
            print(json.dumps(plan_json(plan), indent=2))
            return 0
        out.mkdir(parents=True, exist_ok=True)
        F.write_receipt(out, F.build_receipt(
            status=F.STATUS_RUNNING, source=source_block(row, plan["revision"], {}),
            issue=plan["issue"], ledger_json=None, members=args.members,
            variable=row["primary_variable"], units=row["units"], lattice=None, frames=[]))
        receipt = run(args, row, plan, out)
        F.write_receipt(out, receipt)
        print(f"READY {receipt_path}")
        return 0
    except F.NowcastRefusal as exc:
        print(str(exc), file=sys.stderr)
        if not args.plan_only:
            out.mkdir(parents=True, exist_ok=True)
            F.write_receipt(out, F.build_receipt(
                status=F.STATUS_REFUSED, source=source_block(row, row["revision"], {}),
                issue=plan["issue"] if plan else F.parse_utc(issue_text), ledger_json=None,
                members=args.members, variable=row["primary_variable"], units=row["units"],
                lattice=None, frames=[], refusal=str(exc)))
        return EXIT_REFUSED
    except Exception:
        if not args.plan_only and plan is not None:
            F.write_receipt(out, F.build_receipt(
                status=F.STATUS_FAILED, source=source_block(row, row["revision"], {}),
                issue=plan["issue"], ledger_json=None, members=args.members,
                variable=row["primary_variable"], units=row["units"], lattice=None,
                frames=[], refusal=traceback.format_exc(limit=5)))
        raise


if __name__ == "__main__":
    sys.exit(main())
