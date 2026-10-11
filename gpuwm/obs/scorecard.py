"""One product door for the Rust observation scorecard. No Python math path."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import json
import tempfile
import time
import re

from gpuwm import bridges


def _acquire(manifest: dict, root: Path, scratch: Path, timeout: int, *, acquire=True, data_root=None) -> dict:
    """Schedule the existing public observation doors using metadata rows."""
    from gpuwm.rustwx import verification_inventory
    from gpuwm.verification_visuals import fetch_stations, fetch_radar
    prepared = json.loads(json.dumps(manifest))
    def resolve(value, parent):
        if "${DATA_ROOT}" in value:
            if data_root is None:
                raise ValueError("this recipe needs --data-root to locate its recorded inputs")
            value = value.replace("${DATA_ROOT}", str(data_root.resolve()))
        return str((parent / value).resolve())
    failures = []
    for case in prepared.get("cases", ()):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case["id"]) or case["id"] in (".", ".."):
            raise ValueError("case id must be one safe folder name; acquisition cannot write outside its private scratch folder")
        case["receipts"] = [resolve(p, root) for p in case.get("receipts", ())]
        if case.get("pairs"):
            case["pairs"]["path"] = resolve(case["pairs"]["path"], root)
        paths = []
        for index, source in enumerate(case.get("requests", ())):
            original = (root / source).resolve()
            request = json.loads(original.read_text(encoding="utf-8-sig"))
            for key in ("stations_path", "station_table_path"):
                if request.get(key):
                    request[key] = resolve(request[key], original.parent)
            for arm in request.get("arms", ()):
                for key in ("path", "grid_path", "previous_path", "points_path"):
                    if arm.get(key):
                        arm[key] = resolve(arm[key], original.parent)
            for radar in request.get("radar", ()):
                for key in ("path", "grid_path"):
                    radar[key] = resolve(radar[key], original.parent)
            folder = scratch / case["id"] / str(index)
            inventory = verification_inventory(request, workdir=folder / "inventory", timeout=timeout) if acquire else {}
            bbox = inventory.get("bbox")
            for row in prepared.get("acquisition", ()) if acquire else ():
                try:
                    if row["kind"] == "radar" and (not re.fullmatch(r"[A-Za-z0-9_.-]+", row["quantity"]) or row["quantity"] in (".", "..")):
                        raise ValueError("radar quantity must be one safe folder name; acquisition cannot leave its private scratch folder")
                    if (row["kind"] == "stations" and row["source"] != "asos") or (row["kind"] == "radar" and row["source"] != "mrms"):
                        raise ValueError("existing acquisition doors serve ASOS stations and MRMS grids; another source needs its decoded native input")
                    if row["kind"] == "stations" and not request.get("stations_path"):
                        request["station_source"] = row["source"]
                        request["stations_path"] = str(fetch_stations(request, bbox, folder / "stations", timeout=timeout))
                    elif row["kind"] == "radar" and not any(r.get("quantity") == row["quantity"] for r in request.get("radar", ())):
                        request["radar_source"] = row["source"]
                        # An existing output directory makes old and new MRMS
                        # doors choose the invocation-owned object cache.
                        (folder / row["quantity"] / "objects").mkdir(parents=True, exist_ok=True)
                        radar = fetch_radar(request, {"product": row["product"], "key": row["quantity"]},
                                            bbox, folder / row["quantity"], timeout=timeout)
                        request.setdefault("radar", []).append(radar)
                    elif row["kind"] not in ("stations", "radar"):
                        raise ValueError(f"unsupported acquisition kind {row['kind']}; supply a decoded native observation path")
                except (OSError, ValueError, RuntimeError, LookupError, subprocess.TimeoutExpired) as error:
                    failure = f"{case['id']}/{index}/{row['source']}/{row.get('quantity', 'stations')}: {type(error).__name__}: {error}"
                    failures.append(failure)
                    # Preserve other available sources and let required metric
                    # coverage return exit 3 with a concrete missing row.
            path = folder / "request.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(request, indent=2), encoding="utf-8")
            paths.append(str(path))
        case["requests"] = paths
    prepared.setdefault("notes", []).extend(failures)
    return prepared


def _clean_owned(root: Path, report: Path) -> None:
    """Remove only this invocation's private acquired inputs, logging sizes."""
    rows = []
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_symlink() or path.is_file():
            rows.append({"path": str(path), "bytes": path.lstat().st_size})
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    root.rmdir()
    report.write_text(json.dumps({"deleted_files": rows, "bytes": sum(r["bytes"] for r in rows)}, indent=2), encoding="utf-8")

def find_binary() -> Path | None:
    """The scorecard binary this door would run, or None; no contract check."""
    from gpuwm.rustwx import crate_dir
    name = bridges.executable_name("rw_scorecard")
    override = os.environ.get("GPUWM_RW_SCORECARD")
    if override:
        found = bridges.find_artifact("GPUWM_RW_SCORECARD", name)
    else:
        found = next((p for p in (crate_dir() / "target" / "release" / name,
                                 crate_dir() / "target" / "debug" / name)
                      if p.is_file()), None)
        found = found or bridges.find_artifact("GPUWM_RW_SCORECARD", name)
    return found


def binary() -> Path:
    found = find_binary()
    if found is None:
        raise RuntimeError("the Rust observation scorecard is missing; build tools/rustwx "
                           "with cargo build --release -p rw-wrfbatch --bin rw_scorecard "
                           "--bin rw_verify, or stage it with gpuwm fetch-bridges")
    found = bridges.accept_resolved(found.resolve())
    ok, evidence = bridges.bridge_abi_matches("rw_scorecard", found)
    if not ok:
        raise RuntimeError(f"{found}: {evidence}; rebuild rw_scorecard so its manifest "
                           "and case bootstrap match this product")
    return found


def _command_main_locked(args) -> int:
    scratch = None
    try:
        from gpuwm.rustwx import verification_binary
        command = [str(binary()), "--manifest", str(args.manifest), "--out", str(args.out)]
        # CSV replay does not need the field reader. Resolve it only for native requests.
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        if args.acquire or args.data_root:
            args.out.mkdir(parents=True, exist_ok=True)
            scratch = Path(tempfile.mkdtemp(prefix=".scorecard-inputs-", dir=args.out)).resolve()
            prepared = _acquire(manifest, args.manifest.resolve().parent, scratch, args.timeout,
                                acquire=args.acquire, data_root=args.data_root)
            staged = scratch / "campaign.json"
            staged.write_text(json.dumps(prepared, indent=2), encoding="utf-8")
            command[command.index("--manifest") + 1] = str(staged)
        if any(case.get("requests") for case in manifest.get("cases", ())):
            command += ["--verify-binary", str(verification_binary())]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        return subprocess.run(command, env=env, check=False).returncode
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(f"obs scorecard: {error}")
        return 2
    finally:
        if scratch is not None:
            _clean_owned(scratch, args.out / f"deleted-acquisition-{time.time_ns()}.json")


def command_main(args) -> int:
    from gpuwm.verification_visuals import _run_lock
    try:
        with _run_lock(args.out, name=".scorecard.lock"):
            return _command_main_locked(args)
    except BlockingIOError:
        print("obs scorecard: this output is in use by another scorecard; writing now would mix receipts from two model comparisons. Retry when it finishes.")
        return 2
    except OSError as error:
        print(f"obs scorecard: cannot lock output {args.out}: {error}")
        return 2


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser("scorecard", help="score recorded cases and compare arms with a case bootstrap")
    parser.add_argument("manifest", type=Path, help="versioned cases, arms, observation sources and metrics")
    parser.add_argument("--out", type=Path, required=True, help="output folder for the JSON receipt, the case and campaign tables and the native sheets")
    parser.add_argument("--acquire", action="store_true", help="fetch and decode public sources named by acquisition metadata; delete these private inputs after scoring")
    parser.add_argument("--data-root", type=Path, help="recorded input root substituted for ${DATA_ROOT} in request paths")
    parser.add_argument("--timeout", type=int, default=180, help="seconds allowed for each existing observation door")
    parser.set_defaults(func=command_main)
