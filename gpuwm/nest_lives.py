"""One row per storm-nest life from a run's receipts (``evidence/nest-lives.json``).

Kept apart from :mod:`gpuwm.storm_nests` (the authoring door, which reaches
the domain wizard and the planner) so the forecast runner that writes the
summary pulls in nothing but json: the standalone preparation package stages
the runner's imports and excludes the wizard.
"""

from __future__ import annotations

from pathlib import Path


def nest_lives(outdir, run_end_s=None) -> list[dict]:
    """One row per nest life from a run's receipts (``evidence/nest-lives.json``).

    Reads ``spawn_receipts.jsonl`` (births, retirements with their stop
    reason) and ``relocation_receipts.dNN.json`` (follow moves).  A life
    still running when the run ended has ``stop_reason = "run-ended"`` and
    stops at ``run_end_s`` (the run's length; else the last spawn boundary).
    """
    import json

    outdir = Path(outdir)
    lives: dict[tuple[int, int], dict] = {}
    current: dict[int, tuple[int, int]] = {}
    end_t = None
    path = outdir / "spawn_receipts.jsonl"
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    for row in rows:
        t = row.get("elapsed_seconds")
        if t is not None:
            end_t = max(float(t), end_t or 0.0)
        for life in row.get("lifecycle_receipts") or ():
            if not life.get("retire"):
                continue
            gid = int(life["grid_id"])
            key = current.pop(gid, None)
            if key is not None and key in lives:
                lives[key].update(
                    stop_s=float(life.get("t", t or 0.0)),
                    stop_reason=life.get("stop_reason") or str(life.get("decision")),
                    stop_evidence={k: life[k] for k in (
                        "max_value", "threshold", "quiet_for_s", "merged_into",
                        "overlap_fraction") if k in life})
        for born in row.get("born") or ():
            gid, episode = int(born["grid_id"]), int(born.get("episode", 1))
            spawn = born.get("spawn_receipt") or {}
            trigger = spawn.get("trigger") or spawn.get("trigger_receipt") or {}
            lives[(gid, episode)] = {
                "grid_id": gid, "episode": episode,
                "spawn_s": float(t) if t is not None else None,
                "spawn_placement": born.get("placement"),
                "trigger": {k: trigger[k] for k in (
                    "field", "trigger", "threshold", "peak", "peak_value",
                    "centroid", "max_value") if isinstance(trigger, dict) and k in trigger},
                "track": [{"t": float(t) if t is not None else None,
                           "placement": born.get("placement"), "event": "spawn"}],
                "stop_s": None, "stop_reason": None}
            current[gid] = (gid, episode)
    for receipt_path in sorted(outdir.glob("relocation_receipts.d*.json")):
        try:
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for move in payload.get("receipts") or ():
            gid = move.get("grid_id")
            t = move.get("elapsed_seconds", move.get("t"))
            # The relocation runner writes placement_to (gpuwm-nest-relocation-
            # runner.v1); the 1 h door run's summary read none of it and said
            # "0 moves" for two nests that moved (box D, 2026-10-21).
            if move.get("event", "relocated") != "relocated":
                continue
            placement = (move.get("placement_to") or move.get("new_placement")
                         or move.get("placement_after") or move.get("to"))
            if gid is None or t is None or placement is None:
                continue
            if isinstance(placement, dict):
                placement = [placement.get("i_parent_start"), placement.get("j_parent_start")]
            for life in lives.values():
                if (life["grid_id"] == int(gid) and life["spawn_s"] is not None
                        and float(t) >= life["spawn_s"]
                        and (life["stop_s"] is None or float(t) <= life["stop_s"])):
                    life["track"].append({"t": float(t), "placement": list(placement),
                                          "event": "move"})
    for life in lives.values():
        life["track"].sort(key=lambda row: (row["t"] is None, row["t"]))
        if life["stop_reason"] is None:
            life["stop_reason"] = "run-ended"
            life["stop_s"] = float(run_end_s) if run_end_s is not None else end_t
    return [lives[key] for key in sorted(lives, key=lambda k: (lives[k]["spawn_s"] or 0.0, k))]


def write_nest_lives(outdir, run_end_s=None) -> Path | None:
    """Write ``evidence/nest-lives.json`` beside a run, or nothing if no nest lived."""
    import json

    lives = nest_lives(outdir, run_end_s=run_end_s)
    if not lives:
        return None
    path = Path(outdir) / "evidence" / "nest-lives.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "gpuwm-nest-lives.v1", "lives": lives},
                               indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    for life in lives:
        print(f"storm nest d{life['grid_id']:02d} episode {life['episode']}: born "
              f"{life['spawn_s']} s at {life['spawn_placement']}, "
              f"{sum(1 for row in life['track'] if row['event'] == 'move')} move(s), "
              f"stopped {life['stop_s']} s ({life['stop_reason']})", flush=True)
    return path


__all__ = ["nest_lives", "write_nest_lives"]
