"""Ensemble bench harness (EXPERIMENTAL).

Reports wall-seconds per member and wall-seconds per simulated minute,
from the numbers the manifest already records.  It performs no
extrapolation to member counts or domain sizes that were not run: a
tiny-domain rate says nothing about a storm-scale one, and the
thirty-member realtime question is answered on real domains or not at
all.
"""

from __future__ import annotations

from pathlib import Path

from gpuwm.ensemble.manifest import (
    ENSEMBLE_MANIFEST_NAME, ENSEMBLE_MANIFEST_SCHEMA, read_manifest,
)

_HEADER = ("member", "status", "sim_s", "wall_s", "wall_s_per_sim_min",
           "final_state_sha256")

NO_CLAIM = (
    "measured on the domain and run length above only; no realtime or "
    "member-count claim is implied or supported by these numbers")


def bench_rows(manifest) -> list[dict]:
    """One row per member, with the derived rate where it is defined."""
    rows = []
    for record in manifest.get("members", ()):
        sim = record.get("sim_seconds")
        wall = record.get("wall_seconds")
        rate = None
        if isinstance(sim, (int, float)) and isinstance(wall, (int, float)) \
                and sim > 0.0:
            rate = float(wall) * 60.0 / float(sim)
        sha = record.get("final_state_sha256")
        rows.append({
            "member": int(record["index"]),
            "status": record.get("status"),
            "sim_s": None if sim is None else float(sim),
            "wall_s": None if wall is None else float(wall),
            "wall_s_per_sim_min": rate,
            "final_state_sha256": None if sha is None else str(sha),
        })
    return rows


def _cell(key: str, value) -> str:
    if value is None:
        return "-"
    if key == "final_state_sha256":
        return str(value)[:12]
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_table(rows, *, title: str | None = None) -> str:
    """A plain fixed-width table; no dependency, no colour, no guessing."""
    cells = [[_cell(key, row.get(key)) for key in _HEADER] for row in rows]
    widths = [max(len(_HEADER[i]), *(len(row[i]) for row in cells)) if cells
              else len(_HEADER[i]) for i in range(len(_HEADER))]
    lines = []
    if title:
        lines.append(title)
    lines.append("  ".join(head.ljust(width)
                           for head, width in zip(_HEADER, widths)))
    lines.append("  ".join("-" * width for width in widths))
    for row in cells:
        lines.append("  ".join(value.ljust(width)
                               for value, width in zip(row, widths)))
    rates = [row["wall_s_per_sim_min"] for row in rows
             if row["wall_s_per_sim_min"] is not None]
    walls = [row["wall_s"] for row in rows if row["wall_s"] is not None]
    if rates:
        ordered = sorted(rates)
        median = ordered[len(ordered) // 2]
        lines.append("")
        lines.append(f"members timed        : {len(rates)}")
        lines.append(f"total wall seconds   : {sum(walls):.3f}")
        lines.append(f"median wall s/sim-min: {median:.3f}")
        lines.append(f"min / max wall s/sim-min: {ordered[0]:.3f} / "
                     f"{ordered[-1]:.3f}")
    lines.append("")
    lines.append(f"note: {NO_CLAIM}")
    return "\n".join(lines)


def bench_from_manifest(ens_root: str | Path, *,
                        title: str | None = None) -> tuple[list[dict], str]:
    """Rows and rendered table for a finished (or partial) ensemble."""
    manifest = read_manifest(Path(ens_root) / ENSEMBLE_MANIFEST_NAME,
                             schema=ENSEMBLE_MANIFEST_SCHEMA)
    rows = bench_rows(manifest)
    heading = title
    if heading is None:
        heading = (f"ensemble bench: {manifest.get('n_members')} members, "
                   f"base config {manifest.get('base_config')}")
    return rows, format_table(rows, title=heading)
