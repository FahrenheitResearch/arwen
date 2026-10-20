#!/usr/bin/env python3
"""Fetch one authority row's public Hugging Face weights without a token."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="fcn3", help="authority ID (default: fcn3)")
    parser.add_argument("--out", type=Path, required=True, help="new weights-cache directory")
    args = parser.parse_args()
    table = json.loads((ROOT / "gpuwm/authorities/earth2-models.v1.json").read_text())
    rows = table["models"]
    if args.model not in rows:
        parser.error("unknown model; choose: " + ", ".join(sorted(rows)))
    weights = rows[args.model].get("weights") or {}
    if weights.get("gated") or weights.get("account_needed"):
        parser.error(f"{args.model}: gated or account-required weights are refused")
    if weights.get("gated") is not False or weights.get("account_needed") is not False:
        parser.error(f"{args.model}: public account-free access is not declared")
    if weights.get("host") != "huggingface" or "/" not in weights.get("repository", ""):
        parser.error(f"{args.model}: no Hugging Face repository is declared")
    if args.out.exists():
        parser.error("--out must be a new directory, so this receipt measures its own download")
    from huggingface_hub import snapshot_download

    started = time.monotonic()
    args.out.mkdir(parents=True)
    snapshot = Path(snapshot_download(
        repo_id=weights["repository"], revision=weights.get("revision"),
        cache_dir=str(args.out), token=False,
    ))
    files = [
        {"path": str(p.relative_to(snapshot)), "bytes": p.stat().st_size}
        for p in sorted(snapshot.rglob("*")) if p.is_file()
    ]
    receipt = {
        "model": args.model, "repository": weights["repository"],
        "requested_revision": weights.get("revision"), "resolved_revision": snapshot.name,
        "token": False, "wall_seconds": round(time.monotonic() - started, 3),
        "files": files, "bytes": sum(p["bytes"] for p in files),
    }
    (args.out / "hf-fetch-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
