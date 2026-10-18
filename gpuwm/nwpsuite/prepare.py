"""The suite's CPU prepare door: fetch and prepare one cycle, then publish its authority.

A derived-config forecast row runs on an existing prepared bundle, so the
cards it holds never wait on a download or a preparation.  This door is the
zero-card stage the controller launches ahead of that forecast for each cycle.
It runs the ordinary ``gpuwm go CONFIG --prepare-only`` and, only when that
door reports PREPARED, writes the ``gpuwm.suite.prepared.v1`` receipt the
forecast row checks before queue admission: the prepared root, the published
experiment configuration and WPS namelist the forecast binds, and their hashes.

Exit codes are the door's own (0 prepared, 75 source behind, 2 refused, 1
failed), so the controller's source-late ladder and refusal handling apply.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

from gpuwm.cycle.ledger import write_receipt_atomically


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepared_line(stdout):
    """The door's last PREPARED JSON line, or None."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                document = json.loads(line)
            except ValueError:
                continue
            if document.get("status") == "PREPARED":
                return document
    return None


def publish(document, *, receipt, suite_sha256, row, cycle, source, config, go_forecast=False):
    """Write the suite's prepared authority from the door's PREPARED report."""
    prepared_root = Path(document["prepared_root"]).resolve()
    experiment = document.get("experiment_config")
    namelist = document.get("wps_namelist")
    if not experiment:
        # Breakage it prevents: the forecast would bind a configuration the
        # bundle's source manifest does not name, and the runner refuses it
        # only after the cards are held.
        raise ValueError("suite_prepare_authority: the preparation reported no published experiment configuration")
    if not namelist and go_forecast:
        # Breakage it prevents: the go forecast door names the bundle's WPS
        # authority; without it the forecast would be refused holding cards.
        raise ValueError("suite_prepare_authority: the preparation published no WPS namelist for the go forecast door")
    files = [Path(experiment).resolve()]
    if namelist:
        files.append(Path(namelist).resolve())
    proof = prepared_root / "proof.json"
    if proof.is_file():
        files.append(proof)
    raw = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    forcing = str(raw.get("fetch", {}).get("source", "")).strip().lower() or None
    write_receipt_atomically(Path(receipt), dict(
        schema="gpuwm.suite.prepared.v1", suite_sha256=suite_sha256, row=row, source=source,
        # gpuwm go forces initial and lateral boundaries from its one [fetch] source.
        lbc_source=forcing, cycle=cycle, prepared_root=str(prepared_root),
        experiment_config=str(files[0]), wps_namelist=str(files[1]) if namelist else None,
        files=[dict(path=str(p), sha256=_sha256(p)) for p in files]))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m gpuwm.nwpsuite.prepare")
    parser.add_argument("--config", required=True, help="the cycle's derived experiment configuration")
    parser.add_argument("--outdir", required=True, help="this attempt's preparation folder")
    parser.add_argument("--receipt", required=True, help="the suite prepared receipt to publish")
    parser.add_argument("--suite-sha256", required=True)
    parser.add_argument("--row", required=True)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--go-forecast", action="store_true", help="the forecast is the go door, which binds the WPS namelist")
    args = parser.parse_args(argv)
    command = [sys.executable, "-m", "gpuwm.cli", "go", args.config, "--prepare-only",
               "--outdir", args.outdir, "--run-stamp", "off"]
    print("suite prepare: " + " ".join(command), flush=True)
    # Relay the door's stage lines as they come, so a live suite shows them.
    said = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as door:
        for line in door.stdout:
            said.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
    if door.returncode:
        return door.returncode
    document = prepared_line("".join(said))
    if document is None:
        print("suite prepare: gpuwm go exited 0 without a PREPARED report", file=sys.stderr)
        return 1
    try:
        publish(document, receipt=args.receipt, suite_sha256=args.suite_sha256, row=args.row,
                cycle=args.cycle, source=args.source, config=args.config,
                go_forecast=args.go_forecast)
    except (OSError, ValueError) as error:
        print(f"suite prepare: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "PUBLISHED", "receipt": str(Path(args.receipt).resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
