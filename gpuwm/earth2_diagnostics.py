"""Run diagnostic authority rows through an isolated Earth2Studio interpreter."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

from gpuwm.earth2_door import _named_row, diagnostic_rows
from gpuwm.earth2_run import earth2_python


def run_diagnostic(name, source, out, device="cuda"):
    key, row = _named_row(name, diagnostic_rows(), kind="diagnostic")
    from gpuwm.earth2_licensing import require_licence, accepted_licences
    require_licence(key, row)
    if row.get("runtime_refusal"):
        raise ValueError(f"diagnostic {key}: {row['runtime_refusal']}")
    if row.get("weights", {}).get("account_needed"):
        raise ValueError(f"diagnostic {key}: weights require an account")
    source, out = Path(source).resolve(), Path(out).resolve()
    if not source.exists():
        raise ValueError(f"diagnostic {key}: no export at {source}")
    if source.is_dir() and source.suffix != ".zarr":
        candidates = sorted(source.rglob("*.zarr")) + sorted(source.rglob("*.nc"))
        if len(candidates) != 1:
            raise ValueError(f"diagnostic input {source} must identify exactly one exported dataset; found {len(candidates)}")
        source = candidates[0]
    out.mkdir(parents=True, exist_ok=True)
    request = out / "request.json"
    request.write_text(json.dumps(dict(model=key, row=row, input=str(source), output=str(out / "diagnostic.nc"), device=device, accept_licence=list(accepted_licences())), indent=2) + "\n")
    command = [earth2_python(), str(Path(__file__).with_name("earth2_diagnostic_runner.py")), str(request)]
    with (out / "runner.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"diagnostic {key} exited {result.returncode}; see {out / 'runner.log'}")
    return 0


def register_cli(subparsers):
    parser = subparsers.add_parser("earth2-diagnose", help="run diagnostic authority rows on an Earth2Studio export")
    parser.add_argument("store", nargs="?", type=Path, help="consumer export, alias of --input")
    parser.add_argument("--model", help=", ".join(sorted(diagnostic_rows())))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--for", dest="consumers", help="comma-separated consumer rows")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.set_defaults(func=main)


def main(args):
    try:
        if args.model:
            if args.consumers or args.store or args.input is None:
                raise ValueError("--model requires --input; use either --model/--input or STORE/--for")
            return run_diagnostic(args.model, args.input, args.out, args.device)
        if args.consumers:
            if args.store and args.input:
                raise ValueError("provide STORE or --input, once")
            args.store = args.store or args.input
            if args.store is None:
                raise ValueError("--for requires STORE or --input")
            from gpuwm.earth2_diagnose import main as consumer_main
            return consumer_main(args)
        raise ValueError("provide --model with --input, or STORE with --for")
    except (ValueError, RuntimeError, OSError) as error:
        print(f"earth2-diagnose: {error}", file=sys.stderr)
        return 2
