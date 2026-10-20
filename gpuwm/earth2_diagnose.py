"""``gpuwm earth2-diagnose``: run NVIDIA Earth-2 diagnostic models on a WOOF export.

The export is what ``gpuwm export --earth2`` writes: Earth2Studio lexicon
channels in the layout Earth2Studio's own IO backends write, so
Earth2Studio's ``InferenceOutputSource`` reads it unchanged.  Which models
can run, their Earth2Studio class and the channels they read are the
``consumers`` rows of ``gpuwm/authorities/earth2-models.v1.json``; adding a
diagnostic is a row there, not code here.

The models run in the interpreter ``GPUWM_EARTH2_PYTHON`` names (the one with
``earth2studio`` and the row's extra installed), or in this one when it can
import ``earth2studio``.  Nothing here installs packages, creates accounts or
logs in; a row whose weights need an account is refused by name.

Outputs, in ``--out``: one CF NetCDF per field diagnostic (``<row>.nc``,
dims time/lat/lon, the row's output variables), one CSV of tracks per
tracker row (``<row>-tracks.csv``), and ``earth2-diagnose.json`` (the rows,
inputs, timings and value ranges).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

TABLE = Path(__file__).with_name("authorities") / "earth2-models.v1.json"
PYTHON_ENV = "GPUWM_EARTH2_PYTHON"
RUNNER = Path(__file__).with_name("earth2_diagnose_runner.py")


def consumer_rows() -> dict:
    return json.loads(TABLE.read_text(encoding="utf-8")).get("consumers", {})


def resolve(names: str, accept_licence=()) -> list[str]:
    from gpuwm.earth2_licensing import require_licence

    rows = consumer_rows()
    wanted = [n.strip().lower() for n in names.split(",") if n.strip()]
    if not wanted:
        raise ValueError("--for needs one or more of: " + ", ".join(sorted(rows)))
    out = []
    for name in wanted:
        if name not in rows:
            raise ValueError(f"--for {name} is not a consumers row of {TABLE.name}; rows: "
                             + ", ".join(sorted(rows)))
        row = rows[name]
        # A row whose weights are not permissive (restricted or unresolved
        # terms) never runs by default: the exact licence must be named.
        require_licence(name, row, accept_licence=accept_licence)
        table = json.loads(TABLE.read_text(encoding="utf-8"))
        authorities = [(key, authority) for key, authority in table["models"].items()
                       if authority.get("kind") == "diagnostic"
                       and authority.get("consumer", key) == name]
        for key, authority in authorities:
            if authority.get("runtime_refusal"):
                raise ValueError(f"--for {name}: {authority['runtime_refusal']}")
            from gpuwm.earth2_licensing import require_licence
            require_licence(key, authority, accept_licence=accept_licence)
        if (row.get("weights") or {}).get("account_needed"):
            raise ValueError(f"--for {name}: its weights need an account at "
                             f"{row['weights'].get('host')}; this tool never creates one or logs in")
        if row.get("role") == "conditioning":
            raise ValueError(f"--for {name}: a conditioning consumer (a prognostic model that "
                             "takes this export as its boundary forcing), not a diagnostic; "
                             "run it through Earth2Studio with InferenceOutputSource as its "
                             "conditioning_data_source")
        out.append(name)
    return out


def earth2_python() -> str:
    named = os.environ.get(PYTHON_ENV)
    for candidate in [named] if named else [sys.executable]:
        try:
            probe = subprocess.run([candidate, "-c", "import earth2studio"],
                                   capture_output=True, text=True)
        except OSError:
            continue
        if probe.returncode == 0:
            return candidate
    raise RuntimeError(
        f"no interpreter with earth2studio: set {PYTHON_ENV} to a Python that has "
        "`pip install earth2studio==0.19.0` plus the extras the rows name")


def main(args) -> int:
    try:
        names = resolve(args.consumers, getattr(args, "accept_licence", ()) or ())
        python = earth2_python()
    except (ValueError, RuntimeError) as exc:
        print(f"gpuwm earth2-diagnose: {exc}", file=sys.stderr)
        return 2
    store = Path(args.store).resolve()
    if not store.exists():
        print(f"gpuwm earth2-diagnose: no export at {store}", file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cmd = [python, "-I", str(RUNNER), "--store", str(store), "--table", str(TABLE),
           "--for", ",".join(names), "--out", str(out), "--device", args.device]
    return subprocess.call(cmd)


def register_cli(sub) -> None:
    parser = sub.add_parser(
        "earth2-diagnose",
        help="run NVIDIA Earth-2 diagnostic models (gust, precipitation, ClimateNet, TC tracker) on a WOOF export",
        description=(
            "Run Earth2Studio diagnostic models on the Earth2Studio-shaped store `gpuwm export "
            "--earth2` wrote. The models and the channels they read are the consumers rows of "
            "earth2-models.v1.json. Needs an interpreter with earth2studio (GPUWM_EARTH2_PYTHON)."),
    )
    parser.add_argument("store", help="the export (zarr or netcdf) written by gpuwm export --earth2")
    parser.add_argument("--for", dest="consumers", required=True, metavar="ROWS",
                        help="consumers rows, comma separated, e.g. windgust-afno,precip-afno,climatenet,tc-tracker")
    parser.add_argument("--out", required=True, metavar="DIR", help="output folder")
    parser.add_argument("--device", default="cuda", help="torch device (default cuda)")
    from gpuwm.earth2_licensing import add_licence_argument
    add_licence_argument(parser)
    parser.set_defaults(func=main)
