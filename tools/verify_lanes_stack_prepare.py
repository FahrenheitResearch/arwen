"""Stage pinned public column inputs on G1, keeping writes inside this lane."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if LANE != Path("/work/pool-verify-lanes-stack"):
        parser.error("Stage column inputs on G1 only")
    if LANE not in args.out.resolve().parents:
        parser.error("Write the preparation receipt inside this lane")
    from gpuwm.core.thompson_contract import CLASSIC_TABLE_ASSETS, FORK_TABLE_ASSETS, validate_table_assets
    from gpuwm.tempo_assets import TEMPO_TABLE_ASSETS, ensure_tempo_tables
    from gpuwm.table_assets import stage_classic_tables
    from gpuwm.physics_compat import packaged_thompson_table_root
    from gpuwm.thompson_fork_assets import ensure_thompson_fork_tables
    roots = {}
    rows = []
    assets = {"classic": CLASSIC_TABLE_ASSETS, "fork": FORK_TABLE_ASSETS,
              "tempo": TEMPO_TABLE_ASSETS}
    names = {"classic": "thompson-tables", "fork": "thompson-fork-tables",
             "tempo": "tempo-tables"}
    for name, expected in assets.items():
        candidates = [Path(base) / names[name] for base in
                      ("/workspace/shared-prepared", "/work/shared-prepared")]
        if name == "classic":
            candidates.append(packaged_thompson_table_root())
        chosen = None
        for candidate in candidates:
            if all((candidate / item.filename).is_file() for item in expected):
                validate_table_assets(candidate, expected)
                chosen = candidate
                break
        root = chosen or LANE / "tables" / name
        roots[name] = root
        if chosen is None:
            with (LANE / "MANIFEST.txt").open("a") as stream:
                stream.write(str(root) + "\n")
            if name == "classic":
                code = stage_classic_tables(SimpleNamespace(root=str(root)))
                if code:
                    raise RuntimeError(f"Classic table acquisition exit={code}")
            elif name == "fork":
                ensure_thompson_fork_tables(root)
            else:
                ensure_tempo_tables(root)
        validate_table_assets(root, expected)
        rows.append({"set": name, "root": str(root), "reused_existing": chosen is not None,
                     "assets": [{"file": item.filename, "bytes": item.bytes,
                                 "sha256": item.sha256} for item in expected]})
        args.out.write_text(json.dumps({"tables": rows}, indent=2) + "\n")
        print(f"READY {name} {root}", flush=True)
    reference = LANE / "tempo-reference"
    if reference.exists():
        raise RuntimeError("Use the existing owned TEMPO reference; no duplicate preparation")
    with (LANE / "MANIFEST.txt").open("a") as stream:
        stream.write(str(reference) + "\n")
    command = ["bash", str(ROOT / "tools/tempo_ref/oracle/build_reference.sh"),
               str(reference), str(roots["tempo"])]
    with (LANE / "tempo-reference.log").open("w") as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    files = []
    for name in ("tempo_blob.bin", "tempo_ref_columns.bin", "columns_in.bin",
                 "columns_catalog.json", "ref_out.bin"):
        path = reference / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append({"file": name, "bytes": path.stat().st_size, "sha256": digest})
    env = {"GPUWM_THOMPSON_TABLE_ROOT": str(roots["classic"]),
           "GPUWM_THOMPSON_FORK_TABLE_ROOT": str(roots["fork"]),
           "GPUWM_TEMPO_TABLE_ROOT": str(roots["tempo"]),
           "TEMPO_TABLE_DIR": str(roots["tempo"]), "TEMPO_ORACLE_DIR": str(reference),
           "TEMPO_BLOB": str(reference / "tempo_blob.bin"),
           "TEMPO_REF_COLUMNS": str(reference / "tempo_ref_columns.bin"),
           "GPUWM_TEMPO_BRIDGE": str(ROOT / "tools/rustwx/target/release/librw_tempo.so")}
    # Paths are controlled task paths, never user shell text or credentials.
    env_path = LANE / "column-env.sh"
    env_path.write_text("".join(f"export {key}={value}\n" for key, value in env.items()))
    args.out.write_text(json.dumps({"tables": rows, "reference": files,
                                    "command": command, "environment": env}, indent=2) + "\n")
    with (LANE / "MANIFEST.txt").open("a") as stream:
        for path in (args.out, env_path, LANE / "tempo-reference.log"):
            stream.write(str(path) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
