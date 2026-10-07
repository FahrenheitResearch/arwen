"""Hash native reference inputs, compilation flags and completed products."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def file_record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def collect(source: Path, official: Path, build: Path, run: Path) -> dict:
    suffixes = {".F", ".F90", ".f", ".f90", ".c", ".h", ".cmake", ".inc"}
    sources = {}
    divergences = {}
    for folder in ("phys", "dyn_em", "Registry", "main", "frame", "share", "external", "cmake", "tools"):
        for path in sorted((source / folder).rglob("*")):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path.suffix not in suffixes and folder != "Registry" and path.name != "CMakeLists.txt":
                continue
            relative = path.relative_to(source).as_posix()
            sources[relative] = file_record(path)
            original = official / relative
            if not original.is_file():
                divergences[relative] = {"kind": "added", "compiled": sources[relative]}
            elif digest(original) != sources[relative]["sha256"]:
                divergences[relative] = {"kind": "modified", "official": file_record(original), "compiled": sources[relative]}
    source_manifest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    archive_inputs = {path.name: file_record(path) for name in ("wrf471-official.tar.gz", "mynn-pinned.tar.gz", "noahmp-pinned.tar.gz")
                      if (path := source.parent / name).is_file()}
    external_manifests = {name: {**file_record(source / name), "contents": (source / name).read_text()}
                          for name in (".gitmodules", "arch/Externals.cfg") if (source / name).is_file()}
    executables = {name: file_record(build / "main" / name) for name in ("ideal", "wrf")}
    flags = {str(path.relative_to(build)): {**file_record(path), "contents": path.read_text()}
             for path in sorted(build.rglob("flags.make"))}
    inputs = {name: file_record(run / name) for name in ("namelist.input", "namelist.fire", "input_sounding", "wrfinput_d01")}
    case_divergences = {name: {"reference": file_record(official / "test" / "em_fire" / name), "run": inputs[name]}
                        for name in ("namelist.input", "namelist.fire", "input_sounding")
                        if digest(official / "test" / "em_fire" / name) != inputs[name]["sha256"]}
    markers = {name: (run / name).read_text().strip() for name in ("ideal.exit", "wrf.exit", "run.done", "started.txt", "finished.txt")
               if (run / name).is_file()}
    terminal = "run.done" in markers
    # Only a terminal run can offer final hashes. An active HDF5 writer may
    # still be extending the latest record even when its filename exists.
    outputs = {path.name: file_record(path) for path in sorted(run.glob("wrfout*")) if path.is_file()} if terminal else {}
    log_tail = (run / "wrf.log").read_text(errors="replace")[-8192:]
    incremental = ({**file_record(build / "incremental-build-receipt.json"),
                    "receipt": json.loads((build / "incremental-build-receipt.json").read_text())}
                   if (build / "incremental-build-receipt.json").is_file() else None)
    return {"schema": "sfire-coupled-reference-receipt-v1", "source_root": str(source), "official_source_root": str(official),
            "source_manifest_sha256": source_manifest, "source_files": sources, "source_divergences": divergences,
            "archive_inputs": archive_inputs, "external_manifests": external_manifests,
            "compiler": subprocess.run(["gfortran", "--version"], check=True, text=True, capture_output=True).stdout,
            "executables": executables, "flags": flags, "cmake_cache": file_record(build / "CMakeCache.txt"),
            "incremental_build": incremental,
            "inputs": inputs, "official_case_input_divergences": case_divergences, "markers": markers,
            "terminal": terminal, "completed_successfully": terminal and markers.get("wrf.exit") == "0" and "SUCCESS COMPLETE WRF" in log_tail,
            "outputs": outputs, "wrf_log": file_record(run / "wrf.log") if terminal else {"live": True}, "wrf_log_tail": log_tail}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("official", type=Path)
    parser.add_argument("build", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = collect(args.source, args.official, args.build, args.run)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("terminal", "completed_successfully", "source_manifest_sha256", "official_case_input_divergences")}, sort_keys=True))


if __name__ == "__main__":
    main()
