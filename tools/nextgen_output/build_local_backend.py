#!/usr/bin/env python3
"""Build/stage Linux output with pinned local-file C containers, never Fortran.

Python only downloads, verifies and launches build tools. No field processing.
All generated files, including temporary compiler files, stay in --scratch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import time


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--cargo-home", required=True, type=Path)
    parser.add_argument("--cargo-target", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--fetch", action="store_true", help="fetch missing pinned archives")
    parser.add_argument("--keep-work", action="store_true", help="retain sources/builds for diagnosis")
    args = parser.parse_args()
    if platform.system() != "Linux" or args.jobs < 1:
        parser.error("this recipe requires Linux and a positive build-worker count")
    root = args.scratch.resolve()
    root.mkdir(parents=True, exist_ok=True)
    crate = Path(__file__).resolve().parent
    target = args.cargo_target.resolve()
    prefix, stage = root / "prefix", root / "stage"
    stage.mkdir()  # A completed or partial previous stage is never overwritten.
    tmp = root / "tmp"
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ, TMPDIR=str(tmp), XDG_CACHE_HOME=str(root / "cache"),
               CARGO_HOME=str(args.cargo_home.resolve()), CARGO_TARGET_DIR=str(target))
    # Do not allow caller search paths to silently select a system C backend.
    for key in ("CMAKE_PREFIX_PATH", "HDF5_ROOT", "HDF5_DIR", "RUSTFLAGS"):
        env.pop(key, None)
    rows, cleanup = [], []
    receipt = {"schema": "arwen.nextgen.output-local-backend.v1", "jobs": args.jobs,
               "machine": platform.machine(), "commands": rows}
    record = root / "build-receipt.json"

    def run(name, argv):
        start = time.perf_counter_ns()
        result = subprocess.run(list(map(str, argv)), env=env, cwd=root,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = root / (name + ".log")
        log.write_text(result.stdout)
        rows.append(dict(stage=name, argv=list(map(str, argv)), exit=result.returncode,
                         wall_ms=(time.perf_counter_ns() - start) / 1e6,
                         log=str(log), log_sha256=sha(log)))
        record.write_text(json.dumps(receipt, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {log}")
        return result.stdout

    notices = stage / "notices"
    notices.mkdir()
    sources = json.loads((crate / "backend-sources.json").read_text())
    receipt["sources"] = sources
    for item in sources:
        archive = root / (item["name"] + ".tar.gz")
        if not archive.exists() and args.fetch:
            run("fetch-" + item["name"], ["curl", "--fail", "--location", "--retry", "2",
                "--silent", "--show-error", "--output", archive, item["url"]])
        if not archive.is_file() or sha(archive) != item["archive_sha256"]:
            raise RuntimeError(f"missing or incorrect archive: {archive}")
        source = root / (item["name"] + "-" + item["commit"])
        # Always extract the verified archive; never compile unverified leftovers.
        if source.exists():
            shutil.rmtree(source)
        with tarfile.open(archive) as bundle:
            bundle.extractall(root, filter="data")
        for name, digest in item["notices"].items():
            if sha(source / name) != digest:
                raise RuntimeError(f"changed upstream licence: {source / name}")
        cleanup.extend([source, archive])
    common = ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
              "-DCMAKE_INSTALL_LIBDIR=lib", f"-DCMAKE_INSTALL_PREFIX={prefix}",
              "-DBUILD_SHARED_LIBS=OFF", "-DBUILD_TESTING=OFF"]

    def build(name, options):
        item = next(item for item in sources if item["name"] == name)
        source = root / (name + "-" + item["commit"])
        build_dir = root / ("build-" + name)
        cleanup.append(build_dir)
        run(name + "-configure", ["cmake", "-S", source, "-B", build_dir, *common, *options])
        run(name + "-build", ["cmake", "--build", build_dir, "--parallel", args.jobs])
        run(name + "-install", ["cmake", "--install", build_dir])

    build("zlib", ["-DZLIB_BUILD_EXAMPLES=OFF"])
    build("hdf5", ["-DHDF5_BUILD_HL_LIB=ON", "-DHDF5_BUILD_FORTRAN=OFF",
        "-DHDF5_BUILD_CPP_LIB=OFF", "-DHDF5_BUILD_JAVA=OFF", "-DHDF5_BUILD_TOOLS=OFF",
        "-DHDF5_BUILD_EXAMPLES=OFF", "-DHDF5_ENABLE_PARALLEL=OFF",
        "-DHDF5_ENABLE_ROS3_VFD=OFF", "-DHDF5_ENABLE_MIRROR_VFD=OFF",
        "-DHDF5_ENABLE_SZIP_SUPPORT=OFF", "-DHDF5_ENABLE_Z_LIB_SUPPORT=ON",
        f"-DZLIB_LIBRARY_RELEASE={prefix}/lib/libz.a", f"-DZLIB_INCLUDE_DIR={prefix}/include"])
    build("netcdf-c", [f"-DCMAKE_PREFIX_PATH={prefix}", f"-DHDF5_ROOT={prefix}",
        "-DHDF5_USE_STATIC_LIBRARIES=ON", "-DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=ON",
        f"-DZLIB_LIBRARY={prefix}/lib/libz.a", f"-DZLIB_INCLUDE_DIR={prefix}/include",
        "-DNETCDF_ENABLE_HDF5=ON",
        "-DNETCDF_ENABLE_CDF5=ON", "-DNETCDF_ENABLE_REMOTE_FUNCTIONALITY=OFF",
        "-DNETCDF_ENABLE_DAP=OFF", "-DNETCDF_ENABLE_NCZARR=OFF",
        "-DNETCDF_ENABLE_BYTERANGE=OFF", "-DNETCDF_ENABLE_PLUGINS=OFF",
        "-DNETCDF_ENABLE_FILTER_SZIP=OFF", "-DNETCDF_ENABLE_FILTER_BZ2=OFF",
        "-DNETCDF_ENABLE_FILTER_BLOSC=OFF", "-DNETCDF_ENABLE_FILTER_ZSTD=OFF",
        "-DNETCDF_ENABLE_LIBXML2=OFF", "-DNETCDF_ENABLE_TESTS=OFF",
        "-DNETCDF_ENABLE_EXAMPLES=OFF", "-DNETCDF_BUILD_UTILITIES=OFF"])
    env["GPUWM_OUTPUT_NETCDF_PREFIX"] = str(prefix)
    run("rust-build", ["cargo", "build", "--release", "--offline", "--locked",
                       "--jobs", args.jobs, "--manifest-path", crate / "Cargo.toml"])
    lib = stage / "gpuwm" / "hydro" / "librw_nextgen_output.so"
    lib.parent.mkdir(parents=True)
    cli = stage / "bin" / "rw_nextgen_output"
    cli.parent.mkdir()
    shutil.copy2(target / "release" / lib.name, lib)
    shutil.copy2(target / "release" / cli.name, cli)
    # Match the existing Python loader's sibling-library contract in the stage.
    shutil.copy2(crate.parents[1] / "gpuwm/hydro/output.py", lib.with_name("output.py"))
    for name in ("NOTICE", "NETCDF-C-NOTICE", "HDF5-NOTICE.txt", "ZLIB-NOTICE.txt",
                 "UTF8PROC-NOTICE.txt", "PARQUET-NOTICE.txt", "SNAP-NOTICE.txt"):
        shutil.copy2(crate / name, notices / name)
    shutil.copy2(crate.parents[1] / "LICENSE", notices / "APACHE-2.0.txt")
    receipt["dynamic_dependencies"] = {}
    for artifact in (lib, cli):
        linked = run("ldd-" + artifact.name, ["ldd", artifact])
        needed = run("elf-" + artifact.name, ["readelf", "--dynamic", artifact])
        run("versions-" + artifact.name, ["readelf", "--version-info", artifact])
        receipt["dynamic_dependencies"][str(artifact.relative_to(stage))] = linked
        for line in needed.splitlines():
            if "(NEEDED)" in line:
                name = line.split("[")[1].split("]")[0]
                if name not in {"libc.so.6", "libm.so.6", "libgcc_s.so.1", "ld-linux-x86-64.so.2"}:
                    raise RuntimeError(f"unexpected dynamic dependency in {artifact}: {name}")
    receipt["artifacts"] = {str(p.relative_to(stage)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                            for p in sorted(stage.rglob("*")) if p.is_file()}
    receipt["scope"] = "Linux host build; C containers statically linked, system libc/libm/GCC runtime remain. Not a manylinux/wheel or installed-product qualification."
    # The staged product needs none of these source archives or build objects.
    if not args.keep_work:
        cleanup.extend([prefix, tmp, root / "cache"])
        receipt["cleanup"] = []
        for path in cleanup:
            if path.exists():
                size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size
                receipt["cleanup"].append({"path": str(path), "logical_bytes": size})
                shutil.rmtree(path) if path.is_dir() else path.unlink()
    record.write_text(json.dumps(receipt, indent=2) + "\n")
    shutil.copy2(record, stage / "BACKEND-BUILD.json")
    print(json.dumps({"stage": str(stage), "receipt": str(record), "artifacts": receipt["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
