#!/usr/bin/env python3
"""Bounded local fixtures and receipts for the production root static stage.

The root stage is compiled directly from mapped_direct.py, preserving its
production call body and options. This harness omits forcing decode and later
initialization. The full CLI qualification uses the same filesystem fixtures.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def prepare(args):
    import numpy as np
    import rasterio
    from affine import Affine
    from gpuwm.static.highres_fetch import (COPERNICUS_DEM_TILE_URL,
        copernicus_dem_tile_ids, domain_footprint, one_degree_tile_bbox)
    from gpuwm.static.projection import grids_from_wps_namelist
    from gpuwm.static.build import GeogSelection

    case = args.case_dir.resolve()
    case.mkdir(parents=True, exist_ok=False)
    source_experiment = args.experiment_config.resolve()
    source_namelist = args.wps_namelist.resolve()
    text = source_experiment.read_text()
    text = re.sub(r"(?ms)^\[static.highres\].*\Z", "", text)
    text = "\n".join(line.split("#", 1)[0].rstrip()
                      for line in text.splitlines()) + "\n"
    text = re.sub(r'(?m)^name = ".+"$', 'name = "static-resource-fixture"', text)
    text = re.sub(r"(?m)^nx = \d+$", f"nx = {args.nx}", text)
    text = re.sub(r"(?m)^ny = \d+$", f"ny = {args.ny}", text)
    (case / "experiment.toml").write_text(text, encoding="utf-8")
    wps = source_namelist.read_text()
    wps = re.sub(r"(?m)^( e_we\s*=\s*)\d+", rf"\g<1>{args.nx + 1}", wps)
    wps = re.sub(r"(?m)^( e_sn\s*=\s*)\d+", rf"\g<1>{args.ny + 1}", wps)
    (case / "namelist.wps").write_text(wps, encoding="utf-8")
    grid = grids_from_wps_namelist(case / "namelist.wps")[0]
    bbox = domain_footprint(grid, halo=3)
    files = []

    def record(path, **extra):
        files.append(dict(path=path.relative_to(case).as_posix(),
                          bytes=path.stat().st_size, sha256=sha(path), **extra))

    selection = GeogSelection.from_tokens(case / "geog", "bnu_soil_30s+default")
    for role in ("terrain", "landuse", "soil_top", "soil_bottom", "greenfrac",
                 "lai", "albedo", "snow_albedo", "soil_temperature"):
        directory = selection.path(role)
        directory.mkdir(parents=True)
        step = 1 / 120 if role not in ("lai", "soil_temperature") else (
            1 / 6 if role == "lai" else 1)
        margin = max(1, 5 * step)
        left = math.floor((bbox.lon_min - margin) / step) * step
        bottom = math.floor((bbox.lat_min - margin) / step) * step
        width = math.ceil((bbox.lon_max + margin - left) / step)
        height = math.ceil((bbox.lat_max + margin - bottom) / step)
        width = math.ceil(width / 64) * 64
        height = math.ceil(height / 64) * 64
        months = 12 if role in ("greenfrac", "lai", "albedo") else 1
        category_count = 21 if role == "landuse" else 16
        categorical = role in ("landuse", "soil_top", "soil_bottom")
        index = dict(type="categorical" if categorical else "continuous",
            signed="yes", projection="regular_ll", dx=step, dy=step,
            known_x=1, known_y=1, known_lat=bottom + step / 2,
            known_lon=left + step / 2, wordsize=2, tile_x=64, tile_y=64,
            tile_z=months, scale_factor=1 if categorical or role == "terrain" else .01,
            missing_value=-32768)
        if categorical:
            index.update(category_min=1, category_max=category_count)
        if role == "landuse":
            index.update(mminlu="MODIFIED_IGBP_MODIS_NOAH", iswater=17,
                         islake=21, isurban=13, isice=15)
        (directory / "index").write_text("\n".join(
            f"{key} = {value}" for key, value in index.items()) + "\n",
            encoding="utf-8")
        record(directory / "index", role=role)
        for row in range(0, height, 64):
            for col in range(0, width, 64):
                yy, xx = np.indices((64, 64))
                yy, xx = yy + row, xx + col
                raw = np.empty((months, 64, 64), dtype=">i2")
                for month in range(months):
                    if role == "terrain":
                        values = 100 + (xx % 101) + (yy % 71)
                    elif role == "landuse":
                        values = np.where((xx // 9 + yy // 7) % 17 == 0, 17,
                                          np.where((xx // 5 + yy // 11) % 5 == 0, 1, 10))
                    elif categorical:
                        values = 1 + ((xx // 13 + yy // 17) % 12)
                    elif role == "soil_temperature":
                        values = 28300 + (xx % 13)
                    else:
                        offset = dict(greenfrac=40, lai=200, albedo=2000,
                                      snow_albedo=6000)[role]
                        values = offset + month + (xx + yy) % 5
                    raw[month] = values
                path = directory / f"{col + 1:05d}-{col + 64:05d}.{row + 1:05d}-{row + 64:05d}"
                raw.tofile(path)
                record(path, role=role)

    cache = case / "cache/gpuwm/highres-cache/copernicus_dem_glo30"
    cache.mkdir(parents=True)
    for tile in copernicus_dem_tile_ids(bbox):
        box = one_degree_tile_bbox(tile)
        path = cache / f"Copernicus_DSM_COG_10_{tile}_DEM.tif"
        with rasterio.open(path, "w", driver="GTiff", width=3600, height=3600,
                count=1, dtype="float32", crs="EPSG:4326", nodata=-9999,
                transform=Affine(1 / 3600, 0, box.lon_min, 0, -1 / 3600, box.lat_max),
                compress="deflate", predictor=3, tiled=True, blockxsize=256,
                blockysize=256) as target:
            xx = np.arange(3600)[None, :]
            for row in range(0, 3600, 128):
                count = min(128, 3600 - row)
                yy = np.arange(row, row + count)[:, None]
                values = 250 + ((xx + int(box.lon_min * 3600)) % 83) * .75 \
                    + ((yy + int(box.lat_min * 3600)) % 71) * .35
                # Every source lattice has finite terrain and a small void.
                values = np.where((xx < 16) & (yy < 16), -9999, values)
                target.write(values.astype("float32"), 1,
                    window=rasterio.windows.Window(0, row, 3600, count))
        dump(path.with_name(path.name + ".sha256.json"), dict(
            url=COPERNICUS_DEM_TILE_URL.format(tile=tile), sha256=sha(path),
            bytes=path.stat().st_size, fetched_utc="2026-10-04T00:00:00Z"))
        record(path, role="copernicus-dem-glo30", source_shape=[3600, 3600],
               source_resolution_deg=1 / 3600)
        record(path.with_name(path.name + ".sha256.json"), role="tile-hash-receipt")
    record(case / "experiment.toml", role="experiment")
    record(case / "namelist.wps", role="namelist")
    dump(case / "fixture.json", dict(schema="arwen.box-static-fixture.v1",
        nx=args.nx, ny=args.ny, bbox=bbox.as_dict(), files=files,
        box_experiment_sha256=sha(source_experiment),
        box_namelist_sha256=sha(source_namelist),
        modifications=["Target horizontal dimensions scaled.",
            "Fallback static.highres block removed to reproduce failed default.",
            "Deterministic valid local WPS and 30 m TIFF input fixtures."],
        limitations=["Synthetic source values do not qualify operational terrain.",
            "The root-stage harness omits forcing decode and initialization."]))
    print(f"prepared {args.nx}x{args.ny}, {len(files)} files, {sum(f['bytes'] for f in files)} bytes", flush=True)


def worker(args):
    import psutil
    import numpy as np
    if args.source_root:
        from profile_highres_static import load_pinned_modules
        load_pinned_modules(args.source_root)
    from gpuwm.static import highres, highres_fetch, rust_bridge
    from gpuwm.static.corridor import _write_deterministic_npz
    from gpuwm.experiment import load_experiment
    from gpuwm.static.build import GeogSelection
    from gpuwm.static.projection import grids_from_wps_namelist
    from gpuwm import mapped_direct
    from gpuwm.static.highres_production import load_static_highres, apply_prepared_highres
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    case = args.case_dir.resolve()
    fixture = json.loads((case / "fixture.json").read_text())
    process = psutil.Process()
    phases = []
    started = time.perf_counter()
    report = dict(schema="arwen.box-static-profile.v1", implementation=args.label,
        harness_sha256=sha(__file__), fixture_sha256=sha(case / "fixture.json"),
        library=str(args.library.resolve()), library_sha256=sha(args.library),
        nx=fixture["nx"], ny=fixture["ny"], bbox=fixture["bbox"],
        limitations=fixture["limitations"], threads=2,
        started_unix=time.time(), pid=os.getpid())

    @contextmanager
    def phase(name):
        start = time.perf_counter()
        item = dict(name=name, started_perf=start, started_unix=time.time(),
                    start_rss_bytes=process.memory_info().rss)
        print(f"PHASE {name}", flush=True)
        try:
            yield
        finally:
            item.update(ended_perf=time.perf_counter(), ended_unix=time.time(),
                        wall_s=time.perf_counter() - start,
                        end_rss_bytes=process.memory_info().rss)
            phases.append(item)
            dump(output / "phases.json", phases)

    def observe(module, name):
        original = getattr(module, name)
        def wrapped(*positional, **keywords):
            with phase(name):
                return original(*positional, **keywords)
        setattr(module, name, wrapped)

    try:
        reason = highres.static_compute_workaround()
        if reason:
            raise RuntimeError(f"native qualification refused: {reason}")
        for name in ("highres_derive_window", "highres_resample", "build_fields"):
            observe(rust_bridge, name)
        exp = load_experiment(case / "experiment.toml")
        cfg = exp.root.run
        grid = grids_from_wps_namelist(case / "namelist.wps")[0]
        selection = GeogSelection.from_tokens(case / "geog", "bnu_soil_30s+default")
        static_highres = load_static_highres(case / "experiment.toml", run_config=cfg)
        report["config"] = static_highres.echo()
        report["orchestration_sources"] = {module.__name__: dict(
            path=module.__file__, sha256=sha(module.__file__))
            for module in (mapped_direct, highres, highres_fetch, rust_bridge)}
        tree = ast.parse(Path(mapped_direct.__file__).read_text(encoding="utf-8"))
        node = next(node for node in ast.walk(tree) if isinstance(node, ast.With)
                    and "Prepare root static fields" in ast.get_source_segment(
                        Path(mapped_direct.__file__).read_text(encoding="utf-8"), node))
        report["production_stage"] = dict(path=mapped_direct.__file__, line=node.lineno,
            sha256=hashlib.sha256(ast.get_source_segment(
                Path(mapped_direct.__file__).read_text(encoding="utf-8"), node).encode()).hexdigest())
        namespace = dict(vars(mapped_direct), grid=grid, cfg=cfg, exp=exp,
            selection=selection, static_highres=static_highres,
            geog_root=case / "geog", prebuilt_static=None, prebuilt_receipt=None,
            apply_prepared_highres=apply_prepared_highres)
        with phase("production-root-static"):
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         mapped_direct.__file__, "exec"), namespace)
        fields = namespace["static"]
        report["static_seconds"] = time.perf_counter() - namespace["static_started"]
        report["static_receipt"] = namespace["root_static_receipt"]
        report["field_bytes"] = sum(value.nbytes for value in fields.values())
        report["fields"] = {key: dict(shape=list(value.shape), dtype=value.dtype.str,
            bytes=value.nbytes, sha256=hashlib.sha256(
                np.ascontiguousarray(value).tobytes()).hexdigest())
            for key, value in sorted(fields.items())}
        with phase("deterministic-static-artifact"):
            artifact = output / "static.npz"
            _write_deterministic_npz(artifact, fields)
        report["artifact"] = dict(path=str(artifact), bytes=artifact.stat().st_size,
                                  sha256=sha(artifact))
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(phases=phases, wall_s=time.perf_counter() - started,
                      ended_unix=time.time(), final_rss_bytes=process.memory_info().rss)
        dump(output / "worker.json", report)


def run(args):
    import psutil
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tile_source = args.case_dir.resolve() / "cache/gpuwm/highres-cache/copernicus_dem_glo30"
    tile_target = output / "cache/gpuwm/highres-cache/copernicus_dem_glo30"
    tile_target.mkdir(parents=True)
    for source in tile_source.iterdir():
        if source.name.endswith((".tif", ".sha256.json")):
            os.link(source, tile_target / source.name)
    command = [sys.executable, str(Path(__file__).resolve()), "worker",
               "--case-dir", str(args.case_dir.resolve()), "--output", str(output),
               "--library", str(args.library.resolve()), "--label", args.label]
    if args.source_root:
        command += ["--source-root", str(args.source_root.resolve())]
    environment = dict(os.environ, GPUWM_STATIC_BRIDGE=str(args.library.resolve()),
        GPUWM_NO_LOCAL_GPU="1", CUDA_VISIBLE_DEVICES="-1", RAYON_NUM_THREADS="2",
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2", GPUWM_MAPPED_ENGINE_THREADS="2",
        LOCALAPPDATA=str(output / "cache"), XDG_CACHE_HOME=str(output / "cache"))
    environment.pop("GPUWM_STATIC_PYTHON", None)
    samples = []
    peaks_by_pid = {}
    peak = os_peak = 0
    started = time.perf_counter()
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        allowed = process.cpu_affinity()
        process.cpu_affinity(allowed[:2])
        while child.poll() is None:
            try:
                targets = [process, *process.children(recursive=True)]
                for target in targets:
                    target.cpu_affinity(allowed[:2])
                    memory = target.memory_info()
                    peak = max(peak, memory.rss)
                    os_peak = max(os_peak, getattr(memory, "peak_wset", 0))
                    peaks_by_pid[target.pid] = max(peaks_by_pid.get(target.pid, 0),
                        memory.rss, getattr(memory, "peak_wset", 0))
                    samples.append((time.time(), time.perf_counter(), target.pid,
                                    memory.rss, getattr(memory, "private", 0)))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(.05)
    with (output / "rss.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("unix_time", "perf_time", "pid", "rss_bytes", "private_bytes"))
        writer.writerows(samples)
    report = json.loads((output / "worker.json").read_text())
    for phase in report["phases"]:
        phase["peak_rss_sampled_bytes"] = max((rss for _, stamp, pid, rss, _ in samples
            if pid == report["pid"] and phase["started_perf"] <= stamp <= phase["ended_perf"]),
            default=max(phase["start_rss_bytes"], phase["end_rss_bytes"]))
    report.update(command=command, returncode=child.returncode,
        wall_process_s=time.perf_counter() - started, peak_rss_bytes=max(peak, os_peak),
        peak_rss_sampled_bytes=peak, peak_rss_os_bytes=os_peak, samples=len(samples),
        worker_peak_rss_bytes=peaks_by_pid.get(report["pid"]),
        affinity=allowed[:2], sample_interval_s=.05)
    dump(output / "profile.json", report)
    print(json.dumps(dict(status=report["status"], peak_rss_bytes=report["peak_rss_bytes"],
                          wall_s=report["wall_s"])), flush=True)
    if child.returncode:
        raise SystemExit(child.returncode)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--case-dir", type=Path, required=True)
    prepare_parser.add_argument("--experiment-config", type=Path, required=True)
    prepare_parser.add_argument("--wps-namelist", type=Path, required=True)
    prepare_parser.add_argument("--nx", type=int, required=True)
    prepare_parser.add_argument("--ny", type=int, required=True)
    for name in ("run", "worker"):
        worker_parser = subparsers.add_parser(name)
        worker_parser.add_argument("--case-dir", type=Path, required=True)
        worker_parser.add_argument("--output", type=Path, required=True)
        worker_parser.add_argument("--library", type=Path, required=True)
        worker_parser.add_argument("--label", required=True)
        worker_parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    dict(prepare=prepare, run=run, worker=worker)[args.action](args)


if __name__ == "__main__":
    main()
