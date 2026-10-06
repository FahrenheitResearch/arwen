#!/usr/bin/env python3
"""Profile real native high-resolution statics and retain exact-byte receipts.

The before implementation can be pinned source files and a library built from
the staging commit. Inputs are bounded synthetic GeoTIFFs, never operational
simulation data. Worker processes run the actual native mosaic, warp, and merge
operations. No substitute data-path implementation or GPU is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STAGING_SHA = "00bbfcf9a91723125b4a1328248e90b66e611256"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def load_pinned_modules(source_root):
    """Load exact staging orchestration while retaining the real native ABI."""
    import gpuwm.static
    for name in ("rust_bridge", "highres", "highres_fetch"):
        canonical = "gpuwm.static." + name
        spec = importlib.util.spec_from_file_location(
            canonical, Path(source_root) / "gpuwm/static" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[canonical] = module
        setattr(gpuwm.static, name, module)
        spec.loader.exec_module(module)


def grid_for(nx, ny):
    from gpuwm.static.lambert import LambertGrid
    return LambertGrid(39.0, -98.0, 30.0, 60.0, -98.0,
                       1000.0, 1000.0, nx + 1, ny + 1)


def snapshot_staging(args):
    """Preserve the exact staging compute and orchestration for a rerun."""
    source_root = args.output.resolve()
    source_root.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(
        ["git", "rev-parse", args.staging_sha], cwd=ROOT,
        text=True).strip()
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit,
         "tools/rustwx/crates/static-fields/src",
         "tools/rustwx/crates/rw-libm/src"], cwd=ROOT,
        text=True).splitlines()
    paths += ["tools/rustwx/crates/static-fields/Cargo.toml",
              "tools/rustwx/crates/static-fields/build.rs",
              "tools/rustwx/crates/rw-libm/Cargo.toml",
              "gpuwm/static/highres.py", "gpuwm/static/rust_bridge.py",
              "gpuwm/static/highres_fetch.py", "tools/rustwx/Cargo.lock"]
    files = []
    for name in paths:
        data = subprocess.check_output(["git", "show", f"{commit}:{name}"],
                                       cwd=ROOT)
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append(dict(path=name, bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest()))
    # The crate and its one local dependency remain exact. This reduced
    # workspace excludes unrelated renderer crates and limits build work.
    workspace = '''[workspace]
members = ["crates/static-fields", "crates/rw-libm"]
resolver = "2"
[workspace.package]
edition = "2024"
license = "MIT"
publish = false
rust-version = "1.85"
[workspace.dependencies]
rayon = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
[profile.release]
lto = "thin"
codegen-units = 1
strip = "symbols"
'''
    (source_root / "tools/rustwx/Cargo.toml").write_text(workspace,
                                                        encoding="utf-8")
    write_json(source_root / "manifest.json", dict(staging_sha=commit,
                                                    files=files))
    print(f"preserved {len(files)} staging files from {commit}", flush=True)


def prepare_case(args):
    """Write compressed input rasters a row block at a time."""
    import numpy as np
    import rasterio
    from affine import Affine
    from gpuwm.static.highres_fetch import domain_footprint

    case_dir = args.case_dir.resolve()
    case_dir.mkdir(parents=True, exist_ok=False)
    grid = grid_for(args.nx, args.ny)
    bbox = domain_footprint(grid, halo=3)
    # 0.0025 degree is about 278 m north-south. Longitude spacing is
    # narrower at this latitude. It is explicitly a resource-bounded
    # synthetic source, rather than a full 30 m GLO-30 qualification.
    res = args.source_resolution
    left = np.floor((bbox.lon_min - 0.03) / res) * res
    right = np.ceil((bbox.lon_max + 0.03) / res) * res
    bottom = np.floor((bbox.lat_min - 0.03) / res) * res
    top = np.ceil((bbox.lat_max + 0.03) / res) * res
    width = int(round((right - left) / res))
    height = int(round((top - bottom) / res))
    profile = dict(driver="GTiff", count=1, crs="EPSG:4326",
                   compress="deflate", tiled=True, blockxsize=256,
                   blockysize=256)
    files = []

    def record(path, role, **extra):
        files.append(dict(path=path.name, role=role,
                          sha256=sha256_file(path), bytes=path.stat().st_size,
                          **extra))

    def write_raster(path, nx, ny, row0, col0, dtype, nodata, make_block):
        with rasterio.open(path, "w", width=nx, height=ny, dtype=dtype,
                           nodata=nodata,
                           transform=Affine(res, 0, left + col0 * res,
                                            0, -res, top - row0 * res),
                           predictor=3 if dtype == "float32" else 2,
                           **profile) as target:
            xx = np.arange(col0, col0 + nx, dtype=np.int64)[None, :]
            for offset in range(0, ny, 128):
                count = min(128, ny - offset)
                yy = np.arange(row0 + offset, row0 + offset + count,
                               dtype=np.int64)[:, None]
                target.write(np.asarray(make_block(yy, xx), dtype=dtype), 1,
                             window=rasterio.windows.Window(0, offset, nx,
                                                            count))

    # Four adjoining tiles plus a small shared-ground overlap exercise
    # first-writer ordering, source pixel lattices, and encoded block seams.
    for iy in range(2):
        for ix in range(2):
            row0 = iy * (height // 2)
            col0 = ix * (width // 2)
            ny = height - row0 if iy else height // 2 + 2
            nx = width - col0 if ix else width // 2 + 2
            path = case_dir / f"terrain_{iy}_{ix}.tif"

            def terrain(yy, xx):
                values = (300.0 + (xx % 433) * 0.75 + (yy % 311) * 0.35
                          + np.sin(xx * 0.013) * 80.0
                          + np.cos(yy * 0.017) * 55.0)
                # A missing patch inside the footprint causes a real
                # coverage transition to the 30-arc-second baseline.
                hole = ((xx > width * 3 // 8) & (xx < width * 7 // 16)
                        & (yy > height * 3 // 8) & (yy < height * 7 // 16))
                return np.where(hole, -9999.0, values)

            write_raster(path, nx, ny, row0, col0, "float32", -9999.0,
                         terrain)
            record(path, "terrain")

    path = case_dir / "landcover.tif"
    classes = np.array([11, 21, 31, 41, 42, 52, 71, 82, 90, 12], dtype=np.uint8)

    def landcover(yy, xx):
        values = classes[((xx // 7) + (yy // 11)) % len(classes)]
        hole = ((xx > width * 5 // 8) & (xx < width * 11 // 16)
                & (yy > height * 5 // 8) & (yy < height * 11 // 16))
        return np.where(hole, 0, values)

    write_raster(path, width, height, 0, 0, "uint8", 0, landcover)
    record(path, "landcover")
    depths = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"]
    fractions = dict(sand=[80, 20, 40, 10], silt=[10, 30, 40, 80],
                     clay=[10, 50, 20, 10])
    for depth_index, depth in enumerate(depths):
        for component in ("sand", "silt", "clay"):
            path = case_dir / f"{component}_{depth}.tif"
            table = np.array(fractions[component], dtype=np.int16)

            def soil(yy, xx, table=table, component=component,
                     depth_index=depth_index):
                values = table[((xx // 13) + (yy // 17)) % 4] * 10
                # Keep raw component totals at 100 percent while making
                # top and bottom depth-weighted classes distinguishable.
                delta = depth_index * (5 if component == "sand" else
                                       -5 if component == "silt" else 0)
                hole = ((xx > width * 5 // 16) & (xx < width * 3 // 8)
                        & (yy > height * 5 // 8) & (yy < height * 11 // 16))
                return np.where(hole, -32768, values + delta)

            write_raster(path, width, height, 0, 0, "int16", -32768, soil)
            record(path, "soil", component=component, depth=depth)
    payload = dict(schema="arwen.highres-static-case.v1",
                   nx=args.nx, ny=args.ny, dx_m=1000.0,
                   source_resolution_deg=res,
                   source_shape=[height, width], bbox=bbox.as_dict(),
                   files=files,
                   limitations=["Synthetic inputs at about 250 m, not operational GLO-30 inputs.",
                                "The baseline climatology is a deterministic valid fixture."])
    write_json(case_dir / "case.json", payload)
    print(f"case {args.nx}x{args.ny}: source {width}x{height}; "
          f"{sum(x['bytes'] for x in files)} input bytes", flush=True)


def baseline_fields(nx, ny):
    """A valid Noah baseline with coast and monthly donor variability."""
    import numpy as np
    yy, xx = np.indices((ny, nx))
    ocean = ((xx < nx // 8) & (yy < ny // 3)) | (
        (xx > nx * 7 // 8) & (yy > ny * 2 // 3))
    land = ~ocean
    luf = np.zeros((21, ny, nx), dtype=np.float64)
    luf[9] = land
    luf[16] = ocean
    soil = np.zeros((16, ny, nx), dtype=np.float64)
    soil[5] = land
    soil[13] = ocean
    fields = dict(HGT_M=100.0 + xx * 0.05 + yy * 0.1,
                  LANDUSEF=luf, LANDMASK=land.astype(np.float64),
                  LU_INDEX=np.where(land, 10.0, 17.0),
                  SOILCTOP=soil, SOILCBOT=soil.copy(),
                  SCT_DOM=np.where(land, 6.0, 14.0),
                  SCB_DOM=np.where(land, 6.0, 14.0),
                  SNOALB=np.where(land, 0.6, 0.0),
                  SOILTEMP=np.where(land, 283.0 + xx * 0.0002, 0.0))
    for name, value in (("GREENFRAC", 0.4), ("LAI12M", 2.0),
                        ("ALBEDO12M", 20.0)):
        monthly = np.empty((12, ny, nx), dtype=np.float64)
        for month in range(12):
            monthly[month] = np.where(land, value + month * 0.01 +
                                     (xx % 7) * 0.001, 0.0)
        fields[name] = monthly
    return fields


def array_receipt(fields):
    import numpy as np
    result = {}
    for name in sorted(fields):
        value = np.ascontiguousarray(fields[name])
        result[name] = dict(shape=list(value.shape), dtype=value.dtype.str,
                            bytes=value.nbytes,
                            sha256=hashlib.sha256(memoryview(value).cast("B")).hexdigest())
    return result


def worker(args):
    import numpy as np
    import psutil
    sys.path.insert(0, str(ROOT))
    if args.source_root:
        load_pinned_modules(args.source_root)
    from gpuwm.static import highres, highres_fetch, rust_bridge
    from gpuwm.static.corridor import _write_deterministic_npz
    from gpuwm.static.highres import BoundRaster

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    phases = []
    report = dict(schema="arwen.highres-static-profile.v1",
                  staging_sha=STAGING_SHA, implementation=args.label,
                  library=str(args.library.resolve()),
                  library_sha256=sha256_file(args.library),
                  source_root=str(args.source_root) if args.source_root else str(ROOT),
                  harness_sha256=sha256_file(Path(__file__)),
                  python=platform.python_version(), numpy=np.__version__,
                  platform=platform.platform(), threads=2,
                  pid=os.getpid(), started_unix=time.time(),
                  started_perf=time.perf_counter())

    @contextmanager
    def phase(name):
        started = time.perf_counter()
        record = dict(name=name, started_unix=time.time(),
                      started_perf=started,
                      start_rss_bytes=process.memory_info().rss)
        print(f"PHASE {name}", flush=True)
        try:
            yield
        finally:
            record.update(ended_unix=time.time(),
                          ended_perf=time.perf_counter(),
                          wall_s=time.perf_counter() - started,
                          end_rss_bytes=process.memory_info().rss)
            phases.append(record)
            write_json(output / "phases.json", phases)

    try:
        reason = highres.static_compute_workaround()
        if reason:
            raise RuntimeError(f"native qualification refused: {reason}")
        library = rust_bridge.load()
        report["native_library_loaded"] = library._name
        report["orchestration_sources"] = {
            module.__name__: dict(path=module.__file__,
                                  sha256=sha256_file(module.__file__))
            for module in (highres, highres_fetch, rust_bridge)}
        native_root = (args.source_root or ROOT) / "tools/rustwx/crates/static-fields/src"
        report["native_sources"] = {
            path.relative_to(native_root).as_posix(): sha256_file(path)
            for path in sorted(native_root.rglob("*.rs"))}
        case = json.loads((args.case_dir / "case.json").read_text(encoding="utf-8"))
        nx, ny = case["nx"], case["ny"]
        report.update(nx=nx, ny=ny, dx_m=case["dx_m"],
                      source_shape=case["source_shape"],
                      source_resolution_deg=case["source_resolution_deg"],
                      limitations=case["limitations"])
        grid = grid_for(nx, ny)
        files = case["files"]

        def bind(row, path=None):
            path = path or args.case_dir / row["path"]
            return BoundRaster(path=path, sha256=row["sha256"],
                               expected_bytes=row["bytes"],
                               source_id="synthetic-" + row["role"],
                               role=row["role"], source_url="local:bounded-fixture",
                               license_id="fixture", license_url="local:fixture",
                               nominal_resolution="about 250 m",
                               scale_factor=0.1 if row["role"] == "soil" else 1.0)

        with phase("baseline-allocation"):
            baseline = baseline_fields(nx, ny)
        report["baseline_bytes"] = sum(value.nbytes for value in baseline.values())
        with phase("native-terrain-mosaic"):
            terrain_tiles = [highres_fetch.FetchedFile(
                path=args.case_dir / row["path"], url="local:bounded-fixture",
                sha256=row["sha256"], bytes=row["bytes"],
                fetched_utc="fixture", cache_hit=False)
                             for row in files if row["role"] == "terrain"]
            bbox = highres_fetch.FootprintBBox(**case["bbox"])
            terrain, mosaic_audit = highres_fetch.derive_global_terrain_window(
                terrain_tiles, bbox, output / "cache", sea_level_fill=None,
                source_nodata=-9999.0,
                resolution_deg=case["source_resolution_deg"])
        report["terrain_mosaic"] = dict(sha256=terrain.sha256, bytes=terrain.bytes,
                                        audit=mosaic_audit)
        terrain_bound = BoundRaster(
            path=terrain.path, sha256=terrain.sha256, expected_bytes=terrain.bytes,
            source_id="synthetic-terrain", role="terrain",
            source_url="local:bounded-fixture", license_id="fixture",
            license_url="local:fixture", nominal_resolution="about 250 m")
        landcover_row = next(row for row in files if row["role"] == "landcover")
        with phase("native-landcover-window"):
            landcover_input = highres_fetch.FetchedFile(
                path=args.case_dir / landcover_row["path"],
                url="local:bounded-fixture", sha256=landcover_row["sha256"],
                bytes=landcover_row["bytes"], fetched_utc="fixture",
                cache_hit=False)
            landcover_window = highres_fetch.derive_landcover_window(
                landcover_input, bbox, output / "cache")
        report["landcover_window"] = dict(
            sha256=landcover_window.sha256, bytes=landcover_window.bytes,
            audit=highres_fetch.landcover_window_audit(landcover_window))
        landcover = bind({**landcover_row, "sha256": landcover_window.sha256,
                          "bytes": landcover_window.bytes},
                         path=landcover_window.path)
        soils = {(row["component"], row["depth"]): bind(row)
                 for row in files if row["role"] == "soil"}

        # Timing wrappers observe the real bridge calls. Each wrapper delegates
        # unchanged arguments and unchanged results to the loaded native ABI.
        # They do not supply synthetic outputs, bypass any operation, or alter
        # arithmetic. Their records identify which real operation holds RSS.
        original_resample = rust_bridge.highres_resample

        def timed_resample(request):
            label = "native-resample-" + request["kind"]
            if request["kind"] == "soil-categories":
                label += "-" + request["depth_weights"][0][0]
            with phase(label):
                return original_resample(request)

        rust_bridge.highres_resample = timed_resample
        with phase("full-highres-overrides"):
            overrides, overlay_audit = highres.build_highres_overrides(
                grid, terrain=terrain_bound, landcover=landcover,
                soil_sources=soils,
                baseline_ocean=highres.baseline_ocean_mask(baseline),
                baseline=baseline)
        report["overrides"] = array_receipt(overrides)
        report["overlay_audit"] = overlay_audit
        coverage = overlay_audit["coverage"]["fields"]
        report["fixture_checks"] = {
            "terrain_fallback_exercised": coverage["terrain"]["cells_outside_coverage"] > 0,
            "terrain_blend_exercised": coverage["terrain"]["cells_blended"] > 0,
            "landcover_fallback_exercised": coverage["land_use"]["cells_outside_coverage"] > 0,
            "landcover_blend_exercised": coverage["land_use"]["cells_blended"] > 0,
            "soil_top_fallback_exercised": coverage["soil_top_0_30cm"]["cells_outside_coverage"] > 0,
            "soil_bottom_fallback_exercised": coverage["soil_bottom_30_100cm"]["cells_outside_coverage"] > 0,
            "several_land_categories": len(np.unique(overrides["LU_INDEX"])) >= 5,
            "several_soil_categories": len(np.unique(overrides["SCT_DOM"])) >= 3,
        }
        if not all(report["fixture_checks"].values()):
            raise AssertionError(f"fixture missed required data paths: {report['fixture_checks']}")
        with phase("native-static-merge"):
            merged, merge_audit = highres.merge_highres_overrides(baseline,
                                                                overrides)
        report["merged"] = array_receipt(merged)
        report["merge_audit"] = merge_audit
        del overrides, baseline
        artifact = output / "static.npz"
        with phase("deterministic-static-artifact"):
            _write_deterministic_npz(artifact, merged)
        report["artifact"] = dict(path=str(artifact), bytes=artifact.stat().st_size,
                                   sha256=sha256_file(artifact))
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(phases=phases, ended_unix=time.time(),
                      wall_s=time.perf_counter() - report["started_perf"],
                      final_rss_bytes=process.memory_info().rss)
        write_json(output / "worker.json", report)


def run_profile(args):
    import csv
    import psutil
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "worker",
               "--case-dir", str(args.case_dir.resolve()),
               "--output", str(output), "--library", str(args.library.resolve()),
               "--label", args.label]
    if args.source_root:
        command += ["--source-root", str(args.source_root.resolve())]
    environment = dict(os.environ, GPUWM_STATIC_BRIDGE=str(args.library.resolve()),
                       GPUWM_NO_LOCAL_GPU="1", RAYON_NUM_THREADS="2",
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    environment.pop("GPUWM_STATIC_PYTHON", None)
    wall_started = time.perf_counter()
    samples = []
    peak = 0
    os_peak = 0
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT)
        observed = psutil.Process(child.pid)
        if os.name == "nt":
            observed.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        while child.poll() is None:
            try:
                info = observed.memory_info()
                peak = max(peak, info.rss)
                os_peak = max(os_peak, getattr(info, "peak_wset", 0))
                samples.append((time.time(), info.rss,
                                getattr(info, "private", 0),
                                time.perf_counter()))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.05)
    with (output / "rss.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("unix_time", "rss_bytes", "private_bytes", "perf_time"))
        writer.writerows(samples)
    report = json.loads((output / "worker.json").read_text(encoding="utf-8"))
    report.update(command=command, returncode=child.returncode,
                  wall_process_s=time.perf_counter() - wall_started,
                  peak_rss_sampled_bytes=peak,
                  peak_rss_os_bytes=os_peak,
                  peak_rss_bytes=max(peak, os_peak),
                  sampling_interval_s=0.05, samples=len(samples))
    for phase in report["phases"]:
        readings = [rss for _, rss, _, stamp in samples
                    if phase["started_perf"] <= stamp <= phase["ended_perf"]]
        phase["peak_rss_sampled_bytes"] = max(readings, default=max(
            phase["start_rss_bytes"], phase["end_rss_bytes"]))
    write_json(output / "profile.json", report)
    print(f"{args.label} {report.get('nx')}x{report.get('ny')}: "
          f"{report['status']}, peak {report['peak_rss_bytes'] / 2**30:.3f} GiB, "
          f"wall {report['wall_s']:.3f} s; {output / 'profile.json'}", flush=True)
    return child.returncode


def compare_profiles(args):
    before = json.loads((args.before / "profile.json").read_text(encoding="utf-8"))
    after = json.loads((args.after / "profile.json").read_text(encoding="utf-8"))
    comparisons = {}
    for section in ("overrides", "merged"):
        names = sorted(set(before[section]) | set(after[section]))
        comparisons[section] = {name: before[section].get(name) == after[section].get(name)
                                for name in names}
    a, b = Path(before["artifact"]["path"]), Path(after["artifact"]["path"])
    equal = a.stat().st_size == b.stat().st_size
    offset = 0
    first_difference = None
    with a.open("rb") as left, b.open("rb") as right:
        while True:
            x, y = left.read(8 * 1024 * 1024), right.read(8 * 1024 * 1024)
            if x != y:
                equal = False
                limit = min(len(x), len(y))
                first_difference = offset + next(
                    (i for i in range(limit) if x[i] != y[i]), limit)
                break
            if not x:
                break
            offset += len(x)
    report = dict(schema="arwen.highres-static-identity.v1",
                  before=str(args.before.resolve() / "profile.json"),
                  after=str(args.after.resolve() / "profile.json"),
                  staging_sha=STAGING_SHA, nx=before["nx"], ny=before["ny"],
                  arrays=comparisons,
                  array_bytes_identical=all(all(x.values()) for x in comparisons.values()),
                  artifact_bytes_equal=equal, artifact_bytes_compared=offset,
                  artifact_first_difference=first_difference,
                  artifact_before=before["artifact"], artifact_after=after["artifact"],
                  terrain_mosaic_bytes_identical=(before["terrain_mosaic"]["sha256"] ==
                                                 after["terrain_mosaic"]["sha256"]),
                  terrain_mosaic_audit_equal=(before["terrain_mosaic"]["audit"] ==
                                             after["terrain_mosaic"]["audit"]),
                  landcover_window_bytes_identical=(before["landcover_window"]["sha256"] ==
                                                   after["landcover_window"]["sha256"]),
                  landcover_window_audit_equal=(before["landcover_window"]["audit"] ==
                                               after["landcover_window"]["audit"]),
                  merge_audit_equal=before["merge_audit"] == after["merge_audit"],
                  before_peak_rss_bytes=before["peak_rss_bytes"],
                  after_peak_rss_bytes=after["peak_rss_bytes"],
                  before_wall_s=before["wall_s"], after_wall_s=after["wall_s"],
                  limitations=before["limitations"])
    report["status"] = "passed" if (report["array_bytes_identical"] and equal and
                                               report["terrain_mosaic_bytes_identical"] and
                                               report["landcover_window_bytes_identical"] and
                                               report["landcover_window_audit_equal"] and
                                               report["merge_audit_equal"]) else "failed"
    write_json(args.output, report)
    print(f"identity {before['nx']}x{before['ny']}: {report['status']}; "
          f"compared {offset} static artifact bytes", flush=True)
    return 0 if report["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    snapshot = subs.add_parser("snapshot")
    snapshot.add_argument("--output", required=True, type=Path)
    snapshot.add_argument("--staging-sha", default=STAGING_SHA)
    prepare = subs.add_parser("prepare")
    prepare.add_argument("--case-dir", required=True, type=Path)
    prepare.add_argument("--nx", required=True, type=int)
    prepare.add_argument("--ny", required=True, type=int)
    prepare.add_argument("--source-resolution", type=float, default=0.0025)
    for mode in ("run", "worker"):
        sub = subs.add_parser(mode)
        sub.add_argument("--case-dir", required=True, type=Path)
        sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--library", required=True, type=Path)
        sub.add_argument("--label", required=True)
        sub.add_argument("--source-root", type=Path)
    compare = subs.add_parser("compare")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"snapshot": snapshot_staging, "prepare": prepare_case, "worker": worker,
            "run": run_profile, "compare": compare_profiles}[args.mode](args) or 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
