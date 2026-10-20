#!/usr/bin/env python3
"""Build a HYDRO-WORLD fabric for any lon/lat box: orchestration only.

1. asks `rw_hydro_world tiles` which tile files the source row needs for the box;
2. fetches the missing ones in parallel (bounded timeouts), records each in the
   dataset's SOURCE.md (URL, UTC time, SHA-256, bytes); a tile the service does
   not have (HTTP 404, open sea in the Copernicus tiling) is listed in ABSENT.txt
   so the builder treats it as sea instead of failing;
3. runs `rw_hydro_world build` (all data work is Rust, all cores);
4. optionally runs the engine's reader `rw_hydrofabric prepare` on the result,
   without and with a parent grid in the fabric's own CRS.

usage:
  hydro_world_build.py --bbox W,S,E,N --out FABRIC.gpkg --data-root DIR
      [--row global-copernicus-glo90] [--accept-licence NAME]... [--fetch-only]
      [--exe rw_hydro_world] [--engine rw_hydrofabric --grid-dx 1000]
"""
import argparse, concurrent.futures as cf, datetime, hashlib, json, math, os, sqlite3, subprocess, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(url, dest, timeout=600):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
            t0 = time.time()
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"{url}: download exceeded {timeout} s")
    except urllib.error.HTTPError as e:
        if os.path.exists(part):
            os.remove(part)
        if e.code == 404:
            return "absent"
        raise
    os.replace(part, dest)
    return "fetched"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def engine_prepare(engine, gpkg, out, dx):
    os.makedirs(out, exist_ok=True)
    db = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
    srs_id, = db.execute("select srs_id from gpkg_geometry_columns where table_name='divides'").fetchone()
    wkt, = db.execute("select definition from gpkg_spatial_ref_sys where srs_id=?", (srs_id,)).fetchone()
    x0, y0, x1, y1 = db.execute("select min_x,min_y,max_x,max_y from gpkg_contents where table_name='divides'").fetchone()
    terms, = db.execute("select value from fabric_metadata where key='data_terms'").fetchone()
    db.close()
    gx0, gy0 = math.floor(x0 / dx) * dx, math.floor(y0 / dx) * dx
    grid = {"nx": int(math.ceil((x1 - gx0) / dx)), "ny": int(math.ceil((y1 - gy0) / dx)), "x0": gx0, "y0": gy0, "dx": dx, "dy": dx, "crs": wkt}
    runs = []
    for tag, extra in (("nogrid", {}), ("grid", {"grid": grid})):
        cfg = os.path.join(out, f"config-{tag}.json")
        json.dump({"geopackage": os.path.abspath(gpkg), "output": os.path.join(out, f"prepared-{tag}"), "data_license": terms, **extra}, open(cfg, "w"))
        t = time.time()
        p = subprocess.run([engine, "prepare", cfg], capture_output=True, text=True, timeout=1800)
        runs.append({"tag": tag, "command": f"{engine} prepare {cfg}", "exit": p.returncode, "wall_ms": (time.time() - t) * 1e3,
                     "payloads": len(json.loads(p.stdout)["files"]) if p.returncode == 0 else 0, "stderr": p.stderr[-2000:]})
        if p.returncode == 0:
            open(os.path.join(out, f"manifest-{tag}.json"), "w").write(p.stdout)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=os.environ.get("HYDRO_WORLD_DATA"))
    ap.add_argument("--row", default="global-copernicus-glo90")
    ap.add_argument("--sources", default=os.path.join(HERE, "data", "sources.v1.json"))
    ap.add_argument("--accept-licence", action="append", default=[])
    ap.add_argument("--exe", default="rw_hydro_world")
    ap.add_argument("--engine")
    ap.add_argument("--grid-dx", type=float, default=1000.0)
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if not a.data_root:
        sys.exit("--data-root (or HYDRO_WORLD_DATA) is required")
    # Check the actual Rust licence gate before enumerating or fetching files,
    # including --fetch-only. A named research licence is never a default.
    admission = [a.exe, "admit", a.sources, a.row]
    for licence in a.accept_licence:
        admission += ["--accept-licence", licence]
    admitted = subprocess.run(admission, capture_output=True, text=True, timeout=30)
    if admitted.returncode:
        sys.exit((admitted.stderr or admitted.stdout).strip())
    bbox = f"--bbox={a.bbox}"
    tiles = json.loads(subprocess.run([a.exe, "tiles", "--sources", a.sources, "--row", a.row, bbox, "--data-root", a.data_root], check=True, capture_output=True, text=True).stdout)
    todo = [t for t in tiles if not t["exists"]]
    absent_lists = {}
    for t in tiles:
        p = os.path.join(a.data_root, os.path.dirname(t["local"]), "ABSENT.txt")
        absent_lists.setdefault(p, set(open(p).read().split()) if os.path.exists(p) else set())
    todo = [t for t in todo if os.path.basename(t["local"]) not in absent_lists[os.path.join(a.data_root, os.path.dirname(t["local"]), "ABSENT.txt")]]
    t0 = time.time()
    results = []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch, t["url"], os.path.join(a.data_root, t["local"])): t for t in todo}
        for f in cf.as_completed(futs):
            t = futs[f]
            results.append((t, f.result()))
    for t, status in sorted(results, key=lambda x: x[0]["local"]):
        dest = os.path.join(a.data_root, t["local"])
        tile_dir = os.path.dirname(dest)
        if status == "absent":
            with open(os.path.join(tile_dir, "ABSENT.txt"), "a") as f:
                f.write(os.path.basename(dest) + "\n")
            continue
        src_md = os.path.join(os.path.dirname(tile_dir) if os.path.basename(tile_dir) == "tiles" else tile_dir, "SOURCE.md")
        rel = os.path.relpath(dest, os.path.dirname(src_md))
        with open(src_md, "a") as f:
            f.write(f"| {rel} | {t['url']} | {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} | {sha256(dest)} | {os.path.getsize(dest)} |\n")
    print(json.dumps({"tiles_needed": len(tiles), "fetched": sum(s == "fetched" for _, s in results), "absent_sea": sum(s == "absent" for _, s in results), "fetch_s": round(time.time() - t0, 2)}))
    if a.fetch_only:
        return
    receipt = os.path.splitext(a.out)[0] + ".receipt.json"
    cmd = [a.exe, "build", "--sources", a.sources, "--row", a.row, bbox, "--data-root", a.data_root, "--out", a.out, "--receipt", receipt]
    for l in a.accept_licence:
        cmd += ["--accept-licence", l]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    sys.stderr.write(p.stderr)
    if p.returncode != 0:
        sys.exit(f"build failed (exit {p.returncode})")
    r = json.load(open(receipt))
    print(json.dumps({"fabric": a.out, "receipt": receipt, "total_ms": r["total_ms"], "flowpaths": r["check"]["counts"]["flowpaths"], "errors": r["check"]["errors"], "gaps": r["check"]["contract_gaps"]}))
    if a.engine:
        runs = engine_prepare(a.engine, a.out, os.path.splitext(a.out)[0] + "-engine", a.grid_dx)
        print(json.dumps([{k: v for k, v in x.items() if k != "stderr"} for x in runs]))
        if any(x["exit"] != 0 for x in runs):
            sys.exit("engine prepare failed: " + " | ".join(x["stderr"] for x in runs))


if __name__ == "__main__":
    main()
