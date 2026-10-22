"""Solar products and public NOAA radiation acquisition. Native data paths only."""
from __future__ import annotations
import datetime as dt
import json
import hashlib
from pathlib import Path
import subprocess
import urllib.request
from gpuwm import rustwx


def _renderer():
    path = rustwx.find_renderer()
    if path is None:
        raise RuntimeError(rustwx.renderer_remedy())
    return path


def _catalog(renderer):
    result = subprocess.run([str(renderer), "--solar-catalog"], capture_output=True,
                            text=True, env=rustwx.renderer_env(), check=True)
    return json.loads(result.stdout)


def _fetch(args, catalog):
    row = next((r for r in catalog["observations"] if r["id"] == args.network), None)
    if row is None:
        raise ValueError(f"unknown solar observation network {args.network}")
    day = dt.date.fromisoformat(args.date)
    if day < dt.date(2015, 1, 1):
        raise ValueError("these observation rows describe the post-2015 one-minute archive")
    if args.network == "surfrad" and not args.directory:
        raise ValueError("SURFRAD acquisition requires its archive station directory")
    if not args.prefix.isalnum() or not all(c.isalnum() or c == '_' for c in args.directory):
        raise ValueError("station prefix and directory must be archive tokens")
    url = row["url"].format(directory=args.directory, prefix=args.prefix,
                            year=day.year, yy=f"{day.year % 100:02}",
                            doy=f"{day.timetuple().tm_yday:03}")
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / url.rsplit('/', 1)[1]
    # Atomic replacement prevents a failed transfer from becoming a cached day.
    temporary = target.with_suffix(target.suffix + '.part')
    request = urllib.request.Request(url, headers={"User-Agent": "ArWen-solar/1"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as stream:
        while block := response.read(65536):
            stream.write(block)
    temporary.replace(target)
    receipt = {"schema": "gpuwm.solar-acquisition.v1", "url": url,
               "path": str(target.resolve()), "bytes": target.stat().st_size,
               "network": row, "date": day.isoformat(),
               "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
               "acknowledgement": "Data provided by NOAA Global Monitoring Laboratory"}
    target.with_suffix('.acquisition.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(target)
    return 0


def main(args):
    renderer = _renderer()
    catalog = _catalog(renderer)
    if args.solar_command == "catalog":
        print(json.dumps(catalog, indent=2))
        return 0
    if args.solar_command == "fetch-observations":
        return _fetch(args, catalog)
    if args.request:
        request = args.request.resolve()
    else:
        if not args.wrfout or args.out is None:
            raise ValueError("products requires --request or --wrfout ... --out DIR")
        args.out.mkdir(parents=True, exist_ok=True)
        frames = [{"label": "model", "kind": "wrfout", "path": str(p.resolve()),
                   "fields": {"solar_ghi": "SWDOWN", "solar_dni": "SWDDNI", "solar_dhi": "SWDDIF",
                              "temperature_2m": "T2", "wind_speed_10m": "WSPD10"}}
                  for p in sorted(args.wrfout)]
        data = {"out": str(args.out.resolve()), "frames": frames,
                "products": args.products.split(',') if args.products else []}
        if args.systems:
            data["systems"] = json.loads(args.systems.read_text())
        request = args.out.resolve() / 'solar-request.json'
        request.write_text(json.dumps(data, indent=2) + '\n')
    result = rustwx.run_solar_product_renderer(renderer, request)
    print(json.dumps(result, indent=2))
    return 0


def register_cli(sub):
    parser = sub.add_parser("solar", help="irradiance, PVWatts power, ramps, daily energy and NOAA verification")
    commands = parser.add_subparsers(dest="solar_command", required=True)
    commands.add_parser("catalog", help="show native product, PV system and observation rows").set_defaults(func=main)
    fetch = commands.add_parser("fetch-observations", help="fetch one public NOAA radiation day with an acquisition receipt")
    fetch.add_argument('--network', choices=('surfrad', 'solrad'), required=True)
    fetch.add_argument('--prefix', required=True)
    fetch.add_argument('--directory', default='')
    fetch.add_argument('--date', required=True)
    fetch.add_argument('--out', type=Path, required=True)
    fetch.set_defaults(func=main)
    products = commands.add_parser("products", help="compute and render a native history sequence")
    products.add_argument('--request', type=Path)
    products.add_argument('--wrfout', nargs='+', type=Path)
    products.add_argument('--out', type=Path)
    products.add_argument('--systems', type=Path)
    products.add_argument('--products', help="comma-separated solar slugs; default all")
    products.set_defaults(func=main)
