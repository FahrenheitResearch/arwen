"""Public fire acquisition and CLI orchestration; Rust owns every data transform."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import uuid
from gpuwm.static import rust_bridge
from gpuwm.static.sfire import _json_buffer, _library

ROUTES = Path(__file__).parent / "authorities" / "fire-observation-routes.v1.json"


def prepare(request):
    """Decode sources once, retaining the native metadata receipt."""
    library = _library()
    if not hasattr(library, "gpuwm_static_fire_ignition"):
        raise RuntimeError("Rust static-fields lacks public fire ignition; rebuild static-fields")
    function = library.gpuwm_static_fire_ignition
    function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_uint64)]
    function.restype = ctypes.c_int32
    for symbol in ("gpuwm_static_sfire_metadata_json",):
        getattr(library, symbol).argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        getattr(library, symbol).restype = ctypes.c_int64
    library.gpuwm_static_sfire_metadata_drop.argtypes = [ctypes.c_uint64]
    library.gpuwm_static_sfire_metadata_drop.restype = None
    data, length = _json_buffer(request)
    handle = ctypes.c_uint64()
    started = time.perf_counter()
    if function(data, length, ctypes.byref(handle)):
        raise ValueError(rust_bridge.last_error(library))
    try:
        count = library.gpuwm_static_sfire_metadata_json(handle.value, None, 0)
        if count < 0:
            raise ValueError(rust_bridge.last_error(library))
        output = (ctypes.c_uint8 * count)()
        if library.gpuwm_static_sfire_metadata_json(handle.value, output, count) != count:
            raise ValueError(rust_bridge.last_error(library))
        receipt = json.loads(bytes(output))
        receipt["orchestration_wall_seconds"] = time.perf_counter() - started
        (Path(request["output_directory"]) / "ignition-receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt
    finally:
        library.gpuwm_static_sfire_metadata_drop(handle.value)
        rust_bridge.fieldset_free(handle.value)


def acquire(route_name, incident_id, directory):
    """Fetch an ID-bound public GeoJSON response, leaving decode to Rust."""
    routes = json.loads(ROUTES.read_text(encoding="utf-8"))["sources"]
    route = routes[route_name]
    identifier = "{" + str(uuid.UUID(incident_id.strip("{}"))) + "}"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (route_name + ".geojson")
    pages = []
    offset = 0
    urls = []
    while True:
        query = urllib.parse.urlencode(dict(
            where=f"{route['id_property']} = '{identifier}'", outFields="*",
            returnGeometry="true", outSR=4326, f="geojson",
            orderByFields=route["object_id_property"], resultOffset=offset,
            resultRecordCount=1000))
        url = route["url"] + "/query?" + query
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        # Pagination control is provider response metadata. Geometry remains
        # untouched and the native decoder joins and validates the pages.
        page = json.loads(payload)
        if "error" in page:
            raise ValueError(f"Public fire service refused acquisition: {page['error']}")
        pages.append(payload)
        urls.append(url)
        more = page.get("exceededTransferLimit", page.get("properties", {}).get("exceededTransferLimit", False))
        if not more:
            break
        if not page.get("features"):
            raise ValueError("Public fire service returned an empty truncated page")
        offset += len(page["features"])
    payload = (b'{"type":"gpuwm-fire-geojson-pages-v1","pages":['
               + b",".join(pages) + b"]}")
    path.write_bytes(payload)
    url = urls[0]
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest(),
                source_url=url, license=route["license"],
                id_property=route["id_property"], time_property=route["time_property"])


def request_from_file(path, *, experiment=None, domain=None):
    path = Path(path).resolve()
    request = json.loads(path.read_text(encoding="utf-8"))
    if request.pop("schema", None) != "gpuwm-fire-ignition-v1":
        raise ValueError("Fire ignition request must declare gpuwm-fire-ignition-v1")
    if experiment is not None:
        from gpuwm.static.sfire import resolve_experiment_request
        resolved = resolve_experiment_request(Path(experiment), {}, domain_id=domain)
        for key in ("grid_spec", "sr_x", "sr_y"):
            if key in request and request[key] != resolved[key]:
                raise ValueError(f"Fire ignition {key} conflicts with experiment geometry")
            request[key] = resolved[key]
    for key in ("output_directory", "static_bundle", "output"):
        if key in request:
            request[key] = str((path.parent / request[key]).resolve())
    for key in ("incident", "perimeters", "hotspots"):
        if key not in request:
            continue
        source = dict(request[key])
        if "route" in source:
            route = source.pop("route")
            source = {**acquire(route, request["incident_id"],
                               Path(request["output_directory"]) / "sources"), **source}
        elif "path" in source:
            source["path"] = str((path.parent / source["path"]).resolve())
        request[key] = source
    return request


def run_cli(args):
    receipt = prepare(request_from_file(args.request, experiment=args.experiment, domain=args.domain))
    print(json.dumps(receipt, indent=2))
    return 0


def register_cli(subparsers):
    parser = subparsers.add_parser("fire-ignition", help="Admit public incident/perimeter/FIRMS inputs through Rust")
    parser.add_argument("request", type=Path)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--domain", type=int)
    parser.set_defaults(func=run_cli)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    register_cli(sub)
    args = parser.parse_args(["fire-ignition", *(argv if argv is not None else __import__("sys").argv[1:])])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
