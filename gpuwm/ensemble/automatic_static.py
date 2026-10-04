"""Share original geography before source-dependent terrain/land repairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpuwm.ensemble.automatic_sources import _canonical
from gpuwm.ensemble.physical_store import digest_file


def build_shared_static(*, runner, arguments, output_root):
    """Call the registered route's existing geography build, once.

    No decoded atmosphere, source orography, RootTerrainBlend, soil initial
    state or source-prepared static is consumed. Those remain owned by each
    native source producer. A shared high-resolution overlay is bound to its
    exact configuration, grid and date by the existing native receipt.

    Which build that is comes from the route's row in the preparation
    runner table: its own geography executable, or baseline fields built
    in process and finished here.
    """
    from gpuwm.ensemble.automatic_preparation import _option
    from gpuwm.source_cli import preparation_runners, shared_geography_target
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    config = _option(arguments, "--experiment-config")
    wps = _option(arguments, "--wps-namelist")
    geog = _option(arguments, "--geog-root")
    if geog is None:
        raise ValueError("shared geography requires the original geog-root authority")
    row = preparation_runners().get(runner)
    producer = None if row is None else row.shared_geography
    if producer is None or not producer.builds:
        raise ValueError(f"{runner} has no shared native geography producer")
    cache = root / "native-static.npz"
    receipt = root / "native-static-receipt.json"
    if producer.command is not None:
        import subprocess
        subprocess.run(producer.command(lambda flag: _option(arguments, flag), cache, receipt),
                       check=True)
    else:
        if config is None or wps is None:
            raise ValueError("shared geography requires the exact experiment-config and WPS authorities")
        from gpuwm.native_wrf_contract import (
            write_native_static_cache, write_native_geometry_receipt, verify_native_static_receipt)
        from gpuwm.static.highres_production import apply_prepared_highres
        experiment, grid, cfg, highres = shared_geography_target(config, wps)
        fields, origin, landuse = producer.fields(Path(wps), Path(geog), grid, cfg, highres)
        fields, overlay = apply_prepared_highres(fields, grid, config=highres,
            domain_id=1, case_date=experiment.start_time.date(), landuse_attrs=landuse,
            baseline_receipt=origin)
        write_native_static_cache(cache, fields)
        document = write_native_geometry_receipt(receipt, grid, cfg, cache)
        if origin.get("terrain_smoothing") is not None:
            document["terrain_smoothing"] = origin["terrain_smoothing"]
        if isinstance(overlay, dict) and overlay.get("highres") is not None:
            document["highres"] = overlay["highres"]
        document["ensemble_preblend_static"] = {
            "schema": "gpuwm-ensemble-preblend-static.v1", "runner": runner,
            "geography_only": True, "root_terrain_blend_applied": False,
            "highres_overlay_applied": "highres" in document, "ordinary_origin": origin}
        receipt.write_text(_canonical(document) + "\n", encoding="utf-8")
        verify_native_static_receipt(receipt, cache, grid, cfg)
    result = {"schema": "gpuwm-ensemble-shared-geography.v1", "status": "PASS",
              "runner": runner, "cache": str(cache), "cache_sha256": digest_file(cache),
              "receipt": str(receipt), "receipt_sha256": digest_file(receipt),
              "source_terrain_blending": "each original source producer",
              "source_land_surface_repairs": "each original source producer"}
    (root / "shared-geography.json").write_text(_canonical(result) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_bytes())
    result = build_shared_static(runner=request["runner"], arguments=request["arguments"], output_root=args.output_root)
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
