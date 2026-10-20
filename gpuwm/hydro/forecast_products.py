"""Hydro products at the end of every coupled forecast, with nothing new to type.

Runs once at the forecast's output boundary, never in the step loop: the
realization's GPU history is published by the Rust writer
(``gpuwm.hydro.output``), its renderer request gets the gauges the fabric
matched, and the Rust renderer (``rw_wrfbatch`` NextGen products, through
``gpuwm.nextgen_products``) draws hydrographs and streamflow into
``<forecast out>/hydro-products/<domain>/<product>/<valid-day>/``. Python never
reads a discharge value. Every reason a product is missing is written in the
receipt and printed, never skipped silently.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

SCHEMA = "gpuwm-hydro-forecast-products.v1"


def render_domain(dx_m, grid_id=1):
    """Render-folder domain label, e.g. ``d01-3km`` (render folder law)."""
    km = float(dx_m) / 1000.0
    label = f"{km:g}km" if km >= 1.0 else f"{float(dx_m):g}m"
    return f"d{int(grid_id):02d}-{label}"


def matched_gauges(manifest_path, node_ids):
    """Gauges from the prepared fabric's gauges.json that sit on the roster.

    The renderer roster uses decimal publication IDs ("2420816"); the fabric
    names the gauge's flowpath ("wb-2420816"). Only exact ID matches count.
    """
    if manifest_path is None:
        return [], "no prepared fabric manifest bound to this realization"
    gauges_path = Path(manifest_path).parent / "gauges.json"
    if not gauges_path.is_file():
        return [], f"prepared fabric has no gauges.json ({gauges_path.parent})"
    roster = set(node_ids)
    matched, stations = [], {}
    for row in json.loads(gauges_path.read_text()).get("rows", []):
        feature = str(row.get("feature_id", ""))
        local = feature.rsplit("-", 1)[-1]
        if local.isdigit() and str(int(local)) in roster and str(int(local)) not in stations:
            matched.append(str(int(local)))
            stations[str(int(local))] = row.get("station_id")
    return matched, None if matched else "no fabric gauge lies on the routed roster"


def publish(physics, settings, *, outdir, dx_m, renderer_runner=None):
    """Write the hydro history and its rendered products; return the receipt."""
    started = time.perf_counter()
    driver = getattr(physics, "hydro", None)
    if driver is None:
        return {"schema": SCHEMA, "status": "not-run",
                "reason": "hydro was not attached to this forecast (see hydro_activation)"}
    from .output import History
    histories = [name for name, component in driver.components.items()
                 if isinstance(component, History)]
    if not histories:
        gap = ("realization has no gpuwm.hydro.output:prepare_history stage, so there is no "
               "GPU series to publish (owners c-hydro-routing, nwm-v4-config)")
        print(f"hydro products: {gap}", file=sys.stderr)
        return {"schema": SCHEMA, "status": "gap", "gap": gap}
    receipt = {"schema": SCHEMA, "histories": histories, "products": []}
    try:
        import importlib
        target = (settings or {}).get("output_factory", "gpuwm.hydro.output:prepare")
        module, _, name = target.partition(":")
        writer = getattr(importlib.import_module(module), name)(
            settings=settings, driver=driver, outdir=Path(outdir) / "hydro")
        written = writer.finish(driver, receipt={"scope": "coupled forecast hydro history"})
        receipt["history_output"] = written["output_directory"]
        manifest = (settings or {}).get("settings_bindings", {}).get("fabric_manifest")
        domain = render_domain(dx_m)
        if renderer_runner is None:
            from gpuwm.nextgen_products import run_nextgen_products as renderer_runner
        for history in histories:
            request_path = Path(written["output_directory"]) / history / "renderer-request.json"
            if not request_path.is_file():
                receipt["products"].append({"history": history, "status": "gap",
                                            "gap": "history formats exclude renderer_request"})
                continue
            request = json.loads(request_path.read_text())
            gauges, why = matched_gauges(manifest, request.get("node_ids", []))
            request["gauges"] = gauges
            request["domain"] = domain
            request["out_dir"] = str((Path(outdir) / "hydro-products").resolve())
            request_path.write_text(json.dumps(request, indent=2) + "\n")
            entry = {"history": history, "request": str(request_path), "domain": domain,
                     "gauges": gauges, "gauge_gap": why}
            if why:
                print(f"hydro products: {why}", file=sys.stderr)
            try:
                entry["renderer_receipt"] = str(renderer_runner(request_path))
                entry["status"] = "partial" if why else "rendered"
            except (OSError, RuntimeError) as error:
                entry["status"] = "gap"
                entry["gap"] = f"Rust renderer: {error}"
                print(f"hydro products: {entry['gap']}", file=sys.stderr)
            receipt["products"].append(entry)
        receipt["status"] = ("rendered" if all(p.get("status") == "rendered"
                                               for p in receipt["products"]) else "partial")
    except Exception as error:  # noqa: BLE001 - the forecast output is kept; the failure is named
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(f"hydro products failed: {receipt['error']}", file=sys.stderr)
    receipt["seconds"] = time.perf_counter() - started
    receipt["parallelism"] = "output boundary only: Rust writer and Rust renderer (Rayon); no step-loop work"
    return receipt


def finalize_forecast(physics, *, outdir, dx_m, forecast_seconds, atmosphere_steps):
    """Publish the resident hydro state before a config-driven route releases it.

    The ERA5/config front door also runs through runtime.integrate_prepared_case;
    the prepared-cache reporters alone cannot publish that route's resident state.
    """
    if getattr(physics, "hydro", None) is None:
        return None
    from .cost import receipt, refuse_if_over_budget
    from .runtime import current_settings
    block = {"schema": "gpuwm-coupled-forecast.v1",
             "hydro_cost": receipt(physics, forecast_seconds=forecast_seconds,
                                   atmosphere_steps=atmosphere_steps)}
    error = None
    try:
        block["diagnostics"] = physics.hydro.check_health()
    except RuntimeError as exc:
        block["diagnostics"] = getattr(exc, "diagnostics", {})
        block["health_error"] = str(exc)
        error = exc
    block["hydro_products"] = publish(physics, current_settings(), outdir=outdir, dx_m=dx_m)
    path = Path(outdir) / "hydro-receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(block, indent=2) + "\n")
    if error is not None:
        raise error
    refuse_if_over_budget(block["hydro_cost"])
    if block["hydro_products"]["status"] != "rendered":
        raise RuntimeError(f"coupled hydro products incomplete; see {path}")
    return block
