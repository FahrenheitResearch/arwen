"""Fire-danger orchestration. Rust owns weather I/O, indices and persistence.

Run a native request with ``python -m gpuwm.fire_danger_state run REQUEST.json``.
This door serves Canadian FWI, NFDRS 2016 and KBDI with carried state.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

PRODUCTS = ("fwi_ffmc", "fwi_dmc", "fwi_dc", "fwi_isi", "fwi_bui", "fwi_fwi", "kbdi", "nfdrs_mc1", "nfdrs_mc10", "nfdrs_mc100", "nfdrs_mc1000", "nfdrs_live_herb", "nfdrs_live_wood", "nfdrs_gsi", "nfdrs_erc", "nfdrs_bi", "nfdrs_sc")


def binary() -> Path:
    from gpuwm.bridges import fetch_on_demand, on_demand_failure
    found = _find()
    if found is None and not os.environ.get("GPUWM_FIRE_INDICES") and fetch_on_demand("rw_fire_indices"):
        # On demand since 2.9.4 (PyPI per-file limit): pinned in this release's
        # bundle, staged by the first fire-danger request that needs it.
        found = _find()
    if found is None:
        failure = on_demand_failure("rw_fire_indices")
        raise FileNotFoundError("Build rw_fire_indices with cargo build --release -p fire-indices in tools/rustwx, or set GPUWM_FIRE_INDICES to the binary."
                                + (f" ({failure})" if failure else ""))
    return found


def _find() -> Path | None:
    from gpuwm.bridges import (accept_resolved, default_bridge_dir,
                                legacy_bridge_candidates, packaged_bridge_dir,
                                executable_name)
    root = Path(__file__).resolve().parent.parent
    override = os.environ.get("GPUWM_FIRE_INDICES")
    filename = executable_name("rw_fire_indices")
    candidates = (Path(override).expanduser(),) if override else (
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
        *legacy_bridge_candidates(filename),
        root / "libexec/bridges" / filename,
        root / "tools/rustwx/target/release" / filename,
        root / "tools/rustwx/target/debug" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            resolved = accept_resolved(candidate.resolve())
            result = subprocess.run([str(resolved), "--abi"], capture_output=True, text=True, check=True)
            if result.stdout.strip() != "GPUWM_FIRE_INDICES_V1":
                raise ValueError(f"Fire-danger executable has an incompatible state contract: {resolved}")
            return resolved
    return None


def run(request: Path, *, render: bool = False) -> dict:
    subprocess.run([str(binary()), str(request)], check=True)
    config = json.loads(request.read_text())
    receipt = json.loads(Path(config["receipt"]).read_text())
    if render:
        from gpuwm import rustwx
        renderer = rustwx.find_renderer()
        if renderer is None:
            raise FileNotFoundError("rw_wrfbatch is required for fire-danger maps")
        output = Path(config["output"])
        for daily in receipt["daily"]:
            for product in receipt["products"]:
                result = rustwx.run_cf_renderer(renderer, output, variable=product,
                    out_dir=output.parent / "maps", valid=daily["valid"],
                    receipt=output.parent / f"render-{product}-{daily['local_date']}.json")
                if result:
                    raise RuntimeError(f"Native renderer failed for {product} at {daily['valid']}")
    return receipt


def spinup(config_path: Path) -> dict:
    """Acquire only needed ERA5 fields via the existing generic Rust route.

    ERA5 data are research opt-in under CC-BY-4.0, not a shipping default.
    Configuration names start (first hourly interval after standard noon),
    days, area (S,W,N,E), output root, member, utc_offset_hours, initial_kbdi,
    annual_rain_mm and annual_rain_source. Every acquisition retains a receipt.
    """
    from datetime import datetime, timedelta, timezone
    from gpuwm.era5_arco import ARCO_STORE
    from gpuwm.fetch import ERA5_PRESSURE_LEVELS_HPA
    from gpuwm.zarr_bridge import extract_regular_zarr
    config = json.loads(config_path.read_text())
    if config.get("research_era5_license") != "CC-BY-4.0":
        raise ValueError("ERA5 spin-up requires research_era5_license=CC-BY-4.0")
    start = datetime.fromisoformat(config["start"].replace("Z", "+00:00"))
    if start.tzinfo is None:
        raise ValueError("ERA5 start requires an explicit UTC timezone")
    start = start.astimezone(timezone.utc)
    local_prior = start - timedelta(hours=1) + timedelta(hours=config["utc_offset_hours"])
    if local_prior.hour != 12 or start.minute or start.second or start.microsecond:
        raise ValueError("ERA5 start must be the hourly interval after local standard noon")
    if type(config["days"]) is not int or config["days"] < 14:
        raise ValueError("ERA5 spin-up needs at least 14 whole days")
    out = Path(config["out"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fields = (("2m_temperature", "T2", "K"), ("2m_dewpoint_temperature", "D2", "K"),
              ("10m_u_component_of_wind", "U10", "m s-1"),
              ("10m_v_component_of_wind", "V10", "m s-1"),
              ("total_precipitation", "TP", "m"),
              ("surface_solar_radiation_downwards", "SSRD", "J m-2"))
    inputs = []
    for day in range(config["days"]):
        request = {"store": ARCO_STORE, "times": [
            (start + timedelta(hours=24 * day + hour)).isoformat().replace("+00:00", "Z")
            for hour in range(24)], "area": config["area"],
            "expected_levels_hpa": list(ERA5_PRESSURE_LEVELS_HPA),
            "fields": [{"source": s, "output": n, "units": u, "pressure": False}
                       for s, n, u in fields]}
        target = out / f"forcing-{day:04d}.nc"
        request_path = out / f"acquisition-{day:04d}.json"
        # Acquisition can resume only when the native receipt binds this exact
        # request. Missing receipts are incomplete downloads, never reusable data.
        result_path = request_path.with_suffix(".result.json")
        hash_path = request_path.with_suffix(".sha256")
        reusable = (hash_path.is_file() and target.is_file() and request_path.is_file() and result_path.is_file()
                    and json.loads(request_path.read_text()) == request
                    and json.loads(result_path.read_text()).get("schema") == "arwen.regular-forcing.v1"
                    and hash_path.read_text().strip() == _sha256(target))
        if not reusable:
            if target.exists():
                raise FileExistsError(f"Incomplete prior ERA5 acquisition: {target}")
            extract_regular_zarr(request, request_path=request_path, output=target)
            hash_path.write_text(_sha256(target) + "\n")
        inputs.append(str(target))
    native = {key: config[key] for key in ("member", "utc_offset_hours", "initial_kbdi",
                                           "annual_rain_source")}
    for key in ("annual_rain_mm", "annual_rain_grid", "initial_fwi", "nfdrs", "compute_backend"):
        if key in config:
            native[key] = config[key]
    native.update(inputs=inputs, state_out=str(out / "state.json"),
                  output=str(out / "indices.nc"), receipt=str(out / "receipt.json"))
    request_path = out / "run.json"
    request_path.write_text(json.dumps(native, indent=2) + "\n")
    return run(request_path)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("run")
    command.add_argument("request", type=Path)
    command.add_argument("--render", action="store_true")
    command = sub.add_parser("spinup")
    command.add_argument("request", type=Path)
    for name in ("verify", "verify-nfdrs", "climatology", "compare"):
        command = sub.add_parser(name)
        command.add_argument("request", type=Path)
    args = parser.parse_args()
    if args.command == "spinup":
        spinup(args.request)
    elif args.command in ("verify", "verify-nfdrs", "climatology", "compare"):
        subprocess.run([str(binary()), args.command, str(args.request)], check=True)
    else:
        run(args.request, render=args.render)


if __name__ == "__main__":
    main()
