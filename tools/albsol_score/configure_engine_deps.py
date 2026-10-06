"""Bind only the scorer's three Rust vendor paths to an explicit engine tree."""
import argparse
import json
import os
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-source", required=True, type=Path)
    args = parser.parse_args()
    engine = args.engine_source.resolve(strict=True)
    relative = {
        "grib-core": "tools/grib1_bridge/vendor/grib-core",
        "netcrust": "tools/rustwx/vendor/netcrust",
        "hdf5-reader": "tools/rustwx/vendor/netcrust/vendor/hdf5-reader",
    }
    paths = {name: engine / suffix for name, suffix in relative.items()}
    for path in paths.values():
        if not (path / "Cargo.toml").is_file():
            parser.error(f"Missing Rust dependency manifest: {path / 'Cargo.toml'}")
    vendor = engine / "tools/rustwx/vendor/crates-io"
    if not vendor.is_dir():
        parser.error(f"Missing offline Cargo registry directory: {vendor}")
    manifest = Path(__file__).parent / "Cargo.toml"
    text = manifest.read_text(encoding="utf-8")
    for name, path in paths.items():
        pattern = rf'(?m)^({re.escape(name)}\s*=\s*\{{\s*path\s*=\s*)"[^"\n]*"'
        text, count = re.subn(pattern, lambda match: match[1] + json.dumps(Path(os.path.relpath(path, manifest.parent)).as_posix()), text)
        if count != 1:
            parser.error(f"Expected exactly one path for {name}, found {count}")
    manifest.write_bytes(text.encode("utf-8"))
    cargo_config = manifest.parent / ".cargo/config.toml"
    cargo_config.parent.mkdir(exist_ok=True)
    cargo_config.write_bytes((
        '[source.crates-io]\nreplace-with = "engine-vendored-sources"\n\n'
        '[source.engine-vendored-sources]\ndirectory = '
        + json.dumps(Path(os.path.relpath(vendor, manifest.parent)).as_posix()) + '\n'
    ).encode("utf-8"))
    print(json.dumps({"manifest": str(manifest), "engine_source": str(engine),
                      "dependencies": {name: str(path) for name, path in paths.items()},
                      "cargo_config": str(cargo_config), "vendor_registry": str(vendor)}, indent=2))


if __name__ == "__main__":
    main()
