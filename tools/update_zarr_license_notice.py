#!/usr/bin/env python3
"""Inventory the offline Zarr closure and retain its dependency licence texts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "tools/zarr_bridge"
TARGETS = ("x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc")
COPIES = ("licenses/THIRD-PARTY-LICENSES-zarr-binary.txt",
          "tools/rustwx/assets/basemap/THIRD-PARTY-LICENSES-zarr-binary.txt")
PREFIXES = ("LICENSE", "LICENCE", "COPYING", "COPYRIGHT", "NOTICE")


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def inventory():
    lock = (WORKSPACE / "Cargo.lock").read_bytes()
    packages = []
    for item in tomllib.loads(lock.decode())["package"]:
        if "source" not in item:
            continue
        directory = WORKSPACE / "vendor/crates-io" / (item["name"] + "-" + item["version"])
        package = tomllib.loads((directory / "Cargo.toml").read_text(encoding="utf-8"))["package"]
        record = json.loads((directory / ".cargo-checksum.json").read_text())
        if record["package"] != item["checksum"]:
            raise ValueError(f"{directory.name}: registry checksum differs from the lock")
        packages.append({"name": item["name"], "version": item["version"], "package_sha256": item["checksum"],
                         "license": package.get("license"), "license_file": package.get("license-file")})
    return {"schema": "arwen.zarr-vendor.v1", "lock_sha256": sha(lock),
            "command": "cd tools/zarr_bridge && cargo vendor --locked --offline --versioned-dirs vendor/crates-io",
            "packages": packages}


def selected_packages():
    selected, counts = {}, {}
    for target in TARGETS:
        done = subprocess.run(["cargo", "metadata", "--locked", "--offline", "--format-version", "1",
                               "--filter-platform", target], cwd=WORKSPACE, check=True,
                              stdout=subprocess.PIPE, encoding="utf-8")
        document = json.loads(done.stdout)
        ids = {node["id"] for node in document["resolve"]["nodes"]}
        packages = [package for package in document["packages"] if package["id"] in ids]
        counts[target] = len(packages)
        for package in packages:
            selected[package["id"]] = package
        if target.endswith("linux-gnu"):
            openssl = next(node for node in document["resolve"]["nodes"] if "#openssl-sys@" in node["id"])
            if "vendored" not in openssl["features"]:
                raise ValueError("Linux Zarr transport must link the vendored OpenSSL closure")
    return sorted(selected.values(), key=lambda package: (package["name"], package["version"])), counts


def render(packages, counts):
    texts, rows = {}, []
    supplement_root = WORKSPACE / "additional-notices"
    supplements = json.loads((supplement_root / "SOURCES.json").read_text())["sources"]
    for package in packages:
        directory = Path(package["manifest_path"]).resolve().parent
        directory.relative_to(ROOT)
        # Include nested native-library notices as well as Rust package roots.
        # Path dependencies may have a workspace vendor directory; those crates
        # have their own resolved entry, so avoid sweeping unrelated sources.
        files = sorted(path for path in directory.rglob("*") if path.is_file()
                       and path.name.upper().startswith(PREFIXES)
                       and "target" not in path.relative_to(directory).parts
                       and ("vendor" not in path.relative_to(directory).parts or directory.is_relative_to(WORKSPACE / "vendor")))
        for supplement in supplements:
            if package["name"] in supplement["packages"]:
                path = supplement_root / supplement["file"]
                if sha(path.read_bytes()) != supplement["sha256"]:
                    raise ValueError(f"Supplemental upstream notice changed: {path.name}")
                files.append(path)
        if package["name"] == "rw-zarr" and not files:
            files = [ROOT / "LICENSE"]
        if package["name"] in ("grib-core", "netcdf-writer") and not files:
            files = [ROOT / "tools/rustwx/LICENSE"]
        if package["name"] == "mapped-engine" and not files:
            # This already shipped component remains covered by the original
            # bridge notice. Preserve that record, including its own source
            # and licence distinctions, instead of assigning it a new grant.
            files = [ROOT / "licenses/THIRD-PARTY-LICENSES-bridge-binaries.txt"]
        if not files:
            raise ValueError(f"{package['name']} {package['version']}: no retained licence text")
        references = []
        for path in files:
            raw = path.read_bytes()
            digest = sha(raw)
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            entry = texts.setdefault(digest, {"label": f"ZARR-{len(texts) + 1:03d}", "text": text, "sources": []})
            source = path.relative_to(ROOT).as_posix()
            if source not in entry["sources"]:
                entry["sources"].append(source)
            if entry["label"] not in references:
                references.append(entry["label"])
        rows.append(f"{package['name']} {package['version']} | {package.get('license') or 'see retained grant'} | [{','.join(references)}]")
        if package.get("authors"):
            rows.append("  Authors as declared: " + "; ".join(package["authors"]))
    lines = ["RW_ZARR: LOCKED DEPENDENCIES AND LICENCE TEXTS", "=" * 78,
             "Generated by: python tools/update_zarr_license_notice.py", "",
             "This inventory covers the resolved Linux and Windows build closures,",
             "including build-time packages. It is not a claim that every resolved",
             "package is linked into both binaries. The first-party reader uses the",
             "repository Apache-2.0 licence; dependencies retain the grants below.", "",
             f"Cargo.lock sha256 {sha((WORKSPACE / 'Cargo.lock').read_bytes())}"]
    lines += [f"{target}: {count} resolved packages" for target, count in counts.items()]
    lines += ["", "Supplemental upstream notices omitted from Cargo package archives:"]
    lines += [f"{row['url']} sha256 {row['sha256']}" for row in supplements]
    lines += ["", "Inventory:", *rows]
    for digest, entry in texts.items():
        lines += ["", "-" * 78, f"[{entry['label']}] sha256 {digest}", "As carried by:",
                  *["  " + source for source in entry["sources"]], "-" * 78, entry["text"].replace("\r\n", "\n").rstrip()]
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = (json.dumps(inventory(), indent=2) + "\n").encode()
    packages, counts = selected_packages()
    notice = render(packages, counts)
    outputs = {WORKSPACE / "vendor-manifest.json": manifest, **{ROOT / name: notice for name in COPIES}}
    for path, payload in outputs.items():
        if args.check:
            if not path.is_file() or path.read_bytes() != payload:
                raise SystemExit(f"Zarr dependency notice is stale: {path.relative_to(ROOT)}")
        else:
            path.write_bytes(payload)
    print(f"PASS: {len(packages)} resolved packages, offline vendor inventory and both native notices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
