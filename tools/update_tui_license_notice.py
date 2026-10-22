#!/usr/bin/env python3
"""Refresh both binary notices with the locked TUI dependencies and their texts.

Cargo resolves the shipped Linux and Windows targets offline. The inventory also
includes build dependencies; it does not claim every resolved crate is linked.

Only section 5 is rewritten, from its heading to its END line. Every byte
before the heading and after the END line is kept. Until 2.8.5 everything
from the heading to the end of the file was replaced, and the simulated-radar
vendor notices (BowEcho, the radar volume writers, the bzip2 table grant) had been
appended after the section: the next version bump, which runs this tool,
would have deleted them from the notice a compiled consumer receives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc")
SEPARATOR = "=" * 78
MARKER = SEPARATOR + "\n5.  ARWEN TUI -- LOCKED DEPENDENCIES AND LICENCE TEXTS\n"
END = SEPARATOR + "\nEND ARWEN TUI LOCKED DEPENDENCIES AND LICENCE TEXTS\n"
COPIES = ("licenses/THIRD-PARTY-LICENSES-bridge-binaries.txt",
          "tools/rustwx/assets/basemap/THIRD-PARTY-LICENSES.txt")


def inventory(root: Path) -> dict:
    """Keep the two interface locks' inventory against the shared registry."""
    vendor = root / "tools/rustwx/vendor/crates-io"
    locks, packages = {}, {}
    for workspace in ("arwen-launchpad", "arwen-tui"):
        raw = (root / "tools" / workspace / "Cargo.lock").read_bytes()
        locks[workspace] = hashlib.sha256(raw).hexdigest()
        for item in tomllib.loads(raw.decode())["package"]:
            if "source" not in item:
                continue
            directory = vendor / (item["name"] + "-" + item["version"])
            if not directory.is_dir():
                directory = vendor / item["name"]
            package = tomllib.loads((directory / "Cargo.toml").read_text(encoding="utf-8"))["package"]
            checksum = json.loads((directory / ".cargo-checksum.json").read_text())["package"]
            if ((package["name"], package["version"]) != (item["name"], item["version"])
                    or checksum != item["checksum"]):
                raise ValueError(f"{directory.name}: package identity or checksum differs from the lock")
            key = item["name"], item["version"]
            record = {"name": item["name"], "version": item["version"],
                      "package_sha256": checksum, "license": package.get("license"),
                      "license_file": package.get("license-file")}
            if packages.setdefault(key, record) != record:
                raise ValueError(f"{directory.name}: interface locks disagree")
    return {"command": "python tools/update_tui_license_notice.py (shared registry maintenance: tools/rustwx/VENDOR.md)",
            "locks": locks, "packages": [packages[key] for key in sorted(packages)]}


def resolved_packages(root: Path) -> tuple[list[dict], dict[str, int]]:
    packages, counts = {}, {}
    for target in TARGETS:
        result = subprocess.run(
            ["cargo", "metadata", "--locked", "--offline", "--format-version", "1",
             "--filter-platform", target], cwd=root / "tools/arwen-tui",
            check=True, stdout=subprocess.PIPE, encoding="utf-8")
        document = json.loads(result.stdout)
        selected = {node["id"] for node in document["resolve"]["nodes"]}
        rows = [row for row in document["packages"]
                if row["id"] in selected and row["name"] != "arwen-tui"]
        counts[target] = len(rows)
        for row in rows:
            package = packages.setdefault(row["id"], {**row, "notice_targets": []})
            package["notice_targets"].append(target)
    return sorted(packages.values(), key=lambda p: (p["name"], p["version"])), counts


def render_section(root: Path, packages: list[dict], counts: dict[str, int]) -> str:
    texts, inventory = {}, []
    vendor = (root / "tools/rustwx/vendor/crates-io").resolve()
    for package in packages:
        directory = Path(package["manifest_path"]).resolve().parent
        directory.relative_to(vendor)
        name = f"{package['name']} {package['version']}"
        files = sorted(path for path in directory.iterdir()
                       if path.is_file() and path.name.upper().startswith(
                           ("LICENSE", "LICENCE", "COPYING", "COPYRIGHT", "NOTICE")))
        if not files or not package.get("license"):
            raise ValueError(f"{name}: missing licence grant or local notice text")
        references = []
        for path in files:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            entry = texts.setdefault(digest, {"label": f"TUI-{len(texts) + 1:03d}",
                                            "text": raw.decode("utf-8-sig"), "sources": []})
            entry["sources"].append(path.relative_to(root).as_posix())
            references.append(entry["label"])
        inventory.append(f"  {name} | {package['license']} | [{','.join(references)}]")
    lines = [MARKER.rstrip(), "-" * 78, "",
             "Generated by: python tools/update_tui_license_notice.py", "",
             f"{len(packages)} distinct external crate name-and-version pairs are resolved for",
             "the supported targets below. This includes build-time dependencies and",
             "target-specific dependencies; it is not a count of linked runtime crates.",
             "All LICENSE, LICENCE, COPYING, COPYRIGHT and NOTICE files at each",
             "resolved vendored package root are reproduced, deduplicated by raw SHA-256.",
             "The TUI itself uses the repository Apache-2.0 licence.", ""]
    for name in ("Cargo.toml", "Cargo.lock"):
        path = root / "tools/arwen-tui" / name
        lines.append(f"tools/arwen-tui/{name} sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
    lines += [f"{target}: {counts[target]} external packages" for target in TARGETS]
    lines += ["", "Inventory (grant expressions are copied from the vendored Cargo metadata):", ""]
    lines += inventory
    lines += ["", f"{len(texts)} distinct licence and attribution texts follow."]
    for digest, entry in texts.items():
        lines += ["", "-" * 78, f"[{entry['label']}] sha256 {digest}", "As carried by:"]
        lines += ["  " + source for source in entry["sources"]]
        lines += ["-" * 78, "", entry["text"].replace("\r\n", "\n").rstrip(), ""]
    return "\n".join(lines).rstrip() + "\n"


def compose(existing: str, section: str) -> str:
    """``existing`` with section 5 replaced by ``section`` and nothing else moved.

    A notice that has no section 5 yet gains one at its end. A notice whose
    section 5 has no END line is refused: where the section stops cannot be
    read, and replacing through to the end of the file is how the notices
    after it would be deleted.
    """
    head, heading, rest = existing.partition(MARKER)
    after = ""
    if heading:
        _, end, after = rest.partition(END)
        if not end:
            raise SystemExit(
                "TUI licence notice has section 5 without its END line, so where the "
                "section stops cannot be read; rewriting from the heading to the end "
                "of the file would delete every notice after it (the simulated-radar "
                "vendor notices sat there). Restore the line pair\n" + END)
    return head.rstrip() + "\n\n" + section + "\n" + END + after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without writing either copy")
    args = parser.parse_args()
    manifest = (json.dumps(inventory(ROOT), indent=2) + "\n").encode()
    packages, counts = resolved_packages(ROOT)
    first = (ROOT / COPIES[0]).read_bytes().decode("utf-8")
    expected = compose(first, render_section(ROOT, packages, counts)).encode("utf-8")
    if args.check:
        if (ROOT / "tools/arwen-ui-vendor/manifest.json").read_bytes() != manifest:
            raise SystemExit("TUI vendor inventory is stale; run tools/update_tui_license_notice.py")
        if any((ROOT / path).read_bytes() != expected for path in COPIES):
            raise SystemExit("TUI licence notices are stale; run tools/update_tui_license_notice.py")
    else:
        (ROOT / "tools/arwen-ui-vendor/manifest.json").write_bytes(manifest)
        for path in COPIES:
            (ROOT / path).write_bytes(expected)
    print(f"PASS: {len(packages)} locked TUI dependency packages, two identical binary notices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
