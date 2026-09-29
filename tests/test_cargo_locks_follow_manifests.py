"""Every Rust workspace's Cargo.lock names the dependencies its manifests declare.

The installers and the wheel build compile each workspace with
``cargo build --release --locked --offline`` (install.sh builds
tools/zarr_bridge that way).  ``--locked`` refuses to touch a lock file
that no longer matches the manifests, so a crate that gains a dependency
while one of the lock files that reaches it by path is left behind stops
the install outright: a mapped-engine change added libc and windows-sys,
rw_wps's lock followed and zarr_bridge's did not, and a checkout install
stopped at that build with "cannot update the lock file".

This check is hermetic: it reads Cargo.toml and Cargo.lock files and
needs no Rust toolchain, so it runs with the rest of the Python tests
rather than only when an installer runs.  For each workspace it follows
the path dependencies from the root manifest and asks that every
dependency a path crate declares appears in that crate's entry in the
workspace's lock.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

WORKSPACES = (
    "tools/arwen-launchpad",
    "tools/arwen-tui",
    "tools/grib1_bridge",
    "tools/region_global_dealias",
    "tools/rustwx",
    "tools/rw_wps",
    "tools/zarr_bridge",
)

_SECTIONS = ("dependencies", "build-dependencies")


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _members(root: Path, manifest: dict) -> list[Path]:
    found = []
    if "package" in manifest:
        found.append(root / "Cargo.toml")
    for pattern in manifest.get("workspace", {}).get("members", []):
        if pattern in (".", "./", ""):
            continue
        for candidate in sorted(root.glob(pattern)):
            if (candidate / "Cargo.toml").is_file():
                found.append(candidate / "Cargo.toml")
    return found


def _declared(manifest: dict, sections: tuple[str, ...],
              workspace_deps: dict) -> dict[str, dict]:
    tables = [manifest.get(name, {}) for name in sections]
    for target in manifest.get("target", {}).values():
        tables.extend(target.get(name, {}) for name in sections)
    declared: dict[str, dict] = {}
    for table in tables:
        for key, spec in table.items():
            spec = spec if isinstance(spec, dict) else {"version": spec}
            if spec.get("workspace"):
                inherited = workspace_deps.get(key, {})
                inherited = (inherited if isinstance(inherited, dict)
                             else {"version": inherited})
                spec = {**inherited, **{k: v for k, v in spec.items()
                                        if k != "workspace"}}
            declared[spec.get("package", key)] = spec
    return declared


def _lock_dependency_names(entry: dict) -> set[str]:
    return {item.split(" ")[0] for item in entry.get("dependencies", [])}


def _stale_entries(workspace: str) -> list[str]:
    root = REPOSITORY_ROOT / workspace
    root_manifest = _load(root / "Cargo.toml")
    workspace_deps = root_manifest.get("workspace", {}).get("dependencies", {})
    lock = _load(root / "Cargo.lock")
    path_entries = {entry["name"]: entry for entry in lock["package"]
                    if "source" not in entry}

    members = _members(root, root_manifest)
    queue = [(path, True) for path in members]
    seen: set[Path] = set()
    stale = []
    while queue:
        manifest_path, is_member = queue.pop()
        manifest_path = manifest_path.resolve()
        if manifest_path in seen:
            continue
        seen.add(manifest_path)
        manifest = _load(manifest_path)
        sections = _SECTIONS + (("dev-dependencies",) if is_member else ())
        declared = _declared(manifest, sections, workspace_deps)
        if not is_member:
            # Cargo resolves every feature of a workspace member but only
            # the enabled ones of a crate reached by path, so an optional
            # dependency of such a crate is absent from the lock until a
            # feature turns it on.
            declared = {key: spec for key, spec in declared.items()
                        if not spec.get("optional")}
        for spec in declared.values():
            if "path" in spec:
                queue.append((manifest_path.parent / spec["path"] / "Cargo.toml",
                              False))
        name = manifest["package"]["name"]
        entry = path_entries.get(name)
        if entry is None:
            stale.append(f"{name} ({manifest_path.relative_to(REPOSITORY_ROOT)}) "
                         f"has no entry in {workspace}/Cargo.lock")
            continue
        missing = sorted(set(declared) - _lock_dependency_names(entry))
        if missing:
            stale.append(f"{name} declares {', '.join(missing)}, which "
                         f"{workspace}/Cargo.lock does not list for it")
    return stale


@pytest.mark.parametrize("workspace", WORKSPACES)
def test_the_lock_names_every_declared_dependency(workspace):
    stale = _stale_entries(workspace)
    assert not stale, (
        f"{workspace}/Cargo.lock no longer matches its manifests, so "
        f"`cargo build --locked` refuses it: " + "; ".join(stale)
        + f". Run `cargo metadata --offline --format-version 1` in "
        f"{workspace} and commit the refreshed lock.")


def test_every_workspace_with_a_lock_is_checked():
    locks = {
        str(path.parent.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
        for path in (REPOSITORY_ROOT / "tools").glob("*/Cargo.lock")
    }
    assert locks == set(WORKSPACES)
