"""Explicit preparation donors carried by the human launch and run plan."""

from pathlib import Path


def bindings(values, *, base):
    from gpuwm.source_cli import _role_binding_errors

    if not isinstance(values, (list, tuple)):
        raise ValueError("--supplement bindings must be a list of ROLE=PATH values")
    values = [str(value) for value in values]
    errors = _role_binding_errors(values, "--supplement", unique=False)
    if errors:
        raise ValueError("; ".join(errors))
    result = []
    for value in values:
        role, raw = value.split("=", 1)
        path = Path(raw)
        if not path.is_absolute():
            path = Path(base) / path
        path = path.resolve()
        if any(char in str(path) for char in "\t\r\n"):
            raise ValueError("supplement paths cannot contain tabs or newlines")
        binding = f"{role}={path}"
        if binding in result:
            raise ValueError(f"duplicate supplement binding: {binding}")
        if not path.is_file():
            raise ValueError(f"declared supplement is missing: {path}")
        result.append(binding)
    return result


def validate_route(values, *, chain, source_root=None):
    if not values:
        return
    if chain == "prepared:hrrr":
        from gpuwm.ingest.native_supplements import supplement_bindings

        for _, path in supplement_bindings(values):
            if source_root is not None and not path.is_relative_to(Path(source_root).resolve()):
                raise ValueError(
                    f"HRRR supplement {path} must be inside the source download directory "
                    f"{Path(source_root).resolve()}. Put the donor in that directory, or "
                    "use --data-dir DIR containing the donor; the preparation source "
                    "manifest binds all primary and donor files relative to that directory.")
    elif chain != "prepared:staged":
        raise ValueError(
            f"--supplement is not used by the {chain} route. "
            "Supply donors when preparing an HRRR or mapped source bundle.")


def hrrr_source_manifest(values, *, source_root, fetched_manifest, output):
    """Bind explicit donor bytes in a run-local manifest; retain fetch receipts."""
    from gpuwm.fetch import sha256_file
    from gpuwm.fetch_guard import atomic_write_text
    from gpuwm.ingest.native_supplements import supplement_bindings
    from tools.hrrr_pipeline import _parse_manifest

    if not values:
        return fetched_manifest
    validate_route(values, chain="prepared:hrrr", source_root=source_root)
    entries = _parse_manifest(fetched_manifest)
    for _, path in supplement_bindings(values):
        relative = path.relative_to(Path(source_root).resolve()).as_posix()
        digest = sha256_file(path)
        if relative in entries and entries[relative] != digest:
            raise ValueError(f"supplement differs from the fetched source receipt: {path}")
        entries[relative] = digest
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, "".join(
        f"{digest}  {name}\n" for name, digest in sorted(entries.items())))
    return output
