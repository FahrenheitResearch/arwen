"""Declarations for optional native fields; decoding remains in Rust.

Paths are explicit GRIB collections, each of which may contain several source
times. The source manifest owns the bytes; the decoder owns time, quantity,
units and projected-grid selection. No neighboring file is discovered here.
"""
from pathlib import Path


def supplement_bindings(values, *, base: Path | None = None):
    bindings = []
    seen = set()
    for value in values or ():
        role, separator, raw_path = str(value).partition("=")
        if separator != "=" or role != "PMSL" or not raw_path:
            raise ValueError(
                "native --supplement requires PMSL=GRIB; PMSL is mean-sea-level "
                "pressure in Pa selected from the donor's GRIB metadata")
        path = Path(raw_path)
        if base is not None and not path.is_absolute():
            path = base / path
        path = path.resolve()
        if (role, path) in seen:
            raise ValueError(f"duplicate {role} donor path: {path}")
        if any(character in str(path) for character in "\t\r\n"):
            raise ValueError("native donor paths cannot contain tabs or newlines")
        seen.add((role, path))
        bindings.append((role, path))
    return tuple(bindings)


def series_supplement_suffix(values):
    return "".join(f"\t{role}={path}" for role, path in supplement_bindings(values))


def gate_supplement_fields(gate):
    fields = tuple(gate.get("supplement_fields", "").split(","))
    if fields == ("",):
        return ()
    if (fields != ("PMSL",)
            or gate.get("supplement_units") != "PMSL=Pa"
            or gate.get("supplement_alignment") != "exact_primary_grid_and_source_time"):
        raise ValueError("native supplemental field declaration has invalid units or alignment")
    return fields


def verify_supplement_receipt(receipt):
    """Recheck external donor bytes before sealing their decoded publication."""
    import hashlib
    verified = set()
    for binding in (receipt or {}).get("supplement_bindings", ()):
        path = Path(binding["path"])
        expected = binding["sha256"]
        if (path, expected) in verified:
            continue
        with path.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed != expected:
            raise ValueError(f"PMSL donor changed before publication: {path}")
        verified.add((path, expected))


def native_pressure_policy(namelist_input, declared_case=None):
    """Resolve the same explicit pressure choice before and during preparation."""
    from gpuwm.case_data import preparation_case_policy
    from gpuwm.namelist_import import parse_namelist
    setting = parse_namelist(namelist_input).get("domains", {}).get("sfcp_to_sfcp")
    if setting is not None and len(setting) != 1:
        raise ValueError("native preparation requires one boolean sfcp_to_sfcp namelist entry")
    return preparation_case_policy(
        declared_case, sfcp_to_sfcp=None if setting is None else setting[0])


def require_native_pressure_field(policy, *, bindings=(), bridge_root=None):
    """Reject an absent required quantity before static/decode/init work.

    A sealed bridge supplies its existing field declaration; it does not need
    another donor flag. Its payload authority is checked by the sealed loader.
    """
    if policy["sfcp_to_sfcp"] or bindings:
        return
    if bridge_root is not None:
        from .hrrr import _read_gate
        if "PMSL" in gate_supplement_fields(_read_gate(Path(bridge_root))):
            return
    raise ValueError(
        "sfcp_to_sfcp=false requires analyzed PMSL (mean-sea-level pressure in Pa). "
        "Add --supplement PMSL=GRIB and bind that donor in the source SHA256SUMS, "
        "or use a sealed bridge already carrying PMSL.")
