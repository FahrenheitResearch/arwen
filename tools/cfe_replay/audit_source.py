#!/usr/bin/env python3
"""Audit CFE source metadata before the referee chooses a capture boundary.

This reads pinned Git source text only. It never reads, transforms, hashes, or
compares numerical arrays; those operations belong to tools/nextgen_replay.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import time


PIN = "f9182dfb3d81c407c66c5b782114173893c30cf2"
REPOSITORY = "https://github.com/NOAA-OWP/cfe"
SOURCE_FILES = (
    "include/bmi_cfe.h", "include/cfe.h", "include/conceptual_reservoir.h",
    "include/nash_cascade.h", "src/bmi_cfe.c", "src/giuh.c",
)


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True)


def uncomment(source):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)


def audit(source):
    started = time.perf_counter()
    documents = {name: git(source, "show", f"{PIN}:{name}") for name in SOURCE_FILES}
    licence = git(source, "show", f"{PIN}:LICENSE")
    if "Apache License" not in licence or "Version 2.0, January 2004" not in licence:
        raise ValueError("pinned CFE LICENSE no longer matches the inspected Apache-2.0 text")

    structs = {}
    declaration = re.compile(
        r"(struct\s+\w+|double|int|long(?:\s+int)?|char|surface_water_partition_type)"
        r"\s*(\*)?\s*(\w+)\s*;"
    )
    for name, document in documents.items():
        if not name.endswith(".h"):
            continue
        for match in re.finditer(r"struct\s+(\w+)\s*\{([^{}]*)\}\s*;", uncomment(document), re.S):
            fields = []
            for item in match[2].split(";"):
                if not item.strip():
                    continue
                field = declaration.fullmatch(item.strip() + ";")
                if field is None:
                    raise ValueError(f"unsupported source declaration in {match[1]}: {item.strip()}")
                fields.append((field[3], field[1] + ("*" if field[2] else "")))
            structs[match[1]] = fields

    leaves = {}

    def flatten(struct_name, prefix=""):
        for name, native_type in structs[struct_name]:
            path = prefix + name
            if native_type.startswith("struct ") and not native_type.endswith("*"):
                flatten(native_type.removeprefix("struct "), path + ".")
            else:
                leaves[path] = native_type

    flatten("cfe_state_struct")
    implementation = uncomment(documents["src/bmi_cfe.c"])
    count = int(re.search(r"#define\s+STATE_VAR_NAME_COUNT\s+(\d+)", implementation)[1])
    table = re.search(r"Variable var_info\[\]\s*=\s*\{(.*?)\n\};", implementation, re.S)[1]
    advertised = {
        int(index): {"name": name, "type": native_type, "size": int(size)}
        for index, name, native_type, size in re.findall(
            r'\{\s*(\d+),\s*"([^"]+)",\s*"([^"]+)",\s*(\d+)\s*\}', table
        )
    }
    pointers = re.search(
        r"static int Get_state_var_ptrs\s*\([^)]*\)\s*\{(.*?)return BMI_SUCCESS;",
        implementation, re.S,
    )[1]
    assignments = {
        int(index): expression.strip()
        for index, expression in re.findall(r"ptr_list\[(\d+)\]\s*=\s*([^;]+);", pointers)
    }
    rows = []
    for index in range(count):
        row = {"slot": index, "advertised": advertised[index]}
        expression = assignments.get(index)
        if expression is None:
            row["problem"] = "pointer slot is never assigned"
        else:
            path = re.search(r"state->([\w.]+)", expression)[1]
            native_type = leaves[path]
            address_of = expression.startswith("&")
            row.update(expression=expression, field=path, native_type=native_type)
            if address_of and native_type.endswith("*"):
                row["problem"] = "address of pointer exported instead of pointee bytes"
            else:
                expected = advertised[index]["type"].removesuffix("*")
                actual = native_type.removesuffix("*").replace("long int", "long")
                if expected == "string":
                    expected = "char"
                if actual == "surface_water_partition_type":
                    actual = "int"  # native width must still be recorded by the capture tool
                if expected != actual:
                    row["problem"] = "advertised type differs from pointed-to field type"
        rows.append(row)
    size_function = re.search(
        r"static int Get_state_var_sizes\s*\([^)]*\)\s*\{(.*?)return BMI_SUCCESS;",
        implementation, re.S,
    )[1]
    size_assignments = [
        {"slot": int(index), "advertised_name": advertised[int(index)]["name"], "expression": expression.strip()}
        for index, expression in re.findall(r"var_info\[(\d+)\]\.size\s*=\s*([^;]+);", size_function)
    ]
    mapped = {row["field"] for row in rows if "field" in row}
    return {
        "schema": "nextgen.cfe.source-audit/v1",
        "source": {"repository": REPOSITORY, "revision": PIN,
                   "git_blobs": {name: git(source, "rev-parse", f"{PIN}:{name}").strip()
                                 for name in (*SOURCE_FILES, "LICENSE", "TERMS.md")}},
        "licence": {"spdx": "Apache-2.0", "quote": "Apache License; Version 2.0, January 2004"},
        "scope": "source metadata only; not a native execution, numerical replay, or GPU timing",
        "declared_state_slots": count,
        "assigned_pointer_slots": len(assignments),
        "unassigned_pointer_slots": sorted(set(range(count)) - assignments.keys()),
        "serialization": rows,
        "dynamic_size_assignments": size_assignments,
        "native_leaf_fields": leaves,
        "fields_absent_from_serialization": sorted(set(leaves) - mapped),
        "elapsed_seconds": time.perf_counter() - started,
        "parallelism": "one host process; source-metadata inspection only",
    }


def check_layout(layout, audit_result):
    """Check the port owner's concrete metadata, without interpreting arrays."""
    leaves = audit_result["native_leaf_fields"]
    seen = set()
    conversions = []
    synthetic_options = []
    for group, allowed in (("fields", {"double", "double*", "int"}),
                           ("options", {"int", "surface_water_partition_type"})):
        if not isinstance(layout.get(group), list) or not layout[group]:
            raise ValueError(f"missing nonempty port layout {group}")
        for name in layout[group]:
            if name in seen:
                raise ValueError(f"duplicate port field {name}")
            seen.add(name)
            if group == "options" and name == "nash_surface_params.has_storage":
                if leaves.get("nash_surface_params.nash_storage") != "double*":
                    raise ValueError("synthetic storage presence flag lacks native pointer witness")
                synthetic_options.append({"field": name, "witness": "nash_surface_params.nash_storage",
                                          "operation": "live captured pointee descriptor presence; pointer address never read"})
                continue
            if leaves.get(name) not in allowed:
                raise ValueError(f"port {group} field {name} does not match pinned native type")
            if group == "fields" and leaves[name] == "int":
                conversions.append({"field": name, "native_type": "int", "device_type": "f64",
                                    "operation": "exact integer promotion in the Rust setup packer"})
    if layout.get("offsets") != ["giuh_ordinates", "runoff_queue", "nash_subsurface",
                                "nash_surface", "soil_profile"]:
        raise ValueError("port layout does not match the owner's five ragged offset rows")
    return {"field_count": len(layout["fields"]), "option_count": len(layout["options"]),
            "offset_count": len(layout["offsets"]), "native_type_conversions": conversions,
            "synthetic_options": synthetic_options,
            "scope": "names/types only; not evidence that native fields are initialized or numerical outputs match"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="existing pinned CFE Git checkout (read only)")
    parser.add_argument("output", type=Path, help="new audit JSON under lane scratch")
    parser.add_argument("--layout", type=Path, help="CFE port owner's published field-schema.json")
    args = parser.parse_args()
    result = audit(args.source)
    if args.layout:
        result["port_layout"] = check_layout(json.loads(args.layout.read_text()), result)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "declared_state_slots", "assigned_pointer_slots", "unassigned_pointer_slots", "elapsed_seconds"
    )}))


if __name__ == "__main__":
    main()
