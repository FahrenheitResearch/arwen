"""Compile the unmodified pinned WRF small-step module and C ABI shims."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

PIN = {
    "dyn_em/module_small_step_em.F": "cabf1a177d50fb0096db79644af20cfe6d75217dbe63ab406a7e29bb54c17634",
    "share/module_model_constants.F": "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
}
ROUTINES = ("small_step_prep", "small_step_finish", "calc_p_rho", "calc_coef_w",
            "advance_uv", "advance_mu_t", "advance_w", "sumflux")
REAL_CONFIG = ("dampcoef", "zdamp")
INT_CONFIG = ("damp_opt", "phi_adv_z")


def statements(source):
    pending = ""
    for line in source.splitlines():
        line = line.split("!")[0].strip()
        if not line or line.startswith("#"):
            continue
        pending += " " + line.lstrip("&").rstrip("&").strip()
        if not line.endswith("&"):
            yield pending.strip()
            pending = ""


def generate(source):
    fields = sorted(set(re.findall(r"config_flags%(\w+)", source, re.I)))
    config = {f: "real" if f in REAL_CONFIG else "integer" if f in INT_CONFIG
              else "logical" for f in fields}
    stub = ["module module_configure", "implicit none", "type grid_config_rec_type"]
    for f, t in config.items():
        stub.append(f"{t} :: {f} = " + (".false." if t == "logical" else "0"))
    stub += ["end type", "end module"]
    schema = {}
    wrappers = ["module oracle_wrappers", "use iso_c_binding", "use module_small_step_em",
                "implicit none", "contains"]
    blocks = list(statements(source))
    for routine in ROUTINES:
        start = next(i for i, s in enumerate(blocks)
                     if re.match(rf"subroutine {routine}\s*\(", s, re.I))
        header = blocks[start]
        args = [s.strip().lower() for s in header[header.index("(")+1:header.rindex(")")].split(",")]
        declarations = {}
        for stmt in blocks[start+1:]:
            if "::" not in stmt:
                if stmt.lower().startswith("end subroutine"):
                    break
                continue
            lhs, rhs = stmt.split("::", 1)
            t = re.match(r"\s*(real|integer|logical|type\(grid_config_rec_type\))", lhs, re.I)
            if not t:
                continue
            dim = re.search(r"dimension\s*\((.*?)\)", lhs, re.I)
            intent = re.search(r"intent\s*\((.*?)\)", lhs, re.I)
            for name in rhs.split(","):
                name = name.strip().lower()
                if name in args:
                    declarations[name] = {"type": t[1].lower(),
                                          "dimensions": dim[1].replace(" ", "").lower() if dim else "",
                                          "intent": intent[1].strip().lower() if intent else ""}
            if len(declarations) == len(args):
                break
        if set(args) != set(declarations):
            raise ValueError((routine, set(args)-set(declarations)))
        schema[routine] = {"args": args, "declarations": declarations}
        wrappers.append(f"subroutine oracle_{routine}({','.join(args)}) bind(C)")
        ordered = sorted(args, key=lambda n: (bool(declarations[n]["dimensions"]),
                                              declarations[n]["type"] != "integer", args.index(n)))
        for name in ordered:
            decl = declarations[name]
            t = "real(c_float)" if decl["type"] == "real" else "integer(c_int)"
            dims = decl["dimensions"]
            if name == "config_flags":
                dims = str(len(fields))
            wrappers.append(t + (f", dimension({dims})" if dims else "") + f" :: {name}")
        if "config_flags" in args:
            wrappers.append("type(grid_config_rec_type) :: cfg")
            for i, (field, t) in enumerate(config.items(), 1):
                val = f"config_flags({i})"
                if t == "logical":
                    val += " /= 0"
                elif t == "real":
                    val = f"transfer({val}, cfg%{field})"
                wrappers.append(f"cfg%{field} = {val}")
        call_args = ["cfg" if a == "config_flags" else f"({a}/=0)"
                     if declarations[a]["type"] == "logical" else a for a in args]
        wrappers += [f"call {routine}({','.join(call_args)})", "end subroutine"]
    wrappers.append("end module")
    return "\n".join(stub)+"\n", "\n".join(wrappers)+"\n", {"routines": schema, "config": config}


def build(source_root, output):
    output.mkdir(parents=True, exist_ok=True)
    for rel, digest in PIN.items():
        actual = hashlib.sha256((source_root/rel).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"WRF source {rel} differs from the v4.7.1 pin: {actual}")
    text = (source_root/"dyn_em/module_small_step_em.F").read_text()
    stub, wrappers, schema = generate(text)
    (output/"configure_stub.F90").write_text(stub)
    (output/"wrappers.F90").write_text(wrappers)
    (output/"schema.json").write_text(json.dumps(schema, indent=2)+"\n")
    flags = ["-O0", "-fPIC", "-cpp", "-ffree-form", "-ffree-line-length-none",
             "-ffp-contract=off", "-fcheck=bounds", "-Dwrfmodel", "-DEM_CORE=1", "-DRWORDSIZE=4"]
    commands = []
    for path in (output/"configure_stub.F90", source_root/"share/module_model_constants.F",
                 source_root/"dyn_em/module_small_step_em.F", output/"wrappers.F90"):
        command = ["gfortran", "-c", *flags, str(path.resolve())]
        subprocess.run(command, cwd=output, check=True)
        commands.append(command)
    command = ["gfortran", "-shared", "-o", "libsmallstep_oracle.so", "configure_stub.o",
               "module_model_constants.o", "module_small_step_em.o", "wrappers.o"]
    subprocess.run(command, cwd=output, check=True)
    commands.append(command)
    receipt = {"wrf_version": "4.7.1", "wrf_commit": "f52c197ed39d12e087d02c50f412d90d418f6186",
               "source_sha256": PIN, "commands": commands,
               "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
               "configuration_stub": "Data-only grid_config_rec_type fields read by WRF; no arithmetic or routines are replaced.",
               "hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in output.iterdir() if p.is_file() and p.name != "build-receipt.json"}}
    (output/"build-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"library": str(output/"libsmallstep_oracle.so"), "routines": list(schema["routines"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
