"""Translate a pinned Thompson numerical generation into a CUDA column.

The source generation is metadata. The parser rejects unknown statements;
it keeps Fortran array bounds, real kinds, expression order and loop order.
Initialization values are read back from that generation's Fortran init.
No math implementation is generated: CUDA uses WOOF's existing routines.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("lake_translation", ROOT / "tools/lake_wrf461_oracle/translate.py")
lake = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = lake
spec.loader.exec_module(lake)

TABLE_SHAPES = {
    **dict.fromkeys(("tcg_racg", "tmr_racg", "tcr_gacr", "tmg_gacr", "tnr_racg", "tnr_gacr"), [37, 37, 37, 37]),
    **dict.fromkeys(("tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2", "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2", "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2"), [37, 9, 37, 37]),
    **dict.fromkeys(("tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz"), [37, 37, 45, 55]),
    **dict.fromkeys(("tpi_qcfz", "tni_qcfz"), [37, 100, 45, 55]),
    **dict.fromkeys(("tps_iaus", "tni_iaus", "tpi_ide"), [64, 55]),
    **dict.fromkeys(("t_efrw", "t_efsw"), [100, 100]),
    **dict.fromkeys(("tpc_wev", "tnc_wev", "tnr_rev"), [100, 37, 100]),
    "tnccn_act": [7, 9, 7, 5, 4],
}
ROUTINES = ("rslf", "rsif", "icedemott", "icekoop", "eff_aero", "activ_ncloud", "mp_thompson", "calc_effectrad", "calc_refl10cm")

RADAR_FIELDS = {
    **{n: ("double", ["50"], "") for n in ("xxds", "xdts", "xxdg", "xdtg")},
    "simpson": ("double", ["51"], ""),
    **{n: ("double", [], "") for n in ("k_w", "pi5", "lamda4")},
    **{n: ("cd", [], "") for n in ("m_w_0", "m_i_0")},
    "nrbins": ("int", [], "50"),
    "lamda_radar": ("double", [], "0.10"),
    "melt_outside_s": ("double", [], "0.9d0"),
    "melt_outside_g": ("double", [], "0.9d0"),
}
RADAR_STRINGS = ("mixingrulestring_s", "matrixstring_s", "inclusionstring_s", "hoststring_s", "hostmatrixstring_s", "hostinclusionstring_s", "mixingrulestring_g", "matrixstring_g", "inclusionstring_g", "hoststring_g", "hostmatrixstring_g", "hostinclusionstring_g")


def statements(path):
    active, stack, carry, start = True, [], "", 0
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if line.startswith("#"):
            if line.startswith("#if"):
                stack.append((active, False))
                active = False
            elif line.startswith("#else"):
                parent, take = stack[-1]
                active = parent and not take
            elif line.startswith("#endif"):
                active, _ = stack.pop()
            continue
        if not active:
            continue
        line = line.split("!", 1)[0].strip().lower()
        if not line:
            continue
        if not carry:
            start = lineno
        carry += " " + line.lstrip("&").rstrip("&").strip()
        if not line.endswith("&"):
            for part in lake.split(carry.strip(), ";"):
                yield start, part
            carry = ""
    assert not carry and not stack


def declaration(line):
    if "::" not in line:
        if re.match(r"(?:real|integer|logical|double precision)\s+\w", line):
            m = re.match(r"(double precision|real|integer|logical)\s+(.*)", line)
            line = m[1] + "::" + m[2]
        else:
            return None
    out = lake.declaration(line)
    if line.startswith("double precision") or "kind=r8size" in line.replace(" ", ""):
        for v in out:
            v.typ = "double"
    return out


def parse_source(path):
    module, routines, current, module_zone = {}, {}, None, True
    for n, line in statements(path):
        if line == "contains":
            module_zone = False
        start = re.match(r"(?:(real|double precision)\s+)?(subroutine|function)\s+(\w+)\s*(?:\((.*)\))?$", line)
        if start:
            current = dict(name=start[3], ret="float" if start[2] == "function" else "void", args=lake.split(start[4]) if start[4] else [], vars={}, body=[])
            routines[current["name"]] = current
            continue
        if line.startswith(("end subroutine", "end function")):
            current = None
            continue
        if line.startswith(("module ", "end module", "contains", "implicit ", "use ")):
            continue
        if line.startswith("character"):
            if "::" in line and current is not None:
                for name in lake.split(line.split("::", 1)[1]):
                    current["vars"][name.strip()] = lake.Variable(name.strip(), "int", [])
            continue
        decl = declaration(line)
        if decl is not None:
            if current is None and not module_zone:
                continue
            target = module if current is None else current["vars"]
            for v in decl:
                target[v.name] = v
        elif current is not None:
            current["body"].append((n, line))
    for name, shape in TABLE_SHAPES.items():
        module[name].dims = [str(v) for v in shape]
    for name, value in {"re_qc_bg": "2.49e-6", "re_qi_bg": "4.99e-6", "re_qs_bg": "9.99e-6"}.items():
        module[name] = lake.Variable(name, "float", [], initial=value, const=True)
    for name, (typ, dims, initial) in RADAR_FIELDS.items():
        module[name] = lake.Variable(name, typ, dims, initial=initial, const=bool(initial))
    for name in RADAR_STRINGS:
        module[name] = lake.Variable(name, "int", [], initial="0", const=True)
    return module, {k: routines[k] for k in ROUTINES}


def referenced(module, routines):
    text = " ".join(line for r in routines.values() for _, line in r["body"])
    names = set(re.findall(r"\b[a-z_]\w*\b", text)) & set(module)
    while True:
        more = set(re.findall(r"\b[a-z_]\w*\b", " ".join(module[k].initial + " " + " ".join(module[k].dims) for k in names))) & set(module)
        if more <= names:
            return names
        names |= more


class Expression(lake.Expression):
    def parse(self, minimum=0):
        # Reuse the checked Fortran parser. Array and intrinsic lowering is
        # supplied below; expressions retain their original grouping.
        return super().parse(minimum)


def expr(source, symbols, slices=None):
    source = re.sub(r"'([a-z])'", lambda m: str(ord(m[1])), source)
    source = re.sub(r"any\s*\(\s*(\w+)\s*\.eqv\.\s*\.true\.\s*\)", r"tc_any(\1,kts,kte)", source)
    value = lake.expr(source, symbols, slices)
    # Lower array calls recursively. All dimensions are known, including
    # the live kts:kte bounds of a column's private arrays.
    def lower_arrays(text):
        for name, var in sorted(symbols.items(), key=lambda p: -len(p[0])):
            if not var.dims:
                continue
            pattern = re.compile(r"\b" + re.escape(name) + r"\(")
            while match := pattern.search(text):
                begin, depth, end = match.end(), 1, match.end()
                while depth:
                    depth += (text[end] == "(") - (text[end] == ")")
                    end += 1
                args = lake.split(text[begin:end-1])
                if len(args) != len(var.dims):
                    raise ValueError((name, args, var.dims))
                index, stride = "0", "1"
                for arg, dim in zip(args, var.dims):
                    lo, hi = lake.bounds(dim)
                    index += f"+({arg}-({lake.expr(lo, symbols)}))*({stride})"
                    stride += f"*(({lake.expr(hi, symbols)})-({lake.expr(lo, symbols)})+1)"
                text = text[:match.start()] + name + "[" + index + "]" + text[end:]
        return text
    value = lower_arrays(value)
    value = re.sub(r"\blake_(div|pow|min|max|abs)\b", r"tc_\1", value)
    value = re.sub(r"\bcopysign\b", "tc_sign", value)
    # Keep a REAL parameter exponent REAL even when its value is an
    # integer. gfortran calls powf for x**3.0 and multiplies for x**3;
    # those operations have different float32 words on source columns.
    value = re.sub(r"\b(exp|dexp|alog|dlog|log|alog10|dlog10|sqrt|dsqrt|dabs|dmin1|dmax1|datan|atan|datan2|atan2|sngl|dfloat|present|any|sign)\s*\(", lambda m: "tc_" + {"dexp":"exp", "alog":"log", "dlog":"log", "alog10":"log10", "dlog10":"log10", "dsqrt":"sqrt", "dabs":"abs", "dmin1":"min", "dmax1":"max", "datan":"atan", "datan2":"atan2"}.get(m[1], m[1]) + "(", value)
    return value


def body_line(n, line, symbols, function):
    label = re.match(r"^(\d+)\s+(.*)", line)
    prefix = ""
    if label:
        prefix, line = "label_" + label[1] + ": ;\n", label[2]
    if line == "continue":
        return prefix
    if re.match(r"go\s*to\s+\d+", line):
        return prefix + "goto label_" + re.findall(r"\d+", line)[0] + ";"
    if line.startswith(("write", "print", "call wrf_")):
        return prefix + "// diagnostic only"
    if line == "return":
        return prefix + ("return result;" if function else "return;")
    if function and re.match(function + r"\s*=", line):
        return prefix + "result = " + expr(line.split("=", 1)[1], symbols) + ";"
    if line.startswith("call "):
        return prefix + expr(line[5:], symbols) + ";"
    if line.startswith(("if ", "if(", "else if", "elseif")):
        is_else, start = line.startswith("else"), line.index("(")
        depth, end = 0, 0
        for i in range(start, len(line)):
            depth += (line[i] == "(") - (line[i] == ")")
            if depth == 0:
                end = i
                break
        cond = expr(line[start+1:end], symbols)
        tail = line[end+1:].strip()
        lead = "} else if" if is_else else "if"
        return prefix + lead + "(" + cond + ") {" + ("" if tail == "then" else " " + body_line(n, tail, symbols, function) + " }")
    if line.startswith("do "):
        var, vals = line[3:].split("=", 1)
        var, vals = var.strip(), lake.split(vals)
        step = vals[2] if len(vals) == 3 else "1"
        op = ">=" if step.startswith("-") else "<="
        return prefix + f"for({var}={expr(vals[0], symbols)}; {var}{op}{expr(vals[1], symbols)}; {var}+={expr(step, symbols)}) {{"
    if line in ("enddo", "end do", "endif", "end if"):
        return prefix + "}"
    if line == "else":
        return prefix + "} else {"
    if line in ("cycle", "exit"):
        return prefix + {"cycle": "continue;", "exit": "break;"}[line]
    if line.startswith(("implicit", "data ")):
        return prefix
    if "=" in line:
        lhs, rhs = line.split("=", 1)
        base = lhs.strip().split("(")[0]
        if base in symbols and symbols[base].dims and (":" in lhs or "(" not in lhs):
            variable = symbols[base]
            if len(variable.dims) != 1:
                raise ValueError((n, "multidimensional whole-array assignment", line))
            lo, hi = lake.bounds(variable.dims[0])
            match = re.search(r"\(([^)]*):([^)]*)\)", lhs)
            if match:
                lo, hi = match[1].strip() or lo, match[2].strip() or hi
            return prefix + f"for(int _slice={expr(lo,symbols)};_slice<={expr(hi,symbols)};++_slice) " + expr(base + "(_slice)",symbols) + " = " + expr(rhs.strip(),symbols) + ";"
        return prefix + expr(lhs.strip(), symbols) + " = " + expr(rhs.strip(), symbols) + ";"
    raise ValueError((n, line))


def dump_init(source, directory):
    module, routines = parse_source(source)
    names = sorted(referenced(module, routines) - set(TABLE_SHAPES) - set(RADAR_STRINGS) - {"re_qc_bg", "re_qi_bg", "re_qs_bg"})
    names = [n for n in names if module[n].typ != "char"]
    code = ["subroutine dump_generation_coefficients()", "open(91,file='coefficients.bin',access='stream',form='unformatted',status='replace')"]
    code += ["write(91) " + name for name in names]
    code += ["close(91)", "end subroutine dump_generation_coefficients"]
    text = source.read_text()
    position = text.lower().rindex("end module module_mp_thompson")
    (directory / "module_mp_thompson-dump.F").write_text(text[:position] + "\n" + "\n".join(code) + "\n" + text[position:])
    driver = (ROOT / "tools/thompson_wrf461_oracle/dump_aux_tables.F90").read_text()
    driver = driver.replace("thompson_init,", "dump_generation_coefficients, thompson_init,")
    driver = driver.replace("  open(71", "  call dump_generation_coefficients()\n  open(71")
    (directory / "dump_coefficients.F90").write_text(driver)
    (directory / "coefficient-schema.json").write_text(json.dumps({n: vars(module[n]) for n in names}, indent=2))


def generate(source, directory, output):
    import numpy as np
    module, routines = parse_source(source)
    schema = json.loads((directory / "coefficient-schema.json").read_text())
    raw, offset, values = (directory / "coefficients.bin").read_bytes(), 0, {}
    # Integer dimensions are PARAMETERs in this module and precede arrays
    # in dependency order. Resolve them from the declarations themselves.
    dimenv = {"kts": 1, "kte": 64}
    unresolved = {n: v.initial for n, v in module.items() if v.typ == "int" and v.const and v.initial}
    while unresolved:
        progress = False
        for name, initial in list(unresolved.items()):
            try:
                dimenv[name] = eval(initial, {"__builtins__": {}}, dimenv)
            except NameError:
                continue
            del unresolved[name]
            progress = True
        if not progress:
            raise ValueError(unresolved)
    for name, v in schema.items():
        count = 1
        for dim in v["dims"]:
            lo, hi = lake.bounds(dim)
            count *= eval(hi, {"__builtins__": {}}, dimenv) - eval(lo, {"__builtins__": {}}, dimenv) + 1
        dtype = {"float": "<f4", "double": "<f8", "int": "<i4", "bool": "<i4", "cd": "<f8"}[v["typ"]]
        if v["typ"] == "cd":
            count *= 2
        a = np.frombuffer(raw, dtype, count, offset)
        values[name] = a.tolist()
        offset += a.nbytes
    if offset != len(raw):
        raise ValueError((offset, len(raw)))
    lines = ["// Generated Thompson numerical generation; source and init values are pinned.", "struct ThompsonColumn {", "int nz;"]
    for name in sorted(referenced(module, routines) | set(TABLE_SHAPES)):
        v = module[name]
        if name in TABLE_SHAPES:
            lines.append(f"const {v.typ}* {name};")
        elif name in values:
            def literal(x):
                if v.typ == "bool":
                    return "true" if x else "false"
                if v.typ == "int":
                    return str(x)
                return float(x).hex() + ("f" if v.typ == "float" else "")
            vs = values[name]
            if v.typ == "cd":
                lines.append(f"cd {name} = {{" + ",".join(float(x).hex() for x in vs) + "};")
            elif v.dims:
                # Constants are static device data, rather than a copy in
                # each thread's local column storage.
                lines.insert(1, f"__device__ __constant__ {v.typ} tc_{name}[{len(vs)}] = {{" + ",".join(map(literal, vs)) + "};")
                lines.append(f"const {v.typ}* {name} = tc_{name};")
            else:
                lines.append(f"static constexpr {v.typ} {name} = {literal(vs[0])};")
        else:
            lines.append(f"static constexpr {v.typ} {name} = {expr(v.initial, module)};")
    lines.append("__device__ void rayleigh_soak_wetgraupel(double x,double a,double b,double fm,double mo,cd mw,cd mi,double lambda,double& cback,int,int,int,int,int,int) { cback = refl_rayleigh_soak_wetgraupel(x,a,b,fm,mo,mw,mi,pi5,lamda4); }")
    for routine in routines.values():
        symbols = module | routine["vars"]
        args = []
        for name in routine["args"]:
            v = routine["vars"][name]
            typ = v.typ + ("*" if v.dims else "&" if routine["ret"] == "void" and v.intent == "inout" else "")
            args.append(typ + " " + name)
        lines.append("__device__ __forceinline__ " + routine["ret"] + " " + routine["name"] + "(" + ",".join(args) + ") {")
        function = routine["name"] if routine["ret"] != "void" else None
        if function:
            lines.append(routine["ret"] + " result;")
        for name, v in routine["vars"].items():
            if name in routine["args"]:
                continue
            if v.dims:
                cap = 1
                for dim in v.dims:
                    lo, hi = lake.bounds(dim)
                    cap *= eval(hi, {"__builtins__": {}}, dimenv) - eval(lo, {"__builtins__": {}}, dimenv) + 1
                initial = ""
                if v.initial:
                    if not v.initial.startswith("(/") or not v.initial.endswith("/)"):
                        raise ValueError((name, v.initial))
                    initial = " = {" + ",".join(expr(x,symbols) for x in lake.split(v.initial[2:-2])) + "}"
                capacity = ("THOMPSON_COLUMN_KMAX+1" if any("kte+1" in d for d in v.dims)
                            else "THOMPSON_COLUMN_KMAX" if any("kte" in d for d in v.dims)
                            else str(cap))
                lines.append(f"{v.typ} {name}[{capacity}]{initial};")
            else:
                initial = " = " + expr(v.initial, symbols) if v.initial else ""
                lines.append(f"{v.typ} {name}{initial};")
        for n, line in routine["body"]:
            try:
                body = body_line(n, line, symbols, function)
            except Exception as e:
                raise ValueError((routine["name"], n, line)) from e
            if body:
                lines.append(f"// source:{n}\n{body}")
        if function:
            lines.append("return result;")
        lines.append("}")
    lines.append("};")
    output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("dump-init", "generate"))
    ap.add_argument("source", type=Path)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.action == "dump-init":
        dump_init(a.source, a.directory)
    else:
        generate(a.source, a.directory, a.output)
