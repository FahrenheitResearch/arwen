"""Translate the pinned, complete WRF lake column to auditable CUDA C++.

This is deliberately a closed translation of module_sf_lake.F, not a general
Fortran compiler. Unknown statements fail generation. Array bounds and real
kinds are preserved, including default REAL literals in REAL(8) expressions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).parent / "upstream/module_sf_lake.F"
SOURCE_SHA256 = "b173ad8328ecc0aa6bb6e6306a9c6af4dc876b39dc5d304fe221122983214b98"


def split(text, delim=","):
    parts, depth, start = [], 0, 0
    for i, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == delim and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def statements():
    stack, active, carry, start = [], True, "", 0
    for lineno, raw in enumerate(SOURCE.read_text().splitlines(), 1):
        line = raw.strip()
        if line.startswith("#"):
            if line.startswith("#if"):
                take = "EM_CORE" in line or line.startswith("#ifndef") or "!defined" in line
                stack.append((active, take))
                active = active and take
            elif line.startswith("#else"):
                parent, take = stack[-1]
                active = parent and not take
            elif line.startswith("#endif"):
                active, _ = stack.pop()
            else:
                raise ValueError((lineno, line))
            continue
        if not active:
            continue
        # WRF strings here contain no exclamation marks.
        line = line.split("!", 1)[0].strip().lower()
        if not line:
            continue
        if not carry:
            start = lineno
        carry += " " + line.lstrip("&").rstrip("&").strip()
        if not line.endswith("&"):
            for item in split(carry.strip(), ";"):
                yield start, item
            carry = ""
    assert not carry and not stack


@dataclass
class Variable:
    name: str
    typ: str
    dims: list
    intent: str = ""
    initial: str = ""
    const: bool = False


def declaration(line):
    if "::" not in line:
        return None
    attrs, decls = line.split("::", 1)
    typ = ("double" if attrs.startswith("real(r8)") else
           "float" if attrs.startswith("real") else
           "int" if attrs.startswith("integer") else
           "bool" if attrs.startswith("logical") else "char")
    dim = re.search(r"dimension\s*\((.*?)\)", attrs)
    intent = re.search(r"intent\s*\((.*?)\)", attrs)
    out = []
    for decl in split(decls):
        name_init = decl.split("=", 1)
        m = re.fullmatch(r"(\w+)\s*(?:\((.*)\))?", name_init[0].strip())
        if not m:
            raise ValueError(decl)
        dims = m[2] if m[2] is not None else dim[1] if dim else None
        out.append(Variable(m[1], typ, split(dims) if dims else [],
                            intent[1].strip() if intent else "",
                            name_init[1].strip() if len(name_init) == 2 else "",
                            "parameter" in attrs))
    return out


TOKEN = re.compile(r"\s*(\d+(?:\.\d*)?(?:[ed][+-]?\d+)?(?:_r8)?|\.\d+(?:[ed][+-]?\d+)?(?:_r8)?|\.[a-z]+\.|[a-z_]\w*|\*\*|<=|>=|==|/=|[^\s])")
PREC = {".or.": 1, ".and.": 2, ".eqv.": 2, ".neqv.": 2,
        "==": 3, "/=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3,
        ".eq.": 3, ".ne.": 3, ".lt.": 3, ".gt.": 3, ".le.": 3, ".ge.": 3,
        "+": 4, "-": 4, "*": 5, "/": 5, "**": 7}
OPS = {".or.": "||", ".and.": "&&", ".eqv.": "==", ".neqv.": "!=",
       ".eq.": "==", ".ne.": "!=", ".lt.": "<", ".gt.": ">",
       ".le.": "<=", ".ge.": ">=", "/=": "!="}


class Expression:
    def __init__(self, source, symbols, slices=None):
        self.tokens = TOKEN.findall(source)
        self.i, self.symbols, self.slices = 0, symbols, slices or {}

    def take(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else ""

    def parse(self, minimum=0):
        tok = self.take()
        if tok in ("+", "-", ".not."):
            left = "(" + ("!" if tok == ".not." else tok) + self.parse(3 if tok == ".not." else 6) + ")"
        elif tok == "(":
            left = self.parse()
            assert self.take() == ")"
            left = "(" + left + ")"
        elif re.match(r"\d|\.\d", tok):
            if "_r8" in tok:
                left = tok.replace("_r8", "").replace("d", "e")
                if not any(c in left for c in ".e"):
                    left += ".0"
            elif any(c in tok for c in ".ed"):
                left = tok.replace("d", "e") + ("" if "d" in tok else "f")
            else:
                left = tok
        elif tok in (".true.", ".false."):
            left = tok[1:-1]
        else:
            left = tok
            if self.peek() == "(":
                self.take()
                args, rawargs, begin, depth = [], [], self.i, 0
                while True:
                    t = self.take()
                    if t == "(" : depth += 1
                    elif t == ")" and depth: depth -= 1
                    elif (t == "," or t == ")") and depth == 0:
                        rawargs.append("".join(self.tokens[begin:self.i-1]))
                        begin = self.i
                        if t == ")": break
                for j, arg in enumerate(rawargs):
                    if ":" in arg:
                        lo = arg.split(":")[0]
                        var = self.symbols[tok]
                        lo = lo or bounds(var.dims[j])[0]
                        args.append("(" + expr(lo, self.symbols) + "+_slice)")
                    else:
                        args.append(expr(arg, self.symbols))
                if tok in self.symbols and self.symbols[tok].dims:
                    left = tok + "(" + ",".join(args) + ")"
                elif tok in ("min", "max"):
                    left = "lake_" + tok + "(" + ",".join(args) + ")"
                elif tok in ("real", "dble", "int", "nint"):
                    if tok == "nint": left = "((int)round(" + args[0] + "))"
                    else: left = "((" + {"real":"float", "dble":"double", "int":"int"}[tok] + ")(" + args[0] + "))"
                elif tok in ("sum",):
                    left = "lake_sum(" + rawargs[0].split("(")[0] + ")"
                elif tok == "sign": left = "copysign(" + ",".join(args) + ")"
                elif tok == "abs": left = "lake_abs(" + ",".join(args) + ")"
                else: left = tok + "(" + ",".join(args) + ")"
            elif tok in self.symbols and self.symbols[tok].dims and tok in self.slices:
                left = tok + ".data[_slice]"
        while self.peek() in PREC and PREC[self.peek()] >= minimum:
            op = self.take()
            right = self.parse(PREC[op] + (0 if op == "**" else 1))
            if op == "/": left = "lake_div(" + left + "," + right + ")"
            elif op == "**": left = "lake_pow(" + left + "," + right + ")"
            else: left = "(" + left + OPS.get(op, op) + right + ")"
        return left


def expr(source, symbols, slices=None):
    parser = Expression(source, symbols, slices)
    result = parser.parse()
    if parser.i != len(parser.tokens):
        raise ValueError((source, parser.tokens[parser.i:]))
    return result


def bounds(dim):
    parts = dim.split(":")
    return parts if len(parts) == 2 else ["1", dim]


VALUES = dict(nlevsoil=10, nlevlake=10, nlevsnow=5, lbp=1,ubp=1,lbc=1,ubc=1,
              begg=1,endg=1,begl=1,endl=1,begc=1,endc=1,begp=1,endp=1,
              column=1,num_shlakec=1,num_shlakep=1,ims=1,ime=1,jms=1,jme=1,
              kms=1,kme=2,its=1,ite=1,jts=1,jte=1,kts=1,kte=1,ids=1,ide=1,
              jds=1,jde=1,kds=1,kde=2,lbj=-4,ubj=20,fn=1,numf=1,num_snowc=1,
              num_nosnowc=1,num_nolakec=1)


def capacity(var):
    n = 1
    for dim in var.dims:
        lo, hi = bounds(dim)
        n *= eval(hi, {"__builtins__":{}}, VALUES) - eval(lo, {"__builtins__":{}}, VALUES) + 1
    return n


def define(var, symbols, member=False):
    if var.typ == "char" or var.name == "r8": return ""
    # Unassociated biogeochemistry pointers are declared but never read in
    # this WRF build (CN and DGVM are not defined).
    if ":" in var.dims: return ""
    if var.dims:
        limits = [expr(v, symbols) for dim in var.dims for v in bounds(dim)]
        init = ", " + expr(var.initial, symbols) if var.initial else ""
        return f"LakeStorage<{var.typ},{capacity(var)}> {var.name}{{" + ",".join(limits) + "}" + ";" + (f" {var.name}.fill({expr(var.initial,symbols)});" if var.initial and not member else "")
    const = "static constexpr " if var.const else ""
    initial = " = " + expr(var.initial, symbols) if var.initial else ""
    return f"{const}{var.typ} {var.name}{initial};"


def assignment(line, symbols):
    lhs, rhs = line.split("=", 1)
    lhs, rhs = lhs.strip(), rhs.strip()
    base = lhs.split("(")[0].strip()
    slices = {}
    if base in symbols and symbols[base].dims:
        var = symbols[base]
        if "(" not in lhs:
            slices[base] = True
            count = str(capacity(var))
        else:
            indices = split(lhs[lhs.index("(")+1:-1])
            for j, dim in enumerate(indices):
                if ":" in dim:
                    lo, hi = dim.split(":")
                    dlo, dhi = bounds(var.dims[j])
                    lo, hi = lo or dlo, hi or dhi
                    count = f"({expr(hi,symbols)})-({expr(lo,symbols)})+1"
                    slices[base] = True
    if slices:
        # Whole-array RHS expressions use the same linear element index.
        for name, var in symbols.items():
            if var.dims:
                slices[name] = True
        return f"for(int _slice=0; _slice<{count}; ++_slice) " + expr(lhs,symbols,slices) + " = " + expr(rhs,symbols,slices) + ";"
    return expr(lhs,symbols) + " = " + expr(rhs,symbols) + ";"


def action(line, symbols, lineno):
    if line.startswith("call "):
        call = line[5:].strip()
        if call.startswith(("wrf_message", "wrf_debug")): return "// WRF diagnostic only."
        if call.startswith("wrf_error_fatal"): return f"error = {lineno}; return;"
        return expr(call,symbols) + "; if(error) return;"
    if line == "return": return "return;"
    if line == "cycle": return "continue;"
    if line == "exit": return "break;"
    if "=" in line: return assignment(line,symbols)
    raise ValueError((lineno,line))


def body_line(lineno, line, symbols, function):
    line = re.sub(r"^\w+\s*:\s*(do\b)",r"\1",line)
    line = re.sub(r"^(end\s*do)\s+\w+$",r"\1",line)
    if line.startswith(("implicit ","use ","write(","write (", "print ")): return ""
    if line.startswith(("data ","data(")):
        names, vals = line[4:].strip().split("/",1)
        vals = split(vals.rsplit("/",1)[0])
        name = names.strip()
        if "(" in name: name = name[1:].split("(")[0]
        return " ".join(f"{name}.data[{i}]={expr(v,symbols)};" for i,v in enumerate(vals))
    if line in ("enddo", "end do", "endif", "end if"): return "}"
    if line == "else": return "} else {"
    if line.startswith(("if ","if(","else if", "elseif")):
        is_else = line.startswith(("else",))
        start = line.index("(")
        depth, end = 0, 0
        for i in range(start,len(line)):
            depth += (line[i] == "(") - (line[i] == ")")
            if depth == 0:
                end = i
                break
        condition = expr(line[start+1:end], symbols)
        tail = line[end+1:].strip()
        prefix = "} else if" if is_else else "if"
        if tail == "then": return prefix + "(" + condition + ") {"
        return prefix + "(" + condition + ") { " + action(tail,symbols,lineno) + " }"
    if line.startswith("do while"):
        return "while(" + expr(line[line.index("(")+1:-1],symbols) + ") {"
    if line.startswith("do "):
        v, values = line[3:].split("=",1)
        v, values = v.strip(), split(values)
        start,end = values[:2]
        step = values[2] if len(values)>2 else "1"
        op = ">=" if step.startswith("-") else "<="
        return f"for({v}={expr(start,symbols)}; {v}{op}{expr(end,symbols)}; {v}+={expr(step,symbols)}) {{"
    if function and re.match(function+r"\s*=",line):
        return "return " + expr(line.split("=",1)[1],symbols) + ";"
    return action(line,symbols,lineno)


def generate():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Pinned WRF lake source differs; review the translation before regenerating")
    module, routines, current = {}, [], None
    for n,line in statements():
        start = re.match(r"(?:(real\(r8\)) )?(subroutine|function) (\w+)\s*\((.*)\)",line)
        if start:
            current = dict(name=start[3],ret="double" if start[2]=="function" else "void",args=split(start[4]),vars={},body=[])
            routines.append(current)
            continue
        if line.startswith(("end subroutine", "end function")):
            current = None
            continue
        if line.startswith(("module ","end module", "contains", "implicit ","use ")): continue
        decl = declaration(line)
        if decl is not None:
            dest = module if current is None else current["vars"]
            for var in decl: dest[var.name] = var
        elif current is not None:
            current["body"].append((n,line))
        elif line.startswith(("data ","data(")):
            pass  # module sand/clay data are initialized by the constructor below.
        else: raise ValueError((n,line))
    result = ["// Generated by tools/lake_wrf461_oracle/translate.py from WRF v4.6.1.",
              "// Source SHA256: " + hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "// Public-domain WRF transcription: licenses/LICENSE-WRF-public-domain.txt.",
              "struct LakeColumn {", "int error = 0;",
              "static constexpr float rcp = 0x1.24924ap-2f;"]
    for var in module.values():
        result.append(define(var,module,True))
    result.append("LAKE_HD LakeColumn() {")
    for name in ("filter_shlakec","filter_shlakep","pcolumn","pgridcell","cgridcell","clandunit","lakpoi"):
        result.append(name+".fill(1);")
    for n,line in statements():
        if line.startswith("subroutine "): break
        if line.startswith(("data ","data(")): result.append(body_line(n,line,module,None))
    result.append("}")
    for routine in routines:
        if routine["name"] == "lakedebug": continue
        symbols = module | routine["vars"]
        args = []
        for name in routine["args"]:
            v = routine["vars"][name]
            typ = f"LakeArray<{v.typ}>" if v.dims else v.typ
            prefix = "const " if v.intent=="in" else ""
            args.append(f"{prefix}{typ}& {name}")
        result.append("LAKE_HD " + routine["ret"] + " " + routine["name"] + "(" + ",".join(args) + ") {")
        for var in routine["vars"].values():
            if var.name not in routine["args"]:
                result.append(define(var,symbols))
        for n,line in routine["body"]:
            try:
                value = body_line(n,line,symbols,routine["name"] if routine["ret"]!="void" else None)
            except Exception as e:
                raise ValueError((routine["name"],n,line)) from e
            if value: result.append(f"// WRF:{n}\n{value}")
        result.append("}")
    result.append("};")
    return re.sub(r"\bvoid\b(?=\s*(?:;|=|>|<|\)))", "void_fraction", "\n".join(result)) + "\n"


if __name__ == "__main__":
    output = ROOT / "gpuwm/core/kernels/lake_wrf.cuh"
    output.write_text(generate(),encoding="utf-8",newline="\n")
    print(output)
