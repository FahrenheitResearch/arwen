"""Expand marked expressions into RN intrinsics, preserving their AST order.

This is a development tool. The generated header is the shipped source.
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def expand(source):
    def emit(node, kind):
        if isinstance(node, ast.BinOp):
            op = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}[type(node.op)]
            return f"__{kind}{op}_rn({emit(node.left, kind)}, {emit(node.right, kind)})"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            # Negation changes only the sign bit, including signed zero.
            return f"plm_neg{kind}({emit(node.operand, kind)})"
        if isinstance(node, ast.Call):
            return f"{node.func.id}({', '.join(emit(a, kind) for a in node.args)})"
        if isinstance(node, ast.Subscript):
            return f"{emit(node.value, kind)}[{ast.unparse(node.slice)}]"
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return hex(node.value)
            return repr(node.value) + ("f" if kind == "f" and isinstance(node.value, float) else "")
        raise ValueError(ast.dump(node))

    while match := re.search(r"\b([DF])\(", source):
        start = match.end()
        end, depth = start, 1
        while depth:
            depth += (source[end] == "(") - (source[end] == ")")
            end += 1
        tree = ast.parse(source[start:end-1], mode="eval").body
        source = source[:match.start()] + emit(tree, match[1].lower()) + source[end:]
    return source


if __name__ == "__main__":
    template = (ROOT / "tools/portable_libm64.cuh.in").read_text(encoding="utf-8")
    # Avoid the physics constants supplied by the production loader.
    names = set(re.findall(r"\b[A-Z][A-Z0-9_]*\b", template)) - {"D", "F", "GPUWM_PORTABLE_LIBM64_CUH"}
    code_start = template.index("__device__")
    template = template[:code_start] + re.sub(r"\b[A-Z][A-Z0-9_]*\b", lambda m: "plm_" + m[0] if m[0] in names else m[0], template[code_start:])
    (ROOT / "gpuwm/core/kernels/portable_libm64.cuh").write_text(expand(template), encoding="utf-8", newline="\n")
