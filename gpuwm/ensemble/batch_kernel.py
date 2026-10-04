"""Audited member-outermost launch adapters for direct-pointer CUDA kernels.

The original entry and translation unit are used at one member.  Larger
batches retain the entry's arithmetic body verbatim and prepend member-local
pointer offsets and virtual block coordinates.  Source construction is not an
identity proof: callers must compare every member's output words on a device.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from math import prod
import re
from typing import Mapping, Sequence
from types import MappingProxyType

import numpy as np

from gpuwm.core.device_cache import cuda_cache


class BatchKernelUnsupported(ValueError):
    """The adapter cannot preserve the named kernel's indexing or ownership."""


_AUDITED_TOKENS = frozenset({
    "real", "float", "double", "int", "unsigned", "signed", "long", "short",
    "char", "bool", "void", "const", "volatile", "restrict", "__restrict",
    "__restrict__", "size_t", "int32_t", "uint32_t", "int64_t", "uint64_t",
    "uint3", "dim3", "blockIdx", "gridDim", "threadIdx", "blockDim",
    "__ensemble_virtual_block", "__ensemble_virtual_grid", "__ensemble_coord_type",
})


@dataclass(frozen=True)
class PointerSpec:
    name: str
    role: str
    dtype: str = "float32"

    def __post_init__(self):
        if re.fullmatch(r"[A-Za-z_]\w*", self.name) is None:
            raise ValueError("pointer names must be C identifiers")
        if self.role not in ("member", "shared"):
            raise ValueError("pointer role must be member or shared")
        dtype = np.dtype(self.dtype)
        if dtype.kind not in "biuf" or dtype.itemsize == 0:
            raise TypeError("CUDA pointer dtype must be a scalar boolean, integer, or float")
        if not dtype.isnative:
            raise ValueError("CUDA pointer dtype must have native byte order")
        object.__setattr__(self, "dtype", dtype.str)


@dataclass(frozen=True)
class KernelSpec:
    module: str
    entry: str
    pointers: tuple[PointerSpec, ...]
    options: tuple[str, ...] = ("-std=c++17",)

    def __post_init__(self):
        for value in (self.module, self.entry):
            if re.fullmatch(r"[A-Za-z_]\w*", value) is None:
                raise ValueError("kernel module and entry must be C identifiers")
        if not isinstance(self.pointers, tuple) or not all(
                isinstance(pointer, PointerSpec) for pointer in self.pointers):
            raise TypeError("pointers must be a tuple of PointerSpec values")
        names = [pointer.name for pointer in self.pointers]
        if len(set(names)) != len(names):
            raise ValueError("each pointer parameter needs one ownership role")
        if not isinstance(self.options, tuple) or not all(
                isinstance(option, str) for option in self.options):
            raise TypeError("options must be a tuple of strings")


def _members(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("members must be an integer")
    if value < 1:
        raise ValueError("members must be positive")
    return int(value)


def _masked(source: str) -> str:
    """Hide comments and strings without moving any source offsets."""
    pattern = re.compile(r'/\*[\s\S]*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|'
                         r"'(?:\\.|[^'\\])*'")
    return pattern.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), source)


def _close(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    raise BatchKernelUnsupported("unbalanced CUDA entry delimiters")


def _effective_options(options):
    """Use the compiler hook's own feature/option policy when it is active."""
    from gpuwm import wrf_exact
    return wrf_exact.effective_options(options) if wrf_exact.ENABLED else tuple(options)


def _option_macros(options):
    macros = {"__CUDACC__": "1", "__CUDACC_RTC__": "1"}
    for option in options:
        if option.startswith(("-include", "--pre-include")):
            raise BatchKernelUnsupported(
                "pre-included headers can change signature types/features outside the audited source")
        match = re.fullmatch(r"(?:-D|--define-macro=)([A-Za-z_]\w*)(?:=(.*))?", option)
        if match:
            if match[1] in _AUDITED_TOKENS:
                raise BatchKernelUnsupported(
                    f"macro {match[1]} redefines an audited CUDA type/qualifier/index token")
            macros[match[1]] = "1" if match[2] is None else match[2]
        elif option.startswith(("-D", "--define-macro", "-U", "--undefine-macro")):
            match = re.fullmatch(r"(?:-U|--undefine-macro=)([A-Za-z_]\w*)", option)
            if match is None:
                raise BatchKernelUnsupported("conditional signatures need canonical -DNAME=value/-UNAME options")
            macros.pop(match[1], None)
        match = re.fullmatch(r"(?:-arch|--gpu-architecture)=(?:compute|sm)_(\d+)", option)
        if match:
            macros["__CUDA_ARCH__"] = str(int(match[1]) * 10)
        match = re.fullmatch(r"(?:-std|--std)=c\+\+(11|14|17|20)", option)
        if match:
            macros["__cplusplus"] = {"11": "201103", "14": "201402",
                                      "17": "201703", "20": "202002"}[match[1]]
    return macros


def _integer_condition(expression, macros, trail=()):
    """Evaluate a closed signed-integer CPP condition, never float arithmetic."""
    token_pattern = re.compile(r"\s+|0[xX][0-9a-fA-F]+[uUlL]*|[0-9]+[uUlL]*|"
                               r"[A-Za-z_]\w*|&&|\|\||<<|>>|<=|>=|==|!=|[()!~+*/%<>&^|\-]")
    def tokenize(value):
        tokens, offset = [], 0
        while offset < len(value):
            match = token_pattern.match(value, offset)
            if match is None:
                raise BatchKernelUnsupported(
                    f"conditional signatures permit integer macro expressions only, got {value!r}")
            if not match[0].isspace():
                tokens.append(match[0])
            offset = match.end()
        return tokens

    def expand(items, visited):
        result, offset = [], 0
        while offset < len(items):
            token = items[offset]
            if token == "defined":
                # defined's operand is tested for presence, never expanded.
                length = 4 if offset + 1 < len(items) and items[offset + 1] == "(" else 2
                result.extend(items[offset:offset + length])
                offset += length
                continue
            if token in macros and macros[token] is not None:
                if token in visited:
                    raise BatchKernelUnsupported(f"conditional signature macro {token} is recursive")
                result.extend(expand(tokenize(macros[token]), visited + (token,)))
            else:
                result.append(token)
            offset += 1
        return result

    tokens = expand(tokenize(expression), trail)
    position = 0
    precedence = {"||": 1, "&&": 2, "|": 3, "^": 4, "&": 5,
                  "==": 6, "!=": 6, "<": 7, "<=": 7, ">": 7, ">=": 7,
                  "<<": 8, ">>": 8, "+": 9, "-": 9, "*": 10, "/": 10, "%": 10}

    def parse(minimum=1):
        nonlocal position
        if position == len(tokens):
            raise BatchKernelUnsupported("incomplete integer conditional signature expression")
        token = tokens[position]
        position += 1
        if token in ("!", "~", "+", "-"):
            node = ("unary", token, parse(11))
        elif token == "(":
            node = parse()
            if position == len(tokens) or tokens[position] != ")":
                raise BatchKernelUnsupported("unbalanced integer conditional signature parentheses")
            position += 1
        elif token == "defined":
            parenthesized = position < len(tokens) and tokens[position] == "("
            position += int(parenthesized)
            if position == len(tokens) or re.fullmatch(r"[A-Za-z_]\w*", tokens[position]) is None:
                raise BatchKernelUnsupported("conditional signature defined() needs one macro name")
            node = ("defined", tokens[position])
            position += 1
            if parenthesized:
                if position == len(tokens) or tokens[position] != ")":
                    raise BatchKernelUnsupported("conditional signature defined() needs one macro name")
                position += 1
        elif re.fullmatch(r"[A-Za-z_]\w*", token):
            node = ("macro", token)
        elif re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*", token):
            if re.search(r"[uU]", token):
                raise BatchKernelUnsupported("conditional signatures require signed integer conditions; unsigned arithmetic is unaudited")
            value = token.rstrip("lL")
            try:
                number = int(value, 16 if value.lower().startswith("0x") else
                             8 if len(value) > 1 and value.startswith("0") else 10)
            except ValueError as failure:
                raise BatchKernelUnsupported("invalid integer conditional signature literal") from failure
            node = ("number", number)
        else:
            raise BatchKernelUnsupported("unsupported integer conditional signature syntax")
        while position < len(tokens) and precedence.get(tokens[position], 0) >= minimum:
            operation = tokens[position]
            position += 1
            node = ("binary", operation, node, parse(precedence[operation] + 1))
        return node

    tree = parse()
    if position != len(tokens):
        raise BatchKernelUnsupported(
            f"unsupported conditional signature expression {expression!r}; function macros need explicit integer definitions")

    def bounded(value):
        if not -(2**63) <= value < 2**63:
            raise BatchKernelUnsupported("conditional signature integer expression exceeds the audited signed 64-bit range")
        return value

    def evaluate(node):
        kind = node[0]
        if kind == "number":
            return bounded(node[1])
        if kind == "defined":
            if node[1].startswith("__") and node[1] not in macros:
                raise BatchKernelUnsupported(
                    f"conditional signature needs compiler builtin {node[1]}; bind the actual architecture/standard option")
            return int(node[1] in macros)
        if kind == "macro":
            name = node[1]
            if name in ("true", "false"):
                return int(name == "true")
            if name not in macros:
                if name.startswith("__"):
                    raise BatchKernelUnsupported(
                        f"conditional signature needs compiler builtin {name}; bind the actual architecture/standard option")
                return 0  # CPP's defined semantics for an absent ordinary macro.
            if name in trail or macros[name] is None:
                raise BatchKernelUnsupported(
                    f"conditional signature macro {name} is recursive or function-like; supply an audited integer definition")
            return _integer_condition(macros[name], macros, trail + (name,))
        if kind == "unary":
            operation, value = node[1], evaluate(node[2])
            if operation == "!":
                return int(not value)
            if operation == "-":
                return bounded(-value)
            if operation == "+":
                return value
            if operation == "~":
                return bounded(~value)
        if kind == "binary":
            operation, left = node[1], evaluate(node[2])
            if operation == "&&" and left == 0:
                return 0
            if operation == "||" and left != 0:
                return 1
            right = evaluate(node[3])
            if operation in ("/", "%"):
                if right == 0:
                    raise BatchKernelUnsupported("conditional signature has integer division by zero")
                quotient = (abs(left) // abs(right)) * (-1 if (left < 0) != (right < 0) else 1)
                return bounded(quotient if operation == "/" else left - quotient * right)
            if operation in ("<<", ">>") and (not 0 <= right < 63 or left < 0):
                raise BatchKernelUnsupported("conditional signature shift is outside the audited integer range")
            operations = {"+": lambda: left + right, "-": lambda: left - right,
                          "*": lambda: left * right, "&": lambda: left & right,
                          "|": lambda: left | right, "^": lambda: left ^ right,
                          "<<": lambda: left << right, ">>": lambda: left >> right,
                          "==": lambda: int(left == right), "!=": lambda: int(left != right),
                          "<": lambda: int(left < right), "<=": lambda: int(left <= right),
                          ">": lambda: int(left > right), ">=": lambda: int(left >= right),
                          "&&": lambda: int(left != 0 and right != 0),
                          "||": lambda: int(left != 0 or right != 0)}
            if operation in operations:
                return bounded(operations[operation]())
        raise BatchKernelUnsupported(
            f"conditional signatures permit integer macro expressions only, got {expression!r}")

    return evaluate(tree)


def _active_source(source, options, *, macro_ranges=None):
    """An offset-preserving audit view; emitted arithmetic branches stay original."""
    macros = _option_macros(options)
    lines = source.splitlines(keepends=True)
    masked_lines = _masked(source).splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    result, stack = [], []
    active, index = True, 0
    blank = lambda value: re.sub(r"[^\n]", " ", value)
    while index < len(lines):
        first = index
        text = masked_lines[index]
        while text.rstrip().endswith("\\"):
            text = text.rstrip()[:-1] + " "
            index += 1
            if index >= len(lines):
                raise BatchKernelUnsupported("unterminated preprocessor continuation in CUDA source")
            text += masked_lines[index]
        directive = re.match(r"\s*#\s*([A-Za-z_]\w*)\b(.*)", text, re.S)
        if directive is None:
            result.extend(lines[first:index + 1] if active else
                          [blank(line) for line in lines[first:index + 1]])
            index += 1
            continue
        command, argument = directive[1], directive[2].strip()
        if command in ("if", "ifdef", "ifndef"):
            if command != "if" and re.fullmatch(r"[A-Za-z_]\w*", argument) is None:
                raise BatchKernelUnsupported("conditional signature ifdef/ifndef must name one macro")
            if active and command != "if" and argument.startswith("__") and argument not in macros:
                raise BatchKernelUnsupported(
                    f"conditional signature needs compiler builtin {argument}; bind the actual architecture/standard option")
            selected = (bool(_integer_condition(argument, macros)) if command == "if"
                        else (argument in macros) == (command == "ifdef")) if active else False
            stack.append([active, selected, False])
            active = active and selected
        elif command in ("elif", "else"):
            if not stack or stack[-1][2]:
                raise BatchKernelUnsupported("unmatched/duplicate preprocessor else in CUDA source")
            parent, taken, _ = stack[-1]
            selected = (bool(_integer_condition(argument, macros)) if command == "elif"
                        else True) if parent and not taken else False
            active = parent and not taken and selected
            stack[-1][1] = taken or selected
            stack[-1][2] = command == "else"
        elif command == "endif":
            if not stack:
                raise BatchKernelUnsupported("unmatched preprocessor endif in CUDA source")
            active = stack.pop()[0]
        elif active and command == "define":
            match = re.fullmatch(r"([A-Za-z_]\w*)(.*)", argument, re.S)
            if match is None:
                raise BatchKernelUnsupported("unsupported macro definition in CUDA source")
            if match[1] in _AUDITED_TOKENS:
                raise BatchKernelUnsupported(
                    f"macro {match[1]} redefines an audited CUDA type/qualifier/index token")
            if macro_ranges is not None:
                macro_ranges.append((offsets[first], offsets[index + 1]))
            macros[match[1]] = None if match[2].startswith("(") else match[2].strip()
        elif active and command == "undef":
            if re.fullmatch(r"[A-Za-z_]\w*", argument) is None:
                raise BatchKernelUnsupported("unsupported macro undefinition in CUDA source")
            macros.pop(argument, None)
        elif active and command not in ("pragma", "line"):
            raise BatchKernelUnsupported(
                f"conditional signature audit cannot resolve active #{command}; assemble all headers and feature definitions explicitly")
        result.extend(blank(line) for line in lines[first:index + 1])
        index += 1
    if stack:
        raise BatchKernelUnsupported("unterminated conditional branch in CUDA source")
    return "".join(result)


def _macro_definitions(source):
    """Complete continued definitions, with masked and original text."""
    lines = source.splitlines(keepends=True)
    masked = _masked(source).splitlines(keepends=True)
    index = 0
    while index < len(lines):
        first = index
        while masked[index].rstrip().endswith("\\"):
            index += 1
            if index == len(lines):
                raise BatchKernelUnsupported("unterminated macro definition continuation")
        if re.match(r"\s*#\s*define\b", masked[first]):
            yield "".join(masked[first:index + 1]), "".join(lines[first:index + 1])
        index += 1


def _device_bodies(masked):
    pattern = re.compile(r"\b(__device__|__global__)\b[^;{}]*\([^;{}]*\)\s*\{")
    found = []
    for match in pattern.finditer(masked):
        opening = match.end() - 1
        closing = _close(masked, opening, "{", "}")
        found.append((match[1] == "__device__", opening, closing))
    return tuple(found)


def _coordinate_rewrites(source, options):
    """Audit active coordinate reads and their exact original source offsets."""
    definitions = []
    active = _active_source(source, options, macro_ranges=definitions)
    masked = _masked(active)
    if re.search(r"\b__ensemble_(?:virtual_(?:block|grid)|coord_type)\b", _masked(source)):
        raise BatchKernelUnsupported("generated CUDA coordinate helper name collides with supplied source")
    bodies = _device_bodies(masked)
    macro_reads = []
    for start, end in definitions:
        text = _masked(source[start:end])
        match = re.match(r"\s*#\s*define\s+([A-Za-z_]\w*)", text)
        replacement = match.end()
        if replacement < len(text) and text[replacement] == "(":
            close = _close(text, replacement, "(", ")")
            if re.search(r"\b(?:blockIdx|gridDim)\b", text[replacement + 1:close]):
                raise BatchKernelUnsupported("macro parameters shadow CUDA coordinates and cannot be virtualized")
            replacement = close + 1
        body = text[replacement:]
        if "#" in body:
            raise BatchKernelUnsupported("macro token pasting/stringification can hide CUDA coordinate identifiers")
        for token in re.finditer(r"\b(blockIdx|gridDim)\b", body):
            macro_reads.append((start + replacement + token.start(),
                                start + replacement + token.end(), token[1]))
    reads = []
    helper_reads = False
    for token in re.finditer(r"\b(blockIdx|gridDim)\b", masked):
        containers = [(helper, start, end) for helper, start, end in bodies
                      if start < token.start() < end]
        if not containers:
            raise BatchKernelUnsupported("CUDA coordinates outside active device bodies cannot be safely virtualized")
        helper_reads |= any(helper for helper, _, _ in containers)
        reads.append((token.start(), token.end(), token[1]))
    # Static local storage is device-global unless explicitly block-shared.
    for _, start, end in bodies:
        body = masked[start + 1:end]
        for declaration in re.finditer(r"\bstatic\b([^;{}]*)(?:;|\{)", body):
            text = declaration[1]
            if re.search(r"\b__shared__\b", text):
                continue
            if (re.search(r"\b(?:const|constexpr)\b", text)
                    and re.search(r"\b(?:real|float|double|int|unsigned|signed|long|short|char|bool|size_t|u?int(?:32|64)_t)\b", text)
                    and "*" not in text and "&" not in text):
                continue
            raise BatchKernelUnsupported("mutable helper-local static storage is shared across members")
    replacements = tuple(sorted(set(reads + macro_reads)))
    for start, end, _ in replacements:
        before = _masked(source[max(0, start - 80):start]).rstrip()
        after = _masked(source[end:min(len(source), end + 80)]).lstrip()
        if before.endswith((".", "->", "::", "&")):
            raise BatchKernelUnsupported("qualified/addressed CUDA coordinates cannot be safely virtualized")
        if re.search(r"\b(?:auto|uint3|dim3|int|unsigned|long|short|float|double)\s*$", before):
            raise BatchKernelUnsupported("local declarations shadow CUDA coordinates")
        component = re.match(r"\.\s*[xyz]\b", after)
        if component is None:
            raise BatchKernelUnsupported("whole CUDA coordinate values have unaudited builtin/proxy types; use direct x/y/z components")
        following = after[component.end():].lstrip()
        # Parentheses do not turn a coordinate component into writable storage.
        # Refuse mutations before substituting a readonly builtin with a value.
        following = re.sub(r"^(?:\)\s*)+", "", following)
        if (re.match(r"(?:=(?!=)|(?:<<|>>|[+\-*/%&|^])=|\+\+|--)", following)
                or re.search(r"(?:\+\+|--)\s*\(*\s*$", before)):
            raise BatchKernelUnsupported("CUDA coordinate writes cannot be safely virtualized")
    return helper_reads or bool(macro_reads), replacements


def _rewrite_coordinates(source, replacements):
    pieces, offset = [], 0
    for start, end, token in replacements:
        pieces.extend((source[offset:start], "__ensemble_virtual_block()" if token == "blockIdx"
                       else "__ensemble_virtual_grid()"))
        offset = end
    pieces.append(source[offset:])
    return "".join(pieces)


def _coordinate_helpers(members):
    return ("template<class A, class B> struct __ensemble_coord_type { static constexpr bool same = false; };\n"
            "template<class A> struct __ensemble_coord_type<A, A> { static constexpr bool same = true; };\n"
            + "".join(f"static_assert(__ensemble_coord_type<decltype(::{variable}.{axis}), unsigned int>::same, "
                       '"CUDA coordinate component type differs from the audited unsigned index");\n'
                       for variable in ("blockIdx", "gridDim") for axis in ("x", "y", "z"))
            + "static __device__ __forceinline__ uint3 __ensemble_virtual_block() {\n"
            f"    const unsigned int nx = gridDim.x / {members}u;\n"
            "    uint3 value; value.x = blockIdx.x % nx;\n"
            "    value.y = blockIdx.y; value.z = blockIdx.z; return value;\n}\n"
            "static __device__ __forceinline__ dim3 __ensemble_virtual_grid() {\n"
            f"    return dim3(gridDim.x / {members}u, gridDim.y, gridDim.z);\n}}\n")


def _entry_parts(source: str, spec: KernelSpec, options=None):
    original_masked = _masked(source)
    if re.search(r"\b(?:cooperative_groups|this_grid|grid_group)\b", original_masked):
        raise BatchKernelUnsupported(
            f"{spec.entry}: cooperative grid synchronization can mix different members")
    for masked_definition, original_definition in _macro_definitions(source):
        if re.search(r"%(?:ctaid|nctaid)", original_definition):
            raise BatchKernelUnsupported(
                f"{spec.entry}: macro PTX grid-coordinate reads bypass virtualization")
    active_source = _active_source(source, _effective_options(spec.options) if options is None else options)
    masked = _masked(active_source)
    for variable in re.finditer(r"\b__device__\b([^;{}]*(?:;|\{))", masked):
        declaration_text = variable.group(1)
        if ("(" not in declaration_text
                and not re.search(r"\b(?:const|__constant__)\b", declaration_text)):
            raise BatchKernelUnsupported(
                f"{spec.entry}: writable device globals would be shared by all members")
    # Launch bounds constrain the original compiler's register allocation.
    # Keep the complete declaration in generated source. Only the CUDA
    # attribute's one/two positive literal integers are admitted here; other
    # decorators and macro-valued bounds still require a separate audit.
    launch_bounds = (r"(?:__launch_bounds__\s*\(\s*[1-9][0-9]*\s*"
                     r"(?:,\s*[1-9][0-9]*\s*)?\)\s*)?")
    pattern = re.compile(r"\b__global__\s+" + launch_bounds + r"void\s+" + re.escape(spec.entry)
                         + r"\s*\(")
    matches = list(pattern.finditer(masked))
    if len(matches) != 1:
        raise BatchKernelUnsupported(
            f"{spec.entry}: one unconditional __global__ void definition is required")
    declaration = matches[0]
    start = declaration.end() - 1
    end = _close(masked, start, "(", ")")
    body_start = end + 1
    while body_start < len(masked) and masked[body_start].isspace():
        body_start += 1
    if body_start == len(masked) or masked[body_start] != "{":
        raise BatchKernelUnsupported(f"{spec.entry}: entry is a prototype or decorated definition")
    body_end = _close(masked, body_start, "{", "}")
    signature = masked[start + 1:end]
    pointer_types = {}
    parameter_names = []
    uses_real = False
    scalar_types = {
        "real", "float", "double", "int", "unsigned", "unsigned int",
        "long", "unsigned long", "long long", "unsigned long long",
        "short", "unsigned short", "char", "signed char", "unsigned char", "bool",
        "size_t", "int32_t", "uint32_t", "int64_t", "uint64_t",
    }
    parameters = [] if signature.strip() in ("", "void") else signature.split(",")
    for parameter in parameters:
        parameter = re.sub(r"\b(?:__restrict__|__restrict|restrict)\b", "", parameter)
        parameter = " ".join(parameter.split())
        match = re.fullmatch(r"(.+?)\s*\b([A-Za-z_]\w*)", parameter)
        if match is None or any(token in parameter for token in ("[", "]", "(", ")", "&", "=")):
            raise BatchKernelUnsupported(f"{spec.entry}: unsupported CUDA parameter {parameter!r}")
        typename, name = match.groups()
        parameter_names.append(name)
        if name.startswith("__ensemble_"):
            raise BatchKernelUnsupported(f"{spec.entry}: reserved batch parameter prefix")
        if "*" in typename:
            if typename.count("*") != 1:
                raise BatchKernelUnsupported(
                    f"{spec.entry}: indirect pointer {name} needs a dedicated member adapter")
            pointee, pointer_qualifiers = typename.split("*")
            if pointer_qualifiers.strip():
                raise BatchKernelUnsupported(
                    f"{spec.entry}: pointer-level qualifiers on {name} need a dedicated adapter")
            pointee = pointee.strip()
            basic = re.sub(r"\b(?:const|volatile)\b", "", pointee)
            uses_real |= " ".join(basic.split()) == "real"
            if " ".join(basic.split()) not in scalar_types:
                raise BatchKernelUnsupported(
                    f"{spec.entry}: aggregate pointer {name} can hide cross-member addresses")
            pointer_types[name] = pointee
        else:
            basic = re.sub(r"\bconst\b", "", typename)
            uses_real |= " ".join(basic.split()) == "real"
            if " ".join(basic.split()) not in scalar_types:
                raise BatchKernelUnsupported(
                    f"{spec.entry}: by-value aggregate {name} can hide member pointers")
    if uses_real:
        aliases = re.findall(r"\btypedef\s+([^;{}]+?)\s+real\s*;", masked)
        if ([" ".join(alias.split()) for alias in aliases] != ["float"]
                or re.search(r"\busing\s+real\s*=", masked)):
            raise BatchKernelUnsupported(
                f"{spec.entry}: real parameters require exactly typedef float real; "
                "another alias would invalidate float32 argument storage")
    declared = {pointer.name for pointer in spec.pointers}
    actual = set(pointer_types)
    if declared != actual:
        raise BatchKernelUnsupported(
            f"{spec.entry}: pointer ownership differs from signature; "
            f"missing={sorted(actual - declared)}, extra={sorted(declared - actual)}")
    for pointer in spec.pointers:
        basic = re.sub(r"\b(?:const|volatile)\b", "", pointer_types[pointer.name])
        basic = " ".join(basic.split())
        dtypes = {
            "real": "float32", "float": "float32", "double": "float64",
            "char": "int8", "signed char": "int8", "unsigned char": "uint8",
            "short": "int16", "unsigned short": "uint16",
            "int": "int32", "unsigned": "uint32", "unsigned int": "uint32",
            "long long": "int64", "unsigned long long": "uint64",
            "size_t": "uint64", "int32_t": "int32", "uint32_t": "uint32",
            "int64_t": "int64", "uint64_t": "uint64", "bool": "bool",
        }
        expected_dtype = dtypes.get(basic)
        if expected_dtype is None:
            raise BatchKernelUnsupported(
                f"{spec.entry}: pointer {pointer.name} has an ambiguous CUDA scalar type {basic!r}")
        if np.dtype(pointer.dtype) != np.dtype(expected_dtype):
            raise BatchKernelUnsupported(
                f"{spec.entry}: pointer {pointer.name} declares {basic} but its "
                f"ownership specification uses {np.dtype(pointer.dtype).name}")
        if pointer.role == "shared" and not re.search(
                r"\bconst\b", pointer_types[pointer.name]):
            raise BatchKernelUnsupported(
                f"{spec.entry}: shared pointer {pointer.name} must be const "
                "to prevent members writing one output allocation")
    body = masked[body_start + 1:body_end]
    if "__ensemble_" in body:
        raise BatchKernelUnsupported(f"{spec.entry}: reserved batch local prefix")
    forbidden = (r"\b(?:cooperative_groups|this_grid|grid_group|cudaLaunchDevice|"
                 r"__threadfence|__threadfence_system)\b|<<<|>>>")
    if re.search(forbidden, body):
        raise BatchKernelUnsupported(
            f"{spec.entry}: grid-wide synchronization or device launches can mix members")
    # Inline PTX reads physical coordinates, bypassing the virtual locals.
    if re.search(r"%(?:ctaid|nctaid)", source[body_start:body_end]):
        raise BatchKernelUnsupported(f"{spec.entry}: PTX grid-coordinate reads bypass virtualization")
    for helper in re.finditer(r"\b__device__\b[^;{}]*\([^;{}]*\)\s*\{", masked):
        helper_start = helper.end() - 1
        helper_end = _close(masked, helper_start, "{", "}")
        helper_text = masked[helper_start:helper_end]
        if re.search(r"%(?:ctaid|nctaid)", source[helper_start:helper_end]):
            raise BatchKernelUnsupported(
                f"{spec.entry}: device helpers read physical PTX grid coordinates")
        if re.search(forbidden, helper_text):
            raise BatchKernelUnsupported(
                f"{spec.entry}: device helpers synchronize across blocks or launch kernels")
    return (declaration.start(), start, end, body_start, body_end,
            pointer_types, tuple(parameter_names), active_source[start + 1:end])


def generate_batch_source(source: str, spec: KernelSpec, members: int, *, audit_options=None) -> str:
    """Retain original arithmetic text while making direct pointers member-local.

    N=1 returns the exact input bytes.  For N>1 every pointer must have an
    explicit role.  Shared pointers retain their address; member pointers
    advance by their runtime byte stride.  This adapter preserves block-local
    barriers, warp membership, and each scalar launch's original grid shape.
    """
    members = _members(members)
    if members == 1:
        return source
    options = _effective_options(spec.options) if audit_options is None else tuple(audit_options)
    _, start, end, body_start, _, pointer_types, _, active_signature = _entry_parts(source, spec, options)
    accessor_mode, replacements = _coordinate_rewrites(source, options)
    descriptor = f"__ensemble_strides_{spec.entry}"
    count = max(1, len(spec.pointers))
    type_text = f"struct {descriptor} {{ unsigned long long bytes[{count}]; }};\n"
    locals_text = [
        "\n    const uint3 __ensemble_physical_block = blockIdx;",
        "    const dim3 __ensemble_physical_grid = gridDim;",
        f"    const unsigned int __ensemble_grid_x = __ensemble_physical_grid.x / {members}u;",
        "    const unsigned int __ensemble_member = __ensemble_physical_block.x / __ensemble_grid_x;",
    ]
    if not accessor_mode:
        locals_text += [
            "    const dim3 blockIdx(__ensemble_physical_block.x % __ensemble_grid_x,",
            "                        __ensemble_physical_block.y, __ensemble_physical_block.z);",
            "    const dim3 gridDim(__ensemble_grid_x, __ensemble_physical_grid.y,",
            "                       __ensemble_physical_grid.z);",
        ]
    for index, pointer in enumerate(spec.pointers):
        if pointer.role == "shared":
            continue
        pointee = pointer_types[pointer.name]
        byte_type = "const char" if re.search(r"\bconst\b", pointee) else "char"
        if re.search(r"\bvolatile\b", pointee):
            byte_type = "volatile " + byte_type
        locals_text.append(
            f"    {pointer.name} = reinterpret_cast<{pointee}*>("
            f"reinterpret_cast<{byte_type}*>({pointer.name}) + "
            f"static_cast<unsigned long long>(__ensemble_member) * "
            f"__ensemble_pointer_strides.bytes[{index}]);")
    insertion = "\n".join(locals_text) + "\n"
    signature = active_signature
    separator = ", " if signature.strip() not in ("", "void") else ""
    if signature.strip() == "void":
        signature = ""
    signature += separator + descriptor + " __ensemble_pointer_strides"
    closing = ")\n{" if "#" in source[start + 1:body_start] else source[end:body_start + 1]
    if not accessor_mode:
        return (type_text + source[:start + 1]
                + signature + closing + insertion + source[body_start + 1:])
    prefix = _rewrite_coordinates(source[:start + 1],
                                  tuple(row for row in replacements if row[0] < start + 1))
    suffix = _rewrite_coordinates(source[body_start + 1:],
                                  tuple((lo - body_start - 1, hi - body_start - 1, token)
                                        for lo, hi, token in replacements if lo >= body_start + 1))
    return type_text + _coordinate_helpers(members) + prefix + signature + closing + insertion + suffix


def batch_grid(grid: Sequence[int], members: int) -> tuple[int, int, int]:
    """Flatten members into x while keeping every scalar grid coordinate."""
    members = _members(members)
    if not 1 <= len(grid) <= 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            or value < 1 for value in grid):
        raise ValueError("CUDA grid must have one to three positive integer dimensions")
    gx, gy, gz = tuple(int(value) for value in grid) + (1,) * (3 - len(grid))
    if gx * members > 2**31 - 1 or max(gy, gz) > 65535:
        raise ValueError("member grid exceeds CUDA grid dimension limits")
    return gx * members, gy, gz


def pack_pointer_strides(spec: KernelSpec, strides: Mapping[str, int]) -> np.void:
    """Pack explicit per-pointer byte strides as one by-value CUDA parameter."""
    expected = {pointer.name for pointer in spec.pointers}
    if set(strides) != expected:
        raise ValueError("pointer byte strides must name every declared pointer exactly")
    values = []
    for pointer in spec.pointers:
        stride = strides[pointer.name]
        if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)):
            raise TypeError("pointer byte strides must be integers")
        if stride < 0 or stride > 2**64 - 1:
            raise ValueError("pointer byte stride is outside uint64 range")
        if pointer.role == "shared" and stride != 0:
            raise ValueError(f"shared pointer {pointer.name} must have zero byte stride")
        if pointer.role == "member" and stride == 0:
            raise ValueError(f"member pointer {pointer.name} needs a positive byte stride")
        values.append(int(stride))
    table = np.asarray(values or [0], dtype=np.uint64)
    return table.view(np.dtype(("V", table.nbytes)))[0]


@lru_cache(maxsize=None)
def _parameter_names_in_context(spec: KernelSpec, defines: tuple[tuple[str, int], ...], options):
    from gpuwm.core import kernels
    source = (kernels.module_source_int_defines(spec.module, defines)
              if defines else kernels.module_source(spec.module))
    return _entry_parts(source, spec, options)[-2]


def _runtime_audit_options(spec):
    """The actual architecture CuPy appends, for audit only, never new flags."""
    from cupy.cuda import compiler
    helper = getattr(compiler, "_get_arch_for_options_for_nvrtc", None)
    if helper is None:
        raise BatchKernelUnsupported(
            "installed CuPy does not expose its effective NVRTC architecture; "
            "conditional kernel signatures cannot be matched to compilation")
    flag, _ = helper()
    match = re.fullmatch(r"-arch=(?:sm|compute)_(\d+)", flag)
    if match is None:
        raise BatchKernelUnsupported("installed CuPy returned an unaudited NVRTC architecture flag")
    actual = int(match[1])
    options = _effective_options(spec.options)
    for option in options:
        requested = re.fullmatch(r"(?:-arch|--gpu-architecture)=(?:compute|sm)_(\d+)", option)
        if requested and int(requested[1]) != actual:
            raise BatchKernelUnsupported(
                "requested architecture differs from CuPy's final NVRTC target; "
                "the signature audit would select another ABI")
    return options + (flag,)


def _parameter_names(spec: KernelSpec, defines: tuple[tuple[str, int], ...], options=None):
    return _parameter_names_in_context(spec, defines,
                                       _runtime_audit_options(spec) if options is None else options)


def _pointer_arguments(spec, args, parameter_names):
    """Audit direct pointer storage types without imposing a batch shape."""
    if len(args) != len(parameter_names) or len(set(parameter_names)) != len(parameter_names):
        raise ValueError("CUDA arguments must match the audited entry parameter order")
    for pointer in spec.pointers:
        if pointer.name not in parameter_names:
            raise ValueError(f"pointer {pointer.name} is absent from audited parameter order")
        argument = args[parameter_names.index(pointer.name)]
        if not hasattr(argument, "__cuda_array_interface__"):
            raise TypeError(f"pointer {pointer.name} requires a CUDA array backing")
        if not argument.nbytes:
            raise ValueError(f"pointer {pointer.name} requires nonempty storage")
        if np.dtype(argument.dtype) != np.dtype(pointer.dtype):
            raise TypeError(
                f"pointer {pointer.name} requires {np.dtype(pointer.dtype).name}; "
                f"{np.dtype(argument.dtype).name} storage would be reinterpreted or overrun")
        yield pointer, argument


def validate_pointer_arguments(spec: KernelSpec, members: int, args,
                               strides: Mapping[str, int],
                               parameter_names: Sequence[str]) -> None:
    """Require bounded member-local CUDA storage at each pointer argument.

    Inner axes are contiguous. A padded leading member stride is accepted
    only with allocation pointer/size evidence covering the last logical slab.
    The actual leading stride must equal the supplied pointer byte stride.
    A member-zero view does not describe every member and is refused.
    """
    members = _members(members)
    for pointer, argument in _pointer_arguments(spec, args, parameter_names):
        if pointer.role == "member":
            if len(argument.shape) < 2 or argument.shape[0] != members:
                raise ValueError(
                    f"member pointer {pointer.name} must expose the complete "
                    "(members, ...) logical shape")
            itemsize = np.dtype(argument.dtype).itemsize
            slab_bytes = prod(int(extent) for extent in argument.shape[1:]) * int(itemsize)
            if slab_bytes < 1 or argument.nbytes != members * slab_bytes:
                raise ValueError(f"member pointer {pointer.name} has inconsistent logical payload bytes")
            byte_stride = strides[pointer.name]
            if argument.flags.c_contiguous:
                if byte_stride != slab_bytes:
                    raise ValueError(f"member pointer {pointer.name} requires its exact slab byte stride")
                continue
            actual_strides = getattr(argument, "strides", None)
            if actual_strides is None or len(actual_strides) != len(argument.shape):
                raise ValueError(f"padded member pointer {pointer.name} lacks actual array strides")
            expected = itemsize
            for extent, actual in zip(reversed(argument.shape[1:]), reversed(actual_strides[1:])):
                if actual != expected:
                    raise ValueError(f"padded member pointer {pointer.name} requires contiguous inner axes")
                expected *= int(extent)
            if (actual_strides[0] != byte_stride or byte_stride < slab_bytes
                    or byte_stride % itemsize):
                raise ValueError(
                    f"padded member pointer {pointer.name} needs its actual positive aligned "
                    "leading stride at least as large as one logical slab")
            data = getattr(argument, "data", None)
            memory = getattr(data, "mem", None)
            base = getattr(memory, "ptr", None)
            capacity = getattr(memory, "size", None)
            address = getattr(data, "ptr", None)
            if (memory is None or any(isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) for value in (base, capacity, address))):
                raise ValueError(
                    f"padded member pointer {pointer.name} needs owned allocation pointer/size bounds")
            if type(memory).__name__ == "UnownedMemory" and getattr(memory, "owner", None) is None:
                raise ValueError(f"padded member pointer {pointer.name} lacks allocation ownership evidence")
            base, capacity, address = int(base), int(capacity), int(address)
            span = (members - 1) * int(byte_stride) + slab_bytes
            if (not 0 <= base < 2**64 or not 0 < capacity < 2**64
                    or base + capacity > 2**64 or not 0 <= address < 2**64
                    or address % itemsize or address < base or address + span > base + capacity):
                raise ValueError(
                    f"padded member pointer {pointer.name} exceeds its allocation bounds "
                    "or starts at an unaligned address")
        elif not argument.flags.c_contiguous:
            raise ValueError(f"shared pointer {pointer.name} requires contiguous storage")


@cuda_cache(maxsize=None)
def _compiled(spec: KernelSpec, members: int, defines: tuple[tuple[str, int], ...], audit_options=None):
    from gpuwm.core import kernels
    if members == 1:
        if spec.options != ("-std=c++17",):
            raise ValueError("N=1 uses the original loader's exact compiler options")
        return (kernels.get_kernel_int_defines(spec.module, spec.entry, defines)
                if defines else kernels.get_kernel(spec.module, spec.entry))
    import cupy as cp
    from gpuwm.certify.kernel_manifest import record_module
    source = (kernels.module_source_int_defines(spec.module, defines)
              if defines else kernels.module_source(spec.module))
    source = generate_batch_source(source, spec, members,
                                   audit_options=_runtime_audit_options(spec) if audit_options is None else audit_options)
    key = f"gpuwm.ensemble.batch_kernel:{spec.module}:{spec.entry}[members={members}]"
    options = _effective_options(spec.options)
    module = cp.RawModule(code=source, options=options, name_expressions=None)
    kernels._compile_observed(module, key)
    record_module(key, source=source, options=options, module=module)
    return module.get_function(spec.entry)


def get_batch_kernel(spec: KernelSpec, members: int, *,
                     defines: tuple[tuple[str, int], ...] = ()):
    """Return a lazy raw-kernel adapter with explicit pointer_strides at launch.

    No CuPy import or device query occurs until launch.  The batch's resident
    arrays must expose every logical member slab with complete contiguous
    backing or proven padded allocation bounds; admission is the caller's
    responsibility. One member uses the unchanged original loader and args.
    """
    members = _members(members)

    def launch(grid, block, args, *, pointer_strides=None, **kwargs):
        physical = batch_grid(grid, members)
        if members == 1:
            return _compiled(spec, members, defines)(grid, block, args, **kwargs)
        if pointer_strides is None:
            raise ValueError("a member batch requires explicit pointer byte strides")
        descriptor = pack_pointer_strides(spec, pointer_strides)
        validate_pointer_arguments(spec, members, args, pointer_strides,
                                   _parameter_names(spec, defines))
        return _compiled(spec, members, defines)(
            physical, block, tuple(args) + (descriptor,), **kwargs)

    return launch


def _current_device():
    import cupy as cp
    return int(cp.cuda.runtime.getDevice())


def _argument_owners(original_args, device):
    """Retain every supplied CUDA array owner without copying its storage."""
    owners = []
    array_rows = []
    for argument in original_args:
        if not hasattr(argument, "__cuda_array_interface__"):
            continue
        owner_device = getattr(getattr(argument, "device", None), "id", None)
        if owner_device is None or int(owner_device) != device:
            raise ValueError(
                f"prepared batch argument is not on owning CUDA device {device}; "
                "a bound handle cannot safely read another card's allocation")
        memory = getattr(getattr(argument, "data", None), "mem", None)
        if memory is not None:
            owners.append(memory)
        array_rows.append(MappingProxyType({
            "pointer": int(argument.data.ptr), "dtype": np.dtype(argument.dtype).str,
            "shape": tuple(argument.shape), "strides": tuple(argument.strides),
            "allocation_pointer": getattr(memory, "ptr", None),
            "allocation_bytes": getattr(memory, "size", None),
        }))
    return tuple(owners), tuple(array_rows)


def _finish_prepared_launch(spec, members, device, physical_grid, block,
                            bound_args, kernel, owners, array_rows, audit_options,
                            shared_mem, stream, source_receipt=None):
    block = tuple(block)
    kwargs = {"shared_mem": shared_mem}
    if stream is not None:
        kwargs["stream"] = stream
    receipt = MappingProxyType({"module": spec.module, "entry": spec.entry,
                               "members": members, "device": device,
                               "grid": physical_grid, "block": block,
                               "options": _effective_options(spec.options),
                               "audit_options": audit_options,
                               "arrays": array_rows, **(source_receipt or {})})

    def launch():
        current = _current_device()
        if current != device:
            raise ValueError(
                f"prepared batch belongs to CUDA device {device}, current device is {current}; "
                "select the owning device or bind a new launch before reading its allocations")
        _ = owners
        return kernel(physical_grid, block, bound_args, **kwargs)

    launch.binding_receipt = receipt
    return launch


def prepare_batch_kernel_launch(spec: KernelSpec, members: int, grid, block, args, *,
                                pointer_strides=None,
                                defines: tuple[tuple[str, int], ...] = (),
                                shared_mem: int = 0, stream=None):
    """Bind a fixed launch once, retaining arrays and their allocation owners.

    In-place data updates remain visible. New arrays, layout/scalars, devices
    or grids require a new binding. Submission performs one owning-device
    check and invokes the fixed raw handle without packing or array traversal.
    N=1 binds the original scalar handle/grid/args without a batch descriptor.
    """
    members = _members(members)
    device = _current_device()
    original_args = tuple(args)
    owners, array_rows = _argument_owners(original_args, device)
    if members == 1:
        physical_grid = tuple(grid)
        bound_args = original_args
        audit_options = None
    else:
        physical_grid = batch_grid(grid, members)
        if pointer_strides is None:
            raise ValueError("a prepared member batch requires explicit pointer byte strides")
        descriptor = pack_pointer_strides(spec, pointer_strides)
        audit_options = _runtime_audit_options(spec)
        validate_pointer_arguments(spec, members, original_args, pointer_strides,
                                   _parameter_names(spec, defines, audit_options))
        bound_args = original_args + (descriptor,)
    kernel = _compiled(spec, members, defines, audit_options)
    return _finish_prepared_launch(spec, members, device, physical_grid, block,
                                   bound_args, kernel, owners, array_rows, audit_options,
                                   shared_mem, stream)


@cuda_cache(maxsize=None)
def _compiled_source(source, spec, members, audit_options):
    """Compile one supplied translation unit with separate per-device identity."""
    import cupy as cp
    from gpuwm.core import kernels
    from gpuwm.certify.kernel_manifest import record_module
    _entry_parts(source, spec, audit_options)
    compiled_source = generate_batch_source(source, spec, members, audit_options=audit_options)
    source_hash = sha256(source.encode("utf-8")).hexdigest()
    compiled_hash = sha256(compiled_source.encode("utf-8")).hexdigest()
    key = (f"gpuwm.ensemble.batch_source:{spec.module}:{spec.entry}"
           f"[members={members},source={source_hash}]")
    options = _effective_options(spec.options)
    module = cp.RawModule(code=compiled_source, options=options, name_expressions=None)
    kernels._compile_observed(module, key)
    record_module(key, source=compiled_source, options=options, module=module)
    return module.get_function(spec.entry), source_hash, compiled_hash


def prepare_batch_source_launch(source: str, spec: KernelSpec, members: int,
                                grid, block, args, *, pointer_strides=None,
                                shared_mem: int = 0, stream=None):
    """Bind an explicitly supplied, audited CUDA translation unit.

    Compile options come only from ``spec.options``. Signature and pointer
    roles are audited before compilation. The supplied source and generated
    source hashes are recorded alongside the actual compiler audit context.
    No array is copied. Ownership, device checks and rebinding rules match
    ``prepare_batch_kernel_launch``.

    At N=1 this API compiles the exact supplied source, without batch metadata.
    A caller that expanded a signature macro for larger batches must still
    use ``prepare_batch_kernel_launch`` at N=1 to retain the original loader's
    exact source and handle. Supplied-source equivalence needs its own proof.
    """
    if not isinstance(source, str) or not source.strip():
        raise TypeError("supplied CUDA source must be a nonempty string")
    members = _members(members)
    device = _current_device()
    original_args = tuple(args)
    owners, array_rows = _argument_owners(original_args, device)
    audit_options = _runtime_audit_options(spec)
    parameter_names = _entry_parts(source, spec, audit_options)[-2]
    if len(original_args) != len(parameter_names):
        raise ValueError("CUDA arguments must match the audited supplied entry parameter order")
    if members == 1:
        # This is a new supplied-source boundary. Check storage types while
        # retaining scalar array shapes and the exact source/grid/arguments.
        tuple(_pointer_arguments(spec, original_args, parameter_names))
        physical_grid = tuple(grid)
        bound_args = original_args
    else:
        physical_grid = batch_grid(grid, members)
        if pointer_strides is None:
            raise ValueError("a prepared member batch requires explicit pointer byte strides")
        descriptor = pack_pointer_strides(spec, pointer_strides)
        validate_pointer_arguments(spec, members, original_args, pointer_strides, parameter_names)
        bound_args = original_args + (descriptor,)
    kernel, source_hash, compiled_hash = _compiled_source(source, spec, members, audit_options)
    return _finish_prepared_launch(
        spec, members, device, physical_grid, block, bound_args, kernel,
        owners, array_rows, audit_options, shared_mem, stream,
        {"source_kind": "supplied", "source_sha256": source_hash,
         "compiled_source_sha256": compiled_hash})
