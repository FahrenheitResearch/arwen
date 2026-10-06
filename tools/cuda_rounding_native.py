"""Native cubin/SASS compiler diagnostics without opening a CUDA device."""
from __future__ import annotations

from collections import Counter
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from gpuwm.nvrtc_cache_key import keyed_source
from tools.cuda_rounding_dataflow import ptx_dataflow_signatures


def nvrtc_metadata(library: Path) -> dict:
    """Compiler version and accepted targets without a CUDA driver call."""
    nvrtc = ctypes.CDLL(str(library))
    major, minor, count = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    if nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)):
        raise RuntimeError("nvrtcVersion failed")
    if nvrtc.nvrtcGetNumSupportedArchs(ctypes.byref(count)):
        raise RuntimeError("nvrtcGetNumSupportedArchs failed")
    arches = (ctypes.c_int * count.value)()
    if nvrtc.nvrtcGetSupportedArchs(arches):
        raise RuntimeError("nvrtcGetSupportedArchs failed")
    return {"nvrtc_version": [major.value, minor.value], "supported_architectures": list(arches)}


def compile_cubin(source: str, options: tuple[str, ...], library: Path) -> bytes:
    """NVRTC C API binary read with the exact reported size.

    Binary cubins need every trailing zero byte. A text-oriented wrapper that
    drops a final byte produces an ELF whose final program header is truncated.
    No CUDA runtime or driver API is called here.
    """
    nvrtc = ctypes.CDLL(str(library))
    program = ctypes.c_void_p()
    create = nvrtc.nvrtcCreateProgram
    create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
                      ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
    compile_program = nvrtc.nvrtcCompileProgram
    compile_program.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    destroy = nvrtc.nvrtcDestroyProgram
    destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    options = tuple(options)
    option_buffer = (ctypes.c_char_p * len(options))(*(option.encode() for option in options))
    status = create(ctypes.byref(program), keyed_source(source, options).encode(), b"unit.cu", 0, None, None)
    if status:
        raise RuntimeError(f"nvrtcCreateProgram returned {status}")
    try:
        status = compile_program(program, len(options), option_buffer)
        if status:
            size = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
            nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(f"nvrtcCompileProgram returned {status}: {log.value.decode(errors='replace')}")
        size = ctypes.c_size_t()
        nvrtc.nvrtcGetCUBINSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        status = nvrtc.nvrtcGetCUBINSize(program, ctypes.byref(size))
        if status or not size.value:
            raise RuntimeError(f"nvrtcGetCUBINSize returned {status}, {size.value} bytes")
        blob = ctypes.create_string_buffer(size.value)
        nvrtc.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        status = nvrtc.nvrtcGetCUBIN(program, blob)
        if status:
            raise RuntimeError(f"nvrtcGetCUBIN returned {status}")
        return blob.raw
    finally:
        destroy(ctypes.byref(program))


def compile_sass(source: str, options: tuple[str, ...], *, nvrtc_library: Path,
                 nvdisasm: Path, return_metadata: bool = False,
                 deletion_ledger: Path | None = None):
    """Compile a real sm target and disassemble it with inline source info."""
    temporary_root = os.environ.get("TMPDIR")
    if not temporary_root:
        raise RuntimeError("TMPDIR must name the owned compiler scratch directory")
    Path(temporary_root).mkdir(parents=True, exist_ok=True)
    metadata = {}
    with tempfile.TemporaryDirectory(prefix="rounding-native-", dir=temporary_root) as temporary:
        cubin = Path(temporary) / "unit.cubin"
        blob = compile_cubin(source, options, nvrtc_library)
        cubin.write_bytes(blob)
        metadata = {"cubin_bytes": len(blob), "cubin_sha256": hashlib.sha256(blob).hexdigest()}
        try:
            process = subprocess.run([str(nvdisasm), "--print-code", "--print-line-info-inline", str(cubin)],
                                     check=False, capture_output=True, text=True, timeout=180)
            if process.returncode:
                raise RuntimeError(f"nvdisasm returned {process.returncode}: {process.stderr[:2000]}")
            if not process.stdout.strip():
                raise RuntimeError("nvdisasm produced no native instructions")
            sass = process.stdout
        finally:
            if deletion_ledger is not None:
                ledger = Path(deletion_ledger)
                ledger.parent.mkdir(parents=True, exist_ok=True)
                record = {"file": cubin.relative_to(temporary_root).as_posix(),
                          "operation": "owned_temporary_cleanup", **metadata,
                          "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                          "options": list(options)}
                # One O_APPEND write per short JSON line keeps two compiler
                # workers from overwriting or interleaving deletion records.
                descriptor = os.open(ledger, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
                try:
                    os.write(descriptor, (json.dumps(record) + "\n").encode())
                finally:
                    os.close(descriptor)
    return (sass, metadata) if return_metadata else sass


_SECTION = re.compile(r"\.section\s+\.text\.([\w$]+)")
_LINE = re.compile(r'//##\s+File\s+"([^"]+)",\s+line\s+(\d+)')
_FLOAT = re.compile(r"/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?"
                    r"((?:FFMA|FMUL|FADD|DFMA|DMUL|DADD)(?:\.[A-Z0-9]+)*)\s")
_ANY_INSTRUCTION = re.compile(r"/\*[0-9a-fA-F]+\*/")
_INLINED_AT = re.compile(r'inlined at "[^"]+",\s+line\s+(\d+)')
_NATIVE_INSTRUCTION = re.compile(r"/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?"
                                 r"([A-Z][A-Z0-9.]*)\s+([^;]+);")
_NATIVE_REGISTER = re.compile(r"\b(?:UR|R)\d+\b")


def sass_floating_signatures(sass: str, statement_lines=None) -> dict[tuple, Counter]:
    """Native floating opcodes at actual source lines, grouped by entry."""
    signatures, function, line, chain = {}, "<global>", 0, ()
    pending = []
    statement_lines = statement_lines or {}
    for text in sass.splitlines():
        match = _SECTION.search(text)
        if match:
            function, line, chain, pending = match[1], 0, (), []
        match = _LINE.search(text)
        if match:
            pending.append((int(match[2]), tuple(int(value) for value in _INLINED_AT.findall(text))))
        if _ANY_INSTRUCTION.search(text) and pending:
            line = statement_lines.get(pending[0][0], pending[0][0])
            call_lines = list(pending[0][1]) + [item[0] for item in pending[1:]]
            chain = tuple(dict.fromkeys((1, statement_lines.get(value, value), 0) for value in call_lines))
            pending = []
        match = _FLOAT.search(text)
        if match:
            opcode = match[1]
            instruction = _NATIVE_INSTRUCTION.search(text)
            arguments = instruction[2].split(",") if instruction else []
            # ptxas can encode a multiply as FFMA(a,b,-0). Its negative-zero
            # addend preserves the rounded product's zero sign, so this is
            # not contraction of another arithmetic expression. Positive
            # zero is deliberately not normalized: it can flip a -0 product.
            if opcode.startswith(("FFMA", "DFMA")) and arguments and arguments[-1].strip() == "-RZ":
                opcode = opcode.replace("FFMA", "FMUL", 1).replace("DFMA", "DMUL", 1)
            signatures.setdefault((function, 1, line, 0, chain), Counter())[opcode] += 1
    return signatures


def sass_dataflow_signatures(sass: str, statement_lines=None) -> dict[tuple, Counter]:
    """Native FMA operand DAGs through explicit register definitions.

    Constant-bank operands and global/shared/local loads remain distinct leaves.
    Address lowering and control-flow joins can differ harmlessly, so these
    fingerprints expose candidates instead of claiming numerical proof.
    """
    translated = []
    pending = []
    line, chain = 0, ()
    for text in sass.splitlines():
        match = _SECTION.search(text)
        if match:
            translated.append(f".visible .entry {match[1]}() {{")
            pending, line, chain = [], 0, ()
        match = _LINE.search(text)
        if match:
            pending.append((int(match[2]), tuple(int(value) for value in _INLINED_AT.findall(text))))
        match = _NATIVE_INSTRUCTION.search(text)
        if not match:
            continue
        if pending:
            line = pending[0][0]
            chain = tuple(dict.fromkeys(list(pending[0][1]) + [item[0] for item in pending[1:]]))
            pending = []
        translated.append(f".loc 1 {line} 0 " + " ".join(f"inlined_at 1 {value} 0" for value in chain))
        opcode, arguments = match[1], match[2]
        arguments = arguments.replace(".reuse", "").replace("URZ", "0").replace("RZ", "0")
        arguments = _NATIVE_REGISTER.sub(lambda item: "%" + item[0], arguments)
        parts = opcode.split(".")
        kind = parts[0]
        values = [value.strip() for value in arguments.split(",")]
        if kind in ("FFMA", "DFMA") and values[-1] == "-0":
            kind = "FMUL" if kind == "FFMA" else "DMUL"
            arguments = ", ".join(values[:-1])
        floating = {"FFMA": "fma", "FMUL": "mul", "FADD": "add",
                    "DFMA": "fma", "DMUL": "mul", "DADD": "add"}
        if kind in floating:
            rounding = next((item.lower() for item in parts if item in ("RN", "RZ", "RM", "RP")), "rn")
            flush = ".ftz" if "FTZ" in parts else ""
            precision = "f64" if kind.startswith("D") else "f32"
            opcode = f"{floating[kind]}.{rounding}{flush}.{precision}"
        elif kind.startswith(("LDC", "ULDC")):
            opcode = "ld.param.b64" if "64" in parts else "ld.param.b32"
        elif kind.startswith(("LDG", "LDS", "LDL")):
            space = "global" if kind.startswith("LDG") else "shared" if kind.startswith("LDS") else "local"
            opcode = f"ld.{space}.b64" if "64" in parts else f"ld.{space}.b32"
        elif kind in ("MOV", "UMOV", "S2R", "S2UR"):
            opcode = "mov.b32"
        else:
            opcode = opcode.lower()
        translated.append(f"{opcode} {arguments};")
    return ptx_dataflow_signatures("\n".join(translated), statement_lines)
