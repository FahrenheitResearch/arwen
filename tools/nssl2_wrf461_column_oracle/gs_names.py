"""Names shared by the GS rate-dump attribution tools.

WOOF's fused GS (gpuwm/core/kernels/nssl2_fused_gs.cu) is a per-point
transliteration of WRF's nssl_2mom_gs that keeps WRF's variable names.  The
dump compares every function-scope variable of nssl2_gs_point that WRF's
routine also holds as a gather array name(mgs), at the same program point
(just before the one update, module_mp_nssl_2mom.F:23049).
"""
from __future__ import annotations

import re
from pathlib import Path

DECL = re.compile(r"^(?:const\s+)?(?:float|double)\s+(.*);$")


def _split_top(decl: str) -> list[str]:
    depth, cur, parts = 0, "", []
    for ch in decl:
        if ch in "([{":
            depth += 1
        if ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def woof_scope_names(cu_text: str) -> list[str]:
    """Function-scope float/double variables of nssl2_gs_point, in order."""
    start = cu_text.index("__device__ void nssl2_gs_point(")
    body = cu_text[cu_text.index("{", start):]
    depth, names, pending = 0, [], ""
    for raw in body.split("\n"):
        code = raw.split("//")[0]
        if depth == 1:
            text = (pending + " " + code).strip() if pending else code.strip()
            if text.startswith(("float ", "double ", "const float ", "const double ")):
                if text.endswith(";"):
                    m = DECL.match(text)
                    if m:
                        decl = re.sub(r"\[[^\]]*\]", "", m.group(1))
                        for part in _split_top(decl):
                            n = re.match(r"\s*([A-Za-z_]\w*)", part)
                            if n:
                                names.append(n.group(1))
                    pending = ""
                else:
                    pending = text
            else:
                pending = ""
        depth += code.count("{") - code.count("}")
        if depth <= 0 and names:
            break
    return names


def shared_names(cu_path: Path, wrf_gs_body: str) -> list[str]:
    out = []
    for n in woof_scope_names(cu_path.read_text(encoding="utf-8")):
        if n in out:
            continue
        if re.search(r"\b%s\(mgs\)" % re.escape(n.lower()), wrf_gs_body):
            out.append(n)
    return out
