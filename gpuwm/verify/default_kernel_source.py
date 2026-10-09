"""A kernel module's source as the default compile sees it.

The diffusion compile route selects ``GPUWM_WRF_EXACT_C_DIFFUSION`` by
default. The remaining WRF-exact selectors are strict verification options.
Harnesses that patch kernel text (a workspace witness, a transcription
mutation, a diagnostic clone) must patch the code production runs, not a
branch that compiles away; they read the module through :func:`default_source`.
"""
from __future__ import annotations

import re

#: Diffusion is selected in production; the other exact selectors are unset.
_SELECTOR = re.compile(r"#\s*(if|ifdef|ifndef)\s+(!?)\s*(GPUWM_WRF_EXACT\w*)\s*$")
_OR_SELECTOR = re.compile(r"#\s*if\s+([A-Z0-9_]+(?:\s*\|\|\s*[A-Z0-9_]+)+)\s*$")
_EXACT_SELECTORS = frozenset(("GPUWM_WRF_EXACT", "GPUWM_WRF_EXACT_C_BIGSTEP",
                             "GPUWM_WRF_EXACT_C_ADVECTION", "GPUWM_WRF_EXACT_C_DIFFUSION",
                             "GPUWM_WRF_EXACT_D_DIAGNOSTICS"))
_OPEN = re.compile(r"#\s*(if|ifdef|ifndef)\b")
_ELIF = re.compile(r"#\s*elif\b")
_ELSE = re.compile(r"#\s*else\b")
_ENDIF = re.compile(r"#\s*endif\b")


def default_source(source: str, *, diffusion_selected: bool = True) -> str:
    """``source`` with selectors resolved for production diffusion.

    A selector region keeps the branch the default compile takes and loses
    the other branch and its directive lines; every other line, including
    any other preprocessor conditional, is kept byte for byte.  A selector
    OR expressions containing only the five known selectors are resolved
    too. Any other compound expression, ``#elif`` or ``#define``
    involving an exact selector raises rather than being guessed at.
    """
    kept = []
    frames = []  # (is_selector, taken) per open conditional
    for line in source.splitlines(keepends=True):
        text = line.strip()
        if text.startswith("#"):
            selector = _SELECTOR.fullmatch(text)
            if selector:
                kind, negated, _name = selector.groups()
                # Undefined: ``#if X`` and ``#ifdef X`` are false.
                selected = diffusion_selected and _name == "GPUWM_WRF_EXACT_C_DIFFUSION"
                frames.append((True, selected != (kind == "ifndef" or bool(negated))))
                continue
            selector_or = _OR_SELECTOR.fullmatch(text)
            if selector_or and all(name.strip() in _EXACT_SELECTORS
                                   for name in selector_or.group(1).split("||")):
                frames.append((True, diffusion_selected and any(name.strip() == "GPUWM_WRF_EXACT_C_DIFFUSION"
                                         for name in selector_or.group(1).split("||"))))
                continue
            if _ELIF.match(text):
                if (frames and frames[-1][0]) or "GPUWM_WRF_EXACT" in text:
                    raise ValueError(f"unrecognized WRF-exact selector: {text}")
            elif _ELSE.match(text):
                if frames and frames[-1][0]:
                    frames[-1] = (True, not frames[-1][1])
                    continue
            elif _ENDIF.match(text):
                if not frames:
                    raise ValueError("#endif without an open conditional")
                if frames.pop()[0]:
                    continue
            elif "GPUWM_WRF_EXACT" in text:
                raise ValueError(f"unrecognized WRF-exact selector: {text}")
            elif _OPEN.match(text):
                frames.append((False, True))
        if all(taken for _is_selector, taken in frames):
            kept.append(line)
    if frames:
        raise ValueError("unterminated preprocessor conditional")
    return "".join(kept)
