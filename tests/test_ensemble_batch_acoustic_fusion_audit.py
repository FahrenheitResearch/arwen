"""The Mu/W fusion trial's source audit, held against the installed kernel on a CPU.

THE BREAKAGE THIS PREVENTS
--------------------------
``gpuwm/ensemble/batch_acoustic_fusion.py`` builds one joined column kernel
from the text of the acoustic Mu and W bodies, under a counted audit: three
column intermediates, a closed set of index forms, and a fixed number of Mu
stores.  The audit is a guard, and a guard that has gone stale refuses a
kernel that is fine.  That happened at the 2.8.5 staging tip.  2.8.4
(a11786408) gave ``advance_mu_th_msf`` a strict and a default body side by
side in one conditional block and gave ``advance_w_phi_msf`` a forcing row
named ``kr``.  The audit read the raw text, counted both arms (2, 4 and 6
stores against the audited 1, 2 and 3) and refused every map-factor domain:
21 device tests red, in files no battery list ran, so nothing reported it.

Everything the audit decides is decided from text, so it is asked here
without a card.  The installed kernel must be admitted, and a kernel whose
compiled stores, index forms or store ownership really changed must still be
refused.  The device files (``tests/test_ensemble_batch_acoustic_fusion_gpu.py``
and its dycore sibling) hold the words; this file holds the admission.
"""

from __future__ import annotations

import re

import pytest

from gpuwm.core.kernels import module_source, module_source_int_defines
from gpuwm.ensemble import batch_acoustic_fusion as fusion
from gpuwm.ensemble.batch_kernel import (
    KernelSpec, PointerSpec, _active_source, _close, _masked,
)
from gpuwm.ensemble.batch_state import BatchStateUnsupported

#: The default arithmetic: no strict macro, any real architecture.
OPTIONS = ("-std=c++17", "-arch=sm_80")
PAIRS = {"flat": ("advance_mu_th", "advance_w_phi"),
         "mapped": ("advance_mu_th_msf", "advance_w_phi_msf")}
SOURCES = {"shipped_tier": module_source("acoustic"),
           "deep_tier": module_source_int_defines("acoustic", (("WPHI_MAX_LEV", 257),))}
SOURCE = SOURCES["shipped_tier"]
AUDITED_MU_STORES = {"th_pp_old": 1, "th_pp": 2, "ww_pp": 3}


def _signature_end(source: str, entry: str) -> tuple[int, int]:
    masked = _masked(source)
    declaration = re.search(r"\bvoid\s+" + entry + r"\s*\(", masked)
    assert declaration is not None, f"{entry} left the acoustic translation unit"
    return declaration.end(), _close(masked, declaration.end() - 1, "(", ")")


def _spec(source: str, entry: str) -> KernelSpec:
    """The entry's pointers as the launcher derives them; ownership is not audited here."""
    start, end = _signature_end(source, entry)
    signature = re.sub(r"#if GPUWM_WRF_EXACT\s+.*?#endif", "",
                       _masked(source)[start:end], flags=re.S)
    return KernelSpec("acoustic", entry, tuple(
        PointerSpec(part.split()[-1].lstrip("*"), "member")
        for part in signature.split(",") if "*" in part))


def _body(source: str, entry: str) -> tuple[int, int]:
    masked = _masked(source)
    opening = masked.index("{", _signature_end(source, entry)[1])
    return opening + 1, _close(masked, opening, "{", "}")


def _occurrences(source: str, entry: str, statement: str) -> tuple[list[int], list[int]]:
    """Offsets of one statement in an entry body: (compiled arm, other arms)."""
    start, end = _body(source, entry)
    active = _active_source(source, OPTIONS)
    compiled, other = [], []
    for match in re.finditer(re.escape(statement), source[start:end]):
        offset = start + match.start()
        (compiled if active[offset:offset + len(statement)] == statement
         else other).append(offset)
    return compiled, other


def _fuse(source: str, pair: str, *, shared: bool):
    mu, w = (_spec(source, entry) for entry in PAIRS[pair])
    return fusion.fusion_source(source, mu, w, OPTIONS, shared_intermediates=shared)


def _insert(source: str, offset: int, text: str) -> str:
    return source[:offset] + text + source[offset:]


@pytest.mark.parametrize("tier", sorted(SOURCES))
@pytest.mark.parametrize("pair", sorted(PAIRS))
@pytest.mark.parametrize("shared", (False, True), ids=("simple", "shared"))
def test_the_installed_kernel_is_admitted(tier: str, pair: str, shared: bool) -> None:
    """The stale-audit direction: the kernel this tree ships must not be refused."""

    source = SOURCES[tier]
    fused, names, receipt = _fuse(source, pair, shared=shared)
    appended = fused[len(source):]
    assert fused.startswith(source) and names
    assert "void " + fusion.ENTRY + "(" in appended
    # Only the compiled arm is joined: no conditional line and no strict call.
    assert re.search(r"^[ \t]*#", appended, re.M) is None
    assert "wrf_advance_mu_theta" not in appended
    assert receipt["column_entries"] == PAIRS[pair]
    if not shared:
        assert receipt["shared_phase_inventory"] == []
        return
    mu_phase, w_phase = receipt["shared_phase_inventory"]
    assert mu_phase["mirrored_stores"] == AUDITED_MU_STORES
    assert w_phase["mirrored_stores"] == dict.fromkeys(AUDITED_MU_STORES, 0)
    assert all(count > 0 for count in w_phase["shared_reads"].values()), (
        "the W phase no longer reads one of the three Mu intermediates; the "
        "joined cache would carry a row nothing consumes", w_phase)


def test_the_mapped_mu_kernel_keeps_a_second_body_the_audit_must_not_count() -> None:
    """Non-vacuity for the incident: both arms exist, and only one is compiled.

    If the kernel ever drops its second body this witness has nothing left to
    hold and can be retired with it; the admission test above still stands.
    """

    start, end = _body(SOURCE, "advance_mu_th_msf")
    compiled, other = _occurrences(SOURCE, "advance_mu_th_msf", "ww_pp[c] = 0.0f;")
    assert len(compiled) == 1 and len(other) == 1, (compiled, other)
    # Independent of the audit's own preprocessor view: the default body is
    # the one after the block's last #else.
    divide = SOURCE.rindex("\n#else\n", start, end)
    assert other[0] < divide < compiled[0]
    raw = SOURCE[start:end]
    counted = {name: len(re.findall(r"\b" + name + r"\s*\[[^\]]*\]\s*\+?=(?!=)", raw))
               for name in AUDITED_MU_STORES}
    assert counted == {"th_pp_old": 2, "th_pp": 4, "ww_pp": 6}, counted


@pytest.mark.parametrize("pair", sorted(PAIRS))
def test_a_store_added_to_the_compiled_arm_is_refused(pair: str) -> None:
    """A fourth Omega store would be a global output the joined cache never sees."""

    mu = PAIRS[pair][0]
    compiled, _ = _occurrences(SOURCE, mu, "ww_pp[c] = 0.0f;")
    doctored = _insert(SOURCE, compiled[0], "ww_pp[c] = 0.0f;\n    ")
    with pytest.raises(BatchStateUnsupported, match="Mu intermediate store inventory changed"):
        _fuse(doctored, pair, shared=True)
    # The simple concatenation keeps no cache, so it has no inventory to hold.
    _fuse(doctored, pair, shared=False)


@pytest.mark.parametrize("pair", sorted(PAIRS))
def test_a_store_removed_from_the_compiled_arm_is_refused(pair: str) -> None:
    mu = PAIRS[pair][0]
    statement = "ww_pp[(size_t)nz * st + c] = 0.0f;"
    compiled, _ = _occurrences(SOURCE, mu, statement)
    assert len(compiled) == 1
    doctored = SOURCE[:compiled[0]] + SOURCE[compiled[0] + len(statement):]
    with pytest.raises(BatchStateUnsupported, match="Mu intermediate store inventory changed"):
        _fuse(doctored, pair, shared=True)


def test_a_store_added_to_the_uncompiled_arm_changes_nothing() -> None:
    """The other direction of the fix: an arm the compiler drops is not audited."""

    _, other = _occurrences(SOURCE, "advance_mu_th_msf", "ww_pp[c] = 0.0f;")
    doctored = _insert(SOURCE, other[0], "ww_pp[c] = 0.0f;\n    ")
    fused, _, receipt = _fuse(doctored, "mapped", shared=True)
    clean, _, _ = _fuse(SOURCE, "mapped", shared=True)
    assert receipt["shared_phase_inventory"][0]["mirrored_stores"] == AUDITED_MU_STORES
    assert fused[len(doctored):].split() == clean[len(SOURCE):].split(), (
        "an edit inside an uncompiled arm reached the joined kernel's text")


@pytest.mark.parametrize("pair,statement,replacement", (
    ("flat", "+ ww_pp[(size_t)k * st + c])", "+ ww_pp[(size_t)(k + 0) * st + c])"),
    ("mapped", "+ ww_pp[(size_t)kr * st + c])", "+ ww_pp[(size_t)(kr + 0) * st + c])"),
))
def test_an_index_form_the_audit_has_not_seen_is_refused(
        pair: str, statement: str, replacement: str) -> None:
    """A new index needs its level and ownership stated before it is rewritten."""

    w = PAIRS[pair][1]
    compiled, _ = _occurrences(SOURCE, w, statement)
    assert compiled, f"{w} no longer reads {statement!r} in its compiled arm"
    doctored = SOURCE[:compiled[-1]] + replacement + SOURCE[compiled[-1] + len(statement):]
    with pytest.raises(BatchStateUnsupported, match="fused shared column index changed"):
        _fuse(doctored, pair, shared=True)


@pytest.mark.parametrize("pair", sorted(PAIRS))
def test_a_w_phase_store_to_a_mu_intermediate_is_refused(pair: str) -> None:
    """The W phase only reads the three; a write there changes whose row it is."""

    w = PAIRS[pair][1]
    statement = "real muts = mut + mu_pp[c];"
    compiled, _ = _occurrences(SOURCE, w, statement)
    assert compiled, f"{w} lost the anchor statement {statement!r}"
    doctored = _insert(SOURCE, compiled[-1] + len(statement), "\n    th_pp[c] = muts;")
    with pytest.raises(BatchStateUnsupported,
                       match="the W phase writes an intermediate shared only with Mu"):
        _fuse(doctored, pair, shared=True)


@pytest.mark.parametrize("pair", sorted(PAIRS))
@pytest.mark.parametrize("shared", (False, True), ids=("simple", "shared"))
def test_a_definition_inside_a_column_body_is_refused(pair: str, shared: bool) -> None:
    """The joined phase is the option-resolved body, which carries no directives.

    A macro defined inside a body would be compiled by the separate launch and
    silently absent from the joined one.
    """

    mu = PAIRS[pair][0]
    compiled, _ = _occurrences(SOURCE, mu, "    mu_pp_old[c] = mu_pp[c];")
    assert compiled
    doctored = _insert(SOURCE, compiled[-1], "#define FUSION_AUDIT_PROBE 1\n")
    with pytest.raises(BatchStateUnsupported,
                       match="preprocessor directive other than a conditional"):
        _fuse(doctored, pair, shared=shared)
