"""``PHYSICS.md``'s reachability prose, derived from the things that own it.

WHY THIS FILE EXISTS
--------------------
Twice in one week a reader concluded from this page that MYNN's physics
was hardcoded and could not be run with longwave radiation.  Once it was
an outside collaborator, once one of this project's own agents.  Both
readings were wrong, both were reasonable, and both came from the same
two sentences: a ``reachability`` column that says how a MENU can offer
an option, printed with no hint that the config route composes freely,
and a MYNN row that said the scheme was "admitted only as the coupled
5/5 pair".

Fixing the prose is not enough, because the prose was not wrong when it
was written -- it DRIFTED.  Measured while writing this file:

* the revised MM5 row said ``unreachable`` and "no route allows the
  override" while the registry had it at ``component-override`` and the
  prepared-domain-tree route listed it in
  ``allowed_component_options.surface_layer``;
* the MYNN rows said the PBL was admitted only as the 5/5 pair, while
  the loader accepts ``bl_pbl_physics=5`` beside surface layers 91 and 1
  as well -- 48, 48 and 72 distinct accepted combinations in the
  composition walk.  The real restriction runs the other way and belongs
  to the MYNN SURFACE LAYER, which WRF v4.6.1 admits only with MYNN or
  no PBL.

So this file derives every reachability claim on the page from the
object that owns it, and asserts the page matches:

======================================  ==============================
claim on the page                       authority
======================================  ==============================
each surface-layer row's reachability   ``physics_registry_v2.json``
                                        ``components.*.reachability``
the definition block quoted for         the registry's own
``reachability.state``                  ``authority.reachability_
                                        declaration``
which options are registry-             the registry, cross-checked
``unreachable``, and what a config      against the accepted counts in
naming each one actually gets           the composition-walk receipt
the twelve pinned MYNN knobs            ``gpuwm.config.MYNN_PBL_
                                        OPTION_IDENTITY``
the refusal text quoted for a moved     ``build_experiment`` itself,
knob, and for YSU + MYNN surface        called on a real config
the ``implemented-unverified`` census   the registry
======================================  ==============================

Two of these are executed rather than read, because a quoted error
message is the one kind of documentation a user compares against their
own terminal character by character.
"""
from __future__ import annotations

import json
import pathlib
import re
import tomllib
import warnings

import pytest

from gpuwm.config import MYNN_PBL_OPTION_IDENTITY
from gpuwm.experiment import build_experiment

ROOT = pathlib.Path(__file__).resolve().parent.parent
PHYSICS_MD = ROOT / "docs" / "public" / "PHYSICS.md"
REGISTRY = ROOT / "gpuwm" / "physics_registry_v2.json"
WALK = (ROOT / "docs" / "public" / "receipts"
        / "physics-composition-walk.json")


def _page() -> str:
    return PHYSICS_MD.read_text(encoding="utf-8")


def _registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _walk() -> dict:
    return json.loads(WALK.read_text(encoding="utf-8"))


def _squash(text: str) -> str:
    """Collapse whitespace so a reflowed paragraph still matches."""
    return " ".join(text.split())


def _prose(text: str) -> str:
    """The page as prose: block-quote markers and code ticks removed.

    Only a LEADING ``>`` is a block-quote marker.  Stripping every ``>``
    would also eat the ones inside ``<component>``, which is how the
    first draft of this file failed against a correct page.
    """
    lines = [re.sub(r"^\s*> ?", "", line) for line in text.splitlines()]
    return _squash("\n".join(lines).replace("`", ""))


#: A daylight one-hour window at a plains point, on the smallest grid the
#: vertical contract admits for every scheme here.  Daylight on purpose:
#: the 1.7.1 nocturnal guard would otherwise answer a different question.
_CONFIG = """
[experiment]
name = "physics-md-reachability-claims"
start_time = 2021-06-01T18:00:00
run_seconds = 3600.0
restart_interval_s = 0.0

[projection]
map_proj = "lambert"
ref_lat = 38.0
ref_lon = -97.0
truelat1 = 30.0
truelat2 = 45.0
stand_lon = -97.0

[shared]
nz = 40
ztop = 20000.0
map_proj = 1
moist = true
cudt_minutes = 0.0
mp_physics = 6
bl_pbl_physics = {pbl}
sf_sfclay_physics = {sfclay}
sf_surface_physics = 2
num_soil_layers = 4
ra_lw_physics = 4
ra_sw_physics = 4
{extra}

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
history_interval_s = 3600.0
e_we = 41
e_sn = 41
dx = 3000.0
dy = 3000.0
time_step = 15
specified = true
"""


def _load(*, pbl: int = 5, sfclay: int = 5, extra: str = ""):
    """Push one config through the front door; return the run or raise."""
    raw = tomllib.loads(_CONFIG.format(pbl=pbl, sfclay=sfclay, extra=extra))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_experiment(
            raw, source="physics-md-reachability-claims.toml").root.run


def _refusal(**kwargs) -> str:
    with pytest.raises(ValueError) as caught:
        _load(**kwargs)
    return str(caught.value)


# ---------------------------------------------------------------------------
# The reachability column itself.
# ---------------------------------------------------------------------------

#: page label -> registry component/option, for the one table on the page
#: that prints the reachability column.
_SURFACE_LAYER_ROWS = {
    "MM5 (classic)": "classic-mm5",
    "MYNN": "mynn",
    "MM5 (revised)": "revised-mm5",
}


def test_the_surface_layer_reachability_column_is_the_registry_s():
    """The column claims to be quoted verbatim, so verify it is.

    This is the assertion that would have caught the live drift: the page
    printed ``unreachable`` for the revised MM5 surface layer long after
    the registry had moved it to ``component-override`` and a route had
    started offering it.
    """
    options = _registry()["components"]["surface_layer"]["options"]
    page = _page()

    wrong = []
    for label, option_id in _SURFACE_LAYER_ROWS.items():
        state = options[option_id]["reachability"]["state"]
        maturity = options[option_id]["maturity"]
        pattern = re.compile(
            r"^\| " + re.escape(label) + r" \| \d+ \| ([^|]+?) \| ([^|]+?) \|",
            re.M)
        match = pattern.search(page)
        assert match is not None, (
            f"the surface-layer table has no row for {label!r}")
        if match.group(1).strip() != maturity:
            wrong.append(f"{label}: maturity page={match.group(1).strip()!r} "
                         f"registry={maturity!r}")
        if match.group(2).strip() != state:
            wrong.append(f"{label}: reachability "
                         f"page={match.group(2).strip()!r} "
                         f"registry={state!r}")
    assert wrong == [], (
        "docs/public/PHYSICS.md disagrees with the registry it quotes: "
        + "; ".join(wrong))


def test_the_page_quotes_the_registry_s_own_definition_of_reachability():
    """The definition is the registry's, not this page's paraphrase.

    A paraphrase is what drifted last time.  The page quotes the
    declaration as a block quote, so every clause of the declaration must
    appear in the page with its whitespace collapsed.
    """
    declaration = _squash(
        _registry()["authority"]["reachability_declaration"])
    page = _prose(_page())

    # The page may stop the quote before the trailing sentence about the
    # recomputation test, but every clause that DEFINES a state must be
    # there verbatim.
    definition = declaration.split(" implemented and reachable")[0]
    assert definition in page, (
        "the page no longer quotes the registry's own definition of "
        f"reachability.state verbatim. The registry says:\n{definition}")


def test_every_registry_unreachable_option_is_published_with_what_it_does():
    """`unreachable` is a menu verdict, so the page must print both halves.

    Three of the five registry-``unreachable`` options are accepted by
    ``build_experiment`` when a config names them directly, and two are
    genuinely refused.  Publishing only the registry state is what
    produced "so no config can reach it", which was false.  The page must
    carry a row for every unreachable option and, where the loader
    accepts it, the walk's accepted count.
    """
    registry = _registry()
    walk = _walk()
    page = _page()

    unreachable = {
        f"{group}/{name}": option
        for group, body in registry["components"].items()
        for name, option in body["options"].items()
        if option["reachability"]["state"] == "unreachable"
    }
    assert unreachable, "the registry declares no unreachable option at all"

    problems = []
    for key, option in unreachable.items():
        selectors = option["selectors"]
        if not selectors:
            # Nothing can resolve to it; there is no count to publish.
            continue
        # An option is only as reachable as its LEAST admitted selector:
        # reaching it means setting all of them at once, so a selector
        # that never appears in an accepted run makes the whole option
        # unreachable however popular its siblings are.  wrf-rrtm-dudhia
        # is exactly that case -- ra_sw_physics=1 is accepted 157 times
        # (Dudhia shortwave), ra_lw_physics=1 never, and taking the max
        # here would have published the pair as reachable.
        accepted = min(
            walk["per_axis"][name][str(value)]["accepted"]
            for name, value in selectors.items()
        )
        if accepted == 0:
            continue
        if f"**{accepted} accepted**" not in page:
            problems.append(
                f"{key}: the loader accepts {accepted} combinations naming "
                f"{selectors}, and the page does not say so")
    assert problems == [], (
        "docs/public/PHYSICS.md publishes a registry 'unreachable' state "
        "without the measured loader verdict beside it: "
        + "; ".join(problems))


def test_the_page_does_not_carry_the_retired_template_only_readings():
    """The exact sentences that misled two readers may not come back."""
    page = _squash(_page())
    retired = (
        "so no config can reach it",
        "admitted only as the coupled 5/5 pair",
        "only selectable as the 5/5 pair",
    )
    present = [phrase for phrase in retired if phrase in page]
    assert present == [], (
        "docs/public/PHYSICS.md has regrown a reading that was measured "
        f"false: {present}")


# ---------------------------------------------------------------------------
# What IS pinned.  The scope limit must stay stated, and stated correctly.
# ---------------------------------------------------------------------------

def test_the_page_lists_the_whole_mynn_option_identity():
    """Every pinned knob, with the value the code actually admits.

    A partial list reads as "these few are pinned", which understates a
    real scope limit in the same way the reachability column overstated
    one.
    """
    page = _page()
    missing = [
        name for name in MYNN_PBL_OPTION_IDENTITY
        if f"`{name}`" not in page
    ]
    assert missing == [], (
        "docs/public/PHYSICS.md does not publish every pinned MYNN knob: "
        f"{missing}")
    assert f"{len(MYNN_PBL_OPTION_IDENTITY)} knobs" in page.replace(
        "Twelve", "12"), (
        "the page's count of pinned MYNN knobs is not "
        f"{len(MYNN_PBL_OPTION_IDENTITY)}")


def test_the_quoted_mynn_identity_refusal_is_the_one_the_loader_emits():
    """A user compares a quoted error against their terminal, character
    for character, so this one is executed rather than transcribed."""
    message = _refusal(extra="bl_mynn_mixlength = 2")
    assert _squash(message) in _squash(_page()), (
        "docs/public/PHYSICS.md quotes a MYNN option-identity refusal that "
        f"the loader does not emit. Measured:\n{message}")


def test_the_quoted_pairing_refusal_is_the_one_the_loader_emits():
    """Same, for the one genuine MYNN pairing restriction.

    It belongs to the MYNN SURFACE LAYER, not the MYNN PBL, and the page
    now says so; if that ever inverts again the quoted message goes with
    it.
    """
    message = _refusal(pbl=1, sfclay=5)
    assert _squash(message) in _squash(_page()), (
        "docs/public/PHYSICS.md quotes a PBL/surface-layer refusal that the "
        f"loader does not emit. Measured:\n{message}")


def test_the_mynn_pbl_composes_with_all_three_surface_layers():
    """The control for the sentence above: the restriction is one-sided.

    Without this, "the surface layer restricts the PBL slot" could be
    rewritten back into "the pair is mandatory" and the refusal test
    above would still pass.
    """
    for sfclay in (5, 91, 1):
        run = _load(pbl=5, sfclay=sfclay)
        assert (run.bl_pbl_physics, run.sf_sfclay_physics) == (5, sfclay), (
            "the loader rewrote a switch instead of accepting or refusing "
            f"it: asked 5/{sfclay}, resolved "
            f"{run.bl_pbl_physics}/{run.sf_sfclay_physics}")


def _key_fields(key: str) -> dict[str, int]:
    """The walk's compact combination key, back into a switch dict.

    ``key_of`` in tools/report_physics_composition_walk.py writes
    ``mp8.pbl5.sl5.lsm2.lw4.sw4.cu1.km1``, optionally with a
    ``raagg<n>`` term or a bare ``ack``/``noack`` flag.  The flags carry
    no number and are dropped.
    """
    fields = {}
    for token in key.split("."):
        match = re.fullmatch(r"([a-z]+)(-?\d+)", token)
        if match:
            fields[match.group(1)] = int(match.group(2))
    return fields


def _distinct_accepted_by_pbl_and_surface_layer() -> dict[tuple[int, int], int]:
    """DISTINCT accepted suites per (bl_pbl_physics, sf_sfclay_physics).

    Counted off ``accepted_combinations``, which the walk documents as
    deduplicated configurations rather than attempts.
    """
    counts: dict[tuple[int, int], int] = {}
    for key in _walk()["accepted_combinations"]:
        fields = _key_fields(key)
        pair = (fields["pbl"], fields["sl"])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def test_the_mynn_pairing_counts_on_the_page_are_the_receipts_own():
    """The three numbers in the MYNN scope note, recomputed.

    They were written by hand from a walk run during authoring and
    nothing pinned them, which is how a receipt regeneration could move
    the measurement while the page kept quoting the old figure.

    The mapping is pinned, not just the multiset: the pairings are read
    out of the same sentence as the counts and zipped in the order the
    page prints them, so swapping "5/91, 5/1 and 5/5" without swapping
    "48, 48 and 72" goes red even though both lists are unchanged.

    The 72-versus-73 trap is pinned too.  ``mynn_slice.accepted`` in the
    same receipt reads 73 for the 5/5 pairing because it counts
    ATTEMPTS and the walk re-tries its 5/5 anchor in a second tier.
    Both numbers are correct about different things, which is exactly
    the shape of disagreement a reader cannot resolve alone, so the page
    now states both and this test holds both to the receipt.
    """
    page = _squash(_page())
    match = re.search(
        r"The MYNN \*PBL\* carries no such restriction: (.+?) are all "
        r"accepted -- \*\*(.+?) distinct accepted combinations\*\*", page)
    assert match is not None, (
        "docs/public/PHYSICS.md no longer states the MYNN PBL pairing "
        "counts in the form this test pins; if the sentence was reworded, "
        "reword this pattern with it rather than deleting the pin")

    pairings = [(int(pbl), int(sfclay))
                for pbl, sfclay in re.findall(r"(\d+)/(\d+)",
                                              match.group(1))]
    claimed = [int(number) for number in re.findall(r"\d+", match.group(2))]
    assert len(pairings) == len(claimed) == 3, (
        f"expected three pairings and three counts, read {pairings} "
        f"and {claimed}")

    measured = _distinct_accepted_by_pbl_and_surface_layer()
    wrong = [
        f"{pbl}/{sfclay}: page says {count}, the receipt's "
        f"accepted_combinations has {measured.get((pbl, sfclay), 0)}"
        for (pbl, sfclay), count in zip(pairings, claimed)
        if measured.get((pbl, sfclay), 0) != count
    ]
    assert wrong == [], (
        "docs/public/PHYSICS.md's MYNN pairing counts disagree with "
        "docs/public/receipts/physics-composition-walk.json: "
        + "; ".join(wrong))

    attempts = _walk()["mynn_slice"]["accepted"]
    assert f"reads **{attempts}** for the 5/5 pairing" in page, (
        "the page states the mynn_slice ATTEMPT count so the two numbers "
        f"cannot be read as a contradiction; the receipt now says "
        f"{attempts}")
    assert attempts >= measured[(5, 5)], (
        "attempts cannot be fewer than distinct configurations; one of "
        "the two receipt fields has changed meaning")


def test_the_mynn_pairing_count_pin_can_fail():
    """Control: the recomputation is a measurement, not a tautology.

    If ``_distinct_accepted_by_pbl_and_surface_layer`` silently returned
    whatever the page said, the test above would pass against any
    number.  A pairing the walk never accepts must count zero.
    """
    measured = _distinct_accepted_by_pbl_and_surface_layer()
    assert measured.get((1, 5), 0) == 0, (
        "YSU + MYNN surface layer is the refused pairing quoted on the "
        "page; the walk must not report it accepted")
    assert measured[(5, 5)] > 0


def test_mynn_with_both_radiation_streams_on_is_accepted_and_preserved():
    """The claim the whole incident was about, executed."""
    run = _load(pbl=5, sfclay=5)
    assert (run.ra_lw_physics, run.ra_sw_physics) == (4, 4)
    assert (run.bl_pbl_physics, run.sf_sfclay_physics) == (5, 5)
    for name, admitted in MYNN_PBL_OPTION_IDENTITY.items():
        assert getattr(run, name) == admitted, (
            f"{name} moved off the MYNN option identity while radiation "
            "was on")


# ---------------------------------------------------------------------------
# The maturity rung, which this work did not move.
# ---------------------------------------------------------------------------

def test_the_implemented_unverified_census_on_the_page_is_the_registry_s():
    """The rung is shared, and the page prints how widely.

    The number exists so a reader cannot read ``implemented-unverified``
    on a MYNN row as a mark against MYNN.  A rung change must move it.
    """
    registry = _registry()
    options = [
        option
        for body in registry["components"].values()
        for option in body["options"].values()
    ]
    at_rung = [o for o in options
               if o["maturity"] == "implemented-unverified"]
    claim = f"**{len(at_rung)} of the registry's {len(options)} component options**"
    assert claim in _page(), (
        "docs/public/PHYSICS.md no longer states the implemented-unverified "
        f"census correctly; measured {len(at_rung)} of {len(options)}")


def test_mynn_is_still_implemented_unverified_in_the_registry():
    """Naming a composition is not evidence; the rung must not have moved."""
    components = _registry()["components"]
    for group in ("pbl", "surface_layer"):
        assert components[group]["options"]["mynn"]["maturity"] == (
            "implemented-unverified"), (
            f"the MYNN {group} maturity moved; the page says it did not")
