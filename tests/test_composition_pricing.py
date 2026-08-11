"""Every composition the loader accepts must also PRICE.

THE BLIND SPOT THIS CLOSES
    ``tools/report_physics_composition_walk.py`` measures one gate:
    :func:`gpuwm.experiment.build_experiment`.  A row it marks ACCEPTED
    means the loader resolved the TOML into per-domain ``RunConfig``s with
    every selector preserved -- and nothing more.  The memory preflight is
    a SECOND gate, reached on every real run:
    ``gpuwm/core/model.py`` calls
    :func:`gpuwm.core.preflight.estimate_experiment` unconditionally while
    building the model, and ``gpuwm check`` is that call and little else.
    Its selector tables (``_MICROPHYSICS_KERNEL_MODULES``,
    ``_PBL_KERNEL_MODULES``, ``_SURFACE_LAYER_KERNEL_MODULES``, ...) are
    FAIL-CLOSED: a selector value with no kernel-module row raises rather
    than pricing zero.

    So a scheme could be admitted by the loader, advertised in the schema
    menu, swept by the composition walk, recorded ACCEPTED in a published
    receipt -- and still be unable to run, because nothing ever asked the
    preflight to price it.  That is not hypothetical.  It shipped twice
    into the 1.9 assembly:

      * ``mp_physics = 50`` (P3 one-category) -- accepted at
        gpuwm/config.py's inline mp check, absent from
        ``_MICROPHYSICS_KERNEL_MODULES``;
      * ``bl_pbl_physics = 2`` / ``sf_sfclay_physics = 2`` (MYJ + the Eta
        surface layer) -- accepted via ``PBL_SCHEMES`` /
        ``SURFACE_LAYER_SCHEMES``, absent from both PBL tables.

    Both were caught by a human reading the tables, which is the failure
    mode this file exists to retire.  A new scheme that reaches the loader
    without a kernel-module row now fails HERE, on every cut, by name.

WHAT IS MEASURED
    The receipt's ``accepted_combinations`` -- the measured accepted set,
    regenerated and byte-compared by
    ``tests/test_physics_composition_walk.py`` -- is decoded back into
    combinations, rebuilt through the same front door, and pushed through
    ``estimate_experiment``.  Decoding is not trusted: every key must
    round-trip through the walk's own ``key_of``, and every rebuild must
    still be ACCEPTED, so a lossy decode fails loudly instead of quietly
    testing something else.

WHAT A REFUSAL IS ALLOWED TO BE
    Not every accepted composition has to price.  A preflight refusal is
    legitimate when it is a DECLARED rule: a stated, reasoned refusal that
    names the way through, rather than a missing table row.  Those live in
    :data:`DECLARED_PRICING_RULES`, one entry each, with the predicate
    that says which compositions it may fire on.  A refusal matching no
    declared rule is a defect.  A declared rule that fires on nothing is
    dead text and also fails -- the escape hatch is not allowed to grow
    quietly.
"""
from __future__ import annotations

import json
import re
import tomllib
import warnings
from pathlib import Path

import pytest

from gpuwm.core.preflight import estimate_experiment, physics_kernel_modules
from gpuwm.experiment import build_experiment
from tools import report_physics_composition_walk as walk

MODEL = Path(__file__).resolve().parents[1]
RECEIPT = (
    MODEL / "docs" / "public" / "receipts" / "physics-composition-walk.json"
)

#: The walk's compact key tags, inverted.  ``raagg`` is the legacy
#: ``ra_physics`` aggregate rather than an axis, so it is mapped by hand.
_TAG_TO_AXIS = {tag: name for name, tag in walk._TAG.items()}
_PART = re.compile(r"([a-z]+)(-?\d+)")


def decode(key: str) -> dict[str, object]:
    """One accepted key back into the combination that produced it."""

    combination: dict[str, object] = {}
    for part in key.split("."):
        if part == "ack":
            combination["km_opt_zero_acknowledgement"] = walk.KM_OPT_ZERO_ACK
            continue
        if part == "noack":
            combination["_ack_withheld"] = True
            combination["km_opt_zero_acknowledgement"] = ""
            continue
        if part == "noradack":
            combination["_radiation_ack_withheld"] = True
            continue
        matched = _PART.fullmatch(part)
        if matched is None:
            raise AssertionError(
                f"unrecognised accepted-combination key segment {part!r} in "
                f"{key!r}; this decoder and "
                "tools/report_physics_composition_walk.py:key_of have "
                "drifted apart")
        tag, value = matched.group(1), int(matched.group(2))
        if tag == "raagg":
            combination["ra_physics"] = value
        else:
            combination[_TAG_TO_AXIS[tag]] = value
    return combination


def build(combination: dict[str, object]):
    """Push one combination through the production front door."""

    text = walk.render_config(combination)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_experiment(
            tomllib.loads(text), source="composition-pricing.toml")


# ---------------------------------------------------------------------------
# The declared refusals.
# ---------------------------------------------------------------------------

def _selects_noahmp(combination: dict[str, object]) -> bool:
    return int(combination.get("sf_surface_physics", 0)) == 4


#: Preflight refusals that are RULES, not gaps.  Each entry is
#: ``(id, predicate, signature)``: the predicate says which compositions
#: the rule may fire on, and the signature is a fragment the refusal's own
#: message must contain.  Both halves matter -- a rule that fires outside
#: its predicate is a different defect wearing a declared rule's name.
DECLARED_PRICING_RULES = (
    (
        "noahmp-translation-unit-has-no-measured-composite-frame",
        _selects_noahmp,
        "cannot price the local-memory reservation",
        # gpuwm/core/preflight.py UNMEASURED_KERNEL_MODULES: Noah-MP's
        # driver/energy/thermal fragments compile only through
        # noahmp_kernel_sources.translation_unit_source and fail NVRTC
        # standalone, so their per-thread frame has never been measured.
        # The refusal is deliberate, is not a missing selector row, and
        # names both the way through (sf_surface_physics = 2) and the
        # fix (a CHAINED_TRANSLATION_UNIT_FRAMES row).  It becomes a
        # green row the day that composite frame is measured; nothing
        # here has to be remembered for that to happen, because the
        # dead-rule check below fails the moment it stops firing.
    ),
)


@pytest.fixture(scope="module")
def accepted_keys() -> tuple[str, ...]:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    keys = tuple(receipt["accepted_combinations"])
    assert keys, "the composition walk recorded no accepted combination"
    return keys


@pytest.fixture(scope="module")
def priced(accepted_keys) -> dict[str, object]:
    """Build and price every accepted composition, once, for the file."""

    ok: list[str] = []
    refused: list[tuple[str, dict[str, object], str]] = []
    rebuild_refused: list[tuple[str, str]] = []
    for key in accepted_keys:
        combination = decode(key)
        assert walk.key_of(combination) == key, (
            f"decoding {key!r} did not round-trip through key_of; the "
            "pricing walk would be measuring a different composition than "
            "the one the receipt accepted")
        try:
            experiment = build(combination)
        except Exception as error:  # noqa: BLE001 -- the verdict IS the error
            rebuild_refused.append((key, f"{type(error).__name__}: {error}"))
            continue
        try:
            estimate_experiment(experiment)
        except Exception as error:  # noqa: BLE001 -- the verdict IS the error
            refused.append((key, combination, str(error)))
        else:
            ok.append(key)
    return {"ok": tuple(ok), "refused": tuple(refused),
            "rebuild_refused": tuple(rebuild_refused)}


def test_every_accepted_composition_rebuilds(priced):
    """A receipt row that no longer builds means the decode is wrong."""

    assert priced["rebuild_refused"] == (), (
        "the receipt records these combinations as ACCEPTED but rebuilding "
        "them through build_experiment refused:\n  "
        + "\n  ".join(f"{key}: {message}"
                      for key, message in priced["rebuild_refused"][:10]))


def test_every_accepted_composition_can_be_priced(priced):
    """THE INSTRUMENT.  Loader-accepted implies preflight-priceable.

    An unpriceable composition is a scheme a user can select, that
    ``gpuwm check`` refuses and ``gpuwm run`` dies on.  The only refusals
    allowed through are the declared rules above, and only on the
    compositions their predicate admits.
    """

    undeclared = []
    for key, combination, message in priced["refused"]:
        flat = " ".join(message.split())
        matches = [rule_id
                   for rule_id, predicate, signature in DECLARED_PRICING_RULES
                   if signature in flat and predicate(combination)]
        if not matches:
            undeclared.append((key, flat))
    assert not undeclared, (
        f"{len(undeclared)} composition(s) the loader ACCEPTS cannot be "
        "priced by gpuwm/core/preflight.py, and the refusal is not a "
        "declared composition rule.  Every one of these is a scheme a user "
        "can select and cannot run: `gpuwm check` refuses it and "
        "gpuwm/core/model.py dies on the same call while building the "
        "model.  Add the missing kernel-module row (with honest pricing "
        "derived from the scheme's actual kernels), or declare the refusal "
        "in DECLARED_PRICING_RULES with its reason.\n  "
        + "\n  ".join(f"{key}: {message}" for key, message in undeclared[:10]))


def test_no_declared_pricing_rule_is_dead_text(priced):
    """An escape hatch nothing uses is an escape hatch nobody audits."""

    fired = {rule_id
             for key, combination, message in priced["refused"]
             for rule_id, predicate, signature in DECLARED_PRICING_RULES
             if signature in " ".join(message.split())
             and predicate(combination)}
    declared = {rule_id for rule_id, _, _ in DECLARED_PRICING_RULES}
    assert declared == fired, (
        "these declared pricing rules fired on nothing and are now dead "
        f"text: {sorted(declared - fired)}.  If the underlying refusal was "
        "resolved, delete the entry in the same commit that resolved it.")


def test_the_walk_priced_a_meaningful_share_of_its_accepted_set(priced,
                                                                accepted_keys):
    """Guard against a green that comes from measuring almost nothing."""

    assert len(priced["ok"]) > len(accepted_keys) // 2, (
        f"only {len(priced['ok'])} of {len(accepted_keys)} accepted "
        "compositions priced; a majority behind declared refusals means the "
        "declared-rule list has become the measurement")


# ---------------------------------------------------------------------------
# The two schemes this file was written for, named, so the failure reads.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,combination,expected_modules",
    [
        # P3 one-category on the shipped YSU/MM5/Noah anchor with the
        # Dudhia radiation pair.  NOT RTE+RRTMGP: P3 supplies no snow
        # radius, so gpuwm.config.validate_p3_radiation refuses that
        # pairing at the door (the alternative, found at this gate, was
        # dying at the first radiation call).
        #
        # P3 is a HOST transcription and launches no kernel of its own, so
        # what its row must produce is the shared moisture validator -- and
        # NOT ``refl``, whose absence is the other half of the honest price.
        ("mp_physics=50 (P3 one-category)",
         walk._suite(mp_physics=50, ra_lw_physics=0, ra_sw_physics=1),
         {"present": ("microphysics_validation",), "absent": ("refl",)}),
        # MYJ with its own Eta surface layer, the pairing
        # gpuwm.config.validate_myj_pairing enforces in both directions.
        ("bl_pbl_physics=2 + sf_sfclay_physics=2 (MYJ + Eta)",
         walk._suite(bl_pbl_physics=2, sf_sfclay_physics=2),
         {"present": ("myjpbl", "myjsfc"), "absent": ("ysu", "sfclay")}),
    ],
)
def test_the_1_9_resurrected_schemes_price(label, combination,
                                           expected_modules):
    """mp=50 and MYJ, priced by name, against the modules they launch.

    These two are covered by the receipt walk above; they are ALSO spelled
    out here so that dropping their kernel-module row fails with the
    scheme's name in the test id rather than as one line inside a list of
    991.  Both were accepted-but-unpriceable at b557703ba.

    The module assertions are what stop the blocker from being "fixed" by
    an empty row: a scheme priced at nothing prices fine and reserves
    nothing, which is the under-pricing direction that put a run 1,630 MiB
    over.  ``absent`` pins the two honest exclusions -- P3's ``refl`` and
    MYJ's YSU/MM5 modules -- so a copy-paste row fails too.
    """

    experiment = build(combination)
    modules = physics_kernel_modules(experiment)
    missing = [name for name in expected_modules["present"]
               if name not in modules]
    assert not missing, f"{label}: unpriced kernel module(s) {missing}"
    surplus = [name for name in expected_modules["absent"] if name in modules]
    assert not surplus, (
        f"{label}: priced kernel module(s) {surplus} the scheme never "
        "launches; over-pricing a rail gate refuses runs that would fit")
    estimate = estimate_experiment(experiment)
    assert estimate.alloc_estimate_bytes > 0, label
