"""A memory refusal names only ways out that are not refused again.

When the smallest layout of a ladder does not fit the card, the wizard
names the levers that would move the refusal: a lighter physics suite, a
shallower ladder, a larger card.  Each named lever has to be one the same
door then accepts.  A suite that is cheaper than the refused one but
still over the same budget is not a way out: following it reproduces the
refusal that named it.
"""

from __future__ import annotations

import re

import pytest

from gpuwm.cli import main as cli_main
from gpuwm.core.preflight import EXTERNAL_MARGIN_BYTES, GIB
from gpuwm.domain_wizard import (DEFAULT_PHYSICS_PROFILE,
                                 WIZARD_PHYSICS_PROFILES, LighterProfiles,
                                 _ladder_request_flags,
                                 _lighter_profiles_than,
                                 _minimum_layout_memory_remedy,
                                 profile_route_blocker,
                                 single_domain_runtime_switches)


def _advised(message: str) -> list[str]:
    match = re.search(r"a lighter --physics-profile \(([^)]+)\)", message)
    if match is None:
        return []
    return [name.strip() for name in match.group(1).split(",")]


# One 3 km domain on a small declared card: the default suite does not fit
# even at the minimum layout, and several lighter suites price below it,
# some of them still over the budget.
_SMALL_CARD_REQUEST = ["domain", "--point", "35.5,-97.5", "--root-dx", "3",
                       "--card", "5gb", "--hours", "1", "--source", "hrrr",
                       "--cycle", "2026-09-27T19"]


@pytest.mark.parametrize("choices", [
    None,
    # Scheme choices stay on the request that follows the advice, over the
    # suite it names, so a suite is priced with them.  Thompson over every
    # lighter suite leaves none within the budget; no cumulus leaves some.
    '{"microphysics": "thompson-mp8"}',
    '{"cumulus": "off"}',
])
def test_every_advised_suite_is_admitted_on_the_same_card(
        tmp_path, capsys, choices):
    """Follow each suite the refusal names; none may be refused again."""

    request = [*_SMALL_CARD_REQUEST,
               *([] if choices is None else ["--physics-choices", choices])]
    refused = tmp_path / "refused.toml"
    rc = cli_main([*request, "--out", str(refused)])
    captured = capsys.readouterr()
    message = captured.err + captured.out
    # Watched firing: the minimum-layout memory refusal, not another gate.
    assert rc == 2, message
    assert "does not fit" in message and "minimum layout" in message, message
    assert not refused.exists()
    advised = _advised(message)
    if not advised:
        assert "no lighter shipped --physics-profile fits" in message, message
    for rank, suite in enumerate(advised):
        out = tmp_path / f"advised-{rank}.toml"
        rc = cli_main([*request, "--physics-profile", suite,
                       "--out", str(out)])
        captured = capsys.readouterr()
        follow = captured.err + captured.out
        assert rc == 0 and out.exists(), (
            f"the refusal advised {suite} at #{rank + 1} and the same "
            f"request with it was refused: {follow}")
    # A single-domain ladder at its minimum layout has no shallower
    # ladder to fall back to, so that lever is not named.
    assert "shallower ladder" not in message
    assert "a larger card (this suite needs about" in message


def _named_shallower(message: str) -> str | None:
    match = re.search(r"a shallower ladder \(--ladder (\S+) fits", message)
    return None if match is None else match.group(1)


def test_a_named_shallower_ladder_is_the_deepest_one_that_fits(
        tmp_path, capsys):
    """The ladder lever names the deepest shallower ladder the card fits.

    Across cards that refuse the deepest preset at its minimum layout,
    the shallower ladder named is exactly the deepest shallower preset
    the same request then admits, and none is named when every one of
    them is refused again.
    """

    def _run(ladder: str, vram: str, out) -> tuple[int, str]:
        rc = cli_main(["domain", "--point", "39.1,-94.6", "--ladder", ladder,
                       "--vram-gib", vram, "--hours", "6",
                       "--cycle", "2026-08-12T00", "--out", str(out)])
        captured = capsys.readouterr()
        return rc, captured.err + captured.out

    shallower_presets = ("12-3-1", "12-3", "12")
    named_any = False
    for vram in ("4", "4.5", "4.6"):
        rc, message = _run("12-3-1-0.5", vram, tmp_path / f"deep-{vram}.toml")
        # Watched firing: the minimum-layout memory refusal on every card.
        assert rc == 2, message
        assert "minimum layout" in message, message
        named = _named_shallower(message)
        deepest_admitted = None
        for ladder in shallower_presets:
            follow_rc, _follow = _run(ladder, vram,
                                      tmp_path / f"{ladder}-{vram}.toml")
            if follow_rc == 0:
                deepest_admitted = ladder
                break
        assert named == deepest_admitted, (
            f"on a {vram} GiB card the refusal named {named!r} as the "
            f"shallower ladder, and the deepest the same request admits "
            f"is {deepest_admitted!r}: {message}")
        named_any = named_any or named is not None
    assert named_any, "no card in the walk named a shallower ladder"


def _admissible_lighter(source: str) -> list[str]:
    names = []
    for name in WIZARD_PHYSICS_PROFILES:
        if name == DEFAULT_PHYSICS_PROFILE:
            continue
        try:
            single_domain_runtime_switches(name)
        except ValueError:
            continue
        if profile_route_blocker(name, source) is None:
            names.append(name)
    return names


def test_lighter_suites_are_filtered_by_the_budget():
    """Cheaper is necessary, not sufficient: the suite must fit the budget."""

    source = "gfs"
    candidates = _admissible_lighter(source)
    assert len(candidates) >= 4
    prices = {name: (index + 1) * GIB // 4
              for index, name in enumerate(candidates)}
    prices[DEFAULT_PHYSICS_PROFILE] = (len(candidates) + 1) * GIB // 4
    price = prices.get

    budget = 3 * GIB // 4
    result = _lighter_profiles_than(DEFAULT_PHYSICS_PROFILE, source, price,
                                    budget_bytes=budget)
    assert result.fitting, "suites priced within the budget must be named"
    assert all(prices[name] <= budget for name in result.fitting)
    # Smallest step down first, among the suites that fit.
    assert list(result.fitting) == sorted(
        result.fitting, key=prices.__getitem__, reverse=True)
    assert result.lightest == (candidates[0], prices[candidates[0]])
    assert result.compared

    nothing = _lighter_profiles_than(DEFAULT_PHYSICS_PROFILE, source, price,
                                     budget_bytes=GIB // 8)
    assert nothing.fitting == ()
    assert nothing.lightest == (candidates[0], prices[candidates[0]])


def test_remedy_says_when_no_lighter_suite_fits_and_names_the_card():
    envelope = int(4.28 * GIB)
    budget = int(3.75 * GIB)
    free = budget + EXTERNAL_MARGIN_BYTES
    text = _minimum_layout_memory_remedy(
        lighter=LighterProfiles(lightest=("suite-a", int(4.0 * GIB))),
        envelope_bytes=envelope, free_bytes=free, budget_bytes=budget,
        source="gfs", shallower=None)
    assert _advised(text) == []
    assert "no lighter shipped --physics-profile fits this budget" in text
    assert "suite-a, needs 4.00 GiB here, 0.25 GiB over" in text
    need = (envelope + EXTERNAL_MARGIN_BYTES) / GIB
    assert f"needs about {need:.2f} GiB free" in text
    assert f"presents about {free / GIB:.2f} GiB" in text
    assert "shallower ladder" not in text


def test_remedy_says_when_no_suite_prices_below_the_refused_one():
    budget = int(3.75 * GIB)
    text = _minimum_layout_memory_remedy(
        lighter=LighterProfiles(compared=True),
        envelope_bytes=int(4.28 * GIB),
        free_bytes=budget + EXTERNAL_MARGIN_BYTES, budget_bytes=budget,
        source="gfs", shallower=None)
    assert _advised(text) == []
    assert ("no lighter shipped --physics-profile fits this budget either: "
            "none that gfs can run prices below this one here") in text
    assert "a larger card" in text
    # Nothing was compared (a template's own suite): nothing is said of it.
    silent = _minimum_layout_memory_remedy(
        lighter=LighterProfiles(),
        envelope_bytes=int(4.28 * GIB),
        free_bytes=budget + EXTERNAL_MARGIN_BYTES, budget_bytes=budget,
        source="gfs", shallower=None)
    assert "physics-profile" not in silent


@pytest.mark.parametrize("shallower", [None, "--ladder 12"])
def test_remedy_names_fitting_suites_and_only_a_ladder_that_fits(shallower):
    budget = int(3.75 * GIB)
    text = _minimum_layout_memory_remedy(
        lighter=LighterProfiles(fitting=("suite-b", "suite-c"),
                                lightest=("suite-c", int(3.0 * GIB))),
        envelope_bytes=int(4.28 * GIB),
        free_bytes=budget + EXTERNAL_MARGIN_BYTES, budget_bytes=budget,
        source="gfs", shallower=shallower)
    assert _advised(text) == ["suite-b", "suite-c"]
    assert "no lighter" not in text
    if shallower is None:
        assert "shallower ladder" not in text
    else:
        assert f"a shallower ladder ({shallower} fits" in text
    assert "a larger card" in text


@pytest.mark.parametrize(("ratios", "root_dx_m", "flags"), [
    ((), 12000.0, "--ladder 12"),
    ((4, 3), 12000.0, "--ladder 12-3-1"),
    ((), 3000.0, "--root-dx 3"),
    ((3,), 3000.0, "--root-dx 3 --chain 3"),
    ((3, 2), 4500.0, "--root-dx 4.5 --chain 3,2"),
])
def test_a_named_ladder_is_given_as_the_flags_that_request_it(
        ratios, root_dx_m, flags):
    assert _ladder_request_flags(ratios, root_dx_m) == flags
