"""The nowcast's defaults are pinned to the runs that measured them.

Every assertion here is a *measurement* the default rests on, not a
preference.  If one of these numbers moves, the doc line and the
comment beside it in ``tools/da_nowcast.py`` have to move with it, and
the run that justifies the new value has to exist.

Pure tests: no network, no GPU, no subprocess.  Sites are synthetic ids
-- real station names never enter the tree's generic code or its
fixtures (standing owner rule).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.da_nowcast import (
    CARD_PROFILES, DEFAULT_MEMBERS, DEFAULT_MEMORY_BUDGET_MIB,
    PROFILE_UNSET, WindowPlan, apply_card_profile, build_parser,
    cycle_cmd, plan_window)


def utc(*parts) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


def plan() -> WindowPlan:
    return plan_window(utc(2026, 8, 5, 5, 30), cycles=6,
                       cycle_seconds=900, free_legs=6,
                       now=utc(2026, 8, 5, 5, 45))


def parsed(*extra):
    return build_parser().parse_args(
        ["run", "--site", "qqqq", "--window-end", "latest",
         "--out", "case", *extra])


# ---------------------------------------------------------------------------
# 1. ensemble size
# ---------------------------------------------------------------------------
class TestEnsembleSizeDefault:
    def test_ten_is_the_default_on_the_front_door(self):
        assert DEFAULT_MEMBERS == 10
        assert parsed().members == 10

    def test_the_help_carries_the_measurement_not_a_preference(self):
        text = build_parser().format_help()
        # The argparse help for `run` lives on the subparser; reach it
        # through the same public surface a user does.
        run_help = _run_help()
        assert "measured" in run_help.lower()
        for cited in ("0.0018", "0.0062", "82%"):
            assert cited in run_help, cited
        assert "run" in text

    def test_the_penalty_direction_is_stated_where_a_user_reads_it(self):
        # The finding that makes this default non-obvious: the metric is
        # computed on the ensemble MEAN, and averaging penalises this
        # under-producing forecast.  A user who does not know that will
        # raise N expecting more skill.
        run_help = _run_help()
        assert "ensemble mean" in run_help
        assert "penalis" in run_help or "penaliz" in run_help


def _run_help() -> str:
    parser = build_parser()
    for action in parser._actions:           # noqa: SLF001 - argparse
        if getattr(action, "choices", None) and "run" in (
                action.choices or {}):
            return action.choices["run"].format_help()
    raise AssertionError("no `run` subparser")


# ---------------------------------------------------------------------------
# 2. memory
# ---------------------------------------------------------------------------
class TestMemoryBudgetDefault:
    def test_default_is_the_shipped_measured_to_fit_value(self):
        # 6144 MiB is what the 16 GB frontier ran at: whole-card peak
        # 15,888 MiB, 97.0% of a 16,376 MiB card, 488 MiB spare.
        # Unchanged deliberately -- the ladder that would lower it had
        # not finished.  See DEFAULT_MEMORY_BUDGET_MIB.
        assert DEFAULT_MEMORY_BUDGET_MIB == 6144.0
        assert parsed().memory_budget_mib == 6144.0

    def test_the_knob_reaches_the_cycle_driver(self):
        # It is the ONE term in the memory model an operator controls;
        # before this it was unreachable from the front door.
        argv = cycle_cmd(
            prepared_root="p", authority_dir="a", profile="prof",
            plan=plan(), members=10, obs_files=[], grid_wrfouts=[],
            cycle_out="c", proof_sha="x", manifest_sha="y",
            content_sha="z", seed=1, solve_device="cuda",
            horizontal_loc_m=12000.0, vertical_loc_m=3000.0,
            length_scale_km=50.0, source="gfs")
        assert argv[argv.index("--memory-budget-mib") + 1] == "6144"

    def test_an_explicit_budget_is_carried_through(self):
        argv = cycle_cmd(
            prepared_root="p", authority_dir="a", profile="prof",
            plan=plan(), members=10, obs_files=[], grid_wrfouts=[],
            cycle_out="c", proof_sha="x", manifest_sha="y",
            content_sha="z", seed=1, solve_device="cuda",
            horizontal_loc_m=12000.0, vertical_loc_m=3000.0,
            length_scale_km=50.0, source="gfs",
            memory_budget_mib=2048.0)
        assert argv[argv.index("--memory-budget-mib") + 1] == "2048"


# ---------------------------------------------------------------------------
# 3. the 16 GB profile
# ---------------------------------------------------------------------------
class TestCardProfiles:
    def test_auto_changes_nothing(self):
        args = parsed()
        assert args.profile == "auto"
        assert apply_card_profile(args) == {}
        assert args.vram_gib is None

    def test_16gib_profile_sizes_the_preflight_at_the_measured_card(self):
        # The frontier ran the shipped shape end to end on a 16,376 MiB
        # RTX 4080.  The profile does not shrink the run -- that is the
        # finding -- it points the preflight at the right card.
        args = parsed("--profile", "card-16gib")
        applied = apply_card_profile(args)
        assert applied == {"vram_gib": 16.0}
        assert args.vram_gib == 16.0
        # Nothing else moved: N, dx and the box are the measured shape.
        assert args.members == DEFAULT_MEMBERS
        assert args.dx_km == 3.0
        assert args.box_half_km == 198.0
        assert args.memory_budget_mib == DEFAULT_MEMORY_BUDGET_MIB

    def test_an_explicit_flag_beats_the_profile(self):
        args = parsed("--profile", "card-16gib", "--vram-gib", "24")
        assert apply_card_profile(args) == {}
        assert args.vram_gib == 24.0

    def test_every_profile_key_declares_its_unset_value(self):
        for name, profile in CARD_PROFILES.items():
            for key in profile:
                assert key in PROFILE_UNSET, f"{name}.{key}"

    def test_profiles_are_named_for_hardware_only(self):
        # Standing owner rule: no radar-site names in defaults or
        # generic code.  A profile names a card, or it is the
        # do-nothing default; nothing else is a legal name.
        for name in CARD_PROFILES:
            assert re.fullmatch(r"auto|card-\d+gib", name), name

    def test_the_run_help_survives_being_formatted(self):
        # argparse %-expands help strings, so a bare "97.0%" in a
        # citation takes `--help` down with a ValueError.  The
        # measurements below all carry percentages.
        assert "97.0%" in _run_help()
        assert "82%" in _run_help()


# ---------------------------------------------------------------------------
# 4. what stays off, and stays off silently
# ---------------------------------------------------------------------------
class TestOptInCapabilitiesStayOff:
    ARGV = dict(
        prepared_root="p", authority_dir="a", profile="prof",
        members=10, obs_files=[], grid_wrfouts=[], cycle_out="c",
        proof_sha="x", manifest_sha="y", content_sha="z", seed=1,
        solve_device="cuda", horizontal_loc_m=12000.0,
        vertical_loc_m=3000.0, length_scale_km=50.0, source="gfs")

    def test_the_nest_is_not_in_the_front_door_argv(self):
        # Cost model basis is "computed ... not measured"; the measured
        # A/B had not run.  Opt in on tools/da_cycle_prepared.py.
        argv = cycle_cmd(plan=plan(), **self.ARGV)
        assert not [a for a in argv if str(a).startswith("--nest")]

    def test_concurrent_members_are_not_in_the_front_door_argv(self):
        # Byte-identity proof still open; the lane is not in this tree.
        argv = cycle_cmd(plan=plan(), **self.ARGV)
        assert "--member-workers" not in argv

    def test_the_front_door_offers_no_flag_for_either(self):
        for flag in ("--nest-half-width-km", "--nest-members",
                     "--member-workers"):
            with pytest.raises(SystemExit):
                parsed(flag, "1")


# ---------------------------------------------------------------------------
# 5. the provenance receipt and the code cannot drift apart
# ---------------------------------------------------------------------------
PROVENANCE = (Path(__file__).resolve().parents[1] / "evidence"
              / "da-demo" / "defaults-provenance.json")


class TestDefaultsProvenance:
    @staticmethod
    def doc() -> dict:
        return json.loads(PROVENANCE.read_text(encoding="utf-8"))

    @staticmethod
    def entry(doc: dict, name: str) -> dict:
        for item in doc["defaults"]:
            if item["name"] == name:
                return item
        raise AssertionError(f"no provenance entry for {name}")

    def test_the_receipt_exists_and_names_its_pin(self):
        doc = self.doc()
        assert doc["schema"] == "gpuwm-da.defaults-provenance.v1"
        assert doc["pinned_by"] == "tests/" + Path(__file__).name

    def test_shipped_values_match_the_receipt(self):
        doc = self.doc()
        assert self.entry(doc, "members")["value"] == DEFAULT_MEMBERS
        assert (self.entry(doc, "memory_budget_mib")["value"]
                == DEFAULT_MEMORY_BUDGET_MIB)
        assert self.entry(doc, "profile")["value"] in CARD_PROFILES

    def test_every_entry_is_measured_or_says_it_is_pending(self):
        # The rule the defaults were set under: a default is either a
        # measurement with a receipt, or it is unchanged and listed as
        # pending.  Nothing may be silently preferred.
        for item in self.doc()["defaults"]:
            status = item["status"]
            has_receipt = bool(item.get("measurements")
                               or item.get("receipt"))
            assert has_receipt, item["name"]
            if "pending" in status.lower() or "OFF" in status:
                continue
            assert "measured" in status.lower(), item["name"]

    def test_the_two_off_by_default_capabilities_are_recorded_as_off(self):
        doc = self.doc()
        for name in ("nested free forecast", "concurrent members"):
            item = self.entry(doc, name)
            assert item["value"] is None
            assert item["status"].startswith("OFF")
