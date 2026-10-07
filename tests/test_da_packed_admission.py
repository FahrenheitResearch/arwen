"""Packed DA forecasts run on any number of cards and members.

The packed step used to refuse every roster but 32 or 64 members on
exactly eight cards, although the waves were already general
(gpuwm.da.member_wave.plan_wave); that kept it off every one-to-four-card
box. The per-card width is now the most the priced trajectory fits.
"""
from __future__ import annotations

import inspect

import pytest

GIB = 1024**3


def _cards(n, free_gib, total_gib=None):
    total = (total_gib if total_gib is not None else free_gib)*GIB
    return [{"uuid": f"GPU-{i:08x}-0000-0000-0000-000000000000",
             "total_bytes": total, "free_bytes": free_gib*GIB}
            for i in range(n)]


@pytest.mark.parametrize("free_gib, peak_gib, expected", [
    (96, 20, 4),      # a 96 GB card: four members fit
    (32, 6, 4),
    (32, 10, 2),      # a 32 GB card: two fit, four do not
    (24, 12, 1),      # a 4090: two do not fit, one does (was a refusal)
    (16, 9, 1),       # a 5070 Ti
])
def test_auto_takes_the_widest_pack_that_fits(free_gib, peak_gib, expected):
    from tools.da_cycle_prepared import packed_members_per_card
    # with an MPS controller answering; without one auto stops at 2
    assert packed_members_per_card("auto", _cards(1, free_gib),
                                   peak_gib*GIB,
                                   mps_available=True) == expected


def test_auto_reads_the_tightest_card():
    from tools.da_cycle_prepared import packed_members_per_card
    cards = _cards(3, 96)
    cards[1]["free_bytes"] = 20*GIB
    assert packed_members_per_card("auto", cards, 7*GIB,
                                   mps_available=True) == 2


@pytest.mark.parametrize("cards", [1, 2, 3, 4, 8, 16])
def test_any_card_count_is_accepted(cards):
    from tools.da_cycle_prepared import packed_members_per_card
    assert packed_members_per_card("2", _cards(cards, 32), 10*GIB) == 2


def test_a_member_that_fits_nowhere_and_a_stated_width_that_does_not_fit_are_refused():
    from tools.da_cycle_prepared import packed_members_per_card
    with pytest.raises(ValueError, match="one packed member"):
        packed_members_per_card("auto", _cards(1, 16), 13*GIB)
    with pytest.raises(ValueError, match="4 packed members"):
        packed_members_per_card("4", _cards(1, 24), 6*GIB)
    with pytest.raises(ValueError, match="at least one physical card"):
        packed_members_per_card("auto", [], GIB)


def test_the_driver_no_longer_pins_eight_cards_or_32_64_members():
    from tools import da_cycle_prepared
    source = inspect.getsource(da_cycle_prepared)
    assert "expected_cards = 1 if args.packed_smoke else 8" not in source
    assert "not in (32, 64)" not in source
    assert "packed_members_per_card(" in source.split("def cycle(")[1]
    parser = da_cycle_prepared.build_parser()
    action = next(a for a in parser._actions
                  if a.dest == "forecast_members_per_card")
    assert set(action.choices) == {"serial", "auto", "1", "2", "4"}


def test_plan_wave_spreads_an_odd_roster_over_three_cards():
    """The scheduler side was already general: 10 members, 3 cards, 2 per
    card is one wave of 6 and one of 4, every member placed once."""
    from gpuwm.da import member_wave as wave
    roster = [{"member": i, "argv": ["python", "-m", "tools.da_member_leg"],
               "request_hash": "0"*64, "t_start": 0.0, "t_end": 900.0}
              for i in range(10)]
    cards = _cards(3, 32)
    plan = wave.plan_wave(roster, cards, 2, member_peak_bytes=10*GIB,
                          host_available_bytes=120*GIB)
    assert [len(w) for w in plan["waves"]] == [6, 4]
    assert sorted(job["member"] for w in plan["waves"] for job in w) \
        == list(range(10))
    assert {job["gpu_uuid"] for w in plan["waves"] for job in w} \
        == {card["uuid"] for card in cards}


def test_auto_packing_stops_at_two_without_an_mps_controller():
    """Four members per card need an MPS controller; on a 96 GB card with
    none running, auto picked 4 and every packed run failed at launch (box
    K, 2026-10-05).  auto now takes 2 there, and 4 only when one answers."""
    from tools.da_cycle_prepared import packed_members_per_card
    gib = 1024**3
    cards = [{"uuid": "GPU-x", "free_bytes": 96 * gib, "total_bytes": 96 * gib}]
    assert packed_members_per_card("auto", cards, 17 * gib,
                                   mps_available=False) == 2
    assert packed_members_per_card("auto", cards, 17 * gib,
                                   mps_available=True) == 4
    # a stated width is the caller's (the wave refuses 4 without MPS)
    assert packed_members_per_card("4", cards, 17 * gib,
                                   mps_available=False) == 4


def test_mps_is_unavailable_without_a_pipe_directory(monkeypatch):
    from gpuwm.da import member_wave
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    assert member_wave.mps_available() is False
