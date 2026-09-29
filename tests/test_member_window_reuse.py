"""Reusing an ensemble member's fetch folder for another window prepares that window.

Breakage this prevents: a member fetch for leads f000/f006 was staged and
prepared; fetching the same member into the same folder for a longer or
shorter window and preparing again stopped with "The selected member
receipt has a different file count. Use a clean output directory.",
although every requested file was on disk and verified.  The staged tree
was keyed only by member and cycle, so the earlier window's receipt stood
in for the new one.  Each lead list now stages under its own folder, the
earlier tree and its input list stay exactly as they were, and a reused
tree is still byte-checked against its receipt.
"""
from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import pytest

from gpuwm import member_prep, prep_handoff
from test_audit_area1_acquisition import _member_inputs


def _write_window(spec, steps):
    """The upstream files a fetch of ``steps`` leaves beside the earlier ones."""
    grammar = prep_handoff.load_member_grammar(
        prep_handoff.packaged_member_grammar(spec["set"]))
    cycle = datetime.strptime(spec["cycle"], "%Y-%m-%dT%H")
    for product in grammar.products():
        for step in steps:
            path = Path(spec["inputs"]) / grammar.relative_path(spec["member"], product, cycle, step)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"fixture {product} {step}".encode())
    return grammar


@pytest.mark.parametrize("steps", [[0], [0, 6, 12]])
def test_a_changed_window_is_staged_beside_the_first(tmp_path, monkeypatch, steps):
    handoff, calls, _ = _member_inputs(tmp_path, monkeypatch)
    first = prep_handoff.preparation_arguments(handoff)
    first_list = Path(first[first.index("--input-list") + 1])
    first_listing = first_list.read_bytes()
    first_files = {Path(line): Path(line).read_bytes() for line in first_list.read_text().splitlines()}

    changed = copy.deepcopy(handoff)
    changed["member_prep"]["steps"] = steps
    grammar = _write_window(changed["member_prep"], steps)
    second = prep_handoff.preparation_arguments(changed)
    second_list = Path(second[second.index("--input-list") + 1])
    selected = [Path(line) for line in second_list.read_text().splitlines()]

    assert len(selected) == len(grammar.products()) * len(steps)
    assert all(path.is_file() for path in selected)
    assert second_list != first_list
    # The first window's tree and list are untouched and still reusable.
    assert first_list.read_bytes() == first_listing
    assert all(path.read_bytes() == data for path, data in first_files.items())
    assert prep_handoff.preparation_arguments(handoff) == first
    # Reusing the second window re-verifies rather than restaging.
    staged_calls = len(calls)
    assert prep_handoff.preparation_arguments(changed) == second
    assert len(calls) == staged_calls + len(selected)
    receipts = sorted((tmp_path / "members").rglob(member_prep.RECEIPT_NAME))
    assert len(receipts) == 2


def test_a_reused_window_still_refuses_a_damaged_staged_file(tmp_path, monkeypatch):
    handoff, _, _ = _member_inputs(tmp_path, monkeypatch)
    arguments = prep_handoff.preparation_arguments(handoff)
    listing = Path(arguments[arguments.index("--input-list") + 1])
    Path(listing.read_text().splitlines()[0]).write_bytes(b"damaged input")
    with pytest.raises(ValueError, match="differs from its input"):
        prep_handoff.preparation_arguments(handoff)


def test_the_generation_name_says_which_leads_it_holds():
    name = prep_handoff.lead_generation([0, 6, 12])
    assert name.startswith("f000-f012-")
    assert name != prep_handoff.lead_generation([0, 12])
    assert name == prep_handoff.lead_generation([0, 6, 12])
