"""``gpuwm research --explain SUBCOMMAND`` keeps the flag it was given.

The research subcommands take --explain themselves, and so does
``gpuwm research``.  argparse copies every value the selected child
parsed over the parent's, defaults included, so the child's False
erased a --explain typed before the subcommand: the reader asked for the
explanation and got the one-line refusal plus a pointer telling them to
add --explain, which they already had.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pytest

from gpuwm.cli import build_parser
from gpuwm.explain import explain_enabled


_CHILDREN = [
    ["catalog"],
    ["attributes"],
    ["hardware", "--vram-gib", "12"],
    ["create", "scenario-convection.gentle", "--out", "unused.toml",
     "--point", "40,-100", "--cycle", "2026-09-27T00"],
]


@pytest.mark.parametrize("child", _CHILDREN, ids=lambda words: words[0])
def test_explain_is_the_same_on_either_side_of_the_subcommand(child):
    parser = build_parser()
    before = parser.parse_args(["research", "--explain", *child])
    after = parser.parse_args(["research", *child, "--explain"])
    plain = parser.parse_args(["research", *child])
    assert explain_enabled(before) is True
    assert explain_enabled(after) is True
    assert explain_enabled(plain) is False
    assert plain.explain is False


def test_a_research_parser_built_on_its_own_still_takes_the_child_flag():
    from gpuwm import research_workspaces

    parser = argparse.ArgumentParser()
    research_workspaces.register_cli(parser.add_subparsers(required=True))
    assert explain_enabled(parser.parse_args(
        ["research", "catalog", "--explain"])) is True
    assert explain_enabled(parser.parse_args(["research", "catalog"])) is False


def _research_hardware(*words):
    # No card is measured: the suppression variable stands in for a box
    # whose card cannot be read, which is the refusal that carries the
    # explanation half this test looks for.
    environment = {**os.environ, "GPUWM_NO_LOCAL_GPU": "1",
                   "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(
        [sys.executable, "-m", "gpuwm.cli", "research", *words],
        capture_output=True, text=True, timeout=120, env=environment)


def test_explain_before_the_subcommand_prints_the_explanation():
    typed_first = _research_hardware("--explain", "hardware")
    typed_last = _research_hardware("hardware", "--explain")
    untyped = _research_hardware("hardware")
    for result in (typed_first, typed_last, untyped):
        assert result.returncode == 2, result.stderr
    reason = "The budget decides every grid dimension"
    assert reason in typed_first.stderr, typed_first.stderr
    assert reason in typed_last.stderr, typed_last.stderr
    assert reason not in untyped.stderr, untyped.stderr
