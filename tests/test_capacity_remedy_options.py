"""A sizing refusal offers only the capacity options its own command accepts.

One helper sizes for `gpuwm domain` (which takes --card and --vram-gib),
`gpuwm research hardware`/`create` (--vram-gib only) and `gpuwm
domain-tiles` (neither: it plans against measured memory).  Its refusal
on a machine with no readable card told every one of them to pass
`--card 12gb`, which `research hardware` then rejected with exit 2 and
`domain-tiles` rejected along with `--vram-gib`.
"""

from __future__ import annotations

import argparse
import re

import pytest

from gpuwm import cli
from gpuwm import domain_wizard as wizard

#: A command-line option inside prose; the lookbehind keeps a dash
#: inside a word (``gpu-cu12``) from reading as one.
OPTION = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")

NO_CUPY = "the GPU runtime (CuPy) is not installed"


def _door(*names: str) -> argparse.ArgumentParser:
    parser = cli.build_parser()
    for name in names:
        action = next(item for item in parser._actions
                      if isinstance(item, argparse._SubParsersAction))
        parser = action.choices[name]
    return parser


def _accepted(parser: argparse.ArgumentParser) -> set[str]:
    return {option for action in parser._actions
            for option in action.option_strings}


@pytest.fixture
def no_card(monkeypatch):
    monkeypatch.setattr(wizard, "device_memory_probe_subprocess", lambda: None)
    monkeypatch.setattr(wizard, "device_memory_probe_reason", lambda: NO_CUPY)


def test_research_hardware_offers_a_step_it_then_accepts(no_card, monkeypatch, capsys):
    assert cli.main(["research", "hardware", "--json"]) != 0
    captured = capsys.readouterr()
    offered = set(OPTION.findall(captured.out + captured.err))
    assert "--vram-gib" in offered
    assert offered <= _accepted(_door("research", "hardware")), offered

    def probed():
        raise AssertionError("a declared capacity must not probe the local card")

    monkeypatch.setattr(wizard, "device_memory_probe_subprocess", probed)
    assert cli.main(["research", "hardware", "--json", "--vram-gib", "12"]) == 0


def test_research_create_names_only_its_own_capacity_option(no_card):
    with pytest.raises(ValueError) as refusal:
        wizard.resolve_sizing_budget(None, None, declare=("--vram-gib",))
    offered = set(OPTION.findall(str(refusal.value)))
    assert offered == {"--vram-gib"}
    assert offered <= _accepted(_door("research", "create"))


def test_the_measured_note_names_only_the_doors_options(monkeypatch):
    monkeypatch.setattr(wizard, "device_memory_probe_subprocess",
                        lambda: {"total_bytes": 12 * wizard.GIB,
                                 "free_bytes": 10 * wizard.GIB})
    note = wizard.resolve_sizing_budget(None, None, declare=("--vram-gib",)).note
    assert set(OPTION.findall(note)) == {"--vram-gib"}
    assert set(OPTION.findall(wizard.resolve_sizing_budget(None, None).note)) == {
        "--card", "--vram-gib"}


def test_domain_keeps_both_capacity_options(no_card):
    with pytest.raises(ValueError) as refusal:
        wizard.resolve_sizing_budget(None, None)
    offered = set(OPTION.findall(str(refusal.value)))
    assert offered == {"--card", "--vram-gib"}
    assert offered <= _accepted(_door("domain"))


def test_domain_tiles_offers_no_capacity_option(no_card, monkeypatch, tmp_path):
    from gpuwm import runplan, starter_template
    from gpuwm.core import preflight

    monkeypatch.setattr(preflight, "config_forcing_source", lambda *a, **k: "gfs")
    monkeypatch.setattr(runplan, "streaming_decision", lambda *a, **k: None)
    with pytest.raises(ValueError) as refusal:
        starter_template._tiles_memory_plan(tmp_path / "case.toml", None, original={})
    text = str(refusal.value)
    offered = set(OPTION.findall(text))
    assert offered <= _accepted(_door("domain-tiles")), offered
    assert "machine whose card it plans for" in text


def test_install_cupy_is_offered_only_when_cupy_is_missing(monkeypatch):
    """A probe that imported CuPy and then failed quotes its own error, and
    that error can name a cupy module; the refusal told the reader to
    install the CuPy the probe had just imported."""

    from gpuwm.core import preflight

    assert NO_CUPY == preflight.PROBE_REASON_NO_RUNTIME
    monkeypatch.setattr(wizard, "device_memory_probe_subprocess", lambda: None)
    for reason, offers_install in [
            (NO_CUPY, True),
            ("the probe could not read the card through CUDA (AttributeError: "
             "module 'cupy.cuda.runtime' has no attribute 'deviceGetLimit')",
             False)]:
        monkeypatch.setattr(wizard, "device_memory_probe_reason",
                            lambda reason=reason: reason)
        for declare in (("--card", "--vram-gib"), ()):
            with pytest.raises(ValueError) as refusal:
                wizard.resolve_sizing_budget(None, None, declare=declare)
            text = str(refusal.value)
            assert reason in text
            assert ("pip install" in text) is offers_install, text
            assert ("make the local card readable" in text) is (
                not offers_install), text


def test_an_unknown_capacity_option_is_a_programming_error():
    with pytest.raises(ValueError, match="not a capacity option"):
        wizard.resolve_sizing_budget(None, None, declare=("--gpu",))
