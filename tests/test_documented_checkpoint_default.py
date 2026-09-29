"""The pages say what `gpuwm domain` writes for checkpoints.

The breakage these prevent: DOWNSCALE.md told readers that a single-domain
emission writes ``restart_interval_s = 0.0``, sent them to hand-edit the
file as "a workaround for an unfixed default", and listed "a single-domain
emission produces a parent that cannot be downscaled" among the known
limits; the manual's pipeline chapter repeated it.  The wizard has written
an hourly interval (or the end of a shorter run) for every emission, so a
reader was steered to an edit that changes nothing and told the one-domain
quickstart cannot feed `gpuwm downscale`, which it can.

The emitted value is measured by running each page's own documented
`gpuwm domain` command through the real wizard; no value is restated here.
"""
from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path

import pytest

from gpuwm.cli import main as cli_main


_REPO = Path(__file__).resolve().parents[1]
_DOWNSCALE = _REPO / "docs" / "public" / "DOWNSCALE.md"


def _user_pages() -> list[Path]:
    """Pages a reader follows.  The ``*-runbook.md`` files in ``docs/`` are
    records of one campaign's box and are not held to the current default."""

    pages = [_REPO / "README.md", _REPO / "CONTRIBUTING.md"]
    pages += sorted(page for page in (_REPO / "docs").glob("*.md")
                    if not page.name.endswith("-runbook.md"))
    pages += sorted((_REPO / "docs" / "public").glob("*.md"))
    pages += sorted((_REPO / "docs" / "manual").glob("*.md"))
    return [page for page in pages if page.is_file()]


def _documented_commands(text: str) -> list[str]:
    commands: list[str] = []
    for block in re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S):
        joined = block.replace("\\" + "\n", " ")
        commands += [line.strip() for line in joined.splitlines()]
    return commands


def _emitted_interval(command: str, tmp_path: Path, monkeypatch) -> float:
    """Run a documented `gpuwm domain` command and read the interval it wrote."""

    tokens = shlex.split(command.replace("\\", "/"))[2:]
    tokens = ["2026-07-29T18" if token == "latest" else token for token in tokens]
    if "--geog-root" in tokens:
        geog = tmp_path / "WPS_GEOG"
        geog.mkdir(exist_ok=True)
        tokens[tokens.index("--geog-root") + 1] = str(geog)
    monkeypatch.chdir(tmp_path)
    assert cli_main(["domain", *tokens]) == 0, command
    config = tmp_path / tokens[tokens.index("--out") + 1]
    return float(tomllib.loads(config.read_text(encoding="utf-8"))
                 ["experiment"]["restart_interval_s"])


def _domain_commands(page: Path, *, ladder: bool) -> list[str]:
    return [command for command in _documented_commands(page.read_text(encoding="utf-8"))
            if command.startswith("gpuwm domain ")
            and ("--ladder" in command) == ladder]


#: A sentence attributing a zero or absent checkpoint cadence to what the
#: wizard writes.
_ZERO_CLAIM = re.compile(
    r"restart_interval_s`?\s*=\s*0(?:\.0*)?(?![\d.])|default is 0\b"
    r"|disables? restart|cannot be\s+downscaled|no restart", re.I)
_EMISSION = re.compile(r"\b(emission|emissions|emitted|emits|emit|wizard)\b|gpuwm domain",
                       re.I)
_STATED = re.compile(r"restart_interval_s`?\s*=\s*([0-9][0-9.]*)")


def _prose_sentences(text: str) -> list[str]:
    """Headings, paragraphs and list items outside code fences, split into
    sentences, with emphasis marks removed."""

    prose = re.sub(r"```.*?```", "\n\n", text, flags=re.S)
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n|\n(?=#)|\n(?=\s*[-*] )", prose):
        joined = " ".join(line.strip() for line in paragraph.splitlines())
        joined = joined.replace("**", "")
        sentences += [part for part in re.split(r"(?<=[.;!?])\s+", joined) if part]
    return sentences


@pytest.mark.parametrize("ladder", [False, True], ids=["single-domain", "ladder"])
def test_the_downscale_walk_emits_a_parent_that_checkpoints(tmp_path, monkeypatch, ladder):
    """The page's own `gpuwm domain` commands, single-domain (Route 1) and
    nest ladder (Route 2), write a positive interval: the parent they start
    carries the checkpoint `--parent-restart` binds."""

    commands = _domain_commands(_DOWNSCALE, ladder=ladder)
    assert commands, f"DOWNSCALE.md documents no {'ladder' if ladder else 'single-domain'} domain command"
    for index, command in enumerate(commands):
        workdir = tmp_path / f"walk{index}"
        workdir.mkdir()
        assert _emitted_interval(command, workdir, monkeypatch) > 0, command


def test_the_downscale_walk_has_no_edit_step_for_the_emitted_interval():
    """Route 1 walked the reader through replacing `restart_interval_s = 0.0`
    with `3600.0` in the file the wizard had just written with 3600.0."""

    text = _DOWNSCALE.read_text(encoding="utf-8")
    route1 = text.split("# Route 1", 1)[1].split("# Route 2", 1)[0]
    blocks = re.findall(r"```toml\n(.*?)```", route1, flags=re.S)
    zero = [block for block in blocks if _ZERO_CLAIM.search(block)]
    assert not zero, zero


def test_no_page_says_the_wizard_writes_a_zero_checkpoint_interval(tmp_path, monkeypatch):
    """Every sentence about what `gpuwm domain` emits that names a checkpoint
    cadence names the one the documented command actually writes, and none
    says an emission turns checkpoints off or cannot be downscaled."""

    emitted = _emitted_interval(_domain_commands(_DOWNSCALE, ladder=False)[0],
                                tmp_path, monkeypatch)
    assert emitted > 0
    wrong = []
    scanned = 0
    for page in _user_pages():
        for sentence in _prose_sentences(page.read_text(encoding="utf-8")):
            if not _EMISSION.search(sentence) or "restart" not in sentence.lower() \
                    and "downscaled" not in sentence.lower():
                continue
            scanned += 1
            if _ZERO_CLAIM.search(sentence):
                wrong.append(f"{page.relative_to(_REPO)}: {sentence}")
                continue
            for value in _STATED.findall(sentence):
                if float(value.rstrip(".")) != emitted:
                    wrong.append(f"{page.relative_to(_REPO)}: {sentence}")
    assert scanned >= 3, scanned
    assert not wrong, "\n".join(wrong)
