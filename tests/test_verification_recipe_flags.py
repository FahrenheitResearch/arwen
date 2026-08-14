"""The published reproduce recipe, held against the parsers that run it.

Every command in the verification document's reproduce block is extracted and
fed to the parser that would receive it.  A flag the document invents, or a
flag a command drops, fails here rather than in the hands of the first reader
who tries to follow the recipe.

Two of those flags are load-bearing rather than cosmetic, and the criterion
names why: without ``--start-time`` the comparator writes an empty
``forecast_hour`` for every row, and the acceptance band is keyed by lead;
without ``--done-file`` the poll loop has no terminating condition at all.
The third, the directory-input mode, is what certification requires of the
geography digest.
"""

from __future__ import annotations

import importlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

import certification_fixtures as fixtures

REPO_ROOT = fixtures.REPO_ROOT
VERIFICATION = REPO_ROOT / "docs" / "public" / "VERIFICATION.md"
RECEIPT = (REPO_ROOT / "gpuwm" / "data" / "certification" / "recipe-receipt"
           / "receipt.json")
COMPARATOR = REPO_ROOT / "tools" / "matched_wrfout_stream_compare.py"
SECTION_HEADING = "7. Reproduce this"


def _section_commands() -> list[list[str]]:
    """Every command in the reproduce section's fenced block."""
    lines = VERIFICATION.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines)
                 if line.startswith("#") and line.lstrip("# ").strip()
                 == SECTION_HEADING)
    block: list[str] = []
    fenced = False
    for line in lines[start + 1:]:
        if line.startswith("#") and not fenced:
            break
        if line.startswith("```"):
            if fenced:
                break
            fenced = True
            continue
        if fenced:
            block.append(line)
    assert block, "the reproduce section carries no fenced command block"
    joined = "\n".join(block).replace("\\\n", " ")
    return [shlex.split(line) for line in joined.splitlines() if line.strip()]


COMMANDS = _section_commands()


def _tool_parser(script: str):
    """The parser belonging to the tool the recipe names, found by its path.

    The block carries more than one tool, so the parser is resolved from the
    command's own script path rather than assumed.  A recipe line naming a
    tool that has no ``build_parser`` fails here.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        module = importlib.import_module(Path(script).stem)
    finally:
        sys.path.pop(0)
    return module.build_parser()


def _comparator_parser():
    return _tool_parser(COMPARATOR.name)


def test_the_block_carries_the_four_commands_the_recipe_describes():
    programs = [tokens[0] for tokens in COMMANDS]
    assert programs.count("gpuwm") == 2
    assert programs.count("python") == 2
    assert len(COMMANDS) == 4
    assert [tokens[1] for tokens in COMMANDS if tokens[0] == "python"] == [
        "tools/matched_wrfout_stream_compare.py",
        "tools/matched_wrfout_t0_state_digest.py",
    ]


@pytest.mark.parametrize("index", range(len(COMMANDS)),
                         ids=lambda i: " ".join(COMMANDS[i][:2]))
def test_every_flag_is_accepted_by_the_parser_that_receives_it(index):
    tokens = COMMANDS[index]
    if tokens[0] == "gpuwm":
        from gpuwm.cli import build_parser

        parsed = build_parser().parse_args(tokens[1:])
        assert parsed.command == tokens[1]
        return
    assert tokens[0] == "python"
    assert (REPO_ROOT / tokens[1]).is_file(), tokens[1]
    _tool_parser(tokens[1]).parse_args(tokens[2:])


def test_this_recipe_agrees_with_the_shared_door_registry():
    """This file's check and the whole-corpus one, on one registry.

    This test parses, which is stronger than the corpus-wide rule in
    ``test_docs_extras_agree_with_code.py`` -- parsing catches a missing
    required argument that membership cannot.  The two must nonetheless
    resolve the same command to the same door and read the same option
    set, or the recipe could pass here while the corpus rule called the
    same line undocumented.  Both draw from ``doc_command_parity``.
    """

    import doc_command_parity as shared

    registry = shared.doors()
    for tokens in COMMANDS:
        if tokens[0] != "gpuwm":
            continue
        line = " ".join(tokens)
        resolved = shared.resolve_door(line, registry)
        assert resolved is not None, line
        name, body = resolved
        assert name == f"gpuwm {tokens[1]}", (name, line)
        assert name in registry, name
        defined = shared.door_options(registry[name])
        for flag in shared.FLAG.findall(body):
            assert flag in defined, (
                f"the recipe passes {flag} to `{name}`, which the shared "
                f"registry says it does not define")


def _comparator_tokens() -> list[str]:
    return next(tokens for tokens in COMMANDS
                if tokens[0] == "python"
                and tokens[1].endswith("matched_wrfout_stream_compare.py"))


def _run_tokens() -> list[str]:
    return next(tokens for tokens in COMMANDS
                if tokens[0] == "gpuwm" and tokens[1] == "run")


def test_the_comparator_invocation_passes_start_time_and_done_file():
    tokens = _comparator_tokens()
    assert "--start-time" in tokens
    assert "--done-file" in tokens
    parsed = _comparator_parser().parse_args(tokens[2:])
    assert parsed.start_time, "the recipe passes --start-time with no value"
    assert parsed.done_file is not None


def test_without_start_time_the_lead_column_is_written_empty():
    """The grounding the criterion states, asserted against the source."""
    source = COMPARATOR.read_text(encoding="utf-8")
    assert 'lead = ""' in source
    assert "if start_dt is not None:" in source
    assert '"forecast_hour": lead' in source
    parsed = _comparator_parser().parse_args(
        ["--gpu-dir", "g", "--cpu-dir", "c", "--out-csv", "m.csv"])
    assert parsed.start_time is None
    assert parsed.done_file is None


def test_without_done_file_the_poll_loop_cannot_return():
    source = COMPARATOR.read_text(encoding="utf-8")
    assert ("run_done = args.done_file is not None and args.done_file.exists()"
            in source)
    # The only `return 0` in the loop sits behind that flag.
    loop = source.split("while True:", 1)[1]
    assert loop.count("return 0") == 1
    assert "if run_done and not pending:" in loop


def test_the_run_invocation_passes_the_directory_input_mode_certify_requires():
    from gpuwm.certify.capsule import GEOGRAPHY_CONTENT_ALGORITHM
    from gpuwm.cli import build_parser

    tokens = _run_tokens()
    assert "--directory-input-hash" in tokens
    parsed = build_parser().parse_args(tokens[1:])
    assert parsed.directory_input_hash == "content"
    # And 'content' is the mode whose digest certify accepts.
    assert GEOGRAPHY_CONTENT_ALGORITHM.endswith(parsed.directory_input_hash)


def test_the_doc_cites_a_receipt_that_records_a_real_execution():
    assert RECEIPT.exists(), "the recipe receipt is not committed"
    text = VERIFICATION.read_text(encoding="utf-8")
    assert str(RECEIPT.relative_to(REPO_ROOT).as_posix()) in text.replace(
        "\\", "/")
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    comparator = receipt["comparator"]
    assert comparator["executed"] is True
    assert comparator["exit_code"] == 0
    assert comparator["row_count"] >= 2
    assert comparator["forecast_hour_all_non_empty"] is True
    assert receipt["run"]["executed"] is False
    assert receipt["run"]["flags_accepted_by_the_production_parser"][
        "accepted"] is True


def test_the_receipt_argv_is_the_command_the_document_publishes():
    """The receipt records the amended line, not a convenient variant."""
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    executed = receipt["comparator"]["argv"]
    published = _comparator_tokens()
    executed_flags = {token for token in executed if token.startswith("--")}
    published_flags = {token for token in published if token.startswith("--")}
    assert executed_flags == published_flags
    assert executed[1] == published[1]
    run_flags = {token for token in
                 receipt["run"]["flags_accepted_by_the_production_parser"]
                 and receipt["run"]["argv"] if token.startswith("--")}
    assert {token for token in _run_tokens() if token.startswith("--")} == (
        run_flags)


def test_the_receipt_metrics_csv_is_the_bytes_the_receipt_hashes():
    import hashlib

    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    csv_bytes = (RECEIPT.parent / "metrics.csv").read_bytes()
    assert hashlib.sha256(csv_bytes).hexdigest() == (
        receipt["comparator"]["metrics_csv_sha256"])
    header = csv_bytes.decode("utf-8").splitlines()[0].split(",")
    from gpuwm.certify.band import ROW_KEY_COLUMNS

    assert header[:len(ROW_KEY_COLUMNS)] == list(ROW_KEY_COLUMNS)
    assert set(header[len(ROW_KEY_COLUMNS):]) == set(fixtures.metric_columns())


def test_the_receipt_is_reproducible_through_its_own_tool(tmp_path):
    """The tool that made the receipt still runs, and still terminates."""
    completed = subprocess.run(
        [sys.executable, "tools/verification_recipe_receipt.py",
         "--receipt-dir", str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        timeout=600)
    assert completed.returncode == 0, completed.stderr
    rebuilt = json.loads((tmp_path / "receipt.json").read_text(
        encoding="utf-8"))
    committed = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert rebuilt["comparator"]["exit_code"] == 0
    assert rebuilt["comparator"]["forecast_hour_values"] == (
        committed["comparator"]["forecast_hour_values"])
    assert rebuilt["run"] == committed["run"]


def test_no_reproduce_command_is_a_transcription_of_a_stale_tool():
    """A flag the tool no longer has is a recipe nobody can run."""
    parser = _comparator_parser()
    known = {string for action in parser._actions
             for string in action.option_strings}
    used = {token for token in _comparator_tokens() if token.startswith("--")}
    assert used <= known, sorted(used - known)
    assert re.search(r"--start-time\s+\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}",
                     VERIFICATION.read_text(encoding="utf-8"))
