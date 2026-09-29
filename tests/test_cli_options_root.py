"""The page that lists every option lists the root command's options too.

docs/public/CLI-OPTIONS.md promises the complete command-line surface,
and the generator walked from the root's children: `gpuwm --help-all`,
which the short root help advertises, was on no page.
"""

from __future__ import annotations

from tools.build_cli_options_doc import DOC, doors, render


def test_the_root_parser_is_a_documented_door():
    parser = doors()["gpuwm"]
    flags = {name for action in parser._actions for name in action.option_strings}
    assert "--help-all" in flags


def test_the_generated_and_committed_pages_carry_the_root_section():
    for page in (render(), DOC.read_text(encoding="utf-8")):
        section = page.split("## `gpuwm`\n", 1)[1].split("\n## ", 1)[0]
        assert "`--help-all`" in section
