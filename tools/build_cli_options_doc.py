"""Write ``docs/public/CLI-OPTIONS.md`` from the parsers themselves.

The complete option surface of every documented door, in one page, built
by reading the argparse parsers rather than by remembering to write a
line.  ``tests/test_docs_extras_agree_with_code.py`` holds the committed
page against those same parsers in both directions, so the page cannot
silently fall behind the code and cannot name a flag the code dropped.

Run it after changing any CLI option::

    python -m tools.build_cli_options_doc

It rewrites the generated half of the page and leaves the hand-written
notes above each table alone -- those are keyed by door name in
:data:`NOTES` here, which is version-controlled prose, not scraped text.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "public" / "CLI-OPTIONS.md"

#: Hand-written orientation for the doors whose flags gate whole
#: features.  Everything else is served by its help text alone; these
#: are the ones where a reader needs to know that the flag is the ONLY
#: way in.  Keyed by door name.
NOTES: dict[str, str] = {
    "gpuwm downscale": (
        "`--parent-namelist` (with `--parent-namelist-domain`) is the "
        "entire stock-WRF-parent route this command's own summary "
        "advertises: without it, only a gpuwm parent run can be "
        "downscaled.  `--tiles {on,auto}` and `--child-size` are the "
        "only way to stream a `--point`-derived child, because this "
        "command authors the child TOML itself and a `[tiles]` table "
        "you wrote by hand would be overwritten."),
    "gpuwm go": (
        "`--no-memory-gate` is the only escape from the pre-fetch memory "
        "gate.  The gate runs before the chain downloads anything, and "
        "on a box whose card it cannot see it declines to refuse rather "
        "than blocking a run that would have worked."),
    "gpuwm multi-run": (
        "`--preflight {estimate,alloc,off}` is the only override of the "
        "plan's own preflight mode."),
    "gpuwm run": (
        "`--allow-shared-gpu`, `--gpu-uuid`, `--prep-timeout` and "
        "`--supervisor-max-restarts` are the command-line spellings of "
        "settings STREAMING.md documents only as run-plan keys."),
    "gpuwm render": (
        "`--pair-labels`, `--pair-subtitle` and `--pair-title` title and "
        "label the paired CPU-vs-GPU figure `--pair` composes."),
    "gpuwm-prepared-forecast": (
        "`--tiles JSON` is the only way to stream this route: its "
        "hash-bound experiment cannot carry a `[tiles]` table, so the "
        "table rides on the flag.  `--render-products` (with "
        "`--render-dir`) is render-on-first-committed-frame, off by "
        "absence.  `--materialize-authorities` and `--show-capabilities` "
        "each select a DIFFERENT program with its own options and must "
        "be the first argument on the line."),
    "gpuwm-prepared-tree-forecast": (
        "`--sealed-forcing-extension` selects the append-only "
        "forcing-prefix checkpoint contract."),
    "rw-wps": (
        "`--validate-physics-plan`, `--canonical-physics-plan-output`, "
        "`--extend-root-preparation`, `--sealed-prepared-cache`, "
        "`--domain-source-orography`, `--validate-hrrr-domain` and "
        "`--no-stock-wrf-export` are gates on the preprocessing route; "
        "each is off unless named."),
}

#: Doors that are the same program reached by a second name.
ALIASES = {"gpuwm-wrf-init": "rw-wps"}

#: What the page prints for an option whose parser declares no help.
#: The flag is still NAMED, which is the reachability contract; the
#: missing sentence is a separate, smaller debt, and the parity test
#: ratchets the count down rather than letting it grow.
NO_HELP = "_(the parser declares no help text for this option)_"


def portable(text: str) -> str:
    """Strip machine-specific absolute paths out of a help string.

    Some help strings interpolate a resolved path so the terminal shows
    the reader the real file on THEIR box.  Committed to a document that
    every reader shares, the same string names one developer's checkout
    -- so the page would ship a path nobody else has, and would churn in
    the diff on every machine that regenerated it.  The package-relative
    form says the same thing and belongs to no one.
    """

    import gpuwm

    for root, stand_in in (
            (str(Path(gpuwm.__file__).resolve().parent), "<gpuwm package>"),
            (str(REPO_ROOT), "<repository root>")):
        for spelling in (root, root.replace("\\", "/")):
            text = text.replace(spelling, stand_in)
    return text


def _options(parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    """``(spelling, help)`` for every option except ``--help``."""

    rows: list[tuple[str, str]] = []
    for action in parser._actions:
        names = [o for o in action.option_strings if o.startswith("--")]
        if not names or "--help" in names:
            continue
        spelling = ", ".join(sorted(names))
        if action.choices and not isinstance(action.choices, dict):
            spelling += " {" + ",".join(str(c) for c in action.choices) + "}"
        elif action.metavar:
            spelling += f" {action.metavar}"
        text = portable(" ".join((action.help or "").split()))
        rows.append((spelling, text or NO_HELP))
    return sorted(rows)


def doors() -> dict[str, argparse.ArgumentParser]:
    """Every documented command door, name -> parser."""

    from gpuwm.cli import build_parser

    out: dict[str, argparse.ArgumentParser] = {}

    def walk(parser: argparse.ArgumentParser, prefix: str) -> None:
        """Every command under ``parser``, at every depth.

        Recursive rather than one level deep.  ``gpuwm obs`` is the first
        door with a subcommand tree of its own -- `gpuwm obs radar pack
        --file ...` is three levels -- and a one-level walk resolved every
        one of those to the `gpuwm obs` parser, which owns none of their
        flags.  The docs-parity instrument then reported that the European
        radar page passed nine flags `gpuwm obs` does not define, all nine
        of which the command it actually names defines.
        """

        for action in parser._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for name, sub in sorted(action.choices.items()):
                out[f"{prefix} {name}"] = sub
                walk(sub, f"{prefix} {name}")

    walk(build_parser(), "gpuwm")

    def built(module: str, attr: str = "build_parser"):
        return getattr(importlib.import_module(module), attr)()

    out["rw-wps"] = built("gpuwm.source_cli", "_parser")
    out["gpuwm-mapped-inspect"] = built("gpuwm.mapped_source")
    out["gpuwm-wrf-runtime-check"] = built("gpuwm.native_wrf_distribution")
    out["gpuwm-prepared-forecast"] = built(
        "gpuwm.prepared_single_domain_forecast")
    out["gpuwm-prepared-forecast --materialize-authorities"] = built(
        "gpuwm.prepared_single_domain_forecast", "build_materialize_parser")
    out["gpuwm-prepared-tree-forecast"] = built(
        "gpuwm.prepared_domain_tree_forecast")
    return out


def render() -> str:
    lines = [
        "# Every option, every door",
        "",
        "The complete command-line surface, read off the parsers "
        "themselves.  It exists because a flag that appears in no "
        "document is a feature nobody can reach: `--parent-namelist` "
        "gated the whole stock-WRF-parent route, `--tiles` gated the "
        "only streamed prepared route, and neither was written down "
        "anywhere.",
        "",
        "`tests/test_docs_extras_agree_with_code.py` holds this page "
        "against the parsers in both directions, so it cannot fall "
        "behind the code and cannot name a flag that was removed.  "
        "Regenerate it with `python -m tools.build_cli_options_doc` "
        "after changing any option.",
        "",
        "Options are listed with the help text the tool itself prints.  "
        "Positional arguments and `--help` are omitted; run any door "
        "with `--help` for its usage line.",
        "",
    ]
    built = doors()
    for name in sorted(built):
        rows = _options(built[name])
        lines.append(f"## `{name}`")
        lines.append("")
        note = NOTES.get(name)
        if note:
            lines.append(note)
            lines.append("")
        if name in ALIASES:
            lines.append(f"Same program as `{ALIASES[name]}`.")
            lines.append("")
        if not rows:
            lines.append("Takes no options of its own.")
            lines.append("")
            continue
        lines.append("| option | what it does |")
        lines.append("|---|---|")
        for spelling, text in rows:
            safe = text.replace("|", "\\|")
            lines.append(f"| `{spelling}` | {safe} |")
        lines.append("")
    for alias, target in sorted(ALIASES.items()):
        lines.append(f"## `{alias}`")
        lines.append("")
        lines.append(f"The same program as `{target}`, under its other "
                     f"installed name; every option above applies.")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.build_cli_options_doc",
        description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true",
        help="exit 1 if the committed page is stale instead of writing it")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
        if current != text:
            print(f"{DOC} is stale; run python -m tools.build_cli_options_doc")
            return 1
        print(f"{DOC} is current")
        return 0
    DOC.parent.mkdir(parents=True, exist_ok=True)
    # newline="" so `\n` reaches the file as `\n`.  Python's text mode
    # translates it to `\r\n` on Windows, and the repository promises
    # `* -text` -- committed bytes are checkout bytes on every platform.
    # This tool ran on Windows once and put the only CRLF copy of this
    # page into the object database; a generator that re-creates the
    # defect every time it runs is the defect.
    DOC.write_text(text, encoding="utf-8", newline="")
    print(f"wrote {DOC} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
