"""``gpuwm machines``: the Machines list from a terminal.

The same table and the same checks as the page's Machines list
(:mod:`gpuwm.gui.machines`); every subcommand prints JSON with ``--json``
and a few plain lines otherwise.  ``follow`` is what the page starts in
the background for a forecast on another machine or a render worker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def _print(document: Any, as_json: bool, lines: list[str]) -> None:
    if as_json:
        print(json.dumps(document, indent=2, sort_keys=True, default=str))
    else:
        print("\n".join(lines))


def _row_line(row: dict[str, Any]) -> str:
    card = ", ".join(f"{c.get('name')} {round((c.get('memory_total_mib') or 0) / 1024)} GB"
                     for c in row.get("cards") or [] if isinstance(c, dict) and c.get("memory_total_mib"))
    version = row.get("version_there")
    same = "" if row.get("version_matches", True) else f", gpuwm {version or 'not installed'} (not this version)"
    if not same and isinstance(row.get("not_ready"), dict):
        same = f", gpuwm {version} {row['not_ready'].get('label')}"
    return f"{row.get('name'):<16} {row.get('state', '?'):<10} {row.get('detail') or ''}" \
           + (f"  [{card}]" if card else "") + same


def main(args: argparse.Namespace) -> int:
    from .machines import MachineError, Registry, check_row, check, Machine, install

    registry = Registry()
    try:
        if args.action == "list":
            rows = registry.listing(fresh=True)
            _print({"file": str(registry.file()), "machines": rows}, args.json,
                   [f"Machines file: {registry.file()}"] + [_row_line(row) for row in rows]
                   + ([] if rows else ["No machines yet. Add one: gpuwm machines add NAME USER@HOST"]))
            return 0
        if args.action == "add":
            row: dict[str, Any] = {"name": args.name, "host": args.host, "kind": args.kind}
            for key in ("port", "identity", "workspace", "python", "owner_file", "owner_tag",
                        "geog_root", "data_dir"):
                value = getattr(args, key, None)
                if value is not None:
                    row[key] = value
            for item in args.env or []:
                key, _, value = item.partition("=")
                row.setdefault("env", {})[key] = value
            for item in args.set or []:
                key, _, value = item.partition("=")
                row[key] = value
            row = check_row({k: v for k, v in row.items() if v is not None})
            report = check(Machine(row)) if row["kind"] == "ssh" else None
            registry.put(row)
            _print({"row": row, "check": report}, args.json,
                   [f"Added {row['name']} to {registry.file()}."] + ([_row_line(report)] if report else []))
            return 0
        if args.action == "check":
            report = registry.probe(args.name, fresh=True)
            _print(report, args.json, [_row_line(report)] + ([report["offer"]["words"]] if report.get("offer") else []))
            return 0
        if args.action == "remove":
            registry.remove(args.name)
            _print({"ok": True}, args.json, [f"Removed {args.name}."])
            return 0
        if args.action == "install":
            answer = install(registry, args.name, wheelhouse=args.wheelhouse,
                             from_machine=args.from_machine, dry=args.dry_run)
            _print(answer, args.json, [answer.get("message", "")])
            return 0
        if args.action == "cloud":
            from .cloud import execute, plan

            row = next((r for r in registry.rows() if r.get("name") == args.name), None)
            if row is None:
                raise MachineError(f"There is no machine called {args.name!r}.", "", 404)
            if args.dry_run:
                document = plan(row, args.verb)
                lines = [f"Nothing runs. {args.verb} {args.name} makes these calls:"]
                lines += [f"  {i + 1}. [{step['where']}] {step['command']}" for i, step in enumerate(document["steps"])]
                if document.get("cap_minutes") is not None:
                    lines.append(f"Spend cap: {document['cap_minutes']} min of running "
                                 f"(${document['remaining_usd']} left).")
                _print(document, args.json, lines)
                return 0
            answer = execute(registry, args.name, args.verb)
            _print(answer, args.json, [f"{args.verb} {args.name}: done."])
            return 0
        if args.action == "follow":
            from .remote_runs import Follower

            return Follower(Path(args.root), args.run, registry).run()
    except MachineError as error:
        print(f"gpuwm machines: {error.message}" + (f"\n{error.fix}" if error.fix else ""), file=sys.stderr)
        return 2
    return 2


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser(
        "machines", help="the computers ArWen can run and draw forecasts on (this one, SSH hosts, cloud)")
    command.add_argument("--json", action="store_true", help="print JSON")
    sub = command.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="every machine and what it is doing now")
    add = sub.add_parser("add", help="add an SSH host (key authentication only) or a cloud machine")
    add.add_argument("name")
    add.add_argument("host", nargs="?", default=None, help="user@host or an ssh alias (SSH machines)")
    add.add_argument("--kind", default="ssh", help="ssh (default) or a cloud provider, such as aws")
    add.add_argument("--port", type=int)
    add.add_argument("--identity", help="an SSH key file on this computer")
    add.add_argument("--workspace", help="the folder on the machine ArWen may write into")
    add.add_argument("--python", help="the machine's Python that has gpuwm (default: <workspace>/venv/bin/python)")
    add.add_argument("--owner-file", dest="owner_file", help="the machine's card-sharing OWNER file, if it uses one")
    add.add_argument("--owner-tag", dest="owner_tag", help="the word this computer's runs write in that file")
    add.add_argument("--geog-root", dest="geog_root")
    add.add_argument("--data-dir", dest="data_dir")
    add.add_argument("--env", action="append", metavar="NAME=VALUE", help="environment for runs there")
    add.add_argument("--set", action="append", metavar="FIELD=VALUE", help="any other row field (cloud rows)")
    for word, text in (("check", "check one machine now"), ("remove", "remove a machine's row")):
        one = sub.add_parser(word, help=text)
        one.add_argument("name")
    inst = sub.add_parser("install", help="install this computer's gpuwm version on a machine")
    inst.add_argument("name")
    inst.add_argument("--wheelhouse", required=True, help="the folder holding the gpuwm and gpuwm_data wheels")
    inst.add_argument("--from-machine", dest="from_machine", help="the machine that folder is on (default: this one)")
    inst.add_argument("--dry-run", action="store_true")
    cloud = sub.add_parser("cloud", help="start, stop or terminate a cloud machine")
    cloud.add_argument("verb", choices=("start", "stop", "terminate"))
    cloud.add_argument("name")
    cloud.add_argument("--dry-run", action="store_true", help="print every call; run nothing, spend nothing")
    follow = sub.add_parser("follow", help="mirror a forecast on another machine and its pictures (the page starts this)")
    follow.add_argument("--root", required=True)
    follow.add_argument("run")
    command.set_defaults(func=main)


__all__ = ["main", "register_cli"]
