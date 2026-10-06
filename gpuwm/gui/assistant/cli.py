"""``gpuwm assistant``: the page's assistant from a terminal.

The same routes the panel calls, run in this process against the same
forecasts folder, so a plan made here is the same plan the page shows.

    gpuwm assistant status
    gpuwm assistant on [--yes]                      turn it on; offers the download (sizes and licences first)
    gpuwm assistant off [--remove]                  turn it off; --remove deletes its downloaded model
    gpuwm assistant setup [--model ID] [--yes]      download the server and model (sizes and licences first)
    gpuwm assistant say "WORDS" [--conversation ID] [--json]
    gpuwm assistant use-endpoint URL [--model NAME] [--kind ollama]
    gpuwm assistant unload

The assistant is optional and off until it is turned on; nothing is
downloaded and no model server is started before that.

``say`` prints the plan, its reasons and each typed decision with its
probability.  It never starts a run: the plan's Start line is printed
as the command to paste.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


def _api(root: Path | None):
    from ..api import Api
    from ..jobs import Runner
    from ..server import default_root

    folder = (root or default_root()).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return Api(folder, Runner(), token="cli", port=0, version="cli", bind="127.0.0.1")


def _call(api, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    reply = api.handle(method, path, {}, json.dumps(body or {}).encode("utf-8"), "application/json")
    return reply.status, reply.body


def _fail(body: Any) -> int:
    if body.get("off"):
        # the page's words name its Settings; a terminal turns it on with its own command
        print(f"{body.get('message')} Turn it on with: gpuwm assistant on", file=sys.stderr)
        return 1
    print(f"{body.get('message')} {body.get('fix') or ''}".strip(), file=sys.stderr)
    return 1


def _gib(value: float) -> str:
    return f"{value / 2**30:.1f} GB"


def _print_turn(turn: dict[str, Any]) -> None:
    print(turn.get("reply") or turn.get("error") or "")
    plan = turn.get("plan") or {}
    if plan:
        print()
        for key, value in plan["fields"].items():
            if value not in (None, ""):
                print(f"  {key:10} {value}")
        print()
        for key, why in plan["reasons"].items():
            print(f"  {key}: {why}")
    for decision in turn.get("decisions") or []:
        p = decision.get("probability")
        print(f"  [{decision['question']}] {decision['choice']}"
              f" ({'not measured' if p is None else f'{p:.2f}'}, {decision['backend']})")
    for action in turn.get("actions") or []:
        if action.get("type") == "confirm" and action.get("command"):
            print(f"\nTo start it: {action['command']}")
    print(f"\n{turn.get('seconds', 0):.1f} s, conversation {turn.get('conversation')}")


def _setup(api, model: str | None, yes: bool) -> int:
    body: dict[str, Any] = {"model": model} if model else {}
    status, plan = _call(api, "POST", "/api/assistant/install", {**body, "dry_run": True})
    if status != 200 or not plan.get("ok"):
        return _fail(plan)
    if not plan["items"]:
        print("Everything is already here.")
        return 0
    for item in plan["items"]:
        print(f"  {item['what']}: {_gib(item['bytes'])}, licence {item['licence']['id']} ({item['licence']['url']})")
    print(f"Into {plan['home']}, {_gib(plan['bytes'])} in all, each file checked against its published digest.")
    if not yes:
        if input("Download? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Nothing downloaded. To use a model server you already run instead: "
                  "gpuwm assistant use-endpoint URL --model NAME")
            return 0
    status, reply = _call(api, "POST", "/api/assistant/install", {**body, "confirm": True})
    if status != 200:
        return _fail(reply)
    while True:
        status, state = _call(api, "GET", "/api/assistant")
        progress = state["install"]
        if progress.get("state") in ("done", "failed"):
            break
        if progress.get("total"):
            print(f"\r  {progress.get('what')}: {_gib(progress['bytes'])} of {_gib(progress['total'])}   ",
                  end="", flush=True)
        time.sleep(1)
    print()
    if progress["state"] == "failed":
        print(progress.get("message"), file=sys.stderr)
        return 1
    print("Ready.")
    return 0


def main(arguments: argparse.Namespace) -> int:
    api = _api(arguments.root)
    what = arguments.what
    if what == "on":
        status, body = _call(api, "POST", "/api/assistant/enable", {"on": True})
        if status != 200:
            return _fail(body)
        model = body["model"]
        card = body.get("card_gib")
        which = f"For this card ({card:.0f} GB)" if card else "With no card found"
        print(f"The assistant is on. {which}: {model['name']}, {_gib(model['bytes'])}, "
              f"licence {model['licence']['id']}.")
        if body.get("backend") == "endpoint":
            print(f"It uses the model server at {body.get('endpoint')}.")
            return 0
        return _setup(api, None, arguments.yes)
    if what == "off":
        status, body = _call(api, "POST", "/api/assistant/enable", {"on": False})
        if status != 200:
            return _fail(body)
        print("The assistant is off." + (" Its model left the card." if body.get("stopped") else ""))
        if arguments.remove:
            status, gone = _call(api, "POST", "/api/assistant/remove")
            if status != 200:
                return _fail(gone)
            print(f"Deleted its downloaded model and server ({_gib(gone['freed_bytes'])}).")
        elif body.get("downloaded_bytes"):
            print(f"Its downloaded files ({_gib(body['downloaded_bytes'])}) are kept; "
                  "gpuwm assistant off --remove deletes them.")
        return 0
    if what == "status":
        status, body = _call(api, "GET", "/api/assistant")
        print(json.dumps(body, indent=1))
        return 0
    if what == "unload":
        status, body = _call(api, "POST", "/api/assistant/unload")
        print("The model is off the card." if body.get("stopped") else "The model was not on the card.")
        return 0
    if what == "use-endpoint":
        body = {"backend": "endpoint", "endpoint_url": arguments.url, "endpoint_model": arguments.model or "local",
                "endpoint_kind": arguments.kind}
        status, reply = _call(api, "POST", "/api/assistant/settings", body)
        if status != 200:
            return _fail(reply)
        if reply["settings"].get("enabled") is not True:
            print("Saved. The assistant is off; gpuwm assistant on turns it on.")
        return 0
    if what == "setup":
        return _setup(api, arguments.model, arguments.yes)
    if what == "say":
        body = {"text": " ".join(arguments.words)}
        if arguments.conversation:
            body["conversation"] = arguments.conversation
        status, turn = _call(api, "POST", "/api/assistant/say", body)
        if status != 200:
            return _fail(turn)
        if arguments.json:
            print(json.dumps(turn, indent=1, default=str))
        else:
            _print_turn(turn)
        return 0
    return 2


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser(
        "assistant", help="say what you want to see; get a checked plan (the same assistant as the Weather Library's panel)")
    command.add_argument("--root", type=Path, default=None,
                         help="the folder that holds your forecasts (default: $GPUWM_RUNS_ROOT, else ~/arwen-runs)")
    whats = command.add_subparsers(dest="what", required=True)
    whats.add_parser("status", help="on or off, which model, where it runs, whether it is on the card")
    turn_on = whats.add_parser("on", help="turn the assistant on (it is off until you do); offers the download first")
    turn_on.add_argument("--yes", action="store_true", help="download without asking")
    turn_off = whats.add_parser("off", help="turn the assistant off; its model leaves the card")
    turn_off.add_argument("--remove", action="store_true", help="also delete the model and server it downloaded")
    setup = whats.add_parser("setup", help="download the model server and model, sizes and licences first")
    setup.add_argument("--model", help="a model id from the table (default: picked by your card's memory)")
    setup.add_argument("--yes", action="store_true", help="download without asking")
    say = whats.add_parser("say", help="one message to the assistant")
    say.add_argument("words", nargs="+")
    say.add_argument("--conversation", help="continue this conversation")
    say.add_argument("--json", action="store_true", help="print the whole turn as JSON")
    endpoint = whats.add_parser("use-endpoint", help="use a model server you already run (LM Studio, Ollama, llama.cpp)")
    endpoint.add_argument("url", help="its chat-completions address (the /v1 API that LM Studio, Ollama and llama.cpp serve), like http://127.0.0.1:1234/v1")
    endpoint.add_argument("--model", help="the model's name on that server")
    endpoint.add_argument("--kind", choices=["ollama", "lm-studio", "llama-server", "other"], default="other")
    whats.add_parser("unload", help="take the model off the card")
    command.set_defaults(func=main)


__all__ = ["main", "register_cli"]
