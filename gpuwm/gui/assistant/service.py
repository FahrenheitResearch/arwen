"""The assistant's API routes, settings and conversations.

    GET  /api/assistant                    status: on or off, model, where it runs, loaded, card memory, licence
    GET  /api/assistant/catalog            the models, endpoints and decision services it can use
    GET  /api/assistant/conversations      the saved conversations, newest first
    GET  /api/assistant/conversations/ID   one conversation: every turn, tool call and typed decision
    POST /api/assistant/enable             turn it on or off ({"on": true}); off stops its model server
    POST /api/assistant/remove             delete the model and server it downloaded
    POST /api/assistant/settings           where the model runs and who answers the typed decisions
    POST /api/assistant/install            download the server and model ({"confirm": true} is the click)
    POST /api/assistant/load               put the model on the card
    POST /api/assistant/unload             take it off the card
    POST /api/assistant/say                one turn: {"text", "conversation"?, "form"?}
    POST /api/assistant/decide             one typed decision: {"question": {id, instructions, options}, "state"}
    POST /api/assistant/v1/systemone       the same, on the System One wire (Jev, Kev and jevals speak it)

Settings and conversations live under the forecasts folder's
``.arwen-gui/assistant``; keys are kept there and never sent back.

The assistant is optional and starts off.  Until the person turns it on,
nothing is downloaded, no model server is started, no card memory is
used and no model server is asked anything: install, load, say and the
decision routes answer 409 with the way to turn it on, and a forecast's
fit and start never mention it.  Turning it off stops the bundled server;
``remove`` deletes what it downloaded.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlparse
import uuid

from . import catalog
from .agent import Agent
from .decide import DecisionError, LocalDecider, Question, SystemOneDecider
from .llm import Chat, LlmError
from .local import Local, detect_endpoints, gpu_memory_of, home_words, unload_ollama

BACKENDS = ("bundled", "endpoint")
DECIDERS = ("local", "jev", "kev")
OFF_MESSAGE = "The assistant is off."
OFF_FIX = "Turn it on in Settings, or press Turn on in the assistant panel. Nothing is downloaded until you say so."
FORECAST_NOTE = ("A forecast is running on this card, so the assistant's model stays off it until the "
                 "forecast ends. You can still use every control on the page.")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Assistant:
    def __init__(self, api: Any, local: Local | None = None) -> None:
        self.api = api
        self.local = local or Local()
        self.folder = Path(api.root) / ".arwen-gui" / "assistant"
        self._lock = threading.Lock()
        self._conversation_locks: dict[str, threading.Lock] = {}
        self._conversation_locks_guard = threading.Lock()
        self._install: threading.Thread | None = None
        self.note = ""

    # ---------------------------------------------------------- settings

    def settings(self) -> dict[str, Any]:
        try:
            value = json.loads((self.folder / "settings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            value = {}
        value.setdefault("enabled", False)
        value.setdefault("backend", "bundled")
        value.setdefault("decisions", "local")
        return value

    def enabled(self) -> bool:
        return self.settings().get("enabled") is True

    def require_on(self) -> None:
        """The refusal every route that would download, start or ask a model gives while the assistant is off."""

        from ..api import ApiError

        if not self.enabled():
            raise ApiError(409, OFF_MESSAGE, OFF_FIX, off=True)

    def _save(self, value: dict[str, Any]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / "settings.json"
        path.write_text(json.dumps(value, indent=1, sort_keys=True), encoding="utf-8")
        if os.name != "nt":
            path.chmod(0o600)

    def _card_gib(self) -> float | None:
        try:
            devices = self.api.system().get("devices") or []
        except Exception:  # noqa: BLE001 - no card is a normal answer
            return None
        return devices[0].get("memory_total_gib") if devices else None

    def model_row(self) -> dict[str, Any]:
        chosen = self.settings().get("model")
        return catalog.model(chosen) if chosen and catalog.model(chosen) else catalog.pick_model(self._card_gib())

    def forecast_running(self) -> list[str]:
        from .. import runs

        found = []
        for path in runs.iter_runs(self.api.root):
            try:
                if runs.status(path).get("state") == "running":
                    found.append(runs.summarize(self.api.root, path)["id"])
            except Exception:  # noqa: BLE001 - a broken folder is not a running forecast
                continue
        return found

    def status(self) -> dict[str, Any]:
        settings = self.settings()
        on = settings["enabled"] is True
        row = self.model_row()
        # Off, no model server is asked anything and no run folder is scanned: only files are looked at.
        live = self.local.running() if on and settings["backend"] == "bundled" else None
        running = self.forecast_running() if on else []
        body: dict[str, Any] = {
            "enabled": on, "card_gib": self._card_gib(), "downloaded_bytes": self.local.downloaded_bytes(),
            "backend": settings["backend"], "decisions": settings["decisions"],
            "model": {key: row.get(key) for key in ("id", "name", "bytes", "min_card_gib", "licence", "note", "quant",
                                                     "context", "why")},
            "downloaded": self.local.model_ready(row), "server_found": self.local.server_binary() is not None,
            "loaded": bool(live), "install": dict(self.local.progress), "forecasts_running": running,
            "home": home_words(self.local.root), "note": self.note,
            "endpoint": settings.get("endpoint_url"), "endpoint_model": settings.get("endpoint_model"),
            # Not a secret: the settings form shows it again, so a save that leaves it alone keeps it.
            "decision_url": settings.get("decision_url"),
            "key_set": bool(settings.get("key")), "decision_key_set": bool(settings.get("decision_key")),
        }
        if live:
            body["vram_gib"] = gpu_memory_of(int(live["pid"]))
            body["load_s"] = live.get("load_s")
            body["port"] = live.get("port")
        if running and settings["backend"] == "bundled":
            body["note"] = FORECAST_NOTE
        if not on:
            body["off"] = f"{OFF_MESSAGE} {OFF_FIX}"
        return body

    # ---------------------------------------------------------- the card

    def card_warning(self) -> str | None:
        """Words for a model server on this computer that the gui cannot take off the card, else None.

        A forecast sized to the whole card would run out of memory beside it, so the fit and the
        start say so before the person finds out from a failed forecast.
        """

        settings = self.settings()
        if settings["enabled"] is not True or settings["backend"] != "endpoint":
            return None
        url = str(settings.get("endpoint_url") or "")
        host = (urlparse(url).hostname or "").lower()
        if host not in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
            return None
        if settings.get("endpoint_kind") == "ollama" and settings.get("endpoint_model"):
            return None
        return (f"The model server at {host} is on this computer and may still hold card memory. "
                "Close its model (for example Eject in LM Studio) before a forecast that needs the whole card.")

    def make_room(self) -> dict[str, Any] | None:
        """Called once a forecast has been accepted: the model leaves the card."""

        settings = self.settings()
        if settings["enabled"] is not True:
            return None
        warning = self.card_warning()
        if warning:
            return {"unloaded": False, "warning": warning}
        if settings["backend"] == "bundled":
            try:
                stopped = self.local.stop()
            except LlmError as error:
                # The forecast has started either way; its reply says the model is still on the card.
                self.note = str(error)
                return {"unloaded": False, "warning": self.note}
            if stopped:
                self.note = ("The assistant's model left the card so the forecast has all of it. "
                             "It loads again when you next ask, once the forecast has ended.")
                return {"unloaded": True, "message": self.note}
            return None
        url = str(settings.get("endpoint_url") or "")
        if settings.get("endpoint_kind") == "ollama" and settings.get("endpoint_model"):
            if unload_ollama(url, settings["endpoint_model"]):
                self.note = "Ollama was asked to take its model off the card while the forecast runs."
                return {"unloaded": True, "message": self.note}
        return None

    def chat(self) -> Chat:
        from ..api import ApiError

        self.require_on()
        settings = self.settings()
        if settings["backend"] == "endpoint":
            url = settings.get("endpoint_url")
            if not url:
                raise ApiError(409, "No model endpoint is set.", "Add one in the assistant's settings.")
            row = catalog.model(settings.get("model") or "") or {}
            return Chat(url, settings.get("endpoint_model") or "local", key=settings.get("key"),
                        extra=row.get("request"))
        row = self.model_row()
        live = self.local.running()
        if live and live.get("model") != row["id"]:
            live = None  # another model is on the card; start() swaps it for the chosen one
        if not live:
            if self.forecast_running():
                raise ApiError(409, FORECAST_NOTE, "Wait for the forecast to finish, or connect a model on "
                               "another machine in the assistant's settings.")
            if not self.local.model_ready(row) or self.local.server_binary() is None:
                raise ApiError(409, "The assistant's model is not downloaded yet.",
                               "Press Set up in the assistant panel; it shows the size and licence first.",
                               install=self.local.install_plan(row))
            try:
                live = self.local.start(row)
            except LlmError as error:
                raise ApiError(502, str(error), "Check that the card has free memory.") from error
            self.note = ""
        return Chat(f"http://127.0.0.1:{live['port']}/v1", row["id"], extra=row.get("request"))

    def decider(self, chat: Chat | None):
        from ..api import ApiError

        settings = self.settings()
        kind = settings["decisions"]
        if kind == "local":
            if chat is None:
                chat = self.chat()
            return LocalDecider(chat, backend=f"local:{chat.model}")
        service = catalog.decision_service(kind) or {}
        url = settings.get("decision_url") or service.get("url")
        key = settings.get("decision_key")
        if service.get("needs_key") and not key:
            raise ApiError(409, f"{service.get('name', kind)} needs your API key.", "Add it in the assistant's settings.")
        return SystemOneDecider(url, key=key, backend=kind)

    # ---------------------------------------------------------- conversations

    def _conversation_path(self, conversation: str) -> Path:
        from ..files import require_name

        require_name(conversation, "conversation")
        return self.folder / "conversations" / f"{conversation}.json"

    def conversation(self, conversation: str) -> dict[str, Any]:
        path = self._conversation_path(conversation)
        if not path.is_file():
            raise FileNotFoundError(f"No conversation called {conversation}.")
        return json.loads(path.read_text(encoding="utf-8"))

    def conversations(self) -> dict[str, Any]:
        folder = self.folder / "conversations"
        rows = []
        for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            first = (document.get("turns") or [{}])[0]
            rows.append({"id": path.stem, "started_utc": document.get("started_utc"),
                         "turns": len(document.get("turns") or []), "first": str(first.get("user") or "")[:120]})
        return {"conversations": rows}

    # ---------------------------------------------------------- routes

    def get(self, parts: list[str], query: dict[str, list[str]]):
        from ..api import ApiError, Reply

        if not parts:
            return Reply(body=self.status())
        if parts == ["catalog"]:
            document = catalog.load()
            return Reply(body={"models": document["models"], "brought": catalog.brought(),
                               "endpoints": document["endpoints"],
                               "decision_services": document["decision_services"],
                               # off, no model server on this computer is asked anything, even its name
                               "detected": detect_endpoints() if self.enabled() else [],
                               "server": catalog.server_build()})
        if parts == ["conversations"]:
            return Reply(body=self.conversations())
        if len(parts) == 2 and parts[0] == "conversations":
            return Reply(body=self.conversation(parts[1]))
        raise ApiError(404, "No such endpoint.")

    def post(self, parts: list[str], payload: dict[str, Any], dry: bool):
        from ..api import ApiError, Reply

        if parts == ["settings"]:
            return Reply(body=self._settings(payload, dry))
        if parts == ["enable"]:
            return Reply(body=self._enable(payload, dry))
        if parts == ["remove"]:
            return Reply(body=self._remove(dry))
        if parts == ["unload"]:
            if dry:
                return Reply(body={"ok": True, "dry_run": True, "message": "Nothing ran. This unloads the model."})
            try:
                stopped = self.local.stop()
            except LlmError as error:
                raise ApiError(502, str(error), "Press Unload again in the assistant's panel.") from error
            self.note = "The model is off the card." if stopped else ""
            return Reply(body={"ok": True, "stopped": stopped, **self.status()})
        if parts in (["install"], ["load"], ["say"], ["decide"], ["v1", "systemone"]):
            self.require_on()
        if parts == ["install"]:
            return self._install_route(payload, dry)
        if parts == ["load"]:
            if dry:
                return Reply(body={"ok": True, "dry_run": True, "message": "Nothing ran. This loads the model."})
            self.chat()
            return Reply(body={"ok": True, **self.status()})
        if parts == ["say"]:
            return Reply(body=self._say(payload, dry))
        if parts == ["decide"]:
            question = payload.get("question") or {}
            answer = self._decide_one(question.get("id"), question.get("instructions"), question.get("options"),
                                      payload.get("state"))
            return Reply(body={"ok": True, "answer": answer})
        if parts == ["v1", "systemone"]:
            answers = {}
            for qid, item in (payload.get("questions") or {}).items():
                if (item or {}).get("type") != "choice":
                    raise ApiError(400, "Only choice questions are answered here.", "")
                answer = self._decide_one(qid, item.get("instructions"), item.get("criteria"), payload.get("state"))
                answers[qid] = {"choice": answer["choice"], "probabilities": answer["probabilities"],
                                "confidence": answer["confidence"]}
            return Reply(body={"answers": answers})
        raise ApiError(404, "No such action.")

    def _decide_one(self, qid: Any, instructions: Any, options: Any, state: Any) -> dict[str, Any]:
        from ..api import ApiError

        if not isinstance(options, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in options.items()):
            raise ApiError(400, "options must map each id to its description.", "")
        try:
            question = Question(str(qid or "question"), str(instructions or ""), dict(options))
            return self.decider(None).decide(question, state)
        except DecisionError as error:
            raise ApiError(502, str(error), "") from error

    def _enable(self, payload: dict[str, Any], dry: bool) -> dict[str, Any]:
        """Turn the assistant on or off. On downloads nothing; off stops its model server and keeps the files."""

        from ..api import ApiError

        on = payload.get("on")
        if not isinstance(on, bool):
            raise ApiError(400, "on must be true or false.", "")
        if dry:
            return {"ok": True, "dry_run": True, "enabled": self.enabled(),
                    "message": "Nothing changed. This turns the assistant " + ("on." if on else "off.")}
        value = self.settings()
        value["enabled"] = on
        self._save(value)
        stopped = False
        if not on:
            try:
                stopped = self.local.stop()
            except LlmError as error:
                # The setting is saved off; only its model is still running.
                raise ApiError(502, str(error), "Turn the assistant off again to stop its model.") from error
            self.note = ""
        body = {"ok": True, "stopped": stopped, **self.status()}
        row = self.model_row()
        if on and value["backend"] == "bundled" and not self.local.model_ready(row):
            # What Set up would download, so the page can ask before any of it happens.
            body["install"] = self.local.install_plan(row)
        return body

    def _remove(self, dry: bool) -> dict[str, Any]:
        from ..api import ApiError

        size = self.local.downloaded_bytes()
        if dry:
            return {"ok": True, "dry_run": True, "bytes": size, "home": home_words(self.local.root),
                    "message": "Nothing deleted. This deletes the assistant's downloaded model and server."}
        if self._install is not None and self._install.is_alive():
            raise ApiError(409, "A download is going.", "Wait for it to finish, then remove the model.")
        freed = self.local.remove()
        self.note = ""
        return {"ok": True, "freed_bytes": freed, **self.status()}

    def _settings(self, payload: dict[str, Any], dry: bool) -> dict[str, Any]:
        from ..api import ApiError

        value = self.settings()
        if "backend" in payload:
            if payload["backend"] not in BACKENDS:
                raise ApiError(400, f"backend must be one of {', '.join(BACKENDS)}.", "")
            value["backend"] = payload["backend"]
        if "decisions" in payload:
            if payload["decisions"] not in DECIDERS:
                raise ApiError(400, f"decisions must be one of {', '.join(DECIDERS)}.", "")
            value["decisions"] = payload["decisions"]
        if "model" in payload:
            if payload["model"] is not None and catalog.model(str(payload["model"])) is None:
                raise ApiError(400, "That model is not in the table.", "Pick one the settings list.")
            value["model"] = payload["model"]
        for key in ("endpoint_url", "endpoint_model", "endpoint_kind", "decision_url"):
            if key in payload:
                text = str(payload[key] or "").strip()
                if key.endswith("url") and text and not text.startswith(("http://", "https://")):
                    raise ApiError(400, f"{key} must start with http:// or https://.", "")
                value[key] = text or None
        for key in ("key", "decision_key"):
            if key in payload:
                value[key] = str(payload[key] or "").strip() or None
        if not dry:
            self._save(value)
        shown = {key: item for key, item in value.items() if key not in ("key", "decision_key")}
        return {"ok": True, "dry_run": dry, "settings": shown}

    def _install_route(self, payload: dict[str, Any], dry: bool):
        from ..api import ApiError, Reply

        row = catalog.model(str(payload.get("model") or "")) or self.model_row()
        plan = self.local.install_plan(row)
        if dry or not plan.get("ok") or not plan["items"]:
            return Reply(body={**plan, "dry_run": dry, "message": "Nothing downloaded." if plan["items"]
                               else "Everything is already here."})
        if payload.get("confirm") is not True:
            raise ApiError(409, "Downloading needs your click on the confirm that shows the sizes and licences.",
                           "", install=plan)
        if self._install is not None and self._install.is_alive():
            raise ApiError(409, "A download is already going.", "Wait for it; the panel shows its progress.")
        self.local.progress = {"state": "starting", "model": row["id"]}
        self._install = threading.Thread(target=self.local.install, args=(row,), daemon=True,
                                         name="arwen-assistant-install")
        self._install.start()
        return Reply(body={**plan, "ok": True, "started": True,
                           "message": "Downloading. The panel shows the progress."})

    def _conversation_lock(self, conversation: str) -> threading.Lock:
        with self._conversation_locks_guard:
            return self._conversation_locks.setdefault(conversation,
                                                       threading.Lock())

    def _say(self, payload: dict[str, Any], dry: bool) -> dict[str, Any]:
        from ..api import ApiError

        text = str(payload.get("text") or "").strip()
        if not text:
            raise ApiError(400, "Say what you want to see.", "")
        if len(text) > 2000:
            raise ApiError(400, "That is too long for one message.", "Keep it under 2000 characters.")
        conversation = str(payload.get("conversation") or "") or (
            datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
        path = self._conversation_path(conversation)
        if dry:
            return {"ok": True, "dry_run": True, "conversation": conversation,
                    "message": "Nothing ran. This sends your words to the assistant."}
        form = payload.get("form") if isinstance(payload.get("form"), dict) else None
        # One message at a time per conversation, from reading its history
        # to writing it back.  Two overlapping messages used to both read
        # the same history and each write it back with only their own
        # turn added, so the earlier turn and its pending question were
        # lost although both requests answered.
        with self._conversation_lock(conversation):
            return self._say_locked(conversation, path, text, form)

    def _say_locked(self, conversation: str, path: Path, text: str,
                    form: dict | None) -> dict[str, Any]:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            document = {"id": conversation, "started_utc": _utc(), "turns": []}
        with self._lock:
            chat = self.chat()
            agent = Agent(self.api, chat)
            agent.decide_fn = self.decider(agent.chat).decide
            turns = document["turns"]
            pending = turns[-1].get("pending") if turns else None
            turn = agent.turn(text, history=turns, form=form, pending=pending)
        record = turn.record()
        record["at_utc"] = _utc()
        record["model"] = chat.model
        if turn.question:
            request = (pending or {}).get("request") or text
            record["pending"] = {"missing": "place", "request": request}
        turns.append(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        _replace_text(path, json.dumps(document, indent=1, default=str))
        return {"ok": True, "conversation": conversation, **record, "note": self.note}


def _replace_text(path: Path, text: str) -> None:
    """Write ``path`` whole or not at all.

    A page reading the conversation while it is replaced holds the file
    open, and Windows refuses a replace over an open file for as long as
    that read lasts; a few short retries outlast it.
    """

    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(text, encoding="utf-8")
    try:
        for delay in (0.02, 0.05, 0.1, 0.2, None):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if delay is None or os.name != "nt":
                    raise
                time.sleep(delay)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


__all__ = ["Assistant", "BACKENDS", "DECIDERS", "FORECAST_NOTE", "OFF_FIX", "OFF_MESSAGE"]
