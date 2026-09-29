"""``gpuwm gui``: the local web page's HTTP server.

A standard-library ``ThreadingHTTPServer`` (one thread per request,
HTTP/1.0).  Every request passes :class:`.auth.Guard` first (Host, Origin,
loopback peer, token).  ``/`` and ``/static/...`` serve the page;
``/api/...`` is :class:`.api.Api`; ``/api/runs/RUN/events`` is a
Server-Sent Events stream of the run's ``events.jsonl``, resumed from the
browser's ``Last-Event-ID`` after a reconnect.

The server never imports CuPy: GPU work runs in the run's own process,
and a meta-path guard makes an accidental import here fail loudly
instead of opening a CUDA context beside a forecast.
"""

from __future__ import annotations

import argparse
import errno
import http.server
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any, Iterator
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

from . import DEFAULT_PORT, PORT_TRIES, auth, runs
from .api import Api, Reply
from .files import plain_message, scrub
from .files import PathRefused, content_type, download_name, run_file_type, static_file
from .jobs import Runner
from .wiki import ensure_seed

BODY_LIMIT = 4 * 1024 * 1024
#: At most this many live event streams at once; the next is told to retry.
SSE_LIMIT = 16
FILE_CHUNK = 1024 * 1024
SOCKET_TIMEOUT_S = 120.0
POLL_S = 0.25
STATUS_EVERY_S = 1.0
KEEPALIVE_S = 15.0
BLOCKED_MODULES = ("cupy", "cupyx", "cupy_backends")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; font-src 'self'; "
        "style-src 'self'; script-src 'self'; connect-src 'self'; "
        "object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
}


class _NoGpuImports:
    """A meta-path finder that refuses CuPy inside the GUI server."""

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name.split(".", 1)[0] in BLOCKED_MODULES:
            raise ImportError(f"the GUI server never imports {name}: GPU work runs in the run's own process")
        return None


def block_gpu_imports() -> None:
    if not any(isinstance(finder, _NoGpuImports) for finder in sys.meta_path):
        sys.meta_path.insert(0, _NoGpuImports())


def _version() -> str:
    try:
        from gpuwm import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


def default_root() -> Path:
    """Where runs live unless --root says otherwise: ``GPUWM_RUNS_ROOT``, else ~/arwen-runs."""

    env = os.environ.get("GPUWM_RUNS_ROOT")
    return Path(env).expanduser() if env else Path.home() / "arwen-runs"


class GuiServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"
    request_queue_size = 64

    def __init__(self, address: tuple[str, int], *, root: Path, runner: Runner,
                 token: str | None = None, allow_remote: bool = False,
                 extra_hosts: tuple[str, ...] = (), quiet: bool = True, machines: Any = None,
                 owner_file: str | None = None) -> None:
        host = address[0]
        self.address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        # Set before binding: a port already in use closes the server inside
        # super().__init__, and server_close reads it.
        self.stopping = threading.Event()
        super().__init__(address, Handler)
        self.port = int(self.server_address[1])
        self.bind_host = host
        self.token = token or auth.new_token()
        self.guard = auth.Guard(self.token, self.port, host, allow_remote, tuple(extra_hosts))
        self.root = root
        self.quiet = quiet
        self.api = Api(root, runner, token=self.token, port=self.port, version=_version(), bind=host,
                       machines=machines, owner_file=owner_file)
        self.streams = threading.BoundedSemaphore(SSE_LIMIT)

    @property
    def url(self) -> str:
        host = self.bind_host if self.bind_host not in auth.WILDCARD_BINDS else "127.0.0.1"
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}/?token={self.token}"

    def server_close(self) -> None:
        # A port already in use fails inside the base constructor, which
        # closes the socket before this class has set its own fields.
        stopping = getattr(self, "stopping", None)
        if stopping is not None:
            stopping.set()
        try:
            super().server_close()
        finally:
            api = getattr(self, "api", None)
            if api is not None:
                api.close()


class _Caseless:
    """A read-only header mapping looked up case-insensitively."""

    def __init__(self, lowered: dict[str, str]) -> None:
        self._data = lowered

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key.lower(), default)


class Handler(http.server.BaseHTTPRequestHandler):
    server: GuiServer
    server_version = "arwen-gui"
    sys_version = ""
    protocol_version = "HTTP/1.0"
    timeout = SOCKET_TIMEOUT_S

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        if not self.server.quiet:
            super().log_message(format, *args)

    def _headers(self, status: int, kind: str, length: int | None,
                 extra: dict[str, str] | None = None, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        for key, value in {**SECURITY_HEADERS, **(extra or {})}.items():
            self.send_header(key, value)
        self.end_headers()

    def _json(self, status: int, body: Any, extra: dict[str, str] | None = None) -> None:
        data = (json.dumps(body, sort_keys=True, allow_nan=False, default=str) + "\n").encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(data), extra)
        if self.command != "HEAD":
            self.wfile.write(data)

    def _file(self, path: Path, cache: str = "no-store", run_file: bool = False) -> None:
        extra: dict[str, str] = {}
        if run_file:
            kind, inline = run_file_type(path)
            extra["Content-Security-Policy"] = "sandbox; default-src 'none'"
        else:
            kind, inline = content_type(path), True
        size = path.stat().st_size
        if not inline:
            extra["Content-Disposition"] = f'attachment; filename="{download_name(path.name)}"'
        self._headers(200, kind, size, extra, cache)
        if self.command == "HEAD":
            return
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(FILE_CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        self._dispatch("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("HEAD")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._json(405, {"ok": False, "message": "Only GET and POST are used here.", "fix": ""})

    do_DELETE = do_PATCH = do_OPTIONS = do_PUT  # noqa: N815

    def _dispatch(self, method: str) -> None:
        try:
            self._route(method)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as error:  # noqa: BLE001 - one bad request never takes the server down
            try:
                import traceback

                traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
                self._json(500, {"ok": False, "message": "The page server hit an error. The full reason is in its log.",
                                 "fix": "Try again; if it repeats, send the page server's log."})
            except OSError:
                pass

    def _route(self, method: str) -> None:
        parts = urlsplit(self.path)
        raw_path = parts.path or "/"
        query = parse_qs(parts.query, keep_blank_values=True)
        # A second Host, Origin, token or cookie header would let one check
        # read a value another does not; such a request is refused outright.
        for name in ("Host", "Origin", auth.TOKEN_HEADER, "Cookie"):
            if len(self.headers.get_all(name) or []) > 1:
                self._drain_body()
                self._json(400, {"ok": False, "message": f"The request has more than one {name} header.", "fix": ""})
                return
        view = _Caseless({key.lower(): value for key, value in self.headers.items()})
        decision = self.server.guard.check(method, raw_path, view, query, self.client_address[0])
        if not decision.allowed:
            self._drain_body()
            self._json(decision.status, {"ok": False, "message": decision.message,
                                         "fix": "Open the link gpuwm gui printed." if decision.status == 403 else ""})
            return
        if decision.set_cookie:
            self._headers(303, "text/plain; charset=utf-8", 0,
                          {"Location": "/", "Set-Cookie": self.server.guard.cookie_header()})
            return
        if raw_path in ("/", "/index.html"):
            self._static("index.html")
            return
        if raw_path.startswith("/static/"):
            self._static(raw_path[len("/static/"):])
            return
        if raw_path.startswith("/api/"):
            self._api(method, raw_path, query, view)
            return
        if raw_path == "/favicon.ico":
            self._headers(204, "image/x-icon", 0)
            return
        self._json(404, {"ok": False, "message": "No such page.", "fix": "Go to /."})

    def _drain_body(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return
        if 0 < length <= BODY_LIMIT:
            self.rfile.read(length)

    def _static(self, relative: str) -> None:
        try:
            path = static_file(relative)
        except (PathRefused, FileNotFoundError):
            self._json(404, {"ok": False, "message": "No such file.", "fix": ""})
            return
        self._file(path, cache="no-cache")

    def _api(self, method: str, raw_path: str, query: dict[str, list[str]], headers: _Caseless) -> None:
        body = b""
        if method == "POST":
            try:
                length = int(headers.get("content-length") or 0)
            except ValueError:
                length = -1
            if length < 0:
                self._json(400, {"ok": False, "message": "Bad Content-Length.", "fix": ""})
                return
            if length > BODY_LIMIT:
                self._json(413, {"ok": False, "message": "The request is too large.", "fix": ""})
                return
            body = self.rfile.read(length) if length else b""
        segments = [part for part in raw_path.split("/") if part]
        if method in ("GET", "HEAD") and len(segments) == 4 and segments[:2] == ["api", "runs"] \
                and segments[3] == "events":
            self._events(unquote(segments[2]), headers, query)
            return
        reply: Reply = self.server.api.handle(method, raw_path, query, body, headers.get("content-type") or "")
        if reply.file is not None:
            self._file(reply.file, run_file=reply.run_file)
            return
        self._json(reply.status, reply.body, reply.headers)

    # ---- Server-Sent Events

    def _events(self, run_id: str, headers: _Caseless, query: dict[str, list[str]]) -> None:
        try:
            rundir = runs.existing_run(self.server.root, run_id)
        except (PathRefused, FileNotFoundError) as error:
            self._json(404, {"ok": False, "message": scrub(str(error)), "fix": ""})
            return
        if not self.server.streams.acquire(blocking=False):
            self._json(503, {"ok": False, "message": "Too many live views are open.",
                             "fix": "Close a few Weather Library tabs and reload."}, {"Retry-After": "5"})
            return
        try:
            since = parse_event_id(headers.get("last-event-id") or (query.get("since") or [""])[0])
            self._headers(200, "text/event-stream; charset=utf-8", None,
                          {"X-Accel-Buffering": "no", "Connection": "close"})
            if self.command == "HEAD":
                return
            self.wfile.write(b"retry: 2000\n\n")
            self.wfile.flush()
            gone = threading.Event()
            for event, event_id, data in follow(rundir, since, gone, self.server.stopping):
                if event == "keepalive":
                    chunk = b": keepalive\n\n"
                else:
                    lines = [f"id: {event_id}"] if event_id is not None else []
                    lines.append(f"event: {event}")
                    if isinstance(data, dict):
                        # The page's event lines show words; paths and paste-able commands stay in events.jsonl.
                        # The folders a refusal names as the place to act stay in its words, as on the notice.
                        folders = data.get("folders")
                        folders = [item for item in folders if isinstance(item, str)] \
                            if isinstance(folders, list) else []
                        data = {**data, **{key: plain_message(data[key], folders, paragraph=key == "remedy")
                                           for key in ("message", "remedy") if isinstance(data.get(key), str)}}
                    lines.append("data: " + json.dumps(data, sort_keys=True, allow_nan=False, default=str))
                    chunk = ("\n".join(lines) + "\n\n").encode("utf-8")
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except OSError:
                    gone.set()
                    break
        finally:
            self.server.streams.release()


def parse_event_id(value: str | None) -> int | None:
    try:
        return int(str(value).strip()) if value not in (None, "") else None
    except ValueError:
        return None


def follow(rundir: Path, since: int | None, gone: threading.Event,
           stopping: threading.Event) -> Iterator[tuple[str, int | None, Any]]:
    """(event, id, data) for one run: every record of events.jsonl after
    ``since`` (its ``sequence``), then new ones as they are written, a
    ``status`` event whenever the derived state changes, and a keepalive
    comment when nothing else was sent for a while."""

    tail = runs.EventTail(rundir / runs.EVENTS)
    last_status: tuple[Any, ...] | None = None
    next_status = 0.0
    last_sent = time.monotonic()
    while not (gone.is_set() or stopping.is_set()):
        for record in tail.read_new():
            sequence = record.get("sequence")
            if not isinstance(sequence, int):
                continue
            if since is not None and sequence <= since:
                continue
            last_sent = time.monotonic()
            yield (str(record.get("event") or "record"), sequence, record)
        now = time.monotonic()
        if now >= next_status:
            next_status = now + STATUS_EVERY_S
            status = runs.status(rundir)
            key = (status.get("state"), status.get("percent"), status.get("stage"),
                   status.get("outputs_committed"), status.get("phase"),
                   status.get("basemap_missing"), status.get("follow_lost"),
                   json.dumps(status.get("library_warnings") or [], sort_keys=True),
                   # Where a preparation is (its step, download, boundary times, time in the stage): the page's
                   # line moves with it rather than holding "preparing" until the first model step.
                   json.dumps(status.get("preparation"), sort_keys=True))
            if key != last_status:
                last_status = key
                last_sent = now
                yield ("status", None, status)
        if now - last_sent >= KEEPALIVE_S:
            last_sent = now
            yield ("keepalive", None, None)
        stopping.wait(POLL_S)


# ---------------------------------------------------------------- building

def build_server(root: Path | str, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 allow_remote: bool = False, extra_hosts: tuple[str, ...] = (),
                 runner: Runner | None = None, token: str | None = None, quiet: bool = True,
                 port_tries: int = 1, block_gpu: bool = False, machines: Any = None,
                 owner_file: str | None = None) -> GuiServer:
    """Bind a GUI server (not yet serving); tries ``port_tries`` ports from ``port``."""

    auth.require_bind(host, allow_remote)
    if block_gpu:
        block_gpu_imports()
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    ensure_seed(root_path)
    last_error: OSError | None = None
    for attempt in range(max(1, port_tries)):
        try:
            return GuiServer((host, port + attempt if port else 0), root=root_path,
                             runner=runner or Runner(), token=token, allow_remote=allow_remote,
                             extra_hosts=extra_hosts, quiet=quiet, machines=machines,
                             owner_file=owner_file)
        except OSError as error:
            if error.errno not in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", -1), 10013) or not port:
                raise
            last_error = error
    assert last_error is not None
    raise last_error


def serve_in_thread(server: GuiServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2},
                              name="arwen-gui-server", daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------- the command

def register_cli(subparsers: argparse._SubParsersAction) -> None:
    from gpuwm.cli_numbers import int_between

    command = subparsers.add_parser(
        "gui",
        help=f"open the Weather Library in your browser (a local page on port {DEFAULT_PORT}); "
             "closing it never stops a forecast")
    command.add_argument("--root", type=Path, default=None,
                         help="the folder that holds your forecasts (default: $GPUWM_RUNS_ROOT, else ~/arwen-runs)")
    # A port outside 0..65535 reached socket.bind() and left as an
    # OverflowError traceback; the parser names the option instead.
    command.add_argument("--port", type=int_between(0, 65535), default=None,
                         help=f"port to listen on (default {DEFAULT_PORT}, or the next free one of "
                              f"{PORT_TRIES}); 0 = any free port")
    command.add_argument("--bind", default="127.0.0.1",
                         help="address to listen on (default 127.0.0.1, this computer only)")
    command.add_argument("--allow-remote", action="store_true",
                         help="allow a --bind address other computers can reach, and requests from them")
    command.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                         help="with --allow-remote: another host name the browser uses to reach this server")
    command.add_argument("--owner-file", default=None, metavar="PATH",
                         help="the card-sharing OWNER file every forecast started or queued from the page "
                              "takes the card through (default: $GPUWM_GPU_OWNER_FILE, else none)")
    command.add_argument("--no-open", action="store_true", help="print the link only; do not open a browser")
    command.add_argument("--verbose", action="store_true", help="log every request on stderr")
    command.set_defaults(func=gui_main)


def gui_main(arguments: argparse.Namespace) -> int:
    root = arguments.root if arguments.root is not None else default_root()
    port = DEFAULT_PORT if arguments.port is None else int(arguments.port)
    try:
        server = build_server(root, host=arguments.bind, port=port, allow_remote=arguments.allow_remote,
                              extra_hosts=tuple(arguments.allow_host), quiet=not arguments.verbose,
                              port_tries=PORT_TRIES if arguments.port is None else 1, block_gpu=True,
                              owner_file=arguments.owner_file)
    except auth.BindRefused as error:
        print(f"gpuwm gui: {error.what}\n{error.fix}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"gpuwm gui: could not listen on port {port} ({error}).\n"
              "Pass --port with another number, or close the other Weather Library page server.", file=sys.stderr)
        return 2
    lines = [
        "The Weather Library is ready. Open this link in your browser:",
        f"  {server.url}",
        f"Forecasts folder: {server.root}",
    ]
    if server.bind_host in ("127.0.0.1", "localhost", "::1"):
        lines.append(f"From another computer: ssh -L {server.port}:localhost:{server.port} "
                     f"{socket.gethostname()}, then open the link there.")
    lines.append("Closing this stops the page, never a forecast. Press Ctrl+C to close it.")
    print("\n".join(lines), flush=True)
    server.api.warm()
    # Queued forecasts start themselves while the page server runs, open page or not.
    server.api.queue.start()
    if not arguments.no_open:
        try:
            webbrowser.open(server.url)
        except Exception:  # noqa: BLE001 - the printed link is the door either way
            pass
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nClosed the page server. Forecasts keep going; gpuwm gui opens it again.", flush=True)
    finally:
        server.server_close()
    return 0


__all__ = ["GuiServer", "Handler", "block_gpu_imports", "build_server", "default_root", "follow",
           "gui_main", "parse_event_id", "register_cli", "serve_in_thread"]
