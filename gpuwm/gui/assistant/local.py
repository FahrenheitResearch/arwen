"""The local model: download it, start it, stop it, and keep it off the card while a forecast runs.

The bundled way is llama.cpp's ``llama-server`` (MIT) and one GGUF model
file, both rows of ``catalog.json``, downloaded on first use into the
assistant's home (``GPUWM_ASSISTANT_HOME``, else ``~/.arwen/assistant``)
and checked against their published SHA-256 before use.  A download is
an install, so it only happens after the person's click on a confirm
that shows the sizes and the licences.

The server listens on 127.0.0.1 only.  Its process id (with the process's
creation time), port and model are written to ``server.json`` in the home,
so a restarted page finds it.  Only that very process is ever stopped: a
record whose PID now names another program (reused after a crash or a
reboot) is dropped without a signal, because starting a forecast stops
the model automatically and a stale record made that stop end an
unrelated program.

A forecast sizes itself to the card's memory, so the model must not sit
on the card while one runs: the assistant's ``make_room`` stops the bundled
server (or asks Ollama to unload) before a run starts, and the page says
so.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile
import threading
import time
from typing import Any, Callable
import urllib.request
import zipfile

from . import catalog
from .llm import LlmError, get_json

CHUNK = 4 * 1024 * 1024
START_TIMEOUT_S = 180.0
DEFAULT_PORT = 8089


def home() -> Path:
    env = os.environ.get("GPUWM_ASSISTANT_HOME")
    return Path(env).expanduser() if env else Path.home() / ".arwen" / "assistant"


def home_words(root: Path) -> str:
    """Where the model is kept, in words a page can show: never this computer's own path.

    Under the home folder it is named from there (``the .arwen/assistant folder in your home folder``);
    anywhere else it is the folder GPUWM_ASSISTANT_HOME names, which the person set themselves.
    """

    if os.environ.get("GPUWM_ASSISTANT_HOME"):
        return "the folder GPUWM_ASSISTANT_HOME names"
    try:
        inside = Path(root).resolve().relative_to(Path.home().resolve())
    except (ValueError, OSError):
        return "the folder GPUWM_ASSISTANT_HOME names"
    return f"the {inside.as_posix()} folder in your home folder" if inside.parts else "your home folder"


def _free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return int(probe.getsockname()[1])
    raise OSError("No free port for the model server.")


def _ours(record: dict[str, Any] | None) -> bool:
    """The model server this record names is running now (same PID and creation time); a bare PID is not."""

    from gpuwm import proc_identity

    if not isinstance(record, dict):
        return False
    return proc_identity.alive(record.get("process"), record.get("pid"))


def gpu_memory_of(pid: int) -> float | None:
    """GiB the process holds on the card, from nvidia-smi; None when it cannot say."""

    try:
        done = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in done.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            try:
                return round(float(parts[1]) / 1024, 2)
            except ValueError:
                return None
    return None


def library_dirs(folder: Path) -> list[str]:
    """Folders under ``folder`` holding shared libraries (.so, .dll)."""

    found: list[str] = []
    if not folder.is_dir():
        return found
    for path in folder.rglob("*"):
        if path.is_file() and (".so" in path.suffixes or path.suffix.lower() == ".dll"):
            parent = str(path.parent)
            if parent not in found:
                found.append(parent)
    return found


class Local:
    """The bundled llama-server and its model files under one home folder."""

    #: Start and stop, one at a time (each instance sets its own; this one serves an instance made without it).
    _process_lock = threading.RLock()

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or home()
        self._lock = threading.Lock()
        # Start and stop, one at a time: a load and an unload from two requests must not both decide on one
        # record.  Kept apart from the download's lock, so an unload for a forecast never waits on a download.
        self._process_lock = threading.RLock()
        self.progress: dict[str, Any] = {"state": "idle"}

    # ---------------------------------------------------------- files

    def model_path(self, row: dict[str, Any]) -> Path:
        return self.root / "models" / row["file"]

    def server_dir(self, build: dict[str, Any]) -> Path:
        return self.root / "server" / build["id"]

    def server_binary(self, build: dict[str, Any] | None = None) -> Path | None:
        """The llama-server to run: the bundled build, else one on PATH."""

        build = build or catalog.server_build()
        if build is not None:
            folder = self.server_dir(build)
            if (folder / ".ok").is_file():
                for path in sorted(folder.rglob(build["binary"])):
                    if path.is_file():
                        return path
        found = shutil.which("llama-server")
        return Path(found) if found else None

    def model_ready(self, row: dict[str, Any]) -> bool:
        path = self.model_path(row)
        marker = path.with_suffix(path.suffix + ".ok")
        return path.is_file() and marker.is_file() and path.stat().st_size == row["bytes"]

    def install_plan(self, row: dict[str, Any]) -> dict[str, Any]:
        """What a first use downloads: each file with its size, digest and licence."""

        items = []
        build = catalog.server_build()
        if self.server_binary(build) is None:
            if build is None:
                return {"ok": False, "message": "There is no llama.cpp build for this computer in the table.",
                        "fix": "Install llama-server yourself (it only has to be on PATH), or connect LM Studio "
                               "or Ollama in the assistant's settings.", "items": []}
            for archive in build["archives"]:
                items.append({"what": "llama.cpp server" if "cudart" not in archive["url"] else "CUDA runtime for it",
                              "url": archive["url"], "bytes": archive["bytes"], "sha256": archive["sha256"],
                              "licence": archive.get("licence") or build["licence"]})
        if not self.model_ready(row):
            items.append({"what": row["name"], "url": row["url"], "bytes": row["bytes"], "sha256": row["sha256"],
                          "licence": row["licence"]})
        total = sum(item["bytes"] for item in items)
        return {"ok": True, "model": row["id"], "items": items, "bytes": total,
                "gib": round(total / 2**30, 2), "home": home_words(self.root)}

    def _download(self, url: str, dest: Path, sha256: str, size: int, label: str) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(dest.suffix + ".part")
        digest = hashlib.sha256()
        got = 0
        request = urllib.request.Request(url, headers={"User-Agent": "arwen-assistant"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                stream.write(chunk)
                digest.update(chunk)
                got += len(chunk)
                self.progress.update({"state": "downloading", "what": label, "bytes": got, "total": size})
        if digest.hexdigest() != sha256 or got != size:
            partial.unlink(missing_ok=True)
            raise LlmError(f"{label} did not match its published digest, so it was deleted. Try again.")
        partial.replace(dest)

    def install(self, row: dict[str, Any], on_done: Callable[[dict[str, Any]], None] | None = None) -> None:
        """Download what :meth:`install_plan` lists; progress in :attr:`progress`."""

        with self._lock:
            try:
                build = catalog.server_build()
                if self.server_binary(build) is None and build is not None:
                    folder = self.server_dir(build)
                    folder.mkdir(parents=True, exist_ok=True)
                    for archive in build["archives"]:
                        name = archive["url"].rsplit("/", 1)[-1]
                        target = self.root / "downloads" / name
                        self._download(archive["url"], target, archive["sha256"], archive["bytes"], name)
                        self.progress.update({"state": "unpacking", "what": name})
                        if name.endswith(".zip"):
                            with zipfile.ZipFile(target) as bundle:
                                bundle.extractall(folder)
                        else:
                            with tarfile.open(target) as bundle:
                                bundle.extractall(folder, filter="data")
                        target.unlink(missing_ok=True)
                    for path in folder.rglob(build["binary"]):
                        path.chmod(0o755)
                    (folder / ".ok").write_text("ok\n", encoding="utf-8")
                if not self.model_ready(row):
                    path = self.model_path(row)
                    self._download(row["url"], path, row["sha256"], row["bytes"], row["name"])
                    path.with_suffix(path.suffix + ".ok").write_text(row["sha256"] + "\n", encoding="utf-8")
                self.progress = {"state": "done", "model": row["id"]}
            except Exception as error:  # noqa: BLE001 - reported on the page
                self.progress = {"state": "failed", "message": str(error)}
        if on_done is not None:
            on_done(self.progress)

    def downloaded_bytes(self) -> int:
        """Bytes the assistant has downloaded into its home: models, the server and any partial download."""

        total = 0
        for part in ("models", "server", "downloads"):
            folder = self.root / part
            if folder.is_dir():
                total += sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
        return total

    def remove(self) -> int:
        """Stop the server and delete everything the assistant downloaded; the bytes freed.

        Only the home's own ``models``, ``server`` and ``downloads`` folders and the server's
        record and log go; nothing outside the home is touched.
        """

        with self._lock:
            self.stop()
            freed = self.downloaded_bytes()
            for part in ("models", "server", "downloads"):
                shutil.rmtree(self.root / part, ignore_errors=True)
            (self.root / "server.log").unlink(missing_ok=True)
            self.progress = {"state": "idle"}
        return freed

    # ---------------------------------------------------------- the server

    def _record(self) -> dict[str, Any] | None:
        try:
            return json.loads((self.root / "server.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def running(self) -> dict[str, Any] | None:
        """The bundled server's record when it is alive and answering, else None."""

        record = self._record()
        if not _ours(record):
            return None
        try:
            get_json(f"http://127.0.0.1:{record['port']}/health", timeout=2.0)
        except LlmError:
            return None
        return record

    def start(self, row: dict[str, Any], *, gpu: int = 0) -> dict[str, Any]:
        with self._process_lock:
            return self._start(row, gpu=gpu)

    def _start(self, row: dict[str, Any], *, gpu: int) -> dict[str, Any]:
        live = self.running()
        if live and live.get("model") == row["id"]:
            return live
        if _ours(self._record()):
            # A server that stopped answering still holds its model on the card: it is stopped before another
            # starts, where replacing its record left it running with nothing that could stop it.
            self.stop()
        binary = self.server_binary()
        if binary is None or not self.model_ready(row):
            raise LlmError("The model is not downloaded yet.")
        port = _free_port(DEFAULT_PORT)
        argv = [str(binary), "-m", str(self.model_path(row)), "--host", "127.0.0.1", "--port", str(port),
                "-ngl", "999", "-c", str(row.get("context") or 16384), "--jinja", "-np", "1",
                *[str(arg) for arg in row.get("server_args") or []]]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # The CUDA runtime archive unpacks beside the server, not always into its folder.
        libraries = [str(binary.parent)] + library_dirs(self.root / "server")
        name = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[name] = os.pathsep.join([*libraries, env.get(name, "")])
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            with (self.root / "server.log").open("ab") as log:
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           env=env, cwd=str(binary.parent), creationflags=flags,
                                           start_new_session=os.name != "nt")
        except OSError as error:
            raise LlmError(f"The model server could not start: {error.strerror or error}. "
                           "Download the assistant again in Settings.") from error
        from gpuwm import proc_identity

        # Its identity is read before anything can reap it; then a thread reaps it when it exits, so an ended
        # server is not left as a zombie that still answers to its PID for as long as the page server runs.
        identity = proc_identity.identify(process.pid)
        threading.Thread(target=process.wait, name="assistant-server-reaper", daemon=True).start()
        record = {"pid": process.pid, "process": identity, "port": port,
                  "model": row["id"], "argv": argv, "gpu": gpu,
                  "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (self.root / "server.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        started = time.monotonic()
        while time.monotonic() - started < START_TIMEOUT_S:
            if process.poll() is not None:
                (self.root / "server.json").unlink(missing_ok=True)
                raise LlmError(f"The model server exited ({process.returncode}). See {self.root / 'server.log'}.")
            try:
                health = get_json(f"http://127.0.0.1:{port}/health", timeout=2.0)
                if str(health.get("status", "ok")) == "ok":
                    record["load_s"] = round(time.monotonic() - started, 1)
                    (self.root / "server.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
                    return record
            except LlmError:
                pass
            time.sleep(0.5)
        self.stop()
        raise LlmError(f"The model server did not answer within {int(START_TIMEOUT_S)} s.")

    def stop(self) -> bool:
        """Stop the bundled server this home started: True when it was running and is gone.

        Its record is kept until the process has gone, so a stop that did
        not take can be tried again; one dropped first left a server that
        still held the card with nothing that could find it.  Raises
        :class:`LlmError` when the server does not end.
        """

        with self._process_lock:
            return self._stop()

    def _stop(self) -> bool:
        from gpuwm import proc_identity
        import signal

        record = self._record()
        path = self.root / "server.json"
        if not _ours(record):
            path.unlink(missing_ok=True)
            return False
        identity = record.get("process")
        # SIGTERM, then SIGKILL; Windows ends the tree at once.  Each through the process's own handle, so a
        # server that ended in between is never a program given its PID since.
        for sig in (signal.SIGTERM, *(() if os.name == "nt" else (signal.SIGKILL,))):
            try:
                proc_identity.signal_process(identity, sig, tree=True)
            except OSError as error:
                raise LlmError(f"The model server could not stop: {error}. Try Unload again.") from error
            for _ in range(40):
                if not _ours(record):
                    path.unlink(missing_ok=True)
                    return True
                time.sleep(0.25)
        raise LlmError("The model server did not stop and may still hold card memory. Try Unload again.")


def detect_endpoints(timeout: float = 1.5) -> list[dict[str, Any]]:
    """The catalog's endpoints that answer on this computer, with their model names."""

    found = []
    for row in catalog.endpoints():
        try:
            document = get_json(f"{row['url']}/models", timeout=timeout)
        except LlmError:
            continue
        names = [str(item.get("id")) for item in document.get("data") or [] if item.get("id")]
        found.append({**row, "models": names})
    return found


def unload_ollama(base_url: str, model: str) -> bool:
    """Ask an Ollama endpoint to drop its model from the card now (keep_alive 0)."""

    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    body = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
    request = urllib.request.Request(f"{root}/api/generate", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=20):
            return True
    except OSError:
        return False


__all__ = ["Local", "detect_endpoints", "gpu_memory_of", "home", "home_words", "unload_ollama"]
