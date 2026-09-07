"""Drive the real terminal binary with ordinary keyboard and mouse input.

Requires requirements-tui-journeys.txt. Recipes and evidence are JSON; no app
state is injected. A passed journey proves only the actions in its recipe.
"""
from __future__ import annotations

import argparse
import codecs
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import re
import select
import signal
import site
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Terminal:
    def __init__(self, argv, cwd, env, rows, columns):
        import psutil
        if os.name == "nt":
            from winpty import PtyProcess
            # pywinpty treats integer 0 as a missing backend and consults the
            # parent's environment; the string keeps ConPTY explicit.
            self.proc = PtyProcess.spawn(argv, cwd=str(cwd), env=env,
                                         dimensions=(rows, columns), backend="0")
            self.fd = self.proc.fileobj
            self.backend = "windows-conpty"
        else:
            import fcntl
            import pty
            import struct
            import termios
            master, slave = pty.openpty()
            fcntl.ioctl(slave, termios.TIOCSWINSZ,
                        struct.pack("HHHH", rows, columns, 0, 0))
            self.proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=slave,
                                         stdout=slave, stderr=slave,
                                         start_new_session=True)
            os.close(slave)
            self.fd = master
            self.backend = "posix-pty"
        self.process = psutil.Process(self.pid)
        self.children = {}

    def track_children(self):
        import psutil
        for parent in [self.process, *list(self.children.values())]:
            try:
                for child in parent.children(recursive=True):
                    self.children[(child.pid, child.create_time())] = child
            except psutil.NoSuchProcess:
                pass

    @property
    def pid(self):
        return self.proc.pid

    def alive(self):
        return self.proc.isalive() if os.name == "nt" else self.proc.poll() is None

    def exit_code(self):
        return self.proc.exitstatus if os.name == "nt" else self.proc.poll()

    def write(self, value):
        if os.name == "nt":
            self.proc.write(value)
        else:
            os.write(self.fd, value.encode("utf-8"))

    def read(self, timeout):
        if not select.select([self.fd], [], [], timeout)[0]:
            return b""
        if os.name == "nt":
            return self.fd.recv(65536)
        try:
            return os.read(self.fd, 65536)
        except OSError as error:
            if error.errno != 5:
                raise
            return b""

    def close(self):
        import psutil
        self.track_children()
        failures = []
        try:
            if os.name == "nt":
                self.proc.close(force=True)
            elif self.alive():
                os.killpg(self.pid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.pid, signal.SIGKILL)
                    self.proc.wait(timeout=5)
        except Exception as error:
            failures.append(f"terminal close: {type(error).__name__}: {error}")
        finally:
            # Workers may own separate process groups. Only processes observed
            # as descendants of this launched TUI are eligible for cleanup;
            # psutil's Process identity also protects against PID reuse.
            owned = [*self.children.values(), self.process]
            for process in owned:
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    pass
                except psutil.Error as error:
                    failures.append(f"terminate {process.pid}: {error}")
            _, alive = psutil.wait_procs(owned, timeout=3)
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
                except psutil.Error as error:
                    failures.append(f"kill {process.pid}: {error}")
            _, alive = psutil.wait_procs(alive, timeout=3)
            if os.name == "nt":
                self.fd.close()
            else:
                os.close(self.fd)
        return {"tracked_children": sorted({p.pid for p in self.children.values()}),
                "remaining_pids": [p.pid for p in alive], "errors": failures}


KEYS = {"enter": "\r", "escape": "\x1b", "tab": "\t",
        "up": "\x1b[A", "down": "\x1b[B", "right": "\x1b[C", "left": "\x1b[D",
        "backspace": "\x7f", "f1": "\x1bOP", "f2": "\x1bOQ", "f3": "\x1bOR",
        "f4": "\x1bOS", "f5": "\x1b[15~", "f6": "\x1b[17~", "f7": "\x1b[18~"}
KEYS.update({f"ctrl+{chr(value + 96)}": chr(value) for value in range(1, 27)})


class Journey:
    def __init__(self, args, recipe):
        import pyte
        self.args, self.recipe = args, recipe
        self.root = args.out.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.steps = []
        self.raw = (self.root / "terminal.ansi").open("wb")
        self.screen = pyte.Screen(recipe.get("columns", 120), recipe.get("rows", 36))
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.stream = pyte.Stream(self.screen)
        self.terminal = None
        self.expected_exit = None
        self.checked_jobs = {}

    def expand(self, value):
        return str(value).replace("{workspace}", str(self.root)).replace("{runs}", str(self.root / "runs"))

    def owned_path(self, value):
        path = Path(self.expand(value)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Evidence path escapes the journey directory: {value}")
        return path

    def text(self):
        return "\n".join(self.screen.display)

    def pump(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.terminal.track_children()
            data = self.terminal.read(min(0.05, max(0, end - time.monotonic())))
            if data:
                self.raw.write(data)
                self.raw.flush()
                self.stream.feed(self.decoder.decode(data))
            elif not self.terminal.alive():
                break

    def wait(self, condition, description, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.pump(0.1)
            if condition():
                return
            if not self.terminal.alive():
                raise AssertionError(f"Terminal exited before {description}")
        raise AssertionError(f"Timed out after {timeout}s waiting for {description}")

    def capture(self, index, name):
        stem = f"{index:03d}-{name}"
        (self.root / (stem + ".txt")).write_text(self.text(), encoding="utf-8")
        # HTML preserves physical cells, foreground and background emitted by the
        # actual terminal. This is a screen capture, not an app-state mockup.
        rows = []
        for row in range(self.screen.lines):
            cells = []
            for column in range(self.screen.columns):
                cell = self.screen.buffer[row][column]
                def color(value, default):
                    if value == "default":
                        return default
                    return "#" + value if len(value) == 6 else value
                foreground = color(cell.fg, "#d8e4e0")
                background = color(cell.bg, "#101b18")
                if cell.reverse:
                    foreground, background = background, foreground
                cells.append(f'<span style="color:{foreground};background:{background}'
                             + (';font-weight:bold' if cell.bold else '')
                             + '">' + html.escape(cell.data) + '</span>')
            rows.append("".join(cells))
        page = '<!doctype html><meta charset="utf-8"><title>Actual ArWen terminal journey</title>'
        page += '<style>body{background:#101b18;color:#d8e4e0}pre{font:14px/1.25 Consolas,monospace}</style>'
        page += '<p>' + html.escape(self.recipe["name"] + " / " + name) + '</p><pre>' + "\n".join(rows) + '</pre>'
        (self.root / (stem + ".html")).write_text(page, encoding="utf-8")

    def perform(self, action):
        timeout = action.get("timeout_seconds", 20)
        kind = action["action"]
        if kind == "expect":
            values = action["text"]
            values = [values] if isinstance(values, str) else values
            if not values or any(not isinstance(value, str) or not value for value in values):
                raise ValueError("Expected visible text must be nonempty")
            self.wait(lambda: all(self.expand(v) in self.text() for v in values),
                      repr(values), timeout)
        elif kind == "press":
            key = action["key"]
            self.terminal.write(key if len(key) == 1 else KEYS[key.lower()])
            self.pump(0.15)
        elif kind == "paste":
            value = self.expand(action["text"])
            if any(ord(c) < 32 for c in value):
                raise ValueError("Paste must contain printable text; use press for keys")
            self.terminal.write("\x1b[200~" + value + "\x1b[201~")
            if value and action.get("await_echo", True):
                self.wait(lambda: value[-24:] in self.text(), "pasted text to appear", timeout)
            else:
                self.pump(0.15)
        elif kind == "click":
            from wcwidth import wcswidth
            label = self.expand(action["text"])
            if not label:
                raise ValueError("A click requires a nonempty visible label")
            matches = []
            def visible():
                matches.clear()
                for row, line in enumerate(self.screen.display):
                    if action.get("row_prefix") and not line.startswith(action["row_prefix"]):
                        continue
                    start = 0
                    while (column := line.find(label, start)) >= 0:
                        matches.append((row, wcswidth(line[:column])))
                        start = column + len(label)
                if "occurrence" in action:
                    matches[:] = matches[action["occurrence"]:action["occurrence"] + 1]
                return len(matches) == 1
            self.wait(visible, f"one visible clickable {label!r}", timeout)
            row, column = matches[0]
            self.terminal.write(f"\x1b[<0;{column + 2};{row + 1}M\x1b[<0;{column + 2};{row + 1}m")
            self.pump(0.15)
        elif kind == "expect_file":
            path = self.owned_path(action["path"])
            self.wait(lambda: path.is_file(), str(path), timeout)
            if "contains" in action:
                self.wait(lambda: self.expand(action["contains"]) in path.read_text(encoding="utf-8"),
                          f"file content {path}", timeout)
        elif kind == "expect_jobs":
            count = action.get("count", 1)
            expected_exit = action.get("exit_code", 0)
            def complete():
                jobs = list((self.root / "runs" / ".arwen-tui").glob("*/job.json"))
                if len(jobs) > count:
                    raise AssertionError(f"Expected exactly {count} native jobs, found {len(jobs)}")
                if len(jobs) < count:
                    return False
                results = [job.parent / "result.json" for job in jobs]
                for result in results:
                    if result.is_file():
                        self.check_job_result(result, expected_exit)
                return all(path.is_file() for path in results)
            self.wait(complete, f"{count} native jobs to complete with exit {expected_exit}", timeout)
        elif kind == "expect_exit":
            self.wait(lambda: not self.terminal.alive(), "terminal exit", timeout)
            self.expected_exit = action.get("exit_code", 0)
            if self.terminal.exit_code() != self.expected_exit:
                raise AssertionError(f"Terminal exited {self.terminal.exit_code()}, expected {self.expected_exit}")
        elif kind == "absent":
            values = action["text"]
            values = [values] if isinstance(values, str) else values
            for value in map(self.expand, values):
                if value in self.text():
                    raise AssertionError(f"Unexpected visible text {value!r}")
        elif kind == "compare_job_values":
            values = []
            for path in sorted((self.root / "runs" / ".arwen-tui").glob("*/result.json")):
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("cli_args", [None])[0] != action["command"]:
                    continue
                log = (path.parent / "job.log").read_text(encoding="utf-8")
                matches = re.findall(action["pattern"], log)
                if len(matches) != 1:
                    raise AssertionError(f"Expected one captured value in {path.parent / 'job.log'}")
                values.append(matches[0].strip())
            if len(values) != action["count"]:
                raise AssertionError(f"Expected {action['count']} values; got {values}")
            for first, second in action.get("same", []):
                if values[first] != values[second]:
                    raise AssertionError(f"Matching requests did not reuse the cache: {values}")
            for first, second in action.get("different", []):
                if values[first] == values[second]:
                    raise AssertionError(f"Different requests shared a cache: {values}")
            self.steps[-1]["observed_values"] = values
        elif kind == "capture":
            self.pump(0.1)
        else:
            raise ValueError(f"Unknown action {kind}")

    def check_job_result(self, path, expected_exit):
        payload = json.loads(path.read_text(encoding="utf-8"))
        job = json.loads((path.parent / "job.json").read_text(encoding="utf-8"))
        command = job.get("command", []) if isinstance(job, dict) else []
        if (not isinstance(payload, dict)
                or payload.get("schema") != "gpuwm-tui-result-v1"
                or type(payload.get("exit_code")) is not int
                or payload["exit_code"] != expected_exit
                or payload.get("status") != ("completed" if expected_exit == 0 else
                                             "interrupted" if expected_exit == 130 else "failed")
                or not payload.get("ended_at")
                or len(command) < 4 or command[1:3] != ["-m", "gpuwm.cli"]
                or payload.get("cli_args") != command[3:]):
            raise AssertionError(f"Native command receipt does not confirm its recorded request: {path}: {payload}")
        self.checked_jobs[path] = expected_exit

    def run(self):
        env = os.environ.copy()
        for name in list(env):
            if name.upper().startswith(("GPUWM_", "CUPY_", "PYTHONPATH", "PYTHONHOME")) or name == "NO_COLOR":
                env.pop(name)
        env["PYTHONUSERBASE"] = site.getuserbase()
        for key, sub in (("USERPROFILE", ""), ("HOME", ""), ("APPDATA", "roaming"),
                         ("LOCALAPPDATA", "local"), ("XDG_CONFIG_HOME", "config"), ("XDG_CACHE_HOME", "cache")):
            path = self.root / "profile" / sub
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
        env.update(TERM="xterm-256color", PYTHONUNBUFFERED="1")
        if self.args.distribution_manifest is not None:
            env["GPUWM_NATIVE_DISTRIBUTION_MANIFEST"] = str(self.args.distribution_manifest.resolve(strict=True))
        if not self.args.allow_gpu:
            env["GPUWM_NO_LOCAL_GPU"] = "1"
        argv = [str(self.args.tui.resolve()), "--python", str(self.args.python.resolve()),
                "--output", str(self.root / "runs")]
        record = {"schema": "arwen-tui-journey-v1", "name": self.recipe["name"],
                  "scope": self.args.scope, "status": "RUNNING", "argv": argv,
                  "platform": sys.platform,
                  "recipe_sha256": sha256(self.args.recipe), "runner_sha256": sha256(__file__),
                  "tui_sha256": sha256(self.args.tui), "python_sha256": sha256(self.args.python),
                  "gpu_allowed": self.args.allow_gpu, "steps": self.steps,
                  "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        def save():
            temporary = self.root / "receipt.tmp"
            temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
            temporary.replace(self.root / "receipt.json")
        save()
        interrupted = False
        try:
            if not isinstance(self.recipe.get("steps"), list) or not self.recipe["steps"]:
                raise ValueError("A journey needs a nonempty list of ordinary-input steps")
            probe_code = (
                "import gpuwm, importlib.metadata as m, json, sys; from pathlib import Path; "
                "from gpuwm.runtime_manifest import provenance; "
                "from gpuwm.provenance import resolve, providing_distribution; "
                "b=resolve(); d=providing_distribution(Path(gpuwm.__file__).parent); "
                "print(json.dumps({'python':sys.version,'engine_file':gpuwm.__file__,"
                "'engine_version':b.reported_version,'binding':b.as_dict(),"
                "'direct_url':json.loads(d.read_text('direct_url.json') or '{}') if d else {},"
                "'provenance':provenance(Path(gpuwm.__file__).parent.parent,gpuwm_version=b.reported_version)}))")
            probe = subprocess.run([str(self.args.python), "-c", probe_code],
                    cwd=self.root, env=env, capture_output=True, text=True, timeout=30, check=True)
            record["engine"] = json.loads(probe.stdout)
            if self.args.scope == "installed-artifact":
                engine = record["engine"]
                if (engine["binding"]["install_kind"] != "wheel"
                        or engine["binding"]["metadata_is_borrowed"]
                        or engine["binding"]["versions_agree"] is False
                        or engine["direct_url"].get("dir_info", {}).get("editable")
                        or engine["provenance"]["identity_source"] in {"git", "installed-editable-source"}):
                    raise AssertionError("installed-artifact scope requires an installed wheel, not an editable checkout")
            record["terminal_dependencies"] = {name: importlib.metadata.version(name)
                    for name in (("pyte", "wcwidth", "psutil", "pywinpty") if os.name == "nt" else ("pyte", "wcwidth", "psutil"))}
            self.terminal = Terminal(argv, self.root, env, self.screen.lines, self.screen.columns)
            self.screen.write_process_input = self.terminal.write
            record["pid"] = self.terminal.pid
            record["terminal_backend"] = self.terminal.backend
            save()
            for index, action in enumerate(self.recipe["steps"], 1):
                name = action.get("name", action["action"])
                if not name.replace("-", "").replace("_", "").isalnum():
                    raise ValueError("Step names must be letters, digits, hyphens or underscores")
                step = {"index": index, "name": name, "action": action, "status": "RUNNING"}
                self.steps.append(step)
                save()
                start = time.monotonic()
                try:
                    self.perform(action)
                    step["status"] = "PASS"
                finally:
                    step["seconds"] = round(time.monotonic() - start, 3)
                    self.capture(index, name)
                    save()
            self.pump(0.1)
            if self.expected_exit is None and not self.terminal.alive():
                raise AssertionError(f"Terminal exited unexpectedly with code {self.terminal.exit_code()}")
            # Ending a recipe cannot certify an unfinished command merely
            # because it forgot an expect_jobs action.
            for job in (self.root / "runs" / ".arwen-tui").glob("*/job.json"):
                result = job.parent / "result.json"
                if not result.is_file():
                    raise AssertionError(f"Native command is unfinished: {job}")
                self.check_job_result(result, self.checked_jobs.get(result, 0))
            record["status"] = "PASS"
        except KeyboardInterrupt:
            interrupted = True
            record["status"] = "FAIL"
            record["error"] = "Journey interrupted by keyboard input"
            if self.steps and self.steps[-1]["status"] == "RUNNING":
                self.steps[-1]["status"] = "FAIL"
        except Exception as error:
            record["status"] = "FAIL"
            record["error"] = f"{type(error).__name__}: {error}"
            if self.steps and self.steps[-1]["status"] == "RUNNING":
                self.steps[-1]["status"] = "FAIL"
        finally:
            if self.terminal is not None:
                try:
                    record["alive_before_close"] = self.terminal.alive()
                    record["exit_code_before_close"] = self.terminal.exit_code()
                    if (record["status"] == "PASS" and self.expected_exit is None
                            and not record["alive_before_close"]):
                        record["status"] = "FAIL"
                        record["error"] = "The terminal exited without an expect_exit action"
                    record["cleanup"] = self.terminal.close()
                    record["alive_after_close"] = self.terminal.alive()
                    if (record["alive_after_close"] or record["cleanup"]["remaining_pids"]
                            or record["cleanup"]["errors"]):
                        raise RuntimeError(f"Process cleanup did not finish cleanly: {record['cleanup']}")
                except Exception as error:
                    record["status"] = "FAIL"
                    record["cleanup_error"] = f"{type(error).__name__}: {error}"
                    record.setdefault("error", record["cleanup_error"])
            for name, path in (("tui", self.args.tui), ("python", self.args.python),
                               ("recipe", self.args.recipe), ("runner", __file__)):
                try:
                    if sha256(path) != record[name + "_sha256"]:
                        raise RuntimeError(f"The {name} changed while the journey was running")
                except Exception as error:
                    record["status"] = "FAIL"
                    record.setdefault("error", f"{type(error).__name__}: {error}")
            self.raw.close()
            record["terminal_sha256"] = sha256(self.root / "terminal.ansi")
            record["evidence_files"] = [{"path": str(path.relative_to(self.root)),
                                         "bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in sorted(self.root.rglob("*")) if path.is_file()
                    and path.name not in ("receipt.json", "receipt.tmp")
                    and "profile" not in path.relative_to(self.root).parts]
            record["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            save()
        print(json.dumps({key: record[key] for key in ("name", "scope", "status")}
                         | {"receipt": str(self.root / "receipt.json"), "error": record.get("error")}))
        return 130 if interrupted else 0 if record["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tui", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--scope", choices=("development", "installed-artifact"), required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--distribution-manifest", type=Path)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    return Journey(args, recipe).run()


if __name__ == "__main__":
    raise SystemExit(main())
