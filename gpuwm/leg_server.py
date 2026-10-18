"""A resident WRF-input member-leg runner: one long-lived process per card slot runs many legs.

Breakage it prevents (single-core inventory round 2, 2026-10-16, 4x RTX PRO 6000 box): every DA member leg
started two fresh Python processes (the door and its worker), and each one paid interpreter start, imports,
CUDA context creation, the Thompson table load, the history writer's start and the first step's kernel-module
loads on one thread before its first step, while the card waited.

The server runs exactly the door's code: ``gpuwm.wrfinput_forecast.main(argv)`` with the leg's own argv, so the
door's plan review, output claim, card lease and admission all happen per leg as before.  The one change is
where the door's worker body runs: the door starts it as ``[sys.executable, -m, gpuwm.wrfinput_forecast, ...,
--_worker, ...]`` through ``subprocess.run(command, env=..., check=False)``; inside a leg this server answers
exactly that call by running the same ``main(argv)`` in this process, with the environment the child would have
had, when the lease grants the one card this server was started on.  Any other call, or any other grant, runs
the ordinary child process.  ``gpuwm/wrfinput_forecast.py`` is not changed, so the preparation receipt (which
binds that adapter's source bytes) and every checkpoint identity stay the same as the child path's.

The CUDA context, compiled kernels, imported modules and the history writer stay warm between legs; device
memory pools are handed back after every leg so the next leg's admission measures the card as the child path
would.  A leg that fails (any non-zero status or exception) ends the server after its reply, so the next leg
starts in a fresh process: a failed leg never leaves state behind for another member.

Protocol (one JSON object per line):
  server -> caller on start:  {"ready": true, "protocol": 1, "pid": N}
  caller -> server:           {"argv": [...door args...], "log": "/path/to/leg.log"}
  server -> caller:           {"rc": N, "wall_seconds": S, "legs": K, "in_process": bool, "exiting": bool}
EOF on stdin ends the server.  Each leg's output (the door and the worker part) is appended to its log file,
where the child processes' output went.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time


PROTOCOL = 1


class _InProcessWorker:
    """Answers the door's worker ``subprocess.run`` with an in-process ``main(argv)`` (see the module docstring)."""

    def __init__(self):
        self.pinned = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
        self.mask = None
        self.active = False
        self.used = False
        self.real_run = subprocess.run

    def _claims(self, args, kwargs):
        if not self.active or len(args) != 1 or set(kwargs) != {"env", "check"} or kwargs["check"] is not False:
            return False
        command, env = args[0], kwargs["env"]
        if (not isinstance(command, list) or len(command) < 4 or command[0] != sys.executable
                or command[1:3] != ["-m", "gpuwm.wrfinput_forecast"] or "--_worker" not in command
                or not isinstance(env, dict)):
            return False
        mask = env.get("CUDA_VISIBLE_DEVICES") or ""
        if len(self.pinned) != 1 or len([d for d in mask.split(",") if d.strip()]) != 1:
            return False
        if self.mask is not None and mask != self.mask:
            return False
        # the child's environment is this process's own with the lease's mask, and nothing else
        return {k: v for k, v in env.items() if k != "CUDA_VISIBLE_DEVICES"} == \
            {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}

    def run(self, *args, **kwargs):
        if not self._claims(args, kwargs):
            return self.real_run(*args, **kwargs)
        import traceback
        from gpuwm import wrfinput_forecast
        command = list(args[0])
        mask = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
        self.mask = mask
        self.used = True
        saved_env, saved_argv = os.environ.get("CUDA_VISIBLE_DEVICES"), sys.argv
        os.environ["CUDA_VISIBLE_DEVICES"] = mask
        sys.argv = [wrfinput_forecast.__file__, *command[3:]]
        self.active = False
        try:
            code = wrfinput_forecast.main(command[3:])
            rc = 0 if code is None else int(code)
        except SystemExit as stop:
            rc = 0 if stop.code is None else (stop.code if isinstance(stop.code, int) else 1)
            if stop.code is not None and not isinstance(stop.code, int):
                print(stop.code, file=sys.stderr)
        except BaseException:
            traceback.print_exc()        # what a child's unhandled exception printed; its status was 1
            rc = 1
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            self.active = True
            sys.argv = saved_argv
            if saved_env is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved_env
        return subprocess.CompletedProcess(command, rc)


def _release_device_memory():
    """Hand freed device and pinned blocks back, so the next leg's admission sees the card's real free memory."""
    import gc
    gc.collect()
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _run_leg(worker, argv, log_path):
    """One leg with fds 1 and 2 on ``log_path``; returns its status (the door's exit code)."""
    import traceback
    from gpuwm import wrfinput_forecast
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    saved_argv = sys.argv
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        sys.argv = [wrfinput_forecast.__file__, *argv]
        worker.used = False
        worker.active = True
        try:
            code = wrfinput_forecast.main(list(argv))
            rc = 0 if code is None else int(code)
        except SystemExit as stop:
            rc = 0 if stop.code is None else (stop.code if isinstance(stop.code, int) else 1)
            if stop.code is not None and not isinstance(stop.code, int):
                print(stop.code, file=sys.stderr)
        except BaseException:
            traceback.print_exc()
            rc = 1
        finally:
            worker.active = False
            sys.argv = saved_argv
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        os.close(log_fd)
    return rc


def serve(requests, replies):
    worker = _InProcessWorker()
    subprocess.run = worker.run
    legs = 0
    replies.write(json.dumps({"ready": True, "protocol": PROTOCOL, "pid": os.getpid()}) + "\n")
    replies.flush()
    for line in requests:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        t0 = time.perf_counter()
        rc = _run_leg(worker, request["argv"], request["log"])
        legs += 1
        exiting = rc != 0
        if not exiting:
            _release_device_memory()
        replies.write(json.dumps({"rc": rc, "wall_seconds": round(time.perf_counter() - t0, 3), "legs": legs,
                                  "in_process": worker.used, "exiting": exiting}) + "\n")
        replies.flush()
        if exiting:
            return rc
    return 0


def main():
    # Requests and replies travel on private copies of stdin and stdout: fd 1 is pointed at each leg's log and
    # fd 0 is /dev/null (a leg's helpers never read it).
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    requests = os.fdopen(os.dup(0), "r")
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(2, 1)
    return serve(requests, replies)


if __name__ == "__main__":
    raise SystemExit(main())
