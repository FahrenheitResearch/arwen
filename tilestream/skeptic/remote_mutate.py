"""Break one defence at a time; run the graph gate; see whether it notices.

A negative control that cannot fail proves nothing.  The graphcap workstream
ships three gate controls (``graph_reuse='run'``, ``graph_key='none'``,
``graph_scalars=False``) and several defences with NO control at all -- the
private per-buffer memory pool, the per-buffer health ledger, the one-shot
``settle`` graph, the node-count floor.  Each break below is applied to a
pristine copy of the tree; ``test_gate --graph-only`` then either catches it
or does not, and a break the gate does not catch is a defence that is
asserted rather than proved.

Reverting is done from a pristine copy rather than from git, because the
tree this runs on is an rsync of a worktree and has no .git of its own.
"""
import os
import shutil
import subprocess
import sys
import time

TREE = os.path.dirname(os.path.abspath(__file__))
PRISTINE = os.path.join(TREE, "_pristine")
TOUCHED = ["tilestream/graphcap.py", "gpuwm/core/health_ledger.py",
           "tilestream/driver.py"]

MUTATIONS = {
    "noop_replay": (
        "tilestream/graphcap.py",
        "    def launch(self, stream) -> None:\n"
        "        if self.settle is not None:",
        "    def launch(self, stream) -> None:\n"
        "        self.replays += 1\n"
        "        return\n"
        "        if self.settle is not None:"),
    "no_settle": (
        "tilestream/graphcap.py",
        "        if captures[0][1:3] != captures[-1][1:3]:\n"
        "            settle = captures[0][0]",
        "        if False:\n"
        "            settle = captures[0][0]"),
    "shared_pool": (
        "tilestream/graphcap.py",
        "            if self._pool is None:\n"
        "                self._pool = cp.cuda.MemoryPool()",
        "            if self._pool is None:\n"
        "                global _SHARED_POOL_BREAK\n"
        "                try:\n"
        "                    _SHARED_POOL_BREAK\n"
        "                except NameError:\n"
        "                    _SHARED_POOL_BREAK = cp.cuda.MemoryPool()\n"
        "                self._pool = _SHARED_POOL_BREAK"),
    "shared_ledger": (
        "tilestream/graphcap.py",
        "        if self.ledger is None:\n"
        "            self.ledger = health_ledger.HealthLedger()",
        "        if self.ledger is None:\n"
        "            global _SHARED_LEDGER_BREAK\n"
        "            try:\n"
        "                _SHARED_LEDGER_BREAK\n"
        "            except NameError:\n"
        "                _SHARED_LEDGER_BREAK = health_ledger.HealthLedger()\n"
        "            self.ledger = _SHARED_LEDGER_BREAK"),
    "ledger_store": (
        "gpuwm/core/health_ledger.py",
        "        cp.bitwise_or(self._slots[i:i + 1], status[:1].reshape(1),\n"
        "                      out=self._slots[i:i + 1])",
        "        self._slots[i:i + 1] = status[:1].reshape(1)"),
    "no_node_floor": (
        "tilestream/graphcap.py",
        "MIN_PLAUSIBLE_NODES = 32",
        "MIN_PLAUSIBLE_NODES = 0"),
    # The scalar-delta re-application is the third gate control's subject.
    # Breaking it a DIFFERENT way -- applying the delta twice -- is the same
    # class of bug the control is meant to cover, and tests whether the
    # control covers the class or only its own switch.
    "double_scalars": (
        "tilestream/graphcap.py",
        "        if self.replay_scalars and held.scalars_delta and self.set_scalars_fn:\n"
        "            self.set_scalars_fn(state, apply_scalar_delta(\n"
        "                self.scalars_fn(state), held.scalars_delta))",
        "        if self.replay_scalars and held.scalars_delta and self.set_scalars_fn:\n"
        "            self.set_scalars_fn(state, apply_scalar_delta(\n"
        "                self.scalars_fn(state), held.scalars_delta))\n"
        "            self.set_scalars_fn(state, apply_scalar_delta(\n"
        "                self.scalars_fn(state), held.scalars_delta))"),
}


def ensure_pristine() -> None:
    if os.path.isdir(PRISTINE):
        return
    os.makedirs(PRISTINE, exist_ok=True)
    for rel in TOUCHED:
        dst = os.path.join(PRISTINE, rel.replace("/", "__"))
        shutil.copy2(os.path.join(TREE, rel), dst)


def revert() -> None:
    for rel in TOUCHED:
        shutil.copy2(os.path.join(PRISTINE, rel.replace("/", "__")),
                     os.path.join(TREE, rel))


def apply(name: str) -> None:
    rel, old, new = MUTATIONS[name]
    full = os.path.join(TREE, rel)
    with open(full) as fh:
        text = fh.read()
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{name}: anchor matched {n} times in {rel}")
    with open(full, "w") as fh:
        fh.write(text.replace(old, new))


def run_gate(tag: str, args) -> tuple[int, str]:
    log = os.path.join(TREE, f"_mut_{tag}.log")
    env = dict(os.environ, PYTHONPATH=TREE)
    with open(log, "w") as fh:
        p = subprocess.run([sys.executable, "-m", "tilestream.test_gate"]
                           + list(args), cwd=TREE, stdout=fh,
                           stderr=subprocess.STDOUT, env=env, timeout=7200)
    with open(log) as fh:
        text = fh.read()
    return p.returncode, text


def verdict(rc: int, text: str) -> str:
    if "GATE PASSED" in text:
        return "GATE PASSED  (break NOT caught)"
    if "GATE FAILED" in text:
        n = text.split("GATE FAILED -- ")[1].split(" ")[0]
        return f"GATE FAILED  (break CAUGHT, {n} problems)"
    if "out of memory" in text.lower() or "OutOfMemory" in text:
        return "INCONCLUSIVE  (out of memory -- the machine failed, not the gate)"
    return f"CRASHED rc={rc}  (break caught only as a crash)"


def main() -> int:
    ensure_pristine()
    args = ["--graph-only"]
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else \
        ["BASELINE"] + list(MUTATIONS)
    for name in names:
        revert()
        if name != "BASELINE":
            apply(name)
        t0 = time.time()
        rc, text = run_gate(name, args)
        print(f"{name:16s} {verdict(rc, text):58s} {time.time() - t0:6.0f}s",
              flush=True)
    revert()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
