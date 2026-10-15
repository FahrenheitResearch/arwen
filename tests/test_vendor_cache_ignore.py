from pathlib import Path
import subprocess


def test_wps_vendored_bytecode_is_ignored_without_hiding_source():
    root = Path(__file__).resolve().parents[1]
    paths = ["tools/rw_wps/vendor/a/b/__pycache__/module.cpython-311.pyc",
             "tools/rw_wps/vendor/a/__pycache__/module.pyc",
             "tools/rw_wps/vendor/a/src/module.py"]
    # Bytes, not text: a text-mode pipe on Windows writes each newline as CRLF, and git then reads every
    # path with a trailing carriage return (public ci windows-2025 at the 2.9.0 r7 snapshot).
    run = subprocess.run(["git", "-C", str(root), "check-ignore", "--no-index", "--stdin"],
                         input=("\n".join(paths) + "\n").encode("utf-8"), capture_output=True)
    assert run.returncode == 0, run.stderr.decode("utf-8", "replace")
    assert run.stdout.decode("utf-8").splitlines() == paths[:2]
