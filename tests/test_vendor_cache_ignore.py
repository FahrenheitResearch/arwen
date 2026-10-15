from pathlib import Path
import subprocess


def test_wps_vendored_bytecode_is_ignored_without_hiding_source():
    root = Path(__file__).resolve().parents[1]
    paths = ["tools/rw_wps/vendor/a/b/__pycache__/module.cpython-311.pyc",
             "tools/rw_wps/vendor/a/__pycache__/module.pyc",
             "tools/rw_wps/vendor/a/src/module.py"]
    run = subprocess.run(["git", "-C", str(root), "check-ignore", "--no-index", "--stdin"],
                         input="\n".join(paths) + "\n", text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    assert run.stdout.splitlines() == paths[:2]
