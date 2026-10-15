"""Native launcher behavior, including killing a blocked renderer process."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import pytest

from gpuwm import bridge_assets, bridges

NAMES = ("rw_ensbatch", "rw_compare", "rw_verify", "rw_scorecard", "rw_nowcast_frames")


@pytest.fixture(scope="module")
def binaries():
    root = Path(os.environ.get("RW_LAUNCHER_TEST_DIR", Path(__file__).resolve().parents[1] /
                               "tools/rustwx/target/release"))
    suffix = ".exe" if os.name == "nt" else ""
    for name in (*NAMES, "rw_wrfbatch"):
        assert (root / (name + suffix)).is_file(), f"build rw-wrfbatch before launcher proof: {root / (name + suffix)}"
    return root, suffix


@pytest.mark.parametrize("name", NAMES)
def test_launcher_contains_its_contract_and_stamp(binaries, name):
    root, suffix = binaries
    path = root / (name + suffix)
    bridge_assets.verify_contract_marker(name, path)
    revisions = bridge_assets.embedded_source_revisions(path.read_bytes())
    assert len(revisions) == 1
    assert revisions == bridge_assets.embedded_source_revisions((root / ("rw_wrfbatch" + suffix)).read_bytes())
    assert subprocess.run([str(path), "--abi"], capture_output=True).returncode == 0


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    root = tmp_path_factory.mktemp("launcher-probe")
    source = root / "probe.rs"
    source.write_text(r'''
use std::{io::{self, Read}, time::Duration};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("block") => {
            std::fs::write(&args[2], std::process::id().to_string()).unwrap();
            loop { std::thread::sleep(Duration::from_secs(1)); }
        }
        _ => {
            for arg in &args { println!("arg={arg}"); }
            let mut input = String::new(); io::stdin().read_to_string(&mut input).unwrap();
            print!("stdin={input}"); eprintln!("probe stderr"); std::process::exit(39);
        }
    }
}
''')
    suffix = ".exe" if os.name == "nt" else ""
    executable = root / ("rw_wrfbatch" + suffix)
    subprocess.run(["rustc", str(source), "-o", str(executable)], check=True, capture_output=True)
    return root


@pytest.mark.parametrize("name", NAMES)
def test_argument_stdin_streams_and_exit_code(binaries, probe, name):
    root, suffix = binaries
    launcher = probe / (name + suffix)
    shutil.copy2(root / launcher.name, launcher)
    args = [str(launcher), "echo", "", "a b", 'a"b', "c\\d", "snow-雪"]
    completed = subprocess.run(args, input="input text\n", capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 39
    assert completed.stdout == "".join(f"arg={arg}\n" for arg in args) + "stdin=input text\n"
    assert completed.stderr == "probe stderr\n"


def alive(pid):
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        kernel.CloseHandle(ctypes.c_void_p(handle))
        return code.value == 259
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_killed_launcher_leaves_no_renderer(binaries, probe):
    root, suffix = binaries
    launcher = probe / ("rw_compare" + suffix)
    shutil.copy2(root / launcher.name, launcher)
    pid_file = probe / "child.pid"
    child = subprocess.Popen([str(launcher), "block", str(pid_file)])
    pid = None
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists(), "renderer did not start"
        pid = int(pid_file.read_text())
        assert alive(pid)
        if os.name != "nt":
            assert pid == child.pid, "Linux launcher must replace itself"
        child.kill()
        child.wait(timeout=10)
        deadline = time.monotonic() + 10
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not alive(pid), f"renderer {pid} survived its launcher"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        if pid is not None and alive(pid):
            os.kill(pid, signal.SIGTERM)
