"""The public CI's jobs hold on the tree the release publishes.

ci failed on the public repository for 2.7.6, 2.7.7 and 2.8.0 (run
36518598383 at 2.8.0) while publish succeeded on all three.  Each test here
names the job and the breakage it keeps out:

* syntax: five vendored scripts needed Python 3.12 and the job ran the 3.11
  floor the package declares;
* cpu: three card-test modules imported cupy through a sibling test module,
  so collection ended with three errors and nothing ran;
* oracles: on Windows every listed deck path kept a carriage return through
  bash, so pytest found no file and no oracle ran;
* publish (tests/test_publish_workflow_state_machine.py): nothing read ci.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.battery import python_floor, run_must_run_gates, run_stage1  # noqa: E402

#: The modules that reached cupy only through a sibling test module.
SIBLING_CARD_MODULES = ("tests/test_da_cycle_join_gpu.py",
                        "tests/test_noahmp_cold_start_device.py",
                        "tests/test_noahmp_device_wiring.py")


@pytest.fixture(scope="module")
def ci():
    return yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))


def _runs(job) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


# --- syntax -----------------------------------------------------------------

def test_the_syntax_job_installs_the_python_the_package_declares(ci):
    steps = ci["jobs"]["syntax"]["steps"]
    floor = [step for step in steps if step.get("id") == "floor"]
    assert len(floor) == 1 and "tools/battery/python_floor.py" in floor[0]["run"]
    setup = [step for step in steps if step.get("uses", "").startswith("actions/setup-python@")]
    assert [step["with"]["python-version"] for step in setup] == ["${{ steps.floor.outputs.version }}"]
    assert steps.index(floor[0]) < steps.index(setup[0])
    compile_line = _runs(ci["jobs"]["syntax"]).split("compileall", 1)[1].split()
    assert set(python_floor.SYNTAX_JOB_ROOTS) <= set(compile_line), compile_line
    assert python_floor.declared_floor(ROOT) == python_floor.declared_floor(ROOT / "gpuwm-data")


def test_every_file_the_syntax_job_compiles_parses_under_the_declared_floor():
    floor = python_floor.declared_floor(ROOT)
    problems = []
    for path in python_floor.syntax_job_files(ROOT):
        raw = path.read_bytes()
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError:
            source = raw.decode("latin-1")
        problems += python_floor.newer_syntax(source, floor, path.relative_to(ROOT).as_posix())
    assert not problems, (
        f"Python {floor[0]}.{floor[1]} (requires-python) cannot read these; the public CI's "
        "syntax job compiles them under it. A vendored crate's script re-vendored from "
        "upstream brings its own syntax back (libc's etc/libc-util.py did):\n" + "\n".join(problems))


@pytest.mark.parametrize("source, newer", [
    ('print(f"{E.YEL}Skipping {fulldesc} ({", ".join(t.skip)}){E.RST}")\n', True),
    ("print(f\"{E.YEL}Skipping {fulldesc} ({', '.join(t.skip)}){E.RST}\")\n", False),
    ("x = f\"{'\\n'.join(y)}\"\n", True),
    ('x = f"{y # why\n}"\n', True),
    ('x = f"{f"{1}"}"\n', True),
    ("x = f'''{'a'}'''\n", False),
    ('x = f"{y!r:>{w}}"\n', False),
    ("type Alias = int\n", True),
    ("def f[T](x: T): return x\n", True),
])
def test_the_floor_scan_reads_what_the_floor_can_and_cannot(source, newer):
    found = python_floor.newer_syntax(source, (3, 11))
    assert bool(found) is newer, found


def test_a_floor_without_a_lower_bound_is_refused(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = "<4"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="lower bound"):
        python_floor.declared_floor(tmp_path)


# --- cpu --------------------------------------------------------------------

def test_card_modules_reaching_cupy_through_a_sibling_are_dropped_whole():
    from conftest import _cupy_install_scope

    for name in SIBLING_CARD_MODULES:
        whole, _functions = _cupy_install_scope(str(ROOT / name))
        assert whole and "imports test_" in whole, (name, whole)


def test_the_sibling_closure_follows_a_rootless_directory_only(tmp_path):
    from conftest import _cupy_install_scope

    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "test_scratch_helper.py").write_text("import cupy\n", encoding="utf-8")
    (loose / "test_scratch_user.py").write_text(
        "from test_scratch_helper import x\n\n\ndef test_x():\n    pass\n", encoding="utf-8")
    _cupy_install_scope.cache_clear()
    whole, _ = _cupy_install_scope(str(loose / "test_scratch_user.py"))
    assert whole == "imports test_scratch_helper"
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "test_scratch_helper.py").write_text("import cupy\n", encoding="utf-8")
    (package / "test_scratch_user.py").write_text(
        "from test_scratch_helper import x\n\n\ndef test_x():\n    pass\n", encoding="utf-8")
    _cupy_install_scope.cache_clear()
    whole, _ = _cupy_install_scope(str(package / "test_scratch_user.py"))
    assert whole is None, "inside a package a bare name is not the sibling file"
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    (guarded / "test_scratch_helper.py").write_text("import cupy\n", encoding="utf-8")
    (guarded / "test_scratch_user.py").write_text(
        "try:\n    from test_scratch_helper import x\nexcept ImportError:\n    x = None\n",
        encoding="utf-8")
    _cupy_install_scope.cache_clear()
    whole, _ = _cupy_install_scope(str(guarded / "test_scratch_user.py"))
    assert whole is None, "a guarded sibling import is not a card dependence"


def test_the_card_modules_collect_on_an_install_without_cupy(tmp_path):
    """The public cpu job's own selection over the three modules, cupy absent."""
    hide = ("import sys; sys.modules['cupy'] = None; import pytest; "
            "raise SystemExit(pytest.main(sys.argv[1:]))")
    env = dict(os.environ, GPUWM_NO_LOCAL_GPU="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-c", hide, "-q", "-p", "no:cacheprovider", "--basetemp",
         str(tmp_path / "bt"), "-m",
         "not gpu and not slow and not network and not static_platform_qualification",
         *SIBLING_CARD_MODULES],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr
    assert result.returncode in (0, 5), output[-4000:]
    assert "ERROR" not in output and "during collection" not in output, output[-4000:]
    assert f"{len(SIBLING_CARD_MODULES)} module(s) SKIPPED whole" in output, output[-4000:]


def test_the_cpu_job_uses_the_complete_release_list_and_platform_worker_counts(ci):
    job = ci["jobs"]["cpu"]
    assert "python tools/battery/run_stage1.py --manifest tools/battery/stage1_files.txt" in _runs(job)
    assert "--native-manifest tools/battery/stage1_native_files.txt --platform ${{ matrix.platform }} --minimum 389 --" in _runs(job)
    assert '--basetemp="$RUNNER_TEMP/b"' in _runs(job)
    assert "-n ${{ matrix.workers }} --dist loadfile" in _runs(job)
    assert {item["os"]: item["workers"] for item in job["strategy"]["matrix"]["include"]} == {
        "ubuntu-24.04": 2, "windows-2025": 4}
    assert {item["os"]: item["platform"] for item in job["strategy"]["matrix"]["include"]} == {
        "ubuntu-24.04": "linux", "windows-2025": "windows"}
    files = run_stage1.selected_files(ROOT, platform="linux")
    assert set(run_stage1.listed_files(ROOT)) <= set(files)
    assert set(run_stage1.publication_files(ROOT)) <= set(files)
    assert set(run_stage1.CONTRACT_FILES) <= set(files)
    assert len(files) == len(set(files)) and all((ROOT / name).is_file() for name in files)


def _stage1_root(tmp_path, text):
    folder = tmp_path / "tools/battery"
    folder.mkdir(parents=True)
    (folder / "stage1_files.txt").write_bytes(text.replace("\n", "\r\n").encode())
    (folder / "stage1_native_files.txt").write_text("# no native-dependent fixture files\n", encoding="utf-8")
    for name in run_stage1.CONTRACT_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_control(): pass\n", encoding="utf-8")
    return tmp_path


def test_stage1_runner_names_missing_files_before_pytest(tmp_path, monkeypatch, capsys):
    root = _stage1_root(tmp_path, "tests/test_absent.py\n")
    monkeypatch.setattr(run_stage1, "publication_files", lambda root: [])
    assert run_stage1.main(["--root", str(root)]) == 2
    assert "tests/test_absent.py" in capsys.readouterr().err


def test_stage1_runner_rejects_duplicate_files_instead_of_inflating_coverage(tmp_path, capsys):
    root = _stage1_root(tmp_path, "tests/test_present.py\ntests/test_present.py\n")
    assert run_stage1.main(["--root", str(root)]) == 2
    assert "duplicates coverage" in capsys.readouterr().err


def test_stage1_runner_passes_the_full_union_without_shell_translation(tmp_path, monkeypatch):
    root = _stage1_root(tmp_path, "# release list\ntests/test_release.py\n")
    for name in ("tests/test_release.py", "tests/test_publication.py"):
        (root / name).write_text("def test_x(): pass\n", encoding="utf-8")
    monkeypatch.setattr(run_stage1, "publication_files", lambda root: ["tests/test_publication.py"])
    calls = []
    monkeypatch.setattr(run_stage1.subprocess, "call", lambda command, **kwargs: calls.append((command, kwargs)) or 0)
    arguments = ["-q", "-n", "2", "-m", "not gpu and not slow and not network"]
    assert run_stage1.main(["--root", str(root), "--minimum", "1", "--", *arguments]) == 0
    assert calls == [([sys.executable, "-m", "pytest", *arguments, "tests/test_release.py",
                      "tests/test_publication.py", *run_stage1.CONTRACT_FILES], {"cwd": root})]


def test_stage1_runner_holds_the_declared_minimum(tmp_path, monkeypatch, capsys):
    root = _stage1_root(tmp_path, "tests/test_present.py\n")
    monkeypatch.setattr(run_stage1, "publication_files", lambda root: [])
    assert run_stage1.main(["--root", str(root), "--minimum", "2"]) == 2
    assert "fewer than the 2" in capsys.readouterr().err


def test_stage1_runner_reads_the_explicit_manifest_argument(tmp_path, monkeypatch):
    root = _stage1_root(tmp_path, "tests/test_default_absent.py\n")
    selected = "tools/battery/selected-stage1.txt"
    (root / selected).write_bytes(b"tests/test_selected.py\r\n")
    (root / "tests/test_selected.py").write_text("def test_x(): pass\n", encoding="utf-8")
    monkeypatch.setattr(run_stage1, "publication_files", lambda root: [])
    calls = []
    monkeypatch.setattr(run_stage1.subprocess, "call", lambda command, **kwargs: calls.append(command) or 0)
    assert run_stage1.main(["--root", str(root), "--manifest", selected]) == 0
    assert "tests/test_selected.py" in calls[0] and "tests/test_default_absent.py" not in calls[0]


def test_windows_runner_reads_the_explicit_native_partition_and_preserves_linux_coverage(tmp_path, monkeypatch):
    root = _stage1_root(tmp_path, "tests/test_native.py\ntests/test_portable.py\n")
    for name in ("tests/test_native.py", "tests/test_portable.py", "tests/test_publication.py"):
        (root / name).write_text("def test_x(): pass\n", encoding="utf-8")
    partition = "tools/battery/selected-native.txt"
    (root / partition).write_bytes(b"tests/test_native.py # rw_netcdf reads a native fixture\r\n")
    monkeypatch.setattr(run_stage1, "publication_files", lambda root: ["tests/test_publication.py"])
    calls = []
    monkeypatch.setattr(run_stage1.subprocess, "call", lambda command, **kwargs: calls.append(command) or 0)
    assert run_stage1.main(["--root", str(root), "--native-manifest", partition, "--platform", "Windows"]) == 0
    assert "tests/test_native.py" not in calls[0]
    assert "tests/test_portable.py" in calls[0] and "tests/test_publication.py" in calls[0]
    assert set(run_stage1.CONTRACT_FILES) <= set(calls[0])
    linux = set(run_stage1.selected_files(root, native_manifest=partition, platform="linux"))
    windows = set(run_stage1.selected_files(root, native_manifest=partition, platform="windows"))
    assert linux - windows == {"tests/test_native.py"}
    assert set(run_stage1.listed_files(root)) <= linux | windows
    assert run_stage1.native_files(root, partition) == {"tests/test_native.py": "rw_netcdf reads a native fixture"}


@pytest.mark.parametrize("text, message", [
    ("tests/test_absent.py # rw_netcdf reads an external file\n", "no maintained Linux Stage 1 coverage"),
    ("tests/test_present.py\n", "names no required artifact"),
    ("tests/test_present.py # rw_netcdf\ntests/test_present.py # netcdf-writer\n", "duplicates coverage"),
])
def test_native_partition_refuses_uncovered_unexplained_or_duplicate_rows(tmp_path, monkeypatch, capsys, text, message):
    root = _stage1_root(tmp_path, "tests/test_present.py\n")
    (root / run_stage1.NATIVE_MANIFEST).write_text(text, encoding="utf-8")
    monkeypatch.setattr(run_stage1, "publication_files", lambda root: [])
    assert run_stage1.main(["--root", str(root), "--platform", "windows"]) == 2
    assert message in capsys.readouterr().err


# --- the Windows cpu partition, emulated on any host -------------------------
# Public CI 37665304517 failed 12 native-dependent files on the windows-2025
# cpu job that no private leg ran without natives.  The emulation that now
# catches that class must hide every native route and select what the job
# selects.

def test_the_windows_partition_emulation_reaches_no_native_route(tmp_path):
    from tools.battery import emulate_windows_cpu_partition as emulation
    scripts = "Scripts" if os.name == "nt" else "bin"
    gate = tmp_path / "gate-venv"
    environ = {"PATH": os.pathsep.join([str(gate / scripts), str(tmp_path / "system")]),
               "VIRTUAL_ENV": str(gate), "LANG": "C.UTF-8",
               "GPUWM_CPU_PREPROCESS_BRIDGE": "a", "GPUWM_ENSEMBLE_PREPARATION_BRIDGE": "a",
               "GPUWM_NCWRITE_BRIDGE": "a", "GPUWM_SUPEROB_BRIDGE": "a", "GPUWM_RW_NETCDF": "a",
               "RUSTWX_ASSETS_DIR": "a", "XDG_CACHE_HOME": "a", "PYTHONPATH": "a", "CARGO_TARGET_DIR": "a"}
    home, venv = tmp_path / "home", tmp_path / "venv"
    env = emulation.emulated_environment(environ, home, venv)
    assert [key for key in env if key.startswith(("GPUWM_", "RUSTWX_", "XDG_"))] == ["GPUWM_NO_LOCAL_GPU"]
    assert env["GPUWM_NO_LOCAL_GPU"] == "1" and env["CUDA_VISIBLE_DEVICES"] == ""
    assert "PYTHONPATH" not in env and "CARGO_TARGET_DIR" not in env
    assert env["HOME"] == env["USERPROFILE"] == str(home) and env["VIRTUAL_ENV"] == str(venv)
    assert env["PATH"].split(os.pathsep) == [str(venv / scripts), str(tmp_path / "system")]
    assert env["LANG"] == "C.UTF-8"


def test_the_windows_partition_emulation_names_every_in_tree_native_directory(tmp_path):
    from tools.battery import emulate_windows_cpu_partition as emulation
    built = ["tools/rustwx/target", "tools/rw_wps/crates/target", "libexec/bridges", "gpuwm/libexec/bridges"]
    assert emulation.reachable_native_directories(tmp_path) == []
    for name in built:
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "tools/rustwx/src").mkdir()
    assert emulation.reachable_native_directories(tmp_path) == sorted(tmp_path / name for name in built)


def test_the_windows_partition_emulation_never_removes_a_directory_it_did_not_make(tmp_path, capsys):
    from tools.battery import emulate_windows_cpu_partition as emulation
    kept = tmp_path / "work" / "kept.txt"
    kept.parent.mkdir()
    kept.write_text("not the emulation's\n", encoding="utf-8")
    assert emulation.main(["--work-dir", str(kept.parent)]) == 2
    assert "is not empty" in capsys.readouterr().err
    assert kept.read_text(encoding="utf-8") == "not the emulation's\n"


def test_the_windows_partition_emulation_selects_what_the_cpu_job_selects(ci):
    from tools.battery import emulate_windows_cpu_partition as emulation
    assert emulation.cpu_leg(ROOT, "windows")["os"] == "windows-2025"
    assert emulation.cpu_leg(ROOT, "linux")["os"] == "ubuntu-24.04"
    assert f'-m "{emulation.CPU_JOB_MARKERS}"' in _runs(ci["jobs"]["cpu"])
    header = (ROOT / run_stage1.NATIVE_MANIFEST).read_text(encoding="utf-8")
    assert "python tools/battery/emulate_windows_cpu_partition.py --work-dir" in header


# --- oracles ----------------------------------------------------------------

def test_the_oracles_job_hands_the_list_to_the_runner_not_to_a_shell(ci):
    run = _runs(ci["jobs"]["oracles"])
    assert "tools/battery/run_must_run_gates.py --minimum 20" in run
    assert "mapfile" not in run and "${decks" not in run
    assert "-p tools.battery.no_silent_skip" in run


@pytest.mark.parametrize("job", ["cpu", "oracles"])
def test_numeric_hosted_jobs_pin_the_measured_platform_numpy_profiles(ci, job):
    strategy = ci["jobs"][job]["strategy"]["matrix"]
    assert {item["os"]: item["numpy"] for item in strategy["include"]} == {
        "ubuntu-24.04": "2.4.6", "windows-2025": "2.2.6"}
    assert 'python -m pip install -e ./gpuwm-data -e ".[dev]" numpy==${{ matrix.numpy }}' in _runs(ci["jobs"][job])


def test_every_oracles_entry_exists():
    decks = run_must_run_gates.listed_decks(ROOT)
    assert len(decks) >= 20
    assert not run_must_run_gates.missing_decks(decks, ROOT)


def _manifest_root(tmp_path, text: str, newline: str = "\n") -> pathlib.Path:
    battery = tmp_path / "tools" / "battery"
    battery.mkdir(parents=True)
    (battery / "must_run_gates.txt").write_bytes(text.replace("\n", newline).encode("utf-8"))
    return tmp_path


def test_the_runner_refuses_a_missing_deck_by_name(tmp_path, capsys):
    root = _manifest_root(tmp_path, "# decks\ntests/test_present.py\ntests/test_gone.py\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_present.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    assert run_must_run_gates.main(["--root", str(root)]) == 2
    error = capsys.readouterr().err
    assert "tests/test_gone.py" in error and "tests/test_present.py" not in error


def test_the_runner_reads_a_crlf_list_as_plain_paths_and_holds_the_minimum(tmp_path, capsys):
    root = _manifest_root(tmp_path, "tests/test_present.py\n", newline="\r\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_present.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    assert run_must_run_gates.listed_decks(root) == ["tests/test_present.py"]
    assert run_must_run_gates.main(["--root", str(root), "--minimum", "2"]) == 2
    assert "fewer than the 2" in capsys.readouterr().err


def test_the_runner_passes_every_manifest_deck_to_pytest_without_shell_translation(monkeypatch):
    calls = []
    monkeypatch.setattr(run_must_run_gates.subprocess, "call",
                        lambda command, **kwargs: calls.append((command, kwargs)) or 0)
    arguments = ["-q", "-p", "no:cacheprovider", "-p", "tools.battery.no_silent_skip",
                 "-m", "not gpu and not network"]
    assert run_must_run_gates.main(["--minimum", "20", "--", *arguments]) == 0
    assert calls == [([sys.executable, "-m", "pytest", *arguments,
                      *run_must_run_gates.listed_decks(ROOT)], {"cwd": ROOT})]
