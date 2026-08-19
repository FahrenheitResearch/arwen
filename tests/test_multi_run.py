"""CPU-only tests for process-level multi-run orchestration."""

from __future__ import annotations

import hashlib
import json
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import gpuwm.cli as cli
import gpuwm.multi_run as multi_run
from gpuwm.config_authority import (CONFIG_PAYLOAD_ENV, CONFIG_SHA256_ENV,
                                    CONFIG_SOURCE_ENV)
from gpuwm.multi_run import (GroupOutcome, ProcessRecord, child_environment,
                             load_plan, resolve_devices)
from gpuwm.supervisor import (GPUIdentity, GPU_LOCK_ROOT_ENV,
                              SHARED_INPUT_AUTHORITY_ROOT_ENV)


def _plan_text(*, preflight: str = "off",
               devices: tuple[object, ...] = (0, 1),
               same_config: bool = False,
               nested_cache: bool = False,
               same_outdir: bool = False) -> str:
    lines = [
        'schema = "gpuwm.multi-run-plan/v1"',
        'summary = "production-summary.json"',
        f'preflight = "{preflight}"',
    ]
    for index, device in enumerate(devices):
        suffix = chr(ord("a") + index)
        config = "config_a.toml" if same_config else f"config_{suffix}.toml"
        outdir = "output_a" if same_outdir else f"output_{suffix}"
        cache = (f"output_{suffix}/cache" if nested_cache and index == 0
                 else f"cache_{suffix}")
        rendered_device = (str(device) if isinstance(device, int)
                           else json.dumps(device))
        lines.extend([
            "",
            "[[run]]",
            f'name = "run_{suffix}"',
            f"device = {rendered_device}",
            f'config = "{config}"',
            f'outdir = "{outdir}"',
            f'scratch = "scratch_{suffix}"',
            f'cache = "{cache}"',
        ])
    return "\n".join(lines) + "\n"


def _write_plan(tmp_path: Path, **kwargs) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    devices = kwargs.get("devices", (0, 1))
    for index in range(len(devices)):
        suffix = chr(ord("a") + index)
        (tmp_path / f"config_{suffix}.toml").write_text(
            "[experiment]\nname = 'fixture'\n", encoding="utf-8")
    plan = tmp_path / "plan.toml"
    plan.write_text(_plan_text(**kwargs), encoding="utf-8")
    return plan


def _write_module_plan(tmp_path: Path, *, run_count: int = 1) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\nname = 'fixture'\n", encoding="utf-8")
    lines = [
        'schema = "gpuwm.multi-run-plan/v1"',
        'summary = "production-summary.json"',
        'preflight = "estimate"',
    ]
    for index in range(run_count):
        suffix = chr(ord("a") + index)
        lines.extend([
            "",
            "[[run]]",
            f'name = "module_{suffix}"',
            f"device = {index}",
            'module = "gpuwm.prepared_domain_tree_forecast"',
            'inputs = ["prepared", "experiment.toml"]',
            'args = [',
            '  "--prepared-root", "{input0}",',
            '  "--experiment-config", "{input1}",',
            '  "--outdir", "{outdir}",',
            ']',
            f'outdir = "module-output-{suffix}"',
            f'scratch = "module-scratch-{suffix}"',
            f'cache = "module-cache-{suffix}"',
        ])
    plan = tmp_path / "module-plan.toml"
    plan.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan


def _inventory() -> tuple[GPUIdentity, ...]:
    return (
        GPUIdentity("GPU-alpha", "600.00", "card alpha", 0),
        GPUIdentity("GPU-beta", "600.00", "card beta", 1),
        GPUIdentity("GPU-gamma", "600.00", "card gamma", 2),
    )


def test_plan_accepts_one_to_n_runs_and_resolves_indices(tmp_path):
    one = load_plan(_write_plan(tmp_path / "one", devices=(2,)))
    assert len(one.runs) == 1
    assert resolve_devices(one, _inventory())[0].gpu.uuid == "GPU-gamma"

    many_root = tmp_path / "many"
    many_root.mkdir()
    many = load_plan(_write_plan(many_root, devices=(0, 1, 2)))
    assert [entry.gpu.uuid for entry in resolve_devices(many, _inventory())] \
        == ["GPU-alpha", "GPU-beta", "GPU-gamma"]


def test_plan_rejects_zero_runs(tmp_path):
    plan = tmp_path / "plan.toml"
    plan.write_text(
        'schema = "gpuwm.multi-run-plan/v1"\npreflight = "off"\n',
        encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        load_plan(plan)


@pytest.mark.parametrize("options,match", [
    ({"same_outdir": True}, "overlaps"),
    ({"nested_cache": True}, "overlaps"),
])
def test_plan_rejects_shared_or_nested_paths(tmp_path, options, match):
    plan = _write_plan(tmp_path, **options)
    with pytest.raises(ValueError, match=match):
        load_plan(plan)


def test_shared_read_only_config_is_allowed_for_identity_runs(tmp_path):
    plan_path = _write_plan(tmp_path, same_config=True)
    plan = load_plan(plan_path)
    assert plan.runs[0].config == plan.runs[1].config
    assert plan.runs[0].inputs == plan.runs[1].inputs
    assert plan.sha256 == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    expected_config_hash = hashlib.sha256(
        plan.runs[0].config.read_bytes()).hexdigest()
    assert plan.runs[0].input_bindings[0].sha256 == expected_config_hash
    assert plan.runs[1].input_bindings[0] \
        == plan.runs[0].input_bindings[0]


def test_summary_cannot_overlap_a_mutable_run_root(tmp_path):
    plan = _write_plan(tmp_path)
    text = plan.read_text(encoding="utf-8")
    plan.write_text(text.replace(
        'summary = "production-summary.json"',
        'summary = "output_a/production-summary.json"'),
        encoding="utf-8")
    with pytest.raises(ValueError, match="summary.*overlaps"):
        load_plan(plan)


def test_module_runner_uses_shell_free_argv_and_shared_read_only_inputs(
        tmp_path):
    plan = load_plan(_write_module_plan(tmp_path, run_count=2))
    runs = resolve_devices(plan, _inventory())
    assert plan.runs[0].inputs == plan.runs[1].inputs
    assert plan.runs[0].input_bindings[0].kind == "directory"
    assert plan.runs[0].input_bindings[0].sha256 is None
    assert plan.runs[0].input_bindings[1].sha256 == hashlib.sha256(
        plan.runs[0].inputs[1].read_bytes()).hexdigest()

    command = multi_run._run_command(runs[0])
    assert command[:4] == (
        multi_run.sys.executable, "-m", "gpuwm.multi_run", "worker")
    assert "gpuwm.prepared_domain_tree_forecast" in command
    assert str(plan.runs[0].outdir) in command
    assert str(plan.runs[0].inputs[0]) in command
    assert str(plan.runs[0].inputs[1]) in command
    assert isinstance(command, tuple)

    multi_run._prepare_directories(plan)
    assert plan.runs[0].scratch_dir.is_dir()
    assert plan.runs[0].cache.is_dir()
    assert (plan.runs[0].cache / "cupy").is_dir()
    assert (plan.runs[0].cache / "cuda").is_dir()
    # The production runner owns the atomic output-directory claim.
    assert not plan.runs[0].outdir.exists()


def test_single_prepared_runner_binds_all_schema_known_path_flags(tmp_path):
    (tmp_path / "prepared").mkdir()
    (tmp_path / "experiment.toml").write_text(
        "[experiment]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "namelist.wps").write_text(
        "&share\n/\n", encoding="utf-8")
    plan_path = tmp_path / "single-plan.toml"
    plan_path.write_text("\n".join([
        'schema = "gpuwm.multi-run-plan/v1"',
        'summary = "summary.json"',
        'preflight = "off"',
        '[[run]]',
        'name = "single"',
        'device = 0',
        'module = "gpuwm.prepared_single_domain_forecast"',
        'inputs = ["prepared", "experiment.toml", "namelist.wps"]',
        'args = [',
        '  "--prepared-root", "{input0}",',
        '  "--experiment-config", "{input1}",',
        '  "--wps-namelist", "{input2}",',
        '  "--outdir", "{outdir}",',
        ']',
        'outdir = "output"',
        'scratch = "scratch"',
        'cache = "cache"',
        '',
    ]), encoding="utf-8")
    plan = load_plan(plan_path)
    assert plan.runs[0].module \
        == "gpuwm.prepared_single_domain_forecast"
    assert [binding.kind for binding in plan.runs[0].input_bindings] \
        == ["directory", "file", "file"]


def test_module_form_requires_declared_inputs_and_outdir_placeholder(tmp_path):
    plan = _write_module_plan(tmp_path)
    text = plan.read_text(encoding="utf-8")
    plan.write_text(text.replace(
        'inputs = ["prepared", "experiment.toml"]', "inputs = []"),
        encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_plan(plan)

    plan = _write_module_plan(tmp_path / "other")
    text = plan.read_text(encoding="utf-8")
    plan.write_text(text.replace('"{outdir}"', '"fixed-output"'),
                    encoding="utf-8")
    with pytest.raises(ValueError, match=r"--outdir.*\{outdir\}"):
        load_plan(plan)

    plan = _write_module_plan(tmp_path / "placeholder-typo")
    text = plan.read_text(encoding="utf-8")
    plan.write_text(text.replace(
        '"--outdir", "{outdir}",',
        '"--outdir", "{outdir}", "--label", "{outdirx}",'),
        encoding="utf-8")
    with pytest.raises(ValueError, match="unknown placeholder.*outdirx"):
        load_plan(plan)


def test_module_form_rejects_duplicate_or_alternate_outdir_options(tmp_path):
    duplicate = _write_module_plan(tmp_path / "duplicate")
    text = duplicate.read_text(encoding="utf-8")
    duplicate.write_text(text.replace(
        '  "--outdir", "{outdir}",',
        '  "--outdir", "{outdir}", "--outdir", "escaped-output",'),
        encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one canonical --outdir"):
        load_plan(duplicate)

    alternate = _write_module_plan(tmp_path / "alternate")
    text = alternate.read_text(encoding="utf-8")
    alternate.write_text(text.replace(
        '  "--outdir", "{outdir}",', '  "--outdir={outdir}",'),
        encoding="utf-8")
    with pytest.raises(ValueError, match="alternate output option"):
        load_plan(alternate)


def test_module_form_requires_canonical_declared_input_path_flags(tmp_path):
    missing = _write_module_plan(tmp_path / "missing")
    text = missing.read_text(encoding="utf-8")
    missing.write_text(text.replace(
        '  "--prepared-root", "{input0}",\n', ""), encoding="utf-8")
    with pytest.raises(ValueError, match="--prepared-root exactly once"):
        load_plan(missing)

    duplicate = _write_module_plan(tmp_path / "duplicate-input")
    text = duplicate.read_text(encoding="utf-8")
    duplicate.write_text(text.replace(
        '  "--experiment-config", "{input1}",',
        '  "--experiment-config", "{input1}", '
        '"--experiment-config", "{input1}",'), encoding="utf-8")
    with pytest.raises(ValueError, match="--experiment-config exactly once"):
        load_plan(duplicate)

    alternate = _write_module_plan(tmp_path / "alternate-input")
    text = alternate.read_text(encoding="utf-8")
    alternate.write_text(text.replace(
        '"--prepared-root", "{input0}"',
        '"--prepared-root={input0}"'), encoding="utf-8")
    with pytest.raises(ValueError, match="alternate path option"):
        load_plan(alternate)


def test_module_form_rejects_duplicate_declared_input_paths(tmp_path):
    plan = _write_module_plan(tmp_path)
    text = plan.read_text(encoding="utf-8")
    plan.write_text(text.replace(
        'inputs = ["prepared", "experiment.toml"]',
        'inputs = ["prepared", "prepared", "experiment.toml"]'),
        encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate declared input"):
        load_plan(plan)


def test_module_form_rejects_literal_cross_run_input_escape(tmp_path):
    plan = _write_module_plan(tmp_path, run_count=2)
    before, second = plan.read_text(encoding="utf-8").split(
        'name = "module_b"', 1)
    second = second.replace(
        '"--prepared-root", "{input0}"',
        '"--prepared-root", "module-output-a"', 1)
    plan.write_text(before + 'name = "module_b"' + second, encoding="utf-8")
    with pytest.raises(ValueError, match="exact declared.*inputN"):
        load_plan(plan)


def test_module_form_rejects_input_placeholder_on_non_path_flag(tmp_path):
    plan = _write_module_plan(tmp_path)
    text = plan.read_text(encoding="utf-8")
    plan.write_text(text.replace(
        '  "--outdir", "{outdir}",',
        '  "--io-mode", "{input0}", "--outdir", "{outdir}",'),
        encoding="utf-8")
    with pytest.raises(ValueError, match="only as exact values"):
        load_plan(plan)


@pytest.mark.parametrize("module", [
    "gpuwm.cli",
    "gpuwm.multi_run",
    "gpuwm.ignored_output_runner",
])
def test_module_form_rejects_unsupported_or_output_ignoring_modules(
        tmp_path, module):
    plan = _write_module_plan(tmp_path / module.rsplit(".", 1)[-1])
    text = plan.read_text(encoding="utf-8")
    plan.write_text(text.replace(
        'module = "gpuwm.prepared_domain_tree_forecast"',
        f'module = "{module}"'), encoding="utf-8")
    with pytest.raises(ValueError, match="supported prepared forecast"):
        load_plan(plan)


def test_locked_module_worker_reuses_uuid_lock_before_calling_target(
        tmp_path, monkeypatch):
    events = []
    outdir = tmp_path / "new-output"
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\nname='fixture'\n", encoding="utf-8")

    class _Lock:
        def __init__(self, uuid, *, run_id):
            events.append(("lock-created", uuid, run_id))

        def __enter__(self):
            events.append(("lock-entered",))
            return self

        def __exit__(self, *_exc):
            events.append(("lock-exited",))

    class _Module:
        @staticmethod
        def main(arguments):
            events.append(("main", tuple(arguments)))
            return 6

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-alpha")
    monkeypatch.setattr(multi_run, "GPUFileLock", _Lock)
    monkeypatch.setattr(
        multi_run, "preflight_exclusive_gpu",
        lambda uuid, **kwargs: events.append(("preflight", uuid, kwargs)))
    monkeypatch.setattr(
        multi_run.importlib, "import_module", lambda name: _Module())

    rc = multi_run._locked_module_main(
        gpu_uuid="GPU-alpha",
        module_name="gpuwm.prepared_domain_tree_forecast",
        outdir=outdir,
        inputs=(prepared, config),
        arguments=(
            "--prepared-root", str(prepared),
            "--experiment-config", str(config),
            "--outdir", str(outdir)))
    assert rc == 6
    assert [event[0] for event in events] == [
        "lock-created", "lock-entered", "preflight", "main", "lock-exited"]
    assert events[2][2]["approved_pids"] == {multi_run.os.getpid()}


@pytest.mark.parametrize("arguments,match", [
    (("--outdir", "{out}", "--outdir", "escaped"),
     "exactly one canonical"),
    (("--outdir=escaped",), "alternate output option"),
])
def test_locked_module_worker_revalidates_outdir_contract_before_locking(
        tmp_path, monkeypatch, arguments, match):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-alpha")
    monkeypatch.setattr(
        multi_run, "GPUFileLock",
        lambda *_args, **_kwargs: pytest.fail("lock must not be acquired"))
    with pytest.raises(ValueError, match=match):
        multi_run._locked_module_main(
            gpu_uuid="GPU-alpha",
            module_name="gpuwm.prepared_domain_tree_forecast",
            outdir=tmp_path / "output", inputs=(), arguments=arguments)


def test_locked_check_worker_masks_and_locks_before_importing_cli(
        tmp_path, monkeypatch):
    events = []
    held = {"value": False}

    class _Lock:
        def __init__(self, uuid, *, run_id):
            events.append(("lock-created", uuid, run_id))

        def __enter__(self):
            held["value"] = True
            events.append(("lock-entered",))
            return self

        def __exit__(self, *_exc):
            events.append(("lock-exited",))
            held["value"] = False

    class _Cli:
        @staticmethod
        def main(arguments):
            assert held["value"]
            assert multi_run.os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-beta"
            events.append(("cli-main", tuple(arguments)))
            return 5

    def imported(name):
        assert held["value"]
        events.append(("import", name))
        return _Cli()

    config = tmp_path / "config.toml"
    config.write_text("[experiment]\nname='fixture'\n", encoding="utf-8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-beta")
    monkeypatch.setattr(multi_run, "GPUFileLock", _Lock)
    monkeypatch.setattr(
        multi_run, "preflight_exclusive_gpu",
        lambda uuid, **kwargs: events.append(("preflight", uuid, kwargs)))
    monkeypatch.setattr(multi_run.importlib, "import_module", imported)

    rc = multi_run._locked_check_main(
        gpu_uuid="GPU-beta", config=config, mode="alloc")
    assert rc == 5
    assert [event[0] for event in events] == [
        "lock-created", "lock-entered", "preflight", "import", "cli-main",
        "lock-exited"]
    assert events[2][2]["approved_pids"] == {multi_run.os.getpid()}
    assert events[4][1] == ("check", str(config), "--alloc")


def test_plan_requires_new_mutable_directories(tmp_path):
    plan = _write_plan(tmp_path)
    (tmp_path / "output_a").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        load_plan(plan)


def test_plan_requires_new_create_only_shared_authority_store(tmp_path):
    plan = _write_plan(tmp_path)
    store = tmp_path / "production-summary.json.input-authorities"
    store.mkdir()
    with pytest.raises(ValueError, match="input authority store.*already exists"):
        load_plan(plan)


def test_device_aliases_cannot_select_one_physical_gpu_twice(tmp_path):
    plan = load_plan(_write_plan(
        tmp_path, devices=(0, "GPU-alpha")))
    with pytest.raises(ValueError, match="unique physical device"):
        resolve_devices(plan, _inventory())


def test_child_environment_preserves_parent_and_isolates_runtime_paths(
        tmp_path):
    plan = load_plan(_write_plan(tmp_path, devices=(1,)))
    run = resolve_devices(plan, _inventory())[0]
    parent = {
        "KEEP_ME": "unchanged",
        "PATH": "parent-path",
        "TMPDIR": str(tmp_path / "parent-temp"),
    }
    environment = child_environment(run, parent)

    expected = dict(parent)
    expected.update({
        "CUDA_VISIBLE_DEVICES": "GPU-beta",
        "CUDA_CACHE_PATH": str(run.spec.cache / "cuda"),
        "CUPY_CACHE_DIR": str(run.spec.cache / "cupy"),
        GPU_LOCK_ROOT_ENV: str(
            (tmp_path / "parent-temp" / "gpuwm" / "locks").resolve()),
        SHARED_INPUT_AUTHORITY_ROOT_ENV: str(
            plan.input_authority_store),
        CONFIG_PAYLOAD_ENV: str(run.spec.captured_config.resolve()),
        CONFIG_SOURCE_ENV: str(run.spec.config.resolve()),
        CONFIG_SHA256_ENV: run.spec.input_bindings[0].sha256,
        "TEMP": str(run.spec.scratch_dir),
        "TMP": str(run.spec.scratch_dir),
        "TMPDIR": str(run.spec.scratch_dir),
    })
    assert environment == expected


def test_captured_config_is_only_byte_authority_through_route_check_supervisor(
        tmp_path, monkeypatch):
    """Same-size source mutation cannot affect any post-capture config use."""

    import gpuwm.case_data as case_data
    import gpuwm.core.preflight as core_preflight
    import gpuwm.ingest.preflight as ingest_preflight
    import gpuwm.supervisor as supervisor

    plan_path = _write_plan(tmp_path, devices=(0,), preflight="off")
    config = tmp_path / "config_a.toml"
    original = (
        b"[experiment]\nname='aaaaaaaa'\n"
        b"[case_data]\nmarker=true\n")
    mutated = original.replace(b"aaaaaaaa", b"bbbbbbbb")
    assert len(mutated) == len(original)
    config.write_bytes(original)
    plan = load_plan(plan_path)
    run = resolve_devices(plan, _inventory())[0]
    multi_run._prepare_directories(plan)
    assert run.spec.captured_config.read_bytes() == original
    environment = child_environment(run, {})
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    metadata = config.stat()
    descriptor = os.open(config, os.O_WRONLY | os.O_TRUNC)
    try:
        os.write(descriptor, mutated)
    finally:
        os.close(descriptor)
    os.utime(config, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))

    real_path_open = Path.open
    source = config.resolve()

    def guarded_open(path, *args, **kwargs):
        if Path(path).resolve() == source:
            pytest.fail("mutable original config was opened after capture")
        return real_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    observed_payloads = []
    fake_exp = SimpleNamespace(name="captured", domains=())

    def load_bytes(payload, *, source, base_dir, **kwargs):
        # **kwargs so a keyword the real loader grows does not land here
        # as a TypeError.  2.3.3 added `require_met_inputs` for `gpuwm
        # static`, every caller kept passing it, and this double -- which
        # is about WHICH BYTES reach the loader and about nothing else --
        # started failing on an argument it has no opinion about.
        observed_payloads.append(payload)
        assert source == str(config.resolve())
        assert Path(base_dir) == config.parent.resolve()
        return fake_exp, object()

    monkeypatch.setattr(case_data, "load_experiment_case_bytes", load_bytes)
    monkeypatch.setattr(supervisor, "supervise_from_cli", lambda _args: 0)
    args = cli.build_parser().parse_args([
        "run", str(config), "--outdir", str(tmp_path / "cli-output")])
    assert cli._dispatch(args) == 0

    monkeypatch.setattr(
        ingest_preflight, "preflight_report",
        lambda *_args: SimpleNamespace(ok=True, format=lambda: "PASS"))
    assert ingest_preflight._check_command(
        Namespace(config=config, json=False)) == 0
    assert core_preflight._load_experiment_any(config) is fake_exp
    assert core_preflight.config_forcing_source(config) == "era5"

    class _StopAfterConfigDiscovery(RuntimeError):
        pass

    def stop_after_config(path, **kwargs):
        assert Path(path) == config.resolve()
        assert kwargs["config_bytes"] == original
        raise _StopAfterConfigDiscovery

    monkeypatch.setattr(supervisor, "resolved_input_hashes",
                        stop_after_config)
    with pytest.raises(_StopAfterConfigDiscovery):
        supervisor.supervise_experiment(
            config, tmp_path / "supervisor-output", poll_seconds=0.05)

    descriptor = os.open(config, os.O_WRONLY | os.O_TRUNC)
    try:
        os.write(descriptor, original)
    finally:
        os.close(descriptor)
    os.utime(config, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    assert len(observed_payloads) >= 3
    assert all(payload == original for payload in observed_payloads)
    assert config.stat().st_mtime_ns == metadata.st_mtime_ns


def test_group_launches_every_child_before_polling_and_aggregates_failures(
        tmp_path):
    plan = load_plan(_write_plan(tmp_path))
    runs = resolve_devices(plan, _inventory())
    multi_run._prepare_directories(plan)
    created = []
    exit_codes = (0, 7)

    class _Process:
        def __init__(self, index):
            self.index = index
            self.pid = 50_000 + index

        def poll(self):
            # A sequential launch-and-wait implementation fails here: the
            # first child must not be polled before the second is launched.
            assert len(created) == len(runs)
            return exit_codes[self.index]

        def terminate(self):  # pragma: no cover - a safety tripwire
            raise AssertionError("multi-run must not terminate children")

        def kill(self):  # pragma: no cover - a safety tripwire
            raise AssertionError("multi-run must not kill children")

    def popen(command, **kwargs):
        process = _Process(len(created))
        created.append((process, command, kwargs))
        return process

    outcome = multi_run._execute_group(
        runs, phase="run", command_for=multi_run._run_command,
        log_for=lambda run: run.spec.log, popen=popen,
        poll_seconds=0.001)
    assert not outcome.interrupted
    assert [record.exit_code for record in outcome.records] == [0, 7]
    assert [record.status for record in outcome.records] \
        == ["succeeded", "failed"]
    assert outcome.started_at_utc is not None
    assert outcome.completed_at_utc is not None
    assert outcome.wall_seconds is not None
    assert outcome.wall_seconds >= 0.0
    assert all(item[2]["stdin"] is multi_run.subprocess.DEVNULL
               for item in created)
    assert all("shell" not in item[2] for item in created)


def test_config_preflight_checks_overlap_on_distinct_locked_worker_routes(
        tmp_path):
    plan = load_plan(_write_plan(tmp_path, preflight="alloc"))
    runs = resolve_devices(plan, _inventory())
    multi_run._prepare_directories(plan)
    created = []

    class _Process:
        def __init__(self, index):
            self.pid = 55_000 + index

        def poll(self):
            assert len(created) == len(runs)
            return 0

    def popen(command, **kwargs):
        process = _Process(len(created))
        created.append((process, tuple(command), kwargs))
        return process

    outcome = multi_run._execute_group(
        runs, phase="check",
        command_for=lambda run: multi_run._check_command(run, "alloc"),
        log_for=lambda run: run.spec.preflight_log, popen=popen,
        poll_seconds=0.001)
    assert not outcome.interrupted
    assert [item[1][3] for item in created] == [
        "check-worker", "check-worker"]
    assert [item[2]["env"]["CUDA_VISIBLE_DEVICES"] for item in created] == [
        "GPU-alpha", "GPU-beta"]
    assert len({item[2]["env"][GPU_LOCK_ROOT_ENV] for item in created}) == 1
    assert [
        item[1][item[1].index("--gpu-uuid") + 1] for item in created
    ] == ["GPU-alpha", "GPU-beta"]
    assert all("--mode" in item[1] and "alloc" in item[1]
               for item in created)


def test_module_entry_preserves_shell_metacharacters_as_plain_argv(
        tmp_path, monkeypatch):
    observed = {}
    outdir = tmp_path / "output"
    prepared = tmp_path / "prepared"
    config = tmp_path / "config.toml"

    def run(**kwargs):
        observed.update(kwargs)
        return 0

    monkeypatch.setattr(multi_run, "_locked_module_main", run)
    rc = multi_run._module_entry([
        "worker", "--gpu-uuid", "GPU-alpha", "--module",
        "gpuwm.prepared_single_domain_forecast", "--outdir",
        str(outdir), "--input", str(prepared), "--input", str(config),
        "--", "--label", "a && b; $(c)", "--outdir", str(outdir),
    ])
    assert rc == 0
    assert observed["module_name"] \
        == "gpuwm.prepared_single_domain_forecast"
    assert observed["arguments"] == [
        "--label", "a && b; $(c)", "--outdir", str(outdir)]
    assert observed["inputs"] == (prepared.resolve(), config.resolve())


def test_check_worker_entry_routes_config_and_mode(tmp_path, monkeypatch):
    observed = {}
    config = tmp_path / "config.toml"
    config.write_text("[experiment]\nname='fixture'\n", encoding="utf-8")

    def run(**kwargs):
        observed.update(kwargs)
        return 4

    monkeypatch.setattr(multi_run, "_locked_check_main", run)
    rc = multi_run._module_entry([
        "check-worker", "--gpu-uuid", "GPU-beta", "--config",
        str(config), "--mode", "estimate",
    ])
    assert rc == 4
    assert observed == {
        "gpu_uuid": "GPU-beta",
        "config": config.resolve(),
        "mode": "estimate",
    }


def test_ctrl_c_reports_owned_children_without_terminating_them(tmp_path):
    plan = load_plan(_write_plan(tmp_path))
    runs = resolve_devices(plan, _inventory())
    multi_run._prepare_directories(plan)
    processes = []

    class _InterruptedProcess:
        def __init__(self, index):
            self.pid = 60_000 + index
            self.signals = []

        def poll(self):
            raise KeyboardInterrupt

        def terminate(self):
            self.signals.append("terminate")

        def kill(self):
            self.signals.append("kill")

    def popen(_command, **_kwargs):
        process = _InterruptedProcess(len(processes))
        processes.append(process)
        return process

    outcome = multi_run._execute_group(
        runs, phase="run", command_for=multi_run._run_command,
        log_for=lambda run: run.spec.log, popen=popen,
        poll_seconds=0.001)
    assert outcome.interrupted
    assert all(record.status == "running_unobserved"
               for record in outcome.records)
    assert [record.pid for record in outcome.records] == [60_000, 60_001]
    assert all(process.signals == [] for process in processes)


@pytest.mark.parametrize("locked_exception", [False, True])
def test_arbitrary_second_launch_failure_preserves_first_pid_in_summary(
        tmp_path, monkeypatch, locked_exception):
    plan_path = _write_plan(tmp_path, preflight="off")
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)
    processes = []

    class _NoWritableAttributes(BaseException):
        __slots__ = ()

        def __setattr__(self, _name, _value):
            raise AttributeError("exception attributes are sealed")

    injected = (_NoWritableAttributes("sealed launch failure")
                if locked_exception else RuntimeError("second launch failed"))

    class _FirstProcess:
        pid = 424242

        def __init__(self):
            self.signals = []

        def poll(self):  # pragma: no cover - second launch fails first
            raise AssertionError("first process must remain unobserved")

        def terminate(self):
            self.signals.append("terminate")

        def kill(self):
            self.signals.append("kill")

    def popen(_command, **_kwargs):
        if not processes:
            process = _FirstProcess()
            processes.append(process)
            return process
        raise injected

    monkeypatch.setattr(multi_run.subprocess, "Popen", popen)
    with pytest.raises(multi_run.GroupExecutionError) as raised:
        multi_run.multi_run_main(Namespace(
            plan=plan_path, summary=None, preflight=None))

    assert raised.value.original is injected
    assert raised.value.__cause__ is injected
    assert processes[0].signals == []
    summary = json.loads(
        (tmp_path / "production-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["runs"][0]["process"]["pid"] == 424242
    assert summary["runs"][0]["process"]["status"] == "running_unobserved"
    assert summary["runs"][1]["process"]["status"] == "launch_unobserved"
    assert summary["orchestration"]["known_child_pids"] == [424242]
    assert summary["orchestration"]["unobserved_child_pids"] == [424242]


def _record(name: str, phase: str, log: Path, exit_code: int) -> ProcessRecord:
    return ProcessRecord(
        name=name, phase=phase, command=(phase, name), log=log,
        status="succeeded" if exit_code == 0 else "failed",
        pid=70_000, started_at_utc="2026-08-01T00:00:00Z",
        completed_at_utc="2026-08-01T00:00:01Z",
        duration_seconds=1.0, exit_code=exit_code)


def test_check_failure_warns_but_does_not_block_successful_runs(
        tmp_path, monkeypatch, capsys):
    plan_path = _write_plan(tmp_path, preflight="estimate")
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)
    calls = []

    def execute(runs, *, phase, **_kwargs):
        calls.append(phase)
        codes = (4, 0) if phase == "check" else (0, 0)
        records = tuple(_record(
            run.spec.name, phase,
            run.spec.preflight_log if phase == "check" else run.spec.log,
            code) for run, code in zip(runs, codes))
        return GroupOutcome(
            records, False, "2026-08-01T00:00:00Z",
            "2026-08-01T00:00:01.25Z", 1.25)

    monkeypatch.setattr(multi_run, "_execute_group", execute)
    rc = multi_run.multi_run_main(Namespace(
        plan=plan_path, summary=None, preflight=None))

    assert rc == 0
    assert calls == ["check", "run"]
    assert "launch continues" in capsys.readouterr().err
    summary = json.loads(
        (tmp_path / "production-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete_with_warnings"
    assert summary["exit_code"] == 0
    assert summary["runs"][0]["preflight"]["exit_code"] == 4
    assert summary["runs"][0]["process"]["exit_code"] == 0
    assert summary["plan_sha256"] == hashlib.sha256(
        plan_path.read_bytes()).hexdigest()
    assert summary["runs"][0]["config_sha256"] == hashlib.sha256(
        (tmp_path / "config_a.toml").read_bytes()).hexdigest()
    assert summary["runs"][0]["gpu_lock_root"] == str(
        multi_run._parent_lock_root(multi_run.os.environ))
    assert summary["runs"][0]["input_authorities"][0]["binding"] \
        == "prelaunch_sha256"
    assert summary["file_authority_verification"]["status"] == "PASS"
    assert summary["file_authority_verification"]["schema"] \
        == "gpuwm.file-authority-monitor/v1"
    assert summary["file_authority_verification"]["started_at_utc"]
    assert summary["file_authority_verification"]["stopped_at_utc"]
    assert summary["input_authority_store"] == {
        "content_deduplication": "one_create_only_file_per_sha256",
        "entries": [],
        "path": str(
            tmp_path / "production-summary.json.input-authorities"),
        "status": "EMPTY",
    }
    assert summary["execution_timing"] == {
        "completed_child_duration_sum_seconds": 2.0,
        "concurrent_wall_seconds": 1.25,
        "overlap_ratio": 1.6,
        "overlap_ratio_basis": (
            "completed_child_duration_sum_seconds / "
            "concurrent_wall_seconds; this measures process overlap, not "
            "performance against a serial baseline"),
        "observed_child_duration_sum_seconds": 2.0,
        "window_completed_at_utc": "2026-08-01T00:00:01.25Z",
        "window_started_at_utc": "2026-08-01T00:00:00Z",
    }


def test_module_runner_delegates_preflight_and_launches(tmp_path, monkeypatch,
                                                        capsys):
    plan_path = _write_module_plan(tmp_path)
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)
    phases = []

    def execute(runs, *, phase, **_kwargs):
        phases.append(phase)
        records = tuple(_record(
            run.spec.name, phase, run.spec.log, 0) for run in runs)
        return GroupOutcome(records, False)

    monkeypatch.setattr(multi_run, "_execute_group", execute)
    rc = multi_run.multi_run_main(Namespace(
        plan=plan_path, summary=None, preflight=None))
    assert rc == 0
    assert phases == ["run"]
    assert "delegated to production module" in capsys.readouterr().out
    summary = json.loads(
        (tmp_path / "production-summary.json").read_text(encoding="utf-8"))
    assert summary["runs"][0]["module"] \
        == "gpuwm.prepared_domain_tree_forecast"
    assert summary["runs"][0]["preflight"] is None


def test_failure_aggregation_is_nonzero_and_keeps_every_result(
        tmp_path, monkeypatch):
    plan_path = _write_plan(tmp_path, preflight="off")
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)

    def execute(runs, *, phase, **_kwargs):
        assert phase == "run"
        return GroupOutcome(tuple(
            _record(run.spec.name, phase, run.spec.log, code)
            for run, code in zip(runs, (0, 9))), False,
            "2026-08-01T00:00:00Z", "2026-08-01T00:00:01Z", 1.0)

    monkeypatch.setattr(multi_run, "_execute_group", execute)
    rc = multi_run.multi_run_main(Namespace(
        plan=plan_path, summary=None, preflight=None))
    summary = json.loads(
        (tmp_path / "production-summary.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert summary["status"] == "failed"
    assert [row["process"]["exit_code"] for row in summary["runs"]] == [0, 9]
    assert summary["execution_timing"][
        "completed_child_duration_sum_seconds"] == 2.0
    assert summary["execution_timing"]["overlap_ratio"] is None
    assert "not every forecast succeeded" in summary["execution_timing"][
        "overlap_ratio_basis"]


def test_file_authority_mutation_before_launch_refuses_all_children(
        tmp_path, monkeypatch):
    plan_path = _write_plan(tmp_path, preflight="off")
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)
    real_prepare = multi_run._prepare_directories

    def prepare_then_mutate(plan, state=None):
        real_prepare(plan, state)
        (tmp_path / "config_a.toml").write_text(
            "[experiment]\nname='changed-before-launch'\n",
            encoding="utf-8")

    monkeypatch.setattr(multi_run, "_prepare_directories", prepare_then_mutate)
    monkeypatch.setattr(
        multi_run, "_execute_group",
        lambda *_args, **_kwargs: pytest.fail("no child may launch"))

    rc = multi_run.multi_run_main(Namespace(
        plan=plan_path, summary=None, preflight=None))
    summary = json.loads(
        (tmp_path / "production-summary.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert summary["status"] == "failed"
    assert summary["file_authority_verification"]["status"] == "FAIL"
    changed = [row for row in summary["file_authority_verification"]["files"]
               if row["status"] == "FAIL"]
    assert [Path(row["path"]).name for row in changed] == ["config_a.toml"]
    assert all(row["process"]["status"] == "not_started"
               for row in summary["runs"])


def test_file_authority_mutation_during_children_fails_final_receipt(
        tmp_path, monkeypatch):
    plan_path = _write_plan(tmp_path, preflight="off")
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)

    def execute(runs, *, phase, **_kwargs):
        assert phase == "run"
        (tmp_path / "config_b.toml").write_text(
            "[experiment]\nname='changed-during-run'\n", encoding="utf-8")
        records = tuple(
            _record(run.spec.name, phase, run.spec.log, 0) for run in runs)
        return GroupOutcome(
            records, False, "2026-08-01T00:00:00Z",
            "2026-08-01T00:00:01Z", 1.0)

    monkeypatch.setattr(multi_run, "_execute_group", execute)
    rc = multi_run.multi_run_main(Namespace(
        plan=plan_path, summary=None, preflight=None))
    summary = json.loads(
        (tmp_path / "production-summary.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert summary["status"] == "failed"
    assert summary["file_authority_verification"]["status"] == "FAIL"
    assert summary["execution_timing"]["overlap_ratio"] is None
    assert "authority changed" in summary["execution_timing"][
        "overlap_ratio_basis"]


@pytest.mark.parametrize("target_name", ["plan.toml", "config_a.toml"])
def test_authority_monitor_sticks_after_transient_mutation_and_restore(
        tmp_path, target_name):
    plan = load_plan(_write_plan(tmp_path, preflight="off"))
    target = tmp_path / target_name
    original = target.read_bytes()
    monitor = multi_run.FileAuthorityMonitor(
        plan, poll_seconds=0.001).start()

    target.write_bytes(original + b"\n# transient mutation\n")
    assert monitor.wait_for_failure(2.0)
    target.write_bytes(original)
    verification = monitor.stop()

    assert verification["status"] == "FAIL"
    record = next(
        item for item in verification["files"]
        if Path(item["path"]).name == target_name)
    assert record["status"] == "FAIL"
    assert any(
        observation["status"] == "CHANGED"
        and observation["observed_at_utc"]
        for observation in record["observations"])


def test_transient_config_mutation_during_checks_prevents_forecast_launch(
        tmp_path, monkeypatch):
    plan_path = _write_plan(tmp_path, preflight="estimate")
    config = tmp_path / "config_a.toml"
    original = config.read_bytes()
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)
    monkeypatch.setattr(multi_run, "AUTHORITY_MONITOR_POLL_SECONDS", 0.001)
    phases = []

    def execute(runs, *, phase, **_kwargs):
        phases.append(phase)
        assert phase == "check"
        config.write_bytes(original + b"\n# transient\n")
        multi_run.time.sleep(0.02)
        config.write_bytes(original)
        records = tuple(_record(
            run.spec.name, phase, run.spec.preflight_log, 0) for run in runs)
        return GroupOutcome(records, False)

    monkeypatch.setattr(multi_run, "_execute_group", execute)
    rc = multi_run.multi_run_main(Namespace(
        plan=plan_path, summary=None, preflight=None))
    summary = json.loads(
        (tmp_path / "production-summary.json").read_text(encoding="utf-8"))

    assert rc == 1
    assert phases == ["check"]
    assert summary["status"] == "failed"
    assert summary["orchestration"]["stage"] == "config_checks"
    assert summary["file_authority_verification"]["status"] == "FAIL"
    assert all(row["process"]["status"] == "not_started"
               for row in summary["runs"])


def test_interrupted_main_writes_atomic_summary_and_returns_130(
        tmp_path, monkeypatch):
    plan_path = _write_plan(tmp_path, preflight="off")
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)

    def execute(runs, *, phase, **_kwargs):
        records = tuple(ProcessRecord(
            run.spec.name, phase, (phase,), run.spec.log,
            status="running_unobserved", pid=80_000 + index,
            started_at_utc="2026-08-01T00:00:00Z",
            duration_seconds=0.5)
            for index, run in enumerate(runs))
        return GroupOutcome(records, True)

    monkeypatch.setattr(multi_run, "_execute_group", execute)
    rc = multi_run.multi_run_main(Namespace(
        plan=plan_path, summary=None, preflight=None))
    summary_path = tmp_path / "production-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 130
    assert summary["status"] == "interrupted"
    assert summary["exit_code"] == 130
    assert [row["process"]["pid"] for row in summary["runs"]] \
        == [80_000, 80_001]
    assert summary["execution_timing"][
        "completed_child_duration_sum_seconds"] is None
    assert summary["execution_timing"][
        "observed_child_duration_sum_seconds"] == 1.0
    assert summary["execution_timing"]["overlap_ratio"] is None
    assert "interrupted" in summary["execution_timing"][
        "overlap_ratio_basis"]
    assert summary["orchestration"]["known_child_pids"] \
        == [80_000, 80_001]
    assert summary["orchestration"]["unobserved_child_pids"] \
        == [80_000, 80_001]
    assert summary["orchestration"][
        "summary_create_only_capability_proven"] is True
    assert not list(tmp_path.glob("production-summary.json.*.tmp"))


def test_interrupt_before_summary_capability_returns_130_without_receipt(
        tmp_path, monkeypatch, capsys):
    plan_path = _write_plan(tmp_path, preflight="off")
    monkeypatch.setattr(
        multi_run, "query_gpus",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt))

    rc = multi_run.multi_run_main(Namespace(
        plan=plan_path, summary=None, preflight=None))

    assert rc == 130
    assert not (tmp_path / "production-summary.json").exists()
    stderr = capsys.readouterr().err
    assert "device_discovery" in stderr
    assert "capability had not been established" in stderr


def test_interrupt_during_directory_claim_records_claimed_paths(
        tmp_path, monkeypatch):
    plan_path = _write_plan(tmp_path, preflight="off")
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)

    def interrupt_after_one_claim(plan, state):
        claimed = plan.runs[0].scratch_dir
        claimed.mkdir()
        state.claimed_directories.append(claimed)
        raise KeyboardInterrupt

    monkeypatch.setattr(
        multi_run, "_prepare_directories", interrupt_after_one_claim)
    rc = multi_run.multi_run_main(Namespace(
        plan=plan_path, summary=None, preflight=None))
    summary = json.loads(
        (tmp_path / "production-summary.json").read_text(encoding="utf-8"))

    assert rc == 130
    assert summary["status"] == "interrupted"
    assert summary["orchestration"]["stage"] == "directory_claim"
    assert summary["orchestration"]["parent_claimed_directories"] \
        == [str((tmp_path / "scratch_a").resolve())]
    assert summary["file_authority_verification"]["status"] == "NOT_STARTED"


def test_interrupt_during_authority_monitor_start_publishes_receipt(
        tmp_path, monkeypatch):
    plan_path = _write_plan(tmp_path, preflight="off")
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)
    monkeypatch.setattr(
        multi_run.FileAuthorityMonitor, "start",
        lambda _self: (_ for _ in ()).throw(KeyboardInterrupt))

    rc = multi_run.multi_run_main(Namespace(
        plan=plan_path, summary=None, preflight=None))
    summary = json.loads(
        (tmp_path / "production-summary.json").read_text(encoding="utf-8"))

    assert rc == 130
    assert summary["status"] == "interrupted"
    assert summary["orchestration"]["stage"] == "authority_monitor_start"
    assert summary["file_authority_verification"]["status"] == "FAIL"


@pytest.mark.parametrize(("preflight", "phase", "stage"), [
    ("estimate", "check", "config_checks"),
    ("off", "run", "forecast"),
])
def test_outer_interrupt_during_child_stage_publishes_receipt(
        tmp_path, monkeypatch, preflight, phase, stage):
    plan_path = _write_plan(tmp_path, preflight=preflight)
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)

    def interrupt(_runs, *, phase: str, **_kwargs):
        assert phase == phase_expected
        raise KeyboardInterrupt

    phase_expected = phase
    monkeypatch.setattr(multi_run, "_execute_group", interrupt)
    rc = multi_run.multi_run_main(Namespace(
        plan=plan_path, summary=None, preflight=None))
    summary = json.loads(
        (tmp_path / "production-summary.json").read_text(encoding="utf-8"))

    assert rc == 130
    assert summary["status"] == "interrupted"
    assert summary["orchestration"]["stage"] == stage
    assert summary["file_authority_verification"]["stopped_at_utc"]


def test_unexpected_post_start_exception_stops_monitor_and_records_failure(
        tmp_path, monkeypatch):
    plan_path = _write_plan(tmp_path, preflight="off")
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)
    monkeypatch.setattr(
        multi_run, "_execute_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected post-start failure")))

    with pytest.raises(RuntimeError, match="injected post-start failure"):
        multi_run.multi_run_main(Namespace(
            plan=plan_path, summary=None, preflight=None))
    summary = json.loads(
        (tmp_path / "production-summary.json").read_text(encoding="utf-8"))

    assert summary["status"] == "failed"
    assert summary["orchestration"]["stage"] == "forecast"
    assert "injected post-start failure" in summary["orchestration"]["error"]
    assert summary["file_authority_verification"]["stopped_at_utc"]
    assert not any(
        thread.name == "gpuwm-file-authority-monitor"
        for thread in multi_run.threading.enumerate())


def test_interrupt_during_final_publication_retries_create_only_receipt(
        tmp_path, monkeypatch):
    plan_path = _write_plan(tmp_path, preflight="off")
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)
    monkeypatch.setattr(
        multi_run, "_execute_group",
        lambda runs, *, phase, **_kwargs: GroupOutcome(tuple(
            _record(run.spec.name, phase, run.spec.log, 0) for run in runs),
            False))
    real_write = multi_run._write_summary
    calls = []

    def interrupt_once(path, payload):
        calls.append(payload["status"])
        if len(calls) == 1:
            raise KeyboardInterrupt
        return real_write(path, payload)

    monkeypatch.setattr(multi_run, "_write_summary", interrupt_once)
    rc = multi_run.multi_run_main(Namespace(
        plan=plan_path, summary=None, preflight=None))
    summary = json.loads(
        (tmp_path / "production-summary.json").read_text(encoding="utf-8"))

    assert rc == 130
    assert calls == ["complete", "interrupted"]
    assert summary["status"] == "interrupted"
    assert summary["orchestration"]["stage"] == "summary_publication"


def test_main_publication_race_preserves_other_writer(tmp_path, monkeypatch,
                                                       capsys):
    plan_path = _write_plan(tmp_path, preflight="off")
    summary_path = tmp_path / "production-summary.json"
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)
    monkeypatch.setattr(
        multi_run, "_execute_group",
        lambda runs, *, phase, **_kwargs: GroupOutcome(tuple(
            _record(run.spec.name, phase, run.spec.log, 0) for run in runs),
            False))

    def race(_path, _payload):
        summary_path.write_text('{"publisher":"other"}\n', encoding="utf-8")
        raise ValueError("racing summary exists")

    monkeypatch.setattr(multi_run, "_write_summary", race)
    rc = multi_run.multi_run_main(Namespace(
        plan=plan_path, summary=None, preflight=None))

    assert rc == 1
    assert json.loads(summary_path.read_text(encoding="utf-8")) \
        == {"publisher": "other"}
    assert "another invocation was preserved" in capsys.readouterr().err


def test_summary_publication_is_create_only_under_a_deterministic_race(
        tmp_path, monkeypatch):
    summary = tmp_path / "summary.json"
    real_link = multi_run.os.link

    def racing_link(source, destination):
        Path(destination).write_text("racing publisher\n", encoding="utf-8")
        return real_link(source, destination)

    monkeypatch.setattr(multi_run.os, "link", racing_link)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        multi_run._write_summary(summary, {"status": "ours"})
    assert summary.read_text(encoding="utf-8") == "racing publisher\n"
    assert not list(tmp_path.glob("summary.json.*.tmp"))


def test_summary_create_only_capability_is_proven_before_child_launch(
        tmp_path, monkeypatch):
    plan_path = _write_plan(tmp_path, preflight="off")
    monkeypatch.setattr(multi_run, "query_gpus", _inventory)
    monkeypatch.setattr(
        multi_run.os, "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("hard links unsupported")))
    monkeypatch.setattr(
        multi_run, "_execute_group",
        lambda *_args, **_kwargs: pytest.fail("no child may launch"))

    with pytest.raises(ValueError, match="does not support.*no child"):
        multi_run.multi_run_main(Namespace(
            plan=plan_path, summary=None, preflight=None))
    assert not (tmp_path / "output_a").exists()
    assert not (tmp_path / "scratch_a").exists()
    assert not list(tmp_path.glob(".production-summary.json.*"))


def test_cli_registers_multi_run_parser():
    args = cli.build_parser().parse_args(["multi-run", "plan.toml"])
    assert args.command == "multi-run"
    assert args.func is multi_run.multi_run_main
    assert args.preflight is None


def test_lock_root_override_is_independent_of_isolated_tmpdir(
        tmp_path, monkeypatch):
    from gpuwm.supervisor import default_lock_path

    shared = tmp_path / "shared-locks"
    monkeypatch.setenv(GPU_LOCK_ROOT_ENV, str(shared))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "isolated-temp"))
    assert default_lock_path("GPU-alpha").parent == shared


def test_resolved_lock_root_is_passed_exactly_despite_child_temp_isolation(
        tmp_path):
    plan = load_plan(_write_plan(tmp_path, devices=(1,)))
    shared = (tmp_path / "machine-wide-locks").resolve()
    run = resolve_devices(
        plan, _inventory(), lock_root=shared)[0]

    environment = child_environment(run, {
        "TMPDIR": str(tmp_path / "unrelated-parent-temp")})

    assert environment[GPU_LOCK_ROOT_ENV] == str(shared)
    assert environment["TMPDIR"] == str(run.spec.scratch_dir)


def test_default_tmp_lock_root_cannot_nest_under_scratch(tmp_path):
    lock_root = multi_run._parent_lock_root({"TMPDIR": "/tmp"})
    scratch = lock_root.parent
    plan_path = _write_plan(tmp_path, devices=(0,))
    plan_path.write_text(
        plan_path.read_text(encoding="utf-8").replace(
            'scratch = "scratch_a"',
            f'scratch = {json.dumps(str(scratch))}'),
        encoding="utf-8")
    plan = load_plan(plan_path)

    with pytest.raises(ValueError, match="shared GPU lock root.*overlaps"):
        multi_run._validate_lock_root_isolation(plan, lock_root)
