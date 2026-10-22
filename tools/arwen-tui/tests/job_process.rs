#[path = "../src/job.rs"]
pub mod job;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
fn python() -> PathBuf {
    PathBuf::from(std::env::var_os("GPUWM_TUI_TEST_PYTHON").expect("set test Python path"))
}
fn scratch(label: &str) -> PathBuf {
    // Parallel tests read one clock tick on the 2.8.6 windows-2025 runner and collided on this name; the counter keeps each call distinct.
    static NEXT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("arwen-tui-{label}-{}-{stamp}-{}", std::process::id(), NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
    fs::create_dir(&path).unwrap();
    path
}
/// How long a test lets a real Python child run before calling it hung. A
/// hang guard, not a speed bound: the 2.9.4 R8 gate runs these tests beside
/// the other cargo rows on one loaded desktop, where `python -m gpuwm.cli
/// --help` (2.6 s on a quiet machine) ran past the old 20 s bound.
const HANG_GUARD: Duration = Duration::from_secs(600);
fn wait(job: &mut job::Job) -> i32 {
    let deadline = Instant::now() + HANG_GUARD;
    loop {
        if let Some(code) = job.poll().unwrap() {
            return code;
        }
        assert!(
            Instant::now() < deadline,
            "worker timed out: {}",
            job.log_tail(20)
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}
fn await_file(path: &Path) {
    let deadline = Instant::now() + HANG_GUARD;
    while !path.exists() {
        assert!(Instant::now() < deadline, "missing {}", path.display());
        std::thread::sleep(Duration::from_millis(20));
    }
}
fn await_started(job: &mut job::Job) {
    let deadline = Instant::now() + HANG_GUARD;
    while !job.dir.join("start").is_file() {
        assert!(job.poll().unwrap().is_none(), "startup failed: {}", job.log_tail(20));
        assert!(Instant::now() < deadline, "worker did not become ready");
        std::thread::sleep(Duration::from_millis(20));
    }
}
#[test]
fn real_cli_help_failure_and_raw_logs() {
    let root = scratch("cli");
    let cwd = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut help = job::Job::start_with_module_path(&python(), "--help", &[], &root.join("help"), &cwd, Some(&cwd)).unwrap();
    assert_eq!(wait(&mut help), 0);
    assert!(help.log_tail(200).contains("usage:"));
    assert_eq!(help.command[2], "gpuwm.cli");
    let result: serde_json::Value =
        serde_json::from_slice(&fs::read(help.dir.join("result.json")).unwrap()).unwrap();
    assert_eq!(result["status"], "completed");
    let process: serde_json::Value =
        serde_json::from_slice(&fs::read(help.dir.join("process.json")).unwrap()).unwrap();
    assert_eq!(process["pid"], result["pid"]);
    assert_eq!(process["cli_args"], result["cli_args"]);
    let mut bad = job::Job::start_with_module_path(
        &python(),
        "not-an-arwen-command",
        &[],
        &root.join("bad"),
        &cwd, Some(&cwd),
    )
    .unwrap();
    assert_eq!(wait(&mut bad), 2);
    assert!(bad.log_tail(200).contains("invalid choice"));
    assert!(job::Job::start_with_module_path(&python(), "--help", &[], &help.dir, &cwd, Some(&cwd)).is_err());
}
// Only the fixture CLI is substituted. The worker is copied byte for byte;
// the actual process group / JobObject and file logging run on this machine.
fn fixture(root: &Path) {
    fs::write(root.join("memory-exit1.log"), include_str!("fixtures/check-memory-exit1.log")).unwrap();
    // Captured from real Windows and Linux Check invocations, declared budget.
    fs::write(root.join("memory-exit1-linux.log"), include_str!("fixtures/check-memory-exit1-linux.log")).unwrap();
    fs::write(root.join("memory-exit1-compact.log"), include_str!("fixtures/check-memory-exit1-compact.log")).unwrap();
    let package = root.join("gpuwm");
    fs::create_dir(&package).unwrap();
    fs::write(package.join("__init__.py"), "").unwrap();
    fs::copy(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../gpuwm/tui_worker.py"),
        package.join("tui_worker.py"),
    )
    .unwrap();
    fs::write(package.join("cli.py"), r#"
import sys
def main(argv):
    import subprocess, time, os, json, pathlib
    if len(argv) > 1 and argv[1] == 'shared-recovery':
        mode = argv[2]
        folder = pathlib.Path(os.environ['GPUWM_CONFIGURATION_RECOVERY_DIR'])
        if mode != 'no-draft':
            folder.mkdir()
            (folder/'draft.toml').write_bytes(b"name='complete rejected candidate'\n")
            (folder/'recovery.json').write_text(json.dumps({'schema':'arwen.configuration-recovery.v1','status':'memory-refused'}))
        print(json.dumps({'schema':'arwen.configuration-error.v1','kind':'input' if mode == 'ordinary' else 'memory','created':False,'error':'budget refusal'}), flush=True)
        if mode == 'mismatch':
            (folder.parent/'result.json').write_text(json.dumps({'schema':'gpuwm-tui-result-v1','status':'failed','exit_code':2,'cli_args':['another-command']}))
            os._exit(2)
        return 0 if mode == 'success' else 2
    if argv[0] in ('go', 'check'):
        mode = argv[1]
        memory_one = mode.startswith('memory1-')
        mode = mode.removeprefix('memory1-')
        code = 1 if memory_one else (2 if argv[0] == 'go' else 4)
        if memory_one:
            name = 'memory-exit1-linux.log' if mode.startswith('linux') else 'memory-exit1-compact.log' if mode == 'compact' else 'memory-exit1.log'
            print(pathlib.Path(name).read_text(), flush=True)
            if mode in ('mixed', 'linux-mixed'):
                print('unrelated_physics_gate: FAIL', flush=True)
        elif mode == 'fits-toml':
            print('go: memory -- the forecast is the memory-binding phase at 4.93 GiB; it fits the 5.16 GiB budget', flush=True)
            print("gpuwm go: Cannot declare ('shared',) twice (line 95, column 8)", flush=True)
        elif argv[0] == 'go':
            print('go: memory -- the forecast needs 8.28 GiB; that EXCEEDS the 7.22 GiB budget', flush=True)
            print('gpuwm go: this configuration will not fit: that EXCEEDS the 7.22 GiB budget. Refusing here, BEFORE the fetch stage.', flush=True)
        else:
            print('  BINDING PHASE: forecast EXCEEDS the 7.22 GiB budget.', flush=True)
            print('  WARNING: observed peak envelope 8.28 GiB exceeds the WDDM budget 7.22 GiB', flush=True)
        if mode == 'corrupt':
            pathlib.Path(argv[2]).write_text('{not-json')
        elif mode == 'mismatch':
            pathlib.Path(argv[2]).write_text(json.dumps({'schema':'gpuwm-tui-result-v1', 'status':'failed', 'exit_code':code, 'cli_args':['another-command']}))
        if mode in ('missing', 'corrupt', 'mismatch'):
            os._exit(code)
        return code
    if argv[0].startswith('fixture-receipt-'):
        if argv[0].endswith('corrupt'):
            pathlib.Path(argv[1]).write_text('{not-json')
        elif argv[0].endswith('mismatch'):
            pathlib.Path(argv[1]).write_text(json.dumps({
                'schema':'gpuwm-tui-result-v1', 'status':'completed',
                'exit_code':0, 'cli_args':['another-command']}))
        os._exit(0)
    if argv[0] == 'fixture-finish':
        print('before detach', flush=True)
        time.sleep(0.4)
        print('after detach', flush=True)
        return 0
    child_code = "import time\nwhile True:\n open('grandchild.log','a').write('x')\n time.sleep(.03)"
    if argv[0] == 'fixture-ignore-stop':
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        child_code = "import signal\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\n" + child_code
    child = subprocess.Popen([sys.executable, '-u', '-c', child_code])
    print('parent and grandchild started', flush=True)
    try:
        return child.wait()
    except KeyboardInterrupt:
        child.wait(timeout=5)
        raise
"#).unwrap();
}
#[test]
fn detach_keeps_worker_and_logs_alive() {
    let root = scratch("detach");
    fixture(&root);
    let directory = root.join("job");
    let args = vec![
        "path with spaces".into(),
        "literal \"quote\"".into(),
        "日本語".into(),
    ];
    let mut running =
        job::Job::start_with_module_path(&python(), "fixture-finish", &args, &directory, &root, Some(&root)).unwrap();
    await_started(&mut running);
    assert!(running.poll().unwrap().is_none());
    drop(running);
    await_file(&directory.join("result.json"));
    let value: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.join("result.json")).unwrap()).unwrap();
    assert_eq!(value["exit_code"], 0);
    assert_eq!(
        value["cli_args"],
        serde_json::json!(["fixture-finish", args[0], args[1], args[2]])
    );
    assert!(fs::read_to_string(directory.join("job.log"))
        .unwrap()
        .contains("after detach"));
}
#[test]
fn stop_covers_grandchild_and_leaves_unrelated_process_running() {
    let root = scratch("stop");
    fixture(&root);
    #[cfg(unix)]
    let worker_python = {
        use std::os::unix::fs::PermissionsExt;
        // A background launcher passes SIG_IGN across exec. This helper is
        // owned by this fixture and changes no signal disposition in the test
        // process or its other concurrently running jobs.
        let launcher = root.join("python-with-ignored-interrupt");
        let executable = python().to_string_lossy().replace('\'', "'\\''");
        fs::write(&launcher, format!("#!/bin/sh\ntrap '' INT\nexec '{executable}' \"$@\"\n")).unwrap();
        fs::set_permissions(&launcher, fs::Permissions::from_mode(0o755)).unwrap();
        launcher
    };
    #[cfg(not(unix))]
    let worker_python = python();
    let mut unrelated = Command::new(python())
        .arg("-c")
        .arg("import time; time.sleep(30)")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut running =
            job::Job::start_with_module_path(&worker_python, "fixture-wait", &[], &root.join("job"), &root, Some(&root)).unwrap();
        await_started(&mut running);
        let heartbeat = root.join("grandchild.log");
        await_file(&heartbeat);
        running.stop().unwrap();
        assert_eq!(wait(&mut running), 130);
        assert!(running.interrupted());
        assert!(!running.memory_refused);
        std::thread::sleep(Duration::from_millis(150));
        let length = fs::metadata(&heartbeat).unwrap().len();
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(fs::metadata(&heartbeat).unwrap().len(), length);
        assert!(unrelated.try_wait().unwrap().is_none());
        assert!(running.dir.join("result.json").exists());
    }));
    let _ = unrelated.kill();
    let _ = unrelated.wait();
    if let Err(error) = result {
        std::panic::resume_unwind(error);
    }
}

#[test]
fn memory_recovery_requires_the_matching_worker_receipt_and_actual_refusal() {
    let root = scratch("memory-recovery");
    fixture(&root);
    for command in ["go", "check"] {
        for mode in ["valid", "missing", "corrupt", "mismatch"] {
            let directory = root.join(format!("{command}-{mode}"));
            let args = vec![
                mode.into(),
                directory.join("result.json").to_string_lossy().into_owned(),
            ];
            let mut running =
                job::Job::start_with_module_path(&python(), command, &args, &directory, &root, Some(&root)).unwrap();
            assert!(!running.memory_refused);
            assert_eq!(wait(&mut running), if command == "go" { 2 } else { 4 });
            assert_eq!(running.memory_refused, mode == "valid", "{command} {mode}");
        }
    }
    let mut unrelated = job::Job::start_with_module_path(
        &python(),
        "go",
        &["fits-toml".into()],
        &root.join("fits-toml"),
        &root, Some(&root),
    )
    .unwrap();
    assert_eq!(wait(&mut unrelated), 2);
    assert!(!unrelated.memory_refused);
    for mode in ["valid", "missing", "corrupt", "mismatch", "mixed", "linux", "linux-mixed", "compact"] {
        let directory = root.join(format!("check1-{mode}"));
        let args = vec![format!("memory1-{mode}"), directory.join("result.json").to_string_lossy().into_owned()];
        let mut running = job::Job::start_with_module_path(&python(), "check", &args, &directory, &root, Some(&root)).unwrap();
        assert_eq!(wait(&mut running), 1);
        assert_eq!(running.memory_refused, matches!(mode, "valid" | "linux" | "compact"), "check1 {mode}");
    }
}

#[test]
fn shared_configuration_recovery_does_not_depend_on_the_creator_command_name() {
    let root = scratch("shared-recovery");
    fixture(&root);
    for creator in ["case-catalog", "research", "future-configuration-creator"] {
        for mode in ["valid", "ordinary", "success", "mismatch", "no-draft"] {
            let directory = root.join(format!("{creator}-{mode}"));
            let mut running = job::Job::start_with_module_path(&python(), creator,
                &["shared-recovery".into(), mode.into()], &directory, &root, Some(&root)).unwrap();
            assert_eq!(wait(&mut running), if mode == "success" { 0 } else { 2 });
            assert_eq!(running.memory_refused, matches!(mode, "valid" | "no-draft"));
            assert_eq!(running.configuration_recovery().is_some(), mode == "valid");
            if let Some(path) = running.configuration_recovery() {
                assert_eq!(fs::read_to_string(path).unwrap(), "name='complete rejected candidate'\n");
            }
        }
    }
}

#[test]
fn os_zero_without_matching_receipt_is_never_success() {
    for mode in ["missing", "corrupt", "mismatch"] {
        let root = scratch(mode);
        fixture(&root);
        let directory = root.join("job");
        let action = format!("fixture-receipt-{mode}");
        let args = vec![directory.join("result.json").to_string_lossy().into_owned()];
        let mut running = job::Job::start_with_module_path(&python(), &action, &args, &directory, &root, Some(&root)).unwrap();
        assert_eq!(wait(&mut running), 1);
        assert!(running
            .log_tail(30)
            .contains("could not verify its completion"));
        let record: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("launcher-result.json")).unwrap())
                .unwrap();
        assert_eq!(record["os_exit_code"], 0);
        assert_eq!(record["exit_code"], 1);
        assert_eq!(record["worker_receipt_valid"], false);
        if mode == "corrupt" {
            assert_eq!(
                fs::read_to_string(directory.join("result.json")).unwrap(),
                "{not-json"
            );
        }
    }
}

#[test]
fn a_diagnostic_write_failure_still_delivers_the_terminal_result() {
    let root = scratch("diagnostic-failure");
    fixture(&root);
    let directory = root.join("job");
    fs::create_dir(&directory).unwrap();
    fs::write(directory.join("launcher-result.tmp"), "preserve existing bytes").unwrap();
    let mut running = job::Job::start_with_module_path(
        &python(), "fixture-receipt-missing", &[], &directory, &root, Some(&root)).unwrap();
    assert_eq!(wait(&mut running), 1);
    assert_eq!(running.poll().unwrap(), Some(1));
    assert!(running.completion_notice.as_ref().unwrap().contains("Could not save"));
    assert_eq!(fs::read_to_string(directory.join("launcher-result.tmp")).unwrap(), "preserve existing bytes");
}

#[cfg(unix)]
#[test]
fn an_ignored_interrupt_escalates_automatically_or_on_an_explicit_retry() {
    for retry in [false, true] {
        let root = scratch(if retry { "stop-retry" } else { "stop-grace" });
        fixture(&root);
        let mut running = job::Job::start_with_module_path(
            &python(), "fixture-ignore-stop", &[], &root.join("job"), &root, Some(&root)).unwrap();
        await_started(&mut running);
        let heartbeat = root.join("grandchild.log");
        await_file(&heartbeat);
        let started = Instant::now();
        running.stop().unwrap();
        assert!(running.is_stopping());
        if retry {
            running.stop().unwrap();
            assert!(running.stop_message().contains("Force stop"));
        }
        assert_ne!(wait(&mut running), 0);
        assert!(running.interrupted());
        assert!(started.elapsed() < Duration::from_secs(if retry { 3 } else { 8 }));
        std::thread::sleep(Duration::from_millis(150));
        let length = fs::metadata(&heartbeat).unwrap().len();
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(fs::metadata(&heartbeat).unwrap().len(), length);
        assert!(running.dir.join("result.json").is_file());
    }
}

#[test]
fn slow_python_startup_keeps_input_responsive_and_can_stop_before_the_cli() {
    let root = scratch("slow-startup");
    fixture(&root);
    // The package import parks on a gate this test never opens, so the
    // interpreter cannot become ready while the test runs. A launcher that
    // waited for the interpreter would therefore wait until the import gives
    // up (25 s, under the 30 s the old launcher froze input for), and the
    // import leaves a marker when it does. So the property is asserted by
    // ORDER, not by a wall clock: `start` returned while Python was still
    // inside its import. The previous form asserted `start` took under 2 s,
    // which a loaded Windows host (an Intel Xeon 8488C running the R8 gate
    // at 32 jobs) missed while the launcher did nothing wrong.
    fs::write(
        root.join("gpuwm/__init__.py"),
        "import pathlib, sys, time\n\
         pathlib.Path('import-started').write_text('')\n\
         deadline = time.monotonic() + 25\n\
         while not pathlib.Path('import-release').exists():\n\
         \x20   if time.monotonic() > deadline:\n\
         \x20       pathlib.Path('import-gave-up').write_text('')\n\
         \x20       sys.exit(7)\n\
         \x20   time.sleep(0.05)\n",
    )
    .unwrap();
    let mut running = job::Job::start_with_module_path(
        &python(), "fixture-finish", &[], &root.join("job"), &root, Some(&root)).unwrap();
    assert!(
        !root.join("import-gave-up").exists(),
        "UI waited for the delayed interpreter: start returned only after the import gave up"
    );
    assert!(running.poll().unwrap().is_none());
    assert!(!running.dir.join("start").exists());
    // Stop while Python is demonstrably mid-import, not before it ran.
    await_file(&root.join("import-started"));
    assert!(running.poll().unwrap().is_none(), "the parked import ended: {}", running.log_tail(20));
    running.stop().unwrap();
    assert_ne!(wait(&mut running), 0);
    assert!(!running.dir.join("start").exists());
    assert!(!running.log_tail(100).contains("before detach"));
}

#[test]
fn an_unready_worker_times_out_without_releasing_the_command() {
    let root = scratch("startup-deadline");
    fixture(&root);
    fs::write(root.join("gpuwm/__init__.py"), "import time\ntime.sleep(3)\n").unwrap();
    let mut running = job::Job::start_with_module_path(
        &python(), "fixture-finish", &[], &root.join("job"), &root, Some(&root)).unwrap();
    running.started = Instant::now() - Duration::from_secs(31);
    assert_ne!(wait(&mut running), 0);
    assert!(!running.dir.join("start").exists());
    assert!(running.completion_notice.as_ref().unwrap().contains("within 30 seconds"));
}
