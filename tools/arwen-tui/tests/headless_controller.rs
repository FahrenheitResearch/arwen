//! Exercise the real desktop controller process with a bounded workspace peer.
//! Only the peer and CLI payload are fixtures; request handling, job ownership,
//! Python selection, worker startup, and controller lifetime are production code.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{env, fs, path::{Path, PathBuf}, process::{Child, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH}};

fn document(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn write_json(path: &Path, value: Value) {
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}
fn digest(path: &Path) -> String { format!("{:x}", Sha256::digest(fs::read(path).unwrap())) }
fn await_file(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !path.is_file() {
        assert!(Instant::now() < deadline, "missing {}", path.display());
        std::thread::sleep(Duration::from_millis(20));
    }
}
fn send(handoff: &Value, id: &str, action: &str, fields: Value) -> Value {
    let control = Path::new(handoff["control_dir"].as_str().unwrap());
    let mut request = fields;
    request["schema"] = json!("arwen.companion-request.v1");
    request["session_id"] = handoff["session_id"].clone();
    request["id"] = json!(id);
    request["action"] = json!(action);
    let temporary = control.join("requests").join(format!("{id}.tmp"));
    write_json(&temporary, request);
    fs::rename(temporary, control.join("requests").join(format!("{id}.json"))).unwrap();
    let response = control.join("responses").join(format!("{id}.json"));
    await_file(&response);
    let value = document(&response);
    assert_eq!(value["ok"], true, "{value}");
    value
}
fn workspace_peer(path: &Path) {
    println!("Workspace startup diagnostic on stdout.");
    eprintln!("Workspace startup diagnostic on stderr.");
    let handoff = document(path);
    assert_eq!(handoff["python"].as_str().unwrap(), env::var("ARWEN_TEST_SELECTED_PYTHON").unwrap());
    assert_eq!(handoff["engine_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(handoff["tui_version"], env!("CARGO_PKG_VERSION"));
    let cwd = PathBuf::from(handoff["cwd"].as_str().unwrap());
    let status_path = Path::new(handoff["status_path"].as_str().unwrap());
    await_file(status_path);
    let status = document(status_path);
    assert_eq!(status["session_id"], handoff["session_id"]);
    assert_eq!(status["target"]["kind"], "local");
    write_json(&cwd.join("observed-handoff.json"), handoff.clone());
    match env::var("ARWEN_TEST_PEER_MODE").unwrap().as_str() {
        "idle" | "slow-probe" => return,
        "failed" => std::process::exit(7),
        "worker" => {},
        other => panic!("unknown peer mode {other}"),
    }
    let config = cwd.join("forecast.toml");
    fs::write(&config, "name = 'Controller acceptance'\n").unwrap();
    send(&handoff, "open-setup", "open_config", json!({"config_path":config}));
    let plan = cwd.join("plan.json");
    write_json(&plan, json!({"schema":"gpuwm.run-plan.v1", "route":"prepared",
        "config":{"path":config}, "output_root":cwd.join("forecast-output")}));
    let launched = send(&handoff, "launch-reviewed", "launch_plan", json!({"plan_path":plan,
        "plan_sha256":digest(&plan), "config_sha256":digest(&config), "target":{"kind":"local"}}));
    write_json(&cwd.join("launch-response.json"), launched);
    await_file(&cwd.join("worker-started.json"));
    fs::write(cwd.join("workspace-closing"), "").unwrap();
}

struct Process(Child);
impl Process {
    fn wait(&mut self) -> std::process::ExitStatus {
        let deadline = Instant::now() + Duration::from_secs(25);
        loop {
            if let Some(status) = self.0.try_wait().unwrap() { return status; }
            assert!(Instant::now() < deadline, "controller timed out");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}
fn scratch(mode: &str) -> PathBuf {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let root = env::temp_dir().join(format!("arwen headless {mode} {} {stamp}", std::process::id()));
    fs::create_dir(&root).unwrap();
    root
}
fn command(root: &Path, python: &Path, mode: &str) -> Command {
    let package = root.join("fixture-modules").join("gpuwm");
    fs::create_dir_all(&package).unwrap();
    fs::write(package.join("__init__.py"), format!("__version__ = '{}'\n", env!("CARGO_PKG_VERSION"))).unwrap();
    fs::write(package.join("tui_worker.py"), include_str!("../../../gpuwm/tui_worker.py")).unwrap();
    fs::write(package.join("cli.py"), r#"
import json, pathlib, sys, time
def main(argv):
    root = pathlib.Path.cwd()
    (root / 'worker-started.json').write_text(json.dumps({'python':sys.executable, 'argv':argv}))
    deadline = time.monotonic() + 15
    while not (root / 'finish-worker').exists():
        if time.monotonic() >= deadline:
            return 9
        time.sleep(.02)
    (root / 'worker-completed').write_text('complete')
    return 0
"#).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_arwen-tui"));
    command.args(["--headless-companion", "--python"]).arg(python)
        .arg("--companion").arg(env::current_exe().unwrap())
        .arg("--output").arg(root.join("runs"))
        .current_dir(root)
        .env("ARWEN_TEST_SELECTED_PYTHON", python)
        .env("ARWEN_TEST_PEER_MODE", mode)
        .env("GPUWM_TUI_PYTHON", root.join("wrong-inherited-python.exe"))
        .env("ARWEN_PYTHON", root.join("wrong-gui-python.exe"))
        .env("PYTHONPATH", package.parent().unwrap())
        .env("PYTHONNOUSERSITE", "1").env("PYTHONSAFEPATH", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("APPDATA", root.join("appdata"))
        .env("LOCALAPPDATA", root.join("localappdata"))
        .env("XDG_CONFIG_HOME", root.join("config"))
        .stdin(Stdio::null())
        .stdout(Stdio::from(fs::File::create(root.join("controller.stdout.log")).unwrap()))
        .stderr(Stdio::from(fs::File::create(root.join("controller.stderr.log")).unwrap()));
    #[cfg(windows)] {
        use std::os::windows::process::CommandExt;
        command.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    }
    command
}
fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.get(1).map(String::as_str) == Some("-P") {
        let root = env::current_dir().unwrap();
        write_json(&root.join("probe-started.json"), json!({"pid":std::process::id()}));
        match env::var("ARWEN_TEST_PEER_MODE").unwrap().as_str() {
            "hung-probe" => std::thread::sleep(Duration::from_secs(60)),
            "slow-probe" => std::thread::sleep(Duration::from_millis(150)),
            "malformed-probe" => { println!("{{not a version}}"); return; },
            other => panic!("unknown probe mode {other}"),
        }
        println!("{}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if args.get(1).map(String::as_str) == Some("--handoff") {
        workspace_peer(Path::new(&args[2]));
        return;
    }
    let python = PathBuf::from(env::var_os("GPUWM_TUI_TEST_PYTHON").expect("set test Python path"));
    assert!(python.is_absolute() && python.is_file());
    for mode in ["idle", "worker", "failed", "missing", "snapshot", "missing-python", "malformed-probe", "slow-probe", "hung-probe"] {
        let root = scratch(mode);
        let mut launch = command(&root, &python, mode);
        if mode == "missing" { launch.arg("--companion").arg(root.join("unavailable-workspace.exe")); }
        if mode == "snapshot" { launch.arg("--snapshot").arg(root.join("snapshot.html")); }
        if mode == "missing-python" { launch.arg("--python").arg(root.join("unavailable-python.exe")); }
        if mode.ends_with("-probe") {
            let fixture = env::current_exe().unwrap();
            launch.arg("--python").arg(&fixture).env("ARWEN_TEST_SELECTED_PYTHON", fixture);
        }
        let began = Instant::now();
        let mut process = Process(launch.spawn().unwrap());
        if mode == "worker" {
            await_file(&root.join("workspace-closing"));
            std::thread::sleep(Duration::from_millis(500));
            assert!(process.0.try_wait().unwrap().is_none(), "controller left its active worker");
            let worker = document(&root.join("worker-started.json"));
            assert_eq!(Path::new(worker["python"].as_str().unwrap()).canonicalize().unwrap(), python.canonicalize().unwrap());
            assert_eq!(worker["argv"][0], "run-plan");
            fs::write(root.join("finish-worker"), "").unwrap();
        }
        let status = process.wait();
        let stderr = fs::read_to_string(root.join("controller.stderr.log")).unwrap();
        assert_eq!(status.success(), matches!(mode, "idle" | "worker" | "slow-probe"), "{mode}: {stderr}");
        assert!(!stderr.contains("interactive terminal"), "headless mode reached terminal initialization");
        if mode == "missing" { assert!(stderr.contains("Visual workspace is not installed"), "{stderr}"); }
        if mode == "snapshot" { assert!(stderr.contains("read-only snapshot"), "{stderr}"); }
        if mode == "missing-python" { assert!(stderr.contains("Cannot start the selected ArWen runtime"), "{stderr}"); }
        if mode == "malformed-probe" { assert!(stderr.contains("invalid ArWen version"), "{stderr}"); }
        if mode == "hung-probe" {
            assert!(stderr.contains("timed out after 20 seconds"), "{stderr}");
            assert!(began.elapsed() < Duration::from_secs(24));
            assert!(!root.join("observed-handoff.json").is_file());
            let pid = document(&root.join("probe-started.json"))["pid"].as_u64().unwrap() as u32;
            #[cfg(windows)] unsafe {
                use windows_sys::Win32::{Foundation::CloseHandle, System::Threading::{GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION}};
                let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
                if !process.is_null() {
                    let mut code = 259;
                    assert_ne!(GetExitCodeProcess(process, &mut code), 0);
                    CloseHandle(process);
                    assert_ne!(code, 259, "timed-out version process is still alive");
                }
            }
            #[cfg(unix)] unsafe { assert_ne!(libc::kill(pid as i32, 0), 0, "timed-out version process was not reaped"); }
        }
        if matches!(mode, "idle" | "worker" | "failed" | "slow-probe") {
            let handoff = document(&root.join("observed-handoff.json"));
            let status = document(Path::new(handoff["status_path"].as_str().unwrap()));
            assert_eq!(status["state"], "closed");
            let log_path = Path::new(handoff["control_dir"].as_str().unwrap()).join("companion.log");
            let log = fs::read_to_string(&log_path).unwrap();
            assert!(log.contains("[ArWen TUI] Visual workspace launch"));
            assert!(log.contains("Workspace startup diagnostic on stdout."));
            assert!(log.contains("Workspace startup diagnostic on stderr."));
            if mode == "failed" {
                let escaped = format!("{:?}", log_path.display().to_string());
                assert!(stderr.contains(escaped.trim_matches('"')), "{stderr}");
            }
        }
        if mode == "worker" {
            assert!(root.join("worker-completed").is_file());
            let response = document(&root.join("launch-response.json"));
            let job = PathBuf::from(response["job_dir"].as_str().unwrap());
            let result = document(&job.join("result.json"));
            assert_eq!(result["status"], "completed");
            assert_eq!(result["exit_code"], 0);
        }
        println!("PASS headless controller {mode}: {}", root.display());
    }
}
