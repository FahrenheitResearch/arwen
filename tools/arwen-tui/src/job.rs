//! Owned CLI jobs. Dropping the UI detaches; only an explicit stop signals work.
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Instant;

pub struct Job {
    pub dir: PathBuf,
    pub action: String,
    pub command: Vec<String>,
    pub started: Instant,
    pub outcome: Option<i32>,
    /// Set only after a matching worker receipt and a specific memory refusal.
    pub memory_refused: bool,
    child: Child,
    stopping: bool,
    stop_started: Option<Instant>,
    force_stopped: bool,
    startup_released: bool,
    pub completion_notice: Option<String>,
    #[cfg(windows)]
    owner: windows::Owner,
}

impl Job {
    pub fn start(
        python: &Path,
        action: &str,
        args: &[String],
        directory: &Path,
        cwd: &Path,
    ) -> io::Result<Self> {
        fs::create_dir_all(directory)?;
        let dir = directory.canonicalize()?;
        let cwd = cwd.canonicalize()?;
        let mut command = vec![
            python.to_string_lossy().into_owned(),
            "-m".into(),
            "gpuwm.cli".into(),
            action.into(),
        ];
        command.extend_from_slice(args);
        // create_new claims this directory; never replace another job's log.
        let mut receipt = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join("job.json"))?;
        serde_json::to_writer_pretty(
            &mut receipt,
            &serde_json::json!({
                "schema": "gpuwm-tui-job-v1", "command": command,
                "cwd": cwd, "action": action,
            }),
        )?;
        receipt.write_all(b"\n")?;
        receipt.sync_all()?;
        let log = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join("job.log"))?;
        let mut launch = Command::new(python);
        #[cfg(windows)]
        let owner = windows::Owner::new()?;
        launch
            .arg("-u")
            .arg("-m")
            .arg("gpuwm.tui_worker")
            .arg("--job-dir")
            .arg(&dir);
        #[cfg(windows)]
        launch.arg("--windows-job").arg(&owner.name);
        launch
            .arg("--")
            .arg(action)
            .args(args)
            .env("GPUWM_CONFIGURATION_RECOVERY_DIR", dir.join("configuration-recovery"))
            .current_dir(&cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            launch.process_group(0);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            launch.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
        }
        let mut child = launch.spawn()?;
        #[cfg(windows)]
        if let Err(error) = owner.assign(&child) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        // The UI's ordinary poll releases the ready worker. Waiting here froze
        // input for up to 30 seconds, including during automatic node log polls.
        Ok(Self {
            dir,
            action: action.into(),
            command,
            started: Instant::now(),
            outcome: None,
            memory_refused: false,
            child,
            stopping: false,
            stop_started: None,
            force_stopped: false,
            startup_released: false,
            completion_notice: None,
            #[cfg(windows)]
            owner,
        })
    }

    pub fn poll(&mut self) -> io::Result<Option<i32>> {
        self.poll_inner(true)
    }

    fn poll_inner(&mut self, allow_start: bool) -> io::Result<Option<i32>> {
        if self.outcome.is_some() {
            return Ok(self.outcome);
        }
        if !self.startup_released && !self.stopping {
            if self.started.elapsed() >= std::time::Duration::from_secs(30) {
                self.completion_notice = Some("Python did not become ready within 30 seconds. The command was not released; check the configured Python executable and open its log.".into());
                self.force_stop()?;
                self.startup_released = true;
            } else if allow_start && self.dir.join("ready").is_file() {
                let release = OpenOptions::new().write(true).create_new(true)
                    .open(self.dir.join("start")).and_then(|marker| marker.sync_all());
                if let Err(error) = release {
                    self.completion_notice = Some(format!("Could not start the ready command: {error}. Open its log for details."));
                    self.force_stop()?;
                }
                self.startup_released = true;
            }
        }
        if !self.force_stopped && self.stop_started.is_some_and(|at|
            at.elapsed() >= std::time::Duration::from_secs(5)) {
            self.force_stop()?;
        }
        let Some(status) = self.child.try_wait()? else {
            return Ok(None);
        };
        #[cfg(unix)]
        let code = {
            use std::os::unix::process::ExitStatusExt;
            status
                .code()
                .unwrap_or_else(|| -status.signal().unwrap_or(1))
        };
        #[cfg(not(unix))]
        let code = status.code().unwrap_or(1);
        let receipt = fs::read(self.dir.join("result.json"))
            .map_err(|error| format!("worker completion receipt is missing or unreadable: {error}"))
            .and_then(|bytes| {
                serde_json::from_slice::<serde_json::Value>(&bytes)
                    .map_err(|error| format!("worker completion receipt is malformed: {error}"))
            });
        let expected_status = if code == 0 {
            "completed"
        } else if code == 130 {
            "interrupted"
        } else {
            "failed"
        };
        let problem = match receipt {
            Err(error) => Some(error),
            Ok(value) => {
                if value["schema"] != "gpuwm-tui-result-v1"
                    || value["exit_code"].as_i64() != Some(i64::from(code))
                    || value["status"] != expected_status
                    || value["cli_args"] != serde_json::json!(&self.command[3..])
                {
                    Some(
                        "worker completion receipt does not match this command and OS exit status"
                            .to_string(),
                    )
                } else {
                    None
                }
            }
        };
        let outcome = if problem.is_some() && code == 0 {
            1
        } else {
            code
        };
        self.outcome = Some(outcome);
        self.memory_refused = problem.is_none()
            && !self.interrupted()
            && memory_refusal(&self.action, outcome, &self.log_tail(1000));
        if let Some(problem) = problem {
            let mut message = if self.stopping {
                format!("Stopped owned process tree (OS exit {code}). {problem}. Partial outputs remain; only an already durable checkpoint can be resumed.")
            } else {
                format!("The command exited with code {code}, but ArWen could not verify its completion. Open the log for details before using its output. Completion check: {problem}.")
            };
            if let Some(prior) = self.completion_notice.take() { message = format!("{prior} {message}"); }
            self.completion_notice = Some(message.clone());
            // Once the process has exited, a diagnostic-file failure must not
            // strand the UI before it receives that terminal result.
            let persist = || -> io::Result<()> {
            let mut log = OpenOptions::new()
                .append(true)
                .open(self.dir.join("job.log"))?;
            writeln!(log, "\n[TUI launcher] {message}")?;
            log.sync_all()?;
            let diagnostic = serde_json::json!({
                "schema": "gpuwm-tui-launcher-result-v1", "exit_code": outcome,
                "os_exit_code": code, "status": if self.stopping { "stopped" } else { "failed" },
                "worker_receipt_valid": false, "message": message,
            });
            let temporary = self.dir.join("launcher-result.tmp");
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)?;
            serde_json::to_writer_pretty(&mut file, &diagnostic)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, self.dir.join("launcher-result.json"))?;
            // Preserve malformed/mismatched worker evidence. Forced termination
            // cannot run its finally block, so only a missing result is filled.
            if !self.dir.join("result.json").exists() {
                fs::copy(
                    self.dir.join("launcher-result.json"),
                    self.dir.join("result.json"),
                )?;
            }
            Ok(())
            };
            if let Err(error) = persist() {
                self.completion_notice = Some(format!("{message} Could not save the completion details: {error}."));
            }
        }
        Ok(self.outcome)
    }

    pub fn stop(&mut self) -> io::Result<()> {
        // Stopping a pending worker must never release its CLI as a side effect.
        if self.poll_inner(false)?.is_some() {
            return Ok(());
        }
        if self.stopping {
            return self.force_stop();
        }
        #[cfg(unix)]
        {
            // The child is still owned and unreaped: its process-group id
            // cannot refer to a later unrelated job through PID reuse.
            let result = unsafe { libc::kill(-(self.child.id() as i32), libc::SIGINT) };
            if result != 0 {
                let error = io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ESRCH) {
                    return Err(error);
                }
            }
        }
        #[cfg(windows)]
        {
            self.owner.terminate(130)?;
            self.force_stopped = true;
        }
        self.stopping = true;
        self.stop_started = Some(Instant::now());
        Ok(())
    }

    fn force_stop(&mut self) -> io::Result<()> {
        #[cfg(unix)]
        {
            // The child remains owned and unreaped until poll observes exit.
            // A retry must signal the same group, never silently claim success.
            if unsafe { libc::kill(-(self.child.id() as i32), libc::SIGKILL) } != 0 {
                let error = io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ESRCH) { return Err(error); }
            }
        }
        #[cfg(windows)]
        self.owner.terminate(130)?;
        self.force_stopped = true;
        Ok(())
    }

    pub fn is_stopping(&self) -> bool { self.stopping && self.outcome.is_none() }

    pub fn stop_message(&self) -> &'static str {
        if self.outcome.is_some() { "This command has finished." }
        else if self.force_stopped { "Force stop requested for this run; waiting for it to exit. Partial output is kept." }
        else if self.stopping { "Stop requested. Remaining work will be force-stopped after five seconds. Saved checkpoints and partial output are kept." }
        else { "No stop has been requested." }
    }

    /// Last lines of the combined raw log; bounded even for a huge forecast log.
    pub fn log_tail(&self, limit: usize) -> String {
        read_log_tail(&self.dir.join("job.log"), limit)
    }

    pub fn interrupted(&self) -> bool {
        self.outcome
            .is_some_and(|code| code != 0 && (self.stopping || code == 130))
    }
    pub fn configuration_recovery(&self) -> Option<PathBuf> {
        if !self.memory_refused {
            return None;
        }
        configuration_recovery_path(&self.dir)
    }
}

fn configuration_recovery_path(directory: &Path) -> Option<PathBuf> {
    let folder = directory.join("configuration-recovery");
    let receipt: serde_json::Value = serde_json::from_reader(
        File::open(folder.join("recovery.json")).ok()?.take(64 * 1024)).ok()?;
    if receipt["schema"] != "arwen.configuration-recovery.v1" || receipt["status"] != "memory-refused" {
        return None;
    }
    let path = folder.join("draft.toml");
    if !fs::symlink_metadata(&path).ok()?.file_type().is_file() { return None; }
    let path = path.canonicalize().ok()?;
    let owned = directory.canonicalize().ok()?.join("configuration-recovery");
    path.starts_with(owned).then_some(path)
}

/// Recognize the existing CLI admission boundary, not an arbitrary occurrence
/// of "memory" or a successful memory estimate preceding an unrelated failure.
pub fn memory_refusal(action: &str, code: i32, log: &str) -> bool {
    if code != 0 && log.lines().any(|line| {
        serde_json::from_str::<serde_json::Value>(line).ok().is_some_and(|value| {
            value["schema"] == "arwen.configuration-error.v1" && value["kind"] == "memory"
                && value["created"] == false && value["error"].is_string()
        })
    }) { return true; }
    match (action, code) {
        ("go", 2) => {
            log.lines().any(|line| {
                let line = line.trim_start();
                line.starts_with("go: memory -- ")
                    && line.contains("EXCEEDS")
                    && line.contains("budget")
            }) && log.lines().any(|line| {
                line.trim_start()
                    .starts_with("gpuwm go: this configuration will not fit:")
            })
        }
        ("check", 1 | 4) => {
            let envelope = log
                .lines()
                .any(|line| line.trim_start().starts_with("BINDING PHASE:"))
                && log.lines().any(|line| {
                    let line = line.trim_start();
                    line.starts_with("WARNING: observed peak envelope ")
                        && line.contains(" exceeds the ")
                        && line.contains("budget")
                });
            if code == 4 {
                return envelope;
            }
            // Check returns 1 when an allocation-budget gate fails too. Do
            // not widen this to arbitrary Check failures: only these named
            // memory verdicts, plus their exact summary, may report FAIL.
            let gates = [
                "alloc_fits_wddm_budget",
                "alloc_fits_vram_budget",
                "alloc_measured_le_estimate",
                "alloc_estimate_le_wddm_budget",
                "alloc_estimate_le_vram_budget",
            ];
            let heading = log
                .lines()
                .any(|line| line.starts_with("gpuwm memory preflight: FAIL ("));
            let allocation = log.lines().any(|line| {
                gates
                    .iter()
                    .any(|name| line.trim() == format!("{name}: FAIL"))
            });
            let only_memory_failures = log.lines().all(|line| {
                let line = line.trim();
                let upper = line.to_ascii_uppercase();
                if !upper.contains("FAIL") && !upper.contains("ERROR") && !upper.contains("TRACEBACK") { return true; }
                line.starts_with("gpuwm memory preflight: FAIL (")
                    || gates.iter().any(|name| line == format!("{name}: FAIL"))
                    || (line.starts_with("WARNING: observed peak envelope ")
                        && line.contains("exceeds the ") && line.contains("budget")
                        && line.contains("exit code 1: a gate above FAILED as well, and the harder verdict wins."))
            });
            envelope && heading && allocation && only_memory_failures
        }
        // Host forcing exhaustion (check exit 5) needs different remedies;
        // putting forecast tiles in host RAM does not fix source decode RAM.
        _ => false,
    }
}

/// Startup failures can have a log even when no owned Job was created.
pub fn read_log_tail(path: &Path, limit: usize) -> String {
    if limit == 0 {
        return String::new();
    }
    let read = || -> io::Result<String> {
        let mut file = File::open(path)?;
        let length = file.metadata()?.len();
        let start = length.saturating_sub(128 * 1024);
        file.seek(SeekFrom::Start(start))?;
        let mut bytes = Vec::new();
        file.take(128 * 1024).read_to_end(&mut bytes)?;
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<_> = text.lines().collect();
        Ok(lines[lines.len().saturating_sub(limit)..].join("\n"))
    };
    read().unwrap_or_else(|error| format!("Could not read job log: {error}"))
}

#[cfg(windows)]
mod windows {
    use super::*;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, TerminateJobObject,
    };

    pub struct Owner {
        handle: HANDLE,
        pub name: String,
    }
    impl Owner {
        pub fn new() -> io::Result<Self> {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(io::Error::other)?
                .as_nanos();
            let name = format!("Local\\ArwenTui-{}-{nonce}", std::process::id());
            let wide: Vec<u16> = name.encode_utf16().chain(Some(0)).collect();
            let handle = unsafe { CreateJobObjectW(std::ptr::null(), wide.as_ptr()) };
            if handle.is_null() {
                Err(io::Error::last_os_error())
            } else {
                Ok(Self { handle, name })
            }
        }
        pub fn assign(&self, child: &Child) -> io::Result<()> {
            if unsafe { AssignProcessToJobObject(self.handle, child.as_raw_handle() as HANDLE) }
                == 0
            {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        }
        pub fn terminate(&self, code: u32) -> io::Result<()> {
            if unsafe { TerminateJobObject(self.handle, code) } == 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        }
    }
    impl Drop for Owner {
        fn drop(&mut self) {
            // No KILL_ON_JOB_CLOSE: closing the TUI deliberately detaches.
            unsafe {
                CloseHandle(self.handle);
            }
        }
    }
}
