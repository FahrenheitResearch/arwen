//! Private local credential IPC. Keys never enter job commands or log files.
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
    time::{Duration, Instant},
};

pub const DEFAULT_URL: &str = "https://cds.climate.copernicus.eu/api";

pub struct Status {
    pub configured: bool,
    pub path: String,
    pub source: String,
    pub url: String,
    pub editable: bool,
}

impl Status {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|_| "CDS credential status could not be read.")?;
        if value["schema"].as_str() != Some("arwen.cds-credentials.v1") {
            return Err("This Python installation does not support the CDS key panel.".into());
        }
        let string = |name: &str, limit| value[name].as_str()
            .filter(|s| s.len() <= limit && !s.chars().any(char::is_control))
            .map(str::to_owned).ok_or_else(|| "Invalid CDS credential status.".to_owned());
        let source = string("source", 32)?;
        if !matches!(source.as_str(), "file" | "environment" | "missing") {
            return Err("Invalid CDS credential source.".into());
        }
        Ok(Self {
            configured: value["configured"].as_bool().ok_or("Invalid CDS credential status.")?,
            editable: value["editable"].as_bool().ok_or("Invalid CDS credential status.")?,
            path: string("path", 8192)?, source, url: string("url", 2048)?,
        })
    }
}

#[derive(Default)]
pub struct Client {
    pub status: Option<Status>,
    pub notice: String,
    pending: Option<Receiver<Result<Status, String>>>,
    context: Option<(PathBuf, PathBuf)>,
    saving: bool,
}

impl Client {
    pub fn busy(&self) -> bool { self.pending.is_some() }
    pub fn summary(&self) -> &'static str {
        match &self.status {
            Some(status) if status.configured => "Configured",
            Some(_) => "Not configured",
            None if self.busy() => "Loading",
            None => "Unavailable",
        }
    }
    pub fn ensure(&mut self, python: &Path, cwd: &Path) {
        if !self.busy() && self.context.as_ref() != Some(&(python.to_owned(), cwd.to_owned())) {
            self.status = None;
            self.refresh(python, cwd);
        }
    }
    pub fn refresh(&mut self, python: &Path, cwd: &Path) {
        if !self.busy() { self.start(python, cwd, None); }
    }
    pub fn save(&mut self, python: &Path, cwd: &Path, key: String) {
        if self.busy() { return; }
        let Some(status) = &self.status else { return; };
        if !status.editable {
            self.notice = "CDS is set by environment variables. Change those variables and reopen ArWen.".into();
            return;
        }
        let url = if status.url.is_empty() { DEFAULT_URL } else { &status.url }.to_owned();
        self.start(python, cwd, Some((url, key)));
    }
    fn start(&mut self, python: &Path, cwd: &Path, save: Option<(String, String)>) {
        self.saving = save.is_some();
        self.notice = if self.saving { "Saving CDS key…" } else { "Reading local CDS settings…" }.into();
        self.context = Some((python.to_owned(), cwd.to_owned()));
        let python = python.to_owned();
        let cwd = cwd.to_owned();
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || { let _ = sender.send(request(&python, &cwd, save)); });
        self.pending = Some(receiver);
    }
    pub fn poll(&mut self) {
        let Some(receiver) = self.pending.take() else { return; };
        match receiver.try_recv() {
            Ok(Ok(status)) => {
                self.status = Some(status);
                self.notice = if self.saving { "CDS key saved." } else { "Local settings loaded." }.into();
            }
            Ok(Err(error)) => self.notice = error,
            Err(TryRecvError::Empty) => self.pending = Some(receiver),
            Err(TryRecvError::Disconnected) => self.notice = "CDS credential operation did not finish.".into(),
        }
    }
}

pub struct Form {
    key: String,
    pub editing: bool,
}

impl Form {
    pub fn new() -> Self { Self { key: String::new(), editing: false } }
    pub fn masked(&self) -> String {
        if self.key.is_empty() { "(blank keeps the saved key)".into() }
        else { "•".repeat(self.key.chars().count().min(32)) }
    }
    pub fn append(&mut self, value: &str) {
        if self.editing {
            for character in value.chars().filter(|c| !c.is_control()) {
                if self.key.len() + character.len_utf8() > 4096 { break; }
                self.key.push(character);
            }
        }
    }
    pub fn backspace(&mut self) { self.key.pop(); }
    pub fn clear(&mut self) { self.key.clear(); }
    pub fn take_key(&mut self) -> String { std::mem::take(&mut self.key) }
}

fn request(python: &Path, cwd: &Path, save: Option<(String, String)>) -> Result<Status, String> {
    let saving = save.is_some();
    let mut command = Command::new(python);
    command.args(["-m", "gpuwm", "cds-credentials", "--json"])
        .current_dir(cwd).stdout(Stdio::piped()).stderr(Stdio::null());
    if saving { command.arg("--save").stdin(Stdio::piped()); }
    else { command.stdin(Stdio::null()); }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = command.spawn().map_err(|_| "Could not open the CDS key service. Check the Python setting (F9).")?;
    if let Some((url, key)) = save {
        let mut stdin = child.stdin.take().ok_or("CDS key input is unavailable.")?;
        thread::spawn(move || {
            if let Ok(mut input) = serde_json::to_vec(&serde_json::json!({"url": url, "key": key})) {
                let _ = stdin.write_all(&input);
                input.fill(0);
            }
        });
    }
    let stdout = child.stdout.take().ok_or("CDS credential output is unavailable.")?;
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let mut output = Vec::new();
        let result = stdout.take(65_537).read_to_end(&mut output).map(|_| output);
        let _ = sender.send(result);
    });
    let deadline = Instant::now() + Duration::from_secs(20);
    let outcome = loop {
        match child.try_wait() {
            Ok(Some(outcome)) => break outcome,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(40)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("CDS credential operation timed out. Check the displayed file location.".into());
            }
        }
    };
    if !outcome.success() {
        return Err(if saving {
            "Could not save the CDS key. Check the key, file access and environment overrides."
        } else { "Could not read CDS settings. Check the Python setting (F9) and credential location." }.into());
    }
    let output = receiver.recv_timeout(Duration::from_secs(1))
        .map_err(|_| "CDS credential status did not arrive.")?
        .map_err(|_| "CDS credential status could not be read.")?;
    if output.len() > 65_536 { return Err("CDS credential status is too large.".into()); }
    Status::parse(&output)
}
