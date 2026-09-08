//! Catalog documents remain data; Python owns schema, source and native admission.
use super::{absolute, button_bar, display_path, key_hit, safe, Hit, HitRegion, Request};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::Style,
    widgets::{List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};
use serde_json::Value;
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver, TryRecvError},
    thread::JoinHandle,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Path,
    Files,
    List,
    Detail,
    Preview,
}
#[derive(Clone, Copy)]
enum Operation {
    List,
    Detail,
    Preview,
}
struct Query {
    id: u64,
    args: Vec<String>,
    sent: bool,
    started: Instant,
    operation: Operation,
}
struct Worker {
    child: Child,
    input: ChildStdin,
    responses: Receiver<Result<Value, String>>,
    err: Option<JoinHandle<Vec<u8>>>,
    active: Option<u64>,
    failed: bool,
}
fn drain(mut stream: impl Read + Send + 'static) -> JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut output = Vec::new();
        let mut block = [0; 8192];
        while let Ok(n) = stream.read(&mut block) {
            if n == 0 {
                break;
            }
            if output.len() < 16 * 1024 * 1024 {
                output.extend_from_slice(&block[..n]);
            }
        }
        output
    })
}
impl Worker {
    fn spawn(python: &Path, cwd: &Path) -> Result<Self, String> {
        let mut command = Command::new(python);
        command
            .args(["-X", "utf8", "-m", "gpuwm.case_catalog", "--tui-server"])
            .current_dir(cwd)
            .env("GPUWM_NO_LOCAL_GPU", "1")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command
            .spawn()
            .map_err(|e| format!("Could not read catalog: {e}"))?;
        let input = child.stdin.take().unwrap();
        let output = child.stdout.take().unwrap();
        let (sender, responses) = mpsc::channel();
        std::thread::spawn(move || {
            let mut output = BufReader::new(output);
            loop {
                let mut line = Vec::new();
                let result = match (&mut output).take(16 * 1024 * 1024 + 1).read_until(b'\n', &mut line) {
                    Ok(0) => break,
                    Ok(n) if n > 16 * 1024 * 1024 || !line.ends_with(b"\n") => {
                        let _ = sender.send(Err("Catalog worker response was incomplete or too large.".into()));
                        break;
                    }
                    Ok(_) => serde_json::from_slice::<Value>(&line)
                        .map_err(|e| format!("Catalog returned invalid JSON: {e}")),
                    Err(error) => Err(format!("Could not read catalog response: {error}")),
                };
                if sender.send(result).is_err() {
                    break;
                }
            }
        });
        let err = Some(drain(child.stderr.take().unwrap()));
        Ok(Self {
            child,
            input,
            responses,
            err,
            active: None,
            failed: false,
        })
    }
    fn send(&mut self, query: &mut Query) -> Result<(), String> {
        let mut payload = serde_json::to_vec(&query.args).map_err(|e| e.to_string())?;
        payload.push(b'\n');
        if payload.len() > 64 * 1024 {
            return Err("Catalog request is too long; shorten the search or path.".into());
        }
        self.input.write_all(&payload).and_then(|_| self.input.flush())
            .map_err(|e| format!("Could not query catalog: {e}"))?;
        self.active = Some(query.id);
        query.sent = true;
        Ok(())
    }
    fn poll(&mut self) -> Option<(u64, Result<Value, String>)> {
        let id = self.active?;
        let result = match self.responses.try_recv() {
            Ok(Ok(value)) if value["schema"] == "arwen.case-error.v1" => {
                Err(string(&value, "error"))
            }
            Ok(Err(error)) => { self.failed = true; Err(error) },
            Ok(result) => result,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => {
                self.failed = true;
                Err("Catalog worker stopped. Retry to reopen the catalog.".into())
            }
        };
        self.active = None;
        Some((id, result))
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        if let Some(err) = self.err.take() {
            let _ = err.join();
        }
    }
}

pub enum Intent {
    Keep,
    Close,
    Review(Request),
}
pub struct Form {
    python: PathBuf,
    cwd: PathBuf,
    page: Page,
    query: Option<Query>,
    worker: Option<Worker>,
    next_query: u64,
    search_due: Option<Instant>,
    pub notice: String,
    catalog: String,
    search: String,
    rows: Vec<Value>,
    selected: usize,
    offset: usize,
    total: usize,
    detail: Value,
    preview: Value,
    tier: usize,
    source: Option<usize>,
    field: usize,
    out: String,
    scroll: u16,
    directory: PathBuf,
    files: Vec<PathBuf>,
    file_selected: usize,
}
const TIERS: [&str; 3] = ["lower", "recommended", "upper"];
fn string(value: &Value, key: &str) -> String {
    value[key].as_str().unwrap_or("").to_owned()
}
fn pretty(value: &Value) -> String {
    fn label(text: &str) -> String {
        let mut result = text.replace('_', " ");
        if let Some(first) = result.get_mut(..1) {
            first.make_ascii_uppercase();
        }
        result
    }
    fn scalar(value: &Value) -> String {
        match value {
            Value::String(s) => s.clone(),
            Value::Null => "None specified".into(),
            _ => value.to_string(),
        }
    }
    fn lines(value: &Value, depth: usize) -> String {
        let indent = "  ".repeat(depth.min(6));
        match value {
            Value::Object(rows) => rows
                .iter()
                .map(|(key, value)| {
                    if value.is_object()
                        || value
                            .as_array()
                            .is_some_and(|rows| rows.iter().any(|v| v.is_object() || v.is_array()))
                    {
                        format!("{indent}{}\n{}", label(key), lines(value, depth + 1))
                    } else {
                        format!("{indent}{}: {}", label(key), lines(value, 0))
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
            Value::Array(rows) => {
                if rows.is_empty() {
                    "None listed".into()
                } else if rows.iter().all(|v| !v.is_array() && !v.is_object()) {
                    rows.iter().map(scalar).collect::<Vec<_>>().join("; ")
                } else {
                    rows.iter()
                        .map(|v| lines(v, depth))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                }
            }
            _ => scalar(value),
        }
    }
    lines(value, 0)
}
impl Form {
    pub fn new(python: &Path, cwd: &Path) -> Self {
        Self::with_catalog(
            python,
            cwd,
            std::env::var("GPUWM_TUI_CASE_CATALOG").unwrap_or_default(),
        )
    }
    pub fn from_catalog(python: &Path, cwd: &Path, path: &Path) -> Self {
        Self::with_catalog(python, cwd, display_path(path))
    }
    fn with_catalog(python: &Path, cwd: &Path, catalog: String) -> Self {
        let mut form = Self {
            python: python.into(),
            cwd: cwd.into(),
            page: Page::List,
            query: None,
            worker: None,
            next_query: 0,
            search_due: None,
            notice: String::new(),
            catalog,
            search: String::new(),
            rows: vec![],
            selected: 0,
            offset: 0,
            total: 0,
            detail: Value::Null,
            preview: Value::Null,
            tier: 1,
            source: None,
            field: 0,
            out: String::new(),
            scroll: 0,
            directory: cwd.into(),
            files: vec![],
            file_selected: 0,
        };
        if form.catalog.trim().is_empty() {
            form.search();
        } else {
            form.open();
        }
        form
    }
    fn start(&mut self, operation: Operation, mut args: Vec<String>) {
        self.query = None;
        if !self.catalog.is_empty() {
            args.extend(["--catalog".into(), self.catalog.clone()]);
        }
        args.push("--json".into());
        if self.worker.is_none() {
            match Worker::spawn(&self.python, &self.cwd) {
                Ok(worker) => self.worker = Some(worker),
                Err(error) => { self.notice = error; return; }
            }
        }
        self.next_query += 1;
        self.query = Some(Query { id: self.next_query, args, sent: false,
            started: Instant::now(), operation });
        self.notice = match operation {
            Operation::List => "Loading cases... Esc cancels.",
            Operation::Detail => "Opening case... Esc cancels.",
            Operation::Preview => "Checking selection... Esc cancels.",
        }.into();
    }
    fn search(&mut self) {
        self.search_due = None;
        self.start(
            Operation::List,
            vec![
                "list".into(),
                "--query".into(),
                self.search.clone(),
                "--offset".into(),
                self.offset.to_string(),
                "--limit".into(),
                "100".into(),
            ],
        );
    }
    fn schedule_search(&mut self) {
        self.query = None;
        self.offset = 0;
        self.search_due = Some(Instant::now() + Duration::from_millis(180));
        self.notice = "Updating search...".into();
    }
    fn open(&mut self) {
        let path = absolute(
            PathBuf::from(super::unquote(self.catalog.trim())),
            &self.cwd,
        );
        if !path.is_file() {
            self.page = Page::Path;
            self.notice = "Choose an existing ZIP, JSON or TOML catalog file.".into();
            return;
        }
        self.catalog = display_path(&path);
        self.offset = 0;
        self.search();
    }
    fn options(&self) -> &[Value] {
        self.detail["case"]["source_options"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
    fn selection_args(&self, command: &str) -> Result<Vec<String>, String> {
        let option = self
            .source
            .and_then(|i| self.options().get(i))
            .ok_or("Choose a source and initialization with Left/Right before previewing.")?;
        Ok(vec![
            command.into(),
            string(&self.detail["case"], "id"),
            "--tier".into(),
            TIERS[self.tier].into(),
            "--source-option".into(),
            string(option, "id"),
        ])
    }
    fn prepare_preview(&mut self) {
        match self.selection_args("preview") {
            Ok(args) => {
                self.scroll = 0;
                self.start(Operation::Preview, args)
            }
            Err(error) => self.notice = error,
        }
    }
    pub fn poll(&mut self) {
        if self.search_due.is_some_and(|due| Instant::now() >= due) {
            self.search();
        }
        let response = self.worker.as_mut().and_then(Worker::poll);
        if self.worker.as_ref().is_some_and(|worker| worker.failed) {
            self.worker = None;
            self.query = None;
            self.notice = response.and_then(|(_, result)| result.err())
                .unwrap_or_else(|| "Catalog worker stopped. Retry to reopen the catalog.".into());
            return;
        }
        let Some(query) = &mut self.query else { return };
        if query.started.elapsed() >= Duration::from_secs(60) {
            self.query = None;
            self.worker = None;
            self.notice = "Catalog query timed out. Your selections are preserved; retry or choose another catalog.".into();
            return;
        }
        if !query.sent {
            if let Some(worker) = &mut self.worker {
                if worker.active.is_none() {
                    if let Err(error) = worker.send(query) {
                        self.query = None;
                        self.worker = None;
                        self.notice = error;
                    }
                }
            }
            return;
        }
        let Some((id, result)) = response else { return };
        self.complete(id, result);
    }
    fn complete(&mut self, id: u64, result: Result<Value, String>) {
        let Some(query) = &self.query else { return };
        if id != query.id { return; }
        let operation = query.operation;
        self.query = None;
        match result {
            Err(error) => self.notice = format!("Catalog: {error}"),
            Ok(value) => match operation {
                Operation::List => {
                    if let Some(path) = value["provenance"]["source"].as_str() {
                        self.catalog = path.to_owned();
                    }
                    self.total = value["total"].as_u64().unwrap_or(0) as usize;
                    self.rows = value["cases"].as_array().cloned().unwrap_or_default();
                    self.selected = 0;
                    self.page = Page::List;
                    self.notice=format!("{} matches. Type to search; Enter opens the selected case. PgUp/PgDn changes catalog pages.",self.total);
                }
                Operation::Detail => {
                    self.detail = value;
                    self.page = Page::Detail;
                    self.scroll = 0;
                    self.field = 0;
                    self.tier = 1;
                    let recommendation = string(&self.detail["case"], "recommended_source_option");
                    self.source = if self.options().len() == 1 {
                        Some(0)
                    } else {
                        self.options()
                            .iter()
                            .position(|option| string(option, "id") == recommendation)
                    };
                    let base = format!(
                        "{}-{}",
                        string(&self.detail["case"], "id"),
                        TIERS[self.tier]
                    );
                    let mut path = self.cwd.join(format!("{base}.toml"));
                    let mut suffix = 2;
                    while path.exists() {
                        path = self.cwd.join(format!("{base}-{suffix}.toml"));
                        suffix += 1;
                    }
                    self.out = display_path(&path);
                    self.notice="Tab selects a field; Left/Right changes tier or source. PgUp/PgDn scrolls all case details.".into();
                }
                Operation::Preview => {
                    self.preview = value;
                    self.page = Page::Preview;
                    self.scroll = 0;
                    self.notice="Review the area, source, physics and catalog provenance. Enter reviews configuration creation; no forecast starts.".into();
                }
            },
        }
    }
    fn create_request(&self) -> Result<Request, String> {
        if self.preview.is_null() {
            return Err("Preview this case before creating a configuration.".into());
        }
        let path = absolute(PathBuf::from(super::unquote(self.out.trim())), &self.cwd);
        if self.out.trim().is_empty() || path.extension().is_none_or(|s| s != "toml") {
            return Err("Choose a new .toml configuration path.".into());
        }
        if path.exists() {
            return Err(
                "That configuration path exists. Go Back and choose a new filename.".into(),
            );
        }
        let mut args = self.selection_args("create")?;
        args.extend([
            "--catalog".into(),
            self.catalog.clone(),
            "--out".into(),
            display_path(&path),
            "--expected-catalog-sha256".into(),
            string(&self.preview["provenance"], "original_sha256"),
            "--geometry-only".into(),
            "--json".into(),
        ]);
        Ok(Request {
            command: "case-catalog".into(),
            args,
            created: Some(path),
            title: "Create editable configuration from reviewed case",
        })
    }
    fn browse(&mut self) {
        self.query = None;
        self.search_due = None;
        let candidate = absolute(
            PathBuf::from(super::unquote(self.catalog.trim())),
            &self.cwd,
        );
        self.directory = if candidate.is_dir() {
            candidate
        } else {
            candidate.parent().unwrap_or(&self.cwd).into()
        };
        self.read_directory();
        self.page = Page::Files;
    }
    fn read_directory(&mut self) {
        self.files = fs::read_dir(&self.directory)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.is_dir()
                    || p.extension().is_some_and(|e| {
                        e.eq_ignore_ascii_case("json")
                            || e.eq_ignore_ascii_case("toml")
                            || e.eq_ignore_ascii_case("zip")
                    })
            })
            .collect();
        self.files.sort_by_key(|p| {
            (
                !p.is_dir(),
                p.file_name().map(|n| n.to_string_lossy().to_lowercase()),
            )
        });
        self.file_selected = 0;
        self.notice="Choose a folder or catalog. Backspace opens the parent folder; Esc returns to the path.".into();
    }
    fn edit(&mut self) -> Option<&mut String> {
        match self.page {
            Page::Path => Some(&mut self.catalog),
            Page::List => Some(&mut self.search),
            Page::Detail if self.field == 2 => Some(&mut self.out),
            _ => None,
        }
    }
    pub fn paste(&mut self, text: &str) {
        if let Some(value) = self.edit() {
            value.push_str(text.trim_end_matches(['\r', '\n']));
            if self.page == Page::List {
                self.schedule_search();
            }
        }
    }
    pub fn select(&mut self, index: usize) {
        match self.page {
            Page::Files => self.file_selected = index.min(self.files.len().saturating_sub(1)),
            Page::List => self.selected = index.min(self.rows.len().saturating_sub(1)),
            Page::Detail => self.field = index.min(2),
            _ => {}
        }
    }
    pub fn key(&mut self, key: KeyEvent) -> Intent {
        if key.code == KeyCode::Esc {
            self.query = None;
            self.search_due = None;
            self.page = match self.page {
                Page::Path => return Intent::Close,
                Page::Files | Page::List => Page::Path,
                Page::Detail => Page::List,
                Page::Preview => Page::Detail,
            };
            self.scroll = 0;
            return Intent::Keep;
        }
        if key.code == KeyCode::F(4) {
            self.catalog.clear();
            self.search.clear();
            self.offset = 0;
            self.page = Page::List;
            self.search();
            return Intent::Keep;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('u') {
            if let Some(value) = self.edit() {
                value.clear();
                if self.page == Page::List {
                    self.schedule_search();
                }
            }
            return Intent::Keep;
        }
        if matches!(key.code, KeyCode::Char(_) | KeyCode::Backspace)
            && !key.modifiers.contains(KeyModifiers::CONTROL)
        {
            if let Some(value) = self.edit() {
                match key.code {
                    KeyCode::Char(c) => value.push(c),
                    KeyCode::Backspace => {
                        value.pop();
                    }
                    _ => {}
                }
                if self.page == Page::List {
                    self.schedule_search();
                }
                return Intent::Keep;
            }
        }
        match self.page {
            Page::Path => match key.code {
                KeyCode::Enter if self.query.is_none() => self.open(),
                KeyCode::F(2) => self.browse(),
                _ => {}
            },
            Page::Files => match key.code {
                KeyCode::Up => self.file_selected = self.file_selected.saturating_sub(1),
                KeyCode::Down => {
                    self.file_selected =
                        (self.file_selected + 1).min(self.files.len().saturating_sub(1))
                }
                KeyCode::Backspace => {
                    if let Some(parent) = self.directory.parent() {
                        self.directory = parent.into();
                        self.read_directory();
                    }
                }
                KeyCode::Enter => {
                    if let Some(path) = self.files.get(self.file_selected).cloned() {
                        if path.is_dir() {
                            self.directory = path;
                            self.read_directory()
                        } else {
                            self.catalog = display_path(&path);
                            self.open()
                        }
                    }
                }
                _ => {}
            },
            Page::List => match key.code {
                KeyCode::F(2) => self.browse(),
                KeyCode::Up => self.selected = self.selected.saturating_sub(1),
                KeyCode::Down => {
                    self.selected = (self.selected + 1).min(self.rows.len().saturating_sub(1))
                }
                KeyCode::PageDown if self.offset + 100 < self.total => {
                    self.offset += 100;
                    self.search()
                }
                KeyCode::PageUp => {
                    self.offset = self.offset.saturating_sub(100);
                    self.search()
                }
                KeyCode::Enter if self.query.is_none() && self.search_due.is_none() => {
                    if let Some(row) = self.rows.get(self.selected) {
                        self.start(Operation::Detail, vec!["show".into(), string(row, "id")]);
                    }
                }
                _ => {}
            },
            Page::Detail => match key.code {
                KeyCode::Tab | KeyCode::Enter => self.field = (self.field + 1) % 3,
                KeyCode::BackTab => self.field = (self.field + 2) % 3,
                KeyCode::Left | KeyCode::Right if self.field < 2 => {
                    let right = key.code == KeyCode::Right;
                    if self.field == 0 {
                        self.tier = (self.tier + if right { 1 } else { 2 }) % 3;
                    } else {
                        let count = self.options().len();
                        if count > 0 {
                            self.source = Some(
                                self.source
                                    .map(|n| (n + if right { 1 } else { count - 1 }) % count)
                                    .unwrap_or(0),
                            );
                        }
                    }
                    self.preview = Value::Null;
                }
                KeyCode::F(3) if self.query.is_none() => self.prepare_preview(),
                KeyCode::Up => self.scroll = self.scroll.saturating_sub(1),
                KeyCode::Down => self.scroll = self.scroll.saturating_add(1),
                KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(8),
                KeyCode::PageDown => self.scroll = self.scroll.saturating_add(8),
                _ => {}
            },
            Page::Preview => match key.code {
                KeyCode::Enter if self.query.is_none() => match self.create_request() {
                    Ok(request) => return Intent::Review(request),
                    Err(error) => self.notice = error,
                },
                KeyCode::BackTab => {
                    self.page = Page::Detail;
                    self.scroll = 0;
                }
                KeyCode::Up => self.scroll = self.scroll.saturating_sub(1),
                KeyCode::Down => self.scroll = self.scroll.saturating_add(1),
                KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(8),
                KeyCode::PageDown => self.scroll = self.scroll.saturating_add(8),
                KeyCode::Home => self.scroll = 0,
                KeyCode::End => self.scroll = u16::MAX,
                _ => {}
            },
        }
        Intent::Keep
    }
    fn content(&self) -> String {
        let case = &self.detail["case"];
        if self.page == Page::Preview {
            format!("{}{}\nTier: {}\nSource: {}  Cycle UTC: {}\nOutput: {}\nGPU memory: checked for the selected target at Review / Run.\n\nGeometry\n{}\n\nPhysics profile\n{}\nNative overrides\n{}\n\nSource guidance\n{}\n\nCatalog recommendations\n{}\n\nCatalog provenance\n{}",
                if self.preview["synthetic"].as_bool()==Some(true){"SYNTHETIC EXAMPLE — "}else{""},string(&self.preview,"title"),string(&self.preview,"tier"),string(&self.preview,"source"),string(&self.preview,"cycle"),self.out,
                pretty(&self.preview["geometry"]),string(&self.preview,"physics_profile"),pretty(&self.preview["native_overrides"]),pretty(&self.preview["source_availability"]),pretty(&self.preview["recommendations"]),pretty(&self.preview["provenance"]))
        } else {
            format!(
                "{}{}\n{}\n\nAll catalog details (scroll to read)\n{}",
                if case["synthetic"].as_bool() == Some(true) {
                    "SYNTHETIC EXAMPLE — "
                } else {
                    ""
                },
                string(case, "title"),
                string(case, "summary"),
                pretty(case)
            )
        }
    }
    pub fn draw(
        &mut self,
        frame: &mut Frame,
        hits: &mut Vec<HitRegion>,
        body: Rect,
        buttons: Rect,
    ) {
        match self.page {
            Page::Path => {
                frame.render_widget(Paragraph::new(format!("Open another case catalog\n\nCatalog path (ZIP, JSON or TOML)\n{}\n\nThe built-in historical catalog is available with F4. Open another catalog here to use its listed initializations, domain tiers, physics and research notes. Creation opens an editable TOML configuration.\n\nType or paste; Ctrl+U clears the path. Enter opens the file.",safe(&self.catalog))).wrap(Wrap{trim:false}),body);
                button_bar(
                    frame,
                    hits,
                    buttons,
                    &[
                        ("Open catalog", key_hit(KeyCode::Enter)),
                        ("Browse (F2)", key_hit(KeyCode::F(2))),
                        ("Built-in (F4)", key_hit(KeyCode::F(4))),
                        ("Close", key_hit(KeyCode::Esc)),
                    ],
                );
            }
            Page::Files | Page::List => {
                let parts =
                    Layout::vertical([Constraint::Length(2), Constraint::Min(1)]).split(body);
                let is_files = self.page == Page::Files;
                frame.render_widget(
                    Paragraph::new(if is_files {
                        format!("Folder: {}", display_path(&self.directory))
                    } else {
                        format!(
                            "Search: {}\n{}–{} of {} · Enter opens details",
                            safe(&self.search),
                            if self.total == 0 { 0 } else { self.offset + 1 },
                            self.offset + self.rows.len(),
                            self.total
                        )
                    })
                    .wrap(Wrap { trim: false }),
                    parts[0],
                );
                let labels = if is_files {
                    self.files
                        .iter()
                        .map(|p| {
                            format!(
                                "{}{}",
                                if p.is_dir() { "[folder] " } else { "" },
                                p.file_name().unwrap_or_default().to_string_lossy()
                            )
                        })
                        .collect::<Vec<_>>()
                } else {
                    self.rows
                        .iter()
                        .map(|r| {
                            format!(
                                "{}{} · {}",
                                if r["synthetic"].as_bool() == Some(true) {
                                    "[SYNTHETIC] "
                                } else {
                                    ""
                                },
                                string(r, "title"),
                                string(r, "id")
                            )
                        })
                        .collect()
                };
                let mut state = ListState::default().with_selected(Some(if is_files {
                    self.file_selected
                } else {
                    self.selected
                }));
                frame.render_stateful_widget(
                    List::new(
                        labels
                            .iter()
                            .map(|s| ListItem::new(safe(s)))
                            .collect::<Vec<_>>(),
                    )
                    .highlight_symbol("> ")
                    .highlight_style(super::theme::selected()),
                    parts[1],
                    &mut state,
                );
                for index in
                    state.offset()..labels.len().min(state.offset() + parts[1].height as usize)
                {
                    hits.push(HitRegion {
                        area: Rect::new(
                            parts[1].x,
                            parts[1].y + (index - state.offset()) as u16,
                            parts[1].width,
                            1,
                        ),
                        action: Hit::CaseItem(index),
                    });
                }
                button_bar(
                    frame,
                    hits,
                    buttons,
                    &[
                        (if is_files { "Back" } else { "Change catalog" }, key_hit(KeyCode::Esc)),
                        ("Open (Enter)", key_hit(KeyCode::Enter)),
                        (
                            if is_files { "Parent" } else { "Previous page" },
                            key_hit(if is_files {
                                KeyCode::Backspace
                            } else {
                                KeyCode::PageUp
                            }),
                        ),
                        ("Next page", key_hit(KeyCode::PageDown)),
                    ],
                );
            }
            Page::Detail | Page::Preview => {
                let text_area = if self.page == Page::Detail {
                    let parts =
                        Layout::vertical([Constraint::Length(3), Constraint::Min(1)]).split(body);
                    let option = self
                        .source
                        .and_then(|i| self.options().get(i))
                        .map(|o| {
                            format!(
                                "{} · {} · {}",
                                string(o, "id"),
                                string(o, "source"),
                                string(o, "cycle_utc")
                            )
                        })
                        .unwrap_or_else(|| "Choose explicitly with Left/Right".into());
                    let fields = [
                        format!("Tier: {}  (Left/Right)", TIERS[self.tier]),
                        format!("Source / initialization: {option}"),
                        format!(
                            "New TOML: {}",
                            super::guide_value_tail(
                                &self.out,
                                usize::from(parts[0].width.saturating_sub(12))
                            )
                        ),
                    ];
                    for (index, value) in fields.iter().enumerate() {
                        let area =
                            Rect::new(parts[0].x, parts[0].y + index as u16, parts[0].width, 1);
                        let text = if index == self.field {
                            format!("> {value}")
                        } else {
                            format!("  {value}")
                        };
                        frame.render_widget(
                            Paragraph::new(safe(&text)).style(if index == self.field {
                                super::theme::selected()
                            } else {
                                Style::default()
                            }),
                            area,
                        );
                        hits.push(HitRegion {
                            area,
                            action: Hit::CaseItem(index),
                        });
                    }
                    parts[1]
                } else {
                    body
                };
                let content = safe(&self.content());
                let count = Paragraph::new(content.clone())
                    .wrap(Wrap { trim: false })
                    .line_count(text_area.width)
                    .saturating_sub(text_area.height as usize)
                    .min(u16::MAX as usize) as u16;
                self.scroll = self.scroll.min(count);
                frame.render_widget(
                    Paragraph::new(content)
                        .wrap(Wrap { trim: false })
                        .scroll((self.scroll, 0)),
                    text_area,
                );
                button_bar(
                    frame,
                    hits,
                    buttons,
                    &[
                        ("Back", key_hit(KeyCode::Esc)),
                        (
                            if self.page == Page::Detail {
                                "Preview (F3)"
                            } else {
                                "Review creation"
                            },
                            key_hit(if self.page == Page::Detail {
                                KeyCode::F(3)
                            } else {
                                KeyCode::Enter
                            }),
                        ),
                        ("PgUp", key_hit(KeyCode::PageUp)),
                        ("PgDn", key_hit(KeyCode::PageDown)),
                    ],
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn form() -> Form {
        let mut form = Form::new(Path::new("unused"), Path::new("."));
        form.page = Page::Detail;
        form.detail = serde_json::json!({"case":{"id":"example","source_options":[{"id":"first"},{"id":"second"}]}});
        form.out = "case-test-unique.toml".into();
        form
    }
    #[test]
    fn source_requires_explicit_choice_and_review_pins_catalog() {
        let mut form = form();
        assert!(form.selection_args("preview").is_err());
        form.source = Some(1);
        form.preview = serde_json::json!({"provenance":{"original_sha256":"abc123"}});
        let request = form.create_request().unwrap();
        assert_eq!(request.command, "case-catalog");
        assert!(request
            .args
            .windows(2)
            .any(|p| p == ["--source-option", "second"]));
        assert!(request
            .args
            .windows(2)
            .any(|p| p == ["--expected-catalog-sha256", "abc123"]));
        assert!(!request.args.iter().any(|v| v == "go" || v == "sim"));
        assert!(request.args.iter().any(|v| v == "--geometry-only"));
        assert!(!request.args.iter().any(|v| v == "--vram-gib" || v == "--card"));
    }
    #[test]
    fn existing_output_is_refused_before_geometry_creation() {
        let mut form = form();
        form.source = Some(0);
        form.preview = serde_json::json!({});
        form.out = "Cargo.toml".into();
        assert!(form.create_request().is_err());
    }
    #[test]
    fn catalog_text_is_literal_and_navigation_preserves_values() {
        let mut form = form();
        form.source = Some(0);
        form.field = 2;
        form.out.clear();
        form.paste("quoted $(shell) case.toml\r\n");
        assert_eq!(form.out, "quoted $(shell) case.toml");
        form.page = Page::Preview;
        form.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(form.page == Page::Detail);
        assert_eq!(form.source, Some(0));
        assert_eq!(form.out, "quoted $(shell) case.toml");
    }
    #[test]
    fn cancelled_and_superseded_catalog_responses_do_not_change_the_current_page() {
        let mut form = form();
        form.page = Page::List;
        form.query = Some(Query { id: 2, args: vec![], sent: true,
            started: Instant::now(), operation: Operation::List });
        let list = serde_json::json!({"total":1,"cases":[{"id":"new","title":"New search"}]});
        form.complete(1, Ok(serde_json::json!({"case":{"id":"stale"}})));
        assert!(form.page == Page::List);
        assert_eq!(form.query.as_ref().unwrap().id, 2);
        form.complete(2, Ok(list.clone()));
        assert_eq!(form.rows[0]["id"], "new");
        assert!(form.query.is_none());
        form.query = Some(Query { id: 3, args: vec![], sent: true,
            started: Instant::now(), operation: Operation::List });
        form.browse();
        form.complete(3, Ok(list));
        assert!(form.page == Page::Files);
        assert!(form.query.is_none());
    }
    #[test]
    fn every_case_screen_keeps_actions_and_editable_fields_visible() {
        use ratatui::{backend::TestBackend, Terminal};
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            for page in [
                Page::Path,
                Page::Files,
                Page::List,
                Page::Detail,
                Page::Preview,
            ] {
                let mut form = form();
                form.page = page;
                form.out = format!("{}tail-forecast.toml", "long folder/".repeat(30));
                form.rows = vec![serde_json::json!({"id":"a","title":"Case A"})];
                form.files = vec![PathBuf::from("catalog.zip")];
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                let mut hits = vec![];
                terminal
                    .draw(|frame| {
                        form.draw(
                            frame,
                            &mut hits,
                            Rect::new(2, 5, width - 4, height - 11),
                            Rect::new(2, height - 4, width - 4, 2),
                        )
                    })
                    .unwrap();
                assert!(hits
                    .iter()
                    .all(|hit| hit.area.right() <= width && hit.area.bottom() <= height));
                assert!(hits
                    .iter()
                    .any(|hit| matches!(hit.action, Hit::Key(KeyCode::Esc, _))));
                if page == Page::Detail {
                    for index in 0..3 {
                        assert!(hits
                            .iter()
                            .any(|hit| matches!(hit.action, Hit::CaseItem(i) if i==index)));
                    }
                    let screen = terminal
                        .backend()
                        .buffer()
                        .content
                        .iter()
                        .map(|c| c.symbol())
                        .collect::<String>();
                    assert!(screen.contains("tail-forecast.toml"));
                }
            }
        }
    }
}
