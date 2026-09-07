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
    io::Read,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
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
    child: Child,
    out: Option<JoinHandle<Vec<u8>>>,
    err: Option<JoinHandle<Vec<u8>>>,
    started: Instant,
    operation: Operation,
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
impl Query {
    fn spawn(
        python: &Path,
        cwd: &Path,
        args: &[String],
        operation: Operation,
    ) -> Result<Self, String> {
        let mut command = Command::new(python);
        command
            .args(["-X", "utf8", "-m", "gpuwm.cli", "case-catalog"])
            .args(args)
            .current_dir(cwd)
            .env("GPUWM_NO_LOCAL_GPU", "1")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .stdin(Stdio::null())
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
        let out = Some(drain(child.stdout.take().unwrap()));
        let err = Some(drain(child.stderr.take().unwrap()));
        Ok(Self {
            child,
            out,
            err,
            started: Instant::now(),
            operation,
        })
    }
    fn poll(&mut self) -> Option<Result<Value, String>> {
        let status = match self.child.try_wait() {
            Ok(None) if self.started.elapsed() < Duration::from_secs(60) => return None,
            Ok(None) => {
                let _ = self.child.kill();
                let _ = self.child.wait();
                return Some(Err("Catalog query timed out. Your selections are preserved; retry or choose another catalog.".into()));
            }
            Err(e) => return Some(Err(format!("Could not observe catalog query: {e}"))),
            Ok(Some(status)) => status,
        };
        let out = self
            .out
            .take()
            .and_then(|t| t.join().ok())
            .unwrap_or_default();
        let err = self
            .err
            .take()
            .and_then(|t| t.join().ok())
            .unwrap_or_default();
        let parsed = serde_json::from_slice::<Value>(&out);
        Some(if status.success() {
            parsed.map_err(|e| format!("Catalog returned invalid JSON: {e}"))
        } else {
            Err(parsed
                .ok()
                .and_then(|v| v["error"].as_str().map(str::to_owned))
                .unwrap_or_else(|| {
                    safe(&String::from_utf8_lossy(&err))
                        .chars()
                        .take(1800)
                        .collect()
                }))
        })
    }
}
impl Drop for Query {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
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
    vram: String,
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
        let example = cwd.join("gpuwm/data/case-catalog/example.json");
        Self {
            python: python.into(),
            cwd: cwd.into(),
            page: Page::Path,
            query: None,
            search_due: None,
            notice:
                "Open your ZIP, JSON or TOML catalog. The bundled example is explicitly synthetic."
                    .into(),
            catalog: std::env::var("GPUWM_TUI_CASE_CATALOG").unwrap_or_else(|_| {
                if example.is_file() {
                    display_path(&example)
                } else {
                    String::new()
                }
            }),
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
            vram: String::new(),
            scroll: 0,
            directory: cwd.into(),
            files: vec![],
            file_selected: 0,
        }
    }
    fn start(&mut self, operation: Operation, mut args: Vec<String>) {
        self.query = None;
        args.extend(["--catalog".into(), self.catalog.clone(), "--json".into()]);
        match Query::spawn(&self.python, &self.cwd, &args, operation) {
            Ok(query) => {
                self.query = Some(query);
                self.notice =
                    "Reading catalog and validating selected values... Esc cancels.".into()
            }
            Err(error) => self.notice = error,
        }
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
        let Some(query) = &mut self.query else { return };
        let operation = query.operation;
        let Some(result) = query.poll() else { return };
        self.query = None;
        match result {
            Err(error) => self.notice = format!("Catalog: {error}"),
            Ok(value) => match operation {
                Operation::List => {
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
            "--json".into(),
        ]);
        if !self.vram.trim().is_empty() {
            let value =
                self.vram.trim().parse::<f64>().map_err(|_| {
                    "VRAM must be a positive GiB value, or leave it blank to detect."
                })?;
            if !value.is_finite() || value <= 0.0 {
                return Err("VRAM must be a positive, finite GiB value.".into());
            }
            args.extend(["--vram-gib".into(), self.vram.trim().into()]);
        }
        Ok(Request {
            command: "case-catalog".into(),
            args,
            created: Some(path),
            title: "Create editable configuration from reviewed case",
        })
    }
    fn browse(&mut self) {
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
            Page::Detail if self.field == 3 => Some(&mut self.vram),
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
            Page::Detail => self.field = index.min(3),
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
                KeyCode::Enter => self.open(),
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
                KeyCode::Tab | KeyCode::Enter => self.field = (self.field + 1) % 4,
                KeyCode::BackTab => self.field = (self.field + 3) % 4,
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
            format!("{}{}\nTier: {}\nSource: {}  Cycle UTC: {}\nOutput: {}\nVRAM GiB: {}\n\nGeometry\n{}\n\nPhysics profile\n{}\nNative overrides\n{}\n\nSource guidance\n{}\n\nCatalog recommendations\n{}\n\nCatalog provenance\n{}\n\n{}",
                if self.preview["synthetic"].as_bool()==Some(true){"SYNTHETIC EXAMPLE — "}else{""},string(&self.preview,"title"),string(&self.preview,"tier"),string(&self.preview,"source"),string(&self.preview,"cycle"),self.out,
                if self.vram.is_empty(){"detect at creation"}else{&self.vram},pretty(&self.preview["geometry"]),string(&self.preview,"physics_profile"),pretty(&self.preview["native_overrides"]),pretty(&self.preview["source_availability"]),pretty(&self.preview["recommendations"]),pretty(&self.preview["provenance"]),string(&self.preview,"native_admission"))
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
                frame.render_widget(Paragraph::new(format!("Open a case catalog\n\nCatalog path (ZIP, JSON or TOML)\n{}\n\nUse a catalog supplied by its author, or inspect the bundled synthetic format example. Cases keep their listed initializations, domain tiers, physics and research notes. Creation opens an editable TOML configuration.\n\nType or paste; Ctrl+U clears the path.",safe(&self.catalog))).wrap(Wrap{trim:false}),body);
                button_bar(
                    frame,
                    hits,
                    buttons,
                    &[
                        ("Open catalog", key_hit(KeyCode::Enter)),
                        ("Browse (F2)", key_hit(KeyCode::F(2))),
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
                        ("Back", key_hit(KeyCode::Esc)),
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
                        Layout::vertical([Constraint::Length(4), Constraint::Min(1)]).split(body);
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
                        format!(
                            "VRAM GiB: {}",
                            if self.vram.is_empty() {
                                "detect at creation"
                            } else {
                                &self.vram
                            }
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
        form.vram = "12".into();
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
    }
    #[test]
    fn invalid_capacity_and_existing_output_are_refused() {
        let mut form = form();
        form.source = Some(0);
        form.preview = serde_json::json!({});
        for value in ["NaN", "-1", "inf", "text"] {
            form.vram = value.into();
            assert!(form.create_request().is_err());
        }
        form.vram = "12".into();
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
                    for index in 0..4 {
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
