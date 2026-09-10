mod clipboard;
mod cds_credentials;
mod companion;
mod domains;
mod calendar;
mod cases;
mod editor;
mod file_drop;
mod file_drop_input;
mod guide;
mod job;
mod node_ui;
mod plotsettings;
mod remote;
mod run_view;
#[cfg(test)]
mod run_view_process_tests;
mod research;
mod research_browser;
mod scenario;
mod theme;
mod workflows;

use crossterm::{
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent,
        MouseEventKind,
    },
    execute,
};
use editor::{display_path, Editor};
use guide::{Guide, Kind, Request};
use job::Job;
use ratatui::{
    backend::TestBackend,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};
use std::{
    env, fs, io,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use theme::{AMBER, BACK, INK, MUTED, TEAL};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Tab {
    Home,
    Overview,
    Settings,
    Logs,
}
enum Dialog {
    Workflows(workflows::Browser),
    Scenario(scenario::Form, String),
    Details {
        text: String,
        offset: usize,
        notice: Option<String>,
    },
    Domains(usize),
    Plots(plotsettings::Form),
    Nodes,
    DomainMenu(usize, usize),
    DomainForm(domains::Form, Option<usize>),
    DomainRemoval(domains::Removal, u16),
    Era5Provider(usize, String),
    CdsCredentials(cds_credentials::Form),
    Choice(bool, usize),
    Guide(Guide),
    Calendar(Guide, calendar::Form),
    Cases(cases::Form),
    Summary(Guide, usize),
    Review(Request, Option<Guide>, u16),
    Path(&'static str, String),
    Browser(PathBuf, Vec<PathBuf>, usize),
    DroppedFiles(Vec<PathBuf>, usize),
    Stop,
    Quit,
    Help(HelpScroll),
}

#[derive(Default)]
struct HelpScroll {
    offset: u16,
    max_offset: u16,
    page_rows: u16,
}

const HELP_TEXT: &str = "Click tabs, settings, actions and buttons.\n\
K Cases: browse built-in historical cases, choose source and tier\n\
Drop a TOML or catalog file to open it; several files open a chooser\n\
W Research: choose a weather question and configuration\n\
I Scenario: review initial-state warm bubbles\n\
H Home   V Overview   D/Ctrl+D Domains   F/E Settings   L Logs\n\
B / Ctrl+P Plots: presets, searchable selection and preferences\n\
R / Ctrl+R Nodes: SSH profiles, remote jobs, logs, stop and resume\n\
Click status or Ctrl+L for error details and log actions.\n\
G / Ctrl+G Geography (F11 is often terminal fullscreen)\n\
O Open configuration   Ctrl+O Paste path\n\
Ctrl+S Save   F12 Save As\n\
Ctrl+Z Undo   Ctrl+Y Redo   Ctrl+U Discard draft (in Settings)\n\
Ctrl+Q Quit anywhere; Ctrl+C reviews stopping a local command.\n\
F1 / ? opens help over a form; Esc returns to your answers.\n\
After memory refusal: Ctrl+F Fit domain; Ctrl+T Tile streaming.\n\
In domain forms: F2 applies changes to the draft.\n\
U ERA5 provider: Google ARCO or Copernicus CDS (ERA5 inputs only).\n\
F5 Check   F6 Plan   F7 Review and run\n\
F3 Output folder   F4 Prepared folder   F8 Run prepared\n\
F9 Python   F10 Installation check\n\
In forms: Enter next, Shift+Tab previous, Ctrl+A all settings.\n\
Type or paste a value; Ctrl+U clears it. Long values show the end.\n\
In settings review: Ctrl+N next, Enter edit selected row.\n\
While editing TOML, letters remain text. Click a tab or Esc to leave.\n\
Read-only captures: --snapshot FILE.html --snapshot-screen SCREEN\n\
SCREEN includes modes, mode:ID, research:ID, guide and help.\n\
Scenario capture needs --config. Set --snapshot-width and --snapshot-height.\n\
Help: Up/Down or wheel scroll; PgUp/PgDn page; Home/End jump.\n\
Esc closes help. Editing or opening a configuration starts no command.";
#[derive(Clone, Copy)]
enum Action {
    Check,
    Plan,
    Run,
    Prepared,
    Doctor,
}
const ACTIONS: [(&str, Action); 5] = [
    ("F5  Check configuration and resources", Action::Check),
    ("F6  Review launch plan", Action::Plan),
    ("F7  Prepare and run forecast", Action::Run),
    ("F8  Run existing preparation", Action::Prepared),
    ("F10 Check installation", Action::Doctor),
];

#[derive(Clone, Copy)]
enum Hit {
    Workflow(usize),
    Research(usize),
    Details,
    FitCurrent,
    TileCurrent,
    Domains,
    Plots,
    PlotItem(usize),
    Nodes,
    DomainField(usize),
    DomainSelect(usize),
    Era5Provider,
    CdsCredentials,
    Quit,
    JobResult,
    OpenCompanion,
    DomainValue(usize),
    GuideGrid(usize),
    CalendarDay(u8),
    CalendarHour(u8),
    CaseItem(usize),
    Key(KeyCode, KeyModifiers),
    View(Tab),
    Home(usize),
    Choice(usize),
    Summary(usize),
    Browser(usize),
    Action(usize),
    Editor(Rect),
}
struct HitRegion {
    area: Rect,
    action: Hit,
}

struct StartupFailure {
    message: String,
    log: Option<PathBuf>,
}

struct App {
    editor: Option<Editor>,
    python: PathBuf,
    output: PathBuf,
    prepared: PathBuf,
    geog_root: PathBuf,
    cwd: PathBuf,
    tab: Tab,
    dialog: Option<Dialog>,
    dialog_stack: Vec<Dialog>,
    saved_guides: Vec<Guide>,
    job: Option<Job>,
    startup_failure: Option<StartupFailure>,
    status: String,
    selected: usize,
    log_offset: usize,
    local_raw_logs: bool,
    exit: bool,
    exit_after_job: bool,
    input_enabled: bool,
    // Canonical path consumed by the active job, cleared on observed exit.
    active_config: Option<PathBuf>,
    memory_recovery_config: Option<PathBuf>,
    pending_config: Option<PathBuf>,
    pending_workflow: Option<&'static str>,
    guide_cache: Option<Guide>,
    case_cache: Option<cases::Form>,
    plot_catalog: plotsettings::Catalog,
    cds: cds_credentials::Client,
    companion: companion::Controller,
    companion_remote: Option<CompanionRemoteRequest>,
    companion_waiting: Option<QueuedCompanionRequest>,
    focus_logs_pending: Option<(companion::Request, Instant, String)>,
    run_views: run_view::Manager,
    companion_artifacts: Option<serde_json::Value>,
    tui_map_review: Option<companion::Request>,
    tui_map_waiting: Option<companion::Request>,
    tui_map_launch: Option<companion::Request>,
    discard_setup_review: bool,
    nodes: remote::Controller,
    node_panel: node_ui::Panel,
    plots_from_nodes: bool,
    hits: Vec<HitRegion>,
    #[cfg(test)]
    clipboard_hook: Option<fn(&str) -> Result<(), String>>,
}

struct CompanionRemoteRequest {
    request: companion::Request,
    node: remote::Node,
    source: serde_json::Value,
}

struct QueuedCompanionRequest {
    request: companion::Request,
    session_id: Option<String>,
    queued_at: Instant,
}
const COMPANION_QUEUE_TIMEOUT: Duration = Duration::from_secs(30);

fn focus_console_window(){
    #[cfg(all(windows,not(test)))]
    unsafe {
        #[link(name="kernel32")]
        unsafe extern "system" {fn GetConsoleWindow()->*mut std::ffi::c_void;}
        #[link(name="user32")]
        unsafe extern "system" {fn IsIconic(window:*mut std::ffi::c_void)->i32;fn ShowWindow(window:*mut std::ffi::c_void,command:i32)->i32;fn SetForegroundWindow(window:*mut std::ffi::c_void)->i32;}
        let window=GetConsoleWindow();
        if !window.is_null(){if IsIconic(window)!=0{ShowWindow(window,9);}SetForegroundWindow(window);}
    }
}

fn passive_node_operation(operation: &remote::Operation) -> bool {
    matches!(operation, remote::Operation::Probe | remote::Operation::Logs { .. }
        | remote::Operation::Status { .. } | remote::Operation::ArtifactIndex { .. }
        | remote::Operation::SyncArtifacts { .. } | remote::Operation::SyncProcessedFrameV2 { .. } | remote::Operation::SyncNativePlots { .. })
}
fn passive_companion_action(action: &companion::Action) -> bool {
    // SelectTarget only enters the remote request lane for a refresh of the
    // already selected, SHA-bound SSH target; target changes stay separate.
    matches!(action, companion::Action::ArtifactIndex { .. } | companion::Action::SyncArtifacts { .. }
        | companion::Action::SelectTarget | companion::Action::SyncProcessedFrameV2 { .. } | companion::Action::SyncNativePlots { .. })
}
fn passive_companion_lane(action: Option<&companion::Action>, operation: Option<&remote::Operation>) -> bool {
    action.is_none_or(passive_companion_action) && operation.is_none_or(passive_node_operation)
}
fn companion_activity_state(local_busy: bool, node_pending: bool) -> &'static str {
    if local_busy || node_pending { "running" } else { "ready" }
}
fn companion_launch_available(
    local_busy: bool, queued: bool, node: Option<&remote::Node>, job: Option<&serde_json::Value>,
    action: Option<&companion::Action>, operation: Option<&remote::Operation>,
) -> bool {
    if local_busy || queued || !passive_companion_lane(action, operation) { return false; }
    let Some(node) = node else { return action.is_none() && operation.is_none(); };
    // A remembered job without its matching terminal status is unresolved.
    // This is an advisory for the GUI; dispatch still performs its full review.
    match (node.last_job.as_deref(), job) {
        (None, None) => true,
        (Some(id), Some(job)) => job["id"] == id && matches!(job["state"].as_str(),
            Some("stopped" | "interrupted" | "completed" | "failed" | "cancelled")),
        _ => false,
    }
}

fn safe(text: &str) -> String {
    text.chars()
        .filter(|c| *c == '\n' || *c == '\t' || !c.is_control())
        .collect()
}
fn absolute(path: PathBuf, cwd: &Path) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    }
}
fn unquote(s: &str) -> &str {
    s.strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .unwrap_or(s)
}

fn log_display_rows(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return Vec::new();
    }
    let mut rows = Vec::new();
    for line in text.lines() {
        let mut row = String::new();
        let mut used = 0;
        for grapheme in line.graphemes(true) {
            let visible = if grapheme == "\t" { "    " } else { grapheme };
            let cells = UnicodeWidthStr::width(visible);
            if used > 0 && used + cells > width {
                rows.push(std::mem::take(&mut row));
                used = 0;
            }
            row.push_str(visible);
            used += cells;
        }
        rows.push(row);
    }
    rows
}

fn draw_log(frame: &mut Frame, area: Rect, text: &str, offset: usize, title: &str) {
    // Job::log_tail bounds the input. Wrap first so scrolling and End count the
    // same display rows that the terminal shows, including long command lines.
    let rows = log_display_rows(&safe(text), area.width.saturating_sub(2) as usize);
    let end = rows.len().saturating_sub(offset.min(rows.len()));
    let start = end.saturating_sub(area.height.saturating_sub(2) as usize);
    let visible = rows[start..end]
        .iter()
        .map(|s| Line::raw(s.as_str()))
        .collect::<Vec<_>>();
    frame.render_widget(Paragraph::new(visible).block(panel(title)), area);
}

fn local_forecast_action(action: &str, command: &[String]) -> bool {
    matches!(action, "run-plan" | "go" | "sim" | "run" | "resume")
        && !command.iter().any(|arg| matches!(arg.as_str(), "--dry-run" | "--physics-profiles" | "--help"))
}

fn draw_forecast_progress(frame: &mut Frame, area: Rect, status: &serde_json::Value) {
    let text = node_ui::job_progress_text(status, area.width < 90 || area.height < 14);
    frame.render_widget(
        Paragraph::new(safe(&text)).wrap(Wrap { trim: false }).block(panel(" Forecast progress ")),
        area,
    );
}

impl App {
    fn new() -> io::Result<Self> {
        let cwd = env::current_dir()?;
        #[cfg(test)]
        let nodes = remote::Controller::load(&cwd);
        #[cfg(not(test))]
        let nodes = remote::Controller::load_user(&cwd);
        let mut app = Self {
            editor: None, python: env::var_os("GPUWM_TUI_PYTHON").map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("python")),
            output: cwd.join("arwen-runs"), prepared: PathBuf::new(), geog_root: PathBuf::new(), cwd,
            tab: Tab::Home, dialog: None, dialog_stack: Vec::new(), saved_guides: Vec::new(), job: None, startup_failure: None,
            status: "Choose a forecast mode, open a configuration, or continue a run. Each launch starts with a review.".into(),
            selected: 0, log_offset: 0, local_raw_logs: false, exit: false, input_enabled: false, active_config: None, memory_recovery_config: None, pending_config: None, guide_cache: None, case_cache: None, hits: Vec::new(),
            plot_catalog: plotsettings::Catalog::default(),
            cds: cds_credentials::Client::default(),
            companion: companion::Controller::default(),
            companion_remote: None,
            companion_waiting: None,
            focus_logs_pending: None,
            run_views: run_view::Manager::default(),
            companion_artifacts: None,
            tui_map_review: None,
            tui_map_waiting: None,
            tui_map_launch: None,
            discard_setup_review: false,
            pending_workflow: None,
            nodes, node_panel: node_ui::Panel::default(), plots_from_nodes: false,
            #[cfg(test)]
            clipboard_hook: None,
            exit_after_job: false,
        };
        if let Some(error) = app.nodes.load_error.clone() {
            app.open_nodes();
            app.node_panel.error(format!("{error}\n\nNode preferences: {}\nCorrect the file, then return to Nodes and choose Reload.", display_path(&app.nodes.path)));
        } else if app.nodes.store.selected().is_some() {
            app.open_nodes();
            app.status =
                "Saved node selected. Jobs reconnects to its runs; Start reviews a new one.".into();
        }
        Ok(app)
    }
    fn set_viewport(&mut self, width: u16, height: u16) {
        self.input_enabled = width >= 65 && height >= 20;
    }
    fn dirty(&self) -> bool {
        self.editor.as_ref().is_some_and(|e| e.dirty)
    }
    fn overlay(&mut self, dialog: Dialog) {
        if let Some(previous) = self.dialog.take() {
            self.dialog_stack.push(previous);
        }
        self.dialog = Some(dialog);
    }
    fn close_overlay(&mut self) {
        self.dialog = self.dialog_stack.pop();
    }
    fn request_quit(&mut self) {
        if matches!(self.dialog, Some(Dialog::Quit)) { return; }
        if self.busy() || self.dirty() || self.dialog.is_some() || !self.saved_guides.is_empty() {
            self.overlay(Dialog::Quit);
        } else {
            self.exit = true;
        }
    }
    fn show_job_result(&mut self) {
        if let Some(Dialog::Details { offset, .. }) = &mut self.dialog {
            *offset = 0;
            return;
        }
        let previous = self.dialog.take();
        self.show_details();
        if let Some(Dialog::Details { offset, .. }) = &mut self.dialog { *offset = 0; }
        if let Some(previous) = previous { self.dialog_stack.push(previous); }
    }
    fn job_result(&self) -> Option<(String, Option<PathBuf>, bool)> {
        if let Some(failure) = &self.startup_failure {
            return Some(("FAILED TO START · click for details".into(), failure.log.clone(), true));
        }
        let job = self.job.as_ref()?;
        let code = job.outcome?;
        let state = if job.interrupted() { "STOPPED" } else if code == 0 { "COMPLETED" } else { "FAILED" };
        Some((format!("{state} · {} · exit {code} · click for details", job.action),
            Some(job.dir.join("job.log")), code != 0))
    }
    fn retain_guide(&mut self, guide: Guide) {
        self.saved_guides.retain(|saved| saved.kind != guide.kind
            || saved.workflow != guide.workflow || saved.research != guide.research);
        self.saved_guides.push(guide);
        self.status = "Setup answers retained for this session. Esc resumes them, or reopen the same weather task.".into();
    }
    fn busy(&self) -> bool {
        self.job.as_ref().is_some_and(|j| j.outcome.is_none())
    }
    fn failed(&self) -> bool {
        self.startup_failure.is_some()
            || self
                .job
                .as_ref()
                .is_some_and(|j| j.outcome.is_some_and(|code| code != 0) && !j.interrupted())
    }
    fn badge(&self) -> &'static str {
        if self.nodes.store.selected().is_some() {
            if self.nodes.pending.as_ref().is_some_and(|request|matches!(request.operation,remote::Operation::Probe))&&self.nodes.view.runtime.is_none()&&self.nodes.view.status.is_none() {
                return "CONNECTING";
            }
            if self.nodes.view.connection_error.is_some() {
                return "DISCONNECTED";
            }
            return match self
                .nodes
                .view
                .status
                .as_ref()
                .and_then(|v| v["state"].as_str())
            {
                Some("running" | "starting") => "RUNNING",
                Some("stopping") => "STOPPING",
                Some("completed") => "COMPLETED",
                Some("failed") => "FAILED",
                Some("stopped" | "interrupted" | "cancelled") => "STOPPED",
                Some(_) => "CHECK STATUS",
                None if self.nodes.view.runtime.is_some() => "CONNECTED",
                None => "RECONNECT",
            };
        }
        if self.busy() {
            if self.job.as_ref().is_some_and(Job::is_stopping) { "STOPPING" } else { "RUNNING" }
        } else if self.failed() {
            if self.dirty() {
                "FAILED · DRAFT"
            } else {
                "FAILED"
            }
        } else if self.job.as_ref().is_some_and(Job::interrupted) {
            if self.dirty() {
                "INTERRUPTED · DRAFT"
            } else {
                "INTERRUPTED"
            }
        } else if self.dirty() {
            "UNSAVED DRAFT"
        } else if self.job.is_some() {
            "COMPLETED"
        } else {
            "READY"
        }
    }
    fn memory_recovery_available(&self) -> bool {
        self.failed()
            && self
                .editor
                .as_ref()
                .is_some_and(|editor| self.memory_recovery_config.as_ref() == Some(&editor.path))
    }
    fn log_path(&self) -> Option<PathBuf> {
        if let Some(failure) = &self.startup_failure {
            failure.log.clone()
        } else {
            self.job.as_ref().map(|job| job.dir.join("job.log"))
        }
    }
    fn log_text(&self) -> String {
        self.job
            .as_ref()
            .filter(|_| self.startup_failure.is_none())
            .map(|job| job.log_tail(1000))
            .or_else(|| self.log_path().map(|path| job::read_log_tail(&path, 1000)))
            .filter(|text| !text.trim().is_empty())
            .unwrap_or_else(|| {
                self.startup_failure
                    .as_ref()
                    .map(|failure| failure.message.clone())
                    .unwrap_or_else(|| {
                        "No command output yet. Choose a command from Overview.".into()
                    })
            })
    }
    fn has_local_forecast_job(&self) -> bool {
        self.startup_failure.is_none()
            && self.job.as_ref().is_some_and(|job| local_forecast_action(&job.action, &job.command))
    }
    fn show_details(&mut self) {
        let mut text = self.status.clone();
        if let Some(failure) = &self.startup_failure {
            if text != failure.message {
                text.push_str(&format!(
                    "\n\nLatest command FAILED to start:\n{}",
                    failure.message
                ));
            }
        } else if let Some(job) = &self.job {
            let outcome = match job.outcome {
                None => "RUNNING".to_owned(),
                Some(code) if job.interrupted() => format!("INTERRUPTED (exit {code})"),
                Some(0) => "COMPLETED (exit 0)".to_owned(),
                Some(code) => format!("FAILED (exit {code})"),
            };
            text.push_str(&format!("\n\nLatest command: {} — {outcome}", job.action));
        }
        if let Some(notice) = self.job.as_ref().and_then(|job| job.completion_notice.as_ref()) {
            if !text.contains(notice) { text.push_str(&format!("\n{notice}")); }
        }
        if let Some(path) = self.log_path() {
            text.push_str(&format!(
                "\n\nSaved log file:\n{}\n\nRecent output (up to 1000 lines / 128 KiB):\n{}",
                display_path(&path),
                self.log_text()
            ));
        }
        self.dialog = Some(Dialog::Details {
            text: safe(&text),
            notice: None,
            // Failures normally explain the cause at the end of the output.
            // Home still exposes the complete status and saved file path.
            offset: if self.log_path().is_some() && !self.memory_recovery_available() {
                usize::MAX
            } else {
                0
            },
        });
    }
    fn copy_diagnostic(&self, details: &str, full_log: bool) -> String {
        let content = if full_log {
            self.log_path()
                .ok_or_else(|| "No saved log exists for this command. Use Copy details.".into())
                .and_then(|path| clipboard::read_log(&path))
        } else {
            Ok(details.to_owned())
        };
        let result = content.and_then(|text| self.copy_text(&text));
        match result {
            Ok(bytes) => format!(
                "Copied {} ({bytes} bytes). Paste into another app.",
                if full_log {
                    "complete saved log"
                } else {
                    "details"
                }
            ),
            Err(error) => format!("Copy failed: {error}"),
        }
    }
    fn copy_text(&self, text:&str)->Result<usize,String>{
        #[cfg(test)]
        if let Some(copy)=self.clipboard_hook{return copy(text).map(|_|text.len());}
        clipboard::copy(text).map(|_|text.len())
    }
    fn node_log_text(&self)->String{
        let node=self.nodes.store.selected();
        let mut value=format!("Node: {}\nJob: {}\nState: {}\n\nRecent retained node output (up to 128 KiB):\n{}",
            node.map(|n|n.name.as_str()).unwrap_or("No node selected"),
            node.and_then(|n|n.last_job.as_deref()).unwrap_or("No job selected"),
            self.nodes.view.status.as_ref().and_then(|v|v["state"].as_str()).unwrap_or("Unknown"),
            self.nodes.view.log);
        if let Some(error)=self.nodes.view.status.as_ref().and_then(|v|v["error"].as_str()){
            value.push_str(&format!("\n\nNative failure: {error}\n"));
        }
        safe(&value)
    }
    fn save_log_text(&self, text:&str)->Result<PathBuf,String>{
        use std::io::Write;
        let directory=self.output.join(".arwen-tui").join("logs");
        fs::create_dir_all(&directory).map_err(|e|e.to_string())?;
        let path=directory.join(format!("arwen-log-{}-{}.txt",std::process::id(),remote::stamp()));
        let mut file=fs::OpenOptions::new().write(true).create_new(true).open(&path).map_err(|e|e.to_string())?;
        file.write_all(text.as_bytes()).and_then(|_|file.sync_all()).map_err(|e|e.to_string())?;
        path.canonicalize().map_err(|e|e.to_string())
    }
    fn open_log_text(&self, text:&str)->String{
        let result=self.save_log_text(text).and_then(|path|{
            #[cfg(not(test))]
            {
                let program=if cfg!(target_os="windows"){"notepad.exe"}else if cfg!(target_os="macos"){"open"}else{"xdg-open"};
                std::process::Command::new(program).arg(&path).spawn()
                    .map_err(|e|format!("Saved {}. Could not open the text editor: {e}",display_path(&path)))?;
            }
            Ok(path)
        });
        match result{Ok(path)=>format!("Opened plain-text log: {}",display_path(&path)),Err(error)=>error}
    }
    fn fit_current(&mut self) {
        // A refusal (for example an unsaved draft) must remain visible in the
        // workspace status instead of being hidden behind the previous modal.
        self.dialog = None;
        if !self.memory_recovery_available() {
            self.status = "This failure is not a confirmed memory refusal. Open Details to read and copy its cause.".into();
            return;
        }
        self.begin_guide(Kind::Fit);
        let Some(Dialog::Guide(guide)) = &mut self.dialog else {
            return;
        };
        let Some(editor) = &self.editor else {
            return;
        };
        guide.questions[0].value = display_path(&editor.path);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let fitted = editor.path.with_file_name(format!(
            "{}.fit-{stamp}.toml",
            editor
                .path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
        ));
        guide.questions[6].value = display_path(&fitted);
        if let Ok(doc) = editor.text().parse::<toml_edit::DocumentMut>() {
            if let Some(projection) = doc.get("projection") {
                let coordinate = |key| {
                    projection.get(key).and_then(|value| {
                        value
                            .as_float()
                            .filter(|n| n.is_finite())
                            .map(|n| n.to_string())
                            .or_else(|| value.as_integer().map(|n| n.to_string()))
                    })
                };
                if let (Some(lat), Some(lon)) = (coordinate("ref_lat"), coordinate("ref_lon")) {
                    guide.questions[1].value = format!("{lat},{lon}");
                }
            }
            if let Some(start) = doc
                .get("experiment")
                .and_then(|table| table.get("start_time"))
            {
                guide.questions[3].value = start
                    .as_str()
                    .map(str::to_owned)
                    .or_else(|| {
                        start
                            .as_value()
                            .and_then(|value| value.as_datetime())
                            .map(ToString::to_string)
                    })
                    .unwrap_or_default();
            }
        }
        self.status = "Review the current location and time, then fit a NEW configuration to this GPU. The original and its physics stay unchanged; no forecast starts.".into();
    }
    fn tile_current(&mut self) {
        self.dialog = None;
        if !self.memory_recovery_available() {
            self.status = "This failure is not a confirmed memory refusal. Open Details to read and copy its cause.".into();
            return;
        }
        self.begin_guide(Kind::Tiles);
        let Some(Dialog::Guide(mut guide)) = self.dialog.take() else {
            return;
        };
        let Some(editor) = &self.editor else {
            return;
        };
        guide.questions[0].value = display_path(&editor.path);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        guide.questions[1].value = display_path(&editor.path.with_file_name(format!(
            "{}.tiles-{stamp}.toml", editor.path.file_stem().unwrap_or_default().to_string_lossy()
        )));
        self.status = "Keep the domain area, resolution and physics. The engine plans GPU tiles and checks available GPU and system RAM. Review before creating a new configuration.".into();
        self.dialog = Some(Dialog::Summary(guide, 0));
    }
    fn reset_setup(&mut self) {
        // Explicit Reset discards only editable setup. The local Job, its
        // active_config, the SSH controller and its selected node remain owned.
        self.discard_setup_review |= self.nodes.pending.as_ref().is_some_and(|request|
            matches!(request.operation, remote::Operation::ReviewPlan { .. }
                | remote::Operation::Start { preview: true, .. }
                | remote::Operation::Resume { preview: true, .. }));
        self.editor = None;
        self.prepared = PathBuf::new();
        self.memory_recovery_config = None;
        self.pending_config = None;
        self.pending_workflow = None;
        self.guide_cache = None;
        self.case_cache = None;
        self.saved_guides.clear();
        self.tui_map_review = None;
        self.tui_map_waiting = None;
        self.tui_map_launch = None;
        self.dialog = None;
        self.dialog_stack.clear();
        fn is_review(screen: &node_ui::Screen) -> bool {
            match screen {
                node_ui::Screen::Review { .. } => true,
                node_ui::Screen::Error { previous, .. } => is_review(previous),
                _ => false,
            }
        }
        if is_review(&self.node_panel.screen) { self.node_panel.open(&self.nodes); }
        self.tab = Tab::Home;
        self.selected = 0;
        self.hits.clear();
        self.status = "Setup reset. Choose a mode or open a configuration. Saved files, the selected target and running jobs are preserved.".into();
    }
    fn accept_setup_review_update(&mut self, update: &remote::Update) -> bool {
        let stale = self.discard_setup_review && matches!(update,
            remote::Update::Preview { .. } | remote::Update::PlanReviewed(_));
        self.discard_setup_review = false;
        if stale {
            self.status = "Setup reset; the earlier launch review was discarded. Running jobs remain available.".into();
        }
        !stale
    }
    fn open(&mut self, path: PathBuf) {
        if self.dirty() {
            self.status =
                "Save with Ctrl+S, or export the draft with F12 before opening another configuration.".into();
            return;
        }
        let path = absolute(path, &self.cwd);
        let catalog = path.extension().is_some_and(|extension| {
            extension.eq_ignore_ascii_case("zip") || extension.eq_ignore_ascii_case("json")
        }) || (path.extension().is_some_and(|extension| extension.eq_ignore_ascii_case("toml"))
            && fs::read_to_string(&path).ok()
                .and_then(|text| text.parse::<toml_edit::DocumentMut>().ok())
                .and_then(|document| document.get("schema").and_then(|v| v.as_str()).map(str::to_owned))
                .is_some_and(|schema| schema.starts_with("arwen.case-catalog")));
        if catalog {
            self.begin_cases_from(Some(path));
            return;
        }
        match Editor::load(path) {
            Ok(e) => {
                self.memory_recovery_config = None;
                self.status = format!(
                    "Opened {}. Next: Enter reviews the launch plan; E edits every setting.",
                    display_path(&e.path)
                );
                self.selected = 1;
                self.editor = Some(e);
                self.tab = Tab::Overview;
            }
            Err(e) => self.status = format!("Could not open configuration: {e}"),
        }
    }
    fn browse(&mut self, dir: PathBuf) {
        let items = fs::read_dir(&dir).map(|list| {
            let mut items = list
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| {
                    p.is_dir()
                        || p.extension().is_some_and(|e| {
                            e.eq_ignore_ascii_case("toml")
                                || e.eq_ignore_ascii_case("json")
                                || e.eq_ignore_ascii_case("zip")
                        })
                })
                .collect::<Vec<_>>();
            items.sort_by_key(|p| {
                (
                    !p.is_dir(),
                    p.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_lowercase(),
                )
            });
            if let Some(parent) = dir.parent() {
                items.insert(0, parent.to_path_buf());
            }
            items
        });
        match items {
            Ok(items) => self.dialog = Some(Dialog::Browser(dir, items, 0)),
            Err(e) => self.status = format!("Could not read folder: {e}"),
        }
    }
    fn save(&mut self) {
        if !self.dirty() {
            self.status = "No unsaved changes.".into();
            return;
        }
        if let Some(e) = &mut self.editor {
            if self.active_config.as_ref() == Some(&e.path) {
                self.status = "This configuration is in use by the running command. Keep editing your draft, or F12 saves it as a separate file.".into();
                return;
            }
            self.status = match e.save() {
                Ok(p) => {
                    self.memory_recovery_config = None;
                    format!(
                        "Saved. Next: F6 reviews the launch plan. Backup: {}",
                        display_path(&p)
                    )
                }
                Err(e) => e,
            };
        }
    }
    fn era5_provider(&self) -> Option<&'static str> {
        let doc = self.editor.as_ref()?.text().parse::<toml_edit::DocumentMut>().ok()?;
        let fetch = doc.get("fetch")?.as_table_like()?;
        if fetch.get("source").and_then(toml_edit::Item::as_str) != Some("era5") { return None; }
        Some(match fetch.get("era5_provider") {
            None => "cds",
            Some(value) => match value.as_str() {
                Some("cds") => "cds", Some("arco") => "arco", _ => "unknown",
            },
        })
    }
    fn begin_era5_provider(&mut self) {
        let Some(provider) = self.era5_provider() else { return; };
        self.dialog = Some(Dialog::Era5Provider(usize::from(provider == "cds"),
            self.editor.as_ref().unwrap().text()));
    }
    fn begin_cds_credentials(&mut self) {
        self.cds.refresh(&self.python, &self.cwd);
        self.overlay(Dialog::CdsCredentials(cds_credentials::Form::new()));
    }
    fn companion_context(&self) -> serde_json::Value {
        let config = self.editor.as_ref().map(|editor| &editor.path);
        let hash = config.and_then(|path| fs::read(path).ok()).map(|bytes| companion::digest(&bytes));
        serde_json::json!({"config_path":config,"config_sha256":hash,"python":self.python,"cwd":self.cwd,
            "output_root":self.output,"geog_root":(!self.geog_root.as_os_str().is_empty()).then_some(&self.geog_root),
            "prepared_root":(!self.prepared.as_os_str().is_empty()).then_some(&self.prepared),
            "render_products":self.plot_spec().unwrap_or_else(|_| "all".into()),
            "plot_preferences_path":config.map(|path| plotsettings::sidecar(path)),
            "current_job_dir":self.job.as_ref().map(|job| &job.dir),"target":self.companion_target(),
            "available_targets":self.available_companion_targets()})
    }
    fn available_companion_targets(&self)->Vec<serde_json::Value>{
        let mut values=vec![serde_json::json!({"kind":"local","name":"Local computer"})];
        values.extend(self.nodes.store.nodes.iter().map(|node|serde_json::json!({"kind":"ssh","node_id":node.id,
            "connection_sha256":companion::digest(node.connection_key().as_bytes()),"name":node.name})));
        values
    }
    fn select_companion_target(&mut self,target:&companion::Target)->Result<String,String>{
        let id=match target{
            companion::Target::Local=>None,
            companion::Target::Ssh{node_id,connection_sha256}=>{
                let node=self.nodes.store.nodes.iter().find(|node|node.id==*node_id).ok_or("That saved node no longer exists. Refresh the targets.")?;
                if companion::digest(node.connection_key().as_bytes())!=*connection_sha256{return Err("That node connection changed. Refresh the targets before selecting it.".into());}
                node.validate(false)?;
                Some(node_id.clone())
            }
        };
        if self.nodes.pending.as_ref().is_some_and(|pending|matches!(pending.operation,remote::Operation::Probe)
            &&Some(&pending.node.id)==id.as_ref()&&self.checked_companion_target(Some(target)).is_ok()){
            return Ok("Connecting to the selected node…".into());
        }
        self.nodes.select(id)?;
        self.tui_map_waiting=None;self.tui_map_launch=None;
        if self.nodes.store.selected().is_none(){return Ok("Target: Local computer.".into());}
        self.nodes.begin(remote::Operation::Probe,&self.python,&self.output,&self.cwd).map_err(|error|{
            self.nodes.view.connection_error=Some(error.clone());error
        })?;
        Ok("Connecting to the selected node…".into())
    }
    fn companion_target(&self)->serde_json::Value {
        let Some(node)=self.nodes.store.selected() else{return serde_json::json!({"kind":"local","name":"Local computer"});};
        let runtime=self.nodes.view.runtime.as_ref();
        let capability=|name:&str|runtime.is_some_and(|r|r["capabilities"][name]==true);
        let hardware=runtime.and_then(|r|r.get("probe")).filter(|p|p["measured_unix_ms"].is_u64()&&p["devices"].is_array())
            .map(|p|serde_json::json!({"measured_unix_ms":p["measured_unix_ms"],"devices":p["devices"],"sizing":p["sizing"],"host_memory":p["host_memory"]}));
        serde_json::json!({"kind":"ssh","node_id":node.id,"name":node.name,
            "connection_sha256":companion::digest(node.connection_key().as_bytes()),"workspace":node.workspace,
            "remote_geog_root":(!node.geography.is_empty()).then_some(&node.geography),"hardware":hardware,
            "capabilities":{"stage_plan_v1":capability("stage_plan_v1"),"review_plan_v1":capability("review_plan_v1"),"start_plan_v1":capability("start_plan_v1"),"artifact_sync_v1":capability("artifact_sync_v1"),
                "artifact_index_v1":capability("artifact_index_v1"),"artifact_sequence_v1":capability("artifact_sequence_v1"),
                "processed_frame_v2":capability("processed_frame_v2"),"processed_member_stream_v2":capability("processed_member_stream_v2")}})
    }
    fn checked_companion_target(&self,target:Option<&companion::Target>)->Result<Option<remote::Node>,String>{
        match (self.nodes.store.selected(),target) {
            (None,None|Some(companion::Target::Local))=>Ok(None),
            (Some(node),Some(companion::Target::Ssh{node_id,connection_sha256}))
                if node.id==*node_id&&companion::digest(node.connection_key().as_bytes())==*connection_sha256=>Ok(Some(node.clone())),
            _=>Err("The execution target changed or its SSH binding is missing. Select the intended target and review again.".into()),
        }
    }
    fn checked_companion_plan(&self,request:&companion::Request,path:&std::path::Path)->Result<serde_json::Value,String>{
        if self.dirty(){return Err("Save the TUI draft before reviewing or launching a map plan.".into());}
        let plan=companion::read_json(path,4*1024*1024)?;
        if plan["schema"]!="gpuwm.run-plan.v1"{return Err("Unsupported run-plan schema.".into());}
        let bytes=fs::read(path).map_err(|e|e.to_string())?;
        let plan_hash=companion::digest(&bytes);
        if request.plan_sha256.as_deref()!=Some(plan_hash.as_str()){return Err("The saved map plan changed; review again.".into());}
        let config=PathBuf::from(plan["config"]["path"].as_str().ok_or("Remote review needs a saved configuration path.")?);
        let config=if config.is_absolute(){config}else{path.parent().unwrap_or(&self.cwd).join(config)};
        let config=config.canonicalize().map_err(|e|format!("Cannot open the saved map configuration: {e}"))?;
        if self.editor.as_ref().map(|e|&e.path)!=Some(&config){return Err("Open this saved map configuration in the TUI before reviewing its remote launch.".into());}
        let config_hash=companion::digest(&fs::read(&config).map_err(|e|e.to_string())?);
        if request.config_sha256.as_deref()!=Some(config_hash.as_str()){return Err("The saved map configuration changed; review again.".into());}
        Ok(serde_json::json!({"plan_path":path.canonicalize().map_err(|e|e.to_string())?,"plan_sha256":plan_hash,
            "config_path":config,"config_sha256":config_hash}))
    }
    fn begin_companion_remote(&mut self,request:companion::Request)->Result<(),String>{
        if self.companion_waiting.is_some()||self.tui_map_waiting.is_some(){return Err("A forecast action is already waiting for this node. It will continue automatically.".into());}
        if self.companion_remote.is_some()||self.nodes.pending.is_some(){
            let interactive=matches!(request.action,companion::Action::ReviewPlan(_)|companion::Action::LaunchPlan(_)|companion::Action::StopJob(_)|companion::Action::SelectTarget);
            let passive=passive_companion_lane(self.companion_remote.as_ref().map(|pending|&pending.request.action),
                self.nodes.pending.as_ref().map(|pending|&pending.operation));
            if !interactive||!passive{return Err("A node request is in progress. Wait for it to finish, then retry this action.".into());}
            if self.busy(){return Err("A local job is already running.".into());}
            self.checked_companion_target(request.target.as_ref())?.ok_or("Select an SSH node for this action.")?;
            if let companion::Action::ReviewPlan(path)|companion::Action::LaunchPlan(path)=&request.action{self.checked_companion_plan(&request,path)?;}
            let action=match request.action{companion::Action::ReviewPlan(_)=>"Review",companion::Action::LaunchPlan(_)=>"Launch",companion::Action::SelectTarget=>"Hardware refresh",_=>"Stop"};
            self.status=format!("Waiting for the current node request; {action} will continue automatically.");
            self.companion_waiting=Some(QueuedCompanionRequest{request,
                session_id:self.companion.session.as_ref().map(|session|session.id.clone()),queued_at:Instant::now()});
            return Ok(());
        }
        let node=self.checked_companion_target(request.target.as_ref())?.ok_or("Select an SSH node for this action.")?;
        let mut source=serde_json::Value::Null;
        let operation=match &request.action {
            // checked_companion_target above guarantees the exact current
            // node/connection. Do not call nodes.select: it clears the active
            // job/log view, even when the requested node is already selected.
            companion::Action::SelectTarget=>remote::Operation::Probe,
            companion::Action::ReviewPlan(path)=>{
                if self.busy(){return Err("A local job is already running.".into());}
                let target=self.companion_target();
                if target["capabilities"]["review_plan_v1"]!=true||target["capabilities"]["stage_plan_v1"]!=true {
                    return Err("Connect to the selected node in Nodes and confirm its current ArWen runtime supports staged map plans. No remote work was started.".into());
                }
                source=self.checked_companion_plan(&request,path)?;
                remote::Operation::ReviewPlan{plan:path.clone(),plan_sha256:request.plan_sha256.clone().unwrap(),config_sha256:request.config_sha256.clone().unwrap(),
                    output:format!("{}/arwen-map-{}",node.output_directory().trim_end_matches('/'),remote::stamp())}
            }
            companion::Action::LaunchPlan(path)=>{
                if self.busy(){return Err("A local job is already running.".into());}
                if self.nodes.view.status.as_ref().is_some_and(|job|matches!(job["state"].as_str(),Some("starting"|"running"|"stopping"|"ownership_mismatch"|"lost"))){
                    return Err("The selected node's current job is still active or unresolved. Inspect its status before starting another map plan.".into());
                }
                source=self.checked_companion_plan(&request,path)?;
                let id=request.review_id.as_deref().ok_or("Remote launch needs its review ID.")?;
                let review_path=self.companion.session.as_ref().ok_or("The companion session closed.")?.review_path(id)?;
                let bytes=fs::read(&review_path).map_err(|e|format!("Cannot read the completed node review: {e}"))?;
                if request.review_sha256.as_deref()!=Some(companion::digest(&bytes).as_str()){return Err("The completed node review changed; review again.".into());}
                let review=companion::read_json(&review_path,2*1024*1024)?;
                if review["schema"]!="arwen.companion-remote-review.v1"||review["review_id"]!=id
                    ||review["source"]!=source||review["target"]!=request.target.as_ref().unwrap().value()
                    ||review["node_settings_sha256"]!=companion::digest(serde_json::json!([node.output_directory(),node.geography]).to_string().as_bytes()) {
                    return Err("The node, its input locations, or saved configuration changed after review. Review again.".into());
                }
                Self::checked_companion_inputs(&review["remote_review"])?;
                let mut remote_review=review["remote_review"].clone();
                if remote_review["source_blobs"].as_array().is_some_and(|blobs|!blobs.is_empty()){
                    remote_review["local_source_manifest"]=serde_json::json!(review_path);
                }
                remote::Operation::StartPlan{review:remote_review}
            }
            companion::Action::StopJob(id)=>{
                if node.last_job.as_deref()!=Some(id.as_str())||self.nodes.view.status.as_ref().and_then(|j|j["id"].as_str())!=Some(id.as_str()){
                    return Err("That job is not the selected node's current recorded job. Refresh Nodes before stopping it.".into());
                }
                remote::Operation::Stop{job:id.clone()}
            }
            companion::Action::SyncArtifacts{job,domain,sequence,reader_leases}=>{
                if node.last_job.as_deref()!=Some(job.as_str())||self.nodes.view.status.as_ref().and_then(|j|j["id"].as_str())!=Some(job.as_str()){
                    return Err("That artifact job is not the selected node's current recorded job. Refresh Nodes first.".into());
                }
                if self.companion_target()["capabilities"]["artifact_sync_v1"]!=true{
                    return Err("Reconnect the selected node to confirm committed frame transfer support.".into());
                }
                if sequence.is_some()&&self.companion_target()["capabilities"]["artifact_sequence_v1"]!=true{
                    return Err("Reconnect the selected node to confirm forecast-time selection support.".into());
                }
                let cache=self.companion.session.as_ref().ok_or("The companion session closed.")?.directory
                    .join("remote-artifacts").join(&node.id).join(job);
                remote::Operation::SyncArtifacts{job:job.clone(),domain:*domain,cache,sequence:*sequence,reader_leases:*reader_leases}
            }
            companion::Action::ArtifactIndex{job,domain,after_sequence}=>{
                if node.last_job.as_deref()!=Some(job.as_str())||self.nodes.view.status.as_ref().and_then(|j|j["id"].as_str())!=Some(job.as_str()){
                    return Err("That timeline job is not the selected node's current recorded job. Refresh Nodes first.".into());
                }
                if self.companion_target()["capabilities"]["artifact_index_v1"]!=true{
                    return Err("Reconnect the selected node to confirm native forecast timeline support.".into());
                }
                remote::Operation::ArtifactIndex{job:job.clone(),domain:*domain,after_sequence:*after_sequence}
            }
            companion::Action::SyncProcessedFrame{job,domain,sequence}=>{
                if node.last_job.as_deref()!=Some(job.as_str())||self.nodes.view.status.as_ref().and_then(|status|status["id"].as_str())!=Some(job.as_str()){
                    return Err("That converted frame is not the selected node's current recorded job. Open an older run through Runs.".into());
                }
                let cache=self.companion.session.as_ref().ok_or("The companion session closed.")?.directory
                    .join("processed-store").join(&node.id).join(job);
                remote::Operation::SyncProcessedFrame{job:job.clone(),domain:*domain,cache,sequence:*sequence}
            }
            companion::Action::SyncProcessedFrameV2{job,domain,sequence,options,reader_leases,cache_bytes}=>{
                if node.last_job.as_deref()!=Some(job.as_str())||self.nodes.view.status.as_ref().and_then(|status|status["id"].as_str())!=Some(job.as_str()){
                    return Err("This viewer frame is not the selected node's current job. Open a saved run through My forecasts.".into());
                }
                if self.companion_target()["capabilities"]["processed_frame_v2"]!=true{
                    return Err("Reconnect the node after installing the compact native viewer update.".into());
                }
                let cache=self.output.join(".arwen-viewer-cache").join(&node.id).join(job);
                remote::Operation::SyncProcessedFrameV2{job:job.clone(),domain:*domain,cache,sequence:*sequence,
                    options:options.clone(),reader_leases:*reader_leases,cache_bytes:*cache_bytes}
            }
            companion::Action::SyncNativePlots{job,domain,sequence}=>{
                if node.last_job.as_deref()!=Some(job.as_str()){return Err("Open this saved job through My forecasts to retrieve its plots.".into());}
                remote::Operation::SyncNativePlots{job:job.clone(),domain:*domain,sequence:*sequence,cache:self.output.join(".arwen-native-plots-cache").join(&node.id).join(job)}
            }
            _=>return Err("Unsupported remote companion operation.".into()),
        };
        self.nodes.begin(operation,&self.python,&self.output,&self.cwd)?;
        self.status="Contacting the selected node; the TUI is retaining the request and its log.".into();
        self.companion_remote=Some(CompanionRemoteRequest{request,node,source});
        Ok(())
    }
    fn continue_queued_companion_remote(&mut self){
        let Some(queued)=self.companion_waiting.as_ref()else{return;};
        let same_session=queued.session_id==self.companion.session.as_ref().map(|session|session.id.clone());
        let stale=if !same_session{Some("The companion session changed while the node action was waiting; no action was dispatched.".to_owned())}
            else if queued.queued_at.elapsed()>=COMPANION_QUEUE_TIMEOUT{Some("The queued node action expired after 30 seconds; no action was dispatched. Retry when the node responds.".to_owned())}
            else if let Err(error)=self.checked_companion_target(queued.request.target.as_ref()){Some(error)}
            else if let companion::Action::ReviewPlan(path)|companion::Action::LaunchPlan(path)=&queued.request.action{
                self.checked_companion_plan(&queued.request,path).err()
            }else{None};
        if stale.is_none()&&(self.nodes.pending.is_some()||self.companion_remote.is_some()){return;}
        let queued=self.companion_waiting.take().expect("checked queued node action");
        let request=queued.request;
        let result=match stale{Some(error)=>Err(error),None=>self.begin_companion_remote(request.clone())};
        if let Err(error)=result{
            self.status=error.clone();
            if same_session{if let Some(session)=&self.companion.session{
                let _=session.respond_with(&request.id,&request.name,Err(error),serde_json::json!({"target":request.target.as_ref().map(companion::Target::value)}));
            }}
        }
    }
    fn finish_companion_remote(&mut self,update:&remote::Update){
        let Some(pending)=self.companion_remote.take()else{return;};
        let target=pending.request.target.as_ref().unwrap().value();
        let mut details=serde_json::json!({"target":target});
        let result=(||->Result<String,String>{
            self.checked_companion_target(pending.request.target.as_ref())?;
            match(update,&pending.request.action){
                (remote::Update::Failed(error),_)=>Err(error.clone()),
                (remote::Update::Connected,companion::Action::SelectTarget)=>Ok("The selected node probe completed.".into()),
                (remote::Update::PlanReviewed(remote_review),companion::Action::ReviewPlan(path))=>{
                    Self::checked_companion_inputs(remote_review)?;
                    if self.checked_companion_plan(&pending.request,path)?!=pending.source{return Err("The local selection changed during the node review.".into());}
                    let review=serde_json::json!({"schema":"arwen.companion-remote-review.v1","review_id":pending.request.id,
                        "created_unix_ms":companion::now_ms(),"target":target,"source":pending.source,
                        "node_settings_sha256":companion::digest(serde_json::json!([pending.node.output_directory(),pending.node.geography]).to_string().as_bytes()),
                        "remote_review":remote_review});
                    let(path,hash)=self.companion.session.as_ref().ok_or("The companion session closed.")?.save_review(&pending.request.id,&review)?;
                    if self.tui_map_review.as_ref().is_some_and(|request|request.id==pending.request.id){
                        let mut launch=pending.request.clone();launch.id=format!("tui-launch-{}",remote::stamp());launch.name="launch_plan".into();
                        launch.action=match &pending.request.action{companion::Action::ReviewPlan(path)=>companion::Action::LaunchPlan(path.clone()),_=>unreachable!()};
                        launch.review_id=Some(pending.request.id.clone());launch.review_sha256=Some(hash.clone());self.tui_map_launch=Some(launch);
                        self.node_panel.screen=node_ui::Screen::Review{operation:remote::Operation::StartPlan{review:remote_review.clone()},review:remote_review.clone()};
                        self.node_panel.notice="Current saved map setup reviewed on the selected node. Enter starts only these reviewed inputs.".into();
                        self.dialog=Some(Dialog::Nodes);
                    }
                    details["review_id"]=serde_json::json!(pending.request.id);details["review_path"]=serde_json::json!(path);details["review_sha256"]=serde_json::json!(hash);
                    Ok("Node review ready. No forecast started.".into())
                }
                (remote::Update::Started(id),companion::Action::LaunchPlan(_))=>{
                    details["job_id"]=serde_json::json!(id);details["job_dir"]=serde_json::Value::Null;
                    details["remote_output_root"]=self.nodes.view.status.as_ref().map(|j|j["outdir"].clone()).unwrap_or_default();
                    Ok("The reviewed plan started on the selected node.".into())
                }
                (remote::Update::Stopped(id),companion::Action::StopJob(_))=>{
                    details["job_id"]=serde_json::json!(id);Ok("The selected node confirmed job termination.".into())
                }
                (remote::Update::ArtifactsSynced(reply),companion::Action::SyncArtifacts{job,..})=>{
                    if self.nodes.store.selected().and_then(|node|node.last_job.as_deref())!=Some(job.as_str())
                        ||reply["artifacts"]["job_id"]!=*job{return Err("The selected artifact job changed during transfer.".into());}
                    details["job_id"]=serde_json::json!(job);details["transferred_bytes"]=reply["transferred_bytes"].clone();
                    details["waiting"]=reply["artifacts"]["waiting"].clone();
                    if let Some(recovery)=reply.get("cache_recovery"){
                        self.companion_artifacts=None;
                        details["cache_recovery"]=recovery.clone();
                        return Ok("Refreshing a retained frame whose cached bytes changed.".into());
                    }
                    if reply["artifacts"]["waiting"]==true{return Ok("Waiting for this domain's first committed node frame.".into());}
                    let mut artifacts=reply["artifacts"].clone();artifacts["schema"]=serde_json::json!("arwen.remote-artifacts.v1");
                    artifacts["target"]=target.clone();
                    let(path,hash)=self.companion.session.as_ref().ok_or("The companion session closed.")?.save_artifacts(&pending.request.id,&artifacts)?;
                    details["artifact_manifest_path"]=serde_json::json!(path);details["artifact_manifest_sha256"]=serde_json::json!(hash);
                    self.companion_artifacts=Some(details.clone());
                    Ok("Committed node frame ready.".into())
                }
                (remote::Update::ArtifactIndexed(reply),companion::Action::ArtifactIndex{job,..})=>{
                    if self.nodes.store.selected().and_then(|node|node.last_job.as_deref())!=Some(job.as_str())
                        ||reply["artifact_index"]["job_id"]!=*job{return Err("The selected timeline job changed during discovery.".into());}
                    let mut index=reply["artifact_index"].clone();index["schema"]=serde_json::json!("arwen.remote-artifact-index.v1");
                    index["target"]=target.clone();
                    let(path,hash)=self.companion.session.as_ref().ok_or("The companion session closed.")?.save_artifacts(&pending.request.id,&index)?;
                    details["job_id"]=serde_json::json!(job);details["waiting"]=index["waiting"].clone();
                    details["artifact_index_path"]=serde_json::json!(path);details["artifact_index_sha256"]=serde_json::json!(hash);
                    Ok("Native forecast timeline page ready.".into())
                }
                (remote::Update::ProcessedFrameSynced(reply),companion::Action::SyncProcessedFrame{job,..})=>{
                    if self.nodes.store.selected().and_then(|node|node.last_job.as_deref())!=Some(job.as_str())||reply["processed_frame"]["job_id"]!=*job{
                        return Err("The selected converted-frame job changed during transfer.".into());
                    }
                    details["job_id"]=serde_json::json!(job);details["processed_frame"]=reply["processed_frame"].clone();
                    let result=remote::processed_message(&details["processed_frame"]);
                    if let Ok(message)=&result{if details["processed_frame"]["processing"].is_object(){details["processed_frame"]["processing"]["message"]=serde_json::json!(message);}}
                    result
                }
                (remote::Update::ProcessedFrameSyncedV2(reply),companion::Action::SyncProcessedFrameV2{job,..})=>{
                    if self.nodes.store.selected().and_then(|node|node.last_job.as_deref())!=Some(job.as_str())||reply["processed_frame"]["job_id"]!=*job{
                        return Err("The selected viewer job changed during transfer.".into());
                    }
                    details["job_id"]=serde_json::json!(job);details["processed_frame"]=reply["processed_frame"].clone();
                    let result=remote::processed_message(&details["processed_frame"]);
                    if let Ok(message)=&result{if details["processed_frame"]["processing"].is_object(){details["processed_frame"]["processing"]["message"]=serde_json::json!(message);}}
                    result
                }
                (remote::Update::NativePlotsSynced(reply),companion::Action::SyncNativePlots{job,..})=>{
                    details["job_id"]=serde_json::json!(job);details["native_plots"]=reply["native_plots"].clone();
                    Ok(if reply["native_plots"]["waiting"]==true{"Native plots are still being prepared."}else{"Native plot gallery ready."}.into())
                }
                _=>Err("The node returned a different operation; no success was assumed.".into()),
            }
        })();
        self.status=match &result{Ok(message)|Err(message)=>message.clone()};
        if let Some(session)=&self.companion.session{if let Err(error)=session.respond_with(&pending.request.id,&pending.request.name,result,details){self.status=error;}}
    }
    fn checked_companion_inputs(review:&serde_json::Value)->Result<(),String>{
        let inputs=review["source_inputs"].as_object().filter(|m|!m.is_empty()&&m.len()<=22).ok_or("The remote review has no bounded selected-input manifest.")?;
        for(name,expected)in inputs{
            let path=std::path::Path::new(name);
            let hash=expected.as_str().filter(|h|h.len()==64&&h.bytes().all(|c|c.is_ascii_hexdigit())).ok_or("The review contains an invalid source-input SHA-256.")?;
            if !path.is_absolute()||path.is_symlink()||fs::metadata(path).map(|m|!m.is_file()||m.len()>65536).unwrap_or(true)
                ||fs::read(path).map(|b|companion::digest(&b)!=hash).unwrap_or(true){
                return Err(format!("Selected local input '{}' changed or is unavailable. Review the map plan again.",path.file_name().unwrap_or_default().to_string_lossy()));
            }
        }
        Ok(())
    }
    fn use_nodes_file(&mut self,path:PathBuf)->Result<(),String>{
        if !path.is_absolute()||!path.is_file(){return Err("--nodes-file requires an absolute existing node profile JSON file.".into());}
        let controller=remote::Controller::load_path(path);
        if let Some(error)=&controller.load_error{return Err(format!("Cannot open the selected node profile file: {error}"));}
        self.nodes=controller;
        Ok(())
    }
    fn prepare_startup_node(&mut self)->Result<(),String>{
        let node=self.nodes.store.selected().ok_or("--connect-node requires an active saved node profile; choose its ID in the selected node profile file.")?;
        node.validate(false)?;
        self.open_nodes();
        Ok(())
    }
    fn open_companion(&mut self) {
        if self.dirty() {
            self.status = "Save or Save As before opening the visual workspace. Your draft is preserved.".into();
            return;
        }
        if let Err(error) = self.plot_spec() { self.status = error; return; }
        let context = self.companion_context();
        self.status = match self.companion.open(&self.cwd, &self.output, context) { Ok(message) | Err(message) => message };
        self.publish_companion_status(true);
        if self.nodes.store.selected().is_some()&&self.nodes.pending.is_none()&&self.nodes.view.runtime.is_none(){self.node_request(remote::Operation::Probe);}
    }
    /// The desktop owns the visual window; the existing controller continues
    /// handling its requests and worker receipts without an interactive console.
    fn run_headless_companion(&mut self) -> Result<(), String> {
        self.open_companion();
        if self.companion.child_status().is_err() { return Err(self.status.clone()); }
        loop {
            self.poll();
            let workspace = self.companion.child_status();
            // Keep releasing/polling an owned worker even if its GUI closes.
            // The only operation that stops that worker is an explicit request.
            if workspace.as_ref().is_ok_and(|status| status.is_none()) || self.busy() {
                std::thread::sleep(Duration::from_millis(150));
                continue;
            }
            remote::remove_poll_records(&self.output);
            return match workspace {
                Ok(Some(status)) if status.success() => Ok(()),
                Ok(Some(status)) => Err(format!("The visual workspace closed with {status}. Diagnostic log: {}",
                    self.companion.session.as_ref().map(|session| session.directory.join("companion.log").display().to_string()).unwrap_or_else(|| "unavailable".into()))),
                Err(error) => Err(format!("Cannot read the visual workspace status: {error}")),
                Ok(None) => unreachable!("a running workspace keeps the controller alive"),
            };
        }
    }
    fn publish_companion_status(&mut self, force: bool) {
        if self.companion.session.is_none() { return; }
        if !force && self.companion.session.as_ref().is_some_and(|session| !session.due()) { return; }
        let mut status = self.companion_context();
        status["state"] = serde_json::json!(companion_activity_state(self.busy(),self.nodes.pending.is_some()));
        status["launch_available"] = serde_json::json!(companion_launch_available(self.busy(),
            self.companion_waiting.is_some()||self.tui_map_waiting.is_some(),self.nodes.store.selected(),self.nodes.view.status.as_ref(),
            self.companion_remote.as_ref().map(|pending|&pending.request.action),self.nodes.pending.as_ref().map(|pending|&pending.operation)));
        status["draft_dirty"] = serde_json::json!(self.dirty());
        status["queued_node_action"]=self.companion_waiting.as_ref().map(|queued|serde_json::json!({
            "id":queued.request.id,"action":queued.request.name,"state":"waiting","message":self.status,
            "expires_after_seconds":COMPANION_QUEUE_TIMEOUT.as_secs()})).unwrap_or_default();
        status["job"] = if let Some(node)=self.nodes.store.selected(){
            self.nodes.view.status.as_ref().map(|job|serde_json::json!({"job_id":job["id"],"job_dir":null,
                "target":{"kind":"ssh","node_id":node.id,"connection_sha256":companion::digest(node.connection_key().as_bytes())},
                "action":if job["action"]=="start-plan"{"run-plan"}else{job["action"].as_str().unwrap_or("remote")},
                "state":job["state"],"exit_code":job["exit_code"],"remote_output_root":job["outdir"],
                "source_config_path":job["source_config_path"],"source_config_sha256":job["source_config_sha256"],
                "error":job["error"],"stage":job["stage"],"phase":job["phase"],"phase_updated_unix_ms":job["phase_updated_unix_ms"],"model_elapsed_seconds":job["model_elapsed_seconds"],"valid_time":job["valid_time"],"render_summary":job["render_summary"],"progress":job["progress"],"pipeline_progress":job["pipeline_progress"],
                "background_maps":job["background_maps"],"native_plots":job["native_plots"],
                "manifest_ready":false,"log_path":null,"progress_path":null,"events_path":null,"ready_dir":null})).unwrap_or_default()
        }else{self.job.as_ref().map(|job| companion::job_status(job, &self.output)).unwrap_or(serde_json::Value::Null)};
        status["target_connection_error"]=serde_json::json!(self.nodes.view.connection_error);
        if let Some(artifacts)=&self.companion_artifacts{
            if status["job"]["job_id"]==artifacts["job_id"]&&status["job"]["target"]==artifacts["target"]{
                for key in ["artifact_manifest_path","artifact_manifest_sha256"]{status["job"][key]=artifacts[key].clone();}
            }
        }
        if let Some(session) = &mut self.companion.session {
            if let Err(error) = session.publish(status, force) { self.status = format!("Visual workspace status: {error}"); }
        }
    }
    fn poll_run_views(&mut self) {
        for reply in self.run_views.poll() {
            if let Some(session)=self.companion.session.as_ref().filter(|session|session.id==reply.session_id){
                let _=session.respond_with(&reply.request.id,&reply.request.name,reply.result,reply.details);
            }
        }
    }
    fn finish_focus_logs_request(&mut self){
        let Some((request,started,session_id))=self.focus_logs_pending.take()else{return;};
        if self.companion.session.as_ref().is_none_or(|session|session.id!=session_id){return;}
        if started.elapsed()<Duration::from_secs(30)&&(self.nodes.pending.is_some()||self.companion_remote.is_some()||self.companion_waiting.is_some()){
            self.focus_logs_pending=Some((request,started,session_id));return;
        }
        let result=(||{
            if started.elapsed()>=Duration::from_secs(30){return Err("The node is still busy. Retry opening progress.".into());}
            let companion::Action::FocusJobLogs(job)=&request.action else{unreachable!()};
            let Some(companion::Target::Ssh{node_id,connection_sha256})=&request.target else{return Err("Select the saved SSH node for this job.".into());};
            let node=self.nodes.store.nodes.iter().find(|node|node.id==*node_id).ok_or("The requested node no longer exists.")?;
            if companion::digest(node.connection_key().as_bytes())!=*connection_sha256{return Err("The saved node connection changed. Refresh before opening its logs.".into());}
            if self.nodes.store.active.as_ref()!=Some(node_id){self.nodes.select(Some(node_id.clone()))?;}
            self.nodes.remember_job(job)?;
            self.nodes.view.log.clear();self.nodes.view.cursor=0;self.nodes.view.status=None;
            self.nodes.begin(remote::Operation::Logs{job:job.clone(),cursor:0},&self.python,&self.output,&self.cwd)?;
            self.view(Tab::Logs);focus_console_window();
            Ok(format!("Progress opened for saved job {job}."))
        })();
        if let Some(session)=&self.companion.session{let _=session.respond(&request.id,&request.name,result,None);}
    }

    fn poll_companion_requests(&mut self) {
        self.finish_focus_logs_request();
        for request in self.companion.requests() {
            if matches!(request.action,companion::Action::FocusJobLogs(_)){
                if let Some(session)=&self.companion.session {
                    if self.focus_logs_pending.is_none(){self.focus_logs_pending=Some((request,Instant::now(),session.id.clone()));}
                    else{let _=session.respond(&request.id,&request.name,Err("A progress-window request is already pending.".into()),None);}
                }
                continue;
            }
            if run_view::Manager::handles(&request.action){
                let result=self.companion.session.as_ref().ok_or_else(||"The companion session closed.".to_owned())
                    .and_then(|session|self.run_views.begin(request.clone(),session,&self.nodes.store,&self.python,&self.output,&self.cwd));
                if let Some(session)=&self.companion.session{match result{
                    Ok(Some(reply))=>{let _=session.respond_with(&request.id,&request.name,reply.result,reply.details);}
                    Ok(None)=>{}
                    Err(error)=>{let _=session.respond_with(&request.id,&request.name,Err(error),serde_json::json!({"target":request.target.as_ref().map(companion::Target::value)}));}
                }}
                continue;
            }
            // Reuse the serialized, expiring foreground queue for a same-node
            // sizing refresh. Switching targets still follows the ordinary
            // selection refusal while any node request is in progress. An
            // already running Probe keeps the existing coalescing response.
            let sizing_refresh=matches!(request.action,companion::Action::SelectTarget)
                &&matches!(self.checked_companion_target(request.target.as_ref()),Ok(Some(_)))
                &&!self.nodes.pending.as_ref().is_some_and(|pending|matches!(pending.operation,remote::Operation::Probe));
            let remote_action=sizing_refresh||(matches!(&request.action,companion::Action::ReviewPlan(_)|companion::Action::LaunchPlan(_)|companion::Action::StopJob(_)|companion::Action::SyncArtifacts{..}|companion::Action::SyncProcessedFrame{..}|companion::Action::SyncProcessedFrameV2{..}|companion::Action::SyncNativePlots{..}|companion::Action::ArtifactIndex{..})
                && (self.nodes.store.selected().is_some()||matches!(request.target,Some(companion::Target::Ssh{..}))));
            if remote_action {
                if let Err(error)=self.begin_companion_remote(request.clone()){
                    self.status=error.clone();if let Some(session)=&self.companion.session{let _=session.respond_with(&request.id,&request.name,Err(error),serde_json::json!({"target":request.target.as_ref().map(companion::Target::value)}));}
                }
                self.publish_companion_status(true);continue;
            }
            let mut job = None;
            let result = match &request.action {
                companion::Action::BrowseRuns|companion::Action::OpenRun(_)|companion::Action::CloseRun(_)=>Err("Close a saved run through its own read-only viewer.".into()),
                companion::Action::ReviewPlan(_) => Err("Choose and connect an SSH node before remote review.".into()),
                companion::Action::SyncArtifacts{..}|companion::Action::SyncProcessedFrame{..}|companion::Action::SyncProcessedFrameV2{..}|companion::Action::SyncNativePlots{..}|companion::Action::ArtifactIndex{..}=>Err("Choose and connect the recorded SSH node before retrieving frames.".into()),
                companion::Action::LaunchPlan(path) => {
                    if self.busy() { Err("A local job is already running.".into()) }
                    else if self.nodes.store.selected().is_some() { Err("Select Local computer before launching a companion plan.".into()) }
                    else if self.dirty() { Err("Save the TUI draft before launching a plan.".into()) }
                    else if self.checked_companion_target(request.target.as_ref()).is_err(){Err("The execution target changed. Review again.".into())}
                    else if (request.plan_sha256.is_some()||request.config_sha256.is_some())&&self.checked_companion_plan(&request,path).is_err(){Err("The saved plan or configuration changed. Review again.".into())}
                    else {
                        match companion::read_json(path, 4 * 1024 * 1024) {
                            Ok(plan) if plan["schema"] == "gpuwm.run-plan.v1" => {
                                let previous = self.job.as_ref().map(|job| job.dir.clone());
                                self.start_command("run-plan", &[path.to_string_lossy().into_owned()], None);
                                job = self.job.as_ref().filter(|job| Some(&job.dir) != previous.as_ref()).map(|job| job.dir.clone());
                                if job.is_some() {
                                    self.active_config = plan["config"]["path"].as_str().map(PathBuf::from)
                                        .map(|config| if config.is_absolute() { config } else { path.parent().unwrap_or(&self.cwd).join(config) })
                                        .and_then(|config| config.canonicalize().ok());
                                    Ok("Run plan accepted.".into())
                                } else { Err(self.status.clone()) }
                            }
                            Ok(_) => Err("Unsupported run-plan schema.".into()),
                            Err(error) => Err(error),
                        }
                    }
                }
                companion::Action::StopJob(id) => {
                    match &mut self.job {
                        Some(current) if current.dir == PathBuf::from(id) => {
                            job = Some(current.dir.clone());
                            current.stop().map(|_| current.stop_message().to_owned()).map_err(|e| e.to_string())
                        }
                        _ => Err("That job is not the current TUI job.".into()),
                    }
                }
                companion::Action::OpenConfig(path) => {
                    if self.dirty() { Err("Save or Save As before opening another configuration.".into()) }
                    else if !path.is_file() || !path.extension().is_some_and(|extension| extension.eq_ignore_ascii_case("toml")) {
                        Err("Choose an existing TOML configuration.".into())
                    } else {
                        let expected = path.canonicalize().ok();
                        self.open(path.clone());
                        if self.editor.as_ref().map(|editor| &editor.path) == expected.as_ref() { Ok("Configuration opened.".into()) }
                        else { Err(self.status.clone()) }
                    }
                }
                companion::Action::ResetSetup => {
                    self.reset_setup();
                    Ok(self.status.clone())
                }
                companion::Action::FocusLogs => {
                    if request.target.is_some()&&self.checked_companion_target(request.target.as_ref()).is_err(){Err("The requested log target changed. Choose the intended node again.".into())}
                    else{self.view(Tab::Logs);focus_console_window();Ok("Logs selected in the control center.".into())}
                }
                companion::Action::FocusJobLogs(_)=>unreachable!("saved job focus is queued before dispatch"),
                companion::Action::FocusNodes => {self.open_nodes();Ok("Node targets opened in the control center.".into())}
                companion::Action::FocusSetup => {
                    self.dialog=None;self.dialog_stack.clear();self.view(Tab::Settings);
                    Ok(if self.editor.is_some(){"Configuration settings opened in the control center."}else{"Create or open a configuration on Home."}.into())
                }
                companion::Action::SelectTarget=>self.select_companion_target(request.target.as_ref().expect("typed target")),
            };
            self.status = match &result { Ok(message) | Err(message) => message.clone() };
            if result.is_err() { job = None; }
            if let Some(session) = &self.companion.session {
                let response=if matches!(request.action,companion::Action::SelectTarget){session.respond_with(&request.id,&request.name,result,
                    serde_json::json!({"target":request.target.as_ref().map(companion::Target::value)}))}else{session.respond(&request.id, &request.name, result, job.as_deref())};
                if let Err(error) = response {
                    self.status = format!("Visual workspace response: {error}");
                }
            }
            self.publish_companion_status(true);
        }
    }
    fn set_era5_provider(&mut self, provider: &str, original: &str) -> Result<(), String> {
        if !matches!(provider, "cds" | "arco") { return Err("Choose an ERA5 provider.".into()); }
        let editor = self.editor.as_mut().ok_or("Open an ERA5 configuration first.")?;
        if editor.text() != original {
            return Err("The draft changed. Reopen the ERA5 provider choice.".into());
        }
        let mut doc = original.parse::<toml_edit::DocumentMut>()
            .map_err(|_| "Correct the TOML in Settings first.")?;
        if doc.get("fetch").and_then(|fetch| fetch.get("source"))
            .and_then(toml_edit::Item::as_str) != Some("era5") {
            return Err("Provider selection needs [fetch].source = \"era5\".".into());
        }
        let forcing = doc.get("case_data").and_then(|data| data.get("forcing"))
            .and_then(toml_edit::Item::as_array)
            .filter(|paths| paths.len() == 1)
            .and_then(|paths| paths.get(0)).and_then(toml_edit::Value::as_str)
            .ok_or("Custom ERA5 inputs: edit provider and forcing paths in Settings.")?;
        let name_at = forcing.rfind(['/', '\\']).map_or(0, |index| index + 1);
        if !matches!(&forcing[name_at..], "era5-combined.grib" | "era5-combined.nc") {
            return Err("Custom ERA5 inputs: edit provider and forcing paths in Settings.".into());
        }
        let next_forcing = format!("{}{}", &forcing[..name_at],
            if provider == "arco" { "era5-combined.nc" } else { "era5-combined.grib" });
        let fetch = doc.get_mut("fetch").and_then(toml_edit::Item::as_table_like_mut)
            .ok_or("Correct [fetch] in Settings first.")?;
        let mut value = toml_edit::Value::from(provider);
        if let Some(old) = fetch.get("era5_provider").and_then(toml_edit::Item::as_value) {
            *value.decor_mut() = old.decor().clone();
        }
        fetch.insert("era5_provider", toml_edit::Item::Value(value));
        let paths = doc.get_mut("case_data").and_then(|data| data.get_mut("forcing"))
            .and_then(toml_edit::Item::as_array_mut).unwrap();
        let path = paths.get_mut(0).unwrap();
        let mut value = toml_edit::Value::from(next_forcing);
        *value.decor_mut() = path.decor().clone();
        *path = value;
        editor.replace_draft(doc.to_string());
        Ok(())
    }
    fn export_dialog(&mut self) {
        if let Some(e) = &self.editor {
            self.dialog = Some(Dialog::Path(
                "Save draft as",
                display_path(&e.suggested_draft_path()),
            ));
        } else {
            self.status = "Open a configuration first (O).".into();
        }
    }
    fn save_as(&mut self, path: PathBuf) -> bool {
        let Some(e) = &mut self.editor else {
            return false;
        };
        let path = absolute(path, e.path.parent().unwrap_or(&self.cwd));
        let previous = e.path.clone();
        match e.save_as(path) {
            Ok(saved) => {
                self.memory_recovery_config = None;
                self.status = format!(
                    "{} {}.{}{}",
                    if saved.syntax_error.is_some() {
                        "Saved syntactically invalid draft:"
                    } else {
                        "Saved new configuration:"
                    },
                    display_path(&saved.path),
                    if saved.syntax_error.is_some() {
                        " Correct the TOML before running; Check will report the syntax error."
                    } else {
                        " Check it before running."
                    },
                    if saved.moved_folder {
                        " Relative data paths now resolve from this new folder; review them."
                    } else {
                        " Relative data paths keep their original folder."
                    }
                );
                if let Err(error) = plotsettings::inherit(&previous, &saved.path) {
                    self.status
                        .push_str(&format!(" Plot settings were not copied: {error}"));
                }
                true
            }
            Err(error) => {
                self.status = error;
                false
            }
        }
    }
    fn argv(&self, action: Action) -> Result<(&'static str, Vec<String>), String> {
        if matches!(action, Action::Doctor) {
            return Ok(("doctor", vec![]));
        }
        let editor = self
            .editor
            .as_ref()
            .ok_or("Open a configuration first (O).")?;
        if editor.dirty {
            return Err("Save your draft with Ctrl+S before checking or launching it.".into());
        }
        let config = editor.path.to_string_lossy().into_owned();
        let output = self.output.to_string_lossy().into_owned();
        let (command, mut args) = match action {
            Action::Check => ("check", vec![config]),
            Action::Plan => (
                "go",
                vec![config, "--outdir".into(), output, "--dry-run".into()],
            ),
            Action::Run => ("go", vec![config, "--outdir".into(), output]),
            Action::Prepared => {
                if self.prepared.as_os_str().is_empty() {
                    return Err("Choose the prepared folder with F4, then press F8.".into());
                }
                (
                    "sim",
                    vec![
                        self.prepared.to_string_lossy().into_owned(),
                        "--experiment-config".into(),
                        config,
                        "--outdir".into(),
                        output,
                    ],
                )
            }
            Action::Doctor => unreachable!(),
        };
        if matches!(action, Action::Plan | Action::Run) && !self.geog_root.as_os_str().is_empty() {
            args.extend([
                "--geog-root".into(),
                self.geog_root.to_string_lossy().into_owned(),
            ]);
        }
        if matches!(action, Action::Plan | Action::Run | Action::Prepared) {
            args.extend([
                if matches!(action, Action::Prepared) {
                    "--render-products"
                } else {
                    "--products"
                }
                .into(),
                self.plot_spec()?,
            ]);
        }
        Ok((command, args))
    }
    fn plot_selection(&self) -> Result<plotsettings::Selection, String> {
        if let Some(node) = self.nodes.store.selected() {
            return match &node.plot_products {
                Some(spec) => Ok(plotsettings::Selection {
                    label: node.plot_label.clone().unwrap_or_else(|| "Custom".into()),
                    spec: plotsettings::normalize(spec)?,
                }),
                None => Ok(plotsettings::Selection::default()),
            };
        }
        self.editor
            .as_ref()
            .map(|editor| plotsettings::load(&editor.path))
            .unwrap_or_else(|| Ok(plotsettings::Selection::default()))
    }
    fn plot_spec(&self) -> Result<String, String> {
        self.plot_selection().map(|selection| selection.spec)
    }
    fn begin_plots(&mut self) {
        if self.nodes.store.selected().is_some() {
            if self.nodes.pending.is_some() {
                self.status = "Wait for the node request before changing plots.".into();
                return;
            }
            self.plots_from_nodes = true;
            self.plot_catalog.request(&self.python, &self.cwd);
            match self.plot_selection() {
                Ok(selection) => {
                    self.dialog = Some(Dialog::Plots(plotsettings::Form::from_selection(selection)))
                }
                Err(error) => self.status = error,
            }
            return;
        }
        self.plots_from_nodes = false;
        let Some(editor) = &self.editor else {
            self.status = "Create or open a configuration first, then choose Plots.".into();
            return;
        };
        self.plot_catalog.request(&self.python, &self.cwd);
        self.dialog = Some(Dialog::Plots(plotsettings::Form::new(&editor.path)));
    }
    fn open_nodes(&mut self) {
        self.node_panel.open(&self.nodes);
        self.dialog = Some(Dialog::Nodes);
    }
    fn current_setup_uses_staged_node(&self)->bool{
        self.editor.is_some()&&self.nodes.store.selected().is_some_and(|node|node.config.trim().is_empty())
    }
    fn current_setup_review_request(&mut self)->Result<companion::Request,String>{
        if self.dirty(){return Err("Save the current setup before reviewing its node forecast.".into());}
        let config=self.editor.as_ref().ok_or("Open or create a setup before reviewing its node forecast.")?.path.clone();
        let node=self.nodes.store.selected().ok_or("Choose a node for this setup.")?.clone();
        let products=self.plot_spec()?;
        let config_sha256=companion::digest(&fs::read(&config).map_err(|e|e.to_string())?);
        self.companion.ensure_session(&self.output)?;
        let directory=self.companion.session.as_ref().unwrap().directory.join("workspace");
        fs::create_dir_all(&directory).map_err(|e|e.to_string())?;
        let id=format!("tui-review-{}",remote::stamp());let path=directory.join(format!("{id}.json"));
        let plan=serde_json::json!({"schema":"gpuwm.run-plan.v1","name":id,"route":"prepared",
            "config":{"path":config},"output_root":self.output.join(format!("run-{id}")),
            "run_options":{"render_products":products}});
        let bytes=serde_json::to_vec_pretty(&plan).map_err(|e|e.to_string())?;
        use std::io::Write;
        let mut file=fs::OpenOptions::new().write(true).create_new(true).open(&path).map_err(|e|e.to_string())?;
        file.write_all(&bytes).and_then(|_|file.sync_all()).map_err(|e|e.to_string())?;
        Ok(companion::Request{id,name:"review_plan".into(),action:companion::Action::ReviewPlan(path),
            target:Some(companion::Target::Ssh{node_id:node.id.clone(),connection_sha256:companion::digest(node.connection_key().as_bytes())}),
            plan_sha256:Some(companion::digest(&bytes)),config_sha256:Some(config_sha256),review_id:None,review_sha256:None})
    }
    fn review_current_setup_on_node(&mut self)->Result<(),String>{
        if self.busy()||self.companion_waiting.is_some()||self.tui_map_waiting.is_some()
            ||!passive_companion_lane(self.companion_remote.as_ref().map(|p|&p.request.action),
                self.nodes.pending.as_ref().map(|p|&p.operation)){
            return Err("Another forecast action is in progress. Follow its status before reviewing this setup.".into());
        }
        let request=self.current_setup_review_request()?;
        self.tui_map_launch=None;self.tui_map_review=Some(request.clone());
        if self.companion_target()["capabilities"]["review_plan_v1"]==true{
            self.begin_companion_remote(request)?;
            self.node_panel.notice=if self.companion_waiting.is_some(){self.status.clone()}else{"Checking the current saved setup and GPU memory on the selected node…".into()};
        }else{
            if self.nodes.pending.is_none(){self.nodes.begin(remote::Operation::Probe,&self.python,&self.output,&self.cwd)?;}
            self.tui_map_waiting=Some(request);
            self.node_panel.notice="Connecting to the selected node, then reviewing the current saved setup…".into();
        }
        self.status=self.node_panel.notice.clone();self.dialog=Some(Dialog::Nodes);Ok(())
    }
    fn continue_current_setup_review(&mut self,update:&remote::Update){
        let Some(request)=self.tui_map_waiting.take()else{return;};
        if let remote::Update::Failed(error)=update{self.node_panel.error(error.clone());return;}
        let result=if self.checked_companion_target(request.target.as_ref()).is_err(){Err("The selected node changed while connecting. Review the current setup again.".into())}
            else if self.companion_target()["capabilities"]["review_plan_v1"]==true{self.begin_companion_remote(request)}
            else if matches!(update,remote::Update::Connected){Err("The connected node's runtime does not support staged map plans. Update its matched ArWen runtime.".into())}
            else{self.tui_map_waiting=Some(request);self.nodes.begin(remote::Operation::Probe,&self.python,&self.output,&self.cwd)};
        if let Err(error)=result{self.status=error.clone();self.node_panel.error(error);}
    }
    fn node_request(&mut self, operation: remote::Operation) {
        if self.companion_waiting.is_some(){self.node_panel.notice="A forecast action is waiting for the current node request and will continue automatically.".into();return;}
        if matches!(operation,remote::Operation::Start{preview:true,..})&&self.current_setup_uses_staged_node(){
            if let Err(error)=self.review_current_setup_on_node(){self.status=error.clone();self.node_panel.error(error);}
            return;
        }
        if matches!(operation,remote::Operation::StartPlan{..}){
            let result=self.tui_map_launch.clone().ok_or_else(||"This map review is no longer current. Review the saved setup again.".to_owned())
                .and_then(|request|self.begin_companion_remote(request));
            if let Err(error)=result{self.status=error.clone();self.node_panel.error(error);}
            return;
        }
        let action = operation.action();
        let mutates = operation.mutates();
        match self
            .nodes
            .begin(operation, &self.python, &self.output, &self.cwd)
        {
            Ok(()) => {
                self.status = format!(
                    "Contacting {}: {action}. {}",
                    self.nodes
                        .store
                        .selected()
                        .map(|n| n.host.as_str())
                        .unwrap_or("node"),
                    if mutates {
                        "If the connection is lost, refresh Jobs before retrying."
                    } else {
                        ""
                    }
                );
                self.node_panel.notice = self.status.clone();
            }
            Err(error) => {
                self.node_panel.error(error.clone());
                self.status = error;
            }
        }
    }
    fn node_intent(&mut self, intent: node_ui::Intent) {
        match intent {
            node_ui::Intent::Keep => {}
            node_ui::Intent::Close => {
                if self.nodes.pending.is_some() {
                    self.status = self.node_panel.notice.clone();
                }
                self.dialog = None;
            }
            node_ui::Intent::Plots => self.begin_plots(),
            node_ui::Intent::CopyLogs => {
                self.node_panel.notice=match self.copy_text(&self.node_log_text()){
                    Ok(bytes)=>format!("Copied node logs ({bytes} bytes). Paste into another app."),
                    Err(error)=>format!("Copy failed: {error}. O opens a plain-text log."),
                };
                self.status=self.node_panel.notice.clone();
            }
            node_ui::Intent::OpenLog => {
                self.node_panel.notice=self.open_log_text(&self.node_log_text());
                self.status=self.node_panel.notice.clone();
            }
            node_ui::Intent::Remove(id) => match self.nodes.remove_node(&id) {
                Ok(()) => {
                    self.node_panel.open(&self.nodes);
                    self.status = "Saved node removed. Remote jobs and files were not changed.".into();
                    self.node_panel.notice = self.status.clone();
                }
                Err(error) => {
                    self.status = error.clone();
                    self.node_panel.error(error);
                }
            },
            node_ui::Intent::Request(operation) => self.node_request(operation),
            node_ui::Intent::Select(id) => {
                let target=id.map(|id|self.nodes.store.nodes.iter().find(|node|node.id==id)
                    .map(|node|companion::Target::Ssh{node_id:node.id.clone(),connection_sha256:companion::digest(node.connection_key().as_bytes())})
                    .ok_or_else(||"The selected saved node no longer exists.".to_owned())).transpose();
                match target.and_then(|target|self.select_companion_target(&target.unwrap_or(companion::Target::Local))){
                    Ok(message)=>{self.status=message.clone();self.node_panel.notice=message;}
                    Err(error)=>{self.status=error.clone();self.node_panel.error(error);}
                }
            }
            node_ui::Intent::Save(node) => {
                let id = node.id.clone();
                match self
                    .nodes
                    .save_node(node)
                    .and_then(|_| self.nodes.select(Some(id)))
                {
                    Ok(()) => {
                        self.node_panel.open(&self.nodes);
                        self.node_panel.notice = "Node saved and selected. Connect checks its ArWen installation; Start reviews a new job.".into();
                    }
                    Err(error) => {
                        if let node_ui::Screen::Edit { editing, .. } = &mut self.node_panel.screen {
                            *editing = false;
                        }
                        self.status = error.clone();
                        self.node_panel.error(error);
                    }
                }
            }
            node_ui::Intent::ChooseJob(id) => {
                if let Err(error) = self.nodes.remember_job(&id) {
                    self.node_panel.error(error);
                }
                self.nodes.view.log.clear();
                self.nodes.view.cursor = 0;
                self.nodes.view.status = None;
                self.node_request(remote::Operation::Logs { job: id, cursor: 0 });
            }
            node_ui::Intent::Reload => match self.nodes.reload() {
                Ok(()) => self.node_panel.open(&self.nodes),
                Err(error) => self.node_panel.error(error),
            },
        }
    }
    fn node_action(&mut self, action: Action) -> bool {
        if self.nodes.store.selected().is_none() || matches!(action, Action::Doctor) {
            return false;
        }
        self.dialog = Some(Dialog::Nodes);
        if matches!(action, Action::Prepared)
            && self
                .nodes
                .store
                .selected()
                .is_some_and(|n| n.prepared.is_empty())
        {
            self.node_panel.notice = "Edit this node and set Prepared folder on node, then choose Start to review a run using those inputs.".into();
            return true;
        }
        let operation = match action {
            Action::Check => remote::Operation::Probe,
            Action::Plan | Action::Run | Action::Prepared => match self.plot_spec() {
                Ok(products) => remote::Operation::Start {
                    products,
                    preview: true,
                    binding: None,
                },
                Err(error) => {
                    self.node_panel.notice = error;
                    return true;
                }
            },
            Action::Doctor => unreachable!(),
        };
        self.node_request(operation);
        true
    }
    fn launch(&mut self, action: Action) {
        if self.node_action(action) {
            return;
        }
        if self.busy() {
            self.status =
                "A command is already running. Follow it in Logs, or press X to stop this run."
                    .into();
            return;
        }
        let (command, args) = match self.argv(action) {
            Ok(v) => v,
            Err(e) => {
                self.status = e;
                return;
            }
        };
        self.start_command(command, &args, None);
    }
    fn review_action(&mut self, action: Action) {
        if self.node_action(action) {
            return;
        }
        match self.argv(action) {
            Ok((command, args)) => {
                self.dialog = Some(Dialog::Review(
                    Request {
                        command: command.into(),
                        args,
                        created: None,
                        title: "Prepare and run",
                    },
                    None,
                    0,
                ))
            }
            Err(error) => self.status = error,
        }
    }
    fn start_command(&mut self, command: &str, args: &[String], created: Option<PathBuf>) {
        if self.nodes.store.selected().is_some()
            && matches!(command, "go" | "sim" | "check" | "resume" | "downscale")
        {
            self.status = "A node is selected. Use Nodes → Start, or select Local computer to run this local configuration.".into();
            return;
        }
        if self.busy() {
            self.status = "A command is already running. L follows its log.".into();
            return;
        }
        self.memory_recovery_config = None;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let directory = self
            .output
            .join(".arwen-tui")
            .join(format!("{stamp}-{}", std::process::id()));
        match Job::start(&self.python, command, args, &directory, &self.cwd) {
            Ok(job) => {
                self.startup_failure = None;
                self.pending_workflow = if created.is_some() {
                    self.guide_cache.as_ref().and_then(|guide| guide.workflow)
                } else {
                    None
                };
                self.pending_config = created;
                self.active_config = if matches!(
                    command,
                    "doctor" | "domain" | "research" | "run-plan" | "sources" | "case-catalog"
                ) {
                    None
                } else {
                    self.editor.as_ref().map(|e| e.path.clone())
                };
                self.status = format!(
                    "Started {command}. Full output is retained in {}",
                    display_path(&directory)
                );
                self.job = Some(job);
                self.tab = Tab::Logs;
                self.log_offset = 0;
                self.local_raw_logs = false;
            }
            Err(e) => {
                self.status =
                    format!("Could not start {command}: {e}. F9 changes the Python executable.");
                // Keep diagnostics tied to this attempt, never to a prior completed job.
                let path = directory.join("job.log");
                self.startup_failure = Some(StartupFailure {
                    message: self.status.clone(),
                    log: path.is_file().then_some(path),
                });
                self.job = None;
                self.active_config = None;
                self.pending_config = None;
                self.pending_workflow = None;
                self.tab = Tab::Logs;
                self.log_offset = 0;
                self.local_raw_logs = false;
            }
        }
    }
    fn poll(&mut self) {
        self.poll_companion_requests();
        self.poll_run_views();
        self.cds.poll();
        if self.tab == Tab::Overview || self.era5_provider().is_some()
            || matches!(self.dialog, Some(Dialog::CdsCredentials(_))) {
            self.cds.ensure(&self.python, &self.cwd);
        }
        self.plot_catalog.poll();
        if let Some(Dialog::Calendar(_, form)) = &mut self.dialog {
            form.poll();
        }
        if let Some(Dialog::Cases(form)) = &mut self.dialog {
            form.poll();
        }
        if let Some(update) = self.nodes.poll() {
            self.finish_companion_remote(&update);
            let current_setup = self.accept_setup_review_update(&update);
            if current_setup { self.node_panel.update(&update); }
            if matches!(update,remote::Update::Connected)
                &&self.nodes.store.selected().and_then(|node|node.last_job.as_ref()).is_some()
                &&matches!(self.node_panel.screen,node_ui::Screen::Nodes|node_ui::Screen::Jobs){
                self.node_panel.screen=node_ui::Screen::Job;
                self.node_panel.notice="Reconnecting to this node's saved forecast job…".into();
            }
            match &update {
                remote::Update::Preview { .. } if current_setup => self.status = "Node launch review is ready in Nodes.".into(),
                remote::Update::Started(id) => self.status = format!("Remote job {id} started. Nodes shows its status and log; closing the TUI leaves it running."),
                remote::Update::Stopped(id) => self.status = format!("Node confirmed termination of {id}."),
                remote::Update::Failed(error) => self.status = error.clone(),
                _ => {},
            }
            self.continue_current_setup_review(&update);
        }
        // Drain a captured interactive request before automatic log refresh can
        // occupy the serialized node channel again. Expiry is checked even
        // while an unresponsive read is still pending.
        self.continue_queued_companion_remote();
        if self.companion_waiting.is_none()&&(matches!(self.dialog, Some(Dialog::Nodes))||self.companion.session.is_some()) && self.node_panel.should_refresh_connected(&self.nodes,self.companion.session.is_some())
        {
            if let Some(job) = self.nodes.store.selected().and_then(|n| n.last_job.clone()) {
                self.node_request(remote::Operation::Logs {
                    job,
                    cursor: self.nodes.view.cursor,
                });
            }
        }
        let was_busy = self.busy();
        if let Some(job) = &mut self.job {
            if job.outcome.is_none() {
                match job.poll() {
                    Ok(Some(code)) => {
                        self.memory_recovery_config = if job.memory_refused {
                            self.active_config.clone()
                        } else {
                            None
                        };
                        self.status = if code == 0
                            && job.action == "go"
                            && job.command.iter().any(|a| a == "--dry-run")
                        {
                            "Launch plan ready. Next: F7 reviews the Prepare and run command, or V returns to your configuration.".into()
                        } else if code == 0 {
                            format!(
                                "{} completed. V returns to Overview; Ctrl+L shows details. Logs: {}",
                                job.action,
                                display_path(&job.dir)
                            )
                        } else if job.memory_refused {
                            "Memory admission refused. Fit domain can resize the grid; Tile streaming keeps its geometry and plans GPU tiles using system RAM. Open Details to compare and review.".into()
                        } else if job.interrupted() {
                            format!("{} INTERRUPTED (exit {code}). Click this summary or Ctrl+L for details. Partial output remains in the saved log folder.", job.action)
                        } else {
                            format!(
                                "{} FAILED (exit {code}). Click this summary or Ctrl+L for error details.",
                                job.action
                            )
                        }
                    }
                    Err(e) => self.status = format!("Could not read command status: {e}"),
                    _ => {}
                }
                if let Some(notice) = &job.completion_notice {
                    self.status.push_str(&format!(" {notice}"));
                }
            }
        }
        if self.job.as_ref().is_some_and(|job| job.outcome.is_some()) {
            self.active_config = None;
            if let Some(path) = self.pending_config.take() {
                let workflow = self.pending_workflow.take().and_then(workflows::mode);
                if self.job.as_ref().is_some_and(|job| job.outcome == Some(0)) {
                    self.open(path.clone());
                    self.guide_cache = None;
                    self.case_cache = None;
                    self.status="Configuration created. Next: Enter reviews the plan. E edits all TOML settings; nothing has been prepared or forecast yet.".into();
                    if self.job.as_ref().is_some_and(|job| job.action == "research") {
                        self.status = "Research configuration created. D reviews actual grids; Ctrl+L shows fitted coverage and movement limits. F6 reviews the plan; no forecast has started.".into();
                    }
                    if let Some(mode) = workflow {
                        match plotsettings::Selection::preset_id(mode.preset)
                            .and_then(|selection| plotsettings::save(&path, &selection, None)) {
                            Ok(()) => self.status = format!("{} setup created with its plot set. D shapes domains and tracking; F6 reviews the plan. No forecast has started.", mode.title),
                            Err(error) => self.status = format!("Configuration created; mode plots were not saved: {error}. B opens Plots for review."),
                        }
                    }
                } else if let Some(draft) = self.job.as_ref().and_then(Job::configuration_recovery) {
                    self.open_configuration_recovery(draft);
                } else if self.memory_recovery_available() {
                    self.status = "This memory plan does not fit. Your configuration is preserved. Ctrl+F reviews fitting its grids; Ctrl+T reviews tile streaming. Details explain the planner's limit.".into();
                    self.show_details();
                } else {
                    self.status="Configuration creation FAILED. Click this summary or Ctrl+L for the engine's error; Esc returns to your saved answers.".into();
                }
            }
        }
        if was_busy && !self.busy() && matches!(self.dialog, Some(Dialog::Details { .. })) {
            self.show_details();
        }
        if self.exit_after_job && !self.busy() {
            self.exit = true;
        }
        self.publish_companion_status(false);
    }
    fn open_configuration_recovery(&mut self, draft: PathBuf) {
        if self.dirty() {
            self.status = format!("Configuration exceeds GPU memory. Its recovery draft is saved at {}. Save your open edits before opening it to Fit domain or Tile streaming.", display_path(&draft));
            return;
        }
        self.open(draft.clone());
        if self.editor.as_ref().is_some_and(|editor| editor.path == draft) {
            self.memory_recovery_config = Some(draft);
            self.status = "Configuration exceeds GPU memory. Saved as a recovery draft; no runnable configuration was created. Ctrl+F fits its grids to this GPU; Ctrl+T reviews tile streaming while keeping its geometry.".into();
            self.show_details();
        }
    }
    fn view(&mut self, tab: Tab) {
        if tab == Tab::Logs && self.nodes.store.selected().is_some() {
            self.dialog = Some(Dialog::Nodes);
            if self
                .nodes
                .store
                .selected()
                .and_then(|n| n.last_job.as_ref())
                .is_some()
            {
                self.node_panel.screen = node_ui::Screen::Job;
            } else {
                self.node_request(remote::Operation::List);
            }
            return;
        }
        if tab == Tab::Overview && self.nodes.store.selected().is_some() {
            self.open_nodes();
            return;
        }
        if matches!(tab, Tab::Overview | Tab::Settings) && self.editor.is_none() {
            self.status = "Create or open a configuration first: choose New forecast or Open existing on Home.".into();
            self.tab = Tab::Home;
            self.selected = 0;
            return;
        }
        self.tab = tab;
        if tab == Tab::Home {
            self.selected = 0;
        }
        if tab == Tab::Overview {
            self.selected = 1;
        }
    }
    fn mouse(&mut self, event: MouseEvent) {
        if event.kind == MouseEventKind::Down(MouseButton::Left) {
            if let Some(action) = self.hits.iter().rev().find(|hit|
                matches!(hit.action, Hit::Quit | Hit::JobResult)
                && event.column >= hit.area.x && event.column < hit.area.right()
                && event.row >= hit.area.y && event.row < hit.area.bottom()).map(|hit| hit.action) {
                match action { Hit::Quit => self.request_quit(), Hit::JobResult => self.show_job_result(), _ => {} }
                self.hits.clear();
                return;
            }
        }
        if !self.input_enabled {
            return;
        }
        if matches!(self.dialog, Some(Dialog::Scenario(..))) {
            if let Some(Dialog::Scenario(mut form, original)) = self.dialog.take() {
                let intent = form.mouse(event);
                self.scenario_intent(form, original, intent);
            }
            return;
        }
        if matches!(self.dialog, Some(Dialog::Nodes)) {
            let products = self.plot_spec();
            let intent = self.node_panel.mouse(event, &self.nodes, products);
            self.node_intent(intent);
            return;
        }
        if matches!(
            event.kind,
            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown
        ) {
            let code = if event.kind == MouseEventKind::ScrollUp {
                KeyCode::Up
            } else {
                KeyCode::Down
            };
            self.key(KeyEvent::new(code, KeyModifiers::NONE));
            return;
        }
        if event.kind != MouseEventKind::Down(MouseButton::Left) {
            return;
        }
        let Some(hit) = self
            .hits
            .iter()
            .rev()
            .find(|h| {
                event.column >= h.area.x
                    && event.column < h.area.right()
                    && event.row >= h.area.y
                    && event.row < h.area.bottom()
            })
            .map(|h| h.action)
        else {
            return;
        };
        match hit {
            Hit::Workflow(index) => {
                if let Some(Dialog::Workflows(browser)) = &mut self.dialog {
                    browser.choose(index);
                }
            }
            Hit::Research(index) => {
                if let Some(Dialog::Workflows(browser)) = &mut self.dialog {
                    browser.choose_row(index);
                }
            }
            Hit::Details => self.show_details(),
            Hit::FitCurrent => self.fit_current(),
            Hit::TileCurrent => self.tile_current(),
            Hit::Domains => self.begin_domains(),
            Hit::Plots => self.begin_plots(),
            Hit::Nodes => self.open_nodes(),
            Hit::PlotItem(index) => {
                if let Some(Dialog::Plots(form)) = &mut self.dialog {
                    form.choose(index, &self.plot_catalog);
                }
            }
            Hit::DomainSelect(index) => {
                if let Some(Dialog::Domains(selected)) = &mut self.dialog { *selected = index; }
            }
            Hit::Era5Provider => self.begin_era5_provider(),
            Hit::CdsCredentials => self.begin_cds_credentials(),
            Hit::Quit => self.request_quit(),
            Hit::JobResult => self.show_job_result(),
            Hit::OpenCompanion => self.open_companion(),
            Hit::DomainField(index) => {
                if let Some(Dialog::DomainForm(form, editing)) = &mut self.dialog {
                    form.selected = index;
                    *editing = Some(index);
                }
            }
            Hit::DomainValue(index) => {
                if let Some(Dialog::DomainForm(form, Some(field))) = &mut self.dialog {
                    if let Some(value) = form.fields[*field].choices().get(index) {
                        form.fields[*field].value = (*value).into();
                    }
                }
            }
            Hit::GuideGrid(step) => {
                if let Some(Dialog::Guide(mut guide) | Dialog::Summary(mut guide, _)) =
                    self.dialog.take()
                {
                    guide.step = step;
                    guide.summary_edit = true;
                    self.dialog = Some(Dialog::Guide(guide));
                }
            }
            Hit::CalendarDay(day) => {
                if let Some(Dialog::Calendar(_, form)) = &mut self.dialog { form.day(day); }
            }
            Hit::CalendarHour(hour) => {
                if let Some(Dialog::Calendar(_, form)) = &mut self.dialog { form.hour(hour); }
            }
            Hit::CaseItem(index) => {
                if let Some(Dialog::Cases(form)) = &mut self.dialog { form.select(index); }
            }
            Hit::Key(code, modifiers) => self.key(KeyEvent::new(code, modifiers)),
            Hit::View(tab) => self.view(tab),
            Hit::Home(index) => {
                self.selected = index;
                self.home_choice();
            }
            Hit::Action(index) => {
                self.selected = index;
                self.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
            }
            Hit::Choice(index) | Hit::Summary(index) | Hit::Browser(index) => {
                match self.dialog.as_mut() {
                    Some(Dialog::Choice(_, selected))
                    | Some(Dialog::Summary(_, selected))
                    | Some(Dialog::Browser(_, _, selected))
                    | Some(Dialog::DroppedFiles(_, selected))
                    | Some(Dialog::Domains(selected))
                    | Some(Dialog::Era5Provider(selected, _))
                    | Some(Dialog::DomainMenu(_, selected)) => *selected = index,
                    _ => return,
                }
                self.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
            }
            Hit::Editor(rect) => {
                if let Some(e) = &mut self.editor {
                    e.row = (e.top + event.row.saturating_sub(rect.y) as usize)
                        .min(e.lines.len().saturating_sub(1));
                    let target = e.left + event.column.saturating_sub(rect.x + 7) as usize;
                    e.col = Editor::character_at_display_column(&e.lines[e.row], target);
                }
            }
        }
        // A changed dialog/tab must never accept coordinates from the old drawing.
        self.hits.clear();
    }
    fn key(&mut self, key: KeyEvent) {
        if key.kind == KeyEventKind::Release {
            return;
        }
        let mut key = key;
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        // Escape and safety controls work before modal routing and at any size.
        if ctrl && matches!(key.code, KeyCode::Char('q' | 'Q')) {
            self.request_quit();
            return;
        }
        if ctrl && matches!(key.code, KeyCode::Char('c' | 'C')) {
            if self.busy() {
                if !matches!(self.dialog, Some(Dialog::Stop)) { self.overlay(Dialog::Stop); }
            } else {
                self.request_quit();
            }
            return;
        }
        if key.code == KeyCode::F(1)
            || (key.code == KeyCode::Char('?') && (self.dialog.is_some() || self.tab != Tab::Settings))
        {
            if matches!(self.dialog, Some(Dialog::Help(_))) {
                self.close_overlay();
            } else {
                self.overlay(Dialog::Help(HelpScroll::default()));
            }
            return;
        }
        if !self.input_enabled {
            if matches!(self.dialog, Some(Dialog::Quit | Dialog::Stop | Dialog::Help(_)))
                && matches!(key.code, KeyCode::Esc | KeyCode::Char('y' | 'Y' | 'n' | 'N'))
            {
                let dialog = self.dialog.take().unwrap();
                self.dialog_key(dialog, key);
            }
            return;
        }
        if ctrl && self.memory_recovery_available()
            && matches!(self.dialog, None | Some(Dialog::Details { .. }))
        {
            match key.code {
                KeyCode::Char('f' | 'F') => { self.fit_current(); return; }
                KeyCode::Char('t' | 'T') => { self.tile_current(); return; }
                _ => {}
            }
        }
        // Text entry preserves case. Workspace shortcuts accept either case.
        if self.dialog.is_none() && self.tab != Tab::Settings {
            if let KeyCode::Char(c) = key.code {
                key.code = KeyCode::Char(c.to_ascii_lowercase());
            }
        }
        if let Some(dialog) = self.dialog.take() {
            self.dialog_key(dialog, key);
            return;
        }
        if self.tab == Tab::Overview && ctrl && matches!(key.code, KeyCode::Char('z' | 'Z' | 'y' | 'Y')) {
            if let Some(editor) = &mut self.editor {
                let undo = matches!(key.code, KeyCode::Char('z' | 'Z'));
                let changed = if undo { editor.undo() } else { editor.redo() };
                self.status = if changed {
                    if undo { "Undid the last draft edit." } else { "Redid the draft edit." }
                } else { "No draft edit to restore." }.into();
            }
            return;
        }
        if self.tab != Tab::Settings && key.code == KeyCode::Char('u') && !ctrl {
            self.begin_era5_provider();
            return;
        }
        if (ctrl && matches!(key.code, KeyCode::Char('r' | 'R')))
            || (self.tab != Tab::Settings && key.code == KeyCode::Char('r'))
        {
            self.open_nodes();
            return;
        }
        if ctrl && matches!(key.code, KeyCode::Char('l' | 'L')) {
            self.show_details();
            return;
        }
        if ctrl && matches!(key.code, KeyCode::Char('s' | 'S')) {
            if key.modifiers.contains(KeyModifiers::SHIFT) {
                self.export_dialog();
            } else {
                self.save();
            }
            return;
        }
        if ctrl && key.code == KeyCode::Char('o') {
            self.dialog = Some(Dialog::Path("Configuration path", String::new()));
            return;
        }
        if (ctrl && matches!(key.code, KeyCode::Char('d' | 'D')))
            || (self.tab != Tab::Settings && key.code == KeyCode::Char('d'))
        {
            self.begin_domains();
            return;
        }
        if (ctrl && matches!(key.code, KeyCode::Char('p' | 'P')))
            || (self.tab != Tab::Settings && key.code == KeyCode::Char('b'))
        {
            self.begin_plots();
            return;
        }
        if !ctrl && self.tab == Tab::Logs && key.code == KeyCode::Char('g') && self.has_local_forecast_job() {
            self.local_raw_logs = !self.local_raw_logs;
            self.log_offset = 0;
            return;
        }
        if (ctrl && matches!(key.code, KeyCode::Char('g' | 'G')))
            || (self.tab != Tab::Settings && key.code == KeyCode::Char('g'))
        {
            self.dialog = Some(Dialog::Path(
                "Geography folder",
                display_path(&self.geog_root),
            ));
            return;
        }
        if self.tab != Tab::Settings {
            match key.code {
                KeyCode::Char('k') => {
                    self.begin_cases();
                    return;
                }
                KeyCode::Char('i') => {
                    self.begin_scenario();
                    return;
                }
                KeyCode::Char('w') => {
                    self.begin_workflows();
                    return;
                }
                KeyCode::Char('f' | 'e') => {
                    self.view(Tab::Settings);
                    return;
                }
                KeyCode::Char('v') => {
                    self.view(Tab::Overview);
                    return;
                }
                KeyCode::Char('l') => {
                    self.view(Tab::Logs);
                    return;
                }
                KeyCode::Char('?') => {
                    self.dialog = Some(Dialog::Help(HelpScroll::default()));
                    return;
                }
                _ => {}
            }
        }
        match key.code {
            KeyCode::F(1) => {
                self.dialog = Some(Dialog::Help(HelpScroll::default()));
                return;
            }
            KeyCode::F(2) => {
                self.browse(self.cwd.clone());
                return;
            }
            KeyCode::F(3) => {
                self.dialog = Some(Dialog::Path(
                    "Output folder",
                    self.output.to_string_lossy().into(),
                ));
                return;
            }
            KeyCode::F(4) => {
                self.dialog = Some(Dialog::Path(
                    "Prepared folder",
                    self.prepared.to_string_lossy().into(),
                ));
                return;
            }
            KeyCode::F(5) => {
                self.launch(Action::Check);
                return;
            }
            KeyCode::F(6) => {
                self.launch(Action::Plan);
                return;
            }
            KeyCode::F(7) => {
                self.review_action(Action::Run);
                return;
            }
            KeyCode::F(8) => {
                self.review_action(Action::Prepared);
                return;
            }
            KeyCode::F(9) => {
                self.dialog = Some(Dialog::Path(
                    "Python executable",
                    self.python.to_string_lossy().into(),
                ));
                return;
            }
            KeyCode::F(11) => {
                self.dialog = Some(Dialog::Path(
                    "Geography folder",
                    display_path(&self.geog_root),
                ));
                return;
            }
            KeyCode::F(12) => {
                self.export_dialog();
                return;
            }
            KeyCode::F(10) => {
                self.launch(Action::Doctor);
                return;
            }
            KeyCode::Esc => {
                if let Some(guide) = self.guide_cache.take() {
                    self.dialog = Some(Dialog::Guide(guide));
                } else if let Some(form) = self.case_cache.take() {
                    self.dialog = Some(Dialog::Cases(form));
                } else if let Some(guide) = self.saved_guides.pop() {
                    self.dialog = Some(Dialog::Guide(guide));
                } else {
                    self.tab = if self.editor.is_some() {
                        Tab::Overview
                    } else {
                        Tab::Home
                    };
                }
                return;
            }
            _ => {}
        }
        if self.tab == Tab::Settings {
            if let Some(e) = &mut self.editor {
                if ctrl {
                    self.status = match key.code {
                        KeyCode::Char('z' | 'Z') => if e.undo() { "Undid the last draft edit." } else { "No earlier draft edit to undo." },
                        KeyCode::Char('y' | 'Y') => if e.redo() { "Redid the draft edit." } else { "No draft edit to redo." },
                        KeyCode::Char('u' | 'U') => if e.discard_draft() { "Restored the saved configuration. Ctrl+Z can undo this discard." } else { "The draft already matches the saved configuration." },
                        _ => return,
                    }.into();
                } else {
                    e.key(key.code);
                }
            }
            return;
        }
        if key.code == KeyCode::Char('h') {
            self.tab = Tab::Home;
            self.selected = 0;
            return;
        }
        if self.tab == Tab::Home {
            match key.code {
                KeyCode::Up => self.selected = self.selected.saturating_sub(1),
                KeyCode::Down => self.selected = (self.selected + 1).min(4),
                KeyCode::Enter => self.home_choice(),
                KeyCode::Char('n') => self.begin_workflows(),
                KeyCode::Char('t') => self.begin_guide(Kind::Fit),
                KeyCode::Char('k') => self.begin_cases(),
                KeyCode::Char('o') => self.dialog = Some(Dialog::Choice(false, 0)),
                KeyCode::Char('c') => self.dialog = Some(Dialog::Choice(true, 0)),
                KeyCode::Char('s') => self.start_command("sources", &[], None),
                KeyCode::Char('p') => {
                    self.start_command("run-plan", &["--physics-profiles".into()], None)
                }
                KeyCode::Char('v') if self.editor.is_some() => {
                    self.tab = Tab::Overview;
                    self.selected = 1;
                }
                KeyCode::Char('q') => {
                    self.request_quit();
                }
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Char('q') => {
                self.request_quit();
            }
            KeyCode::Char('y' | 'Y') if self.tab==Tab::Logs=>{
                self.status=match self.copy_text(&self.log_text()){
                    Ok(bytes)=>format!("Copied logs ({bytes} bytes). Paste into another app."),
                    Err(error)=>format!("Copy failed: {error}. O opens a plain-text log."),
                };
            }
            KeyCode::Char('o' | 'O') if self.tab==Tab::Logs=>self.status=self.open_log_text(&self.log_text()),
            KeyCode::Char('o') => {
                self.dialog = Some(Dialog::Path("Configuration path", String::new()))
            }
            KeyCode::Char('e') => {
                if self.editor.is_some() {
                    self.tab = Tab::Settings
                } else {
                    self.status = "Open a configuration first (O).".into()
                }
            }
            KeyCode::Char('l') => self.view(Tab::Logs),
            KeyCode::Char('v') => self.tab = Tab::Overview,
            KeyCode::Char('x') if self.nodes.store.selected().is_some() => {
                self.dialog = Some(Dialog::Nodes);
                if let Some(job) = self.nodes.store.selected().and_then(|n| n.last_job.clone()) {
                    self.node_panel.screen = node_ui::Screen::Stop { job };
                } else {
                    self.node_panel.notice = "Choose a job from the node's Jobs list first.".into();
                }
            }
            KeyCode::Char('x') if self.busy() => self.dialog = Some(Dialog::Stop),
            KeyCode::Char('?') => self.dialog = Some(Dialog::Help(HelpScroll::default())),
            KeyCode::Tab => {
                self.tab = if self.tab == Tab::Overview {
                    Tab::Logs
                } else {
                    Tab::Overview
                }
            }
            KeyCode::Up if self.tab == Tab::Overview => {
                self.selected = self.selected.saturating_sub(1)
            }
            KeyCode::Down if self.tab == Tab::Overview => {
                self.selected = (self.selected + 1).min(ACTIONS.len() - 1)
            }
            KeyCode::Enter if self.tab == Tab::Overview => {
                let action = ACTIONS[self.selected].1;
                if matches!(action, Action::Run | Action::Prepared) {
                    self.review_action(action)
                } else {
                    self.launch(action)
                }
            }
            KeyCode::Up | KeyCode::PageUp if self.tab == Tab::Logs => self.log_offset += 10,
            KeyCode::Down | KeyCode::PageDown if self.tab == Tab::Logs => {
                self.log_offset = self.log_offset.saturating_sub(10)
            }
            KeyCode::End if self.tab == Tab::Logs => self.log_offset = 0,
            _ => {}
        }
    }
    fn home_choice(&mut self) {
        match self.selected {
            0 => self.begin_workflows(),
            1 => self.dialog = Some(Dialog::Choice(false, 0)),
            2 => self.dialog = Some(Dialog::Choice(true, 0)),
            3 => self.begin_guide(Kind::Fit),
            _ => self.begin_cases(),
        }
    }
    fn begin_cases(&mut self) {
        self.begin_cases_from(None);
    }
    fn begin_cases_from(&mut self, catalog: Option<PathBuf>) {
        if self.nodes.store.selected().is_some() {
            self.open_nodes();
            self.node_panel.notice = "Choose Local computer to create a local configuration from a case catalog.".into();
            return;
        }
        if self.dirty() {
            self.status = "Save the current draft with Ctrl+S or F12 before creating a case configuration.".into();
            return;
        }
        self.guide_cache = None;
        self.dialog = Some(Dialog::Cases(match catalog {
            Some(path) => cases::Form::from_catalog(&self.python, &self.cwd, &path),
            None => self.case_cache.take().unwrap_or_else(|| cases::Form::new(&self.python, &self.cwd)),
        }));
        self.status = "Browse case catalogs, inspect their source and physics choices, then create editable TOML.".into();
    }
    fn begin_workflows(&mut self) {
        self.dialog = Some(Dialog::Workflows(workflows::Browser::default()));
        self.status =
            "Choose a weather task. Each mode opens an editable setup and a reviewed command."
                .into();
    }
    fn workflow_route(&mut self, index: usize, route: workflows::Route) {
        let Some(mode) = workflows::MODES.get(index) else {
            return;
        };
        match route {
            workflows::Route::New | workflows::Route::Downscale => {
                let fresh = self.begin_guide_named(if route == workflows::Route::New {
                    Kind::New
                } else {
                    Kind::Downscale
                }, Some(mode.id), None);
                if let Some(Dialog::Guide(guide)) = &mut self.dialog {
                    if fresh { guide.apply_workflow(mode, &self.cwd); }
                    self.status = if route == workflows::Route::New {
                        format!(
                            "{} · {}. All choices are reviewed before a command starts.",
                            mode.title, mode.setup
                        )
                    } else {
                        "Downscale actual archived history: review parent physics, boundary cadence and target GPU capacity. Planning is the default action.".into()
                    };
                }
            }
            workflows::Route::CurrentPlots => {
                // Preserve the sidecar's optimistic-concurrency evidence and ask
                // for the ordinary plot review before changing an existing setup.
                self.begin_plots();
                if let Some(Dialog::Plots(form)) = &mut self.dialog {
                    if !form.has_error() {
                        if let Some(index) = plotsettings::presets()
                            .iter()
                            .position(|preset| preset.id == mode.preset)
                        {
                            form.choose(index, &self.plot_catalog);
                        }
                    }
                }
            }
            workflows::Route::Tracking => {
                self.begin_domains();
                self.status = "Select an existing child domain, then Follow a storm. Review the tracking field, search region, units and corridor before saving.".into();
            }
            workflows::Route::Open => {
                self.dialog = Some(Dialog::Path("Configuration path", String::new()))
            }
            workflows::Route::Render => {
                let fresh = self.begin_guide_named(Kind::Render, Some(mode.id), None);
                if let Some(Dialog::Guide(guide)) = &mut self.dialog {
                    if fresh { guide.apply_workflow(mode, &self.cwd); }
                }
            }
            workflows::Route::Scenario => self.begin_scenario(),
        }
    }
    fn research_route(&mut self, index: usize, route: workflows::Route) {
        let Some(row) = research::config(index) else {
            return;
        };
        let Some(family) = research::family_for_config(row)
            .and_then(|id| workflows::MODES.iter().position(|m| m.id == id))
        else {
            return;
        };
        match route {
            workflows::Route::New | workflows::Route::Downscale | workflows::Route::Render => {
                let kind = match route {
                    workflows::Route::New => Kind::Research,
                    workflows::Route::Downscale => Kind::Downscale,
                    _ => Kind::Render,
                };
                let fresh = self.begin_guide_named(kind, None, Some(research::text(row, "id")));
                if let Some(Dialog::Guide(guide)) = &mut self.dialog {
                    if fresh { guide.apply_research(row, &self.cwd); }
                    self.status = format!("{} · {}. Review the resource choices and exact command; creation does not start a forecast.", research::text(row, "title"), research::method(row));
                }
            }
            workflows::Route::CurrentPlots => {
                self.begin_plots();
                if let Some(Dialog::Plots(form)) = &mut self.dialog {
                    if !form.has_error() {
                        form.selection = plotsettings::Selection {
                            label: research::text(row, "title").into(),
                            spec: research::strings(row, "diagnostics").join(","),
                        };
                        form.mode = plotsettings::Mode::Review;
                        form.selected = 0;
                        form.notice = "Review this experiment's requested diagnostics before saving. Missing history fields are reported by the native renderer.".into();
                    }
                }
            }
            workflows::Route::Open => {
                self.dialog = Some(Dialog::Path("Configuration path", String::new()));
                self.status = format!("{} requires your supplied scenario state. Open its actual TOML, then review Domains, Tracking and the requested plots against this research question.", research::text(row, "title"));
            }
            _ => self.workflow_route(family, route),
        }
    }
    fn begin_scenario(&mut self) {
        if self.nodes.store.selected().is_some() {
            self.open_nodes();
            self.node_panel.notice = "Initial-state edits apply to a local TOML draft. Select Local computer and open the scenario configuration to edit it.".into();
            return;
        }
        let Some(editor) = &self.editor else {
            self.dialog = Some(Dialog::Path("Configuration path", String::new()));
            self.status =
                "Open the scenario's TOML, then I opens its initial-state controls.".into();
            return;
        };
        let original = editor.text();
        match scenario::Form::new(original.clone()) {
            Ok(form) => self.dialog = Some(Dialog::Scenario(form, original)),
            Err(error) => {
                self.status = format!(
                    "Cannot open initial-state controls: {error}. F opens complete settings."
                )
            }
        }
    }
    fn scenario_intent(
        &mut self,
        form: scenario::Form,
        original: String,
        intent: scenario::Intent,
    ) {
        match intent {
            scenario::Intent::Keep => self.dialog = Some(Dialog::Scenario(form, original)),
            scenario::Intent::Cancel => self.dialog = None,
            scenario::Intent::Apply(text) => {
                if let Some(editor) = &mut self.editor {
                    if editor.text() == original {
                        editor.replace_draft(text);
                        self.status = "Initial-state changes applied to your draft. F12 saves a separate scenario; F5 checks it before any preparation or run.".into();
                        self.dialog = None;
                        self.tab = Tab::Overview;
                    } else {
                        self.status = "The draft changed while Scenario Lab was open. Close and reopen the form before applying.".into();
                        self.dialog = Some(Dialog::Scenario(form, original));
                    }
                }
            }
        }
    }
    fn begin_domains(&mut self) {
        self.status = "Select a domain. Edit, add a child, or remove it from the draft.".into();
        self.dialog = Some(Dialog::Domains(0));
    }
    fn edit_domain(&mut self, index: usize, section: domains::Section) {
        let form = self.editor.as_ref().ok_or_else(|| "Open a configuration first.".to_owned())
            .and_then(|editor| domains::Form::new(editor.text(), index, section));
        match form {
            Ok(form) => self.dialog = Some(Dialog::DomainForm(form, None)),
            Err(error) => {
                self.status = error;
                self.dialog = Some(Dialog::Domains(index));
            }
        }
    }
    fn domain_labels(&self) -> Result<Vec<String>, String> {
        match &self.editor {
            Some(editor) => domains::labels(&editor.text()),
            None => Ok(Vec::new()),
        }
    }
    fn begin_guide(&mut self, kind: Kind) {
        self.begin_guide_named(kind, None, None);
    }
    fn begin_guide_named(&mut self, kind: Kind, workflow: Option<&'static str>, research: Option<&'static str>) -> bool {
        if self.nodes.store.selected().is_some() {
            self.open_nodes();
            self.node_panel.notice = "A Linux node is selected. Start uses its saved remote configuration; Jobs → Resume continues a node job. Choose Local computer to create or run a local configuration.".into();
            return false;
        }
        if self.dirty() {
            self.status =
                "Save the current draft with Ctrl+S or F12 before starting another workflow."
                    .into();
            return false;
        }
        if let Some(index) = self.saved_guides.iter().position(|saved|
            saved.kind == kind && saved.workflow == workflow && saved.research == research)
        {
            self.dialog = Some(Dialog::Guide(self.saved_guides.remove(index)));
            self.status = "Resumed your retained setup answers. Continue editing or review all settings with Ctrl+A.".into();
            return false;
        }
        let mut guide = Guide::new(kind, &self.cwd, &self.output);
        guide.workflow = workflow;
        guide.research = research;
        if kind == Kind::New {
            let mut query = std::process::Command::new(&self.python);
            query
                .args([
                    "-c",
                    "from gpuwm.domain_interactive import DEFAULT_SOURCE; print(DEFAULT_SOURCE)",
                ])
                .current_dir(&self.cwd);
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                query.creation_flags(0x08000000);
            }
            if let Ok(result) = query.output() {
                if result.status.success() {
                    let value = String::from_utf8_lossy(&result.stdout).trim().to_string();
                    if !value.is_empty() && !value.contains(char::is_whitespace) {
                        guide.questions[7].value = value;
                    }
                }
            }
        }
        self.status = "Fill in the fields, then review the command.".into();
        self.dialog = Some(Dialog::Guide(guide));
        true
    }
    fn dialog_key(&mut self, dialog: Dialog, key: KeyEvent) {
        let dialog = match dialog {
            Dialog::Cases(mut form) => {
                match form.key(key) {
                    cases::Intent::Keep => self.dialog = Some(Dialog::Cases(form)),
                    cases::Intent::Close => self.dialog = None,
                    cases::Intent::Review(request) => {
                        self.case_cache = Some(form);
                        self.dialog = Some(Dialog::Review(request, None, 0));
                    }
                }
                return;
            }
            Dialog::Scenario(mut form, original) => {
                let intent = form.key(key);
                self.scenario_intent(form, original, intent);
                return;
            }
            Dialog::Workflows(mut browser) => {
                match browser.key(key) {
                    workflows::Intent::Keep => self.dialog = Some(Dialog::Workflows(browser)),
                    workflows::Intent::Close => self.dialog = None,
                    workflows::Intent::Start(index, route) => self.workflow_route(index, route),
                    workflows::Intent::StartConfig(index, route) => {
                        self.research_route(index, route)
                    }
                }
                return;
            }
            Dialog::Nodes => {
                self.dialog = Some(Dialog::Nodes);
                let products = self.plot_spec();
                let intent = self.node_panel.key(key, &self.nodes, products);
                self.node_intent(intent);
                return;
            }
            Dialog::Plots(mut form) => {
                match form.key(key, &self.plot_catalog) {
                    plotsettings::Intent::Cancel => {
                        if self.plots_from_nodes {
                            self.dialog = Some(Dialog::Nodes);
                        }
                    }
                    plotsettings::Intent::Keep => self.dialog = Some(Dialog::Plots(form)),
                    plotsettings::Intent::Save => {
                        if self.plots_from_nodes {
                            let result = self
                                .nodes
                                .store
                                .selected()
                                .cloned()
                                .ok_or("Choose a node first.".to_string())
                                .and_then(|mut node| {
                                    node.plot_label = Some(form.selection.label.clone());
                                    node.plot_products = Some(form.selection.spec.clone());
                                    self.nodes.save_node(node)
                                });
                            match result {
                                Ok(()) => {
                                    self.status = format!(
                                        "Saved {} for the selected node.",
                                        form.selection.summary()
                                    );
                                    self.node_panel.notice = self.status.clone();
                                    self.dialog = Some(Dialog::Nodes);
                                }
                                Err(error) => {
                                    form.notice = error;
                                    self.dialog = Some(Dialog::Plots(form));
                                }
                            }
                            return;
                        }
                        let result = self
                            .editor
                            .as_ref()
                            .ok_or("No configuration is open.".to_owned())
                            .and_then(|editor| {
                                plotsettings::save(
                                    &editor.path,
                                    &form.selection,
                                    form.original.as_deref(),
                                )
                            });
                        match result {
                            Ok(()) => {
                                self.status = format!("Saved {}. Plot settings: {}. Future TUI Plan/Run launches use this request; no forecast or rendering started.",
                                    form.selection.summary(), display_path(&plotsettings::sidecar(&self.editor.as_ref().unwrap().path)));
                            }
                            Err(error) => {
                                form.notice = error;
                                self.dialog = Some(Dialog::Plots(form));
                            }
                        }
                    }
                }
                return;
            }
            other => other,
        };
        if key.code == KeyCode::Esc {
            self.dialog = match dialog {
                Dialog::DomainMenu(index, _) => Some(Dialog::Domains(index)),
                Dialog::DomainRemoval(removal, _) => Some(Dialog::Domains(removal.index)),
                Dialog::CdsCredentials(mut form) if form.editing => {
                    form.editing = false;
                    Some(Dialog::CdsCredentials(form))
                }
                Dialog::CdsCredentials(_) => self.dialog_stack.pop(),
                Dialog::DomainForm(form, Some(_)) => Some(Dialog::DomainForm(form, None)),
                Dialog::DomainForm(form, None) if matches!(form.section, domains::Section::Geometry | domains::Section::AddChild) => Some(Dialog::Domains(form.index)),
                Dialog::DomainForm(form, None) => Some(Dialog::DomainMenu(form.index, 0)),
                Dialog::Guide(g) if g.summary_edit => Some(Dialog::Summary(g.clone(), g.step + 1)),
                Dialog::Guide(g) => { self.retain_guide(g); None },
                Dialog::Calendar(g, _) => Some(Dialog::Guide(g)),
                Dialog::Summary(mut g, _) => {
                    g.summary_edit = false;
                    Some(Dialog::Guide(g))
                }
                Dialog::Review(_, Some(g), _) => Some(Dialog::Summary(g, 0)),
                Dialog::Review(request, None, _) if request.command == "case-catalog" => self.case_cache.take().map(Dialog::Cases),
                Dialog::Quit => { self.exit_after_job = false; self.dialog_stack.pop() },
                Dialog::Help(_) | Dialog::Stop => self.dialog_stack.pop(),
                Dialog::Details { .. } if !self.dialog_stack.is_empty() => self.dialog_stack.pop(),
                _ => None,
            };
            return;
        }
        match dialog {
            Dialog::Details {
                text,
                mut offset,
                mut notice,
            } => {
                match key.code {
                    KeyCode::Char('c' | 'C') => notice = Some(self.copy_diagnostic(&text, false)),
                    KeyCode::Char('y' | 'Y') => notice = Some(self.copy_diagnostic(&text, true)),
                    KeyCode::Up => offset = offset.saturating_sub(1),
                    KeyCode::Down => offset = offset.saturating_add(1),
                    KeyCode::PageUp => offset = offset.saturating_sub(10),
                    KeyCode::PageDown => offset = offset.saturating_add(10),
                    KeyCode::Home => offset = 0,
                    KeyCode::End => offset = usize::MAX,
                    KeyCode::Char('l' | 'L') if self.log_path().is_some() => {
                        self.local_raw_logs = true;
                        self.view(Tab::Logs);
                        self.log_offset = 0;
                        return;
                    }
                    _ => {}
                }
                self.dialog = Some(Dialog::Details {
                    text,
                    offset,
                    notice,
                });
            }
            Dialog::Domains(mut selected) => {
                let count = self.domain_labels().map(|rows| rows.len()).unwrap_or(0);
                match key.code {
                    KeyCode::Char('s' | 'S') if key.modifiers.contains(KeyModifiers::CONTROL) => self.save(),
                    KeyCode::Char('z' | 'Z') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        if let Some(editor) = &mut self.editor {
                            self.status = if editor.undo() { "Undid the last draft edit." } else { "No earlier draft edit to undo." }.into();
                        }
                        let count = self.domain_labels().map(|rows| rows.len()).unwrap_or(0);
                        selected = selected.min(count.saturating_sub(1));
                    }
                    KeyCode::Up => selected = selected.saturating_sub(1),
                    KeyCode::Down => selected = (selected + 1).min(count + 1),
                    KeyCode::Enter | KeyCode::Char('e' | 'E') => {
                        if selected < count {
                            self.edit_domain(selected, domains::Section::Geometry);
                        } else {
                            self.begin_guide(if selected == count {
                                Kind::New
                            } else {
                                Kind::Downscale
                            });
                        }
                        return;
                    }
                    KeyCode::Char('a' | 'A') if selected < count => {
                        self.edit_domain(selected, domains::Section::AddChild);
                        return;
                    }
                    KeyCode::Char('m' | 'M') if selected < count => {
                        self.dialog = Some(Dialog::DomainMenu(selected, 0));
                        return;
                    }
                    KeyCode::Delete | KeyCode::Char('r' | 'R') if selected < count => {
                        if let Some(editor) = &self.editor {
                            match domains::Removal::new(editor.text(), selected) {
                                Ok(removal) => {
                                    self.dialog = Some(Dialog::DomainRemoval(removal, 0));
                                    return;
                                }
                                Err(error) => self.status = error,
                            }
                        }
                    }
                    _ => {}
                }
                self.dialog = Some(Dialog::Domains(selected));
            }
            Dialog::DomainMenu(index, mut selected) => {
                match key.code {
                    KeyCode::Up => selected = selected.saturating_sub(1),
                    KeyCode::Down => selected = (selected + 1).min(domains::SECTIONS.len() - 1),
                    KeyCode::Enter => {
                        if let Some(editor) = &self.editor {
                            match domains::Form::new(
                                editor.text(),
                                index,
                                domains::SECTIONS[selected].0,
                            ) {
                                Ok(form) => {
                                    self.dialog = Some(Dialog::DomainForm(form, None));
                                    return;
                                }
                                Err(error) => self.status = error,
                            }
                        }
                    }
                    _ => {}
                }
                self.dialog = Some(Dialog::DomainMenu(index, selected));
            }
            Dialog::DomainRemoval(removal, mut offset) => {
                match key.code {
                    KeyCode::Up => offset = offset.saturating_sub(1),
                    KeyCode::Down => offset = offset.saturating_add(1),
                    KeyCode::PageUp => offset = offset.saturating_sub(8),
                    KeyCode::PageDown => offset = offset.saturating_add(8),
                    KeyCode::Home => offset = 0,
                    KeyCode::F(2) => {
                        if let Some(editor) = &mut self.editor {
                            if editor.text() != removal.original {
                                self.status = "The draft changed. Reopen Domains before removing anything.".into();
                            } else {
                                editor.replace_draft(removal.apply());
                                self.status = format!("Removed {} domain(s) from the draft. Ctrl+Z undoes; Ctrl+S saves.", removal.rows.len());
                                let count = self.domain_labels().map(|rows| rows.len()).unwrap_or(1);
                                self.dialog = Some(Dialog::Domains(removal.index.min(count.saturating_sub(1))));
                                return;
                            }
                        }
                    }
                    _ => {}
                }
                self.dialog = Some(Dialog::DomainRemoval(removal, offset));
            }
            Dialog::Era5Provider(mut selected, original) => {
                match key.code {
                    KeyCode::Char('c' | 'C') => {
                        self.dialog = Some(Dialog::Era5Provider(selected, original));
                        self.begin_cds_credentials();
                        return;
                    }
                    KeyCode::Up => selected = selected.saturating_sub(1),
                    KeyCode::Down => selected = (selected + 1).min(1),
                    KeyCode::Enter => {
                        let (provider, label) = if selected == 0 { ("arco", "Google ARCO") } else { ("cds", "Copernicus CDS") };
                        match self.set_era5_provider(provider, &original) {
                            Ok(()) => {
                                self.status = format!("ERA5: {label}. Ctrl+S saves; Ctrl+Z undoes.");
                                self.tab = Tab::Overview;
                                return;
                            }
                            Err(error) => self.status = error,
                        }
                    }
                    _ => {}
                }
                self.dialog = Some(Dialog::Era5Provider(selected, original));
            }
            Dialog::CdsCredentials(mut form) => {
                let editable = self.cds.status.as_ref().is_some_and(|status| status.editable) && !self.cds.busy();
                if key.code == KeyCode::F(2) && editable {
                    self.cds.save(&self.python, &self.cwd, form.take_key());
                    form.editing = false;
                } else if key.code == KeyCode::F(5) && !self.cds.busy() {
                    self.cds.refresh(&self.python, &self.cwd);
                } else if editable && form.editing {
                    match key.code {
                        KeyCode::Enter | KeyCode::Tab => form.editing = false,
                        KeyCode::Backspace => form.backspace(),
                        KeyCode::Char('u' | 'U') if key.modifiers.contains(KeyModifiers::CONTROL) => form.clear(),
                        KeyCode::Char(character) if !key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => form.append(&character.to_string()),
                        _ => {}
                    }
                } else if editable && matches!(key.code, KeyCode::Enter | KeyCode::Char('e' | 'E')) {
                    form.editing = true;
                }
                self.dialog = Some(Dialog::CdsCredentials(form));
            }
            Dialog::DomainForm(mut form, mut editing) => {
                if key.code == KeyCode::F(2) || (key.code == KeyCode::Enter && key.modifiers.contains(KeyModifiers::CONTROL)) {
                    match form.apply() {
                        Ok(text) => {
                            if let Some(editor) = &mut self.editor {
                                if editor.text() != form.original {
                                    self.status = "The draft changed while this form was open. Reopen Domains to edit its current values.".into();
                                } else {
                                    editor.replace_draft(text);
                                    self.status = "Domain updated in the draft. Ctrl+S saves; F5 checks.".into();
                                    let index = if form.section == domains::Section::AddChild {
                                        self.domain_labels().map(|rows| rows.len().saturating_sub(1)).unwrap_or(form.index)
                                    } else { form.index };
                                    self.dialog = Some(Dialog::Domains(index));
                                    return;
                                }
                            }
                        }
                        Err(error) => self.status = error,
                    }
                } else if let Some(index) = editing {
                    let field = &mut form.fields[index];
                    match key.code {
                        KeyCode::Enter => editing = None,
                        KeyCode::Tab => {
                            form.selected = (index + 1) % form.fields.len();
                            editing = Some(form.selected);
                        }
                        KeyCode::BackTab => {
                            form.selected = index.saturating_sub(1);
                            editing = Some(form.selected);
                        }
                        KeyCode::Backspace => {
                            field.value.pop();
                        }
                        KeyCode::Char('u' | 'U')
                            if key.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            field.value.clear()
                        }
                        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                            field.value.push(c)
                        }
                        KeyCode::Left | KeyCode::Right if !field.choices().is_empty() => {
                            let choices = field.choices();
                            let at = choices.iter().position(|s| *s == field.value).unwrap_or(0);
                            let next = if key.code == KeyCode::Left {
                                (at + choices.len() - 1) % choices.len()
                            } else {
                                (at + 1) % choices.len()
                            };
                            field.value = choices[next].into();
                        }
                        _ => {}
                    }
                } else {
                    match key.code {
                        KeyCode::Up => form.selected = form.selected.saturating_sub(1),
                        KeyCode::Down => {
                            form.selected = (form.selected + 1).min(form.fields.len() - 1)
                        }
                        KeyCode::Enter => editing = Some(form.selected),
                        _ => {}
                    }
                }
                self.dialog = Some(Dialog::DomainForm(form, editing));
            }
            Dialog::Choice(continuing, mut selected) => {
                let count = if continuing { 2 } else { 3 };
                match key.code {
                    KeyCode::Up => selected = selected.saturating_sub(1),
                    KeyCode::Down => selected = (selected + 1).min(count - 1),
                    KeyCode::Enter => {
                        if !continuing && selected == 0 {
                            self.dialog = Some(Dialog::Path("Configuration path", String::new()));
                        } else {
                            self.begin_guide(if continuing {
                                if selected == 0 {
                                    Kind::Resume
                                } else {
                                    Kind::Prepared
                                }
                            } else if selected == 1 {
                                Kind::Wrf
                            } else {
                                Kind::MetEm
                            });
                        }
                        return;
                    }
                    _ => {}
                }
                self.dialog = Some(Dialog::Choice(continuing, selected));
            }
            Dialog::Summary(mut guide, mut selected) => {
                guide.sync_choices();
                selected = selected.min(guide.questions.len());
                if (key.code == KeyCode::Enter && selected == 0)
                    || (matches!(key.code, KeyCode::Char('n' | 'N')) && key.modifiers.contains(KeyModifiers::CONTROL))
                {
                    if let Some(index) = guide.questions.iter().position(|question|
                        matches!(question.flag, "--cycle" | "--start-time")
                            && question.value.trim().eq_ignore_ascii_case("latest"))
                    {
                        guide.step = index;
                        guide.summary_edit = true;
                        let mut form = calendar::Form::new(&self.python, &self.cwd,
                            guide.date_query_args(), &guide.questions[index].value);
                        form.latest();
                        self.dialog = Some(Dialog::Calendar(guide, form));
                        return;
                    }
                }
                match key.code {
                    KeyCode::Up => selected = selected.saturating_sub(1),
                    KeyCode::Down => selected = (selected + 1).min(guide.questions.len()),
                    KeyCode::Home | KeyCode::Tab => selected = 0,
                    KeyCode::Char('n' | 'N') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        match guide.request(&self.cwd) {
                            Ok(request) => {
                                self.dialog = Some(Dialog::Review(request, Some(guide), 0));
                                return;
                            }
                            Err(error) => self.status = error,
                        }
                    }
                    KeyCode::Enter => {
                        if selected == 0 {
                            match guide.request(&self.cwd) {
                                Ok(request) => {
                                    self.dialog = Some(Dialog::Review(request, Some(guide), 0));
                                    return;
                                }
                                Err(error) => self.status = error,
                            }
                        } else {
                            guide.step = selected - 1;
                            guide.summary_edit = true;
                            self.dialog = Some(Dialog::Guide(guide));
                            return;
                        }
                    }
                    _ => {}
                }
                self.dialog = Some(Dialog::Summary(guide, selected));
            }
            Dialog::Guide(mut guide) => {
                let resolve_latest = matches!(key.code, KeyCode::Enter | KeyCode::Tab)
                    && guide.questions[guide.step].value.trim().eq_ignore_ascii_case("latest");
                if guide.date_question() && (key.code == KeyCode::F(3) || resolve_latest) {
                    let mut form = calendar::Form::new(&self.python, &self.cwd,
                        guide.date_query_args(), &guide.questions[guide.step].value);
                    if resolve_latest { form.latest(); }
                    self.dialog = Some(Dialog::Calendar(guide, form));
                    return;
                }
                let field = &mut guide.questions[guide.step];
                match key.code {
                    KeyCode::Enter | KeyCode::Tab => {
                        if let Err(error) = guide.validate_current() {
                            self.status = error;
                        } else if !guide.advance() {
                            let selected = if guide.summary_edit {
                                guide.step + 1
                            } else {
                                0
                            };
                            self.status = "Settings are ready to review. Click a row to edit, or choose Next below.".into();
                            self.dialog = Some(Dialog::Summary(guide, selected));
                            return;
                        }
                    }
                    KeyCode::BackTab => guide.previous(),
                    KeyCode::Backspace => {
                        field.value.pop();
                    }
                    KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        guide.sync_choices();
                        self.status =
                            "All settings stay editable. Choose Next when you are ready to review."
                                .into();
                        self.dialog = Some(Dialog::Summary(guide, 0));
                        return;
                    }
                    KeyCode::F(2) => {
                        let flag = field.flag;
                        let help_command = match guide.kind {
                            Kind::Downscale => Some("downscale"),
                            Kind::Tiles => Some("domain-tiles"),
                            _ => None,
                        };
                        self.guide_cache = Some(guide);
                        if let Some(command) = help_command {
                            self.start_command(command, &["--help".into()], None)
                        } else if flag == "--physics-profile" {
                            self.start_command("run-plan", &["--physics-profiles".into()], None)
                        } else {
                            self.start_command("sources", &[], None)
                        }
                        return;
                    }
                    KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        field.value.clear()
                    }
                    KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                        field.value.push(c)
                    }
                    _ => {}
                }
                self.dialog = Some(Dialog::Guide(guide));
            }
            Dialog::Calendar(mut guide, mut form) => {
                if let Some(cycle) = form.key(key) {
                    guide.questions[guide.step].value = cycle;
                    self.status = "Selected the exact UTC period. Continue to review the configuration before creating it.".into();
                    self.dialog = Some(Dialog::Guide(guide));
                } else {
                    self.dialog = Some(Dialog::Calendar(guide, form));
                }
            }
            Dialog::Review(request, guide, mut offset) => {
                if key.code == KeyCode::Enter && key.kind == KeyEventKind::Press {
                    if self.busy() {
                        self.status = "A command is already running. Close this review and follow its log before starting another.".into();
                        self.dialog = Some(Dialog::Review(request, guide, offset));
                        return;
                    }
                    if let Some(path) = &request.created {
                        if path.exists() {
                            self.status =
                                "The configuration path now exists. Choose another new path."
                                    .into();
                            self.dialog = Some(Dialog::Review(request, guide, offset));
                            return;
                        }
                    }
                    self.guide_cache = guide;
                    self.start_command(&request.command, &request.args, request.created);
                } else if key.code == KeyCode::BackTab {
                    self.dialog = guide.map(|g| Dialog::Summary(g, 0)).or_else(|| {
                        if request.command == "case-catalog" { self.case_cache.take().map(Dialog::Cases) } else { None }
                    });
                } else {
                    match key.code {
                        KeyCode::Up => offset = offset.saturating_sub(1),
                        KeyCode::Down => offset = offset.saturating_add(1),
                        KeyCode::PageUp => offset = offset.saturating_sub(8),
                        KeyCode::PageDown => offset = offset.saturating_add(8),
                        _ => {}
                    }
                    self.dialog = Some(Dialog::Review(request, guide, offset));
                }
            }
            Dialog::Path(label, mut value) => match key.code {
                KeyCode::F(2) if label == "Configuration path" => {
                    let path = absolute(PathBuf::from(unquote(value.trim())), &self.cwd);
                    let folder = if path.is_dir() {
                        path
                    } else {
                        path.parent().unwrap_or(&self.cwd).to_path_buf()
                    };
                    self.browse(folder);
                }
                KeyCode::Enter => {
                    let path = PathBuf::from(unquote(value.trim()));
                    if path.as_os_str().is_empty()
                        && !matches!(label, "Prepared folder" | "Geography folder")
                    {
                        self.status = "Enter a path, or Esc to keep the current value.".into();
                        self.dialog = Some(Dialog::Path(label, value));
                        return;
                    }
                    match label {
                        "Save draft as" => {
                            if !self.save_as(path) {
                                self.dialog = Some(Dialog::Path(label, value));
                            }
                        }
                        "Configuration path" => self.open(path),
                        "Output folder" => self.output = absolute(path, &self.cwd),
                        "Geography folder" => {
                            self.geog_root = if path.as_os_str().is_empty() {
                                PathBuf::new()
                            } else {
                                absolute(path, &self.cwd)
                            };
                            self.status = if self.geog_root.as_os_str().is_empty() {
                                "Geography folder cleared; Plan and Run use the CLI default.".into()
                            } else {
                                "Geography folder set for Plan and Run. Declared case-data paths retain CLI priority.".into()
                            };
                        }
                        "Prepared folder" => {
                            self.prepared = if path.as_os_str().is_empty() {
                                path
                            } else {
                                absolute(path, &self.cwd)
                            }
                        }
                        "Python executable" => self.python = path,
                        _ => {}
                    }
                }
                KeyCode::Backspace => {
                    value.pop();
                    self.dialog = Some(Dialog::Path(label, value));
                }
                KeyCode::Char(c) if key.modifiers.contains(KeyModifiers::CONTROL) && c == 'u' => {
                    self.dialog = Some(Dialog::Path(label, String::new()))
                }
                KeyCode::Char(c) => {
                    value.push(c);
                    self.dialog = Some(Dialog::Path(label, value));
                }
                _ => self.dialog = Some(Dialog::Path(label, value)),
            },
            Dialog::Browser(dir, items, mut selected) => match key.code {
                KeyCode::Up => {
                    selected = selected.saturating_sub(1);
                    self.dialog = Some(Dialog::Browser(dir, items, selected));
                }
                KeyCode::Down => {
                    selected = (selected + 1).min(items.len().saturating_sub(1));
                    self.dialog = Some(Dialog::Browser(dir, items, selected));
                }
                KeyCode::Backspace => self.browse(dir.parent().unwrap_or(&dir).to_path_buf()),
                KeyCode::Enter => {
                    if let Some(p) = items.get(selected) {
                        if p.is_dir() {
                            self.browse(p.clone())
                        } else {
                            self.open(p.clone())
                        }
                    }
                }
                KeyCode::Char('/') => {
                    self.dialog = Some(Dialog::Path("Configuration path", String::new()))
                }
                _ => self.dialog = Some(Dialog::Browser(dir, items, selected)),
            },
            Dialog::DroppedFiles(items, mut selected) => match key.code {
                KeyCode::Up => {
                    selected = selected.saturating_sub(1);
                    self.dialog = Some(Dialog::DroppedFiles(items, selected));
                }
                KeyCode::Down => {
                    selected = (selected + 1).min(items.len().saturating_sub(1));
                    self.dialog = Some(Dialog::DroppedFiles(items, selected));
                }
                KeyCode::Enter => {
                    if let Some(path) = items.get(selected) {
                        self.open(path.clone());
                    }
                }
                _ => self.dialog = Some(Dialog::DroppedFiles(items, selected)),
            },
            Dialog::Stop => {
                if matches!(key.code, KeyCode::Char('y' | 'Y')) && key.kind == KeyEventKind::Press {
                    if let Some(job) = &mut self.job {
                        self.status = match job.stop() { Ok(()) => job.stop_message().into(), Err(e) => format!("Could not stop this run: {e}") };
                    }
                    self.close_overlay();
                } else if matches!(key.code, KeyCode::Char('n' | 'N')) {
                    self.close_overlay();
                } else {
                    self.dialog = Some(Dialog::Stop)
                }
            }
            Dialog::Quit => {
                if matches!(key.code, KeyCode::Char('y' | 'Y')) && key.kind == KeyEventKind::Press {
                    if self.busy() {
                        let job = self.job.as_mut().unwrap();
                        match job.stop() {
                            Ok(()) => { self.exit_after_job = true; self.status = format!("{} Closing after worker termination is confirmed.", job.stop_message()); }
                            Err(error) => self.status = format!("Could not stop this run: {error}. The workspace remains open."),
                        }
                        self.dialog = Some(Dialog::Quit);
                    } else {
                        self.exit = true;
                    }
                } else if matches!(key.code, KeyCode::Char('s' | 'S')) {
                    self.save();
                    self.close_overlay();
                } else if matches!(key.code, KeyCode::Char('n' | 'N')) {
                    self.exit_after_job = false;
                    self.close_overlay();
                } else {
                    self.dialog = Some(Dialog::Quit)
                }
            }
            Dialog::Help(mut scroll) => {
                scroll.offset = match key.code {
                    KeyCode::Up => scroll.offset.saturating_sub(1),
                    KeyCode::Down => scroll.offset.saturating_add(1).min(scroll.max_offset),
                    KeyCode::PageUp => scroll.offset.saturating_sub(scroll.page_rows.max(1)),
                    KeyCode::PageDown => scroll
                        .offset
                        .saturating_add(scroll.page_rows.max(1))
                        .min(scroll.max_offset),
                    KeyCode::Home => 0,
                    KeyCode::End => scroll.max_offset,
                    _ => scroll.offset,
                };
                self.dialog = Some(Dialog::Help(scroll));
            }
            Dialog::Plots(_) | Dialog::Nodes | Dialog::Workflows(_) | Dialog::Scenario(..) | Dialog::Cases(_) => {
                unreachable!()
            }
        }
    }
    fn paste(&mut self, value: String) {
        if !self.input_enabled {
            return;
        }
        if let Some(Dialog::CdsCredentials(form)) = &mut self.dialog {
            if self.cds.status.as_ref().is_some_and(|status| status.editable) && !self.cds.busy() {
                form.append(&value);
            }
            return;
        }
        let open_files = self.dialog.is_none() || matches!(self.dialog,
            Some(Dialog::Browser(..) | Dialog::DroppedFiles(..)
                | Dialog::Path("Configuration path", _) | Dialog::Choice(false, _)
                | Dialog::Cases(_)));
        if open_files {
            if let Some(paths) = file_drop::paths(&value, &self.cwd) {
                if self.dirty() {
                    self.status = "Save with Ctrl+S or F12 before opening a dropped file. Your draft is unchanged.".into();
                    return;
                }
                self.dialog = None;
                if paths.len() == 1 {
                    self.open(paths[0].clone());
                } else {
                    self.status = format!("{} files dropped. Choose one to open; no forecast starts.", paths.len());
                    self.dialog = Some(Dialog::DroppedFiles(paths, 0));
                }
                return;
            }
        }
        if matches!(self.dialog, Some(Dialog::Nodes)) {
            self.node_panel.paste(&value);
        } else if let Some(Dialog::Plots(form)) = &mut self.dialog {
            form.paste(&value);
        } else if let Some(Dialog::DomainForm(form, Some(index))) = &mut self.dialog {
            form.fields[*index]
                .value
                .push_str(value.trim_end_matches(['\r', '\n']));
        } else if let Some(Dialog::Scenario(form, _)) = &mut self.dialog {
            form.paste(&value);
        } else if let Some(Dialog::Workflows(browser)) = &mut self.dialog {
            browser.paste(&value);
        } else if let Some(Dialog::Guide(guide)) = &mut self.dialog {
            guide.questions[guide.step]
                .value
                .push_str(value.trim_end_matches(['\r', '\n']));
        } else if let Some(Dialog::Calendar(_, form)) = &mut self.dialog {
            form.paste(&value);
        } else if let Some(Dialog::Cases(form)) = &mut self.dialog {
            form.paste(&value);
        } else if let Some(Dialog::Path(_, text)) = &mut self.dialog {
            text.push_str(value.trim_end_matches(['\r', '\n']));
        } else if self.dialog.is_none() && self.tab == Tab::Settings {
            if let Some(e) = &mut self.editor {
                e.insert(&value);
            }
        }
    }
}

fn panel(title: &str) -> Block<'_> {
    theme::panel(title)
}

fn guide_value_tail(value: &str, width: usize) -> String {
    let visible = safe(value).replace('\n', "↵").replace('\t', " ");
    if UnicodeWidthStr::width(visible.as_str()) <= width {
        return visible;
    }
    if width == 0 {
        return String::new();
    }
    let mut used = 1; // Keep a leading ellipsis when earlier input is hidden.
    let mut tail = Vec::new();
    for grapheme in visible.graphemes(true).rev() {
        let cells = UnicodeWidthStr::width(grapheme);
        if used + cells > width {
            break;
        }
        tail.push(grapheme);
        used += cells;
    }
    format!("…{}", tail.into_iter().rev().collect::<String>())
}
fn popup(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width.saturating_sub(2));
    let h = h.min(area.height.saturating_sub(2));
    Rect::new(
        area.x + (area.width - w) / 2,
        area.y + (area.height - h) / 2,
        w,
        h,
    )
}
fn fact(label: &str, value: impl Into<String>) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<15}: "), Style::default().fg(MUTED)),
        Span::raw(safe(&value.into())),
    ])
}

fn compact_overview(app: &App, width: u16, height: u16) -> Vec<Line<'static>> {
    let clip = |text: String, limit: usize| {
        if UnicodeWidthStr::width(text.as_str()) <= limit {
            return text;
        }
        let mut result = String::new();
        let mut used = 0;
        for ch in text.chars() {
            let cells = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
            if used + cells > limit.saturating_sub(1) {
                break;
            }
            result.push(ch);
            used += cells;
        }
        result.push('…');
        result
    };
    let plots = app
        .plot_selection()
        .map(|selection| selection.summary())
        .unwrap_or_else(|_| "Settings need review (B)".into());
    let Some(editor) = &app.editor else {
        return vec![
            Line::styled("Choose a mode to begin", theme::heading()),
            Line::raw("N Modes · O Open existing · C Continue"),
            Line::styled(
                "Check settings, review the plan, then launch.",
                Style::default().fg(MUTED),
            ),
        ];
    };
    let Ok(doc) = editor.text().parse::<toml_edit::DocumentMut>() else {
        return vec![
            Line::styled("TOML needs review · F opens settings", theme::notice(true)),
            Line::raw("The complete draft remains editable."),
            Line::raw(format!("Plots: {plots}")),
        ];
    };
    let value = |table: &str, key: &str| {
        doc.get(table).and_then(|item| item.get(key)).map(|item| {
            item.as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| item.to_string().trim().to_owned())
        })
    };
    let name = safe(&value("experiment", "name").unwrap_or_else(|| "Forecast draft".into()));
    let cycle = value("fetch", "cycle")
        .or_else(|| value("experiment", "start_time"))
        .map(|value| clip(safe(&value), 19))
        .unwrap_or_else(|| "start unset".into());
    let duration = value("experiment", "run_seconds")
        .and_then(|value| value.parse::<f64>().ok())
        .map(|seconds| format!("{:.1} h", seconds / 3600.0))
        .unwrap_or_else(|| "duration unset".into());
    let domains = doc.get("domain").and_then(|item| item.as_array_of_tables());
    let domain = if let Some(domains) = domains {
        let first = domains
            .iter()
            .next()
            .map(|domain| {
                let get = |key: &str| {
                    domain
                        .get(key)
                        .map(|item| item.to_string().trim().to_owned())
                        .unwrap_or_else(|| "?".into())
                };
                format!(
                    " · d{} {} × {} · dx {} m",
                    get("grid_id"),
                    get("nx"),
                    get("ny"),
                    get("dx")
                )
            })
            .unwrap_or_default();
        format!("Domains: {}{first}", domains.len())
    } else {
        "Domains: none defined · D opens domains".into()
    };
    let mut lines = if height <= 3 {
        vec![Line::styled(
            format!(
                "{} · {cycle} · {duration}",
                clip(
                    name,
                    usize::from(width).saturating_sub(cycle.len() + duration.len() + 6)
                )
            ),
            theme::heading(),
        )]
    } else {
        vec![
            Line::styled(clip(name, width as usize), theme::heading()),
            Line::styled(
                format!("Start: {cycle} · {duration}"),
                Style::default().fg(MUTED),
            ),
        ]
    };
    lines.push(Line::raw(clip(domain, width as usize)));
    lines.push(Line::styled(
        clip(format!("Plots: {plots}"), width as usize),
        Style::default().fg(theme::LEAF),
    ));
    if height >= 5 {
        let source = value("fetch", "source")
            .or_else(|| value("case_data", "source"))
            .unwrap_or_else(|| "not selected".into());
        let source = match app.era5_provider() {
            Some("arco") => "ERA5 · Google ARCO".into(),
            Some("cds") => "ERA5 · Copernicus CDS".into(),
            Some(_) => "ERA5 · choose provider (U)".into(),
            None => source,
        };
        lines.push(Line::styled(
            format!(
                "Input: {} · F Settings · D Domains · B Plots",
                safe(&source)
            ),
            Style::default().fg(MUTED),
        ));
    }
    lines
}

fn button_bar(frame: &mut Frame, hits: &mut Vec<HitRegion>, area: Rect, buttons: &[(&str, Hit)]) {
    button_bar_selected(frame, hits, area, buttons, None);
}

fn button_bar_selected(
    frame: &mut Frame,
    hits: &mut Vec<HitRegion>,
    area: Rect,
    buttons: &[(&str, Hit)],
    selected: Option<usize>,
) {
    let mut x = area.x;
    let mut y = area.y;
    for (index, (label, action)) in buttons.iter().enumerate() {
        let width = (UnicodeWidthStr::width(*label) as u16 + 4).min(area.width);
        if x + width > area.right() {
            x = area.x;
            y += 1;
        }
        if y >= area.bottom() {
            break;
        }
        let rect = Rect::new(x, y, width, 1);
        let primary = matches!(action, Hit::Key(KeyCode::Enter | KeyCode::F(7), _));
        let style = if selected == Some(index) {
            theme::selected()
        } else {
            theme::button(primary)
        };
        let text = if selected == Some(index) { format!("[>{label}<]") } else { format!("[ {label} ]") };
        frame.render_widget(Paragraph::new(text).style(style), rect);
        hits.push(HitRegion {
            area: rect,
            action: *action,
        });
        x += width + 1;
    }
}
fn key_hit(code: KeyCode) -> Hit {
    Hit::Key(code, KeyModifiers::NONE)
}

fn clickable_list(
    frame: &mut Frame,
    hits: &mut Vec<HitRegion>,
    area: Rect,
    mut rows: Vec<(Vec<Line<'static>>, Hit)>,
    selected: Option<usize>,
) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    // Ratatui omits an item that cannot fit as a whole. Bound only its
    // presentation; the complete setting remains in Guide and Request.
    for (lines, _) in &mut rows {
        if lines.len() > area.height as usize {
            lines.truncate(area.height as usize);
            *lines.last_mut().unwrap() = Line::raw("... (value continues)");
        }
    }
    let mut state = ListState::default().with_selected(selected);
    let items = rows
        .iter()
        .map(|(lines, _)| ListItem::new(lines.clone()))
        .collect::<Vec<_>>();
    frame.render_stateful_widget(
        List::new(items)
            .highlight_symbol("> ")
            .highlight_style(theme::selected()),
        area,
        &mut state,
    );
    let mut y = area.y;
    for (lines, action) in rows.iter().skip(state.offset()) {
        if y >= area.bottom() {
            break;
        }
        let height = lines.len() as u16;
        if height > area.bottom() - y {
            break;
        }
        hits.push(HitRegion {
            area: Rect::new(area.x, y, area.width, height),
            action: *action,
        });
        y += height;
    }
}
fn draw_dialog(frame: &mut Frame, app: &mut App, area: Rect) {
    if app.dialog.is_none() {
        return;
    }
    app.hits.clear();
    if let Some(Dialog::Scenario(form, _)) = &mut app.dialog {
        let backdrop = Rect::new(
            area.x,
            area.y + 3,
            area.width,
            area.height.saturating_sub(3),
        );
        form.draw(frame, backdrop);
        return;
    }
    if let Some(Dialog::Workflows(browser)) = &app.dialog {
        let backdrop = Rect::new(
            area.x,
            area.y + 3,
            area.width,
            area.height.saturating_sub(3),
        );
        browser.draw(
            frame,
            backdrop,
            &mut app.hits,
            app.editor.is_some() || app.nodes.store.selected().is_some(),
        );
        return;
    }
    if matches!(app.dialog, Some(Dialog::Nodes)) {
        let plots = app
            .plot_selection()
            .map(|v| v.summary())
            .unwrap_or_else(|e| e);
        let backdrop = Rect::new(
            area.x,
            area.y + 3,
            area.width,
            area.height.saturating_sub(3),
        );
        app.node_panel.draw(frame, backdrop, &app.nodes, &plots);
        return;
    }
    // A modal has its own controls. Background tabs/hotkeys are not actionable.
    let backdrop = Rect::new(
        area.x,
        area.y + 3,
        area.width,
        area.height.saturating_sub(3),
    );
    frame.render_widget(Clear, backdrop);
    theme::backdrop(frame, backdrop);
    let rect = if matches!(app.dialog, Some(Dialog::Calendar(..))) && area.height < 24 {
        backdrop // Preserve six complete weeks and 24 hours at 65 x 20.
    } else {
        popup(backdrop, 100, backdrop.height)
    };
    let title = match app.dialog.as_ref().unwrap() {
        Dialog::Nodes | Dialog::Workflows(_) | Dialog::Scenario(..) => unreachable!(),
        Dialog::Details { .. } => "Command / error details".into(),
        Dialog::Domains(_) => "Domains".into(),
        Dialog::Plots(form) => format!(
            "Plots - {}",
            match form.mode {
                plotsettings::Mode::Presets => "choose a preset or customize",
                plotsettings::Mode::Products => "search and select products",
                plotsettings::Mode::Text => "advanced selector list",
                plotsettings::Mode::Review => "review every selected product",
            }
        ),
        Dialog::DomainMenu(_, _) => "Domain settings".into(),
        Dialog::DomainForm(form, _) => form.title.clone(),
        Dialog::DomainRemoval(removal, _) => removal.title.clone(),
        Dialog::Era5Provider(..) => "ERA5 provider".into(),
        Dialog::CdsCredentials(_) => "Copernicus CDS key".into(),
        Dialog::Choice(true, _) => "Continue forecast".into(),
        Dialog::Choice(false, _) => "Open existing".into(),
        Dialog::Guide(g) => format!("{} - {}", g.title(), g.progress()),
        Dialog::Calendar(..) => "Choose source date and cycle - UTC".into(),
        Dialog::Cases(..) => "Cases - catalog, source and configuration".into(),
        Dialog::Summary(g, _) => format!("{} - review settings", g.title()),
        Dialog::Review(..) => "Review before starting (Up/Down to scroll)".into(),
        Dialog::Path("Configuration path", _) => "Configuration or catalog path".into(),
        Dialog::Path(label, _) => (*label).into(),
        Dialog::Browser(dir, ..) => format!("Open - {}", display_path(dir)),
        Dialog::DroppedFiles(..) => "Choose a dropped file to open".into(),
        Dialog::Stop => "Stop this run?".into(),
        Dialog::Quit => "Close workspace?".into(),
        Dialog::Help(_) => "ArWen help".into(),
    };
    let block = panel(&title).border_style(Style::default().fg(theme::SKY));
    let inside = block.inner(rect);
    frame.render_widget(block, rect);
    let notice_height = if matches!(app.dialog, Some(Dialog::Calendar(..))) && inside.height <= 16 {
        1
    } else if let Some(Dialog::Details {
        notice: Some(notice),
        ..
    }) = &app.dialog
    {
        log_display_rows(notice, usize::from(inside.width))
            .len()
            .clamp(2, 4) as u16
    } else {
        2
    };
    let parts = Layout::vertical([
        Constraint::Min(3),
        Constraint::Length(notice_height),
        Constraint::Length(if matches!(app.dialog, Some(Dialog::Details { .. })) && app.memory_recovery_available() { 3 } else { 2 }),
    ])
    .split(inside);
    let body = parts[0];
    if let Some(Dialog::Help(scroll)) = &mut app.dialog {
        scroll.page_rows = body.height.max(1);
        scroll.max_offset = Paragraph::new(HELP_TEXT)
            .wrap(Wrap { trim: false })
            .line_count(body.width)
            .saturating_sub(usize::from(body.height))
            .min(usize::from(u16::MAX)) as u16;
        scroll.offset = scroll.offset.min(scroll.max_offset);
    }
    let note = if let Some(Dialog::Calendar(_, form)) = &app.dialog {
        form.notice.clone()
    } else if let Some(Dialog::Cases(form)) = &app.dialog {
        form.notice.clone()
    } else if let Some(Dialog::Plots(form)) = &app.dialog {
        if form.has_error() {
            form.notice.clone()
        } else if let Some(error) = &app.plot_catalog.error {
            format!("Catalog unavailable: {error} Saved selections are preserved.")
        } else if app.plot_catalog.loading {
            format!("Loading catalog. {}", form.notice)
        } else {
            if form.mode == plotsettings::Mode::Presets
                && form.notice.starts_with("Presets request plots only.")
            {
                plotsettings::presets()
                    .get(form.selected)
                    .map(|preset| preset.description.clone())
                    .unwrap_or_else(|| form.notice.clone())
            } else {
                form.notice.clone()
            }
        }
    } else if let Some(Dialog::Details { notice, .. }) = &app.dialog {
        notice.clone().unwrap_or_else(|| {
            "Up/Down or wheel scroll · Home/End jump · Shift+drag selects in Windows Terminal"
                .into()
        })
    } else if let Some(Dialog::Help(scroll)) = &app.dialog {
        format!(
            "Up/Down or wheel · PgUp/PgDn · Home/End · Esc closes\nScroll {} / {}",
            scroll.offset, scroll.max_offset
        )
    } else {
        safe(&app.status)
    };
    frame.render_widget(
        Paragraph::new(note)
            .wrap(Wrap { trim: false })
            .style(theme::notice(
                matches!(app.dialog, Some(Dialog::Details { .. }))
                    || matches!(&app.dialog, Some(Dialog::Plots(form)) if form.has_error()),
            )),
        parts[1],
    );
    let buttons = parts[2];
    if let Some(Dialog::Cases(form)) = &mut app.dialog {
        form.draw(frame, &mut app.hits, body, buttons);
        return;
    }
    if let Some(Dialog::Details { text, offset, .. }) = &mut app.dialog {
        let rows = log_display_rows(text, usize::from(body.width));
        *offset = (*offset).min(rows.len().saturating_sub(usize::from(body.height)));
    }
    match app.dialog.as_ref().unwrap() {
        Dialog::Nodes | Dialog::Workflows(_) | Dialog::Scenario(..) => unreachable!(),
        Dialog::Plots(form) => {
            plotsettings::draw(frame, &mut app.hits, form, &app.plot_catalog, body, buttons)
        }
        Dialog::Details { text, offset, .. } => {
            let rows = log_display_rows(text, usize::from(body.width));
            frame.render_widget(
                Paragraph::new(
                    rows.into_iter()
                        .skip(*offset)
                        .take(usize::from(body.height))
                        .map(Line::raw)
                        .collect::<Vec<_>>(),
                ),
                body,
            );
            let mut actions = vec![("C Copy details", key_hit(KeyCode::Char('c')))];
            if app.log_path().is_some() {
                actions.push(("Y Copy log", key_hit(KeyCode::Char('y'))));
                actions.push(("L View log", key_hit(KeyCode::Char('l'))));
            }
            actions.push(("Close", key_hit(KeyCode::Esc)));
            actions.extend([
                ("Up", key_hit(KeyCode::PageUp)),
                ("Down", key_hit(KeyCode::PageDown)),
                ("End", key_hit(KeyCode::End)),
            ]);
            if app.memory_recovery_available() {
                actions.push(("Ctrl+F Fit domain", Hit::FitCurrent));
                actions.push(("Ctrl+T Tile streaming", Hit::TileCurrent));
            }
            button_bar(frame, &mut app.hits, buttons, &actions);
        }
        Dialog::Domains(selected) => {
            let (mut labels, note) = match app.domain_labels() {
                Ok(rows) => (rows, "Select a domain. Changes stay in your draft.".to_owned()),
                Err(error) => (Vec::new(), error),
            };
            labels.push("New forecast with nested grids".into());
            labels.push("Downscale an archived forecast (offline child)".into());
            let count = labels.len() - 2;
            let parts = Layout::vertical([Constraint::Length(2), Constraint::Min(1)]).split(body);
            frame.render_widget(Paragraph::new(note).wrap(Wrap { trim: false }), parts[0]);
            let rows = labels
                .into_iter()
                .enumerate()
                .map(|(i, label)| (vec![Line::raw(label)], if i < count { Hit::DomainSelect(i) } else { Hit::Choice(i) }))
                .collect();
            clickable_list(frame, &mut app.hits, parts[1], rows, Some(*selected));
            let mut actions = vec![("Back", key_hit(KeyCode::Esc)),
                (if *selected < count { "Edit (Enter)" } else { "Open (Enter)" }, key_hit(KeyCode::Enter))];
            if *selected < count {
                actions.extend([("Add child (A)", key_hit(KeyCode::Char('a'))),
                    ("Remove (Del)", key_hit(KeyCode::Delete)),
                    ("Advanced (M)", key_hit(KeyCode::Char('m')))]);
            }
            button_bar(frame, &mut app.hits, buttons, &actions);
        }
        Dialog::DomainRemoval(removal, offset) => {
            let mut lines = vec![Line::raw("Remove these domains from the draft:"), Line::raw("")];
            lines.extend(removal.rows.iter().map(|row| Line::raw(row.clone())));
            if removal.removes_relocation {
                lines.push(Line::raw(""));
                lines.push(Line::raw("Also remove their relocation settings and scheduled moves."));
            }
            lines.push(Line::raw(""));
            lines.push(Line::raw("F2 removes. Esc cancels. Up/Down scrolls."));
            let content = Paragraph::new(lines).wrap(Wrap { trim: false });
            let max_offset = content.line_count(body.width).saturating_sub(body.height as usize).min(u16::MAX as usize) as u16;
            let offset = (*offset).min(max_offset);
            frame.render_widget(content.scroll((offset, 0)), body);
            if let Some(Dialog::DomainRemoval(_, stored)) = &mut app.dialog { *stored = offset; }
            button_bar(frame, &mut app.hits, buttons, &[
                ("Cancel", key_hit(KeyCode::Esc)),
                ("F2 Remove from draft", key_hit(KeyCode::F(2))),
            ]);
        }
        Dialog::Era5Provider(selected, _) => {
            let parts = Layout::vertical([Constraint::Length(2), Constraint::Min(1)]).split(body);
            frame.render_widget(Paragraph::new("Choose the ERA5 download provider."), parts[0]);
            clickable_list(frame, &mut app.hits, parts[1], vec![
                (vec![Line::raw("Google ARCO")], Hit::Choice(0)),
                (vec![Line::raw("Copernicus CDS")], Hit::Choice(1)),
            ], Some(*selected));
            button_bar(frame, &mut app.hits, buttons, &[
                ("Cancel", key_hit(KeyCode::Esc)),
                ("Use provider (Enter)", key_hit(KeyCode::Enter)),
                ("CDS key (C)", key_hit(KeyCode::Char('c'))),
            ]);
        }
        Dialog::CdsCredentials(form) => {
            let mut lines = vec![Line::styled(format!("CDS key: {}", app.cds.summary()), theme::heading())];
            if let Some(status) = &app.cds.status {
                lines.extend([
                    Line::raw(format!("File: {}", status.path)),
                    Line::raw(format!("Source: {}", status.source)),
                    Line::raw(format!("Endpoint: {}", status.url)),
                ]);
                if status.editable {
                    lines.push(Line::styled(format!("{}: {}", if form.editing { "Enter key >" } else { "New key" }, form.masked()), theme::heading()));
                    lines.push(Line::raw(if form.editing { "Type or paste. Enter finishes editing; F2 saves." }
                        else { "Enter changes the key. F2 saves." }));
                } else {
                    lines.push(Line::raw("Set by environment variables. Change them and reopen ArWen."));
                }
            }
            lines.push(Line::raw(app.cds.notice.clone()));
            frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), body);
            let mut actions = vec![("Back", key_hit(KeyCode::Esc))];
            if !app.cds.busy() {
                if app.cds.status.as_ref().is_some_and(|status| status.editable) {
                    actions.push((if form.editing { "Done editing" } else { "Enter / Change key" }, key_hit(KeyCode::Enter)));
                    actions.push(("F2 Save key", key_hit(KeyCode::F(2))));
                }
                actions.push(("F5 Refresh", key_hit(KeyCode::F(5))));
            }
            button_bar(frame, &mut app.hits, buttons, &actions);
        }
        Dialog::DomainMenu(index, selected) => {
            let label = app
                .domain_labels()
                .ok()
                .and_then(|rows| rows.get(*index).cloned())
                .unwrap_or_default();
            let parts = Layout::vertical([Constraint::Length(3), Constraint::Min(1)]).split(body);
            frame.render_widget(Paragraph::new(label).wrap(Wrap { trim: false }), parts[0]);
            let rows = domains::SECTIONS
                .iter()
                .enumerate()
                .map(|(i, (_, label))| (vec![Line::raw(*label)], Hit::Choice(i)))
                .collect();
            clickable_list(frame, &mut app.hits, parts[1], rows, Some(*selected));
            button_bar(
                frame,
                &mut app.hits,
                buttons,
                &[
                    ("Domains", key_hit(KeyCode::Esc)),
                    ("Edit (Enter)", key_hit(KeyCode::Enter)),
                ],
            );
        }
        Dialog::DomainForm(form, Some(index)) => {
            let field = &form.fields[*index];
            let parts = Layout::vertical([Constraint::Min(2), Constraint::Length(2)]).split(body);
            frame.render_widget(Paragraph::new(format!("{}\n> {}\n\n{}\n\nEnter keeps this field. Ctrl+U clears.", field.label, safe(&field.value), field.help)).wrap(Wrap { trim: false }), parts[0]);
            let choices = field
                .choices()
                .iter()
                .enumerate()
                .map(|(i, value)| (*value, Hit::DomainValue(i)))
                .collect::<Vec<_>>();
            button_bar(frame, &mut app.hits, parts[1], &choices);
            button_bar(
                frame,
                &mut app.hits,
                buttons,
                &[
                    ("Back to fields", key_hit(KeyCode::Esc)),
                    ("Keep field (Enter)", key_hit(KeyCode::Enter)),
                    ("Next field", key_hit(KeyCode::Tab)),
                ],
            );
        }
        Dialog::DomainForm(form, None) => {
            let parts = Layout::vertical([Constraint::Length(3), Constraint::Min(1)]).split(body);
            let signal_issue = form.signal_issue();
            let note = if form
                .fields
                .first()
                .is_some_and(|f| f.label == "Policy enabled" && f.value == "false")
            {
                "F2 removes this policy and its advanced settings. Esc cancels."
            } else if let Some(issue) = &signal_issue {
                issue
            } else {
                &form.note
            };
            frame.render_widget(Paragraph::new(note).wrap(Wrap { trim: false }), parts[0]);
            let rows = form
                .fields
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    let value = if field.value.is_empty() {
                        "(omitted)"
                    } else {
                        &field.value
                    };
                    let text = format!("{}: {}", field.label, safe(value));
                    let lines = log_display_rows(&text, parts[1].width.saturating_sub(2) as usize)
                        .into_iter()
                        .map(Line::raw)
                        .collect();
                    (lines, Hit::DomainField(index))
                })
                .collect();
            clickable_list(frame, &mut app.hits, parts[1], rows, Some(form.selected));
            button_bar(
                frame,
                &mut app.hits,
                buttons,
                &[
                    ("Cancel", key_hit(KeyCode::Esc)),
                    ("Edit field", key_hit(KeyCode::Enter)),
                    (
                        "F2 Apply to draft",
                        key_hit(KeyCode::F(2)),
                    ),
                ],
            );
        }
        Dialog::Summary(g, selected) => {
            let list_parts =
                Layout::vertical([Constraint::Length(2), Constraint::Min(1)]).split(body);
            frame.render_widget(Paragraph::new("Review or click any setting to edit it.\nReady? Click Next below, or press Ctrl+N from any row."), list_parts[0]);
            let rows = g
                .questions
                .iter()
                .enumerate()
                .map(|(index, q)| {
                    let value = if q.value.is_empty() {
                        "Default / not entered"
                    } else {
                        &q.value
                    };
                    let text = format!("{}: {}", q.label, safe(value));
                    let lines =
                        log_display_rows(&text, list_parts[1].width.saturating_sub(2) as usize)
                            .into_iter()
                            .map(Line::raw)
                            .collect();
                    (lines, Hit::Summary(index + 1))
                })
                .collect();
            clickable_list(
                frame,
                &mut app.hits,
                list_parts[1],
                rows,
                selected.checked_sub(1),
            );
            button_bar(
                frame,
                &mut app.hits,
                buttons,
                &[
                    ("Back", key_hit(KeyCode::Esc)),
                    (
                        "Next: review command (Ctrl+N)",
                        Hit::Key(KeyCode::Char('n'), KeyModifiers::CONTROL),
                    ),
                ],
            );
        }
        Dialog::Guide(g) => {
            let q = &g.questions[g.step];
            let label = Paragraph::new(q.label)
                .style(theme::heading())
                .wrap(Wrap { trim: false });
            let label_rows = label.line_count(body.width).min(2) as u16;
            let parts = Layout::vertical([
                Constraint::Length(label_rows),
                Constraint::Min(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(if g.kind == Kind::New { 2 } else { 0 }),
            ])
            .split(body);
            frame.render_widget(label, parts[0]);
            let help = match g.geometry_hint() {
                Some(hint) => format!("{}\n\n{hint}", q.help),
                None => q.help.to_owned(),
            };
            frame.render_widget(Paragraph::new(help).wrap(Wrap { trim: false }), parts[1]);
            // The editable value has its own reserved row: verbose help can
            // never push the current input off the smallest supported screen.
            let value = guide_value_tail(&q.value, usize::from(parts[2].width.saturating_sub(3)));
            let cursor_x = parts[2].x + 2 + UnicodeWidthStr::width(value.as_str()) as u16;
            frame.render_widget(
                Paragraph::new(format!("> {value}"))
                    .style(Style::default().fg(INK).bg(theme::RAISED)),
                parts[2],
            );
            if parts[2].height > 0 && parts[2].width >= 3 {
                frame.set_cursor_position((cursor_x, parts[2].y));
            }
            frame.render_widget(
                Paragraph::new("Type or paste · Ctrl+U clears · Enter continues")
                    .style(Style::default().fg(MUTED)),
                parts[3],
            );
            if g.kind == Kind::New {
                button_bar(
                    frame,
                    &mut app.hits,
                    parts[4],
                    &[
                        ("Root grid spacing", Hit::GuideGrid(3)),
                        ("Nested grid ratios", Hit::GuideGrid(4)),
                    ],
                );
            }
            button_bar(
                frame,
                &mut app.hits,
                buttons,
                &[
                    (
                        "Back",
                        key_hit(if g.summary_edit {
                            KeyCode::Esc
                        } else {
                            KeyCode::BackTab
                        }),
                    ),
                    (
                        if g.summary_edit {
                            "Save setting (Enter)"
                        } else {
                            "Next (Enter)"
                        },
                        key_hit(KeyCode::Enter),
                    ),
                    (
                        "All settings",
                        Hit::Key(KeyCode::Char('a'), KeyModifiers::CONTROL),
                    ),
                    (
                        if g.date_question() {
                            "Calendar (F3)"
                        } else if matches!(g.kind, Kind::Downscale | Kind::Tiles) {
                            "CLI help"
                        } else {
                            "Catalog"
                        },
                        key_hit(KeyCode::F(if g.date_question() { 3 } else { 2 })),
                    ),
                    ("Cancel", key_hit(KeyCode::Esc)),
                ],
            );
        }
        Dialog::Calendar(_, form) => form.draw(frame, &mut app.hits, body, buttons),
        Dialog::Cases(_) => unreachable!(),
        Dialog::Review(request, guide, offset) => {
            let description = if request.command == "render" {
                "Render the listed forecast history with the native Rust renderer. An ordinary output root gets a fresh run folder; a path inside an existing run folder reuses it and can replace matching images. The source label is printed on the plots."
            } else if request.command == "downscale"
                && request.args.iter().any(|a| a == "--dry-run")
            {
                "Validate the archived parent and write the derived child plan. No forecast runs; the new output directory may be created."
            } else if request.command == "domain-tiles" {
                "Preserve the domain area, resolution and physics. The engine plans GPU tile dimensions and checks GPU/system RAM, then creates a new streaming configuration. Check and Plan come next; no forecast starts."
            } else if request.created.is_some() {
                "Create an editable configuration, then review the forecast plan. No forecast starts yet."
            } else if request.command == "downscale" {
                "Run a standalone child forecast from the archived parent. The engine validates the archive and selected settings before integration."
            } else {
                "Start this command using your exact settings."
            };
            frame.render_widget(
                Paragraph::new(format!(
                    "{}\n\n{}{}\n\nEquivalent CLI command:\n{}",
                    request.title,
                    description,
                    guide.as_ref().and_then(|g| g.workflow).and_then(workflows::mode)
                        .map(|mode| format!("\n\nMode: {}\n{}{}", mode.title, mode.limits,
                            if request.created.is_some() { format!("\nPlot set: {} (saved after configuration creation succeeds).", mode.preset) }
                            else if request.command == "downscale" { "\nDownscale creates history; return to this mode's Plot history action to render it.".into() }
                            else { String::new() })).unwrap_or_default(),
                    guide::command_text(&app.python, request)
                ))
                .wrap(Wrap { trim: false })
                .scroll((*offset, 0)),
                body,
            );
            button_bar(
                frame,
                &mut app.hits,
                buttons,
                &[
                    (
                        "Back",
                        key_hit(if guide.is_some() {
                            KeyCode::BackTab
                        } else {
                            KeyCode::Esc
                        }),
                    ),
                    (
                        if request.command == "downscale"
                            && request.args.iter().any(|a| a == "--dry-run")
                        {
                            "Write downscale plan (Enter)"
                        } else if request.command == "downscale" {
                            "Run offline child (Enter)"
                        } else if request.created.is_some() {
                            "Create configuration (Enter)"
                        } else {
                            "Run command (Enter)"
                        },
                        key_hit(KeyCode::Enter),
                    ),
                ],
            );
        }
        Dialog::Choice(continuing, selected) => {
            let labels = if *continuing {
                vec![
                    "Continue from a saved checkpoint",
                    "Run an existing prepared bundle",
                ]
            } else {
                vec![
                    "Open a configuration or case catalog (TOML, ZIP, JSON)",
                    "Use WRF real.exe inputs (directory)",
                    "Use WPS met_em inputs (directory)",
                ]
            };
            clickable_list(
                frame,
                &mut app.hits,
                body,
                labels
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (vec![Line::raw(*s)], Hit::Choice(i)))
                    .collect(),
                Some(*selected),
            );
            button_bar(
                frame,
                &mut app.hits,
                buttons,
                &[
                    ("Back", key_hit(KeyCode::Esc)),
                    ("Open selected (Enter)", key_hit(KeyCode::Enter)),
                ],
            );
        }
        Dialog::Path(label, value) => {
            frame.render_widget(
                Paragraph::new(format!(
                    "{}\n\n> {}\n\nType or paste the path. Ctrl+U clears it.\n{}",
                    if *label == "Configuration path" { "Configuration (TOML) or case catalog (ZIP, JSON, TOML)" } else { label },
                    safe(value),
                    if *label == "Save draft as" {
                        "Use a new file name; your current file is kept."
                    } else {
                        ""
                    }
                ))
                .wrap(Wrap { trim: false }),
                body,
            );
            let mut actions = vec![
                ("Cancel", key_hit(KeyCode::Esc)),
                ("Use this path (Enter)", key_hit(KeyCode::Enter)),
            ];
            if *label == "Configuration path" {
                actions.push(("Browse (F2)", key_hit(KeyCode::F(2))));
            }
            button_bar(frame, &mut app.hits, buttons, &actions);
        }
        Dialog::Browser(dir, items, selected) => {
            let rows = items
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let name = if p == dir.parent().unwrap_or(dir) {
                        "..".into()
                    } else {
                        p.file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned()
                    };
                    (
                        vec![Line::raw(format!(
                            "{} {}",
                            if p.is_dir() { "+" } else { " " },
                            safe(&name)
                        ))],
                        Hit::Browser(i),
                    )
                })
                .collect();
            clickable_list(frame, &mut app.hits, body, rows, Some(*selected));
            button_bar(
                frame,
                &mut app.hits,
                buttons,
                &[
                    ("Back", key_hit(KeyCode::Esc)),
                    ("Parent folder", key_hit(KeyCode::Backspace)),
                    ("Paste path", key_hit(KeyCode::Char('/'))),
                ],
            );
        }
        Dialog::DroppedFiles(items, selected) => {
            let rows = items.iter().enumerate().map(|(index, path)| (
                vec![Line::raw(guide_value_tail(&display_path(path), usize::from(body.width.saturating_sub(4))))], Hit::Browser(index)
            )).collect();
            clickable_list(frame, &mut app.hits, body, rows, Some(*selected));
            button_bar(frame, &mut app.hits, buttons, &[
                ("Cancel", key_hit(KeyCode::Esc)),
                ("Open (Enter)", key_hit(KeyCode::Enter)),
            ]);
        }
        Dialog::Stop => {
            frame.render_widget(Paragraph::new("Stop this run and its preparation/forecast workers?\n\nPartial outputs are kept. Resume uses the last checkpoint already saved.").wrap(Wrap { trim: false }), body);
            button_bar(
                frame,
                &mut app.hits,
                buttons,
                &[
                    ("Keep running", key_hit(KeyCode::Char('n'))),
                    ("Stop this run", key_hit(KeyCode::Char('y'))),
                ],
            );
        }
        Dialog::Quit => {
            let close_label = if app.exit_after_job { "Y Force stop and quit" }
                else if app.busy() { "Y Stop and quit" } else { "Y Close TUI" };
            frame.render_widget(
                Paragraph::new(format!(
                    "{}{}",
                    if app.exit_after_job {
                        "Stopping the local command. Closing after worker termination is confirmed. Stay cancels closing; it does not undo the stop request.\n\n"
                    } else if app.busy() {
                        "Closing stops this local command and its workers. The workspace waits for termination; partial outputs and saved checkpoints are kept.\n\n"
                    } else {
                        ""
                    },
                    if app.dirty() {
                        "Unsaved editor changes will be discarded when you close."
                    } else {
                        "Close the terminal workspace?"
                    }
                ))
                .wrap(Wrap { trim: false }),
                body,
            );
            button_bar(
                frame,
                &mut app.hits,
                buttons,
                &[
                    ("N Stay", key_hit(KeyCode::Char('n'))),
                    ("S Save draft", key_hit(KeyCode::Char('s'))),
                    (close_label, key_hit(KeyCode::Char('y'))),
                ],
            );
        }
        Dialog::Help(scroll) => {
            frame.render_widget(
                Paragraph::new(HELP_TEXT)
                    .wrap(Wrap { trim: false })
                    .scroll((scroll.offset, 0)),
                body,
            );
            button_bar(
                frame,
                &mut app.hits,
                buttons,
                &[
                    ("Close help", key_hit(KeyCode::Esc)),
                    ("PgUp", key_hit(KeyCode::PageUp)),
                    ("PgDn", key_hit(KeyCode::PageDown)),
                    ("Home", key_hit(KeyCode::Home)),
                    ("End", key_hit(KeyCode::End)),
                ],
            );
        }
    }
}

fn draw_header_actions(frame: &mut Frame, app: &mut App, area: Rect) {
    if area.width >= 17 && area.height > 0 {
        button_bar(frame, &mut app.hits, Rect::new(area.right() - 17, area.y, 17, 1),
            &[("Quit (Ctrl+Q)", Hit::Quit)]);
    }
    if area.height < 3 { return; }
    if let Some((result, path, failed)) = app.job_result() {
        let result_area = Rect::new(area.x, area.y + 1, area.width, 2);
        let log = path.map(|path| {
            let shown = path.strip_prefix(&app.output).unwrap_or(&path);
            format!("Log: {}", guide_value_tail(&display_path(shown), area.width.saturating_sub(5) as usize))
        }).unwrap_or_else(|| "No log file was created. Click for details.".into());
        let style = if failed { theme::notice(true) } else {
            Style::default().fg(theme::BACK).bg(theme::LEAF)
        };
        frame.render_widget(Paragraph::new(vec![
            Line::styled(result, style.add_modifier(Modifier::BOLD)),
            Line::styled(log, style),
        ]).style(style), result_area);
        app.hits.push(HitRegion { area: result_area, action: Hit::JobResult });
    }
}

fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    app.set_viewport(area.width, area.height);
    app.hits.clear();
    theme::backdrop(frame, area);
    if area.width < 65 || area.height < 20 {
        let controls = match app.dialog {
            Some(Dialog::Quit) if app.exit_after_job => "Stopping. N / Esc cancels quit; Y forces stop and quits.",
            Some(Dialog::Quit) if app.busy() => "Quit: Y stops workers and quits; N / Esc stays.",
            Some(Dialog::Quit) => "Quit: Y closes and discards drafts; N / Esc stays.",
            Some(Dialog::Stop) => "Stop local run: Y confirms; N / Esc keeps running.",
            Some(Dialog::Help(_)) => "Ctrl+Q quit; Ctrl+C stop review; Esc closes help.",
            _ => "Ctrl+Q quits; Ctrl+C reviews stopping a local run.",
        };
        frame.render_widget(Paragraph::new(format!("ArWen — resize to at least 65 × 20.\nEditing paused. {controls}")).wrap(Wrap { trim: false }), area);
        draw_header_actions(frame, app, area);
        theme::finish(frame);
        return;
    }
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(2),
        Constraint::Min(4),
        Constraint::Length(3),
        Constraint::Length(2),
    ])
    .split(area);
    let badge = app.badge();
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("  ArWen  ", theme::heading()),
            Span::styled("2.7 preview", Style::default().fg(theme::LEAF)),
            Span::raw(
                app.nodes
                    .store
                    .selected()
                    .map(|n| format!("  NODE: {}  ", safe(&n.name)))
                    .unwrap_or_else(|| "  Local computer  ".into()),
            ),
            Span::styled(
                badge,
                Style::default().fg(AMBER).add_modifier(Modifier::BOLD),
            ),
        ]))
        .style(Style::default().bg(theme::RAISED))
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(theme::BORDER)),
        ),
        rows[0],
    );
    button_bar_selected(
        frame,
        &mut app.hits,
        rows[1],
        &[
            ("H Home", Hit::View(Tab::Home)),
            ("V Overview", Hit::View(Tab::Overview)),
            ("W Modes", key_hit(KeyCode::Char('w'))),
            ("D Domains", Hit::Domains),
            ("I Scenario", key_hit(KeyCode::Char('i'))),
            ("B Plots", Hit::Plots),
            ("R Nodes", Hit::Nodes),
            ("F Settings", Hit::View(Tab::Settings)),
            ("L Logs", Hit::View(Tab::Logs)),
        ],
        Some(match app.tab {
            Tab::Home => 0,
            Tab::Overview => 1,
            Tab::Settings => 7,
            Tab::Logs => 8,
        }),
    );
    match app.tab {
        Tab::Home => {
            let spacious = rows[2].height >= 20 && area.width >= 105;
            let descriptive = rows[2].height >= 16;
            let parts = Layout::vertical([
                Constraint::Length(if spacious { 3 } else if rows[2].height < 11 { 1 } else { 2 }),
                Constraint::Min(if descriptive { 12 } else { 7 }),
                Constraint::Length(2),
            ])
            .split(rows[2]);
            frame.render_widget(
                Paragraph::new(vec![
                    Line::styled("What are you exploring?", theme::heading()),
                    Line::styled(
                        "Click a task, or use Up/Down and Enter.",
                        Style::default().fg(MUTED),
                    ),
                ]),
                parts[0],
            );
            button_bar(frame, &mut app.hits, Rect::new(parts[0].x, parts[0].y, parts[0].width, 1),
                &[("Open visual workspace", Hit::OpenCompanion)]);
            let columns = if spacious {
                Layout::horizontal([
                    Constraint::Percentage(60),
                    Constraint::Length(1),
                    Constraint::Min(25),
                ])
                .split(parts[1])
            } else {
                Layout::horizontal([
                    Constraint::Min(1),
                    Constraint::Length(0),
                    Constraint::Length(0),
                ])
                .split(parts[1])
            };
            let choices_area = columns[0];
            let choices = [
                (
                    "Choose a forecast mode",
                    "Severe storms, tropical, winter, fire weather and more",
                ),
                ("Open existing", "Configuration, case catalog, WRF inputs, or WPS met_em"),
                (
                    "Continue forecast",
                    "Locate a real checkpoint or an existing prepared bundle",
                ),
                (
                    "Fit starter TOML",
                    "Size your editable configuration to an area and GPU",
                ),
                (
                    "Browse case catalog",
                    "Historical cases, source dates, domain tiers and physics notes",
                ),
            ];
            let item_height = if descriptive { 2 } else { 1 };
            let items = choices
                .iter()
                .map(|(title, description)| {
                    let mut lines = vec![Line::styled(
                        *title,
                        Style::default()
                            .fg(theme::LEAF)
                            .add_modifier(Modifier::BOLD),
                    )];
                    if descriptive {
                        lines.push(Line::styled(*description, Style::default().fg(MUTED)));
                    }
                    ListItem::new(lines)
                })
                .collect::<Vec<_>>();
            let mut selected = ListState::default().with_selected(Some(app.selected));
            frame.render_stateful_widget(
                List::new(items)
                    .block(panel(" Start here "))
                    .highlight_symbol("> ")
                    .highlight_style(theme::selected()),
                choices_area,
                &mut selected,
            );
            for i in selected.offset()..choices.len() {
                let y = choices_area.y + 1 + (i - selected.offset()) as u16 * item_height;
                if y + item_height <= choices_area.bottom().saturating_sub(1) {
                    app.hits.push(HitRegion {
                        area: Rect::new(
                            choices_area.x + 1,
                            y,
                            choices_area.width.saturating_sub(2),
                            item_height,
                        ),
                        action: Hit::Home(i),
                    });
                }
            }
            if spacious {
                let target = app
                    .nodes
                    .store
                    .selected()
                    .map(|node| safe(&node.name))
                    .unwrap_or_else(|| "Local computer".into());
                let draft = app
                    .editor
                    .as_ref()
                    .map(|editor| display_path(&editor.path))
                    .unwrap_or_else(|| "No configuration open".into());
                frame.render_widget(
                    Paragraph::new(vec![
                        Line::styled("Your forecast workspace", theme::heading()),
                        Line::raw(""),
                        Line::styled("01  Choose a weather use", Style::default().fg(theme::LEAF)),
                        Line::styled("02  Shape the domain and inputs", Style::default().fg(INK)),
                        Line::styled("03  Review the exact launch", Style::default().fg(INK)),
                        Line::raw(""),
                        fact("Target", target),
                        fact("Draft", draft),
                        fact(
                            "Plots",
                            app.plot_selection()
                                .map(|selection| selection.summary())
                                .unwrap_or_else(|error| error),
                        ),
                        Line::raw(""),
                        Line::styled(
                            "Nothing runs until you choose it.",
                            Style::default().fg(MUTED),
                        ),
                    ])
                    .wrap(Wrap { trim: false })
                    .block(panel(" At a glance ")),
                    columns[2],
                );
            }
            frame.render_widget(Paragraph::new("Choose a mode · Browse cases · Review and launch\nN Modes  K Cases  O Open  C Continue  T Fit  S Sources  P Physics").wrap(Wrap{trim:false}).style(Style::default().fg(MUTED)),parts[2]);
        }
        Tab::Overview => {
            let wide = area.width >= 100;
            let cols = if wide {
                Layout::horizontal([Constraint::Percentage(58), Constraint::Percentage(42)])
                    .split(rows[2])
            } else {
                Layout::vertical([Constraint::Min(3), Constraint::Length(7)]).split(rows[2])
            };
            let mut lines = vec![];
            if let Some(e) = &app.editor {
                lines.push(fact("Configuration", display_path(&e.path).to_string()));
                lines.push(fact("CDS key", app.cds.summary()));
                if let Some(status) = &app.cds.status { lines.push(fact("CDS file", &status.path)); }
                lines.push(fact(
                    "Plots",
                    app.plot_selection()
                        .map(|value| value.summary())
                        .unwrap_or_else(|_| "Settings need review (B)".into()),
                ));
                lines.push(Line::raw(""));
                match e.text().parse::<toml_edit::DocumentMut>() {
                    Ok(doc) => {
                        for (label, table, key) in [
                            ("Name", "experiment", "name"),
                            ("Starts", "experiment", "start_time"),
                            ("Duration (s)", "experiment", "run_seconds"),
                            ("Input source", "fetch", "source"),
                            ("Input source", "case_data", "source"),
                            ("Vertical levels", "shared", "nz"),
                            ("Microphysics", "shared", "mp_physics"),
                        ] {
                            if let Some(v) = doc.get(table).and_then(|t| t.get(key)) {
                                lines.push(fact(label, v.to_string().trim().to_string()));
                            }
                        }
                        if let Some(provider) = app.era5_provider() {
                            lines.push(fact("ERA5 provider [U]", match provider {
                                "arco" => "Google ARCO", "cds" => "Copernicus CDS", _ => "Choose provider",
                            }));
                        }
                        if let Some(domains) =
                            doc.get("domain").and_then(|v| v.as_array_of_tables())
                        {
                            lines.push(fact("Domains", domains.len().to_string()));
                            for domain in domains.iter() {
                                let get = |k: &str| {
                                    domain
                                        .get(k)
                                        .map(|v| v.to_string().trim().to_string())
                                        .unwrap_or_else(|| "—".into())
                                };
                                lines.push(Line::styled(
                                    format!(
                                        "  d{}    {} × {}    dx {} m",
                                        get("grid_id"),
                                        get("nx"),
                                        get("ny"),
                                        get("dx")
                                    ),
                                    Style::default().fg(MUTED),
                                ));
                            }
                        }
                    }
                    Err(e) => lines.push(Line::styled(
                        format!("TOML: {e}"),
                        Style::default().fg(AMBER),
                    )),
                }
                lines.push(Line::raw(""));
                lines.push(Line::styled(
                    "D edits domains, tracking and lifecycle; E opens all settings.",
                    Style::default().fg(TEAL),
                ));
            } else {
                lines.extend([
                    Line::styled(
                        "Start with your experiment",
                        Style::default().fg(TEAL).add_modifier(Modifier::BOLD),
                    ),
                    Line::raw(""),
                    Line::raw("O opens a configuration from disk."),
                    Line::raw("Ctrl+O accepts a pasted file path."),
                    Line::raw(""),
                    Line::raw("Review and edit every setting in the TOML."),
                    Line::raw("Check it, review the plan, then launch."),
                ]);
                lines.push(fact("CDS key", app.cds.summary()));
                if let Some(status) = &app.cds.status { lines.push(fact("CDS file", &status.path)); }
            }
            lines.push(Line::raw(""));
            lines.push(fact("Outputs [F3]", display_path(&app.output).to_string()));
            lines.push(fact(
                "Prepared [F4]",
                if app.prepared.as_os_str().is_empty() {
                    "Choose only to run existing preparation".into()
                } else {
                    display_path(&app.prepared).to_string()
                },
            ));
            lines.push(fact(
                "Geography [G]",
                if app.geog_root.as_os_str().is_empty() {
                    "CLI default".to_owned()
                } else {
                    display_path(&app.geog_root)
                },
            ));
            lines.push(fact("Python [F9]", display_path(&app.python).to_string()));
            let configuration = if wide {
                Paragraph::new(lines)
                    .wrap(Wrap { trim: false })
                    .block(panel(" Configuration "))
            } else {
                Paragraph::new(compact_overview(app, cols[0].width, cols[0].height))
                    .style(Style::default().fg(INK).bg(theme::SURFACE))
            };
            frame.render_widget(configuration, cols[0]);
            let right = if wide {
                Layout::vertical([Constraint::Length(9), Constraint::Min(3)]).split(cols[1])
            } else {
                Layout::vertical([Constraint::Min(1), Constraint::Length(0)]).split(cols[1])
            };
            let node_target = app.nodes.store.selected().is_some();
            let choices = ACTIONS
                .iter()
                .map(|(label, action)| ListItem::new(if node_target {
                    match action {
                        Action::Check => "F5  Connect to node",
                        Action::Plan => "F6  Review start on node",
                        Action::Run => "F7  Review start on node",
                        Action::Prepared => "F8  Review prepared start on node",
                        Action::Doctor => "F10 Check local installation",
                    }
                } else { *label }))
                .collect::<Vec<_>>();
            let mut state = ListState::default().with_selected(Some(app.selected));
            frame.render_stateful_widget(
                List::new(choices)
                    .block(panel(" Next step · Enter selects "))
                    .highlight_symbol("› ")
                    .highlight_style(theme::selected()),
                right[0],
                &mut state,
            );
            let visual = Rect::new(right[0].x + 1, right[0].y, right[0].width.saturating_sub(2), 1);
            frame.render_widget(Paragraph::new("─".repeat(visual.width as usize)).style(Style::default().fg(theme::BORDER).bg(theme::SURFACE)), visual);
            button_bar(frame, &mut app.hits, visual, &[("Open visual workspace", Hit::OpenCompanion)]);
            for i in state.offset()..ACTIONS.len() {
                let y = right[0].y + 1 + (i - state.offset()) as u16;
                if y < right[0].bottom().saturating_sub(1) {
                    app.hits.push(HitRegion {
                        area: Rect::new(right[0].x + 1, y, right[0].width.saturating_sub(2), 1),
                        action: Hit::Action(i),
                    });
                }
            }
            let notes = if let Some(job) = &app.job {
                format!("Command: {}\nElapsed: {:.0} s\n\n{}\n\nL shows recent output. Ctrl+L opens details.{}\nSaved log: {}",job.action,job.started.elapsed().as_secs_f64(),job.outcome.map(|c|format!("Exited with {c}")).unwrap_or_else(||"Running".into()),if app.busy() { " X stops this run." } else { "" },display_path(&job.dir.join("job.log")))
            } else if let Some(failure) = &app.startup_failure {
                format!(
                    "FAILED to start\n\n{}\n\nCtrl+L opens details.",
                    failure.message
                )
            } else {
                "No command is running.\n\nCheck and Plan use the same validation and routing as the CLI.\n\nRun starts preparation and the forecast with your saved settings.\n\nF1: keys and help".into()
            };
            frame.render_widget(
                Paragraph::new(notes)
                    .wrap(Wrap { trim: false })
                    .block(panel(" Session ")),
                right[1],
            );
        }
        Tab::Settings => {
            if let Some(e) = &mut app.editor {
                let block = panel(" Complete TOML · Ctrl+S save · F12 Save As · Esc overview ");
                let inside = block.inner(rows[2]);
                frame.render_widget(block, rows[2]);
                button_bar(frame, &mut app.hits, Rect::new(inside.x, inside.y, inside.width, 1), &[
                    ("Ctrl+Z Undo", Hit::Key(KeyCode::Char('z'), KeyModifiers::CONTROL)),
                    ("Ctrl+Y Redo", Hit::Key(KeyCode::Char('y'), KeyModifiers::CONTROL)),
                    ("Ctrl+U Discard", Hit::Key(KeyCode::Char('u'), KeyModifiers::CONTROL)),
                ]);
                let inside = Rect::new(inside.x, inside.y + 1, inside.width, inside.height.saturating_sub(1));
                let height = inside.height as usize;
                let width = inside.width.saturating_sub(7) as usize;
                e.ensure_viewport(width, height);
                let lines = e
                    .lines
                    .iter()
                    .enumerate()
                    .skip(e.top)
                    .take(height)
                    .map(|(n, line)| {
                        let text = Editor::visible_line(line, e.left, width);
                        let color = if text.trim_start().starts_with('#') {
                            MUTED
                        } else if text.trim_start().starts_with('[') {
                            TEAL
                        } else {
                            INK
                        };
                        Line::from(vec![
                            Span::styled(format!("{:>5}  ", n + 1), Style::default().fg(MUTED)),
                            Span::styled(safe(&text), Style::default().fg(color)),
                        ])
                        .style(Style::default().bg(if n == e.row {
                            theme::RAISED
                        } else {
                            theme::SURFACE
                        }))
                    })
                    .collect::<Vec<_>>();
                frame.render_widget(Paragraph::new(lines), inside);
                app.hits.push(HitRegion {
                    area: inside,
                    action: Hit::Editor(inside),
                });
                let dx = Editor::display_column(&e.lines[e.row], e.col).saturating_sub(e.left);
                frame.set_cursor_position((
                    inside.x + 7 + (dx as u16).min(inside.width.saturating_sub(8)),
                    inside.y + (e.row - e.top) as u16,
                ));
            }
        }
        Tab::Logs => {
            let forecast = app.has_local_forecast_job();
            let log_rows=Layout::vertical([Constraint::Min(2),Constraint::Length(1)]).split(rows[2]);
            if forecast && !app.local_raw_logs {
                let status = companion::job_status(app.job.as_ref().expect("forecast job"), &app.output);
                draw_forecast_progress(frame, log_rows[0], &status);
            } else {
                let title = if app.busy() {
                    " Live log · X stop · Up/Down scroll · End follow "
                } else {
                    " Recent output · Up/Down scroll · End follow "
                };
                draw_log(frame, log_rows[0], &app.log_text(), app.log_offset, title);
            }
            let mut buttons = Vec::new();
            if forecast {
                buttons.push((if app.local_raw_logs {"G Progress"} else {"G Raw logs"}, key_hit(KeyCode::Char('g'))));
            }
            buttons.extend([("Y Copy logs",key_hit(KeyCode::Char('y'))),("O Open log",key_hit(KeyCode::Char('o')))]);
            if app.busy() { buttons.push(("X Stop", key_hit(KeyCode::Char('x')))); }
            button_bar(frame,&mut app.hits,log_rows[1],&buttons);
        }
    }
    let status_rows =
        Layout::vertical([Constraint::Length(2), Constraint::Length(1)]).split(rows[3]);
    frame.render_widget(
        Paragraph::new(safe(&app.status))
            .style(theme::notice(
                app.startup_failure.is_some()
                    || app
                        .job
                        .as_ref()
                        .and_then(|job| job.outcome)
                        .is_some_and(|code| code != 0),
            ))
            .wrap(Wrap { trim: false }),
        status_rows[0],
    );
    app.hits.push(HitRegion {
        area: status_rows[0],
        action: Hit::Details,
    });
    let mut status_actions = vec![(if app.memory_recovery_available() { "Details" } else { "Ctrl+L Details" }, Hit::Details)];
    if app.memory_recovery_available() {
        status_actions.push(("Ctrl+F Fit domain", Hit::FitCurrent));
        status_actions.push(("Ctrl+T Tile streaming", Hit::TileCurrent));
    }
    button_bar(frame, &mut app.hits, status_rows[1], &status_actions);
    let mut actions = vec![
            (
                if app.nodes.store.selected().is_some() {
                    "F5 Connect"
                } else {
                    "F5 Check"
                },
                key_hit(KeyCode::F(5)),
            ),
            (
                if app.nodes.store.selected().is_some() {
                    "F6 Review"
                } else {
                    "F6 Plan"
                },
                key_hit(KeyCode::F(6)),
            ),
            ("F7 Run", key_hit(KeyCode::F(7))),
    ];
    if let Some(provider) = app.era5_provider() {
        actions.push((match provider {
            "arco" => "ERA5: Google (U)", "cds" => "ERA5: CDS (U)", _ => "ERA5 provider (U)",
        }, Hit::Era5Provider));
    }
    if app.tab == Tab::Overview || app.era5_provider().is_some() {
        actions.push((match app.cds.status.as_ref() {
            Some(status) if status.configured => "CDS key: set",
            Some(_) => "CDS key: missing",
            None => "CDS key",
        }, Hit::CdsCredentials));
    }
    actions.extend([
            (
                "Geography",
                Hit::Key(KeyCode::Char('g'), KeyModifiers::CONTROL),
            ),
            ("Save", Hit::Key(KeyCode::Char('s'), KeyModifiers::CONTROL)),
            ("Help", key_hit(KeyCode::F(1))),
    ]);
    button_bar(frame, &mut app.hits, rows[4], &actions);
    draw_dialog(frame, app, area);
    draw_header_actions(frame, app, area);
    theme::finish(frame);
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
fn color_css(color: Color, foreground: bool) -> String {
    let (r, g, b) = match color {
        Color::Rgb(r, g, b) => (r, g, b),
        Color::Reset => return color_css(if foreground { INK } else { BACK }, foreground),
        Color::Indexed(index) if index >= 232 => {
            let level = 8 + (index - 232) * 10;
            (level, level, level)
        }
        Color::Indexed(index) if index >= 16 => {
            let value = index - 16;
            let level = |part: u8| if part == 0 { 0 } else { 55 + part * 40 };
            (level(value / 36), level((value % 36) / 6), level(value % 6))
        }
        other => {
            let index = match other {
                Color::Black => 0,
                Color::Red => 1,
                Color::Green => 2,
                Color::Yellow => 3,
                Color::Blue => 4,
                Color::Magenta => 5,
                Color::Cyan => 6,
                Color::Gray => 7,
                Color::DarkGray => 8,
                Color::LightRed => 9,
                Color::LightGreen => 10,
                Color::LightYellow => 11,
                Color::LightBlue => 12,
                Color::LightMagenta => 13,
                Color::LightCyan => 14,
                Color::White => 15,
                Color::Indexed(value) => value,
                _ => unreachable!(),
            } as usize;
            [
                (0, 0, 0),
                (128, 0, 0),
                (0, 128, 0),
                (128, 128, 0),
                (0, 0, 128),
                (128, 0, 128),
                (0, 128, 128),
                (192, 192, 192),
                (128, 128, 128),
                (255, 0, 0),
                (0, 255, 0),
                (255, 255, 0),
                (0, 0, 255),
                (255, 0, 255),
                (0, 255, 255),
                (255, 255, 255),
            ][index]
        }
    };
    format!("#{r:02x}{g:02x}{b:02x}")
}

fn modifier_names(modifiers: Modifier) -> Vec<&'static str> {
    [
        (Modifier::BOLD, "bold"),
        (Modifier::DIM, "dim"),
        (Modifier::ITALIC, "italic"),
        (Modifier::UNDERLINED, "underlined"),
        (Modifier::SLOW_BLINK, "slow_blink"),
        (Modifier::RAPID_BLINK, "rapid_blink"),
        (Modifier::REVERSED, "reversed"),
        (Modifier::HIDDEN, "hidden"),
        (Modifier::CROSSED_OUT, "crossed_out"),
    ]
    .into_iter()
    .filter_map(|(flag, name)| modifiers.contains(flag).then_some(name))
    .collect()
}

fn cell_document(
    buffer: &ratatui::buffer::Buffer,
    screen: &str,
    cursor: Option<(u16, u16)>,
) -> serde_json::Value {
    let mut cells = Vec::with_capacity(buffer.area.width as usize * buffer.area.height as usize);
    for y in buffer.area.y..buffer.area.bottom() {
        for x in buffer.area.x..buffer.area.right() {
            let cell = &buffer[(x, y)];
            cells.push(serde_json::json!({
                "x": x, "y": y, "symbol": cell.symbol(), "fg": color_css(cell.fg, true),
                "bg": color_css(cell.bg, false), "fg_raw": format!("{:?}", cell.fg),
                "bg_raw": format!("{:?}", cell.bg), "modifiers": cell.modifier.bits(),
                "modifier_names": modifier_names(cell.modifier), "skip": cell.skip,
            }));
        }
    }
    serde_json::json!({"schema": "arwen.tui.cells.v1", "width": buffer.area.width,
        "height": buffer.area.height, "screen": screen,
        "cursor": cursor.map(|(x, y)| serde_json::json!({"x": x, "y": y, "visible": true})),
        "color_fallback": "Reset uses the theme; named and indexed colors use the standard ANSI/xterm palette.",
        "cells": cells})
}

fn buffer_html(
    buffer: &ratatui::buffer::Buffer,
    screen: &str,
    cursor: Option<(u16, u16)>,
) -> String {
    let mut content = String::new();
    for y in buffer.area.y..buffer.area.bottom() {
        for x in buffer.area.x..buffer.area.right() {
            let cell = &buffer[(x, y)];
            let (foreground, background) = if cell.modifier.contains(Modifier::REVERSED) {
                (color_css(cell.bg, false), color_css(cell.fg, true))
            } else {
                (color_css(cell.fg, true), color_css(cell.bg, false))
            };
            let mut glyph_style = String::new();
            if cell.modifier.contains(Modifier::BOLD) {
                glyph_style.push_str("font-weight:700;");
            }
            if cell.modifier.contains(Modifier::DIM) {
                glyph_style.push_str("opacity:.6;");
            }
            if cell.modifier.contains(Modifier::ITALIC) {
                glyph_style.push_str("font-style:italic;");
            }
            if cell.modifier.contains(Modifier::HIDDEN) {
                glyph_style.push_str("visibility:hidden;");
            }
            let mut decorations = Vec::new();
            if cell.modifier.contains(Modifier::UNDERLINED) {
                decorations.push("underline");
            }
            if cell.modifier.contains(Modifier::CROSSED_OUT) {
                decorations.push("line-through");
            }
            if !decorations.is_empty() {
                glyph_style.push_str(&format!("text-decoration:{};", decorations.join(" ")));
            }
            if cell
                .modifier
                .intersects(Modifier::SLOW_BLINK | Modifier::RAPID_BLINK)
            {
                glyph_style.push_str(if cell.modifier.contains(Modifier::RAPID_BLINK) {
                    "animation:blink .5s steps(1) infinite;"
                } else {
                    "animation:blink 1s steps(1) infinite;"
                });
            }
            let cursor_style = if cursor == Some((x, y)) {
                "box-shadow:inset 0 -2px #e7f6ef;"
            } else {
                ""
            };
            content.push_str(&format!(
                "<span class=cell data-x={x} data-y={y} data-fg=\"{}\" data-bg=\"{}\" data-modifiers={} style=\"color:{foreground};background:{background};{cursor_style}\"><span class=glyph style=\"{glyph_style}\">{}</span></span>",
                escape(&format!("{:?}", cell.fg)), escape(&format!("{:?}", cell.bg)), cell.modifier.bits(), escape(cell.symbol())
            ));
        }
    }
    format!("<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\"><title>ArWen — {screen}</title><style>\
        body{{margin:0;background:#08191d;color:#e7f6ef;padding:24px;font:14px system-ui}}\
        header{{margin:0 0 16px;color:#97bab3}}\
        .terminal{{display:grid;grid-template-columns:repeat({width},1ch);grid-auto-rows:1.35em;width:{width}ch;font:16px/1.35 Consolas,'Cascadia Mono',monospace;font-variant-ligatures:none}}\
        .cell{{position:relative;width:1ch;height:1.35em;white-space:pre}}\
        .glyph{{position:absolute;left:0;top:0;z-index:1;white-space:pre}}\
        @keyframes blink{{50%{{opacity:0}}}}@media(prefers-reduced-motion:reduce){{.glyph{{animation:none!important}}}}\
        </style><header>Actual ArWen terminal cells · {screen} · {width} × {height}</header><main class=terminal aria-label=\"Actual ArWen terminal screen\">{content}</main></html>",
        screen=escape(screen), width=buffer.area.width, height=buffer.area.height)
}

fn snapshot_screen(app: &mut App, screen: &str) -> Result<(), String> {
    match screen {
        "current" => (),
        "home" | "overview" | "settings" | "logs" => {
            app.dialog = None;
            app.view(match screen {
                "home" => Tab::Home,
                "overview" => Tab::Overview,
                "settings" => Tab::Settings,
                _ => Tab::Logs,
            });
        }
        "help" => app.dialog = Some(Dialog::Help(HelpScroll::default())),
        "nodes" => app.open_nodes(),
        "domains" => app.dialog = Some(Dialog::Domains(0)),
        "scenario" => {
            let original = app
                .editor
                .as_ref()
                .ok_or("Scenario snapshot needs --config")?
                .text();
            app.dialog = Some(Dialog::Scenario(
                scenario::Form::new(original.clone())?,
                original,
            ));
        }
        "plots" => {
            app.dialog = Some(Dialog::Plots(plotsettings::Form::from_selection(
                app.plot_selection()?,
            )))
        }
        "guide" => app.dialog = Some(Dialog::Guide(Guide::new(Kind::New, &app.cwd, &app.output))),
        "modes" => app.dialog = Some(Dialog::Workflows(workflows::Browser::default())),
        value if value.starts_with("mode:") => {
            let id = &value[5..];
            let index = workflows::MODES
                .iter()
                .position(|mode| mode.id == id)
                .ok_or_else(|| format!("Unknown snapshot mode {id}"))?;
            let mut browser = workflows::Browser::default();
            browser.choose(index);
            app.dialog = Some(Dialog::Workflows(browser));
        }
        value if value.starts_with("research:") => {
            let id = &value[9..];
            let index = research::rows("configurations")
                .iter()
                .position(|row| row["id"] == id)
                .ok_or_else(|| format!("Unknown research configuration {id}"))?;
            let mut browser = workflows::Browser::default();
            browser.choose_config(index);
            app.dialog = Some(Dialog::Workflows(browser));
        }
        _ => return Err(format!("Unknown snapshot screen {screen}")),
    }
    Ok(())
}

fn snapshot_buffer(
    app: &mut App,
    width: u16,
    height: u16,
) -> io::Result<(ratatui::buffer::Buffer, Option<(u16, u16)>)> {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend)?;
    terminal.draw(|f| draw(f, app))?;
    let position = terminal.get_cursor_position()?;
    let cursor = (app.input_enabled
        && (matches!(app.dialog, Some(Dialog::Guide(_)))
            || (app.tab == Tab::Settings && app.dialog.is_none() && app.editor.is_some())))
    .then_some((position.x, position.y));
    Ok((terminal.backend().buffer().clone(), cursor))
}

fn snapshot(app: &mut App, path: &Path, width: u16, height: u16, screen: &str) -> io::Result<()> {
    let (buffer, cursor) = snapshot_buffer(app, width, height)?;
    fs::write(path, buffer_html(&buffer, screen, cursor))?;
    fs::write(
        path.with_extension("cells.json"),
        serde_json::to_vec_pretty(&cell_document(&buffer, screen, cursor))?,
    )
}

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let mut app = App::new()?;
    let mut args = env::args().skip(1);
    let mut snap = None;
    let (mut snapshot_width, mut snapshot_height) = (120, 36);
    let mut snapshot_selection = String::from("current");
    let mut open_companion = false;
    let mut headless_companion = false;
    let mut connect_node = false;
    let mut show_progress = false;
    let mut explicit_nodes = false;
    let mut configured = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--companion" => {
                app.companion.explicit_path = Some(absolute(PathBuf::from(args.next().ok_or("--companion needs an executable")?), &app.cwd));
            }
            "--open-companion" => open_companion = true,
            "--headless-companion" => headless_companion = true,
            "--connect-node" => connect_node = true,
            "--show-progress" => { connect_node = true; show_progress = true; },
            "--nodes-file" => {
                if explicit_nodes{return Err("--nodes-file may be supplied only once.".into());}
                app.use_nodes_file(PathBuf::from(args.next().ok_or("--nodes-file needs an absolute profile JSON path")?))?;
                explicit_nodes=true;
            }
            "--config" => {
                if configured{return Err("--config may be supplied only once.".into());}
                app.open(PathBuf::from(args.next().ok_or("--config needs a path")?));configured=true;
            }
            "--python" => {
                app.python = PathBuf::from(args.next().ok_or("--python needs an executable")?)
            }
            "--output" => {
                app.output = absolute(
                    PathBuf::from(args.next().ok_or("--output needs a folder")?),
                    &app.cwd,
                )
            }
            "--prepared" => {
                app.prepared = absolute(
                    PathBuf::from(args.next().ok_or("--prepared needs a folder")?),
                    &app.cwd,
                )
            }
            "--geog-root" => {
                let value = PathBuf::from(args.next().ok_or("--geog-root needs a folder")?);
                app.geog_root = if value.as_os_str().is_empty() {
                    value
                } else {
                    absolute(value, &app.cwd)
                };
            }
            "--snapshot" => {
                snap = Some(PathBuf::from(
                    args.next().ok_or("--snapshot needs an HTML path")?,
                ))
            }
            "--snapshot-width" => {
                snapshot_width = args
                    .next()
                    .ok_or("--snapshot-width needs columns")?
                    .parse::<u16>()?
            }
            "--snapshot-height" => {
                snapshot_height = args
                    .next()
                    .ok_or("--snapshot-height needs rows")?
                    .parse::<u16>()?
            }
            "--snapshot-screen" => {
                snapshot_selection = args.next().ok_or("--snapshot-screen needs a screen name")?
            }
            "--help" | "-h" => {
                println!("Visual workspace: --companion PATH (or ARWEN_COMPANION); --open-companion opens it at startup.\n");
                println!("Desktop controller: --headless-companion opens the visual workspace without an interactive terminal. It keeps an owned forecast running after the window closes.\n");
                println!("Node profiles: --nodes-file ABSOLUTE_JSON selects one explicit profile store; --connect-node opens Nodes and probes its active profile without starting a forecast.\n");
                println!("Progress window: --show-progress connects the active saved node and opens its current job status without starting a forecast.\n");
                println!("ArWen terminal workspace (2.7 preview)\nUsage: arwen-tui [--config FILE] [--python EXECUTABLE] [--output DIR] [--prepared DIR] [--geog-root DIR]\n\nStart with W Research to choose a weather question and configuration. I Scenario edits initial-state warm bubbles; D Domains edits following and tracking in an open configuration. Open existing or Continue forecast resumes your own workflow. Click options, tabs and buttons. K opens the built-in historical cases. O accepts TOML configurations and catalog ZIP/JSON files; F2 browses. Drop files to open them without starting a forecast. F/E edits all settings; V shows overview; G opens geography. Ctrl+S saves. F6 reviews the plan; F7 reviews the exact launch command.\nNo command starts automatically. F1 shows all keys; Up/Down or wheel, PgUp/PgDn and Home/End scroll help; Esc closes it.\n\nRead-only capture: --snapshot FILE.html [--snapshot-width COLUMNS] [--snapshot-height ROWS] [--snapshot-screen SCREEN]. Produces styled HTML and FILE.cells.json from the actual terminal cells. SCREEN: home, overview, settings, logs, help, nodes, domains, plots, guide, modes, mode:ID, research:ID, scenario (needs --config), or current. Default size: 120 x 36.");
                return Ok(());
            }
            "--version" => {
                println!("arwen-tui {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            _ => return Err(format!("Unknown option {arg}; use --help").into()),
        }
    }
    if let Some(path) = snap {
        if headless_companion { return Err("--headless-companion cannot be combined with a read-only snapshot.".into()); }
        if open_companion { return Err("--open-companion cannot be combined with a read-only snapshot.".into()); }
        if connect_node{return Err("--connect-node cannot be combined with a read-only snapshot.".into());}
        if !(1..=400).contains(&snapshot_width) || !(1..=160).contains(&snapshot_height) {
            return Err("Snapshot dimensions must be 1..400 columns and 1..160 rows".into());
        }
        snapshot_screen(&mut app, &snapshot_selection)?;
        return Ok(snapshot(
            &mut app,
            &path,
            snapshot_width,
            snapshot_height,
            &snapshot_selection,
        )?);
    }
    if headless_companion {
        if connect_node { app.prepare_startup_node()?; }
        return app.run_headless_companion().map_err(Into::into);
    }
    use std::io::IsTerminal;
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Err(
            "Open ArWen TUI in an interactive terminal; use --help for launch options.".into(),
        );
    }
    if connect_node { app.prepare_startup_node()?; }
    if show_progress { app.view(Tab::Logs); }
    let mut terminal = ratatui::init();
    execute!(io::stdout(), EnableBracketedPaste, EnableMouseCapture)?;
    if connect_node { app.node_request(remote::Operation::Probe); }
    if open_companion { app.open_companion(); }
    let result = (|| -> io::Result<()> {
        while !app.exit {
            app.poll();
            terminal.draw(|f| draw(f, &mut app))?;
            if event::poll(Duration::from_millis(150))? {
                let pending = file_drop_input::read(event::read()?, &app.cwd)?;
                for (index, input) in pending.into_iter().enumerate() {
                    if index > 0 {
                        app.poll();
                        terminal.draw(|f| draw(f, &mut app))?;
                    }
                    match input {
                    Event::Key(k) => {
                        let size = terminal.size()?;
                        app.set_viewport(size.width, size.height);
                        app.key(k);
                    }
                    Event::Paste(v) => {
                        let size = terminal.size()?;
                        app.set_viewport(size.width, size.height);
                        app.paste(v);
                    }
                    Event::Mouse(event) => app.mouse(event),
                    Event::Resize(w, h) => {
                        app.set_viewport(w, h);
                        app.hits.clear();
                    }
                    _ => {}
                    }
                    if app.exit {
                        break;
                    }
                }
            }
        }
        Ok(())
    })();
    let _ = execute!(io::stdout(), DisableBracketedPaste, DisableMouseCapture);
    ratatui::restore();
    remote::remove_poll_records(&app.output);
    if let Some(job) = &app.job {
        println!(
            "ArWen {}: {}. Logs: {}",
            job.action,
            if job.outcome.is_none() {
                "worker termination was not confirmed; check the saved log"
            } else {
                "finished"
            },
            display_path(&job.dir)
        );
    }
    if let Some(node) = app.nodes.store.selected() {
        if let Some(id) = &node.last_job {
            println!(
                "Node {}: job {}. Reopen Nodes → Jobs to reconnect.",
                safe(&node.host),
                id
            );
        }
    }
    result?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::SKY;

    #[test]
    fn guide_pins_current_label_and_unicode_input_tail_in_supported_viewports() {
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            for kind in [Kind::New, Kind::Research, Kind::Downscale] {
                let mut app = loaded_app("name = 'guide value'\n");
                let mut guide = Guide::new(kind, &app.cwd, &app.output);
                guide.step = guide
                    .questions
                    .iter()
                    .position(|q| q.flag == "--out" || q.flag == "--outdir")
                    .unwrap();
                let label = guide.questions[guide.step].label;
                let value = format!(
                    "C:/{}case-界-e\u{301}-end.toml",
                    "long directory/".repeat(12)
                );
                guide.questions[guide.step].value = value.clone();
                app.dialog = Some(Dialog::Guide(guide));
                let (buffer, cursor) = snapshot_buffer(&mut app, width, height).unwrap();
                let text = render_at(&mut app, width, height);
                assert!(text.contains(label), "{kind:?} at {width}x{height}: {text}");
                assert!(text.contains("…") && text.contains("-end.toml"), "{text}");
                assert!(text.contains("Ctrl+U clears"), "{text}");
                let (x, y) = cursor.expect("Guide exposes the insertion cursor");
                assert!(x < width && y < height);
                assert_eq!(buffer[(x - 1, y)].symbol(), "l");
                let Some(Dialog::Guide(guide)) = &app.dialog else {
                    panic!("Guide remains open");
                };
                assert_eq!(guide.questions[guide.step].value, value);
                capture_screen(&format!("guide-input-{kind:?}"), width, height, &mut app);
                assert!(app.job.is_none());
            }
        }
    }

    #[test]
    fn help_scrolls_by_page_wheel_and_end_with_bounds_and_escape_close() {
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let mut app = loaded_app("name = 'help navigation'\n");
            app.key(press(KeyCode::F(1)));
            let first = render_at(&mut app, width, height);
            assert!(
                first.contains("W Research") && first.contains("I Scenario"),
                "{first}"
            );
            assert!(first.contains("PgUp/PgDn"), "{first}");
            app.key(press(KeyCode::PageDown));
            let Some(Dialog::Help(scroll)) = &app.dialog else {
                panic!("Help stays open while paging");
            };
            assert_eq!(scroll.offset, scroll.page_rows.min(scroll.max_offset));
            app.key(press(KeyCode::End));
            let last = render_at(&mut app, width, height);
            assert!(
                last.contains("Esc closes help.") && last.contains("command."),
                "{last}"
            );
            assert!(last.contains("research:ID"), "{last}");
            capture_screen("help-last-page", width, height, &mut app);
            for _ in 0..100 {
                app.key(press(KeyCode::PageDown));
            }
            let Some(Dialog::Help(scroll)) = &app.dialog else {
                panic!("Help stays open at its end");
            };
            assert_eq!(scroll.offset, scroll.max_offset);
            let end = scroll.offset;
            app.mouse(MouseEvent {
                kind: MouseEventKind::ScrollUp,
                column: 10,
                row: 10,
                modifiers: KeyModifiers::NONE,
            });
            let Some(Dialog::Help(scroll)) = &app.dialog else {
                panic!("Wheel retains help");
            };
            assert_eq!(scroll.offset, end.saturating_sub(1));
            app.key(press(KeyCode::PageUp));
            app.key(press(KeyCode::Home));
            assert_eq!(render_at(&mut app, width, height), first);
            app.key(press(KeyCode::Char('a')));
            assert!(matches!(app.dialog, Some(Dialog::Help(_))));
            app.key(press(KeyCode::End));
            render_at(&mut app, 120, 36);
            let Some(Dialog::Help(scroll)) = &app.dialog else {
                panic!("Resize retains help");
            };
            assert!(scroll.offset <= scroll.max_offset);
            app.key(press(KeyCode::Esc));
            assert!(app.dialog.is_none() && app.job.is_none());
        }
    }

    #[test]
    fn styled_snapshot_retains_physical_cells_colors_modifiers_and_cursor() {
        let mut buffer = ratatui::buffer::Buffer::empty(Rect::new(0, 0, 4, 2));
        buffer[(0, 0)]
            .set_symbol("<&\"")
            .set_fg(Color::Rgb(1, 2, 3))
            .set_bg(Color::Rgb(4, 5, 6));
        buffer[(0, 0)].modifier = Modifier::BOLD | Modifier::UNDERLINED | Modifier::REVERSED;
        buffer[(3, 0)].skip = true;
        buffer.set_string(0, 1, "界é", Style::default().fg(SKY).bg(BACK));
        let document = cell_document(&buffer, "capture <test>", Some((2, 1)));
        assert_eq!(document["width"], 4);
        assert_eq!(document["height"], 2);
        assert_eq!(document["cells"].as_array().unwrap().len(), 8);
        assert_eq!(document["cells"][0]["fg"], "#010203");
        assert_eq!(document["cells"][0]["bg"], "#040506");
        assert_eq!(
            document["cells"][0]["modifier_names"],
            serde_json::json!(["bold", "underlined", "reversed"])
        );
        assert_eq!(document["cells"][3]["skip"], true);
        assert_eq!(document["cells"][4]["symbol"], "界");
        assert_eq!(document["cells"][6]["symbol"], "é");
        assert_eq!(
            document["cursor"],
            serde_json::json!({"x": 2, "y": 1, "visible": true})
        );
        let html = buffer_html(&buffer, "capture <test>", Some((2, 1)));
        assert_eq!(html.matches("class=cell ").count(), 8);
        assert!(html.contains("color:#040506;background:#010203"));
        assert!(html.contains("font-weight:700;") && html.contains("text-decoration:underline;"));
        assert!(html.contains("&lt;&amp;&quot;") && html.contains("capture &lt;test&gt;"));
        assert!(html.contains("data-x=2 data-y=1") && html.contains("box-shadow:inset 0 -2px"));
        assert!(html.contains("grid-template-columns:repeat(4,1ch)"));
        assert_eq!(color_css(Color::Indexed(196), true), "#ff0000");
        assert_eq!(color_css(Color::Indexed(255), false), "#eeeeee");
        assert_eq!(color_css(Color::Reset, false), color_css(BACK, false));
    }

    #[test]
    fn home_choices_and_all_navigation_remain_visible_with_clear_focus() {
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let mut app = loaded_app("name = 'visual navigation'\n");
            app.tab = Tab::Home;
            for selected in 0..5 {
                app.selected = selected;
                let (buffer, cursor) = snapshot_buffer(&mut app, width, height).unwrap();
                assert!(cursor.is_none());
                for index in 0..5 {
                    let hit = app
                        .hits
                        .iter()
                        .find(|hit| matches!(hit.action, Hit::Home(i) if i == index))
                        .unwrap_or_else(|| {
                            panic!("Home choice {index} is missing at {width}x{height}")
                        });
                    assert!(hit.area.right() <= width && hit.area.bottom() <= height);
                    if index == selected {
                        let cell = &buffer[(hit.area.x + 2, hit.area.y)];
                        let monochrome = env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty());
                        assert_eq!(cell.bg, if monochrome { Color::Reset } else { SKY }, "Selected home row at {width}x{height}");
                        assert_eq!(cell.fg, if monochrome { Color::Reset } else { BACK });
                        assert!(cell.modifier.contains(Modifier::BOLD | Modifier::UNDERLINED));
                    }
                }
                for tab in [Tab::Home, Tab::Overview, Tab::Settings, Tab::Logs] {
                    assert!(app
                        .hits
                        .iter()
                        .any(|hit| matches!(hit.action, Hit::View(value) if value == tab)));
                }
                for code in ['w', 'i'] {
                    assert!(app.hits.iter().any(|hit| matches!(hit.action, Hit::Key(KeyCode::Char(value), _) if value == code)));
                }
                assert!(app
                    .hits
                    .iter()
                    .any(|hit| matches!(hit.action, Hit::Domains)));
                assert!(app.hits.iter().any(|hit| matches!(hit.action, Hit::Plots)));
                assert!(app.hits.iter().any(|hit| matches!(hit.action, Hit::Nodes)));
            }
            capture_screen("home-mode-focus", width, height, &mut app);
            assert!(app.job.is_none() && !app.dirty());
        }
    }

    #[test]
    fn snapshot_settings_exports_cursor_and_scenario_selection_is_read_only() {
        let original = include_str!("../../../configs/nest_lifecycle_20240521_4km.toml");
        let mut app = loaded_app(original);
        snapshot_screen(&mut app, "settings").unwrap();
        let (buffer, cursor) = snapshot_buffer(&mut app, 65, 20).unwrap();
        let (x, y) = cursor.expect("Editor cursor exported");
        assert!(x < buffer.area.width && y < buffer.area.height);
        assert!(snapshot_buffer(&mut app, 64, 19).unwrap().1.is_none());
        snapshot_screen(&mut app, "scenario").unwrap();
        assert!(matches!(app.dialog, Some(Dialog::Scenario(..))));
        let (_, cursor) = snapshot_buffer(&mut app, 65, 20).unwrap();
        assert!(cursor.is_none());
        assert_eq!(app.editor.as_ref().unwrap().text(), original);
        assert!(app.job.is_none() && !app.dirty());
    }

    #[test]
    fn compact_overview_keeps_forecast_facts_and_every_next_action_visible() {
        let original = "[experiment]\nname = 'Storm overview'\nstart_time = 2026-09-05T18:00:00\nrun_seconds = 10800.0\n[[domain]]\ngrid_id = 1\nnx = 48\nny = 36\ndx = 4000.0\n";
        for (width, height) in [(65, 20), (80, 24)] {
            let mut app = loaded_app(original);
            app.view(Tab::Overview);
            let text = render_at(&mut app, width, height);
            for fact in [
                "Storm overview",
                "2026-09-05T18:00:00",
                "3.0 h",
                "Domains: 1",
                "48 × 36",
                "dx 4000.0 m",
                "Plots:",
            ] {
                assert!(
                    text.contains(fact),
                    "Missing {fact} at {width}x{height}:\n{text}"
                );
            }
            for index in 0..ACTIONS.len() {
                assert!(
                    app.hits
                        .iter()
                        .any(|hit| matches!(hit.action, Hit::Action(value) if value == index)),
                    "Missing action {index} at {width}x{height}"
                );
            }
            capture_screen("overview-compact-facts", width, height, &mut app);
            assert!(app.job.is_none() && !app.dirty());
        }
    }

    thread_local! {
        static COPIED: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    fn record_copy(text: &str) -> Result<(), String> {
        COPIED.with(|copies| copies.borrow_mut().push(text.into()));
        Ok(())
    }

    fn fixture_plot_catalog(app: &mut App) {
        let names = plotsettings::presets()
            .iter()
            .flat_map(|preset| preset.products.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        app.plot_catalog = plotsettings::Catalog::fixture(&app.python, names);
    }

    #[test]
    fn pending_node_launch_escape_never_claims_that_the_dispatch_was_cancelled() {
        let Some(python) = env::var_os("GPUWM_TUI_TEST_PYTHON") else {
            return;
        };
        let mut app = loaded_app("a=1\n");
        let local = app.editor.as_ref().unwrap().path.clone();
        let directory = local.parent().unwrap().to_path_buf();
        app.nodes = remote::Controller::load(&directory);
        let mut node = remote::Node::blank();
        node.host = "fixture-node".into();
        node.workspace = "/node/work".into();
        app.nodes.store.active = Some(node.id.clone());
        app.nodes.store.nodes.push(node.clone());
        let job_dir = directory.join("short-control-request");
        // A harmless real worker supplies the pending handle. No SSH or forecast runs.
        let job = Job::start(&PathBuf::from(python), "version", &[], &job_dir, &directory).unwrap();
        let operation = remote::Operation::Start {
            products: "none".into(),
            preview: false,
            binding: None,
        };
        app.nodes.pending = Some(remote::Request {
            node,
            operation: operation.clone(),
            job,
        });
        app.node_panel.screen = node_ui::Screen::Review {
            operation,
            review: serde_json::json!({}),
        };
        app.dialog = Some(Dialog::Nodes);
        app.key(press(KeyCode::Esc));
        assert!(app.dialog.is_none());
        assert!(app.nodes.pending.is_some());
        assert!(app.status.contains("request continues") && !app.status.contains("no job started"));
        let mut request = app.nodes.pending.take().unwrap();
        let limit = std::time::Instant::now() + Duration::from_secs(45);
        while request.job.poll().unwrap().is_none() {
            assert!(std::time::Instant::now() < limit);
            std::thread::sleep(Duration::from_millis(20));
        }
        app.nodes.view.status = Some(serde_json::json!({"state":"running"}));
        assert_eq!(app.badge(), "RUNNING");
        app.nodes.view.connection_error = Some("SSH lost".into());
        assert_eq!(app.badge(), "DISCONNECTED");
        drop(request);
        fs::remove_dir_all(&job_dir).unwrap();
        fs::remove_file(local).unwrap();
        fs::remove_dir(directory).unwrap();
    }
    #[test]
    fn nodes_profiles_plots_and_reviews_preserve_the_local_configuration() {
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let original = "# Local draft is a different configuration\na=1\n";
            let mut app = loaded_app(original);
            let local = app.editor.as_ref().unwrap().path.clone();
            let directory = local.parent().unwrap().to_path_buf();
            app.nodes = remote::Controller::load(&directory);
            fixture_plot_catalog(&mut app);
            let screen = render_at(&mut app, width, height);
            assert!(screen.contains("R Nodes"), "{screen}");
            click_hit(&mut app, |hit| matches!(hit, Hit::Nodes));
            let mut node = remote::Node::blank();
            node.host = "weather-node".into();
            node.workspace = "/srv/weather".into();
            node.config = "/srv/weather/Storm case.toml".into();
            node.last_job = Some("recorded-job".into());
            app.node_intent(node_ui::Intent::Save(node));
            assert!(app.nodes.store.selected().is_some());
            let screen = render_at(&mut app, width, height);
            capture_screen("nodes-profiles", width, height, &mut app);
            assert!(
                screen.contains("NODE: Linux node") && screen.contains("P Connect"),
                "{screen}"
            );
            app.key(press(KeyCode::Char('b')));
            let Some(Dialog::Plots(form)) = &mut app.dialog else {
                panic!("remote plot picker missing")
            };
            form.selection = plotsettings::Selection::preset(1);
            form.mode = plotsettings::Mode::Review;
            let wanted = form.selection.spec.clone();
            render_at(&mut app, width, height);
            capture_screen("nodes-plots-review", width, height, &mut app);
            app.key(press(KeyCode::Enter));
            assert!(matches!(app.dialog, Some(Dialog::Nodes)));
            assert_eq!(app.plot_spec().unwrap(), wanted);
            assert_eq!(
                remote::Store::load(&app.nodes.path)
                    .unwrap()
                    .selected()
                    .unwrap()
                    .plot_products
                    .as_deref(),
                Some(wanted.as_str())
            );
            assert!(!plotsettings::sidecar(&local).exists());
            assert_eq!(fs::read_to_string(&local).unwrap(), original);
            assert!(app.job.is_none() && app.nodes.pending.is_none());
            for (name, screen) in [
                (
                    "nodes-edit",
                    node_ui::Screen::Edit {
                        node: app.nodes.store.selected().unwrap().clone(),
                        field: 9,
                        editing: true,
                    },
                ),
                ("nodes-jobs", node_ui::Screen::Jobs),
                ("nodes-job", node_ui::Screen::Job),
                (
                    "nodes-stop",
                    node_ui::Screen::Stop {
                        job: "recorded-job".into(),
                    },
                ),
                (
                    "nodes-resume",
                    node_ui::Screen::Resume {
                        job: "recorded-job".into(),
                        checkpoint: "latest".into(),
                        output: String::new(),
                        field: 0,
                    },
                ),
                (
                    "nodes-launch-review",
                    node_ui::Screen::Review {
                        operation: remote::Operation::Start {
                            products: wanted.clone(),
                            preview: true,
                            binding: None,
                        },
                        review: serde_json::json!({"config":"/srv/weather/Storm case.toml","outdir":"/srv/weather/arwen-unique","products":wanted,"config_sha256":"a".repeat(64),"input_sha256":"b".repeat(64)}),
                    },
                ),
            ] {
                app.node_panel.screen = screen;
                render_at(&mut app, width, height);
                capture_screen(name, width, height, &mut app);
            }
            app.start_command("go", &[display_path(&local)], None);
            assert!(
                app.job.is_none(),
                "Selecting a node must not run a local configuration"
            );
            assert!(app.status.contains("A node is selected"));
            let mut invalid = app.nodes.store.selected().unwrap().clone();
            invalid.host.clear();
            app.node_panel.screen = node_ui::Screen::Edit {
                node: invalid,
                field: 1,
                editing: true,
            };
            app.dialog = Some(Dialog::Nodes);
            let text = render_at(&mut app, width, height);
            let (y, line) = text
                .lines()
                .enumerate()
                .find(|(_, line)| line.contains("Ctrl+S Save"))
                .unwrap();
            let x = line.find("Ctrl+S Save").unwrap();
            app.mouse(MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: x as u16,
                row: y as u16,
                modifiers: KeyModifiers::NONE,
            });
            let text = render_at(&mut app, width, height);
            capture_screen("nodes-invalid-save", width, height, &mut app);
            assert!(text.contains("SSH host must be"), "{text}");
            assert_eq!(app.nodes.store.selected().unwrap().host, "weather-node");
            app.node_intent(node_ui::Intent::Select(None));
            assert_eq!(app.plot_spec().unwrap().split(',').count(), 25);
            assert_eq!(fs::read_to_string(&local).unwrap(), original);
            fs::remove_file(&app.nodes.path).unwrap();
            fs::remove_file(&local).unwrap();
            fs::remove_dir(directory).unwrap();
        }
    }
    #[test]
    fn plots_presets_search_clicks_review_and_persistence_at_all_terminal_sizes() {
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let original = "# Keep my science and formatting\nname = 'Storm case'\n";
            let mut app = loaded_app(original);
            fixture_plot_catalog(&mut app);
            let path = app.editor.as_ref().unwrap().path.clone();
            let screen = render_at(&mut app, width, height);
            assert!(screen.contains("B Plots"), "{screen}");
            click_hit(&mut app, |hit| matches!(hit, Hit::Plots));
            let screen = render_at(&mut app, width, height);
            capture_screen("plots-presets", width, height, &mut app);
            assert!(
                screen.contains("General (25 plots)") && screen.contains("Customize"),
                "{screen}"
            );
            for index in 0..plotsettings::presets().len() {
                let Some(Dialog::Plots(form)) = &mut app.dialog else {
                    panic!("Plots not open");
                };
                form.mode = plotsettings::Mode::Presets;
                form.selected = index;
                render_at(&mut app, width, height);
                click_hit(
                    &mut app,
                    |hit| matches!(hit, Hit::PlotItem(i) if i == index),
                );
                let screen = render_at(&mut app, width, height);
                assert!(screen.contains("Save plots"), "{screen}");
                capture_screen(
                    &format!("plots-preset-{index}-review"),
                    width,
                    height,
                    &mut app,
                );
                app.key(press(KeyCode::End));
                let screen = render_at(&mut app, width, height);
                let last = plotsettings::presets()[index].products.last().unwrap();
                assert!(screen.contains(last), "{screen}");
                capture_screen(
                    &format!("plots-preset-{index}-last"),
                    width,
                    height,
                    &mut app,
                );
            }
            app.key(press(KeyCode::F(2)));
            app.key(press(KeyCode::End));
            render_at(&mut app, width, height);
            click_hit(
                &mut app,
                |hit| matches!(hit, Hit::PlotItem(i) if i == plotsettings::presets().len() + 1),
            );
            render_at(&mut app, width, height);
            click_hit(&mut app, |hit| matches!(hit, Hit::Key(KeyCode::F(5), _)));
            for query in ["sea-level", "2m_temperature"] {
                app.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
                app.paste(query.into());
                let screen = render_at(&mut app, width, height);
                assert!(screen.contains("Search:"), "{screen}");
                click_hit(&mut app, |hit| matches!(hit, Hit::PlotItem(0)));
            }
            let screen = render_at(&mut app, width, height);
            assert!(screen.contains("2 selected"), "{screen}");
            capture_screen("plots-custom-search", width, height, &mut app);
            click_hit(&mut app, |hit| matches!(hit, Hit::Key(KeyCode::F(4), _)));
            let screen = render_at(&mut app, width, height);
            assert!(
                screen.contains("Save plots") && screen.contains("first frame only"),
                "{screen}"
            );
            assert!(screen.contains("Review the request"), "{screen}");
            capture_screen("plots-custom-review", width, height, &mut app);
            click_hit(&mut app, |hit| matches!(hit, Hit::Key(KeyCode::Enter, _)));
            assert!(app.dialog.is_none());
            assert!(app.status.contains("Saved Custom"));
            assert_eq!(app.plot_spec().unwrap(), "mslp_10m_winds,2m_temperature");
            assert_eq!(fs::read_to_string(&path).unwrap(), original);
            assert_eq!(app.editor.as_ref().unwrap().text(), original);
            assert!(!app.dirty());
            assert!(app.job.is_none());
            let mut reopened = App::new().unwrap();
            reopened.open(path.clone());
            assert_eq!(reopened.plot_spec().unwrap(), app.plot_spec().unwrap());
            app.begin_plots();
            app.key(press(KeyCode::F(3)));
            app.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
            app.paste("var:SNOWH,total_qpf".into());
            let screen = render_at(&mut app, width, height);
            assert!(screen.contains("var:SNOWH,total_qpf"), "{screen}");
            capture_screen("plots-advanced-edit", width, height, &mut app);
            app.key(press(KeyCode::Esc));
            app.key(press(KeyCode::Esc));
            assert_eq!(app.plot_spec().unwrap(), "mslp_10m_winds,2m_temperature");
            let export = path.with_file_name("exported.toml");
            assert!(app.save_as(export.clone()));
            assert_eq!(
                plotsettings::load(&export).unwrap().spec,
                "mslp_10m_winds,2m_temperature"
            );
            fs::write(plotsettings::sidecar(&export), "{broken").unwrap();
            app.begin_plots();
            app.plot_catalog.error = Some("renderer is unavailable".into());
            let screen = render_at(&mut app, width, height);
            assert!(screen.contains("Invalid plot settings JSON"), "{screen}");
            capture_screen("plots-settings-error", width, height, &mut app);
            assert!(app.argv(Action::Run).is_err());
            assert_eq!(fs::read_to_string(&export).unwrap(), original);
        }
    }

    #[test]
    fn plot_requests_reach_the_real_go_and_prepared_cli_parsers() {
        let mut app = loaded_app("name='argv only'\n");
        app.prepared = app.cwd.join("prepared input with spaces");
        let default = app.plot_spec().unwrap();
        assert_eq!(default.split(',').count(), 25);
        assert_eq!(App::new().unwrap().plot_spec().unwrap(), default);
        let config = app.editor.as_ref().unwrap().path.clone();
        for wanted in [
            default,
            "all".into(),
            "none".into(),
            "var:SNOWH,total_qpf".into(),
        ] {
            let previous = fs::read(plotsettings::sidecar(&config)).ok();
            plotsettings::save(
                &config,
                &plotsettings::Selection {
                    label: "Explicit".into(),
                    spec: wanted.clone(),
                },
                previous.as_deref(),
            )
            .unwrap();
            let mut commands = Vec::new();
            for action in [Action::Plan, Action::Run, Action::Prepared] {
                let (name, args) = app.argv(action).unwrap();
                let mut command = vec![name.to_string()];
                command.extend(args);
                commands.push(command);
            }
            let mut guide = Guide::new(Kind::Prepared, &app.cwd, &app.output);
            guide.questions[0].value = display_path(&app.prepared);
            guide.questions[1].value = display_path(&config);
            let request = guide.request(&app.cwd).unwrap();
            let mut command = vec![request.command];
            command.extend(request.args);
            commands.push(command);
            let python = env::var_os("GPUWM_TUI_TEST_PYTHON")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("python"));
            let code = "import json,sys\nfrom gpuwm.cli import build_parser\np=build_parser()\nfor words in json.loads(sys.argv[1]):\n a=p.parse_args(words)\n assert a.render_products == sys.argv[2], (words,a.render_products)\nassert p.parse_args(['go','untouched.toml']).render_products is None\nassert p.parse_args(['sim','prepared','--experiment-config','untouched.toml','--outdir','out']).render_products is None\nprint('PASS: actual CLI parser accepted all plot arguments; CLI defaults unchanged')";
            let output = std::process::Command::new(python)
                .args([
                    "-c",
                    code,
                    &serde_json::to_string(&commands).unwrap(),
                    &wanted,
                ])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        for action in [Action::Check, Action::Doctor] {
            assert!(!app
                .argv(action)
                .unwrap()
                .1
                .iter()
                .any(|arg| arg.contains("products")));
        }
        fs::write(plotsettings::sidecar(&config), "broken").unwrap();
        assert!(app.argv(Action::Run).is_err());
        assert!(app.argv(Action::Check).is_ok());
    }

    #[test]
    fn memory_refusal_classification_does_not_turn_other_failures_into_resizing() {
        let go = "go: memory -- the forecast needs 8.28 GiB; that EXCEEDS the 7.22 GiB budget\ngpuwm go: this configuration will not fit: the forecast needs 8.28 GiB; that EXCEEDS the 7.22 GiB budget\n";
        assert!(job::memory_refusal("go", 2, go));
        for code in [0, 1, 4, 5, 130] {
            assert!(!job::memory_refusal("go", code, go));
        }
        assert!(!job::memory_refusal("doctor", 2, go));
        assert!(!job::memory_refusal("go", 2, "go: memory -- 4.93 GiB fits the 5.16 GiB budget\ngpuwm go: Cannot declare ('shared',) twice (line 95, column 8)\n"));
        assert!(!job::memory_refusal(
            "go",
            2,
            "# remedy example: gpuwm go: this configuration will not fit:\n"
        ));
        assert!(!job::memory_refusal("go", 2, "go: memory -- it fits the budget\ngpuwm go: this configuration will not fit: unrelated fixture\n"));
        for budget in ["budget", "WDDM budget"] {
            let check = format!("  BINDING PHASE: forecast EXCEEDS the 7.22 GiB budget.\n  WARNING: observed peak envelope 8.28 GiB exceeds the {budget} 7.22 GiB\n");
            assert!(job::memory_refusal("check", 4, &check));
            for code in [0, 1, 2, 3, 5, 130] {
                assert!(!job::memory_refusal("check", code, &check));
            }
        }
        let check_one = include_str!("../tests/fixtures/check-memory-exit1.log");
        assert!(job::memory_refusal("check", 1, check_one));
        for extra in [
            "\nphysics_gate: FAIL\n",
            "\ninput preflight: FAILED\n",
            "\nGPU readiness: ERROR\n",
            "\nTraceback (most recent call last):\n",
        ] {
            assert!(
                !job::memory_refusal("check", 1, &format!("{check_one}{extra}")),
                "{extra}"
            );
        }
        assert!(!job::memory_refusal(
            "check",
            1,
            &check_one.replace(
                "alloc_estimate_le_wddm_budget: FAIL",
                "alloc_estimate_le_wddm_budget: PASS"
            )
        ));
        assert!(!job::memory_refusal(
            "check",
            1,
            &check_one.replace("WARNING: observed peak envelope", "OTHER NOTE:")
        ));
    }

    #[test]
    fn tile_recovery_reviews_a_new_file_and_expires_when_the_configuration_changes() {
        let original = include_str!("../../../configs/gfs_12km_quickstart.toml");
        let mut app = loaded_app(original);
        let path = app.editor.as_ref().unwrap().path.clone();
        let log = app.output.join("memory-refusal-ui-fixture.log");
        fs::create_dir_all(&app.output).unwrap();
        fs::write(&log, "go: memory -- the forecast needs 8.28 GiB; that EXCEEDS the 7.22 GiB budget\ngpuwm go: this configuration will not fit: the forecast needs 8.28 GiB; that EXCEEDS the 7.22 GiB budget\n").unwrap();
        app.status = "Memory admission refused. Review Fit domain or Tile streaming.".into();
        app.startup_failure = Some(StartupFailure {
            message: "Memory refusal UI fixture".into(),
            log: Some(log),
        });
        app.memory_recovery_config = Some(path.clone()); // Classification is tested through real workers separately.
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            app.dialog = None;
            let screen = render_at(&mut app, width, height);
            assert!(
                screen.contains("Fit domain") && screen.contains("Tile streaming"),
                "{screen}"
            );
            click_hit(&mut app, |hit| matches!(hit, Hit::Details));
            let screen = render_at(&mut app, width, height);
            assert!(
                screen.contains("C Copy details")
                    && screen.contains("Y Copy log")
                    && screen.contains("L View log")
                    && screen.contains("Fit domain")
                    && screen.contains("Tile streaming")
                    && screen.contains("Close"),
                "{screen}"
            );
            capture_screen("memory-recovery-choices", width, height, &mut app);
            if width == 65 {
                app.key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL));
            } else {
                click_hit(&mut app, |hit| matches!(hit, Hit::TileCurrent));
            }
            let Some(Dialog::Summary(guide, 0)) = &app.dialog else {
                panic!("short streaming review");
            };
            assert_eq!(guide.kind, Kind::Tiles);
            assert_eq!(guide.questions.len(), 3);
            assert_eq!(guide.questions[0].value, display_path(&path));
            assert_eq!(guide.questions[2].value, "auto");
            let request = guide.request(&app.cwd).unwrap();
            let new_path = request.created.clone().unwrap();
            assert_ne!(new_path, path);
            assert_eq!(
                new_path.parent().unwrap().canonicalize().unwrap(),
                path.parent().unwrap().canonicalize().unwrap()
            );
            assert_eq!(request.command, "domain-tiles");
            assert_eq!(
                request.args,
                vec![
                    display_path(&path),
                    "--out".into(),
                    display_path(&new_path),
                    "--mode=auto".into(),
                    "--write".into()
                ]
            );
            render_at(&mut app, width, height);
            capture_screen("memory-tile-settings", width, height, &mut app);
            click_hit(&mut app, |hit| {
                matches!(hit, Hit::Key(KeyCode::Char('n'), _))
            });
            assert!(
                matches!(&app.dialog, Some(Dialog::Review(request, _, _)) if request.command == "domain-tiles")
            );
            let screen = render_at(&mut app, width, height);
            assert!(
                screen.contains("Create configuration (Enter)")
                    && screen.contains("Preserve the domain area"),
                "{screen}"
            );
            capture_screen("memory-tile-command-review", width, height, &mut app);
            let mut command_visible = false;
            for _ in 0..30 {
                let screen = render_at(&mut app, width, height);
                if screen.contains("domain-tiles") {
                    capture_screen("memory-tile-command-scrolled", width, height, &mut app);
                    command_visible = true;
                    break;
                }
                app.key(press(KeyCode::Down));
            }
            assert!(
                command_visible,
                "The reviewed command must be reachable by scrolling"
            );
            assert!(app.job.is_none());
            assert!(!new_path.exists());
            assert_eq!(fs::read_to_string(&path).unwrap(), original);
            app.key(press(KeyCode::Esc));
            assert!(matches!(app.dialog, Some(Dialog::Summary(_, _))));
            app.dialog = None;
        }
        app.editor.as_mut().unwrap().insert("# unsaved\n");
        let draft = app.editor.as_ref().unwrap().text();
        app.show_details();
        render_at(&mut app, 65, 20);
        click_hit(&mut app, |hit| matches!(hit, Hit::TileCurrent));
        assert!(app.dialog.is_none());
        assert!(app.status.contains("Save the current draft"));
        assert_eq!(app.editor.as_ref().unwrap().text(), draft);
        assert_eq!(fs::read_to_string(&path).unwrap(), original);
        app.save();
        assert!(!app.memory_recovery_available());
        app.memory_recovery_config = Some(path.clone());
        let other = path.with_file_name("other-configuration.toml");
        fs::write(&other, original).unwrap();
        app.open(other);
        assert!(!app.memory_recovery_available());
        let screen = render_at(&mut app, 65, 20);
        assert!(!screen.contains("Fit domain") && !screen.contains("Tile streaming"));
        app.tile_current();
        assert!(app.dialog.is_none() && app.status.contains("not a confirmed memory refusal"));
    }

    #[test]
    fn a_creation_recovery_draft_enters_the_existing_fit_tiles_and_run_workflow() {
        let original = include_str!("../../../configs/gfs_12km_quickstart.toml");
        let mut app = loaded_app(original);
        let existing = app.editor.as_ref().unwrap().path.clone();
        let draft = existing.with_file_name("rejected-complete-draft.toml");
        fs::write(&draft, original).unwrap();
        app.startup_failure = Some(StartupFailure { message: "Confirmed memory refusal".into(), log: None });
        app.open_configuration_recovery(draft.clone());
        assert_eq!(app.editor.as_ref().unwrap().path, draft);
        assert!(app.memory_recovery_available());
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let screen = render_at(&mut app, width, height);
            assert!(screen.contains("Fit domain") && screen.contains("Tile streaming"), "{screen}");
        }
        app.tile_current();
        let Some(Dialog::Summary(guide, _)) = &app.dialog else { panic!("Shared streaming review"); };
        let request = guide.request(&app.cwd).unwrap();
        assert_eq!(request.command, "domain-tiles");
        assert_eq!(guide.questions[0].value, display_path(&draft));
        app.fit_current();
        let Some(Dialog::Guide(guide)) = &app.dialog else { panic!("Shared fit guide"); };
        assert_eq!(guide.kind, Kind::Fit);
        assert_eq!(guide.questions[0].value, display_path(&draft));
        assert_eq!(app.argv(Action::Run).unwrap().0, "go");
        assert_eq!(fs::read_to_string(&existing).unwrap(), original);
        app.editor.as_mut().unwrap().insert("# unsaved\n");
        let edited = app.editor.as_ref().unwrap().text();
        app.open_configuration_recovery(existing);
        assert_eq!(app.editor.as_ref().unwrap().text(), edited);
        assert!(app.status.contains("Save your open edits"));
    }

    #[test]
    fn node_logs_have_visible_copy_and_plain_text_open_without_changing_job_or_draft(){
        let original="# untouched saved configuration\na=1\n";
        let mut app=loaded_app(original);
        let saved=app.editor.as_ref().unwrap().path.clone();
        app.editor.as_mut().unwrap().insert("# draft\n");
        let draft=app.editor.as_ref().unwrap().text();
        let mut node=remote::Node::blank();node.name="Node fixture".into();node.host="fixture-node".into();node.last_job=Some("job-fixture".into());
        app.nodes.store.active=Some(node.id.clone());app.nodes.store.nodes.push(node.clone());
        app.nodes.view.log="native prepare output café 界\nFAILED explicit forcing\n".into();
        app.nodes.view.status=Some(serde_json::json!({"id":"job-fixture","state":"failed","error":"Exact native failure"}));
        app.node_panel.screen=node_ui::Screen::Job;
        app.dialog=None;app.tab=Tab::Overview;app.key(press(KeyCode::Char('l')));
        assert!(matches!(app.dialog,Some(Dialog::Nodes)));
        app.clipboard_hook=Some(record_copy);
        for(width,height)in[(65,20),(80,24),(120,36)]{
            let screen=render_at(&mut app,width,height);
            assert!(screen.contains("Y Copy logs")&&screen.contains("O Open log"),"{screen}");
        }
        let text=app.node_log_text();
        app.key(press(KeyCode::Char('y')));
        COPIED.with(|copies|assert_eq!(copies.borrow().last(),Some(&text)));
        assert!(app.node_panel.notice.contains("Copied node logs"));
        let path=app.save_log_text(&text).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(),text);
        assert_eq!(path.extension().and_then(|s|s.to_str()),Some("txt"));
        app.key(press(KeyCode::Char('o')));
        assert!(app.node_panel.notice.contains("Opened plain-text log"));
        assert_eq!(app.nodes.store.selected().unwrap().id,node.id);
        assert_eq!(app.nodes.store.selected().unwrap().last_job.as_deref(),Some("job-fixture"));
        assert_eq!(app.editor.as_ref().unwrap().text(),draft);
        assert_eq!(fs::read_to_string(saved).unwrap(),original);
        app.clipboard_hook=Some(|_|Err("clipboard unavailable".into()));
        app.key(press(KeyCode::Char('y')));
        assert!(app.node_panel.notice.contains("O opens a plain-text log"));
    }
    #[test]
    fn diagnostic_copy_buttons_copy_full_text_and_keep_failures_visible() {
        let original = "# original configuration\na=1\n";
        let mut app = loaded_app(original);
        let log = app.output.join("clipboard-fixture.log");
        fs::create_dir_all(&app.output).unwrap();
        let full_log = format!(
            "FIRST SAVED LINE\n{}\nLAST SAVED LINE 界 🌦\n",
            "Long diagnostic café. ".repeat(8000)
        );
        fs::write(&log, &full_log).unwrap();
        app.startup_failure = Some(StartupFailure {
            message: "fixture failure".into(),
            log: Some(log.clone()),
        });
        app.status = "FAILED fixture command".into();
        app.clipboard_hook = Some(record_copy);
        app.editor.as_mut().unwrap().insert("# unsaved draft\n");
        let draft = app.editor.as_ref().unwrap().text();
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            app.show_details();
            let screen = render_at(&mut app, width, height);
            for label in ["C Copy details", "Y Copy log", "L View log", "Close"] {
                assert!(screen.contains(label), "missing {label}: {screen}");
            }
            assert!(!screen.contains("Fit domain") && !screen.contains("Tile streaming"));
            let Some(Dialog::Details { text, .. }) = &app.dialog else {
                panic!("details");
            };
            let details = text.clone();
            assert!(details.contains(&display_path(&log)));
            assert!(!details.contains("FIRST SAVED LINE")); // Recent view is bounded.
            click_hit(&mut app, |hit| {
                matches!(hit, Hit::Key(KeyCode::Char('c'), _))
            });
            COPIED.with(|copies| assert_eq!(copies.borrow().last().unwrap(), &details));
            assert!(render_at(&mut app, width, height).contains("Copied details"));
            click_hit(&mut app, |hit| {
                matches!(hit, Hit::Key(KeyCode::Char('y'), _))
            });
            COPIED.with(|copies| assert_eq!(copies.borrow().last().unwrap(), &full_log));
            let screen = render_at(&mut app, width, height);
            assert!(screen.contains("Copied complete saved log"), "{screen}");
            capture_screen("clipboard-copy-success", width, height, &mut app);
            assert_eq!(app.status, "FAILED fixture command");
            assert_eq!(app.badge(), "FAILED · DRAFT");
            app.clipboard_hook = Some(|_| {
                Err("Clipboard unavailable. Open the saved log; Home shows its path.".into())
            });
            app.key(press(KeyCode::Char('c')));
            let screen = render_at(&mut app, width, height);
            assert!(
                screen.contains("Copy failed:")
                    && screen.contains("Clipboard unavailable")
                    && screen.contains("shows its path."),
                "{screen}"
            );
            assert!(matches!(&app.dialog, Some(Dialog::Details { text, .. }) if text == &details));
            capture_screen("clipboard-copy-failure", width, height, &mut app);
            app.clipboard_hook = Some(record_copy);
            let count = COPIED.with(|copies| copies.borrow().len());
            render_at(&mut app, 64, 19);
            app.key(press(KeyCode::Char('c')));
            app.mouse(MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: 4,
                row: 15,
                modifiers: KeyModifiers::NONE,
            });
            COPIED.with(|copies| assert_eq!(copies.borrow().len(), count));
            assert_eq!(app.editor.as_ref().unwrap().text(), draft);
            assert_eq!(
                fs::read_to_string(&app.editor.as_ref().unwrap().path).unwrap(),
                original
            );
            assert!(app.job.is_none());
        }
        fs::remove_file(&log).unwrap();
        render_at(&mut app, 65, 20);
        app.key(press(KeyCode::Char('y')));
        assert!(render_at(&mut app, 65, 20).contains("Cannot read saved log"));
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "Writes harmless fixtures to the real Windows clipboard; run explicitly in an interactive user session"]
    fn windows_user_clipboard_roundtrip_from_details_and_full_log() {
        fn read_our_fixture() -> String {
            use windows_sys::Win32::System::{
                DataExchange::{CloseClipboard, GetClipboardData, OpenClipboard},
                Memory::{GlobalLock, GlobalSize, GlobalUnlock},
            };
            // Called only after our successful fixture write. Never inspect the
            // user's previous clipboard, and do not print clipboard contents.
            unsafe {
                let deadline = std::time::Instant::now() + Duration::from_secs(2);
                while OpenClipboard(std::ptr::null_mut()) == 0 {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "Clipboard remained busy after fixture write"
                    );
                    std::thread::sleep(Duration::from_millis(25));
                }
                let memory = GetClipboardData(13);
                assert!(!memory.is_null());
                let data = GlobalLock(memory).cast::<u16>();
                assert!(!data.is_null());
                let units = std::slice::from_raw_parts(data, GlobalSize(memory) / 2);
                let end = units.iter().position(|unit| *unit == 0).unwrap();
                let text = String::from_utf16(&units[..end]).unwrap();
                GlobalUnlock(memory);
                CloseClipboard();
                text
            }
        }
        let mut app = loaded_app("# harmless clipboard test\na=1\n");
        let log = app.output.join("harmless-clipboard-fixture.log");
        fs::create_dir_all(&app.output).unwrap();
        let fixture = format!(
            "ArWen clipboard verification — café 界 🌦\n{}\nEND FIXTURE\n",
            "literal $() 'quotes' `text`\n".repeat(7000)
        );
        fs::write(&log, &fixture).unwrap();
        app.startup_failure = Some(StartupFailure {
            message: "Harmless clipboard verification fixture".into(),
            log: Some(log.clone()),
        });
        app.show_details();
        render_at(&mut app, 65, 20);
        let Some(Dialog::Details { text, .. }) = &app.dialog else {
            panic!("details");
        };
        let expected_details = text.clone();
        click_hit(&mut app, |hit| {
            matches!(hit, Hit::Key(KeyCode::Char('c'), _))
        });
        assert!(
            matches!(&app.dialog, Some(Dialog::Details { notice: Some(notice), .. }) if notice.starts_with("Copied details"))
        );
        assert_eq!(read_our_fixture(), expected_details);
        render_at(&mut app, 65, 20);
        click_hit(&mut app, |hit| {
            matches!(hit, Hit::Key(KeyCode::Char('y'), _))
        });
        assert!(
            matches!(&app.dialog, Some(Dialog::Details { notice: Some(notice), .. }) if notice.starts_with("Copied complete saved log"))
        );
        assert_eq!(read_our_fixture(), fixture);
        assert_eq!(fs::read_to_string(&log).unwrap(), fixture);
        fs::remove_file(log).unwrap();
        clipboard::copy("ArWen clipboard verification passed.").unwrap();
    }

    #[test]
    fn status_details_are_clickable_scrollable_and_preserve_editor_input() {
        let original = "# unchanged café\na=1\n";
        let mut app = loaded_app(original);
        app.tab = Tab::Settings;
        app.editor.as_mut().unwrap().insert("# unsaved draft\n");
        let draft = app.editor.as_ref().unwrap().text();
        app.status = format!(
            "Could not save: {}\nFINAL ERROR DETAIL",
            "long path with spaces/界/".repeat(70)
        );
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            app.dialog = None;
            let screen = render_at(&mut app, width, height);
            assert!(screen.contains("Ctrl+L Details"), "{screen}");
            // The yellow summary itself is actionable, including its second row.
            let area = app
                .hits
                .iter()
                .find(|hit| matches!(hit.action, Hit::Details) && hit.area.height == 2)
                .unwrap()
                .area;
            app.mouse(MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: area.right() - 1,
                row: area.bottom() - 1,
                modifiers: KeyModifiers::NONE,
            });
            assert!(matches!(app.dialog, Some(Dialog::Details { .. })));
            let screen = render_at(&mut app, width, height);
            assert!(
                screen.contains("Command / error details") && screen.contains("Close"),
                "{screen}"
            );
            assert!(!app
                .hits
                .iter()
                .any(|hit| matches!(hit.action, Hit::View(_) | Hit::Action(_) | Hit::Editor(_))));
            click_hit(&mut app, |hit| matches!(hit, Hit::Key(KeyCode::End, _)));
            let screen = render_at(&mut app, width, height);
            assert!(screen.contains("FINAL ERROR DETAIL"), "{screen}");
            let before = match &app.dialog {
                Some(Dialog::Details { offset, .. }) => *offset,
                _ => panic!("details"),
            };
            app.mouse(MouseEvent {
                kind: MouseEventKind::ScrollUp,
                column: 10,
                row: 10,
                modifiers: KeyModifiers::NONE,
            });
            assert!(
                matches!(&app.dialog, Some(Dialog::Details { offset, .. }) if *offset == before.saturating_sub(1))
            );
            app.key(press(KeyCode::Home));
            assert!(render_at(&mut app, width, height).contains("Could not save"));
            capture_screen("error-details-long-message", width, height, &mut app);
            click_hit(&mut app, |hit| matches!(hit, Hit::Key(KeyCode::Esc, _)));
            assert!(app.dialog.is_none());
            render_at(&mut app, width, height);
            let button = app
                .hits
                .iter()
                .find(|hit| matches!(hit.action, Hit::Details) && hit.area.height == 1)
                .unwrap()
                .area;
            app.mouse(MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: button.x + 1,
                row: button.y,
                modifiers: KeyModifiers::NONE,
            });
            assert!(matches!(app.dialog, Some(Dialog::Details { .. })));
            app.key(press(KeyCode::Esc));
            app.key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL));
            assert!(matches!(app.dialog, Some(Dialog::Details { .. })));
            render_at(&mut app, 64, 19);
            app.key(press(KeyCode::Esc));
            assert!(matches!(app.dialog, Some(Dialog::Details { .. })));
            assert_eq!(app.editor.as_ref().unwrap().text(), draft);
            assert_eq!(
                fs::read_to_string(&app.editor.as_ref().unwrap().path).unwrap(),
                original
            );
            assert!(app.job.is_none());
        }
    }

    #[test]
    fn fit_recovery_prefills_existing_values_and_proposes_only_a_new_file() {
        let original = "[experiment]\nstart_time = 2026-09-05T18:00:00 # exact UTC\nrun_seconds = 10800\n[projection]\nref_lat = 35.5 # requested center\nref_lon = -97 # requested center\n[shared]\nmp_physics = 10\n";
        let mut app = loaded_app(original);
        let path = app.editor.as_ref().unwrap().path.clone();
        app.start_command("go", &[], None); // Intentionally missing test interpreter; no CLI can run.
        assert_eq!(app.badge(), "FAILED");
        assert!(!app.memory_recovery_available());
        // A preclassified memory refusal for this editor drives the UI fixture;
        // worker-process tests separately prove that classification is earned.
        app.memory_recovery_config = Some(path.clone());
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            app.dialog = None;
            let screen = render_at(&mut app, width, height);
            assert!(screen.contains("Fit domain") && screen.contains("Tile streaming"));
            click_hit(&mut app, |hit| matches!(hit, Hit::Details));
            render_at(&mut app, width, height);
            if width == 65 {
                app.key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::CONTROL));
            } else {
                click_hit(&mut app, |hit| matches!(hit, Hit::FitCurrent));
            }
            let Some(Dialog::Guide(guide)) = &app.dialog else {
                panic!("Fit recovery guide");
            };
            assert_eq!(guide.kind, Kind::Fit);
            assert_eq!(
                PathBuf::from(&guide.questions[0].value)
                    .canonicalize()
                    .unwrap(),
                path
            );
            assert_eq!(guide.questions[1].value, "35.5,-97");
            assert_eq!(guide.questions[3].value, "2026-09-05T18:00:00");
            assert!(guide.questions[4].value.is_empty()); // Preserve duration.
            assert!(guide.questions[5].value.is_empty()); // Measure local available memory.
            let request = guide.request(&app.cwd).unwrap();
            assert_eq!(request.command, "domain-fit");
            let created = request.created.unwrap();
            assert_ne!(created, path);
            assert!(!created.exists());
            assert!(request.args.iter().any(|arg| arg == "--write"));
            assert!(request.args.iter().any(|arg| arg == "--point=35.5,-97"));
            assert_eq!(fs::read_to_string(&path).unwrap(), original);
            assert!(!app.dirty() && app.job.is_none());
            render_at(&mut app, width, height);
            capture_screen("error-fit-recovery", width, height, &mut app);
        }
        app.dialog = None;
        app.editor.as_mut().unwrap().insert("# unsaved\n");
        let draft = app.editor.as_ref().unwrap().text();
        app.show_details();
        render_at(&mut app, 65, 20);
        click_hit(&mut app, |hit| matches!(hit, Hit::FitCurrent));
        assert!(!matches!(app.dialog, Some(Dialog::Guide(_))));
        assert!(app.dialog.is_none());
        assert!(app.status.contains("Save the current draft"));
        assert!(render_at(&mut app, 65, 20).contains("Save the current draft"));
        assert_eq!(app.badge(), "FAILED · DRAFT");
        assert_eq!(app.editor.as_ref().unwrap().text(), draft);
        assert_eq!(fs::read_to_string(path).unwrap(), original);
    }

    fn wait_for_ui_job(app: &mut App) {
        let deadline = std::time::Instant::now() + Duration::from_secs(45);
        while app.busy() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(30));
            app.poll();
        }
        assert!(!app.busy(), "{}", app.log_text());
    }

    #[test]
    fn native_exit_two_and_later_startup_failure_open_the_correct_attempt() {
        let Some(python) = env::var_os("GPUWM_TUI_TEST_PYTHON") else {
            return;
        };
        let original = include_str!("../../../configs/gfs_12km_quickstart.toml");
        let mut app = loaded_app(original);
        app.python = python.into();
        assert_eq!(app.badge(), "READY");
        app.start_command("go", &[], None); // Missing required CONFIG: parser exits before any forecast/data work.
        assert_eq!(app.badge(), "RUNNING");
        app.show_details();
        wait_for_ui_job(&mut app);
        assert!(
            matches!(&app.dialog, Some(Dialog::Details { text, .. }) if text.contains("FAILED (exit 2)") && !text.contains("— RUNNING"))
        );
        assert_eq!(
            app.job.as_ref().unwrap().outcome,
            Some(2),
            "{}",
            app.log_text()
        );
        let failed_log = app.log_path().unwrap();
        let result: serde_json::Value = serde_json::from_slice(
            &fs::read(app.job.as_ref().unwrap().dir.join("result.json")).unwrap(),
        )
        .unwrap();
        assert!(
            result["error"].is_null(),
            "Ordinary exit 2 must be readable without an exception field"
        );
        // Preserve the actual parser failure, then append a bounded display fixture for the reported memory refusal.
        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&failed_log)
            .unwrap();
        writeln!(file, "\nMemory gate: 8.28 GiB need exceeds 7.22 GiB budget.\nRefused before downloading forcing data.").unwrap();
        drop(file);
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            app.dialog = None;
            let screen = render_at(&mut app, width, height);
            assert!(
                screen.lines().next().unwrap().contains("FAILED"),
                "{screen}"
            );
            assert!(!screen.lines().next().unwrap().contains("READY"));
            assert!(screen.contains("Ctrl+L Details"));
            assert!(!screen.contains("Fit domain") && !screen.contains("Tile streaming"));
            capture_screen("error-exit-two", width, height, &mut app);
            click_hit(&mut app, |hit| matches!(hit, Hit::Details));
            let screen = render_at(&mut app, width, height);
            assert!(
                screen.contains("8.28 GiB") && screen.contains("7.22 GiB"),
                "{screen}"
            );
            assert!(
                screen.contains("L View log") && screen.contains("Close"),
                "{screen}"
            );
            capture_screen("error-exit-two-details", width, height, &mut app);
            if let Some(Dialog::Details { text, .. }) = &app.dialog {
                assert!(text.contains(&display_path(&failed_log)));
                assert!(text.contains("FAILED (exit 2)"));
            }
            app.log_offset = 400;
            click_hit(&mut app, |hit| {
                matches!(hit, Hit::Key(KeyCode::Char('l'), _))
            });
            assert!(app.dialog.is_none() && app.tab == Tab::Logs && app.log_offset == 0);
            assert!(render_at(&mut app, width, height).contains("8.28 GiB"));
        }
        app.editor.as_mut().unwrap().insert("# next attempt\n");
        assert_eq!(app.badge(), "FAILED · DRAFT");
        app.start_command("version", &[], None);
        wait_for_ui_job(&mut app);
        assert_eq!(
            app.job.as_ref().unwrap().outcome,
            Some(0),
            "{}",
            app.log_text()
        );
        assert_eq!(app.badge(), "UNSAVED DRAFT");
        app.editor.as_mut().unwrap().dirty = false;
        assert_eq!(app.badge(), "COMPLETED");
        app.job.as_mut().unwrap().outcome = Some(130);
        assert_eq!(app.badge(), "INTERRUPTED");
        app.job.as_mut().unwrap().outcome = Some(1); // Includes OS-zero jobs rejected by completion validation.
        assert_eq!(app.badge(), "FAILED");
        app.python = app.output.join("no-interpreter-next-attempt.exe");
        app.start_command("go", &[], None);
        assert!(app.job.is_none() && app.startup_failure.is_some());
        assert_ne!(app.log_path().unwrap(), failed_log);
        app.show_details();
        let Some(Dialog::Details { text, .. }) = &app.dialog else {
            panic!("startup details");
        };
        assert!(text.contains("Could not start go"));
        assert!(!text.contains(&display_path(&failed_log)));
        assert!(!text.contains("8.28 GiB"));
        assert_eq!(
            fs::read_to_string(&app.editor.as_ref().unwrap().path).unwrap(),
            original
        );
    }

    #[test]
    fn domains_mouse_edits_apply_only_to_draft_and_cancel_preserves_original() {
        let original = include_str!("../../../configs/nest_lifecycle_20240521_4km.toml");
        let mut app = loaded_app(original);
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let text = render_at(&mut app, width, height);
            assert!(text.contains("D Domains"));
            assert!(app.hits.iter().any(|h| matches!(h.action, Hit::Domains)));
        }
        click_hit(&mut app, |h| matches!(h, Hit::Domains));
        render_at(&mut app, 80, 24);
        click_hit(&mut app, |h| matches!(h, Hit::DomainSelect(1)));
        assert!(matches!(app.dialog, Some(Dialog::Domains(1))));
        assert_eq!(app.editor.as_ref().unwrap().text(), original);
        assert!(!app.dirty());
        render_at(&mut app, 80, 24);
        click_hit(&mut app, |h| matches!(h, Hit::Key(KeyCode::Char('m'), _)));
        render_at(&mut app, 80, 24);
        click_hit(&mut app, |h| matches!(h, Hit::Choice(0)));
        render_at(&mut app, 80, 24);
        click_hit(&mut app, |h| matches!(h, Hit::DomainField(0)));
        app.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
        app.paste("not a number".into());
        app.key(press(KeyCode::Enter));
        app.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL));
        assert!(app.status.contains("valid"));
        assert_eq!(app.editor.as_ref().unwrap().text(), original);
        assert!(!app.dirty());
        app.key(press(KeyCode::Esc));
        assert!(matches!(app.dialog, Some(Dialog::Domains(1))));
        app.key(press(KeyCode::Enter));
        if let Some(Dialog::DomainForm(form, _)) = &mut app.dialog {
            form.fields[0].value = "180".into();
        } else {
            panic!("geometry form");
        }
        app.key(press(KeyCode::F(2)));
        assert!(app.dirty());
        let editor = app.editor.as_ref().unwrap();
        assert_eq!(fs::read_to_string(&editor.path).unwrap(), original);
        assert_eq!(
            editor.text().parse::<toml_edit::DocumentMut>().unwrap()["domain"][1]["nx"]
                .as_integer(),
            Some(180)
        );
        assert!(app.job.is_none());
    }

    #[test]
    fn tracking_choices_are_clickable_and_cancel_does_not_enable_them() {
        let original = include_str!("../../../configs/nest_lifecycle_20240521_4km.toml");
        let mut app = loaded_app(original);
        app.dialog = Some(Dialog::DomainMenu(1, 2));
        app.key(press(KeyCode::Enter));
        render_at(&mut app, 65, 20);
        click_hit(&mut app, |h| matches!(h, Hit::DomainField(0)));
        render_at(&mut app, 65, 20);
        click_hit(&mut app, |h| matches!(h, Hit::DomainValue(1)));
        app.key(press(KeyCode::Enter));
        app.key(press(KeyCode::Down));
        app.key(press(KeyCode::Enter));
        render_at(&mut app, 65, 20);
        click_hit(&mut app, |h| matches!(h, Hit::DomainValue(0)));
        assert!(render_at(&mut app, 65, 20).contains("> pressure"));
        app.key(press(KeyCode::Right));
        if let Some(Dialog::DomainForm(form, _)) = &app.dialog {
            assert_eq!(form.fields[1].value, "uh");
        }
        app.key(press(KeyCode::Esc));
        app.key(press(KeyCode::Esc));
        assert_eq!(app.editor.as_ref().unwrap().text(), original);
        assert!(!app.dirty());
        assert!(app.job.is_none());
    }

    #[test]
    fn new_nest_shortcut_and_historical_downscale_are_visible_without_commands() {
        let mut app = App::new().unwrap();
        app.dialog = Some(Dialog::Guide(complete_guide(&app)));
        render_at(&mut app, 65, 20);
        click_hit(&mut app, |h| matches!(h, Hit::GuideGrid(4)));
        if let Some(Dialog::Guide(guide)) = &mut app.dialog {
            assert_eq!(guide.step, 4);
            guide.questions[4].value = "3,2".into();
        }
        app.key(press(KeyCode::Enter));
        assert!(matches!(app.dialog, Some(Dialog::Summary(_, 5))));
        app.begin_domains();
        render_at(&mut app, 80, 24);
        click_hit(&mut app, |h| matches!(h, Hit::Choice(1)));
        assert!(matches!(&app.dialog, Some(Dialog::Guide(g)) if g.kind == Kind::Downscale));
        assert!(app.job.is_none());
    }

    fn capture_screen(name: &str, width: u16, height: u16, app: &mut App) {
        if let Some(dir) = env::var_os("GPUWM_TUI_SNAPSHOT_DIR") {
            let dir = PathBuf::from(dir);
            fs::create_dir_all(&dir).unwrap();
            let name = format!("{name}-{width}x{height}");
            let (buffer, cursor) = snapshot_buffer(app, width, height).unwrap();
            let mut text = String::new();
            for y in 0..height {
                for x in 0..width {
                    text.push_str(buffer[(x, y)].symbol());
                }
                text.push('\n');
            }
            fs::write(dir.join(format!("{name}.txt")), text).unwrap();
            fs::write(
                dir.join(format!("{name}.html")),
                buffer_html(&buffer, &name, cursor),
            )
            .unwrap();
            fs::write(
                dir.join(format!("{name}.cells.json")),
                serde_json::to_vec_pretty(&cell_document(&buffer, &name, cursor)).unwrap(),
            )
            .unwrap();
        }
    }

    #[test]
    fn every_domain_field_and_action_remains_visible_at_supported_sizes() {
        let original = include_str!("../../../configs/nest_lifecycle_20240521_4km.toml");
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let mut app = loaded_app(original);
            app.begin_domains();
            render_at(&mut app, width, height);
            capture_screen("domains", width, height, &mut app);
            app.dialog = Some(Dialog::DomainMenu(1, 0));
            render_at(&mut app, width, height);
            capture_screen("domain-menu", width, height, &mut app);
            for (section, _) in domains::SECTIONS {
                let form = domains::Form::new(original.into(), 1, section).unwrap();
                let fields = form.fields.len();
                app.dialog = Some(Dialog::DomainForm(form, None));
                for index in 0..fields {
                    if let Some(Dialog::DomainForm(form, _)) = &mut app.dialog {
                        form.selected = index;
                    }
                    let screen = render_at(&mut app, width, height);
                    assert!(screen.contains("Apply to draft"), "{screen}");
                    assert!(
                        app.hits
                            .iter()
                            .any(|h| matches!(h.action, Hit::DomainField(i) if i == index)),
                        "{screen}"
                    );
                    if index == 0 || index + 1 == fields {
                        capture_screen(
                            &format!("{section:?}-fields-{index}"),
                            width,
                            height,
                            &mut app,
                        );
                    }
                    click_hit(&mut app, |h| matches!(h, Hit::DomainField(i) if i == index));
                    let screen = render_at(&mut app, width, height);
                    assert!(screen.contains("Keep field"), "{screen}");
                    assert!(screen.contains("> "), "{screen}");
                    capture_screen(
                        &format!("{section:?}-edit-{index}"),
                        width,
                        height,
                        &mut app,
                    );
                    if let Some(Dialog::DomainForm(form, _)) = &app.dialog {
                        for choice in 0..form.fields[index].choices().len() {
                            assert!(
                                app.hits.iter().any(
                                    |h| matches!(h.action, Hit::DomainValue(i) if i == choice)
                                ),
                                "{screen}"
                            );
                        }
                    }
                    app.key(press(KeyCode::Esc));
                }
                app.key(press(KeyCode::Esc));
            }
            assert_eq!(app.editor.as_ref().unwrap().text(), original);
            assert!(!app.dirty());
            assert!(app.job.is_none());
        }
    }

    #[test]
    fn downscale_fields_and_plan_run_review_stay_visible_without_starting() {
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let mut app = loaded_app("a=1\n");
            let mut guide = Guide::new(Kind::Downscale, &app.cwd, &app.output);
            guide.questions[0].value = "archived parent".into();
            guide.questions[1].value = "35.3,-97.5".into();
            guide.questions[3].value = "parent.gpuwmrst".into();
            guide.questions[11].value = "900".into();
            guide.questions[18].value = app
                .output
                .join("new-downscaled-child")
                .to_string_lossy()
                .into();
            for index in 0..guide.questions.len() {
                guide.step = index;
                app.dialog = Some(Dialog::Guide(guide.clone()));
                let screen = render_at(&mut app, width, height);
                capture_screen(&format!("downscale-edit-{index}"), width, height, &mut app);
                assert!(screen.contains("> "), "{screen}");
                assert!(screen.contains("All settings"), "{screen}");
            }
            app.dialog = Some(Dialog::Summary(guide.clone(), guide.questions.len()));
            let screen = render_at(&mut app, width, height);
            capture_screen("downscale-summary", width, height, &mut app);
            assert!(screen.contains("Next: review command"), "{screen}");
            for action in ["plan", "run"] {
                guide.questions[19].value = action.into();
                let request = guide.request(&app.cwd).unwrap();
                app.dialog = Some(Dialog::Review(request, Some(guide.clone()), 0));
                let screen = render_at(&mut app, width, height);
                capture_screen(
                    &format!("downscale-review-{action}"),
                    width,
                    height,
                    &mut app,
                );
                assert!(
                    screen.contains(if action == "plan" {
                        "Write downscale plan"
                    } else {
                        "Run offline child"
                    }),
                    "{screen}"
                );
                let mut reviewed = screen;
                for offset in 1..=32 {
                    app.key(press(KeyCode::Down));
                    let screen = render_at(&mut app, width, height);
                    if [8, 16].contains(&offset) {
                        capture_screen(
                            &format!("downscale-review-{action}-scroll-{offset}"),
                            width,
                            height,
                            &mut app,
                        );
                    }
                    reviewed.push_str(&screen);
                }
                assert!(reviewed.contains("--point"), "{reviewed}");
                assert!(reviewed.contains("--out"), "{reviewed}");
                assert_eq!(reviewed.contains("--dry-run"), action == "plan");
            }
            assert!(app.job.is_none());
            assert!(!app.dirty());
        }
    }

    fn render_at(app: &mut App, width: u16, height: u16) -> String {
        let mut term = Terminal::new(TestBackend::new(width, height)).unwrap();
        term.draw(|f| draw(f, app)).unwrap();
        let mut text = String::new();
        for y in 0..height {
            for x in 0..width {
                text.push_str(term.backend().buffer()[(x, y)].symbol());
            }
            text.push('\n');
        }
        text
    }
    fn click_hit(app: &mut App, matches: impl Fn(Hit) -> bool) {
        let area = app
            .hits
            .iter()
            .find(|h| matches(h.action))
            .expect("visible clickable action")
            .area;
        app.mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: area.x,
            row: area.y,
            modifiers: KeyModifiers::NONE,
        });
    }
    fn complete_guide(app: &App) -> Guide {
        let mut g = Guide::new(Kind::New, &app.cwd, &app.output);
        g.questions[1].value = "35.3,-97.5".into();
        g.questions[7].value = "gfs".into();
        g.questions[8].value = "2026-09-05T00".into();
        g
    }
    #[test]
    fn calendar_keeps_all_days_hours_and_actions_reachable_at_supported_sizes() {
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let mut app = loaded_app("a=1\n");
            let mut guide = complete_guide(&app);
            guide.step = 8;
            let original = guide.questions[8].value.clone();
            let form = calendar::Form::fixture(serde_json::json!({
                "schema":"gpuwm.source-availability.v1", "source_id":"era5-l137",
                "display_name":"ERA5 model-level analyses", "hours":6,
                "cycle_hours":(0..24).collect::<Vec<_>>(), "cycle_grid":{},
                "earliest":"1940-01-01T00", "latest_candidate":"2026-09-01T12",
                "probeable":false, "latest_supported":true,
            }), "2026-08-05T00"); // August needs all six calendar rows.
            app.dialog = Some(Dialog::Calendar(guide, form));
            let screen = render_at(&mut app, width, height);
            assert!(screen.contains("Use date") && screen.contains("Latest expected"), "{screen}");
            for day in 1..=31 { assert!(app.hits.iter().any(|h| matches!(h.action, Hit::CalendarDay(value) if value == day)), "day {day}: {screen}"); }
            for hour in 0..24 { assert!(app.hits.iter().any(|h| matches!(h.action, Hit::CalendarHour(value) if value == hour)), "hour {hour}: {screen}"); }
            click_hit(&mut app, |h| matches!(h, Hit::CalendarDay(31)));
            render_at(&mut app, width, height);
            click_hit(&mut app, |h| matches!(h, Hit::CalendarHour(18)));
            capture_screen("source-calendar", width, height, &mut app);
            app.key(press(KeyCode::Esc));
            let Some(Dialog::Guide(guide)) = &app.dialog else { panic!("back to the date question") };
            assert_eq!(guide.questions[8].value, original); // Cancel never changes a date.
            assert!(app.job.is_none());
        }
    }
    #[test]
    fn calendar_manual_entry_preserves_exact_values_and_default_latest_opens_a_check() {
        let mut app = loaded_app("a=1\n");
        let mut guide = complete_guide(&app);
        guide.step = 8;
        guide.questions[8].value = "latest".into();
        app.dialog = Some(Dialog::Guide(guide.clone()));
        app.key(press(KeyCode::Enter));
        assert!(matches!(app.dialog, Some(Dialog::Calendar(..))));
        assert!(app.job.is_none());
        let form = calendar::Form::fixture(serde_json::json!({
            "cycle_hours":[0,6,12,18], "cycle_grid":{},
            "earliest":"2021-03-22T12", "latest_candidate":"2026-09-06T12",
        }), "2026-09-05T00");
        app.dialog = Some(Dialog::Calendar(guide, form));
        app.paste("2026-09-05T日本語".into());
        app.key(press(KeyCode::Enter));
        assert!(matches!(app.dialog, Some(Dialog::Calendar(..))));
        app.paste("2026-09-05T06:00:00Z".into());
        app.key(press(KeyCode::Enter));
        let Some(Dialog::Guide(guide)) = &app.dialog else { panic!("date applied") };
        assert_eq!(guide.questions[8].value, "2026-09-05T06");
        assert_eq!(guide.questions[7].value, "gfs");
        assert_eq!(guide.questions[9].value, "6");
        let mut guide = guide.clone();
        guide.questions[8].value = "latest".into();
        app.dialog = Some(Dialog::Summary(guide, 0));
        app.key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::CONTROL));
        assert!(matches!(&app.dialog, Some(Dialog::Calendar(guide, _)) if guide.summary_edit && guide.step == 8));
        assert!(app.job.is_none());
    }
    #[test]
    fn list_clicks_match_whole_visible_rows_and_oversized_values_remain_visible() {
        let mut app = loaded_app("a=1\n");
        app.dialog = Some(Dialog::Summary(complete_guide(&app), 1));
        let mut term = Terminal::new(TestBackend::new(40, 8)).unwrap();
        term.draw(|f| {
            clickable_list(
                f,
                &mut app.hits,
                Rect::new(0, 0, 40, 8),
                vec![
                    (vec![Line::raw("first"); 5], Hit::Summary(1)),
                    (vec![Line::raw("second"); 5], Hit::Summary(2)),
                ],
                Some(0),
            )
        })
        .unwrap();
        assert_eq!(app.hits.len(), 1);
        assert_eq!(term.backend().buffer()[(2, 6)].symbol(), " ");
        app.mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 2,
            row: 6,
            modifiers: KeyModifiers::NONE,
        });
        assert!(matches!(app.dialog, Some(Dialog::Summary(_, 1))));
        app.hits.clear();
        term.draw(|f| {
            clickable_list(
                f,
                &mut app.hits,
                Rect::new(0, 0, 40, 8),
                vec![(vec![Line::raw("long value"); 12], Hit::Summary(1))],
                Some(0),
            )
        })
        .unwrap();
        assert_eq!(app.hits.len(), 1);
        assert_eq!(term.backend().buffer()[(2, 0)].symbol(), "l");
        assert_eq!(term.backend().buffer()[(2, 7)].symbol(), ".");
        click_hit(&mut app, |h| matches!(h, Hit::Summary(1)));
        assert!(matches!(app.dialog, Some(Dialog::Guide(_))));
    }
    #[test]
    fn summary_next_remains_visible_and_clickable_at_every_terminal_size() {
        for (w, h) in [(65, 20), (80, 24), (120, 36)] {
            let mut app = loaded_app("a=1\n");
            let mut g = complete_guide(&app);
            g.questions[12].value = format!("{}forecast.toml", "long path/".repeat(15));
            app.dialog = Some(Dialog::Summary(g, 15));
            let screen = render_at(&mut app, w, h);
            assert!(screen.contains("Next: review command"), "{screen}");
            assert!(!app.hits.iter().any(|h| matches!(h.action, Hit::View(_))));
            capture_screen("summary", w, h, &mut app);
            click_hit(&mut app, |hit| {
                matches!(hit, Hit::Key(KeyCode::Char('n'), _))
            });
            assert!(matches!(app.dialog, Some(Dialog::Review(..))));
            assert!(app.job.is_none());
        }
    }
    #[test]
    fn mouse_edits_scrolled_setting_and_returns_without_losing_answers() {
        let mut app = loaded_app("a=1\n");
        let mut guide = complete_guide(&app);
        guide.questions.iter_mut().find(|q| q.flag == "@advanced").unwrap().value = "on".into();
        guide.sync_choices();
        let extra = guide.questions.iter().position(|q| q.flag == "@extra").unwrap();
        app.dialog = Some(Dialog::Summary(guide, extra + 1));
        render_at(&mut app, 65, 20);
        click_hit(&mut app, |hit| matches!(hit, Hit::Summary(index) if index == extra + 1));
        let Some(Dialog::Guide(g)) = &mut app.dialog else {
            panic!("setting editor");
        };
        assert_eq!(g.step, extra);
        g.questions[extra].value = "[\"--projection\",\"lambert\"]".into();
        render_at(&mut app, 65, 20);
        click_hit(&mut app, |hit| matches!(hit, Hit::Key(KeyCode::Enter, _)));
        let Some(Dialog::Summary(g, selected)) = &app.dialog else {
            panic!("settings retained");
        };
        assert_eq!(*selected, extra + 1);
        assert_eq!(g.questions[extra].value, "[\"--projection\",\"lambert\"]");
        assert_eq!(g.questions[7].value, "gfs");
        app.key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::CONTROL));
        assert!(matches!(app.dialog, Some(Dialog::Review(..))));
        app.key(press(KeyCode::Esc));
        assert!(matches!(app.dialog, Some(Dialog::Summary(..))));
        assert!(app.job.is_none());
    }
    #[test]
    fn workspace_shortcuts_mouse_tabs_and_geography_do_not_steal_editor_text() {
        let mut app = loaded_app("a=1\n");
        app.tab = Tab::Home;
        app.key(press(KeyCode::Char('F')));
        assert!(app.tab == Tab::Settings);
        app.key(press(KeyCode::Char('V')));
        assert!(app.editor.as_ref().unwrap().text().starts_with('V'));
        render_at(&mut app, 80, 24);
        click_hit(&mut app, |hit| matches!(hit, Hit::View(Tab::Overview)));
        assert!(app.tab == Tab::Overview);
        app.key(press(KeyCode::Char('G')));
        assert!(matches!(
            app.dialog,
            Some(Dialog::Path("Geography folder", _))
        ));
        app.key(press(KeyCode::Esc));
        app.key(press(KeyCode::Char('f')));
        assert!(app.tab == Tab::Settings);
        app.key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::CONTROL));
        assert!(matches!(
            app.dialog,
            Some(Dialog::Path("Geography folder", _))
        ));
        app.key(press(KeyCode::Esc));
        app.key(press(KeyCode::Esc));
        app.key(press(KeyCode::Char('V')));
        assert!(app.tab == Tab::Overview);
        assert!(app.job.is_none());
    }
    #[test]
    fn missing_configuration_explains_navigation_and_mouse_resize_is_safe() {
        let mut app = App::new().unwrap();
        render_at(&mut app, 80, 24);
        click_hit(&mut app, |hit| matches!(hit, Hit::View(Tab::Settings)));
        assert!(app.tab == Tab::Home);
        assert!(app.status.contains("Create or open"));
        render_at(&mut app, 80, 24);
        let area = app
            .hits
            .iter()
            .find(|h| matches!(h.action, Hit::Home(0)))
            .unwrap()
            .area;
        render_at(&mut app, 40, 10);
        app.mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: area.x,
            row: area.y,
            modifiers: KeyModifiers::NONE,
        });
        assert!(app.dialog.is_none());
        assert!(app.job.is_none());
    }
    #[test]
    fn editor_mouse_places_cursor_by_display_cells_without_splitting_graphemes() {
        assert_eq!(
            Editor::character_at_display_column(&"界a".chars().collect::<Vec<_>>(), 1),
            0
        );
        assert_eq!(
            Editor::character_at_display_column(&"界a".chars().collect::<Vec<_>>(), 2),
            1
        );
        assert_eq!(
            Editor::character_at_display_column(&"e\u{301}x".chars().collect::<Vec<_>>(), 1),
            2
        );
    }
    #[test]
    fn every_screen_and_small_terminal_render_without_running_a_command() {
        let mut app = App::new().unwrap();
        for (w, h) in [(64, 19), (65, 20), (80, 24), (120, 36)] {
            for tab in [Tab::Home, Tab::Overview, Tab::Settings, Tab::Logs] {
                app.tab = tab;
                let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
                term.draw(|f| draw(f, &mut app)).unwrap();
                assert!(app.job.is_none());
            }
        }
    }

    #[test]
    fn new_open_continue_are_actions_not_an_implicit_browser() {
        let mut app = App::new().unwrap();
        app.set_viewport(80, 24);
        assert!(app.tab == Tab::Home);
        app.key(press(KeyCode::Enter));
        assert!(matches!(app.dialog, Some(Dialog::Workflows(_))));
        app.key(press(KeyCode::Enter));
        assert!(matches!(app.dialog, Some(Dialog::Workflows(ref browser)) if browser.details));
        app.key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::CONTROL));
        assert!(matches!(app.dialog, Some(Dialog::Guide(_))));
        app.key(press(KeyCode::Esc));
        app.key(press(KeyCode::Char('o')));
        assert!(matches!(app.dialog, Some(Dialog::Choice(false, 0))));
        app.key(press(KeyCode::Enter));
        assert!(matches!(
            app.dialog,
            Some(Dialog::Path("Configuration path", _))
        ));
        app.key(press(KeyCode::Esc));
        app.key(press(KeyCode::Char('c')));
        assert!(matches!(app.dialog, Some(Dialog::Choice(true, 0))));
        assert!(app.job.is_none());
    }
    #[test]
    fn five_essentials_then_any_setting_is_editable_without_running() {
        let mut app = loaded_app("a=1\n");
        let mut guide = Guide::new(Kind::New, &app.cwd, &app.output);
        guide.questions[7].value = "gfs".into();
        app.dialog = Some(Dialog::Guide(guide));
        for (step, value) in [
            (1, "35.3,-97.5"),
            (7, "gfs"),
            (8, "2026-09-05T00"),
            (9, "6"),
            (12, "new-short-flow.toml"),
        ] {
            let Some(Dialog::Guide(g)) = &mut app.dialog else {
                panic!("Expected essential question");
            };
            assert_eq!(g.step, step);
            g.questions[step].value = value.into();
            app.key(press(KeyCode::Enter));
        }
        let Some(Dialog::Summary(g, _)) = app.dialog.take() else {
            panic!("Expected all-settings summary");
        };
        assert_eq!(g.questions.len(), 15);
        app.dialog = Some(Dialog::Summary(g, 4));
        app.key(press(KeyCode::Enter));
        let Some(Dialog::Guide(g)) = &mut app.dialog else {
            panic!("Expected selected setting editor");
        };
        assert_eq!(g.step, 3);
        g.questions[3].value = "48.125".into();
        app.key(press(KeyCode::Enter));
        let Some(Dialog::Summary(g, _)) = &app.dialog else {
            panic!("Expected return to summary");
        };
        assert_eq!(g.questions[3].value, "48.125");
        assert!(app.job.is_none());
    }

    #[test]
    fn native_guided_emission_opens_configuration_at_plan_next_step() {
        let Some(python) = env::var_os("GPUWM_TUI_TEST_PYTHON") else {
            return;
        };
        let mut app = loaded_app("a=1\n");
        app.python = python.into();
        let directory = app.output.parent().unwrap().to_path_buf();
        let mut guide = Guide::new(Kind::New, &directory, &app.output);
        guide.questions[0].value = "Native TUI proof".into();
        guide.questions[1].value = "35.3,-97.5".into();
        guide.questions[3].value = "12.125".into();
        guide.questions[6].value = "16".into();
        guide.questions[7].value = "gfs".into();
        guide.questions[8].value = "2026-09-05T00".into();
        let request = guide.request(&directory).unwrap();
        let created = request.created.clone().unwrap();
        app.dialog = Some(Dialog::Summary(guide, 12));
        render_at(&mut app, 80, 24);
        click_hit(&mut app, |hit| {
            matches!(hit, Hit::Key(KeyCode::Char('n'), _))
        });
        assert!(matches!(app.dialog, Some(Dialog::Review(..))));
        render_at(&mut app, 80, 24);
        click_hit(&mut app, |hit| matches!(hit, Hit::Key(KeyCode::Enter, _)));
        assert!(app.job.is_some(), "{}", app.status);
        let deadline = std::time::Instant::now() + Duration::from_secs(45);
        while app.busy() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
            app.poll();
        }
        assert_eq!(
            app.job.as_ref().unwrap().outcome,
            Some(0),
            "{}",
            app.job.as_ref().unwrap().log_tail(60)
        );
        assert_eq!(
            app.editor.as_ref().unwrap().path,
            created.canonicalize().unwrap()
        );
        assert!(app.tab == Tab::Overview);
        assert_eq!(app.selected, 1);
        let text = fs::read_to_string(&created).unwrap();
        let doc = text.parse::<toml_edit::DocumentMut>().unwrap();
        assert_eq!(
            doc["domain"][0]["dx"]
                .as_float()
                .unwrap_or_else(|| doc["domain"][0]["dx"].as_integer().unwrap() as f64),
            12125.0
        );
        assert!(app.status.contains("Next: Enter reviews the plan"));
        let original = fs::read(&created).unwrap();
        let mut fit = Guide::new(Kind::Fit, &directory, &app.output);
        fit.questions[0].value = created.to_string_lossy().into_owned();
        fit.questions[1].value = "35.3,-97.5".into();
        fit.questions[3].value = "2026-09-05T00".into();
        fit.questions[5].value = "16".into();
        let request = fit.request(&directory).unwrap();
        let fitted = request.created.clone().unwrap();
        app.start_command(&request.command, &request.args, request.created);
        let deadline = std::time::Instant::now() + Duration::from_secs(45);
        while app.busy() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
            app.poll();
        }
        assert_eq!(
            app.job.as_ref().unwrap().outcome,
            Some(0),
            "{}",
            app.job.as_ref().unwrap().log_tail(60)
        );
        assert_eq!(fs::read(&created).unwrap(), original);
        assert_eq!(
            app.editor.as_ref().unwrap().path,
            fitted.canonicalize().unwrap()
        );
        assert!(app.tab == Tab::Overview);
        assert_eq!(app.selected, 1);
        assert!(fitted.with_extension("fit.json").exists());
        assert!(app
            .job
            .as_ref()
            .unwrap()
            .log_tail(200)
            .contains("No forecast has started"));
    }
    #[test]
    fn log_control_sequences_are_displayed_as_text() {
        assert_eq!(safe("a\x1b[2J\0b\nc"), "a[2Jb\nc");
    }
    #[test]
    fn weather_modes_are_mouse_accessible_and_keep_launch_reviewed() {
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let mut app = loaded_app("# preserve the user's setup\na=1\n");
            let original = app.editor.as_ref().unwrap().text();
            for index in 0..workflows::MODES.len() {
                app.begin_workflows();
                if let Some(Dialog::Workflows(browser)) = &mut app.dialog {
                    browser.selected = index;
                }
                render_at(&mut app, width, height);
                click_hit(
                    &mut app,
                    |hit| matches!(hit, Hit::Workflow(value) if value == index),
                );
                assert!(matches!(&app.dialog, Some(Dialog::Workflows(browser)) if browser.details));
                render_at(&mut app, width, height);
                assert!(app
                    .hits
                    .iter()
                    .any(|hit| matches!(hit.action, Hit::Research(_))));
                assert!(app
                    .hits
                    .iter()
                    .any(|hit| matches!(hit.action, Hit::Key(KeyCode::Enter, _))));
                app.key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::CONTROL));
                let Some(Dialog::Guide(guide)) = &app.dialog else {
                    panic!("Mode must open editable questions");
                };
                assert_eq!(guide.workflow, Some(workflows::MODES[index].id));
                assert_eq!(guide.questions[4].value, workflows::MODES[index].chain);
                assert!(
                    guide.questions[10].value.is_empty(),
                    "a mode must not override source physics"
                );
                assert!(app.job.is_none());
                assert_eq!(app.editor.as_ref().unwrap().text(), original);
            }
        }
    }

    #[test]
    fn every_research_configuration_has_visible_actions_and_a_reviewable_native_route() {
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let mut app = loaded_app("# retained source\na=1\n");
            for (index, row) in research::rows("configurations").iter().enumerate() {
                let mut browser = workflows::Browser::default();
                browser.choose_config(index);
                app.dialog = Some(Dialog::Workflows(browser));
                let first = render_at(&mut app, width, height);
                assert!(
                    first.contains("Enter Set up")
                        || first.contains("Enter Downscale")
                        || first.contains("Enter Open scenario")
                );
                app.key(press(KeyCode::End));
                let last = render_at(&mut app, width, height);
                assert!(last.contains("Esc Back"));
                let last_source = research::strings(row, "citation_ids")
                    .last()
                    .copied()
                    .unwrap();
                let citation = research::rows("citations")
                    .iter()
                    .find(|v| v["id"] == last_source)
                    .unwrap();
                let ending = research::text(citation, "url")
                    .trim_end_matches('/')
                    .rsplit('/')
                    .next()
                    .unwrap();
                assert!(
                    last.replace('\n', "").replace(' ', "").contains(ending),
                    "{} end was clipped: {last}",
                    row["id"]
                );
                click_hit(&mut app, |hit| matches!(hit, Hit::Key(KeyCode::Enter, _)));
                if research::setup_route(row) == workflows::Route::Open {
                    assert!(matches!(
                        app.dialog,
                        Some(Dialog::Path("Configuration path", _))
                    ));
                    assert!(app.job.is_none());
                    continue;
                }
                let Some(Dialog::Guide(guide)) = &app.dialog else {
                    panic!("{} must open editable setup", row["id"]);
                };
                assert_eq!(guide.research, Some(research::text(row, "id")));
                assert_eq!(
                    guide.kind,
                    if row["method"] == "archived_downscale" {
                        Kind::Downscale
                    } else {
                        Kind::Research
                    }
                );
                assert!(guide.workflow.is_none());
                assert!(app.job.is_none());
                assert_eq!(
                    app.editor.as_ref().unwrap().text(),
                    "# retained source\na=1\n"
                );
            }
        }
    }

    #[test]
    fn mode_plot_review_cancel_preserves_existing_preferences_and_toml() {
        let mut app = loaded_app("a=1\n");
        let path = app.editor.as_ref().unwrap().path.clone();
        let selection = plotsettings::Selection::preset(0);
        plotsettings::save(&path, &selection, None).unwrap();
        let before = fs::read(plotsettings::sidecar(&path)).unwrap();
        fixture_plot_catalog(&mut app);
        app.workflow_route(6, workflows::Route::CurrentPlots);
        assert!(
            matches!(&app.dialog, Some(Dialog::Plots(form)) if form.mode == plotsettings::Mode::Review && form.selection.label == "Fire weather")
        );
        app.key(press(KeyCode::Esc));
        assert_eq!(fs::read(plotsettings::sidecar(&path)).unwrap(), before);
        assert_eq!(app.editor.as_ref().unwrap().text(), "a=1\n");
        assert!(app.job.is_none());
    }

    #[test]
    fn scenario_entry_and_cancel_preserve_file_and_editor_and_never_start_a_job() {
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let text = "# unmodified source\n[projection]\nref_lat=35.0\nref_lon=-97.0\n";
            let mut app = loaded_app(text);
            render_at(&mut app, width, height);
            app.key(press(KeyCode::Char('i')));
            assert!(matches!(app.dialog, Some(Dialog::Scenario(..))));
            assert!(render_at(&mut app, width, height).contains("warm bubbles"));
            app.key(press(KeyCode::Esc));
            assert!(app.dialog.is_none());
            assert_eq!(app.editor.as_ref().unwrap().text(), text);
            assert_eq!(
                fs::read_to_string(&app.editor.as_ref().unwrap().path).unwrap(),
                text
            );
            assert!(!app.dirty());
            assert!(app.job.is_none());
        }
    }

    fn loaded_app(text: &str) -> App {
        let dir = env::temp_dir().join(format!(
            "arwen-ui-controls-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&dir).unwrap();
        let path = dir.join("experiment.toml");
        fs::write(&path, text).unwrap();
        let mut app = App::new().unwrap();
        app.open(path);
        app.output = dir.join("jobs");
        app.python = dir.join("no-interpreter-unit-test.exe");
        app.set_viewport(120, 36);
        app
    }
    #[test]
    fn companion_timeline_receipt_and_corrupt_cache_recovery_are_typed_and_job_bound(){
        let mut app=loaded_app("a=1\n");
        let mut node=remote::Node::blank();node.host="fixture-node".into();node.last_job=Some("job-1".into());
        app.nodes.store.active=Some(node.id.clone());app.nodes.store.nodes.push(node.clone());
        let target=companion::Target::Ssh{node_id:node.id.clone(),connection_sha256:companion::digest(node.connection_key().as_bytes())};
        app.companion.session=Some(companion::Session::test_session(&app.output).unwrap());
        let session=app.companion.session.as_ref().unwrap().directory.clone();
        let render_summary=serde_json::json!({"schema":"gpuwm.render-summary.v1","rendered_png_count":25,"skipped_families":[{"name":"qpf_1h","reasons":["Previous frame is unavailable"]}]});
        let native_receipt:serde_json::Value=serde_json::from_str(include_str!("../tests/fixtures/native-progress-newcastle-2013.json")).unwrap();let progress=native_receipt["native_result"]["progress"].clone();
        app.nodes.view.status=Some(serde_json::json!({"id":"job-1","action":"start-plan","state":"completed","render_summary":render_summary,"phase":"preparing:prepare-case","phase_updated_unix_ms":123456,"progress":progress}));
        app.publish_companion_status(true);
        let status=companion::read_json(&session.join("status.json"),128*1024).unwrap();
        assert_eq!(status["job"]["render_summary"],render_summary);
        assert_eq!(status["job"]["progress"],progress);
        assert_eq!(status["job"]["phase"],"preparing:prepare-case");
        assert_eq!(status["job"]["phase_updated_unix_ms"],123456);
        let request=companion::Request{id:"timeline-1".into(),name:"artifact_index".into(),
            action:companion::Action::ArtifactIndex{job:"job-1".into(),domain:3,after_sequence:0},target:Some(target.clone()),
            plan_sha256:None,config_sha256:None,review_id:None,review_sha256:None};
        app.companion_remote=Some(CompanionRemoteRequest{request:request.clone(),node:node.clone(),source:serde_json::Value::Null});
        app.finish_companion_remote(&remote::Update::ArtifactIndexed(serde_json::json!({"artifact_index":{
            "schema":"gpuwm.remote-artifact-index.v1","job_id":"job-1","domain":3,"waiting":true,"entries":[],"next_after_sequence":null,"latest_sequence":null}})));
        let response=companion::read_json(&session.join("responses/timeline-1.json"),65536).unwrap();
        assert_eq!(response["ok"],true);
        let saved=PathBuf::from(response["artifact_index_path"].as_str().unwrap());
        let bytes=fs::read(&saved).unwrap();assert_eq!(response["artifact_index_sha256"],companion::digest(&bytes));
        let value:serde_json::Value=serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["schema"],"arwen.remote-artifact-index.v1");assert_eq!(value["target"],target.value());
        let mut request=request;request.id="recovery-1".into();request.name="sync_artifacts".into();
        request.action=companion::Action::SyncArtifacts{job:"job-1".into(),domain:3,sequence:Some(42),reader_leases:true};
        app.companion_remote=Some(CompanionRemoteRequest{request,node,source:serde_json::Value::Null});
        app.companion_artifacts=Some(serde_json::json!({"job_id":"job-1","artifact_manifest_path":"old"}));
        let recovery=serde_json::json!({"schema":"arwen.artifact-cache-recovery.v1","sha256":"a".repeat(64),"reason":"corrupt_retained_object"});
        app.finish_companion_remote(&remote::Update::ArtifactsSynced(serde_json::json!({"artifacts":{"job_id":"job-1","waiting":true,"frames":[]},"transferred_bytes":0,"cache_recovery":recovery})));
        let response=companion::read_json(&session.join("responses/recovery-1.json"),65536).unwrap();
        assert_eq!(response["cache_recovery"],recovery);assert_eq!(response["ok"],true);
        assert!(app.companion_artifacts.is_none()&&response["artifact_manifest_path"].is_null());
    }
    #[test]
    fn companion_open_configuration_preserves_remote_target_and_hardware_identity(){
        let mut app=loaded_app("a=1\n");
        let directory=app.editor.as_ref().unwrap().path.parent().unwrap().to_path_buf();
        app.nodes=remote::Controller::load(&directory);
        let mut node=remote::Node::blank();node.host="fixture-node".into();node.workspace="/node/work".into();
        app.nodes.store.active=Some(node.id.clone());app.nodes.store.nodes.push(node.clone());
        app.nodes.view.runtime=Some(serde_json::json!({"capabilities":{"stage_plan_v1":true,"review_plan_v1":true,"start_plan_v1":true},
            "probe":{"measured_unix_ms":1234,"devices":[{"index":0,"uuid":"GPU-fixture","memory_total_bytes":16000000000u64,"memory_free_bytes":15000000000u64}],
                "sizing":{"schema":"arwen.target-sizing.v1","total_bytes":16000000000u64,"free_bytes":15000000000u64,"profile":{"name":"measured fixture"}}}}));
        let next=directory.join("saved-map-candidate.toml");fs::write(&next,"a=2\n").unwrap();
        app.open(next.clone());
        assert_eq!(app.nodes.store.selected().unwrap().id,node.id);
        assert_eq!(app.editor.as_ref().unwrap().path,next.canonicalize().unwrap());
        let target=app.companion_target();assert_eq!(target["hardware"]["measured_unix_ms"],1234);
        assert_eq!(target["hardware"]["sizing"],app.nodes.view.runtime.as_ref().unwrap()["probe"]["sizing"]);
        assert_eq!(target["connection_sha256"],companion::digest(node.connection_key().as_bytes()));
        assert!(app.checked_companion_target(None).is_err());
        assert!(app.checked_companion_target(Some(&companion::Target::Ssh{node_id:node.id.clone(),connection_sha256:companion::digest(node.connection_key().as_bytes())})).unwrap().is_some());
        let selected=app.nodes.store.active.clone();app.open_nodes();assert_eq!(app.nodes.store.active,selected);
        assert!(app.nodes.pending.is_none()&&app.job.is_none());
    }
    #[test]
    fn companion_reset_setup_discards_draft_and_review_but_preserves_files_target_and_job_context(){
        let mut app=loaded_app("a=1\n");
        let saved=app.editor.as_ref().unwrap().path.clone();
        let original=fs::read(&saved).unwrap();
        app.nodes=remote::Controller::load(saved.parent().unwrap());
        let mut node=remote::Node::blank();node.host="fixture-node".into();node.workspace="/owned/work".into();
        node.last_job=Some("running-fixture".into());
        app.nodes.store.active=Some(node.id.clone());app.nodes.store.nodes.push(node);
        app.nodes.view.status=Some(serde_json::json!({"id":"running-fixture","state":"running"}));
        app.nodes.view.runtime=Some(serde_json::json!({"probe":{"measured_unix_ms":1234}}));
        let target=app.companion_target();
        app.active_config=Some(saved.clone());
        app.pending_config=Some(saved.clone());app.memory_recovery_config=Some(saved.clone());
        app.prepared=saved.with_file_name("prepared");app.geog_root=saved.with_file_name("geography");
        app.tab=Tab::Settings;app.key(press(KeyCode::Char('#')));assert!(app.dirty());
        app.guide_cache=Some(Guide::new(Kind::New,&app.cwd,&app.output));
        app.saved_guides.push(Guide::new(Kind::New,&app.cwd,&app.output));
        app.dialog=Some(Dialog::Help(HelpScroll::default()));
        app.dialog_stack.push(Dialog::Help(HelpScroll::default()));
        app.node_panel.screen=node_ui::Screen::Review{operation:remote::Operation::StartPlan{review:serde_json::json!({})},review:serde_json::json!({})};
        let session=companion::Session::test_session(&app.output).unwrap();
        let directory=session.directory.clone();let session_id=session.id.clone();
        app.companion.session=Some(session);
        fs::write(directory.join("requests/reset-fixture.json"),serde_json::to_vec(&serde_json::json!({
            "schema":"arwen.companion-request.v1","session_id":session_id,"id":"reset-fixture","action":"reset_setup"})).unwrap()).unwrap();
        app.poll_companion_requests();
        let response=companion::read_json(&directory.join("responses/reset-fixture.json"),8192).unwrap();
        assert_eq!(response["ok"],true);assert_eq!(response["action"],"reset_setup");
        let status=companion::read_json(&directory.join("status.json"),65536).unwrap();
        assert!(status["config_path"].is_null());assert_eq!(status["draft_dirty"],false);
        assert!(app.editor.is_none()&&app.pending_config.is_none()&&app.memory_recovery_config.is_none());
        assert!(app.prepared.as_os_str().is_empty());
        assert_eq!(app.geog_root,saved.with_file_name("geography"));
        assert!(app.guide_cache.is_none()&&app.saved_guides.is_empty()&&app.dialog.is_none()&&app.dialog_stack.is_empty());
        assert!(app.tui_map_review.is_none()&&app.tui_map_waiting.is_none()&&app.tui_map_launch.is_none());
        assert_eq!(app.active_config,Some(saved.clone()));assert_eq!(app.companion_target(),target);
        assert_eq!(app.nodes.view.status.as_ref().unwrap()["state"],"running");
        assert_eq!(fs::read(&saved).unwrap(),original);
        assert!(!matches!(app.node_panel.screen,node_ui::Screen::Review{..}));
        app.open(saved.clone());
        assert_eq!(app.editor.as_ref().unwrap().text(),"a=1\n");
        assert_eq!(app.editor.as_ref().unwrap().path,saved);
        assert_eq!(app.companion_context()["geog_root"],serde_json::json!(saved.with_file_name("geography")));
    }
    #[test]
    fn reset_setup_rejects_late_review_without_discarding_a_job_started_update(){
        let mut app=loaded_app("a=1\n");
        app.reset_setup();app.discard_setup_review=true;
        assert!(!app.accept_setup_review_update(&remote::Update::Preview{
            operation:remote::Operation::StartPlan{review:serde_json::json!({})},review:serde_json::json!({})}));
        assert!(app.editor.is_none());assert!(!app.discard_setup_review);
        app.discard_setup_review=true;
        assert!(app.accept_setup_review_update(&remote::Update::Started("running-fixture".into())));
        assert!(!app.discard_setup_review);
    }
    #[test]
    fn reset_setup_leaves_an_owned_running_local_job_alive(){
        let mut app=loaded_app("a=1\n");
        let python=PathBuf::from(env::var_os("GPUWM_TUI_TEST_PYTHON").expect("set test Python path"));
        let root=app.output.parent().unwrap().join("owned-reset-fixture");
        let package=root.join("gpuwm");fs::create_dir_all(&package).unwrap();
        fs::write(package.join("__init__.py"),"").unwrap();
        fs::write(package.join("tui_worker.py"),include_str!("../../../gpuwm/tui_worker.py")).unwrap();
        fs::write(package.join("cli.py"),"import time\ndef main(argv=None):\n print('benign reset fixture alive', flush=True)\n time.sleep(30)\n return 0\n").unwrap();
        let owned=Job::start_with_module_path(&python,"benign-reset-fixture",&[],&root.join("job"),&root,Some(&root)).unwrap();
        let directory=owned.dir.clone();app.job=Some(owned);
        app.active_config=Some(app.editor.as_ref().unwrap().path.clone());
        let active=app.active_config.clone();
        app.reset_setup();
        assert_eq!(app.job.as_ref().unwrap().dir,directory);assert_eq!(app.active_config,active);
        assert!(app.busy());assert!(app.job.as_mut().unwrap().poll().unwrap().is_none());
        // Cleanup terminates only this test's deliberately owned sleeping helper.
        app.job.as_mut().unwrap().stop().unwrap();
        let deadline=std::time::Instant::now()+Duration::from_secs(10);
        while app.job.as_mut().unwrap().poll().unwrap().is_none(){
            assert!(std::time::Instant::now()<deadline);std::thread::sleep(Duration::from_millis(20));
        }
    }
    #[test]
    fn companion_current_setup_plan_keeps_source_target_and_products_without_remote_paths(){
        let mut app=loaded_app("a=1\n");
        let directory=app.editor.as_ref().unwrap().path.parent().unwrap().to_path_buf();
        app.nodes=remote::Controller::load(&directory);
        let mut node=remote::Node::blank();node.host="fixture-node".into();node.workspace="/owned/work".into();
        app.nodes.store.nodes.push(node.clone());app.nodes.store.active=Some(node.id.clone());
        assert!(app.current_setup_uses_staged_node());
        let available=app.available_companion_targets();assert_eq!(available.len(),2);assert_eq!(available[1]["node_id"],node.id);
        let request=app.current_setup_review_request().unwrap();
        let companion::Action::ReviewPlan(path)=&request.action else{panic!("staged review");};
        let plan=companion::read_json(path,65536).unwrap();
        assert_eq!(plan["route"],"prepared");assert_eq!(plan["config"]["path"],serde_json::json!(app.editor.as_ref().unwrap().path));
        assert_eq!(plan["run_options"]["render_products"],app.plot_spec().unwrap());
        assert!(app.checked_companion_plan(&request,path).is_ok());assert!(app.nodes.pending.is_none()&&app.job.is_none());
        let wrong=companion::Target::Ssh{node_id:node.id.clone(),connection_sha256:"0".repeat(64)};
        assert!(app.select_companion_target(&wrong).is_err());assert_eq!(app.nodes.store.active,Some(node.id));
        app.nodes.store.nodes[0].config="/explicit/advanced.toml".into();assert!(!app.current_setup_uses_staged_node());
    }
    #[test]
    fn companion_explicit_nodes_and_focus_setup_preserve_the_visible_draft(){
        let mut app=loaded_app("a=1\n");
        let directory=app.editor.as_ref().unwrap().path.parent().unwrap().to_path_buf();
        let path=directory.join("explicit-node-profiles.json");
        let mut controller=remote::Controller::load_path(path.clone());
        let mut node=remote::Node::blank();node.host="fixture-node".into();node.workspace="/owned/node/work".into();
        controller.store.active=Some(node.id.clone());controller.store.nodes.push(node.clone());controller.store.save(&path).unwrap();
        let original=fs::read(&path).unwrap();
        assert!(app.use_nodes_file(PathBuf::from("relative-profiles.json")).is_err());
        app.use_nodes_file(path.clone()).unwrap();app.prepare_startup_node().unwrap();
        assert_eq!(app.nodes.path,path);assert_eq!(app.nodes.store.active,Some(node.id.clone()));
        assert!(matches!(app.dialog,Some(Dialog::Nodes)));assert!(app.nodes.pending.is_none()&&app.job.is_none());
        app.editor.as_mut().unwrap().insert("# unsaved\n");let draft=app.editor.as_ref().unwrap().text();
        let session=companion::Session::test_session(&directory.join("focus-control")).unwrap();
        fs::write(session.directory.join("requests/focus-setup.json"),serde_json::to_vec(&serde_json::json!({
            "schema":"arwen.companion-request.v1","session_id":session.id,"id":"focus-setup","action":"focus_setup"})).unwrap()).unwrap();
        app.companion.session=Some(session);app.poll_companion_requests();
        assert_eq!(app.tab,Tab::Settings);assert!(app.dialog.is_none());assert!(app.dirty());
        assert_eq!(app.editor.as_ref().unwrap().text(),draft);assert_eq!(app.nodes.store.active,Some(node.id));
        assert_eq!(fs::read(path).unwrap(),original);assert!(app.nodes.pending.is_none()&&app.job.is_none());
    }
    #[test]
    fn companion_launch_available_stays_true_across_passive_poll_activity_cycles(){
        let mut node=remote::Node::blank();node.last_job=Some("job-1".into());
        let job=serde_json::json!({"id":"job-1","state":"failed"});
        let operations=[remote::Operation::Probe,remote::Operation::Status{job:"job-1".into()},
            remote::Operation::Logs{job:"job-1".into(),cursor:0},
            remote::Operation::ArtifactIndex{job:"job-1".into(),domain:1,after_sequence:0},
            remote::Operation::SyncArtifacts{job:"job-1".into(),domain:1,cache:PathBuf::new(),sequence:None,reader_leases:false}];
        for operation in &operations{
            for pending in [Some(operation),None,Some(operation),None]{
                assert_eq!(companion_activity_state(false,pending.is_some()),if pending.is_some(){"running"}else{"ready"});
                assert!(companion_launch_available(false,false,Some(&node),Some(&job),None,pending));
            }
        }
        for action in [companion::Action::SelectTarget,companion::Action::ArtifactIndex{job:"job-1".into(),domain:1,after_sequence:0},
            companion::Action::SyncArtifacts{job:"job-1".into(),domain:1,sequence:None,reader_leases:false}]{
            assert!(companion_launch_available(false,false,Some(&node),Some(&job),Some(&action),Some(&operations[0])));
        }
    }
    #[test]
    fn companion_launch_available_refuses_foreground_requests_and_preserves_local_ready(){
        let node=remote::Node::blank();
        let operations=[remote::Operation::List,
            remote::Operation::ReviewPlan{plan:PathBuf::new(),plan_sha256:"a".repeat(64),config_sha256:"b".repeat(64),output:"/tmp/review".into()},
            remote::Operation::StartPlan{review:serde_json::Value::Null},remote::Operation::Stop{job:"job-1".into()},
            remote::Operation::Start{products:"all".into(),preview:true,binding:None},
            remote::Operation::Resume{job:"job-1".into(),checkpoint:"/tmp/checkpoint".into(),output:"/tmp/output".into(),preview:false,binding:None},
            remote::Operation::SyncProcessedFrame{job:"job-1".into(),domain:1,cache:PathBuf::new(),sequence:None}];
        for operation in &operations{
            assert!(!companion_launch_available(false,false,Some(&node),None,None,Some(operation)));
        }
        for action in [companion::Action::ReviewPlan(PathBuf::new()),companion::Action::LaunchPlan(PathBuf::new()),
            companion::Action::StopJob("job-1".into()),companion::Action::SyncProcessedFrame{job:"job-1".into(),domain:1,sequence:None}]{
            assert!(!companion_launch_available(false,false,Some(&node),None,Some(&action),None));
        }
        assert!(companion_launch_available(false,false,Some(&node),None,None,None));
        assert!(!companion_launch_available(true,false,Some(&node),None,None,None));
        assert!(!companion_launch_available(false,true,Some(&node),None,None,None));
        assert!(companion_launch_available(false,false,None,None,None,None));
        assert!(!companion_launch_available(true,false,None,None,None,None));
        assert!(!companion_launch_available(false,false,None,None,None,Some(&remote::Operation::Probe)));
    }
    #[test]
    fn compact_viewer_queue_does_not_disable_an_otherwise_ready_run_button(){
        let node=remote::Node::blank();
        let options=remote::ViewerOptions::from_value(&serde_json::json!({"prefetch_sequences":[2,3]})).unwrap();
        let action=companion::Action::SyncProcessedFrameV2{job:"job-1".into(),domain:1,sequence:Some(1),options:options.clone(),reader_leases:true,cache_bytes:None};
        let operation=remote::Operation::SyncProcessedFrameV2{job:"job-1".into(),domain:1,sequence:Some(1),options,reader_leases:true,cache_bytes:None,cache:PathBuf::new()};
        assert!(passive_companion_action(&action));assert!(passive_node_operation(&operation));
        assert!(companion_launch_available(false,false,Some(&node),None,Some(&action),Some(&operation)));
        assert!(!companion_launch_available(false,true,Some(&node),None,Some(&action),Some(&operation)));
    }
    #[test]
    fn companion_launch_available_requires_matching_terminal_remote_job(){
        let mut node=remote::Node::blank();node.last_job=Some("job-1".into());
        for state in ["starting","running","stopping","ownership_mismatch","lost","unknown",""]{
            let job=serde_json::json!({"id":"job-1","state":state});
            assert!(!companion_launch_available(false,false,Some(&node),Some(&job),None,None),"{state}");
        }
        assert!(!companion_launch_available(false,false,Some(&node),None,None,None));
        for state in ["stopped","interrupted","completed","failed","cancelled"]{
            let mut job=serde_json::json!({"id":"job-1","state":state});
            assert!(companion_launch_available(false,false,Some(&node),Some(&job),None,None),"{state}");
            job["id"]=serde_json::json!("old-job");
            assert!(!companion_launch_available(false,false,Some(&node),Some(&job),None,None));
        }
        node.last_job=None;
        let unrecorded=serde_json::json!({"id":"job-1","state":"running"});
        assert!(!companion_launch_available(false,false,Some(&node),Some(&unrecorded),None,None));
    }
    #[test]
    fn companion_launch_available_status_keeps_queue_and_tui_review_exclusive(){
        let(mut app,request,control)=companion_queue_fixture();
        app.nodes.store.nodes[0].last_job=Some("job-1".into());
        app.nodes.view.status=Some(serde_json::json!({"id":"job-1","state":"failed"}));
        app.publish_companion_status(true);
        let before=companion::read_json(&control.join("status.json"),65536).unwrap();
        assert_eq!(before["state"],"ready");assert_eq!(before["launch_available"],true);
        app.begin_companion_remote(request.clone()).unwrap();
        app.publish_companion_status(true);
        let queued=companion::read_json(&control.join("status.json"),65536).unwrap();
        assert_eq!(queued["state"],"ready");assert_eq!(queued["launch_available"],false);
        assert_eq!(queued["queued_node_action"]["state"],"waiting");
        assert!(!control.join("responses/queued-review.json").exists());
        assert!(app.begin_companion_remote(request.clone()).unwrap_err().contains("already waiting"));
        app.companion_waiting=None;app.companion_remote=None;app.tui_map_waiting=Some(request);
        app.publish_companion_status(true);
        assert_eq!(companion::read_json(&control.join("status.json"),65536).unwrap()["launch_available"],false);
        assert!(app.nodes.pending.is_none()&&app.job.is_none());
    }
    #[test]
    fn companion_queue_only_yields_read_only_node_operations(){
        let readonly=[remote::Operation::Probe,remote::Operation::Status{job:"job-1".into()},
            remote::Operation::Logs{job:"job-1".into(),cursor:0},
            remote::Operation::ArtifactIndex{job:"job-1".into(),domain:1,after_sequence:0},
            remote::Operation::SyncArtifacts{job:"job-1".into(),domain:1,cache:PathBuf::new(),sequence:None,reader_leases:false}];
        assert!(readonly.iter().all(passive_node_operation));
        let exclusive=[remote::Operation::ReviewPlan{plan:PathBuf::new(),plan_sha256:"a".repeat(64),config_sha256:"b".repeat(64),output:"/tmp/review".into()},
            remote::Operation::StartPlan{review:serde_json::Value::Null},remote::Operation::Stop{job:"job-1".into()},
            remote::Operation::SyncProcessedFrame{job:"job-1".into(),domain:1,cache:PathBuf::new(),sequence:None}];
        assert!(exclusive.iter().all(|operation|!passive_node_operation(operation)));
    }
    fn companion_queue_fixture()->(App,companion::Request,PathBuf){
        let mut app=loaded_app("a=1\n");let config=app.editor.as_ref().unwrap().path.clone();let directory=config.parent().unwrap();
        app.nodes=remote::Controller::load(directory);
        let mut node=remote::Node::blank();node.host="fixture-node".into();node.workspace="/node/work".into();
        app.nodes.store.active=Some(node.id.clone());app.nodes.store.nodes.push(node.clone());
        let session=companion::Session::test_session(&directory.join("control")).unwrap();
        let control=session.directory.clone();app.companion.session=Some(session);
        let plan=directory.join("queued-plan.json");fs::write(&plan,serde_json::to_vec(&serde_json::json!({"schema":"gpuwm.run-plan.v1","config":{"path":config}})).unwrap()).unwrap();
        let request=companion::Request{id:"queued-review".into(),name:"review_plan".into(),action:companion::Action::ReviewPlan(plan.clone()),
            target:Some(companion::Target::Ssh{node_id:node.id.clone(),connection_sha256:companion::digest(node.connection_key().as_bytes())}),
            plan_sha256:Some(companion::digest(&fs::read(plan).unwrap())),config_sha256:Some(companion::digest(&fs::read(config).unwrap())),review_id:None,review_sha256:None};
        let mut background=request.clone();background.id="background-artifacts".into();background.name="sync_artifacts".into();
        background.action=companion::Action::SyncArtifacts{job:"job-1".into(),domain:1,sequence:None,reader_leases:false};
        app.companion_remote=Some(CompanionRemoteRequest{request:background,node,source:serde_json::Value::Null});
        (app,request,control)
    }
    #[test]
    fn companion_queue_retains_one_interactive_request_without_false_ack(){
        let(mut app,request,control)=companion_queue_fixture();
        app.begin_companion_remote(request.clone()).unwrap();
        assert_eq!(app.companion_waiting.as_ref().unwrap().request.id,request.id);
        assert_eq!(app.companion_remote.as_ref().unwrap().request.id,"background-artifacts");
        assert!(app.status.contains("Review will continue automatically"));
        assert!(!control.join("responses/queued-review.json").exists());
        app.continue_queued_companion_remote();assert!(app.companion_waiting.is_some());
        let mut second=request.clone();second.id="second".into();
        assert!(app.begin_companion_remote(second).unwrap_err().contains("already waiting"));
        app.node_request(remote::Operation::Logs{job:"job-1".into(),cursor:0});
        assert!(app.nodes.pending.is_none()&&app.job.is_none());
        assert_eq!(app.companion_waiting.as_ref().unwrap().request.id,request.id);
        app.publish_companion_status(true);
        assert_eq!(companion::read_json(&control.join("status.json"),65536).unwrap()["queued_node_action"]["state"],"waiting");
    }
    #[test]
    fn companion_queue_releases_after_read_and_rechecks_capability_without_dispatch(){
        let(mut app,request,control)=companion_queue_fixture();app.begin_companion_remote(request).unwrap();
        app.companion_remote=None;app.continue_queued_companion_remote();
        // The queue is consumed, then the original remote-review validation
        // runs. No runtime capability was granted by this CPU-only fixture.
        assert!(app.companion_waiting.is_none()&&app.companion_remote.is_none());
        let response=companion::read_json(&control.join("responses/queued-review.json"),65536).unwrap();
        assert_eq!(response["ok"],false);assert!(response["message"].as_str().unwrap().contains("Connect to the selected node"));
        assert!(app.nodes.pending.is_none()&&app.job.is_none());
    }
    #[test]
    fn companion_queue_expires_while_read_is_pending_and_never_dispatches_late(){
        let(mut app,request,control)=companion_queue_fixture();app.begin_companion_remote(request).unwrap();
        app.companion_waiting.as_mut().unwrap().queued_at=Instant::now()-COMPANION_QUEUE_TIMEOUT;
        app.continue_queued_companion_remote();
        assert!(app.companion_waiting.is_none()&&app.companion_remote.is_some());
        let response=companion::read_json(&control.join("responses/queued-review.json"),65536).unwrap();
        assert_eq!(response["ok"],false);assert!(response["message"].as_str().unwrap().contains("expired"));
        app.companion_remote=None;app.continue_queued_companion_remote();
        assert!(app.nodes.pending.is_none()&&app.job.is_none());
    }
    #[test]
    fn companion_queue_cancels_changed_session_target_and_saved_inputs(){
        for changed in 0..4{
            let(mut app,request,control)=companion_queue_fixture();app.begin_companion_remote(request.clone()).unwrap();
            match changed{
                0=>{app.companion.session.as_mut().unwrap().id="replacement-session".into();},
                1=>{app.nodes.store.nodes[0].host="changed-host".into();},
                2=>{fs::write(&app.editor.as_ref().unwrap().path,"a=2\n").unwrap();},
                _=>{let companion::Action::ReviewPlan(path)=&request.action else{unreachable!()};fs::write(path,"{}").unwrap();},
            }
            app.continue_queued_companion_remote();assert!(app.companion_waiting.is_none());
            assert!(app.nodes.pending.is_none()&&app.job.is_none());
            if changed==0{assert!(!control.join("responses/queued-review.json").exists());}
            else{assert_eq!(companion::read_json(&control.join("responses/queued-review.json"),65536).unwrap()["ok"],false);}
        }
    }
    #[test]
    fn companion_queue_never_queues_behind_mutation_or_processed_frame_work(){
        for action in [companion::Action::LaunchPlan(PathBuf::from("plan.json")),companion::Action::SyncProcessedFrame{job:"job-1".into(),domain:1,sequence:None}]{
            let(mut app,request,_)=companion_queue_fixture();app.companion_remote.as_mut().unwrap().request.action=action;
            assert!(app.begin_companion_remote(request).unwrap_err().contains("node request is in progress"));
            assert!(app.companion_waiting.is_none()&&app.nodes.pending.is_none()&&app.job.is_none());
        }
    }
    #[test]
    fn companion_queue_launch_revalidates_the_completed_review_before_dispatch(){
        let(mut app,mut request,control)=companion_queue_fixture();
        let(path,hash)=app.companion.session.as_ref().unwrap().save_review("completed-review",&serde_json::json!({"fixture":"review hash is rechecked before remote execution"})).unwrap();
        let companion::Action::ReviewPlan(plan)=request.action else{unreachable!()};
        request.action=companion::Action::LaunchPlan(plan);request.name="launch_plan".into();
        request.review_id=Some("completed-review".into());request.review_sha256=Some(hash);
        app.begin_companion_remote(request).unwrap();
        assert!(!control.join("responses/queued-review.json").exists());
        fs::write(path,"{}").unwrap();app.companion_remote=None;app.continue_queued_companion_remote();
        let response=companion::read_json(&control.join("responses/queued-review.json"),65536).unwrap();
        assert_eq!(response["ok"],false);assert!(response["message"].as_str().unwrap().contains("completed node review changed"));
        assert!(app.companion_waiting.is_none()&&app.nodes.pending.is_none()&&app.job.is_none());
    }
    #[test]
    fn companion_queue_stop_revalidates_the_recorded_job_before_dispatch(){
        let(mut app,mut request,control)=companion_queue_fixture();
        request.action=companion::Action::StopJob("old-job".into());request.name="stop_job".into();
        app.begin_companion_remote(request).unwrap();
        app.nodes.store.nodes[0].last_job=Some("new-job".into());app.nodes.view.status=Some(serde_json::json!({"id":"new-job","state":"running"}));
        app.companion_remote=None;app.continue_queued_companion_remote();
        let response=companion::read_json(&control.join("responses/queued-review.json"),65536).unwrap();
        assert_eq!(response["ok"],false);assert!(response["message"].as_str().unwrap().contains("not the selected node's current recorded job"));
        assert!(app.companion_waiting.is_none()&&app.nodes.pending.is_none()&&app.job.is_none());
    }
    fn companion_sizing_refresh_fixture()->(App,companion::Request,PathBuf){
        let(mut app,mut request,control)=companion_queue_fixture();
        request.id="sizing-refresh".into();request.name="select_target".into();request.action=companion::Action::SelectTarget;
        request.plan_sha256=None;request.config_sha256=None;
        app.nodes.store.nodes[0].last_job=Some("running-user-forecast".into());
        app.nodes.store.save(&app.nodes.path).unwrap();
        app.nodes.view.status=Some(serde_json::json!({"id":"running-user-forecast","state":"running","model_elapsed_seconds":120.0}));
        app.nodes.view.log="Retained forecast progress\n".into();app.nodes.view.cursor=27;
        app.nodes.view.runtime=Some(serde_json::json!({"probe":{"sizing":{"measured_unix_ms":1}}}));
        (app,request,control)
    }
    #[test]
    fn companion_sizing_refresh_protocol_queues_without_clearing_the_active_job(){
        let(mut app,request,control)=companion_sizing_refresh_fixture();
        let before=fs::read(&app.nodes.path).unwrap();let job=app.nodes.view.status.clone();let runtime=app.nodes.view.runtime.clone();
        fs::write(control.join("requests/sizing-refresh.json"),serde_json::to_vec(&serde_json::json!({
            "schema":"arwen.companion-request.v1","session_id":app.companion.session.as_ref().unwrap().id,
            "id":request.id,"action":"select_target","target":request.target.as_ref().unwrap().value()})).unwrap()).unwrap();
        app.poll_companion_requests();
        assert_eq!(app.companion_waiting.as_ref().unwrap().request.id,request.id);
        assert!(app.status.contains("Hardware refresh will continue automatically"));
        assert!(!control.join("responses/sizing-refresh.json").exists());
        assert_eq!(app.nodes.view.status,job);assert_eq!(app.nodes.view.runtime,runtime);
        assert_eq!(app.nodes.view.log,"Retained forecast progress\n");assert_eq!(app.nodes.view.cursor,27);
        assert_eq!(fs::read(&app.nodes.path).unwrap(),before);
        app.node_request(remote::Operation::Logs{job:"running-user-forecast".into(),cursor:27});
        assert!(app.nodes.pending.is_none()&&app.job.is_none());
        app.publish_companion_status(true);
        let status=companion::read_json(&control.join("status.json"),65536).unwrap();
        assert_eq!(status["queued_node_action"]["action"],"select_target");
        assert_eq!(status["job"]["job_id"],"running-user-forecast");assert_eq!(status["job"]["state"],"running");
    }
    #[test]
    fn companion_sizing_refresh_expiry_session_and_target_binding_prevent_late_probe(){
        for changed in 0..3{
            let(mut app,request,control)=companion_sizing_refresh_fixture();app.begin_companion_remote(request).unwrap();
            match changed{
                0=>app.companion_waiting.as_mut().unwrap().queued_at=Instant::now()-COMPANION_QUEUE_TIMEOUT,
                1=>app.companion.session.as_mut().unwrap().id="replacement-session".into(),
                _=>app.nodes.store.nodes[0].host="changed-host".into(),
            }
            app.continue_queued_companion_remote();
            assert!(app.companion_waiting.is_none()&&app.companion_remote.is_some());
            if changed==1{assert!(!control.join("responses/sizing-refresh.json").exists());}
            else{assert_eq!(companion::read_json(&control.join("responses/sizing-refresh.json"),65536).unwrap()["ok"],false);}
            app.companion_remote=None;app.continue_queued_companion_remote();
            assert!(app.nodes.pending.is_none()&&app.job.is_none());
            assert_eq!(app.nodes.store.nodes[0].last_job.as_deref(),Some("running-user-forecast"));
            assert_eq!(app.nodes.view.status.as_ref().unwrap()["state"],"running");
        }
    }
    #[test]
    fn companion_sizing_refresh_dispatches_only_probe_and_preserves_job_on_transport_error(){
        let(mut app,request,control)=companion_sizing_refresh_fixture();
        app.python=control.join("missing-python-for-cpu-protocol-test.exe");
        app.output=control.join("owned-node-requests");app.cwd=control.clone();
        let job=app.nodes.view.status.clone();let runtime=app.nodes.view.runtime.clone();
        let before=fs::read(&app.nodes.path).unwrap();
        app.begin_companion_remote(request).unwrap();app.companion_remote=None;
        app.continue_queued_companion_remote();
        // The ordinary launch boundary records its exact argv before the
        // deliberately absent interpreter refuses. No process or SSH starts.
        let attempts=fs::read_dir(app.output.join(".arwen-tui")).unwrap().collect::<Result<Vec<_>,_>>().unwrap();
        assert_eq!(attempts.len(),1);
        let receipt=companion::read_json(&attempts[0].path().join("job.json"),65536).unwrap();
        let argv=receipt["command"].as_array().unwrap();
        assert_eq!(argv[3],"remote");assert_eq!(argv[4],"probe");
        assert!(!argv.iter().any(|arg|arg=="start"||arg=="start-plan"||arg=="stop"));
        let response=companion::read_json(&control.join("responses/sizing-refresh.json"),65536).unwrap();
        assert_eq!(response["ok"],false);
        assert!(app.nodes.pending.is_none()&&app.companion_waiting.is_none()&&app.companion_remote.is_none()&&app.job.is_none());
        assert_eq!(app.nodes.view.status,job);assert_eq!(app.nodes.view.runtime,runtime);assert_eq!(fs::read(&app.nodes.path).unwrap(),before);
        assert_eq!(app.nodes.view.log,"Retained forecast progress\n");assert_eq!(app.nodes.view.cursor,27);
    }
    #[test]
    fn companion_sizing_refresh_completes_only_on_matching_probe_result(){
        for success in [false,true]{
            let(mut app,request,control)=companion_sizing_refresh_fixture();
            let node=app.nodes.store.selected().unwrap().clone();let job=app.nodes.view.status.clone();
            let target=request.target.as_ref().unwrap().value();
            app.companion_remote=Some(CompanionRemoteRequest{request,node,source:serde_json::Value::Null});
            if success{app.finish_companion_remote(&remote::Update::Connected);}
            else{app.finish_companion_remote(&remote::Update::Failed("fixture probe failure".into()));}
            let response=companion::read_json(&control.join("responses/sizing-refresh.json"),65536).unwrap();
            assert_eq!(response["ok"],success);assert_eq!(response["target"],target);
            assert!(app.companion_remote.is_none()&&app.nodes.pending.is_none()&&app.job.is_none());
            assert_eq!(app.nodes.view.status,job);assert_eq!(app.nodes.store.nodes[0].last_job.as_deref(),Some("running-user-forecast"));
        }
    }
    #[test]
    fn companion_sizing_refresh_never_queues_target_switch_or_overlaps_mutation(){
        let(mut app,mut request,_)=companion_sizing_refresh_fixture();
        let node=app.nodes.store.selected().unwrap().clone();
        request.target=Some(companion::Target::Ssh{node_id:node.id.clone(),connection_sha256:"0".repeat(64)});
        assert!(app.begin_companion_remote(request).is_err());assert!(app.companion_waiting.is_none());
        for action in [companion::Action::LaunchPlan(PathBuf::from("plan.json")),companion::Action::SyncProcessedFrame{job:"running-user-forecast".into(),domain:1,sequence:None}]{
            let(mut app,request,_)=companion_sizing_refresh_fixture();
            app.companion_remote.as_mut().unwrap().request.action=action;
            assert!(app.begin_companion_remote(request).unwrap_err().contains("node request is in progress"));
            assert!(app.companion_waiting.is_none()&&app.nodes.pending.is_none()&&app.job.is_none());
        }
    }
    #[test]
    fn companion_remote_review_is_source_bound_and_does_not_publish_a_job(){
        let mut app=loaded_app("a=1\n");
        let config=app.editor.as_ref().unwrap().path.clone();let directory=config.parent().unwrap().to_path_buf();
        app.nodes=remote::Controller::load(&directory);
        let mut node=remote::Node::blank();node.host="fixture-node".into();node.workspace="/node/work".into();
        app.nodes.store.active=Some(node.id.clone());app.nodes.store.nodes.push(node.clone());
        let session=companion::Session::test_session(&directory.join("control")).unwrap();
        let control=session.directory.clone();app.companion.session=Some(session);
        let path=directory.join("map-plan.json");fs::write(&path,serde_json::to_vec(&serde_json::json!({"schema":"gpuwm.run-plan.v1","config":{"path":config}})).unwrap()).unwrap();
        let request=companion::Request{id:"review-fixture".into(),name:"review_plan".into(),action:companion::Action::ReviewPlan(path.clone()),
            target:Some(companion::Target::Ssh{node_id:node.id.clone(),connection_sha256:companion::digest(node.connection_key().as_bytes())}),
            plan_sha256:Some(companion::digest(&fs::read(&path).unwrap())),config_sha256:Some(companion::digest(&fs::read(&config).unwrap())),review_id:None,review_sha256:None};
        let source=app.checked_companion_plan(&request,&path).unwrap();
        app.companion_remote=Some(CompanionRemoteRequest{request:request.clone(),node:node.clone(),source:source.clone()});
        let wps=directory.join("selected.namelist.wps");fs::write(&wps,"&share\n max_dom = 1,\n/\n").unwrap();
        let mut inputs=serde_json::Map::new();inputs.insert(config.to_string_lossy().into_owned(),serde_json::json!(companion::digest(&fs::read(&config).unwrap())));
        inputs.insert(path.to_string_lossy().into_owned(),serde_json::json!(companion::digest(&fs::read(&path).unwrap())));
        inputs.insert(wps.to_string_lossy().into_owned(),serde_json::json!(companion::digest(&fs::read(&wps).unwrap())));
        app.finish_companion_remote(&remote::Update::PlanReviewed(serde_json::json!({"memory":{"measured":true,"refuse":false},"source_inputs":inputs})));
        let response=companion::read_json(&control.join("responses/review-fixture.json"),65536).unwrap();
        assert_eq!(response["ok"],true);assert!(response["job_id"].is_null());assert!(response["job_dir"].is_null());
        let review_path=PathBuf::from(response["review_path"].as_str().unwrap());
        assert_eq!(response["review_sha256"],companion::digest(&fs::read(&review_path).unwrap()));
        assert_eq!(companion::read_json(&review_path,65536).unwrap()["source"],source);
        let mut launch=request.clone();launch.id="launch-fixture".into();launch.name="launch_plan".into();
        launch.action=companion::Action::LaunchPlan(path.clone());launch.review_id=Some(request.id.clone());
        launch.review_sha256=Some(response["review_sha256"].as_str().unwrap().into());
        fs::write(&wps,"&share\n max_dom = 2,\n/\n").unwrap();
        let error=app.begin_companion_remote(launch).unwrap_err();
        assert!(error.contains("selected.namelist.wps")&&error.contains("changed"),"{error}");
        fs::write(&config,"a=3\n").unwrap();assert!(app.checked_companion_plan(&request,&path).is_err());
        assert!(app.nodes.pending.is_none()&&app.job.is_none());
    }
    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn ordinary_open_lists_catalogs_and_keeps_the_loaded_configuration() {
        let mut app = loaded_app("a=1\n");
        let original = app.editor.as_ref().unwrap().path.clone();
        let folder = original.parent().unwrap().to_path_buf();
        let catalogs = [
            ("historical.ZIP", "archive bytes are read by the catalog backend"),
            ("historical.json", "{}"),
            ("historical.toml", "schema = 'arwen.case-catalog.v1'\n"),
        ];
        for (name, contents) in catalogs {
            fs::write(folder.join(name), contents).unwrap();
        }
        fs::write(folder.join("not-a-configuration.txt"), "unrelated").unwrap();
        for (name, _) in catalogs {
            app.dialog = None;
            app.tab = Tab::Home;
            app.key(press(KeyCode::Char('o')));
            app.key(press(KeyCode::Enter));
            assert!(matches!(&app.dialog, Some(Dialog::Path("Configuration path", _))));
            app.paste(display_path(&folder));
            app.key(press(KeyCode::F(2)));
            let Some(Dialog::Browser(_, items, selected)) = &mut app.dialog else {
                panic!("Open must show its file browser");
            };
            assert!(!items.iter().any(|p| p.ends_with("not-a-configuration.txt")));
            *selected = items.iter().position(|p| p.ends_with(name)).unwrap();
            app.key(press(KeyCode::Enter));
            assert!(matches!(app.dialog, Some(Dialog::Cases(_))), "{name}");
            assert_eq!(app.editor.as_ref().unwrap().path, original);
            assert_eq!(app.editor.as_ref().unwrap().text(), "a=1\n");
            assert!(app.job.is_none());
        }
    }

    #[test]
    fn dropped_file_opens_without_running_and_unsaved_drafts_are_preserved() {
        let mut app = loaded_app("a=1\n");
        let original = app.editor.as_ref().unwrap().path.clone();
        let other = original.with_file_name("storm space-界.toml");
        fs::write(&other, "b=2\n").unwrap();
        app.tab = Tab::Home;
        app.paste(format!("\"{}\"", other.display()));
        assert_eq!(app.editor.as_ref().unwrap().text(), "b=2\n");
        assert!(app.editor.as_ref().unwrap().path.ends_with("storm space-界.toml"));
        assert!(app.job.is_none());
        assert_eq!(fs::read_to_string(&original).unwrap(), "a=1\n");

        app.tab = Tab::Settings;
        app.editor.as_mut().unwrap().insert("# unsaved edit\n");
        let draft = app.editor.as_ref().unwrap().text();
        app.paste(format!("\"{}\"", original.display()));
        assert_eq!(app.editor.as_ref().unwrap().text(), draft);
        assert!(app.editor.as_ref().unwrap().path.ends_with("storm space-界.toml"));
        assert!(app.status.contains("draft is unchanged"));
        assert_eq!(fs::read_to_string(&other).unwrap(), "b=2\n");

        app.paste("\n[render]\nproducts=['t2']\n".into());
        assert!(app.editor.as_ref().unwrap().text().contains("products=['t2']"));
        assert!(app.job.is_none());
    }

    #[test]
    fn multiple_dropped_files_require_selection_and_catalogs_use_cases() {
        let mut app = loaded_app("a=1\n");
        let original = app.editor.as_ref().unwrap().path.clone();
        let catalog = original.with_file_name("weather cases.zip");
        fs::write(&catalog, "catalog backend validates archive bytes").unwrap();
        app.paste(format!("\"{}\" \"{}\"", original.display(), catalog.display()));
        assert!(matches!(&app.dialog, Some(Dialog::DroppedFiles(paths, 0)) if paths.len()==2));
        assert_eq!(app.editor.as_ref().unwrap().path, original);
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let screen = render_at(&mut app, width, height);
            assert!(screen.contains("Choose a dropped file"));
            assert!(screen.contains("weather cases.zip") && screen.contains("experiment.toml"));
            assert!(app.hits.iter().any(|hit| matches!(hit.action, Hit::Key(KeyCode::Enter, _))));
        }
        app.key(press(KeyCode::Down));
        app.key(press(KeyCode::Enter));
        assert!(matches!(app.dialog, Some(Dialog::Cases(_))));
        assert_eq!(app.editor.as_ref().unwrap().path, original);
        assert_eq!(fs::read_to_string(&original).unwrap(), "a=1\n");
        assert!(app.job.is_none());
    }

    #[test]
    fn small_loaded_window_blocks_mutations_but_retains_quit_controls() {
        let mut app = loaded_app("a=1\n");
        let mut term = Terminal::new(TestBackend::new(64, 19)).unwrap();
        term.draw(|f| draw(f, &mut app)).unwrap();
        for code in [
            KeyCode::F(5),
            KeyCode::F(6),
            KeyCode::F(7),
            KeyCode::F(8),
            KeyCode::F(10),
            KeyCode::Enter,
        ] {
            app.key(press(code));
        }
        assert!(app.job.is_none());
        assert!(!app.output.exists());
        app.editor.as_mut().unwrap().insert("# draft\n");
        let draft = app.editor.as_ref().unwrap().text();
        let path = app.editor.as_ref().unwrap().path.clone();
        app.tab = Tab::Settings;
        for key in [
            press(KeyCode::Char('z')),
            press(KeyCode::Backspace),
            press(KeyCode::Delete),
            press(KeyCode::Enter),
            KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL),
            press(KeyCode::F(12)),
        ] {
            app.key(key);
        }
        app.paste("corrupt draft".into());
        assert_eq!(app.editor.as_ref().unwrap().text(), draft);
        assert_eq!(fs::read_to_string(path).unwrap(), "a=1\n");
        app.dialog = Some(Dialog::Path("Save draft as", "unchanged".into()));
        app.paste("changed".into());
        app.key(press(KeyCode::Enter));
        assert!(matches!(&app.dialog, Some(Dialog::Path(_, value)) if value == "unchanged"));
        app.dialog = Some(Dialog::Quit);
        app.key(press(KeyCode::Char('n')));
        assert!(!app.exit);
        let mut large = Terminal::new(TestBackend::new(80, 24)).unwrap();
        large.draw(|f| draw(f, &mut app)).unwrap();
        app.key(press(KeyCode::Esc));
        app.tab = Tab::Settings;
        app.key(press(KeyCode::Char('z')));
        assert_ne!(app.editor.as_ref().unwrap().text(), draft);
        app.set_viewport(64, 19);
        app.key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::CONTROL));
        app.key(press(KeyCode::Char('y')));
        assert!(app.exit);
    }

    #[test]
    fn global_help_and_quit_restore_guide_and_retained_workflow_answers() {
        let mut app = loaded_app("a=1\n");
        let mode = workflows::MODES.iter().position(|m| m.id == "convective").unwrap_or(0);
        app.workflow_route(mode, workflows::Route::New);
        let Some(Dialog::Guide(g)) = &mut app.dialog else { panic!("guide"); };
        g.questions[0].value = "35.3,-97.5".into();
        g.step = 2;
        for help in [KeyCode::F(1), KeyCode::Char('?')] {
            app.key(press(help));
            assert!(matches!(app.dialog, Some(Dialog::Help(_))));
            app.key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::CONTROL));
            assert!(matches!(app.dialog, Some(Dialog::Quit)));
            app.key(press(KeyCode::Char('n')));
            assert!(matches!(app.dialog, Some(Dialog::Help(_))));
            app.key(press(KeyCode::Esc));
            assert!(matches!(&app.dialog, Some(Dialog::Guide(g)) if g.step == 2 && g.questions[0].value == "35.3,-97.5"));
        }
        app.key(press(KeyCode::Esc));
        assert!(app.dialog.is_none());
        assert_eq!(app.saved_guides.len(), 1);
        app.workflow_route(mode, workflows::Route::New);
        assert!(matches!(&app.dialog, Some(Dialog::Guide(g)) if g.step == 2 && g.questions[0].value == "35.3,-97.5"));
        assert!(app.saved_guides.is_empty());
        assert!(app.job.is_none());
    }

    #[test]
    fn settings_undo_redo_discard_use_keys_and_visible_buttons() {
        let mut app = loaded_app("a=1\n");
        app.tab = Tab::Settings;
        app.key(press(KeyCode::Char('#')));
        let draft = app.editor.as_ref().unwrap().text();
        app.key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::CONTROL));
        assert_eq!(app.editor.as_ref().unwrap().text(), "a=1\n");
        render_at(&mut app, 65, 20);
        click_hit(&mut app, |hit| matches!(hit, Hit::Key(KeyCode::Char('y'), KeyModifiers::CONTROL)));
        assert_eq!(app.editor.as_ref().unwrap().text(), draft);
        app.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
        assert_eq!(app.editor.as_ref().unwrap().text(), "a=1\n");
        app.key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::CONTROL));
        assert_eq!(app.editor.as_ref().unwrap().text(), draft);
        assert_eq!(fs::read_to_string(&app.editor.as_ref().unwrap().path).unwrap(), "a=1\n");
        assert!(app.job.is_none());
    }

    #[test]
    fn quit_from_a_tiny_guide_waits_for_real_local_worker_termination() {
        let mut app = loaded_app("a=1\n");
        let python = PathBuf::from(env::var_os("GPUWM_TUI_TEST_PYTHON").expect("set test Python path"));
        let root = app.output.parent().unwrap().join("owned-quit-fixture");
        let package = root.join("gpuwm");
        fs::create_dir_all(&package).unwrap();
        fs::write(package.join("__init__.py"), "").unwrap();
        fs::write(package.join("tui_worker.py"), include_str!("../../../gpuwm/tui_worker.py")).unwrap();
        fs::write(package.join("cli.py"), "import time\ndef main(argv=None):\n print('benign quit-control fixture started', flush=True)\n time.sleep(30)\n return 0\n").unwrap();
        app.job = Some(Job::start_with_module_path(&python, "benign-quit-control", &[], &root.join("job"), &root, Some(&root)).unwrap());
        app.dialog = Some(Dialog::Guide(complete_guide(&app)));
        app.set_viewport(64, 19);
        app.key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::CONTROL));
        assert!(matches!(app.dialog, Some(Dialog::Quit)));
        app.key(press(KeyCode::Char('y')));
        assert!(!app.exit, "Quit must wait for the worker outcome");
        assert!(app.exit_after_job);
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !app.exit {
            app.poll();
            assert!(std::time::Instant::now() < deadline, "owned worker did not stop: {}", app.status);
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(!app.busy());
        assert!(app.job.as_ref().unwrap().interrupted());
        assert_ne!(app.job.as_ref().unwrap().outcome, Some(0));
    }

    #[test]
    fn repeated_quit_does_not_discard_a_draft() {
        let mut app = loaded_app("a=1\n");
        app.editor.as_mut().unwrap().insert("# draft\n");
        app.key(press(KeyCode::Char('q')));
        assert!(matches!(app.dialog, Some(Dialog::Quit)));
        app.key(KeyEvent::new_with_kind(
            KeyCode::Char('q'),
            KeyModifiers::NONE,
            KeyEventKind::Repeat,
        ));
        app.key(press(KeyCode::Char('q'))); // terminals without repeat reporting
        assert!(!app.exit && app.dirty());
        app.key(press(KeyCode::Char('y')));
        assert!(app.exit);
    }

    #[test]
    fn active_configuration_is_protected_while_a_separate_draft_remains_editable() {
        let mut app = loaded_app("a=1\n");
        let active = app.editor.as_ref().unwrap().path.clone();
        app.active_config = Some(active.clone());
        app.editor.as_mut().unwrap().insert("# next run\n");
        app.save();
        assert!(app.dirty());
        assert_eq!(fs::read_to_string(&active).unwrap(), "a=1\n");
        assert!(app.status.contains("running command"));
        assert!(app.save_as(PathBuf::from("next.toml")));
        assert_ne!(app.editor.as_ref().unwrap().path, active);
        assert_eq!(app.active_config.as_ref(), Some(&active));
        app.editor.as_mut().unwrap().insert("# more changes\n");
        app.save();
        assert!(!app.dirty());
        assert_eq!(fs::read_to_string(&active).unwrap(), "a=1\n");
    }

    #[test]
    fn conflict_export_is_accessible_by_both_save_as_keys_and_keeps_external_bytes() {
        let mut app = loaded_app("# café 🌧\na=1\n");
        let original = app.editor.as_ref().unwrap().path.clone();
        app.editor.as_mut().unwrap().insert("[incomplete");
        let draft = app.editor.as_ref().unwrap().text();
        fs::write(&original, "a=2\n").unwrap();
        app.save();
        assert!(app.dirty());
        app.key(KeyEvent::new(
            KeyCode::Char('S'),
            KeyModifiers::CONTROL | KeyModifiers::SHIFT,
        ));
        assert!(matches!(app.dialog, Some(Dialog::Path("Save draft as", _))));
        app.key(press(KeyCode::Esc));
        app.key(press(KeyCode::F(12)));
        let target = match &app.dialog {
            Some(Dialog::Path("Save draft as", value)) => PathBuf::from(value),
            _ => panic!("no export dialog"),
        };
        assert_eq!(
            fs::canonicalize(target.parent().unwrap()).unwrap(),
            original.parent().unwrap()
        );
        app.key(press(KeyCode::Enter));
        assert_eq!(fs::read_to_string(&target).unwrap(), draft);
        assert_eq!(fs::read_to_string(&original).unwrap(), "a=2\n");
        assert!(app.status.contains("syntactically invalid"));
        // Syntax validation is still the real CLI's job; saved incomplete
        // content is forwarded honestly rather than changed or substituted.
        let (command, args) = app.argv(Action::Check).unwrap();
        assert_eq!(command, "check");
        assert_eq!(PathBuf::from(&args[0]), app.editor.as_ref().unwrap().path);
    }

    #[test]
    fn loaded_unicode_cursor_and_delete_follow_the_visible_character_after_resize() {
        let mut app = loaded_app(&format!("# {}X marker\na=1\n", "界".repeat(50)));
        app.tab = Tab::Settings;
        app.editor.as_mut().unwrap().col = 52;
        for (w, h) in [(120, 36), (65, 20), (80, 24)] {
            let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
            term.draw(|f| draw(f, &mut app)).unwrap();
            let cursor = term.get_cursor_position().unwrap();
            assert_eq!(term.backend().buffer()[(cursor.x, cursor.y)].symbol(), "X");
        }
        app.key(press(KeyCode::Delete));
        app.key(press(KeyCode::Char('Z')));
        app.key(press(KeyCode::Left));
        let mut term = Terminal::new(TestBackend::new(65, 20)).unwrap();
        term.draw(|f| draw(f, &mut app)).unwrap();
        let cursor = term.get_cursor_position().unwrap();
        assert_eq!(term.backend().buffer()[(cursor.x, cursor.y)].symbol(), "Z");
        assert!(app
            .editor
            .as_ref()
            .unwrap()
            .text()
            .contains(&format!("{}Z marker", "界".repeat(50))));
    }
    #[test]
    fn geography_folder_is_optional_and_only_forwarded_to_go() {
        let mut app = loaded_app("name = 'unchanged'\n");
        for action in [Action::Plan, Action::Run] {
            assert!(!app
                .argv(action)
                .unwrap()
                .1
                .iter()
                .any(|v| v == "--geog-root"));
        }
        app.key(KeyEvent::new(KeyCode::F(11), KeyModifiers::NONE));
        assert!(matches!(
            app.dialog,
            Some(Dialog::Path("Geography folder", _))
        ));
        app.paste("geography data".into());
        app.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let expected = app
            .cwd
            .join("geography data")
            .to_string_lossy()
            .into_owned();
        for action in [Action::Plan, Action::Run] {
            let (command, args) = app.argv(action).unwrap();
            assert_eq!(command, "go");
            assert!(args
                .windows(2)
                .any(|pair| pair == ["--geog-root", expected.as_str()]));
        }
        app.prepared = app.cwd.join("prepared");
        for action in [Action::Check, Action::Prepared, Action::Doctor] {
            assert!(!app
                .argv(action)
                .unwrap()
                .1
                .iter()
                .any(|v| v == "--geog-root"));
        }
        app.key(KeyEvent::new(KeyCode::F(11), KeyModifiers::NONE));
        app.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
        app.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(app.geog_root.as_os_str().is_empty());
        assert_eq!(app.editor.as_ref().unwrap().text(), "name = 'unchanged'\n");
    }
    #[test]
    fn local_forecast_progress_keeps_native_timing_and_nonforecast_output_readable() {
        for action in ["run-plan", "go", "sim", "run", "resume"] {
            assert!(local_forecast_action(action, &[]));
        }
        for action in ["check", "doctor", "sources", "domain", "case-catalog"] {
            assert!(!local_forecast_action(action, &[]));
        }
        for flag in ["--dry-run", "--physics-profiles", "--help"] {
            assert!(!local_forecast_action("run-plan", &[flag.into()]));
        }
        let fixture:serde_json::Value=serde_json::from_str(include_str!("../tests/fixtures/native-progress-newcastle-2013.json")).unwrap();
        let mut status=serde_json::json!({"state":"running","phase":"integrating","progress":fixture["native_result"]["progress"]});
        for (width,height) in [(65,12),(80,16),(120,24)] {
            let mut terminal=Terminal::new(TestBackend::new(width,height)).unwrap();
            terminal.draw(|frame|draw_forecast_progress(frame,frame.area(),&status)).unwrap();
            let buffer=terminal.backend().buffer();
            let rendered=(0..height).map(|y|(0..width).map(|x|buffer[(x,y)].symbol()).collect::<String>()).collect::<Vec<_>>().join("\n");
            for label in ["Forecast progress","RUNNING","Steps 360","Wall time","Checkpoint"] {
                assert!(rendered.contains(label),"Missing {label} at {width}x{height}:\n{rendered}");
            }
            assert!(!rendered.contains("RENDER_SUMMARY"));
        }
        status["state"]=serde_json::json!("completed");
        assert!(node_ui::job_progress_text(&status,true).starts_with("COMPLETED · Finished"));
    }

    #[test]
    fn local_forecast_progress_toggle_preserves_copy_open_stop_and_geography() {
        let Some(python)=env::var_os("GPUWM_TUI_TEST_PYTHON").map(PathBuf::from) else{return};
        let mut app=loaded_app("# unchanged local setup\na=1\n");
        let root=app.editor.as_ref().unwrap().path.parent().unwrap().to_path_buf();
        let package=root.join("gpuwm");fs::create_dir(&package).unwrap();
        fs::write(package.join("__init__.py"),"").unwrap();
        fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../gpuwm/tui_worker.py"),package.join("tui_worker.py")).unwrap();
        fs::write(package.join("cli.py"),"def main(argv=None):\n return 0\n").unwrap();
        app.job=Some(Job::start_with_module_path(&python,"run-plan",&[],&root.join("job"),&root,Some(&root)).unwrap());
        let raw="native raw fixture log café\n";
        fs::write(app.job.as_ref().unwrap().dir.join("job.log"),raw).unwrap();
        app.tab=Tab::Logs;app.clipboard_hook=Some(record_copy);
        let draft=app.editor.as_ref().unwrap().text();
        for (width,height) in [(65,20),(80,24),(120,36)] {
            let screen=render_at(&mut app,width,height);
            for label in ["Forecast progress","G Raw logs","Y Copy logs","O Open log","X Stop"] {
                assert!(screen.contains(label),"Missing {label} at {width}x{height}:\n{screen}");
            }
            assert!(!screen.contains("native raw fixture"));
            app.log_offset=100;
            app.key(press(KeyCode::Char('G')));
            assert!(app.local_raw_logs&&app.log_offset==0&&app.dialog.is_none());
            let screen=render_at(&mut app,width,height);
            assert!(screen.contains("native raw fixture")&&screen.contains("G Progress"),"{screen}");
            app.key(press(KeyCode::PageUp));assert_eq!(app.log_offset,10);
            app.key(press(KeyCode::End));assert_eq!(app.log_offset,0);
            app.key(press(KeyCode::Char('y')));
            COPIED.with(|copies|assert!(copies.borrow().last().unwrap().contains(raw.trim())));
            app.key(press(KeyCode::Char('o')));assert!(app.status.contains(".txt"));
            app.key(press(KeyCode::Char('g')));assert!(!app.local_raw_logs);
            app.key(KeyEvent::new(KeyCode::Char('g'),KeyModifiers::CONTROL));
            assert!(matches!(app.dialog,Some(Dialog::Path("Geography folder",_))));app.key(press(KeyCode::Esc));
            app.key(press(KeyCode::Char('x')));assert!(matches!(app.dialog,Some(Dialog::Stop)));app.key(press(KeyCode::Esc));
            assert_eq!(app.editor.as_ref().unwrap().text(),draft);
        }
        let deadline=std::time::Instant::now()+Duration::from_secs(10);
        while app.job.as_mut().unwrap().poll().unwrap().is_none(){
            assert!(std::time::Instant::now()<deadline);std::thread::sleep(Duration::from_millis(10));
        }
        assert!(render_at(&mut app,80,24).contains("COMPLETED"));
        app.job.as_mut().unwrap().action="sources".into();
        assert!(!app.has_local_forecast_job());
        let screen=render_at(&mut app,80,24);assert!(!screen.contains("G Raw logs"));
    }

    #[test]
    fn loaded_long_log_tail_scroll_and_resize_show_latest_display_rows() {
        let mut app = loaded_app("name = 'log control'\n");
        app.tab = Tab::Logs;
        let long = format!(
            "--input={} --next={}",
            "界界e\u{301}🙂/".repeat(30),
            "a".repeat(220)
        );
        let text = format!(
            "{}\nstep 79\nstep 80 LATEST OUTPUT",
            (0..40)
                .map(|_| long.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        );
        let logfile = app
            .editor
            .as_ref()
            .unwrap()
            .path
            .parent()
            .unwrap()
            .join("job.log");
        fs::write(&logfile, &text).unwrap();
        let loaded = fs::read_to_string(logfile).unwrap();
        for width in [120, 65, 80] {
            app.set_viewport(width, 24);
            let mut term = Terminal::new(TestBackend::new(width, 24)).unwrap();
            let render = |term: &mut Terminal<TestBackend>, offset: usize| {
                term.draw(|frame| draw_log(frame, frame.area(), &loaded, offset, "Log"))
                    .unwrap();
                let buffer = term.backend().buffer();
                (0..24)
                    .map(|y| {
                        (0..width)
                            .map(|x| buffer[(x, y)].symbol())
                            .collect::<String>()
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            app.key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE));
            assert!(render(&mut term, app.log_offset).contains("step 80 LATEST OUTPUT"));
            app.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
            assert!(!render(&mut term, app.log_offset).contains("LATEST OUTPUT"));
            app.key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE));
            assert!(render(&mut term, app.log_offset).contains("step 80 LATEST OUTPUT"));
            let wrapped = log_display_rows(&long, usize::from(width - 2));
            assert!(wrapped
                .iter()
                .all(|row| UnicodeWidthStr::width(row.as_str()) <= usize::from(width - 2)));
            assert_eq!(wrapped.concat(), long);
        }
    }
}
