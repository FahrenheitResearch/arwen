//! Questions collect exact CLI arguments; the existing engine owns all science.
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    New,
    Research,
    Fit,
    Tiles,
    Wrf,
    MetEm,
    Resume,
    Prepared,
    Downscale,
    Render,
}
#[derive(Clone)]
pub struct Question {
    pub label: &'static str,
    pub flag: &'static str,
    pub help: &'static str,
    pub value: String,
    pub required: bool,
}
#[derive(Clone)]
pub struct Guide {
    pub kind: Kind,
    pub workflow: Option<&'static str>,
    pub research: Option<&'static str>,
    pub questions: Vec<Question>,
    pub step: usize,
    pub summary_edit: bool,
}
#[derive(Clone)]
pub struct Request {
    pub command: String,
    pub args: Vec<String>,
    pub created: Option<PathBuf>,
    pub title: &'static str,
}
fn q(
    label: &'static str,
    flag: &'static str,
    help: &'static str,
    value: impl Into<String>,
    required: bool,
) -> Question {
    Question {
        label,
        flag,
        help,
        value: value.into(),
        required,
    }
}

fn unused_config_path(path: PathBuf) -> PathBuf {
    // Native writers publish a TOML and same-stem companions such as
    // .namelist.wps, .d01-target.json and .fit.json. Reserve the whole stem
    // when choosing a default; explicit paths still reach the normal writer
    // checks unchanged. This read does not allocate or reserve any file.
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let extension = path.extension().unwrap_or_default().to_string_lossy();
    let normalize = |value: &str| {
        if cfg!(windows) { value.to_lowercase() } else { value.to_owned() }
    };
    let occupied: Vec<_> = std::fs::read_dir(parent)
        .into_iter().flatten().filter_map(Result::ok)
        .map(|entry| normalize(&entry.file_name().to_string_lossy())).collect();
    for suffix in 1u64.. {
        let candidate_stem = if suffix == 1 { stem.to_string() } else { format!("{stem}-{suffix}") };
        let compared = normalize(&candidate_stem);
        let prefix = format!("{compared}.");
        if !occupied.iter().any(|name| name == &compared || name.starts_with(&prefix)) {
            return if suffix == 1 { path } else {
                parent.join(format!("{candidate_stem}.{extension}"))
            };
        }
    }
    unreachable!("a directory cannot occupy every u64 filename suffix")
}

impl Guide {
    pub fn validate_current(&self) -> Result<(), String> {
        let field = &self.questions[self.step];
        let value = field.value.trim();
        if self.kind == Kind::New && field.flag == "@location" && value.is_empty()
            && self.questions[2].value.trim().is_empty() {
            return Err("Enter latitude,longitude or a GeoJSON boundary file before continuing.".into());
        }
        if field.required && value.is_empty() { return Err(format!("{} is required.", field.label)); }
        if field.flag == "--hardware-class" && !matches!(value, "auto" | "8" | "12" | "16" | "24" | "32") {
            return Err("Research GPU profile must be auto, 8, 12, 16, 24 or 32. Type only the number; use 8 for an 8 GiB target.".into());
        }
        if field.flag == "@advanced" && !matches!(value, "off" | "on") { return Err("Advanced CLI options must be off or on.".into()); }
        if field.flag == "@extra" && !value.is_empty() && serde_json::from_str::<Vec<String>>(value).is_err() {
            return Err("Advanced arguments need a list of quoted strings, for example [\"--projection\",\"lambert\"]. Clear this field to continue without them.".into());
        }
        Ok(())
    }
    pub fn new(kind: Kind, cwd: &Path, output: &Path) -> Self {
        let out = output.to_string_lossy().into_owned();
        let mut questions=match kind {
   Kind::Research=>vec![
    q("Location: latitude,longitude OR GeoJSON file", "@location", "Centre your research question, or select a GeoJSON footprint. Inspect the generated coverage and nest corridor before preparing.", "", true),
    q("Input source", "--source", "The native source registry validates coverage, physics and available times. GFS supplies a global starting point.", "gfs", true),
    q("Start cycle (UTC)", "--cycle", "YYYY-MM-DDTHH. Choose the event and period you intend to study. latest may use the network.", "", true),
    q("Forecast duration (hours)", "--hours", "The recommendation's study window. The native source horizon remains authoritative.", "6", true),
    q("Research GPU profile: auto, 8, 12, 16, 24, 32", "--hardware-class", "Type auto or a bare number such as 8. Start with auto, or choose 8 for an 8 GiB target. A profile selects grid spacing and vertical detail; it does not promise a fit or increase memory. A 24 or 32 profile on an 8 GiB target can fail the question's minimum study area. Try auto or 8 for the same question before changing the question.", "auto", true),
    q("Target GPU capacity (GiB, optional)", "--vram-gib", "Blank measures this computer's capacity AND free memory. Enter capacity only when planning for another GPU; that becomes an estimate rather than a measurement.", "", false),
    q("New research configuration file", "--out", "A NEW TOML and its companion files. The generated configuration exposes actual geometry, tracker and scenario settings for review; no forecast starts.", unused_config_path(cwd.join("research.toml")).to_string_lossy(), true),
    q("Experiment name (optional)", "--name", "A descriptive label for this particular case and comparison.", "", false),
    q("Physics starting suite (optional)", "--physics-profile", "Blank uses the native source default. An explicit suite must be compatible with the source, time window and method.", "", false),
   ],
   Kind::New=>vec![
    q("Forecast name", "--name", "A name for this forecast; all settings remain editable before launch.", "My forecast", false),
    q("Location: latitude,longitude OR GeoJSON file", "@location", "A point sizes a rectangular grid around your centre using the GPU budget. A GeoJSON boundary preserves the requested area inside the projected grids. Type decimal degrees such as 29.87,-90.07, or a boundary file. Ctrl+A opens all settings.", "", false),
    q("GeoJSON boundary file (optional)", "--polygon", "Polygon or MultiPolygon file. Use either a center OR a boundary, not both.", "", false),
    q("Root grid spacing (km)", "--root-dx", "An editable 12 km starting point. Exact numbers pass to the existing domain command.", "12", false),
    q("Nested grid ratios (optional)", "--chain", "Comma-separated ratios, for example 3,2. Empty creates one domain.", "", false),
    q("Vertical levels (optional)", "--nz", "Empty keeps the engine default. Full eta and domain settings can be edited in TOML afterward.", "", false),
    q("GPU memory (GiB, optional)", "--vram-gib", "Empty asks the native wizard to measure the local GPU. Enter capacity for another machine.", "", false),
    q("Input source", "--source", "The shown recommendation comes from the installed guided CLI. Any registered ID or alias is accepted. F2 shows sources; Esc returns here.", "", true),
    q("Start cycle (UTC)", "--cycle", "latest opens a check for the newest complete period. F3 opens the source calendar with cycle hours and archive guidance. You can also type an exact YYYY-MM-DDTHH date in UTC. Analysis services show a publication estimate when no object probe exists.", "latest", true),
    q("Forecast duration (hours)", "--hours", "Exact whole hours accepted by domain. Full TOML later exposes run_seconds.", "6", true),
    q("Physics starting suite (optional)", "--physics-profile", "Empty keeps the native source default. F2 shows the engine's suites. Full physics remains editable afterward.", "", false),
    q("Root output interval (seconds, optional)", "--history-interval", "Empty keeps the engine default; the engine checks compatibility with the exact time step.", "", false),
    q("New configuration file", "--out", "A NEW .toml file. The native wizard also writes its companion WPS namelist. Existing files are not replaced here.", unused_config_path(cwd.join("forecast.toml")).to_string_lossy(), true),
    q("Forcing data directory (optional)", "--data-dir", "Empty keeps the native domain default. No forecast starts when the configuration is created.", "", false),
   ],
   Kind::Fit=>vec![
    q("Editable starter TOML", "", "A complete ArWen configuration. Edit its physics and grid spacing before fitting; those choices are preserved.", "", true),
    q("Location: latitude,longitude OR GeoJSON file", "@location", "Point fits a centered layout to the GPU. GeoJSON preserves the full requested area.", "", true),
    q("Input source (optional)", "--source", "Empty uses the starter's fetch source. Supply a source only when the starter has no fetch source.", "", false),
    q("Start time (UTC)", "--start-time", "Enter the intended UTC start, for example 2026-09-05T00. This explicitly updates the starter's time and fetch cycle.", "", true),
    q("Duration (hours, optional)", "--hours", "Empty preserves the duration in the starter. Enter a value to change it explicitly.", "", false),
    q("GPU memory (GiB, optional)", "--vram-gib", "Empty detects this machine's GPU. Enter capacity when sizing for another machine.", "", false),
    q("New fitted configuration", "--out", "A new TOML file. Review all changes in the command log; the starter is never overwritten.", unused_config_path(cwd.join("fitted.toml")).to_string_lossy(), true),
   ],
   Kind::Tiles=>vec![
    q("Current configuration", "", "Keep this configuration's domain area, resolution, timing and physics. The original file is preserved.", "", true),
    q("New streaming configuration", "--out", "A new TOML file. The engine checks current GPU and system RAM and plans tile dimensions before creating it. No forecast starts.", unused_config_path(cwd.join("streaming.toml")).to_string_lossy(), true),
    q("Streaming mode: auto or on", "--mode", "auto streams when needed; on forces streaming. Both ask the engine to plan GPU tile dimensions with the full domain in system RAM.", "auto", true),
   ],
   Kind::Render=>vec![
    q("Forecast history file or folder", "@history", "An actual wrfout file, or a folder containing wrfout files. Compatible files form a timeline for windowed plots; separate runs, domains and lifecycle episodes stay separate. Discovery includes child folders, skips symbolic links, and is bounded. The exact files appear in the command review.", "", true),
    q("Requested plots", "--products", "Comma-separated canonical products. The task mode supplies a starting selection; all is also accepted. Missing fields/windows are reported by the native renderer.", "all", true),
    q("Frame index or all", "--timeidx", "all plots every saved frame; a nonnegative index selects one record within each file.", "all", true),
    q("Image output folder", "--out", "An ordinary output root gets a fresh timestamped run folder. Choosing a folder already inside run-YYYY... reuses that run and may replace matching plots; review the path deliberately.", cwd.join("weather-plots").to_string_lossy(), true),
    q("Source label shown on plots", "--source-label", "Name the model and whether this is a hypothetical or changed-initial-state run. This label is printed on the actual images.", "External / unspecified forecast", true),
    q("Image size in pixels", "--size", "Width x height, for example 1200x900. Use the native renderer's supported dimensions.", "1200x900", true),
   ],
   Kind::Downscale=>vec![
    q("Archived parent wrfout folder or file", "", "Choose actual parent history from ArWen or stock WRF. The engine checks its physics, times, geometry and boundary cadence.", "", true),
    q("Child centre: latitude,longitude", "--point", "For an ArWen parent, derive a child around this point. Alternatively leave blank and supply a child configuration under All settings.", "", false),
    q("Existing child RunConfig TOML", "--child-config", "Alternative to a centre point. This standalone child uses specified=true and nested=false; set ratio and parent start indices below.", "", false),
    q("Parent ArWen restart (physics evidence)", "--parent-restart", "The parent run's restart, or leave blank and provide its stock-WRF namelist in All settings. The engine decides which evidence is required.", "", false),
    q("Parent stock-WRF namelist.input", "--parent-namelist", "Alternative to an ArWen parent restart. Select the producing namelist, never an unrelated physics preset.", "", false),
    q("Parent domain ID", "--parent-domain", "Optional domain selector when the archive contains several domains.", "", false),
    q("Refinement ratio", "--ratio", "Required with a child configuration. For a centre point, blank keeps the native default.", "", false),
    q("Child size: nx or nx,ny", "--child-size", "Optional explicit extent for a centre-point child. Blank lets the engine size it using the requested GPU capacity.", "", false),
    q("Child southwest parent index i", "--i-parent-start", "Required with a child configuration; a centre-point child derives its placement.", "", false),
    q("Child southwest parent index j", "--j-parent-start", "Required with a child configuration; these are 1-based parent cells.", "", false),
    q("Child surface warm-start file", "--child-surface-from", "Child-grid wrfinput/history carrying land identity and soil warm start. Required when the child's surface physics needs it.", "", false),
    q("Maximum parent boundary interval (seconds)", "--max-boundary-interval-seconds", "Explicit acceptable forcing cadence. Alternatively explicitly accept the archive cadence in the next setting. The two choices are mutually exclusive.", "", false),
    q("Accept parent archive cadence: true/false", "@accept-parent-cadence", "An explicit scientific choice: true uses the archive's own cadence as the ceiling; blank or false does not accept it. Cannot combine with a maximum interval.", "", false),
    q("Child duration (hours)", "--hours", "Optional centre-point run window. Blank keeps the full available parent window.", "", false),
    q("Child output interval (seconds)", "--output-interval-seconds", "Optional centre-point history cadence. Blank keeps the native inherited cadence.", "", false),
    q("Child vertical levels: N,STRETCH", "--child-levels", "Optional independent vertical ladder clustered toward the ground. Enter the intended stretch; the engine validates the vertical remap.", "", false),
    q("GPU memory capacity (GiB)", "--vram-gib", "Blank measures this computer's total and free GPU memory. Enter capacity for another machine; this turns off automatic measurement. A supplied child configuration or explicit child size keeps its own dimensions and is priced on whichever card this setting names.", "", false),
    q("Streaming for point child: on/auto", "--tiles", "Optional centre-point streaming selection. For a child configuration, edit its own [tiles] settings instead.", "", false),
    q("New downscaled output directory", "--out", "A new directory. Planning can write the derived TOML and reports; an existing directory is never replaced.", cwd.join("downscaled-run").to_string_lossy(), true),
    q("Action: plan or run", "@downscale-mode", "plan invokes --dry-run to validate and write the derived plan without a forecast. run starts the offline child only after you review and confirm the exact command.", "plan", true),
    q("Parent namelist domain column", "--parent-namelist-domain", "Optional column of the stock-WRF namelist corresponding to this parent (native default 1). This is distinct from selecting a wrfout domain ID.", "", false),
    q("Measure local GPU free memory: true/false", "@auto-vram", "true measures actual total and free memory: it fits a point child when no size is given, and prices an explicit child size or a supplied child configuration on the measured card otherwise. Entering a capacity turns this off. false uses explicit capacity or the native 24 GiB default when none is supplied.", "true", false),
   ],
   Kind::Wrf|Kind::MetEm=>vec![q("Existing input directory", if kind==Kind::Wrf {"--wrfinput"}else{"--met-em"}, if kind==Kind::Wrf {"Folder containing wrfinput_d0*, wrfbdy_d01 and the producing namelist.input. Its physics stays authoritative."}else{"Folder containing met_em.d0*.nc and the producing namelist.input. Its settings stay authoritative."}, "", true), q("Shorten duration (seconds, optional)","--run-seconds","Empty preserves the producing namelist duration.","",false),q("Forecast output directory","--outdir","Run output will be written here. Review the exact command before starting.",out,true)],
   Kind::Resume=>vec![q("Original configuration file","","The same ArWen TOML used by the interrupted run. Checkpoint identity is checked by the existing engine.","",true),q("Checkpoint file or latest","--from","Use an actual gpuwmrst checkpoint, or latest to let the engine locate a valid set in the output directory.","latest",true),q("Existing forecast output directory","--outdir","The interrupted run's wrfout/checkpoint directory. A log folder alone is not a checkpoint.",out,true)],
   Kind::Prepared=>vec![q("Prepared bundle directory","","The actual native preparation folder; not a Downloads folder or raw input file.","",true),q("ArWen configuration file","--experiment-config","The configuration belonging to this prepared bundle.","",true),q("Producing WPS namelist (optional)","--wps-namelist","Select the exact WPS authority used during prep when required by the bundle.","",false),q("Forecast output directory","--outdir","Run output will be written here. The engine validates preparation identity.",out,true)],
  };
        if kind != Kind::Tiles {
            questions.push(q("Advanced CLI options: off/on","@advanced","Leave off for the ordinary guided setup. Turn on to expose optional exact CLI arguments. Turning off removes those custom arguments from this guide.","off",false));
        }
        Self {
            kind,
            workflow: None,
            research: None,
            questions,
            step: if kind == Kind::New { 1 } else { 0 },
            summary_edit: false,
        }
    }
    pub fn apply_workflow(&mut self, mode: &'static crate::workflows::Mode, cwd: &Path) {
        self.workflow = Some(mode.id);
        if self.kind == Kind::New {
            self.questions[0].value = mode.title.into();
            self.questions[4].value = mode.chain.into();
            self.questions[9].value = mode.hours.into();
            self.questions[12].value = unused_config_path(cwd.join(format!("{}.toml", mode.id))).to_string_lossy().into_owned();
        } else if self.kind == Kind::Render {
            if let Ok(selection) = crate::plotsettings::Selection::preset_id(mode.preset) { self.questions[1].value = selection.spec; }
            if mode.scenario { self.questions[4].value = "HYPOTHETICAL scenario / source unspecified".into(); }
        }
    }
    pub fn date_question(&self) -> bool {
        matches!(self.questions[self.step].flag, "--cycle" | "--start-time")
    }
    pub fn date_query_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        for question in &self.questions {
            if matches!(question.flag, "--source" | "--hours") && !question.value.trim().is_empty() {
                args.extend([question.flag.into(), question.value.trim().into()]);
            }
        }
        if self.kind == Kind::Fit && !self.questions[0].value.trim().is_empty() {
            args.extend(["--config".into(), self.questions[0].value.trim().into()]);
        }
        args
    }
    pub fn geometry_hint(&self) -> Option<String> {
        if self.kind == Kind::Research && matches!(self.step, 4 | 5) {
            return Some(format!("Selected profile: {}. Target capacity: {}. Creation checks the full configuration and required study area before writing it.",
                self.questions[4].value, if self.questions[5].value.trim().is_empty() { "measure this computer".into() } else { format!("{} GiB (declared estimate)", self.questions[5].value) }));
        }
        if self.kind != Kind::New || !matches!(self.step, 1 | 2 | 3 | 4) { return None; }
        let mut spacing = self.questions[3].value.trim().parse::<f64>().ok()?;
        if !spacing.is_finite() || spacing <= 0.0 { return None; }
        let mut chain = vec![format!("d01 {spacing} km")];
        for (index, ratio) in self.questions[4].value.split(',').filter(|v| !v.trim().is_empty()).enumerate() {
            let ratio = ratio.trim().parse::<u32>().ok().filter(|r| *r >= 2)?;
            spacing /= f64::from(ratio);
            chain.push(format!("d{:02} ≈{:.3} km", index + 2, spacing));
        }
        Some(format!("Grid spacing: {}. Nest ratios refine spacing; they do not choose the geographic extent. Review the generated domains before running.", chain.join(" → ")))
    }
    pub fn apply_research(&mut self, row: &'static serde_json::Value, cwd: &Path) {
        self.research = Some(crate::research::text(row, "id"));
        if self.kind == Kind::Research {
            self.questions[3].value = row["geometry"]["forecast_hours"].as_f64().unwrap_or(6.0).to_string();
            self.questions[6].value = unused_config_path(cwd.join(format!("{}.toml", crate::research::text(row, "id")))).to_string_lossy().into_owned();
            self.questions[7].value = crate::research::text(row, "title").into();
        } else if self.kind == Kind::Render {
            self.questions[1].value = crate::research::strings(row, "diagnostics").join(",");
            if row["scenario"].is_object() || crate::research::family_for_config(row) == Some("hypothetical_tropical") {
                self.questions[4].value = "HYPOTHETICAL research scenario / source unspecified".into();
            }
        } else if self.kind == Kind::Downscale {
            self.questions[6].value = row["geometry"]["nest_ratios"].as_array()
                .and_then(|ratios| ratios.first()).and_then(|ratio| ratio.as_u64())
                .map(|ratio| ratio.to_string()).unwrap_or_default();
            self.questions[6].help = "This recommendation's refinement ratio. Review the actual parent spacing: a 4 km parent with ratio 3 produces a 1.33 km child. A different parent changes the study geometry; the archive must supply the full requested period and surface state.";
            self.questions[18].value = cwd.join(format!("{}-downscaled", crate::research::text(row, "id"))).to_string_lossy().into_owned();
            self.questions[13].value = row["geometry"]["forecast_hours"].as_f64().unwrap_or(6.0).to_string();
            self.questions[14].value = row["geometry"]["history_interval_s"].as_f64().unwrap_or(300.0).to_string();
            self.questions[21].value = "true".into();
            // Child output cadence does not constrain the archived parent's
            // input cadence. Keep the native parent-cadence choice visible;
            // users can instead set a stricter explicit maximum in this guide.
            self.questions[11].value.clear();
            self.questions[12].value = "true".into();
        }
    }
    fn essentials(&self) -> Option<&'static [usize]> {
        match self.kind {
            Kind::New => Some(&[1, 7, 8, 9, 12]),
            Kind::Research => Some(&[0, 1, 2, 3, 4, 5, 6]),
            Kind::Fit => Some(&[0, 1, 3, 4, 6]),
            Kind::Downscale => Some(&[0, 1, 3, 11, 16, 18, 19]),
            _ => None,
        }
    }
    pub fn progress(&self) -> String {
        if self.summary_edit {
            return "Edit setting".into();
        }
        if let Some(order) = self.essentials() {
            return format!(
                "Essential {}/{}",
                order.iter().position(|s| *s == self.step).unwrap_or(0) + 1,
                order.len()
            );
        }
        format!("Question {}/{}", self.step + 1, self.questions.iter().filter(|q| !q.flag.starts_with("@advanced") && q.flag != "@extra").count())
    }
    pub fn sync_choices(&mut self) {
        let advanced = self.questions.iter().any(|q| q.flag == "@advanced" && q.value.trim() == "on");
        if advanced && !self.questions.iter().any(|q| q.flag == "@extra") {
            self.questions.push(q("Advanced: exact CLI arguments (JSON)", "@extra", "Optional argument strings, for example [\"--projection\",\"lambert\"]. Leave blank to add nothing. These arguments go to the native parser without a shell.", "", false));
        } else if !advanced {
            self.questions.retain(|q| q.flag != "@extra");
        }
        if self.kind == Kind::Downscale {
            // Only a declared capacity turns measuring off: an explicit
            // child size or a supplied configuration is priced on the
            // measured card when measuring stays on.
            if !self.questions[16].value.trim().is_empty() {
                self.questions[21].value = "false".into();
            }
            if !self.questions[11].value.trim().is_empty() {
                self.questions[12].value = "false".into();
            }
        }
        self.step = self.step.min(self.questions.len().saturating_sub(1));
    }
    pub fn advance(&mut self) -> bool {
        self.sync_choices();
        if self.summary_edit {
            return false;
        }
        if let Some(order) = self.essentials() {
            let at = order.iter().position(|s| *s == self.step).unwrap_or(0);
            if at + 1 < order.len() {
                self.step = order[at + 1];
                true
            } else {
                false
            }
        } else if let Some(next) = ((self.step + 1)..self.questions.len()).find(|i| !matches!(self.questions[*i].flag, "@advanced" | "@extra")) {
            self.step = next;
            true
        } else {
            false
        }
    }
    pub fn previous(&mut self) {
        if !self.summary_edit {
            if let Some(order) = self.essentials() {
                let at = order.iter().position(|s| *s == self.step).unwrap_or(0);
                self.step = order[at.saturating_sub(1)];
                return;
            }
        }
        self.step = self.step.saturating_sub(1);
    }
    pub fn title(&self) -> &'static str {
        if let Some(row) = self.research.and_then(crate::research::config_by_id) { return crate::research::text(row, "title"); }
        if self.kind == Kind::New {
            if let Some(mode) = self.workflow.and_then(crate::workflows::mode) { return mode.title; }
        }
        match self.kind {
            Kind::New => "New forecast",
            Kind::Research => "Research configuration",
            Kind::Fit => "Fit an editable starter",
            Kind::Tiles => "Keep domain geometry with tile streaming",
            Kind::Wrf => "Use WRF inputs",
            Kind::MetEm => "Use WPS met_em",
            Kind::Resume => "Continue from checkpoint",
            Kind::Prepared => "Run prepared bundle",
            Kind::Downscale => "Downscale archived forecast",
            Kind::Render => "Plot existing forecast history",
        }
    }
    pub fn request(&self, cwd: &Path) -> Result<Request, String> {
        for field in &self.questions {
            if field.required && field.value.trim().is_empty() {
                return Err(format!("{} is required.", field.label));
            }
        }
        if let Some(field) = self.questions.iter().find(|q| q.flag == "--hardware-class") {
            if !matches!(field.value.trim(), "auto" | "8" | "12" | "16" | "24" | "32") {
                return Err("Research GPU profile must be auto, 8, 12, 16, 24 or 32. Type only the number; use 8 for an 8 GiB target.".into());
            }
        }
        if self.kind == Kind::New {
            let point = &self.questions[1].value;
            let polygon = &self.questions[2].value;
            if point.trim().is_empty() == polygon.trim().is_empty() {
                return Err("Enter either center coordinates or a GeoJSON boundary file.".into());
            }
        }
        if self.kind == Kind::Tiles && !matches!(self.questions[2].value.trim(), "auto" | "on") {
            return Err("Streaming mode must be auto or on.".into());
        }
        let command = match self.kind {
            Kind::New => "domain",
            Kind::Research => "research",
            Kind::Fit => "domain-fit",
            Kind::Tiles => "domain-tiles",
            Kind::Wrf | Kind::MetEm => "run",
            Kind::Resume => "resume",
            Kind::Prepared => "sim",
            Kind::Downscale => "downscale",
            Kind::Render => "render",
        };
        let mut args = vec![];
        if self.kind == Kind::Research {
            args.extend(["create".into(), self.research.ok_or("Choose a research configuration first.")?.into()]);
        }
        let mut created = None;
        let mut history = Vec::new();
        for field in &self.questions {
            let value = field.value.trim();
            if value.is_empty() {
                continue;
            }
            if field.flag == "@history" {
                history = history_files(value, cwd)?;
                continue;
            }
            if field.flag == "@extra" {
                let extra: Vec<String> = serde_json::from_str(value).map_err(|_| {
                    "Advanced arguments need a list of quoted strings, for example [\"--projection\",\"lambert\"]. Clear this field to continue without them.".to_owned()
                })?;
                if matches!(
                    self.kind,
                    Kind::New | Kind::Research | Kind::Fit | Kind::Tiles | Kind::Downscale
                ) && extra
                    .iter()
                    .any(|a| a == "--out" || a.starts_with("--out="))
                {
                    return Err("Set the configuration path in its named question, not additional arguments.".into());
                }
                args.extend(extra);
                continue;
            }
            if field.flag == "@advanced" {
                if !matches!(value, "off" | "on") { return Err("Advanced CLI options must be off or on.".into()); }
                continue;
            }
            if field.flag == "@location" {
                let coordinates = value.split(',').collect::<Vec<_>>();
                let point = coordinates.len() == 2
                    && coordinates.iter().all(|v| v.trim().parse::<f64>().is_ok());
                args.push(format!(
                    "{}={}",
                    if point { "--point" } else { "--polygon" },
                    value
                ));
                continue;
            }
            if field.flag == "@downscale-mode" {
                match value {
                    "plan" => args.push("--dry-run".into()),
                    "run" => {}
                    _ => return Err("Downscale action must be plan or run.".into()),
                }
                continue;
            }
            if field.flag == "@accept-parent-cadence" {
                match value {
                    "true" => args.push("--accept-parent-cadence".into()),
                    "false" => {}
                    _ => return Err("Accept parent cadence must be true, false, or blank.".into()),
                }
                continue;
            }
            if field.flag == "@auto-vram" {
                // A declared capacity takes precedence over the automatic
                // default, including requests assembled before Next is used.
                // An explicit child size or a supplied child configuration
                // does not: the engine prices either on the measured card
                // when measuring is on (the drawn-box case).
                if self.kind == Kind::Downscale && !self.questions[16].value.trim().is_empty() { continue; }
                match value {
                    "true" => args.push("--auto-vram".into()),
                    "false" => {},
                    _ => return Err("Measure local GPU memory must be true, false, or blank.".into()),
                }
                continue;
            }
            if field.flag == "--out" {
                let path = PathBuf::from(value);
                let path = if path.is_absolute() {
                    path
                } else {
                    cwd.join(path)
                };
                if path.exists() && self.kind != Kind::Render {
                    return Err(if self.kind == Kind::Downscale {
                        "That output directory already exists. Choose a new downscaled output directory."
                    } else {
                        "That configuration already exists. Choose a new filename, or open the existing file from Home."
                    }.into());
                }
                if !matches!(self.kind, Kind::Downscale | Kind::Render) {
                    created = Some(path.clone());
                }
                args.extend([field.flag.into(), path.to_string_lossy().into_owned()]);
            } else if field.flag.is_empty() {
                args.push(value.into())
            } else {
                args.push(format!("{}={}", field.flag, value))
            }
        }
        if matches!(self.kind, Kind::Fit | Kind::Tiles) {
            args.push("--write".into());
        }
        if self.kind == Kind::Downscale {
            let has = |flag: &str| {
                self.questions
                    .iter()
                    .any(|q| q.flag == flag && !q.value.trim().is_empty())
            };
            if has("--point") == has("--child-config") {
                return Err(
                    "Choose either child centre coordinates or an existing child configuration."
                        .into(),
                );
            }
            if args.iter().any(|a| a == "--auto-vram") && has("--vram-gib") {
                return Err("Measuring this GPU and declaring a capacity are two answers to one budget. Leave the capacity blank, or set Measure local GPU to false.".into());
            }
            if has("--parent-restart") && has("--parent-namelist") {
                return Err(
                    "Choose either an ArWen parent restart or a stock-WRF parent namelist.".into(),
                );
            }
            if has("--max-boundary-interval-seconds")
                && args.iter().any(|a| a == "--accept-parent-cadence")
            {
                return Err(
                    "Choose a maximum boundary interval or accept the archive cadence, not both."
                        .into(),
                );
            }
        }
        if self.kind == Kind::Prepared {
            let config = PathBuf::from(self.questions[1].value.trim());
            let config = if config.is_absolute() {
                config
            } else {
                cwd.join(config)
            };
            let plots = crate::plotsettings::load(&config)?;
            args.extend(["--render-products".into(), plots.spec]);
        }
        if self.kind == Kind::Render {
            args.push("--series".into());
            args.extend(["--engine".into(), "rust".into(), "--".into()]);
            args.extend(history);
            if cfg!(windows) && args.iter().map(|value| value.encode_utf16().count() * 2 + 3).sum::<usize>() > 24_000 {
                return Err("The history paths exceed the Windows command-size budget. Choose a smaller domain/date folder.".into());
            }
        }
        Ok(Request {
            command: command.into(),
            args,
            created,
            title: self.title(),
        })
    }
}
fn history_files(value: &str, cwd: &Path) -> Result<Vec<String>, String> {
    let path = PathBuf::from(value);
    let path = if path.is_absolute() { path } else { cwd.join(path) };
    if path.is_file() { return Ok(vec![path.to_string_lossy().into_owned()]); }
    if !path.is_dir() { return Err("Choose an existing forecast history file or folder.".into()); }
    let mut pending = vec![path];
    let mut files = Vec::new();
    let mut visited = 0;
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory).map_err(|e| format!("Cannot read history folder: {e}"))? {
            let entry = entry.map_err(|e| format!("Cannot read history entry: {e}"))?;
            visited += 1;
            if visited > 20_000 { return Err("History folder contains over 20,000 entries. Choose a smaller run or domain folder.".into()); }
            let kind = entry.file_type().map_err(|e| format!("Cannot inspect history entry: {e}"))?;
            if kind.is_symlink() { continue; }
            if kind.is_dir() { pending.push(entry.path()); }
            else if kind.is_file() && entry.file_name().to_string_lossy().starts_with("wrfout")
                && has_netcdf_header(&entry.path())? {
                files.push(entry.path().to_string_lossy().into_owned());
                if files.len() > 512 { return Err("Over 512 history files found. Choose a smaller domain/date folder for this render.".into()); }
            }
        }
    }
    if files.is_empty() { return Err("No wrfout files found. Choose the folder containing actual forecast history.".into()); }
    files.sort();
    Ok(files)
}
fn has_netcdf_header(path: &Path) -> Result<bool, String> {
    use std::io::Read;
    let mut bytes = [0u8; 8];
    let mut file = std::fs::File::open(path).map_err(|e| format!("Cannot inspect history {}: {e}", path.display()))?;
    let count = file.read(&mut bytes).map_err(|e| format!("Cannot read history header {}: {e}", path.display()))?;
    Ok((count >= 4 && &bytes[..3] == b"CDF" && matches!(bytes[3], 1 | 2 | 5))
        || (count == 8 && &bytes == b"\x89HDF\r\n\x1a\n"))
}

pub fn command_text(python: &Path, request: &Request) -> String {
    let mut all = vec![
        python.to_string_lossy().into_owned(),
        "-m".into(),
        "gpuwm.cli".into(),
        request.command.clone(),
    ];
    all.extend(request.args.clone());
    let quoted = all
        .iter()
        .map(|value| {
            format!(
                "'{}'",
                if cfg!(windows) {
                    value.replace('\'', "''")
                } else {
                    value.replace('\'', "'\"'\"'")
                }
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    if cfg!(windows) {
        format!("& {quoted}")
    } else {
        quoted
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn repeat_directory(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("arwen-repeat-{label}-{}-{}",
            std::process::id(), std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&root).unwrap();
        root
    }

    fn output_path(guide: &Guide) -> PathBuf {
        PathBuf::from(&guide.questions.iter().find(|question| question.flag == "--out").unwrap().value)
    }

    #[test]
    fn repeated_creation_defaults_keep_the_original_configuration_and_companions() {
        let root = repeat_directory("defaults");
        for (kind, stem) in [(Kind::New, "forecast"), (Kind::Research, "research"),
                             (Kind::Fit, "fitted"), (Kind::Tiles, "streaming")] {
            let first = output_path(&Guide::new(kind, &root, &root.join("runs")));
            assert_eq!(first, root.join(format!("{stem}.toml")));
            assert!(!first.exists());
            std::fs::write(&first, b"original configuration").unwrap();
            let companion = first.with_extension("namelist.wps");
            std::fs::write(&companion, b"original authority").unwrap();
            let second = output_path(&Guide::new(kind, &root, &root.join("runs")));
            assert_eq!(second, root.join(format!("{stem}-2.toml")));
            assert!(!second.exists());
            std::fs::write(&second, b"second configuration").unwrap();
            let third = output_path(&Guide::new(kind, &root, &root.join("runs")));
            assert_eq!(third, root.join(format!("{stem}-3.toml")));
            assert!(!third.exists());
            assert_eq!(std::fs::read(first).unwrap(), b"original configuration");
            assert_eq!(std::fs::read(companion).unwrap(), b"original authority");
            assert_eq!(std::fs::read(second).unwrap(), b"second configuration");
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn companion_only_collisions_also_reserve_the_configuration_stem() {
        let root = repeat_directory("companions");
        let extensions = ["namelist.wps", "d01-target.json", "namelist.input",
                          "stock.namelist.input", "fit.json", "toml.arwen-plots.json"];
        let mut retained = Vec::new();
        for (index, extension) in extensions.iter().enumerate() {
            let stem = if index == 0 { "forecast".to_owned() } else { format!("forecast-{}", index + 1) };
            let companion = root.join(format!("{stem}.{extension}"));
            std::fs::write(&companion, extension.as_bytes()).unwrap();
            retained.push((companion, extension.as_bytes()));
            let next = output_path(&Guide::new(Kind::New, &root, &root.join("runs")));
            assert_eq!(next, root.join(format!("forecast-{}.toml", index + 2)));
            assert!(!next.exists());
        }
        for (path, bytes) in retained { assert_eq!(std::fs::read(path).unwrap(), bytes); }
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), extensions.len());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn workflow_and_research_defaults_suffix_but_explicit_paths_stay_exact() {
        let root = repeat_directory("presets");
        let mode = crate::workflows::mode("general").unwrap();
        let existing = root.join("general.toml");
        std::fs::write(&existing, b"first workflow").unwrap();
        let mut guide = Guide::new(Kind::New, &root, &root.join("runs"));
        guide.apply_workflow(mode, &root);
        assert_eq!(output_path(&guide), root.join("general-2.toml"));
        guide.questions[1].value = "35.3,-97.5".into();
        guide.questions[7].value = "gfs".into();
        guide.questions[8].value = "2026-09-06T00".into();
        let explicit = root.join("User chosen path.toml");
        guide.questions[12].value = explicit.to_string_lossy().into_owned();
        assert_eq!(guide.request(&root).unwrap().created, Some(explicit.clone()));
        assert!(!explicit.exists());
        guide.questions[12].value = existing.to_string_lossy().into_owned();
        assert!(guide.request(&root).err().unwrap().contains("already exists"));
        assert_eq!(output_path(&guide), existing);
        assert_eq!(std::fs::read(existing).unwrap(), b"first workflow");
        static ROW: std::sync::LazyLock<serde_json::Value> = std::sync::LazyLock::new(||
            serde_json::json!({"id":"study", "title":"Study", "geometry":{"forecast_hours":6}}));
        std::fs::write(root.join("study.namelist.wps"), b"research authority").unwrap();
        let mut research = Guide::new(Kind::Research, &root, &root.join("runs"));
        research.apply_research(&ROW, &root);
        assert_eq!(output_path(&research), root.join("study-2.toml"));
        assert_eq!(std::fs::read(root.join("study.namelist.wps")).unwrap(), b"research authority");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn render_mode_reviews_real_history_paths_and_explicit_native_products() {
        let root = std::env::temp_dir().join(format!("arwen-render-guide-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let history = root.join("Operator's history");
        std::fs::create_dir_all(history.join("d02")).unwrap();
        for path in [history.join("wrfout_d01_first.nc"), history.join("d02/wrfout_d02_first.nc")] { std::fs::write(path, b"CDF\x01\0\0\0\0native renderer validates the remaining NetCDF").unwrap(); }
        std::fs::write(history.join("report.json"), b"{}").unwrap();
        std::fs::write(history.join("wrfout_d01_first.json"), b"{}").unwrap();
        std::fs::write(history.join("wrfout_d01_preview.png"), b"\x89PNG\r\n\x1a\n").unwrap();
        let mut guide = Guide::new(Kind::Render, &root, &root.join("runs"));
        guide.apply_workflow(crate::workflows::mode("hypothetical_tropical").unwrap(), &root);
        guide.questions[0].value = history.to_string_lossy().into_owned();
        let request = guide.request(&root).unwrap();
        assert_eq!(request.command, "render");
        assert!(request.created.is_none());
        assert!(request.args.iter().any(|value| value.starts_with("--products=mslp_10m_winds,")));
        assert!(request.args.iter().any(|value| value.starts_with("--source-label=HYPOTHETICAL")));
        assert!(request.args.iter().any(|value| value == "--series"));
        let delimiter = request.args.iter().position(|value| value == "--").unwrap();
        assert_eq!(request.args[delimiter-2..delimiter], ["--engine", "rust"]);
        assert_eq!(request.args[delimiter+1..].len(), 2);
        assert!(request.args[delimiter+1..].iter().all(|value| Path::new(value).is_file()));
        assert!(!root.join("weather-plots").exists());
        guide.questions[0].value = root.join("missing").to_string_lossy().into_owned();
        assert!(guide.request(&root).err().unwrap().contains("existing forecast"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn streaming_recovery_uses_the_engine_planner_and_an_explicit_new_file() {
        let directory =
            std::env::temp_dir().join(format!("arwen-tiles-guide-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("original.toml");
        std::fs::write(&source, "# untouched\na=1\n").unwrap();
        let mut guide = Guide::new(Kind::Tiles, &directory, &directory.join("run"));
        guide.questions[0].value = source.to_string_lossy().into_owned();
        assert_eq!(guide.questions.len(), 3);
        for mode in ["auto", "on"] {
            guide.questions[2].value = mode.into();
            let request = guide.request(&directory).unwrap();
            assert_eq!(request.command, "domain-tiles");
            assert!(request.args.contains(&format!("--mode={mode}")));
            assert!(request.args.contains(&"--write".to_owned()));
            assert!(!request.args.iter().any(|arg| arg.contains("tile_n")
                || arg.contains("--point")
                || arg.contains("--hours")));
            assert!(!request.created.unwrap().exists());
        }
        guide.questions[2].value = "off".into();
        assert!(guide
            .request(&directory)
            .err()
            .unwrap()
            .contains("auto or on"));
        guide.questions[2].value = "auto".into();
        guide.questions[1].value = source.to_string_lossy().into_owned();
        assert!(guide.request(&directory).is_err());
        assert_eq!(
            std::fs::read_to_string(source).unwrap(),
            "# untouched\na=1\n"
        );
    }

    #[test]
    fn fitting_uses_existing_template_authority_and_an_explicit_write() {
        let mut g = Guide::new(Kind::Fit, Path::new("."), Path::new("runs"));
        g.questions[0].value = "Operator's starter.toml".into();
        g.questions[1].value = "area.geojson".into();
        g.questions[3].value = "2026-09-05T00".into();
        g.questions[6].value = "new-fitted-test-nonexistent.toml".into();
        let r = g.request(Path::new(".")).unwrap();
        assert_eq!(r.command, "domain-fit");
        assert_eq!(r.args[0], "Operator's starter.toml");
        assert!(r.args.contains(&"--polygon=area.geojson".into()));
        assert!(r.args.contains(&"--write".into()));
        assert!(!r.args.iter().any(|a| a.starts_with("--physics")
            || a.starts_with("--root-dx")
            || a.starts_with("--chain")
            || a.starts_with("--vram")));
        assert!(r.created.is_some());
        for expected in [0, 1, 3, 4, 6] {
            assert_eq!(g.step, expected);
            g.advance();
        }
    }
    #[test]
    fn exact_arguments_and_unlisted_sources_are_retained() {
        let mut g = Guide::new(Kind::New, Path::new("."), Path::new("runs"));
        g.questions[1].value = "-33.87,151.21".into();
        g.questions[3].value = "48.125".into();
        g.questions[7].value = "future-source-alias".into();
        g.questions[8].value = "2026-09-05T00".into();
        g.questions[12].value = "new-guide-test-nonexistent.toml".into();
        let r = g.request(Path::new(".")).unwrap();
        assert!(r.args.contains(&"--point=-33.87,151.21".into()));
        assert!(r.args.contains(&"--root-dx=48.125".into()));
        assert!(r.args.contains(&"--source=future-source-alias".into()));
        assert!(!r.args.iter().any(|s| s.starts_with("--physics-profile")));
    }
    #[test]
    fn external_inputs_do_not_invent_physics_overrides() {
        for kind in [Kind::Wrf, Kind::MetEm] {
            let mut g = Guide::new(kind, Path::new("."), Path::new("runs"));
            g.questions[0].value = "input folder".into();
            let r = g.request(Path::new(".")).unwrap();
            assert_eq!(r.command, "run");
            assert!(!r
                .args
                .iter()
                .any(|s| s.contains("rrtmg") || s.contains("vertical-grid")));
        }
    }
    #[test]
    fn checkpoint_and_prepared_use_existing_engine_doors() {
        let mut g = Guide::new(Kind::Resume, Path::new("."), Path::new("prior-run"));
        g.questions[0].value = "same.toml".into();
        let r = g.request(Path::new(".")).unwrap();
        assert_eq!(r.command, "resume");
        assert!(r.args.contains(&"--from=latest".into()));
        assert!(r.args.contains(&"--outdir=prior-run".into()));
    }
    #[test]
    fn research_downscale_keeps_parent_cadence_independent_of_child_output() {
        let mut count = 0;
        for row in crate::research::rows("configurations").iter().filter(|row| row["method"] == "archived_downscale") {
            count += 1;
            let mut guide = Guide::new(Kind::Downscale, Path::new("."), Path::new("runs"));
            guide.apply_research(row, Path::new("."));
            guide.questions[0].value = "hourly-parent-history".into();
            guide.questions[1].value = "35.3,-97.5".into();
            guide.questions[3].value = "parent.gpuwmrst".into();
            assert!(guide.questions[11].value.is_empty());
            assert_eq!(guide.questions[12].value, "true");
            assert_eq!(guide.questions[14].value, row["geometry"]["history_interval_s"].as_u64().unwrap().to_string());
            let request = guide.request(Path::new(".")).unwrap();
            assert!(request.args.iter().any(|arg| arg == "--accept-parent-cadence"));
            assert!(!request.args.iter().any(|arg| arg.starts_with("--max-boundary-interval-seconds")));
        }
        assert_eq!(count, 23);
    }

    #[test]
    fn explicit_downscale_capacity_replaces_the_automatic_default() {
        let mut guide = Guide::new(Kind::Downscale, Path::new("."), Path::new("runs"));
        guide.apply_research(crate::research::config_by_id("convective-initiation.fine").unwrap(), Path::new("."));
        guide.questions[0].value = "parent history".into();
        guide.questions[1].value = "35.3,-97.5".into();
        assert!(guide.request(Path::new(".")).unwrap().args.iter().any(|arg| arg == "--auto-vram"));
        guide.questions[16].value = "8".into();
        let request = guide.request(Path::new(".")).unwrap();
        assert!(request.args.contains(&"--vram-gib=8".into()));
        assert!(!request.args.iter().any(|arg| arg == "--auto-vram"));
        guide.step = 16;
        guide.advance();
        assert_eq!(guide.questions[21].value, "false");
        assert_eq!(guide.questions[16].value, "8");
    }

    #[test]
    fn ordinary_guides_never_ask_for_raw_arguments_without_advanced_opt_in() {
        for kind in [Kind::New, Kind::Research, Kind::Fit, Kind::Wrf, Kind::MetEm,
                     Kind::Resume, Kind::Prepared, Kind::Downscale, Kind::Render] {
            let mut guide = Guide::new(kind, Path::new("."), Path::new("runs"));
            loop {
                assert!(!matches!(guide.questions[guide.step].flag, "@advanced" | "@extra"));
                if !guide.advance() { break; }
            }
            assert!(!guide.questions.iter().any(|q| q.flag == "@extra"));
            let index = guide.questions.iter().position(|q| q.flag == "@advanced").unwrap();
            guide.questions[index].value = "on".into();
            guide.step = index;
            guide.summary_edit = true;
            assert!(!guide.advance());
            assert!(guide.questions.iter().any(|q| q.flag == "@extra"));
        }
    }

    #[test]
    fn malformed_advanced_arguments_give_plain_recovery_and_can_be_disabled() {
        let mut guide = Guide::new(Kind::Wrf, Path::new("."), Path::new("runs"));
        guide.questions[0].value = "parent inputs".into();
        guide.questions.iter_mut().find(|q| q.flag == "@advanced").unwrap().value = "on".into();
        guide.sync_choices();
        guide.questions.iter_mut().find(|q| q.flag == "@extra").unwrap().value = "--projection lambert".into();
        let error = guide.request(Path::new(".")).err().unwrap();
        assert!(error.contains("list of quoted strings"));
        assert!(error.contains("Clear this field"));
        assert!(!error.contains("line 1 column"));
        guide.questions.iter_mut().find(|q| q.flag == "@advanced").unwrap().value = "off".into();
        guide.sync_choices();
        assert!(!guide.questions.iter().any(|q| q.flag == "@extra"));
        assert!(guide.request(Path::new(".")).is_ok());
    }

    #[test]
    fn ordinary_profile_tokens_and_required_location_have_field_level_recovery() {
        let mut guide = Guide::new(Kind::Research, Path::new("."), Path::new("runs"));
        guide.step = 4;
        for invalid in ["8 GiB", "8gb", "high"] {
            guide.questions[4].value = invalid.into();
            assert!(guide.validate_current().unwrap_err().contains("Type only the number"));
        }
        for valid in ["auto", "8", "12", "16", "24", "32"] {
            guide.questions[4].value = valid.into();
            assert!(guide.validate_current().is_ok());
        }
        let mut guide = Guide::new(Kind::New, Path::new("."), Path::new("runs"));
        assert!(guide.validate_current().is_err());
        guide.questions[2].value = "study.geojson".into();
        assert!(guide.validate_current().is_ok());
    }

    #[test]
    fn every_research_recipe_builds_exact_requests_accepted_by_the_native_parser() {
        use std::io::Write;
        use std::process::{Command, Stdio};
        let root = repeat_directory("all-research-argv");
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
        let python = std::env::var_os("GPUWM_TUI_TEST_PYTHON").unwrap_or_else(|| "python".into());
        // Actual NetCDF parent frames, restart physics evidence and a loadable
        // supplied configuration exercise the path/argument boundary. Native
        // preparation/rendering have their separate array-level controls.
        let fixture_script = r#"
import datetime, pathlib, sys
root=pathlib.Path(sys.argv[1]); sys.path.insert(0,str(pathlib.Path.cwd()/'tests'))
from test_offline_child import _history
from test_downscale_cli import _restart_evidence, _SURFACE_PARENT_CONFIG, _give_the_parent_a_real_projection, _add_parent_surface
from test_case_data import make_case_toml
parent=root/'parent history'; parent.mkdir()
for hour in range(7):
    path=parent/f'wrfout_d01_1974-04-03_{12+hour:02d}_00_00'
    _history(path,datetime.datetime(1974,4,3,12)+datetime.timedelta(hours=hour),ny=18,nx=20)
    _give_the_parent_a_real_projection(path,ny=18,nx=20)
    _add_parent_surface(path,ny=18,nx=20)
_restart_evidence(root/'parent physics.npz',dict(_SURFACE_PARENT_CONFIG,nx=20,ny=18,nz=2,grid_id=1))
supplied=root/'supplied scenario'; supplied.mkdir(); make_case_toml(supplied)
"#;
        let fixture = Command::new(&python).args(["-c", fixture_script]).arg(&root).current_dir(&repository).output().unwrap();
        assert!(fixture.status.success(), "{}", String::from_utf8_lossy(&fixture.stderr));
        let history = root.join("parent history");
        let restart = root.join("parent physics.npz");
        let supplied = root.join("supplied scenario/case.toml");
        let mut records = Vec::new();
        let mut supplied_count = 0;
        for row in crate::research::rows("configurations") {
            let id = crate::research::text(row, "id");
            let mut render = Guide::new(Kind::Render, &root, &root.join("runs"));
            render.apply_research(row, &root);
            render.questions[0].value = history.to_string_lossy().into_owned();
            render.questions[3].value = root.join(format!("{id}-plots")).to_string_lossy().into_owned();
            let request = render.request(&root).unwrap();
            assert!(request.created.is_none());
            records.push(serde_json::json!({"row":row,"command":request.command,"args":request.args}));
            let kind = match crate::research::setup_route(row) {
                crate::workflows::Route::New => Kind::Research,
                crate::workflows::Route::Downscale => Kind::Downscale,
                crate::workflows::Route::Open => {
                    supplied_count += 1;
                    assert!(row["requires_existing_state"].as_bool().unwrap());
                    records.push(serde_json::json!({"row":row,"command":"supplied","path":supplied}));
                    continue;
                }
                other => panic!("unexpected setup route {other:?}"),
            };
            let mut guide = Guide::new(kind, &root, &root.join("runs"));
            guide.apply_research(row, &root);
            if kind == Kind::Research {
                for (index, value) in [(0,"39.5,-84.0"),(1,"gfs"),(2,"2026-09-05T18"),(4,"8"),(5,"8")] {
                    guide.questions[index].value = value.into();
                }
            } else {
                guide.questions[0].value = history.to_string_lossy().into_owned();
                guide.questions[1].value = "39.5,-84.0".into();
                guide.questions[3].value = restart.to_string_lossy().into_owned();
                guide.questions[16].value = "8".into();
                guide.sync_choices();
            }
            let request = guide.request(&root).unwrap();
            assert_eq!(request.created.is_some(), kind == Kind::Research);
            records.push(serde_json::json!({"row":row,"command":request.command,"args":request.args}));
        }
        assert_eq!(supplied_count, 3);
        let parser_script = r#"
import json,pathlib,sys
from gpuwm.cli import build_parser
from gpuwm.case_data import load_experiment_case
from gpuwm.research_workspaces import create_workspace
parser=build_parser(); records=json.load(sys.stdin); counts={}
for item in records:
    row=item['row']; command=item['command']; counts[command]=counts.get(command,0)+1
    if command=='supplied':
        exp,data=load_experiment_case(pathlib.Path(item['path']))
        assert exp.domains and data is not None
        args=parser.parse_args(['research','create',row['id'],'--point=39.5,-84.0','--cycle=2026-09-05T18','--vram-gib=8','--out='+item['path']])
        try:create_workspace(args)
        except ValueError as error:assert 'existing scenario state' in str(error),str(error)
        else:raise AssertionError('supplied scenario was replaced with a fresh source')
        continue
    args=parser.parse_args([command,*item['args']])
    if command=='render':
        assert args.series and args.engine=='rust' and args.timeidx=='all'
        assert args.products.split(',')==row['diagnostics']
        assert all(pathlib.Path(path).is_file() for path in args.wrfout)
    elif command=='research':
        assert args.configuration_id==row['id'] and args.source=='gfs'
        assert args.cycle=='2026-09-05T18' and args.point=='39.5,-84.0'
        assert args.hardware_class=='8' and args.vram_gib==8
        assert args.hours==row['geometry']['forecast_hours']
        assert not args.out.exists()
    elif command=='downscale':
        assert args.dry_run and args.vram_gib==8 and not args.auto_vram
        assert pathlib.Path(args.parent_restart).is_file() and all(pathlib.Path(path).is_dir() for path in args.parent)
        assert args.ratio==row['geometry']['nest_ratios'][0]
        assert args.hours==row['geometry']['forecast_hours']
        assert args.output_interval_seconds==row['geometry']['history_interval_s']
        assert args.accept_parent_cadence and args.max_boundary_interval_seconds is None
        assert not args.out.exists()
assert counts=={'render':114,'research':88,'downscale':23,'supplied':3},counts
print(json.dumps(counts))
"#;
        let mut child = Command::new(&python).args(["-c", parser_script]).current_dir(&repository)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        child.stdin.take().unwrap().write_all(&serde_json::to_vec(&records).unwrap()).unwrap();
        let result = child.wait_with_output().unwrap();
        assert!(result.status.success(), "Native parser rejected a research request:\n{}\n{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
        assert!(!root.join("runs").exists());
    }

    #[test]
    fn historical_downscale_preserves_paths_and_reviews_plan_or_run_explicitly() {
        let mut guide = Guide::new(Kind::Downscale, Path::new("."), Path::new("runs"));
        guide.questions[0].value = "Operator's history/parent files".into();
        guide.questions[1].value = "35.3,-97.5".into();
        guide.questions[3].value = "Operator's evidence/parent.gpuwmrst".into();
        guide.questions[10].value = "child surface.nc".into();
        guide.questions[11].value = "900".into();
        guide.questions[18].value = "new child's output".into();
        let request = guide.request(Path::new(".")).unwrap();
        assert_eq!(request.command, "downscale");
        assert_eq!(request.args[0], "Operator's history/parent files");
        assert!(request
            .args
            .contains(&"--parent-restart=Operator's evidence/parent.gpuwmrst".into()));
        assert!(request
            .args
            .contains(&"--child-surface-from=child surface.nc".into()));
        assert!(request.args.contains(&"--dry-run".into()));
        assert!(request.created.is_none());
        guide.questions[19].value = "run".into();
        assert!(!guide
            .request(Path::new("."))
            .unwrap()
            .args
            .contains(&"--dry-run".into()));
        guide.questions[12].value = "true".into();
        assert!(guide
            .request(Path::new("."))
            .err()
            .unwrap()
            .contains("not both"));
    }
}
