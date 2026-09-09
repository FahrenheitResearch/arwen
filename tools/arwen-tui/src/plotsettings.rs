//! Per-configuration plot requests. The engine still owns product availability.
use super::{
    button_bar, clickable_list, display_path, key_hit, safe, Hit, HitRegion, INK, MUTED, TEAL,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::Line,
    widgets::{Paragraph, Wrap},
    Frame,
};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::{mpsc, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

const SCHEMA: &str = "gpuwm-tui-plots-v1";
const MAX_FILE: u64 = 64 * 1024;
const PRESETS: &str = include_str!("../../../gpuwm/data/tui/plot-presets.json");
const PRESET_NOTICE: &str =
    "Presets request plots only. Missing history fields or times are reported by the renderer.";
const PRODUCT_NOTICE: &str =
    "Type to search. Click a row or press Space to toggle it. Review lists every selected product.";
const REVIEW_NOTICE: &str =
    "Review the request, then Save plots. This does not start rendering or a forecast.";
const TEXT_NOTICE: &str =
    "Edit comma-separated selectors. Use Ctrl+U to clear, then Review before saving.";

#[derive(Clone, Debug)]
pub struct Preset {
    pub id: String,
    pub label: String,
    pub description: String,
    pub products: Vec<String>,
}

pub fn presets() -> &'static [Preset] {
    static DATA: OnceLock<Vec<Preset>> = OnceLock::new();
    DATA.get_or_init(|| {
        let document: serde_json::Value =
            serde_json::from_str(PRESETS).expect("bundled plot presets");
        document["presets"]
            .as_array()
            .expect("preset list")
            .iter()
            .map(|row| Preset {
                id: row["id"].as_str().unwrap().into(),
                label: row["label"].as_str().unwrap().into(),
                description: row["description"].as_str().unwrap().into(),
                products: row["products"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_str().unwrap().into())
                    .collect(),
            })
            .collect()
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub label: String,
    pub spec: String,
}

impl Default for Selection {
    fn default() -> Self {
        Self::preset(0)
    }
}

impl Selection {
    pub fn preset_id(id: &str) -> Result<Self, String> {
        presets().iter().position(|preset| preset.id == id)
            .map(Self::preset)
            .ok_or_else(|| format!("Unknown plot preset: {id}"))
    }
    pub fn preset(index: usize) -> Self {
        let preset = &presets()[index];
        Self {
            label: preset.label.clone(),
            spec: preset.products.join(","),
        }
    }
    pub fn summary(&self) -> String {
        match self.spec.as_str() {
            "all" => "All available plots".into(),
            "none" => "No plots".into(),
            _ => format!("{} · {} selected", self.label, self.spec.split(',').count()),
        }
    }
}

pub fn sidecar(config: &Path) -> PathBuf {
    let mut name = config.as_os_str().to_owned();
    name.push(".arwen-plots.json");
    name.into()
}

fn read(path: &Path) -> Result<Option<Vec<u8>>, String> {
    match fs::metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!(
            "Cannot read plot settings {}: {error}",
            display_path(path)
        )),
        Ok(metadata) if metadata.len() > MAX_FILE => {
            Err("Plot settings exceed 64 KiB. Open the sidecar to correct it.".into())
        }
        Ok(_) => fs::read(path).map(Some).map_err(|error| error.to_string()),
    }
}

pub fn normalize(spec: &str) -> Result<String, String> {
    if spec.len() > 32 * 1024 {
        return Err("The product list exceeds 32 KiB.".into());
    }
    if spec.chars().any(char::is_control) {
        return Err("Product selectors cannot contain control characters.".into());
    }
    let tokens: Vec<_> = spec.split(',').map(str::trim).collect();
    if tokens.iter().any(|value| value.is_empty()) {
        return Err(
            "Choose at least one product, or choose None. Empty list entries are not allowed."
                .into(),
        );
    }
    if tokens.len() > 1 && tokens.iter().any(|value| matches!(*value, "all" | "none")) {
        return Err(
            "All and None must stand alone. Choose individual products to customize the list."
                .into(),
        );
    }
    // Preserve order and explicit selectors. The live engine validates their meaning.
    Ok(tokens.join(","))
}

fn decode(bytes: &[u8]) -> Result<Selection, String> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("Invalid plot settings JSON: {error}"))?;
    if value["schema"] != SCHEMA {
        return Err("Unknown plot settings schema. Review Plots before launching.".into());
    }
    let spec = normalize(
        value["products"]
            .as_str()
            .ok_or("Plot settings need a products string.")?,
    )?;
    Ok(Selection {
        label: value["label"].as_str().unwrap_or("Custom").into(),
        spec,
    })
}

pub fn load(config: &Path) -> Result<Selection, String> {
    read(&sidecar(config))?
        .map(|bytes| decode(&bytes))
        .unwrap_or_else(|| Ok(Selection::default()))
}

pub fn save(config: &Path, selection: &Selection, original: Option<&[u8]>) -> Result<(), String> {
    let path = sidecar(config);
    if read(&path)?.as_deref() != original {
        return Err(
            "Plot settings changed outside this dialog. Cancel and reopen Plots before saving."
                .into(),
        );
    }
    let spec = normalize(&selection.spec)?;
    let bytes = serde_json::to_vec_pretty(&serde_json::json!({
        "schema": SCHEMA, "label": selection.label, "products": spec,
    }))
    .map_err(|error| error.to_string())?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp = path.with_file_name(format!(".arwen-plots-{}-{stamp}.tmp", std::process::id()));
    let result = (|| {
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        output.write_all(&bytes)?;
        output.write_all(b"\n")?;
        output.sync_all()?;
        drop(output);
        fs::rename(&temp, &path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result.map_err(|error| {
        format!(
            "Could not save plot settings {}: {error}",
            display_path(&path)
        )
    })
}

/// Carry an explicitly saved choice when the editor exports a NEW configuration.
/// An existing destination choice remains its authority.
pub fn inherit(source: &Path, destination: &Path) -> Result<(), String> {
    if source == destination || read(&sidecar(destination))?.is_some() {
        return Ok(());
    }
    if let Some(bytes) = read(&sidecar(source))? {
        save(destination, &decode(&bytes)?, None)?;
    }
    Ok(())
}

#[derive(Default)]
pub struct Catalog {
    python: Option<PathBuf>,
    receiver: Option<mpsc::Receiver<Result<Vec<String>, String>>>,
    pub products: Vec<String>,
    pub error: Option<String>,
    pub loading: bool,
}

impl Catalog {
    #[cfg(test)]
    pub fn fixture(python: &Path, products: Vec<String>) -> Self {
        Self {
            python: Some(python.to_owned()),
            products,
            ..Self::default()
        }
    }
    pub fn request(&mut self, python: &Path, cwd: &Path) {
        if self.python.as_deref() == Some(python) && (self.loading || !self.products.is_empty()) {
            return;
        }
        let python = python.to_owned();
        let cwd = cwd.to_owned();
        self.python = Some(python.clone());
        self.products.clear();
        self.error = None;
        self.loading = true;
        let (sender, receiver) = mpsc::channel();
        self.receiver = Some(receiver);
        std::thread::spawn(move || {
            let _ = sender.send(query_catalog(&python, &cwd));
        });
    }
    pub fn poll(&mut self) {
        let Some(receiver) = &self.receiver else {
            return;
        };
        match receiver.try_recv() {
            Ok(result) => {
                self.loading = false;
                self.receiver = None;
                match result {
                    Ok(products) => {
                        self.products = products;
                        self.error = None;
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                self.loading = false;
                self.receiver = None;
                self.error =
                    Some("Catalog query ended without a result. Reopen Plots to retry.".into());
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "arwen-plot-settings-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("Operator's forecast.toml");
        fs::write(&path, "# complete science stays here\na=1\n").unwrap();
        path
    }

    #[test]
    fn explicit_plot_choices_persist_without_mutating_science_and_exports_inherit() {
        let config = config();
        let original = fs::read(&config).unwrap();
        assert_eq!(load(&config).unwrap().spec.split(',').count(), 25);
        assert!(!sidecar(&config).exists());
        let selection = Selection {
            label: "Custom".into(),
            spec: "var:SNOWH,2m_temperature,total_qpf".into(),
        };
        save(&config, &selection, None).unwrap();
        assert_eq!(load(&config).unwrap(), selection);
        assert_eq!(fs::read(&config).unwrap(), original);
        let export = config.with_file_name("other.toml");
        fs::write(&export, &original).unwrap();
        inherit(&config, &export).unwrap();
        assert_eq!(load(&export).unwrap(), selection);
        let previous = fs::read(sidecar(&export)).unwrap();
        let all = Selection {
            label: "All".into(),
            spec: "all".into(),
        };
        save(&export, &all, Some(&previous)).unwrap();
        inherit(&config, &export).unwrap();
        assert_eq!(load(&export).unwrap(), all);
        assert!(save(&export, &selection, Some(&previous))
            .unwrap_err()
            .contains("changed outside"));
        fs::write(sidecar(&export), "{invalid-json").unwrap();
        assert!(load(&export).is_err()); // Corruption must not silently run General.
    }

    #[test]
    fn catalog_failure_never_becomes_a_different_or_partial_product_menu() {
        assert!(
            parse_catalog(&serde_json::json!({"products":null,"error":"Stage rw_wrfbatch"}))
                .unwrap_err()
                .contains("Stage")
        );
        assert!(parse_catalog(&serde_json::json!({"products":[{"name":"2m_temperature"}],"parse_warning":"count mismatch"})).is_err());
        assert!(parse_catalog(&serde_json::json!({"products":[{}]})).is_err());
        assert_eq!(
            parse_catalog(
                &serde_json::json!({"products":[{"name":"total_qpf"},{"name":"2m_temperature"}]})
            )
            .unwrap(),
            ["2m_temperature", "total_qpf"]
        );
    }

    #[test]
    fn custom_picker_preserves_explicit_selectors_and_searches_human_labels() {
        let path = config();
        let mut form = Form::new(&path);
        form.selection = Selection {
            label: "Custom".into(),
            spec: "var:SNOWH,total_qpf".into(),
        };
        let catalog = Catalog::fixture(
            Path::new("python"),
            vec![
                "total_qpf".into(),
                "2m_temperature".into(),
                "mslp_10m_winds".into(),
            ],
        );
        form.customize(&catalog);
        form.query = "sea-level".into();
        assert_eq!(form.product_rows(&catalog), ["mslp_10m_winds"]);
        form.choose(0, &catalog);
        assert_eq!(form.selection.spec, "var:SNOWH,total_qpf,mslp_10m_winds");
        form.key(KeyEvent::new(KeyCode::F(4), KeyModifiers::NONE), &catalog);
        assert_eq!(form.mode, Mode::Review);
        assert!(!sidecar(&path).exists());
        assert!(matches!(
            form.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE), &catalog),
            Intent::Keep
        ));
        assert!(matches!(
            form.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE), &catalog),
            Intent::Cancel
        ));
        assert!(!sidecar(&path).exists());
        assert!(normalize("all,total_qpf").is_err());
        assert!(normalize("total_qpf,,2m_temperature").is_err());
        assert_eq!(
            normalize(" var:SNOWH, total_qpf ").unwrap(),
            "var:SNOWH,total_qpf"
        );
    }
}

fn query_catalog(python: &Path, cwd: &Path) -> Result<Vec<String>, String> {
    let mut command = Command::new(python);
    command
        .args(["-P", "-B", "-m", "gpuwm.tui_products", "--catalog"])
        .current_dir(cwd)
        .env("GPUWM_NO_LOCAL_GPU", "1");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    }
    let output = command
        .output()
        .map_err(|error| format!("Cannot load the installed plot catalog: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "Plot catalog failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("Invalid catalog response: {error}"))?;
    parse_catalog(&value)
}

fn parse_catalog(value: &serde_json::Value) -> Result<Vec<String>, String> {
    if let Some(warning) = value["parse_warning"].as_str() {
        return Err(warning.into());
    }
    if let Some(error) = value["error"].as_str() {
        return Err(error.into());
    }
    let rows = value["products"]
        .as_array()
        .ok_or("The installed renderer returned no product catalog. Check the installation.")?;
    let mut names = rows
        .iter()
        .map(|row| {
            row["name"]
                .as_str()
                .map(str::to_owned)
                .ok_or("Catalog entry has no product name.".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if names.is_empty() {
        return Err("The installed renderer returned an empty product catalog.".into());
    }
    names.sort();
    names.dedup();
    Ok(names)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Presets,
    Products,
    Text,
    Review,
}

pub struct Form {
    pub mode: Mode,
    pub selection: Selection,
    pub selected: usize,
    pub query: String,
    pub notice: String,
    pub original: Option<Vec<u8>>,
}

pub enum Intent {
    Keep,
    Cancel,
    Save,
}

impl Form {
    /// The Nodes UI can edit a node profile's choice without a local TOML.
    /// Intent::Save returns the selection; the caller owns its persistence.
    pub fn from_selection(selection: Selection) -> Self {
        Self {
            mode: Mode::Presets,
            selection,
            selected: 0,
            query: String::new(),
            notice: PRESET_NOTICE.into(),
            original: None,
        }
    }
    pub fn new(config: &Path) -> Self {
        let loaded = read(&sidecar(config));
        let original = loaded.as_ref().ok().and_then(|value| value.clone());
        let selection = loaded.and_then(|bytes| {
            bytes
                .map(|data| decode(&data))
                .unwrap_or_else(|| Ok(Selection::default()))
        });
        let notice = selection
            .as_ref()
            .err()
            .cloned()
            .unwrap_or_else(|| PRESET_NOTICE.into());
        let mut form = Self::from_selection(selection.unwrap_or(Selection {
            label: "Custom".into(),
            spec: String::new(),
        }));
        form.notice = notice;
        form.original = original;
        form
    }
    pub fn has_error(&self) -> bool {
        ![PRESET_NOTICE, PRODUCT_NOTICE, REVIEW_NOTICE, TEXT_NOTICE].contains(&self.notice.as_str())
    }
    pub fn tokens(&self) -> Vec<String> {
        self.selection
            .spec
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .collect()
    }
    pub fn product_rows(&self, catalog: &Catalog) -> Vec<String> {
        let mut names = catalog.products.clone();
        names.extend(
            self.tokens()
                .into_iter()
                .filter(|name| !matches!(name.as_str(), "all" | "none")),
        );
        names.extend(
            presets()
                .iter()
                .flat_map(|preset| preset.products.iter())
                .filter(|name| name.starts_with("var:"))
                .cloned(),
        );
        names.sort();
        names.dedup();
        let query = self.query.to_lowercase();
        names.retain(|name| {
            name.to_lowercase().contains(&query)
                || product_label(name).to_lowercase().contains(&query)
        });
        names
    }
    pub fn customize(&mut self, catalog: &Catalog) {
        if self.selection.spec == "all" {
            if catalog.products.is_empty() {
                self.notice =
                    "Load the installed catalog before expanding All into individual selections."
                        .into();
                return;
            }
            self.selection.spec = catalog.products.join(",");
        } else if self.selection.spec == "none" {
            self.selection.spec.clear();
        }
        self.mode = Mode::Products;
        self.selected = 0;
        self.query.clear();
        self.notice = PRODUCT_NOTICE.into();
    }
    pub fn choose(&mut self, index: usize, catalog: &Catalog) {
        match self.mode {
            Mode::Presets => {
                self.selected = index;
                self.selection = if index < presets().len() {
                    Selection::preset(index)
                } else if index == presets().len() {
                    Selection {
                        label: "All".into(),
                        spec: "all".into(),
                    }
                } else {
                    Selection {
                        label: "None".into(),
                        spec: "none".into(),
                    }
                };
                self.mode = Mode::Review;
                self.selected = 0;
                self.notice = REVIEW_NOTICE.into();
            }
            Mode::Products => {
                let rows = self.product_rows(catalog);
                if let Some(product) = rows.get(index) {
                    self.selected = index;
                    let mut selected = self.tokens();
                    if selected.contains(product) {
                        selected.retain(|name| name != product);
                    } else {
                        selected.push(product.clone());
                    }
                    self.selection = Selection {
                        label: "Custom".into(),
                        spec: selected.join(","),
                    };
                }
            }
            Mode::Review => self.selected = index,
            Mode::Text => {}
        }
    }
    pub fn key(&mut self, key: KeyEvent, catalog: &Catalog) -> Intent {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if key.code == KeyCode::Esc {
            if self.mode == Mode::Presets {
                return Intent::Cancel;
            }
            self.mode = Mode::Presets;
            self.selected = 0;
            self.notice = PRESET_NOTICE.into();
            return Intent::Keep;
        }
        if key.code == KeyCode::F(2) {
            self.mode = Mode::Presets;
            self.selected = 0;
            self.notice = PRESET_NOTICE.into();
            return Intent::Keep;
        }
        if key.code == KeyCode::F(3) {
            self.mode = Mode::Text;
            self.notice = TEXT_NOTICE.into();
            return Intent::Keep;
        }
        if key.code == KeyCode::F(4) || (ctrl && key.code == KeyCode::Enter) {
            match normalize(&self.selection.spec) {
                Ok(spec) => {
                    self.selection.spec = spec;
                    self.mode = Mode::Review;
                    self.selected = 0;
                    self.notice = REVIEW_NOTICE.into();
                }
                Err(error) => self.notice = error,
            }
            return Intent::Keep;
        }
        if key.code == KeyCode::F(5) {
            self.customize(catalog);
            return Intent::Keep;
        }
        if self.mode == Mode::Text {
            match key.code {
                KeyCode::Char('u' | 'U') if ctrl => self.selection.spec.clear(),
                KeyCode::Backspace => {
                    self.selection.spec.pop();
                }
                KeyCode::Char(c) if !ctrl && !c.is_control() => {
                    self.selection.spec.push(c);
                    self.selection.label = "Custom".into();
                }
                KeyCode::Enter => match normalize(&self.selection.spec) {
                    Ok(spec) => {
                        self.selection.spec = spec;
                        self.selection.label = "Custom".into();
                        self.mode = Mode::Review;
                        self.selected = 0;
                        self.notice = REVIEW_NOTICE.into();
                    }
                    Err(error) => self.notice = error,
                },
                _ => {}
            }
            return Intent::Keep;
        }
        let count = match self.mode {
            Mode::Presets => presets().len() + 2,
            Mode::Products => self.product_rows(catalog).len(),
            Mode::Review => self.tokens().len(),
            Mode::Text => 0,
        };
        match key.code {
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => self.selected = (self.selected + 1).min(count.saturating_sub(1)),
            KeyCode::PageUp => self.selected = self.selected.saturating_sub(5),
            KeyCode::PageDown => self.selected = (self.selected + 5).min(count.saturating_sub(1)),
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = count.saturating_sub(1),
            KeyCode::Enter if self.mode == Mode::Review => return Intent::Save,
            KeyCode::Enter => self.choose(self.selected, catalog),
            KeyCode::Char(' ') if self.mode == Mode::Products => {
                self.choose(self.selected, catalog)
            }
            KeyCode::Char('u' | 'U') if ctrl && self.mode == Mode::Products => {
                self.query.clear();
                self.selected = 0;
            }
            KeyCode::Backspace if self.mode == Mode::Products => {
                self.query.pop();
                self.selected = 0;
            }
            KeyCode::Char(c) if self.mode == Mode::Products && !ctrl && !c.is_control() => {
                self.query.push(c);
                self.selected = 0;
            }
            _ => {}
        }
        Intent::Keep
    }
    pub fn paste(&mut self, text: &str) {
        if self.mode == Mode::Text {
            self.selection
                .spec
                .push_str(&text.replace(['\r', '\n', '\t'], ""));
            self.selection.label = "Custom".into();
        } else if self.mode == Mode::Products {
            self.query
                .push_str(&safe(text).replace(['\r', '\n', '\t'], ""));
            self.selected = 0;
        }
    }
}

pub fn product_label(name: &str) -> String {
    match name {
        "mslp_10m_winds" => "Sea-level pressure and 10 m winds".into(),
        "total_qpf" | "qpf_total" => "Total precipitation".into(),
        "qpf_1h" => "1-hour precipitation".into(),
        "qpf_6h" => "6-hour precipitation".into(),
        "qpf_12h" => "12-hour precipitation".into(),
        "qpf_24h" => "24-hour precipitation".into(),
        "sbcape" => "Surface-based CAPE".into(),
        "mlcape" => "Mixed-layer CAPE".into(),
        "mucape" => "Most-unstable CAPE".into(),
        "sbcin" => "Surface-based convective inhibition".into(),
        "mlcin" => "Mixed-layer convective inhibition".into(),
        "sblcl" => "Surface-based cloud-base height (LCL)".into(),
        "dcape" => "Downdraft CAPE".into(),
        "srh_0_1km" => "0-1 km storm-relative helicity".into(),
        "srh_0_3km" => "0-3 km storm-relative helicity".into(),
        "uh_2to5km" => "2-5 km updraft helicity".into(),
        "uh_2to5km_run_max" => "Run-maximum 2-5 km updraft helicity".into(),
        "stp_fixed" => "Significant tornado parameter (fixed layer)".into(),
        "ehi_0_1km" => "0-1 km energy-helicity index".into(),
        "ehi_0_3km" => "0-3 km energy-helicity index".into(),
        "bulk_shear_0_1km" => "0-1 km bulk wind shear".into(),
        "bulk_shear_0_6km" => "0-6 km bulk wind shear".into(),
        "var:SNOWH" => "Snow depth (stored SNOWH)".into(),
        "var:SNOW" => "Snow water equivalent (stored SNOW)".into(),
        "all" => "All available plots".into(),
        "none" => "No plots".into(),
        _ => {
            let text = name.replace('_', " ");
            let mut chars = text.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => text,
            }
        }
    }
}

pub fn draw(
    frame: &mut Frame,
    hits: &mut Vec<HitRegion>,
    form: &Form,
    catalog: &Catalog,
    body: Rect,
    buttons: Rect,
) {
    let header_rows = if matches!(form.mode, Mode::Products | Mode::Review) {
        3
    } else {
        2
    };
    let parts = Layout::vertical([Constraint::Length(header_rows), Constraint::Min(2)]).split(body);
    let title = format!(
        "{}{}",
        form.selection.summary(),
        if form.mode == Mode::Presets {
            " (current request)"
        } else {
            ""
        }
    );
    let heading = if form.mode == Mode::Products {
        format!(
            "{} selected · {} catalog products\nSearch: {}_",
            form.tokens().len(),
            catalog.products.len(),
            safe(&form.query)
        )
    } else if form.mode == Mode::Review {
        format!("{title}\nGo: saved frames; local F8: first frame only")
    } else {
        title
    };
    frame.render_widget(
        Paragraph::new(heading)
            .style(Style::default().fg(TEAL))
            .wrap(Wrap { trim: false }),
        parts[0],
    );
    match form.mode {
        Mode::Presets => {
            let mut rows = presets()
                .iter()
                .map(|preset| format!("{} ({} plots)", preset.label, preset.products.len()))
                .collect::<Vec<_>>();
            rows.extend(["All available plots".into(), "None - skip plots".into()]);
            clickable_list(
                frame,
                hits,
                parts[1],
                rows.into_iter()
                    .enumerate()
                    .map(|(index, label)| (vec![Line::raw(label)], Hit::PlotItem(index)))
                    .collect(),
                Some(form.selected),
            );
            button_bar(
                frame,
                hits,
                buttons,
                &[
                    ("Choose", key_hit(KeyCode::Enter)),
                    ("Customize", key_hit(KeyCode::F(5))),
                    ("Edit list", key_hit(KeyCode::F(3))),
                    ("Review", key_hit(KeyCode::F(4))),
                    ("Cancel", key_hit(KeyCode::Esc)),
                ],
            );
        }
        Mode::Products => {
            let selected = form.tokens();
            let rows = form.product_rows(catalog);
            if rows.is_empty() {
                let message = if catalog.loading {
                    "Loading the installed renderer catalog..."
                } else if catalog.error.is_some() {
                    "Catalog unavailable. Read the message below; saved/custom selectors remain intact."
                } else {
                    "No products match. Clear the search to see the catalog."
                };
                frame.render_widget(Paragraph::new(message).wrap(Wrap { trim: false }), parts[1]);
            } else {
                clickable_list(
                    frame,
                    hits,
                    parts[1],
                    rows.into_iter()
                        .enumerate()
                        .map(|(index, name)| {
                            let label = format!(
                                "[{}] {}",
                                if selected.contains(&name) { "x" } else { " " },
                                product_label(&name)
                            );
                            (vec![Line::raw(label)], Hit::PlotItem(index))
                        })
                        .collect(),
                    Some(form.selected),
                );
            }
            button_bar(
                frame,
                hits,
                buttons,
                &[
                    ("Toggle", key_hit(KeyCode::Char(' '))),
                    ("Review", key_hit(KeyCode::F(4))),
                    ("Presets", key_hit(KeyCode::F(2))),
                    ("Edit list", key_hit(KeyCode::F(3))),
                    (
                        "Clear search",
                        Hit::Key(KeyCode::Char('u'), KeyModifiers::CONTROL),
                    ),
                ],
            );
        }
        Mode::Text => {
            let mut rows =
                super::log_display_rows(&safe(&form.selection.spec), parts[1].width as usize);
            let capacity = parts[1].height as usize;
            if rows.len() >= capacity && capacity > 0 {
                rows = rows[rows.len() - capacity..].to_vec();
            }
            rows.push("_".into());
            frame.render_widget(
                Paragraph::new(rows.into_iter().map(Line::raw).collect::<Vec<_>>())
                    .style(Style::default().fg(INK)),
                parts[1],
            );
            button_bar(
                frame,
                hits,
                buttons,
                &[
                    ("Review", key_hit(KeyCode::Enter)),
                    ("Picker", key_hit(KeyCode::F(5))),
                    ("Clear", Hit::Key(KeyCode::Char('u'), KeyModifiers::CONTROL)),
                    ("Back", key_hit(KeyCode::Esc)),
                ],
            );
        }
        Mode::Review => {
            let rows = form
                .tokens()
                .into_iter()
                .enumerate()
                .map(|(index, token)| {
                    (
                        vec![
                            Line::raw(format!("{}. {}", index + 1, product_label(&token))),
                            Line::styled(format!("   {token}"), Style::default().fg(MUTED)),
                        ],
                        Hit::PlotItem(index),
                    )
                })
                .collect();
            clickable_list(frame, hits, parts[1], rows, Some(form.selected));
            button_bar(
                frame,
                hits,
                buttons,
                &[
                    ("Save plots", key_hit(KeyCode::Enter)),
                    ("Customize", key_hit(KeyCode::F(5))),
                    ("Edit list", key_hit(KeyCode::F(3))),
                    ("Back", key_hit(KeyCode::Esc)),
                ],
            );
        }
    }
}
