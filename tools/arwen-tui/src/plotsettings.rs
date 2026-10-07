//! Per-configuration plot requests. The engine still owns product availability.
use super::{
    button_bar, clickable_list, display_path, key_hit, safe, Hit, HitRegion, AMBER, INK, MUTED,
    TEAL,
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
    collections::BTreeMap,
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
/// How many unserved products a preset row names before it counts the rest.
const NAMED_UNSERVED: usize = 4;
/// How many wrapped lines one recorded reason gets in Review.
const REASON_LINES: usize = 3;

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
            _ => format!("{} · {} selected", self.label, product_terms(&self.spec).len()),
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

/// The products of one product list, each section term whole.
///
/// The renderer's own rule (`split_product_spec` in
/// tools/rustwx/crates/rw-wrfbatch/src/section.rs), which
/// `gpuwm.rustwx.product_spec_terms` mirrors: a section's level list is
/// comma-separated too (`xsec:wa=1,2,5@5`), so a token that follows an
/// `xsec:` term whose last term opened a level list, and that is a level,
/// continues that list instead of naming a product. The continuation may
/// carry the term that closes the list (`0.1/wa` in
/// `xsec:QCLOUD=0.01,0.1/wa`). Split on every comma, the picker listed
/// `0.1/wa` as a product of its own.
pub fn product_terms(spec: &str) -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();
    for token in spec.split(',').map(str::trim).filter(|token| !token.is_empty()) {
        let continues = terms.last().is_some_and(|prior| {
            prior.starts_with(SECTION_PREFIX)
                && level_list_open(prior)
                && continues_level_list(token)
        });
        match terms.last_mut() {
            Some(prior) if continues => {
                prior.push(',');
                prior.push_str(token);
            }
            _ => terms.push(token.to_owned()),
        }
    }
    terms
}

const SECTION_PREFIX: &str = "xsec:";

/// True when the term's last `/` part carries an `=` level list that a
/// following level may continue.
fn level_list_open(term: &str) -> bool {
    let last = term.rsplit('/').next().unwrap_or(term);
    last.contains('=') && !last.rsplit('=').next().unwrap_or("").contains('@')
}

/// A level (`-10`, `0.5`, `10@5`), or the list's last level followed by
/// the term that closes it (`0.1/wa`, `10@5/tk=-20`).
fn continues_level_list(token: &str) -> bool {
    match token.split_once('/') {
        Some((level, rest)) => is_level(level) && !rest.trim().is_empty(),
        None => is_level(token),
    }
}

/// What the renderer parses as a level, with an optional highlight.
fn is_level(token: &str) -> bool {
    let (level, highlight) = match token.split_once('@') {
        Some((level, highlight)) => (level, Some(highlight)),
        None => (token, None),
    };
    let numeric = |text: &str| !text.is_empty() && text.trim().parse::<f32>().is_ok();
    numeric(level) && highlight.map_or(true, numeric)
}

pub fn normalize(spec: &str) -> Result<String, String> {
    if spec.len() > 32 * 1024 {
        return Err("The product list exceeds 32 KiB.".into());
    }
    if spec.chars().any(char::is_control) {
        return Err("Product selectors cannot contain control characters.".into());
    }
    if spec.split(',').map(str::trim).any(str::is_empty) {
        return Err(
            "Choose at least one product, or choose None. Empty list entries are not allowed."
                .into(),
        );
    }
    let tokens = product_terms(spec);
    if tokens.len() > 1 && tokens.iter().any(|value| matches!(value.as_str(), "all" | "none")) {
        return Err(
            "All and None must stand alone. Choose individual products to customize the list."
                .into(),
        );
    }
    // Preserve order and explicit selectors. The live engine validates their meaning.
    Ok(tokens.join(","))
}

pub fn for_section(spec: &str, has_section: bool) -> Result<String, String> {
    let spec = normalize(spec)?;
    let sections = product_terms(&spec).into_iter()
        .filter(|term| term.starts_with(SECTION_PREFIX)).collect::<Vec<_>>();
    if !has_section && !sections.is_empty() {
        return Err(format!("Cross-section plots {} need a latitude,longitude,latitude,longitude line; without it the renderer cannot locate the slice. Use a forecast door that carries a section line, or remove these plots from this request.", sections.join(", ")));
    }
    Ok(spec)
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

/// One reply from the catalog query.
///
/// The product menu can fail -- it takes a resolvable renderer -- while
/// the packaged lane record cannot: it is a file in the install. They
/// travel together and are kept separately for that reason, so a reader
/// whose renderer is missing still gets the statement about what this
/// install will not draw instead of an empty dialog.
struct Answer {
    products: Result<Vec<String>, String>,
    unavailable: BTreeMap<String, String>,
    basis: Option<String>,
}

#[derive(Default)]
pub struct Catalog {
    python: Option<PathBuf>,
    receiver: Option<mpsc::Receiver<Answer>>,
    pub products: Vec<String>,
    /// Per product, the recorded reason this install will not draw it,
    /// as `gpuwm.tui_products.preset_availability` states it. Merged
    /// across presets because the record answers per product, not per
    /// preset, and a custom request is not a preset at all.
    pub unavailable: BTreeMap<String, String>,
    /// What that statement was decided from, in the catalog's own words.
    pub availability_basis: Option<String>,
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
    #[cfg(test)]
    pub fn with_availability(
        python: &Path,
        products: Vec<String>,
        unavailable: &[(&str, &str)],
        basis: &str,
    ) -> Self {
        Self {
            unavailable: unavailable
                .iter()
                .map(|(name, reason)| ((*name).to_owned(), (*reason).to_owned()))
                .collect(),
            availability_basis: Some(basis.to_owned()),
            ..Self::fixture(python, products)
        }
    }
    /// Which of `products` this install is recorded as unable to draw.
    pub fn unserved(&self, products: &[String]) -> Vec<String> {
        products
            .iter()
            .filter(|name| self.unavailable.contains_key(*name))
            .cloned()
            .collect()
    }
    /// The recorded reason for one product, or nothing said about it.
    /// Absence is not a verdict: the record states reasons, never a
    /// roster, so a product it does not carry simply runs.
    pub fn reason(&self, product: &str) -> Option<&str> {
        self.unavailable.get(product).map(String::as_str)
    }
    pub fn request(&mut self, python: &Path, cwd: &Path) {
        if self.python.as_deref() == Some(python) && (self.loading || !self.products.is_empty()) {
            return;
        }
        let python = python.to_owned();
        let cwd = cwd.to_owned();
        self.python = Some(python.clone());
        self.products.clear();
        self.unavailable.clear();
        self.availability_basis = None;
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
            Ok(answer) => {
                self.loading = false;
                self.receiver = None;
                // The record is kept whichever way the menu went: what
                // this install cannot draw does not become unknown
                // because the renderer could not be asked.
                self.unavailable = answer.unavailable;
                self.availability_basis = answer.basis;
                match answer.products {
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
        // Parallel tests read one clock tick on the 2.8.6 windows-2025 runner and collided on this name; the counter keeps each call distinct.
        static NEXT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "arwen-plot-settings-{}-{stamp}-{}",
            std::process::id(),
            NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
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
        // 22: simulated_ir_satellite left the general preset when it
        // gained its reason, and 10m_wind_gusts, precipitation_type and
        // cloud_cover left it when every run was measured drawing 20 of
        // 24 -- no wrfout carries their fields -- with cloud_cover_levels,
        // which a wrfout does carry, in place of the total
        // (gpuwm/data/tui/plot-presets.json).
        assert_eq!(load(&config).unwrap().spec.split(',').count(), 22);
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
    fn the_picker_states_which_chosen_products_this_install_will_not_draw() {
        // The catalog document has carried preset_availability since
        // the availability authority landed, and nothing a reader sees
        // carried it: the picker read the product names out of the same
        // document and dropped the statement beside them.
        //
        // No shipped preset names a product this lane cannot draw any
        // more (gpuwm/data/tui/plot-presets.json), so the statement is
        // proven on a record that names two of the general preset's own
        // products: the mechanism, not the shipped record, is under test.
        let layers =
            "Recorded as unavailable for this fixture: the layer panel is not a total cloud fraction.";
        let water = "Recorded as unavailable for this fixture.";
        let catalog = Catalog::with_availability(
            Path::new("python"),
            vec!["total_qpf".into()],
            &[("cloud_cover_levels", layers), ("precipitable_water", water)],
            "the packaged lane record, plus the renderer's own requirement rows",
        );
        let rows = preset_rows(&catalog);
        assert_eq!(rows[0].len(), 2, "the preset row states what is unserved");
        assert!(
            rows[0][1].contains("cloud_cover_levels") && rows[0][1].contains("precipitable_water"),
            "{}",
            rows[0][1]
        );
        assert!(
            preset_rows(&Catalog::fixture(Path::new("python"), vec![]))
                .iter()
                .all(|lines| lines.len() == 1),
            "nothing recorded, nothing stated"
        );

        let form = Form::from_selection(Selection::preset(0));
        let review = review_rows(&form, &catalog, 200);
        let cloud = review
            .iter()
            .find(|lines| lines[1].trim() == "cloud_cover_levels")
            .expect("the general preset requests it");
        assert!(cloud[2].contains("not drawn by this install"), "{cloud:?}");
        assert!(
            cloud.iter().skip(2).any(|line| line.contains("total cloud fraction")),
            "{cloud:?}"
        );
        let qpf = review
            .iter()
            .find(|lines| lines[1].trim() == "total_qpf")
            .expect("the general preset requests it");
        assert_eq!(qpf.len(), 2, "a product with no recorded reason is not judged");

        // A renderer that cannot be asked costs the menu, not the
        // record: the statement is a packaged file either way.
        let answer = parse_answer(&serde_json::json!({
            "error": "rw_wrfbatch is not built",
            "preset_availability": {"general": {"cloud_cover_levels": layers}},
            "preset_availability_basis": "the packaged lane record only",
        }));
        assert!(answer.products.is_err());
        assert_eq!(answer.unavailable["cloud_cover_levels"], layers);
        assert_eq!(answer.basis.as_deref(), Some("the packaged lane record only"));
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

    #[test]
    fn a_section_term_keeps_its_level_list_as_one_picker_row() {
        // The renderer's own rule (rw-wrfbatch section.rs
        // split_product_spec, mirrored by gpuwm.rustwx.product_spec_terms):
        // `0.1/wa` closes the level list of `xsec:QCLOUD=0.01,0.1/wa` and is
        // not a product. Split on every comma, the picker listed it as a row.
        let spec = "composite_reflectivity,xsec:QCLOUD=0.01,0.1/wa";
        let mut form = Form::new(&config());
        let mut rows = |spec: &str| {
            form.selection = Selection {
                label: "Custom".into(),
                spec: spec.into(),
            };
            (form.tokens(), form.selection.summary())
        };
        assert_eq!(
            rows(spec),
            (
                vec!["composite_reflectivity".to_owned(), "xsec:QCLOUD=0.01,0.1/wa".to_owned()],
                "Custom · 2 selected".to_owned()
            )
        );
        assert_eq!(
            rows("xsec:wa=1, 2 ,5@5/tk=-20,total_qpf,xsec:tk/wa").0,
            ["xsec:wa=1,2,5@5/tk=-20", "total_qpf", "xsec:tk/wa"]
        );
        // A level list closed by its `@` highlight takes no more levels, and
        // a number after a store product is that store product's neighbour.
        assert_eq!(rows("xsec:wa=1@5,2").0, ["xsec:wa=1@5", "2"]);
        assert_eq!(rows("total_qpf,5").0, ["total_qpf", "5"]);
        assert_eq!(normalize(&format!(" {spec} ")).unwrap(), spec);
        assert_eq!(
            normalize("xsec:wa=1, 2 ,5@5,total_qpf").unwrap(),
            "xsec:wa=1,2,5@5,total_qpf"
        );
        assert!(normalize("xsec:QCLOUD=0.01,0.1/wa,all").is_err());
        assert!(normalize("xsec:QCLOUD=0.01,,0.1/wa").is_err());
    }

    #[test]
    fn plots_without_a_section_line_hide_and_refuse_cross_sections() {
        let spec = "xsec:wa=1,2/temperature";
        let catalog = Catalog::fixture(Path::new("python"), vec!["total_qpf".into(), spec.into()]);
        let mut form = Form::from_selection(Selection { label: "All".into(), spec: "all".into() });
        assert!(!form.product_rows(&catalog).iter().any(|name| name.starts_with("xsec:")));
        form.customize(&catalog);
        assert_eq!(form.selection.spec, "total_qpf");
        form.selection.spec = spec.into();
        form.mode = Mode::Review;
        let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        assert!(matches!(form.key(enter, &catalog), Intent::Keep));
        assert!(form.notice.contains(spec) && form.notice.contains("cannot locate the slice"));
        form.has_section = true;
        assert!(form.product_rows(&catalog).contains(&spec.to_owned()));
        assert!(matches!(form.key(enter, &catalog), Intent::Save));
        assert_eq!(form.selection.spec, spec);
    }
}

fn query_catalog(python: &Path, cwd: &Path) -> Answer {
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
    let output = match command.output() {
        Ok(output) => output,
        Err(error) => {
            return Answer::failed(format!("Cannot load the installed plot catalog: {error}"))
        }
    };
    if !output.status.success() {
        return Answer::failed(format!(
            "Plot catalog failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    match serde_json::from_slice(&output.stdout) {
        Ok(value) => parse_answer(&value),
        Err(error) => Answer::failed(format!("Invalid catalog response: {error}")),
    }
}

impl Answer {
    fn failed(error: String) -> Self {
        Self {
            products: Err(error),
            unavailable: BTreeMap::new(),
            basis: None,
        }
    }
}

/// The catalog document as the picker reads it: the menu, and what the
/// install says it will not draw. One document, so the dialog and the
/// research recipe door cannot disagree about one product.
fn parse_answer(value: &serde_json::Value) -> Answer {
    let mut unavailable = BTreeMap::new();
    if let Some(presets) = value["preset_availability"].as_object() {
        for rows in presets.values() {
            for (product, reason) in rows.as_object().into_iter().flatten() {
                if let Some(reason) = reason.as_str() {
                    unavailable.insert(product.clone(), safe(reason));
                }
            }
        }
    }
    Answer {
        products: parse_catalog(value),
        unavailable,
        basis: value["preset_availability_basis"].as_str().map(safe),
    }
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
    pub has_section: bool,
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
            has_section: false,
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
        product_terms(&self.selection.spec)
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
            (self.has_section || !name.starts_with(SECTION_PREFIX))
                && (name.to_lowercase().contains(&query)
                    || product_label(name).to_lowercase().contains(&query))
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
            self.selection.spec = catalog.products.iter()
                .filter(|name| self.has_section || !name.starts_with(SECTION_PREFIX))
                .cloned().collect::<Vec<_>>().join(",");
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
            match for_section(&self.selection.spec, self.has_section) {
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
                KeyCode::Enter => match for_section(&self.selection.spec, self.has_section) {
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
            KeyCode::Enter if self.mode == Mode::Review => {
                match for_section(&self.selection.spec, self.has_section) {
                    Ok(spec) => { self.selection.spec = spec; return Intent::Save; }
                    Err(error) => self.notice = error,
                }
            }
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

/// The preset list as the picker draws it, as text.
///
/// Line one is the preset; where this install is recorded as unable to
/// draw some of its products, line two names them. The rows are built
/// here rather than inside the drawing so the statement can be read in
/// a test without a terminal.
pub fn preset_rows(catalog: &Catalog) -> Vec<Vec<String>> {
    presets()
        .iter()
        .map(|preset| {
            let mut lines = vec![format!("{} ({} plots)", preset.label, preset.products.len())];
            let unserved = catalog.unserved(&preset.products);
            if !unserved.is_empty() {
                let named = unserved
                    .iter()
                    .take(NAMED_UNSERVED)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ");
                let rest = unserved.len().saturating_sub(NAMED_UNSERVED);
                lines.push(format!(
                    "   {} not drawn by this install: {named}{}. Review names each reason.",
                    unserved.len(),
                    if rest > 0 {
                        format!(", and {rest} more")
                    } else {
                        String::new()
                    }
                ));
            }
            lines
        })
        .collect()
}

/// The review list as the picker draws it, as text: the product, the
/// selector, and the recorded reason this install will not draw it.
/// A request is never narrowed to what this box happens to serve; the
/// reader chooses with the reason in front of them.
pub fn review_rows(form: &Form, catalog: &Catalog, width: usize) -> Vec<Vec<String>> {
    form.tokens()
        .into_iter()
        .enumerate()
        .map(|(index, token)| {
            let mut lines = vec![
                format!("{}. {}", index + 1, product_label(&token)),
                format!("   {token}"),
            ];
            if let Some(reason) = catalog.reason(&token) {
                let text = format!("not drawn by this install: {reason}");
                let wrapped = super::log_display_rows(&text, width.saturating_sub(3).max(24));
                for (row, line) in wrapped.into_iter().enumerate() {
                    if row == REASON_LINES {
                        lines.push("   ... (reason continues)".into());
                        break;
                    }
                    lines.push(format!("   {line}"));
                }
            }
            lines
        })
        .collect()
}

/// One line of styling law for both lists: the product, then its
/// selector, then the recorded reason, in that order.
fn styled(lines: Vec<String>) -> Vec<Line<'static>> {
    lines
        .into_iter()
        .enumerate()
        .map(|(index, line)| match index {
            0 => Line::raw(line),
            1 => Line::styled(line, Style::default().fg(MUTED)),
            _ => Line::styled(line, Style::default().fg(AMBER)),
        })
        .collect()
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
            let mut rows = preset_rows(catalog);
            let stated = rows.iter().any(|lines| lines.len() > 1);
            rows.push(vec!["All available plots".into()]);
            rows.push(vec!["None - skip plots".into()]);
            // The statement earns its basis line only where it says
            // something, and only where the terminal has room for both;
            // an install that draws every listed product gets the plain
            // list it always had.
            let basis = if stated {
                catalog.availability_basis.as_deref()
            } else {
                None
            };
            let (list, footer) = match basis {
                Some(_) if parts[1].height >= 5 => {
                    let split = Layout::vertical([
                        Constraint::Min(3),
                        Constraint::Length(2),
                    ])
                    .split(parts[1]);
                    (split[0], Some(split[1]))
                }
                _ => (parts[1], None),
            };
            clickable_list(
                frame,
                hits,
                list,
                rows.into_iter()
                    .enumerate()
                    .map(|(index, lines)| (styled(lines), Hit::PlotItem(index)))
                    .collect(),
                Some(form.selected),
            );
            if let (Some(area), Some(basis)) = (footer, basis) {
                frame.render_widget(
                    Paragraph::new(format!("Availability basis: {basis}"))
                        .style(Style::default().fg(MUTED))
                        .wrap(Wrap { trim: false }),
                    area,
                );
            }
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
            let rows = review_rows(form, catalog, parts[1].width as usize)
                .into_iter()
                .enumerate()
                .map(|(index, lines)| (styled(lines), Hit::PlotItem(index)))
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
