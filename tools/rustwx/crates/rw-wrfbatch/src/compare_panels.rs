//! N-panel comparison sheets: any number of model panels, each under its own
//! label, beside any number of references, in the column order the caller
//! names, one header, one colour table, one projection and extent, one PNG.
//!
//! ```text
//! rw_compare --store-root DIR --out-dir DIR [--reference NAME[,NAME...]]
//!            --panel "LABEL=WRFOUT" --panel "LABEL=pmm:M0,M1,..." ...
//!            [--row --panel ... --panel ...]... [--order TOKEN,TOKEN,...]
//!            [--sheet-name STEM] [every other rw_compare option]
//! ```
//!
//! A panel SOURCE is one history file, or an ensemble reduction over member
//! files: `mean:`, `pmm:` (probability-matched mean) or `max:` followed by a
//! comma list whose items may carry `*` and `?` in the file name.  Every file
//! goes through the same hardened wrfout import into its own store and the
//! product's plane is read exactly as the two-input sheet reads its run, so a
//! one-file panel is the same picture `rw_compare` draws for that file.  The
//! reductions are taken on the stored planes, in stored units, before the
//! sheet style's conversion, by `rustwx_ensemble`.
//!
//! `--order` lists the row's columns: a 1-based panel number or a reference
//! name per column.  Every panel and every chosen reference appears exactly
//! once; a column order that drops one is refused rather than drawn short.
//! Without `--order` the panels come first, then the references in
//! `--reference` order.
//!
//! `--row` starts a new row: the rows are stacked under one header (several
//! valid times, say), each row's references are resolved at that row's own
//! valid time, and every row holds the same number of columns.
//!
//! Two panel sources are not history files, and both draw composite
//! reflectivity only (`rw_wrfbatch::nowcast_frames`):
//!
//! * `frames:ROOT@VALID` -- the frame valid at VALID of a
//!   `gpuwm-obs.nowcast-frames.v1` root (a nowcast, or MRMS frames), mapped
//!   onto the row's grid through latitude and longitude and member-reduced
//!   the way the heating window adapter reduces it;
//! * `ttenref:WINDOW` -- the column maximum of a heating window
//!   (`gpuwm-obs.radar-tten-ref.v1`, a window directory or its `ref.json`),
//!   so the heating a run was given can be seen; its valid time is the
//!   window's end.
//!
//! Their grid is the row's: a row that also draws a history file uses that
//! file's grid, and a row of these panels alone uses the `--panel-grid
//! WRFOUT` file's.  A window must already be on that grid (same shape), and
//! a panel's valid time must be the row's.

use super::*;
use rustwx_ensemble::{MemberStack, NanPolicy, PmmTieRule, ensemble_mean, probability_matched_mean};

/// What a model panel draws from its member files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PanelOp {
    /// One history file's plane.
    Field,
    Mean,
    Pmm,
    Max,
}

impl PanelOp {
    fn parse(token: &str) -> Option<Self> {
        match token {
            "mean" => Some(Self::Mean),
            "pmm" => Some(Self::Pmm),
            "max" => Some(Self::Max),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Field => "field",
            Self::Mean => "mean",
            Self::Pmm => "pmm",
            Self::Max => "max",
        }
    }

    fn describe(self, members: usize) -> String {
        match self {
            Self::Field => String::new(),
            Self::Mean => format!("ensemble mean of {members} members"),
            Self::Pmm => format!("probability-matched mean of {members} members"),
            Self::Max => format!("ensemble maximum of {members} members"),
        }
    }
}

/// A panel drawn from something that is not a history file.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum ExternalPanel {
    /// `frames:ROOT@VALID`.
    Frames { root: PathBuf, valid: DateTime<Utc> },
    /// `ttenref:WINDOW`.
    Window { path: PathBuf },
}

impl ExternalPanel {
    fn name(&self) -> &'static str {
        match self {
            Self::Frames { .. } => "frames",
            Self::Window { .. } => "ttenref",
        }
    }
}

/// One `--panel LABEL=SOURCE`.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PanelArg {
    pub label: String,
    pub op: PanelOp,
    pub files: Vec<PathBuf>,
    /// A frames or heating-window panel; `files` is empty then.
    pub external: Option<ExternalPanel>,
}

/// `LABEL=FILE`, or `LABEL=OP:FILE,FILE,...` with OP one of mean, pmm, max.
pub(super) fn parse_panel(text: &str) -> Result<PanelArg, String> {
    let (label, source) = text
        .split_once('=')
        .ok_or_else(|| format!("--panel {text:?} is not LABEL=SOURCE"))?;
    let label = label.trim();
    if label.is_empty() {
        return Err(format!("--panel {text:?} has no label"));
    }
    let source = source.trim();
    if let Some(rest) = source.strip_prefix("frames:") {
        let (root, valid) = rest.rsplit_once('@').ok_or_else(|| {
            format!("--panel {label:?}: frames:ROOT@VALID names a root and the frame's valid time")
        })?;
        let valid = rw_wrfbatch::nowcast_frames::parse_utc(valid)
            .map_err(|error| format!("--panel {label:?}: {error}"))?;
        let root = PathBuf::from(root.trim());
        if !root.join("nowcast.json").is_file() {
            return Err(format!("--panel {label:?}: {} holds no nowcast.json", root.display()));
        }
        return Ok(PanelArg {
            label: label.to_string(),
            op: PanelOp::Field,
            files: Vec::new(),
            external: Some(ExternalPanel::Frames { root, valid }),
        });
    }
    if let Some(rest) = source.strip_prefix("ttenref:") {
        let path = PathBuf::from(rest.trim());
        let receipt = if path.is_dir() { path.join("ref.json") } else { path.clone() };
        if !receipt.is_file() {
            return Err(format!("--panel {label:?}: {} is not a heating window", path.display()));
        }
        return Ok(PanelArg {
            label: label.to_string(),
            op: PanelOp::Field,
            files: Vec::new(),
            external: Some(ExternalPanel::Window { path }),
        });
    }
    let (op, list) = match source.split_once(':') {
        Some((head, rest)) => match PanelOp::parse(head) {
            Some(op) => (op, rest),
            // `C:/...`: a drive letter, not an operation.
            None => (PanelOp::Field, source),
        },
        None => (PanelOp::Field, source),
    };
    let files = rw_wrfbatch::da_sheet::expand_file_list(list)?;
    match (op, files.len()) {
        (PanelOp::Field, 1) => {}
        (PanelOp::Field, count) => {
            return Err(format!(
                "--panel {label:?} names {count} files; a field panel draws one file (use \
                 mean:, pmm: or max: to reduce several)"
            ));
        }
        (_, count) if count < 2 => {
            return Err(format!(
                "--panel {label:?}: {} over {count} member file(s) is not an ensemble reduction",
                op.name()
            ));
        }
        _ => {}
    }
    Ok(PanelArg {
        label: label.to_string(),
        op,
        files,
        external: None,
    })
}

/// A column of a row: model panel `index` or the reference named.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Column {
    Panel(usize),
    Reference(String),
}

/// The row's columns from `--order`, checked: every panel and every chosen
/// reference exactly once.  References the observation manifest adds beside
/// the chosen ones (`extras`) may be named; those not named go last.
pub(super) fn column_order(
    order: Option<&[String]>,
    panels: usize,
    chosen: &[String],
    extras: &[String],
) -> Result<Vec<Column>, String> {
    let Some(order) = order else {
        let mut columns: Vec<Column> = (0..panels).map(Column::Panel).collect();
        columns.extend(chosen.iter().chain(extras).cloned().map(Column::Reference));
        return Ok(columns);
    };
    let mut columns = Vec::with_capacity(order.len());
    for token in order {
        let column = match token.parse::<usize>() {
            Ok(number) if (1..=panels).contains(&number) => Column::Panel(number - 1),
            Ok(number) => {
                return Err(format!(
                    "--order names panel {number}; the row has panels 1 to {panels}"
                ));
            }
            Err(_) if chosen.iter().chain(extras).any(|name| name == token) => {
                Column::Reference(token.clone())
            }
            Err(_) => {
                return Err(format!(
                    "--order names {token:?}, which is neither a panel number nor a chosen \
                     reference ({})",
                    chosen.join(", ")
                ));
            }
        };
        if columns.contains(&column) {
            return Err(format!("--order names {token:?} twice"));
        }
        columns.push(column);
    }
    let missing: Vec<String> = (0..panels)
        .filter(|index| !columns.contains(&Column::Panel(*index)))
        .map(|index| (index + 1).to_string())
        .chain(
            chosen
                .iter()
                .filter(|name| !columns.contains(&Column::Reference((*name).clone())))
                .cloned(),
        )
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "--order leaves out {}; every panel and every reference is drawn exactly once",
            missing.join(", ")
        ));
    }
    for extra in extras {
        if !columns.contains(&Column::Reference(extra.clone())) {
            columns.push(Column::Reference(extra.clone()));
        }
    }
    Ok(columns)
}

/// One member file, imported into its own store.
struct ImportedMember {
    file: PathBuf,
    store_root: PathBuf,
    summary: rw_wrfbatch::wrf_process::WrfProcessSummary,
    slot: u16,
    lead_seconds: u64,
    valid_unix: i64,
    by_valid: BTreeMap<i64, u16>,
}

fn import_member(
    args: &Args,
    specs: &[ProductSpec],
    index: usize,
    file: &Path,
    wants_accumulation: bool,
) -> Result<ImportedMember, String> {
    let mut inputs = vec![file.to_path_buf()];
    let sibling = if wants_accumulation { sibling_an_hour_earlier(file) } else { None };
    for path in args.context.iter().chain(sibling.iter()) {
        if !inputs.iter().any(|have| same_file(have, path)) {
            inputs.push(path.clone());
        }
    }
    let store_root = args.store_root.join("panel-stores").join(format!("m{index:04}"));
    let summary = import(inputs, &store_root, import_options(specs))?;
    for note in &summary.notes {
        eprintln!("IMPORT_NOTE\t{}\t{note}", file.display());
    }
    let mut by_valid = BTreeMap::new();
    let mut found = None;
    for (slot, source_file) in &summary.frame_sources {
        let source = StoreFieldSource::open(&store_root, &summary.model, &summary.run, *slot)
            .map_err(|error| format!("open the frame of {}: {error}", source_file.display()))?;
        let (lead, valid) = frame_time(source.exact_time(), &summary.run, *slot).ok_or_else(|| {
            format!(
                "{} was stored with no valid time this binary can read; a panel with no valid \
                 time cannot share a sheet",
                source_file.display()
            )
        })?;
        by_valid.insert(valid, *slot);
        if same_file(source_file, file) {
            found = Some((*slot, lead, valid));
        }
    }
    let (slot, lead_seconds, valid_unix) =
        found.ok_or_else(|| format!("{}: the import stored no frame for it", file.display()))?;
    Ok(ImportedMember {
        file: file.to_path_buf(),
        store_root,
        summary,
        slot,
        lead_seconds,
        valid_unix,
        by_valid,
    })
}

/// Two coordinate sets that are one grid: same size, every cell within
/// about 11 m.
fn same_grid(lat_a: &[f32], lon_a: &[f32], lat_b: &[f32], lon_b: &[f32]) -> bool {
    lat_a.len() == lat_b.len()
        && lon_a.len() == lon_b.len()
        && lat_a.iter().zip(lat_b).all(|(a, b)| (a - b).abs() <= 1.0e-4)
        && lon_a.iter().zip(lon_b).all(|(a, b)| (a - b).abs() <= 1.0e-4)
}

/// The finite maximum over members at each point, NaN where none is finite.
fn member_max(stack: &MemberStack) -> Vec<f64> {
    (0..stack.points())
        .map(|point| {
            (0..stack.len())
                .map(|member| stack.member(member)[point])
                .filter(|value| value.is_finite())
                .fold(f64::NAN, |best, value| if best.is_nan() || value > best { value } else { best })
        })
        .collect()
}

/// A model panel's plane, ready for the sheet style.
struct ModelPanel {
    label: String,
    op: PanelOp,
    files: Vec<PathBuf>,
    values: Vec<f32>,
    times: SheetTimes,
    /// What a frames or window panel shows, for its subtitle.
    detail: Option<String>,
    /// A frames or window panel's provenance, for the sheet receipt.
    source: Option<serde_json::Value>,
}

/// The planes of one row, read and reduced, with the row's grid.
struct RowPlanes {
    lat: Vec<f32>,
    lon: Vec<f32>,
    projection: Option<GridProjection>,
    ny: usize,
    nx: usize,
    model_id: ModelId,
    first_field: RunField,
    panels: Vec<ModelPanel>,
    domain_token: String,
    domain_label: Option<String>,
}

/// The sheet times of one panel: its own start and lead, the row's
/// reference cycle (fixed by the row's first panel) and the reference lead.
fn panel_times(
    args: &Args,
    has_forecast: bool,
    cycle: &mut Option<DateTime<Utc>>,
    lead_seconds: u64,
    valid_unix: i64,
) -> Result<Result<SheetTimes, String>, String> {
    let run_init = utc(valid_unix - lead_seconds as i64)?;
    let row_cycle = match *cycle {
        Some(cycle) => cycle,
        None => {
            let chosen = match args.cycle.as_ref().filter(|_| has_forecast) {
                Some((date, hour)) => DateTime::parse_from_rfc3339(&format!(
                    "{}-{}-{}T{hour:02}:00:00Z",
                    &date[..4],
                    &date[4..6],
                    &date[6..8]
                ))
                .map_err(|error| format!("--cycle: {error}"))?
                .with_timezone(&Utc),
                None => run_init,
            };
            *cycle = Some(chosen);
            chosen
        }
    };
    let reference_lead_seconds = valid_unix - row_cycle.timestamp();
    if has_forecast && (reference_lead_seconds < 0 || reference_lead_seconds % 3_600 != 0) {
        return Ok(Err(format!(
            "valid {} is not a whole number of hours after cycle {}; the reference publishes \
             hourly forecasts",
            utc(valid_unix)?.to_rfc3339(),
            row_cycle.to_rfc3339()
        )));
    }
    let reference_lead = u16::try_from(if has_forecast { reference_lead_seconds / 3_600 } else { 0 })
        .map_err(|_| format!("lead {reference_lead_seconds} s is out of range"))?;
    Ok(Ok(SheetTimes {
        run_init,
        run_lead_seconds: lead_seconds,
        reference_cycle: row_cycle,
        reference_lead,
        valid: utc(valid_unix)?,
    }))
}

/// One frames or window panel read onto the row grid.
struct ExternalRead {
    values: Vec<f32>,
    start: DateTime<Utc>,
    valid: DateTime<Utc>,
    detail: String,
    record: serde_json::Value,
}

/// A frames or window panel's plane on the row grid, with its start, its
/// valid time, its subtitle detail and its provenance.
fn external_plane(
    external: &ExternalPanel,
    label: &str,
    lat: &[f32],
    lon: &[f32],
    ny: usize,
    nx: usize,
) -> Result<ExternalRead, String> {
    use rw_wrfbatch::nowcast_frames::{frame_on_grid, read_frames_receipt, window_column_max};
    match external {
        ExternalPanel::Frames { root, valid } => {
            let receipt =
                read_frames_receipt(root).map_err(|error| format!("--panel {label:?}: {error}"))?;
            let plane = frame_on_grid(&receipt, *valid, lat, lon, ny, nx)
                .map_err(|error| format!("--panel {label:?}: {error}"))?;
            if plane.covered_cells == 0 {
                return Err(format!(
                    "--panel {label:?}: no cell of the row's grid lies inside {}'s lattice",
                    root.display()
                ));
            }
            let source = if receipt.source_id.is_empty() {
                receipt.source_kind.clone()
            } else {
                receipt.source_id.clone()
            };
            let members = if receipt.members > 1 {
                format!("{} members, ", receipt.members)
            } else {
                String::new()
            };
            let detail = format!(
                "{source}{} | {members}{}",
                if receipt.causal { "" } else { " (not causal)" },
                plane.reduction
            );
            let record = serde_json::json!({
                "kind": "frames", "root": root, "receipt_sha256": receipt.receipt_sha256,
                "source_kind": receipt.source_kind, "source_id": receipt.source_id,
                "issue_time": receipt.issue_time.to_rfc3339(), "causal": receipt.causal,
                "members": receipt.members, "reduction": plane.reduction,
                "covered_cells": plane.covered_cells, "outside_cells": plane.outside_cells,
            });
            Ok(ExternalRead {
                values: plane.values,
                start: receipt.issue_time,
                valid: *valid,
                detail,
                record,
            })
        }
        ExternalPanel::Window { path } => {
            let window =
                window_column_max(path).map_err(|error| format!("--panel {label:?}: {error}"))?;
            if (window.ny, window.nx) != (ny, nx) {
                return Err(format!(
                    "--panel {label:?}: the window {} is {}x{} and the row's grid is {ny}x{nx}; a \
                     window is drawn on the grid it was made for and nothing is regridded here",
                    window.receipt.display(),
                    window.ny,
                    window.nx
                ));
            }
            let start = window.start.filter(|start| *start <= window.valid).unwrap_or(window.valid);
            let class = window.lead_class.clone().unwrap_or_else(|| "unlabelled".to_string());
            let detail = format!("heating window column maximum | lead class {class}");
            let record = serde_json::json!({
                "kind": "ttenref", "receipt": window.receipt, "window_end": window.valid.to_rfc3339(),
                "lead_class": window.lead_class, "source_id": window.source_id,
                "grid_identity_sha256": window.grid_identity_sha256, "nz": window.nz,
                "echo_columns": window.echo_columns, "clear_columns": window.clear_columns,
                "no_coverage_columns": window.no_coverage_columns,
            });
            Ok(ExternalRead { values: window.values, start, valid: window.valid, detail, record })
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn read_row(
    args: &Args,
    spec: &ProductSpec,
    row: &[PanelArg],
    members: &[ImportedMember],
    member_of: &dyn Fn(&Path) -> usize,
    has_forecast: bool,
) -> Result<Result<(RowPlanes, DateTime<Utc>, u16), String>, String> {
    let mut geometry: Option<(Vec<f32>, Vec<f32>, Option<GridProjection>, usize, usize)> = None;
    let mut first_field: Option<RunField> = None;
    let mut model_id: Option<ModelId> = None;
    let mut slots: Vec<Option<ModelPanel>> = (0..row.len()).map(|_| None).collect();
    let mut row_valid: Option<(i64, PathBuf)> = None;
    let mut cycle: Option<DateTime<Utc>> = None;
    for (index, panel) in row.iter().enumerate() {
        if panel.external.is_some() {
            continue;
        }
        let mut planes: Vec<Vec<f32>> = Vec::with_capacity(panel.files.len());
        let mut panel_time: Option<(u64, i64)> = None;
        for file in &panel.files {
            let member = &members[member_of(file)];
            match &row_valid {
                None => row_valid = Some((member.valid_unix, member.file.clone())),
                Some((valid, first)) if *valid != member.valid_unix => {
                    return Err(format!(
                        "{} is valid at {} and {} at {}; one row of a sheet is one valid time",
                        member.file.display(),
                        utc(member.valid_unix)?.to_rfc3339(),
                        first.display(),
                        utc(*valid)?.to_rfc3339()
                    ));
                }
                Some(_) => {}
            }
            let source = StoreFieldSource::open(
                &member.store_root,
                &member.summary.model,
                &member.summary.run,
                member.slot,
            )
            .map_err(|error| format!("{}: open store: {error}", member.file.display()))?;
            let (lat, lon) = {
                let (lat, lon) = source.grid_coordinates();
                (lat.to_vec(), lon.to_vec())
            };
            match &geometry {
                None => {
                    let grid = source.full_grid();
                    let (ny, nx) = (grid.shape.ny, grid.shape.nx);
                    drop(grid);
                    geometry = Some((lat, lon, source.projection().cloned(), ny, nx));
                    model_id = Some(member.summary.model.parse().map_err(|error| {
                        format!("store model slug {:?}: {error}", member.summary.model)
                    })?);
                }
                Some((have_lat, have_lon, ..)) => {
                    if !same_grid(have_lat, have_lon, &lat, &lon) {
                        return Err(format!(
                            "{} is not on the grid of the row's first panel; panels of one \
                             sheet share one grid and nothing is regridded here",
                            member.file.display()
                        ));
                    }
                }
            }
            let field = match read_run_field(
                &source,
                spec,
                member.lead_seconds,
                member.valid_unix,
                &member.by_valid,
                &member.store_root,
                &member.summary,
            )? {
                Ok(field) => field,
                Err(reason) => return Ok(Err(format!("{}: {reason}", member.file.display()))),
            };
            if let Some(first) = &first_field {
                if !same_units(&first.units, &field.units) {
                    return Err(format!(
                        "{} stores {} in {:?} and the row's first panel in {:?}",
                        member.file.display(),
                        spec.title,
                        field.units,
                        first.units
                    ));
                }
            }
            planes.push(field.values.clone());
            if first_field.is_none() {
                first_field = Some(field);
            }
            if panel_time.is_none() {
                panel_time = Some((member.lead_seconds, member.valid_unix));
            }
        }
        let (lead_seconds, valid_unix) = panel_time.expect("a panel names at least one file");
        let times = match panel_times(args, has_forecast, &mut cycle, lead_seconds, valid_unix)? {
            Ok(times) => times,
            Err(reason) => return Ok(Err(reason)),
        };
        let values: Vec<f32> = match panel.op {
            PanelOp::Field => planes.remove(0),
            op => {
                let (ny, nx) = geometry.as_ref().map(|g| (g.3, g.4)).expect("set above");
                let stack = MemberStack::new(
                    ny,
                    nx,
                    planes
                        .iter()
                        .enumerate()
                        .map(|(number, plane)| {
                            (number as u32, plane.iter().map(|v| f64::from(*v)).collect())
                        })
                        .collect(),
                )
                .map_err(|error| format!("--panel {:?}: {error}", panel.label))?;
                let reduced = match op {
                    PanelOp::Mean => ensemble_mean(&stack, NanPolicy::Mask),
                    PanelOp::Pmm => {
                        probability_matched_mean(&stack, NanPolicy::Mask, PmmTieRule::FlatIndex)
                    }
                    PanelOp::Max => Ok(member_max(&stack)),
                    PanelOp::Field => unreachable!("handled above"),
                }
                .map_err(|error| format!("--panel {:?}: {error}", panel.label))?;
                reduced.into_iter().map(|value| value as f32).collect()
            }
        };
        slots[index] = Some(ModelPanel {
            label: panel.label.clone(),
            op: panel.op,
            files: panel.files.clone(),
            values,
            times,
            detail: None,
            source: None,
        });
    }
    if row.iter().any(|panel| panel.external.is_some()) {
        if spec.name != "refc" {
            return Ok(Err(format!(
                "frames: and ttenref: panels carry composite reflectivity only, not {}",
                spec.name
            )));
        }
        if geometry.is_none() {
            let grid_file = args.panel_grid.as_ref().ok_or(
                "a row of frames: or ttenref: panels alone needs --panel-grid WRFOUT to name the \
                 grid it is drawn on",
            )?;
            let member = &members[member_of(grid_file)];
            let source = StoreFieldSource::open(
                &member.store_root,
                &member.summary.model,
                &member.summary.run,
                member.slot,
            )
            .map_err(|error| format!("{}: open store: {error}", member.file.display()))?;
            let (lat, lon) = {
                let (lat, lon) = source.grid_coordinates();
                (lat.to_vec(), lon.to_vec())
            };
            let grid = source.full_grid();
            let (ny, nx) = (grid.shape.ny, grid.shape.nx);
            drop(grid);
            geometry = Some((lat, lon, source.projection().cloned(), ny, nx));
            model_id = Some(member.summary.model.parse().map_err(|error| {
                format!("store model slug {:?}: {error}", member.summary.model)
            })?);
            // The grid file's own reflectivity sets the sheet's style and
            // units; its values are never drawn.
            first_field = Some(
                match read_run_field(
                    &source,
                    spec,
                    member.lead_seconds,
                    member.valid_unix,
                    &member.by_valid,
                    &member.store_root,
                    &member.summary,
                )? {
                    Ok(field) => field,
                    Err(reason) => {
                        return Ok(Err(format!("--panel-grid {}: {reason}", member.file.display())));
                    }
                },
            );
        }
        let units = first_field.as_ref().map(|field| field.units.clone()).unwrap_or_default();
        if !same_units(&units, "dBZ") {
            return Err(format!(
                "the row's reflectivity is stored in {units:?}; frames and windows are dBZ"
            ));
        }
        let (lat, lon, _, ny, nx) = geometry.as_ref().expect("set above");
        for (index, panel) in row.iter().enumerate() {
            let Some(external) = &panel.external else { continue };
            let read = external_plane(external, &panel.label, lat, lon, *ny, *nx)?;
            let source_path = match external {
                ExternalPanel::Frames { root, .. } => root.clone(),
                ExternalPanel::Window { path } => path.clone(),
            };
            match &row_valid {
                None => row_valid = Some((read.valid.timestamp(), source_path)),
                Some((have, first)) if *have != read.valid.timestamp() => {
                    return Err(format!(
                        "--panel {:?} is valid at {} and {} at {}; one row of a sheet is one \
                         valid time",
                        panel.label,
                        read.valid.to_rfc3339(),
                        first.display(),
                        utc(*have)?.to_rfc3339()
                    ));
                }
                Some(_) => {}
            }
            let lead_seconds = u64::try_from((read.valid - read.start).num_seconds()).map_err(|_| {
                format!(
                    "--panel {:?}: valid {} precedes its start {}",
                    panel.label, read.valid, read.start
                )
            })?;
            let times = match panel_times(
                args,
                has_forecast,
                &mut cycle,
                lead_seconds,
                read.valid.timestamp(),
            )? {
                Ok(times) => times,
                Err(reason) => return Ok(Err(reason)),
            };
            slots[index] = Some(ModelPanel {
                label: panel.label.clone(),
                op: PanelOp::Field,
                files: Vec::new(),
                values: read.values,
                times,
                detail: Some(read.detail),
                source: Some(serde_json::json!({"panel_source": external.name(), "record": read.record})),
            });
        }
    }
    let panels: Vec<ModelPanel> =
        slots.into_iter().map(|slot| slot.expect("every panel read")).collect();
    let (lat, lon, projection, ny, nx) = geometry.ok_or("a row needs at least one panel")?;
    let first_file = row
        .iter()
        .find_map(|panel| panel.files.first().cloned())
        .or_else(|| args.panel_grid.clone())
        .ok_or("a row needs a history file or --panel-grid for its domain")?;
    let (domain_token, domain_label) = domain_tokens(&first_file);
    let cycle = cycle.expect("set with the first panel");
    let lead = panels[0].times.reference_lead;
    Ok(Ok((
        RowPlanes {
            lat,
            lon,
            projection,
            ny,
            nx,
            model_id: model_id.expect("set with the grid"),
            first_field: first_field.expect("set with the first panel"),
            panels,
            domain_token,
            domain_label,
        },
        cycle,
        lead,
    )))
}

/// Draw every N-panel sheet the invocation asks for.
pub(super) fn run(
    args: &Args,
    specs: &[ProductSpec],
    forecasts: &[ReferenceSpec],
) -> Result<(), String> {
    let chosen: Vec<String> = if args.reference.is_empty() {
        Vec::new()
    } else {
        reference_names(&args.reference)?
    };
    let has_forecast = forecasts
        .iter()
        .any(|spec| chosen.iter().any(|name| name == spec.name));
    let wants_accumulation = specs
        .iter()
        .any(|spec| matches!(spec.run, RunPlane::HourlyAccumulation(_)));

    // Every distinct file once, each into its own store: two members at one
    // valid time are two runs, and one store would merge them into one frame.
    let mut distinct: Vec<PathBuf> = Vec::new();
    for row in &args.panels {
        for panel in row {
            for file in &panel.files {
                if !distinct.iter().any(|have| same_file(have, file)) {
                    distinct.push(file.clone());
                }
            }
        }
    }
    // The grid a row of frames or windows alone is drawn on.
    if let Some(grid) = &args.panel_grid {
        if !distinct.iter().any(|have| same_file(have, grid)) {
            distinct.push(grid.clone());
        }
    }
    // Plain scoped threads, a few at a time, and never rayon workers: an
    // import blocks its caller on a channel while the importer itself runs
    // on the rayon pool, so imports issued FROM that pool deadlock as soon
    // as there are more files than pool threads.
    let concurrent = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, 8);
    let mut imported: Vec<Result<ImportedMember, String>> = Vec::with_capacity(distinct.len());
    for (chunk_index, chunk) in distinct.chunks(concurrent).enumerate() {
        let results: Vec<Result<ImportedMember, String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = chunk
                .iter()
                .enumerate()
                .map(|(offset, file)| {
                    let index = chunk_index * concurrent + offset;
                    scope.spawn(move || import_member(args, specs, index, file, wants_accumulation))
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|_| Err("a member import panicked".to_string()))
                })
                .collect()
        });
        imported.extend(results);
    }
    let mut members = Vec::with_capacity(imported.len());
    for member in imported {
        members.push(member?);
    }
    println!("IMPORTED\t{} member file(s)", members.len());
    let member_of = |file: &Path| -> usize {
        members
            .iter()
            .position(|member| same_file(&member.file, file))
            .expect("every panel file was imported")
    };

    let theme = rustwx_render::active_theme();
    let station_error = args.station_mode == "error";
    let (mut rendered, mut skipped, mut failed) = (0usize, 0usize, 0usize);
    for spec in specs {
        let outcome = (|| -> Result<Result<PathBuf, String>, String> {
            let mut sheet_rows: Vec<Vec<rustwx_render::RgbaImage>> = Vec::new();
            let mut row_receipts = Vec::new();
            let mut header_units = String::new();
            let mut first_times: Option<SheetTimes> = None;
            let mut first_domain: Option<(String, Option<String>)> = None;
            let mut last_valid = None;
            let mut model_overlays_drawn = false;
            let mut sheet_style_meta: Option<(rustwx_render::RenderDensity, LegendControls)> = None;
            let mut column_tokens: Vec<String> = Vec::new();
            let mut coverage_drawn = false;
            let scratch_root = args.store_root.join("panels").join(format!("sheet_{}", spec.slug));
            for (row_index, row) in args.panels.iter().enumerate() {
                let (planes, _cycle, _lead) =
                    match read_row(args, spec, row, &members, &member_of, has_forecast)? {
                        Ok(planes) => planes,
                        Err(reason) => return Ok(Err(format!("row {}: {reason}", row_index + 1))),
                    };
                let geometry = (
                    planes.lat.as_slice(),
                    planes.lon.as_slice(),
                    planes.projection.as_ref(),
                    planes.ny,
                    planes.nx,
                );
                let times = planes.panels[0].times;
                let ResolvedReferences { style, references } = match resolve_references(
                    args,
                    &chosen,
                    forecasts,
                    spec,
                    times,
                    geometry,
                    planes.model_id,
                    &planes.first_field,
                )? {
                    Ok(resolved) => resolved,
                    Err(reason) => return Ok(Err(format!("row {}: {reason}", row_index + 1))),
                };
                let extras: Vec<String> = references
                    .iter()
                    .map(|reference| reference.name.clone())
                    .filter(|name| !chosen.contains(name))
                    .collect();
                let columns = column_order(args.order.as_deref(), row.len(), &chosen, &extras)?;
                // The row's radar footprint: the first observed reference's
                // unobserved cells, outlined on every model panel so a false
                // echo is only judged where a radar could have seen one.
                let row_coverage: Option<Vec<bool>> =
                    references.iter().find_map(|reference| reference.no_coverage.clone());
                coverage_drawn |= row_coverage.is_some();
                let outline = row_coverage
                    .as_deref()
                    .map_or(CoverageDraw::None, CoverageDraw::Outline);
                if row_index == 0 {
                    column_tokens = columns
                        .iter()
                        .map(|column| match column {
                            Column::Panel(index) => (index + 1).to_string(),
                            Column::Reference(name) => name.clone(),
                        })
                        .collect();
                }
                let scratch = scratch_root.join(format!("row{row_index}"));
                std::fs::create_dir_all(&scratch)
                    .map_err(|error| format!("create {}: {error}", scratch.display()))?;
                let marks = |values: &[f32]| -> Result<Option<MapOverlays>, String> {
                    match (args.stations.as_deref(), spec.station_quantity) {
                        (Some(path), Some(quantity)) => station_marks(
                            path, quantity, times.valid, geometry, values, &style, station_error,
                        )
                        .map(Some),
                        _ => Ok(None),
                    }
                };
                let station_label = |overlays: Option<&MapOverlays>| {
                    overlays
                        .map(|marks| {
                            let count = marks.value_layers.first().map_or(0, |layer| layer.points.len());
                            format!(
                                " | station {} dots n={count}",
                                if station_error { "obs minus forecast" } else { "observed" }
                            )
                        })
                        .unwrap_or_default()
                };
                let mut images = Vec::with_capacity(columns.len());
                let mut receipts = Vec::with_capacity(columns.len());
                for (column_index, column) in columns.iter().enumerate() {
                    let cell_scratch = scratch.join(format!("c{column_index}"));
                    std::fs::create_dir_all(&cell_scratch)
                        .map_err(|error| format!("create {}: {error}", cell_scratch.display()))?;
                    let frame = PanelFrame {
                        geometry,
                        width: args.width,
                        height: args.height,
                        slug: spec.slug,
                        style: &style,
                        scratch: &cell_scratch,
                    };
                    match column {
                        Column::Panel(index) => {
                            let panel = &planes.panels[*index];
                            let values: Vec<f32> =
                                panel.values.iter().map(|value| style.convert.apply(*value)).collect();
                            let overlays = marks(&values)?;
                            model_overlays_drawn |= overlays.is_some();
                            let time_text = if overlays.is_some() {
                                format!("{}{}", panel.times.run_subtitle(), station_label(overlays.as_ref()))
                            } else {
                                panel.times.run_panel_subtitle()
                            };
                            let left = match (&panel.detail, panel.op) {
                                (Some(detail), _) => format!("{detail} | {time_text}"),
                                (None, PanelOp::Field) => time_text,
                                (None, op) => format!("{} | {time_text}", op.describe(panel.files.len())),
                            };
                            let right = match &panel.source {
                                Some(source) => format!(
                                    "source: {}",
                                    source["panel_source"].as_str().unwrap_or("panel")
                                ),
                                None => args.run_source_subtitle.clone(),
                            };
                            images.push(draw_one_panel(
                                frame,
                                &theme,
                                "run",
                                PanelText {
                                    title: panel.label.clone(),
                                    left,
                                    right,
                                },
                                values,
                                style.scale.clone(),
                                style.cbar_tick_step,
                                style.legend,
                                &style.contour_levels,
                                overlays.as_ref(),
                                outline,
                            )?);
                            receipts.push(serde_json::json!({
                                "column": column_index,
                                "kind": if panel.source.is_some() { "external" } else { "model" },
                                "label": panel.label,
                                "op": panel.source.as_ref().and_then(|source| source["panel_source"].as_str()).unwrap_or(panel.op.name()),
                                "members": panel.files,
                                "source": panel.source,
                                "run_init": panel.times.run_init.to_rfc3339(),
                                "run_lead_seconds": panel.times.run_lead_seconds,
                                "valid": panel.times.valid.to_rfc3339(),
                                "station_count": overlays.as_ref().and_then(|marks| marks.value_layers.first()).map(|layer| layer.points.len()),
                            }));
                        }
                        Column::Reference(name) => {
                            let reference = references
                                .iter()
                                .find(|reference| &reference.name == name)
                                .ok_or_else(|| format!("reference {name} did not resolve"))?;
                            let overlays = marks(&reference.values)?;
                            images.push(draw_one_panel(
                                frame,
                                &theme,
                                "reference",
                                PanelText {
                                    title: reference.label.clone(),
                                    left: format!("{}{}", reference.subtitle, station_label(overlays.as_ref())),
                                    right: format!("source: {}", reference.source_label),
                                },
                                reference.values.clone(),
                                style.scale.clone(),
                                style.cbar_tick_step,
                                style.legend,
                                &style.contour_levels,
                                overlays.as_ref(),
                                reference
                                    .no_coverage
                                    .as_deref()
                                    .map_or(outline, CoverageDraw::Fill),
                            )?);
                            let mut receipt = reference.receipt.clone();
                            receipt["column"] = serde_json::json!(column_index);
                            receipt["station_count"] = serde_json::json!(overlays.as_ref().and_then(|marks| marks.value_layers.first()).map(|layer| layer.points.len()));
                            receipts.push(receipt);
                        }
                    }
                }
                if row_index == 0 {
                    header_units = style.display_units.clone();
                    first_times = Some(times);
                    first_domain = Some((planes.domain_token.clone(), planes.domain_label.clone()));
                    sheet_style_meta = Some((style.density, style.legend));
                } else if style.display_units != header_units {
                    return Err(format!(
                        "row {} draws {} and row 1 draws {header_units}; one sheet is one colour table",
                        row_index + 1,
                        style.display_units
                    ));
                }
                last_valid = Some(times.valid);
                let shared_start = planes.panels.iter().all(|panel| {
                    panel.times.run_init == times.run_init
                        && panel.times.run_lead_seconds == times.run_lead_seconds
                });
                row_receipts.push(serde_json::json!({
                    "row": row_index, "valid": times.valid.to_rfc3339(),
                    "shared_start": shared_start, "columns": receipts,
                }));
                sheet_rows.push(images);
            }
            let times = first_times.ok_or("no row was drawn")?;
            let (domain_token, domain_label) = first_domain.expect("set with row 0");
            let mut facts = Vec::new();
            match last_valid {
                Some(last) if sheet_rows.len() > 1 => facts.push(format!(
                    "Valid {} to {} ({} rows)",
                    hour_label(times.valid),
                    hour_label(last),
                    sheet_rows.len()
                )),
                _ => facts.push(format!("Valid {}", hour_label(times.valid))),
            }
            if sheet_rows.len() == 1 && row_receipts[0]["shared_start"] == serde_json::json!(true) {
                facts.push(times.run_subtitle());
            }
            if let Some(label) = &domain_label {
                facts.push(label.clone());
            }
            if !chosen.is_empty() {
                facts.push(format!("References: {}", chosen.join(", ")));
            }
            if coverage_drawn {
                facts.push(NO_COVERAGE_LEGEND.to_string());
            }
            let sheet = rw_wrfbatch::compare::compose_sheet_rows(
                &sheet_rows,
                &SheetHeader {
                    title: format!("{} ({header_units})", spec.title),
                    subtitle: facts.join(" | "),
                },
            )?;
            let sheet = match sheet_style_meta {
                Some((density, legend)) if station_error && model_overlays_drawn => append_error_key(
                    &sheet,
                    &header_units,
                    rustwx_render::ColormapBuildOptions { render_density: density, legend },
                )?,
                _ => sheet,
            };
            let stem = args.sheet_name.clone().map(|name| safe_component(&name, "sheet")).unwrap_or_else(|| {
                format!(
                    "panels{}_vs_{}_{}_{}{}_{}",
                    args.panels[0].len(),
                    if chosen.is_empty() { "none".to_string() } else { chosen.join("_") },
                    spec.slug,
                    times.stem_token(),
                    if sheet_rows.len() > 1 { format!("_rows{}", sheet_rows.len()) } else { String::new() },
                    safe_component(&domain_token, "native_grid")
                )
            });
            let day = times.valid.format("%Y-%m-%d").to_string();
            let path = match args.layout {
                Layout::Nested => layout_path(
                    &args.out_dir,
                    &domain_token,
                    &format!("compare_panels_{}", spec.slug),
                    &day,
                    &stem,
                ),
                Layout::Flat => args.out_dir.join(format!("{stem}.png")),
            };
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
            rustwx_render::save_rgba_png_profile_with_options(
                &sheet,
                &path,
                &PngWriteOptions { compression: PngCompressionMode::default() },
            )
            .map_err(|error| format!("write {}: {error}", path.display()))?;
            if let Some(flat) = &args.flat_dir {
                std::fs::create_dir_all(flat).map_err(|error| error.to_string())?;
                copy_flat_sheet(&path, &flat.join(format!("{stem}.png")))?;
            }
            let receipt = serde_json::json!({
                "schema": "gpuwm.compare-sheet.v3", "product": spec.name, "slug": spec.slug,
                "title": spec.title, "display_units": header_units,
                "run_source_label": args.source_label, "domain": domain_token,
                "order": column_tokens, "rows": row_receipts,
                "panel_size": [args.width, args.height], "sheet": path,
            });
            std::fs::write(
                path.with_extension("json"),
                serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            let _ = std::fs::remove_dir_all(&scratch_root);
            Ok(Ok(path))
        })();
        match outcome {
            Ok(Ok(path)) => {
                rendered += 1;
                println!("RENDERED\t{}\tpanels\t{}", spec.name, path.display());
            }
            Ok(Err(reason)) => {
                skipped += 1;
                println!("SKIPPED\t{}\tpanels\t{reason}", spec.name);
            }
            Err(message) => {
                failed += 1;
                eprintln!("FAILED\t{}\tpanels\t{message}", spec.name);
            }
        }
    }
    println!("FINISHED rendered={rendered} skipped={skipped} failed={failed}");
    if rendered == 0 || failed > 0 {
        return Err(format!(
            "comparison incomplete: rendered={rendered} skipped={skipped} failed={failed}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panel_names_its_label_its_reduction_and_its_members() {
        let dir = std::env::temp_dir().join(format!("rw-panels-parse-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["m_1.nc", "m_2.nc", "m_10.nc", "other.txt"] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
        let one = dir.join("m_1.nc");
        let parsed = parse_panel(&format!("WOOF DA member 0={}", one.display())).unwrap();
        assert_eq!(parsed.label, "WOOF DA member 0");
        assert_eq!(parsed.op, PanelOp::Field);
        assert_eq!(parsed.files, vec![one.clone()]);
        let pmm = parse_panel(&format!("mean=pmm:{}", dir.join("m_*.nc").display())).unwrap();
        assert_eq!(pmm.op, PanelOp::Pmm);
        let names: Vec<String> = pmm.files.iter().map(|f| f.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, vec!["m_1.nc", "m_2.nc", "m_10.nc"], "natural member order");
        assert!(parse_panel("no label").is_err());
        assert!(parse_panel(&format!("=x{}", one.display())).is_err());
        assert!(parse_panel(&format!("x=max:{}", one.display())).is_err(), "a reduction needs two members");
        assert!(parse_panel(&format!("x={},{}", one.display(), one.display())).is_err());
        assert!(parse_panel(&format!("x=mean:{}", dir.join("none_*.nc").display())).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_column_order_names_every_panel_and_reference_once() {
        let chosen = vec!["mrms".to_string(), "hrrr".to_string()];
        let order: Vec<String> = ["mrms", "1", "2", "3", "hrrr"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            column_order(Some(&order), 3, &chosen, &[]).unwrap(),
            vec![
                Column::Reference("mrms".into()),
                Column::Panel(0),
                Column::Panel(1),
                Column::Panel(2),
                Column::Reference("hrrr".into()),
            ]
        );
        assert_eq!(
            column_order(None, 2, &chosen, &[]).unwrap(),
            vec![Column::Panel(0), Column::Panel(1), Column::Reference("mrms".into()), Column::Reference("hrrr".into())]
        );
        let short: Vec<String> = ["mrms", "1", "hrrr"].iter().map(|s| s.to_string()).collect();
        assert!(column_order(Some(&short), 2, &chosen, &[]).unwrap_err().contains("leaves out 2"));
        let twice: Vec<String> = ["1", "1", "mrms", "hrrr"].iter().map(|s| s.to_string()).collect();
        assert!(column_order(Some(&twice), 1, &chosen, &[]).is_err());
        let unknown: Vec<String> = ["1", "rrfs"].iter().map(|s| s.to_string()).collect();
        assert!(column_order(Some(&unknown), 1, &[], &[]).is_err());
        let extra = vec!["obs".to_string()];
        assert_eq!(
            column_order(Some(&["1".to_string()]), 1, &[], &extra).unwrap(),
            vec![Column::Panel(0), Column::Reference("obs".into())]
        );
    }

    #[test]
    fn the_member_maximum_ignores_missing_members() {
        let stack = MemberStack::new(1, 3, vec![(0, vec![1.0, f64::NAN, 5.0]), (1, vec![4.0, f64::NAN, -1.0])]).unwrap();
        let max = member_max(&stack);
        assert_eq!(max[0], 4.0);
        assert!(max[1].is_nan());
        assert_eq!(max[2], 5.0);
    }
}
