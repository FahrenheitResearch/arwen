//! Guided edits to existing TOML. Python remains the configuration authority.
use toml_edit::{DocumentMut, InlineTable, Item, Table, TableLike, Value};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Section {
    Geometry,
    AddChild,
    Follow,
    Spawn,
    Retire,
    Rearm,
}

pub const SECTIONS: [(Section, &str); 6] = [
    (Section::Geometry, "Grid, placement and delayed start"),
    (Section::AddChild, "Add a finer child nest"),
    (Section::Follow, "Follow a storm, vortex or model attribute"),
    (Section::Spawn, "Start a nest on time or weather criteria"),
    (Section::Retire, "Retire a triggered nest"),
    (Section::Rearm, "Re-arm a retired nest slot"),
];

#[derive(Clone, Copy)]
enum Type {
    Integer,
    Number,
    Date,
    Levels,
    Box,
    Choice(&'static [&'static str]),
}

pub struct Field {
    pub label: &'static str,
    pub help: &'static str,
    pub value: String,
    initial: String,
    path: String,
    kind: Type,
    required: bool,
}

impl Field {
    pub fn choices(&self) -> &'static [&'static str] {
        match self.kind {
            Type::Choice(values) => values,
            _ => &[],
        }
    }

    fn parsed(&self) -> Result<Option<Value>, String> {
        let text = self.value.trim();
        if text.is_empty() {
            return if self.required {
                Err(format!("{} is required.", self.label))
            } else {
                Ok(None)
            };
        }
        if let Type::Choice(choices) = self.kind {
            if !choices.contains(&text) {
                return Err(format!("{}: choose {}.", self.label, choices.join(", ")));
            }
            return Ok(Some(Value::from(text)));
        }
        let parsed = format!("value = {text}\n")
            .parse::<DocumentMut>()
            .map_err(|_| format!("{}: enter a valid {}.", self.label, self.format_hint()))?;
        if parsed.len() != 1 {
            return Err(format!("{} takes one value.", self.label));
        }
        let value = parsed
            .get("value")
            .and_then(Item::as_value)
            .ok_or_else(|| format!("{} takes one value.", self.label))?;
        let number =
            |v: &Value| v.as_integer().is_some() || v.as_float().is_some_and(f64::is_finite);
        let valid = match self.kind {
            Type::Integer => value.as_integer().is_some(),
            Type::Number => number(value),
            Type::Date => value
                .as_datetime()
                .is_some_and(|d| d.date.is_some() && d.time.is_some() && d.offset.is_none()),
            Type::Levels => {
                number(value)
                    || value
                        .as_array()
                        .is_some_and(|a| !a.is_empty() && a.iter().all(number))
            }
            Type::Box => value
                .as_array()
                .is_some_and(|a| a.len() == 4 && a.iter().all(|v| v.as_integer().is_some())),
            Type::Choice(_) => unreachable!(),
        };
        if !valid {
            return Err(format!(
                "{}: enter a valid {}.",
                self.label,
                self.format_hint()
            ));
        }
        let mut value = value.clone();
        value.decor_mut().clear();
        Ok(Some(value))
    }

    fn format_hint(&self) -> &'static str {
        match self.kind {
            Type::Integer => "whole number",
            Type::Number => "finite number",
            Type::Date => "UTC date/time without quotes or Z (2026-09-05T06:00:00)",
            Type::Levels => "pressure level or array such as [850, 700, 500]",
            Type::Box => "array of four parent indices: [i_lo, j_lo, i_hi, j_hi]",
            Type::Choice(_) => "listed choice",
        }
    }
}

pub struct Form {
    pub section: Section,
    pub index: usize,
    pub selected: usize,
    pub fields: Vec<Field>,
    pub title: String,
    pub note: String,
    pub original: String,
    global_follow: bool,
    child_id: i64,
}

fn domains(doc: &DocumentMut) -> Result<&toml_edit::ArrayOfTables, String> {
    doc.get("domain").and_then(Item::as_array_of_tables)
        .filter(|a| !a.is_empty())
        .ok_or_else(|| "Open or create a configuration with [[domain]] entries first. All settings remains available for other TOML formats.".into())
}

pub fn labels(text: &str) -> Result<Vec<String>, String> {
    let doc = text
        .parse::<DocumentMut>()
        .map_err(|e| format!("Correct the TOML in Settings first: {e}"))?;
    Ok(domains(&doc)?
        .iter()
        .map(|d| {
            let show = |key| {
                d.get(key)
                    .map(|v| v.to_string().trim().to_owned())
                    .unwrap_or_else(|| "?".into())
            };
            let policy = if d.contains_key("spawn") {
                "triggered start"
            } else if d.contains_key("start_time") {
                "scheduled start"
            } else {
                "normal start"
            };
            format!(
                "d{}  | parent {} | {} x {} cells | {policy}{}",
                show("grid_id"),
                show("parent_id"),
                show("nx"),
                show("ny"),
                if d.contains_key("follow") {
                    " | follows weather"
                } else {
                    ""
                }
            )
        })
        .collect())
}

fn get<'a>(table: &'a dyn TableLike, path: &str) -> Option<&'a Item> {
    let mut keys = path.split('.');
    let mut item = table.get(keys.next()?)?;
    for key in keys {
        item = item.as_table_like()?.get(key)?;
    }
    Some(item)
}

fn set(table: &mut dyn TableLike, path: &str, value: Option<Value>) -> Result<(), String> {
    if let Some((head, tail)) = path.split_once('.') {
        if table.get(head).is_none() {
            if value.is_none() {
                return Ok(());
            }
            table.insert(head, Item::Value(Value::InlineTable(InlineTable::new())));
        }
        let child = table
            .get_mut(head)
            .and_then(Item::as_table_like_mut)
            .ok_or_else(|| format!("{head} must be a table; correct it in Settings first."))?;
        return set(child, tail, value);
    }
    if let Some(mut value) = value {
        if let Some(old) = table.get(path).and_then(Item::as_value) {
            *value.decor_mut() = old.decor().clone();
        }
        if let Some(item) = table.get_mut(path) {
            *item = Item::Value(value);
        } else {
            table.insert(path, Item::Value(value));
        }
    } else {
        table.remove(path);
    }
    Ok(())
}

impl Form {
    pub fn new(text: String, index: usize, section: Section) -> Result<Self, String> {
        let doc = text
            .parse::<DocumentMut>()
            .map_err(|e| format!("Correct the TOML first: {e}"))?;
        let all = domains(&doc)?;
        let domain = all.get(index).ok_or("Choose an existing domain.")?;
        let grid_id = domain
            .get("grid_id")
            .and_then(Item::as_integer)
            .ok_or("Domain grid_id must be an integer.")?;
        let root = domain.get("parent_id").and_then(Item::as_integer) == Some(0);
        if root
            && matches!(
                section,
                Section::Follow | Section::Spawn | Section::Retire | Section::Rearm
            )
        {
            return Err("Choose a child nest for tracking or lifecycle settings. Add a finer child nest from this domain first.".into());
        }
        let relocation = doc.get("relocation").and_then(Item::as_table_like);
        let global_follow = section == Section::Follow
            && relocation.is_some_and(|r| {
                r.get("grid_id").and_then(Item::as_integer) == Some(grid_id)
                    && r.contains_key("follow")
            });
        if global_follow && domain.contains_key("follow") {
            return Err("This nest has both domain.follow and relocation.follow. Resolve the two owners in Settings before guided editing.".into());
        }
        let target: &dyn TableLike = if global_follow {
            relocation.unwrap()
        } else {
            domain
        };
        let policy = match section {
            Section::Follow => Some("follow"),
            Section::Spawn => Some("spawn"),
            Section::Retire => Some("retire"),
            Section::Rearm => Some("rearm"),
            _ => None,
        };
        let child_id = all
            .iter()
            .filter_map(|d| d.get("grid_id").and_then(Item::as_integer))
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or("No available domain ID.")?;
        let mut fields = Vec::new();
        if let Some(policy) = policy {
            let enabled = target.contains_key(policy).to_string();
            fields.push(Field { label: "Policy enabled", help: "Choose true to configure this policy. Applying false removes this entire policy table, including any advanced values in it. Other policies remain unchanged.", value: enabled.clone(), initial: enabled,
                path: "@enabled".into(), kind: Type::Choice(&["false", "true"]), required: true });
        }
        let mut add =
            |label: &'static str, key: &str, help: &'static str, kind: Type, required: bool| {
                let value = if section == Section::AddChild {
                    String::new()
                } else {
                    get(target, key)
                        .map(|item| {
                            item.as_str().map(str::to_owned).unwrap_or_else(|| {
                                if let Some(value) = item.as_value() {
                                    let mut value = value.clone();
                                    value.decor_mut().clear();
                                    value.to_string()
                                } else {
                                    item.to_string().trim().to_owned()
                                }
                            })
                        })
                        .unwrap_or_default()
                };
                fields.push(Field {
                    label,
                    help,
                    initial: value.clone(),
                    value,
                    path: key.into(),
                    kind,
                    required,
                });
            };
        match section {
            Section::Geometry | Section::AddChild => {
                add("Grid width (nx, cells)", "nx", "Number of unstaggered cells. The engine checks refinement divisibility, boundary clearance and GPU memory.", Type::Integer, true);
                add("Grid height (ny, cells)", "ny", "Number of unstaggered cells north-south. No area or physics is changed until you apply this draft.", Type::Integer, true);
                if !root || section == Section::AddChild {
                    add("Refinement ratio", "parent_grid_ratio", "Child spacing is parent spacing divided by this whole-number ratio. The engine validates supported ratios.", Type::Integer, true);
                    add("Time-step ratio", "parent_time_step_ratio", "Parent time step divided by this ratio gives the child time step. Enter the intended value; F5 checks compatibility.", Type::Integer, true);
                    add("West-east parent start (i)", "i_parent_start", "1-based parent cell at the child's southwest corner. For triggered weather placement this is the declared placeholder.", Type::Integer, true);
                    add("South-north parent start (j)", "j_parent_start", "1-based parent cell at the child's southwest corner. The engine checks containment and boundary clearance.", Type::Integer, true);
                } else {
                    add("Root spacing east-west (m)", "dx", "Root grid spacing in metres. F5 checks the complete nested hierarchy after changing it.", Type::Number, true);
                    add("Root spacing north-south (m)", "dy", "Optional root spacing in metres; blank keeps the engine's matching dx default. Existing nested geometry remains as declared.", Type::Number, false);
                }
                add("History interval (seconds)", "history_interval_s", "Required output cadence for this domain. F5 checks it against the model clock and any tracking cadence.", Type::Number, true);
                if !root || section == Section::AddChild {
                    add("Delayed start (UTC, optional)", "start_time", "Unquoted UTC date/time, e.g. 2026-09-05T06:00:00. Blank removes a delayed start. A nest cannot declare both start_time and a spawn trigger.", Type::Date, false);
                }
            }
            Section::Follow => {
                add("Track signal", "follow.field", "pressure: vortex; uh: rotation; reflectivity: echoes; attribute: a supported native state field. Clear incompatible signal settings deliberately and review the threshold units.", Type::Choice(&["pressure", "uh", "reflectivity", "attribute"]), true);
                add("Signal threshold", "follow.threshold", "Pressure: height depth m, or hPa ceiling at level_hpa=0. UH: m2/s2. Echo: dBZ. Attribute: theta in K, qv/qc/qr in kg/kg dry air, w in m/s. Maximum follows values above this; minimum follows values below it.", Type::Number, true);
                add("Pressure level(s), hPa", "follow.level_hpa", "Pressure only. Blank uses the engine's 850 hPa surface; 0 selects mean sea-level pressure. A list such as [850, 700, 500] tracks the mean of those centres.", Type::Levels, false);
                add("Vortex centroid radius (km)", "follow.radius_km", "Pressure only. Blank keeps the engine default; define how far from the pressure extremum the centroid searches.", Type::Number, false);
                add("Reflectivity fallback (dBZ)", "follow.fallback_threshold", "Required for UH: echo threshold before rotation develops. Leave blank for pressure or reflectivity; those signals refuse a fallback.", Type::Number, false);
                add("Model attribute", "follow.attribute", "Attribute signal only. theta = total potential temperature (K); qv/qc/qr = vapour/cloud/rain mixing ratio (kg/kg dry air); w = vertical velocity at mass levels (m/s). This is not RH, air temperature or a hail-size diagnostic.", Type::Choice(&["theta", "qv", "qc", "qr", "w"]), false);
                add("Attribute maximum or minimum", "follow.extremum", "Attribute signal only. max follows the threshold-exceeding feature; min follows values below the threshold. Both use the native weighted centroid and movement limits.", Type::Choice(&["max", "min"]), false);
                add("Attribute vertical reduction", "follow.reduction", "Attribute signal only. Column maximum/minimum/mean uses model mass levels. The mean is unweighted across levels. model_level selects one explicit model index; it is not a pressure or height surface.", Type::Choice(&["column_max", "column_min", "column_mean", "model_level"]), false);
                add("Attribute model level (zero-based)", "follow.model_level", "Required only with model_level reduction: 0 is the lowest mass level. The engine checks this against actual available levels. Clear it for column reductions.", Type::Integer, false);
                add(
                    "Search margin (parent cells)",
                    "follow.search_margin_cells",
                    "Search beyond the nest footprint by this many parent cells.",
                    Type::Integer,
                    true,
                );
                add(
                    "Minimum shift (parent cells)",
                    "follow.min_shift_cells",
                    "Movement dead band in whole parent cells.",
                    Type::Integer,
                    true,
                );
                add(
                    "Maximum shift (parent cells)",
                    "follow.max_shift_cells",
                    "Largest requested displacement in one tracking event.",
                    Type::Integer,
                    true,
                );
                add(
                    "Tracking cooldown (seconds)",
                    "follow.cooldown_seconds",
                    "Minimum model time between tracking moves.",
                    Type::Number,
                    true,
                );
                for (label, key, help, kind, required) in [
                    ("Evaluation cadence (seconds)", "cadence_seconds", "Tracking is evaluated at cycle boundaries. Match the signal's availability; F5 checks history and model-clock compatibility.", Type::Number, true),
                    ("Move bound (parent cells)", "max_move_parent_cells", "Optional movement bound; blank uses the engine default.", Type::Integer, false),
                    ("Minimum overlap fraction", "min_overlap_fraction", "Optional minimum old/new footprint overlap as a fraction. F5 checks the engine's supported range.", Type::Number, false),
                ] {
                    let path = if global_follow { key.to_owned() } else { format!("follow.{key}") };
                    add(label, &path, help, kind, required);
                }
            }
            Section::Spawn => {
                add("Start trigger", "spawn.trigger", "time: model-time boundary; pressure, uh, reflectivity: parent weather. When changing trigger, clear fields marked for other triggers. Dormant nests reserve VRAM.", Type::Choice(&["time", "pressure", "uh", "reflectivity"]), true);
                add("Start after model seconds (time only)", "spawn.at_s", "Seconds since experiment start. Use this for delayed activation that can also retire and re-arm.", Type::Number, false);
                add("Signal threshold (weather only)", "spawn.threshold", "UH: m2/s2. Reflectivity: dBZ. Pressure: height depth in m, or absolute hPa ceiling when level_hpa is 0.", Type::Number, false);
                add(
                    "Watch starts after seconds",
                    "spawn.earliest_s",
                    "Weather triggers only: earliest model time to evaluate the parent signal.",
                    Type::Number,
                    false,
                );
                add("Watch ends after seconds", "spawn.latest_s", "Weather triggers only: latest model time to allow activation. No activation occurs after the window closes.", Type::Number, false);
                add("Search box (parent indices)", "spawn.search_box", "Optional [i_lo, j_lo, i_hi, j_hi], 1-based inclusive parent cells. Blank keeps the native search area.", Type::Box, false);
                add("Pressure level, hPa", "spawn.level_hpa", "Pressure only: one level, blank means 850 hPa; 0 means sea-level pressure. Threshold units change with this choice.", Type::Number, false);
                add(
                    "Vortex centroid radius (km)",
                    "spawn.radius_km",
                    "Pressure only; blank keeps the engine default.",
                    Type::Number,
                    false,
                );
            }
            Section::Retire => {
                add("Retirement trigger", "retire.trigger", "time: episode age; weather: quiet signal under the child. Clear fields marked for other triggers when changing. Retirement requires a spawn policy.", Type::Choice(&["time", "pressure", "uh", "reflectivity"]), true);
                add("Retire at episode age (seconds)", "retire.at_s", "Time trigger only. Seconds since this nest episode became active, not since experiment start.", Type::Number, false);
                add("Quiet-signal threshold", "retire.threshold", "Weather only: UH m2/s2, reflectivity dBZ, or pressure-surface height depth m. At pressure level 0 this is the sea-level hPa ceiling the storm has filled past.", Type::Number, false);
                add(
                    "Quiet signal sustained for (seconds)",
                    "retire.sustained_s",
                    "Optional continuous quiet period. Blank uses the engine default.",
                    Type::Number,
                    false,
                );
                add(
                    "Minimum active lifetime (seconds)",
                    "retire.min_lifetime_s",
                    "Optional minimum episode lifetime before retirement can occur.",
                    Type::Number,
                    false,
                );
                add("Pressure level, hPa", "retire.level_hpa", "Pressure only: blank means 850 hPa; 0 selects sea-level pressure. Use the same intended signal surface as the start policy.", Type::Number, false);
            }
            Section::Rearm => {
                add("Maximum activations", "rearm.max_firings", "Total allowed firings of this nest slot. Requires both spawn and retirement policies.", Type::Integer, true);
                add(
                    "Cooldown before re-arming (seconds)",
                    "rearm.cooldown_s",
                    "Model seconds to wait after retirement before the slot can activate again.",
                    Type::Number,
                    true,
                );
            }
        }
        let title = format!(
            "d{grid_id:02} - {}",
            SECTIONS.iter().find(|s| s.0 == section).unwrap().1
        );
        let note = if section == Section::AddChild {
            format!("New d{child_id:02}, parent d{grid_id:02}. Enter the intended geometry; shared settings are inherited. No scientific preset is selected.")
        } else if global_follow {
            "Editing this nest's existing [relocation.follow]. Manual moves, containment and track output stay in the complete draft.".into()
        } else {
            "Only changed fields are applied. Blank optional values remove their keys. Save explicitly, then use F5 Check / F6 Plan for full validation.".into()
        };
        Ok(Self {
            section,
            index,
            selected: 0,
            fields,
            title,
            note,
            original: text,
            global_follow,
            child_id,
        })
    }

    pub fn signal_issue(&self) -> Option<String> {
        let value = |path: &str| {
            self.fields
                .iter()
                .find(|f| f.path == path)
                .map(|f| f.value.trim())
                .unwrap_or("")
        };
        if value("@enabled") == "false" {
            return None;
        }
        let selector = match self.section {
            Section::Follow => "follow.field",
            Section::Spawn => "spawn.trigger",
            Section::Retire => "retire.trigger",
            _ => return None,
        };
        let signal = value(selector);
        if signal.is_empty() {
            return None;
        }
        let mut incompatible = Vec::new();
        match self.section {
            Section::Follow => {
                if signal != "pressure" {
                    incompatible.push("follow.level_hpa");
                }
                if signal != "uh" {
                    incompatible.push("follow.fallback_threshold");
                }
                if signal != "attribute" {
                    incompatible.extend(["follow.attribute", "follow.extremum", "follow.reduction", "follow.model_level"]);
                } else if value("follow.reduction") != "model_level" {
                    incompatible.push("follow.model_level");
                }
            }
            Section::Spawn => {
                if signal == "time" {
                    incompatible.extend([
                        "spawn.threshold",
                        "spawn.earliest_s",
                        "spawn.latest_s",
                        "spawn.search_box",
                        "spawn.level_hpa",
                        "spawn.radius_km",
                    ]);
                } else {
                    incompatible.push("spawn.at_s");
                    if signal != "pressure" {
                        incompatible.extend(["spawn.level_hpa", "spawn.radius_km"]);
                    }
                }
            }
            Section::Retire => {
                incompatible.push(if signal == "time" {
                    "retire.threshold"
                } else {
                    "retire.at_s"
                });
                if signal != "pressure" {
                    incompatible.push("retire.level_hpa");
                }
            }
            _ => {}
        }
        for path in incompatible {
            if let Some(field) = self
                .fields
                .iter()
                .find(|f| f.path == path && !f.value.trim().is_empty())
            {
                return Some(format!("{signal} refuses {}. Edit that field and Ctrl+U to clear it. Values are kept until you change them.", field.label));
            }
        }
        if self.section == Section::Follow
            && signal == "uh"
            && value("follow.fallback_threshold").is_empty()
        {
            return Some("UH tracking requires Reflectivity fallback (dBZ). Enter the echo threshold used before rotation develops.".into());
        }
        if self.section == Section::Follow && signal == "attribute" {
            for path in ["follow.attribute", "follow.extremum", "follow.reduction"] {
                if value(path).is_empty() {
                    let label = self.fields.iter().find(|field| field.path == path).unwrap().label;
                    return Some(format!("Attribute tracking requires {label}. Choose the actual field and vertical meaning before applying."));
                }
            }
            if value("follow.reduction") == "model_level" && value("follow.model_level").parse::<u32>().is_err() {
                return Some("Attribute model level must be a nonnegative whole index when reduction is model_level. The engine validates the actual vertical extent.".into());
            }
        }
        None
    }

    pub fn apply(&self) -> Result<String, String> {
        if let Some(issue) = self.signal_issue() {
            return Err(issue);
        }
        let mut doc = self
            .original
            .parse::<DocumentMut>()
            .map_err(|e| e.to_string())?;
        let disabled = self
            .fields
            .first()
            .is_some_and(|f| f.path == "@enabled" && f.value.trim() == "false");
        let all = domains(&doc)?;
        let domain = all
            .get(self.index)
            .ok_or("The selected domain no longer exists.")?;
        let policy = match self.section {
            Section::Follow => Some("follow"),
            Section::Spawn => Some("spawn"),
            Section::Retire => Some("retire"),
            Section::Rearm => Some("rearm"),
            _ => None,
        };
        if !disabled {
            if self.section == Section::Spawn && domain.contains_key("start_time") {
                return Err("This nest already has start_time. Clear its delayed start in Grid settings before enabling a trigger.".into());
            }
            if self.section == Section::Retire && !domain.contains_key("spawn") {
                return Err("Configure a triggered start before enabling retirement.".into());
            }
            if self.section == Section::Rearm
                && (!domain.contains_key("spawn") || !domain.contains_key("retire"))
            {
                return Err(
                    "Configure both triggered start and retirement before enabling re-arm.".into(),
                );
            }
        }
        if disabled
            && self.section == Section::Spawn
            && (domain.contains_key("retire") || domain.contains_key("rearm"))
        {
            return Err(
                "Disable re-arm and retirement before removing their start trigger.".into(),
            );
        }
        if disabled && self.section == Section::Retire && domain.contains_key("rearm") {
            return Err("Disable re-arm before removing retirement.".into());
        }
        let mut additions = None;
        if self.section == Section::AddChild {
            let parent_id = domain
                .get("grid_id")
                .and_then(Item::as_integer)
                .ok_or("Parent grid_id is missing.")?;
            let mut child = Table::new();
            child["grid_id"] = toml_edit::value(self.child_id);
            child["parent_id"] = toml_edit::value(parent_id);
            child["specified"] = toml_edit::value(false);
            child["nested"] = toml_edit::value(true);
            additions = Some(child);
        }
        let target: &mut dyn TableLike = if let Some(child) = additions.as_mut() {
            child
        } else if self.global_follow {
            doc["relocation"]
                .as_table_like_mut()
                .ok_or("Relocation must be a table.")?
        } else {
            doc["domain"]
                .as_array_of_tables_mut()
                .unwrap()
                .get_mut(self.index)
                .unwrap()
        };
        if disabled {
            target.remove(policy.unwrap());
        } else {
            for field in &self.fields {
                let parsed = field.parsed()?;
                if field.path == "@enabled" {
                    continue;
                }
                if field.value != field.initial || self.section == Section::AddChild {
                    set(target, &field.path, parsed)?;
                }
            }
            if matches!(self.section, Section::Spawn | Section::Retire) {
                let prefix = policy.unwrap();
                let trigger = get(target, &format!("{prefix}.trigger")).and_then(Item::as_str);
                let required = if trigger == Some("time") {
                    vec!["at_s"]
                } else if self.section == Section::Spawn {
                    vec!["threshold", "earliest_s", "latest_s"]
                } else {
                    vec!["threshold"]
                };
                for key in required {
                    if get(target, &format!("{prefix}.{key}")).is_none() {
                        return Err(format!(
                            "{prefix} trigger {} requires {key}.",
                            trigger.unwrap_or("(choose one)")
                        ));
                    }
                }
            }
            if matches!(self.section, Section::Geometry)
                && target.contains_key("spawn")
                && target.contains_key("start_time")
            {
                return Err("A nest cannot have both a start trigger and a delayed UTC start. Disable its start trigger first, or leave delayed start blank.".into());
            }
        }
        if let Some(child) = additions {
            doc["domain"].as_array_of_tables_mut().unwrap().push(child);
        }
        let updated = doc.to_string();
        updated
            .parse::<DocumentMut>()
            .map_err(|e| format!("Draft TOML is invalid: {e}"))?;
        Ok(updated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const CONFIG: &str = "# retained header\n[experiment]\nname = 'Keep me'\n[unknown]\nvalue = 42 # untouched\n[[domain]]\ngrid_id=1\nparent_id=0\nnx=300\nny=300\nhistory_interval_s=300.0\ndx=12000.0\ndy=12000.0\n[[domain]]\ngrid_id=2\nparent_id=1\nnx=90 # child width\nny=90\nparent_grid_ratio=3\nparent_time_step_ratio=3\ni_parent_start=21\nj_parent_start=22\nhistory_interval_s=300.0\n";
    fn put(form: &mut Form, key: &str, value: &str) {
        form.fields
            .iter_mut()
            .find(|f| f.path == key)
            .unwrap()
            .value = value.into();
    }
    #[test]
    fn only_selected_fields_change_and_comments_survive() {
        let mut form = Form::new(CONFIG.into(), 1, Section::Geometry).unwrap();
        assert_eq!(form.apply().unwrap(), CONFIG);
        put(&mut form, "nx", "120");
        let result = form.apply().unwrap();
        assert_eq!(
            result,
            CONFIG.replace("nx=90 # child width", "nx=120 # child width")
        );
    }
    #[test]
    fn child_creation_is_explicit_and_inherits_shared_settings() {
        let mut form = Form::new(CONFIG.into(), 1, Section::AddChild).unwrap();
        assert!(form.apply().is_err());
        for (key, value) in [
            ("nx", "90"),
            ("ny", "120"),
            ("parent_grid_ratio", "3"),
            ("parent_time_step_ratio", "3"),
            ("i_parent_start", "11"),
            ("j_parent_start", "12"),
            ("history_interval_s", "300.0"),
        ] {
            put(&mut form, key, value);
        }
        let result = form.apply().unwrap();
        assert!(result.starts_with(CONFIG));
        let doc = result.parse::<DocumentMut>().unwrap();
        let child = &doc["domain"].as_array_of_tables().unwrap().get(2).unwrap();
        assert_eq!(child["grid_id"].as_integer(), Some(3));
        assert_eq!(child["parent_id"].as_integer(), Some(2));
        assert_eq!(child["nested"].as_bool(), Some(true));
        assert!(child.get("mp_physics").is_none());
    }
    #[test]
    fn lifecycle_edits_are_real_and_conflicting_activation_is_refused() {
        let mut form = Form::new(CONFIG.into(), 1, Section::Spawn).unwrap();
        put(&mut form, "@enabled", "true");
        put(&mut form, "spawn.trigger", "time");
        put(&mut form, "spawn.at_s", "300");
        let spawned = form.apply().unwrap();
        let mut geometry = Form::new(spawned.clone(), 1, Section::Geometry).unwrap();
        put(&mut geometry, "start_time", "2026-09-05T06:00:00");
        assert!(geometry.apply().unwrap_err().contains("both"));
        let mut retire = Form::new(spawned, 1, Section::Retire).unwrap();
        put(&mut retire, "@enabled", "true");
        put(&mut retire, "retire.trigger", "time");
        put(&mut retire, "retire.at_s", "900");
        let retired = retire.apply().unwrap();
        let mut rearm = Form::new(retired, 1, Section::Rearm).unwrap();
        put(&mut rearm, "@enabled", "true");
        put(&mut rearm, "rearm.max_firings", "2");
        put(&mut rearm, "rearm.cooldown_s", "300");
        let final_doc = rearm.apply().unwrap().parse::<DocumentMut>().unwrap();
        let child = final_doc["domain"]
            .as_array_of_tables()
            .unwrap()
            .get(1)
            .unwrap();
        assert_eq!(get(child, "spawn.at_s").unwrap().as_integer(), Some(300));
        assert_eq!(get(child, "retire.at_s").unwrap().as_integer(), Some(900));
        assert_eq!(
            get(child, "rearm.max_firings").unwrap().as_integer(),
            Some(2)
        );
    }
    #[test]
    fn syntax_and_types_cannot_corrupt_the_draft() {
        for invalid in ["3.5", "\"12\"", "12\n[evil]\nx=4", "not-a-number"] {
            let mut form = Form::new(CONFIG.into(), 1, Section::Geometry).unwrap();
            put(&mut form, "nx", invalid);
            assert!(form.apply().is_err(), "{invalid}");
        }
    }
    #[test]
    fn global_follower_preserves_advanced_settings_and_manual_moves() {
        let text = format!("{CONFIG}\n[relocation]\nenabled=true\ngrid_id=2\ncadence_seconds=900.0\n[relocation.follow]\nfield='pressure'\nthreshold=25.0\nsearch_margin_cells=20\nmin_shift_cells=2\nmax_shift_cells=8\ncooldown_seconds=3600.0\nrefine_grid_id=2 # preserve\n[[relocation.move]]\nat_seconds=1800.0\ndi_parent_cells=1\ndj_parent_cells=1\n");
        let mut form = Form::new(text.clone(), 1, Section::Follow).unwrap();
        put(&mut form, "follow.threshold", "30.0");
        let result = form.apply().unwrap();
        assert_eq!(result, text.replace("threshold=25.0", "threshold=30.0"));
        put(&mut form, "@enabled", "false");
        let disabled = form.apply().unwrap().parse::<DocumentMut>().unwrap();
        assert!(disabled["relocation"].get("follow").is_none());
        assert!(disabled["relocation"].get("move").is_some());
    }

    #[test]
    fn signal_changes_name_incompatible_fields_and_keep_them_until_explicitly_cleared() {
        for (section, selector, signal, incompatible, label) in [
            (
                Section::Follow,
                "follow.field",
                "uh",
                "follow.level_hpa",
                "Pressure level(s)",
            ),
            (
                Section::Follow,
                "follow.field",
                "reflectivity",
                "follow.fallback_threshold",
                "Reflectivity fallback",
            ),
            (
                Section::Spawn,
                "spawn.trigger",
                "time",
                "spawn.threshold",
                "Signal threshold",
            ),
            (
                Section::Spawn,
                "spawn.trigger",
                "pressure",
                "spawn.at_s",
                "Start after",
            ),
            (
                Section::Spawn,
                "spawn.trigger",
                "reflectivity",
                "spawn.radius_km",
                "Vortex centroid",
            ),
            (
                Section::Retire,
                "retire.trigger",
                "time",
                "retire.threshold",
                "Quiet-signal",
            ),
            (
                Section::Retire,
                "retire.trigger",
                "uh",
                "retire.at_s",
                "Retire at",
            ),
        ] {
            let mut form = Form::new(CONFIG.into(), 1, section).unwrap();
            put(&mut form, "@enabled", "true");
            put(&mut form, selector, signal);
            put(&mut form, incompatible, "50");
            let error = form.apply().unwrap_err();
            assert!(error.contains(label), "{error}");
            assert!(error.contains("Ctrl+U"), "{error}");
            assert_eq!(form.original, CONFIG);
            assert_eq!(
                form.fields
                    .iter()
                    .find(|f| f.path == incompatible)
                    .unwrap()
                    .value,
                "50"
            );
            put(&mut form, incompatible, "");
            if section == Section::Follow && signal == "uh" {
                assert!(form
                    .signal_issue()
                    .unwrap()
                    .contains("requires Reflectivity fallback"));
                put(&mut form, "follow.fallback_threshold", "40");
            }
            assert!(form.signal_issue().is_none());
        }
    }

    #[test]
    fn generated_nested_tracking_and_lifecycle_settings_pass_the_python_loader() {
        let Some(python) = std::env::var_os("GPUWM_TUI_TEST_PYTHON") else {
            return;
        };
        let validate = |text: &str| {
            use std::io::Write;
            use std::process::{Command, Stdio};
            let config_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
            let mut child = Command::new(&python).args(["-c", "import pathlib,sys,tomllib; from gpuwm.experiment import build_experiment_from_config_tables; base=pathlib.Path(sys.argv[1]); exp=build_experiment_from_config_tables(tomllib.loads(sys.stdin.read()), source=str(base/'tui-draft.toml'), base_dir=base); assert len(exp.domains)==2; print('native configuration accepted')"])
                .arg(&config_dir).current_dir(config_dir.parent().unwrap()).env("GPUWM_NO_LOCAL_GPU", "1")
                .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
            child
                .stdin
                .take()
                .unwrap()
                .write_all(text.as_bytes())
                .unwrap();
            let result = child.wait_with_output().unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
        };
        let source = include_str!("../../../configs/gfs_12km_quickstart.toml");
        let mut nest = Form::new(source.into(), 0, Section::AddChild).unwrap();
        for (key, value) in [
            ("nx", "30"),
            ("ny", "30"),
            ("parent_grid_ratio", "3"),
            ("parent_time_step_ratio", "3"),
            ("i_parent_start", "11"),
            ("j_parent_start", "11"),
            ("history_interval_s", "1800.0"),
        ] {
            put(&mut nest, key, value);
        }
        let nested = nest.apply().unwrap();
        validate(&nested);
        let mut follow = Form::new(nested.clone(), 1, Section::Follow).unwrap();
        for (key, value) in [
            ("@enabled", "true"),
            ("follow.field", "pressure"),
            ("follow.threshold", "25.0"),
            ("follow.level_hpa", "850.0"),
            ("follow.search_margin_cells", "10"),
            ("follow.min_shift_cells", "1"),
            ("follow.max_shift_cells", "4"),
            ("follow.cooldown_seconds", "1800.0"),
            ("follow.cadence_seconds", "1800.0"),
        ] {
            put(&mut follow, key, value);
        }
        let tracked = follow.apply().unwrap();
        validate(&tracked);
        for attribute in ["theta", "qv", "qc", "qr", "w"] {
            for reduction in ["column_max", "column_min", "column_mean", "model_level"] {
                let mut field = Form::new(tracked.clone(), 1, Section::Follow).unwrap();
                put(&mut field, "follow.field", "attribute");
                assert!(field.apply().is_err());
                put(&mut field, "follow.level_hpa", "");
                put(&mut field, "follow.attribute", attribute);
                put(&mut field, "follow.extremum", "max");
                put(&mut field, "follow.reduction", reduction);
                put(&mut field, "follow.threshold", if attribute == "theta" { "300.0" } else { "0.01" });
                if reduction == "model_level" {
                    assert!(field.apply().is_err());
                    put(&mut field, "follow.model_level", "0");
                }
                let draft = field.apply().unwrap();
                validate(&draft);
                let mut unchanged = Form::new(draft.clone(), 1, Section::Follow).unwrap();
                assert_eq!(unchanged.apply().unwrap(), draft);
                put(&mut unchanged, "follow.field", "reflectivity");
                assert!(unchanged.signal_issue().unwrap().contains("refuses Model attribute"));
            }
        }
        let mut switched_follow = Form::new(tracked.clone(), 1, Section::Follow).unwrap();
        put(&mut switched_follow, "follow.field", "uh");
        assert!(switched_follow.apply().is_err());
        put(&mut switched_follow, "follow.level_hpa", "");
        put(&mut switched_follow, "follow.fallback_threshold", "40.0");
        let uh = switched_follow.apply().unwrap();
        validate(&uh);
        let mut switched_follow = Form::new(uh, 1, Section::Follow).unwrap();
        put(&mut switched_follow, "follow.field", "reflectivity");
        assert!(switched_follow.apply().is_err());
        put(&mut switched_follow, "follow.fallback_threshold", "");
        validate(&switched_follow.apply().unwrap());
        let mut spawn = Form::new(tracked, 1, Section::Spawn).unwrap();
        for (key, value) in [
            ("@enabled", "true"),
            ("spawn.trigger", "pressure"),
            ("spawn.threshold", "25.0"),
            ("spawn.level_hpa", "850.0"),
            ("spawn.earliest_s", "3600.0"),
            ("spawn.latest_s", "18000.0"),
        ] {
            put(&mut spawn, key, value);
        }
        let spawned = spawn.apply().unwrap();
        validate(&spawned);
        let mut switched_spawn = Form::new(spawned.clone(), 1, Section::Spawn).unwrap();
        put(&mut switched_spawn, "spawn.trigger", "time");
        put(&mut switched_spawn, "spawn.at_s", "3600.0");
        assert!(switched_spawn.apply().is_err());
        for key in [
            "spawn.threshold",
            "spawn.level_hpa",
            "spawn.earliest_s",
            "spawn.latest_s",
        ] {
            put(&mut switched_spawn, key, "");
        }
        let timed = switched_spawn.apply().unwrap();
        validate(&timed);
        let mut switched_spawn = Form::new(timed, 1, Section::Spawn).unwrap();
        put(&mut switched_spawn, "spawn.trigger", "reflectivity");
        assert!(switched_spawn.apply().is_err());
        for (key, value) in [
            ("spawn.at_s", ""),
            ("spawn.threshold", "40.0"),
            ("spawn.earliest_s", "3600.0"),
            ("spawn.latest_s", "18000.0"),
        ] {
            put(&mut switched_spawn, key, value);
        }
        validate(&switched_spawn.apply().unwrap());
        let mut retire = Form::new(spawned, 1, Section::Retire).unwrap();
        for (key, value) in [
            ("@enabled", "true"),
            ("retire.trigger", "time"),
            ("retire.at_s", "3600.0"),
        ] {
            put(&mut retire, key, value);
        }
        let mut rearm = Form::new(retire.apply().unwrap(), 1, Section::Rearm).unwrap();
        for (key, value) in [
            ("@enabled", "true"),
            ("rearm.max_firings", "2"),
            ("rearm.cooldown_s", "1800.0"),
        ] {
            put(&mut rearm, key, value);
        }
        validate(&rearm.apply().unwrap());
        let mut delayed = Form::new(nested, 1, Section::Geometry).unwrap();
        put(&mut delayed, "start_time", "2026-08-20T13:00:00");
        validate(&delayed.apply().unwrap());
    }
}
