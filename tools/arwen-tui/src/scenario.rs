//! Initial-state authoring. This panel edits a TOML draft; it never prepares or runs a model.
use crate::theme;
use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::{
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};
use toml_edit::{ArrayOfTables, DocumentMut, InlineTable, Item, Table, TableLike, Value};
use unicode_width::UnicodeWidthChar;

const KEYS: [&str; 7] = [
    "center_lat",
    "center_lon",
    "center_height_m",
    "radius_km",
    "depth_m",
    "amplitude_k",
    "rh_preserve",
];
const LABELS: [&str; 7] = [
    "Latitude (degrees N)",
    "Longitude (degrees E)",
    "Center height (m AGL)",
    "Horizontal radius (km)",
    "Vertical half-depth (m)",
    "Peak theta increase (K)",
    "Preserve relative humidity",
];
const HELP: [&str; 7] = [
    "-90 to 90 degrees. The engine checks that the center is inside the coarse domain.",
    "-180 to 180 degrees; west is negative. Choose the intended initiation location.",
    "Height above ground in metres, at least 0. The peak occurs at this height.",
    "Positive radius in kilometres. The engine refuses a bubble that touches no cells.",
    "Positive HALF-depth in metres, from center to the top or bottom of the ellipse.",
    "Potential-temperature increase: greater than 0 and at most 10 K. Warm bubbles only.",
    "True adjusts water vapor inside the bubble to keep RH. False leaves vapor unchanged.",
];

#[derive(Clone)]
struct Bubble {
    source: Option<usize>,
    values: [String; 7],
}

struct Edit {
    index: Option<usize>,
    values: [String; 7],
    selected: usize,
    cursor: usize,
}

enum Screen {
    List,
    Edit(Edit),
    Review,
}

#[derive(Clone, Copy)]
enum Hit {
    Key(KeyCode),
    Bubble(usize),
    Field(usize),
}

/// A result for the caller. Only `Apply` supplies a replacement editor draft.
pub enum Intent {
    Keep,
    Cancel,
    Apply(String),
}

/// A reversible warm-bubble editor, with its own keyboard and mouse hit regions.
pub struct Form {
    original: String,
    document: DocumentMut,
    initial: Vec<Bubble>,
    bubbles: Vec<Bubble>,
    defaults: [String; 7],
    screen: Screen,
    selected: usize,
    notice: Option<String>,
    offset: usize,
    review_height: usize,
    review_width: usize,
    hits: Vec<(Rect, Hit)>,
}

fn numeric(item: Option<&Item>) -> Option<String> {
    let value = item?.as_value()?;
    value
        .as_integer()
        .map(|v| v.to_string())
        .or_else(|| value.as_float().map(|v| v.to_string()))
}

fn coordinate_label(value: &str, positive: &str, negative: &str) -> String {
    match value.parse::<f64>() {
        Ok(number) if number.is_finite() => format!(
            "{} {}", number.abs(), if number.is_sign_negative() { negative } else { positive }
        ),
        _ => value.to_owned(),
    }
}

fn load_bubble(table: &dyn TableLike, source: usize) -> Result<Bubble, String> {
    if let Some((key, _)) = table.iter().find(|(key, _)| !KEYS.contains(key)) {
        return Err(format!(
            "Bubble {} has unsupported key {key}. Correct it in Settings first.",
            source + 1
        ));
    }
    let mut values = std::array::from_fn(|_| String::new());
    for (index, key) in KEYS.iter().enumerate() {
        if let Some(item) = table.get(key) {
            values[index] = if index == 6 {
                item.as_bool().map(|v| v.to_string())
            } else {
                numeric(Some(item))
            }
            .ok_or_else(|| {
                format!(
                    "Bubble {}: {key} must be {}. Correct it in Settings first.",
                    source + 1,
                    if index == 6 { "a boolean" } else { "a number" }
                )
            })?;
        } else if index == 6 {
            values[index] = "false".into();
        }
    }
    Ok(Bubble {
        source: Some(source),
        values,
    })
}

fn parsed(values: &[String; 7]) -> Result<[Value; 7], String> {
    let mut numbers = [0.0; 6];
    for index in 0..6 {
        let value = values[index]
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .ok_or_else(|| format!("{}: enter one finite number.", LABELS[index]))?;
        numbers[index] = value;
    }
    if !(-90.0..=90.0).contains(&numbers[0]) {
        return Err("Latitude must be between -90 and 90 degrees.".into());
    }
    if !(-180.0..=180.0).contains(&numbers[1]) {
        return Err("Longitude must be between -180 and 180 degrees.".into());
    }
    if numbers[2] < 0.0 {
        return Err("Center height must be at least 0 m above ground.".into());
    }
    for index in 3..6 {
        if numbers[index] <= 0.0 {
            return Err(format!("{} must be greater than zero.", LABELS[index]));
        }
    }
    if numbers[5] > 10.0 {
        return Err("Peak theta increase must be at most 10 K.".into());
    }
    let rh = values[6]
        .parse::<bool>()
        .map_err(|_| "Preserve relative humidity must be true or false.".to_owned())?;
    Ok(std::array::from_fn(|index| {
        if index == 6 {
            Value::from(rh)
        } else {
            Value::from(numbers[index])
        }
    }))
}

fn update_values(
    table: &mut dyn TableLike,
    bubble: &Bubble,
    initial: Option<&Bubble>,
) -> Result<(), String> {
    let values = parsed(&bubble.values)?;
    for (index, mut value) in values.into_iter().enumerate() {
        if initial.is_some_and(|old| old.values[index] == bubble.values[index]) {
            continue;
        }
        if let Some(old) = table.get(KEYS[index]).and_then(Item::as_value) {
            *value.decor_mut() = old.decor().clone();
        }
        table.insert(KEYS[index], Item::Value(value));
    }
    Ok(())
}

impl Form {
    /// Read the current draft, preserving its TOML representation and comments.
    pub fn new(text: String) -> Result<Self, String> {
        let document = text
            .parse::<DocumentMut>()
            .map_err(|e| format!("Correct the TOML in Settings first: {e}"))?;
        let mut bubbles = Vec::new();
        if let Some(item) = document.get("perturbation") {
            let table = item
                .as_table_like()
                .ok_or("[perturbation] must be a table. Correct it in Settings first.")?;
            if let Some((key, _)) = table.iter().find(|(key, _)| *key != "bubbles") {
                return Err(format!(
                    "Unsupported perturbation setting {key}. Correct it in Settings first."
                ));
            }
            if let Some(entries) = table.get("bubbles") {
                if let Some(tables) = entries.as_array_of_tables() {
                    for (index, table) in tables.iter().enumerate() {
                        bubbles.push(load_bubble(table, index)?);
                    }
                } else if let Some(array) = entries.as_array() {
                    for (index, value) in array.iter().enumerate() {
                        let table = value
                            .as_inline_table()
                            .ok_or("Each perturbation bubble must be a table.")?;
                        bubbles.push(load_bubble(table, index)?);
                    }
                } else {
                    return Err("Use [[perturbation.bubbles]] entries or an array of inline tables in Settings.".into());
                }
            }
        }
        let projection = document.get("projection").and_then(Item::as_table_like);
        let defaults = [
            numeric(projection.and_then(|p| p.get("ref_lat"))).unwrap_or_default(),
            numeric(projection.and_then(|p| p.get("ref_lon"))).unwrap_or_default(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            "false".into(),
        ];
        Ok(Self {
            original: text,
            document,
            initial: bubbles.clone(),
            bubbles,
            defaults,
            screen: Screen::List,
            selected: 0,
            notice: None,
            offset: 0,
            review_height: 1,
            review_width: 60,
            hits: Vec::new(),
        })
    }

    fn start_edit(&mut self, index: Option<usize>) {
        let values = index
            .map(|i| self.bubbles[i].values.clone())
            .unwrap_or_else(|| self.defaults.clone());
        let cursor = values[0].len();
        self.screen = Screen::Edit(Edit {
            index,
            values,
            selected: 0,
            cursor,
        });
        self.notice = None;
    }

    fn keep_bubble(&mut self) {
        let Screen::Edit(edit) = &self.screen else {
            return;
        };
        if let Err(error) = parsed(&edit.values) {
            self.notice = Some(error);
            return;
        }
        if let Some(index) = edit.index {
            self.bubbles[index].values = edit.values.clone();
            self.selected = index;
        } else {
            self.bubbles.push(Bubble {
                source: None,
                values: edit.values.clone(),
            });
            self.selected = self.bubbles.len() - 1;
        }
        self.screen = Screen::List;
        self.notice = None;
    }

    fn changed(&self) -> bool {
        self.bubbles.len() != self.initial.len()
            || self
                .bubbles
                .iter()
                .enumerate()
                .any(|(index, b)| b.source != Some(index) || b.values != self.initial[index].values)
    }

    fn review_available(&self) -> bool {
        self.changed() || (self.bubbles.is_empty() && self.document.contains_key("perturbation"))
    }

    fn draft(&self) -> Result<String, String> {
        for (index, bubble) in self.bubbles.iter().enumerate() {
            parsed(&bubble.values).map_err(|e| format!("Bubble {}: {e}", index + 1))?;
        }
        if !self.changed() {
            // An existing empty block is invalid; removing it explicitly disables perturbations.
            if self.bubbles.is_empty() && self.document.contains_key("perturbation") {
                let mut document = self.document.clone();
                document.remove("perturbation");
                return Ok(document.to_string());
            }
            return Ok(self.original.clone());
        }
        let mut document = self.document.clone();
        if self.bubbles.is_empty() {
            document.remove("perturbation");
            return Ok(document.to_string());
        }
        let previous = self
            .document
            .get("perturbation")
            .and_then(Item::as_table_like)
            .and_then(|p| p.get("bubbles"));
        let result = if let Some(array) = previous.and_then(Item::as_array) {
            let mut output = array.clone();
            output.clear();
            for bubble in &self.bubbles {
                let mut table = bubble
                    .source
                    .and_then(|i| array.get(i))
                    .and_then(Value::as_inline_table)
                    .cloned()
                    .unwrap_or_else(InlineTable::new);
                update_values(
                    &mut table,
                    bubble,
                    bubble.source.and_then(|i| self.initial.get(i)),
                )?;
                if bubble.source.is_some() {
                    output.push_formatted(Value::InlineTable(table));
                } else {
                    output.push(Value::InlineTable(table));
                }
            }
            Item::Value(Value::Array(output))
        } else {
            let source = previous.and_then(Item::as_array_of_tables);
            let mut output = ArrayOfTables::new();
            for bubble in &self.bubbles {
                let mut table = bubble
                    .source
                    .and_then(|i| source.and_then(|s| s.get(i)))
                    .cloned()
                    .unwrap_or_else(Table::new);
                update_values(
                    &mut table,
                    bubble,
                    bubble.source.and_then(|i| self.initial.get(i)),
                )?;
                output.push(table);
            }
            Item::ArrayOfTables(output)
        };
        if !document.contains_key("perturbation") {
            let mut table = Table::new();
            table.set_implicit(true);
            document.insert("perturbation", Item::Table(table));
        }
        document
            .get_mut("perturbation")
            .and_then(Item::as_table_like_mut)
            .ok_or("[perturbation] must be a table.")?
            .insert("bubbles", result);
        Ok(document.to_string())
    }

    fn review_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for (index, old) in self.initial.iter().enumerate() {
            match self.bubbles.iter().find(|b| b.source == Some(index)) {
                None => {
                    lines.push(format!("REMOVE bubble {}", index + 1));
                    for (field, label) in LABELS.iter().enumerate() {
                        lines.push(format!("  {label}: {}", old.values[field]));
                    }
                }
                Some(new) => {
                    for (field, label) in LABELS.iter().enumerate() {
                        if old.values[field] != new.values[field] {
                            lines.push(format!(
                                "Bubble {} · {label}: {} → {}",
                                index + 1,
                                old.values[field],
                                new.values[field]
                            ));
                        }
                    }
                }
            }
        }
        for (index, bubble) in self
            .bubbles
            .iter()
            .enumerate()
            .filter(|(_, b)| b.source.is_none())
        {
            lines.push(format!("ADD bubble {}", index + 1));
            for (field, label) in LABELS.iter().enumerate() {
                lines.push(format!("  {label}: {}", bubble.values[field]));
            }
        }
        if self.bubbles.is_empty() && self.document.contains_key("perturbation") {
            lines.push(
                "Remove [perturbation] completely; use the unmodified analyzed initial state."
                    .into(),
            );
        } else if lines.is_empty() {
            lines.push("No changes to the initial-state settings.".into());
        }
        lines
    }

    fn review_rows(&self) -> Vec<String> {
        let width = self.review_width.max(1);
        let mut rows = Vec::new();
        for line in self.review_lines() {
            let mut row = String::new();
            let mut used = 0;
            for character in line.chars() {
                let columns = character.width().unwrap_or(0);
                if used + columns > width && !row.is_empty() {
                    rows.push(std::mem::take(&mut row));
                    used = 0;
                }
                row.push(character);
                used += columns;
            }
            rows.push(row);
        }
        rows
    }

    /// Handle keys locally. Esc cancels an edit or review, then cancels the complete form.
    pub fn key(&mut self, event: KeyEvent) -> Intent {
        if event.kind != KeyEventKind::Press {
            return Intent::Keep;
        }
        let ctrl = event.modifiers.contains(KeyModifiers::CONTROL);
        if event.code == KeyCode::Esc {
            if matches!(self.screen, Screen::List) {
                return Intent::Cancel;
            }
            self.screen = Screen::List;
            self.notice = None;
            return Intent::Keep;
        }
        if matches!(self.screen, Screen::Edit(_)) {
            if event.code == KeyCode::F(2) || (ctrl && event.code == KeyCode::Char('s')) {
                self.keep_bubble();
                return Intent::Keep;
            }
            let Screen::Edit(edit) = &mut self.screen else {
                unreachable!()
            };
            match event.code {
                KeyCode::Tab | KeyCode::Down => {
                    edit.selected = (edit.selected + 1) % 7;
                    edit.cursor = edit.values[edit.selected].len();
                }
                KeyCode::BackTab | KeyCode::Up => {
                    edit.selected = (edit.selected + 6) % 7;
                    edit.cursor = edit.values[edit.selected].len();
                }
                KeyCode::Enter if edit.selected < 6 => {
                    edit.selected += 1;
                    edit.cursor = edit.values[edit.selected].len();
                }
                KeyCode::Enter | KeyCode::Char(' ') | KeyCode::Left | KeyCode::Right
                    if edit.selected == 6 =>
                {
                    edit.values[6] = (edit.values[6] != "true").to_string();
                }
                KeyCode::Char('u') if ctrl && edit.selected < 6 => {
                    edit.values[edit.selected].clear();
                    edit.cursor = 0;
                }
                KeyCode::Left if edit.selected < 6 => edit.cursor = edit.cursor.saturating_sub(1),
                KeyCode::Right if edit.selected < 6 => {
                    edit.cursor = (edit.cursor + 1).min(edit.values[edit.selected].len())
                }
                KeyCode::Home => edit.cursor = 0,
                KeyCode::End => edit.cursor = edit.values[edit.selected].len(),
                KeyCode::Backspace if edit.selected < 6 && edit.cursor > 0 => {
                    edit.cursor -= 1;
                    edit.values[edit.selected].remove(edit.cursor);
                }
                KeyCode::Delete
                    if edit.selected < 6 && edit.cursor < edit.values[edit.selected].len() =>
                {
                    edit.values[edit.selected].remove(edit.cursor);
                }
                KeyCode::Char(c)
                    if !ctrl
                        && edit.selected < 6
                        && (c.is_ascii_digit() || "+-.eE".contains(c)) =>
                {
                    edit.values[edit.selected].insert(edit.cursor, c);
                    edit.cursor += 1;
                }
                _ => {}
            }
            return Intent::Keep;
        }
        if matches!(self.screen, Screen::Review) {
            let last = self.review_rows().len().saturating_sub(self.review_height);
            match event.code {
                KeyCode::Enter => match self.draft() {
                    Ok(text) => return Intent::Apply(text),
                    Err(error) => self.notice = Some(error),
                },
                KeyCode::Up => self.offset = self.offset.saturating_sub(1),
                KeyCode::Down => self.offset = (self.offset + 1).min(last),
                KeyCode::PageUp => self.offset = self.offset.saturating_sub(self.review_height),
                KeyCode::PageDown => self.offset = (self.offset + self.review_height).min(last),
                KeyCode::Home => self.offset = 0,
                KeyCode::End => self.offset = last,
                _ => {}
            }
            return Intent::Keep;
        }
        match event.code {
            KeyCode::Char('a' | 'A') => self.start_edit(None),
            KeyCode::Enter if !self.bubbles.is_empty() => self.start_edit(Some(self.selected)),
            KeyCode::Delete | KeyCode::Char('x' | 'X') if !self.bubbles.is_empty() => {
                self.bubbles.remove(self.selected);
                self.selected = self.selected.min(self.bubbles.len().saturating_sub(1));
                self.notice = None;
            }
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => {
                self.selected = (self.selected + 1).min(self.bubbles.len().saturating_sub(1))
            }
            KeyCode::Char('r' | 'R') | KeyCode::F(2) if self.review_available() => {
                match self.draft() {
                    Ok(_) => {
                        self.screen = Screen::Review;
                        self.offset = 0;
                        self.notice = None;
                    }
                    Err(error) => self.notice = Some(error),
                }
            }
            _ => {}
        }
        Intent::Keep
    }

    /// Paste numeric input into the selected field, rejecting control text and TOML injection.
    pub fn paste(&mut self, text: &str) {
        let Screen::Edit(edit) = &mut self.screen else {
            return;
        };
        let text = text.trim();
        if edit.selected == 6 || text.is_empty() {
            return;
        }
        if !text
            .chars()
            .all(|c| c.is_ascii_digit() || "+-.eE".contains(c))
        {
            self.notice = Some("Paste one numeric value, without units or extra lines.".into());
            return;
        }
        edit.values[edit.selected].insert_str(edit.cursor, text);
        edit.cursor += text.len();
    }

    /// Resolve mouse events using the most recent draw; buttons use the same key path.
    pub fn mouse(&mut self, event: MouseEvent) -> Intent {
        let key = match event.kind {
            MouseEventKind::ScrollUp => Some(KeyCode::Up),
            MouseEventKind::ScrollDown => Some(KeyCode::Down),
            MouseEventKind::Down(MouseButton::Left) => {
                let hit = self
                    .hits
                    .iter()
                    .find(|(r, _)| {
                        event.column >= r.x
                            && event.column < r.right()
                            && event.row >= r.y
                            && event.row < r.bottom()
                    })
                    .map(|(_, h)| *h);
                match hit {
                    Some(Hit::Key(key)) => Some(key),
                    Some(Hit::Bubble(index)) => {
                        self.selected = index;
                        self.start_edit(Some(index));
                        None
                    }
                    Some(Hit::Field(index)) => {
                        if let Screen::Edit(edit) = &mut self.screen {
                            edit.selected = index;
                            edit.cursor = edit.values[index].len();
                            if index == 6 {
                                edit.values[6] = (edit.values[6] != "true").to_string();
                            }
                        }
                        None
                    }
                    None => None,
                }
            }
            _ => None,
        };
        key.map(|key| self.key(KeyEvent::new(key, KeyModifiers::NONE)))
            .unwrap_or(Intent::Keep)
    }

    fn button(&mut self, frame: &mut Frame, area: Rect, label: &str, key: KeyCode, primary: bool) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let width = (label.len() as u16 + 2).min(area.width);
        let rect = Rect::new(area.x, area.y, width, 1);
        frame.render_widget(
            Paragraph::new(format!(" {label} ")).style(theme::button(primary)),
            rect,
        );
        self.hits.push((rect, Hit::Key(key)));
    }

    /// Draw a scrollable list, field editor or change review in the shared workspace theme.
    pub fn draw(&mut self, frame: &mut Frame, area: Rect) {
        self.hits.clear();
        frame.render_widget(Clear, area);
        let block = theme::panel("Initial-state warm bubbles");
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.width < 28 || inner.height < 8 {
            frame.render_widget(
                Paragraph::new("Enlarge the terminal to edit initial conditions. Esc returns.")
                    .style(theme::notice(false))
                    .wrap(Wrap { trim: true }),
                inner,
            );
            return;
        }
        let footer_y = inner.bottom().saturating_sub(2);
        let body_bottom = footer_y.saturating_sub(1);
        let available = body_bottom.saturating_sub(inner.y);
        if let Screen::Edit(edit) = &self.screen {
            let title = edit
                .index
                .map(|i| format!("Edit bubble {}", i + 1))
                .unwrap_or_else(|| "Add bubble · choose location and strength".into());
            frame.render_widget(
                Paragraph::new(title).style(theme::heading()),
                Rect::new(inner.x, inner.y, inner.width, 1),
            );
            let visible = available.saturating_sub(4).min(7).max(1) as usize;
            let first = edit.selected.saturating_sub(visible - 1);
            for (row, index) in (first..7).take(visible).enumerate() {
                let value = if edit.values[index].is_empty() {
                    "<required>"
                } else {
                    &edit.values[index]
                };
                let rect = Rect::new(inner.x, inner.y + 1 + row as u16, inner.width, 1);
                let selected = edit.selected == index;
                let style = if selected {
                    theme::selected()
                } else {
                    Style::default().fg(theme::INK)
                };
                frame.render_widget(
                    Paragraph::new(format!(
                        "{} {}  {}",
                        if selected { "›" } else { " " },
                        LABELS[index],
                        value
                    ))
                    .style(style),
                    rect,
                );
                self.hits.push((rect, Hit::Field(index)));
                if selected && index < 6 {
                    let prefix = 2 + LABELS[index].len() + 2;
                    let x = (prefix + edit.cursor).min(inner.width.saturating_sub(1) as usize);
                    frame.set_cursor_position((inner.x + x as u16, rect.y));
                }
            }
            let help_y = inner.y + 1 + visible as u16;
            frame.render_widget(
                Paragraph::new(HELP[edit.selected])
                    .style(Style::default().fg(theme::MUTED))
                    .wrap(Wrap { trim: true }),
                Rect::new(
                    inner.x,
                    help_y,
                    inner.width,
                    body_bottom.saturating_sub(help_y),
                ),
            );
            self.button(
                frame,
                Rect::new(inner.x, footer_y, 22.min(inner.width), 1),
                "F2 Keep bubble",
                KeyCode::F(2),
                true,
            );
            self.button(
                frame,
                Rect::new(
                    inner.x + 23.min(inner.width),
                    footer_y,
                    inner.width.saturating_sub(23),
                    1,
                ),
                "Esc Back",
                KeyCode::Esc,
                false,
            );
            frame.render_widget(
                Paragraph::new("Tab/↑↓ fields · type value · Ctrl+U clear · Space toggle RH")
                    .style(Style::default().fg(theme::MUTED)),
                Rect::new(inner.x, footer_y + 1, inner.width, 1),
            );
        } else if matches!(self.screen, Screen::Review) {
            frame.render_widget(
                Paragraph::new("Review changes · Apply updates the open draft")
                    .style(theme::heading()),
                Rect::new(inner.x, inner.y, inner.width, 1),
            );
            self.review_width = inner.width as usize;
            let lines = self.review_rows();
            let review_y = inner.y + 2;
            self.review_height = body_bottom.saturating_sub(review_y).max(1) as usize;
            self.offset = self
                .offset
                .min(lines.len().saturating_sub(self.review_height));
            let rows: Vec<_> = lines
                .into_iter()
                .skip(self.offset)
                .take(self.review_height)
                .map(ListItem::new)
                .collect();
            frame.render_widget(
                List::new(rows).style(Style::default().fg(theme::INK)),
                Rect::new(
                    inner.x,
                    review_y,
                    inner.width,
                    body_bottom.saturating_sub(review_y),
                ),
            );
            self.button(
                frame,
                Rect::new(inner.x, footer_y, 23.min(inner.width), 1),
                "Enter Apply to draft",
                KeyCode::Enter,
                true,
            );
            self.button(
                frame,
                Rect::new(
                    inner.x + 24.min(inner.width),
                    footer_y,
                    inner.width.saturating_sub(24),
                    1,
                ),
                "Esc Back",
                KeyCode::Esc,
                false,
            );
            frame.render_widget(
                Paragraph::new("↑↓ / PgUp PgDn review · Save and Check from the workspace")
                    .style(Style::default().fg(theme::MUTED)),
                Rect::new(inner.x, footer_y + 1, inner.width, 1),
            );
        } else {
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(Span::styled(
                        "Shape the initial atmosphere",
                        theme::heading(),
                    )),
                    Line::from("Cosine-squared warm bubbles: 0 < peak <= 10 K."),
                    Line::from("Applied once to initial state; engine checks grid placement."),
                ])
                .style(Style::default().fg(theme::MUTED)),
                Rect::new(inner.x, inner.y, inner.width, 3.min(available)),
            );
            let list_y = inner.y + 4;
            let list_height = body_bottom.saturating_sub(list_y).saturating_sub(2);
            if self.bubbles.is_empty() {
                frame.render_widget(Paragraph::new("No warm bubbles. Add one to explore initiation.\nCoordinates start from the config reference when available.").style(Style::default().fg(theme::INK)).wrap(Wrap { trim: true }), Rect::new(inner.x, list_y, inner.width, list_height));
            } else {
                let rows: Vec<_> = self
                    .bubbles
                    .iter()
                    .enumerate()
                    .map(|(i, b)| {
                        ListItem::new(format!(
                            "{}  {}, {}  · +{} K · r {} km",
                            i + 1,
                            coordinate_label(&b.values[0], "N", "S"),
                            coordinate_label(&b.values[1], "E", "W"),
                            b.values[5],
                            b.values[3]
                        ))
                    })
                    .collect();
                let mut state = ListState::default().with_selected(Some(self.selected));
                frame.render_stateful_widget(
                    List::new(rows)
                        .highlight_style(theme::selected())
                        .highlight_symbol("› ")
                        .style(Style::default().fg(theme::INK)),
                    Rect::new(inner.x, list_y, inner.width, list_height),
                    &mut state,
                );
                for row in 0..list_height as usize {
                    let index = state.offset() + row;
                    if index < self.bubbles.len() {
                        self.hits.push((
                            Rect::new(inner.x, list_y + row as u16, inner.width, 1),
                            Hit::Bubble(index),
                        ));
                    }
                }
            }
            if body_bottom >= inner.y + 6 {
                frame.render_widget(Paragraph::new("Unsupported runtime routes refuse this block.\nStart-time nests receive it; delayed nests get no fresh bubble.").style(Style::default().fg(theme::MUTED)), Rect::new(inner.x, body_bottom - 2, inner.width, 2));
            }
            let review = self.review_available();
            let empty = self.bubbles.is_empty();
            let mut buttons = Vec::new();
            if !empty && !review {
                buttons.push(("Enter Edit", KeyCode::Enter, true));
            }
            buttons.push(("A Add", KeyCode::Char('a'), empty && !review));
            if !empty {
                buttons.push(("X Remove", KeyCode::Char('x'), false));
            }
            if review {
                buttons.push(("R Review", KeyCode::Char('r'), true));
            }
            buttons.push(("Esc Cancel", KeyCode::Esc, false));
            let mut x = inner.x;
            for (label, key, primary) in buttons {
                self.button(
                    frame,
                    Rect::new(x, footer_y, inner.right().saturating_sub(x), 1),
                    label,
                    key,
                    primary,
                );
                x = x.saturating_add(label.len() as u16 + 3).min(inner.right());
            }
            frame.render_widget(
                Paragraph::new(if empty && !review {
                    "A adds a bubble · Changes return to your draft after review"
                } else {
                    "Enter/click edits · Remove last bubble disables perturbations"
                })
                .style(Style::default().fg(theme::MUTED)),
                Rect::new(inner.x, footer_y + 1, inner.width, 1),
            );
        }
        if let Some(notice) = &self.notice {
            frame.render_widget(
                Paragraph::new(notice.as_str()).style(theme::notice(true)),
                Rect::new(inner.x, body_bottom, inner.width, 1),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    const BASE: &str = "# Keep this file comment\n[projection]\nref_lat = 35.5\nref_lon = -97.5\n[shared]\nmp_physics = 10 # Keep physics\n[[domain]]\ngrid_id = 1\nparent_id = 0\n";
    const BUBBLE: &str = "\n# Keep the bubble comment\n[[perturbation.bubbles]]\ncenter_lat = 35.5\ncenter_lon = -97.5\ncenter_height_m = 1500\nradius_km = 10\ndepth_m = 1500\namplitude_k = 3.0 # Keep strength comment\n";

    fn press(form: &mut Form, code: KeyCode) -> Intent {
        form.key(KeyEvent::new(code, KeyModifiers::NONE))
    }
    fn values() -> [String; 7] {
        ["35.5", "-97.5", "1500", "10", "1500", "3", "false"].map(String::from)
    }

    #[test]
    fn no_change_and_cancel_preserve_original_bytes() {
        let text = format!("{BASE}{BUBBLE}");
        let mut form = Form::new(text.clone()).unwrap();
        assert_eq!(form.draft().unwrap(), text);
        press(&mut form, KeyCode::Enter);
        form.paste("9");
        assert!(matches!(press(&mut form, KeyCode::Esc), Intent::Keep));
        assert!(matches!(press(&mut form, KeyCode::Esc), Intent::Cancel));
        assert_eq!(form.draft().unwrap(), text);
    }

    #[test]
    fn edits_preserve_unrelated_settings_and_field_comments() {
        let text = format!("{BASE}{BUBBLE}");
        let mut form = Form::new(text).unwrap();
        form.bubbles[0].values[5] = "4.5".into();
        let output = form.draft().unwrap();
        assert!(output.starts_with(BASE));
        assert!(output.contains("# Keep the bubble comment"));
        assert!(output.contains("amplitude_k = 4.5 # Keep strength comment"));
        assert!(!output.contains("rh_preserve"));
        assert_eq!(
            output.parse::<DocumentMut>().unwrap()["shared"]["mp_physics"].as_integer(),
            Some(10)
        );
    }

    #[test]
    fn numeric_validation_matches_warm_bubble_contract_and_rejects_injection() {
        for (index, value) in [
            (0, "90.1"),
            (1, "-180.1"),
            (2, "-1"),
            (3, "0"),
            (4, "-1"),
            (5, "0"),
            (5, "10.01"),
            (5, "nan"),
            (5, "inf"),
            (5, "true"),
            (5, "3\n[shared]\nmp_physics=0"),
            (6, "1"),
        ] {
            let mut candidate = values();
            candidate[index] = value.into();
            assert!(parsed(&candidate).is_err(), "accepted {index} = {value}");
        }
        let mut boundary = values();
        boundary[2] = "0".into();
        boundary[5] = "10".into();
        assert!(parsed(&boundary).is_ok());
    }

    #[test]
    fn add_and_remove_are_reviewed_before_returning_any_draft() {
        let mut form = Form::new(BASE.into()).unwrap();
        assert!(matches!(press(&mut form, KeyCode::Char('a')), Intent::Keep));
        let Screen::Edit(edit) = &mut form.screen else {
            panic!()
        };
        assert_eq!(edit.values[0], "35.5");
        assert!(edit.values[2].is_empty());
        edit.values = values();
        assert!(matches!(press(&mut form, KeyCode::F(2)), Intent::Keep));
        assert!(matches!(press(&mut form, KeyCode::Char('r')), Intent::Keep));
        assert!(form.review_lines().iter().any(|l| l.contains("ADD bubble")));
        let Intent::Apply(text) = press(&mut form, KeyCode::Enter) else {
            panic!()
        };
        assert_eq!(
            text.parse::<DocumentMut>().unwrap()["perturbation"]["bubbles"]
                .as_array_of_tables()
                .unwrap()
                .len(),
            1
        );
        let mut form = Form::new(text).unwrap();
        press(&mut form, KeyCode::Char('x'));
        assert!(form
            .review_lines()
            .iter()
            .any(|l| l.contains("REMOVE bubble")));
        assert!(!form.draft().unwrap().contains("[perturbation"));
    }

    #[test]
    fn inline_bubble_array_retains_inline_representation_and_other_tables() {
        let text = format!("{BASE}\n[perturbation]\nbubbles = [{{center_lat = 35.5, center_lon = -97.5, center_height_m = 1500, radius_km = 10, depth_m = 1500, amplitude_k = 3}}] # Inline\n");
        let mut form = Form::new(text).unwrap();
        form.bubbles[0].values[5] = "4".into();
        form.bubbles.push(Bubble {
            source: None,
            values: values(),
        });
        let output = form.draft().unwrap();
        let doc = output.parse::<DocumentMut>().unwrap();
        assert_eq!(doc["perturbation"]["bubbles"].as_array().unwrap().len(), 2);
        assert!(output.contains("# Inline"));
        assert!(output.starts_with(BASE));
    }

    #[test]
    fn unknown_features_are_refused_and_location_is_never_invented() {
        assert!(Form::new(format!("{BASE}\n[perturbation]\nvortex = true\n")).is_err());
        assert!(Form::new(format!("{BASE}{BUBBLE}cold_pool = true\n")).is_err());
        let mut form = Form::new("[experiment]\nname='no-location'\n".into()).unwrap();
        press(&mut form, KeyCode::Char('a'));
        let Screen::Edit(edit) = &form.screen else {
            panic!()
        };
        assert!(edit.values[0].is_empty());
        assert!(edit.values[1].is_empty());
    }

    #[test]
    fn malformed_scalar_types_are_refused_and_toml_numeric_formatting_survives() {
        for (old, replacement) in [
            ("center_lat = 35.5", "center_lat = 'é'"),
            ("radius_km = 10", "radius_km = true"),
        ] {
            assert!(Form::new(format!("{BASE}{BUBBLE}").replace(old, replacement)).is_err());
        }
        let text =
            format!("{BASE}{BUBBLE}").replace("center_height_m = 1500", "center_height_m = 1_500");
        let mut form = Form::new(text.clone()).unwrap();
        assert_eq!(form.draft().unwrap(), text);
        form.bubbles[0].values[5] = "4".into();
        assert!(form.draft().unwrap().contains("center_height_m = 1_500"));
    }

    #[test]
    fn compact_editor_shows_each_complete_help_and_positive_amplitude_error() {
        let mut form = Form::new(format!("{BASE}{BUBBLE}")).unwrap();
        form.start_edit(Some(0));
        let mut terminal = Terminal::new(TestBackend::new(65, 20)).unwrap();
        for (index, help) in HELP.iter().enumerate() {
            let Screen::Edit(edit) = &mut form.screen else {
                panic!()
            };
            edit.selected = index;
            terminal
                .draw(|f| form.draw(f, Rect::new(0, 3, 65, 17)))
                .unwrap();
            let buffer = terminal.backend().buffer();
            let rendered = (4..19)
                .map(|y| (1..64).map(|x| buffer[(x, y)].symbol()).collect::<String>())
                .collect::<Vec<_>>()
                .join("\n");
            let normalized = rendered.split_whitespace().collect::<Vec<_>>().join(" ");
            assert!(
                normalized.contains(help),
                "missing help {index}: {normalized}"
            );
        }
        let Screen::Edit(edit) = &mut form.screen else {
            panic!()
        };
        edit.values[5] = "0".into();
        press(&mut form, KeyCode::F(2));
        assert!(form.notice.as_ref().unwrap().contains("greater than zero"));
        terminal.draw(|f| form.draw(f, f.area())).unwrap();
        let buffer = terminal.backend().buffer();
        let rendered: String = (0..20)
            .flat_map(|y| (0..65).map(move |x| buffer[(x, y)].symbol()))
            .collect();
        assert!(rendered.contains("must be greater than zero"));
    }

    #[test]
    fn mouse_actions_reach_review_and_apply_only_the_edited_bubble() {
        fn click(
            form: &mut Form,
            terminal: &mut Terminal<TestBackend>,
            find: impl Fn(&Hit) -> bool,
        ) -> Intent {
            terminal.draw(|f| form.draw(f, f.area())).unwrap();
            let rect = form.hits.iter().find(|(_, hit)| find(hit)).unwrap().0;
            form.mouse(MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: rect.x,
                row: rect.y,
                modifiers: KeyModifiers::NONE,
            })
        }
        let mut form = Form::new(format!("{BASE}{BUBBLE}")).unwrap();
        let mut terminal = Terminal::new(TestBackend::new(65, 20)).unwrap();
        assert!(matches!(
            click(&mut form, &mut terminal, |h| matches!(h, Hit::Bubble(0))),
            Intent::Keep
        ));
        click(&mut form, &mut terminal, |h| matches!(h, Hit::Field(5)));
        form.key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
        form.paste("4.5");
        click(&mut form, &mut terminal, |h| matches!(h, Hit::Field(6)));
        assert!(matches!(
            click(&mut form, &mut terminal, |h| matches!(
                h,
                Hit::Key(KeyCode::F(2))
            )),
            Intent::Keep
        ));
        assert!(matches!(
            click(&mut form, &mut terminal, |h| matches!(
                h,
                Hit::Key(KeyCode::Char('r'))
            )),
            Intent::Keep
        ));
        let Intent::Apply(output) = click(&mut form, &mut terminal, |h| {
            matches!(h, Hit::Key(KeyCode::Enter))
        }) else {
            panic!()
        };
        let doc = output.parse::<DocumentMut>().unwrap();
        let bubble = doc["perturbation"]["bubbles"]
            .as_array_of_tables()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(bubble["amplitude_k"].as_float(), Some(4.5));
        assert_eq!(bubble["rh_preserve"].as_bool(), Some(true));
        assert!(output.starts_with(BASE));
    }

    #[test]
    fn compact_review_wraps_and_scrolls_to_every_changed_value() {
        let mut form = Form::new(format!("{BASE}{BUBBLE}")).unwrap();
        form.bubbles[0].values[0] = "35.1234567890123456789012345678901234567890".into();
        for _ in 0..5 {
            form.bubbles.push(Bubble {
                source: None,
                values: values(),
            });
        }
        press(&mut form, KeyCode::Char('r'));
        let mut terminal = Terminal::new(TestBackend::new(65, 20)).unwrap();
        terminal.draw(|f| form.draw(f, f.area())).unwrap();
        let rows = form.review_rows();
        assert!(rows
            .iter()
            .all(|row| unicode_width::UnicodeWidthStr::width(row.as_str()) <= 63));
        assert!(rows.join("").contains(&form.bubbles[0].values[0]));
        press(&mut form, KeyCode::End);
        terminal.draw(|f| form.draw(f, f.area())).unwrap();
        assert_eq!(form.offset + form.review_height, rows.len());
        let buffer = terminal.backend().buffer();
        let rendered: String = (0..20)
            .flat_map(|y| (0..65).map(move |x| buffer[(x, y)].symbol()))
            .collect();
        assert!(rendered.contains("Preserve relative humidity: false"));
        assert!(rendered.contains("Enter Apply to draft"));
    }

    #[test]
    fn repeated_enter_does_not_apply_a_review() {
        let mut form = Form::new(format!("{BASE}{BUBBLE}")).unwrap();
        form.bubbles[0].values[5] = "4".into();
        press(&mut form, KeyCode::Char('r'));
        assert!(matches!(form.screen, Screen::Review));
        for kind in [KeyEventKind::Repeat, KeyEventKind::Release] {
            let mut event = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
            event.kind = kind;
            assert!(matches!(form.key(event), Intent::Keep));
        }
    }

    #[test]
    fn empty_state_emphasizes_add_and_removed_bubbles_still_reach_review() {
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let mut form = Form::new(BASE.into()).unwrap();
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal
                .draw(|f| form.draw(f, Rect::new(0, 3, width, height - 3)))
                .unwrap();
            let add = form
                .hits
                .iter()
                .find(|(_, h)| matches!(h, Hit::Key(KeyCode::Char('a'))))
                .unwrap()
                .0;
            assert_eq!(terminal.backend().buffer()[(add.x, add.y)].bg, theme::LEAF);
            assert!(!form
                .hits
                .iter()
                .any(|(_, h)| matches!(h, Hit::Key(KeyCode::Char('r' | 'x')))));
            assert!(form
                .hits
                .iter()
                .any(|(_, h)| matches!(h, Hit::Key(KeyCode::Esc))));
            press(&mut form, KeyCode::Char('r'));
            assert!(matches!(form.screen, Screen::List));

            let mut form = Form::new(format!("{BASE}{BUBBLE}")).unwrap();
            press(&mut form, KeyCode::Char('x'));
            terminal
                .draw(|f| form.draw(f, Rect::new(0, 3, width, height - 3)))
                .unwrap();
            let review = form
                .hits
                .iter()
                .find(|(_, h)| matches!(h, Hit::Key(KeyCode::Char('r'))))
                .unwrap()
                .0;
            assert_eq!(
                terminal.backend().buffer()[(review.x, review.y)].bg,
                theme::LEAF
            );
            assert!(!form
                .hits
                .iter()
                .any(|(_, h)| matches!(h, Hit::Key(KeyCode::Char('x')))));
            press(&mut form, KeyCode::Char('r'));
            let Intent::Apply(text) = press(&mut form, KeyCode::Enter) else {
                panic!()
            };
            assert!(!text.contains("[perturbation"));
        }
    }

    #[test]
    fn all_screens_retain_actions_and_mouse_targets_at_supported_sizes() {
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            for (screen, action) in [
                (0, "Enter Edit"),
                (1, "F2 Keep bubble"),
                (2, "Enter Apply to draft"),
            ] {
                let mut form = Form::new(format!("{BASE}{BUBBLE}")).unwrap();
                if screen == 1 {
                    form.start_edit(Some(0));
                }
                if screen == 2 {
                    form.screen = Screen::Review;
                }
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|f| form.draw(f, f.area())).unwrap();
                let buffer = terminal.backend().buffer();
                let rendered: String = (0..height)
                    .flat_map(|y| (0..width).map(move |x| buffer[(x, y)].symbol()))
                    .collect();
                assert!(rendered.contains(action), "{width}x{height}: {action}");
                assert!(form
                    .hits
                    .iter()
                    .all(|(r, _)| r.bottom() <= height && r.right() <= width));
            }
        }
    }
}
