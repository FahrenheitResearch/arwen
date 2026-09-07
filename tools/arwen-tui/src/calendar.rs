//! A source-aware UTC picker. Python's acquisition registry owns availability.
use super::{button_bar, key_hit, safe, Hit, HitRegion, INK, MUTED, TEAL};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Modifier, Style},
    widgets::{Paragraph, Wrap},
    Frame,
};
use serde_json::Value;
use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread::JoinHandle,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Date {
    year: i32,
    month: u8,
    day: u8,
}

impl Date {
    fn leap(year: i32) -> bool {
        year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
    }
    fn month_days(year: i32, month: u8) -> u8 {
        match month {
            2 => {
                if Self::leap(year) {
                    29
                } else {
                    28
                }
            }
            4 | 6 | 9 | 11 => 30,
            _ => 31,
        }
    }
    fn parse(text: &str) -> Option<Self> {
        let bytes = text.as_bytes();
        if bytes.len() < 10
            || bytes[4] != b'-'
            || bytes[7] != b'-'
            || !bytes[..10]
                .iter()
                .enumerate()
                .all(|(i, c)| matches!(i, 4 | 7) || c.is_ascii_digit())
        {
            return None;
        }
        let year = text[..4].parse().ok()?;
        let month = text[5..7].parse().ok()?;
        let day = text[8..10].parse().ok()?;
        if !(1..=9999).contains(&year)
            || !(1..=12).contains(&month)
            || day == 0
            || day > Self::month_days(year, month)
        {
            return None;
        }
        Some(Self { year, month, day })
    }
    fn today() -> Self {
        let mut days = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            / 86400;
        let mut date = Self {
            year: 1970,
            month: 1,
            day: 1,
        };
        while days >= if Self::leap(date.year) { 366 } else { 365 } {
            days -= if Self::leap(date.year) { 366 } else { 365 };
            date.year += 1;
        }
        while days >= u64::from(Self::month_days(date.year, date.month)) {
            days -= u64::from(Self::month_days(date.year, date.month));
            date.month += 1;
        }
        date.day = days as u8 + 1;
        date
    }
    fn text(self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
    fn stamp(self, hour: u8) -> String {
        format!("{}T{hour:02}", self.text())
    }
    fn weekday(self) -> usize {
        let year = self.year - 1;
        let mut days = year * 365 + year / 4 - year / 100 + year / 400 + i32::from(self.day) - 1;
        for month in 1..self.month {
            days += i32::from(Self::month_days(self.year, month));
        }
        (days % 7) as usize // 0001-01-01 was Monday.
    }
    fn shift_month(&mut self, delta: i32) {
        let next = (self.year * 12 + i32::from(self.month) - 1 + delta).clamp(12, 9999 * 12 + 11);
        self.year = next / 12;
        self.month = (next % 12 + 1) as u8;
        self.day = self.day.min(Self::month_days(self.year, self.month));
    }
    fn shift_days(&mut self, delta: i32) {
        for _ in 0..delta.unsigned_abs() {
            if delta > 0 {
                if self.day < Self::month_days(self.year, self.month) {
                    self.day += 1;
                } else if self.year < 9999 || self.month < 12 {
                    self.shift_month(1);
                    self.day = 1;
                }
            } else if self.day > 1 {
                self.day -= 1;
            } else if self.year > 1 || self.month > 1 {
                self.shift_month(-1);
                self.day = Self::month_days(self.year, self.month);
            }
        }
    }
}

fn parse_selection(value: &str, default_hour: u8) -> Result<(Date, u8), String> {
    let text = value.trim().trim_end_matches('Z');
    let date =
        Date::parse(text).ok_or("Enter a real date as YYYY-MM-DD or YYYY-MM-DDTHH (UTC).")?;
    if text.len() == 10 {
        return Ok((date, default_hour));
    }
    if text.as_bytes().get(10) != Some(&b'T')
        || !text
            .as_bytes()
            .get(11..13)
            .is_some_and(|digits| digits.iter().all(u8::is_ascii_digit))
    {
        return Err("Use YYYY-MM-DDTHH in UTC.".into());
    }
    let hour = text[11..13]
        .parse::<u8>()
        .ok()
        .filter(|v| *v < 24)
        .ok_or("UTC hour must be 00 through 23.")?;
    if !matches!(
        &text[13..],
        "" | ":00" | ":00:00" | "+00:00" | ":00+00:00" | ":00:00+00:00"
    ) {
        return Err("Choose an exact UTC hour; minutes and seconds must be zero.".into());
    }
    Ok((date, hour))
}

struct Query {
    child: Child,
    out: Option<JoinHandle<Vec<u8>>>,
    err: Option<JoinHandle<Vec<u8>>>,
    started: Instant,
    latest: bool,
}

fn drain(mut stream: impl Read + Send + 'static) -> JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut result = Vec::new();
        let mut block = [0u8; 8192];
        while let Ok(count) = stream.read(&mut block) {
            if count == 0 {
                break;
            }
            if result.len() < 1024 * 1024 {
                result.extend_from_slice(&block[..count]);
            }
        }
        result
    })
}

impl Query {
    fn spawn(
        python: &Path,
        cwd: &Path,
        arguments: &[String],
        latest: bool,
    ) -> Result<Self, String> {
        let mut command = Command::new(python);
        command
            .args(["-m", "gpuwm.source_availability"])
            .args(arguments)
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("GPUWM_NO_LOCAL_GPU", "1")
            .env("PYTHONDONTWRITEBYTECODE", "1");
        if latest {
            command.arg("--resolve-latest");
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command
            .spawn()
            .map_err(|e| format!("Could not query source dates: {e}"))?;
        let out = Some(drain(child.stdout.take().unwrap()));
        let err = Some(drain(child.stderr.take().unwrap()));
        Ok(Self {
            child,
            out,
            err,
            started: Instant::now(),
            latest,
        })
    }
    fn poll(&mut self) -> Option<Result<Value, String>> {
        match self.child.try_wait() {
            Ok(None)
                if self.started.elapsed()
                    < Duration::from_secs(if self.latest { 180 } else { 30 }) =>
            {
                None
            }
            Ok(None) => {
                let _ = self.child.kill();
                let _ = self.child.wait();
                Some(Err("The availability check timed out. Pick an explicit date or retry Latest; no date was changed.".into()))
            }
            Err(error) => Some(Err(format!(
                "Could not observe the availability check: {error}"
            ))),
            Ok(Some(status)) => {
                let stdout = self
                    .out
                    .take()
                    .and_then(|t| t.join().ok())
                    .unwrap_or_default();
                let stderr = self
                    .err
                    .take()
                    .and_then(|t| t.join().ok())
                    .unwrap_or_default();
                if !status.success() {
                    return Some(Err(format!(
                        "Date check failed: {}",
                        String::from_utf8_lossy(&stderr).trim()
                    )));
                }
                Some(
                    serde_json::from_slice(&stdout)
                        .map_err(|e| format!("The source date query returned invalid data: {e}")),
                )
            }
        }
    }
}

impl Drop for Query {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct Form {
    python: PathBuf,
    cwd: PathBuf,
    arguments: Vec<String>,
    query: Option<Query>,
    pub metadata: Option<Value>,
    date: Date,
    hour: u8,
    text: String,
    manual: bool,
    year_entry: Option<String>,
    initial_date: bool,
    pub notice: String,
}

impl Form {
    pub fn new(python: &Path, cwd: &Path, arguments: Vec<String>, initial: &str) -> Self {
        let parsed = parse_selection(initial, 0).ok();
        let (date, hour) = parsed.unwrap_or((Date::today(), 0));
        let mut form = Self { python: python.to_owned(), cwd: cwd.to_owned(), arguments,
            query: None, metadata: None, date, hour, text: date.stamp(hour), manual: false, year_entry: None,
            initial_date: parsed.is_some(), notice: "Loading the source's cycle schedule… You can still type a date; Esc returns to the guide.".into() };
        form.start(false);
        form
    }
    #[cfg(test)]
    pub fn fixture(metadata: Value, initial: &str) -> Self {
        let (date, hour) = parse_selection(initial, 0).unwrap();
        Self {
            python: PathBuf::new(),
            cwd: PathBuf::new(),
            arguments: Vec::new(),
            query: None,
            metadata: Some(metadata),
            date,
            hour,
            text: date.stamp(hour),
            manual: false,
            year_entry: None,
            initial_date: true,
            notice: "Choose an archive date or check the latest complete cycle.".into(),
        }
    }
    fn start(&mut self, latest: bool) {
        if latest
            && self
                .metadata
                .as_ref()
                .is_some_and(|v| v["latest_supported"] == false)
        {
            self.notice = "Latest is not declared for this source and period. Choose an explicit date or adjust the duration.".into();
            return;
        }
        self.query = None; // Closing/replacing a picker cancels its query process.
        match Query::spawn(&self.python, &self.cwd, &self.arguments, latest) {
            Ok(query) => {
                self.query = Some(query);
                self.notice = if latest { "Checking the newest period with the acquisition service… Esc cancels; no forecast is started." } else { "Loading source dates…" }.into();
            }
            Err(error) => self.notice = error,
        }
    }
    pub fn latest(&mut self) {
        self.start(true);
    }
    pub fn poll(&mut self) {
        let Some(query) = &mut self.query else {
            return;
        };
        let latest = query.latest;
        let Some(result) = query.poll() else {
            return;
        };
        self.query = None;
        match result {
            Err(error) => self.notice = error,
            Ok(value) => {
                if value["schema"] != "gpuwm.source-availability.v1" {
                    self.notice = "This Python installation returned an unsupported source-date schema. Enter a date in the guide.".into();
                    return;
                }
                let selection = if latest {
                    value["selected_cycle"].as_str()
                } else if !self.initial_date {
                    value["latest_candidate"].as_str()
                } else {
                    None
                };
                if let Some((date, hour)) = selection.and_then(|v| parse_selection(v, 0).ok()) {
                    self.date = date;
                    self.hour = hour;
                    self.text = date.stamp(hour);
                    self.manual = false;
                    self.year_entry = None;
                }
                self.notice = if latest {
                    if value["resolution"]["basis"] == "provider_object_probe" {
                        format!("Latest complete: {} UTC. Required final-hour objects passed the provider probe. Use date keeps this exact cycle.", self.text)
                    } else {
                        format!("Latest expected: {} UTC. Publication schedule only; the service verifies the full analysis request during acquisition.", self.text)
                    }
                } else {
                    if value["probeable"] == true {
                        "Choose an expected UTC cycle, or use Latest to check required final-hour objects. Acquisition still checks the complete input set.".into()
                    } else {
                        "Choose an expected UTC period. Latest follows this source's publication schedule; acquisition verifies the complete request.".into()
                    }
                };
                self.metadata = Some(value);
            }
        }
    }
    fn hours(&self) -> Vec<u8> {
        self.metadata
            .as_ref()
            .and_then(|v| v["cycle_hours"].as_array())
            .map(|v| {
                v.iter()
                    .filter_map(|h| h.as_u64().map(|h| h as u8))
                    .collect()
            })
            .unwrap_or_default()
    }
    fn selection_error(&self, date: Date, hour: u8) -> Option<String> {
        let Some(metadata) = &self.metadata else {
            return Some(
                "Wait for the source schedule, or return to the guide for manual date entry."
                    .into(),
            );
        };
        let hours = self.hours();
        if !metadata["cycle_grid"].is_null() && !hours.contains(&hour) {
            return Some("Choose one of the UTC cycles that covers the requested duration.".into());
        }
        let stamp = date.stamp(hour);
        if let Some(start) = metadata["earliest"].as_str() {
            if stamp.as_str() < start {
                return Some(format!("This acquisition layout begins {start} UTC. Earlier history needs another input route."));
            }
        }
        if let Some(end) = metadata["latest_candidate"].as_str() {
            if stamp.as_str() > end {
                return Some(format!(
                    "This period is later than the expected publication boundary: {end} UTC."
                ));
            }
        }
        None
    }
    fn cancel_latest(&mut self) {
        if self.query.as_ref().is_some_and(|q| q.latest) {
            self.query = None;
            self.notice = "Latest check canceled. Choose the intended date and hour.".into();
            if self.metadata.is_none() {
                self.start(false);
            }
        }
    }
    fn sync(&mut self) {
        self.cancel_latest();
        self.manual = false;
        self.year_entry = None;
        self.initial_date = true;
        self.text = self.date.stamp(self.hour);
    }
    pub fn day(&mut self, day: u8) {
        self.date.day = day
            .min(Date::month_days(self.date.year, self.date.month))
            .max(1);
        self.sync();
    }
    pub fn hour(&mut self, hour: u8) {
        if self.manual {
            if let Ok((date, _)) = parse_selection(&self.text, self.hour) {
                self.date = date;
            }
        }
        self.hour = hour.min(23);
        self.sync();
    }
    pub fn paste(&mut self, text: &str) {
        self.cancel_latest();
        if self.year_entry.is_some() {
            self.year_entry = Some(text.trim().into());
            return;
        }
        self.manual = true;
        self.initial_date = true;
        self.text = text.trim().into();
    }
    fn begin_year(&mut self) {
        self.cancel_latest();
        self.initial_date = true;
        self.year_entry = Some(String::new());
        self.notice = "Type a year and press Enter to jump directly. The source's available years still apply.".into();
    }
    fn use_year(&mut self, text: &str) {
        let year = text.parse::<i32>().ok().filter(|year| (1..=9999).contains(year));
        let Some(year) = year else {
            self.notice = "Enter a year from 0001 through 9999, then press Enter.".into();
            return;
        };
        if let Some(metadata) = &self.metadata {
            let first = metadata["earliest"].as_str().and_then(Date::parse);
            let last = metadata["latest_candidate"].as_str().and_then(Date::parse);
            if first.is_some_and(|date| year < date.year) || last.is_some_and(|date| year > date.year) {
                self.notice = format!("This source's expected years are {} through {}. Choose a year in that range.",
                    first.map(|date| date.year.to_string()).unwrap_or_else(|| "undeclared".into()),
                    last.map(|date| date.year.to_string()).unwrap_or_else(|| "undeclared".into()));
                return;
            }
        }
        self.date.year = year;
        self.date.day = self.date.day.min(Date::month_days(year, self.date.month));
        self.sync();
        self.notice = format!("Jumped to {year}. Choose the month, day and UTC hour, then Use date.");
    }
    pub fn key(&mut self, key: KeyEvent) -> Option<String> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if let Some(mut year) = self.year_entry.take() {
            match key.code {
                KeyCode::Enter => {
                    self.year_entry = Some(year.clone());
                    self.use_year(&year);
                    return None;
                }
                KeyCode::Backspace => { year.pop(); }
                KeyCode::Char('u' | 'U') if ctrl => year.clear(),
                KeyCode::Char(c) if !ctrl && c.is_ascii_digit() => {
                    if year.len() < 4 { year.push(c); }
                }
                _ => {}
            }
            self.year_entry = Some(year);
            return None;
        }
        match key.code {
            KeyCode::F(2) => self.begin_year(),
            KeyCode::Char('y' | 'Y') if !self.manual => self.begin_year(),
            KeyCode::F(4) => self.start(true),
            KeyCode::Enter => {
                if self.query.as_ref().is_some_and(|q| q.latest) {
                    self.notice =
                        "The latest-period check is still running; Esc cancels it.".into();
                    return None;
                }
                match parse_selection(&self.text, self.hour) {
                    Err(error) => self.notice = error,
                    Ok((date, hour)) => {
                        if let Some(error) = self.selection_error(date, hour) {
                            self.notice = error;
                        } else {
                            return Some(date.stamp(hour));
                        }
                    }
                }
            }
            KeyCode::Char('u' | 'U') if ctrl => {
                self.cancel_latest();
                self.initial_date = true;
                self.manual = true;
                self.text.clear();
            }
            KeyCode::Backspace => {
                self.cancel_latest();
                self.initial_date = true;
                self.manual = true;
                self.text.pop();
            }
            KeyCode::Char('e' | 'E') if !self.manual => {
                self.cancel_latest();
                self.initial_date = true;
                self.manual = true;
            }
            KeyCode::Char('l' | 'L') if !self.manual => self.start(true),
            KeyCode::Left if !self.manual => {
                self.date.shift_days(-1);
                self.sync();
            }
            KeyCode::Right if !self.manual => {
                self.date.shift_days(1);
                self.sync();
            }
            KeyCode::Up if !self.manual => {
                self.date.shift_days(-7);
                self.sync();
            }
            KeyCode::Down if !self.manual => {
                self.date.shift_days(7);
                self.sync();
            }
            KeyCode::PageUp => {
                self.date.shift_month(if ctrl { -12 } else { -1 });
                self.sync();
            }
            KeyCode::PageDown => {
                self.date.shift_month(if ctrl { 12 } else { 1 });
                self.sync();
            }
            KeyCode::Tab | KeyCode::BackTab => {
                if self.manual {
                    if let Ok((date, hour)) = parse_selection(&self.text, self.hour) {
                        self.date = date;
                        self.hour = hour;
                    }
                    self.sync();
                } else {
                    let hours = self.hours();
                    if !hours.is_empty() {
                        let current = hours.iter().position(|h| *h == self.hour).unwrap_or(0);
                        let next = if key.code == KeyCode::BackTab {
                            (current + hours.len() - 1) % hours.len()
                        } else {
                            (current + 1) % hours.len()
                        };
                        self.hour(hours[next]);
                    }
                }
            }
            KeyCode::Char(c) if !ctrl => {
                self.cancel_latest();
                if !self.manual {
                    self.text.clear();
                    self.manual = true;
                }
                self.initial_date = true;
                self.text.push(c);
            }
            _ => {}
        }
        None
    }
    pub fn draw(&self, frame: &mut Frame, hits: &mut Vec<HitRegion>, body: Rect, buttons: Rect) {
        let compact = body.height < 14;
        let hours = self.hours();
        let slots = (body.width / 5).max(1);
        let hour_rows = (hours.len() as u16).div_ceil(slots).clamp(1, 2);
        let parts = Layout::vertical([
            Constraint::Length(if compact { 1 } else { 2 }),
            Constraint::Length(8),
            Constraint::Length(hour_rows),
            Constraint::Length(1),
            Constraint::Length(if compact { 0 } else { 1 }),
            Constraint::Min(0),
        ])
        .split(body);
        let metadata = self.metadata.as_ref();
        let source = metadata
            .and_then(|v| v["display_name"].as_str())
            .unwrap_or("Source dates");
        let start = metadata
            .and_then(|v| v["earliest"].as_str())
            .unwrap_or("earliest date unknown");
        let end = metadata
            .and_then(|v| v["latest_candidate"].as_str())
            .unwrap_or("checking schedule");
        let heading = if compact {
            format!(
                "{} · UTC · expected {} to {}",
                metadata
                    .and_then(|v| v["source_id"].as_str())
                    .unwrap_or(source),
                start.get(..10).unwrap_or("unknown"),
                end.get(..10).unwrap_or("unknown")
            )
        } else {
            format!(
                "{} · UTC · {} h period\nExpected range: {} → {}",
                safe(source),
                metadata
                    .map(|v| v["hours"].to_string())
                    .unwrap_or_else(|| "?".into()),
                start,
                end
            )
        };
        frame.render_widget(
            Paragraph::new(heading).style(Style::default().fg(INK)),
            parts[0],
        );
        let calendar = parts[1];
        let width = calendar.width.min(56);
        let left = calendar.x + (calendar.width.saturating_sub(width)) / 2;
        let month = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ][usize::from(self.date.month - 1)];
        frame.render_widget(
            Paragraph::new(format!("{month} [{}]", self.date.year))
                .alignment(Alignment::Center)
                .style(Style::default().fg(TEAL).add_modifier(Modifier::BOLD)),
            Rect::new(left + 11, calendar.y, width.saturating_sub(22), 1),
        );
        frame.render_widget(
            Paragraph::new("‹").style(Style::default().fg(TEAL)),
            Rect::new(left + 8, calendar.y, 3, 1),
        );
        frame.render_widget(
            Paragraph::new("›").style(Style::default().fg(TEAL)),
            Rect::new(left + width.saturating_sub(10), calendar.y, 3, 1),
        );
        hits.push(HitRegion {
            area: Rect::new(left + 8, calendar.y, 3, 1),
            action: key_hit(KeyCode::PageUp),
        });
        hits.push(HitRegion {
            area: Rect::new(left + width.saturating_sub(10), calendar.y, 3, 1),
            action: key_hit(KeyCode::PageDown),
        });
        hits.push(HitRegion {
            area: Rect::new(left + 11, calendar.y, width.saturating_sub(22), 1),
            action: key_hit(KeyCode::F(2)),
        });
        for (label, x, key) in [("« Year", left, KeyCode::PageUp),
                               ("Year »", left + width.saturating_sub(7), KeyCode::PageDown)] {
            let area = Rect::new(x, calendar.y, 7, 1);
            frame.render_widget(Paragraph::new(label).style(Style::default().fg(TEAL)), area);
            hits.push(HitRegion { area, action: Hit::Key(key, KeyModifiers::CONTROL) });
        }
        let column = (width / 7).max(1);
        for (index, name) in ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
            .iter()
            .enumerate()
        {
            frame.render_widget(
                Paragraph::new(*name).style(Style::default().fg(MUTED)),
                Rect::new(left + index as u16 * column, calendar.y + 1, column, 1),
            );
        }
        let first = Date {
            day: 1,
            ..self.date
        }
        .weekday();
        for day in 1..=Date::month_days(self.date.year, self.date.month) {
            let index = first + usize::from(day - 1);
            let y = calendar.y + 2 + (index / 7) as u16;
            if y >= calendar.bottom() {
                continue;
            }
            let rect = Rect::new(
                left + (index % 7) as u16 * column,
                y,
                column.saturating_sub(1).max(1),
                1,
            );
            let date = Date { day, ..self.date };
            let expected = self
                .hours()
                .iter()
                .any(|hour| self.selection_error(date, *hour).is_none());
            let mut style = Style::default().fg(if expected { INK } else { MUTED });
            if day == self.date.day {
                style = style.bg(TEAL).fg(super::BACK).add_modifier(Modifier::BOLD);
            }
            let label = if day == self.date.day {
                format!("[{day:2}]")
            } else {
                format!(" {day:2}")
            };
            frame.render_widget(Paragraph::new(label).style(style), rect);
            hits.push(HitRegion {
                area: rect,
                action: Hit::CalendarDay(day),
            });
        }
        for (index, hour) in hours.iter().enumerate() {
            let row = index as u16 / slots;
            if row >= parts[2].height {
                break;
            }
            let rect = Rect::new(
                parts[2].x + index as u16 % slots * 5,
                parts[2].y + row,
                5,
                1,
            );
            let style = if *hour == self.hour {
                Style::default().fg(super::BACK).bg(TEAL)
            } else {
                Style::default().fg(INK)
            };
            let label = if *hour == self.hour {
                format!("[{hour:02}Z]")
            } else {
                format!(" {hour:02}Z ")
            };
            frame.render_widget(Paragraph::new(label).style(style), rect);
            hits.push(HitRegion {
                area: rect,
                action: Hit::CalendarHour(*hour),
            });
        }
        let input = self.year_entry.as_ref().unwrap_or(&self.text);
        let prefix = if self.year_entry.is_some() { "Year > " } else { "> " };
        let visible = super::guide_value_tail(input,
            usize::from(parts[3].width.saturating_sub(prefix.len() as u16 + 1)));
        frame.render_widget(
            Paragraph::new(format!("{prefix}{}", safe(&visible)))
                .style(Style::default().fg(INK).bg(super::theme::RAISED)),
            parts[3],
        );
        hits.push(HitRegion {
            area: parts[3],
            action: key_hit(KeyCode::Char('e')),
        });
        if (self.manual || self.year_entry.is_some()) && parts[3].width > prefix.len() as u16 + 1 {
            use unicode_width::UnicodeWidthStr;
            frame.set_cursor_position((
                parts[3].x + prefix.len() as u16 + UnicodeWidthStr::width(visible.as_str()) as u16,
                parts[3].y,
            ));
        }
        frame.render_widget(
            Paragraph::new("Click [year] or Y: jump to year · Arrows day · Tab hour · PgUp/Dn month · E date")
                .style(Style::default().fg(MUTED)),
            parts[4],
        );
        if let Some(metadata) = metadata {
            let mut details = Vec::new();
            if let Some(transports) = metadata["transports"].as_array() {
                for row in transports {
                    let name = row["transport"].as_str().unwrap_or("transport");
                    let range = if let Some(start) = row["record_start"].as_str() {
                        format!("current archive layout from {start} UTC")
                    } else if let Some(hours) = row["retention_hours"].as_f64() {
                        format!("rolling retention about {hours} hours")
                    } else {
                        "historical range is not declared".into()
                    };
                    details.push(format!("{name}: {range}"));
                }
            }
            if let Some(required) = metadata["requirements"]
                .as_array()
                .filter(|v| !v.is_empty())
            {
                details.push(format!(
                    "Account setup: {}",
                    required
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if let Some(notes) = metadata["notes"].as_array() {
                details.extend(notes.iter().filter_map(Value::as_str).map(str::to_owned));
            }
            frame.render_widget(
                Paragraph::new(details.join("\n"))
                    .wrap(Wrap { trim: false })
                    .style(Style::default().fg(MUTED)),
                parts[5],
            );
        }
        let latest = if metadata.is_some_and(|v| v["probeable"] == false) {
            "Latest expected (F4)"
        } else {
            "Latest complete (F4)"
        };
        button_bar(
            frame,
            hits,
            buttons,
            &[
                ("Back (Esc)", key_hit(KeyCode::Esc)),
                (latest, key_hit(KeyCode::F(4))),
                ("Use date (Enter)", key_hit(KeyCode::Enter)),
            ],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calendar_navigation_handles_leap_years_and_month_ends() {
        let mut date = Date::parse("2024-02-29").unwrap();
        date.shift_days(1);
        assert_eq!(date.text(), "2024-03-01");
        date.shift_days(-1);
        assert_eq!(date.text(), "2024-02-29");
        date.shift_month(12);
        assert_eq!(date.text(), "2025-02-28");
        assert!(Date::parse("2100-02-29").is_none());
        assert!(Date::parse("2000-02-29").is_some());
        assert_eq!(Date::parse("2026-09-01").unwrap().weekday(), 1);
        assert!(parse_selection("2026-09-01T06:15", 0).is_err());
        assert_eq!(parse_selection("2026-09-01", 6).unwrap().1, 6);
    }
    #[test]
    fn manual_entry_and_cycle_selection_enforce_the_loaded_source_window() {
        let mut form = Form::fixture(
            serde_json::json!({"cycle_hours":[0,6,12,18],"cycle_grid":{},"earliest":"2021-03-22T12","latest_candidate":"2026-09-06T06"}),
            "2026-09-05T06",
        );
        form.paste("2020-01-01T00");
        assert!(form
            .key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .is_none());
        assert!(form.notice.contains("begins"));
        form.paste("2026-09-05T07");
        assert!(form
            .key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .is_none());
        form.paste("2026-09-05T12");
        assert_eq!(
            form.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            Some("2026-09-05T12".into())
        );
        form.hour(18);
        assert_eq!(form.text, "2026-09-05T18");
    }

    #[test]
    fn year_entry_jumps_decades_and_preserves_month_day_and_hour() {
        let mut form = Form::fixture(serde_json::json!({
            "cycle_hours":[0,6,12,18], "cycle_grid":{},
            "earliest":"1940-01-01T00", "latest_candidate":"2026-09-06T06"
        }), "2026-05-03T12");
        form.key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE));
        for digit in "1999".chars() {
            form.key(KeyEvent::new(KeyCode::Char(digit), KeyModifiers::NONE));
        }
        assert!(form.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)).is_none());
        assert_eq!(form.text, "1999-05-03T12");
        assert!(form.year_entry.is_none());
        assert_eq!(form.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
                   Some("1999-05-03T12".into()));
    }

    #[test]
    fn pasted_year_handles_leap_day_and_rejects_unavailable_years_without_moving() {
        let mut form = Form::fixture(serde_json::json!({
            "cycle_hours":[0,6,12,18], "cycle_grid":{},
            "earliest":"1940-01-01T00", "latest_candidate":"2026-09-06T06"
        }), "2024-02-29T06");
        form.key(KeyEvent::new(KeyCode::F(2), KeyModifiers::NONE));
        form.paste("1999");
        form.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(form.text, "1999-02-28T06");
        form.key(KeyEvent::new(KeyCode::F(2), KeyModifiers::NONE));
        form.paste("1939");
        form.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(form.notice.contains("1940 through 2026"));
        assert_eq!(form.text, "1999-02-28T06");
        form.paste("not a year");
        form.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(form.notice.contains("Enter a year"));
        assert_eq!(form.text, "1999-02-28T06");
    }

    #[test]
    fn year_header_and_one_year_buttons_are_clickable_at_supported_sizes() {
        use ratatui::{backend::TestBackend, Terminal};
        let form = Form::fixture(serde_json::json!({"cycle_hours":[0,6,12,18]}), "2026-09-06T12");
        for (width, height) in [(51, 14), (66, 18), (100, 29)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height + 2)).unwrap();
            let mut hits = Vec::new();
            terminal.draw(|frame| form.draw(frame, &mut hits,
                Rect::new(0,0,width,height), Rect::new(0,height,width,2))).unwrap();
            for code in [KeyCode::PageUp, KeyCode::PageDown] {
                assert!(hits.iter().any(|hit| matches!(hit.action, Hit::Key(k, m)
                    if k == code && m.contains(KeyModifiers::CONTROL))));
            }
            assert!(hits.iter().any(|hit| matches!(hit.action, Hit::Key(KeyCode::F(2), _))));
            assert!(hits.iter().all(|hit| hit.area.right() <= width && hit.area.bottom() <= height + 2));
        }
    }
}
