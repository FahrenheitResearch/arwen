//! A keyboard and mouse browser for weather families and research recipes.
use crate::{button_bar, clickable_list, key_hit, panel, research as data, safe, workflows::{Intent, Route, MODES}, Hit, HitRegion, INK, MUTED, TEAL};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{layout::{Constraint, Layout, Rect}, style::{Modifier, Style}, text::Line,
    widgets::{Block, Clear, Paragraph, Wrap}, Frame};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Entry { Family(usize), Submode(usize), Leaf(usize), Config(usize) }
impl Entry {
    fn row(self) -> &'static serde_json::Value {
        match self {
            Self::Family(i) => data::rows("families").iter().find(|v| v["id"] == MODES[i].id).expect("family catalog"),
            Self::Submode(i) => &data::rows("submodes")[i],
            Self::Leaf(i) => &data::rows("leaves")[i],
            Self::Config(i) => &data::rows("configurations")[i],
        }
    }
    fn title(self) -> &'static str { data::text(self.row(), "title") }
    fn description(self) -> String {
        if matches!(self, Self::Config(_)) { data::detail(self.row()) }
        else if let Self::Family(i) = self { format!("{}\n\n{}\n\n{}", MODES[i].subtitle, data::text(self.row(), "description"), MODES[i].guidance) }
        else { data::text(self.row(), "description").to_owned() }
    }
    fn search_text(self) -> String {
        fn base(row: &'static serde_json::Value) -> String {
            format!("{} {} {} {} {} {} {}", row["title"], row["description"], row["keywords"], row["research_question"], row["research_objective"], row["id"], data::method(row))
        }
        let mut text = format!("{} {}", base(self.row()), self.label());
        match self {
            Self::Config(_) => {
                if let Some(leaf) = data::rows("leaves").iter().find(|v| v["id"] == self.row()["leaf_id"]) { text.push_str(&base(leaf)); }
            }
            Self::Leaf(_) => {
                for row in data::rows("configurations").iter().filter(|v| v["leaf_id"] == self.row()["id"]) { text.push_str(&base(row)); }
            }
            Self::Submode(_) => {
                for (i, _) in data::rows("leaves").iter().enumerate().filter(|(_, v)| v["submode_id"] == self.row()["id"]) { text.push_str(&Self::Leaf(i).search_text()); }
            }
            Self::Family(i) => {
                for (j, _) in data::rows("submodes").iter().enumerate().filter(|(_, v)| v["family_id"] == MODES[i].id) { text.push_str(&Self::Submode(j).search_text()); }
            }
        }
        text
    }
    fn label(self) -> String {
        match self {
            Self::Family(_) => self.title().into(),
            Self::Submode(_) => format!("{}  ›", self.title()),
            Self::Leaf(_) => format!("{}  · {} setups ›", self.title(), data::strings(self.row(), "configuration_ids").len()),
            Self::Config(_) => format!("{}  · {}", self.title(), data::method(self.row())),
        }
    }
}

#[derive(Default)]
pub struct Browser {
    pub query: String,
    pub selected: usize,
    pub details: bool,
    pub offset: u16,
    pub trail: Vec<Entry>,
    preview: bool,
    notice: String,
    has_config: std::cell::Cell<bool>,
    max_offset: std::cell::Cell<u16>,
}
impl Browser {
    fn matches_text(&self, text: &str) -> bool {
        let text = text.to_lowercase();
        self.query.to_lowercase().split_whitespace().all(|word| text.contains(word))
    }
    pub fn matches(&self) -> Vec<usize> {
        MODES.iter().enumerate().filter_map(|(i, mode)| {
            let text = format!("{} {} {} {}", mode.title, mode.subtitle, mode.keywords, Entry::Family(i).search_text());
            self.matches_text(&text).then_some(i)
        }).collect()
    }
    pub fn index(&self) -> Option<usize> {
        match self.trail.first() { Some(Entry::Family(i)) => Some(*i), _ => self.matches().get(self.selected).copied() }
    }
    fn add_configs(entries: &mut Vec<Entry>, row: &'static serde_json::Value, key: &str) {
        for id in data::strings(row, key) {
            if let Some(i) = data::rows("configurations").iter().position(|v| v["id"] == id) { entries.push(Entry::Config(i)); }
        }
    }
    pub fn entries(&self) -> Vec<Entry> {
        let mut entries = Vec::new();
        match self.trail.last().copied() {
            None => return self.matches().into_iter().map(Entry::Family).collect(),
            Some(Entry::Family(i)) => {
                entries.extend(data::rows("submodes").iter().enumerate().filter_map(|(j, row)| (row["family_id"] == MODES[i].id).then_some(Entry::Submode(j))));
                Self::add_configs(&mut entries, Entry::Family(i).row(), "recommended_config_ids");
            }
            Some(Entry::Submode(i)) => {
                let parent = &data::rows("submodes")[i];
                entries.extend(data::rows("leaves").iter().enumerate().filter_map(|(j, row)| (row["submode_id"] == parent["id"]).then_some(Entry::Leaf(j))));
                Self::add_configs(&mut entries, parent, "recommended_config_ids");
            }
            Some(Entry::Leaf(i)) => Self::add_configs(&mut entries, &data::rows("leaves")[i], "configuration_ids"),
            Some(Entry::Config(_)) => {}
        }
        entries.into_iter().filter(|e| self.matches_text(&e.search_text())).collect()
    }
    pub fn choose(&mut self, index: usize) {
        if index < MODES.len() {
            self.trail = vec![Entry::Family(index)];
            self.query.clear(); self.selected = 0; self.offset = 0; self.details = true; self.preview = false; self.notice.clear();
        }
    }
    pub fn choose_config(&mut self, index: usize) {
        if let Some(row) = data::config(index) {
            if let Some(family) = data::family_for_config(row).and_then(|id| MODES.iter().position(|m| m.id == id)) {
                self.choose(family);
                let leaf = data::rows("leaves").iter().position(|v| v["id"] == row["leaf_id"]).unwrap();
                let submode = data::rows("submodes").iter().position(|v| v["id"] == data::rows("leaves")[leaf]["submode_id"]).unwrap();
                self.trail.extend([Entry::Submode(submode), Entry::Leaf(leaf), Entry::Config(index)]);
            }
        }
    }
    pub fn choose_row(&mut self, index: usize) {
        if let Some(entry) = self.entries().get(index).copied() {
            self.trail.push(entry);
            self.query.clear(); self.selected = 0; self.offset = 0; self.details = true; self.preview = false;
        }
    }
    pub fn paste(&mut self, text: &str) {
        if !self.preview && !matches!(self.trail.last(), Some(Entry::Config(_))) {
            for c in text.chars().filter(|c| !c.is_control()).take(160usize.saturating_sub(self.query.chars().count())) { self.query.push(c); }
            self.selected = 0;
        }
    }
    pub fn key(&mut self, key: KeyEvent) -> Intent {
        if key.code == KeyCode::Esc {
            if self.preview { self.preview = false; self.offset = 0; return Intent::Keep; }
            let Some(previous) = self.trail.pop() else { return Intent::Close; };
            self.query.clear(); self.offset = 0; self.details = !self.trail.is_empty();
            self.selected = self.entries().iter().position(|v| *v == previous).unwrap_or(0);
            return Intent::Keep;
        }
        if self.preview {
            match key.code {
                KeyCode::Up => self.offset = self.offset.saturating_sub(1),
                KeyCode::Down => self.offset = self.offset.saturating_add(1).min(self.max_offset.get()),
                KeyCode::PageUp => self.offset = self.offset.saturating_sub(8),
                KeyCode::PageDown => self.offset = self.offset.saturating_add(8).min(self.max_offset.get()),
                KeyCode::Home => self.offset = 0,
                KeyCode::End => self.offset = self.max_offset.get(),
                KeyCode::Enter => self.choose_row(self.selected),
                KeyCode::F(4) => { self.preview = false; self.offset = 0; },
                _ => {}
            }
            return Intent::Keep;
        }
        if let Some(Entry::Config(i)) = self.trail.last().copied() {
            let row = data::config(i).unwrap();
            if key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) { return Intent::Keep; }
            if !self.has_config.get() && matches!(key.code, KeyCode::Char('p' | 'P' | 'i' | 'I' | 't' | 'T')) {
                self.notice = "Open a configuration to use its plots, tracking or initial-state settings. Esc keeps this research position.".into();
                return Intent::Keep;
            }
            match key.code {
                KeyCode::Up => self.offset = self.offset.saturating_sub(1),
                KeyCode::Down => self.offset = self.offset.saturating_add(1).min(self.max_offset.get()),
                KeyCode::PageUp => self.offset = self.offset.saturating_sub(8),
                KeyCode::PageDown => self.offset = self.offset.saturating_add(8).min(self.max_offset.get()),
                KeyCode::Home => self.offset = 0,
                KeyCode::End => self.offset = self.max_offset.get(),
                KeyCode::Enter | KeyCode::Char('n' | 'N') => return Intent::StartConfig(i, data::setup_route(row)),
                KeyCode::Char('r' | 'R') => return Intent::StartConfig(i, Route::Render),
                KeyCode::Char('p' | 'P') => return Intent::StartConfig(i, Route::CurrentPlots),
                KeyCode::Char('i' | 'I') if row["scenario"].is_object() => return Intent::StartConfig(i, Route::Scenario),
                KeyCode::Char('t' | 'T') if row["tracker"].is_object() => return Intent::StartConfig(i, Route::Tracking),
                _ => {}
            }
            return Intent::Keep;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('u' | 'U') => { self.query.clear(); self.selected = 0; }
                KeyCode::Char('n' | 'N') => if let Some(i) = self.index() { return Intent::Start(i, Route::New); },
                KeyCode::Char('d' | 'D') => if let Some(i) = self.index().filter(|i| MODES[*i].downscale) { return Intent::Start(i, Route::Downscale); },
                _ => {}
            }
            return Intent::Keep;
        }
        match key.code {
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => self.selected = (self.selected + 1).min(self.entries().len().saturating_sub(1)),
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = self.entries().len().saturating_sub(1),
            KeyCode::PageUp => self.selected = self.selected.saturating_sub(6),
            KeyCode::PageDown => self.selected = (self.selected + 6).min(self.entries().len().saturating_sub(1)),
            KeyCode::Enter => self.choose_row(self.selected),
            KeyCode::F(4) => if !self.entries().is_empty() { self.preview = true; self.offset = 0; },
            KeyCode::Backspace => { self.query.pop(); self.selected = 0; },
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::ALT) => self.paste(&c.to_string()),
            _ => {}
        }
        Intent::Keep
    }
    pub fn draw(&self, frame: &mut Frame, area: Rect, hits: &mut Vec<HitRegion>, has_config: bool) {
        self.has_config.set(has_config);
        frame.render_widget(Clear, area);
        frame.render_widget(Block::default().style(Style::default().bg(crate::BACK).fg(INK)), area);
        let block = panel(" Research workspaces · weather, questions, experiments ");
        let inner = block.inner(area); frame.render_widget(block, area);
        let rows = Layout::vertical([Constraint::Length(3), Constraint::Min(4), Constraint::Length(3)]).split(inner);
        if self.preview {
            if let Some(entry) = self.entries().get(self.selected) {
                frame.render_widget(Paragraph::new(format!("{}\nUp/Down scroll · PgDn more · Enter explores this choice", entry.title())).style(Style::default().fg(TEAL)), rows[0]);
                let paragraph = Paragraph::new(entry.description()).wrap(Wrap { trim: false });
                let max_offset = paragraph.line_count(rows[1].width).saturating_sub(rows[1].height as usize).min(u16::MAX as usize) as u16;
                self.max_offset.set(max_offset);
                frame.render_widget(paragraph.scroll((self.offset.min(max_offset), 0)), rows[1]);
                button_bar(frame, hits, rows[2], &[("Enter Explore", key_hit(KeyCode::Enter)), ("Esc Back to choices", key_hit(KeyCode::Esc))]);
            }
            return;
        }
        if let Some(Entry::Config(i)) = self.trail.last().copied() {
            let row = data::config(i).unwrap();
            let title = format!("{}\n{}\n{}", data::text(row, "title"), data::method(row), if self.notice.is_empty() { "Up/Down scroll · PgDn more · settings reviewed before creation" } else { &self.notice });
            frame.render_widget(Paragraph::new(title).style(Style::default().fg(TEAL).add_modifier(Modifier::BOLD)), rows[0]);
            let paragraph = Paragraph::new(data::detail(row)).wrap(Wrap { trim: false });
            let max_offset = paragraph.line_count(rows[1].width).saturating_sub(rows[1].height as usize).min(u16::MAX as usize) as u16;
            self.max_offset.set(max_offset);
            frame.render_widget(paragraph.scroll((self.offset.min(max_offset), 0)), rows[1]);
            let label = match data::setup_route(row) { Route::Downscale => "Enter Downscale", Route::Open => "Enter Open scenario", _ => "Enter Set up" };
            let mut buttons = vec![(label, key_hit(KeyCode::Enter)), ("R Plot history", key_hit(KeyCode::Char('r')))];
            if has_config { buttons.push(("P Plot set", key_hit(KeyCode::Char('p')))); }
            if has_config && row["tracker"].is_object() { buttons.push(("T Tracking", key_hit(KeyCode::Char('t')))); }
            if has_config && row["scenario"].is_object() { buttons.push(("I Initial state", key_hit(KeyCode::Char('i')))); }
            buttons.push(("Esc Back", key_hit(KeyCode::Esc)));
            button_bar(frame, hits, rows[2], &buttons);
            return;
        }
        let breadcrumb = if self.trail.is_empty() { "Choose a weather family".into() } else { self.trail.iter().map(|v| v.title()).collect::<Vec<_>>().join(" › ") };
        let search = format!("{}\nSearch: {}{}\nF4 Details explains the selected choice at any terminal width", breadcrumb, safe(&self.query), if self.query.is_empty() { "type to filter" } else { "" });
        frame.render_widget(Paragraph::new(search).wrap(Wrap { trim: false }).style(Style::default().fg(TEAL)), rows[0]);
        let entries = self.entries();
        let wide = rows[1].width >= 90;
        let columns = Layout::horizontal([Constraint::Percentage(if wide { 52 } else { 100 }), Constraint::Min(0)]).split(rows[1]);
        let list = entries.iter().enumerate().map(|(i, entry)| {
            let shortcut = matches!(entry, Entry::Config(_)) && matches!(self.trail.last(), Some(Entry::Family(_) | Entry::Submode(_)));
            let label = if shortcut { format!("Recommended shortcut: {}", entry.label()) } else { entry.label() };
            (vec![Line::raw(label)], match entry { Entry::Family(n) if self.trail.is_empty() => Hit::Workflow(*n), _ => Hit::Research(i) })
        }).collect();
        clickable_list(frame, hits, columns[0], list, Some(self.selected));
        if wide {
            if let Some(entry) = entries.get(self.selected) {
                frame.render_widget(Paragraph::new(format!("{}\nF4 opens the full, scrollable description\n\n{}", entry.title(), entry.description())).wrap(Wrap { trim: false }).block(panel(" Preview ")), columns[1]);
            }
        }
        if entries.is_empty() { frame.render_widget(Paragraph::new("No match here. Ctrl+U clears the filter; Esc goes up one level.").wrap(Wrap { trim: false }).style(Style::default().fg(MUTED)), columns[0]); }
        let mut buttons = vec![("Enter Explore", key_hit(KeyCode::Enter)), ("F4 Details", key_hit(KeyCode::F(4))), ("Ctrl+N Family starter", Hit::Key(KeyCode::Char('n'), KeyModifiers::CONTROL))];
        if self.index().is_some_and(|i| MODES[i].downscale) {
            buttons.push(("Ctrl+D Downscale archived parent", Hit::Key(KeyCode::Char('d'), KeyModifiers::CONTROL)));
        }
        buttons.extend([("Ctrl+U Clear", Hit::Key(KeyCode::Char('u'), KeyModifiers::CONTROL)), ("Esc Back", key_hit(KeyCode::Esc))]);
        button_bar(frame, hits, rows[2], &buttons);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rendered(browser: &Browser, width: u16, height: u16) -> String {
        let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, height)).unwrap();
        let mut hits = Vec::new();
        terminal.draw(|frame| browser.draw(frame, frame.area(), &mut hits, false)).unwrap();
        let mut text = String::new();
        for y in 0..height {
            for x in 0..width { text.push_str(terminal.backend().buffer()[(x, y)].symbol()); }
            text.push('\n');
        }
        text
    }
    #[test]
    fn each_leaf_has_reachable_setup_and_back_preserves_no_external_action() {
        for (i, row) in data::rows("configurations").iter().enumerate() {
            let mut browser = Browser::default(); browser.choose_config(i);
            assert_eq!(browser.trail.len(), 4);
            let expected = data::setup_route(row);
            assert!(matches!(browser.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)), Intent::StartConfig(index, route) if index == i && route == expected));
            for _ in 0..4 { assert!(matches!(browser.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)), Intent::Keep)); }
            assert!(browser.trail.is_empty());
            assert!(matches!(browser.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)), Intent::Close));
        }
    }
    #[test]
    fn nested_search_and_paste_are_bounded() {
        let mut browser = Browser::default(); browser.paste("derecho");
        assert!(browser.matches().contains(&1));
        browser.choose(1); browser.paste("derecho");
        assert!(browser.entries().iter().any(|v| matches!(v, Entry::Submode(_))));
        browser.query.clear(); browser.paste(&"不存在".repeat(1000));
        assert_eq!(browser.query.chars().count(), 160); assert!(browser.entries().is_empty());
        assert!(matches!(browser.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)), Intent::Keep));
    }
    #[test]
    fn displayed_method_is_searchable_at_each_catalog_level() {
        for (index, row) in data::rows("configurations").iter().enumerate().filter(|(_, row)| row["method"] == "archived_downscale") {
            let mut browser = Browser::default();
            browser.choose_config(index);
            browser.trail.pop();
            browser.paste("Archived-parent downscale");
            assert!(browser.entries().contains(&Entry::Config(index)), "{}", row["id"]);
            while browser.trail.pop().is_some() {
                assert!(!browser.entries().is_empty(), "method became undiscoverable above {}", row["id"]);
            }
        }
    }
    #[test]
    fn ordinary_supercell_search_exposes_the_archived_parent_route() {
        let mut browser = Browser::default();
        browser.paste("supercell downscale");
        let family = MODES.iter().position(|m| m.id == "supercell").unwrap();
        assert_eq!(browser.index(), Some(family));
        assert!(matches!(browser.key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)), Intent::Start(i, Route::Downscale) if i == family));
        browser.choose(family);
        assert!(matches!(browser.key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)), Intent::Start(i, Route::Downscale) if i == family));
    }
    #[test]
    fn full_descriptions_and_family_routes_are_visible_at_small_widths() {
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let mut browser = Browser::default(); browser.paste("supercell");
            let text = rendered(&browser, width, height);
            assert!(text.contains("F4 Details"), "{text}");
            assert!(text.contains("Ctrl+D Downscale archived parent"), "{text}");
            browser.key(KeyEvent::new(KeyCode::F(4), KeyModifiers::NONE));
            let text = rendered(&browser, width, height);
            assert!(text.contains("Storm structure, rotation"), "{text}");
            browser.key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE));
            let text = rendered(&browser, width, height);
            assert!(text.contains("Start a nested forecast"), "{text}");
            browser.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
            assert_eq!(browser.query, "supercell");
        }
    }
    #[test]
    fn unavailable_config_actions_and_modified_letters_keep_the_research_position() {
        let index = data::rows("configurations").iter().position(|r| r["id"] == "scenario-convection.gentle").unwrap();
        let mut browser = Browser::default(); browser.choose_config(index);
        let trail = browser.trail.clone();
        rendered(&browser, 80, 24);
        for key in [KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
                    KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE),
                    KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL)] {
            assert!(matches!(browser.key(key), Intent::Keep));
            assert_eq!(browser.trail, trail);
        }
        assert!(rendered(&browser, 80, 24).contains("Open a configuration"));
    }
}
