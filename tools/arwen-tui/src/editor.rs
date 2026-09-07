//! The complete TOML remains the authority; exports never replace another file.
use crossterm::event::KeyCode;
use std::{
    fs, io,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Remove only Windows' extended-path display prefix; filesystem paths stay canonical.
pub fn display_path(path: &std::path::Path) -> String {
    let value = path.to_string_lossy();
    if let Some(tail) = value.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{tail}");
    }
    value.strip_prefix(r"\\?\").unwrap_or(&value).to_owned()
}

pub struct Editor {
    pub path: PathBuf,
    pub lines: Vec<Vec<char>>,
    pub row: usize,
    pub col: usize,
    pub top: usize,
    /// Horizontal viewport origin in terminal cells, not character indices.
    pub left: usize,
    pub dirty: bool,
    original: String,
    newline: &'static str,
    undo: Vec<DraftState>,
    redo: Vec<DraftState>,
}

struct DraftState {
    text: String,
    row: usize,
    col: usize,
    top: usize,
    left: usize,
}

pub struct DraftSave {
    pub path: PathBuf,
    pub syntax_error: Option<String>,
    pub moved_folder: bool,
}

// Character offsets remain the editor's storage coordinates. All movement and
// display slicing respects graphemes, so an emoji/combining sequence is one key.
fn clusters(line: &[char]) -> Vec<(usize, usize, String, usize)> {
    let text: String = line.iter().collect();
    let mut offset = 0;
    text.graphemes(true)
        .map(|g| {
            let start = offset;
            offset += g.chars().count();
            let visible: String = g
                .chars()
                .filter(|c| *c == '\t' || !c.is_control())
                .collect();
            let visible = visible.replace('\t', "    ");
            let width = visible.width();
            (start, offset, visible, width)
        })
        .collect()
}

impl Editor {
    pub fn character_at_display_column(line: &[char], target: usize) -> usize {
        let mut cells = 0;
        for (start, _, _, width) in clusters(line) {
            if cells + width > target {
                return start;
            }
            cells += width;
        }
        line.len()
    }

    pub fn load(path: PathBuf) -> io::Result<Self> {
        let path = fs::canonicalize(path)?;
        let original = fs::read_to_string(&path)?;
        let newline = if original.contains("\r\n") {
            "\r\n"
        } else {
            "\n"
        };
        let lines = original
            .split('\n')
            .map(|s| s.trim_end_matches('\r').chars().collect())
            .collect();
        Ok(Self {
            path,
            lines,
            row: 0,
            col: 0,
            top: 0,
            left: 0,
            dirty: false,
            original,
            newline,
            undo: Vec::new(),
            redo: Vec::new(),
        })
    }
    pub fn text(&self) -> String {
        if !self.dirty {
            return self.original.clone();
        }
        self.lines
            .iter()
            .map(|line| line.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join(self.newline)
    }
    fn draft_state(&self) -> DraftState {
        DraftState { text: self.text(), row: self.row, col: self.col,
                     top: self.top, left: self.left }
    }
    fn remember(&mut self) {
        self.undo.push(self.draft_state());
        self.redo.clear();
        // Keep the latest edit undoable even for a large file, while bounding
        // accumulated history for ordinary typing and repeated guided edits.
        while self.undo.len() > 1 && (self.undo.len() > 256
            || self.undo.iter().map(|s| s.text.len()).sum::<usize>() > 16 * 1024 * 1024) {
            self.undo.remove(0);
        }
    }
    fn restore_draft(&mut self, state: DraftState) {
        self.dirty = state.text != self.original;
        self.lines = state.text.split('\n')
            .map(|line| line.trim_end_matches('\r').chars().collect()).collect();
        self.row = state.row.min(self.lines.len() - 1);
        self.col = state.col;
        self.top = state.top;
        self.left = state.left;
        self.snap_cursor();
    }
    pub fn undo(&mut self) -> bool {
        let Some(previous) = self.undo.pop() else { return false; };
        self.redo.push(self.draft_state());
        self.restore_draft(previous);
        true
    }
    pub fn redo(&mut self) -> bool {
        let Some(next) = self.redo.pop() else { return false; };
        self.undo.push(self.draft_state());
        self.restore_draft(next);
        true
    }
    pub fn discard_draft(&mut self) -> bool {
        if !self.dirty { return false; }
        self.replace_draft(self.original.clone());
        self.dirty = false;
        true
    }
    /// Replace only the in-memory draft after a guided edit has been reviewed.
    pub fn replace_draft(&mut self, text: String) {
        if text == self.text() {
            return;
        }
        self.remember();
        self.lines = text
            .split('\n')
            .map(|line| line.trim_end_matches('\r').chars().collect())
            .collect();
        self.row = 0;
        self.col = 0;
        self.top = 0;
        self.left = 0;
        self.dirty = true;
    }
    pub fn suggested_draft_path(&self) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        self.path.with_file_name(format!(
            "{}.draft-{stamp}.toml",
            self.path.file_stem().unwrap_or_default().to_string_lossy()
        ))
    }
    /// Save As may preserve an incomplete draft. It never overwrites any path.
    pub fn save_as(&mut self, path: PathBuf) -> Result<DraftSave, String> {
        let parent = path.parent().ok_or("Choose a file in an existing folder")?;
        let parent =
            fs::canonicalize(parent).map_err(|e| format!("Cannot open draft folder: {e}"))?;
        let name = path.file_name().ok_or("Choose a new file name")?;
        let path = parent.join(name);
        let text = self.text();
        let syntax_error = text
            .parse::<toml_edit::DocumentMut>()
            .err()
            .map(|e| e.to_string());
        let mut file = fs::OpenOptions::new().write(true).create_new(true).open(&path)
            .map_err(|e| format!("Draft was not saved: {e}. Choose a new file name; existing files are never replaced."))?;
        use io::Write;
        file.write_all(text.as_bytes()).and_then(|_| file.sync_all())
            .map_err(|e| format!("Draft write failed: {e}. Your full draft remains in the editor; partial file: {}", display_path(&path)))?;
        drop(file);
        #[cfg(unix)]
        fs::File::open(&parent)
            .and_then(|d| d.sync_all())
            .map_err(|e| {
                format!(
                    "Draft file is written at {}, but folder sync failed: {e}",
                    display_path(&path)
                )
            })?;
        let moved_folder = self.path.parent() != Some(parent.as_path());
        self.path = path.clone();
        self.original = text;
        self.dirty = false;
        Ok(DraftSave {
            path,
            syntax_error,
            moved_folder,
        })
    }
    pub fn save(&mut self) -> Result<PathBuf, String> {
        if !self.dirty {
            return Ok(self.path.clone());
        }
        let text = self.text();
        text.parse::<toml_edit::DocumentMut>().map_err(|e| {
            format!("TOML needs a correction: {e}. F12 can export this incomplete draft.")
        })?;
        let current = fs::read_to_string(&self.path).map_err(|e| e.to_string())?;
        if current != self.original {
            return Err("This file changed outside the TUI. Your draft is retained. F12 exports it to a new file without replacing the external changes.".into());
        }
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let parent = self
            .path
            .parent()
            .ok_or("Configuration has no parent folder")?;
        let name = self.path.file_name().unwrap_or_default().to_string_lossy();
        let backup = parent.join(format!(".{name}.arwen-backup-{stamp}"));
        let temp = parent.join(format!(".{name}.arwen-saving-{stamp}"));
        fs::write(&backup, self.original.as_bytes()).map_err(|e| e.to_string())?;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|e| e.to_string())?;
        use io::Write;
        file.write_all(text.as_bytes())
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        drop(file);
        if fs::read_to_string(&self.path).map_err(|e| e.to_string())? != self.original {
            return Err(format!(
                "Configuration changed during save. Draft retained at {}",
                display_path(&temp)
            ));
        }
        fs::rename(&temp, &self.path).map_err(|e| {
            format!(
                "Could not replace configuration: {e}. Draft: {}",
                display_path(&temp)
            )
        })?;
        self.original = text;
        self.dirty = false;
        Ok(backup)
    }
    fn previous(&self) -> usize {
        clusters(&self.lines[self.row])
            .iter()
            .map(|c| c.0)
            .filter(|&p| p < self.col)
            .last()
            .unwrap_or(0)
    }
    fn next(&self) -> usize {
        clusters(&self.lines[self.row])
            .iter()
            .map(|c| c.1)
            .find(|&p| p > self.col)
            .unwrap_or(self.lines[self.row].len())
    }
    fn snap_cursor(&mut self) {
        self.col = self.col.min(self.lines[self.row].len());
        if let Some((start, _, _, _)) = clusters(&self.lines[self.row])
            .into_iter()
            .find(|c| c.0 < self.col && self.col < c.1)
        {
            self.col = start;
        }
    }
    pub fn display_column(line: &[char], col: usize) -> usize {
        clusters(line)
            .iter()
            .take_while(|c| c.1 <= col)
            .map(|c| c.3)
            .sum()
    }
    pub fn visible_line(line: &[char], left: usize, width: usize) -> String {
        let mut text = String::new();
        let mut cell = 0;
        for (_, _, visible, n) in clusters(line) {
            let end = cell + n;
            if cell >= left + width {
                break;
            }
            if cell >= left && end <= left + width {
                text.push_str(&visible);
            } else if end > left && cell < left {
                text.push_str(&" ".repeat((end - left).min(width)));
            }
            cell = end;
        }
        text
    }
    pub fn ensure_viewport(&mut self, width: usize, height: usize) {
        self.snap_cursor();
        if self.row < self.top {
            self.top = self.row;
        }
        if self.row >= self.top + height {
            self.top = self.row.saturating_sub(height.saturating_sub(1));
        }
        let cell = Self::display_column(&self.lines[self.row], self.col);
        let next_width = clusters(&self.lines[self.row])
            .iter()
            .find(|c| c.0 == self.col)
            .map(|c| c.3)
            .unwrap_or(1)
            .max(1);
        if cell < self.left {
            self.left = cell;
        }
        if cell + next_width > self.left + width {
            self.left = (cell + next_width).saturating_sub(width);
        }
    }
    pub fn insert(&mut self, text: &str) {
        let text = text.replace("\r\n", "\n");
        if !text.chars().any(|c| c == '\n' || c == '\t' || !c.is_control()) { return; }
        self.remember();
        for ch in text.chars() {
            if ch == '\n' {
                let tail = self.lines[self.row].split_off(self.col);
                self.row += 1;
                self.lines.insert(self.row, tail);
                self.col = 0;
                self.dirty = true;
            } else if ch == '\t' || !ch.is_control() {
                self.lines[self.row].insert(self.col, ch);
                self.col += 1;
                self.dirty = true;
            }
        }
    }
    pub fn key(&mut self, key: KeyCode) {
        self.snap_cursor();
        match key {
            KeyCode::Char(ch) => self.insert(&ch.to_string()),
            KeyCode::Tab => self.insert("    "),
            KeyCode::Enter => {
                self.remember();
                let tail = self.lines[self.row].split_off(self.col);
                self.row += 1;
                self.lines.insert(self.row, tail);
                self.col = 0;
                self.dirty = true;
            }
            KeyCode::Backspace if self.col > 0 => {
                self.remember();
                let start = self.previous();
                self.lines[self.row].drain(start..self.col);
                self.col = start;
                self.dirty = true;
            }
            KeyCode::Backspace if self.row > 0 => {
                self.remember();
                let line = self.lines.remove(self.row);
                self.row -= 1;
                self.col = self.lines[self.row].len();
                self.lines[self.row].extend(line);
                self.dirty = true;
            }
            KeyCode::Delete if self.col < self.lines[self.row].len() => {
                self.remember();
                let end = self.next();
                self.lines[self.row].drain(self.col..end);
                self.dirty = true;
            }
            KeyCode::Delete if self.row + 1 < self.lines.len() => {
                self.remember();
                let tail = self.lines.remove(self.row + 1);
                self.lines[self.row].extend(tail);
                self.dirty = true;
            }
            KeyCode::Left if self.col > 0 => self.col = self.previous(),
            KeyCode::Left if self.row > 0 => {
                self.row -= 1;
                self.col = self.lines[self.row].len();
            }
            KeyCode::Right if self.col < self.lines[self.row].len() => self.col = self.next(),
            KeyCode::Right if self.row + 1 < self.lines.len() => {
                self.row += 1;
                self.col = 0;
            }
            KeyCode::Up => self.row = self.row.saturating_sub(1),
            KeyCode::Down => self.row = (self.row + 1).min(self.lines.len() - 1),
            KeyCode::PageUp => self.row = self.row.saturating_sub(16),
            KeyCode::PageDown => self.row = (self.row + 16).min(self.lines.len() - 1),
            KeyCode::Home => self.col = 0,
            KeyCode::End => self.col = self.lines[self.row].len(),
            _ => {}
        }
        self.snap_cursor();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(name: &str, text: &str) -> Editor {
        let p = std::env::temp_dir().join(format!(
            "arwen-editor-{}-{name}-{}.toml",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(&p, text).unwrap();
        Editor::load(p).unwrap()
    }
    #[test]
    fn unicode_and_comments_survive_a_targeted_edit() {
        let mut e = fixture(
            "unicode",
            "# café 🌧\r\n[shared]\r\nnz = 49\r\ncustom = [1, 2, 3]\r\n",
        );
        let original = e.text();
        assert!(!e.dirty);
        e.row = 2;
        e.col = 7;
        e.key(KeyCode::Backspace);
        e.insert("0");
        let backup = e.save().unwrap();
        assert_eq!(fs::read_to_string(backup).unwrap(), original);
        assert_eq!(
            e.text(),
            "# café 🌧\r\n[shared]\r\nnz = 40\r\ncustom = [1, 2, 3]\r\n"
        );
    }
    #[test]
    fn external_edit_is_never_overwritten() {
        let mut e = fixture("conflict", "a=1\n");
        e.insert("# draft\n");
        fs::write(&e.path, "a=2\n").unwrap();
        assert!(e.save().unwrap_err().contains("changed outside"));
        assert!(e.dirty);
        assert_eq!(fs::read_to_string(&e.path).unwrap(), "a=2\n");
    }
    #[test]
    fn invalid_toml_stays_in_draft() {
        let mut e = fixture("invalid", "a=1\n");
        e.insert("[");
        assert!(e.save().unwrap_err().contains("TOML"));
        assert_eq!(fs::read_to_string(&e.path).unwrap(), "a=1\n");
    }
    #[test]
    fn save_as_never_replaces_an_existing_file_and_preserves_unknown_settings() {
        let mut e = fixture("saveas", "# keep\n[custom]\nunknown = ['界', 'é']\n");
        let original = e.path.clone();
        let text = e.text();
        assert!(e.save_as(original.clone()).is_err());
        assert_eq!(fs::read_to_string(&original).unwrap(), text);
        let target = e.suggested_draft_path();
        let result = e.save_as(target.clone()).unwrap();
        assert!(!result.moved_folder && result.syntax_error.is_none());
        assert_eq!(fs::read_to_string(target).unwrap(), text);
    }
    #[test]
    fn combining_and_emoji_sequences_move_and_delete_as_visible_characters() {
        let mut e = fixture("graphemes", "é e\u{301} 👩‍🌾X\n");
        e.col = e.lines[0].len();
        e.key(KeyCode::Left);
        assert_eq!(e.lines[0][e.col], 'X');
        e.key(KeyCode::Left);
        assert_eq!(e.lines[0][e.col], '👩');
        e.key(KeyCode::Delete);
        assert_eq!(e.text(), "é e\u{301} X\n");
        e.key(KeyCode::Left);
        e.key(KeyCode::Left);
        assert_eq!(e.lines[0][e.col], 'e');
        e.key(KeyCode::Delete);
        assert_eq!(e.text(), "é  X\n");
    }
    #[test]
    fn guided_changes_undo_and_redo_without_touching_the_saved_file() {
        let mut e = fixture("guided-undo", "# café 🌧\r\n[shared]\r\nnz = 49\r\n");
        let original = e.text();
        e.row = 2;
        e.col = 7;
        let changed = original.replace("49", "64");
        e.replace_draft(changed.clone());
        assert!(e.dirty && e.undo());
        assert_eq!(e.text(), original);
        assert_eq!((e.row, e.col), (2, 7));
        assert!(!e.dirty);
        assert!(e.redo());
        assert_eq!(e.text(), changed);
        assert!(e.dirty);
        assert_eq!(fs::read_to_string(&e.path).unwrap(), original);
    }
    #[test]
    fn paste_is_one_undo_and_a_new_edit_replaces_the_redo_branch() {
        let mut e = fixture("paste-undo", "x=1\n");
        let original = e.text();
        e.insert("# one\n# two\n");
        assert!(e.undo());
        assert_eq!(e.text(), original);
        assert!(!e.undo());
        assert!(e.redo());
        assert!(e.undo());
        e.insert("# different\n");
        assert!(!e.redo());
        assert_eq!(e.text(), "# different\nx=1\n");
    }
    #[test]
    fn discard_is_undoable_and_saved_content_defines_the_dirty_boundary() {
        let mut e = fixture("discard-undo", "x=1\n");
        e.insert("# draft\n");
        let draft = e.text();
        assert!(e.discard_draft());
        assert!(!e.dirty && !e.discard_draft());
        assert!(e.undo());
        assert_eq!(e.text(), draft);
        e.save().unwrap();
        assert!(!e.dirty);
        assert!(e.undo());
        assert!(e.dirty);
        assert_eq!(fs::read_to_string(&e.path).unwrap(), draft);
        assert!(e.redo());
        assert_eq!(e.text(), draft);
        assert!(!e.dirty);
    }
}
