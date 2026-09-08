//! Recognize Windows terminal file drops delivered as a burst of key events.
use crossterm::event::Event;
use std::{io, path::Path};

#[cfg(any(windows, test))]
use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers};

/// Read a bounded Windows key burst, preserving ordinary input in its order.
///
/// Windows console input does not emit crossterm's `Paste` event. A complete
/// burst becomes a paste only when every text token names an existing file
/// accepted by `file_drop`. Native paste events on other platforms are kept.
#[cfg(not(windows))]
pub fn read(first: Event, _cwd: &Path) -> io::Result<Vec<Event>> {
    Ok(vec![first])
}

#[cfg(windows)]
pub fn read(first: Event, cwd: &Path) -> io::Result<Vec<Event>> {
    use crossterm::event;
    use std::time::{Duration, Instant};

    const QUIET: Duration = Duration::from_millis(10);
    const MAX_DURATION: Duration = Duration::from_millis(200);
    const MAX_TEXT_BYTES: usize = 64 * 1024;
    const MAX_EVENTS: usize = 2 * MAX_TEXT_BYTES;

    let Some(first_character) = key_character(&first) else {
        return Ok(vec![first]);
    };
    let start = Instant::now();
    let mut text_bytes = first_character.map_or(0, char::len_utf8);
    let mut events = vec![first];
    let mut complete = true;
    loop {
        let remaining = MAX_DURATION.saturating_sub(start.elapsed());
        if remaining.is_zero() || text_bytes >= MAX_TEXT_BYTES || events.len() >= MAX_EVENTS {
            // The unread remainder might extend a currently valid filename.
            // A truncated prefix must never open a different existing file.
            complete = false;
            break;
        }
        let timeout = QUIET.min(remaining);
        if !event::poll(timeout)? {
            // Reaching the total time cap is not evidence of a quiet gap.
            complete = timeout == QUIET;
            break;
        }
        let next = event::read()?;
        let character = key_character(&next);
        events.push(next);
        match character {
            Some(character) => text_bytes += character.map_or(0, char::len_utf8),
            // Preserve this already-read boundary after the text burst.
            None => break,
        }
    }
    Ok(finish(events, cwd, complete))
}

/// `None` ends a burst; `Some(None)` is a text-key release with no new text.
#[cfg(any(windows, test))]
fn key_character(event: &Event) -> Option<Option<char>> {
    let Event::Key(key) = event else {
        return None;
    };
    if !(key.modifiers - KeyModifiers::SHIFT).is_empty() {
        return None;
    }
    let character = match key.code {
        KeyCode::Char(character) if !character.is_control() => character,
        KeyCode::Enter => '\n',
        KeyCode::Tab => '\t',
        _ => return None,
    };
    Some(match key.kind {
        KeyEventKind::Press | KeyEventKind::Repeat => Some(character),
        KeyEventKind::Release => None,
    })
}

#[cfg(any(windows, test))]
fn finish(events: Vec<Event>, cwd: &Path, complete: bool) -> Vec<Event> {
    if !complete {
        return events;
    }
    let mut text = String::new();
    let mut prefix = 0;
    for event in &events {
        let Some(character) = key_character(event) else {
            break;
        };
        if let Some(character) = character {
            text.push(character);
        }
        prefix += 1;
    }
    if prefix == 0 || super::file_drop::paths(&text, cwd).is_none() {
        return events;
    }
    std::iter::once(Event::Paste(text))
        .chain(events.into_iter().skip(prefix))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEvent, MouseEvent, MouseEventKind};
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    struct Files(PathBuf);
    impl Files {
        fn new() -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "arwen-file-drop-input-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            for name in ["café 日本.toml", "worldwide.ZIP"] {
                fs::write(path.join(name), b"test input").unwrap();
            }
            Self(path)
        }
    }
    impl Drop for Files {
        fn drop(&mut self) {
            assert_eq!(self.0.parent(), Some(std::env::temp_dir().as_path()));
            assert!(self
                .0
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("arwen-file-drop-input-"));
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn keys(text: &str) -> Vec<Event> {
        text.chars()
            .map(|character| {
                Event::Key(KeyEvent::new(
                    match character {
                        '\n' => KeyCode::Enter,
                        '\t' => KeyCode::Tab,
                        character => KeyCode::Char(character),
                    },
                    KeyModifiers::NONE,
                ))
            })
            .collect()
    }

    #[test]
    fn existing_unicode_and_multiple_file_paths_become_a_single_paste() {
        let files = Files::new();
        for text in [
            "\"café 日本.toml\"\n".to_owned(),
            "'café 日本.toml'\tworldwide.ZIP".to_owned(),
            format!("\"{}\"", files.0.join("café 日本.toml").display()),
        ] {
            assert_eq!(
                finish(keys(&text), &files.0, true),
                vec![Event::Paste(text)]
            );
        }
    }

    #[test]
    fn releases_add_no_text_and_shift_and_repeat_characters_are_preserved() {
        let files = Files::new();
        let text = "worldwide.ZIP";
        let mut events = vec![];
        for character in text.chars() {
            let modifiers = if character.is_uppercase() {
                KeyModifiers::SHIFT
            } else {
                KeyModifiers::NONE
            };
            let mut key = KeyEvent::new(KeyCode::Char(character), modifiers);
            key.kind = KeyEventKind::Repeat;
            events.push(Event::Key(key));
            key.kind = KeyEventKind::Release;
            events.push(Event::Key(key));
        }
        assert_eq!(
            finish(events, &files.0, true),
            vec![Event::Paste(text.into())]
        );
    }

    #[test]
    fn ordinary_shortcuts_configuration_text_and_missing_files_are_unchanged() {
        let files = Files::new();
        for text in [
            "q",
            "k\n",
            "[experiment]\nconfig = 'worldwide.ZIP'\n",
            "missing.toml",
            "worldwide.ZIP missing.toml",
        ] {
            let events = keys(text);
            assert_eq!(finish(events.clone(), &files.0, true), events);
        }
        for modifiers in [
            KeyModifiers::CONTROL,
            KeyModifiers::ALT,
            KeyModifiers::SUPER,
            KeyModifiers::HYPER,
            KeyModifiers::META,
            KeyModifiers::CONTROL | KeyModifiers::SHIFT,
        ] {
            let event = Event::Key(KeyEvent::new(KeyCode::Char('s'), modifiers));
            assert_eq!(key_character(&event), None);
            assert_eq!(finish(vec![event.clone()], &files.0, true), vec![event]);
        }
    }

    #[test]
    fn nontext_boundaries_remain_after_the_converted_prefix_in_order() {
        let files = Files::new();
        for boundary in [
            Event::Resize(100, 30),
            Event::Mouse(MouseEvent {
                kind: MouseEventKind::Moved,
                column: 3,
                row: 4,
                modifiers: KeyModifiers::NONE,
            }),
            Event::Key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL)),
            Event::Key(KeyEvent::new(KeyCode::F(2), KeyModifiers::NONE)),
            Event::Paste("ordinary pasted text".into()),
        ] {
            let mut events = keys("worldwide.ZIP");
            events.push(boundary.clone());
            assert_eq!(
                finish(events, &files.0, true),
                vec![Event::Paste("worldwide.ZIP".into()), boundary.clone()]
            );
            assert_eq!(
                finish(vec![boundary.clone()], &files.0, true),
                vec![boundary]
            );
        }
    }

    #[test]
    fn an_incomplete_burst_never_converts_an_existing_path_prefix() {
        let files = Files::new();
        let events = keys("worldwide.ZIP");
        assert_eq!(finish(events.clone(), &files.0, false), events);
    }

    #[cfg(not(windows))]
    #[test]
    fn non_windows_reads_return_the_first_event_without_terminal_input() {
        let event = Event::Key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE));
        assert_eq!(read(event.clone(), Path::new(".")).unwrap(), vec![event]);
    }
}
