//! Interpret pasted terminal file paths as data, without running shell syntax.
use std::path::{Path, PathBuf};

const MAX_TEXT_BYTES: usize = 64 * 1024;
const MAX_FILES: usize = 32;

/// Recognize an entire paste as existing local TOML, ZIP or JSON files.
///
/// Bare paths, quoted paths, a single PowerShell `& 'path'` spelling,
/// backslash-escaped spaces and local file URLs are supported. Every token
/// must be a file; ordinary configuration text or extra command arguments
/// therefore remain ordinary pasted text. Multiple files retain their order.
pub fn paths(text: &str, cwd: &Path) -> Option<Vec<PathBuf>> {
    if text.len() > MAX_TEXT_BYTES
        || text
            .chars()
            .any(|c| c.is_control() && !matches!(c, '\r' | '\n' | '\t'))
    {
        return None;
    }
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    // A single unquoted path may contain spaces and shell metacharacters as
    // literal filename characters. Its existence, never a shell, decides.
    if let Some(path) = candidate(text, cwd) {
        return Some(vec![path]);
    }
    let (text, powershell_prefix) = if let Some(rest) = text.strip_prefix('&') {
        let rest = rest.trim_start();
        if !rest.starts_with(['\'', '"']) {
            return None;
        }
        (rest, true)
    } else {
        (text, false)
    };
    let tokens = tokens(text)?;
    if tokens.is_empty() || tokens.len() > MAX_FILES || (powershell_prefix && tokens.len() != 1) {
        return None;
    }
    tokens.iter().map(|token| candidate(token, cwd)).collect()
}

fn candidate(text: &str, cwd: &Path) -> Option<PathBuf> {
    let decoded;
    let text = if text
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("file://"))
    {
        let rest = &text[7..];
        let rest = if let Some(local) = rest.strip_prefix("localhost/") {
            format!("/{local}")
        } else if rest.starts_with('/') {
            rest.to_owned()
        } else {
            // Network authorities are not local file drops.
            return None;
        };
        if rest.contains(['?', '#']) {
            return None;
        }
        decoded = percent_decode(&rest)?;
        #[cfg(windows)]
        let decoded = decoded
            .strip_prefix('/')
            .filter(|value| {
                value.as_bytes().get(1) == Some(&b':')
                    && value
                        .as_bytes()
                        .first()
                        .is_some_and(u8::is_ascii_alphabetic)
            })
            .unwrap_or(&decoded);
        #[cfg(not(windows))]
        let decoded = decoded.as_str();
        decoded
    } else {
        text
    };
    if text.is_empty() || text.chars().any(char::is_control) {
        return None;
    }
    let path = PathBuf::from(text);
    if !path.extension().is_some_and(|extension| {
        ["toml", "zip", "json"]
            .iter()
            .any(|expected| extension.eq_ignore_ascii_case(expected))
    }) {
        return None;
    }
    let path = if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    };
    path.is_file().then_some(path)
}

fn percent_decode(text: &str) -> Option<String> {
    let mut result = Vec::with_capacity(text.len());
    let mut bytes = text.as_bytes().iter().copied();
    while let Some(byte) = bytes.next() {
        if byte == b'%' {
            let high = char::from(bytes.next()?).to_digit(16)?;
            let low = char::from(bytes.next()?).to_digit(16)?;
            result.push((high * 16 + low) as u8);
        } else {
            result.push(byte);
        }
    }
    String::from_utf8(result).ok()
}

fn tokens(text: &str) -> Option<Vec<String>> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut quoted = None;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if let Some(quote) = quoted {
            if c == quote {
                if quote == '\'' && chars.peek() == Some(&'\'') {
                    chars.next();
                    current.push('\''); // PowerShell's literal apostrophe.
                } else {
                    quoted = None;
                }
            } else if c == '\\' && quote == '"' && chars.peek() == Some(&'"') {
                chars.next();
                current.push('"');
            } else {
                current.push(c);
            }
        } else {
            match c {
                '\'' | '"' => quoted = Some(c),
                c if c.is_whitespace() => {
                    if !current.is_empty() {
                        result.push(std::mem::take(&mut current));
                        if result.len() > MAX_FILES {
                            return None;
                        }
                    }
                }
                '\\' if chars.peek().is_some_and(|next| {
                    next.is_whitespace()
                        || matches!(
                            next,
                            '\'' | '"'
                                | '&'
                                | ';'
                                | '|'
                                | '<'
                                | '>'
                                | '$'
                                | '`'
                                | '('
                                | ')'
                                | '#'
                                | '!'
                                | '{'
                                | '}'
                                | '['
                                | ']'
                                | '*'
                                | '?'
                        )
                }) =>
                {
                    current.push(chars.next()?);
                }
                ';' | '|' | '&' | '<' | '>' | '$' | '`' | '(' | ')' => return None,
                _ => current.push(c),
            }
        }
    }
    if quoted.is_some() {
        return None;
    }
    if !current.is_empty() {
        result.push(current);
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    struct Files(PathBuf);
    impl Files {
        fn new() -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir()
                .join(format!("arwen-file-drop-{}-{unique}", std::process::id()));
            fs::create_dir(&root).unwrap();
            for name in [
                "storm case.toml",
                "worldwide.ZIP",
                "café 日本.json",
                "Drew's case.toml",
                "storm$2026(1)#!.toml",
                "notes.txt",
            ] {
                fs::write(root.join(name), b"test input").unwrap();
            }
            Self(root)
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
                .starts_with("arwen-file-drop-"));
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn bare_quoted_powershell_and_escaped_spaces_name_the_same_file() {
        let files = Files::new();
        let expected = vec![files.0.join("storm case.toml")];
        for text in [
            "storm case.toml",
            "\"storm case.toml\"",
            "'storm case.toml'",
            "& 'storm case.toml'",
            r"storm\ case.toml",
        ] {
            assert_eq!(paths(text, &files.0), Some(expected.clone()), "{text}");
        }
        let absolute = files.0.join("storm case.toml").display().to_string();
        assert_eq!(
            paths(&absolute, Path::new("unrelated")),
            Some(expected.clone())
        );
        assert_eq!(
            paths(&format!("\"{absolute}\""), Path::new("unrelated")),
            Some(expected)
        );
        assert_eq!(
            paths("& 'Drew''s case.toml'", &files.0),
            Some(vec![files.0.join("Drew's case.toml")])
        );
    }

    #[test]
    fn escaped_shell_punctuation_is_literal_filename_data() {
        let files = Files::new();
        let expected = vec![files.0.join("storm$2026(1)#!.toml")];
        for text in [
            "storm$2026(1)#!.toml",
            "'storm$2026(1)#!.toml'",
            r"storm\$2026\(1\)\#\!.toml",
        ] {
            assert_eq!(paths(text, &files.0), Some(expected.clone()), "{text}");
        }
    }

    #[test]
    fn multiple_files_preserve_order_and_unicode() {
        let files = Files::new();
        let expected = vec![
            files.0.join("worldwide.ZIP"),
            files.0.join("café 日本.json"),
            files.0.join("storm case.toml"),
        ];
        for text in [
            "worldwide.ZIP \"café 日本.json\" 'storm case.toml'",
            "worldwide.ZIP\r\n'café 日本.json'\r\n\"storm case.toml\"",
        ] {
            assert_eq!(paths(text, &files.0), Some(expected.clone()));
        }
    }

    #[test]
    fn local_file_urls_decode_spaces_and_unicode_without_network_authorities() {
        let files = Files::new();
        let expected = files.0.join("café 日本.json");
        let raw = expected.to_string_lossy().replace('\\', "/");
        let encoded = raw
            .bytes()
            .map(|b| {
                if b.is_ascii_alphanumeric() || b"/:.-_".contains(&b) {
                    char::from(b).to_string()
                } else {
                    format!("%{b:02X}")
                }
            })
            .collect::<String>();
        let uri = format!(
            "file://{}{}",
            if encoded.starts_with('/') { "" } else { "/" },
            encoded
        );
        assert_eq!(paths(&uri, &files.0), Some(vec![expected]));
        for text in [
            "file://server/path.toml",
            "file:///bad%ZZ.toml",
            "file:///bad%00.toml",
        ] {
            assert_eq!(paths(text, &files.0), None);
        }
    }

    #[test]
    fn config_text_missing_files_directories_and_commands_are_not_file_drops() {
        let files = Files::new();
        for text in [
            "[experiment]\nname = 'storm case.toml'",
            "worldwide.ZIP; echo bad",
            "worldwide.ZIP | cat",
            "& 'worldwide.ZIP' --flag",
            "& 'worldwide.ZIP' 'storm case.toml'",
            "$(worldwide.ZIP)",
            "'worldwide.ZIP' missing.toml",
            "notes.txt",
            "'unterminated.toml",
            "",
            "worldwide.ZIP\x00",
        ] {
            assert_eq!(paths(text, &files.0), None, "{text:?}");
        }
        fs::create_dir(files.0.join("directory.toml")).unwrap();
        assert_eq!(paths("directory.toml", &files.0), None);
    }
}
