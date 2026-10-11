//! Shared CSV transport for observation archives and retained paired scores.
//! Quoted commas, embedded newlines and doubled quotes retain their contents.
use std::io::Write;
use std::path::Path;

pub fn parse(text: &str) -> Result<Vec<Vec<String>>, String> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut closed = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                    closed = true;
                }
            } else {
                field.push(c);
            }
        } else {
            match c {
                '"' if field.is_empty() && !closed => quoted = true,
                ',' => {
                    row.push(std::mem::take(&mut field));
                    closed = false;
                }
                '\r' | '\n' => {
                    if c == '\r' && chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                    closed = false;
                }
                _ if closed => {
                    return Err(
                        "CSV has text after a closing quote; column support is ambiguous".into(),
                    );
                }
                '"' => return Err("CSV has an unquoted quote; column support is ambiguous".into()),
                _ => field.push(c),
            }
        }
    }
    if quoted {
        return Err("CSV ends inside a quoted field".into());
    }
    if !row.is_empty() || !field.is_empty() || closed {
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}
pub fn read(path: &Path) -> Result<Vec<Vec<String>>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    parse(text.trim_start_matches('\u{feff}'))
}
pub fn write_row(writer: &mut impl Write, row: &[String]) -> Result<(), String> {
    for (i, field) in row.iter().enumerate() {
        if i > 0 {
            writer.write_all(b",").map_err(|e| e.to_string())?;
        }
        if field.contains([',', '"', '\n', '\r']) {
            write!(writer, "\"{}\"", field.replace('"', "\"\"")).map_err(|e| e.to_string())?;
        } else {
            writer
                .write_all(field.as_bytes())
                .map_err(|e| e.to_string())?;
        }
    }
    writer.write_all(b"\n").map_err(|e| e.to_string())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quoted_records_round_trip() {
        let rows = vec![
            vec!["station".into(), "source".into()],
            vec!["A".into(), "line one,\nline \"two\"".into()],
        ];
        let mut out = Vec::new();
        for r in &rows {
            write_row(&mut out, r).unwrap();
        }
        assert_eq!(parse(&String::from_utf8(out).unwrap()).unwrap(), rows);
        assert!(parse("a,\"truncated").is_err());
        assert!(parse("a,\"closed\"garbage").is_err());
    }
}
