//! Ordered native history inputs for series beyond command-line limits.

use std::path::{Path, PathBuf};

pub fn read(path: &Path) -> Result<Vec<PathBuf>, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("cannot read input inventory {}: {error}", path.display()))?;
    let values: Vec<String> = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid input inventory {}: {error}", path.display()))?;
    if values.is_empty() || values.iter().any(|value| value.is_empty() || value.contains('\0')) {
        return Err("input inventory must contain nonempty file paths".into());
    }
    let parent = path.parent().unwrap_or(Path::new("."));
    Ok(values
        .into_iter()
        .map(|value| {
            let value = PathBuf::from(value);
            if value.is_absolute() {
                value
            } else {
                parent.join(value)
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_series_keeps_unicode_paths_duplicates_and_order() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "rustwx-input-list-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let inventory = root.join("series.json");
        let mut names: Vec<String> = (0..1441)
            .rev()
            .map(|index| format!("weather 🌦/day {index}/frame {index}.nc"))
            .collect();
        names.push(names[0].clone());
        let absolute = root.join("absolute.nc");
        names.push(absolute.to_string_lossy().into_owned());
        std::fs::write(&inventory, serde_json::to_vec(&names).unwrap()).unwrap();
        let expected: Vec<PathBuf> = names
            .iter()
            .map(|name| {
                let path = PathBuf::from(name);
                if path.is_absolute() {
                    path
                } else {
                    root.join(path)
                }
            })
            .collect();
        assert_eq!(read(&inventory).unwrap(), expected);
        for invalid in [
            b"[]".as_slice(),
            b"[1]".as_slice(),
            b"[\"\"]".as_slice(),
            b"[\"a\\u0000b\"]".as_slice(),
            b"{\"inputs\": [\"a\"]}".as_slice(),
        ] {
            std::fs::write(&inventory, invalid).unwrap();
            assert!(read(&inventory).is_err(), "{invalid:?}");
        }
        std::fs::remove_file(&inventory).unwrap();
        assert!(read(&inventory).unwrap_err().contains("cannot read input inventory"));
        std::fs::remove_dir(&root).unwrap();
    }
}
