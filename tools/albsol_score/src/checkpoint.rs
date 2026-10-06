//! The scorer's bounded checkpoint-plane reader and exact member provenance.
use std::error::Error;
use std::fs;
use std::io::Read;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug)]
pub struct CheckpointPlane {
    pub values: Vec<f64>,
    pub array: String,
}

pub fn read_checkpoint_plane(
    path: &Path,
    name: &str,
    nx: usize,
    ny: usize,
) -> Result<CheckpointPlane> {
    // Select from the ZIP inventory before parsing any NPY data. Only a
    // missing coszen member permits the checkpoint writer's coszen_ref alias;
    // malformed present data and archive/read errors must still be refused.
    let mut archive = zip::ZipArchive::new(fs::File::open(path)?)?;
    let primary = format!("fields/{name}.npy");
    let array = if name == "coszen" && !archive.file_names().any(|entry| entry == primary) {
        "fields/coszen_ref.npy".to_string()
    } else {
        primary
    };
    let mut entry = archive.by_name(&array)?;
    // Preserve the existing C-order, little-endian NPY contract and size bound.
    if entry.size() > ((nx * ny * 8 + 65536) as u64) {
        return Err("checkpoint plane exceeds expected 2-D extent".into());
    }
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;
    if bytes.get(..6) != Some(b"\x93NUMPY") {
        return Err("checkpoint array is not NPY".into());
    }
    let major = *bytes.get(6).ok_or("truncated NPY version")?;
    let start = if major == 1 {
        10
    } else if major == 2 || major == 3 {
        12
    } else {
        return Err("unsupported NPY version".into());
    };
    let size = if start == 10 {
        u16::from_le_bytes(bytes.get(8..10).ok_or("truncated NPY length")?.try_into()?)
            as usize
    } else {
        u32::from_le_bytes(bytes.get(8..12).ok_or("truncated NPY length")?.try_into()?)
            as usize
    };
    let header = std::str::from_utf8(
        bytes
            .get(start..start + size)
            .ok_or("truncated NPY header")?,
    )?;
    let compact: String = header.chars().filter(|c| !c.is_whitespace()).collect();
    if !compact.contains("'fortran_order':False") {
        return Err("checkpoint plane is not C-order".into());
    }
    if !compact.contains(&format!("'shape':({ny},{nx})"))
        && !compact.contains(&format!("'shape':({ny},{nx},)"))
    {
        return Err(format!("checkpoint plane shape differs from ({ny},{nx}): {header}").into());
    }
    let data = &bytes[start + size..];
    let values = if compact.contains("'descr':'<f4'") {
        if data.len() != nx * ny * 4 {
            return Err("checkpoint f32 plane length differs from its shape".into());
        }
        data.chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()) as f64)
            .collect()
    } else if compact.contains("'descr':'<f8'") {
        if data.len() != nx * ny * 8 {
            return Err("checkpoint f64 plane length differs from its shape".into());
        }
        data.chunks_exact(8)
            .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
            .collect()
    } else {
        return Err(format!("unsupported checkpoint plane dtype: {header}").into());
    };
    Ok(CheckpointPlane { values, array })
}

#[cfg(test)]
mod tests {
    use super::read_checkpoint_plane;
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use zip::write::SimpleFileOptions;

    static NEXT_ARCHIVE: AtomicUsize = AtomicUsize::new(0);

    struct Archive {
        path: PathBuf,
    }

    impl Drop for Archive {
        fn drop(&mut self) {
            let bytes = fs::metadata(&self.path).expect("this test's archive metadata").len();
            fs::remove_file(&self.path).expect("delete only this test's tiny archive");
            eprintln!("deleted test archive {} ({bytes} bytes)", self.path.display());
        }
    }

    fn npy(descr: &str, shape: &str, fortran: bool, payload: &[u8]) -> Vec<u8> {
        let mut header = format!(
            "{{'descr': '{descr}', 'fortran_order': {}, 'shape': {shape}, }}",
            if fortran { "True" } else { "False" }
        );
        // NPY v1: ten-byte preamble, padded header ending in newline.
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    fn f32_plane(values: &[f32]) -> Vec<u8> {
        let data: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        npy("<f4", "(2, 2)", false, &data)
    }

    fn f64_plane(values: &[f64]) -> Vec<u8> {
        let data: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        npy("<f8", "(2, 2)", false, &data)
    }

    fn checkpoint(entries: &[(&str, Vec<u8>)]) -> Archive {
        let tmp = PathBuf::from(std::env::var_os("TMPDIR").expect("set owned TMPDIR for tests"));
        assert!(tmp.is_absolute() && tmp.is_dir());
        let path = tmp.join(format!(
            "albsol-scorer-{}-{}.npz",
            std::process::id(),
            NEXT_ARCHIVE.fetch_add(1, Ordering::Relaxed)
        ));
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        // Actual checkpoint metadata member and NPY byte-array representation;
        // the plane reader must select fields without confusing the header.
        let metadata = br#"{"elapsed_seconds":3600.0}"#;
        writer.start_file("__gpuwm_restart_header__.npy", options).unwrap();
        writer
            .write_all(&npy("|u1", &format!("({},)", metadata.len()), false, metadata))
            .unwrap();
        for (name, bytes) in entries {
            writer.start_file(format!("fields/{name}.npy"), options).unwrap();
            writer.write_all(bytes).unwrap();
        }
        writer.finish().unwrap();
        Archive { path }
    }

    #[test]
    fn missing_coszen_reads_f32_reference_and_reports_its_member() {
        let values = [-0.25_f32, 0.0, 0.25, 0.75];
        let archive = checkpoint(&[("coszen_ref", f32_plane(&values))]);
        let selected = read_checkpoint_plane(&archive.path, "coszen", 2, 2).unwrap();
        assert_eq!(selected.values, values.map(f64::from));
        assert_eq!(selected.array, "fields/coszen_ref.npy");
    }

    #[test]
    fn present_f64_coszen_has_precedence_and_reports_its_member() {
        let values = [-0.123456789_f64, 0.0, 0.375, 0.875];
        let archive = checkpoint(&[
            ("coszen", f64_plane(&values)),
            ("coszen_ref", f32_plane(&[1.0; 4])),
        ]);
        let selected = read_checkpoint_plane(&archive.path, "coszen", 2, 2).unwrap();
        assert_eq!(selected.values, values);
        assert_eq!(selected.array, "fields/coszen.npy");
    }

    #[test]
    fn neither_coszen_member_is_an_error() {
        let archive = checkpoint(&[("tsk", f32_plane(&[280.0; 4]))]);
        assert!(read_checkpoint_plane(&archive.path, "coszen", 2, 2).is_err());
    }

    #[test]
    fn malformed_present_coszen_does_not_fall_back_to_a_valid_reference() {
        let archive = checkpoint(&[
            ("coszen", b"not an NPY array".to_vec()),
            ("coszen_ref", f32_plane(&[0.5; 4])),
        ]);
        let error = read_checkpoint_plane(&archive.path, "coszen", 2, 2).unwrap_err();
        assert!(error.to_string().contains("checkpoint array is not NPY"));
    }

    #[test]
    fn another_missing_field_cannot_use_the_coszen_reference() {
        let archive = checkpoint(&[("coszen_ref", f64_plane(&[0.75; 4]))]);
        assert!(read_checkpoint_plane(&archive.path, "tsk", 2, 2).is_err());
        let selected = read_checkpoint_plane(&archive.path, "coszen", 2, 2).unwrap();
        assert_eq!(selected.values, [0.75; 4]);
        assert_eq!(selected.array, "fields/coszen_ref.npy");
    }

    #[test]
    fn present_plane_shape_dtype_order_and_length_errors_remain_errors() {
        let payload: Vec<u8> = [0.5_f32; 4].iter().flat_map(|x| x.to_le_bytes()).collect();
        for (bytes, message) in [
            (npy("<f4", "(1, 4)", false, &payload), "shape differs"),
            (npy("<i4", "(2, 2)", false, &payload), "unsupported checkpoint plane dtype"),
            (npy("<f4", "(2, 2)", true, &payload), "not C-order"),
            (npy("<f4", "(2, 2)", false, &payload[..12]), "length differs"),
        ] {
            let archive = checkpoint(&[
                ("coszen", bytes),
                ("coszen_ref", f32_plane(&[0.25; 4])),
            ]);
            let error = read_checkpoint_plane(&archive.path, "coszen", 2, 2).unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
        }
    }
}
