//! A ZIP writer for the one-download form of an export: every entry STORED
//! (the chunks are already Blosc-compressed, so deflating them again buys
//! nothing and costs the in-place read), ZIP64 records wherever a size, an
//! offset or the entry count passes the classic limits.
//!
//! STORED is what makes the archive open in place: `zarr.storage.ZipStore`
//! reads a chunk with one seek and one read.  Entries are written in sorted
//! path order with a fixed timestamp (1980-01-01 00:00), so two exports of
//! the same datasets are the same archive byte for byte.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{fail, Result};

const LOCAL: u32 = 0x0403_4b50;
const CENTRAL: u32 = 0x0201_4b50;
const END: u32 = 0x0605_4b50;
const END64: u32 = 0x0606_4b50;
const LOCATOR64: u32 = 0x0706_4b50;
const DOS_DATE_1980_01_01: u16 = (0 << 9) | (1 << 5) | 1;
const LIMIT32: u64 = 0xFFFF_FFFF;

struct Entry {
    name: String,
    crc: u32,
    size: u64,
    offset: u64,
}

/// Every file under `root`, as (`archive name`, path), sorted by name.  The
/// archive name is `prefix/relative/path` with forward slashes.
pub fn collect(root: &Path, prefix: &str) -> Result<Vec<(String, PathBuf)>> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                stack.push(path);
            } else {
                let relative = path
                    .strip_prefix(root)
                    .map_err(|e| fail(format!("{e}")))?
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                found.push((format!("{prefix}/{relative}"), path));
            }
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(found)
}

/// Write `files` into a STORED ZIP at `dest`.  Returns the archive's size.
pub fn write(dest: &Path, files: &[(String, PathBuf)]) -> Result<u64> {
    write_with(dest, files, &[])
}

/// Like [`write`], plus entries held in memory; every entry is written in
/// sorted name order.
pub fn write_with(dest: &Path, files: &[(String, PathBuf)], memory: &[(String, Vec<u8>)]) -> Result<u64> {
    enum Src<'a> {
        Disk(&'a Path),
        Memory(&'a [u8]),
    }
    let mut all: Vec<(&str, Src<'_>)> = files.iter().map(|(n, p)| (n.as_str(), Src::Disk(p.as_path()))).collect();
    all.extend(memory.iter().map(|(n, b)| (n.as_str(), Src::Memory(b.as_slice()))));
    all.sort_by(|a, b| a.0.cmp(b.0));
    let partial = dest.with_extension("zip.partial");
    let file = File::create(&partial)
        .map_err(|e| fail(format!("could not create {}: {e}", partial.display())))?;
    let mut out = BufWriter::with_capacity(8 << 20, file);
    let mut entries = Vec::with_capacity(files.len());
    let mut offset: u64 = 0;
    let mut buffer = vec![0u8; 8 << 20];
    for (name, src) in &all {
        let name = name.to_string();
        let size = match src {
            Src::Disk(path) => std::fs::metadata(path)?.len(),
            Src::Memory(bytes) => bytes.len() as u64,
        };
        // CRC first, so the local header carries it and no data descriptor
        // is needed (readers that ignore descriptors still read it).
        let mut hasher = crc32fast::Hasher::new();
        match src {
            Src::Disk(path) => {
                let mut source = File::open(path)?;
                loop {
                    let n = source.read(&mut buffer)?;
                    if n == 0 {
                        break;
                    }
                    hasher.update(&buffer[..n]);
                }
            }
            Src::Memory(bytes) => hasher.update(bytes),
        }
        let crc = hasher.finalize();
        let zip64_local = size >= LIMIT32;
        let name_bytes = name.as_bytes();
        out.write_all(&LOCAL.to_le_bytes())?;
        out.write_all(&(if zip64_local { 45u16 } else { 20u16 }).to_le_bytes())?;
        out.write_all(&0x0800u16.to_le_bytes())?; // UTF-8 names
        out.write_all(&0u16.to_le_bytes())?; // STORED
        out.write_all(&0u16.to_le_bytes())?; // time 00:00
        out.write_all(&DOS_DATE_1980_01_01.to_le_bytes())?;
        out.write_all(&crc.to_le_bytes())?;
        let small = if zip64_local { LIMIT32 as u32 } else { size as u32 };
        out.write_all(&small.to_le_bytes())?;
        out.write_all(&small.to_le_bytes())?;
        out.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
        let extra_len: u16 = if zip64_local { 20 } else { 0 };
        out.write_all(&extra_len.to_le_bytes())?;
        out.write_all(name_bytes)?;
        if zip64_local {
            out.write_all(&1u16.to_le_bytes())?;
            out.write_all(&16u16.to_le_bytes())?;
            out.write_all(&size.to_le_bytes())?;
            out.write_all(&size.to_le_bytes())?;
        }
        let copied = match src {
            Src::Disk(path) => {
                let mut source = File::open(path)?;
                source.seek(SeekFrom::Start(0))?;
                std::io::copy(&mut source, &mut out)?
            }
            Src::Memory(bytes) => {
                out.write_all(bytes)?;
                bytes.len() as u64
            }
        };
        if copied != size {
            return Err(fail(format!("{name} changed while it was being archived")));
        }
        entries.push(Entry { name: name.clone(), crc, size, offset });
        offset += 30 + name_bytes.len() as u64 + u64::from(extra_len) + size;
    }
    let central_start = offset;
    for e in &entries {
        let need_size = e.size >= LIMIT32;
        let need_offset = e.offset >= LIMIT32;
        let mut extra = Vec::new();
        if need_size || need_offset {
            let mut body = Vec::new();
            if need_size {
                body.extend_from_slice(&e.size.to_le_bytes());
                body.extend_from_slice(&e.size.to_le_bytes());
            }
            if need_offset {
                body.extend_from_slice(&e.offset.to_le_bytes());
            }
            extra.extend_from_slice(&1u16.to_le_bytes());
            extra.extend_from_slice(&(body.len() as u16).to_le_bytes());
            extra.extend_from_slice(&body);
        }
        let version: u16 = if extra.is_empty() { 20 } else { 45 };
        let name_bytes = e.name.as_bytes();
        out.write_all(&CENTRAL.to_le_bytes())?;
        out.write_all(&version.to_le_bytes())?; // made by
        out.write_all(&version.to_le_bytes())?; // needed
        out.write_all(&0x0800u16.to_le_bytes())?;
        out.write_all(&0u16.to_le_bytes())?;
        out.write_all(&0u16.to_le_bytes())?;
        out.write_all(&DOS_DATE_1980_01_01.to_le_bytes())?;
        out.write_all(&e.crc.to_le_bytes())?;
        let small = if need_size { LIMIT32 as u32 } else { e.size as u32 };
        out.write_all(&small.to_le_bytes())?;
        out.write_all(&small.to_le_bytes())?;
        out.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
        out.write_all(&(extra.len() as u16).to_le_bytes())?;
        out.write_all(&0u16.to_le_bytes())?; // comment
        out.write_all(&0u16.to_le_bytes())?; // disk
        out.write_all(&0u16.to_le_bytes())?; // internal attributes
        out.write_all(&0u32.to_le_bytes())?; // external attributes
        let small_offset = if need_offset { LIMIT32 as u32 } else { e.offset as u32 };
        out.write_all(&small_offset.to_le_bytes())?;
        out.write_all(name_bytes)?;
        out.write_all(&extra)?;
        offset += 46 + name_bytes.len() as u64 + extra.len() as u64;
    }
    let central_size = offset - central_start;
    let count = entries.len() as u64;
    let need64 = count >= 0xFFFF || central_start >= LIMIT32 || central_size >= LIMIT32;
    if need64 {
        let end64_at = offset;
        out.write_all(&END64.to_le_bytes())?;
        out.write_all(&44u64.to_le_bytes())?;
        out.write_all(&45u16.to_le_bytes())?;
        out.write_all(&45u16.to_le_bytes())?;
        out.write_all(&0u32.to_le_bytes())?;
        out.write_all(&0u32.to_le_bytes())?;
        out.write_all(&count.to_le_bytes())?;
        out.write_all(&count.to_le_bytes())?;
        out.write_all(&central_size.to_le_bytes())?;
        out.write_all(&central_start.to_le_bytes())?;
        out.write_all(&LOCATOR64.to_le_bytes())?;
        out.write_all(&0u32.to_le_bytes())?;
        out.write_all(&end64_at.to_le_bytes())?;
        out.write_all(&1u32.to_le_bytes())?;
        offset += 56 + 20;
    }
    out.write_all(&END.to_le_bytes())?;
    out.write_all(&0u16.to_le_bytes())?;
    out.write_all(&0u16.to_le_bytes())?;
    let small_count = if need64 { 0xFFFF } else { count as u16 };
    out.write_all(&small_count.to_le_bytes())?;
    out.write_all(&small_count.to_le_bytes())?;
    out.write_all(&(if need64 { LIMIT32 as u32 } else { central_size as u32 }).to_le_bytes())?;
    out.write_all(&(if need64 { LIMIT32 as u32 } else { central_start as u32 }).to_le_bytes())?;
    out.write_all(&0u16.to_le_bytes())?;
    offset += 22;
    out.flush()?;
    drop(out);
    std::fs::rename(&partial, dest)
        .map_err(|e| fail(format!("could not move {} into place: {e}", dest.display())))?;
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stored_archive_lists_and_reads_back() {
        let dir = std::env::temp_dir().join(format!("mlx-zip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("data/a.zarr/t")).unwrap();
        std::fs::write(dir.join("data/README.txt"), b"hello\n").unwrap();
        std::fs::write(dir.join("data/a.zarr/t/0.0"), vec![7u8; 1000]).unwrap();
        let files = collect(&dir.join("data"), "data").unwrap();
        let names: Vec<_> = files.iter().map(|f| f.0.as_str()).collect();
        assert_eq!(names, ["data/README.txt", "data/a.zarr/t/0.0"]);
        let dest = dir.join("data-ml.zip");
        let size = write(&dest, &files).unwrap();
        assert_eq!(size, std::fs::metadata(&dest).unwrap().len());
        let archive = crate::zipin::Archive::open(&dest).unwrap();
        let entries = archive.entries();
        assert_eq!(entries.len(), 2);
        let mut out = Vec::new();
        archive.extract(&entries[1], &mut out).unwrap();
        assert_eq!(out, vec![7u8; 1000]);
        let again = dir.join("again-ml.zip");
        write(&again, &files).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), std::fs::read(&again).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
