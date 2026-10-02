//! Read the history files out of a ZIP (the run page's download) one at a
//! time, so the disk holds one frame at a time rather than the whole
//! archive unpacked.  STORED and DEFLATE entries, ZIP64 sizes and offsets,
//! CRC checked on every extraction.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{fail, refuse, Result};

#[derive(Debug, Clone)]
pub struct ZipEntry {
    pub name: String,
    pub method: u16,
    pub crc: u32,
    pub compressed: u64,
    pub size: u64,
    pub local_offset: u64,
}

pub struct Archive {
    path: PathBuf,
    entries: Vec<ZipEntry>,
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

impl Archive {
    pub fn open(path: &Path) -> Result<Self> {
        let bad = |what: &str| {
            refuse(format!(
                "{} is not a readable ZIP ({what}), so no history file could be taken from it",
                path.display()
            ))
        };
        let mut file = File::open(path).map_err(|e| fail(format!("could not open {}: {e}", path.display())))?;
        let len = file.metadata()?.len();
        let tail_len = len.min(65_557);
        file.seek(SeekFrom::Start(len - tail_len))?;
        let mut tail = vec![0u8; tail_len as usize];
        file.read_exact(&mut tail)?;
        let eocd = (0..tail.len().saturating_sub(21))
            .rev()
            .find(|&i| u32_at(&tail, i) == 0x0605_4b50)
            .ok_or_else(|| bad("no end-of-central-directory record"))?;
        let mut count = u64::from(u16_at(&tail, eocd + 10));
        let mut cd_size = u64::from(u32_at(&tail, eocd + 12));
        let mut cd_offset = u64::from(u32_at(&tail, eocd + 16));
        if eocd >= 20 && u32_at(&tail, eocd - 20) == 0x0706_4b50 {
            let end64_at = u64_at(&tail, eocd - 20 + 8);
            let mut end64 = [0u8; 56];
            file.seek(SeekFrom::Start(end64_at))?;
            file.read_exact(&mut end64)?;
            if u32_at(&end64, 0) != 0x0606_4b50 {
                return Err(bad("a ZIP64 locator pointing at no ZIP64 record"));
            }
            count = u64_at(&end64, 32);
            cd_size = u64_at(&end64, 40);
            cd_offset = u64_at(&end64, 48);
        }
        if cd_offset.checked_add(cd_size).is_none_or(|end| end > len) {
            return Err(bad("a central directory past the end of the file"));
        }
        let mut cd = vec![0u8; cd_size as usize];
        file.seek(SeekFrom::Start(cd_offset))?;
        file.read_exact(&mut cd)?;
        if count > cd_size / 46 {
            return Err(bad("entry count exceeds the central directory size"));
        }
        let mut entries = Vec::new();
        let mut at = 0usize;
        while at + 46 <= cd.len() && u32_at(&cd, at) == 0x0201_4b50 {
            let method = u16_at(&cd, at + 10);
            let crc = u32_at(&cd, at + 16);
            let mut compressed = u64::from(u32_at(&cd, at + 20));
            let mut size = u64::from(u32_at(&cd, at + 24));
            let name_len = u16_at(&cd, at + 28) as usize;
            let extra_len = u16_at(&cd, at + 30) as usize;
            let comment_len = u16_at(&cd, at + 32) as usize;
            let mut local_offset = u64::from(u32_at(&cd, at + 42));
            let record_end = at + 46 + name_len + extra_len + comment_len;
            if record_end > cd.len() {
                return Err(bad("truncated central directory entry"));
            }
            let name = String::from_utf8_lossy(&cd[at + 46..at + 46 + name_len]).into_owned();
            let mut extra = &cd[at + 46 + name_len..at + 46 + name_len + extra_len];
            while extra.len() >= 4 {
                let id = u16_at(extra, 0);
                let n = u16_at(extra, 2) as usize;
                let body = &extra[4..(4 + n).min(extra.len())];
                if id == 1 {
                    let mut p = 0;
                    if size == 0xFFFF_FFFF && p + 8 <= body.len() {
                        size = u64_at(body, p);
                        p += 8;
                    }
                    if compressed == 0xFFFF_FFFF && p + 8 <= body.len() {
                        compressed = u64_at(body, p);
                        p += 8;
                    }
                    if local_offset == 0xFFFF_FFFF && p + 8 <= body.len() {
                        local_offset = u64_at(body, p);
                    }
                }
                extra = &extra[(4 + n).min(extra.len())..];
            }
            entries.push(ZipEntry { name, method, crc, compressed, size, local_offset });
            at = record_end;
        }
        if entries.len() as u64 != count || at != cd.len() {
            return Err(bad("central directory does not match its entry count or length"));
        }
        Ok(Archive { path: path.to_path_buf(), entries })
    }

    pub fn entries(&self) -> &[ZipEntry] {
        &self.entries
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Stream one entry's bytes into `out`, checking its CRC.
    pub fn extract(&self, entry: &ZipEntry, out: &mut dyn Write) -> Result<u64> {
        let mut file = File::open(&self.path)?;
        let mut local = [0u8; 30];
        file.seek(SeekFrom::Start(entry.local_offset))?;
        file.read_exact(&mut local)?;
        if u32_at(&local, 0) != 0x0403_4b50 {
            return Err(refuse(format!(
                "{} in {} has no local header where the directory says, so its bytes cannot be trusted",
                entry.name,
                self.path.display()
            )));
        }
        let skip = u64::from(u16_at(&local, 26)) + u64::from(u16_at(&local, 28));
        file.seek(SeekFrom::Start(entry.local_offset + 30 + skip))?;
        let raw = BufReader::with_capacity(8 << 20, file).take(entry.compressed);
        let mut reader: Box<dyn Read> = match entry.method {
            0 => Box::new(raw),
            8 => Box::new(flate2::read::DeflateDecoder::new(raw)),
            other => {
                return Err(refuse(format!(
                    "{} in {} is compressed with ZIP method {other}, which this reader does not decode, so it cannot be read",
                    entry.name,
                    self.path.display()
                )))
            }
        };
        let mut hasher = crc32fast::Hasher::new();
        let mut buffer = vec![0u8; 8 << 20];
        let mut total = 0u64;
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
            out.write_all(&buffer[..n])?;
            total += n as u64;
        }
        if total != entry.size || hasher.finalize() != entry.crc {
            return Err(refuse(format!(
                "{} in {} does not match its recorded size and checksum, so the archive is damaged",
                entry.name,
                self.path.display()
            )));
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn malformed_directory(name_len: u16, count: u16, suffix: &str) {
        let path = std::env::temp_dir().join(format!("mlx-bad-zip-{}-{suffix}.zip", std::process::id()));
        let mut bytes = vec![0u8; 46 + 22];
        bytes[0..4].copy_from_slice(&0x0201_4b50u32.to_le_bytes());
        bytes[28..30].copy_from_slice(&name_len.to_le_bytes());
        bytes[46..50].copy_from_slice(&0x0605_4b50u32.to_le_bytes());
        bytes[56..58].copy_from_slice(&count.to_le_bytes());
        bytes[58..62].copy_from_slice(&46u32.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        let error = match Archive::open(&path) { Err(error) => error, Ok(_) => panic!("damaged ZIP accepted") };
        assert!(error.is_refusal() && error.message().contains("not a readable ZIP"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_truncated_directory_name_is_refused_instead_of_panicking() {
        malformed_directory(100, 1, "truncated-name");
    }

    #[test]
    fn a_directory_entry_count_mismatch_is_refused() {
        malformed_directory(0, 0, "wrong-count");
    }
}
