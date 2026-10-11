//! Ordered, bounded payload consumption. Transport retries and pacing stay in wx-core.
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, serde::Serialize)]
pub struct Receipt {
    pub bytes: u64,
    pub sha256: String,
    pub grib2_messages: usize,
    pub last_byte_unix: f64,
    pub publication: Option<Publication>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Publication {
    pub dev: u64,
    pub ino: u64,
    pub bytes: u64,
    pub mtime_ns: i128,
    pub ctime_ns: i128,
}

#[cfg(unix)]
fn publication(path: &Path) -> Result<Option<Publication>, String> {
    use std::os::unix::fs::MetadataExt;
    let meta = fs::metadata(path).map_err(|e| format!("{}: cannot stat published payload: {e}", path.display()))?;
    Ok(Some(Publication { dev: meta.dev(), ino: meta.ino(), bytes: meta.len(),
        mtime_ns: i128::from(meta.mtime()) * 1_000_000_000 + i128::from(meta.mtime_nsec()),
        ctime_ns: i128::from(meta.ctime()) * 1_000_000_000 + i128::from(meta.ctime_nsec()) }))
}

/// The calls gpuwm.file_change_watch.stat_identity makes, field for field:
/// GetFileInformationByHandle's volume serial, file index and size, and
/// FILE_BASIC_INFO's LastWriteTime and ChangeTime. Not std's metadata:
/// its Windows "created" time is no change time, and the serial and index
/// accessors are unstable.
#[cfg(windows)]
fn publication(path: &Path) -> Result<Option<Publication>, String> {
    use std::ffi::c_void;
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;

    #[repr(C)]
    #[derive(Default)]
    struct ByHandle {
        attributes: u32,
        creation: [u32; 2],
        access: [u32; 2],
        write: [u32; 2],
        volume_serial: u32,
        size_high: u32,
        size_low: u32,
        links: u32,
        index_high: u32,
        index_low: u32,
    }
    #[repr(C)]
    #[derive(Default)]
    struct Basic {
        creation: i64,
        access: i64,
        write: i64,
        change: i64,
        attributes: u32,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetFileInformationByHandle(file: *mut c_void, info: *mut ByHandle) -> i32;
        fn GetFileInformationByHandleEx(file: *mut c_void, class: i32, info: *mut c_void, size: u32) -> i32;
    }
    const FILE_READ_ATTRIBUTES: u32 = 0x80;
    const SHARE_ALL: u32 = 0x1 | 0x2 | 0x4;
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    const FILE_BASIC_INFO: i32 = 0;
    const UNIX_EPOCH_FILETIME: i128 = 116_444_736_000_000_000;

    let cannot = |e: std::io::Error| format!("{}: cannot read the published payload's identity: {e}", path.display());
    let file = fs::OpenOptions::new().access_mode(FILE_READ_ATTRIBUTES).share_mode(SHARE_ALL)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS).open(path).map_err(cannot)?;
    let handle = file.as_raw_handle() as *mut c_void;
    let mut by_handle = ByHandle::default();
    let mut basic = Basic::default();
    // SAFETY: the handle is open for the duration of both calls, and each
    // buffer is a live, writable value of the size the call is told.
    let read = unsafe {
        GetFileInformationByHandle(handle, &mut by_handle) != 0
            && GetFileInformationByHandleEx(handle, FILE_BASIC_INFO, (&mut basic as *mut Basic).cast(),
                std::mem::size_of::<Basic>() as u32) != 0
    };
    if !read {
        return Err(cannot(std::io::Error::last_os_error()));
    }
    Ok(Some(Publication { dev: u64::from(by_handle.volume_serial),
        ino: (u64::from(by_handle.index_high) << 32) | u64::from(by_handle.index_low),
        bytes: (u64::from(by_handle.size_high) << 32) | u64::from(by_handle.size_low),
        mtime_ns: (i128::from(basic.write) - UNIX_EPOCH_FILETIME) * 100,
        ctime_ns: (i128::from(basic.change) - UNIX_EPOCH_FILETIME) * 100 }))
}

#[cfg(not(any(unix, windows)))]
fn publication(_path: &Path) -> Result<Option<Publication>, String> {
    // Without a change-time identity, Python retains its independent disk check.
    Ok(None)
}

pub fn unix_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

/// Holds only Section 0 and the last four bytes, even for a large message.
#[derive(Default)]
struct Envelope {
    header: Vec<u8>,
    remaining: u64,
    tail: Vec<u8>,
    messages: usize,
}

impl Envelope {
    fn push(&mut self, mut bytes: &[u8]) -> Result<(), String> {
        while !bytes.is_empty() {
            if self.remaining == 0 {
                let take = (16 - self.header.len()).min(bytes.len());
                self.header.extend_from_slice(&bytes[..take]);
                bytes = &bytes[take..];
                if self.header.len() < 16 { continue; }
                if &self.header[..4] != b"GRIB" || self.header[7] != 2 {
                    return Err(format!("message {} lacks a GRIB2 indicator", self.messages));
                }
                let length = u64::from_be_bytes(self.header[8..16].try_into().unwrap());
                if length < 20 {
                    return Err(format!("message {} declares invalid length {length}", self.messages));
                }
                self.remaining = length - 16;
                self.header.clear();
                self.tail.clear();
            }
            let take = bytes.len().min(self.remaining.min(usize::MAX as u64) as usize);
            if take > 0 {
                // Only bytes belonging to this message may enter its terminator.
                let part = &bytes[..take];
                if part.len() >= 4 {
                    self.tail.clear();
                    self.tail.extend_from_slice(&part[part.len() - 4..]);
                } else {
                    self.tail.extend_from_slice(part);
                    if self.tail.len() > 4 { self.tail.drain(..self.tail.len() - 4); }
                }
                self.remaining -= take as u64;
                bytes = &bytes[take..];
            }
            if self.remaining == 0 {
                if self.tail != b"7777" {
                    return Err(format!("message {} lacks the 7777 terminator", self.messages));
                }
                self.messages += 1;
            }
        }
        Ok(())
    }

    fn finish(&self) -> Result<usize, String> {
        if self.messages == 0 || self.remaining != 0 || !self.header.is_empty() {
            return Err(format!("truncated GRIB2 stream after {} messages: {} body bytes missing, {} header bytes present",
                self.messages, self.remaining, self.header.len()));
        }
        Ok(self.messages)
    }
}

pub struct Publisher {
    file: File,
    digest: Sha256,
    envelope: Envelope,
    bytes: u64,
    pin: Option<(u64, String)>,
}

impl Publisher {
    pub fn new(path: &Path) -> Result<Self, String> {
        let file = File::options().write(true).create_new(true).open(path)
            .map_err(|e| format!("could not create {}: {e}", path.display()))?;
        Ok(Self { file, digest: Sha256::new(), envelope: Envelope::default(), bytes: 0, pin: None })
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<(), String> {
        if self.pin.is_none() { self.envelope.push(bytes)?; }
        self.file.write_all(bytes).map_err(|e| format!("could not write payload: {e}"))?;
        self.digest.update(bytes);
        self.bytes += bytes.len() as u64;
        Ok(())
    }

    pub fn finish(self, last_byte_unix: f64) -> Result<Receipt, String> {
        let messages = if self.pin.is_none() { self.envelope.finish()? } else { 0 };
        let sha256 = format!("{:x}", self.digest.finalize());
        if let Some((bytes, expected)) = self.pin {
            if bytes != self.bytes || (!expected.is_empty() && expected != sha256) {
                return Err(format!("pinned object refused: received {} bytes SHA-256 {sha256}, expected {bytes} bytes SHA-256 {expected}; other bytes would change geography or lookup coefficients", self.bytes));
            }
        }
        self.file.sync_all().map_err(|e| format!("could not flush payload: {e}"))?;
        Ok(Receipt { bytes: self.bytes, sha256,
            grib2_messages: messages, last_byte_unix, publication: None })
    }
}

/// Publish only a complete verified stream. Remove only our newly created partial.
pub fn publish_with<F>(path: &Path, transfer: F) -> Result<Receipt, crate::net::TransferFault>
where F: FnOnce(&mut Publisher) -> Result<f64, crate::net::TransferFault> {
    publish_config(path, None, transfer)
}

pub fn publish_pinned_with<F>(path: &Path, bytes: u64, sha256: &str, transfer: F)
    -> Result<Receipt, crate::net::TransferFault>
where F: FnOnce(&mut Publisher) -> Result<f64, crate::net::TransferFault> {
    publish_config(path, Some((bytes, sha256.to_owned())), transfer)
}

fn publish_config<F>(path: &Path, pin: Option<(u64, String)>, transfer: F)
    -> Result<Receipt, crate::net::TransferFault>
where F: FnOnce(&mut Publisher) -> Result<f64, crate::net::TransferFault> {
    use crate::net::TransferFault;
    if path.exists() {
        return Err(TransferFault::Refused(format!("refusing to replace an existing file: {}", path.display())));
    }
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let partial = path.with_file_name(format!("{}.part-{}-{stamp}",
        path.file_name().unwrap().to_string_lossy(), std::process::id()));
    let mut publisher = Publisher::new(&partial).map_err(TransferFault::Refused)?;
    publisher.pin = pin;
    let result = transfer(&mut publisher).and_then(|last| publisher.finish(last).map_err(TransferFault::Refused));
    match result {
        Ok(mut receipt) => {
            // Atomic no-replace publication, including concurrent direct CLI calls.
            if let Err(e) = fs::hard_link(&partial, path) {
                let _ = fs::remove_file(&partial);
                return Err(TransferFault::Refused(format!("could not publish {} without replacing existing bytes: {e}", path.display())));
            }
            let _ = fs::remove_file(&partial);
            receipt.publication = publication(path).map_err(TransferFault::Refused)?;
            Ok(receipt)
        }
        Err(error) => {
            let _ = fs::remove_file(&partial);
            Err(match error {
                TransferFault::Network(text) => TransferFault::Network(format!("{}: {text}", path.display())),
                TransferFault::Refused(text) => TransferFault::Refused(format!("{}: {text}", path.display())),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn message(length: usize) -> Vec<u8> {
        let mut bytes = vec![0; length];
        bytes[..4].copy_from_slice(b"GRIB"); bytes[7] = 2;
        bytes[8..16].copy_from_slice(&(length as u64).to_be_bytes());
        bytes[length-4..].copy_from_slice(b"7777"); bytes
    }
    #[test]
    fn split_headers_and_terminators_tile_exactly() {
        let bytes = [message(20), message(113), message(24)].concat();
        for width in 1..=bytes.len() {
            let mut envelope = Envelope::default();
            for part in bytes.chunks(width) { envelope.push(part).unwrap(); }
            assert_eq!(envelope.finish().unwrap(), 3);
        }
    }
    #[test]
    fn refuses_truncated_and_corrupt_middle_records() {
        let mut bytes = [message(20), message(25), message(20)].concat();
        bytes[41] = 0;
        assert!(Envelope::default().push(&bytes).unwrap_err().contains("message 1"));
        for cut in 1..20 {
            let mut env = Envelope::default(); env.push(&message(20)[..cut]).unwrap();
            assert!(env.finish().unwrap_err().contains("truncated"));
        }
    }
}
