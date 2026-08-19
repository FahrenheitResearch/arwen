use std::fs::File;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::error::{Error, Result};

#[derive(Clone)]
enum StorageBacking {
    Bytes(Arc<[u8]>),
    Mmap(Arc<Mmap>),
}

/// An immutable byte range returned by a storage backend.
#[derive(Clone)]
pub struct StorageBuffer {
    backing: StorageBacking,
    start: usize,
    len: usize,
}

impl StorageBuffer {
    pub fn from_vec(bytes: Vec<u8>) -> Self {
        let len = bytes.len();
        Self {
            backing: StorageBacking::Bytes(Arc::<[u8]>::from(bytes)),
            start: 0,
            len,
        }
    }

    pub(crate) fn from_arc_bytes(bytes: Arc<[u8]>, start: usize, len: usize) -> Self {
        Self {
            backing: StorageBacking::Bytes(bytes),
            start,
            len,
        }
    }

    pub(crate) fn from_arc_mmap(mmap: Arc<Mmap>, start: usize, len: usize) -> Self {
        Self {
            backing: StorageBacking::Mmap(mmap),
            start,
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl AsRef<[u8]> for StorageBuffer {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl Deref for StorageBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match &self.backing {
            StorageBacking::Bytes(bytes) => &bytes[self.start..self.start + self.len],
            StorageBacking::Mmap(mmap) => &mmap[self.start..self.start + self.len],
        }
    }
}

/// Random-access, immutable byte storage for HDF5 parsing and reads.
pub trait Storage: Send + Sync {
    /// Total length in bytes.
    fn len(&self) -> u64;

    /// Returns `true` if the storage is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read a byte range from `offset..offset+len`.
    fn read_range(&self, offset: u64, len: usize) -> Result<StorageBuffer>;
}

pub type DynStorage = Arc<dyn Storage>;

/// In-memory storage backed by owned bytes.
pub struct BytesStorage {
    data: Arc<[u8]>,
}

impl BytesStorage {
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data: Arc::<[u8]>::from(data),
        }
    }
}

impl Storage for BytesStorage {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }

    fn read_range(&self, offset: u64, len: usize) -> Result<StorageBuffer> {
        let start = usize::try_from(offset).map_err(|_| Error::OffsetOutOfBounds(offset))?;
        let end = start
            .checked_add(len)
            .ok_or(Error::OffsetOutOfBounds(offset))?;
        if end > self.data.len() {
            return Err(Error::UnexpectedEof {
                offset,
                needed: len as u64,
                available: self.len().saturating_sub(offset),
            });
        }
        Ok(StorageBuffer::from_arc_bytes(self.data.clone(), start, len))
    }
}

/// In-memory storage backed by a read-only memory map.
pub struct MmapStorage {
    mmap: Arc<Mmap>,
}

impl MmapStorage {
    pub fn new(mmap: Mmap) -> Self {
        Self {
            mmap: Arc::new(mmap),
        }
    }
}

impl Storage for MmapStorage {
    fn len(&self) -> u64 {
        self.mmap.len() as u64
    }

    fn read_range(&self, offset: u64, len: usize) -> Result<StorageBuffer> {
        let start = usize::try_from(offset).map_err(|_| Error::OffsetOutOfBounds(offset))?;
        let end = start
            .checked_add(len)
            .ok_or(Error::OffsetOutOfBounds(offset))?;
        if end > self.mmap.len() {
            return Err(Error::UnexpectedEof {
                offset,
                needed: len as u64,
                available: self.len().saturating_sub(offset),
            });
        }
        Ok(StorageBuffer::from_arc_mmap(self.mmap.clone(), start, len))
    }
}

/// File-backed storage that serves explicit byte ranges via positional reads.
pub struct FileStorage {
    file: Arc<File>,
    len: u64,
}

impl FileStorage {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            file: Arc::new(file),
            len,
        })
    }
}

impl Storage for FileStorage {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_range(&self, offset: u64, len: usize) -> Result<StorageBuffer> {
        let needed = u64::try_from(len).map_err(|_| Error::OffsetOutOfBounds(offset))?;
        let end = offset
            .checked_add(needed)
            .ok_or(Error::OffsetOutOfBounds(offset))?;
        if end > self.len {
            return Err(Error::UnexpectedEof {
                offset,
                needed,
                available: self.len.saturating_sub(offset),
            });
        }

        let mut buf = vec![0u8; len];
        read_exact_at(self.file.as_ref(), &mut buf, offset)?;
        Ok(StorageBuffer::from_vec(buf))
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;

    while !buf.is_empty() {
        let n = file.read_at(buf, offset)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "failed to fill whole buffer",
            ));
        }
        offset += n as u64;
        buf = &mut buf[n..];
    }
    Ok(())
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;

    while !buf.is_empty() {
        let n = file.seek_read(buf, offset)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "failed to fill whole buffer",
            ));
        }
        offset += n as u64;
        buf = &mut buf[n..];
    }
    Ok(())
}
