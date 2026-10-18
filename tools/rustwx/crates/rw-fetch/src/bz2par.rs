//! Block-parallel bzip2 decoding with bounded memory and byte-identical output.
//!
//! THE BREAKAGE: `rw_fetch unpack` decoded every `.tar.bz2` on one thread
//! (`MultiBzDecoder`), so the WPS greenfrac archive (0.99 GB to 11.2 GB) took
//! 4-5 minutes while 23 cores sat idle.
//!
//! A bzip2 stream is a sequence of independent blocks, each starting with the
//! 48-bit magic 0x314159265359 at an arbitrary bit offset and carrying its own
//! CRC; a stream ends with 0x177245385090 and the combined CRC. A producer
//! thread scans the compressed bytes (in parallel per chunk) for those magics
//! and cuts each block's bits into a one-block stream ("BZh9" + block +
//! end-of-stream + the block CRC); workers decode those with libbz2, and the
//! reader hands the decoded blocks out in order, checking every stream's
//! combined CRC as `bzip2 -d` does. Every block is checked by its own CRC, so
//! a magic that occurs by chance inside compressed data (about once per 35 TB
//! of input) only makes the two pieces fail; the reader then decodes them
//! joined. A chance end-of-stream magic is accepted only when a valid stream
//! header or the end of the file follows it. Memory is bounded by a fixed
//! number of blocks in flight.
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

const BLOCK_MAGIC: u64 = 0x3141_5926_5359;
const EOS_MAGIC: u64 = 0x1772_4538_5090;
const MASK48: u64 = (1 << 48) - 1;
/// Bits after an end-of-stream magic that must be present to judge it: CRC (32),
/// padding (<8), the next header (32) and the next magic (48).
const EOS_LOOKAHEAD: u64 = 48 + 32 + 8 + 32 + 48;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind { Block, Eos }

/// A run of bits `[start, end)` inside one shared input buffer.
#[derive(Clone)]
struct Range { buf: Arc<Vec<u8>>, start: u64, end: u64 }

enum Item {
    Decoded { data: Vec<u8>, crc: u32, ranges: Vec<Range> },
    Failed { ranges: Vec<Range>, reason: String },
    StreamEnd { stored: u32 },
    Error(String),
    Done,
}

struct Job { index: u64, range: Range }

#[derive(Clone, Default)]
pub(crate) struct Options {
    pub threads: usize,
    pub chunk: usize,
    /// Tests only: absolute bit offsets reported as chance magics.
    pub fake: Vec<(u64, bool)>,
}

pub(crate) struct ParBz2 {
    results: Option<Receiver<(u64, Item)>>,
    tokens: Option<SyncSender<()>>,
    pending: BTreeMap<u64, Item>,
    next: u64,
    out: Vec<u8>,
    pos: usize,
    combined: u32,
    finished: bool,
    handles: Vec<JoinHandle<()>>,
}

pub(crate) fn open(path: &Path) -> io::Result<ParBz2> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    open_with(path, Options { threads, chunk: 16 << 20, fake: Vec::new() })
}

pub(crate) fn open_with(path: &Path, options: Options) -> io::Result<ParBz2> {
    let file = File::open(path)?;
    let threads = options.threads.max(1);
    // Blocks in flight or decoded and waiting: bounds memory at a few per core.
    let slots = 2 * threads + 2;
    let (token_tx, token_rx) = mpsc::sync_channel::<()>(slots);
    for _ in 0..slots { token_tx.send(()).expect("fresh channel has room"); }
    let (result_tx, result_rx) = mpsc::channel::<(u64, Item)>();
    let (job_tx, job_rx) = mpsc::channel::<Job>();
    let job_rx = Arc::new(Mutex::new(job_rx));
    let mut handles = Vec::with_capacity(threads + 1);
    for _ in 0..threads {
        let jobs = Arc::clone(&job_rx);
        let results = result_tx.clone();
        handles.push(std::thread::spawn(move || loop {
            let job = { let guard = jobs.lock().unwrap(); guard.recv() };
            let Ok(job) = job else { return };
            let ranges = vec![job.range];
            let item = match decode_ranges(&ranges) {
                Ok((data, crc)) => Item::Decoded { data, crc, ranges },
                Err(reason) => Item::Failed { ranges, reason },
            };
            if results.send((job.index, item)).is_err() { return; }
        }));
    }
    handles.push(std::thread::spawn(move || {
        let mut producer = Producer { file, options, jobs: job_tx, results: result_tx.clone(), tokens: token_rx, index: 0 };
        let end = match producer.run() {
            Ok(()) => Item::Done,
            Err(e) => Item::Error(e),
        };
        let _ = result_tx.send((producer.index, end));
    }));
    Ok(ParBz2 { results: Some(result_rx), tokens: Some(token_tx), pending: BTreeMap::new(), next: 0,
                out: Vec::new(), pos: 0, combined: 0, finished: false, handles })
}

impl Drop for ParBz2 {
    fn drop(&mut self) {
        // Closing the token and result channels stops the producer, which closes
        // the job channel and so stops every worker.
        self.tokens.take();
        self.results.take();
        self.pending.clear();
        for handle in self.handles.drain(..) { let _ = handle.join(); }
    }
}

fn corrupt(message: String) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, message) }

impl ParBz2 {
    fn take(&mut self, index: u64) -> io::Result<Item> {
        loop {
            if let Some(item) = self.pending.remove(&index) { return Ok(item); }
            let results = self.results.as_ref().expect("reader is open");
            let (at, item) = results.recv().map_err(|_| corrupt("bzip2 decoder threads stopped".into()))?;
            self.pending.insert(at, item);
        }
    }

    fn release(&self, count: usize) {
        if let Some(tokens) = &self.tokens { for _ in 0..count { let _ = tokens.send(()); } }
    }

    /// Fetch the next decoded block (joining pieces split at a chance magic).
    fn advance(&mut self) -> io::Result<bool> {
        loop {
            let item = self.take(self.next)?;
            self.next += 1;
            match item {
                Item::Done => { self.finished = true; return Ok(false); }
                Item::Error(e) => return Err(corrupt(e)),
                Item::StreamEnd { stored } => {
                    self.release(1);
                    if stored != self.combined {
                        return Err(corrupt(format!("stream CRC {stored:08x} differs from the blocks' combined CRC {:08x}", self.combined)));
                    }
                    self.combined = 0;
                }
                Item::Decoded { data, crc, .. } => {
                    self.release(1);
                    self.combined = self.combined.rotate_left(1) ^ crc;
                    if data.is_empty() { continue; }
                    self.out = data;
                    self.pos = 0;
                    return Ok(true);
                }
                Item::Failed { mut ranges, reason } => {
                    let mut used = 1;
                    let mut last = reason;
                    loop {
                        // A chance magic splits one block in two; more pieces is damage.
                        if used >= 4 { return Err(corrupt(format!("bzip2 block refused: {last}"))); }
                        match self.take(self.next)? {
                            Item::Decoded { ranges: more, .. } | Item::Failed { ranges: more, .. } => {
                                self.next += 1;
                                used += 1;
                                ranges.extend(more);
                                match decode_ranges(&ranges) {
                                    Ok((data, crc)) => {
                                        self.release(used);
                                        self.combined = self.combined.rotate_left(1) ^ crc;
                                        self.out = data;
                                        self.pos = 0;
                                        if self.out.is_empty() { break; }
                                        return Ok(true);
                                    }
                                    Err(e) => last = e,
                                }
                            }
                            Item::Error(e) => return Err(corrupt(e)),
                            other => {
                                self.pending.insert(self.next, other);
                                return Err(corrupt(format!("bzip2 block refused: {last}")));
                            }
                        }
                    }
                }
            }
        }
    }
}

impl Read for ParBz2 {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() { return Ok(0); }
        while self.pos == self.out.len() {
            if self.finished || !self.advance()? { return Ok(0); }
        }
        let n = buf.len().min(self.out.len() - self.pos);
        buf[..n].copy_from_slice(&self.out[self.pos..self.pos + n]);
        self.pos += n;
        if self.pos == self.out.len() { self.out = Vec::new(); self.pos = 0; }
        Ok(n)
    }
}

fn bits_at(buf: &[u8], bit: u64, count: u32) -> u64 {
    let mut value = 0u64;
    for k in 0..count as u64 {
        let p = bit + k;
        value = (value << 1) | ((buf[(p / 8) as usize] >> (7 - (p % 8))) & 1) as u64;
    }
    value
}

fn magic_at(buf: &[u8], bit: u64) -> Option<Kind> {
    if bit + 48 > buf.len() as u64 * 8 { return None; }
    match bits_at(buf, bit, 48) { BLOCK_MAGIC => Some(Kind::Block), EOS_MAGIC => Some(Kind::Eos), _ => None }
}

/// For each value of the second byte of a 64-bit window, the shifts at which
/// either magic could start in the window's first byte.
fn shift_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    for magic in [BLOCK_MAGIC, EOS_MAGIC] {
        for shift in 0..8 {
            let window = magic << (16 - shift);
            table[((window >> 48) & 0xff) as usize] |= 1 << shift;
        }
    }
    table
}

/// Every magic starting at a bit in `[lo, hi)`, in order.
fn scan(buf: &[u8], lo: u64, hi: u64, table: &[u8; 256]) -> Vec<(u64, Kind)> {
    let bits = buf.len() as u64 * 8;
    let mut out = Vec::new();
    let mut i = (lo / 8) as usize;
    while (i as u64) * 8 < hi {
        if i + 1 < buf.len() {
            let mut shifts = table[buf[i + 1] as usize];
            while shifts != 0 {
                let s = shifts.trailing_zeros() as u64;
                shifts &= shifts - 1;
                let p = i as u64 * 8 + s;
                if p >= lo && p < hi && p + 48 <= bits {
                    let mut window = 0u64;
                    for k in 0..8 { window = (window << 8) | *buf.get(i + k).unwrap_or(&0) as u64; }
                    match (window >> (16 - s)) & MASK48 {
                        BLOCK_MAGIC => out.push((p, Kind::Block)),
                        EOS_MAGIC => out.push((p, Kind::Eos)),
                        _ => {}
                    }
                }
            }
        }
        i += 1;
    }
    out
}

fn scan_parallel(buf: &[u8], lo: u64, threads: usize, table: &[u8; 256]) -> Vec<(u64, Kind)> {
    let bits = buf.len() as u64 * 8;
    if lo >= bits { return Vec::new(); }
    let pieces = threads.clamp(1, 64);
    let span = ((bits - lo) / pieces as u64 / 8 + 1) * 8;
    if pieces == 1 || bits - lo < 1 << 20 { return scan(buf, lo, bits, table); }
    std::thread::scope(|scope| {
        let parts: Vec<_> = (0..pieces as u64).map(|k| {
            let a = lo + k * span;
            let b = (a + span).min(bits);
            scope.spawn(move || if a < b { scan(buf, a, b, table) } else { Vec::new() })
        }).collect();
        parts.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

struct BitWriter { bytes: Vec<u8>, bits: u64 }

impl BitWriter {
    fn push(&mut self, value: u64, count: u32) {
        for k in (0..count).rev() { self.push_bit(((value >> k) & 1) as u8); }
    }
    fn push_bit(&mut self, bit: u8) {
        if self.bits % 8 == 0 { self.bytes.push(0); }
        if bit != 0 { *self.bytes.last_mut().unwrap() |= 0x80 >> (self.bits % 8); }
        self.bits += 1;
    }
    fn append(&mut self, buf: &[u8], start: u64, end: u64) {
        let mut p = start;
        // Bitwise to the next source byte boundary, then whole source bytes shifted in.
        while p < end && p % 8 != 0 { self.push_bit((buf[(p / 8) as usize] >> (7 - p % 8)) & 1); p += 1; }
        let shift = (self.bits % 8) as u32;
        while p + 8 <= end {
            let byte = buf[(p / 8) as usize];
            if shift == 0 { self.bytes.push(byte); } else {
                *self.bytes.last_mut().unwrap() |= byte >> shift;
                self.bytes.push(byte << (8 - shift));
            }
            self.bits += 8;
            p += 8;
        }
        while p < end { self.push_bit((buf[(p / 8) as usize] >> (7 - p % 8)) & 1); p += 1; }
    }
}

/// Decode consecutive block bits as one single-block stream; returns the data and the block CRC.
fn decode_ranges(ranges: &[Range]) -> Result<(Vec<u8>, u32), String> {
    let first = &ranges[0];
    let crc = bits_at(&first.buf, first.start + 48, 32) as u32;
    let total: u64 = ranges.iter().map(|r| r.end - r.start).sum();
    let mut writer = BitWriter { bytes: Vec::with_capacity((total / 8 + 32) as usize), bits: 0 };
    writer.bytes.extend_from_slice(b"BZh9");
    writer.bits = 32;
    for r in ranges { writer.append(&r.buf, r.start, r.end); }
    writer.push(EOS_MAGIC, 48);
    writer.push(crc as u64, 32);
    let mut data = Vec::with_capacity(1 << 20);
    let mut decoder = bzip2::read::BzDecoder::new(writer.bytes.as_slice());
    decoder.read_to_end(&mut data).map_err(|e| e.to_string())?;
    Ok((data, crc))
}

struct Producer {
    file: File,
    options: Options,
    jobs: Sender<Job>,
    results: Sender<(u64, Item)>,
    tokens: Receiver<()>,
    index: u64,
}

enum State { Header(u64), Block(u64) }

impl Producer {
    fn slot(&self) -> Result<(), String> { self.tokens.recv().map_err(|_| "reader closed".to_string()) }

    fn segment(&mut self, buf: &Arc<Vec<u8>>, start: u64, end: u64) -> Result<(), String> {
        self.slot()?;
        let job = Job { index: self.index, range: Range { buf: Arc::clone(buf), start, end } };
        self.index += 1;
        self.jobs.send(job).map_err(|_| "decoder threads stopped".to_string())
    }

    fn stream_end(&mut self, stored: u32) -> Result<(), String> {
        self.slot()?;
        self.results.send((self.index, Item::StreamEnd { stored })).map_err(|_| "reader closed".to_string())?;
        self.index += 1;
        Ok(())
    }

    fn run(&mut self) -> Result<(), String> {
        let table = shift_table();
        let mut buf: Vec<u8> = Vec::new();
        let mut base: u64 = 0; // absolute bit offset of buf[0]
        let mut state = State::Header(0);
        let mut streams = 0u64;
        let mut eof = false;
        let mut carried: Vec<(u64, Kind)> = Vec::new();
        let mut scanned: u64 = 0;
        while !eof {
            let mut chunk = vec![0u8; self.options.chunk.max(64)];
            let mut filled = 0;
            while filled < chunk.len() {
                let n = self.file.read(&mut chunk[filled..]).map_err(|e| e.to_string())?;
                if n == 0 { eof = true; break; }
                filled += n;
            }
            buf.extend_from_slice(&chunk[..filled]);
            drop(chunk);
            let shared = Arc::new(std::mem::take(&mut buf));
            let bits = shared.len() as u64 * 8;
            let from = match state { State::Header(p) | State::Block(p) => p };
            // Only bits not scanned in an earlier round are scanned; earlier hits are carried.
            let lo = from.max(scanned);
            let mut cands: Vec<(u64, Kind)> = carried.drain(..).filter(|c| c.0 >= from && c.0 < lo).collect();
            cands.extend(scan_parallel(&shared, lo, self.options.threads, &table));
            for &(at, eos) in &self.options.fake {
                if at >= base && at - base >= lo && at - base + 48 <= bits {
                    cands.push((at - base, if eos { Kind::Eos } else { Kind::Block }));
                }
            }
            cands.sort_by_key(|c| c.0);
            cands.dedup_by_key(|c| c.0);
            let mut next = 0usize;
            loop {
                match state {
                    State::Header(pos) => {
                        if pos == bits && eof {
                            if streams == 0 { return Err("empty input is not a bzip2 stream".into()); }
                            return Ok(());
                        }
                        // header (32) + a magic (48) + an empty stream's CRC (32)
                        if pos + 32 + 48 + 32 > bits {
                            if eof { return Err("truncated or trailing data after the last bzip2 stream".into()); }
                            break;
                        }
                        let b = (pos / 8) as usize;
                        if &shared[b..b + 3] != b"BZh" || !(b'1'..=b'9').contains(&shared[b + 3]) {
                            return Err("not a bzip2 stream header".into());
                        }
                        streams += 1;
                        let first = pos + 32;
                        match magic_at(&shared, first) {
                            Some(Kind::Block) => state = State::Block(first),
                            Some(Kind::Eos) => {
                                self.stream_end(bits_at(&shared, first + 48, 32) as u32)?;
                                state = State::Header((first + 80).div_ceil(8) * 8);
                            }
                            None => return Err("bzip2 stream header is not followed by a block".into()),
                        }
                    }
                    State::Block(start) => {
                        while next < cands.len() && cands[next].0 < start + 48 { next += 1; }
                        let Some(&(at, kind)) = cands.get(next) else {
                            if eof { return Err("truncated bzip2 stream: a block has no end".into()); }
                            break;
                        };
                        match kind {
                            Kind::Block => {
                                self.segment(&shared, start, at)?;
                                state = State::Block(at);
                            }
                            Kind::Eos => {
                                if !eof && at + EOS_LOOKAHEAD > bits { break; }
                                let after = (at + 80).div_ceil(8) * 8;
                                let real = after <= bits && ((after == bits && eof) || {
                                    let b = (after / 8) as usize;
                                    after + 80 <= bits && &shared[b..b + 3] == b"BZh" && (b'1'..=b'9').contains(&shared[b + 3])
                                        && magic_at(&shared, after + 32).is_some()
                                });
                                if real {
                                    self.segment(&shared, start, at)?;
                                    self.stream_end(bits_at(&shared, at + 48, 32) as u32)?;
                                    state = State::Header(after);
                                } else {
                                    next += 1; // a chance end magic inside block data
                                }
                            }
                        }
                    }
                }
            }
            // Carry the unfinished part (from the open block or header) into the next round.
            let keep = match state { State::Header(p) | State::Block(p) => p / 8 * 8 };
            buf = shared[(keep / 8) as usize..].to_vec();
            carried = cands.into_iter().filter(|c| c.0 >= keep).map(|(p, k)| (p - keep, k)).collect();
            // Every start p with p + 48 <= bits was checked this round.
            scanned = (bits + 1).saturating_sub(48).saturating_sub(keep);
            base += keep;
            state = match state { State::Header(p) => State::Header(p - keep), State::Block(p) => State::Block(p - keep) };
        }
        unreachable!("the loop returns at the end of input")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn sample(len: usize, seed: u64) -> Vec<u8> {
        // Mixed runs and noise: exercises RLE runs and many blocks at level 1.
        let mut x = seed;
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            if x % 23 == 0 { out.extend(std::iter::repeat((x >> 8) as u8).take((x >> 16) as usize % 300)); }
            else { out.extend_from_slice(&x.to_le_bytes()[..(x as usize % 8) + 1]); }
        }
        out.truncate(len);
        out
    }

    fn compress(data: &[u8], level: u32) -> Vec<u8> {
        let mut e = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::new(level));
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    fn reference(stream: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        bzip2::read::MultiBzDecoder::new(stream).read_to_end(&mut out).unwrap();
        out
    }

    fn parallel(stream: &[u8], options: Options) -> io::Result<Vec<u8>> {
        let dir = std::env::temp_dir().join(format!("rw-bz2par-{}-{}", std::process::id(), rand_name()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.bz2");
        std::fs::write(&path, stream).unwrap();
        let mut out = Vec::new();
        let result = open_with(&path, options).and_then(|mut r| r.read_to_end(&mut out));
        std::fs::remove_dir_all(&dir).unwrap();
        result.map(|_| out)
    }

    fn rand_name() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        N.fetch_add(1, Ordering::Relaxed)
    }

    fn opts(threads: usize, chunk: usize) -> Options { Options { threads, chunk, fake: Vec::new() } }

    #[test]
    fn multi_block_multi_stream_output_is_byte_identical() {
        let a = sample(2_500_000, 7);
        let b = sample(700_000, 11);
        let stream = [compress(&a, 1), compress(&[], 9), compress(&b, 3)].concat();
        let expected = reference(&stream);
        assert_eq!(expected.len(), a.len() + b.len());
        for (threads, chunk) in [(1, 64), (3, 4096), (8, 100_003), (24, 16 << 20)] {
            assert_eq!(parallel(&stream, opts(threads, chunk)).unwrap(), expected, "threads {threads} chunk {chunk}");
        }
    }

    #[test]
    fn a_chance_magic_inside_a_block_is_joined_back() {
        let data = sample(1_000_000, 3);
        let stream = compress(&data, 1);
        let table = shift_table();
        let blocks: Vec<u64> = scan(&stream, 0, stream.len() as u64 * 8, &table).into_iter()
            .filter(|c| c.1 == Kind::Block).map(|c| c.0).collect();
        assert!(blocks.len() >= 5, "{blocks:?}");
        let fake = vec![(blocks[1] + 1000, false), (blocks[2] + 77, true), (blocks[3] + 5, false), (blocks[3] + 4000, false)];
        let out = parallel(&stream, Options { threads: 4, chunk: 8192, fake }).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn damaged_and_truncated_streams_are_refused() {
        let data = sample(600_000, 5);
        let stream = compress(&data, 1);
        let mut damaged = stream.clone();
        let middle = damaged.len() / 2;
        damaged[middle] ^= 0x10;
        assert!(parallel(&damaged, opts(4, 4096)).is_err());
        assert!(parallel(&stream[..stream.len() - 3], opts(4, 4096)).is_err());
        assert!(parallel(b"", opts(2, 4096)).is_err());
        assert!(parallel(&[stream.as_slice(), b"junk-after-stream"].concat(), opts(2, 4096)).is_err());
    }
}
