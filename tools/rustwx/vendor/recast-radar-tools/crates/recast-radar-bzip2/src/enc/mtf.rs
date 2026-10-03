//! Move-to-front and the zero-run code (RUNA/RUNB) over the BWT last column.
//!
//! Symbols: `RUNA = 0`, `RUNB = 1`, `i + 1` for a move-to-front index
//! `i >= 1`, and `EOB = n_in_use + 1` at the end. A run of `z` index-0
//! symbols is written as the bijective base-2 digits of `z`, least
//! significant first (RUNA = 1, RUNB = 2). This is libbzip2's
//! `generateMTFValues`.
//!
//! On radar data the indices are large (about half are 16 or more). The
//! list is 32 words of 8 entries, and a table gives the word that holds
//! each byte value, so a move shifts whole words up to that word (one table
//! update per word) and searches only within it. Every load reads a word
//! that was stored whole, so no load waits on a partial store.

pub(crate) const RUNA: u16 = 0;
pub(crate) const RUNB: u16 = 1;

const LO: u64 = u64::from_le_bytes([1; 8]);
const HI: u64 = u64::from_le_bytes([0x80; 8]);

/// Write the symbols for the last column `l[..n]` into `out`, counting them
/// in `freq`; returns (symbol count, number of byte values in use). `l` is
/// followed by at least 8 readable bytes (their values do not matter).
#[inline(never)]
pub(crate) fn encode(
    l: &[u8],
    n: usize,
    in_use: &[bool; 256],
    out: &mut [u16],
    freq: &mut [u32; 258],
) -> (usize, usize) {
    // The list holds byte values; its initial order is ascending, which is
    // the order of libbzip2's `unseqToSeq` numbering. Entries past the
    // bytes in use are never reached: every byte of `l` is in use.
    let mut bytes = [0u8; 256];
    let mut n_in_use = 0usize;
    for (b, &used) in in_use.iter().enumerate() {
        if used {
            bytes[n_in_use] = b as u8;
            n_in_use += 1;
        }
    }
    let mut list = List::new(&bytes, n_in_use);
    freq.fill(0);
    let eob = n_in_use + 1;
    let mut wr = 0usize;
    let mut zrun = 0u32;
    // Walk the column a run of equal bytes at a time: after a run its byte
    // is at the front of the list, so the next run (a different byte)
    // always moves a byte forward, and a run of r bytes adds r - 1 zeros.
    let mut p = 0usize;
    while p < n {
        let c = l[p];
        // Most runs are one byte long.
        let r = if l[p + 1] != c {
            1
        } else {
            run_len(l, p, n, c)
        };
        p += r;
        if list.front() == c {
            // Only the first run can start with the front byte.
            zrun += r as u32;
            continue;
        }
        if zrun > 0 {
            wr = put_run(out, wr, freq, zrun);
        }
        zrun = r as u32 - 1;
        let idx = list.move_to_front(c);
        out[wr] = (idx + 1) as u16;
        freq[idx + 1] += 1;
        wr += 1;
    }
    if zrun > 0 {
        wr = put_run(out, wr, freq, zrun);
    }
    out[wr] = eob as u16;
    freq[eob] += 1;
    wr += 1;
    (wr, n_in_use)
}

/// The move-to-front list: 32 words of 8 entries (entry `8k + b` in byte
/// `b` of word `k`), and the word that holds each byte value.
struct List {
    words: [u64; 32],
    word_of: [u8; 256],
}

impl List {
    fn new(bytes: &[u8; 256], n: usize) -> List {
        let mut words = [0u64; 32];
        for (w, chunk) in words.iter_mut().zip(bytes.chunks_exact(8)) {
            let mut a = [0u8; 8];
            a.copy_from_slice(chunk);
            *w = u64::from_le_bytes(a);
        }
        let mut word_of = [0u8; 256];
        for (e, &b) in bytes[..n].iter().enumerate() {
            word_of[b as usize] = (e / 8) as u8;
        }
        List { words, word_of }
    }

    /// Move `c` (in the list) to the front and return its index. The words
    /// before the one holding `c` move up one byte each, taking the top byte
    /// of the word below (whose byte then belongs to the next word); in the
    /// word of `c` only the bytes below it move.
    #[inline(always)]
    fn move_to_front(&mut self, c: u8) -> usize {
        let kc = usize::from(self.word_of[usize::from(c)]) & 31;
        let mut carry = u64::from(c);
        for k in 0..kc {
            let w = self.words[k];
            self.words[k] = (w << 8) | carry;
            carry = w >> 56;
            self.word_of[carry as usize] = (k + 1) as u8;
        }
        let w = self.words[kc];
        // High bit set in the bytes equal to c (exact for the lowest one).
        let x = w ^ (u64::from(c) * LO);
        let t = x.wrapping_sub(LO) & !x & HI;
        let b = (t.trailing_zeros() >> 3) & 7;
        let keep = u64::MAX >> (56 - 8 * b);
        self.words[kc] = ((w << 8) & keep) | (w & !keep) | carry;
        self.word_of[usize::from(c)] = 0;
        8 * kc + b as usize
    }

    fn front(&self) -> u8 {
        self.words[0] as u8
    }
}

/// Eight bytes of `s` at `at`, first byte lowest.
#[inline(always)]
fn word(s: &[u8], at: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&s[at..at + 8]);
    u64::from_le_bytes(w)
}

/// Length of the run of `c` starting at `l[p]` (at least 1, at most
/// `n - p`).
#[inline(always)]
fn run_len(l: &[u8], p: usize, n: usize, c: u8) -> usize {
    let pattern = u64::from(c) * LO;
    let mut q = p;
    loop {
        let x = word(l, q) ^ pattern;
        if x != 0 {
            return (q + (x.trailing_zeros() / 8) as usize).min(n) - p;
        }
        q += 8;
        if q >= n {
            return n - p;
        }
    }
}

/// RUNA/RUNB digits of a run of `z >= 1` zeros.
#[inline]
fn put_run(out: &mut [u16], mut wr: usize, freq: &mut [u32; 258], z: u32) -> usize {
    let mut z = z - 1;
    loop {
        let sym = if z & 1 == 0 { RUNA } else { RUNB };
        out[wr] = sym;
        freq[sym as usize] += 1;
        wr += 1;
        if z < 2 {
            break;
        }
        z = (z - 2) / 2;
    }
    wr
}
