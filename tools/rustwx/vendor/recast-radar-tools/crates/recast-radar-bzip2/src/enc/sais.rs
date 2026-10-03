//! SA-IS suffix sorting (Nong, Zhang and Chan, "Two efficient algorithms
//! for linear time suffix array construction", 2011), with a final pass that
//! writes the Burrows-Wheeler last column instead of the suffix array.
//!
//! Suffixes are ordered as if the text ended with a unique sentinel smaller
//! than every symbol. [`bwt`] is used on the least rotation of a block, where
//! that order is also the order of the block's cyclic rotations (see
//! `bwt.rs`).
//!
//! Layout follows the space-saving scheme of Yuta Mori's sais-lite: the
//! suffix array doubles as work space for the reduced string of the
//! recursion. An entry written during an induction pass carries in its sign
//! whether the pass that later scans it must induce its predecessor:
//!
//! * `j > 0`: suffix `j`, whose predecessor `j - 1` is induced by the pass
//!   that scans this entry;
//! * `j <= 0`: nothing (more) to induce from this slot in this pass.
//!
//! Speed on radar data comes from four choices:
//!
//! * stage 1 sorts the LMS substrings directly (counting sort on their first
//!   symbols, then packed integer keys; see [`sorted_lms`]) and names them
//!   in the same pass, instead of inducing over the whole text twice: it
//!   touches only the LMS positions, about a third of the text;
//! * the induction passes are branch-free: every slot is processed, and a
//!   slot with nothing to induce writes to a dummy entry (`sa[n]`, and
//!   `l[n]` for the last column) instead of branching, because on radar
//!   data about half of the slots are skipped in no predictable pattern;
//! * the text is stored after one pad symbol (`tp[0]`, with `tp[x + 1]` the
//!   symbol at `x`), set to the largest or smallest symbol value before each
//!   pass so that the predecessor test of suffix 0 needs no special case;
//! * suffix types are a bitmap computed once per level, from which the LMS
//!   positions are read wherever they are needed.
//!
//! Every index fits in `i32`: blocks are at most 900,000 symbols.

/// Symbol type of a level: bytes at the top, names (`i32`) in the recursion.
pub(crate) trait Sym: Copy + PartialEq + PartialOrd {
    /// Bytes (the top level) rather than names.
    const BYTES: bool;
    /// Larger than or equal to every symbol.
    const HIGH: Self;
    fn idx(self) -> usize;
}

impl Sym for u8 {
    const BYTES: bool = true;
    const HIGH: u8 = u8::MAX;
    #[inline(always)]
    fn idx(self) -> usize {
        self as usize
    }
}

impl Sym for i32 {
    const BYTES: bool = false;
    const HIGH: i32 = i32::MAX;
    #[inline(always)]
    fn idx(self) -> usize {
        self as usize
    }
}

/// Bucket sizes of every symbol.
fn counts<T: Sym>(t: &[T], c: &mut [i32]) {
    c.fill(0);
    for &x in t {
        c[x.idx()] += 1;
    }
}

/// `b[x]` = first slot of bucket `x`.
fn bucket_starts(c: &[i32], b: &mut [i32]) {
    let mut sum = 0;
    for (bi, &ci) in b.iter_mut().zip(c) {
        *bi = sum;
        sum += ci;
    }
}

/// `b[x]` = one past the last slot of bucket `x`.
fn bucket_ends(c: &[i32], b: &mut [i32]) {
    let mut sum = 0;
    for (bi, &ci) in b.iter_mut().zip(c) {
        sum += ci;
        *bi = sum;
    }
}

/// Words of an `n`-bit bitmap.
fn words(n: usize) -> usize {
    n / 64 + 1
}

/// LMS bitmap of `t` into `bits` (bit `i % 64` of word `i / 64` is position
/// `i`); returns the number of LMS positions.
///
/// Position `n - 1` is L-type (the sentinel follows it); `i` is S-type when
/// `t[i] < t[i + 1]`, or when they are equal and `i + 1` is S-type; an LMS
/// position is an S-type position whose left neighbour is L-type.
#[inline(never)]
fn lms_bitmap<T: Sym>(t: &[T], bits: &mut [u64]) -> usize {
    let n = t.len();
    let bits = &mut bits[..words(n)];
    bits.fill(0);
    if n < 2 {
        return 0;
    }
    // S-type bits, right to left (position n - 1 is L-type: bit 0).
    let mut s_next = 0u64;
    let mut next = t[n - 1];
    for (w, word) in bits.iter_mut().enumerate().rev() {
        let lo = w * 64;
        let hi = (lo + 64).min(n - 1);
        let mut acc = 0u64;
        for p in (lo..hi.max(lo)).rev() {
            let cur = t[p];
            let s = u64::from(cur < next) | (u64::from(cur == next) & s_next);
            acc |= s << (p - lo);
            s_next = s;
            next = cur;
        }
        *word = acc;
    }
    // LMS = S-type with an L-type left neighbour; position 0 never is
    // (the carry into bit 0 is 1).
    let mut carry = 1u64;
    let mut m = 0usize;
    for w in bits.iter_mut() {
        let s = *w;
        *w = s & !((s << 1) | carry);
        carry = s >> 63;
        m += w.count_ones() as usize;
    }
    m
}

/// LMS positions in increasing order.
#[inline(always)]
fn for_each_lms(bits: &[u64], mut f: impl FnMut(usize)) {
    for (w, &word) in bits.iter().enumerate() {
        let mut x = word;
        while x != 0 {
            f(w * 64 + x.trailing_zeros() as usize);
            x &= x - 1;
        }
    }
}

/// The predecessor index of an induction slot: `j - 1` for `j > 0`, and 0
/// (a harmless index) otherwise.
#[inline(always)]
fn pred(j: i32) -> usize {
    (j - 1).max(0) as usize
}

/// Solve the reduced problem (when `names < m`) and leave the LMS suffixes,
/// sorted, in `sa[..m]`. Expects the sorted LMS positions in `sa[..m]` and
/// the reduced string (names in text order) at `sa[n + 1 - m..=n]`.
fn finish_lms(
    n: usize,
    sa: &mut [i32],
    lms: &[u64],
    m: usize,
    names: usize,
    pool: &mut [i32],
    bpool: &mut [u64],
) {
    if names >= m {
        return;
    }
    {
        // Recursion: its SA (with dummy) is sa[..m + 1], its padded text
        // sa[n - m..n + 1]; 2m + 1 <= n keeps them apart.
        let (lo, hi) = sa.split_at_mut(n - m);
        suffix_array(&mut hi[..m + 1], &mut lo[..m + 1], names, pool, bpool);
    }
    // LMS positions in text order at sa[n + 1 - m..], then map ranks.
    let mut j = n + 1 - m;
    for_each_lms(lms, |p| {
        sa[j] = p as i32;
        j += 1;
    });
    let (lo, hi) = sa.split_at_mut(n + 1 - m);
    for x in lo[..m].iter_mut() {
        *x = hi[*x as usize];
    }
}

/// Below this many LMS positions, stage 1 on bytes buckets on the first
/// byte (256 counters) instead of the first two (65,536).
pub(crate) const SMALL_LMS: usize = 1 << 12;

#[cfg(test)]
thread_local! {
    /// [`SMALL_LMS`] for this thread, so that the unit tests can put small
    /// texts through either bucketing.
    pub(crate) static SMALL_LMS_FOR_TEST: core::cell::Cell<usize> =
        const { core::cell::Cell::new(SMALL_LMS) };
}

#[inline(always)]
fn small_lms() -> usize {
    #[cfg(test)]
    {
        SMALL_LMS_FOR_TEST.with(core::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        SMALL_LMS
    }
}

/// Bits of an LMS rank in a packed sort item: blocks have at most 900,000
/// bytes, so fewer than 2^19 LMS positions at any level.
const RANK_BITS: u32 = 19;
/// Set in a sorted item that starts a new name.
const NEW: u64 = 1 << 63;

/// How LMS substrings are packed for the direct sort at one level.
#[derive(Clone, Copy)]
struct Keying {
    /// Symbols that pick the bucket (and are not in the packed key).
    first: usize,
    /// Bits per packed symbol.
    width: u32,
    /// Symbols packed after the bucket symbols.
    tail: usize,
    /// Symbol value past the end of a substring (above every symbol).
    end: u64,
}

/// An LMS substring: start, number of symbols, and whether it is the last
/// one (which ends with the sentinel).
#[derive(Clone, Copy)]
struct Sub {
    p: usize,
    real: usize,
    last: bool,
}

impl Sub {
    /// The substring of LMS rank `r` (text order) from the positions in `pos`.
    #[inline(always)]
    fn of(pos: &[i32], r: usize, n: usize) -> Sub {
        let p = pos[r] as usize;
        match pos.get(r + 1) {
            Some(&q) => Sub {
                p,
                real: q as usize - p + 1,
                last: false,
            },
            None => Sub {
                p,
                real: n - p,
                last: true,
            },
        }
    }

    /// Symbol `k` for the direct sort: symbol `x` is `x + 1`, the sentinel
    /// is 0, and past the end every symbol is `end`.
    #[inline(always)]
    fn ext<T: Sym>(self, t: &[T], k: usize, end: u64) -> u64 {
        if k < self.real {
            t[self.p + k].idx() as u64 + 1
        } else if self.last && k == self.real {
            0
        } else {
            end
        }
    }

    /// The packed tail symbols, then one bit set when the substring goes
    /// on past them (no end or sentinel among them).
    #[inline(always)]
    fn key<T: Sym>(self, t: &[T], kg: Keying) -> u64 {
        let mut key = 0u64;
        for k in kg.first..kg.first + kg.tail {
            key = (key << kg.width) | self.ext(t, k, kg.end);
        }
        (key << 1) | u64::from(self.real >= kg.first + kg.tail)
    }

    /// Order of two substrings whose symbols before `from` are equal.
    fn cmp_from<T: Sym>(self, other: Sub, t: &[T], from: usize, end: u64) -> core::cmp::Ordering {
        let mut k = from;
        loop {
            let a = self.ext(t, k, end);
            let b = other.ext(t, k, end);
            if a != b || a == end || a == 0 {
                return a.cmp(&b);
            }
            k += 1;
        }
    }
}

/// Stage 1: the LMS substrings of the text in `tp[1..]` (symbols in
/// `0..k`) sorted directly, instead of by two induction passes over the
/// whole text, and named in the same pass. A counting sort on the first
/// symbols (two bytes, one byte in small texts, or one name) puts packed
/// items (the next few symbols, an "open" bit, the text-order rank) in
/// bucket order; each bucket is then sorted as plain integers, and runs of
/// equal open items (substrings longer than the key) by comparison.
///
/// Substring order: symbols compare by value, the sentinel is below every
/// symbol, and where one substring's symbols are a prefix of another's, the
/// shorter one is larger (at the position where it ends it is S-type and
/// the longer one L-type). This agrees with the order of the LMS suffixes
/// whenever two substrings differ, which is all the recursion needs.
///
/// `cnt` holds a counter per bucket (65,536 for bytes, of which 256 are
/// used in small texts; `k` for names).
/// Leaves the LMS suffixes, sorted, in `sa[..m]`, solving the reduced
/// problem when names repeat.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn sorted_lms<T: Sym>(
    tp: &[T],
    sa: &mut [i32],
    lms: &[u64],
    m: usize,
    k: usize,
    cnt: &mut [i32],
    pool: &mut [i32],
    bpool: &mut [u64],
) {
    if T::BYTES && m >= small_lms() {
        sorted_lms_by::<T, true>(tp, sa, lms, m, k, cnt, pool, bpool);
    } else {
        sorted_lms_by::<T, false>(tp, sa, lms, m, k, cnt, pool, bpool);
    }
}

/// [`sorted_lms`], bucketing bytes on their first two symbols (`PAIRS`) or
/// on the first one; names always bucket on the first. A constant, so the
/// key packing of bytes compiles to fixed shifts either way.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn sorted_lms_by<T: Sym, const PAIRS: bool>(
    tp: &[T],
    sa: &mut [i32],
    lms: &[u64],
    m: usize,
    k: usize,
    cnt: &mut [i32],
    pool: &mut [i32],
    bpool: &mut [u64],
) {
    let n = tp.len() - 1;
    let sa = &mut sa[..n + 1];
    if m <= 1 {
        let mut first = 0usize;
        for_each_lms(lms, |p| first = p);
        sa[..n].fill(0);
        sa[0] = first as i32;
        return;
    }
    let t = &tp[1..];
    // Symbols of the key: 9 bits for bytes, enough for k + 1 for names; at
    // most 43 bits, so rank and flag fit beside them. Bytes bucket on their
    // first two symbols, except in small texts, where filling and scanning
    // 65,536 counters would cost more than the sort they save.
    let kg = if T::BYTES {
        Keying {
            first: if PAIRS { 2 } else { 1 },
            width: 9,
            tail: 4,
            end: 257,
        }
    } else {
        let width = 64 - (k as u64 + 1).leading_zeros();
        Keying {
            first: 1,
            width,
            tail: (43 / width as usize).min(4),
            end: k as u64 + 1,
        }
    };
    // An LMS position is never the last (that one is L-type), so `p + 1`
    // is in the text.
    let bucket = |p: usize| {
        if T::BYTES && PAIRS {
            t[p].idx() << 8 | t[p + 1].idx()
        } else {
            t[p].idx()
        }
    };
    let cnt = if T::BYTES && !PAIRS {
        &mut cnt[..256]
    } else {
        cnt
    };
    let base = n + 1 - m;
    // LMS positions in text order at sa[base..=n].
    {
        let mut j = base;
        for_each_lms(lms, |p| {
            sa[j] = p as i32;
            j += 1;
        });
    }
    let (lo, pos) = sa.split_at_mut(base);
    // The items live at the front of `bpool` only until the names are
    // written; the recursion then reuses that space.
    let items = &mut bpool[..m];
    // Bucket sizes, then the packed items in bucket order.
    cnt.fill(0);
    for &p in pos.iter() {
        cnt[bucket(p as usize)] += 1;
    }
    let mut sum = 0i32;
    for x in cnt.iter_mut() {
        let c = *x;
        *x = sum;
        sum += c;
    }
    for r in 0..m {
        let s = Sub::of(pos, r, n);
        let b = bucket(s.p);
        items[cnt[b] as usize] = s.key(t, kg) << RANK_BITS | r as u64;
        cnt[b] += 1;
    }
    let rank_of = |item: u64| (item & ((1 << RANK_BITS) - 1)) as usize;
    let from = kg.first + kg.tail;
    // Sort each bucket; order runs of equal open keys by comparison; flag
    // every item that starts a new name.
    let mut start = 0usize;
    let mut names = 0usize;
    for &end in cnt.iter() {
        let end = end as usize;
        if end == start {
            continue;
        }
        let bucket = &mut items[start..end];
        bucket.sort_unstable();
        let mut i = 0usize;
        while i < bucket.len() {
            let key = bucket[i] >> RANK_BITS;
            let mut j = i + 1;
            while j < bucket.len() && bucket[j] >> RANK_BITS == key {
                j += 1;
            }
            if key & 1 == 0 || j == i + 1 {
                // Closed keys: equal keys are equal substrings.
                bucket[i] |= NEW;
                names += 1;
            } else {
                // Open keys: the substrings go on past the key.
                let run = &mut bucket[i..j];
                run.sort_unstable_by(|&a, &b| {
                    Sub::of(pos, rank_of(a), n).cmp_from(
                        Sub::of(pos, rank_of(b), n),
                        t,
                        from,
                        kg.end,
                    )
                });
                let mut prev = Sub::of(pos, rank_of(run[0]), n);
                run[0] |= NEW;
                names += 1;
                for x in run[1..].iter_mut() {
                    let s = Sub::of(pos, rank_of(*x), n);
                    if prev.cmp_from(s, t, from, kg.end) != core::cmp::Ordering::Equal {
                        *x |= NEW;
                        names += 1;
                    }
                    prev = s;
                }
            }
            i = j;
        }
        start = end;
    }
    // Positions in sorted order into sa[..m]; names in text order into
    // pos (the reduced string at sa[base..=n]).
    let mut name = -1i32;
    for (x, &item) in lo[..m].iter_mut().zip(items.iter()) {
        name += i32::from(item >= NEW);
        let r = rank_of(item);
        *x = pos[r];
        pos[r] = name;
    }
    finish_lms(n, sa, lms, m, names, pool, bpool);
}

/// Final L pass of [`suffix_array`] (high pad): every scanned entry is
/// complemented, so the ones with work left for the S pass become positive
/// and the rest negative. `b` holds the bucket starts plus a dummy entry.
#[inline(never)]
fn final_l<T: Sym>(tp: &[T], sa: &mut [i32], b: &mut [i32]) {
    let n = tp.len() - 1;
    let sa = &mut sa[..n + 1];
    let dummy = n;
    let bd = b.len() - 1;
    {
        let j = n - 1;
        let cj = tp[j + 1];
        sa[b[cj.idx()] as usize] = (j as i32) ^ -i32::from(tp[j] < cj);
        b[cj.idx()] += 1;
    }
    for i in 0..n {
        let j = sa[i];
        sa[i] = !j;
        let jm = pred(j);
        let c0 = tp[jm + 1];
        let v = (jm as i32) ^ -i32::from(tp[jm] < c0);
        let ci = c0.idx();
        let bc = b[ci];
        let valid = j > 0;
        sa[if valid { bc as usize } else { dummy }] = v;
        b[if valid { ci } else { bd }] = bc + 1;
    }
}

/// Final S pass of [`suffix_array`] (high pad); complemented entries are
/// restored. `b` holds the bucket ends plus a dummy entry.
#[inline(never)]
fn final_s<T: Sym>(tp: &[T], sa: &mut [i32], b: &mut [i32]) {
    let n = tp.len() - 1;
    let sa = &mut sa[..n + 1];
    let dummy = n;
    let bd = b.len() - 1;
    for i in (0..n).rev() {
        let j = sa[i];
        let jm = pred(j);
        let c0 = tp[jm + 1];
        let v = (jm as i32) ^ -i32::from(tp[jm] > c0);
        let ci = c0.idx();
        let valid = j > 0;
        let bc = b[ci] - 1;
        b[if valid { ci } else { bd }] = bc;
        sa[if valid { bc as usize } else { dummy }] = v;
        sa[i] = if valid { j } else { !j };
    }
}

/// Stage 2 set-up: the sorted LMS suffixes in `sa[..m]` moved to the ends of
/// their buckets (keeping their order), every other slot cleared.
#[inline(never)]
fn place_lms<T: Sym>(tp: &[T], sa: &mut [i32], m: usize, c: &[i32], b: &mut [i32]) {
    let n = tp.len() - 1;
    bucket_ends(c, b);
    sa[m..n].fill(0);
    for i in (0..m).rev() {
        let p = sa[i];
        sa[i] = 0;
        let cp = tp[p as usize + 1].idx();
        b[cp] -= 1;
        sa[b[cp] as usize] = p;
    }
}

/// Suffix array of the text in `tp[1..]` (symbols in `0..k`; `tp[0]` is a
/// pad slot) into `sa[..n]`; `sa[n]` is scratch.
fn suffix_array<T: Sym>(
    tp: &mut [T],
    sa: &mut [i32],
    k: usize,
    pool: &mut [i32],
    bpool: &mut [u64],
) {
    let n = tp.len() - 1;
    if n == 0 {
        return;
    }
    let (cb, pool) = pool.split_at_mut(2 * k + 1);
    let (c, b) = cb.split_at_mut(k);
    let (lms, bpool) = bpool.split_at_mut(words(n));
    counts(&tp[1..], c);
    let m = lms_bitmap(&tp[1..], lms);
    if T::BYTES {
        // Only for tests: the encoder's top level is [`bwt`].
        let (cnt, pool) = pool.split_at_mut(1 << 16);
        sorted_lms(tp, sa, lms, m, k, cnt, pool, bpool);
    } else {
        sorted_lms(tp, sa, lms, m, k, b, pool, bpool);
    }
    place_lms(tp, sa, m, c, b);
    // Both passes use the high pad: suffix 0 is stored as 0 when L-type and
    // as !0 when S-type, like every suffix whose predecessor needs no work.
    tp[0] = T::HIGH;
    bucket_starts(c, b);
    final_l(tp, sa, b);
    bucket_ends(c, b);
    final_s(tp, sa, b);
}

/// Scratch `i32`s needed by [`bwt`] for a text of `n` bytes: the top
/// level's two-byte bucket counts, and bucket arrays (sizes, and pointers
/// plus a dummy entry) for every recursion level; each level has less than
/// half the symbols of the one above.
pub(crate) fn pool_len(n: usize) -> usize {
    2 * n + (1 << 16) + 64
}

/// Scratch `u64`s needed by [`bwt`]: the tail keys of the top level's LMS
/// substrings, and an LMS bitmap and its rank table per level.
pub(crate) fn bitmap_pool_len(n: usize) -> usize {
    n / 2 + n / 16 + 256
}

/// Last column of the sorted suffixes of the text in `tp[1..=n]`, where the
/// row of suffix 0 takes the last byte (the cyclic rotation), into
/// `l[..n]`. Returns the row of suffix `target`. `tp[0]` is a pad slot and
/// `tp[n + 1]` scratch; `sa` and `l` need `n + 1` entries, `pool`
/// [`pool_len`]`(n)` and `bpool` [`bitmap_pool_len`]`(n)`.
pub(crate) fn bwt(
    tp: &mut [u8],
    sa: &mut [i32],
    l: &mut [u8],
    target: usize,
    pool: &mut [i32],
    bpool: &mut [u64],
) -> usize {
    let n = tp.len() - 2;
    if n == 0 {
        return 0;
    }
    let mut c = [0i32; 256];
    let mut b = [0i32; 257];
    let (lms, bpool) = bpool.split_at_mut(words(n));
    {
        let text = &mut tp[..n + 1];
        counts(&text[1..], &mut c);
        let m = lms_bitmap(&text[1..], lms);
        let (cnt, pool) = pool.split_at_mut(1 << 16);
        sorted_lms(text, sa, lms, m, 256, cnt, pool, bpool);
        place_lms(text, sa, m, &c, &mut b);
    }
    // Skipped slots read the symbol after the text: the last byte, which is
    // what the row of suffix 0 (the only skipped slot that is a real row)
    // needs.
    tp[n + 1] = tp[n];
    tp[0] = u8::MAX;
    bucket_starts(&c, &mut b);
    let orig = bwt_l(tp, sa, l, &mut b, target);
    tp[0] = 0;
    bucket_ends(&c, &mut b);
    bwt_s(tp, sa, l, &mut b, target, orig)
}

/// `a` where `mask` is all ones, `b` where it is zero: arithmetic, so the
/// compiler keeps the induction loops free of data-dependent branches.
#[inline(always)]
fn sel(mask: usize, a: usize, b: usize) -> usize {
    b ^ ((a ^ b) & mask)
}

/// All ones when `x` holds, zero otherwise.
#[inline(always)]
fn mask(x: bool) -> usize {
    usize::from(x).wrapping_neg()
}

/// Final L pass of [`bwt`] (high pad). Writes the last-column byte of every
/// row it scans: correct for the rows it induces from and for the row of
/// suffix 0, garbage for rows the S pass rewrites. Returns the row of
/// `target` when it is L-type.
#[inline(never)]
fn bwt_l(tp: &[u8], sa: &mut [i32], l: &mut [u8], b: &mut [i32; 257], target: usize) -> usize {
    let n = tp.len() - 2;
    let sa = &mut sa[..n + 1];
    let l = &mut l[..n + 1];
    let dummy = n;
    let mut orig = 0usize;
    {
        let j = n - 1;
        let cj = tp[j + 1];
        let pos = b[cj as usize] as usize;
        b[cj as usize] += 1;
        sa[pos] = (j as i32) ^ -i32::from(tp[j] < cj);
        if j == target {
            orig = pos;
        }
    }
    for i in 0..n {
        let j = sa[i];
        let vm = mask(j > 0);
        let jm = sel(vm, j.wrapping_sub(1) as usize, n);
        let c0 = tp[jm + 1];
        l[i] = c0;
        let v = (jm as i32) ^ -i32::from(tp[jm] < c0);
        let ci = c0 as usize;
        let bc = b[ci];
        let pos = sel(vm, bc as usize, dummy);
        sa[pos] = v;
        b[sel(vm, ci, 256)] = bc + 1;
        sa[i] = (vm as i32) | (!j & !(vm as i32));
        // Skipped slots have jm == n, never the target: this branch is
        // taken once.
        if jm == target {
            orig = pos;
        }
    }
    orig
}

/// Final S pass of [`bwt`] (low pad). An LMS suffix gets its byte when
/// placed. Returns the row of `target`, given its row `orig` from the L
/// pass.
#[inline(never)]
fn bwt_s(
    tp: &[u8],
    sa: &mut [i32],
    l: &mut [u8],
    b: &mut [i32; 257],
    target: usize,
    mut orig: usize,
) -> usize {
    let n = tp.len() - 2;
    let sa = &mut sa[..n + 1];
    let l = &mut l[..n + 1];
    let dummy = n;
    // The pointer of the bucket used last lives in `cur` (bucket `c1`), so
    // slots that induce into the same bucket as the one before chain
    // through a register instead of a store and a load of `b`.
    let mut c1 = 256usize;
    let mut cur = 0usize;
    for i in (0..n).rev() {
        let j = sa[i];
        let vm = mask(j > 0);
        let jm = sel(vm, j.wrapping_sub(1) as usize, n);
        let c0 = tp[jm + 1];
        let prev = tp[jm];
        l[sel(mask(j >= 0), i, dummy)] = c0;
        let lm = mask(prev > c0);
        let v = (jm as i32) | (lm as i32);
        // A valid slot in a new bucket writes the old pointer back and
        // loads the new one; any other slot keeps `c1` and `cur`.
        let ci = sel(vm, c0 as usize, c1);
        let switch = mask(ci != c1);
        b[c1] = cur as i32;
        let loaded = b[sel(switch, ci, 256)] as usize;
        cur = sel(switch, loaded, cur).wrapping_sub(vm & 1);
        c1 = ci;
        let pos = sel(vm, cur, dummy);
        sa[pos] = v;
        l[sel(vm & lm, pos, dummy)] = prev;
        // Skipped slots have jm == n, never the target: this branch is
        // taken once.
        if jm == target {
            orig = pos;
        }
    }
    orig
}

#[cfg(test)]
pub(crate) fn suffix_array_for_test(t: &[u8]) -> Vec<i32> {
    let mut tp = vec![0u8; t.len() + 1];
    tp[1..].copy_from_slice(t);
    let mut sa = vec![0i32; t.len() + 1];
    let mut pool = vec![0i32; pool_len(t.len()) + 2 * 256 + 1];
    let mut bpool = vec![0u64; bitmap_pool_len(t.len())];
    suffix_array(&mut tp, &mut sa, 256, &mut pool, &mut bpool);
    sa.truncate(t.len());
    sa
}
