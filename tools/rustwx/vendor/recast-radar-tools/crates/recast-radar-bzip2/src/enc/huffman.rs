//! Huffman coding of one block's symbols, bit for bit as libbzip2 1.0.8's
//! `sendMTFValues` and `BZ2_hbMakeCodeLengths` do it: the number of tables
//! from the symbol count, initial tables from a frequency partition, four
//! refinement passes (each 50-symbol group picks its cheapest table, then
//! every table is rebuilt from the groups that picked it), code lengths
//! limited to 17 bits by halving the weights, canonical codes, and
//! move-to-front coded selectors.

use super::bits::BitWriter;

pub(crate) const GROUP: usize = 50;
const N_ITERS: usize = 4;
const MAX_LEN: u32 = 17;
const MAX_ALPHA: usize = 258;
const LESSER_ICOST: u8 = 0;
const GREATER_ICOST: u8 = 15;
/// Bits per table in a packed cost word: a group costs at most
/// 50 * 17 = 850 < 1024.
const COST_BITS: u32 = 10;
const COST_MASK: u64 = (1 << COST_BITS) - 1;

/// Symbol-indexed tables are this wide (a power of two above the 258
/// symbols), so a masked symbol needs no bounds check.
const WIDE: usize = 512;
const WIDE_MASK: usize = WIDE - 1;

/// Tables and selectors of one block (reused between blocks).
pub(crate) struct Tables {
    len: [[u8; MAX_ALPHA]; 6],
    rfreq: [[u32; WIDE]; 6],
    /// `code << 5 | len` per table and symbol.
    code: [[u32; WIDE]; 6],
    pack: [u64; WIDE],
    pub(crate) selectors: Vec<u8>,
    heap: HeapScratch,
}

impl Tables {
    pub(crate) fn new() -> Box<Tables> {
        Box::new(Tables {
            len: [[0; MAX_ALPHA]; 6],
            rfreq: [[0; WIDE]; 6],
            code: [[0; WIDE]; 6],
            pack: [0; WIDE],
            selectors: Vec::new(),
            heap: HeapScratch {
                heap: [0; HEAP_LEN],
                weight: [0; NODES_LEN],
                parent: [0; NODES_LEN],
            },
        })
    }
}

/// Work arrays of [`make_code_lengths`], kept with the tables instead of
/// zeroed on the stack for each of its (up to 24) calls per block, which
/// was a fixed cost of about 6 KB of stores per call. Every entry a call
/// reads, it has written earlier in the same call. The lengths are powers of
/// two so that indices masked with [`hp`] and [`nd`] need no bounds checks.
struct HeapScratch {
    /// The binary heap of node numbers (1-based; `heap[0]` is a sentinel).
    heap: [usize; HEAP_LEN],
    /// Node weights: frequency above the low byte, subtree depth in it.
    weight: [u32; NODES_LEN],
    /// Parent of each node, then (after the tree is built) its depth.
    parent: [i32; NODES_LEN],
}

/// Heap slots: more than the `MAX_ALPHA + 1` a heap can use.
const HEAP_LEN: usize = 512;
/// Tree nodes: more than the `2 * MAX_ALPHA` a tree can have.
const NODES_LEN: usize = 1024;

/// A heap slot index, masked into range (it always is).
#[inline(always)]
fn hp(i: usize) -> usize {
    i & (HEAP_LEN - 1)
}

/// A node index, masked into range (it always is).
#[inline(always)]
fn nd(i: usize) -> usize {
    i & (NODES_LEN - 1)
}

/// Costs of one group under every table, packed 10 bits per table.
#[inline(always)]
fn group_cost(pack: &[u64; WIDE], group: &[u16]) -> u64 {
    let mut cost = 0u64;
    for &s in group {
        cost += pack[usize::from(s) & WIDE_MASK];
    }
    cost
}

/// Count a group's symbols into `rf`.
#[inline(always)]
fn count_group(rf: &mut [u32; WIDE], group: &[u16]) {
    for &s in group {
        rf[usize::from(s) & WIDE_MASK] += 1;
    }
}

/// The first table with the strictly smallest cost.
#[inline(always)]
fn cheapest(cost: u64, n_groups: usize) -> usize {
    let mut bt = 0usize;
    let mut bc = cost & COST_MASK;
    for t in 1..n_groups {
        let c = (cost >> (COST_BITS * t as u32)) & COST_MASK;
        let better = c < bc;
        bc = if better { c } else { bc };
        bt = if better { t } else { bt };
    }
    bt
}

/// Write the symbol map, tables, selectors and coded symbols of a block.
#[inline(never)]
pub(crate) fn send(
    tb: &mut Tables,
    mtfv: &[u16],
    freq: &[u32; 258],
    in_use: &[bool; 256],
    n_in_use: usize,
    bw: &mut BitWriter<'_>,
) {
    let n_mtf = mtfv.len();
    let alpha = n_in_use + 2;
    let n_groups = match n_mtf {
        0..200 => 2,
        200..600 => 3,
        600..1200 => 4,
        1200..2400 => 5,
        _ => 6,
    };

    // Initial tables: split the symbol range into n_groups bands of about
    // equal frequency; a table costs 0 inside its band and 15 outside.
    for t in tb.len.iter_mut() {
        t[..alpha].fill(GREATER_ICOST);
    }
    {
        let mut n_part = n_groups;
        let mut rem_f = n_mtf as i64;
        let mut gs: i64 = 0;
        while n_part > 0 {
            let t_freq = rem_f / n_part as i64;
            let mut ge = gs - 1;
            let mut a_freq: i64 = 0;
            while a_freq < t_freq && ge < alpha as i64 - 1 {
                ge += 1;
                a_freq += i64::from(freq[ge as usize]);
            }
            if ge > gs && n_part != n_groups && n_part != 1 && (n_groups - n_part) % 2 == 1 {
                a_freq -= i64::from(freq[ge as usize]);
                ge -= 1;
            }
            let row = &mut tb.len[n_part - 1];
            for (v, l) in row[..alpha].iter_mut().enumerate() {
                let v = v as i64;
                *l = if v >= gs && v <= ge {
                    LESSER_ICOST
                } else {
                    GREATER_ICOST
                };
            }
            n_part -= 1;
            gs = ge + 1;
            rem_f -= a_freq;
        }
    }

    let n_selectors = n_mtf.div_ceil(GROUP);
    tb.selectors.clear();
    tb.selectors.resize(n_selectors, 0);
    for _ in 0..N_ITERS {
        for t in 0..n_groups {
            tb.rfreq[t][..alpha].fill(0);
        }
        // Costs of all tables for one symbol, packed 10 bits per table.
        for v in 0..alpha {
            let mut p = 0u64;
            for t in 0..n_groups {
                p |= u64::from(tb.len[t][v]) << (COST_BITS * t as u32);
            }
            tb.pack[v] = p;
        }
        // Whole groups as fixed-size chunks (the loops unroll), then the
        // short last group.
        let whole = n_mtf / GROUP * GROUP;
        for (g, group) in mtfv[..whole].chunks_exact(GROUP).enumerate() {
            let bt = cheapest(group_cost(&tb.pack, group), n_groups);
            tb.selectors[g] = bt as u8;
            count_group(&mut tb.rfreq[bt], group);
        }
        if whole < n_mtf {
            let group = &mtfv[whole..];
            let bt = cheapest(group_cost(&tb.pack, group), n_groups);
            tb.selectors[n_selectors - 1] = bt as u8;
            count_group(&mut tb.rfreq[bt], group);
        }
        for t in 0..n_groups {
            make_code_lengths(
                &mut tb.heap,
                &mut tb.len[t],
                &tb.rfreq[t][..alpha],
                alpha,
                MAX_LEN,
            );
        }
    }

    // Canonical codes, shortest first, ties by symbol.
    for t in 0..n_groups {
        let len = &tb.len[t][..alpha];
        let min = len.iter().copied().min().unwrap_or(1);
        let max = len.iter().copied().max().unwrap_or(1);
        let mut next = 0u32;
        for n in min..=max {
            for (i, &l) in len.iter().enumerate() {
                if l == n {
                    tb.code[t][i] = (next << 5) | u32::from(l);
                    next += 1;
                }
            }
            next <<= 1;
        }
    }

    // Symbol map: 16 bits of used 16-byte ranges, then 16 bits per range.
    let mut used16 = 0u32;
    for (r, chunk) in in_use.chunks(16).enumerate() {
        if chunk.iter().any(|&u| u) {
            used16 |= 0x8000 >> r;
        }
    }
    bw.put(16, used16);
    for chunk in in_use.chunks(16) {
        if chunk.iter().any(|&u| u) {
            let mut bits = 0u32;
            for (j, &u) in chunk.iter().enumerate() {
                if u {
                    bits |= 0x8000 >> j;
                }
            }
            bw.put(16, bits);
        }
    }

    // Selectors, move-to-front coded, in unary.
    bw.put(3, n_groups as u32);
    bw.put(15, n_selectors as u32);
    let mut pos = [0u8, 1, 2, 3, 4, 5];
    for &s in &tb.selectors {
        let mut j = 0usize;
        while pos[j] != s {
            j += 1;
        }
        pos.copy_within(..j, 1);
        pos[0] = s;
        // j ones then a zero (j <= 5).
        bw.put(j as u32 + 1, ((1u32 << j) - 1) << 1);
    }

    // Code lengths, delta coded.
    for t in 0..n_groups {
        let len = &tb.len[t];
        let mut curr = u32::from(len[0]);
        bw.put(5, curr);
        for &l in &len[..alpha] {
            let l = u32::from(l);
            while curr < l {
                bw.put(2, 2);
                curr += 1;
            }
            while curr > l {
                bw.put(2, 3);
                curr -= 1;
            }
            bw.put(1, 0);
        }
    }

    // The symbols.
    for (g, group) in mtfv.chunks(GROUP).enumerate() {
        let code = &tb.code[usize::from(tb.selectors[g]) % 6];
        for &s in group {
            let e = code[usize::from(s) & WIDE_MASK];
            bw.put(e & 31, e >> 5);
        }
    }
}

/// libbzip2's `BZ2_hbMakeCodeLengths`: Huffman code lengths from a binary
/// heap whose weights carry the subtree depth in their low byte (so equal
/// frequencies merge the shallower tree first), with zero frequencies
/// counted as 1; while a length exceeds `max_len`, every frequency `f`
/// becomes `1 + f / 2` and the code is rebuilt.
#[inline(never)]
fn make_code_lengths(
    scratch: &mut HeapScratch,
    len: &mut [u8],
    freq: &[u32],
    alpha: usize,
    max_len: u32,
) {
    let HeapScratch {
        heap,
        weight,
        parent,
    } = scratch;

    for i in 0..alpha {
        weight[nd(i + 1)] = freq[i].max(1) << 8;
    }
    loop {
        let mut n_nodes = alpha;
        let mut n_heap = 0usize;
        heap[0] = 0;
        weight[0] = 0;
        parent[0] = -2;
        parent[1..=alpha].fill(-1);
        for i in 1..=alpha {
            n_heap += 1;
            heap[hp(n_heap)] = i;
            up_heap(heap, weight, n_heap);
        }
        while n_heap > 1 {
            let n1 = heap[1];
            heap[1] = heap[hp(n_heap)];
            n_heap -= 1;
            down_heap(heap, weight, n_heap, 1);
            let n2 = heap[1];
            heap[1] = heap[hp(n_heap)];
            n_heap -= 1;
            down_heap(heap, weight, n_heap, 1);
            n_nodes += 1;
            parent[nd(n1)] = n_nodes as i32;
            parent[nd(n2)] = n_nodes as i32;
            let (w1, w2) = (weight[nd(n1)], weight[nd(n2)]);
            weight[nd(n_nodes)] = ((w1 & !0xff) + (w2 & !0xff)) | (1 + (w1 & 0xff).max(w2 & 0xff));
            parent[nd(n_nodes)] = -1;
            n_heap += 1;
            heap[hp(n_heap)] = n_nodes;
            up_heap(heap, weight, n_heap);
        }
        // Depths from the root down, in place of the parents: every node
        // was created before its parent, so going down from the root (the
        // last node), a node's parent already holds its depth. libbzip2
        // walks up from each leaf instead; the lengths are the same.
        parent[nd(n_nodes)] = 0;
        for k in (1..n_nodes).rev() {
            parent[nd(k)] = parent[nd(parent[nd(k)] as usize)] + 1;
        }
        let mut too_long = false;
        for (i, l) in len[..alpha].iter_mut().enumerate() {
            let j = parent[nd(i + 1)] as u32;
            *l = j as u8;
            too_long |= j > max_len;
        }
        if !too_long {
            break;
        }
        for w in weight[1..=alpha].iter_mut() {
            *w = (1 + (*w >> 8) / 2) << 8;
        }
    }
}

#[inline]
fn up_heap(heap: &mut [usize; HEAP_LEN], weight: &[u32; NODES_LEN], mut z: usize) {
    let tmp = heap[hp(z)];
    while weight[nd(tmp)] < weight[nd(heap[hp(z >> 1)])] {
        heap[hp(z)] = heap[hp(z >> 1)];
        z >>= 1;
    }
    heap[hp(z)] = tmp;
}

#[inline]
fn down_heap(heap: &mut [usize; HEAP_LEN], weight: &[u32; NODES_LEN], n_heap: usize, mut z: usize) {
    let tmp = heap[hp(z)];
    loop {
        let mut yy = z << 1;
        if yy > n_heap {
            break;
        }
        if yy < n_heap && weight[nd(heap[hp(yy + 1)])] < weight[nd(heap[hp(yy)])] {
            yy += 1;
        }
        if weight[nd(tmp)] < weight[nd(heap[hp(yy)])] {
            break;
        }
        heap[hp(z)] = heap[hp(yy)];
        z = yy;
    }
    heap[hp(z)] = tmp;
}
