//! Burrows-Wheeler transform of one block: sorted cyclic rotations.
//!
//! bzip2 sorts the rotations of a block, not its suffixes. Rotating the
//! block does not change its set of rotations, so the block is first
//! rotated to a text `t` whose suffix order (with a sentinel smaller than
//! every byte) is also the order of its rotations, suffix-sorted by SA-IS in
//! linear time, and `origPtr` is the row of the rotation that starts where
//! the original block does. The two orders can only disagree for a suffix
//! that is a proper prefix of a longer one, and two rotations of `t` qualify:
//!
//! * `t` ends with a byte that occurs nowhere else in the block: then no
//!   suffix is a prefix of another (the shorter one ends with that byte,
//!   and the longer one has a different byte at the same offset);
//! * otherwise `t` is the least rotation (a power of a Lyndon word): if
//!   suffix `j > i` is a prefix of suffix `i`, the rotation at `j`
//!   continues with the whole of `t` while the rotation at `i` continues
//!   with some rotation of `t`, which is never smaller.
//!
//! The sorted order of distinct rotations is unique, so the last column
//! and `origPtr` equal libbzip2's for every block whose rotations are all
//! distinct. A block that is an exact repetition of a shorter string has
//! equal rotations; their rows are identical, so any of them is a valid
//! `origPtr`, and this one can differ from the row libbzip2 picks.

use super::sais;

/// Start of the least rotation of `s = ss[..n]`, given `ss = s ++ s`, and
/// whether `s` is an exact repetition of a shorter string.
///
/// Two-candidate scan (the "minimum expression" algorithm): `i` and `j`
/// are candidate starts; when their rotations first differ at offset `k`,
/// no start in `i..=i + k` (or `j..=j + k`, for the larger one) can be
/// least. Only starts of runs of the smallest byte value can be least (a
/// rotation that starts inside such a run, or with a larger byte, is
/// beaten by the one at the start of the run), so candidates skip to the
/// next run start: the scan stays linear and takes far fewer steps than
/// one per byte. Least starts are never ruled out, so a periodic `s` (two
/// or more least starts) ends with two candidates whose rotations are
/// equal, and an aperiodic one with a candidate past `n`.
#[inline(never)]
pub(crate) fn least_rotation(ss: &[u8], n: usize) -> (usize, bool) {
    let ss = &ss[..2 * n];
    let Some(&m) = ss[..n].iter().min() else {
        return (0, false);
    };
    let mut i = next_start(ss, n, m, 0);
    if i >= n {
        // Every byte is m.
        return (0, n > 1);
    }
    let mut j = next_start(ss, n, m, i + 1);
    while i < n && j < n {
        let x = &ss[i..i + n];
        let y = &ss[j..j + n];
        let k = common_prefix(x, y);
        if k == n {
            return (i.min(j), true);
        }
        if x[k] > y[k] {
            i = next_start(ss, n, m, i + k + 1);
        } else {
            j = next_start(ss, n, m, j + k + 1);
        }
        if i == j {
            j = next_start(ss, n, m, j + 1);
        }
    }
    (i.min(j), false)
}

/// First start of a run of `m` (a position holding `m` whose cyclic
/// predecessor does not) at or after `from`, or `n` when there is none.
fn next_start(ss: &[u8], n: usize, m: u8, from: usize) -> usize {
    let pattern = u64::from(m) * 0x0101_0101_0101_0101;
    let mut p = from;
    while p < n {
        // Next m, a word at a time.
        if p + 8 <= n {
            let x = word(ss, p) ^ pattern;
            let z = x.wrapping_sub(0x0101_0101_0101_0101) & !x & 0x8080_8080_8080_8080;
            if z == 0 {
                p += 8;
                continue;
            }
            p += (z.trailing_zeros() / 8) as usize;
        } else if ss[p] != m {
            p += 1;
            continue;
        }
        let prev = if p == 0 { ss[n - 1] } else { ss[p - 1] };
        if prev != m {
            return p;
        }
        // Inside a run: skip to its end.
        while p < n && ss[p] == m {
            p += 1;
        }
    }
    n
}

/// Length of the common prefix of `x` and `y` (equal lengths), a word at a
/// time.
fn common_prefix(x: &[u8], y: &[u8]) -> usize {
    let n = x.len();
    let mut k = 0usize;
    while k + 8 <= n {
        let d = word(x, k) ^ word(y, k);
        if d != 0 {
            return k + (d.trailing_zeros() / 8) as usize;
        }
        k += 8;
    }
    while k < n && x[k] == y[k] {
        k += 1;
    }
    k
}

/// Eight bytes of `s` at `at`, first byte lowest.
#[inline(always)]
fn word(s: &[u8], at: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&s[at..at + 8]);
    u64::from_le_bytes(w)
}

/// The BWT of one block.
pub(crate) struct Bwt {
    /// Row of the rotation that starts where the block does.
    pub orig_ptr: u32,
    /// The block is an exact repetition of a shorter string.
    pub periodic: bool,
}

/// BWT of the block in `block[1..=n]`, whose byte histogram is `counts`,
/// using `block[n + 1..=2n + 1]` as scratch for the rotated text and the
/// byte before it as the SA-IS pad slot. Writes the last column to
/// `l[..n]`; `sa` and `l` need `n + 1` entries.
pub(crate) fn bwt(
    block: &mut [u8],
    n: usize,
    counts: &[u32; 256],
    sa: &mut [i32],
    l: &mut [u8],
    pool: &mut [i32],
    bpool: &mut [u64],
) -> Bwt {
    if n == 0 {
        return Bwt {
            orig_ptr: 0,
            periodic: false,
        };
    }
    block.copy_within(1..=n, n + 1);
    // Rotate so that a byte that occurs once ends the text, or else to the
    // least rotation. k is the start of the rotation in the block.
    let unique = counts.iter().position(|&c| c == 1);
    let (k, periodic) = match unique {
        Some(byte) => {
            let at = block[1..=n]
                .iter()
                .position(|&x| x == byte as u8)
                .unwrap_or(0);
            ((at + 1) % n, false)
        }
        None => least_rotation(&block[1..], n),
    };
    let target = if k == 0 { 0 } else { n - k };
    let orig = sais::bwt(&mut block[k..=k + n + 1], sa, l, target, pool, bpool);
    Bwt {
        orig_ptr: orig as u32,
        periodic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: sort the rotations by comparing them in full; the row of
    /// rotation 0 is the first row equal to it.
    fn naive(s: &[u8]) -> (Vec<u8>, usize) {
        let n = s.len();
        let rot = |i: usize| s[i..].iter().chain(&s[..i]).copied().collect::<Vec<u8>>();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by_key(|&i| rot(i));
        let l = idx.iter().map(|&i| s[(i + n - 1) % n]).collect();
        let r0 = rot(0);
        let orig = idx.iter().position(|&i| rot(i) == r0).unwrap_or(0);
        (l, orig)
    }

    /// [`check_with`] with stage 1 bucketing on two bytes and on one.
    fn check(s: &[u8]) {
        for small_lms in [0, usize::MAX] {
            sais::SMALL_LMS_FOR_TEST.with(|x| x.set(small_lms));
            check_with(s);
        }
        sais::SMALL_LMS_FOR_TEST.with(|x| x.set(sais::SMALL_LMS));
    }

    fn check_with(s: &[u8]) {
        let n = s.len();
        let mut block = vec![0u8; 2 * n + 2];
        block[1..=n].copy_from_slice(s);
        let mut counts = [0u32; 256];
        for &x in s {
            counts[x as usize] += 1;
        }
        let mut sa = vec![0i32; n + 1];
        let mut l = vec![0u8; n + 1];
        let mut pool = vec![0i32; sais::pool_len(n)];
        let mut bpool = vec![0u64; sais::bitmap_pool_len(n)];
        let r = bwt(
            &mut block, n, &counts, &mut sa, &mut l, &mut pool, &mut bpool,
        );
        l.truncate(n);
        let orig = r.orig_ptr as usize;
        let (want_l, want_orig) = naive(s);
        assert_eq!(l, want_l, "last column of {s:?}");
        let periodic = (1..n).any(|p| n.is_multiple_of(p) && (0..n).all(|i| s[i] == s[i % p]));
        assert_eq!(r.periodic, periodic, "periodicity of {s:?}");
        // Equal rotations (a periodic block) have identical rows, so any
        // of them is a valid origPtr: check that it inverts.
        if want_orig != orig {
            assert_eq!(invert(&l, orig), s, "origPtr {orig} of {s:?}");
        }
    }

    /// Inverse BWT (the decoder's LF walk).
    fn invert(l: &[u8], orig: usize) -> Vec<u8> {
        let n = l.len();
        let mut count = [0usize; 256];
        for &c in l {
            count[c as usize] += 1;
        }
        let mut start = [0usize; 256];
        let mut sum = 0;
        for c in 0..256 {
            start[c] = sum;
            sum += count[c];
        }
        let mut tt = vec![0usize; n];
        for (i, &c) in l.iter().enumerate() {
            tt[start[c as usize]] = i;
            start[c as usize] += 1;
        }
        let mut out = Vec::with_capacity(n);
        let mut p = tt[orig];
        for _ in 0..n {
            out.push(l[p]);
            p = tt[p];
        }
        out
    }

    #[test]
    fn small_strings_match_the_naive_sort() {
        // Every string over {a, b, c} up to length 7, and over {a, b} up to 12.
        for (alpha, max) in [(3u32, 7usize), (2, 12)] {
            for len in 1..=max {
                let total = alpha.pow(len as u32);
                for code in 0..total {
                    let mut x = code;
                    let s: Vec<u8> = (0..len)
                        .map(|_| {
                            let d = (x % alpha) as u8;
                            x /= alpha;
                            b'a' + d
                        })
                        .collect();
                    check(&s);
                }
            }
        }
    }

    #[test]
    fn least_rotation_is_least() {
        // Every string over {a, b, c} up to length 8 and over {a, b} up to
        // 14: the start is a least rotation, and the periodic flag is exact.
        for (alpha, max) in [(3u32, 8usize), (2, 14)] {
            for len in 1..=max {
                for code in 0..alpha.pow(len as u32) {
                    let mut x = code;
                    let s: Vec<u8> = (0..len)
                        .map(|_| {
                            let d = (x % alpha) as u8;
                            x /= alpha;
                            b'a' + d
                        })
                        .collect();
                    let n = s.len();
                    let ss: Vec<u8> = s.iter().chain(&s).copied().collect();
                    let (k, periodic) = least_rotation(&ss, n);
                    let best = (0..n).map(|i| &ss[i..i + n]).min();
                    assert_eq!(Some(&ss[k..k + n]), best, "{s:?}");
                    let want =
                        (1..n).any(|p| n.is_multiple_of(p) && (0..n).all(|i| s[i] == s[i % p]));
                    assert_eq!(periodic, want, "periodicity of {s:?}");
                }
            }
        }
    }

    #[test]
    fn sais_suffix_array_matches_sort() {
        for s in [
            &b"mississippi"[..],
            b"abracadabra",
            b"aaaaaaaa",
            b"abcabcabc",
            b"cbacbacba",
        ] {
            let mut want: Vec<i32> = (0..s.len() as i32).collect();
            want.sort_by_key(|&i| &s[i as usize..]);
            for small_lms in [0, usize::MAX] {
                sais::SMALL_LMS_FOR_TEST.with(|x| x.set(small_lms));
                let sa = sais::suffix_array_for_test(s);
                assert_eq!(sa, want, "{s:?}");
            }
            sais::SMALL_LMS_FOR_TEST.with(|x| x.set(sais::SMALL_LMS));
        }
    }
}
