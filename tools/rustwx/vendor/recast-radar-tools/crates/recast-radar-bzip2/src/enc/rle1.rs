//! Initial run-length stage (RLE1) and block boundaries.
//!
//! Input runs of one byte are cut into pieces of at most 255; a piece of
//! length 1 to 3 is stored as is, a longer one as four copies and a count
//! byte (`len - 4`). The block boundaries follow libbzip2 1.0.8's
//! `BZ2_bzBuffToBuffCompress` (one call with `BZ_FINISH`):
//!
//! * a piece is added to the block when the byte after it is read (or at
//!   the end of the input);
//! * before each input byte is read, a block holding at least
//!   `nblockMAX = 100000 * level - 19` bytes is closed, and the piece
//!   in progress starts the next block;
//! * at the end of the input the piece in progress joins the current block,
//!   even when that block is already full (it then holds at most
//!   `nblockMAX + 9` bytes).
//!
//! The block CRC covers the input bytes of the pieces in the block, which
//! are one contiguous input range.

/// One block's worth of input: `block[..nblock]` holds its RLE1 bytes and
/// `in_use` the byte values present; input consumed up to the returned
/// offset.
#[inline(never)]
pub(crate) fn fill_block(
    input: &[u8],
    start: usize,
    nblock_max: usize,
    block: &mut [u8],
    in_use: &mut [bool; 256],
) -> (usize, usize) {
    let len = input.len();
    let mut nb = 0usize;
    let mut i = start;
    in_use.fill(false);
    while i < len {
        // Fast path: eight bytes, each different from the next, are eight
        // one-byte pieces (the block cannot fill during them).
        if i + 9 <= len && nb + 8 < nblock_max {
            let a = word(input, i);
            let x = a ^ word(input, i + 1);
            if x.wrapping_sub(LO) & !x & HI == 0 {
                block[nb..nb + 8].copy_from_slice(&a.to_le_bytes());
                for b in a.to_le_bytes() {
                    in_use[b as usize] = true;
                }
                nb += 8;
                i += 8;
                continue;
            }
        }
        let c = input[i];
        let run_end = run_end(input, i, c);
        let run = run_end - i;
        in_use[c as usize] = true;
        if run < 4 {
            let dst = &mut block[nb..nb + 3];
            dst.fill(c);
            nb += run;
        } else {
            let dst = &mut block[nb..nb + 5];
            dst[..4].fill(c);
            dst[4] = (run - 4) as u8;
            in_use[run - 4] = true;
            nb += 5;
        }
        i = run_end;
        if nb >= nblock_max && i < len {
            // The byte at `i` was read before the check that closes the
            // block. If it is the last input byte, the end-of-input flush
            // adds its one-byte piece to this block.
            if i + 1 == len {
                let c = input[i];
                in_use[c as usize] = true;
                block[nb] = c;
                nb += 1;
                i = len;
            }
            break;
        }
    }
    (nb, i)
}

const LO: u64 = u64::from_le_bytes([1; 8]);
const HI: u64 = u64::from_le_bytes([0x80; 8]);

/// Eight bytes of `s` at `at`, first byte lowest.
#[inline(always)]
fn word(s: &[u8], at: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&s[at..at + 8]);
    u64::from_le_bytes(w)
}

/// End of the piece that starts at `i` with byte `c`: at most 255 bytes.
#[inline(always)]
fn run_end(input: &[u8], i: usize, c: u8) -> usize {
    let len = input.len();
    let lim = len.min(i + 255);
    let mut j = i + 1;
    if j >= lim || input[j] != c {
        return j;
    }
    let pattern = u64::from_ne_bytes([c; 8]);
    while j + 8 <= lim {
        let w = u64::from_le_bytes([
            input[j],
            input[j + 1],
            input[j + 2],
            input[j + 3],
            input[j + 4],
            input[j + 5],
            input[j + 6],
            input[j + 7],
        ]) ^ pattern;
        if w != 0 {
            return j + (w.trailing_zeros() / 8) as usize;
        }
        j += 8;
    }
    while j < lim && input[j] == c {
        j += 1;
    }
    j
}
