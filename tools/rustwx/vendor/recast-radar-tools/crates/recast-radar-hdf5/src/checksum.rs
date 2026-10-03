//! HDF5 checksums: Jenkins lookup3 for 1.8+ metadata, Fletcher-32 for the
//! `fletcher32` data filter.

/// Bob Jenkins' lookup3 `hashlittle` over little-endian words with an
/// initial value of 0: `H5_checksum_metadata`, the checksum of superblocks
/// v2/v3, v2 object headers, fractal heaps, v2 B-trees, fixed and extensible
/// arrays.
pub(crate) fn lookup3(data: &[u8]) -> u32 {
    let init = 0xdead_beef_u32.wrapping_add(data.len() as u32);
    let (mut a, mut b, mut c) = (init, init, init);
    let word = |block: &[u8; 12], at: usize| {
        u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]])
    };
    let mut rest = data;
    // A final block of exactly 12 bytes goes through the tail path below.
    while rest.len() > 12
        && let Some((block, tail)) = rest.split_first_chunk::<12>()
    {
        a = a.wrapping_add(word(block, 0));
        b = b.wrapping_add(word(block, 4));
        c = c.wrapping_add(word(block, 8));
        // mix(a, b, c)
        a = a.wrapping_sub(c) ^ c.rotate_left(4);
        c = c.wrapping_add(b);
        b = b.wrapping_sub(a) ^ a.rotate_left(6);
        a = a.wrapping_add(c);
        c = c.wrapping_sub(b) ^ b.rotate_left(8);
        b = b.wrapping_add(a);
        a = a.wrapping_sub(c) ^ c.rotate_left(16);
        c = c.wrapping_add(b);
        b = b.wrapping_sub(a) ^ a.rotate_left(19);
        a = a.wrapping_add(c);
        c = c.wrapping_sub(b) ^ b.rotate_left(4);
        b = b.wrapping_add(a);
        rest = tail;
    }
    if rest.is_empty() {
        // hashlittle: a zero-length tail skips the final mix entirely.
        return c;
    }
    // The 1..=12 byte tail reads as three zero-padded words (the C switch
    // adds only the bytes present, which is the same thing).
    let mut tail = [0u8; 12];
    for (slot, byte) in tail.iter_mut().zip(rest) {
        *slot = *byte;
    }
    a = a.wrapping_add(word(&tail, 0));
    b = b.wrapping_add(word(&tail, 4));
    c = c.wrapping_add(word(&tail, 8));
    // final(a, b, c)
    c = (c ^ b).wrapping_sub(b.rotate_left(14));
    a = (a ^ c).wrapping_sub(c.rotate_left(11));
    b = (b ^ a).wrapping_sub(a.rotate_left(25));
    c = (c ^ b).wrapping_sub(b.rotate_left(16));
    a = (a ^ c).wrapping_sub(c.rotate_left(4));
    b = (b ^ a).wrapping_sub(a.rotate_left(14));
    c = (c ^ b).wrapping_sub(b.rotate_left(24));
    c
}

/// `H5_checksum_fletcher32`: Fletcher-32 over big-endian 16-bit words, an
/// odd trailing byte taken as the high byte of a final word.
pub(crate) fn fletcher32(data: &[u8]) -> u32 {
    let (mut sum1, mut sum2) = (0u32, 0u32);
    let (words, tail) = data.as_chunks::<2>();
    // 360 is the most sums that cannot overflow 32 bits between reductions.
    for block in words.chunks(360) {
        for pair in block {
            sum1 = sum1.wrapping_add((u32::from(pair[0]) << 8) | u32::from(pair[1]));
            sum2 = sum2.wrapping_add(sum1);
        }
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }
    if let Some(byte) = tail.first() {
        sum1 = sum1.wrapping_add(u32::from(*byte) << 8);
        sum2 = sum2.wrapping_add(sum1);
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }
    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    (sum2 << 16) | sum1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Jenkins lookup3 (hashlittle) known-answer vectors. The 30-byte phrase
    /// with init 0 is the published lookup3 self-test value; the shorter
    /// vectors pin every tail-length branch class (empty, <4, exactly 12 =
    /// one full block, 13 = block + 1-byte tail). Real object-header,
    /// fractal-heap and B-tree checksums are checked by every real-file test.
    #[test]
    fn lookup3_matches_reference_vectors() {
        assert_eq!(lookup3(b""), 0xdead_beef);
        assert_eq!(lookup3(b"Four score and seven years ago"), 0x1777_0551);
        assert_eq!(lookup3(b"abc"), 0x0e39_7631);
        assert_eq!(lookup3(b"0123456789ab"), 0x1065_e50a);
        assert_eq!(lookup3(b"0123456789abc"), 0x7351_ce56);
    }
}
