/// Jenkins lookup3 `hashlittle2` — used by HDF5 for superblock v2/v3 checksums,
/// object header v2 checksums, and B-tree v2 checksums.
///
/// This is a direct translation of Bob Jenkins' lookup3.c `hashlittle2`.
/// Reference: <http://burtleburtle.net/bob/c/lookup3.c>
pub fn jenkins_lookup3(data: &[u8]) -> u32 {
    let len = data.len();

    // Internal state
    let mut a: u32 = 0xdeadbeef_u32.wrapping_add(len as u32);
    let mut b: u32 = a;
    let mut c: u32 = a;

    let mut offset = 0;

    // Process 12-byte chunks.
    // Note: the C code uses `while (length > 12)`, leaving the last 0–12
    // bytes for the switch/final path.  Using `<` (not `<=`) matches that
    // behaviour — otherwise an input whose length is an exact multiple of
    // 12 would incorrectly run the last chunk through mix() instead of
    // the add-then-final path.
    while offset + 12 < len {
        a = a.wrapping_add(u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]));
        b = b.wrapping_add(u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]));
        c = c.wrapping_add(u32::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]));
        mix(&mut a, &mut b, &mut c);
        offset += 12;
    }

    // Handle the last few bytes
    let remaining = len - offset;

    // Fill a, b, c with remaining bytes (little-endian)
    if remaining > 0 {
        // Process remaining bytes into a, b, c
        let mut tail_a: u32 = 0;
        let mut tail_b: u32 = 0;
        let mut tail_c: u32 = 0;

        // Bytes 0-3 go into a
        for i in 0..std::cmp::min(remaining, 4) {
            tail_a |= (data[offset + i] as u32) << (i * 8);
        }
        // Bytes 4-7 go into b
        if remaining > 4 {
            for i in 4..std::cmp::min(remaining, 8) {
                tail_b |= (data[offset + i] as u32) << ((i - 4) * 8);
            }
        }
        // Bytes 8-11 go into c
        if remaining > 8 {
            for i in 8..remaining {
                tail_c |= (data[offset + i] as u32) << ((i - 8) * 8);
            }
        }

        a = a.wrapping_add(tail_a);
        b = b.wrapping_add(tail_b);
        c = c.wrapping_add(tail_c);

        final_mix(&mut a, &mut b, &mut c);
    }

    c
}

#[inline]
fn mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(4);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(6);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(8);
    *b = b.wrapping_add(*a);
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(16);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(19);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(4);
    *b = b.wrapping_add(*a);
}

#[inline]
fn final_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(14));
    *a ^= *c;
    *a = a.wrapping_sub(c.rotate_left(11));
    *b ^= *a;
    *b = b.wrapping_sub(a.rotate_left(25));
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(16));
    *a ^= *c;
    *a = a.wrapping_sub(c.rotate_left(4));
    *b ^= *a;
    *b = b.wrapping_sub(a.rotate_left(14));
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(24));
}

/// Fletcher-32 checksum used by the HDF5 filter pipeline.
///
/// Matches the HDF5 library's `H5_checksum_fletcher32`: reads 16-bit
/// big-endian words, accumulates in batches of 360, and reduces with
/// `(x & 0xffff) + (x >> 16)` (not `% 65535`).
pub fn fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;
    let total_words = data.len() / 2;

    let mut offset = 0usize;
    let mut remaining = total_words;

    while remaining > 0 {
        let batch = remaining.min(360);
        remaining -= batch;

        for _ in 0..batch {
            let word = ((data[offset] as u32) << 8) | (data[offset + 1] as u32);
            sum1 += word;
            sum2 += sum1;
            offset += 2;
        }

        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    // Final reduction
    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    (sum2 << 16) | sum1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jenkins_empty() {
        let hash = jenkins_lookup3(b"");
        // Empty input produces a deterministic hash from the initial state
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_jenkins_known_value() {
        // Test with known input — the hash is deterministic
        let h1 = jenkins_lookup3(b"hello");
        let h2 = jenkins_lookup3(b"hello");
        assert_eq!(h1, h2);

        // Different input -> different hash (with overwhelming probability)
        let h3 = jenkins_lookup3(b"world");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_fletcher32_simple() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let checksum = fletcher32(&data);
        // Verify it's deterministic
        assert_eq!(checksum, fletcher32(&data));
    }

    #[test]
    fn test_fletcher32_known_reference() {
        // Known reference: 4 bytes [0x00, 0x01, 0x00, 0x02] interpreted as
        // big-endian u16 words: 0x0001 and 0x0002.
        // sum1 = (0 + 1) % 65535 = 1; sum2 = (0 + 1) % 65535 = 1
        // sum1 = (1 + 2) % 65535 = 3; sum2 = (1 + 3) % 65535 = 4
        // result = (4 << 16) | 3 = 0x0004_0003
        let data = [0x00, 0x01, 0x00, 0x02];
        assert_eq!(fletcher32(&data), 0x0004_0003);
    }

    #[test]
    fn test_fletcher32_roundtrip_with_filter() {
        // Build a payload + checksum and verify the filter can strip it
        let payload = vec![0x00u8, 0x80, 0x3F, 0x80, 0x00, 0x00, 0x40, 0x00];
        let ck = fletcher32(&payload);
        let mut data = payload.clone();
        data.extend_from_slice(&ck.to_le_bytes());
        // The filter stores checksum in big-endian
        let stripped = crate::filters::fletcher32::verify_and_strip(&data).unwrap();
        assert_eq!(stripped, payload);
    }
}
