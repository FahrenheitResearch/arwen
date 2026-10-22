// Seeded from the project hdf5-reader checksum implementation.
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
