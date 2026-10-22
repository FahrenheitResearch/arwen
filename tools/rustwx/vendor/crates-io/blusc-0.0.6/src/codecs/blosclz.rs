const MAX_COPY: usize = 32;
const MAX_DISTANCE: usize = 8191;
const MAX_FARDISTANCE: usize = 65535 + MAX_DISTANCE - 1;
const HASH_LOG: usize = 14;

/// Decompresses a raw BloscLZ-compressed block into `output`.
///
/// Returns the number of bytes written to `output`. This operates on a single
/// block — the outer Blosc framing and filter pipeline are handled by
/// [`crate::internal::decompress`].
pub fn decompress(input: &[u8], output: &mut [u8]) -> usize {
    if input.is_empty() {
        return 0;
    }
    let mut ip = 0;
    let mut op = 0;
    let ip_limit = input.len();
    let op_limit = output.len();

    let mut ctrl = (input[ip] & 31) as u32;
    ip += 1;

    loop {
        if ctrl >= 32 {
            let mut len = (ctrl >> 5) as i32 - 1;
            let ofs = (ctrl & 31) as i32 * 256;

            if len == 6 {
                loop {
                    if ip >= ip_limit {
                        return 0;
                    }
                    let code = input[ip];
                    ip += 1;
                    len += code as i32;
                    if code != 255 {
                        break;
                    }
                }
            }

            if ip >= ip_limit {
                return 0;
            }
            let code = input[ip];
            ip += 1;
            len += 3;

            let mut distance = ofs + code as i32;

            if code == 255 {
                if ofs == 31 * 256 {
                    if ip + 1 >= ip_limit {
                        return 0;
                    }
                    let ofs_new = (input[ip] as i32) << 8 | (input[ip + 1] as i32);
                    ip += 2;
                    distance = ofs_new + MAX_DISTANCE as i32;
                }
            }

            distance += 1;

            if op + len as usize > op_limit {
                return 0;
            }
            if (op as i32) < distance {
                return 0;
            }

            let ref_pos = op - distance as usize;

            if ip >= ip_limit {
                break;
            }
            ctrl = input[ip] as u32;
            ip += 1;

            if distance == 1 {
                let val = output[ref_pos];
                for i in 0..len as usize {
                    output[op + i] = val;
                }
            } else {
                if distance as usize >= len as usize {
                    output.copy_within(ref_pos..ref_pos + len as usize, op);
                } else {
                    for i in 0..len as usize {
                        output[op + i] = output[ref_pos + i];
                    }
                }
            }
            op += len as usize;
        } else {
            ctrl += 1;
            if op + ctrl as usize > op_limit {
                return 0;
            }
            if ip + ctrl as usize > ip_limit {
                return 0;
            }

            output[op..op + ctrl as usize].copy_from_slice(&input[ip..ip + ctrl as usize]);
            op += ctrl as usize;
            ip += ctrl as usize;

            if ip >= ip_limit {
                break;
            }
            ctrl = input[ip] as u32;
            ip += 1;
        }
    }

    op
}

#[inline]
fn hash_function(seq: u32, hashlog: usize) -> usize {
    if hashlog == 0 {
        return 0;
    }
    (seq.wrapping_mul(2654435761) >> (32 - hashlog)) as usize
}

/// Equivalent to C get_match: returns index one past last matching byte.
/// Compares bytes starting from ip and ref_pos, up to ip_bound.
/// Returns the position after the last matching byte (mimics C post-increment behavior).
#[inline]
fn get_match(input: &[u8], mut ip: usize, ip_bound: usize, mut ref_pos: usize) -> usize {
    // Mimics C get_match (non-STRICT_ALIGN path):
    //   while (ip < (ip_bound - 8)) {
    //     if (*(int64_t*)ref != *(int64_t*)ip) {
    //       while (*ref++ == *ip++) {}   // no bounds check!
    //       return ip;
    //     }
    //     ip += 8; ref += 8;
    //   }
    //   while ((ip < ip_bound) && (*ref++ == *ip++)) {}
    //   return ip;
    while ip + 8 <= ip_bound {
        if ref_pos + 8 <= input.len() {
            let ip_bytes = u64::from_le_bytes(input[ip..ip + 8].try_into().unwrap());
            let ref_bytes = u64::from_le_bytes(input[ref_pos..ref_pos + 8].try_into().unwrap());
            if ip_bytes != ref_bytes {
                // Byte-by-byte with no ip_bound check (matches C behavior)
                while ref_pos < input.len() && input[ref_pos] == input[ip] {
                    ref_pos += 1;
                    ip += 1;
                }
                // C post-increment: ip advances one more on mismatch (always, no bounds check)
                ip += 1;
                return ip;
            } else {
                ip += 8;
                ref_pos += 8;
            }
        } else {
            break;
        }
    }
    // Scalar remainder: while ((ip < ip_bound) && (*ref++ == *ip++)) {}
    loop {
        if ip >= ip_bound {
            break;
        }
        if ref_pos >= input.len() {
            break;
        }
        let matches = input[ref_pos] == input[ip];
        // C post-increment: both advance regardless of match result
        ref_pos += 1;
        ip += 1;
        if !matches {
            break;
        }
    }
    ip
}

/// Equivalent to C get_run: for run-length detection (distance==0, i.e. repeated byte).
/// Unlike get_match, get_run does NOT post-increment ip on mismatch.
/// C get_run (non-STRICT_ALIGN path):
///   uint8_t x = ip[-1];
///   int64_t value; memset(&value, x, 8);
///   while (ip < (ip_bound - 8)) {
///     value2 = ((int64_t*)ref)[0];
///     if (value != value2) { while (*ref++ == x) ip++; return ip; }
///     ip += 8; ref += 8;
///   }
///   while ((ip < ip_bound) && (*ref++ == x)) ip++;
///   return ip;
#[inline]
fn get_run(input: &[u8], mut ip: usize, ip_bound: usize, mut ref_pos: usize) -> usize {
    let x = input[ip - 1]; // The repeated byte value
    let xval = x as u64;
    let broadcast = xval * 0x0101010101010101u64;
    // 8-byte fast path
    while ip + 8 <= ip_bound {
        if ref_pos + 8 <= input.len() {
            let ref_bytes = u64::from_ne_bytes(input[ref_pos..ref_pos + 8].try_into().unwrap());
            if broadcast != ref_bytes {
                // C: while (*ref++ == x) ip++;
                while ref_pos < input.len() && input[ref_pos] == x {
                    ref_pos += 1;
                    ip += 1;
                }
                return ip;
            } else {
                ip += 8;
                ref_pos += 8;
            }
        } else {
            break;
        }
    }
    // Scalar remainder: while ((ip < ip_bound) && (*ref++ == x)) ip++;
    while ip < ip_bound && ref_pos < input.len() && input[ref_pos] == x {
        ref_pos += 1;
        ip += 1;
    }
    ip
}

/// Entropy probing: estimates the compression ratio for a portion of the buffer.
/// Returns the estimated compression ratio. Also populates htab (but it will be
/// re-initialized before the main compression loop, so this is just for the ratio estimate).
/// Mirrors C get_cratio().
fn get_cratio(
    input: &[u8],
    base: usize,
    maxlen: usize,
    minlen: usize,
    ipshift: usize,
    htab: &mut [usize],
    hashlog: usize,
) -> f64 {
    let hashlen = 1usize << hashlog;
    let limit = if maxlen > hashlen { hashlen } else { maxlen };
    if limit < 13 {
        return f64::MAX;
    } // too small to probe

    let ip_bound = base + limit - 1;
    let ip_limit_probe = base + limit - 12;

    // Initialize hash table to 0 (distances of 0)
    for i in 0..hashlen {
        htab[i] = 0;
    }

    let mut ip = base;
    let mut oc: i32 = 0;
    let mut copy: u32 = 4;
    oc += 5;

    while ip < ip_limit_probe {
        let anchor = ip;

        let seq = u32::from_le_bytes(input[ip..ip + 4].try_into().unwrap());
        let hval = hash_function(seq, hashlog);
        let ref_pos = base + htab[hval]; // relative to base
        let distance = anchor - ref_pos; // could underflow if ref_pos > anchor, but htab starts at 0

        htab[hval] = anchor - base;

        if distance == 0 || distance >= MAX_FARDISTANCE {
            // LITERAL2
            oc += 1;
            ip = anchor + 1;
            copy += 1;
            if copy == MAX_COPY as u32 {
                copy = 0;
                oc += 1;
            }
            continue;
        }

        // Check first 4 bytes
        if ref_pos + 4 <= input.len()
            && ip + 4 <= input.len()
            && input[ref_pos..ref_pos + 4] == input[ip..ip + 4]
        {
            // match found
        } else {
            // LITERAL2
            oc += 1;
            ip = anchor + 1;
            copy += 1;
            if copy == MAX_COPY as u32 {
                copy = 0;
                oc += 1;
            }
            continue;
        }

        ip = anchor + 4;
        let dist = distance - 1;

        // get_run_or_match
        let ref_after = ref_pos + 4;
        if dist == 0 {
            ip = get_run(input, ip, ip_bound, ref_after);
        } else {
            ip = get_match(input, ip, ip_bound, ref_after);
        }

        ip -= ipshift;
        let len = ip - anchor;
        if (len as i32) < minlen as i32 {
            // LITERAL2
            oc += 1;
            ip = anchor + 1;
            copy += 1;
            if copy == MAX_COPY as u32 {
                copy = 0;
                oc += 1;
            }
            continue;
        }

        if copy == 0 {
            oc -= 1;
        }
        copy = 0;

        if dist < MAX_DISTANCE {
            if len >= 7 {
                oc += ((len as i32 - 7) / 255) + 1;
            }
            oc += 2;
        } else {
            if len >= 7 {
                oc += ((len as i32 - 7) / 255) + 1;
            }
            oc += 4;
        }

        // Update hash at match boundary
        if ip + 4 <= input.len() {
            let seq = u32::from_le_bytes(input[ip..ip + 4].try_into().unwrap());
            let hval = hash_function(seq, hashlog);
            htab[hval] = ip - base;
        }
        ip += 1;
        ip += 1;
        oc += 1;
    }

    let ic = (ip - base) as f64;
    if oc <= 0 {
        return f64::MAX;
    }
    ic / oc as f64
}

/// Compresses `input` into `output` using the BloscLZ algorithm.
///
/// - `clevel`: compression level (1–9). Higher levels try harder at the cost of speed.
///
/// Returns the number of compressed bytes written to `output`, or 0 if the data
/// is incompressible at the given level.
pub fn compress(clevel: i32, input: &[u8], output: &mut [u8]) -> usize {
    if input.is_empty() {
        return 0;
    }
    let length = input.len();
    let op_limit = output.len();
    if length < 16 || op_limit < 66 {
        return 0;
    }

    // ipshift and minlen constants (see C blosclz.c comments)
    let ipshift: usize = 4;
    let minlen: usize = 4;

    let hashlog: usize = match clevel {
        0 => 0,
        1 => HASH_LOG - 2,
        2 => HASH_LOG - 1,
        _ => HASH_LOG,
    };
    let hash_size = 1usize << hashlog;
    let mut htab = vec![0usize; hash_size];

    // Entropy probing: estimate compression ratio and bail early if too low.
    // The probe length depends on clevel.
    let maxlen = if clevel < 2 {
        length / 8
    } else if clevel < 4 {
        length / 4
    } else if clevel < 7 {
        length / 2
    } else {
        length
    };
    let shift = length - maxlen;
    let cratio = get_cratio(input, shift, maxlen, minlen, ipshift, &mut htab, hashlog);
    let cratio_thresholds: [f64; 10] = [0.0, 2.0, 1.5, 1.2, 1.2, 1.2, 1.2, 1.15, 1.1, 1.0];
    if cratio < cratio_thresholds[clevel as usize] {
        return 0;
    }

    let mut ip: usize = 0;
    let ip_bound = length - 1;
    let ip_limit = length - 12;
    let mut op: usize = 0;

    // Re-initialize hash table (C does this after entropy probing)
    for i in 0..hash_size {
        htab[i] = 0;
    }

    // Start with literal copy
    let mut copy: usize = 4;
    output[op] = (MAX_COPY - 1) as u8;
    op += 1;
    output[op] = input[ip];
    op += 1;
    ip += 1;
    output[op] = input[ip];
    op += 1;
    ip += 1;
    output[op] = input[ip];
    op += 1;
    ip += 1;
    output[op] = input[ip];
    op += 1;
    ip += 1;

    // Main loop
    while ip < ip_limit {
        let anchor = ip;

        // Find potential match
        let seq = u32::from_le_bytes(input[ip..ip + 4].try_into().unwrap());
        let hval = hash_function(seq, hashlog);
        let ref_pos = htab[hval]; // absolute position in input

        // Calculate distance
        let distance = anchor.wrapping_sub(ref_pos);

        // Update hash table
        htab[hval] = anchor;

        if distance == 0 || distance >= MAX_FARDISTANCE {
            // LITERAL
            if op + 2 > op_limit {
                break;
            }
            output[op] = input[anchor];
            op += 1;
            ip = anchor + 1;
            copy += 1;
            if copy == MAX_COPY {
                copy = 0;
                output[op] = (MAX_COPY - 1) as u8;
                op += 1;
            }
            continue;
        }

        // Check first 4 bytes for match
        if ref_pos + 4 <= input.len() && input[ref_pos..ref_pos + 4] == input[ip..ip + 4] {
            // Match found
        } else {
            // LITERAL
            if op + 2 > op_limit {
                break;
            }
            output[op] = input[anchor];
            op += 1;
            ip = anchor + 1;
            copy += 1;
            if copy == MAX_COPY {
                copy = 0;
                output[op] = (MAX_COPY - 1) as u8;
                op += 1;
            }
            continue;
        }

        // Last matched byte
        ip = anchor + 4;

        // Distance is biased
        let dist = distance - 1;

        // get_run_or_match: zero distance means a run
        let ref_after = ref_pos + 4;
        if dist == 0 {
            ip = get_run(input, ip, ip_bound, ref_after);
        } else {
            ip = get_match(input, ip, ip_bound, ref_after);
        }

        // Length is biased: '1' means a match of 3 bytes
        ip -= ipshift;

        let len = ip - anchor;

        // Encoding short lengths is expensive during decompression
        if len < minlen || (len <= 5 && dist >= MAX_DISTANCE) {
            // LITERAL - reset ip to anchor+1
            ip = anchor;
            if op + 2 > op_limit {
                break;
            }
            output[op] = input[anchor];
            op += 1;
            ip = anchor + 1;
            copy += 1;
            if copy == MAX_COPY {
                copy = 0;
                output[op] = (MAX_COPY - 1) as u8;
                op += 1;
            }
            continue;
        }

        // If we have copied something, adjust the copy count
        if copy > 0 {
            // copy is biased, '0' means 1 byte copy
            output[op - copy - 1] = (copy - 1) as u8;
        } else {
            // Back, to overwrite the copy count
            op -= 1;
        }
        // Reset literal counter
        copy = 0;

        // Encode the match
        if dist < MAX_DISTANCE {
            if len < 7 {
                // MATCH_SHORT
                if op + 2 > op_limit {
                    break;
                }
                output[op] = ((len as u8) << 5) + ((dist >> 8) as u8);
                op += 1;
                output[op] = (dist & 255) as u8;
                op += 1;
            } else {
                // MATCH_LONG
                if op + 1 > op_limit {
                    break;
                }
                output[op] = (7u8 << 5) + ((dist >> 8) as u8);
                op += 1;
                let mut l = len - 7;
                while l >= 255 {
                    if op + 1 > op_limit {
                        return 0;
                    }
                    output[op] = 255;
                    op += 1;
                    l -= 255;
                }
                if op + 2 > op_limit {
                    return 0;
                }
                output[op] = l as u8;
                op += 1;
                output[op] = (dist & 255) as u8;
                op += 1;
            }
        } else {
            // Far distance
            let dist_far = dist - MAX_DISTANCE;
            if len < 7 {
                // MATCH_SHORT_FAR
                if op + 4 > op_limit {
                    break;
                }
                output[op] = ((len as u8) << 5) + 31;
                op += 1;
                output[op] = 255;
                op += 1;
                output[op] = (dist_far >> 8) as u8;
                op += 1;
                output[op] = (dist_far & 255) as u8;
                op += 1;
            } else {
                // MATCH_LONG_FAR
                if op + 1 > op_limit {
                    break;
                }
                output[op] = (7u8 << 5) + 31;
                op += 1;
                let mut l = len - 7;
                while l >= 255 {
                    if op + 1 > op_limit {
                        return 0;
                    }
                    output[op] = 255;
                    op += 1;
                    l -= 255;
                }
                if op + 4 > op_limit {
                    return 0;
                }
                output[op] = l as u8;
                op += 1;
                output[op] = 255;
                op += 1;
                output[op] = (dist_far >> 8) as u8;
                op += 1;
                output[op] = (dist_far & 255) as u8;
                op += 1;
            }
        }

        // Update the hash at match boundary
        if ip + 4 <= input.len() {
            let seq = u32::from_le_bytes(input[ip..ip + 4].try_into().unwrap());
            let hval = hash_function(seq, hashlog);
            htab[hval] = ip;
            ip += 1;
            if clevel == 9 {
                // In some situations, including a second hash proves useful
                let seq_shifted = seq >> 8;
                let hval_shifted = hash_function(seq_shifted, hashlog);
                htab[hval_shifted] = ip;
            }
            ip += 1;
        } else {
            ip += 2;
        }

        if op + 1 > op_limit {
            break;
        }
        // Assuming literal copy
        output[op] = (MAX_COPY - 1) as u8;
        op += 1;
    }

    // Left-over as literal copy
    while ip <= ip_bound {
        if op + 2 > op_limit {
            return 0;
        }
        output[op] = input[ip];
        op += 1;
        ip += 1;
        copy += 1;
        if copy == MAX_COPY {
            copy = 0;
            output[op] = (MAX_COPY - 1) as u8;
            op += 1;
        }
    }

    // Adjust final copy length
    if copy > 0 {
        output[op - copy - 1] = (copy - 1) as u8;
    } else {
        op -= 1;
    }

    // Marker for blosclz
    output[0] |= 1 << 5;

    op
}
