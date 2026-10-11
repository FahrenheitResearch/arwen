/// Unshuffle bytes — reverses the HDF5 shuffle filter.
///
/// The shuffle filter rearranges bytes so that byte 0 of all elements comes first,
/// then byte 1 of all elements, etc. This improves compression of numeric arrays
/// where adjacent elements often have similar high bytes.
///
/// On read, we need to unshuffle: given the shuffled bytes, reconstruct the
/// original element order.
pub fn unshuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let n_elements = data.len() / element_size;
    if n_elements == 0 {
        return data.to_vec();
    }

    let mut output = vec![0u8; data.len()];
    let complete = n_elements * element_size;

    // Byte plane `b` holds byte `b` of every element.  Walking the output
    // one element at a time with the planes pre-split keeps every access
    // in bounds by construction, so the loop carries no per-byte checks
    // (the indexed form with two bounds tests per byte cost ~1 s on an
    // 88 MB chunk of NCAR's 2.5 km restart).
    let planes: Vec<&[u8]> = (0..element_size)
        .map(|byte_idx| &data[byte_idx * n_elements..(byte_idx + 1) * n_elements])
        .collect();
    match element_size {
        4 => {
            let (p0, p1, p2, p3) = (planes[0], planes[1], planes[2], planes[3]);
            for (elem, out) in output[..complete].chunks_exact_mut(4).enumerate() {
                out[0] = p0[elem];
                out[1] = p1[elem];
                out[2] = p2[elem];
                out[3] = p3[elem];
            }
        }
        8 => {
            for (elem, out) in output[..complete].chunks_exact_mut(8).enumerate() {
                for (byte_idx, plane) in planes.iter().enumerate() {
                    out[byte_idx] = plane[elem];
                }
            }
        }
        _ => {
            for (elem, out) in output[..complete].chunks_exact_mut(element_size).enumerate() {
                for (byte_idx, plane) in planes.iter().enumerate() {
                    out[byte_idx] = plane[elem];
                }
            }
        }
    }

    // Copy any trailing bytes that don't form a complete element
    if complete < data.len() {
        output[complete..].copy_from_slice(&data[complete..]);
    }

    output
}

/// Shuffle bytes — for testing purposes.
#[cfg(test)]
fn shuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let n_elements = data.len() / element_size;
    let mut output = vec![0u8; data.len()];

    for byte_idx in 0..element_size {
        let dst_start = byte_idx * n_elements;
        for elem in 0..n_elements {
            output[dst_start + elem] = data[elem * element_size + byte_idx];
        }
    }

    let complete = n_elements * element_size;
    if complete < data.len() {
        output[complete..].copy_from_slice(&data[complete..]);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_4byte() {
        let original: Vec<u8> = vec![
            0x01, 0x02, 0x03, 0x04, // element 0
            0x05, 0x06, 0x07, 0x08, // element 1
            0x09, 0x0A, 0x0B, 0x0C, // element 2
        ];
        let shuffled = shuffle(&original, 4);
        // After shuffle: [0x01,0x05,0x09, 0x02,0x06,0x0A, 0x03,0x07,0x0B, 0x04,0x08,0x0C]
        assert_eq!(
            shuffled,
            vec![0x01, 0x05, 0x09, 0x02, 0x06, 0x0A, 0x03, 0x07, 0x0B, 0x04, 0x08, 0x0C]
        );

        let unshuffled = unshuffle(&shuffled, 4);
        assert_eq!(unshuffled, original);
    }

    #[test]
    fn test_single_element() {
        let data = vec![0x01, 0x02, 0x03, 0x04];
        let result = unshuffle(&data, 4);
        assert_eq!(result, data);
    }

    #[test]
    fn test_element_size_1() {
        let data = vec![0x01, 0x02, 0x03];
        let result = unshuffle(&data, 1);
        assert_eq!(result, data);
    }
}
