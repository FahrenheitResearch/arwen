// Affine i16 codec ported from rustwx volume_store/codec.rs (review/grib-wxa-fast-plots-20260605); f32 tile codec is new.

use crate::error::{RwResult, RwStoreError};
use crate::format::{FLAG_CONSTANT, FLAG_EMPTY, FLAG_HAS_MISSING};

/// Quantized sentinel marking a missing (NaN) value.
pub const MISSING_Q: i16 = i16::MIN;
/// Smallest quantized value representing finite data.
pub const Q_MIN: i16 = i16::MIN + 1;
/// Largest quantized value representing finite data.
pub const Q_MAX: i16 = i16::MAX;

/// Result of encoding one chunk of values. The payload is uncompressed;
/// compression (zstd) happens in the writer layer.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedChunk {
    pub flags: u8,
    pub center: f32,
    pub scale: f32,
    pub min: f32,
    pub max: f32,
    pub valid_count: u32,
    pub payload: Vec<u8>,
}

struct ChunkScan {
    valid_min: f32,
    valid_max: f32,
    valid_count: usize,
    has_missing: bool,
}

fn scan_values(values: &[f32]) -> ChunkScan {
    let mut valid_min = f32::INFINITY;
    let mut valid_max = f32::NEG_INFINITY;
    let mut valid_count = 0usize;
    let mut has_missing = false;
    for value in values {
        if value.is_finite() {
            valid_min = valid_min.min(*value);
            valid_max = valid_max.max(*value);
            valid_count += 1;
        } else {
            has_missing = true;
        }
    }
    ChunkScan {
        valid_min,
        valid_max,
        valid_count,
        has_missing,
    }
}

fn empty_chunk() -> EncodedChunk {
    EncodedChunk {
        flags: FLAG_EMPTY,
        center: 0.0,
        scale: 0.0,
        min: f32::NAN,
        max: f32::NAN,
        valid_count: 0,
        payload: Vec::new(),
    }
}

/// Encode a chunk of values as affine-quantized i16 (3D column codec).
///
/// Errors when the value range cannot produce a finite positive scale
/// (e.g. a range that overflows f32) — same guard the original codec had;
/// silently quantizing through a non-finite scale would destroy data.
pub fn encode_affine_i16(values: &[f32]) -> RwResult<EncodedChunk> {
    let scan = scan_values(values);
    if scan.valid_count == 0 {
        return Ok(empty_chunk());
    }

    if (scan.valid_max - scan.valid_min).abs() <= f32::EPSILON {
        if scan.has_missing {
            let mut payload = Vec::with_capacity(values.len() * 2);
            for value in values {
                let q = if value.is_finite() { 0i16 } else { MISSING_Q };
                payload.extend_from_slice(&q.to_le_bytes());
            }
            return Ok(EncodedChunk {
                flags: FLAG_CONSTANT | FLAG_HAS_MISSING,
                center: scan.valid_min,
                scale: 0.0,
                min: scan.valid_min,
                max: scan.valid_max,
                valid_count: scan.valid_count as u32,
                payload,
            });
        }
        return Ok(EncodedChunk {
            flags: FLAG_CONSTANT,
            center: scan.valid_min,
            scale: 0.0,
            min: scan.valid_min,
            max: scan.valid_max,
            valid_count: scan.valid_count as u32,
            payload: Vec::new(),
        });
    }

    // Range math in f64 so extreme f32 spans cannot overflow to inf.
    let range = f64::from(scan.valid_max) - f64::from(scan.valid_min);
    let center = (f64::from(scan.valid_min) + 0.5 * range) as f32;
    let scale = (range / (2.0 * f64::from(Q_MAX))) as f32;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(RwStoreError::Chunk(format!(
            "invalid affine quantization scale {scale} (range {} .. {})",
            scan.valid_min, scan.valid_max
        )));
    }

    let mut payload = Vec::with_capacity(values.len() * 2);
    for value in values {
        let q = if value.is_finite() {
            ((*value - center) / scale)
                .round()
                .clamp(f32::from(Q_MIN), f32::from(Q_MAX)) as i16
        } else {
            MISSING_Q
        };
        payload.extend_from_slice(&q.to_le_bytes());
    }

    let flags = if scan.has_missing {
        FLAG_HAS_MISSING
    } else {
        0
    };
    Ok(EncodedChunk {
        flags,
        center,
        scale,
        min: scan.valid_min,
        max: scan.valid_max,
        valid_count: scan.valid_count as u32,
        payload,
    })
}

/// Decode a chunk encoded by [`encode_affine_i16`].
pub fn decode_affine_i16(
    flags: u8,
    center: f32,
    scale: f32,
    payload: &[u8],
    value_count: usize,
) -> RwResult<Vec<f32>> {
    if flags & FLAG_EMPTY != 0 {
        return Ok(vec![f32::NAN; value_count]);
    }
    if flags & FLAG_CONSTANT != 0 && payload.is_empty() {
        return Ok(vec![center; value_count]);
    }

    let expected_len = value_count * 2;
    if payload.len() != expected_len {
        return Err(RwStoreError::Chunk(format!(
            "dense i16 payload has {} bytes, expected {expected_len}",
            payload.len()
        )));
    }

    let mut values = Vec::with_capacity(value_count);
    for pair in payload.chunks_exact(2) {
        let q = i16::from_le_bytes(pair.try_into().unwrap());
        if q == MISSING_Q {
            values.push(f32::NAN);
        } else if flags & FLAG_CONSTANT != 0 {
            values.push(center);
        } else {
            values.push(center + scale * f32::from(q));
        }
    }
    Ok(values)
}

/// Encode a chunk of values as raw little-endian f32 (2D tile codec).
/// Constant-with-missing chunks are dense-encoded with NaNs inline.
///
/// The constant-tile shortcut requires EXACT constancy: an earlier
/// epsilon-based check (`range <= f32::EPSILON`) silently collapsed any
/// tile whose absolute value range was below ~1.2e-7 to a single value —
/// which destroyed real fields with tiny absolute magnitudes (HRRR 8 m
/// smoke mass density spans 1e-25..1e-7 kg/m^3, so essentially every tile
/// flattened). This codec is the store's lossless lane; nothing here may
/// round.
pub fn encode_f32_tile(values: &[f32]) -> EncodedChunk {
    let scan = scan_values(values);
    if scan.valid_count == 0 {
        return empty_chunk();
    }

    if scan.valid_max == scan.valid_min && !scan.has_missing {
        return EncodedChunk {
            flags: FLAG_CONSTANT,
            center: scan.valid_min,
            scale: 0.0,
            min: scan.valid_min,
            max: scan.valid_max,
            valid_count: scan.valid_count as u32,
            payload: Vec::new(),
        };
    }

    let mut payload = Vec::with_capacity(values.len() * 4);
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }

    let flags = if scan.has_missing {
        FLAG_HAS_MISSING
    } else {
        0
    };
    EncodedChunk {
        flags,
        center: 0.5 * (scan.valid_min + scan.valid_max),
        scale: 0.0,
        min: scan.valid_min,
        max: scan.valid_max,
        valid_count: scan.valid_count as u32,
        payload,
    }
}

/// Decode a chunk encoded by [`encode_f32_tile`].
pub fn decode_f32_tile(
    flags: u8,
    center: f32,
    payload: &[u8],
    value_count: usize,
) -> RwResult<Vec<f32>> {
    if flags & FLAG_EMPTY != 0 {
        return Ok(vec![f32::NAN; value_count]);
    }
    if flags & FLAG_CONSTANT != 0 && payload.is_empty() {
        return Ok(vec![center; value_count]);
    }

    let expected_len = value_count * 4;
    if payload.len() != expected_len {
        return Err(RwStoreError::Chunk(format!(
            "dense f32 payload has {} bytes, expected {expected_len}",
            payload.len()
        )));
    }

    let mut values = Vec::with_capacity(value_count);
    for quad in payload.chunks_exact(4) {
        values.push(f32::from_le_bytes(quad.try_into().unwrap()));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{FLAG_CONSTANT, FLAG_EMPTY, FLAG_HAS_MISSING};

    #[test]
    fn affine_round_trip_keeps_error_bounded() {
        let values = vec![
            -1500.0,
            -2.5,
            0.0,
            0.015,
            f32::NAN,
            750.25,
            2000.0,
            f32::NAN,
            -0.875,
            1234.5,
        ];
        let encoded = encode_affine_i16(&values).expect("encode");
        assert!(encoded.flags & FLAG_HAS_MISSING != 0);
        assert_eq!(encoded.payload.len(), values.len() * 2);
        let decoded = decode_affine_i16(
            encoded.flags,
            encoded.center,
            encoded.scale,
            &encoded.payload,
            values.len(),
        )
        .unwrap();
        let tolerance = encoded.scale + 1e-4;
        for (source, round_trip) in values.iter().zip(decoded.iter()) {
            if source.is_finite() {
                assert!(
                    (source - round_trip).abs() <= tolerance,
                    "source {source} decoded {round_trip} exceeds tolerance {tolerance}"
                );
            } else {
                assert!(round_trip.is_nan());
            }
        }
    }

    #[test]
    fn affine_constant_chunk_has_no_payload() {
        let values = vec![12.5; 16];
        let encoded = encode_affine_i16(&values).expect("encode");
        assert_eq!(encoded.flags, FLAG_CONSTANT);
        assert!(encoded.payload.is_empty());
        let decoded = decode_affine_i16(
            encoded.flags,
            encoded.center,
            encoded.scale,
            &encoded.payload,
            values.len(),
        )
        .unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn affine_constant_with_missing_keeps_sentinel_payload() {
        let values = vec![7.25, f32::NAN, 7.25, 7.25, f32::NAN, 7.25];
        let encoded = encode_affine_i16(&values).expect("encode");
        assert_eq!(encoded.flags, FLAG_CONSTANT | FLAG_HAS_MISSING);
        assert_eq!(encoded.payload.len(), values.len() * 2);
        for (value, pair) in values.iter().zip(encoded.payload.chunks_exact(2)) {
            let q = i16::from_le_bytes(pair.try_into().unwrap());
            if value.is_finite() {
                assert_eq!(q, 0);
            } else {
                assert_eq!(q, i16::MIN);
            }
        }
        let decoded = decode_affine_i16(
            encoded.flags,
            encoded.center,
            encoded.scale,
            &encoded.payload,
            values.len(),
        )
        .unwrap();
        for (source, round_trip) in values.iter().zip(decoded.iter()) {
            if source.is_finite() {
                assert_eq!(*round_trip, 7.25);
            } else {
                assert!(round_trip.is_nan());
            }
        }
    }

    #[test]
    fn affine_extreme_range_stays_finite_and_bounded() {
        // Regression: max - min overflows f32 to inf; the original port
        // silently produced scale = inf and decoded everything to NaN.
        // With f64 range math this must encode and round-trip normally.
        let values = vec![-3.0e38_f32, 3.0e38_f32];
        let encoded = encode_affine_i16(&values).expect("encode");
        assert!(encoded.scale.is_finite() && encoded.scale > 0.0);
        let decoded = decode_affine_i16(
            encoded.flags,
            encoded.center,
            encoded.scale,
            &encoded.payload,
            values.len(),
        )
        .unwrap();
        for (source, round_trip) in values.iter().zip(decoded.iter()) {
            assert!(round_trip.is_finite());
            assert!(
                (source - round_trip).abs() <= encoded.scale,
                "extreme-range error {} exceeds scale {}",
                (source - round_trip).abs(),
                encoded.scale
            );
        }
    }

    #[test]
    fn affine_all_missing_uses_empty_flag() {
        let values = vec![f32::NAN; 8];
        let encoded = encode_affine_i16(&values).expect("encode");
        assert_eq!(encoded.flags, FLAG_EMPTY);
        assert!(encoded.payload.is_empty());
        assert_eq!(encoded.valid_count, 0);
        let decoded = decode_affine_i16(
            encoded.flags,
            encoded.center,
            encoded.scale,
            &encoded.payload,
            values.len(),
        )
        .unwrap();
        assert_eq!(decoded.len(), values.len());
        assert!(decoded.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn f32_tile_round_trips_bit_exact() {
        let values = vec![
            -273.15,
            f32::NAN,
            0.0,
            1013.25,
            -0.5,
            f32::NAN,
            6.25e4,
            3.0e-3,
        ];
        let encoded = encode_f32_tile(&values);
        assert!(encoded.flags & FLAG_HAS_MISSING != 0);
        let expected_payload: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(encoded.payload, expected_payload);
        let decoded = decode_f32_tile(
            encoded.flags,
            encoded.center,
            &encoded.payload,
            values.len(),
        )
        .unwrap();
        assert_eq!(decoded.len(), values.len());
        for (source, round_trip) in values.iter().zip(decoded.iter()) {
            assert_eq!(source.to_bits(), round_trip.to_bits());
        }
    }

    /// Regression: tiles whose values differ by less than f32::EPSILON in
    /// ABSOLUTE terms are not constant. The old epsilon shortcut collapsed
    /// them to a single value, flattening real small-magnitude fields
    /// (HRRR 8 m smoke mass density lives entirely below 1.3e-7 kg/m^3).
    /// The lossless 2D lane must round-trip them bit-exactly.
    #[test]
    fn f32_tile_keeps_sub_epsilon_ranges_bit_exact() {
        let values = vec![1.226e-25_f32, 1.04e-9, 3.3e-8, 1.0e-7];
        assert!(
            (values[3] - values[0]).abs() <= f32::EPSILON,
            "test data must span less than f32::EPSILON to pin the regression"
        );
        let encoded = encode_f32_tile(&values);
        assert_eq!(
            encoded.flags & FLAG_CONSTANT,
            0,
            "a non-constant tile must not take the constant shortcut"
        );
        let decoded = decode_f32_tile(
            encoded.flags,
            encoded.center,
            &encoded.payload,
            values.len(),
        )
        .unwrap();
        for (source, round_trip) in values.iter().zip(decoded.iter()) {
            assert_eq!(source.to_bits(), round_trip.to_bits());
        }
    }

    #[test]
    fn f32_tile_constant_and_empty() {
        // Constant: exact reproduction via center, no payload.
        let constant = vec![3.5; 10];
        let encoded = encode_f32_tile(&constant);
        assert_eq!(encoded.flags, FLAG_CONSTANT);
        assert!(encoded.payload.is_empty());
        let decoded = decode_f32_tile(
            encoded.flags,
            encoded.center,
            &encoded.payload,
            constant.len(),
        )
        .unwrap();
        assert_eq!(decoded, constant);

        // Empty: all NaN in, all NaN out, no payload.
        let empty = vec![f32::NAN; 4];
        let encoded = encode_f32_tile(&empty);
        assert_eq!(encoded.flags, FLAG_EMPTY);
        assert!(encoded.payload.is_empty());
        let decoded =
            decode_f32_tile(encoded.flags, encoded.center, &encoded.payload, empty.len()).unwrap();
        assert!(decoded.iter().all(|value| value.is_nan()));

        // Constant with missing: dense-encoded raw f32 with NaNs inline.
        let constant_missing = vec![2.0, f32::NAN, 2.0];
        let encoded = encode_f32_tile(&constant_missing);
        assert_eq!(encoded.flags, FLAG_HAS_MISSING);
        assert_eq!(encoded.payload.len(), constant_missing.len() * 4);
        let expected_payload: Vec<u8> = constant_missing
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(encoded.payload, expected_payload);
        let decoded = decode_f32_tile(
            encoded.flags,
            encoded.center,
            &encoded.payload,
            constant_missing.len(),
        )
        .unwrap();
        for (source, round_trip) in constant_missing.iter().zip(decoded.iter()) {
            assert_eq!(source.to_bits(), round_trip.to_bits());
        }
    }
}
