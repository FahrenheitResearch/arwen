//! Binary chunk index record: 64-byte packed representation of a single
//! stored chunk's location, codec statistics, and tile coordinates.

use crate::error::{RwResult, RwStoreError};

/// A single entry in the rw-store chunk index.
///
/// Byte layout (little-endian, 64 bytes):
/// ```text
///  0- 1  var_id      u16
///  2      kind        u8
///  3      flags       u8
///  4- 7  tile_y      u32
///  8-11  tile_x      u32
/// 12-19  offset      u64
/// 20-23  len         u32
/// 24-27  raw_len     u32
/// 28-31  center      f32
/// 32-35  scale       f32
/// 36-39  min         f32
/// 40-43  max         f32
/// 44-47  valid_count u32
/// 48-63  reserved    (zeros)
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkRecord {
    pub var_id: u16,
    pub kind: u8,
    pub flags: u8,
    pub tile_y: u32,
    pub tile_x: u32,
    pub offset: u64,
    pub len: u32,
    pub raw_len: u32,
    pub center: f32,
    pub scale: f32,
    pub min: f32,
    pub max: f32,
    pub valid_count: u32,
}

impl ChunkRecord {
    /// Append exactly 64 bytes to `out`.
    pub fn pack_into(&self, out: &mut Vec<u8>) {
        out.reserve(64);
        out.extend_from_slice(&self.var_id.to_le_bytes()); // 0-1
        out.push(self.kind); // 2
        out.push(self.flags); // 3
        out.extend_from_slice(&self.tile_y.to_le_bytes()); // 4-7
        out.extend_from_slice(&self.tile_x.to_le_bytes()); // 8-11
        out.extend_from_slice(&self.offset.to_le_bytes()); // 12-19
        out.extend_from_slice(&self.len.to_le_bytes()); // 20-23
        out.extend_from_slice(&self.raw_len.to_le_bytes()); // 24-27
        out.extend_from_slice(&self.center.to_le_bytes()); // 28-31
        out.extend_from_slice(&self.scale.to_le_bytes()); // 32-35
        out.extend_from_slice(&self.min.to_le_bytes()); // 36-39
        out.extend_from_slice(&self.max.to_le_bytes()); // 40-43
        out.extend_from_slice(&self.valid_count.to_le_bytes()); // 44-47
        out.extend_from_slice(&[0u8; 16]); // 48-63 reserved
    }

    /// Parse a `ChunkRecord` from the first 64 bytes of `bytes`.
    pub fn unpack(bytes: &[u8]) -> RwResult<Self> {
        if bytes.len() < 64 {
            return Err(RwStoreError::Format(format!(
                "index record requires 64 bytes, got {}",
                bytes.len()
            )));
        }
        let var_id = u16::from_le_bytes(bytes[0..2].try_into().unwrap());
        let kind = bytes[2];
        let flags = bytes[3];
        let tile_y = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let tile_x = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let offset = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        let len = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let raw_len = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        let center = f32::from_le_bytes(bytes[28..32].try_into().unwrap());
        let scale = f32::from_le_bytes(bytes[32..36].try_into().unwrap());
        let min = f32::from_le_bytes(bytes[36..40].try_into().unwrap());
        let max = f32::from_le_bytes(bytes[40..44].try_into().unwrap());
        let valid_count = u32::from_le_bytes(bytes[44..48].try_into().unwrap());
        Ok(Self {
            var_id,
            kind,
            flags,
            tile_y,
            tile_x,
            offset,
            len,
            raw_len,
            center,
            scale,
            min,
            max,
            valid_count,
        })
    }

    /// Sort key: (var_id, kind, tile_y, tile_x) for lexicographic ordering.
    pub fn sort_key(&self) -> (u16, u8, u32, u32) {
        (self.var_id, self.kind, self.tile_y, self.tile_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record() -> ChunkRecord {
        ChunkRecord {
            var_id: 0xABCD,
            kind: 1,
            flags: 3,
            tile_y: 0x0102_0304,
            tile_x: 0x0506_0708,
            offset: 0xDEAD_BEEF_CAFE_1234,
            len: 0x1111_2222,
            raw_len: 0x3333_4444,
            center: 1.5_f32,
            scale: 0.25_f32,
            min: f32::NAN,
            max: f32::NAN,
            valid_count: 0x9999_AAAA,
        }
    }

    #[test]
    fn index_record_round_trips_through_64_bytes() {
        let r = make_record();
        let mut buf = Vec::new();
        r.pack_into(&mut buf);
        assert_eq!(buf.len(), 64);

        let r2 = ChunkRecord::unpack(&buf).unwrap();

        assert_eq!(r2.var_id, r.var_id);
        assert_eq!(r2.kind, r.kind);
        assert_eq!(r2.flags, r.flags);
        assert_eq!(r2.tile_y, r.tile_y);
        assert_eq!(r2.tile_x, r.tile_x);
        assert_eq!(r2.offset, r.offset);
        assert_eq!(r2.len, r.len);
        assert_eq!(r2.raw_len, r.raw_len);
        assert_eq!(r2.center.to_bits(), r.center.to_bits());
        assert_eq!(r2.scale.to_bits(), r.scale.to_bits());
        // NaN fields compared via to_bits
        assert!(r2.min.is_nan(), "min should remain NaN");
        assert_eq!(r2.min.to_bits(), r.min.to_bits());
        assert!(r2.max.is_nan(), "max should remain NaN");
        assert_eq!(r2.max.to_bits(), r.max.to_bits());
        assert_eq!(r2.valid_count, r.valid_count);
    }

    #[test]
    fn index_record_pack_layout_is_exact() {
        let r = ChunkRecord {
            var_id: 0x0102,
            kind: 0x03,
            flags: 0x04,
            tile_y: 0x0506_0708,
            tile_x: 0x090A_0B0C,
            offset: 0x0D0E_0F10_1112_1314,
            len: 0x1516_1718,
            raw_len: 0x191A_1B1C,
            center: 0.0_f32,
            scale: 1.0_f32,
            min: -1.0_f32,
            max: 1.0_f32,
            valid_count: 0x1D1E_1F20,
        };

        let mut buf = Vec::new();
        r.pack_into(&mut buf);

        // var_id at [0..2] little-endian
        assert_eq!(&buf[0..2], &[0x02, 0x01], "var_id LE bytes");
        // kind at [2]
        assert_eq!(buf[2], 0x03, "kind");
        // flags at [3]
        assert_eq!(buf[3], 0x04, "flags");
        // tile_y at [4..8] LE
        assert_eq!(&buf[4..8], &[0x08, 0x07, 0x06, 0x05], "tile_y LE bytes");
        // tile_x at [8..12] LE
        assert_eq!(&buf[8..12], &[0x0C, 0x0B, 0x0A, 0x09], "tile_x LE bytes");
        // offset at [12..20] LE
        assert_eq!(
            &buf[12..20],
            &[0x14, 0x13, 0x12, 0x11, 0x10, 0x0F, 0x0E, 0x0D],
            "offset LE bytes"
        );
        // len at [20..24] LE
        assert_eq!(&buf[20..24], &[0x18, 0x17, 0x16, 0x15], "len LE bytes");
        // raw_len at [24..28] LE
        assert_eq!(&buf[24..28], &[0x1C, 0x1B, 0x1A, 0x19], "raw_len LE bytes");
        // reserved at [48..64] must be zeros
        assert_eq!(&buf[48..64], &[0u8; 16], "reserved zeros");
    }

    #[test]
    fn records_sort_key_orders_by_var_kind_tile() {
        let mut records = [
            ChunkRecord {
                var_id: 2,
                kind: 0,
                tile_y: 0,
                tile_x: 0,
                ..make_record()
            },
            ChunkRecord {
                var_id: 1,
                kind: 1,
                tile_y: 5,
                tile_x: 3,
                ..make_record()
            },
            ChunkRecord {
                var_id: 1,
                kind: 0,
                tile_y: 10,
                tile_x: 2,
                ..make_record()
            },
            ChunkRecord {
                var_id: 1,
                kind: 0,
                tile_y: 10,
                tile_x: 1,
                ..make_record()
            },
            ChunkRecord {
                var_id: 1,
                kind: 0,
                tile_y: 5,
                tile_x: 0,
                ..make_record()
            },
        ];

        records.sort_by_key(|r| r.sort_key());

        let keys: Vec<_> = records.iter().map(|r| r.sort_key()).collect();
        assert_eq!(
            keys,
            vec![
                (1, 0, 5, 0),
                (1, 0, 10, 1),
                (1, 0, 10, 2),
                (1, 1, 5, 3),
                (2, 0, 0, 0),
            ],
            "records should be sorted by (var_id, kind, tile_y, tile_x)"
        );
    }
}
