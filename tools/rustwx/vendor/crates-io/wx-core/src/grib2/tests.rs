#[cfg(test)]
mod tests {
    use crate::grib2::unpack::BitReader;
    use crate::grib2::grid::grid_latlon;
    use crate::grib2::parser::{
        DataRepresentation, GridDefinition, ProductDefinition, Grib2File, Grib2Message,
    };
    use crate::grib2::unpack::unpack_message;

    // ========================================================================
    // BitReader tests
    // ========================================================================

    #[test]
    fn bitreader_read_bits_various_widths() {
        // 0xFF = 1111_1111, 0x00 = 0000_0000, 0xAB = 1010_1011, 0xCD = 1100_1101
        let data = &[0xFF, 0x00, 0xAB, 0xCD];
        let mut reader = BitReader::new(data);

        // Read 1 bit: should be 1 (MSB of 0xFF)
        assert_eq!(reader.read_bits(1), 1);

        // Read 3 bits: should be 0b111 = 7 (next 3 bits of 0xFF)
        assert_eq!(reader.read_bits(3), 0b111);

        // Read 5 bits: should be 0b11110 = 30 (remaining 4 bits of 0xFF + first bit of 0x00)
        assert_eq!(reader.read_bits(5), 0b11110);

        // We've consumed 9 bits so far. Now read 8 bits:
        // bits 9..16 = remaining 7 bits of 0x00 (0000000) + first bit of 0xAB (1)
        // = 0000_0001 = 1
        assert_eq!(reader.read_bits(8), 0b0000_0001);

        // Consumed 17 bits. Read 12 bits:
        // bits 17..28 = remaining 7 bits of 0xAB (010_1011) + first 5 bits of 0xCD (1100_1)
        // = 0101_0111_1001 = 0x579
        assert_eq!(reader.read_bits(12), 0b0101_0111_1001);
    }

    #[test]
    fn bitreader_read_full_bytes() {
        let data = &[0xAB, 0xCD, 0xEF, 0x12];
        let mut reader = BitReader::new(data);

        // Read 16 bits = 0xABCD
        assert_eq!(reader.read_bits(16), 0xABCD);

        // Read 16 bits = 0xEF12
        assert_eq!(reader.read_bits(16), 0xEF12);
    }

    #[test]
    fn bitreader_read_32_bits() {
        let data = &[0x12, 0x34, 0x56, 0x78];
        let mut reader = BitReader::new(data);
        assert_eq!(reader.read_bits(32), 0x12345678);
    }

    #[test]
    fn bitreader_read_zero_bits() {
        let data = &[0xFF];
        let mut reader = BitReader::new(data);
        assert_eq!(reader.read_bits(0), 0);
        // Position should not advance
        assert_eq!(reader.remaining_bits(), 8);
    }

    // ========================================================================
    // BitReader signed tests
    // ========================================================================

    #[test]
    fn bitreader_read_signed_bits_positive() {
        // 8 bits: sign=0, magnitude=0b1010101 = 85
        // 0_1010101 = 0x55
        let data = &[0x55];
        let mut reader = BitReader::new(data);
        assert_eq!(reader.read_signed_bits(8), 85);
    }

    #[test]
    fn bitreader_read_signed_bits_negative() {
        // 8 bits: sign=1, magnitude=0b1010101 = 85 => -85
        // 1_1010101 = 0xD5
        let data = &[0xD5];
        let mut reader = BitReader::new(data);
        assert_eq!(reader.read_signed_bits(8), -85);
    }

    #[test]
    fn bitreader_read_signed_bits_zero() {
        // 8 bits: sign=0, magnitude=0 => 0
        let data = &[0x00];
        let mut reader = BitReader::new(data);
        assert_eq!(reader.read_signed_bits(8), 0);
    }

    #[test]
    fn bitreader_read_signed_bits_negative_zero() {
        // 8 bits: sign=1, magnitude=0 => -0 = 0
        let data = &[0x80];
        let mut reader = BitReader::new(data);
        assert_eq!(reader.read_signed_bits(8), 0);
    }

    #[test]
    fn bitreader_read_signed_bits_1bit() {
        // 1-bit signed: always returns 0 per implementation (only sign bit, no magnitude)
        let data = &[0xFF];
        let mut reader = BitReader::new(data);
        assert_eq!(reader.read_signed_bits(1), 0);
    }

    #[test]
    fn bitreader_read_signed_bits_16bit() {
        // 16 bits: sign=1, magnitude=0x0064 = 100 => -100
        // 1_000000001100100 = 0x8064
        let data = &[0x80, 0x64];
        let mut reader = BitReader::new(data);
        assert_eq!(reader.read_signed_bits(16), -100);
    }

    // ========================================================================
    // BitReader alignment and remaining bits
    // ========================================================================

    #[test]
    fn bitreader_align_to_byte() {
        let data = &[0xFF, 0xAA];
        let mut reader = BitReader::new(data);

        // Read 3 bits, then align
        reader.read_bits(3);
        assert_eq!(reader.remaining_bits(), 13);

        reader.align_to_byte();
        assert_eq!(reader.remaining_bits(), 8); // aligned to byte 1

        // Next read should start at byte index 1
        assert_eq!(reader.read_bits(8), 0xAA);
    }

    #[test]
    fn bitreader_align_to_byte_already_aligned() {
        let data = &[0xFF, 0xAA];
        let mut reader = BitReader::new(data);

        // Read exactly 8 bits (already aligned)
        reader.read_bits(8);
        reader.align_to_byte(); // should be a no-op
        assert_eq!(reader.remaining_bits(), 8);
        assert_eq!(reader.read_bits(8), 0xAA);
    }

    #[test]
    fn bitreader_remaining_bits() {
        let data = &[0xFF, 0x00, 0xAB]; // 24 bits total
        let mut reader = BitReader::new(data);

        assert_eq!(reader.remaining_bits(), 24);

        reader.read_bits(5);
        assert_eq!(reader.remaining_bits(), 19);

        reader.read_bits(19);
        assert_eq!(reader.remaining_bits(), 0);
    }

    #[test]
    fn bitreader_remaining_bits_empty() {
        let data: &[u8] = &[];
        let reader = BitReader::new(data);
        assert_eq!(reader.remaining_bits(), 0);
    }

    // ========================================================================
    // Grid tests
    // ========================================================================

    #[test]
    fn grid_latlon_regular_3x3() {
        // 3x3 regular lat/lon grid from (0,0) to (2,2)
        let grid = GridDefinition {
            template: 0,
            nx: 3,
            ny: 3,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 2.0,
            lon2: 2.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let (lats, lons) = grid_latlon(&grid);
        assert_eq!(lats.len(), 9);
        assert_eq!(lons.len(), 9);

        // Row 0 (j=0): lat=0
        assert!((lats[0] - 0.0).abs() < 1e-10);
        assert!((lons[0] - 0.0).abs() < 1e-10);

        assert!((lats[1] - 0.0).abs() < 1e-10);
        assert!((lons[1] - 1.0).abs() < 1e-10);

        assert!((lats[2] - 0.0).abs() < 1e-10);
        assert!((lons[2] - 2.0).abs() < 1e-10);

        // Row 1 (j=1): lat=1
        assert!((lats[3] - 1.0).abs() < 1e-10);
        assert!((lons[3] - 0.0).abs() < 1e-10);

        assert!((lats[4] - 1.0).abs() < 1e-10);
        assert!((lons[4] - 1.0).abs() < 1e-10);

        // Row 2 (j=2): lat=2
        assert!((lats[8] - 2.0).abs() < 1e-10);
        assert!((lons[8] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn grid_latlon_lambert_hrrr_like() {
        // HRRR-like Lambert Conformal grid, 5x5 subset
        let grid = GridDefinition {
            template: 30,
            nx: 5,
            ny: 5,
            lat1: 21.138123,
            lon1: 237.280472,
            lat2: 0.0,  // not used for Lambert
            lon2: 0.0,
            dx: 3000.0,
            dy: 3000.0,
            latin1: 38.5,
            latin2: 38.5,
            lov: 262.5,
            scan_mode: 0x40,
            ..GridDefinition::default()
        };

        let (lats, lons) = grid_latlon(&grid);
        assert_eq!(lats.len(), 25);
        assert_eq!(lons.len(), 25);

        // First point should match the specified lat1/lon1
        assert!(
            (lats[0] - 21.138123).abs() < 0.01,
            "First point lat: expected ~21.138123, got {}",
            lats[0]
        );
        assert!(
            (lons[0] - 237.280472).abs() < 0.01,
            "First point lon: expected ~237.280472, got {}",
            lons[0]
        );

        // Center point (2,2) = index 12 should be reasonable (not at poles)
        let center_lat = lats[12];
        let center_lon = lons[12];
        assert!(
            center_lat > 20.0 && center_lat < 60.0,
            "Center lat should be reasonable CONUS value, got {}",
            center_lat
        );
        assert!(
            center_lon > 200.0 && center_lon < 300.0,
            "Center lon should be reasonable CONUS value, got {}",
            center_lon
        );

        // All latitudes should be in a reasonable range
        for (i, &lat) in lats.iter().enumerate() {
            assert!(
                lat > 10.0 && lat < 70.0,
                "Lat[{}] = {} out of reasonable range",
                i, lat
            );
        }
    }

    // ========================================================================
    // Simple packing (Template 0) test
    // ========================================================================

    #[test]
    fn unpack_simple_packing_template0() {
        // Test the GRIB2 simple packing formula: Y = (R + X * 2^E) * 10^(-D)
        // With R=273.15, E=0, D=0: Y = 273.15 + X
        // Pack 9 values: 0, 1, 2, 3, 4, 5, 6, 7, 8 as 16-bit unsigned integers
        // Expected output: 273.15, 274.15, 275.15, ...

        let reference_value: f32 = 273.15;
        let binary_scale: i16 = 0;
        let decimal_scale: i16 = 0;
        let bits_per_value: u8 = 16;

        // Pack 9 values as big-endian 16-bit integers
        let packed_values: Vec<u16> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut raw_data = Vec::new();
        for v in &packed_values {
            raw_data.extend_from_slice(&v.to_be_bytes());
        }

        let msg = Grib2Message {
            discipline: 0,
            reference_time: chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                template: 0,
                nx: 3,
                ny: 3,
                lat1: 0.0,
                lon1: 0.0,
                lat2: 2.0,
                lon2: 2.0,
                dx: 1.0,
                dy: 1.0,
                scan_mode: 0,
                ..GridDefinition::default()
            },
            product: ProductDefinition::default(),
            data_rep: DataRepresentation {
                template: 0,
                reference_value,
                binary_scale,
                decimal_scale,
                bits_per_value,
                ..DataRepresentation::default()
            },
            bitmap: None,
            raw_data,
        };

        let values = unpack_message(&msg).unwrap();
        assert_eq!(values.len(), 9);

        for (i, &val) in values.iter().enumerate() {
            let expected = reference_value as f64 + i as f64;
            assert!(
                (val - expected).abs() < 1e-4,
                "values[{}]: expected {}, got {}",
                i, expected, val
            );
        }
    }

    #[test]
    fn unpack_simple_packing_with_scales() {
        // Test with non-zero scales: R=0.0, E=1 (binary_scale=1), D=1 (decimal_scale=1)
        // Formula: Y = (R + X * 2^E) * 10^(-D)
        // Y = (0 + X * 2) * 0.1 = X * 0.2
        // Pack values [0, 5, 10, 15] as 8-bit
        // Expected: [0.0, 1.0, 2.0, 3.0]

        let raw_data: Vec<u8> = vec![0, 5, 10, 15];

        let msg = Grib2Message {
            discipline: 0,
            reference_time: chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                template: 0,
                nx: 2,
                ny: 2,
                lat1: 0.0,
                lon1: 0.0,
                lat2: 1.0,
                lon2: 1.0,
                dx: 1.0,
                dy: 1.0,
                scan_mode: 0,
                ..GridDefinition::default()
            },
            product: ProductDefinition::default(),
            data_rep: DataRepresentation {
                template: 0,
                reference_value: 0.0,
                binary_scale: 1,
                decimal_scale: 1,
                bits_per_value: 8,
                ..DataRepresentation::default()
            },
            bitmap: None,
            raw_data,
        };

        let values = unpack_message(&msg).unwrap();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[1] - 1.0).abs() < 1e-10);
        assert!((values[2] - 2.0).abs() < 1e-10);
        assert!((values[3] - 3.0).abs() < 1e-10);
    }

    // ========================================================================
    // Parser edge case tests
    // ========================================================================

    #[test]
    fn parser_empty_data_returns_empty_file() {
        // Empty data should return a Grib2File with no messages (not an error),
        // because from_bytes just finds no GRIB magic and returns empty vec.
        let result = Grib2File::from_bytes(&[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().messages.len(), 0);
    }

    #[test]
    fn parser_truncated_grib_header() {
        // Just "GRIB" + a few bytes, but not enough for a full 16-byte indicator section
        let data = b"GRIB\x00\x00\x02";
        let result = Grib2File::from_bytes(data);
        // Should either error or return empty (since pos + 16 > len check)
        // The while loop condition `pos + 16 <= data.len()` will be false,
        // so it returns Ok with empty messages
        assert!(result.is_ok());
        assert_eq!(result.unwrap().messages.len(), 0);
    }

    #[test]
    fn parser_invalid_magic_returns_empty() {
        // Data that doesn't contain "GRIB" magic
        let data = b"THIS_IS_NOT_A_VALID_DATA_FILE_XYZ";
        let result = Grib2File::from_bytes(data);
        // find_magic returns None, so we get Ok with empty messages
        assert!(result.is_ok());
        assert_eq!(result.unwrap().messages.len(), 0);
    }

    #[test]
    fn parser_grib_magic_wrong_edition() {
        // Valid GRIB magic but edition=1 (not 2)
        // Section 0 is 16 bytes: "GRIB" + 2 reserved + discipline + edition + 8-byte length
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"GRIB");
        data[6] = 0;   // discipline
        data[7] = 1;   // edition = 1 (unsupported)
        // total length = 32 (as u64 big-endian at offset 8)
        data[15] = 32;

        let result = Grib2File::from_bytes(&data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Unsupported GRIB edition"),
            "Expected edition error, got: {}",
            err
        );
    }

    #[test]
    fn parser_grib_message_extends_beyond_data() {
        // Valid GRIB2 header but total_length points beyond actual data
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(b"GRIB");
        data[6] = 0;    // discipline
        data[7] = 2;    // edition = 2
        // total_length = 1000 (way beyond our 20 bytes)
        data[8..16].copy_from_slice(&1000u64.to_be_bytes());

        let result = Grib2File::from_bytes(&data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("extends beyond"),
            "Expected 'extends beyond' error, got: {}",
            err
        );
    }

    // ========================================================================
    // Integration test: download real HRRR data (ignored by default)
    // ========================================================================

    #[cfg(feature = "network")]
    #[test]
    #[ignore]
    fn integration_download_and_parse_hrrr_tmp_2m() {
        use crate::download::{DownloadClient, parse_idx, find_entries, byte_ranges};
        use crate::models::hrrr::HrrrConfig;

        // Use a known historical date that should be available on AWS
        let date = "20240115";
        let hour = 12;
        let fhour = 0;

        let client = DownloadClient::new().expect("Failed to create download client");

        // Get the idx file
        let idx_url = HrrrConfig::idx_url(date, hour, "sfc", fhour);
        let idx_text = client
            .get_text(&idx_url)
            .expect("Failed to download idx file");

        let entries = parse_idx(&idx_text);
        assert!(!entries.is_empty(), "IDX file should have entries");

        // Find TMP:2 m above ground
        let pattern = HrrrConfig::sfc_temp_2m();
        let selected = find_entries(&entries, pattern);
        assert!(
            !selected.is_empty(),
            "Should find TMP:2 m above ground in idx"
        );

        let ranges = byte_ranges(&entries, &selected);

        // Download just the TMP:2m message
        let grib_url = HrrrConfig::aws_url(date, hour, "sfc", fhour);
        let data = client
            .get_ranges(&grib_url, &ranges)
            .expect("Failed to download GRIB2 data");

        // Parse
        let grib = Grib2File::from_bytes(&data).expect("Failed to parse GRIB2");
        assert!(
            !grib.messages.is_empty(),
            "Should have at least one message"
        );

        let msg = &grib.messages[0];
        let nx = msg.grid.nx as usize;
        let ny = msg.grid.ny as usize;

        // Unpack
        let values = unpack_message(msg).expect("Failed to unpack message");

        // Verify dimensions
        assert_eq!(
            values.len(),
            nx * ny,
            "values.len() ({}) should equal nx*ny ({}x{} = {})",
            values.len(),
            nx,
            ny,
            nx * ny
        );

        // All values should be finite
        for (i, &v) in values.iter().enumerate() {
            assert!(
                v.is_finite(),
                "values[{}] is not finite: {}",
                i, v
            );
        }

        // Temperature should be in a physically reasonable range (200-350 K)
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            min_val >= 200.0 && max_val <= 350.0,
            "Temperature range [{}, {}] K is outside reasonable bounds [200, 350]",
            min_val,
            max_val
        );

        eprintln!(
            "HRRR TMP:2m — grid {}x{}, values: {}, range: [{:.1}, {:.1}] K",
            nx, ny, values.len(), min_val, max_val
        );
    }

    // ========================================================================
    // Template 5.200: Run Length Encoding (RLE) tests
    // ========================================================================

    #[test]
    fn unpack_rle_basic() {
        // RLE with 8-bit values: (value=1, count=3), (value=2, count=2), (value=0, count=4)
        // Should produce: [1, 1, 1, 2, 2, 0, 0, 0, 0]
        let raw_data: Vec<u8> = vec![
            1, 3,   // value=1, count=3
            2, 2,   // value=2, count=2
            0, 4,   // value=0, count=4
        ];

        let msg = Grib2Message {
            discipline: 0,
            reference_time: chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                template: 0,
                nx: 3,
                ny: 3,
                lat1: 0.0,
                lon1: 0.0,
                lat2: 2.0,
                lon2: 2.0,
                dx: 1.0,
                dy: 1.0,
                scan_mode: 0,
                ..GridDefinition::default()
            },
            product: ProductDefinition::default(),
            data_rep: DataRepresentation {
                template: 200,
                reference_value: 0.0,
                binary_scale: 0,
                decimal_scale: 0,
                bits_per_value: 8,
                ..DataRepresentation::default()
            },
            bitmap: None,
            raw_data,
        };

        let values = unpack_message(&msg).unwrap();
        assert_eq!(values.len(), 9);
        // First 3 values should be 1.0
        assert!((values[0] - 1.0).abs() < 1e-10);
        assert!((values[1] - 1.0).abs() < 1e-10);
        assert!((values[2] - 1.0).abs() < 1e-10);
        // Next 2 should be 2.0
        assert!((values[3] - 2.0).abs() < 1e-10);
        assert!((values[4] - 2.0).abs() < 1e-10);
        // Last 4 should be 0.0
        assert!((values[5] - 0.0).abs() < 1e-10);
        assert!((values[6] - 0.0).abs() < 1e-10);
        assert!((values[7] - 0.0).abs() < 1e-10);
        assert!((values[8] - 0.0).abs() < 1e-10);
    }

    // ========================================================================
    // Template 5.4: IEEE Floating Point tests
    // ========================================================================

    #[test]
    fn unpack_ieee_f32() {
        // Pack three f32 values as big-endian bytes
        let v1: f32 = 273.15;
        let v2: f32 = 300.0;
        let v3: f32 = -10.5;
        let v4: f32 = 0.0;

        let mut raw_data = Vec::new();
        raw_data.extend_from_slice(&v1.to_be_bytes());
        raw_data.extend_from_slice(&v2.to_be_bytes());
        raw_data.extend_from_slice(&v3.to_be_bytes());
        raw_data.extend_from_slice(&v4.to_be_bytes());

        let msg = Grib2Message {
            discipline: 0,
            reference_time: chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                template: 0,
                nx: 2,
                ny: 2,
                lat1: 0.0,
                lon1: 0.0,
                lat2: 1.0,
                lon2: 1.0,
                dx: 1.0,
                dy: 1.0,
                scan_mode: 0,
                ..GridDefinition::default()
            },
            product: ProductDefinition::default(),
            data_rep: DataRepresentation {
                template: 4,
                bits_per_value: 32,
                ..DataRepresentation::default()
            },
            bitmap: None,
            raw_data,
        };

        let values = unpack_message(&msg).unwrap();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 273.15).abs() < 0.01);
        assert!((values[1] - 300.0).abs() < 0.01);
        assert!((values[2] - (-10.5)).abs() < 0.01);
        assert!((values[3] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn unpack_ieee_f64() {
        let v1: f64 = 273.15;
        let v2: f64 = 1.0e-15;
        let v3: f64 = -999.999;
        let v4: f64 = std::f64::consts::PI;

        let mut raw_data = Vec::new();
        raw_data.extend_from_slice(&v1.to_be_bytes());
        raw_data.extend_from_slice(&v2.to_be_bytes());
        raw_data.extend_from_slice(&v3.to_be_bytes());
        raw_data.extend_from_slice(&v4.to_be_bytes());

        let msg = Grib2Message {
            discipline: 0,
            reference_time: chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                template: 0,
                nx: 2,
                ny: 2,
                lat1: 0.0,
                lon1: 0.0,
                lat2: 1.0,
                lon2: 1.0,
                dx: 1.0,
                dy: 1.0,
                scan_mode: 0,
                ..GridDefinition::default()
            },
            product: ProductDefinition::default(),
            data_rep: DataRepresentation {
                template: 4,
                bits_per_value: 64,
                ..DataRepresentation::default()
            },
            bitmap: None,
            raw_data,
        };

        let values = unpack_message(&msg).unwrap();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 273.15).abs() < 1e-12);
        assert!((values[1] - 1.0e-15).abs() < 1e-25);
        assert!((values[2] - (-999.999)).abs() < 1e-10);
        assert!((values[3] - std::f64::consts::PI).abs() < 1e-15);
    }

    // ========================================================================
    // Template 5.61: Simple packing with log pre-processing
    // ========================================================================

    #[test]
    fn unpack_simple_log_round_trip() {
        // The log pre-processing stores log10(value + 1).
        // We create values where the simple-packed result represents log10(value + 1).
        // With R=0, E=0, D=0, X maps to X directly.
        // So the simple unpacked value is X, then the log transform gives 10^X - 1.
        //
        // Let's pack X=0,1,2 as 8-bit values:
        //   X=0 -> 10^0 - 1 = 0.0
        //   X=1 -> 10^1 - 1 = 9.0
        //   X=2 -> 10^2 - 1 = 99.0
        //   X=3 -> 10^3 - 1 = 999.0

        let raw_data: Vec<u8> = vec![0, 1, 2, 3];

        let msg = Grib2Message {
            discipline: 0,
            reference_time: chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                template: 0,
                nx: 2,
                ny: 2,
                lat1: 0.0,
                lon1: 0.0,
                lat2: 1.0,
                lon2: 1.0,
                dx: 1.0,
                dy: 1.0,
                scan_mode: 0,
                ..GridDefinition::default()
            },
            product: ProductDefinition::default(),
            data_rep: DataRepresentation {
                template: 61,
                reference_value: 0.0,
                binary_scale: 0,
                decimal_scale: 0,
                bits_per_value: 8,
                ..DataRepresentation::default()
            },
            bitmap: None,
            raw_data,
        };

        let values = unpack_message(&msg).unwrap();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 0.0).abs() < 1e-10, "expected 0.0, got {}", values[0]);
        assert!((values[1] - 9.0).abs() < 1e-10, "expected 9.0, got {}", values[1]);
        assert!((values[2] - 99.0).abs() < 1e-8, "expected 99.0, got {}", values[2]);
        assert!((values[3] - 999.0).abs() < 1e-6, "expected 999.0, got {}", values[3]);
    }

    // ========================================================================
    // Template 3.1: Rotated grid coordinate transform tests
    // ========================================================================

    #[test]
    fn rotated_grid_identity_transform() {
        use crate::grib2::grid::rotated_to_geographic;

        // With south pole at (-90, 0) and no rotation, the rotated grid
        // is identical to the regular grid (no transformation).
        let (lat, lon) = rotated_to_geographic(45.0, 10.0, -90.0, 0.0, 0.0);
        assert!(
            (lat - 45.0).abs() < 0.01,
            "Expected lat ~45.0, got {}",
            lat
        );
        assert!(
            (lon - 10.0).abs() < 0.01,
            "Expected lon ~10.0, got {}",
            lon
        );
    }

    #[test]
    fn rotated_grid_dwd_like() {
        use crate::grib2::grid::rotated_to_geographic;

        // DWD ICON uses south_pole_lat = -40.0, south_pole_lon = -170.0
        // The rotated equator (rot_lat=0, rot_lon=0) maps to the point
        // on the great circle 90 degrees from the rotated south pole.
        // With alpha = 40 deg (north pole lat), the rotated equator at rlon=0
        // maps to lat = 90 - alpha = 50 deg, lon = sp_lon = -170 deg.
        let (lat, lon) = rotated_to_geographic(0.0, 0.0, -40.0, -170.0, 0.0);
        assert!(
            (lat - 50.0).abs() < 1.0,
            "Expected lat ~50.0, got {}",
            lat
        );
        assert!(
            (lon - (-170.0)).abs() < 1.0,
            "Expected lon ~-170.0, got {}",
            lon
        );
    }

    #[test]
    fn rotated_grid_latlon_generation() {
        // Test that grid_latlon works for a small rotated grid
        let grid = GridDefinition {
            template: 1,
            nx: 3,
            ny: 3,
            lat1: -1.0,
            lon1: -1.0,
            lat2: 1.0,
            lon2: 1.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            south_pole_lat: -90.0,   // identity rotation
            south_pole_lon: 0.0,
            rotation_angle: 0.0,
            ..GridDefinition::default()
        };

        let (lats, lons) = grid_latlon(&grid);
        assert_eq!(lats.len(), 9);
        assert_eq!(lons.len(), 9);
        // With identity rotation, center point should be near (0, 0)
        // (after normalization, lon may be 180 due to +PI offset)
        // All lats should be in [-90, 90]
        for &lat in &lats {
            assert!(
                lat >= -90.0 && lat <= 90.0,
                "Lat {} out of range",
                lat
            );
        }
    }

    // ========================================================================
    // Template 5.42: CCSDS stub test
    // ========================================================================

    #[test]
    fn unpack_ccsds_returns_clear_error() {
        let msg = Grib2Message {
            discipline: 0,
            reference_time: chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                template: 0,
                nx: 2,
                ny: 2,
                ..GridDefinition::default()
            },
            product: ProductDefinition::default(),
            data_rep: DataRepresentation {
                template: 42,
                bits_per_value: 16,
                ccsds_flags: 0x00A0,
                ccsds_block_size: 16,
                ccsds_rsi: 128,
                ..DataRepresentation::default()
            },
            bitmap: None,
            raw_data: vec![0u8; 100],
        };

        let result = unpack_message(&msg);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("CCSDS") || err.contains("AEC"),
            "Error should mention CCSDS/AEC, got: {}",
            err
        );
    }
}
