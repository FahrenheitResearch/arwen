//! Integration tests for the GEMPAK grid reader.

use metrust::io::gempak::GempakGrid;

fn push_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_be_bytes());
}

fn push_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_be_bytes());
}

/// Build a minimal synthetic GEMPAK grid file in memory.
fn build_test_file(kx: usize, ky: usize, value: f32) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();

    // 0..28: GEMPAK header
    out.extend_from_slice(b"GEMPAK DATA MANAGEMENT FILE ");

    // Layout
    let pd_start_word: usize = 8;
    let file_keys_ptr = pd_start_word + 24; // PD = 20 i32 + 12 skip + 1 f32 = 24 words
    let file_keys_size = 3;
    let navb_ptr = file_keys_ptr + file_keys_size;
    let navb_size_word = 1;
    let navb_content = 256;
    let anlb_ptr = navb_ptr + navb_size_word + navb_content;
    let anlb_size_word = 1;
    let row_keys_ptr = anlb_ptr + anlb_size_word;
    let col_keys_count = 10;
    let col_keys_ptr = row_keys_ptr;
    let parts_ptr = col_keys_ptr + col_keys_count;
    let parts_count = 1;
    let parts_total = parts_count * 4;
    let row_headers_ptr = parts_ptr + parts_total;
    let rows = 1;
    let row_header_size = 1;
    let col_headers_ptr = row_headers_ptr + rows * row_header_size;
    let columns = 1;
    let col_header_size = 1 + col_keys_count;
    let data_mgmt_ptr = col_headers_ptr + columns * col_header_size;
    let data_mgmt_size = 32;
    let data_block_ptr = data_mgmt_ptr + data_mgmt_size;
    let data_block_entries = rows * columns * parts_count;
    let data_start = data_block_ptr + data_block_entries;
    let grid_data_length = (kx * ky) as i32;
    let data_content_length = 1 + 1 + grid_data_length;

    // Product description
    push_i32(&mut out, 1); // version
    push_i32(&mut out, 1); // file_headers
    push_i32(&mut out, file_keys_ptr as i32);
    push_i32(&mut out, rows as i32);
    push_i32(&mut out, 0); // row_keys count
    push_i32(&mut out, row_keys_ptr as i32);
    push_i32(&mut out, row_headers_ptr as i32);
    push_i32(&mut out, columns as i32);
    push_i32(&mut out, col_keys_count as i32);
    push_i32(&mut out, col_keys_ptr as i32);
    push_i32(&mut out, col_headers_ptr as i32);
    push_i32(&mut out, parts_count as i32);
    push_i32(&mut out, parts_ptr as i32);
    push_i32(&mut out, data_mgmt_ptr as i32);
    push_i32(&mut out, data_mgmt_size as i32);
    push_i32(&mut out, data_block_ptr as i32);
    push_i32(&mut out, 3); // file_type = grid
    push_i32(&mut out, 0); // data_source = model
    push_i32(&mut out, 0); // machine_type
    push_i32(&mut out, -9999); // missing_int
    out.extend_from_slice(&[0u8; 12]); // padding
    push_f32(&mut out, -9999.0); // missing_float

    // File keys
    out.extend_from_slice(b"NAVB");
    push_i32(&mut out, 256);
    push_i32(&mut out, 1);

    // Navigation block
    push_i32(&mut out, 256);
    push_f32(&mut out, 0.0);
    out.extend_from_slice(b"CED\0");
    push_f32(&mut out, 1.0);
    push_f32(&mut out, 1.0);
    push_f32(&mut out, kx as f32);
    push_f32(&mut out, ky as f32);
    push_f32(&mut out, 20.0);
    push_f32(&mut out, -120.0);
    push_f32(&mut out, 50.0);
    push_f32(&mut out, -60.0);
    push_f32(&mut out, 0.0);
    push_f32(&mut out, -90.0);
    push_f32(&mut out, 0.0);
    let nav_written = 12 * 4 + 4;
    out.extend_from_slice(&vec![0u8; 256 * 4 - nav_written]);

    // Analysis block (none)
    push_i32(&mut out, 0);

    // Column keys
    for name in &["GDT1", "GTM1", "GDT2", "GTM2", "GLV1", "GLV2", "GVCD", "GPM1", "GPM2", "GPM3"] {
        let mut key_bytes = [b' '; 4];
        for (i, b) in name.bytes().enumerate() {
            if i < 4 { key_bytes[i] = b; }
        }
        out.extend_from_slice(&key_bytes);
    }

    // Parts
    out.extend_from_slice(b"GRID");
    push_i32(&mut out, 1);
    push_i32(&mut out, 5);
    push_i32(&mut out, 0);

    // Row headers
    push_i32(&mut out, 9999);

    // Column headers
    push_i32(&mut out, 9999);
    push_i32(&mut out, 250101); // GDT1
    push_i32(&mut out, 0);      // GTM1
    push_i32(&mut out, 0);      // GDT2
    push_i32(&mut out, 0);      // GTM2
    push_i32(&mut out, 500);    // GLV1
    push_i32(&mut out, -1);     // GLV2
    push_i32(&mut out, 1);      // GVCD = PRES
    out.extend_from_slice(b"HGHT");
    out.extend_from_slice(b"    ");
    out.extend_from_slice(b"    ");

    // Data management
    for _ in 0..32 { push_i32(&mut out, 0); }

    // Data block pointer
    push_i32(&mut out, data_start as i32);

    // Data content
    push_i32(&mut out, data_content_length);
    push_i32(&mut out, 0);
    push_i32(&mut out, 0); // packing = none
    for _ in 0..(kx * ky) {
        push_f32(&mut out, value);
    }

    out
}

#[test]
fn test_header_validation() {
    let bad = vec![0u8; 100];
    assert!(GempakGrid::from_bytes(&bad).is_err());
}

#[test]
fn test_non_grid_file() {
    let mut data = build_test_file(3, 3, 1.0);
    let ft_offset = 28 + 16 * 4;
    let ft_bytes = 1i32.to_be_bytes();
    data[ft_offset..ft_offset + 4].copy_from_slice(&ft_bytes);
    let result = GempakGrid::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_debug_structure() {
    let data = build_test_file(4, 3, 5500.0);
    let gem = GempakGrid::from_bytes(&data).expect("parse failed");
    let info = gem.grid_info();
    assert!(gem.nx > 0);
    assert!(gem.ny > 0);
    assert_eq!(info.len(), 1);
}

#[test]
fn test_parse_synthetic_grid() {
    let data = build_test_file(4, 3, 5500.0);
    let gem = GempakGrid::from_bytes(&data).expect("parse failed");

    assert_eq!(gem.nx, 4);
    assert_eq!(gem.ny, 3);
    assert_eq!(gem.grid_type, "grid");
    assert_eq!(gem.source, "model");
    assert_eq!(gem.grids.len(), 1);

    let g = &gem.grids[0];
    assert_eq!(g.parameter, "HGHT");
    assert!((g.level - 500.0).abs() < 0.01);
    assert_eq!(g.coordinate, "PRES");
    assert_eq!(g.data.len(), 12);

    for v in &g.data {
        assert!((*v - 5500.0).abs() < 0.01, "expected 5500.0, got {}", v);
    }
}

#[test]
fn test_navigation_info() {
    let data = build_test_file(10, 10, 0.0);
    let gem = GempakGrid::from_bytes(&data).unwrap();

    let nav = gem.navigation.as_ref().expect("nav should be present");
    assert_eq!(nav.projection, "CED");
    assert!((nav.lower_left_lat - 20.0).abs() < 0.01);
    assert!((nav.lower_left_lon - (-120.0)).abs() < 0.01);
    assert!((nav.upper_right_lat - 50.0).abs() < 0.01);
    assert!((nav.upper_right_lon - (-60.0)).abs() < 0.01);
}

#[test]
fn test_grid_info() {
    let data = build_test_file(5, 5, 100.0);
    let gem = GempakGrid::from_bytes(&data).unwrap();
    let info = gem.grid_info();
    assert_eq!(info.len(), 1);
    assert!(info[0].contains("HGHT"));
    assert!(info[0].contains("500"));
}

#[test]
fn test_find_grids() {
    let data = build_test_file(5, 5, 100.0);
    let gem = GempakGrid::from_bytes(&data).unwrap();
    assert_eq!(gem.find_grids("HGHT").len(), 1);
    assert_eq!(gem.find_grids("TMPK").len(), 0);
    assert_eq!(gem.find_grids("hght").len(), 1);
}

#[test]
fn test_get_grid() {
    let data = build_test_file(5, 5, 273.15);
    let gem = GempakGrid::from_bytes(&data).unwrap();
    let g = gem.get_grid("HGHT", 500.0).expect("should find grid");
    assert_eq!(g.data.len(), 25);
    assert!(gem.get_grid("HGHT", 850.0).is_none());
}
