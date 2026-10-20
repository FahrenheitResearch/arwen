//! The GeoTIFF reader decodes the three real inputs byte-for-byte as GDAL does
//! (deflate + floating-point predictor for Copernicus GLO-90, deflate tiles for
//! WorldCover, uncompressed int16 for the SoilGrids WCS). Reference SHA-256 from
//! rasterio/GDAL 3.10 on the lab data host. Needs the fetched inputs: run with --include-ignored
//! and HYDRO_WORLD_DATA pointing at the fetched hydro-world inputs.
use hydro_world_params::tiff::{sample_f32, GeoTiff};
use sha2::{Digest, Sha256};

fn data() -> String {
    std::env::var("HYDRO_WORLD_DATA").unwrap_or_else(|_| "hydro-world-data".into())
}

#[test]
#[ignore]
fn copernicus_glo90_float_predictor_matches_gdal() {
    let t = GeoTiff::open(&format!("{}/copernicus-dem/glo90/Copernicus_DSM_COG_30_N50_00_E007_00_DEM.tif", data())).unwrap();
    assert_eq!((t.width, t.height), (800, 1200));
    assert!((t.origin_x - 6.999375).abs() < 1e-9 && (t.origin_y - 51.000416666666666).abs() < 1e-9);
    let v = t.read_all_f32().unwrap();
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    assert_eq!(format!("{:x}", Sha256::digest(&b)), "bcc22ffd2dc64b427a91ca9a1819b07f179a49538859c77af0742a38df023ad0");
}

#[test]
#[ignore]
fn worldcover_window_matches_gdal() {
    let t = GeoTiff::open(&format!("{}/worldcover/v200-2021/ESA_WorldCover_10m_2021_v200_N48E006_Map.tif", data())).unwrap();
    let (r0, c0, n) = (5000usize, 9000usize, 2048usize);
    let mut w = vec![0u8; n * n];
    for b in 0..t.n_blocks() {
        let (x0, y0, bw, bh, row_len) = t.block_rect(b);
        if x0 + bw <= c0 || x0 >= c0 + n || y0 + bh <= r0 || y0 >= r0 + n {
            continue;
        }
        let d = t.decode_block(b).unwrap();
        for r in 0..bh {
            for c in 0..bw {
                let (y, x) = (y0 + r, x0 + c);
                if y >= r0 && y < r0 + n && x >= c0 && x < c0 + n {
                    w[(y - r0) * n + (x - c0)] = sample_f32(t.kind, &d, r * row_len + c) as u8;
                }
            }
        }
    }
    assert_eq!(format!("{:x}", Sha256::digest(&w)), "3d1c8b05b5e8a462cc99f0f7ce1a2995b9751f4abe856546df96a983fe57ca92");
}

#[test]
#[ignore]
fn soilgrids_wcs_int16_matches_gdal() {
    let t = GeoTiff::open(&format!("{}/soilgrids/wcs-4326-0p0025/rhine_clay_15-30cm_mean.tif", data())).unwrap();
    assert_eq!((t.width, t.height), (3400, 3200));
    let v = t.read_all_f32().unwrap();
    let b: Vec<u8> = v.iter().flat_map(|x| (*x as i16).to_le_bytes()).collect();
    assert_eq!(format!("{:x}", Sha256::digest(&b)), "a3a1d72df501a2d52aa80aed30eacc3cf33e56bc13ea8acbeaadb8f0d968a3f3");
}
