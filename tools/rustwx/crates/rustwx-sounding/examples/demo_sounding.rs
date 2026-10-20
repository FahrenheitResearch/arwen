use rustwx_sounding::{SoundingColumn, SoundingMetadata, write_full_sounding_png};
use std::path::PathBuf;

fn main() {
    let column = SoundingColumn {
        pressure_hpa: vec![
            1000.0, 975.0, 950.0, 925.0, 900.0, 875.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0,
            550.0, 500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0,
        ],
        height_m_msl: vec![
            240.0, 430.0, 620.0, 820.0, 1040.0, 1290.0, 1560.0, 2140.0, 2740.0, 3370.0, 4040.0,
            4750.0, 5510.0, 6340.0, 7240.0, 8230.0, 9340.0, 10610.0, 12110.0, 13940.0,
        ],
        temperature_c: vec![
            25.0, 23.6, 22.1, 20.4, 18.2, 15.9, 13.3, 9.2, 5.1, 0.0, -4.7, -9.5, -15.0, -21.5,
            -28.9, -36.2, -43.6, -51.8, -57.8, -60.5,
        ],
        dewpoint_c: vec![
            18.0, 17.1, 15.8, 14.2, 12.0, 9.8, 7.2, 2.1, -2.8, -7.0, -11.8, -17.0, -22.4, -29.2,
            -35.8, -42.1, -48.3, -55.4, -61.0, -66.0,
        ],
        u_ms: vec![
            4.0, 5.0, 5.5, 6.0, 7.0, 8.5, 10.0, 12.0, 14.0, 17.0, 20.0, 23.0, 27.0, 31.0, 35.0,
            39.0, 43.0, 47.0, 51.0, 55.0,
        ],
        v_ms: vec![
            2.0, 2.4, 3.0, 3.8, 4.8, 6.0, 7.4, 10.0, 12.0, 14.0, 15.8, 18.0, 20.0, 22.0, 24.5,
            27.0, 30.0, 33.0, 36.0, 39.0,
        ],
        omega_pa_s: Vec::new(),
        metadata: SoundingMetadata {
            station_id: "RUSTWX".to_string(),
            valid_time: "2026-04-14T20:00:00Z".to_string(),
            latitude_deg: Some(35.4),
            longitude_deg: Some(-97.6),
            elevation_m: Some(360.0),
            sample_method: Some("nearest".to_string()),
            box_radius_lat_deg: None,
            box_radius_lon_deg: None,
        },
    };

    let proof_dir = workspace_proof_dir();
    std::fs::create_dir_all(&proof_dir).expect("proof dir");
    let output = proof_dir.join("rustwx_sounding_demo.png");
    write_full_sounding_png(&column, &output).expect("render sounding");
    println!("{}", output.display());
}

fn workspace_proof_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("proof")
}
