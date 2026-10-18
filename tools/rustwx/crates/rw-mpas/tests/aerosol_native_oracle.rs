//! Replay the complete pristine MPAS module's horizontal, initial and LBC output.
//! Native harness: tests/aerosol_oracle; runner: hex/tree/tests/test_init_door.py::test_complete_native_aerosol_module.
use rw_mpas::init::aerosols::{self, Climatology};
use std::io::Write;
use std::path::Path;

/// Read real columns with the rustwx decoder, preserving stored binary32 words.
/// The reference driver reconstructs pressure_base with the pristine case7 slice
/// (mpas_init_atm_cases.F:5274-5280) and checks its rho_base against the init file.
#[test]
fn prepare_real_columns() {
    let Some(manifest) = std::env::var_os("MPAS_AEROSOL_REAL_MANIFEST") else {
        return;
    };
    let rows: serde_json::Value =
        serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
    let rows = rows.as_array().unwrap();
    assert!(!rows.is_empty());
    let mut summary = Vec::new();
    for row in rows {
        let file = netcrust::File::open(row["source"].as_str().unwrap()).unwrap();
        let lat = file.read_array::<f32>("latCell").unwrap();
        let lon = file.read_array::<f32>("lonCell").unwrap();
        let land = file.read_array::<i32>("landmask").unwrap();
        let zgrid = file.read_array::<f32>("zgrid").unwrap();
        let metric = file.read_array::<f32>("zz").unwrap();
        let rho = file.read_array::<f32>("rho_base").unwrap();
        let nc = lat.len();
        assert_eq!(lon.len(), nc);
        assert_eq!(land.len(), nc);
        assert_eq!(zgrid.ndim(), 2);
        assert_eq!(zgrid.shape()[0], nc);
        let nv = zgrid.shape()[1] - 1;
        assert_eq!(metric.shape(), [nc, nv]);
        assert_eq!(rho.shape(), [1, nc, nv]);
        let output = Path::new(row["wire"].as_str().unwrap());
        std::fs::create_dir_all(output.parent().unwrap()).unwrap();
        let mut wire = std::io::BufWriter::new(std::fs::File::create(output).unwrap());
        for n in [nc, nv] {
            wire.write_all(&i32::try_from(n).unwrap().to_be_bytes())
                .unwrap();
        }
        for values in [&lat, &lon] {
            for &value in values {
                wire.write_all(&value.to_bits().to_be_bytes()).unwrap();
            }
        }
        for &value in &land {
            wire.write_all(&value.to_be_bytes()).unwrap();
        }
        for values in [&zgrid, &metric, &rho] {
            for &value in values {
                wire.write_all(&value.to_bits().to_be_bytes()).unwrap();
            }
        }
        wire.flush().unwrap();
        summary.push(serde_json::json!({
            "case": row["case"], "n_cells": nc, "n_vert_levels": nv,
            "source": row["source"], "wire": row["wire"],
            "rho_base_words_checked_by_native_driver": nc * nv,
            "climatology": "specified by replay runner"
        }));
    }
    let output = std::env::var_os("MPAS_AEROSOL_REAL_SUMMARY").unwrap();
    std::fs::write(output, serde_json::to_vec_pretty(&summary).unwrap()).unwrap();
}

struct Wire {
    data: Vec<u8>,
    at: usize,
}
impl Wire {
    fn bytes(&mut self, n: usize) -> &[u8] {
        let a = self.at;
        self.at += n;
        &self.data[a..self.at]
    }
    fn int(&mut self) -> usize {
        i32::from_be_bytes(self.bytes(4).try_into().unwrap()) as usize
    }
    fn floats(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| f32::from_bits(u32::from_be_bytes(self.bytes(4).try_into().unwrap())))
            .collect()
    }
    fn text(&mut self) -> String {
        String::from_utf8(self.bytes(64).to_vec())
            .unwrap()
            .trim()
            .into()
    }
}
fn words(label: &str, got: &[f32], native: &[f32]) {
    assert_eq!(got.len(), native.len());
    let mismatches: Vec<_> = got
        .iter()
        .zip(native)
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .take(12)
        .map(|(i, (a, b))| {
            format!(
                "{i}: {a} [{:08x}] != {b} [{:08x}]",
                a.to_bits(),
                b.to_bits()
            )
        })
        .collect();
    assert!(mismatches.is_empty(), "{label}: {}", mismatches.join(", "));
}
fn flatten_months(months: &[Vec<f32>], nc: usize, ng: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for c in 0..nc {
        for k in 0..ng {
            for m in 0..12 {
                out.push(months[m][c * ng + k]);
            }
        }
    }
    out
}
fn scalar_words(label: &str, state: &aerosols::AerosolState, native: &[f32]) {
    let interleaved: Vec<_> = state
        .nifa
        .iter()
        .zip(&state.nwfa)
        .flat_map(|(&a, &b)| [a, b])
        .collect();
    words(label, &interleaved, native);
}
#[test]
fn complete_pristine_module_replay() {
    let Some(root) = std::env::var_os("MPAS_AEROSOL_ORACLE") else {
        eprintln!(
            "Set MPAS_AEROSOL_ORACLE to the native replay corpus to run the Fortran comparison"
        );
        return;
    };
    let root = Path::new(&root);
    let mut cases = 0;
    for entry in std::fs::read_dir(root).unwrap() {
        let folder = entry.unwrap().path();
        if !folder.join("replay.bin").exists() {
            continue;
        }
        cases += 1;
        eprintln!("REPLAY {}", folder.file_name().unwrap().to_string_lossy());
        let mut wire = Wire {
            data: std::fs::read(folder.join("replay.bin")).unwrap(),
            at: 0,
        };
        let (nc, ng, nv) = (wire.int(), wire.int(), wire.int());
        let date = wire.text();
        let timestamp = wire.text();
        let lat = wire.floats(nc);
        let lon = wire.floats(nc);
        let land: Vec<_> = (0..nc).map(|_| wire.int() as i32).collect();
        let pressure = wire.floats(nc * nv);
        let nifa = wire.floats(nc * ng * 12);
        let nwfa = wire.floats(nc * ng * 12);
        let pwif = wire.floats(nc * ng * 12);
        let initial = wire.floats(nc * nv * 2);
        let lbc_start = wire.floats(nc * nv * 2);
        let lbc_later = wire.floats(nc * nv * 2);
        let missing_initial = wire.floats(nc * nv * 2);
        let missing_lbc = wire.floats(nc * nv * 2);
        assert_eq!(wire.at, wire.data.len());
        let input = folder.join(aerosols::NATIVE_FILENAME);
        let (clim, audit) = aerosols::horizontal(&input, &lat, &lon, &land, ng).unwrap();
        assert_eq!(audit.records_interpolated, 36 * ng);
        words(
            "nifa climatology",
            &flatten_months(&clim.nifa, nc, ng),
            &nifa,
        );
        words(
            "nwfa climatology",
            &flatten_months(&clim.nwfa, nc, ng),
            &nwfa,
        );
        words(
            "pressure climatology",
            &flatten_months(&clim.pressure, nc, ng),
            &pwif,
        );
        let (state, _) =
            aerosols::initialize(&input, &lat, &lon, &land, ng, &date, &pressure, nv).unwrap();
        scalar_words("initial", &state, &initial);
        let mut cache: Option<Climatology> = None;
        let lbc = aerosols::initialize_lbc(
            &input, &date, &date, &lat, &lon, &land, ng, &pressure, nv, &mut cache,
        )
        .unwrap()
        .unwrap();
        scalar_words("lbc start", &lbc, &lbc_start);
        let lbc = aerosols::initialize_lbc(
            &input, &timestamp, &date, &lat, &lon, &land, ng, &pressure, nv, &mut cache,
        )
        .unwrap()
        .unwrap();
        scalar_words("lbc reuse", &lbc, &lbc_later);
        let missing = folder.join("missing-climatology");
        let (zero, receipt) =
            aerosols::initialize(&missing, &lat, &lon, &land, ng, &date, &pressure, nv).unwrap();
        scalar_words("missing initial", &zero, &missing_initial);
        assert!(missing_lbc.iter().all(|&v| v == 321.0));
        assert!(!receipt.source_present);
        assert!(zero
            .nifa
            .iter()
            .chain(&zero.nwfa)
            .all(|&v| v.to_bits() == 0));
        assert!(aerosols::initialize_lbc(
            &missing, &timestamp, &date, &lat, &lon, &land, ng, &pressure, nv, &mut cache
        )
        .unwrap()
        .is_none());
        // Native aborts if a present file disappears before either reader open
        // (mpas_init_atm_thompson_aerosols.F:257-262,298-303). Rust propagates the I/O failure
        // at its reader boundary; no scalar output is accepted for that input.
        assert!(matches!(
            aerosols::horizontal(&missing, &lat, &lon, &land, ng),
            Err(rw_mpas::error::MpasError::Io(_))
        ));
        eprintln!(
            "PASS {}: {} horizontal and {} initial/LBC words",
            folder.file_name().unwrap().to_string_lossy(),
            nc * ng * 36,
            nc * nv * 6
        );
    }
    assert!(cases >= 2, "no native aerosol replay cases found");
}
