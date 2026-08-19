//! Test-only golden-fixture IO (`golden/lane2/*`, written by
//! `golden/gen_lane2_goldens.py` running the real Python on real and
//! synthetic source data).

use std::path::{Path, PathBuf};

/// Root of the committed lane-2 goldens.
pub fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("golden").join("lane2")
}

/// This account's home directory, on either platform.
///
/// Used to COMPOSE the reference-tree default instead of spelling it
/// out: a written-out default is one developer's absolute path, and the
/// release snapshot's machine-path gate refuses to build a tree that
/// ships one (`tests/test_release_snapshot_machine_paths.py`).
fn home() -> PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_default()
}

/// The reference WPS_GEOG tree, if present on this box (same
/// convention as tests/test_static_rust_parity.py).
pub fn geog_root() -> Option<PathBuf> {
    let root = std::env::var("GPUWM_STATIC_PARITY_GEOG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            home()
                .join("Downloads")
                .join("WRF_1974_MP55_reference_bundle")
                .join("static")
                .join("WPS_GEOG")
        });
    if root.join("topo_gmted2010_30s").join("index").is_file() {
        Some(root)
    } else {
        None
    }
}

pub struct GoldenArray {
    pub dims: Vec<usize>,
    pub data: ArrayData,
}

pub enum ArrayData {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
}

pub fn read_arr(path: &Path) -> GoldenArray {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|err| panic!("golden {} unreadable: {err}", path.display()));
    assert!(
        bytes.len() >= 10 && &bytes[..8] == b"GWARR1\x00\x00",
        "golden {} has a bad header",
        path.display()
    );
    let code = bytes[8];
    let ndim = bytes[9] as usize;
    let mut dims = Vec::with_capacity(ndim);
    let mut off = 10;
    for _ in 0..ndim {
        dims.push(u64::from_le_bytes(
            bytes[off..off + 8].try_into().unwrap(),
        ) as usize);
        off += 8;
    }
    let count: usize = dims.iter().product();
    let payload = &bytes[off..];
    let data = match code {
        0 => ArrayData::F64(
            payload
                .chunks_exact(8)
                .take(count)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        1 => ArrayData::F32(
            payload
                .chunks_exact(4)
                .take(count)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        2 => ArrayData::I64(
            payload
                .chunks_exact(8)
                .take(count)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        3 => ArrayData::U8(payload[..count].to_vec()),
        4 => {
            let mut out = Vec::with_capacity(count);
            for pair in payload.chunks_exact(16) {
                let value =
                    i64::from_le_bytes(pair[..8].try_into().unwrap());
                let run =
                    u64::from_le_bytes(pair[8..].try_into().unwrap());
                for _ in 0..run {
                    out.push(value);
                }
            }
            assert_eq!(out.len(), count, "RLE golden {} truncated",
                       path.display());
            ArrayData::I64(out)
        }
        other => panic!("golden {} has dtype code {other}", path.display()),
    };
    let got = match &data {
        ArrayData::F64(v) => v.len(),
        ArrayData::F32(v) => v.len(),
        ArrayData::I64(v) => v.len(),
        ArrayData::U8(v) => v.len(),
    };
    assert_eq!(got, count, "golden {} is short", path.display());
    GoldenArray { dims, data }
}

pub fn read_f64(path: &Path) -> (Vec<usize>, Vec<f64>) {
    let arr = read_arr(path);
    match arr.data {
        ArrayData::F64(v) => (arr.dims, v),
        _ => panic!("golden {} is not f64", path.display()),
    }
}

pub fn read_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let arr = read_arr(path);
    match arr.data {
        ArrayData::F32(v) => (arr.dims, v),
        _ => panic!("golden {} is not f32", path.display()),
    }
}

pub fn read_i64(path: &Path) -> (Vec<usize>, Vec<i64>) {
    let arr = read_arr(path);
    match arr.data {
        ArrayData::I64(v) => (arr.dims, v),
        _ => panic!("golden {} is not i64", path.display()),
    }
}

pub fn read_bool(path: &Path) -> (Vec<usize>, Vec<bool>) {
    let arr = read_arr(path);
    match arr.data {
        ArrayData::U8(v) => {
            (arr.dims, v.into_iter().map(|b| b != 0).collect())
        }
        _ => panic!("golden {} is not u8/bool", path.display()),
    }
}

pub fn json(path: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("golden {} unreadable: {err}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|err| panic!("golden {} bad JSON: {err}", path.display()))
}

pub fn hex_f64(v: &serde_json::Value) -> f64 {
    f64::from_bits(
        u64::from_str_radix(v.as_str().expect("hex string"), 16)
            .expect("hex f64"),
    )
}

pub fn hex_f64_vec(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(hex_f64).collect()
}

/// One committed real-domain golden package
/// (`golden/lane2/real_domain_<tag>/`), usable only where the real
/// WPS_GEOG tree is present.
pub struct Package {
    pub dir: PathBuf,
    pub meta: serde_json::Value,
    pub geog: PathBuf,
}

pub fn load_package(tag: &str) -> Option<Package> {
    let geog = geog_root()?;
    let dir = golden_dir().join(format!("real_domain_{tag}"));
    let meta = json(&dir.join("package.json"));
    Some(Package { dir, meta, geog })
}

impl Package {
    pub fn bin(&self, name: &serde_json::Value) -> PathBuf {
        self.dir.join(name.as_str().expect("file name"))
    }

    /// Assemble the mesh from the committed twin/public transform
    /// outputs (the lane-1 seam's job at integration).
    pub fn mesh(&self) -> crate::sampler::SamplerMesh {
        let halo = self.meta["halo"].as_u64().unwrap() as usize;
        let nx = self.meta["nx"].as_u64().unwrap() as usize;
        let ny = self.meta["ny"].as_u64().unwrap() as usize;
        let mi = &self.meta["mesh_in"];
        let (_, lat32) = read_f32(&self.bin(&mi["lat32"]));
        let (_, lon32) = read_f32(&self.bin(&mi["lon32"]));
        let (_, lat64) = read_f64(&self.bin(&mi["lat64"]));
        let (_, lon64) = read_f64(&self.bin(&mi["lon64"]));
        let (_, latc32) = read_f32(&self.bin(&mi["latc32"]));
        let (_, lonc64) = read_f64(&self.bin(&mi["lonc64"]));
        crate::sampler::SamplerMesh::from_twin_outputs(
            nx + 2 * halo,
            ny + 2 * halo,
            lat32,
            lon32,
            &lat64,
            lon64,
            latc32,
            lonc64,
            self.meta["is_lambert"].as_bool().unwrap(),
            hex_f64(&self.meta["dx"]),
        )
        .expect("mesh assembly")
    }

    /// Public alias for tests outside this module.
    pub fn cells_key_public(
        key: &serde_json::Value,
    ) -> crate::sampler::CellsKey {
        Self::cells_key(key)
    }

    fn cells_key(key: &serde_json::Value) -> crate::sampler::CellsKey {
        let bits = |v: &serde_json::Value| {
            u64::from_str_radix(v.as_str().expect("hex"), 16)
                .expect("hex bits")
        };
        let shape: Vec<usize> = key["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        crate::sampler::CellsKey {
            dx: bits(&key["dx"]),
            dy: bits(&key["dy"]),
            known_x: bits(&key["known_x"]),
            known_y: bits(&key["known_y"]),
            known_lat: bits(&key["known_lat"]),
            known_lon: bits(&key["known_lon"]),
            x0: key["x0"].as_i64().unwrap(),
            y0: key["y0"].as_i64().unwrap(),
            nz: shape[0],
            ny: shape[1],
            nx: shape[2],
        }
    }

    /// A fixture-driven sampler: mesh from the committed transform
    /// outputs, pixel cells from the committed Python `pixel_cells`
    /// results (the grid transform itself is lane 1's).
    pub fn sampler(&self) -> crate::sampler::DomainSampler<'static> {
        use std::collections::BTreeMap;
        use std::sync::Arc;
        let halo = self.meta["halo"].as_u64().unwrap() as usize;
        let nx = self.meta["nx"].as_u64().unwrap() as usize;
        let ny = self.meta["ny"].as_u64().unwrap() as usize;
        let mut fixtures = BTreeMap::new();
        for (_, entry) in self.meta["windows"].as_object().unwrap() {
            let Some(cells) = entry["cells"].as_str() else {
                continue;
            };
            let (_, flat) = read_i64(&self.dir.join(cells));
            fixtures
                .insert(Self::cells_key(&entry["key"]), Arc::new(flat));
        }
        crate::sampler::DomainSampler::from_parts(
            None,
            hex_f64(&self.meta["dx"]),
            halo,
            nx,
            ny,
            self.mesh(),
            fixtures,
        )
        .expect("sampler assembly")
    }

    /// The nine resolved dataset paths for the build.
    pub fn geog_paths(&self) -> crate::fields::GeogPaths {
        let dir = |field: &str| {
            self.geog.join(
                self.meta["windows"][field]["dir"].as_str().unwrap(),
            )
        };
        crate::fields::GeogPaths {
            terrain: dir("terrain"),
            landuse: dir("landuse"),
            soil_top: dir("soil_top"),
            soil_bottom: dir("soil_bottom"),
            greenfrac: dir("greenfrac"),
            lai: dir("lai"),
            albedo: dir("albedo"),
            snow_albedo: dir("snow_albedo"),
            soil_temperature: dir("soil_temperature"),
        }
    }
}

/// Bitwise f64 comparison with a first-mismatch report.
pub fn assert_bits_f64(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if g.to_bits() != w.to_bits() {
            panic!(
                "{label}: first bit mismatch at [{k}]: got {g:?} \
                 ({:#018x}), want {w:?} ({:#018x})",
                g.to_bits(),
                w.to_bits()
            );
        }
    }
}

/// Bitwise f32 comparison with a first-mismatch report.
pub fn assert_bits_f32(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if g.to_bits() != w.to_bits() {
            panic!(
                "{label}: first bit mismatch at [{k}]: got {g:?} \
                 ({:#010x}), want {w:?} ({:#010x})",
                g.to_bits(),
                w.to_bits()
            );
        }
    }
}
