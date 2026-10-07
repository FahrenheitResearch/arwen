//
// WOOF post-processor, group E: read the 2D carriers (and TKE) group E needs
// from a
// WRF-format WOOF history (variable names per the WRF ARW user's guide and
// spec 3.1), first time record.

use super::Carriers;

/// Owned carriers of one frame.
#[derive(Clone, Debug, Default)]
pub struct OwnedCarriers {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub vars: std::collections::BTreeMap<&'static str, Vec<f32>>,
    /// Where TKE came from (`TKE_PBL`, `QKE / 2`), when present.
    pub tke_source: Option<&'static str>,
    pub valid_time: String,
}

const REQUIRED: [&str; 5] = ["HGT", "PSFC", "U10", "V10", "XLAT"];
const OPTIONAL: [&str; 10] = ["UST", "T2", "TH2", "Q2", "HFX", "QFX", "LH", "SINALPHA", "COSALPHA", "PBLH"];

impl OwnedCarriers {
    /// Reads a history file.
    pub fn read(path: &std::path::Path) -> Result<Self, String> {
        let f = netcrust::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let dim = |name: &str| f.dimension(name).map(|d| d.len()).ok_or_else(|| format!("dimension {name} absent"));
        let nx = dim("west_east")?;
        let ny = dim("south_north")?;
        let nz = dim("bottom_top")?;
        let read = |name: &str| -> Result<Vec<f32>, String> {
            f.read_f64_first_record_or_all(name)
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .map_err(|e| format!("read {name}: {e}"))
        };
        let mut vars = std::collections::BTreeMap::new();
        for name in REQUIRED {
            vars.insert(name, read(name)?);
        }
        for name in OPTIONAL {
            if f.variable(name).is_some() {
                vars.insert(name, read(name)?);
            }
        }
        let mut tke_source = None;
        if f.variable("TKE_PBL").is_some() {
            vars.insert("TKE", read("TKE_PBL")?);
            tke_source = Some("TKE_PBL");
        } else if f.variable("QKE").is_some() {
            let q = read("QKE")?;
            vars.insert("TKE", q.into_iter().map(|x| x * 0.5).collect());
            tke_source = Some("QKE / 2");
        }
        let valid_time = crate::history::read_times(&f)
            .ok()
            .and_then(|v| v.into_iter().next())
            .unwrap_or_default();
        Ok(Self { nx, ny, nz, vars, tke_source, valid_time })
    }

    fn get(&self, name: &str) -> Option<&[f32]> {
        self.vars.get(name).map(Vec::as_slice)
    }

    /// Borrowed carriers over Group A's shared state of the same frame.
    #[must_use]
    pub fn carriers<'a>(&'a self, state: &'a crate::state::ColumnState) -> Carriers<'a> {
        let r = |n: &str| self.get(n).expect("required carrier was read");
        Carriers {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            state,
            tke: self.get("TKE"),
            hgt: r("HGT"),
            psfc: r("PSFC"),
            u10: r("U10"),
            v10: r("V10"),
            ust: self.get("UST"),
            t2: self.get("T2"),
            th2: self.get("TH2"),
            q2: self.get("Q2"),
            hfx: self.get("HFX"),
            qfx: self.get("QFX"),
            lh: self.get("LH"),
            sinalpha: self.get("SINALPHA"),
            cosalpha: self.get("COSALPHA"),
        }
    }
}
