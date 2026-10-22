//! Export actual forecast storm columns for the independent WRF HAILCAST oracle.
//! NetCDF reading and all profile transforms remain native.
use serde_json::json;
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};
use wrf_core::WrfFile;
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<String> = env::args().collect();
    let mut paths: Vec<PathBuf> = fs::read_dir(&a[1])?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("wrfout_")
        })
        .collect();
    paths.sort();
    for path in paths {
        let f = WrfFile::open(&path)?;
        let n = f.nxy();
        let hail = f.read_var("HAILCAST_DIAM_MAX", 0)?;
        let bad: Vec<usize> = hail
            .iter()
            .enumerate()
            .filter_map(|(i, x)| (!x.is_finite()).then_some(i))
            .collect();
        println!("{} nonfinite={}", path.display(), bad.len());
        if bad.is_empty() {
            continue;
        }
        let t = f.read_var("T", 0)?;
        let nz = t.len() / n;
        let p = f.read_var("P", 0)?;
        let pb = f.read_var("PB", 0)?;
        let ph = f.read_var("PH", 0)?;
        let phb = f.read_var("PHB", 0)?;
        let w = f.read_var("W", 0)?;
        let ht = f.read_var("HGT", 0)?;
        let q: Vec<Vec<f64>> = ["QVAPOR", "QICE", "QCLOUD", "QRAIN", "QSNOW", "QGRAUP"]
            .iter()
            .map(|s| f.read_var(s, 0))
            .collect::<Result<_, _>>()?;
        let cols: Vec<usize> = (0..n)
            .filter(|&c| !hail[c].is_finite() || (0..nz).any(|k| w[k * n + c] >= 10.))
            .collect();
        let mut out = BufWriter::new(File::create(&a[2])?);
        out.write_all(&(cols.len() as i32).to_le_bytes())?;
        out.write_all(&(nz as i32).to_le_bytes())?;
        for &c in &cols {
            for v in 0..11 {
                for k in 0..nz {
                    let i = k * n + c;
                    let pressure = (p[i] + pb[i]) as f32;
                    let temp = (t[i] as f32 + 300.) * (pressure / 100000.).powf(287. / 1004.);
                    let x = match v {
                        0 => temp,
                        1 => ((ph[i] + phb[i] + ph[i + n] + phb[i + n]) / (2. * 9.81)) as f32,
                        2 => pressure,
                        3 => {
                            pressure
                                / (287.
                                    * temp
                                    * (1. + 0.609 * q[0][i] as f32 / (1. + q[0][i] as f32)))
                        }
                        4..=9 => q[v - 4][i] as f32,
                        10 => w[i] as f32,
                        _ => unreachable!(),
                    };
                    out.write_all(&x.to_le_bytes())?;
                }
            }
            out.write_all(&(ht[c] as f32).to_le_bytes())?;
            out.write_all(&2000f32.to_le_bytes())?;
        }
        out.flush()?;
        fs::write(
            format!("{}.json", a[2]),
            serde_json::to_vec_pretty(
                &json!({"source":path,"nz":nz,"nx":f.nx,"ny":f.ny,"nonfinite":bad,"columns":cols,"duration_seconds":2000,"density":"ideal moist air from history; instant fields at history time"}),
            )?,
        )?;
        break;
    }
    Ok(())
}
