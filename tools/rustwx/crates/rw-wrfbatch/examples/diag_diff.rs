//! Two wrfouts through the SAME wrf-rust diagnostics (`compute_var`, the
//! renderer's import lane), one difference row per product.
//!
//!     cargo run --release -p rw-wrfbatch --example diag_diff -- A_WRFOUT B_WRFOUT
//!
//! Prints mean(A-B), rms, Pearson correlation, max |A-B| and both means,
//! over the cells where both are finite.  Used by the WOOF-vs-HRRR
//! like-for-like audit (A = the run, B = rw_native_wrfout of the reference).

use rayon::prelude::*;
use rw_wrfbatch::wrf_process::compute_var;
use wrf_core::WrfFile;

const PRODUCTS: &[(&str, &str, Option<&str>, f64)] = &[
    // (label, wrf-core name, units, display scale)
    ("slp hPa", "slp", Some("hPa"), 1.0),
    ("t2 K", "t2", None, 1.0),
    ("rh2m %", "rh2m", None, 1.0),
    ("wspd10 m/s", "wspd10", None, 1.0),
    ("pw mm", "pw", None, 1.0),
    ("sbcape J/kg", "sbcape", None, 1.0),
    ("sbcin J/kg", "sbcin", None, 1.0),
    ("mlcape J/kg", "mlcape", None, 1.0),
    ("mlcin J/kg", "mlcin", None, 1.0),
    ("mucape J/kg", "mucape", None, 1.0),
    ("mucin J/kg", "mucin", None, 1.0),
    ("shear_0_1km m/s", "shear_0_1km", None, 1.0),
    ("shear_0_6km m/s", "shear_0_6km", None, 1.0),
    ("srh1 m2/s2", "srh1", None, 1.0),
    ("srh3 m2/s2", "srh3", None, 1.0),
    ("maxdbz dBZ", "maxdbz", None, 1.0),
    ("freezing_level m", "freezing_level", None, 1.0),
    ("PBLH m", "PBLH", None, 1.0),
    ("td2 C", "dp2m", None, 1.0),
];

fn stats(a: &[f64], b: &[f64]) -> (usize, f64, f64, f64, f64, f64, f64) {
    let pairs: Vec<(f64, f64)> = a.iter().zip(b).filter(|(x, y)| x.is_finite() && y.is_finite()).map(|(x, y)| (*x, *y)).collect();
    let n = pairs.len();
    if n == 0 {
        return (0, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let nf = n as f64;
    let ma = pairs.iter().map(|p| p.0).sum::<f64>() / nf;
    let mb = pairs.iter().map(|p| p.1).sum::<f64>() / nf;
    let mean = ma - mb;
    let rms = (pairs.iter().map(|p| (p.0 - p.1).powi(2)).sum::<f64>() / nf).sqrt();
    let max = pairs.iter().map(|p| (p.0 - p.1).abs()).fold(0.0, f64::max);
    let cov = pairs.iter().map(|p| (p.0 - ma) * (p.1 - mb)).sum::<f64>();
    let va = pairs.iter().map(|p| (p.0 - ma).powi(2)).sum::<f64>();
    let vb = pairs.iter().map(|p| (p.1 - mb).powi(2)).sum::<f64>();
    (n, mean, rms, cov / (va * vb).sqrt(), max, ma, mb)
}

/// The renderer's 10 m gust (wrf_process::push_gust_10m), same arithmetic.
fn gust(f: &WrfFile) -> Option<Vec<f64>> {
    let hagl = compute_var(f, "height_agl", 0, None).ok()?;
    let wspd = compute_var(f, "wspd", 0, None).ok()?;
    let pblh = compute_var(f, "PBLH", 0, None).ok()?;
    let w10 = compute_var(f, "wspd10", 0, None).ok()?;
    let nz = hagl.shape[0];
    let cells = pblh.data.len();
    Some((0..cells).map(|i| {
        let zpbl = pblh.data[i];
        let wind10 = w10.data[i];
        if !zpbl.is_finite() || !wind10.is_finite() { return f64::NAN; }
        let mut k = 0;
        while k + 1 < nz && hagl.data[k * cells + i] < zpbl { k += 1; }
        let top = wspd.data[k * cells + i];
        let damp = 1.0 - (zpbl.max(0.0) / 2000.0).min(0.5);
        (wind10 + (top - wind10) * damp).max(wind10)
    }).collect())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() == 2, "usage: diag_diff A_WRFOUT B_WRFOUT");
    let a = WrfFile::open(&args[0]).expect("open A");
    let b = WrfFile::open(&args[1]).expect("open B");
    println!("A = {}\nB = {}", args[0], args[1]);
    println!("{:<20} {:>9} {:>10} {:>10} {:>8} {:>10} {:>10} {:>10}", "product", "n", "mean A-B", "rms", "corr", "max|A-B|", "mean A", "mean B");
    let rows: Vec<String> = PRODUCTS.par_iter().map(|(label, name, units, _)| {
        let (x, y) = rayon::join(|| compute_var(&a, name, 0, *units), || compute_var(&b, name, 0, *units));
        match (x, y) {
            (Ok(x), Ok(y)) => {
                let (n, mean, rms, corr, max, ma, mb) = stats(&x.data, &y.data);
                format!("{label:<20} {n:>9} {mean:>10.3} {rms:>10.3} {corr:>8.4} {max:>10.2} {ma:>10.3} {mb:>10.3}")
            }
            (x, y) => format!("{label:<20} failed: {:?} / {:?}", x.err(), y.err()),
        }
    }).collect();
    for r in rows { println!("{r}"); }
    if let (Some(x), Some(y)) = (gust(&a), gust(&b)) {
        let (n, mean, rms, corr, max, ma, mb) = stats(&x, &y);
        println!("{:<20} {n:>9} {mean:>10.3} {rms:>10.3} {corr:>8.4} {max:>10.2} {ma:>10.3} {mb:>10.3}", "gust10 m/s");
    }
}
