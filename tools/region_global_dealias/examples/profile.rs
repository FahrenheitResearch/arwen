use std::fs;
use std::path::Path;

fn main() {
    let dir = std::env::args().nth(1).expect("dump dir");
    let cut = std::env::args().nth(2).unwrap_or_else(|| "01".into());
    let d = Path::new(&dir);
    let meta: serde_lite::Value =
        serde_lite::parse(&fs::read_to_string(d.join(format!("raw_cut{cut}.json"))).unwrap());
    let rows = meta.usize("rows");
    let gates = meta.usize("gates");
    let bytes = fs::read(d.join(format!("raw_cut{cut}.bin"))).unwrap();
    let observed: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let nyq = meta.f32_array("nyquist_mps");
    let az = meta.f32_array("azimuth_deg");
    let resolved = region_global_dealias::solver::resolve_nyquist(&nyq, rows);
    let wraps = region_global_dealias::solver::sweep_wraps(
        &az.iter().map(|a| a.rem_euclid(360.0)).collect::<Vec<_>>(),
    );

    let mut best = [f64::MAX; 4];
    let (mut nfeat, mut nedge) = (0, 0);
    for _ in 0..5 {
        let (t, f, e) =
            region_global_dealias::solver::profile_phases(&observed, &resolved, rows, gates, wraps);
        nfeat = f;
        nedge = e;
        let sum: f64 = t.iter().sum();
        if sum < best.iter().sum::<f64>() {
            best = t;
        }
    }
    let total: f64 = best.iter().sum();
    println!("{dir}/cut{cut}  {rows}x{gates}  regions {nfeat}  edges {nedge}  total {total:.1} ms",);
    let names = [
        "interval_limits",
        "find_regions",
        "edge_sum_and_count",
        "merge_loop",
    ];
    for (name, ms) in names.iter().zip(best.iter()) {
        println!("  {name:<20} {ms:7.1} ms  {:5.1}%", ms / total * 100.0);
    }
}

/// Tiny JSON reader so the example needs no dependency.
mod serde_lite {
    pub struct Value(String);
    pub fn parse(s: &str) -> Value {
        Value(s.to_string())
    }
    impl Value {
        pub fn usize(&self, key: &str) -> usize {
            let pat = format!("\"{key}\":");
            let i = self.0.find(&pat).unwrap() + pat.len();
            self.0[i..]
                .trim_start()
                .split(|c: char| !c.is_ascii_digit())
                .next()
                .unwrap()
                .parse()
                .unwrap()
        }
        pub fn f32_array(&self, key: &str) -> Vec<f32> {
            let pat = format!("\"{key}\":[");
            let i = self.0.find(&pat).unwrap() + pat.len();
            let j = self.0[i..].find(']').unwrap() + i;
            self.0[i..j]
                .split(',')
                .map(|v| v.trim().parse::<f32>().unwrap_or(f32::NAN))
                .collect()
        }
    }
}
