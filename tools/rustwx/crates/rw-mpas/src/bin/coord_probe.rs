//! Generate a graded mesh from a spec and census its cell coordination,
//! without writing a grid file.
//!
//! WHY IT EXISTS. `v16.66.195630` shipped with one 4-coordinated cell and no
//! gate anywhere read a coordination number (2026-08-26, gpuwm-hex
//! `tree/evidence/graded-blowup-20260826/`). This is the instrument that
//! reproduced that mesh from its unchanged spec row -- 195,630 cells,
//! histogram `{4: 1, 5: 1028, 6: 193584, 7: 1016, 8: 1}`, cell 195615
//! fourteen indices from the end -- and then measured the same spec through
//! the repaired surgery. It runs the real `generate_graded`, on the CPU, and
//! writes nothing, so a coordination question costs one run instead of a full
//! emit + static + init.
//!
//! Usage: `coord_probe SPEC.json`

use rw_mpas::mesh::density::MeshSpec;
use rw_mpas::mesh::geom::lat_lon;
use rw_mpas::mesh::hierarchy::{DEFAULT_BETA, generate_graded};
use rw_mpas::mesh::hull::delaunay_rings;
use rw_mpas::mesh::lloyd::LloydOptions;
use rw_mpas::mesh::surgery::SurgeryOptions;
use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let text = std::fs::read_to_string(path).expect("spec");
    let spec = MeshSpec::from_json(&text).expect("parse");
    let t0 = std::time::Instant::now();
    let (points, rings, _outcome, choice, reports) = generate_graded(
        &spec,
        50_000,
        &LloydOptions::default(),
        &SurgeryOptions::default(),
        DEFAULT_BETA,
        |s| println!("  {s}"),
    )
    .unwrap_or_else(|e| panic!("refused: {e}"));
    let n = points.len();
    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    for i in 0..n {
        *hist.entry(rings.ring(i).len()).or_default() += 1;
    }
    println!(
        "CELLS {n}  GP({},{})  levels {}  wall {:.1}s",
        choice.m,
        choice.n,
        reports.len(),
        t0.elapsed().as_secs_f64()
    );
    println!("HIST {hist:?}");
    for r in &reports {
        println!(
            "LEVEL {} inserted {} surgery rounds {} ins {} del {} cav {}",
            r.level,
            r.inserted,
            r.surgery.rounds,
            r.surgery.inserted,
            r.surgery.deleted,
            r.surgery.cavity_resamples
        );
    }
    let rings2 = delaunay_rings(&points).unwrap();
    for i in 0..n {
        let d = rings2.ring(i).len();
        if d < 5 {
            let (lat, lon) = lat_lon(points[i]);
            let nbdeg: Vec<usize> = rings2
                .ring(i)
                .iter()
                .map(|&j| rings2.ring(j as usize).len())
                .collect();
            println!(
                "SUB5 cell {i} deg {d} from_end {} lat {:.3} lon {:.3} nb_deg {nbdeg:?}",
                n - 1 - i,
                lat.to_degrees(),
                lon.to_degrees()
            );
        }
    }
}
