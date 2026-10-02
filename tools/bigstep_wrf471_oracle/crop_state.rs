//! Preserve WRF stored FP32 words while cropping Rust rw_netcdf raw dumps.
use std::{env, fs, io::Write, path::Path};

fn main() {
    let a: Vec<String> = env::args().collect();
    assert_eq!(a.len(), 8, "manifest dump_dir output_dir x0 y0 nx ny");
    let x0: usize = a[4].parse().unwrap();
    let y0: usize = a[5].parse().unwrap();
    let nx: usize = a[6].parse().unwrap();
    let ny: usize = a[7].parse().unwrap();
    fs::create_dir_all(&a[3]).unwrap();
    let mut shapes = String::new();
    for (variable_index, row) in fs::read_to_string(&a[1]).unwrap().lines().enumerate() {
        let cols: Vec<&str> = row.split('\t').collect();
        let name = cols[0];
        let dims: Vec<&str> = cols[1].split(',').collect();
        let lens: Vec<usize> = cols[2].split(',').map(|s| s.parse().unwrap()).collect();
        let mut count = lens.clone();
        let mut start = vec![0; dims.len()];
        for (axis, dim) in dims.iter().enumerate() {
            match *dim {
                "Time" => count[axis] = 1,
                "west_east" => { start[axis] = x0; count[axis] = nx; },
                "west_east_stag" => { start[axis] = x0; count[axis] = nx + 1; },
                "south_north" => { start[axis] = y0; count[axis] = ny; },
                "south_north_stag" => { start[axis] = y0; count[axis] = ny + 1; },
                _ => (),
            }
            assert!(start[axis] + count[axis] <= lens[axis]);
        }
        let raw = fs::read(Path::new(&a[2]).join(format!("{variable_index:04}.f64"))).unwrap();
        assert_eq!(raw.len(), lens.iter().product::<usize>() * 8);
        let mut file = fs::File::create(Path::new(&a[3]).join(format!("{name}.f32"))).unwrap();
        for mut linear in 0..count.iter().product::<usize>() {
            let mut source = 0usize;
            let mut stride = 1usize;
            for axis in (0..count.len()).rev() {
                source += (linear % count[axis] + start[axis]) * stride;
                linear /= count[axis];
                stride *= lens[axis];
            }
            let word = f64::from_le_bytes(raw[source * 8..source * 8 + 8].try_into().unwrap());
            let fp32 = word as f32;
            assert!(word.is_nan() || fp32 as f64 == word, "non-FP32 source: {name}");
            file.write_all(&fp32.to_le_bytes()).unwrap();
        }
        let shape: Vec<String> = dims.iter().zip(count).filter(|(d, _)| **d != "Time")
            .map(|(_, n)| n.to_string()).collect();
        shapes.push_str(&format!("{name}\t{}\n", if shape.is_empty() { "1".into() } else { shape.join(",") }));
    }
    fs::write(Path::new(&a[3]).join("shapes.tsv"), shapes).unwrap();
}
