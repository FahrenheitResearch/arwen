//! Independent edge rows must retain the grid bytes at every worker count.

use rw_mpas::mesh::{derive::MpasMesh, emit::{write_grid, Provenance}, hull::delaunay_rings, icosa};

#[test]
fn derived_grid_bytes_do_not_depend_on_worker_count() {
    let points = icosa::seed(12, 0).unwrap();
    let rings = delaunay_rings(&points).unwrap();
    let root = std::env::temp_dir().join(format!("rw-mpas-derive-workers-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let mut reference = None;
    for workers in [1, 2, 8] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();
        let mesh = pool.install(|| MpasMesh::derive(
            points.clone(), vec![1.0; points.len()], &rings, 0.1)).unwrap();
        let path = root.join(format!("{workers}.nc"));
        write_grid(&mesh, &path, &Provenance::default(), true).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        if let Some(expected) = &reference {
            assert_eq!(&bytes, expected, "grid bytes changed at {workers} workers");
        } else {
            reference = Some(bytes);
        }
        std::fs::remove_file(path).unwrap();
    }
    std::fs::remove_dir(root).unwrap();
}
