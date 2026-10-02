//! Native NetCDF extraction and deterministic input-case construction.
//! Arrays are written as little-endian float32 in (z,y,x) order.
use std::{
    collections::BTreeMap,
    env,
    ffi::{CString, c_char, c_int},
    fs,
    io::Write,
    path::Path,
};

#[link(name = "netcdf")]
unsafe extern "C" {
    fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_inq_varid(id: c_int, name: *const c_char, varid: *mut c_int) -> c_int;
    fn nc_inq_varndims(id: c_int, varid: c_int, n: *mut c_int) -> c_int;
    fn nc_inq_vardimid(id: c_int, varid: c_int, dims: *mut c_int) -> c_int;
    fn nc_inq_dimlen(id: c_int, dimid: c_int, n: *mut usize) -> c_int;
    fn nc_get_var_float(id: c_int, varid: c_int, data: *mut f32) -> c_int;
    fn nc_get_att_float(id: c_int, varid: c_int, name: *const c_char, value: *mut f32) -> c_int;
}
#[derive(Clone)]
struct Array {
    shape: Vec<usize>,
    data: Vec<f32>,
}
impl Array {
    fn filled(shape: &[usize], value: f32) -> Self {
        Self {
            shape: shape.to_vec(),
            data: vec![value; shape.iter().product()],
        }
    }
    fn at(&self, k: usize, j: usize, i: usize) -> f32 {
        self.data[(k * self.shape[1] + j) * self.shape[2] + i]
    }
    fn xy(&self, j: usize, i: usize) -> f32 {
        self.data[j * self.shape[1] + i]
    }
    fn put(&mut self, k: usize, j: usize, i: usize, value: f32) {
        let at = (k * self.shape[1] + j) * self.shape[2] + i;
        self.data[at] = value;
    }
}
struct Dataset(c_int);
impl Dataset {
    fn open(path: &str) -> Self {
        let p = CString::new(path).unwrap();
        let mut id = 0;
        let err = unsafe { nc_open(p.as_ptr(), 0, &mut id) };
        assert_eq!(err, 0, "open error {err}");
        Self(id)
    }
    fn read_optional(&self, name: &str) -> Option<Array> {
        let c = CString::new(name).unwrap();
        let mut vid = 0;
        if unsafe { nc_inq_varid(self.0, c.as_ptr(), &mut vid) } != 0 {
            return None;
        }
        let mut nd = 0;
        assert_eq!(unsafe { nc_inq_varndims(self.0, vid, &mut nd) }, 0);
        let mut dims = vec![0; nd as usize];
        assert_eq!(
            unsafe { nc_inq_vardimid(self.0, vid, dims.as_mut_ptr()) },
            0
        );
        let mut shape = Vec::new();
        for dim in dims {
            let mut n = 0;
            assert_eq!(unsafe { nc_inq_dimlen(self.0, dim, &mut n) }, 0);
            shape.push(n);
        }
        let mut data = vec![0.; shape.iter().product()];
        assert_eq!(
            unsafe { nc_get_var_float(self.0, vid, data.as_mut_ptr()) },
            0
        );
        // WRF files carry time as the first dimension; fixture extraction uses time zero.
        if shape.len() > 1 {
            let n: usize = shape[1..].iter().product();
            data.truncate(n);
            shape.remove(0);
        }
        Some(Array { shape, data })
    }
    fn read(&self, name: &str) -> Array {
        self.read_optional(name)
            .unwrap_or_else(|| panic!("missing WRF variable {name}"))
    }
    fn attr(&self, name: &str) -> f32 {
        let c = CString::new(name).unwrap();
        let mut value = 0.;
        assert_eq!(
            unsafe { nc_get_att_float(self.0, -1, c.as_ptr(), &mut value) },
            0,
            "missing attribute {name}"
        );
        value
    }
}
impl Drop for Dataset {
    fn drop(&mut self) {
        unsafe { nc_close(self.0) };
    }
}

fn crop(a: &Array, x: usize, y: usize, nx: usize, ny: usize) -> Array {
    match a.shape.len() {
        1 => a.clone(),
        2 => {
            let mut out = Array::filled(&[ny, nx], 0.);
            for j in 0..ny {
                for i in 0..nx {
                    out.data[j * nx + i] = a.xy(y + j, x + i);
                }
            }
            out
        }
        3 => {
            let nz = a.shape[0];
            let mut out = Array::filled(&[nz, ny, nx], 0.);
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        out.put(k, j, i, a.at(k, y + j, x + i));
                    }
                }
            }
            out
        }
        _ => panic!("unexpected variable dimensions"),
    }
}
fn emit(out: &Path, name: &str, arrays: &BTreeMap<String, Array>, meta: &str) {
    let folder = out.join(name);
    fs::create_dir_all(&folder).unwrap();
    let mut files = Vec::new();
    for (key, a) in arrays {
        let filename = format!("{key}.f32");
        let mut f = fs::File::create(folder.join(&filename)).unwrap();
        for x in &a.data {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
        let shape = a
            .shape
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(",");
        files.push(format!(
            "\"{key}\":{{\"file\":\"{filename}\",\"shape\":[{shape}]}}"
        ));
    }
    fs::write(
        folder.join("arrays.json"),
        format!(
            "{{\"arrays\":{{{}}},\"metadata\":{meta}}}\n",
            files.join(",")
        ),
    )
    .unwrap();
}
fn coupling(a: &mut BTreeMap<String, Array>, dx: f32, dy: f32) {
    let u = &a["u"];
    let v = &a["v"];
    let nz = u.shape[0];
    let ny = u.shape[1];
    let nx = v.shape[2];
    let c1 = &a["c1h"];
    let c2 = &a["c2h"];
    let mut ru = Array::filled(&[nz, ny, nx + 1], 0.);
    let mut rv = Array::filled(&[nz, ny + 1, nx], 0.);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..=nx {
                let muf = a["muf_u"].xy(j, i);
                let m = c1.data[k] * muf + c2.data[k];
                ru.put(k, j, i, (m * u.at(k, j, i)) / a["msfuy"].xy(j, i));
            }
        }
    }
    for k in 0..nz {
        for j in 0..=ny {
            for i in 0..nx {
                let muf = a["muf_v"].xy(j, i);
                let m = c1.data[k] * muf + c2.data[k];
                rv.put(k, j, i, (m * v.at(k, j, i)) / a["msfvx"].xy(j, i));
            }
        }
    }
    // The diagnosed eta mass flux is a fixture input, not an oracle output.
    // Follow calc_ww_cp continuity with a float32 boundary at each operation.
    let dnw = &a["dnw"];
    let msft = &a["msftx"];
    let mut rw = Array::filled(&[nz + 1, ny, nx], 0.);
    let rdx = 1f32 / dx;
    let rdy = 1f32 / dy;
    for j in 0..ny {
        for i in 0..nx {
            let mut div = vec![0f32; nz];
            let mut dmdt = 0f32;
            for k in 0..nz {
                let du = ru.at(k, j, i + 1) - ru.at(k, j, i);
                let dv = rv.at(k, j + 1, i) - rv.at(k, j, i);
                let bracket = rdx * du + rdy * dv;
                div[k] = (msft.xy(j, i) * dnw.data[k]) * bracket;
                dmdt += div[k];
            }
            let mut w = 0f32;
            for k in 1..nz {
                w -= dnw.data[k - 1] * c1.data[k - 1] * dmdt;
                w -= div[k - 1];
                rw.put(k, j, i, w);
            }
        }
    }
    a.insert("ru".into(), ru);
    a.insert("rv".into(), rv);
    a.insert("rw".into(), rw);
}
fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "extract_inputs <wrf-file> <out> <source-sha256>"
    );
    let ds = Dataset::open(&args[1]);
    let out = Path::new(&args[2]);
    fs::create_dir_all(out).unwrap();
    let hash = &args[3];
    let nx = 24usize;
    let ny = 24usize;
    let u = ds.read("U");
    let v = ds.read("V");
    let nz = u.shape[0];
    let full_ny = u.shape[1];
    let full_nx = v.shape[2];
    let dx = ds.attr("DX");
    let dy = ds.attr("DY");
    let hgt = ds.read("HGT");
    let mut max_gradient = 0f32;
    let mut gx = full_nx / 2;
    let mut gy = full_ny / 2;
    for j in 1..full_ny - 1 {
        for i in 1..full_nx - 1 {
            let gradient =
                (hgt.xy(j, i + 1) - hgt.xy(j, i)).abs() + (hgt.xy(j + 1, i) - hgt.xy(j, i)).abs();
            if gradient > max_gradient {
                max_gradient = gradient;
                gx = i;
                gy = j;
            }
        }
    }
    let center = ((full_nx - nx) / 2, (full_ny - ny) / 2);
    let steep = (
        gx.saturating_sub(nx / 2).min(full_nx - nx),
        gy.saturating_sub(ny / 2).min(full_ny - ny),
    );
    let scenarios = [
        ("real_interior", center.0, center.1, "real"),
        ("real_west_boundary", 0, center.1, "boundary"),
        ("real_open_boundary", full_nx - nx, full_ny - ny, "open"),
        ("real_steep_terrain", steep.0, steep.1, "steep"),
        ("map_factor_extremes", center.0, center.1, "map"),
        ("zero_nearzero_tracers", center.0, center.1, "zero"),
        ("southern_reflection", center.0, center.1, "south"),
        ("limiter_moisture_front", center.0, center.1, "front"),
    ];
    let mut raw = BTreeMap::new();
    for name in [
        "W", "T", "QVAPOR", "MU", "MUB", "FNM", "FNP", "RDNW", "RDN", "DNW", "DN", "C1H", "C2H",
        "C1F", "C2F", "ZNW", "ZNU", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "XLAT", "XLONG", "HGT",
        "PH", "PHB",
    ] {
        raw.insert(name.to_string(), ds.read(name));
    }
    for (name, x, y, kind) in scenarios {
        let open = kind == "boundary" || kind == "steep" || kind == "open";
        let specified = kind == "boundary" || kind == "steep";
        let mut a: BTreeMap<String, Array> = BTreeMap::new();
        a.insert("u".into(), crop(&u, x, y, nx + 1, ny));
        a.insert("v".into(), crop(&v, x, y, nx, ny + 1));
        a.insert("w".into(), crop(&raw["W"], x, y, nx, ny));
        for (src, dst) in [
            ("T", "scalar"),
            ("QVAPOR", "q0"),
            ("MU", "mu_perturbation"),
            ("MUB", "mub"),
            ("XLAT", "latitude"),
            ("XLONG", "longitude"),
            ("HGT", "terrain"),
            ("PH", "ph"),
            ("PHB", "phb"),
        ] {
            a.insert(dst.into(), crop(&raw[src], x, y, nx, ny));
        }
        for (src, dst) in [
            ("FNM", "fnm"),
            ("FNP", "fnp"),
            ("RDNW", "rdnw"),
            ("RDN", "rdn"),
            ("DNW", "dnw"),
            ("DN", "dn"),
            ("C1H", "c1h"),
            ("C2H", "c2h"),
            ("C1F", "c1f"),
            ("C2F", "c2f"),
            ("ZNW", "znw"),
            ("ZNU", "znu"),
        ] {
            a.insert(dst.into(), raw[src].clone());
        }
        for value in &mut a.get_mut("scalar").unwrap().data {
            *value += 300f32;
        }
        let mut mass = a["mub"].clone();
        for (m, p) in mass.data.iter_mut().zip(&a["mu_perturbation"].data) {
            *m += p;
        }
        a.insert("muts".into(), mass.clone());
        a.insert("mu_old".into(), mass);
        // Retain source neighbors outside the crop for staggered boundary masses.
        // Only the original physical domain boundary repeats its nearest mass.
        let source_mass = |j: usize, i: usize| raw["MU"].xy(j, i) + raw["MUB"].xy(j, i);
        let mut muf_u = Array::filled(&[ny, nx + 1], 0.);
        for j in 0..ny {
            for i in 0..=nx {
                let sx = x + i;
                let il = sx.saturating_sub(1);
                let ir = sx.min(full_nx - 1);
                muf_u.data[j * (nx + 1) + i] =
                    0.5f32 * (source_mass(y + j, il) + source_mass(y + j, ir));
            }
        }
        let mut muf_v = Array::filled(&[ny + 1, nx], 0.);
        for j in 0..=ny {
            for i in 0..nx {
                let sy = y + j;
                let jb = sy.saturating_sub(1);
                let jt = sy.min(full_ny - 1);
                muf_v.data[j * nx + i] = 0.5f32 * (source_mass(jb, x + i) + source_mass(jt, x + i));
            }
        }
        a.insert("muf_u".into(), muf_u);
        a.insert("muf_v".into(), muf_v);
        for (src, xsize, ysize, xname, yname) in [
            ("MAPFAC_M", nx, ny, "msftx", "msfty"),
            ("MAPFAC_U", nx + 1, ny, "msfux", "msfuy"),
            ("MAPFAC_V", nx, ny + 1, "msfvx", "msfvy"),
        ] {
            let map = crop(&raw[src], x, y, xsize, ysize);
            a.insert(xname.into(), map.clone());
            a.insert(yname.into(), map);
        }
        let mut transform = match kind {
            "map" => {
                for field in ["msftx", "msfty", "msfux", "msfuy", "msfvx", "msfvy"] {
                    let map = a.get_mut(field).unwrap();
                    let xs = map.shape[1];
                    for j in 0..map.shape[0] {
                        for i in 0..xs {
                            map.data[j * xs + i] = if i < xs / 2 { 0.25 } else { 3.0 };
                        }
                    }
                }
                "replace each isotropic map factor by 0.25 on its western half and 3.0 on its eastern half; recompute coupled fluxes"
            }
            "zero" => {
                let q = a.get_mut("q0").unwrap();
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..nx {
                            let value = match (i / 3 + j / 3 + k / 7) % 4 {
                                0 => 0f32,
                                1 => 1e-30f32,
                                2 => f32::MIN_POSITIVE,
                                _ => f32::from_bits(1),
                            };
                            q.put(k, j, i, value);
                        }
                    }
                }
                "replace moisture by blocks of exact zero, 1e-30, smallest normal, and smallest subnormal; preserve winds and total theta"
            }
            "south" => {
                for val in &mut a.get_mut("latitude").unwrap().data {
                    *val = -val.abs();
                }
                for val in &mut a.get_mut("v").unwrap().data {
                    *val = -*val;
                }
                "reflect latitude to the southern hemisphere and reverse meridional winds; recompute coupled fluxes; source is a northern-hemisphere state"
            }
            "front" => {
                let q = a.get_mut("q0").unwrap();
                for k in 0..nz {
                    for j in 0..ny {
                        for i in nx / 2..nx {
                            q.put(k, j, i, 0.);
                        }
                    }
                }
                "replace the eastern half of source moisture with zero to form a sharp moisture front; stage moisture is 1.05 times old moisture"
            }
            "boundary" => {
                "crop beginning at the physical western source boundary; configured specified lateral boundaries"
            }
            "open" => {
                "crop at physical eastern and northern source boundaries; configure all lateral boundaries open"
            }
            "steep" => {
                "crop around the source grid point with the largest sum of absolute east/north terrain differences"
            }
            _ => "central source crop; no prognostic or map-factor changes",
        }.to_string();
        if !open {
            // A periodic crop has one redundant terminal face on each stagger.
            // Close those aliases explicitly rather than treating the source's
            // unclosed crop endpoints as periodic values.
            for field in ["u", "msfux", "msfuy"] {
                let arr = a.get_mut(field).unwrap();
                if arr.shape.len() == 3 {
                    for k in 0..nz {
                        for j in 0..ny {
                            let v = arr.at(k, j, 0);
                            arr.put(k, j, nx, v);
                        }
                    }
                } else {
                    for j in 0..ny {
                        arr.data[j * (nx + 1) + nx] = arr.data[j * (nx + 1)];
                    }
                }
            }
            for field in ["v", "msfvx", "msfvy"] {
                let arr = a.get_mut(field).unwrap();
                if arr.shape.len() == 3 {
                    for k in 0..nz {
                        for i in 0..nx {
                            let v = arr.at(k, 0, i);
                            arr.put(k, ny, i, v);
                        }
                    }
                } else {
                    for i in 0..nx {
                        arr.data[ny * nx + i] = arr.data[i];
                    }
                }
            }
            let mass = &a["muts"];
            let mut muf_u = a["muf_u"].clone();
            let mut muf_v = a["muf_v"].clone();
            for j in 0..ny {
                let m = 0.5f32 * (mass.xy(j, nx - 1) + mass.xy(j, 0));
                muf_u.data[j * (nx + 1)] = m;
                muf_u.data[j * (nx + 1) + nx] = m;
            }
            for i in 0..nx {
                let m = 0.5f32 * (mass.xy(ny - 1, i) + mass.xy(0, i));
                muf_v.data[i] = m;
                muf_v.data[ny * nx + i] = m;
            }
            a.insert("muf_u".into(), muf_u);
            a.insert("muf_v".into(), muf_v);
            transform.push_str("; close periodic redundant U/V and staggered map-factor end faces to first faces and wrap dry-mass face coupling across crop edges");
        } else if kind == "open" {
            // The cropped open domain copies its own boundary-cell dry mass
            // into external mass halos before coupling normal momentum.
            let mass = &a["muts"];
            let mut muf_u = a["muf_u"].clone();
            let mut muf_v = a["muf_v"].clone();
            for j in 0..ny {
                muf_u.data[j * (nx + 1)] = mass.xy(j, 0);
                muf_u.data[j * (nx + 1) + nx] = mass.xy(j, nx - 1);
            }
            for i in 0..nx {
                muf_v.data[i] = mass.xy(0, i);
                muf_v.data[ny * nx + i] = mass.xy(ny - 1, i);
            }
            a.insert("muf_u".into(), muf_u);
            a.insert("muf_v".into(), muf_v);
            transform.push_str("; close all four cropped open boundaries with zero-gradient dry-mass halos using each boundary cell mass for normal face coupling; recompute coupled RU/RV and continuity ROM without changing winds or map factors");
        }
        // FNM/FNP/RDN on WRF output have nz values, including the surface slot;
        // ArWen coordinate objects carry an additional model-top slot.
        for field in ["fnm", "fnp", "rdn"] {
            if a[field].data.len() == nz {
                let arr = a.get_mut(field).unwrap();
                arr.data.push(0.);
                arr.shape[0] = nz + 1;
            }
        }
        for field in ["c1f", "c2f"] {
            assert_eq!(a[field].data.len(), nz + 1, "{field} full-level length");
        }
        coupling(&mut a, dx, dy);
        a.insert("scalar_pd".into(), a["q0"].clone());
        if kind == "front" {
            for value in &mut a.get_mut("scalar_pd").unwrap().data {
                *value *= 1.05f32;
            }
        }
        for (src, dst) in [
            ("u", "u_old"),
            ("v", "v_old"),
            ("w", "w_old"),
            ("scalar", "scalar_old"),
        ] {
            a.insert(dst.into(), a[src].clone());
        }
        for (src, dst) in [
            ("u", "tend_u"),
            ("v", "tend_v"),
            ("w", "tend_w"),
            ("scalar", "tend_scalar"),
            ("q0", "tend_pd"),
        ] {
            let mut t = Array::filled(&a[src].shape, 0.);
            if kind == "map" {
                for (index, value) in t.data.iter_mut().enumerate() {
                    *value = ((index % 17) as f32 - 8f32) * 0.001f32;
                }
            }
            a.insert(dst.into(), t);
        }
        let terrain_min = a["terrain"]
            .data
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let terrain_max = a["terrain"]
            .data
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut crop_gradient = 0f32;
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let gradient = (a["terrain"].xy(j, i + 1) - a["terrain"].xy(j, i)).abs()
                    + (a["terrain"].xy(j + 1, i) - a["terrain"].xy(j, i)).abs();
                crop_gradient = crop_gradient.max(gradient);
            }
        }
        let meta = format!(
            "{{\"case_id\":\"{name}\",\"nx\":{nx},\"ny\":{ny},\"nz\":{nz},\"dx\":{dx},\"dy\":{dy},\"dt\":6.0,\"open_x\":{open},\"open_y\":{open},\"specified\":{specified},\"source_file\":\"{}\",\"source_sha256\":\"{hash}\",\"source_crop_x\":{x},\"source_crop_y\":{y},\"terrain_gradient_max_m\":{crop_gradient},\"source_terrain_selection_gradient_m\":{max_gradient},\"terrain_min_m\":{terrain_min},\"terrain_max_m\":{terrain_max},\"transform\":\"{transform}\",\"flux_provenance\":\"RU/RV generated by float32 hybrid dry-mass coupling and isotropic map-factor division; original source neighbors retained at cropped faces; ROM diagnosed from calc_ww_cp continuity; ROM is not read from WRF output\",\"halo_provenance\":\"source crop has natural staggering; oracle adapter owns halo construction\"}}",
            Path::new(&args[1]).file_name().unwrap().to_str().unwrap()
        );
        emit(out, name, &a, &meta);
        println!("{name}: {nx}x{ny}x{nz}, source crop {x},{y}");
    }
}
