//! rw-libm verification harness (run on node-2, glibc 2.43; verification.log).
//!
//! Build: a binary crate with this file as src/main.rs, depending on
//! `rw-libm = { path = ".../tools/rustwx/crates/rw-libm" }`, and a build.rs
//! that prints `cargo:rustc-link-search=native=$B/cref`,
//! `cargo:rustc-link-arg=-l:libmpfr.so.6` and `cargo:rustc-link-arg=-lgmp`
//! after build_cref.sh has made $B/cref/libcmref.a and libmpref.a.  Run
//! `harness uni`, `harness bi 1000000000`, `harness f64 1000000000` and
//! `harness polar 1000000000` with CORE_MATH_SRC=$B/core-math/src.
//!
//! univariate binary32: every one of the 2^32 inputs, comparing the Rust
//! port with CORE-MATH's C (built without contraction and with native FMA
//! contraction) and with the running glibc; every glibc disagreement is
//! adjudicated by MPFR.  bivariate binary32 and binary64 asin/acos: the
//! CORE-MATH worst-case files, a special-value grid and N random inputs.
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[link(name = "cmref", kind = "static")]
unsafe extern "C" {
    fn cr_sinf(x: f32) -> f32;
    fn cr_cosf(x: f32) -> f32;
    fn cr_tanf(x: f32) -> f32;
    fn cr_atanf(x: f32) -> f32;
    fn cr_expf(x: f32) -> f32;
    fn cr_logf(x: f32) -> f32;
    fn cr_log10f(x: f32) -> f32;
    fn cr_powf(x: f32, y: f32) -> f32;
    fn cr_atan2f(y: f32, x: f32) -> f32;
    fn cr_asin(x: f64) -> f64;
    fn cr_acos(x: f64) -> f64;
    fn crf_sinf(x: f32) -> f32;
    fn crf_cosf(x: f32) -> f32;
    fn crf_tanf(x: f32) -> f32;
    fn crf_atanf(x: f32) -> f32;
    fn crf_expf(x: f32) -> f32;
    fn crf_logf(x: f32) -> f32;
    fn crf_log10f(x: f32) -> f32;
    fn crf_powf(x: f32, y: f32) -> f32;
    fn crf_atan2f(y: f32, x: f32) -> f32;
    fn crf_asin(x: f64) -> f64;
    fn crf_acos(x: f64) -> f64;
}
#[link(name = "mpref", kind = "static")]
unsafe extern "C" {
    fn ref1f(op: i32, x: f32) -> f32;
    fn ref2f(op: i32, a: f32, b: f32) -> f32;
    fn ref1d(op: i32, x: f64) -> f64;
}
#[link(name = "m")]
unsafe extern "C" {
    fn sinf(x: f32) -> f32;
    fn cosf(x: f32) -> f32;
    fn tanf(x: f32) -> f32;
    fn atanf(x: f32) -> f32;
    fn expf(x: f32) -> f32;
    fn logf(x: f32) -> f32;
    fn log10f(x: f32) -> f32;
    fn powf(x: f32, y: f32) -> f32;
    fn atan2f(y: f32, x: f32) -> f32;
    fn asin(x: f64) -> f64;
    fn acos(x: f64) -> f64;
    fn strtod(s: *const i8, end: *mut *mut i8) -> f64;
    fn strtof(s: *const i8, end: *mut *mut i8) -> f32;
    fn gnu_get_libc_version() -> *const i8;
}

fn same32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}
fn same64(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

struct Uni {
    name: &'static str,
    rust: fn(f32) -> f32,
    cref: unsafe extern "C" fn(f32) -> f32,
    cfma: unsafe extern "C" fn(f32) -> f32,
    glibc: unsafe extern "C" fn(f32) -> f32,
    mp: i32,
}

#[derive(Default, Clone, Copy)]
struct Counts {
    n: u64,
    vs_cref: u64,
    vs_cfma: u64,
    vs_glibc: u64,
    rust_wrong: u64,
    glibc_wrong: u64,
    checked_mpfr: u64,
}
impl Counts {
    fn add(&mut self, o: &Counts) {
        self.n += o.n;
        self.vs_cref += o.vs_cref;
        self.vs_cfma += o.vs_cfma;
        self.vs_glibc += o.vs_glibc;
        self.rust_wrong += o.rust_wrong;
        self.glibc_wrong += o.glibc_wrong;
        self.checked_mpfr += o.checked_mpfr;
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)
}

fn exhaustive(u: &Uni, examples: &Mutex<Vec<String>>) -> Counts {
    const CHUNK: u64 = 1 << 20;
    let next = AtomicU64::new(0);
    let total = Mutex::new(Counts::default());
    std::thread::scope(|s| {
        for _ in 0..threads() {
            s.spawn(|| {
                let mut c = Counts::default();
                loop {
                    let start = next.fetch_add(CHUNK, Ordering::Relaxed);
                    if start >= 1u64 << 32 {
                        break;
                    }
                    for b in start..start + CHUNK {
                        let x = f32::from_bits(b as u32);
                        let r = (u.rust)(x);
                        let cr = unsafe { (u.cref)(x) };
                        let cf = unsafe { (u.cfma)(x) };
                        let g = unsafe { (u.glibc)(x) };
                        c.n += 1;
                        if !same32(r, cr) {
                            c.vs_cref += 1;
                            let mut e = examples.lock().unwrap();
                            if e.len() < 40 {
                                e.push(format!("{} x={:#010x} rust={:#010x} cref={:#010x}", u.name, b, r.to_bits(), cr.to_bits()));
                            }
                        }
                        if !same32(r, cf) {
                            c.vs_cfma += 1;
                        }
                        if !same32(r, g) || !same32(r, cr) {
                            if !same32(r, g) {
                                c.vs_glibc += 1;
                            }
                            let m = unsafe { ref1f(u.mp, x) };
                            c.checked_mpfr += 1;
                            if !same32(r, m) {
                                c.rust_wrong += 1;
                                let mut e = examples.lock().unwrap();
                                if e.len() < 40 {
                                    e.push(format!("{} RUST WRONG x={:#010x} rust={:#010x} mpfr={:#010x}", u.name, b, r.to_bits(), m.to_bits()));
                                }
                            }
                            if !same32(g, m) {
                                c.glibc_wrong += 1;
                            }
                        }
                    }
                }
                total.lock().unwrap().add(&c);
            });
        }
    });
    total.into_inner().unwrap()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

fn read_wc(path: &str, two: bool, double: bool) -> Vec<(f64, f64)> {
    let text = std::fs::read_to_string(path).expect(path);
    let mut out = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let l = l.split('#').next().unwrap().trim();
        let parts: Vec<&str> = l.split(',').map(|p| p.trim()).collect();
        let parse = |s: &str| -> f64 {
            let c = CString::new(s).unwrap();
            unsafe { if double { strtod(c.as_ptr(), std::ptr::null_mut()) } else { strtof(c.as_ptr(), std::ptr::null_mut()) as f64 } }
        };
        if two {
            if parts.len() >= 2 {
                out.push((parse(parts[0]), parse(parts[1])));
            }
        } else if !parts.is_empty() {
            out.push((parse(parts[0]), 0.0));
        }
    }
    out
}

fn specials32() -> Vec<f32> {
    let mut v: Vec<f32> = vec![0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::MIN_POSITIVE, f32::MAX, f32::EPSILON];
    v.push(f32::from_bits(1));
    v.push(f32::from_bits(0x007f_ffff));
    v.push(f32::from_bits(0x7f80_0001)); // signalling NaN
    for k in -150..=128 {
        v.push(2f32.powi(k));
    }
    for n in -40..=40 {
        v.push(n as f32);
        v.push(n as f32 + 0.5);
        v.push(n as f32 * 0.25);
    }
    for b in [0x3f7f_ffffu32, 0x3f80_0001, 0x3f7f_fffe, 0x3f80_0002, 0x4b7f_ffff, 0x4b80_0000, 0x4b80_0001, 0x4f00_0000, 0x5f00_0000] {
        v.push(f32::from_bits(b));
    }
    let n = v.len();
    for i in 0..n {
        v.push(-v[i]);
    }
    v
}

#[derive(Clone, Copy)]
struct Bi {
    name: &'static str,
    rust: fn(f32, f32) -> f32,
    cref: unsafe extern "C" fn(f32, f32) -> f32,
    cfma: unsafe extern "C" fn(f32, f32) -> f32,
    glibc: unsafe extern "C" fn(f32, f32) -> f32,
    mp: i32,
}

fn check_pair(f: &Bi, a: f32, b: f32, always_mpfr: bool, c: &mut Counts, ex: &Mutex<Vec<String>>) {
    let r = (f.rust)(a, b);
    let cr = unsafe { (f.cref)(a, b) };
    let cf = unsafe { (f.cfma)(a, b) };
    let g = unsafe { (f.glibc)(a, b) };
    c.n += 1;
    if !same32(r, cr) {
        c.vs_cref += 1;
        let mut e = ex.lock().unwrap();
        if e.len() < 40 {
            e.push(format!("{} a={:e} ({:#010x}) b={:e} ({:#010x}) rust={:#010x} cref={:#010x}", f.name, a, a.to_bits(), b, b.to_bits(), r.to_bits(), cr.to_bits()));
        }
    }
    if !same32(r, cf) {
        c.vs_cfma += 1;
    }
    let differs = !same32(r, g);
    if differs {
        c.vs_glibc += 1;
    }
    if differs || always_mpfr || !same32(r, cr) {
        let m = unsafe { ref2f(f.mp, a, b) };
        c.checked_mpfr += 1;
        if !same32(r, m) {
            c.rust_wrong += 1;
            let mut e = ex.lock().unwrap();
            if e.len() < 40 {
                e.push(format!("{} RUST WRONG a={:#010x} b={:#010x} rust={:#010x} mpfr={:#010x}", f.name, a.to_bits(), b.to_bits(), r.to_bits(), m.to_bits()));
            }
        }
        if !same32(g, m) {
            c.glibc_wrong += 1;
        }
    }
}

fn bivariate(f: Bi, wc: &[(f64, f64)], swap_wc: bool, n_random: u64, ex: &Mutex<Vec<String>>) -> (Counts, Counts, Counts) {
    // worst cases, every one adjudicated by MPFR
    let mut wcc = Counts::default();
    for &(p, q) in wc {
        let (a, b) = if swap_wc { (q as f32, p as f32) } else { (p as f32, q as f32) };
        check_pair(&f, a, b, true, &mut wcc, ex);
    }
    // special-value grid
    let sp = specials32();
    let mut spc = Counts::default();
    for &a in &sp {
        for &b in &sp {
            check_pair(&f, a, b, true, &mut spc, ex);
        }
    }
    // random pairs in four strategies
    let total = Mutex::new(Counts::default());
    let next = AtomicU64::new(0);
    const CHUNK: u64 = 1 << 18;
    std::thread::scope(|s| {
        for t in 0..threads() {
            let total = &total;
            let next = &next;
            s.spawn(move || {
                let mut c = Counts::default();
                let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (t as u64 + 1).wrapping_mul(0xd1b5_4a32_d192_ed03));
                loop {
                    let start = next.fetch_add(CHUNK, Ordering::Relaxed);
                    if start >= n_random {
                        break;
                    }
                    for i in start..(start + CHUNK).min(n_random) {
                        let (a, b) = match (i % 4, f.mp) {
                            (0, _) => (f32::from_bits(rng.next() as u32), f32::from_bits((rng.next() >> 32) as u32)),
                            // pow: x positive anywhere, y aimed at a finite result
                            (1, 0) => {
                                let x = f32::from_bits((rng.next() as u32) & 0x7f7f_ffff);
                                let l2 = (x as f64).log2();
                                let target = -150.0 + 280.0 * rng.unit();
                                let y = if l2 != 0.0 { (target / l2) as f32 } else { 1.0 };
                                (x, y)
                            }
                            // pow: x near 1, large y
                            (2, 0) => {
                                let x = (1.0 + (rng.unit() - 0.5) * 2f64.powi(-((rng.next() % 24) as i32))) as f32;
                                let y = ((rng.unit() - 0.5) * 2f64.powi((rng.next() % 30) as i32)) as f32;
                                (x, y)
                            }
                            // pow: negative x with integer y, and moderate values
                            (3, 0) => {
                                let x = -(rng.unit() * 8.0) as f32;
                                let y = ((rng.next() % 61) as i32 - 30) as f32;
                                (x, y)
                            }
                            // atan2: comparable magnitudes
                            (1, _) => {
                                let ey = (rng.next() % 200) as i32 - 100;
                                let ex = ey + (rng.next() % 61) as i32 - 30;
                                let y = ((rng.unit() * 2.0 - 1.0) * 2f64.powi(ey)) as f32;
                                let x = ((rng.unit() * 2.0 - 1.0) * 2f64.powi(ex)) as f32;
                                (y, x)
                            }
                            // atan2: tiny y/x
                            (2, _) => {
                                let ey = (rng.next() % 200) as i32 - 120;
                                let ex = ey + 24 + (rng.next() % 40) as i32;
                                let y = ((rng.unit() * 2.0 - 1.0) * 2f64.powi(ey)) as f32;
                                let x = ((rng.unit() * 2.0 - 1.0) * 2f64.powi(ex)) as f32;
                                (y, x)
                            }
                            // atan2: unit-scale geography (the orographic use)
                            _ => (((rng.unit() * 2.0 - 1.0) * 3000.0) as f32, ((rng.unit() * 2.0 - 1.0) * 3000.0) as f32),
                        };
                        check_pair(&f, a, b, i % 97 == 0, &mut c, ex);
                    }
                }
                total.lock().unwrap().add(&c);
            });
        }
    });
    (wcc, spc, total.into_inner().unwrap())
}

#[derive(Clone, Copy)]
struct D1 {
    name: &'static str,
    rust: fn(f64) -> f64,
    cref: unsafe extern "C" fn(f64) -> f64,
    cfma: unsafe extern "C" fn(f64) -> f64,
    glibc: unsafe extern "C" fn(f64) -> f64,
    mp: i32,
}

fn check_d(f: &D1, x: f64, always_mpfr: bool, c: &mut Counts, ex: &Mutex<Vec<String>>) {
    let r = (f.rust)(x);
    let cr = unsafe { (f.cref)(x) };
    let cf = unsafe { (f.cfma)(x) };
    let g = unsafe { (f.glibc)(x) };
    c.n += 1;
    if !same64(r, cr) {
        c.vs_cref += 1;
        let mut e = ex.lock().unwrap();
        if e.len() < 40 {
            e.push(format!("{} x={:e} ({:#018x}) rust={:#018x} cref={:#018x}", f.name, x, x.to_bits(), r.to_bits(), cr.to_bits()));
        }
    }
    if !same64(r, cf) {
        c.vs_cfma += 1;
    }
    let differs = !same64(r, g);
    if differs {
        c.vs_glibc += 1;
    }
    if differs || always_mpfr || !same64(r, cr) {
        let m = unsafe { ref1d(f.mp, x) };
        c.checked_mpfr += 1;
        if !same64(r, m) {
            c.rust_wrong += 1;
            let mut e = ex.lock().unwrap();
            if e.len() < 40 {
                e.push(format!("{} RUST WRONG x={:#018x} rust={:#018x} mpfr={:#018x}", f.name, x.to_bits(), r.to_bits(), m.to_bits()));
            }
        }
        if !same64(g, m) {
            c.glibc_wrong += 1;
        }
    }
}

fn double1(f: D1, wc: &[(f64, f64)], n_random: u64, ex: &Mutex<Vec<String>>) -> (Counts, Counts) {
    let mut wcc = Counts::default();
    for &(x, _) in wc {
        check_d(&f, x, true, &mut wcc, ex);
        check_d(&f, -x, true, &mut wcc, ex);
    }
    let total = Mutex::new(Counts::default());
    let next = AtomicU64::new(0);
    const CHUNK: u64 = 1 << 18;
    std::thread::scope(|s| {
        for t in 0..threads() {
            let total = &total;
            let next = &next;
            s.spawn(move || {
                let mut c = Counts::default();
                let mut rng = Rng(0x2545_f491_4f6c_dd1d ^ (t as u64 + 7).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                loop {
                    let start = next.fetch_add(CHUNK, Ordering::Relaxed);
                    if start >= n_random {
                        break;
                    }
                    for i in start..(start + CHUNK).min(n_random) {
                        let x = match i % 4 {
                            // every encoding with |x| <= 1, uniformly over bit patterns
                            0 => f64::from_bits(rng.next() % 0x3ff0_0000_0000_0001 | (rng.next() & (1 << 63))),
                            // uniform in value over [-1, 1]
                            1 => rng.unit() * 2.0 - 1.0,
                            // close to +-1
                            2 => (1.0 - rng.unit() * 2f64.powi(-((rng.next() % 50) as i32))) * if rng.next() & 1 == 0 { 1.0 } else { -1.0 },
                            // the polar projection's use: a ratio of f32 squares
                            _ => {
                                let a = (rng.unit() * 4000.0) as f32;
                                let b = (rng.unit() * 4000.0) as f32;
                                let (a2, b2) = ((a * a) as f64, (b * b) as f64);
                                (a2 - b2) / (a2 + b2)
                            }
                        };
                        check_d(&f, x, i % 97 == 0, &mut c, ex);
                    }
                }
                total.lock().unwrap().add(&c);
            });
        }
    });
    (wcc, total.into_inner().unwrap())
}

fn row(label: &str, c: &Counts) {
    println!(
        "{:<26} n={:>14} rust!=cref(nofma)={:<6} rust!=cref(fma)={:<6} rust!=glibc={:<10} mpfr-checked={:<10} rust-not-CR={:<4} glibc-not-CR={}",
        label, c.n, c.vs_cref, c.vs_cfma, c.vs_glibc, c.checked_mpfr, c.rust_wrong, c.glibc_wrong
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let what = args.get(1).map(String::as_str).unwrap_or("all");
    let n_random: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_000_000_000);
    let wcdir = std::env::var("CORE_MATH_SRC").expect("set CORE_MATH_SRC to core-math/src");
    let v = unsafe { std::ffi::CStr::from_ptr(gnu_get_libc_version()) };
    println!("glibc {} threads {}", v.to_str().unwrap(), threads());
    let ex = Mutex::new(Vec::new());
    if what == "polar" {
        // The polar projection casts (DEG*h)*asin(t) and acos(c) to binary32:
        // how often does the glibc asin/acos (not correctly rounded) change that?
        let deg = (180.0f32 / std::f32::consts::PI) as f64;
        let total = AtomicU64::new(0);
        let dlat = AtomicU64::new(0);
        let dlon = AtomicU64::new(0);
        let draws: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_000_000_000);
        std::thread::scope(|s| {
            for t in 0..threads() {
                let (total, dlat, dlon) = (&total, &dlat, &dlon);
                s.spawn(move || {
                    let mut rng = Rng(0x1234_5678_9abc_def1 ^ (t as u64 + 3).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                    let per = draws / threads() as u64;
                    let (mut a, mut b) = (0u64, 0u64);
                    for _ in 0..per {
                        let xx = ((rng.unit() * 2.0 - 1.0) * 4000.0) as f32;
                        let yy = ((rng.unit() * 2.0 - 1.0) * 4000.0) as f32;
                        let scale = (1000.0 + rng.unit() * 9000.0) as f32;
                        let r2 = (xx * xx + yy * yy) as f64;
                        let gi2 = (scale * scale) as f64;
                        let tt = (gi2 - r2) / (gi2 + r2);
                        if (deg * rw_libm::asin(tt)) as f32 != (deg * unsafe { asin(tt) }) as f32 { a += 1; }
                        let c = ((xx as f64) / r2.sqrt()).clamp(-1.0, 1.0);
                        if rw_libm::acos(c) as f32 != unsafe { acos(c) } as f32 { b += 1; }
                    }
                    total.fetch_add(per, Ordering::Relaxed);
                    dlat.fetch_add(a, Ordering::Relaxed);
                    dlon.fetch_add(b, Ordering::Relaxed);
                });
            }
        });
        println!("polar casts: {} draws, latitude differs {} times, acos-angle differs {} times",
            total.load(Ordering::Relaxed), dlat.load(Ordering::Relaxed), dlon.load(Ordering::Relaxed));
        return;
    }
    if what == "diff" {
        let name = args.get(2).map(String::as_str).unwrap_or("expf");
        let f: (fn(f32) -> f32, unsafe extern "C" fn(f32) -> f32) = match name {
            "sinf" => (rw_libm::sinf, sinf), "cosf" => (rw_libm::cosf, cosf), "expf" => (rw_libm::expf, expf),
            "logf" => (rw_libm::logf, logf), "tanf" => (rw_libm::tanf, tanf), _ => panic!("name"),
        };
        let mut n = 0;
        for b in 0..=u32::MAX {
            let x = f32::from_bits(b);
            let r = (f.0)(x);
            let g = unsafe { (f.1)(x) };
            if !same32(r, g) {
                n += 1;
                if n <= 20 { println!("{name} x={:#010x} ({:e}) rust={:#010x} glibc={:#010x}", b, x, r.to_bits(), g.to_bits()); }
            }
        }
        println!("{name}: {n} differ");
        return;
    }
    let unis = [
        Uni { name: "sinf", rust: rw_libm::sinf, cref: cr_sinf, cfma: crf_sinf, glibc: sinf, mp: 0 },
        Uni { name: "cosf", rust: rw_libm::cosf, cref: cr_cosf, cfma: crf_cosf, glibc: cosf, mp: 1 },
        Uni { name: "tanf", rust: rw_libm::tanf, cref: cr_tanf, cfma: crf_tanf, glibc: tanf, mp: 2 },
        Uni { name: "atanf", rust: rw_libm::atanf, cref: cr_atanf, cfma: crf_atanf, glibc: atanf, mp: 3 },
        Uni { name: "expf", rust: rw_libm::expf, cref: cr_expf, cfma: crf_expf, glibc: expf, mp: 4 },
        Uni { name: "logf", rust: rw_libm::logf, cref: cr_logf, cfma: crf_logf, glibc: logf, mp: 5 },
        Uni { name: "log10f", rust: rw_libm::log10f, cref: cr_log10f, cfma: crf_log10f, glibc: log10f, mp: 6 },
    ];
    for u in &unis {
        if what == "all" || what == "uni" || what == u.name {
            let t0 = std::time::Instant::now();
            let c = exhaustive(u, &ex);
            row(&format!("{} exhaustive 2^32", u.name), &c);
            eprintln!("  ({:.1}s)", t0.elapsed().as_secs_f64());
        }
    }
    let bis = [
        (Bi { name: "powf", rust: rw_libm::powf, cref: cr_powf, cfma: crf_powf, glibc: powf, mp: 0 }, "binary32/pow/powf.wc", false),
        // atan2f.wc lists y,x; ref2f(1, a, b) = atan2(a, b)
        (Bi { name: "atan2f", rust: rw_libm::atan2f, cref: cr_atan2f, cfma: crf_atan2f, glibc: atan2f, mp: 1 }, "binary32/atan2/atan2f.wc", false),
    ];
    for (f, wcf, swap) in bis {
        if what == "all" || what == "bi" || what == f.name {
            let t0 = std::time::Instant::now();
            let wc = read_wc(&format!("{wcdir}/{wcf}"), true, false);
            let (w, s, r) = bivariate(f, &wc, swap, n_random, &ex);
            row(&format!("{} worst cases", f.name), &w);
            row(&format!("{} special grid", f.name), &s);
            row(&format!("{} random pairs", f.name), &r);
            eprintln!("  ({:.1}s)", t0.elapsed().as_secs_f64());
        }
    }
    let ds = [
        (D1 { name: "asin", rust: rw_libm::asin, cref: cr_asin, cfma: crf_asin, glibc: asin, mp: 0 }, "binary64/asin/asin.wc"),
        (D1 { name: "acos", rust: rw_libm::acos, cref: cr_acos, cfma: crf_acos, glibc: acos, mp: 1 }, "binary64/acos/acos.wc"),
    ];
    for (f, wcf) in ds {
        if what == "all" || what == "f64" || what == f.name {
            let t0 = std::time::Instant::now();
            let wc = read_wc(&format!("{wcdir}/{wcf}"), false, true);
            let (w, r) = double1(f, &wc, n_random, &ex);
            row(&format!("{} worst cases (+/-)", f.name), &w);
            row(&format!("{} random", f.name), &r);
            eprintln!("  ({:.1}s)", t0.elapsed().as_secs_f64());
        }
    }
    let e = ex.into_inner().unwrap();
    if !e.is_empty() {
        println!("examples:");
        for l in e {
            println!("  {l}");
        }
    }
}
