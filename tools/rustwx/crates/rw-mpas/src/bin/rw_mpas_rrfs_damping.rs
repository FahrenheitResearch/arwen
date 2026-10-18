//! RRFSx/MPAS-Model src/core_atmosphere/mpas_atm_core.F:1266-1309 at 1d6e370e.
//! Strict single-precision initialization, including GNU host libm and density scaling.
use std::io::{self, Read, Write};
#[cfg_attr(unix, link(name = "m"))]
unsafe extern "C" {
    fn sinf(x: f32) -> f32;
    fn acosf(x: f32) -> f32;
    fn powf(x: f32, y: f32) -> f32;
}
fn u64_value(input: &mut impl Read) -> io::Result<usize> {
    let mut b = [0; 8]; input.read_exact(&mut b)?;
    usize::try_from(u64::from_le_bytes(b)).map_err(|_| io::Error::other("dimension overflow"))
}
fn real(input: &mut impl Read) -> io::Result<f32> {
    let mut b = [0; 4]; input.read_exact(&mut b)?; Ok(f32::from_le_bytes(b))
}
fn run() -> io::Result<()> {
    let mut input = io::stdin().lock(); let mut output = io::stdout().lock();
    let mut magic = [0; 8]; input.read_exact(&mut magic)?;
    if &magic != b"RRFSDSS1" { return Err(io::Error::other("expected RRFSDSS1 ABI")); }
    let nl = u64_value(&mut input)?; let nc = u64_value(&mut input)?;
    let xnutr = real(&mut input)?; let zd = real(&mut input)?;
    let size = (nl + 1).checked_mul(nc).ok_or_else(|| io::Error::other("dimension overflow"))?;
    let mut zgrid = Vec::with_capacity(size);
    for _ in 0..size { zgrid.push(real(&mut input)?); }
    let mut density = Vec::with_capacity(nc);
    for _ in 0..nc { density.push(real(&mut input)?); }
    let pii = unsafe { acosf(-1.0) };
    let mut dss = vec![0.0f32; nl * nc];
    for c in 0..nc {
        let zt = zgrid[nl * nc + c];
        for k in 0..nl {
            let z = 0.5 * (zgrid[k * nc + c] + zgrid[(k + 1) * nc + c]);
            if z > zd {
                let phase = ((0.5 * pii) * (z - zd)) / (zt - zd);
                let sine = unsafe { sinf(phase) };
                let value = xnutr * (sine * sine);
                dss[k * nc + c] = value / unsafe { powf(density[c], 0.25) };
            }
        }
    }
    for value in dss { output.write_all(&value.to_le_bytes())?; }
    Ok(())
}
fn main() {
    if let Err(error) = run() { eprintln!("RRFS damping: {error}"); std::process::exit(1); }
}
