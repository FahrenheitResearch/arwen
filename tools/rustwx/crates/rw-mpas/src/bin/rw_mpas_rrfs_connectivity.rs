//! Complete ordered second-ring initialization for the pinned fork core.
//! MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:1481-1584 at 1d6e370e.
//! Wire indices are zero-based; native pool conversion remains the caller's.
use std::io::{self, Read, Write};

fn word(r: &mut impl Read) -> io::Result<i32> {
    let mut bytes = [0; 4]; r.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}
fn dimension(r: &mut impl Read) -> io::Result<usize> {
    let mut bytes = [0; 8]; r.read_exact(&mut bytes)?;
    usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| io::Error::other("dimension overflow"))
}
fn run() -> io::Result<()> {
    let mut input = io::stdin().lock(); let mut output = io::stdout().lock();
    let mut magic = [0; 8]; input.read_exact(&mut magic)?;
    let preserve = &magic == b"RRFSN2V2";
    if !preserve && &magic != b"RRFSN2V1" { return Err(io::Error::other("expected RRFSN2V1/V2 ABI")); }
    let cells = dimension(&mut input)?; let solved = dimension(&mut input)?;
    let width = dimension(&mut input)?; let outer_width = dimension(&mut input)?;
    let counts: Vec<_> = (0..cells).map(|_| word(&mut input)).collect::<io::Result<_>>()?;
    let neighbors: Vec<_> = (0..cells*width).map(|_| word(&mut input)).collect::<io::Result<_>>()?;
    // The fork overwrites only solved counts and the visited outer-row prefix.
    // V2 carries real pool entry storage, including padding and unsolved rows.
    let mut outer_count = if preserve {
        (0..cells).map(|_| word(&mut input)).collect::<io::Result<Vec<_>>>()?
    } else { vec![0i32; cells] };
    let mut outer = if preserve {
        (0..cells*outer_width).map(|_| word(&mut input)).collect::<io::Result<Vec<_>>>()?
    } else { vec![-1i32; cells*outer_width] };
    if solved > cells { return Err(io::Error::other("invalid native solve extent")); }
    for cell in 0..solved {
        let count = counts[cell] as usize;
        if count > width { return Err(io::Error::other("native active count exceeds row storage")); }
        let mut group = vec![cell as i32];
        group.extend_from_slice(&neighbors[cell*width..cell*width+count]);
        let inner = group.len();
        for edge in 0..count {
            let adjacent = neighbors[cell*width+edge] as usize;
            if adjacent >= cells { return Err(io::Error::other("native neighbor pool row is absent")); }
            if counts[adjacent] as usize > width { return Err(io::Error::other("native adjacent count exceeds row storage")); }
            for slot in 0..counts[adjacent] as usize {
                let candidate = neighbors[adjacent*width+slot];
                if !group.contains(&candidate) { group.push(candidate); }
            }
        }
        let outside = group.len() - inner;
        // The native core itself terminates when MaxEdges2 < nOuterRow.
        if outside > outer_width { return Err(io::Error::other("native MaxEdges2 < nOuterRow")); }
        outer[cell*outer_width..cell*outer_width+outside].copy_from_slice(&group[inner..]);
        outer_count[cell] = outside as i32;
    }
    for value in outer_count.into_iter().chain(outer) { output.write_all(&value.to_le_bytes())?; }
    Ok(())
}
fn main() {
    if let Err(error) = run() { eprintln!("RRFS connectivity: {error}"); std::process::exit(1); }
}
