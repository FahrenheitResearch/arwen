//! Ordered chemistry diagnostics for ordinary host-array history output.
//! Each operation rounds separately to f32, matching the CUDA diagnostic
//! kernels and WRF chem/module_gocart_aerosols.F:112-144.
use std::slice;

/// The caller supplies disjoint contiguous fields and an output buffer.
/// kind: 0 = volume term sum, 1 = lowest-level sum, 2 = column integral.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_chem_diagnostic_f32(
    fields: *const *const f32, nfields: usize,
    species: *const i32, starts: *const i32, opstarts: *const i32,
    codes: *const i32, operands: *const f32, nterms: usize,
    alt: *const f32, dz8w: *const f32, nz: usize, ncol: usize,
    kind: u32, divide_by_alt: u8, column_scale: f32, output: *mut f32,
) -> i32 {
    if fields.is_null() || species.is_null() || starts.is_null()
        || opstarts.is_null() || codes.is_null() || operands.is_null()
        || alt.is_null() || output.is_null() || (kind == 2 && dz8w.is_null()) {
        return 1;
    }
    let Some(cells) = nz.checked_mul(ncol) else { return 2; };
    if nfields == 0 || nterms == 0 || nz == 0 || ncol == 0 || kind > 2
        || divide_by_alt > 1 || cells > (isize::MAX as usize) / 4
        || nfields > (isize::MAX as usize) / std::mem::size_of::<*const f32>()
        || nterms >= (isize::MAX as usize) / 4 {
        return 2;
    }
    let fields = slice::from_raw_parts(fields, nfields);
    if fields.iter().any(|field| field.is_null()) { return 1; }
    let starts = slice::from_raw_parts(starts, nterms + 1);
    let opstarts = slice::from_raw_parts(opstarts, nterms + 1);
    if starts[0] != 0 || opstarts[0] != 0
        || starts.windows(2).any(|w| w[0] < 0 || w[1] <= w[0])
        || opstarts.windows(2).any(|w| w[0] < 0 || w[1] < w[0]) {
        return 3;
    }
    let species = slice::from_raw_parts(species, starts[nterms] as usize);
    let codes = slice::from_raw_parts(codes, opstarts[nterms] as usize);
    let operands = slice::from_raw_parts(operands, codes.len());
    if species.iter().any(|&slot| slot < 0 || slot as usize >= nfields)
        || codes.iter().any(|&code| code != 0 && code != 1) {
        return 3;
    }
    let alt = slice::from_raw_parts(alt, cells);
    let dz8w = if kind == 2 { Some(slice::from_raw_parts(dz8w, cells)) } else { None };
    let fields: Vec<_> = fields.iter().map(|&field| slice::from_raw_parts(field, cells)).collect();
    let output = slice::from_raw_parts_mut(output, if kind == 0 { cells } else { ncol });
    let level = |index: usize| {
        let mut total = 0.0_f32;
        for term in 0..nterms {
            let slots = &species[starts[term] as usize..starts[term + 1] as usize];
            let mut value = fields[slots[0] as usize][index];
            for &slot in &slots[1..] { value += fields[slot as usize][index]; }
            for operation in opstarts[term] as usize..opstarts[term + 1] as usize {
                value = if codes[operation] == 0 { value * operands[operation] }
                    else { value / operands[operation] };
            }
            total += value;
        }
        if divide_by_alt == 1 { total /= alt[index]; }
        total
    };
    for (index, value) in output.iter_mut().enumerate() {
        *value = if kind == 2 {
            let mut sum = 0.0_f32;
            for k in 0..nz {
                let cell = k * ncol + index;
                let dry_mass = (1.0_f32 / alt[cell]) * dz8w.unwrap()[cell];
                sum += level(cell) * dry_mass;
            }
            sum * column_scale
        } else { level(index) };
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_terms_and_dry_column_mass_preserve_float_words() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [10.0_f32, 20.0, 30.0, 40.0];
        let fields = [a.as_ptr(), b.as_ptr()];
        let species = [0_i32, 1];
        let starts = [0_i32, 2];
        let opstarts = [0_i32, 2];
        let codes = [0_i32, 1];
        let operands = [3.0_f32, 7.0];
        let alt = [0.7_f32, 0.8, 0.9, 1.0];
        let dz = [100.0_f32, 200.0, 300.0, 400.0];
        for kind in 0..=2 {
            let mut output = [0.0_f32; 4];
            assert_eq!(unsafe { gpuwm_chem_diagnostic_f32(
                fields.as_ptr(), 2, species.as_ptr(), starts.as_ptr(), opstarts.as_ptr(),
                codes.as_ptr(), operands.as_ptr(), 1, alt.as_ptr(), dz.as_ptr(),
                2, 2, kind, 1, 0.001, output.as_mut_ptr()) }, 0);
            let at = |i: usize| ((((a[i] + b[i]) * 3.0_f32) / 7.0_f32) + 0.0_f32) / alt[i];
            for i in 0..if kind == 0 { 4 } else { 2 } {
                let expected = if kind == 2 {
                    let first = at(i) * ((1.0_f32 / alt[i]) * dz[i]);
                    let second = at(i + 2) * ((1.0_f32 / alt[i + 2]) * dz[i + 2]);
                    ((0.0_f32 + first) + second) * 0.001_f32
                } else { at(i) };
                assert_eq!(output[i].to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn invalid_term_index_refuses_before_output_changes() {
        let values = [1.0_f32];
        let fields = [values.as_ptr()];
        let species = [1_i32];
        let starts = [0_i32, 1];
        let opstarts = [0_i32, 0];
        let mut output = [42.0_f32];
        assert_eq!(unsafe { gpuwm_chem_diagnostic_f32(
            fields.as_ptr(), 1, species.as_ptr(), starts.as_ptr(), opstarts.as_ptr(),
            opstarts.as_ptr(), values.as_ptr(), 1, values.as_ptr(), std::ptr::null(),
            1, 1, 0, 0, 1.0, output.as_mut_ptr()) }, 3);
        assert_eq!(output, [42.0]);
    }
}
