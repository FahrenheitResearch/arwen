//! Round-trip: every file this crate writes is read back by an INDEPENDENT
//! classic reader (`netcdf-reader` 0.3, a different author and a different
//! parse), and the structure and the values must come back unchanged.
//!
//! The point of using a foreign reader rather than a mirror parser of our
//! own is that a shared misreading of the spec cannot make a broken file
//! look correct.  The acceptance scripts apply a THIRD reader on top of
//! this one: netCDF4-python, i.e. the Unidata C library itself.

use netcdf_reader::{NcAttrValue, NcFile, NcFormat as RFormat, NcType as RType};
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};

/// `netcdf-reader`'s attribute enum carries no `PartialEq`, so compare
/// through `Debug`: it prints the variant AND the values, which is
/// exactly the "same type, same numbers" claim being made.
fn attr_eq(got: &NcAttrValue, want: &NcAttrValue) {
    assert_eq!(format!("{got:?}"), format!("{want:?}"));
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("netcdf-writer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{name}.nc"))
}

// --------------------------------------------------------------------------
// Every classic external type survives a write/read cycle.
// --------------------------------------------------------------------------

#[test]
fn cdf2_every_classic_type_roundtrips() {
    let path = tmp("cdf2_types");
    let mut schema = Schema::new(NcFormat::Offset64);
    let n = schema.def_dim("n", 3, false).unwrap();

    let vb = schema.def_var("vbyte", NcType::Byte, &[n]).unwrap();
    let vc = schema.def_var("vchar", NcType::Char, &[n]).unwrap();
    let vs = schema.def_var("vshort", NcType::Short, &[n]).unwrap();
    let vi = schema.def_var("vint", NcType::Int, &[n]).unwrap();
    let vf = schema.def_var("vfloat", NcType::Float, &[n]).unwrap();
    let vd = schema.def_var("vdouble", NcType::Double, &[n]).unwrap();

    let mut w = NcWriter::create(&path, schema).unwrap();
    w.write_var(vb, VarData::I8(&[-1, 0, 127])).unwrap();
    w.write_var(vc, VarData::Char(b"abc")).unwrap();
    w.write_var(vs, VarData::I16(&[-32768, 0, 32767])).unwrap();
    w.write_var(vi, VarData::I32(&[i32::MIN, 0, i32::MAX])).unwrap();
    w.write_var(vf, VarData::F32(&[1.5, -2.25, f32::NAN])).unwrap();
    w.write_var(vd, VarData::F64(&[1e300, -0.0, 3.25])).unwrap();
    w.finish().unwrap();

    let f = NcFile::open(&path).unwrap();
    assert_eq!(f.format(), RFormat::Offset64);
    assert_eq!(f.variables().unwrap().len(), 6);

    assert_eq!(f.variable("vbyte").unwrap().dtype, RType::Byte);
    assert_eq!(f.variable("vchar").unwrap().dtype, RType::Char);
    assert_eq!(f.variable("vshort").unwrap().dtype, RType::Short);
    assert_eq!(f.variable("vint").unwrap().dtype, RType::Int);
    assert_eq!(f.variable("vfloat").unwrap().dtype, RType::Float);
    assert_eq!(f.variable("vdouble").unwrap().dtype, RType::Double);

    assert_eq!(
        f.read_variable::<i8>("vbyte").unwrap().into_raw_vec_and_offset().0,
        vec![-1i8, 0, 127]
    );
    assert_eq!(f.read_variable_as_string("vchar").unwrap(), "abc");
    assert_eq!(
        f.read_variable::<i16>("vshort").unwrap().into_raw_vec_and_offset().0,
        vec![-32768i16, 0, 32767]
    );
    assert_eq!(
        f.read_variable::<i32>("vint").unwrap().into_raw_vec_and_offset().0,
        vec![i32::MIN, 0, i32::MAX]
    );
    let got_f = f.read_variable::<f32>("vfloat").unwrap().into_raw_vec_and_offset().0;
    assert_eq!(got_f[0], 1.5);
    assert_eq!(got_f[1], -2.25);
    assert!(got_f[2].is_nan());
    let got_d = f.read_variable::<f64>("vdouble").unwrap().into_raw_vec_and_offset().0;
    assert_eq!(got_d, vec![1e300, -0.0, 3.25]);
    // -0.0 must stay NEGATIVE zero: a sign bit lost in a byte swap is a
    // silent numeric change no equality test would show.
    assert!(got_d[1].is_sign_negative());
}

#[test]
fn cdf5_extended_types_roundtrip() {
    let path = tmp("cdf5_types");
    let mut schema = Schema::new(NcFormat::Cdf5);
    let n = schema.def_dim("n", 2, false).unwrap();

    let vub = schema.def_var("vubyte", NcType::UByte, &[n]).unwrap();
    let vus = schema.def_var("vushort", NcType::UShort, &[n]).unwrap();
    let vui = schema.def_var("vuint", NcType::UInt, &[n]).unwrap();
    let vi64 = schema.def_var("vint64", NcType::Int64, &[n]).unwrap();
    let vu64 = schema.def_var("vuint64", NcType::UInt64, &[n]).unwrap();

    let mut w = NcWriter::create(&path, schema).unwrap();
    w.write_var(vub, VarData::U8(&[0, 255])).unwrap();
    w.write_var(vus, VarData::U16(&[0, 65535])).unwrap();
    w.write_var(vui, VarData::U32(&[0, u32::MAX])).unwrap();
    w.write_var(vi64, VarData::I64(&[i64::MIN, i64::MAX])).unwrap();
    w.write_var(vu64, VarData::U64(&[0, u64::MAX])).unwrap();
    w.finish().unwrap();

    let f = NcFile::open(&path).unwrap();
    assert_eq!(f.format(), RFormat::Cdf5);
    assert_eq!(
        f.read_variable::<u8>("vubyte").unwrap().into_raw_vec_and_offset().0,
        vec![0u8, 255]
    );
    assert_eq!(
        f.read_variable::<u16>("vushort").unwrap().into_raw_vec_and_offset().0,
        vec![0u16, 65535]
    );
    assert_eq!(
        f.read_variable::<u32>("vuint").unwrap().into_raw_vec_and_offset().0,
        vec![0u32, u32::MAX]
    );
    assert_eq!(
        f.read_variable::<i64>("vint64").unwrap().into_raw_vec_and_offset().0,
        vec![i64::MIN, i64::MAX]
    );
    assert_eq!(
        f.read_variable::<u64>("vuint64").unwrap().into_raw_vec_and_offset().0,
        vec![0u64, u64::MAX]
    );
}

#[test]
fn extended_type_is_refused_in_cdf2() {
    // The concrete breakage: NC_UINT written into a CDF-2 container is a
    // type code no CDF-1/2 reader knows.  netCDF-C fails the open; a
    // reader that guesses reads garbage.  Refuse at definition time and
    // name the format that would accept it.
    let mut schema = Schema::new(NcFormat::Offset64);
    let n = schema.def_dim("n", 2, false).unwrap();
    let err = schema.def_var("v", NcType::UInt, &[n]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("NC_UINT"), "got: {msg}");
    assert!(msg.contains("CDF-5"), "got: {msg}");
}

// --------------------------------------------------------------------------
// The record (unlimited) dimension -- the capability the seed writer had
// no way to express, and the reason a wrfout could not be written at all.
// --------------------------------------------------------------------------

#[test]
fn record_dimension_roundtrips_with_three_records() {
    let path = tmp("records");
    let mut schema = Schema::new(NcFormat::Offset64);
    let t = schema.def_dim("Time", 0, true).unwrap();
    let s = schema.def_dim("DateStrLen", 19, false).unwrap();
    let y = schema.def_dim("south_north", 2, false).unwrap();

    let times = schema.def_var("Times", NcType::Char, &[t, s]).unwrap();
    let t2 = schema.def_var("T2", NcType::Float, &[t, y]).unwrap();
    let itk = schema.def_var("ITIMESTEP", NcType::Int, &[t]).unwrap();

    let mut w = NcWriter::create(&path, schema).unwrap();
    let stamps = [
        b"2026-08-16_00:00:00",
        b"2026-08-16_01:00:00",
        b"2026-08-16_02:00:00",
    ];
    for (rec, stamp) in stamps.iter().enumerate() {
        w.write_record(rec as u64, times, VarData::Char(*stamp)).unwrap();
        w.write_record(rec as u64, t2, VarData::F32(&[280.0 + rec as f32, 281.0 + rec as f32]))
            .unwrap();
        w.write_record(rec as u64, itk, VarData::I32(&[rec as i32 * 10])).unwrap();
    }
    w.finish().unwrap();

    let f = NcFile::open(&path).unwrap();
    let time_dim = f.dimension("Time").unwrap();
    assert!(time_dim.is_unlimited, "Time must be the record dimension");
    assert_eq!(time_dim.size, 3, "numrecs must be patched to 3 at finish()");

    let times_v = f.variable("Times").unwrap();
    assert_eq!(times_v.dtype, RType::Char);
    assert_eq!(times_v.shape(), vec![3, 19]);
    // One string per record, which is how a WRF `Times` variable reads.
    assert_eq!(
        f.read_variable_as_strings("Times").unwrap(),
        vec![
            "2026-08-16_00:00:00".to_string(),
            "2026-08-16_01:00:00".to_string(),
            "2026-08-16_02:00:00".to_string(),
        ]
    );

    assert_eq!(
        f.read_variable::<f32>("T2").unwrap().into_raw_vec_and_offset().0,
        vec![280.0f32, 281.0, 281.0, 282.0, 282.0, 283.0]
    );
    assert_eq!(
        f.read_variable::<i32>("ITIMESTEP").unwrap().into_raw_vec_and_offset().0,
        vec![0, 10, 20]
    );
}

#[test]
fn record_and_fixed_variables_coexist() {
    let path = tmp("mixed");
    let mut schema = Schema::new(NcFormat::Offset64);
    let t = schema.def_dim("Time", 0, true).unwrap();
    let y = schema.def_dim("y", 2, false).unwrap();

    // Definition order deliberately interleaves the two kinds; the file
    // layout must still be "all fixed data, then the record section".
    let fixed_a = schema.def_var("HGT_M", NcType::Float, &[y]).unwrap();
    let rec_a = schema.def_var("T2", NcType::Float, &[t, y]).unwrap();
    let fixed_b = schema.def_var("XLAT_M", NcType::Float, &[y]).unwrap();

    let mut w = NcWriter::create(&path, schema).unwrap();
    w.write_var(fixed_a, VarData::F32(&[10.0, 20.0])).unwrap();
    w.write_var(fixed_b, VarData::F32(&[40.0, 41.0])).unwrap();
    w.write_record(0, rec_a, VarData::F32(&[1.0, 2.0])).unwrap();
    w.write_record(1, rec_a, VarData::F32(&[3.0, 4.0])).unwrap();
    w.finish().unwrap();

    let f = NcFile::open(&path).unwrap();
    assert_eq!(
        f.read_variable::<f32>("HGT_M").unwrap().into_raw_vec_and_offset().0,
        vec![10.0f32, 20.0]
    );
    assert_eq!(
        f.read_variable::<f32>("XLAT_M").unwrap().into_raw_vec_and_offset().0,
        vec![40.0f32, 41.0]
    );
    assert_eq!(
        f.read_variable::<f32>("T2").unwrap().into_raw_vec_and_offset().0,
        vec![1.0f32, 2.0, 3.0, 4.0]
    );
}

#[test]
fn record_dimension_must_lead_a_record_variable() {
    // The concrete breakage: classic NetCDF stores records interleaved,
    // so the record axis MUST be dimension 0 of every record variable.
    // A variable declared (y, Time) has no representable layout; writing
    // one produces a file whose values are transposed with no error.
    let mut schema = Schema::new(NcFormat::Offset64);
    let t = schema.def_dim("Time", 0, true).unwrap();
    let y = schema.def_dim("y", 2, false).unwrap();
    let err = schema.def_var("bad", NcType::Float, &[y, t]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Time"), "got: {msg}");
    assert!(msg.contains("first dimension"), "got: {msg}");
}

#[test]
fn only_one_unlimited_dimension_is_allowed() {
    let mut schema = Schema::new(NcFormat::Offset64);
    schema.def_dim("Time", 0, true).unwrap();
    let err = schema.def_dim("Time2", 0, true).unwrap_err();
    assert!(err.to_string().contains("unlimited"), "got: {err}");
}

// --------------------------------------------------------------------------
// The single-record-variable special case, which netCDF-C implements and
// most re-implementations miss.
// --------------------------------------------------------------------------

#[test]
fn single_record_variable_slabs_are_not_padded() {
    // netCDF-C (nc3internal.c, NC_computeshapes): when exactly one
    // variable uses the record dimension, the record stride is the RAW
    // slab size, not the 4-byte-padded one -- while the header's `vsize`
    // field still carries the padded value.  MEASURED against a file
    // written by netCDF4-python: a char variable with a 3-byte slab has
    // vsize 4 and stride 3.
    //
    // Getting this wrong is not a crash: the reader walks the record
    // section with the wrong stride and returns shifted bytes.
    let path = tmp("single_rec");
    let mut schema = Schema::new(NcFormat::Offset64);
    let t = schema.def_dim("t", 0, true).unwrap();
    let s = schema.def_dim("s", 3, false).unwrap();
    let c = schema.def_var("c", NcType::Char, &[t, s]).unwrap();

    let mut w = NcWriter::create(&path, schema).unwrap();
    w.write_record(0, c, VarData::Char(b"abc")).unwrap();
    w.write_record(1, c, VarData::Char(b"def")).unwrap();
    w.finish().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    // Six data bytes, back to back, no padding between the two records.
    assert!(
        bytes.ends_with(b"abcdef"),
        "record slabs must be contiguous; tail was {:?}",
        &bytes[bytes.len().saturating_sub(12)..]
    );
    // vsize is still the padded value (4), exactly as netCDF-C writes it.
    let vsize_pos = bytes.len() - 6 - 12; // ... vsize(4) begin(8) data(6)
    assert_eq!(
        u32::from_be_bytes(bytes[vsize_pos..vsize_pos + 4].try_into().unwrap()),
        4,
        "vsize field is the PADDED slab size even in the special case"
    );
}

#[test]
fn two_record_variables_pad_each_slab() {
    let path = tmp("two_rec");
    let mut schema = Schema::new(NcFormat::Offset64);
    let t = schema.def_dim("t", 0, true).unwrap();
    let s = schema.def_dim("s", 3, false).unwrap();
    let c = schema.def_var("c", NcType::Char, &[t, s]).unwrap();
    let g = schema.def_var("f", NcType::Float, &[t]).unwrap();

    let mut w = NcWriter::create(&path, schema).unwrap();
    w.write_record(0, c, VarData::Char(b"abc")).unwrap();
    w.write_record(0, g, VarData::F32(&[1.0])).unwrap();
    w.write_record(1, c, VarData::Char(b"def")).unwrap();
    w.write_record(1, g, VarData::F32(&[2.0])).unwrap();
    w.finish().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let mut want = Vec::new();
    want.extend_from_slice(b"abc\0");
    want.extend_from_slice(&1.0f32.to_be_bytes());
    want.extend_from_slice(b"def\0");
    want.extend_from_slice(&2.0f32.to_be_bytes());
    assert!(bytes.ends_with(&want), "each slab pads to 4 when >1 record var");

    let f = NcFile::open(&path).unwrap();
    assert_eq!(
        f.read_variable_as_strings("c").unwrap(),
        vec!["abc".to_string(), "def".to_string()]
    );
    assert_eq!(
        f.read_variable::<f32>("f").unwrap().into_raw_vec_and_offset().0,
        vec![1.0f32, 2.0]
    );
}

// --------------------------------------------------------------------------
// Attributes of every classic type, global and per-variable.
// --------------------------------------------------------------------------

#[test]
fn attributes_of_every_type_roundtrip() {
    let path = tmp("attrs");
    let mut schema = Schema::new(NcFormat::Offset64);
    let n = schema.def_dim("n", 1, false).unwrap();
    schema
        .put_global_attr("TITLE", AttrValue::Text(" OUTPUT FROM GPUWM".into()))
        .unwrap();
    schema
        .put_global_attr("DX", AttrValue::Floats(vec![12000.0]))
        .unwrap();
    schema
        .put_global_attr("MAP_PROJ", AttrValue::Ints(vec![1]))
        .unwrap();
    schema
        .put_global_attr("CEN_LAT_D", AttrValue::Doubles(vec![38.5]))
        .unwrap();
    schema
        .put_global_attr("FLAGS", AttrValue::Shorts(vec![1, 2, 3]))
        .unwrap();
    schema
        .put_global_attr("RAW", AttrValue::Bytes(vec![-1, 2]))
        .unwrap();

    let v = schema.def_var("T2", NcType::Float, &[n]).unwrap();
    schema.put_var_attr(v, "units", AttrValue::Text("K".into())).unwrap();
    schema
        .put_var_attr(v, "_FillValue", AttrValue::Floats(vec![-9999.0]))
        .unwrap();
    schema
        .put_var_attr(v, "coordinates", AttrValue::Text("XLONG XLAT".into()))
        .unwrap();

    let mut w = NcWriter::create(&path, schema).unwrap();
    w.write_var(v, VarData::F32(&[300.0])).unwrap();
    w.finish().unwrap();

    let f = NcFile::open(&path).unwrap();
    let g = |name: &str| f.global_attribute(name).unwrap().value.clone();
    attr_eq(&g("TITLE"), &NcAttrValue::Chars(" OUTPUT FROM GPUWM".into()));
    attr_eq(&g("DX"), &NcAttrValue::Floats(vec![12000.0]));
    attr_eq(&g("MAP_PROJ"), &NcAttrValue::Ints(vec![1]));
    attr_eq(&g("CEN_LAT_D"), &NcAttrValue::Doubles(vec![38.5]));
    attr_eq(&g("FLAGS"), &NcAttrValue::Shorts(vec![1, 2, 3]));
    attr_eq(&g("RAW"), &NcAttrValue::Bytes(vec![-1, 2]));

    let var = f.variable("T2").unwrap();
    attr_eq(
        &var.attribute("units").unwrap().value,
        &NcAttrValue::Chars("K".into()),
    );
    attr_eq(
        &var.attribute("_FillValue").unwrap().value,
        &NcAttrValue::Floats(vec![-9999.0]),
    );
    attr_eq(
        &var.attribute("coordinates").unwrap().value,
        &NcAttrValue::Chars("XLONG XLAT".into()),
    );
}

// --------------------------------------------------------------------------
// Refusals that name their breakage, and determinism.
// --------------------------------------------------------------------------

#[test]
fn finish_refuses_a_hole_in_the_data() {
    // The concrete breakage: a NetCDF file whose data section was never
    // written reads back as whatever the filesystem left there.  There is
    // no in-band signal, so the refusal has to be at finish().
    let path = tmp("hole");
    let mut schema = Schema::new(NcFormat::Offset64);
    let n = schema.def_dim("n", 1, false).unwrap();
    let a = schema.def_var("a", NcType::Float, &[n]).unwrap();
    let _b = schema.def_var("b", NcType::Float, &[n]).unwrap();
    let mut w = NcWriter::create(&path, schema).unwrap();
    w.write_var(a, VarData::F32(&[1.0])).unwrap();
    let err = w.finish().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains('b'), "the unwritten variable is named: {msg}");
}

#[test]
fn finish_refuses_a_hole_in_the_record_section() {
    let path = tmp("rec_hole");
    let mut schema = Schema::new(NcFormat::Offset64);
    let t = schema.def_dim("t", 0, true).unwrap();
    let a = schema.def_var("a", NcType::Float, &[t]).unwrap();
    let b = schema.def_var("b", NcType::Float, &[t]).unwrap();
    let mut w = NcWriter::create(&path, schema).unwrap();
    w.write_record(0, a, VarData::F32(&[1.0])).unwrap();
    w.write_record(0, b, VarData::F32(&[2.0])).unwrap();
    w.write_record(1, a, VarData::F32(&[3.0])).unwrap();
    let err = w.finish().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("record 1"), "got: {msg}");
    assert!(msg.contains('b'), "got: {msg}");
}

#[test]
fn wrong_value_count_is_refused() {
    let path = tmp("count");
    let mut schema = Schema::new(NcFormat::Offset64);
    let n = schema.def_dim("n", 3, false).unwrap();
    let a = schema.def_var("a", NcType::Float, &[n]).unwrap();
    let mut w = NcWriter::create(&path, schema).unwrap();
    let err = w.write_var(a, VarData::F32(&[1.0])).unwrap_err();
    assert!(err.to_string().contains("3 value"), "got: {err}");
}

#[test]
fn wrong_data_type_is_refused() {
    // Writing f64 into an NC_FLOAT variable would silently halve every
    // value's byte width and shift the whole data section.
    let path = tmp("dtype");
    let mut schema = Schema::new(NcFormat::Offset64);
    let n = schema.def_dim("n", 1, false).unwrap();
    let a = schema.def_var("a", NcType::Float, &[n]).unwrap();
    let mut w = NcWriter::create(&path, schema).unwrap();
    let err = w.write_var(a, VarData::F64(&[1.0])).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("NC_FLOAT"), "got: {msg}");
    assert!(msg.contains("NC_DOUBLE"), "got: {msg}");
}

#[test]
fn identical_schemas_produce_identical_bytes() {
    let build = |path: &std::path::Path| {
        let mut schema = Schema::new(NcFormat::Offset64);
        let t = schema.def_dim("Time", 0, true).unwrap();
        let y = schema.def_dim("y", 2, false).unwrap();
        schema
            .put_global_attr("TITLE", AttrValue::Text("determinism".into()))
            .unwrap();
        let v = schema.def_var("T2", NcType::Float, &[t, y]).unwrap();
        schema.put_var_attr(v, "units", AttrValue::Text("K".into())).unwrap();
        let mut w = NcWriter::create(path, schema).unwrap();
        w.write_record(0, v, VarData::F32(&[1.0, 2.0])).unwrap();
        w.finish().unwrap();
        std::fs::read(path).unwrap()
    };
    assert_eq!(build(&tmp("det_a")), build(&tmp("det_b")));
}

#[test]
fn cdf1_rejects_a_data_section_past_the_32_bit_offset_limit() {
    // The concrete breakage: CDF-1 stores `begin` in 32 bits.  A file
    // whose data section crosses 2 GiB wraps the offset and every
    // variable after the wrap reads from the wrong place.  CDF-2 is the
    // fix, and the message has to say so.
    let mut schema = Schema::new(NcFormat::Classic);
    let big = schema.def_dim("big", 700 * 1024 * 1024, false).unwrap();
    schema.def_var("a", NcType::Float, &[big]).unwrap();
    schema.def_var("b", NcType::Float, &[big]).unwrap();
    let err = NcWriter::create(&tmp("cdf1_overflow"), schema).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("CDF-1"), "got: {msg}");
    assert!(msg.contains("CDF-2"), "got: {msg}");
}

#[test]
fn cdf1_and_cdf2_and_cdf5_all_carry_their_own_magic() {
    for (format, magic) in [
        (NcFormat::Classic, b"CDF\x01"),
        (NcFormat::Offset64, b"CDF\x02"),
        (NcFormat::Cdf5, b"CDF\x05"),
    ] {
        let path = tmp(&format!("magic_{magic:?}").replace(['\\', '"', '\'', '{', '}'], "_"));
        let mut schema = Schema::new(format);
        let n = schema.def_dim("n", 1, false).unwrap();
        let v = schema.def_var("a", NcType::Float, &[n]).unwrap();
        let mut w = NcWriter::create(&path, schema).unwrap();
        w.write_var(v, VarData::F32(&[1.0])).unwrap();
        w.finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..4], magic, "{format:?}");
        // And the independent reader agrees it is that format.
        let f = NcFile::open(&path).unwrap();
        let want = match format {
            NcFormat::Classic => RFormat::Classic,
            NcFormat::Offset64 => RFormat::Offset64,
            NcFormat::Cdf5 => RFormat::Cdf5,
        };
        assert_eq!(f.format(), want);
    }
}

#[test]
fn scalar_variables_roundtrip() {
    let path = tmp("scalar");
    let mut schema = Schema::new(NcFormat::Offset64);
    let v = schema.def_var("gain", NcType::Double, &[]).unwrap();
    let mut w = NcWriter::create(&path, schema).unwrap();
    w.write_var(v, VarData::F64(&[2.5])).unwrap();
    w.finish().unwrap();
    let f = NcFile::open(&path).unwrap();
    assert_eq!(
        f.read_variable::<f64>("gain").unwrap().into_raw_vec_and_offset().0,
        vec![2.5]
    );
}

#[test]
fn zero_records_is_a_valid_file() {
    // A tape opened and closed before the first frame lands must still be
    // a readable NetCDF file, not a truncated one.
    let path = tmp("zero_records");
    let mut schema = Schema::new(NcFormat::Offset64);
    let t = schema.def_dim("Time", 0, true).unwrap();
    let _v = schema.def_var("T2", NcType::Float, &[t]).unwrap();
    let w = NcWriter::create(&path, schema).unwrap();
    w.finish().unwrap();
    let f = NcFile::open(&path).unwrap();
    assert_eq!(f.dimension("Time").unwrap().size, 0);
}
