//! Writer output read back with this crate's reader; these tests pin the
//! encodings. Independent readers (h5py, netCDF-C, xarray) check the same
//! structures in the files the examples `h5_write_demo` and
//! `nc4_write_demo` write (`tools/hdf5_writer_check.py`).

use super::*;
use crate::{H5File, Values};

fn read(bytes: &[u8]) -> H5File<'_> {
    H5File::open(bytes).expect("writer output opens")
}

#[test]
fn empty_file_is_a_root_group() {
    let bytes = Writer::new().finish().expect("finish");
    let file = read(&bytes);
    assert_eq!(file.superblock_version(), 2);
    assert!(file.child_names("/").is_empty());
}

#[test]
fn attributes_keep_type_shape_and_order() {
    let mut writer = Writer::new();
    let root = writer.root();
    let what = writer.add_group(root, "what").expect("group");
    writer
        .add_attribute(what, "object", Value::text_nul("PVOL"))
        .expect("attr");
    writer
        .add_attribute(what, "units", Value::text("dBZ"))
        .expect("attr");
    writer
        .add_attribute(what, "empty", Value::text(""))
        .expect("attr");
    writer
        .add_attribute(what, "gain", Value::scalar(Data::F64(vec![0.5])))
        .expect("attr");
    writer
        .add_attribute(what, "nodata", Value::vector(Data::F32(vec![255.0])))
        .expect("attr");
    writer
        .add_attribute(what, "flags", Value::vector(Data::U8(vec![1, 2, 3])))
        .expect("attr");
    writer
        .add_attribute(what, "i64", Value::vector(Data::I64(vec![-5, i64::MAX])))
        .expect("attr");
    writer
        .add_attribute(what, "u16", Value::scalar(Data::U16(vec![65_000])))
        .expect("attr");
    writer
        .add_attribute(what, "i8", Value::scalar(Data::I8(vec![-128])))
        .expect("attr");
    writer
        .add_attribute(
            what,
            "names",
            Value::var_texts(vec!["alpha".into(), String::new(), "γ".into()]),
        )
        .expect("attr");
    writer
        .add_attribute(what, "note", Value::var_text("variable"))
        .expect("attr");
    let bytes = writer.finish().expect("finish");
    let file = read(&bytes);
    let attrs = file.attrs("/what");
    let names: Vec<&str> = attrs.iter().map(|attr| attr.name()).collect();
    assert_eq!(
        names,
        [
            "object", "units", "empty", "gain", "nodata", "flags", "i64", "u16", "i8", "names",
            "note"
        ]
    );
    let attr = |name: &str| file.attr("/what", name).expect(name);
    assert_eq!(attr("object").as_str().as_deref(), Some("PVOL"));
    assert_eq!(attr("units").as_str().as_deref(), Some("dBZ"));
    assert!(attr("empty").is_null());
    assert_eq!(attr("gain").values(), &Values::F64(vec![0.5]));
    assert!(attr("gain").dims().is_empty());
    assert_eq!(attr("nodata").values(), &Values::F32(vec![255.0]));
    assert_eq!(attr("nodata").dims(), [1]);
    assert_eq!(attr("flags").values(), &Values::U8(vec![1, 2, 3]));
    assert_eq!(attr("i64").values(), &Values::I64(vec![-5, i64::MAX]));
    assert_eq!(attr("u16").values(), &Values::U16(vec![65_000]));
    assert_eq!(attr("i8").values(), &Values::I8(vec![-128]));
    assert_eq!(
        attr("names").values(),
        &Values::VarStrings(vec!["alpha".into(), String::new(), "γ".into()])
    );
    assert_eq!(
        attr("note").values(),
        &Values::VarStrings(vec!["variable".into()])
    );
}

#[test]
fn datasets_in_every_layout_read_back() {
    let rows = 300u64;
    let cols = 97u64;
    let values: Vec<u16> = (0..rows * cols).map(|i| (i * 7 % 65_521) as u16).collect();
    let mut writer = Writer::new();
    let root = writer.root();
    let group = writer.add_group(root, "sweep").expect("group");
    writer
        .add_dataset(
            group,
            "contiguous",
            NewDataset::new(
                Value::array(Data::U16(values.clone()), vec![rows, cols]),
                Layout::Contiguous,
            ),
        )
        .expect("dataset");
    writer
        .add_dataset(
            group,
            "compact",
            NewDataset::new(Value::vector(Data::F64(vec![1.5, -2.0])), Layout::Compact),
        )
        .expect("dataset");
    // 7 x 3 chunks with partial edge chunks, shuffled and deflated.
    writer
        .add_dataset(
            group,
            "chunked",
            NewDataset::new(
                Value::array(Data::U16(values.clone()), vec![rows, cols]),
                Layout::Chunked {
                    chunk: vec![45, 40],
                    shuffle: true,
                    deflate: Some(6),
                },
            )
            .with_fill_value(Data::U16(vec![0])),
        )
        .expect("dataset");
    // 300 one-row chunks: a two-level B-tree.
    writer
        .add_dataset(
            group,
            "many_chunks",
            NewDataset::new(
                Value::array(Data::U16(values.clone()), vec![rows, cols]),
                Layout::chunked(vec![1, cols], Some(1)),
            ),
        )
        .expect("dataset");
    writer
        .add_dataset(
            group,
            "scalar",
            NewDataset::new(Value::scalar(Data::I32(vec![-7])), Layout::Contiguous),
        )
        .expect("dataset");
    writer
        .add_dataset(
            group,
            "strings",
            NewDataset::new(
                Value::var_texts(vec!["azimuth_surveillance".into(), "rhi".into()]),
                Layout::Contiguous,
            ),
        )
        .expect("dataset");
    writer
        .add_dataset(
            group,
            "unallocated",
            NewDataset::new(Value::vector(Data::F32(vec![1.0])), Layout::Unallocated),
        )
        .expect_err("an unallocated dataset takes its shape, not elements");
    writer
        .add_dataset(
            group,
            "unallocated",
            NewDataset::new(
                Value {
                    data: Data::F32(Vec::new()),
                    shape: Shape::Simple(vec![5]),
                },
                Layout::Unallocated,
            ),
        )
        .expect("dataset");
    let bytes = writer.finish().expect("finish");
    let file = read(&bytes);
    for name in ["contiguous", "chunked", "many_chunks"] {
        let dataset = file.dataset(&format!("/sweep/{name}")).expect(name);
        assert_eq!(dataset.dims, [rows as usize, cols as usize], "{name}");
        assert_eq!(dataset.values, Values::U16(values.clone()), "{name}");
    }
    let info = file.dataset_info("/sweep/chunked").expect("info");
    assert_eq!(
        info.filters.iter().map(|f| f.id).collect::<Vec<_>>(),
        [2, 1]
    );
    assert_eq!(info.fill_value, Some(vec![0, 0]));
    assert_eq!(
        file.chunk_locations("/sweep/many_chunks")
            .expect("chunks")
            .len(),
        300
    );
    assert_eq!(
        file.dataset("/sweep/compact").expect("compact").values,
        Values::F64(vec![1.5, -2.0])
    );
    let scalar = file.dataset("/sweep/scalar").expect("scalar");
    assert!(scalar.dims.is_empty());
    assert_eq!(scalar.values, Values::I32(vec![-7]));
    assert_eq!(
        file.dataset("/sweep/strings").expect("strings").values,
        Values::VarStrings(vec!["azimuth_surveillance".into(), "rhi".into()])
    );
    let unallocated = file.dataset("/sweep/unallocated").expect("unallocated");
    assert_eq!(unallocated.values, Values::F32(vec![0.0; 5]));
}

#[test]
fn references_and_dimension_scale_attributes_resolve() {
    let mut writer = Writer::new();
    let root = writer.root();
    let time = writer
        .add_dataset(
            root,
            "time",
            NewDataset::new(Value::vector(Data::F64(vec![0.0, 1.0])), Layout::Contiguous),
        )
        .expect("time");
    let field = writer
        .add_dataset(
            root,
            "DBZH",
            NewDataset::new(
                Value::array(Data::I16(vec![1, 2, 3, 4]), vec![2, 2]),
                Layout::Contiguous,
            ),
        )
        .expect("field");
    writer
        .add_attribute(
            field,
            "DIMENSION_LIST",
            Value::vector(Data::ObjectRefLists(vec![vec![time], vec![time]])),
        )
        .expect("attr");
    writer
        .add_attribute(
            time,
            "REFERENCE_LIST",
            Value::vector(Data::DimensionScaleRefs(vec![(field, 0)])),
        )
        .expect("attr");
    writer
        .add_attribute(root, "target", Value::scalar(Data::ObjectRefs(vec![field])))
        .expect("attr");
    let bytes = writer.finish().expect("finish");
    let file = read(&bytes);
    let time_address = file.object("/time").expect("time").address();
    let field_address = file.object("/DBZH").expect("field").address();
    let list = file.attr("/DBZH", "DIMENSION_LIST").expect("list");
    assert_eq!(
        list.values(),
        &Values::Sequences(vec![
            Values::References(vec![Some(time_address)]),
            Values::References(vec![Some(time_address)]),
        ])
    );
    let back = file.attr("/time", "REFERENCE_LIST").expect("back");
    assert_eq!(
        back.values(),
        &Values::Compound {
            elements: 1,
            members: vec![
                (
                    "dataset".to_owned(),
                    Values::References(vec![Some(field_address)])
                ),
                ("dimension".to_owned(), Values::I32(vec![0])),
            ]
        }
    );
    assert_eq!(
        file.attr("/", "target").expect("target").values(),
        &Values::References(vec![Some(field_address)])
    );
}

#[test]
fn large_groups_keep_insertion_order() {
    let mut writer = Writer::new();
    let root = writer.root();
    let names: Vec<String> = (0..40).rev().map(|i| format!("sweep_{i}")).collect();
    for name in &names {
        writer.add_group(root, name).expect("group");
    }
    let bytes = writer.finish().expect("finish");
    let file = read(&bytes);
    assert_eq!(file.child_names("/"), names);
}

#[test]
fn many_variable_length_strings_span_collections() {
    let mut writer = Writer::new();
    let root = writer.root();
    let values: Vec<String> = (0..70_000).map(|i| format!("s{i}")).collect();
    writer
        .add_dataset(
            root,
            "strings",
            NewDataset::new(Value::var_texts(values.clone()), Layout::Contiguous),
        )
        .expect("dataset");
    let bytes = writer.finish().expect("finish");
    let file = read(&bytes);
    assert_eq!(
        file.dataset("/strings").expect("strings").values,
        Values::VarStrings(values)
    );
}

#[test]
fn invalid_input_is_a_typed_error() {
    let mut writer = Writer::new();
    let root = writer.root();
    assert!(matches!(
        writer.add_group(root, "a/b"),
        Err(WriteError::InvalidName(_))
    ));
    writer.add_group(root, "a").expect("group");
    assert!(matches!(
        writer.add_group(root, "a"),
        Err(WriteError::Duplicate(_))
    ));
    assert!(matches!(
        writer.add_attribute(root, "x", Value::array(Data::U8(vec![1]), vec![2])),
        Err(WriteError::Invalid(_))
    ));
    let dataset = writer
        .add_dataset(
            root,
            "d",
            NewDataset::new(Value::scalar(Data::U8(vec![1])), Layout::Contiguous),
        )
        .expect("dataset");
    assert!(matches!(
        writer.add_group(dataset, "child"),
        Err(WriteError::NotAGroup(_))
    ));
    assert!(matches!(
        writer.add_attribute(root, "huge", Value::vector(Data::U8(vec![0; 70_000]))),
        Ok(())
    ));
    assert!(matches!(writer.finish(), Err(WriteError::TooLarge(_))));
}

#[test]
fn output_is_deterministic() {
    let build = || {
        let mut writer = Writer::new();
        let root = writer.root();
        writer
            .add_attribute(root, "a", Value::var_text("x"))
            .expect("attr");
        writer
            .add_dataset(
                root,
                "d",
                NewDataset::new(
                    Value::array(Data::F32(vec![1.0; 1000]), vec![10, 100]),
                    Layout::chunked(vec![5, 100], Some(4)),
                ),
            )
            .expect("dataset");
        writer.finish().expect("finish")
    };
    assert_eq!(build(), build());
}

#[test]
fn bools_and_compounds_read_back() {
    let mut writer = Writer::new();
    let root = writer.root();
    writer
        .add_attribute(root, "malfunc", Value::scalar(Data::Bools(vec![true])))
        .expect("attr");
    let legend = Data::Compound(vec![
        (
            "key".to_owned(),
            Data::FixedStrings {
                bytes: b"rain\0\0snow\0\0".to_vec(),
                size: 6,
                padding: StringPadding::NullTerminate,
                charset: CharSet::Ascii,
            },
        ),
        ("value".to_owned(), Data::I32(vec![1, -2])),
    ]);
    writer
        .add_dataset(
            root,
            "legend",
            NewDataset::new(Value::vector(legend), Layout::Contiguous),
        )
        .expect("dataset");
    let bytes = writer.finish().expect("finish");
    let file = read(&bytes);
    let malfunc = file.attr("/", "malfunc").expect("malfunc");
    assert!(matches!(malfunc.datatype(), crate::Datatype::Enum { .. }));
    assert_eq!(malfunc.values(), &Values::I8(vec![1]));
    let legend = file.dataset("/legend").expect("legend");
    assert_eq!(legend.values.len(), 2);
    let Values::Compound {
        members: columns, ..
    } = legend.values
    else {
        panic!("not a compound");
    };
    assert_eq!(columns[0].0, "key");
    assert_eq!(
        columns[0].1.strings(),
        Some(vec!["rain".to_owned(), "snow".to_owned()])
    );
    assert_eq!(columns[1], ("value".to_owned(), Values::I32(vec![1, -2])));
}
