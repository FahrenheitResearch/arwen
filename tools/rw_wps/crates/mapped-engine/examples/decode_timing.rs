//! A flame-style timing table for one decode, phase by phase.
//!
//! Mirrors `engine::run_decode` exactly, with a clock around each step, so
//! "where does a large complex-packed decode spend its seconds?" is
//! answered by measurement instead of by reading.  Not shipped: this is an
//! example, so it never lands in the wheel's package data.
//!
//! The files are walked SERIALLY here even though the engine decodes
//! them concurrently: the point of this program is to attribute seconds
//! to phases, and a phase whose clock spans overlapping work attributes
//! nothing.  The concurrency inside a phase (messages, fields, digests)
//! is left on, so each row is that phase as the engine actually runs it.
//! A bare run's TOTAL is therefore a lower bound on a serial decode and
//! an upper bound on the engine's, and the engine's own wall clock is
//! measured by running the engine.
//!
//! usage: decode_timing --mapping M.json --input-list F.txt [--output DIR]
//!                      [--breakdown]

use std::time::Instant;

use mapped_engine::assemble::assemble_grib;
use mapped_engine::grib::{
    grib2_identities, grib2_records, validate_grib2_envelopes, wanted_indices, GribRecord,
};
use mapped_engine::model::Mapping;

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut mapping_path = String::new();
    let mut list_path = String::new();
    let mut output: Option<String> = None;
    // `--breakdown` decodes the same messages a SECOND time to rank
    // unpack against frame export against the fingerprint.  It inflates
    // the total by exactly that second decode, so it is opt-in: a bare
    // run's TOTAL is the real one.
    let mut breakdown = false;
    let mut position = 0usize;
    while position < arguments.len() {
        let flag = arguments[position].clone();
        if flag == "--breakdown" {
            breakdown = true;
            position += 1;
            continue;
        }
        let value = arguments
            .get(position + 1)
            .unwrap_or_else(|| panic!("{flag} needs a value"))
            .clone();
        match flag.as_str() {
            "--mapping" => mapping_path = value,
            "--input-list" => list_path = value,
            "--output" => output = Some(value),
            other => panic!("unknown option {other}"),
        }
        position += 2;
    }

    let whole = Instant::now();
    let mapping = Mapping::load(&mapping_path).expect("mapping loads");
    let files = mapped_engine::engine::read_input_list(&list_path).expect("input list");

    let clock = Instant::now();
    let digests = mapped_engine::engine::input_digests(&files).expect("input digests");
    let input_hash = clock.elapsed();

    let declaration = mapping.grid_declaration().expect("grid declaration");
    let mut read = std::time::Duration::ZERO;
    let mut codec = std::time::Duration::ZERO;
    let mut envelopes = std::time::Duration::ZERO;
    let mut parse_one = std::time::Duration::ZERO;
    let mut identity = std::time::Duration::ZERO;
    let mut select = std::time::Duration::ZERO;
    let mut records_phase = std::time::Duration::ZERO;
    let mut parse_two = std::time::Duration::ZERO;
    let mut messages = 0usize;
    let mut selected = 0usize;
    let mut bytes = 0u64;

    let mut records: Vec<GribRecord> = Vec::new();
    for source in &files {
        let clock = Instant::now();
        let raw = std::fs::read(source).expect("input reads");
        bytes += raw.len() as u64;
        read += clock.elapsed();

        let clock = Instant::now();
        let payload = mapped_engine::codec::decoded_payload(raw, source).expect("codec");
        codec += clock.elapsed();

        let clock = Instant::now();
        validate_grib2_envelopes(&payload, source).expect("envelopes");
        envelopes += clock.elapsed();

        let clock = Instant::now();
        let file = grib_core::grib2::Grib2File::from_bytes(&payload).expect("parse");
        parse_one += clock.elapsed();

        let clock = Instant::now();
        let identities = grib2_identities(&file.messages);
        identity += clock.elapsed();
        messages += identities.len();

        let clock = Instant::now();
        let wanted = wanted_indices(&mapping, &identities).expect("selection");
        select += clock.elapsed();
        selected += wanted.len();

        // The second parse the engine used to pay inside `grib2_records`,
        // timed on its own so the table keeps the cost that was removed
        // visible rather than silently absorbed.
        let clock = Instant::now();
        let _reparse = grib_core::grib2::Grib2File::from_bytes(&payload).expect("parse");
        parse_two += clock.elapsed();

        let clock = Instant::now();
        records.extend(
            grib2_records(&file, source, &wanted, &declaration).expect("records"),
        );
        records_phase += clock.elapsed();
    }

    // Inside `grib2_records`: unpack vs frame export vs array build, on the
    // same bytes, so the three can be ranked against each other.
    let mut unpack = std::time::Duration::ZERO;
    let mut export = std::time::Duration::ZERO;
    let mut fingerprint = std::time::Duration::ZERO;
    for source in files.iter().filter(|_| breakdown) {
        let raw = std::fs::read(source).expect("input reads");
        let payload = mapped_engine::codec::decoded_payload(raw, source).expect("codec");
        let file = grib_core::grib2::Grib2File::from_bytes(&payload).expect("parse");
        let identities = grib2_identities(&file.messages);
        let wanted = wanted_indices(&mapping, &identities).expect("selection");
        for index in &wanted {
            let message = &file.messages[*index];
            let clock = Instant::now();
            let raw_values = grib_core::grib2::unpack_message(message).expect("unpack");
            unpack += clock.elapsed();
            let clock = Instant::now();
            if !declaration.is_lambert() {
                let _ = mapped_engine::grib::regular_latlon_frame(message, &raw_values)
                    .expect("frame export");
            }
            export += clock.elapsed();
            let clock = Instant::now();
            let _ = mapped_engine::grib::grid_fingerprint(message);
            fingerprint += clock.elapsed();
        }
    }

    let clock = Instant::now();
    let collection = assemble_grib(&mapping, &records).expect("assembly");
    let assemble = clock.elapsed();

    let clock = Instant::now();
    let frames = mapped_engine::frames::materialize_frames(&mapping, &collection).expect("frames");
    let materialize = clock.elapsed();

    let mut emit = std::time::Duration::ZERO;
    let mut field_hash = std::time::Duration::ZERO;
    let mut header_hash = std::time::Duration::ZERO;
    let mut cells = 0usize;
    if let Some(directory) = output.as_ref() {
        // The emission's own field hashing, timed apart from the write.
        for frame in &frames {
            for field in &frame.fields {
                let flat = mapped_engine::array::contiguous(&field.values);
                cells += flat.len();
                let clock = Instant::now();
                let _ = mapped_engine::digest::array_sha256(field.values.shape(), &flat);
                field_hash += clock.elapsed();
            }
            let clock = Instant::now();
            let _ = mapped_engine::digest::bytes_sha256(
                mapped_engine::engine::canonical_json(&frame.header).as_bytes(),
            );
            header_hash += clock.elapsed();
        }
        let clock = Instant::now();
        mapped_engine::frames::write_frameset(
            std::path::Path::new(directory),
            &mapping,
            &collection,
            &frames,
            &digests,
        )
        .expect("frameset writes");
        emit = clock.elapsed();
    }

    let total = whole.elapsed();
    println!("inputs            {} files, {:.2} GiB", files.len(), bytes as f64 / 1073741824.0);
    println!("messages          {messages} parsed, {selected} selected");
    println!("frame cells       {cells}");
    println!("--- phase ------------------- seconds -- share");
    let row = |name: &str, value: std::time::Duration| {
        println!(
            "{name:26}  {:8.3}   {:5.1}%",
            value.as_secs_f64(),
            100.0 * value.as_secs_f64() / total.as_secs_f64()
        );
    };
    row("input sha256", input_hash);
    row("file read", read);
    row("acquisition codec", codec);
    row("envelope validation", envelopes);
    row("parse (first)", parse_one);
    row("identities", identity);
    row("selector match", select);
    row("parse (second; removed)", parse_two);
    row("grib2_records (whole)", records_phase);
    row("  of which unpack", unpack);
    row("  of which frame export", export);
    row("  of which fingerprint", fingerprint);
    row("assemble", assemble);
    row("materialize_frames", materialize);
    row("emission field sha256", field_hash);
    row("emission header sha256", header_hash);
    row("write_frameset (whole)", emit);
    row("TOTAL", total);
}
