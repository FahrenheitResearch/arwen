// SPDX-License-Identifier: Apache-2.0
use std::fs;
use std::path::Path;
use std::time::Instant;
mod trace;

fn compare(expected: &Path, actual: &Path, width: usize) -> Result<bool, String> {
    if width != 1 && width != 4 && width != 8 {
        return Err("word bytes must be 1, 4 or 8".into());
    }
    let start = Instant::now();
    let (a, b) = std::thread::scope(|s| {
        let a = s.spawn(|| fs::read(expected));
        let b = s.spawn(|| fs::read(actual));
        (a.join().unwrap(), b.join().unwrap())
    });
    let a = a.map_err(|e| e.to_string())?;
    let b = b.map_err(|e| e.to_string())?;
    if a.len() != b.len() || a.is_empty() || a.len() % width != 0 {
        return Err(
            "expected/actual byte sizes must agree and contain nonempty whole words".into(),
        );
    }
    let words = a.len() / width;
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(words);
    let chunk = words.div_ceil(workers);
    let mut differences = Vec::new();
    let count: usize = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (index, (a, b)) in a
            .chunks(chunk * width)
            .zip(b.chunks(chunk * width))
            .enumerate()
        {
            handles.push(s.spawn(move || {
                let mut count = 0;
                let mut first = Vec::new();
                for (i, (a, b)) in a.chunks_exact(width).zip(b.chunks_exact(width)).enumerate() {
                    if a != b {
                        count += 1;
                        if first.len() < 64 {
                            let mut av = [0_u8; 8];
                            let mut bv = [0_u8; 8];
                            av[..width].copy_from_slice(a);
                            bv[..width].copy_from_slice(b);
                            first.push((
                                index * chunk + i,
                                u64::from_le_bytes(av),
                                u64::from_le_bytes(bv),
                            ));
                        }
                    }
                }
                (count, first)
            }));
        }
        handles
            .into_iter()
            .map(|h| {
                let (count, first) = h.join().unwrap();
                differences.extend(first);
                count
            })
            .sum()
    });
    differences.sort_unstable_by_key(|d| d.0);
    differences.truncate(64);
    let examples = differences
        .iter()
        .map(|(i, a, b)| {
            format!(
                "{{\"word\":{i},\"expected_bits\":\"0x{a:0w$x}\",\"actual_bits\":\"0x{b:0w$x}\"}}",
                w = width * 2
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    println!("{{\"schema\":\"troute-bitwise-v1\",\"words\":{words},\"word_bytes\":{width},\"different_words\":{count},\"workers\":{workers},\"seconds\":{},\"examples\":[{examples}],\"pass\":{}}}", start.elapsed().as_secs_f64(), count == 0);
    Ok(count == 0)
}

fn run(args: &[String]) -> Result<bool, String> {
    match args.get(1).map(String::as_str) {
        Some("trace") => trace::run(args),
        Some("compare") if args.len() == 5 => compare(
            Path::new(&args[2]),
            Path::new(&args[3]),
            args[4].parse().map_err(|_| "word bytes")?,
        ),
        Some("split") if args.len() == 4 => {
            let start = Instant::now();
            let raw = fs::read(&args[2]).map_err(|e| e.to_string())?;
            let arrays = troute_mc_replay::split_records(&raw)?;
            let out = Path::new(&args[3]);
            fs::create_dir(out).map_err(|e| e.to_string())?;
            for (name, data) in ["inputs.f32", "caller-seeds.f32", "expected.f32"]
                .iter()
                .zip(arrays)
            {
                fs::write(out.join(name), data).map_err(|e| e.to_string())?;
            }
            println!(
                "{{\"schema\":\"troute-split-v1\",\"calls\":{},\"workers\":{},\"seconds\":{}}}",
                raw.len() / 108,
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1)
                    .min(raw.len() / 108),
                start.elapsed().as_secs_f64()
            );
            Ok(true)
        }
        _ => Err(
            "usage: troute-mc-replay split CALLS.raw NEW_DIR | compare EXPECTED ACTUAL WORD_BYTES"
                .into(),
        ),
    }
}

fn main() {
    match run(&std::env::args().collect::<Vec<_>>()) {
        Ok(pass) => {
            if !pass {
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}
