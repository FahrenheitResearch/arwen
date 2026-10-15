//! rw_mpas_static is one executable with several tools. These run the built
//! binary: every former binary name still reaches its tool, subcommands reach
//! the same tools, the door's contract is unchanged apart from the TOOLS line
//! a caller reads to know the tools exist, and the stream comparator names
//! the netCDF C library when it cannot load it instead of failing to start.
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_rw_mpas_static");
const TOOLS: [(&str, &str); 5] = [
    ("bounded", "rw_mpas_static_bounded"),
    ("words", "rw_mpas_static_words"),
    ("gwd-replace", "rw_mpas_gwd_replace"),
    ("gwd-review", "rw_mpas_gwd_review"),
    ("verify", "rw_mpas_verify"),
];

fn scratch(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("rw-mpas-multicall-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

/// A link (Unix) or copy (Windows) of the binary under another name.
fn alias(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(if cfg!(windows) { format!("{name}.exe") } else { name.to_string() });
    #[cfg(unix)]
    std::os::unix::fs::symlink(BIN, &path).unwrap();
    #[cfg(not(unix))]
    std::fs::copy(BIN, &path).unwrap();
    path
}

/// A plain copy of the binary under another name, as a resolving caller
/// (the gpuwm-hex comparator lookup resolves links) needs it.
fn copy_as(dir: &Path, name: &str) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(if cfg!(windows) { format!("{name}.exe") } else { name.to_string() });
    std::fs::copy(BIN, &path).unwrap();
    path
}

fn run(program: &Path, args: &[&str]) -> Output {
    Command::new(program).args(args).env_remove("RW_MPAS_NETCDF_LIBRARY").output().unwrap()
}

fn text(output: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr))
}

#[test]
fn the_door_abi_keeps_its_contract_and_lists_the_tools() {
    let output = run(Path::new(BIN), &["--abi"]);
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "{stdout}");
    assert!(lines[0].starts_with("rw_mpas_static --grid GRID.nc --out STATIC.nc"));
    assert_eq!(lines[1], "GRID\tNOMINALDX\tVECTORS\tOPERATORS\tGEOGGRID\tGWDBAND\tWROTE\tFINISHED");
    let names: Vec<&str> = TOOLS.iter().map(|(name, _)| *name).collect();
    assert_eq!(lines[2], format!("TOOLS\t{}", names.join("\t")));
    let version = run(Path::new(BIN), &["--version"]);
    assert!(version.status.success() && text(&version).starts_with("rw_mpas_static "));
}

#[test]
fn every_former_binary_name_and_subcommand_reaches_its_tool() {
    let dir = scratch("names");
    for (name, legacy) in TOOLS {
        let by_name = alias(&dir, legacy);
        let staged = alias(&dir, &format!("{legacy}-staged"));
        let copied = copy_as(&dir.join("copies"), legacy);
        // With no arguments every tool refuses with its own usage, never the
        // door's: the door with no arguments prints its help and exits 0.
        let expected = match name {
            "bounded" => (1, "rw_mpas_static_bounded: usage: rw_mpas_static_bounded"),
            "words" => (2, "usage: rw_mpas_static words"),
            "gwd-replace" => (1, "usage: rw_mpas_static gwd-replace"),
            "gwd-review" => (1, "usage: rw_mpas_static gwd-review"),
            "verify" => (2, "rw_mpas_verify: usage: rw_mpas_verify --request JSON --out JSON"),
            _ => unreachable!(),
        };
        for output in [run(Path::new(BIN), &[name]), run(&by_name, &[]), run(&staged, &[]), run(&copied, &[])] {
            assert_eq!(output.status.code(), Some(expected.0), "{name}: {}", text(&output));
            assert!(text(&output).contains(expected.1), "{name}: {}", text(&output));
        }
    }
    let help = run(Path::new(BIN), &[]);
    assert!(help.status.success());
    for (name, legacy) in TOOLS {
        assert!(text(&help).contains(name) && text(&help).contains(legacy), "{}", text(&help));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn the_comparator_answers_its_abi_without_the_netcdf_library() {
    let dir = scratch("abi");
    let legacy = alias(&dir, "rw_mpas_verify");
    for output in [
        Command::new(BIN).args(["verify", "--abi"]).env("RW_MPAS_NETCDF_LIBRARY", dir.join("absent.so")).output().unwrap(),
        Command::new(&legacy).arg("--abi").env("RW_MPAS_NETCDF_LIBRARY", dir.join("absent.so")).output().unwrap(),
    ] {
        assert!(output.status.success(), "{}", text(&output));
        assert_eq!(String::from_utf8(output.stdout).unwrap(), "gpuwm-rw-mpas-verify-v1\n");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn an_unloadable_netcdf_library_is_a_named_refusal() {
    let dir = scratch("refusal");
    let absent = dir.join("libnetcdf-absent.so");
    let stream = dir.join("stream.nc");
    std::fs::write(&stream, b"CDF\x01").unwrap();
    let output = Command::new(BIN)
        .args(["verify", "--describe", stream.to_str().unwrap()])
        .env("RW_MPAS_NETCDF_LIBRARY", &absent)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    let message = text(&output);
    assert!(message.contains("rw_mpas_verify: the netCDF C library is not loadable"), "{message}");
    assert!(message.contains("RW_MPAS_NETCDF_LIBRARY"), "{message}");
    assert!(message.contains("libnetcdf-absent.so"), "{message}");
    std::fs::remove_dir_all(dir).unwrap();
}

/// On a host that has the library, the loaded comparator reads a real
/// classic stream. Skipped (and said so) where no libnetcdf is installed.
#[test]
fn the_loaded_comparator_describes_a_classic_stream_when_the_library_exists() {
    use rw_store::netcdf_classic::{NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef};
    let dir = scratch("describe");
    let stream = dir.join("history.nc");
    let dims = vec![NcDim::record("Time"), NcDim::fixed("nCells", 2), NcDim::fixed("StrLen", 19)];
    let vars = vec![
        NcVarDef::new("xtime", NcType::Char, vec![0, 2]),
        NcVarDef::new("t2m", NcType::Float, vec![0, 1]),
    ];
    let mut file = NcClassicWriter::create(&stream, NcFormat::Offset64, dims, vec![], vars, 2).unwrap();
    for record in 0..2u64 {
        file.put_record_text("xtime", record, &format!("2000-01-01_{record:02}:00:00")).unwrap();
        file.put_record("t2m", record, NcData::Floats(&[280.0, -0.0])).unwrap();
    }
    file.finish().unwrap();
    let output = Command::new(BIN).args(["verify", "--describe", stream.to_str().unwrap()])
        .env_remove("RW_MPAS_NETCDF_LIBRARY").output().unwrap();
    if text(&output).contains("the netCDF C library is not loadable") {
        eprintln!("SKIPPED: no loadable libnetcdf on this host");
        std::fs::remove_dir_all(dir).unwrap();
        return;
    }
    assert!(output.status.success(), "{}", text(&output));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["records"], 2);
    assert_eq!(value["times"][1], "2000-01-01_01:00:00");
    assert_eq!(value["time_source"], "xtime");
    std::fs::remove_dir_all(dir).unwrap();
}
