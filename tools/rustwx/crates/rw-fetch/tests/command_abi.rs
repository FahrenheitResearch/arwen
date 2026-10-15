use std::process::Command;

#[test]
fn abi_declares_the_commands_in_addition_to_record_keys() {
    let result = Command::new(env!("CARGO_BIN_EXE_rw_fetch"))
        .arg("--abi")
        .output()
        .unwrap();
    assert!(result.status.success());
    let abi = String::from_utf8(result.stdout).unwrap();
    let commands = abi.trim().split('\t')
        .find_map(|token| token.strip_prefix("commands="))
        .expect("record keys alone cannot identify the hash and fetch command contract");
    for required in ["fetch", "probe", "latest", "object", "asset", "encoded-asset",
                     "hash", "unpack", "unpack-audit", "decode", "inventory", "objects"] {
        assert!(commands.split(',').any(|command| command == required), "lost {required}");
        // Missing required arguments must be refused by a known command,
        // before any network transfer, rather than by the command dispatcher.
        let refusal = Command::new(env!("CARGO_BIN_EXE_rw_fetch"))
            .arg(required).output().unwrap();
        assert!(!refusal.status.success(), "{required} accepted no arguments");
        let stderr = String::from_utf8(refusal.stderr).unwrap();
        assert!(!stderr.contains("unknown subcommand"), "{stderr}");
    }
}
