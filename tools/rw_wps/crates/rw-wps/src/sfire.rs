//! SFIRE static preparation through the native Rust static-field authority.
//! The installed shell moves opaque source bytes; the Rust bridge computes
//! the refined fuel, terrain and derivative fields and writes the artifact.
use std::ffi::{OsStr,OsString};
use std::path::Path;
use std::process::Command;
use crate::RwWpsError;

pub fn arguments(config:&Path,output:&Path,receipt:Option<&Path>) -> Vec<OsString> {
    let mut args=vec![config.as_os_str().to_owned(),"--output".into(),output.as_os_str().to_owned()];
    if let Some(receipt)=receipt {args.push("--receipt".into());args.push(receipt.as_os_str().to_owned());}
    args
}

pub fn run(config:&Path,output:&Path,receipt:Option<&Path>,engine:&OsStr) -> Result<(),RwWpsError> {
    let status=Command::new(engine).args(arguments(config,output,receipt)).status()
        .map_err(|e|RwWpsError::Engine(format!("SFIRE static preparation could not launch: {e}")))?;
    if !status.success() {
        return Err(RwWpsError::Engine(format!("SFIRE static preparation exited {status}")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_paths_reach_one_static_engine_without_shell_interpolation() {
        let got=arguments(Path::new("config with space.json"),Path::new("fire.npz"),Some(Path::new("receipt.json")));
        assert_eq!(got,vec![OsString::from("config with space.json"),"--output".into(),"fire.npz".into(),"--receipt".into(),"receipt.json".into()]);
    }
}
