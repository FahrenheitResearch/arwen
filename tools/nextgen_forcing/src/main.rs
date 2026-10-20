// SPDX-License-Identifier: Apache-2.0
use arwen_nextgen_forcing::{prepare,Request,Result};
fn main() {
    if let Err(e)=run() {eprintln!("{e}");std::process::exit(1);}
}
fn run()->Result<()> {
    let path=std::env::args_os().nth(1).ok_or("usage: arwen-nextgen-forcing REQUEST.json")?;
    let request:Request=serde_json::from_slice(&std::fs::read(path)?)?;
    println!("{}",serde_json::to_string_pretty(&prepare(request)?)?); Ok(())
}
