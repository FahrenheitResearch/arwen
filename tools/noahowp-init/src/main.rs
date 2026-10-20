// SPDX-License-Identifier: Apache-2.0
fn main() {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 3 && !(a.len() == 5 && a[3] == "--parameters") {
        eprintln!(
            "usage: arwen-noahowp-init REQUEST.json NEW_OUTPUT_DIRECTORY [--parameters DIRECTORY]"
        );
        std::process::exit(2)
    }
    match arwen_noahowp_init::prepare_with_parameters(
        std::path::Path::new(&a[1]),
        std::path::Path::new(&a[2]),
        if a.len() == 5 {
            Some(std::path::Path::new(&a[4]))
        } else {
            None
        },
    ) {
        Ok(v) => println!("{v}"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1)
        }
    }
}
