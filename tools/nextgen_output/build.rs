fn main() {
    println!("cargo:rerun-if-env-changed=GPUWM_OUTPUT_NETCDF_PREFIX");
    if let Some(prefix) = std::env::var_os("GPUWM_OUTPUT_NETCDF_PREFIX") {
        let lib = std::path::PathBuf::from(prefix).join("lib");
        for name in ["netcdf", "hdf5_hl", "hdf5", "z"] {
            assert!(
                lib.join(format!("lib{name}.a")).is_file(),
                "missing static {name} in {}",
                lib.display()
            );
        }
        println!("cargo:rustc-link-search=native={}", lib.display());
        for name in ["netcdf", "hdf5_hl", "hdf5", "z"] {
            println!("cargo:rustc-link-lib=static={name}");
        }
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
        // Keep private C backend symbols out of the Python process namespace.
        println!("cargo:rustc-link-arg=-Wl,--exclude-libs,ALL");
    } else {
        println!("cargo:rustc-link-lib=netcdf");
    }
}
