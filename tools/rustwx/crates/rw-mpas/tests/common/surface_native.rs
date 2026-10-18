//! Test-only orchestration of f0's unmodified strict init executable.
//! Case 8 arguments follow mpas_init_atm_cases.F:272-299 and the immutable
//! surface stream in core_init_atmosphere/Registry.xml:666-678.
use std::fs;
use std::path::Path;
use std::process::Command;

pub fn run(
    binary: &Path,
    static_path: &Path,
    prefix: &Path,
    graph_prefix: &Path,
    start: &str,
    stop: &str,
    interval: i64,
    ranks: usize,
    directory: &Path,
) -> serde_json::Value {
    fs::create_dir_all(directory).unwrap();
    let quote = |s: &str| s.replace('\'', "''");
    fs::write(directory.join("namelist.init_atmosphere"), format!(
        "&nhyd_model\n config_init_case=8\n config_start_time='{}'\n config_stop_time='{}'\n/\n\
         &dimensions\n config_nvertlevels=55\n config_nsoillevels=4\n config_nfglevels=34\n config_nfgsoillevels=4\n/\n\
         &data_sources\n config_sfc_prefix='{}'\n config_fg_interval={}\n/\n\
         &preproc_stages\n config_static_interp=false\n config_native_gwd_static=false\n config_native_gwd_gsl_static=false\n config_vertical_grid=false\n config_met_interp=false\n/\n\
         &io\n config_pio_num_iotasks=0\n config_pio_stride=1\n/\n\
         &decomposition\n config_block_decomp_file_prefix='{}'\n/\n",
        quote(start), quote(stop), quote(&prefix.to_string_lossy()), interval,
        quote(&graph_prefix.to_string_lossy())
    )).unwrap();
    let input = static_path
        .to_string_lossy()
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;");
    fs::write(directory.join("streams.init_atmosphere"), format!(
        "<streams>\n\
         <immutable_stream name=\"input\" type=\"input\" filename_template=\"{input}\" input_interval=\"initial_only\" />\n\
         <immutable_stream name=\"output\" type=\"output\" filename_template=\"unused.nc\" output_interval=\"none\" packages=\"initial_conds\" />\n\
         <immutable_stream name=\"ugwp_oro_data\" type=\"output\" filename_template=\"unused-ugwp.nc\" output_interval=\"none\" packages=\"gwd_gsl_stage_out\" />\n\
         <immutable_stream name=\"surface\" type=\"output\" filename_template=\"native.nc\" output_interval=\"{interval}\" packages=\"sfc_update\" filename_interval=\"none\" />\n\
         <immutable_stream name=\"lbc\" type=\"output\" filename_template=\"unused-lbc.nc\" output_interval=\"none\" packages=\"lbcs\" />\n\
         </streams>\n"
    )).unwrap();
    let output = Command::new("mpiexec")
        .args(["-n", &ranks.to_string()])
        .arg(binary)
        .current_dir(directory)
        .output()
        .unwrap();
    let mut logs = String::from_utf8_lossy(&output.stdout).into_owned();
    logs.push_str(&String::from_utf8_lossy(&output.stderr));
    let mut files: Vec<_> = fs::read_dir(directory)
        .unwrap()
        .map(|p| p.unwrap().path())
        .collect();
    files.sort();
    for path in files {
        if path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("log.init_atmosphere.")
        {
            logs.push_str(&fs::read_to_string(path).unwrap());
        }
    }
    serde_json::json!({"exit_code":output.status.code(),"success":output.status.success(),"ranks":ranks,"logs":logs})
}
