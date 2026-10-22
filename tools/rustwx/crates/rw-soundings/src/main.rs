/// Release verification reads the source revision from the executable bytes.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main() {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    if std::env::args().any(|a| a == "--abi") {
        println!("gpuwm-soundings.v1 plain-run png json html");
        return;
    }
    // One line naming the device renders would use (a real device is
    // created, not just enumerated); exit 1 with the cause when none can be.
    if std::env::args().any(|a| a == "--probe-device") {
        match rw_soundings::render::render_device() {
            Ok(device) => println!("render device: {device}"),
            Err(e) => {
                eprintln!("soundings: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    let result = (|| {
        let path = std::env::args()
            .nth(1)
            .ok_or("usage: rw_soundings REQUEST.json")?;
        let req = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        rw_soundings::batch::run(req)
    })();
    if let Err(e) = result {
        eprintln!("soundings: {e}");
        std::process::exit(1);
    }
}
