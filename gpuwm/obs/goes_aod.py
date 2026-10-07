"""GOES ABI aerosol optical depth orchestration and fixed-grid pack reader."""
from pathlib import Path
import json
import numpy as np
from gpuwm.obs.frontdoor import GOES
from gpuwm.obs.goes_pack import AOD_SCHEMA, read_goes_pack
from gpuwm.obs.obspack import sha256_of


def read_goes_aod(path):
    """Read Rust's 550 nm AOD pack. Only DQF 0 is high quality.

    The source NetCDF flag_values and flag_meanings are checked by Rust.
    Values 1, 2 and 3 denote medium, low and no retrieval respectively.
    Missing DQF is rejected. Retained zero AOD remains zero.
    """
    pack = read_goes_pack(path, expected_schema=AOD_SCHEMA)
    if pack.meta.get("units") != "1" or pack.meta.get("wavelength_nm") != 550:
        raise ValueError("AOD pack must describe dimensionless AOD at 550 nm")
    aod, dqf = pack.plane("aod"), pack.plane("aod_dqf")
    if np.any(np.isfinite(aod) & (dqf != 0)):
        raise ValueError("AOD pack retains pixels outside high-quality DQF")
    return pack


def fetch_goes_aod(start, end, out, *, satellite="G18", sector="C", window=None):
    """List and fetch anonymous AOD C/F granules, then pack each scan in Rust."""
    if satellite not in ("G18", "G19") or sector not in ("C", "F"):
        raise ValueError("AOD verification uses G18/G19 and sector C/F")
    binary = GOES.require()
    ok, detail = GOES.probe(binary)
    if not ok:
        raise RuntimeError(detail)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    record = GOES.run("fetch", ["--satellite", satellite, "--sector", sector,
                    "--products", "AOD", "--start", str(start), "--end", str(end),
                    "--cache", str(out / "cache"), "--out", str(out / "granules")],
                    schema="gpuwm-obs.goes-fetch.v1", binary=binary)
    (out / "fetch.json").write_text(json.dumps(record, indent=2) + "\n")
    packs = []
    for scan in record["scans"]:
        files = scan["files"]
        if not scan["complete"] or len(files) != 1 or files[0]["product"] != "AOD":
            raise ValueError("AOD fetch did not return one granule per complete scan")
        native = Path(files[0]["path"])
        if sha256_of(native) != files[0]["sha256"]:
            raise ValueError("AOD fetched granule digest mismatch")
        path = out / (native.stem + ".goespack")
        args = ["--aod", str(native), "--out", str(path)]
        if window is not None:
            args += ["--window", ",".join(map(str, window))]
        GOES.run("aod", args, schema="gpuwm-obs.goes-aod-build.v1", binary=binary)
        read_goes_aod(path)
        packs.append(path)
    return tuple(packs)


__all__ = ["fetch_goes_aod", "read_goes_aod"]
