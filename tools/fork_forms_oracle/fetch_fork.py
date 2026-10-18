"""Fetch the operational RAP/HRRR branch's source files this lane compiles.

NOAA-EMC/HRRR tag v4.1.21 (commit 40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827),
``sorc/hrrr_wrfarw.fd/WRFV3.9``.  Every file is checked against the SHA256
recorded here before anything is built from it; the six that other oracles
in this tree already pin carry the same hashes (tools/solar_albedo_oracle,
tools/ruc_hrrr_fork_oracle/source-pins.json,
tests/data/upper_wind_fork.json).

    python fetch_fork.py DEST_DIR

The box fetches these from the public repository itself; nothing is copied
between boxes.  Analysis and verification tooling only.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen

COMMIT = "40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827"
TAG = "v4.1.21"
BASE = (f"https://raw.githubusercontent.com/NOAA-EMC/HRRR/{COMMIT}/"
        "sorc/hrrr_wrfarw.fd/WRFV3.9/")
PIN = {
    "phys/module_radiation_driver.F":
        "7464639e53f6f40b4810943ee9fa21a40b0f53f7525c08cc13b84ae4c7d82801",
    "phys/module_surface_driver.F":
        "5aed6cc50973ada58661262ded97d5ce8b1bf5407ad423f3df2f34e64da5612a",
    "phys/module_sf_ruclsm.F":
        "cb2b54114afc247e16ea03d547a510b19842e2ee012da1202eba19105bef6aab",
    "phys/module_sf_sfcdiags_ruclsm.F":
        "1d91266f56d48fc6595f4489006d7ee4a1b858b881ac68363d10862f7e9d7812",
    "phys/module_microphysics_zero_out.F":
        "a57268701b424b07a604ce93f733314183a29bfb39e7e6283f92f8668808fa4f",
    "phys/module_ra_rrtmg_sw.F":
        "04e97b7e3cea6984979fd4b59ea0293d3ed15e2ecd2ba7a7dd81979bd86f3f39",
    "phys/module_ra_rrtmg_lw.F":
        "688fe4f80cc500a8825529c39dcd3d1ecc245e45e406736f61f252dec04407e6",
    "dyn_em/module_small_step_em.F":
        "15a4c618c4428400cceb3bc7f5734e4df9ed28a1db0421fb37a79ca0cda09b4d",
    "dyn_em/module_big_step_utilities_em.F":
        "88b180f1c3cce95ba965d8183a6e51c62df0d58754d941dd17c0db6f22df7239",
    "dyn_em/module_em.F":
        "0a92ea1ca2f73d267ccedf04f37e99fbaded85831186774bba3703aea8491cc1",
    "dyn_em/solve_em.F":
        "84a246f0fcb4b9521a89c437ecccf2f08b1539b996e0e5151617d88e28b88585",
    "share/module_model_constants.F":
        "0d0ee7d243fe673288efafcdcdc470ee028ad8d083648f12225ce2d168f8e87c",
    "Registry/Registry.EM_COMMON":
        "8eed8499de583e7d11a28fe4c5faba0ea2720b9992e8bd750ef7595ad9a1e552",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(dest: Path, names=None) -> dict[str, str]:
    """Download (when absent) and verify; returns {relative path: sha256}."""
    out = {}
    for name in names or PIN:
        path = dest / name
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(BASE + name) as source:
                path.write_bytes(source.read())
        got = sha256(path)
        if got != PIN[name]:
            raise ValueError(f"{name}: SHA256 {got} differs from pinned {PIN[name]}")
        out[name] = got
    return out


if __name__ == "__main__":
    for name, digest in fetch(Path(sys.argv[1])).items():
        print(digest, name)
