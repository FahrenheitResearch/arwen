"""Write the actual GFDRV arguments from a lane-owned WRF diagnostic build."""
from pathlib import Path


def hook(directory):
    path = Path(directory) / "phys/module_cumulus_driver.F"
    source = path.read_text()
    anchor = "         CALL GFDRV(spp_conv,pattern_spp_conv,field_conv,       &\n"
    if source.count(anchor) != 1:
        raise ValueError("one stock GFDRV call is required")
    if "CALL locstage('cu_gf_input')" in source:
        raise ValueError("GF input dumps are already installed")
    block = "         CALL locstage('cu_gf_input')\n"
    for name in ("u", "v", "w", "t", "qv_curr", "p", "pi", "rho", "dz8w", "p8w",
                 "rthften", "rqvften", "rthraten", "rthblten", "rqvblten"):
        block += f"         CALL locdumpk('{name}', {name}, ims,ime,kms,kme,jms,jme, 1)\n"
    for name in ("ht", "hfx", "qfx", "xland", "kpbl"):
        block += f"         CALL locdumpk('{name}', {name}, ims,ime,1,1,jms,jme, 1)\n"
    for name in ("dt", "dx", "ishallow", "clos_choice"):
        block += f"         CALL locdumpk('{name}', {name}, 1,1,1,1,1,1, 1)\n"
    path.write_text(source.replace(anchor, block + anchor))


if __name__ == "__main__":
    import sys
    hook(sys.argv[1])
