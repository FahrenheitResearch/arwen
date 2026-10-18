"""Add write-only terrain-input dumps after the delivered surface hooks."""
from pathlib import Path


def hook(directory):
    path = Path(directory) / "phys/module_surface_driver.F"
    source = path.read_text()
    anchor = "      CALL locstage('p20_noah_in')\n"
    if source.count(anchor) != 1:
        raise ValueError("one installed p20_noah_in surface hook is required")
    if "CALL locdumpk('declin', declin" in source:
        raise ValueError("terrain input dumps are already installed")
    names = ("slope", "slp_azi", "coszen", "hrang", "diffuse_frac", "swnorm", "xlat", "xlong")
    block = "".join(f"      CALL locdumpk('{n}', {n}, ims,ime,1,1,jms,jme, 1)\n" for n in names)
    block += "      CALL locdumpk('declin', declin, 1,1,1,1,1,1, 1)\n"
    terrain_update = (path, source.replace(anchor, anchor + block))
    path = Path(directory) / "dyn_em/module_first_rk_step_part2.F"
    source = path.read_text()
    anchor = "      CALL locstage('p40_mass_coupled')\n"
    if source.count(anchor) != 1 or "CALL locdumpk('rthcuten', grid%rthcuten" in source:
        raise ValueError("one fresh p40_mass_coupled hook is required")
    names = ("rthcuten", "rqvcuten", "rqccuten", "rucuten", "rvcuten")
    # Inactive CU arrays can be one-element allocation placeholders.
    # Reading them with full memory bounds would capture unrelated memory.
    block = "      IF (config_flags%cu_physics > 0) THEN\n"
    block += "".join(f"      CALL locdumpk('{n}', grid%{n}, ims,ime,kms,kme,jms,jme, 1)\n" for n in names)
    block += "      ENDIF\n"
    terrain_update[0].write_text(terrain_update[1])
    path.write_text(source.replace(anchor, anchor + block))


if __name__ == "__main__":
    import sys
    hook(sys.argv[1])
