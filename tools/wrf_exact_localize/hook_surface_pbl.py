"""Write-only surface/PBL trace for a lane-owned WRF 4.6.1 source copy.

Run after hook_wrf8.py. No computation or scheme option is changed.
Each anchor must occur exactly once, so an incompatible source fails closed.
"""
from pathlib import Path


def insert(src, anchor, block, *, before=False):
    if src.count(anchor) != 1:
        raise ValueError(f"WRF trace anchor {anchor!r}: {src.count(anchor)} matches")
    return src.replace(anchor, block + anchor if before else anchor + block)


def dump(tag, fields3=(), fields2=(), *, grid=False):
    lines = [f"      CALL locstage('{tag}')\n"]
    for name, expr in fields3:
        lines.append(f"      CALL locdumpk('{name}', {expr}, ims,ime,kms,kme,jms,jme, 1)\n")
    for name, expr in fields2:
        lines.append(f"      CALL locdumpk('{name}', {expr}, ims,ime,1,1,jms,jme, 1)\n")
    return "".join(lines)


def hook(dyn):
    dyn = Path(dyn)
    for path in (dyn / "module_first_rk_step_part1.F",
                 dyn / "module_first_rk_step_part2.F",
                 dyn.parent / "phys" / "module_surface_driver.F"):
        if "p00_phy_prep" in path.read_text() or "p40_mass_coupled" in path.read_text() or "p10_sfclay_in" in path.read_text():
            raise ValueError(f"surface/PBL hooks already installed in {path}")
    p = dyn / "module_first_rk_step_part1.F"
    s = p.read_text()
    atmos = [(n, expr) for n, expr in (
        ("theta", "th_phy"), ("temperature", "t_phy"), ("exner", "pi_phy"),
        ("pressure", "grid%p_hyd"), ("p_interface", "grid%p_hyd_w"),
        ("u", "grid%u_phy"), ("v", "grid%v_phy"),
        ("rho", "grid%rho"), ("dz", "dz8w"), ("z_interface", "grid%z_at_w"))]
    surface_names = ("ust", "hfx", "qfx", "wspd", "br", "fm", "fh", "znt",
                     "u10", "v10", "tsk", "psfc", "pblh", "xland", "mavail")
    surface = [(n, "grid%" + n) for n in surface_names]
    pbl = [(n, "grid%" + n) for n in
           ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "rqiblten")]
    s = insert(s, "BENCH_END(phy_prep_tim)\n", dump("p00_phy_prep", atmos))
    s = insert(s, "BENCH_START(pbl_driver_tim)\n", dump("p30_ysu_in", atmos, surface))
    s = insert(s, "BENCH_END(pbl_driver_tim)\n", dump("p31_ysu_out", pbl, surface))
    # Scheme outputs are still uncoupled here. calculate_phy_tend multiplies
    # them in place later, before update_phy_ten interpolates the wind faces.
    p.write_text(s)
    p = dyn / "module_first_rk_step_part2.F"
    lines = p.read_text().splitlines(keepends=True)
    anchors = [i for i, line in enumerate(lines)
               if "call calculate_phy_tend (" in line.lower()]
    if len(anchors) != 1:
        raise ValueError(f"calculate_phy_tend call: {anchors}")
    i = anchors[0]
    while lines[i].split("!")[0].rstrip().endswith("&"):
        i += 1
    lines.insert(i + 1, dump("p40_mass_coupled", pbl + [("rthraten", "grid%rthraten")]))
    p.write_text("".join(lines))
    p = dyn.parent / "phys" / "module_surface_driver.F"
    s = p.read_text()
    fields = [(n, "fhh" if n == "fh" else n) for n in surface_names]
    # These are the actual surface-layer wind inputs after driver adjustments.
    sf_atmos = [("u", "u_phytmp"), ("v", "v_phytmp"),
                ("temperature", "t_phy"), ("pressure", "p_phy"),
                ("p_interface", "p8w"), ("dz", "dz8w"), ("qv", "qv_curr")]
    s = insert(s, "     END SELECT sfclay_select\n", dump("p11_sfclay_out", (), fields))
    # Inputs are emitted just before the selector, after the driver's wind prep.
    lines = s.splitlines(keepends=True)
    anchors = [i for i, line in enumerate(lines)
               if "sfclay_select" in line.lower() and "select case" in line.lower()]
    if len(anchors) != 1:
        raise ValueError(f"sfclay selector: {anchors}")
    lines.insert(anchors[0], dump("p10_sfclay_in", sf_atmos, fields))
    s = "".join(lines)
    land_fields = fields + [(n, n) for n in
                            ("glw", "swdown", "albedo", "emiss", "snow", "snowh", "canwat")]
    soil = "".join(f"      CALL locdumpk('{n}', {n}, ims,ime,1,num_soil_layers,jms,jme, 1)\n"
                   for n in ("smois", "tslb", "sh2o"))
    s = insert(s, "     END SELECT sfc_select\n", dump("p21_noah_out", (), land_fields) + soil)
    lines = s.splitlines(keepends=True)
    anchors = [i for i, line in enumerate(lines)
               if "sfc_select" in line.lower() and "select case" in line.lower()]
    if len(anchors) != 1:
        raise ValueError(f"land selector: {anchors}")
    lines.insert(anchors[0], dump("p20_noah_in", sf_atmos, land_fields) + soil)
    p.write_text("".join(lines))


if __name__ == "__main__":
    import sys
    hook(sys.argv[1])
