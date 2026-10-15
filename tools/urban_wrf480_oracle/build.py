"""Build release-pinned urban schemes and expose the 4.8 interface change.

Run on a lane CPU box. Numerical source files are never edited. The adapted
column driver records the new state separately from the legacy CSV columns.
This builds a reference, not a qualified forecast generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PINS = {
    "v4.6.1": "d66e442fccc04111067e29274c9f9eaccc3cef28",
    "v4.7.0": "eeacb821626feb34c62c3c1e169594d886d13adb",
    "v4.7.1": "f52c197ed39d12e087d02c50f412d90d418f6186",
    "v4.8.0": "06d4240ae989cc3e50af412bb472df3d9048783c",
}
URBAN480_SHA256 = "1be801c3e11aad59be8e0e23612bbe71ba4fa29cefc930520fb4c6e2e1169f6a"
SUBMODULES = {
    "GFL": ("https://github.com/NOAA-GSL/GFL", "873b414670323ae766ec4fe1f4b6b5ab891358ed"),
    "MYNN-EDMF": ("https://github.com/NCAR/MYNN-EDMF", "8f8e8b59021249949c04d1686edc5a02ca325627"),
    "MYNN-SFC": ("https://github.com/NCAR/MYNN-SFC", "b66187f54a0c169e36592552634867474e603ac0"),
    "TEMPO": ("https://github.com/NCAR/TEMPO", "881614f68aa65de8f383d56139162049baf600fa"),
    "fire_behavior": ("https://github.com/NCAR/fire_behavior", "eb7758055a5d436f022a634cae780f5512c42bb5"),
    "noahmp": ("https://github.com/NCAR/noahmp", "5da0b241e48ecfd9a2a1bd667ed554765856d589"),
    "physics_mmm": ("https://github.com/NCAR/MMM-physics", "550b5b494758ad2756394e93cb08b5709b8feb83"),
}
SOURCE_FILES = (
    "phys/module_sf_urban.F", "phys/module_sf_noahdrv.F",
    "phys/module_surface_driver.F",
    "Registry/Registry.EM_COMMON", "run/URBPARM.TBL", "run/URBPARM_LCZ.TBL",
    "share/module_model_constants.F", "frame/module_wrf_error.F", ".gitmodules",
)
FLAGS = ["-O0", "-g", "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-fno-tree-vectorize",
         "-ffree-form", "-ffree-line-length-none", "-fallow-argument-mismatch",
         "-cpp", "-Dwrfmodel", "-DEM_CORE=1", "-DNMM_CORE=0",
         "-DRWORDSIZE=4", "-DIWORDSIZE=4", "-DDWORDSIZE=8", "-DLWORDSIZE=4"]
HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "urban_wrf471_oracle"


def run(args, *, cwd=None, allow_fail=False):
    p = subprocess.run(list(map(str, args)), cwd=cwd, capture_output=True, text=True)
    if p.returncode and not allow_fail:
        raise RuntimeError(f"command {args!r}\n{p.stdout}\n{p.stderr}")
    return p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(root):
    root.mkdir(parents=True, exist_ok=True)
    repo = root / "wrf"
    if not repo.exists():
        run(["git", "init", "-q", repo])
    provenance = {"tags": {}, "submodules": {}, "files": {}}
    for tag, pin in PINS.items():
        run(["git", "-C", repo, "fetch", "-q", "--depth=1", "--filter=blob:none",
             "https://github.com/wrf-model/WRF.git", f"refs/tags/{tag}:refs/tags/{tag}"])
        actual = run(["git", "-C", repo, "rev-parse", tag + "^{commit}"]).stdout.strip()
        if actual != pin:
            raise ValueError(f"{tag}: {actual}, expected {pin}")
        provenance["tags"][tag] = actual
        for name in SOURCE_FILES + (("arch/Externals.cfg",) if tag.startswith("v4.7") else ()):
            data = subprocess.check_output(["git", "-C", str(repo), "show", f"{tag}:{name}"])
            path = root / tag / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            provenance["files"][f"{tag}/{name}"] = sha(path)
        noah_pin = run(["git", "-C", repo, "ls-tree", tag, "phys/noahmp"]).stdout.split()[2]
        noah = root / "submodules" / ("noahmp-" + tag)
        noah.parent.mkdir(parents=True, exist_ok=True)
        if not noah.exists():
            run(["git", "init", "-q", noah])
        run(["git", "-C", noah, "fetch", "-q", "--depth=1", "--filter=blob:none",
             "https://github.com/NCAR/noahmp", noah_pin])
        names = run(["git", "-C", noah, "ls-tree", "-r", "--name-only", "FETCH_HEAD"]).stdout.splitlines()
        wrappers = [n for n in names if n.endswith("module_sf_noahmpdrv.F")]
        if tag == "v4.8.0":
            wrappers = [n for n in names if n.endswith("NoahmpUrbanDriverMainMod.F90")]
        if len(wrappers) != 1:
            raise ValueError(f"{tag} Noah-MP wrapper path is ambiguous or absent: {wrappers}")
        path = root / tag / "phys/noahmp" / wrappers[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(["git", "-C", str(noah), "show", f"FETCH_HEAD:{wrappers[0]}"]))
        provenance["files"][f"{tag}/phys/noahmp/{wrappers[0]}"] = sha(path)
        provenance.setdefault("noahmp_wrappers", {})[tag] = {"commit": noah_pin, "path": wrappers[0]}
    for name, (url, pin) in SUBMODULES.items():
        listed = run(["git", "-C", repo, "ls-tree", "v4.8.0", f"phys/{name}"]).stdout.split()
        if len(listed) != 4 or listed[2] != pin or listed[0] != "160000":
            raise ValueError(f"v4.8.0 phys/{name}: unexpected gitlink {listed}")
        sub = root / "submodules" / name
        sub.parent.mkdir(parents=True, exist_ok=True)
        if not sub.exists():
            run(["git", "init", "-q", sub])
        run(["git", "-C", sub, "fetch", "-q", "--depth=1", "--filter=blob:none", url, pin])
        actual = run(["git", "-C", sub, "rev-parse", "FETCH_HEAD"]).stdout.strip()
        if actual != pin:
            raise ValueError(f"{name}: {actual}, expected {pin}")
        provenance["submodules"][name] = actual
    mmm = root / "mmm47"
    if not mmm.exists():
        run(["git", "init", "-q", mmm])
    run(["git", "-C", mmm, "fetch", "-q", "--depth=1", "--filter=blob:none",
         "https://github.com/NCAR/MMM-physics.git", "refs/tags/20240626-MPASv8.2"])
    pin47 = run(["git", "-C", mmm, "rev-parse", "FETCH_HEAD^{commit}"]).stdout.strip()
    # Materialize this dependency to prevent an empty-directory comparison.
    names = run(["git", "-C", mmm, "ls-tree", "-r", "--name-only", "FETCH_HEAD"]).stdout.splitlines()
    materialized = []
    for name in names:
        if name.lower().endswith((".f", ".f90")):
            dest = root / "mmm47-source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(subprocess.check_output(["git", "-C", str(mmm), "show", f"FETCH_HEAD:{name}"]))
            materialized.append({"path": name, "sha256": sha(dest)})
    if not materialized:
        raise ValueError("WRF 4.7 MMM dependency has no materialized Fortran files")
    provenance["mmm47"] = {"tag": "20240626-MPASv8.2", "commit": pin47, "files": materialized}
    return provenance


def driver480(source):
    """Adapt only the call ABI and recording; leave release science untouched."""
    source = source.replace("  real :: lf_urb_s, z0_urb, vegfrac",
        "  real :: lf_urb_s, z0_urb, vegfrac\n"
        "  real :: tvg, tt, xxxvg, tvgl(4), smg(4), cmcg, flxhumvg, flxhumt\n"
        "  integer :: nbs_unit, tree_override, tt_override\n  real :: fvg_override\n"
        "  character(len=40), parameter :: nbs_names(16) = [character(len=40) :: &\n"
        "    'tvg','tt','xxxvg','tvgl1','tvgl2','tvgl3','tvgl4', &\n"
        "    'smg1','smg2','smg3','smg4','cmcg','flxhumvg','flxhumt','chs','chs2']")
    source = source.replace("  call write_table()", """
  tree_override = 0
  tt_override = 0
  fvg_override = 0.0
  if (command_argument_count() >= 13) then
    call get_command_argument(13, arg)
    read(arg, *) tree_override
  end if
  if (command_argument_count() >= 14) then
    call get_command_argument(14, arg)
    read(arg, *) fvg_override
  end if
  if (command_argument_count() >= 15) then
    call get_command_argument(15, arg)
    read(arg, *) tt_override
  end if
  treeoption = tree_override
  ttscheme = tt_override
  fvg_tbl = fvg_override
  call write_table()
  open(newunit=nbs_unit, file=trim(prefix)//'-nbs-words.csv', status='replace')
  write(nbs_unit, '(A)', advance='no') 'utype,scenario,step'
  do i=1,16
    write(nbs_unit, '(A)', advance='no') ','//trim(nbs_names(i))//'_in'
  end do
  do i=1,16
    write(nbs_unit, '(A)', advance='no') ','//trim(nbs_names(i))
  end do
  write(nbs_unit, *)
"""
    )
    source = source.replace("        call write_inputs()", """        call write_inputs()
        write(nbs_unit, '(I0,2(",",I0))', advance='no') utype, sc, st
        call write_nbs_words()
""")
    source = source.replace("                   lf_urb_s, z0_urb, vegfrac)",
        "                   tvg, tt, xxxvg, tvgl, smg, cmcg, flxhumvg, flxhumt, &\n"
        "                   lf_urb_s, z0_urb, vegfrac)")
    source = source.replace("        call write_outputs()", """        call write_outputs()
        call write_nbs_words()
        write(nbs_unit, *)
""")
    source = source.replace("  close(u)\n", "  close(u)\n  close(nbs_unit)\n", 1)
    source = source.replace("    vegfrac = 0.", """    vegfrac = 0.
    tvg = tsurf0
    tt = tsurf0
    xxxvg = 0.
    tvgl = trl
    smg = (/ 0.2, 0.2, 0.2, 0. /)
    cmcg = 0.
    flxhumvg = 0.
    flxhumt = 0.
""")
    # Preserve the older table-recording routine's signature with extra locals.
    source = source.replace("    integer :: numdir, numhgt", """    real :: albvg_, albt_, fvg_, rtree_, rtree_m_, rw_m_, htree_, dtree_
    real :: lai_tree_, lai_veg_, z0vg_, z0hvg_, capvg_, epsvg_, epst_
    integer :: numdir, numhgt""")
    source = source.replace("betr, betb, betg, trlend, tblend, tglend,               &",
        "betr, betb, betg, trlend, tblend, tglend, &\n"
        "                      albvg_, albt_, fvg_, rtree_, rtree_m_, rw_m_, htree_, dtree_, &\n"
        "                      lai_tree_, lai_veg_, z0vg_, z0hvg_, capvg_, epsvg_, epst_, &")
    source = source.replace("contains\n", """contains
  subroutine write_nbs_words()
    real :: values(16)
    values = [tvg,tt,xxxvg,tvgl,smg,cmcg,flxhumvg,flxhumt,chs,chs2]
    write(nbs_unit, '(16(",",Z8.8))', advance='no') transfer(values,[0],16)
  end subroutine write_nbs_words
""", 1)
    return source


def build(root, out, provenance):
    out.mkdir(parents=True, exist_ok=True)
    provenance["compiler"] = run(["gfortran", "--version"]).stdout.splitlines()[0]
    provenance["runtime"] = run(["ldd", "--version"]).stdout.splitlines()[0]
    provenance["flags"] = FLAGS
    provenance["scalar_kinds"] = {"real": 4, "integer": 4, "double": 8, "logical": 4}
    provenance["module_dependencies"] = ["module_model_constants", "module_wrf_error"]
    legacy = (LEGACY / "ucm_column_oracle.F90").read_text()
    provenance["legacy_driver_sha256"] = sha(LEGACY / "ucm_column_oracle.F90")
    for tag in PINS:
        src = root / tag
        if tag == "v4.8.0" and sha(src / "phys/module_sf_urban.F") != URBAN480_SHA256:
            raise ValueError("WRF 4.8 urban source pin changed")
        modes = ("ref", "snan", "zero")
        for mode in modes:
            work = out / tag / mode
            work.mkdir(parents=True, exist_ok=True)
            init = {"ref": [], "snan": ["-finit-real=snan", "-finit-integer=-2147483647", "-fcheck=all"],
                    "zero": ["-finit-real=zero", "-fcheck=all"]}[mode]
            commands = []
            for file in [LEGACY / "stub_ucm.F90", src / "share/module_model_constants.F",
                         src / "frame/module_wrf_error.F", src / "phys/module_sf_urban.F"]:
                command = ["gfortran", "-c", *FLAGS, *init, "-I.", file]
                commands.append(command)
                p = run(command, cwd=work)
                (work / (file.name + ".log")).write_text(p.stdout + p.stderr)
            old = work / "driver-old.F90"
            old.write_text(legacy)
            command = ["gfortran", "-c", *FLAGS, *init, "-I.", old]
            p = run(command, cwd=work, allow_fail=True)
            (work / "interface.log").write_text(p.stdout + p.stderr)
            if tag == "v4.8.0":
                if p.returncode == 0:
                    raise ValueError("negative interface check unexpectedly compiled")
                if "Rank mismatch" not in p.stderr and "Type mismatch" not in p.stderr:
                    raise ValueError("negative interface check failed for an unrelated reason: " + p.stderr)
                driver = work / "driver.F90"
                driver.write_text(driver480(legacy))
                commands.append(["gfortran", "-c", *FLAGS, *init, "-I.", driver])
                p = run(commands[-1], cwd=work)
                (work / "driver.log").write_text(p.stdout + p.stderr)
                obj = "driver.o"
            else:
                if p.returncode:
                    raise ValueError(p.stderr)
                obj = "driver-old.o"
            commands.append(["gfortran", *FLAGS, *init, "-o", "run_ucm", "stub_ucm.o",
                             "module_model_constants.o", "module_wrf_error.o", "module_sf_urban.o", obj])
            run(commands[-1], cwd=work)
            if tag == "v4.8.0" and mode == "snan":
                commands.append(["gfortran", "-c", *FLAGS, *init, "-ffpe-trap=invalid",
                                 "-I.", "-o", "driver-trap.o", driver])
                run(commands[-1], cwd=work)
                commands.append(["gfortran", *FLAGS, *init, "-ffpe-trap=invalid", "-o", "run_ucm_trap",
                                 "stub_ucm.o", "module_model_constants.o", "module_wrf_error.o",
                                 "module_sf_urban.o", "driver-trap.o"])
                run(commands[-1], cwd=work)
            (work / "commands.json").write_text(json.dumps(commands, indent=2, default=str))
            for table in ("URBPARM.TBL", "URBPARM_LCZ.TBL"):
                (work / table).write_bytes((src / "run" / table).read_bytes())
            (work / "source-hashes.json").write_text(json.dumps({name: sha(src / name) for name in SOURCE_FILES}, indent=2))
    return provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    args.sources = args.sources.resolve()
    args.build = args.build.resolve()
    provenance = fetch(args.sources)
    provenance = build(args.sources, args.build, provenance)
    args.receipt.write_text(json.dumps(provenance, indent=2))
    print("Pinned schemes compiled; legacy 4.8 ABI negative check rejected.")


if __name__ == "__main__":
    main()
