"""Rebuild the delivered second-check boundary corpora inside this lane only."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PATCHES = Path(__file__).resolve().parent


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def main():
    lane, wrf = map(Path, sys.argv[1:3])
    tools = lane / "oracle-tools"
    tools.mkdir(exist_ok=True)
    for name in ("urban_wrf471_oracle", "urban_slucm_regime_oracle",
                 "urban_bep_wrf461_oracle", "urban_bem_regime_oracle"):
        shutil.copytree(ROOT / "tools" / name, tools / name, dirs_exist_ok=True)
    # The SLUCM scenario patch is relative to the original driver. Only the
    # six added switch rows from stress.patch are needed alongside it.
    stress = (PATCHES / "slucm-stress.patch").read_bytes()
    switches = stress[stress.index(b"--- a/tools/urban_slucm_regime_oracle/build.sh"):]
    subprocess.run(["patch", "-p2", "--batch", "--forward"], input=switches,
                   cwd=tools, check=True)
    # Apply the already delivered second-check drivers and build repair.
    for name in ("slucm-subnormal.patch", "bep-extra.patch", "bem-subnormal.patch",
                 "bep-build-pipeline.patch"):
        with (PATCHES / name).open() as patch:
            run("patch", "-p2", "--batch", "--forward", cwd=tools, stdin=patch)
    source = lane / "slucm-source"
    source.mkdir(exist_ok=True)
    pairs = {"module_model_constants.F": "share", "module_wrf_error.F": "frame",
             "URBPARM.TBL": "run", "URBPARM_LCZ.TBL": "run"}
    for name, parent in pairs.items():
        (source / name).symlink_to(wrf / parent / name)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/wrf-model/WRF/v4.7.1/phys/module_sf_urban.F",
        source / "module_sf_urban.F")
    assert hashlib.sha256((source / "module_sf_urban.F").read_bytes()).hexdigest() == \
        "623868c74c4b9d579e9c3811e9c334d731394c2afbfea7d693221626fbf0b0ea"
    (source / "SOURCES.sha256").write_text("".join(
        hashlib.sha256((source / name).read_bytes()).hexdigest() + "  " + name + "\n"
        for name in (*pairs, "module_sf_urban.F")))
    # Keep BEM builds to the two exact requested subnormal fixtures and only
    # the reference executable. The unchanged science source is hash checked.
    p = tools / "urban_bem_regime_oracle/build.sh"
    shutil.copyfile(PATCHES / "bem-extra-regimes.csv",
                    tools / "urban_bem_regime_oracle/REGIMES.csv")
    text = p.read_text()
    begin = text.index("variants=(")
    end = text.index("\n)", begin) + 2
    text = text[:begin] + '''variants=(
    "subnormal-nlcd 0 $src/run/URBPARM.TBL S 3 1"
    "subnormal-lcz 1 $src/run/URBPARM_LCZ.TBL W 3 1"
)''' + text[end:]
    text = text.replace('build_one snan "$src/phys/module_sf_bep_bem.F" -finit-real=snan -finit-integer=-2147483647', ':')
    text = text.replace('    run_one snan "$line" "$build/out-snan"', ':')
    p.write_text(text)
    print("prepared", tools)


if __name__ == "__main__":
    main()
