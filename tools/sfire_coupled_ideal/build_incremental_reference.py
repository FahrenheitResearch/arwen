"""Compile the corrected driver and relink a distinct complete WRF reference.

Unchanged native objects are reused by hash from the complete scalar
optimized build. The baseline source, library and executables stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def build(source: Path, baseline: Path, output: Path):
    source, baseline, output = source.resolve(), baseline.resolve(), output.resolve()
    if output.exists() or output == baseline or baseline in output.parents:
        raise ValueError("use a new incremental build directory outside the baseline build")
    output.mkdir(parents=True)
    (output / "main").mkdir()
    module_dir = output / "modules"
    module_dir.mkdir()
    flags_path = baseline / "CMakeFiles/WRF_Core.dir/flags.make"
    make_path = baseline / "CMakeFiles/WRF_Core.dir/build.make"
    flags = dict(re.findall(r"^(Fortran_\w+) = (.*)$", flags_path.read_text(), re.MULTILINE))
    corrections = json.loads((source / "SFIRE_CORRECTIONS.json").read_text())
    units = tuple(corrections["changed_sources"])
    executed = []
    objects = []
    recipes = make_path.read_text().splitlines()
    for unit in units:
        recipe = next(line.strip() for line in recipes
                      if line.startswith("\t/usr/bin/gfortran") and f"/{unit} -o" in line)
        for name, values in flags.items():
            recipe = recipe.replace(f"$({name})", values)
        command = shlex.split(recipe)
        for i, value in enumerate(command):
            if value.startswith("-J"):
                command[i] = "-J" + str(module_dir)
            elif value.endswith("/" + unit):
                command[i] = str(source / unit)
        object_path = output / (Path(unit).name + ".o")
        command[command.index("-o") + 1] = str(object_path)
        executed.append(command)
        subprocess.run(["nice", "-n", "15", *command], cwd=baseline, check=True)
        objects.append(object_path)
    archive = output / "libWRF_Core.a"
    shutil.copy2(baseline / "libWRF_Core.a", archive)
    members = subprocess.run(["ar", "t", str(archive)], text=True, capture_output=True, check=True).stdout.splitlines()
    if any(members.count(path.name) != 1 for path in objects):
        raise ValueError("the complete core archive must contain each corrected module exactly once")
    subprocess.run(["ar", "r", str(archive), *(str(path) for path in objects)], check=True)
    subprocess.run(["ranlib", str(archive)], check=True)
    for target in ("ideal", "wrf"):
        link_path = baseline / f"main/CMakeFiles/{target}.dir/link.txt"
        command = shlex.split(link_path.read_text())
        command[command.index("-o") + 1] = str(output / "main" / target)
        command = [str(archive) if value == "../libWRF_Core.a" else
                   f"-Wl,--dependency-file={output / 'main' / (target + '.link.d')}"
                   if value.startswith("-Wl,--dependency-file=") else value for value in command]
        executed.append(command)
        subprocess.run(["nice", "-n", "15", *command], cwd=baseline / "main", check=True)
    shutil.copy2(baseline / "CMakeCache.txt", output / "CMakeCache.txt")
    shutil.copy2(flags_path, output / "baseline-flags.make")
    (output / "compile-commands.json").write_text(json.dumps(executed, indent=2) + "\n")
    inputs = {str(path.relative_to(baseline)): digest(path)
              for path in sorted(baseline.rglob("*.a"))}
    original_executables = {name: digest(baseline / "main" / name) for name in ("ideal", "wrf")}
    receipt = {"schema": "sfire-incremental-native-build-v1", "baseline_build": str(baseline),
               "corrected_source": str(source), "corrected_driver_sha256": digest(source / "phys/module_fr_fire_driver.F"),
               "corrected_objects_sha256": {path.name: digest(path) for path in objects}, "reused_libraries_sha256": inputs,
               "baseline_executables_sha256": original_executables,
               "corrected_executables_sha256": {name: digest(output / "main" / name) for name in ("ideal", "wrf")},
               "commands": executed, "baseline_flags_sha256": digest(flags_path)}
    (output / "incremental-build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corrected_source", type=Path)
    parser.add_argument("baseline_build", type=Path)
    parser.add_argument("new_build", type=Path)
    args = parser.parse_args()
    result = build(args.corrected_source, args.baseline_build, args.new_build)
    print(json.dumps({key: result[key] for key in ("corrected_driver_sha256", "corrected_executables_sha256")}, sort_keys=True))
