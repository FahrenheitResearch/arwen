#!/usr/bin/env bash
# Fetch the pinned public WRF releases that build.sh compiles, each with the
# MMM physics that release itself names:
#   v4.6.1  phys/physics_mmm is part of the tree;
#   v4.7.0, v4.7.1  arch/Externals.cfg (WRF's manage_externals), which the
#           stock build checks out before compiling phys/;
#   v4.8.0  the phys/physics_mmm git submodule.
#
#   fetch_sources.sh DEST [TAG ...]      (default: all four tags)
#
# build.sh then verifies every pinned commit and source digest.
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "usage: fetch_sources.sh DEST [TAG ...]" >&2
    exit 2
fi
D=$(realpath -m "$1"); shift
tags=("$@")
[[ ${#tags[@]} -gt 0 ]] || tags=(v4.6.1 v4.7.0 v4.7.1 v4.8.0)
mkdir -p "$D"
for tag in "${tags[@]}"; do
    root=$D/$tag
    [[ -d $root/.git ]] || git clone -q --depth 1 --branch "$tag" https://github.com/wrf-model/WRF.git "$root"
    [[ -f $root/phys/physics_mmm/bl_ysu.F90 ]] && continue
    if git -C "$root" config -f .gitmodules --get submodule.phys/physics_mmm.path > /dev/null; then
        git -C "$root" submodule update --init --depth 1 phys/physics_mmm
    elif [[ -f $root/arch/Externals.cfg ]]; then
        # The [MMM-physics] block names repo_url and tag, as checkout_externals reads them.
        url=$(awk -F' *= *' '/^\[/{s=$0} s=="[MMM-physics]" && $1=="repo_url"{print $2}' "$root/arch/Externals.cfg")
        ref=$(awk -F' *= *' '/^\[/{s=$0} s=="[MMM-physics]" && $1=="tag"{print $2}' "$root/arch/Externals.cfg")
        [[ -n $url && -n $ref ]] || { echo "$tag: arch/Externals.cfg names no MMM-physics tag" >&2; exit 3; }
        git clone -q --depth 1 --branch "$ref" "$url" "$root/phys/physics_mmm"
    else
        echo "$tag: no physics_mmm in the tree, no submodule and no Externals.cfg" >&2
        exit 3
    fi
done
