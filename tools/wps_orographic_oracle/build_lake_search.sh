#!/usr/bin/env bash
set -euo pipefail
script=$(cd "$(dirname "$0")" && pwd)
source=$(realpath "${1:?WPS geogrid source directory required}")
output=$(realpath -m "${2:?output directory required}")
mkdir -p "$output"
cd "$output"
for name in constants_module misc_definitions_module parallel_module module_debug module_map_utils bitarray_module queue_module interp_module; do
 cp "$source/$name.F" "$output/$name.F"
done
cp "$source/cio.c" "$output/cio.c"
sha256sum -c "$script/SOURCE.sha256"
for name in constants_module misc_definitions_module parallel_module module_debug module_map_utils bitarray_module queue_module interp_module; do
 gfortran -O0 -cpp -D_GEOGRID -ffree-form -ffree-line-length-none -fno-fast-math -ffp-contract=off -c "$name.F"
done
gcc -O0 -D_UNDERSCORE -c cio.c
gfortran -O0 -ffree-line-length-none -ffp-contract=off "$script/run_lake_search.F90" constants_module.o misc_definitions_module.o parallel_module.o module_debug.o module_map_utils.o bitarray_module.o queue_module.o interp_module.o cio.o -o lake-search-oracle
./lake-search-oracle
sha256sum search-real.bin lake-gcell-real.bin
sha256sum -c "$script/LAKE-FIXTURES.sha256"
