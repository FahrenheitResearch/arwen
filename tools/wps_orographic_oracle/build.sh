#!/usr/bin/env bash
set -euo pipefail
# Compile byte-unmodified WPS v4.6.0 modules in an isolated output directory.
SCRIPT=$(cd "$(dirname "$0")" && pwd)
SOURCE=$(realpath "${1:?WPS geogrid source directory required}")
OUTPUT=$(realpath -m "${2:?output directory required}")
mkdir -p "$OUTPUT"
cd "$OUTPUT"
for name in constants_module misc_definitions_module parallel_module module_debug module_map_utils bitarray_module queue_module interp_module; do
 cp "$SOURCE/$name.F" "$OUTPUT/$name.F"
done
cp "$SOURCE/cio.c" "$OUTPUT/cio.c"
sha256sum -c "$SCRIPT/SOURCE.sha256"
for name in constants_module misc_definitions_module parallel_module module_debug module_map_utils bitarray_module queue_module interp_module; do
 gfortran -O0 -cpp -D_GEOGRID -ffree-form -ffree-line-length-none -fno-fast-math -ffp-contract=off -c "$name.F"
done
gcc -O0 -D_UNDERSCORE -c cio.c
gfortran -O0 -ffree-line-length-none -ffp-contract=off "$SCRIPT/run.F90" constants_module.o misc_definitions_module.o parallel_module.o module_debug.o module_map_utils.o bitarray_module.o queue_module.o interp_module.o cio.o -o oracle
./oracle
sha256sum mesh-real.bin mesh-edge-real.bin inverse-real.bin gcell-real.bin nint-real.bin interp-real.bin
sha256sum -c "$SCRIPT/FIXTURES.sha256"
