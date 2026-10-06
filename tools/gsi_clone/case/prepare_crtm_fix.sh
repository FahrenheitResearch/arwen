#!/usr/bin/env bash
# Lay the CRTM 2.3.0 release's big-endian coefficient files out flat, the
# shape the analysis script's FIXcrtm folder has (one folder holding
# <sensor>.SpcCoeff.bin, <sensor>.TauCoeff.bin, AerosolCoeff.bin,
# CloudCoeff.bin and the emissivity files).
#
#   prepare_crtm_fix.sh <crtm_v2.3.0.tar.gz> <output dir>
#
# GSI is built with -convert big_endian, so it reads the Big_Endian set.
# Where a sensor has both transmittance forms, ODPS is the one kept.
set -euo pipefail
tarball="$1"; out="$2"
work="$out.unpack"
mkdir -p "$out" "$work"
tar xzf "$tarball" -C "$work" --wildcards 'REL-2.3.0/fix/*Big_Endian/*.bin'
find "$work/REL-2.3.0/fix" -path '*ODAS/Big_Endian/*.bin' -exec mv -f {} "$out/" \;
find "$work/REL-2.3.0/fix" -path '*Big_Endian/*.bin' -exec mv -f {} "$out/" \;
rm -rf "$work"
ls "$out" | wc -l
ls -l "$out/AerosolCoeff.bin" "$out/CloudCoeff.bin" "$out/FASTEM5.MWwater.EmisCoeff.bin" \
      "$out/Nalli.IRwater.EmisCoeff.bin" "$out/NPOESS.IRice.EmisCoeff.bin"
