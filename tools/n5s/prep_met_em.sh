#!/usr/bin/env bash
# Prepare the six-hourly 12Z/18Z d01 WPS forcing in disposable WSL storage.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUNDLE=/mnt/c/Users/drew/Downloads/WRF_1974_MP55_reference_bundle
WPS_ROOT=/home/drew/WPS-4.6.0
WORK_ROOT=/home/drew/n5s-work
WORKDIR=
ENV_FILE="$SCRIPT_DIR/wsl-env.sh"
# The retained bundle GRIB omits PMSL and the invariant surface
# geopotential; real.exe (sfcp_to_sfcp) needs them.  A CDS-downloaded
# supplement GRIB (msl + z) is linked alongside the bundle GRIB and the
# case Vtable is extended with the matching rows from WPS's shipped
# Vtable.ECMWF (ground truth for codes 151/129).
SUPPLEMENT_GRIB=

usage() {
  cat <<'EOF'
Usage: prep_met_em.sh [options]

Options:
  --bundle DIR      Read-only WRF reference bundle.
  --wps-root DIR    WPS 4.6.0 build (default /home/drew/WPS-4.6.0).
  --work-root DIR   Disposable parent (default /home/drew/n5s-work).
  --workdir DIR     Explicit new/empty work directory under work-root.
  --env-file FILE   Runtime environment profile (default tools/n5s/wsl-env.sh).
  --supplement-grib FILE
                    CDS supplement GRIB carrying msl (151) + z (129);
                    linked alongside the bundle GRIB and the Vtable is
                    extended with the WPS Vtable.ECMWF rows for both.
  -h, --help        Show this help.

The script never writes to the reference bundle or oracle build.  It runs
ungrib/metgrid only in a disposable work directory and prints that directory
as its final line for use by run_real.sh.
EOF
}

while (($#)); do
  case "$1" in
    --bundle) BUNDLE=$2; shift 2 ;;
    --wps-root) WPS_ROOT=$2; shift 2 ;;
    --work-root) WORK_ROOT=$2; shift 2 ;;
    --workdir) WORKDIR=$2; shift 2 ;;
    --env-file) ENV_FILE=$2; shift 2 ;;
    --supplement-grib) SUPPLEMENT_GRIB=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$ENV_FILE" ]] || { echo "missing runtime environment file: $ENV_FILE" >&2; exit 2; }
# shellcheck source=/dev/null
source "$ENV_FILE"

for required in \
  "$BUNDLE/era5_grib/era5_19740403.grb" \
  "$BUNDLE/era5_grib/Vtable.ERA5_CDO" \
  "$BUNDLE/namelists/namelist.wps" \
  "$WPS_ROOT/ungrib.exe" "$WPS_ROOT/metgrid.exe"; do
  [[ -f "$required" ]] || { echo "missing required input: $required" >&2; exit 2; }
done
for domain in 01 02 03 04; do
  [[ -f "$BUNDLE/geo_em/geo_em.d${domain}.nc" ]] || {
    echo "missing bundle geo_em.d${domain}.nc" >&2; exit 2; }
done

mkdir -p -- "$WORK_ROOT"
WORK_ROOT=$(realpath -- "$WORK_ROOT")
if [[ -z "$WORKDIR" ]]; then
  WORKDIR=$(mktemp -d "$WORK_ROOT/prep-met-em.XXXXXX")
else
  mkdir -p -- "$WORKDIR"
  WORKDIR=$(realpath -- "$WORKDIR")
  [[ "$WORKDIR" == "$WORK_ROOT"/* ]] || {
    echo "--workdir must be below $WORK_ROOT" >&2; exit 2; }
  [[ -z "$(find "$WORKDIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    echo "--workdir must be empty: $WORKDIR" >&2; exit 2; }
fi

# Canonicalize the supplement BEFORE any cd so caller-relative paths
# keep their meaning (verify3 D4).
if [[ -n "$SUPPLEMENT_GRIB" ]]; then
  [[ -f "$SUPPLEMENT_GRIB" ]] || {
    echo "supplement GRIB not found: $SUPPLEMENT_GRIB" >&2; exit 2; }
  SUPPLEMENT_GRIB=$(realpath -- "$SUPPLEMENT_GRIB")
fi

cd -- "$WORKDIR"
ln -s -- "$WPS_ROOT/ungrib.exe" ungrib.exe
ln -s -- "$WPS_ROOT/metgrid.exe" metgrid.exe
ln -s -- "$BUNDLE/era5_grib/era5_19740403.grb" GRIBFILE.AAA
if [[ -n "$SUPPLEMENT_GRIB" ]]; then
  ECMWF_VTABLE="$WPS_ROOT/ungrib/Variable_Tables/Vtable.ECMWF"
  [[ -f "$ECMWF_VTABLE" ]] || {
    echo "WPS Vtable.ECMWF not found: $ECMWF_VTABLE" >&2; exit 2; }
  ln -s -- "$SUPPLEMENT_GRIB" GRIBFILE.AAB
  # Extend a COPY of the case Vtable with the shipped ECMWF rows for
  # GRIB1 code 151 (msl, surface level type 1) and code 129 AT SURFACE
  # LEVEL TYPE 1 (SOILGEO).  Rows are keyed on code AND level type:
  # Vtable.ECMWF also carries code 129 at level type 100 (pressure-level
  # GEOPT), which must NOT satisfy or supply the surface mapping
  # (verify3 D2).  All greps run under an explicit || true so the
  # promised diagnostics are reachable under set -e (verify3 D3).
  cp -- "$BUNDLE/era5_grib/Vtable.ERA5_CDO" Vtable
  : > vtable-extension.txt
  for code in 151 129; do
    have=$(grep -E "^ *${code} \| *1 \|" Vtable || true)
    if [[ -n "$have" ]]; then
      echo "case Vtable already maps GRIB1 code ${code} at surface level type 1" \
        >> vtable-extension.txt
      continue
    fi
    row=$(grep -E "^ *${code} \| *1 \|" "$ECMWF_VTABLE" | head -1 || true)
    [[ -n "$row" ]] || {
      echo "Vtable.ECMWF has no surface (level type 1) row for GRIB1 code ${code}" >&2
      exit 2; }
    # Insert before the trailing separator line so the table stays valid.
    sep_line=$(grep -nE '^-+' Vtable | tail -1 | cut -d: -f1 || true)
    if [[ -n "$sep_line" ]]; then
      sed -i "${sep_line}i\\${row}" Vtable
    else
      printf '%s\n' "$row" >> Vtable
    fi
    printf 'appended from Vtable.ECMWF: %s\n' "$row" >> vtable-extension.txt
  done
else
  ln -s -- "$BUNDLE/era5_grib/Vtable.ERA5_CDO" Vtable
fi
for domain in 01 02 03 04; do
  ln -s -- "$BUNDLE/geo_em/geo_em.d${domain}.nc" "geo_em.d${domain}.nc"
done
if [[ -f "$WPS_ROOT/metgrid/METGRID.TBL.ARW" ]]; then
  ln -s -- "$WPS_ROOT/metgrid/METGRID.TBL.ARW" METGRID.TBL
elif [[ -f "$WPS_ROOT/metgrid/METGRID.TBL" ]]; then
  ln -s -- "$WPS_ROOT/metgrid/METGRID.TBL" METGRID.TBL
else
  echo "WPS METGRID.TBL not found below $WPS_ROOT/metgrid" >&2
  exit 2
fi

python3 "$SCRIPT_DIR/derive_wps_namelist.py" \
  "$BUNDLE/namelists/namelist.wps" namelist.wps
./ungrib.exe >ungrib.stdout.log 2>ungrib.stderr.log
./metgrid.exe >metgrid.stdout.log 2>metgrid.stderr.log

for stamp in 12_00_00 18_00_00; do
  file="met_em.d01.1974-04-03_${stamp}.nc"
  [[ -s "$file" ]] || { echo "metgrid did not produce $file" >&2; exit 1; }
done
# metgrid necessarily sees all four unchanged geo_em domains.  N5S consumes
# only d01 at 18Z; replace child products with the binding 12Z bundle bytes.
for domain in 02 03 04; do
  find "$WORKDIR" -maxdepth 1 -type f -name "met_em.d${domain}.*.nc" -delete
  source="$BUNDLE/met_em/met_em.d${domain}.1974-04-03_12_00_00.nc"
  [[ -f "$source" ]] || { echo "missing bundle child met_em: $source" >&2; exit 2; }
  ln -s -- "$source" "met_em.d${domain}.1974-04-03_12_00_00.nc"
done

python3 - "$WORKDIR" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = {}
for path in sorted(root.glob("met_em.d*.nc")):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    files[path.name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size,
                        "symlink": path.is_symlink()}
(root / "prep-met-em.json").write_text(
    json.dumps({"schema": 1, "workdir": str(root), "artifacts": files},
               indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

printf '%s\n' "$WORKDIR"
