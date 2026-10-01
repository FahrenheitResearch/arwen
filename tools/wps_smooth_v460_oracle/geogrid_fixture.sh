#!/usr/bin/env bash
# Run WPS v4.6.0's own geogrid.exe on one small real 1 km domain once per
# terrain-smoothing setting, plus once on the same lattice widened by the
# 3-cell geogrid halo with no smoothing (WPS's own unsmoothed halo-extended
# HGT_M, the smoother's input).  make_geogrid_fixture.py packs the result.
#
#   geogrid_fixture.sh WPS_BUILD_DIR GEOG_ROOT WORK_DIR
#
# WPS_BUILD_DIR is a WPS tree at the pinned commit (see README.md) built with
# ./configure (serial GNU) and ./compile geogrid.  Each run's GEOGRID.TBL is
# the stock GEOGRID.TBL.ARW with only HGT_M's smooth line changed (or removed
# for "none"), cut to the HGT_M and LANDUSEF entries a GMTED/MODIS-only
# GEOG tree can serve.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: geogrid_fixture.sh WPS_BUILD_DIR GEOG_ROOT WORK_DIR" >&2
    exit 2
fi
wps=$(realpath "$1")
geog=$(realpath "$2")
work=$(realpath -m "$3")
stock=$wps/geogrid/GEOGRID.TBL.ARW
pinned_commit="335c76a111f84503e8b963abaf273ea8053645bb"
if [[ "$(git -C "$wps" rev-parse HEAD)" != "$pinned_commit" ]]; then
    echo "WPS tree is not at the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "$wps" diff --quiet HEAD -- geogrid/src geogrid/GEOGRID.TBL.ARW; then
    echo "the WPS geogrid sources or stock table differ from the pinned commit" >&2
    exit 3
fi
mkdir -p "$work"

# One small 1 km Lambert domain over deep Alpine valleys.
e_we=49; e_sn=41; dx=1000; lat=46.50; lon=9.90
settings=(
  "none|"
  "passes0|smooth_option = smth-desmth_special; smooth_passes=0"
  "special1|smooth_option = smth-desmth_special; smooth_passes=1"
  "special2|smooth_option = smth-desmth_special; smooth_passes=2"
  "special3|smooth_option = smth-desmth_special; smooth_passes=3"
  "smdes1|smooth_option = smth-desmth; smooth_passes=1"
  "smdes2|smooth_option = smth-desmth; smooth_passes=2"
  "smdes3|smooth_option = smth-desmth; smooth_passes=3"
  "121x1|smooth_option = 1-2-1; smooth_passes=1"
  "121x2|smooth_option = 1-2-1; smooth_passes=2"
  "121x3|smooth_option = 1-2-1; smooth_passes=3"
  "121x5|smooth_option = 1-2-1; smooth_passes=5"
)

write_namelist() {  # dir e_we e_sn ref_x ref_y
  cat > "$1/namelist.wps" <<EOF
&share
 wrf_core = 'ARW',
 max_dom = 1,
 start_date = '2026-01-15_00:00:00',
 end_date   = '2026-01-15_00:00:00',
 interval_seconds = 3600,
 io_form_geogrid = 2,
/
&geogrid
 parent_id         = 1,
 parent_grid_ratio = 1,
 i_parent_start    = 1,
 j_parent_start    = 1,
 e_we              = $2,
 e_sn              = $3,
 geog_data_res     = 'default',
 dx = $dx,
 dy = $dx,
 map_proj = 'lambert',
 ref_lat   = $lat,
 ref_lon   = $lon,
 ref_x     = $4,
 ref_y     = $5,
 truelat1  = 45.0,
 truelat2  = 48.0,
 stand_lon = $lon,
 geog_data_path = '$geog',
 opt_geogrid_tbl_path = './',
/
EOF
}

write_table() {  # dir smooth-line
  python3 - "$stock" "$1/GEOGRID.TBL" "$2" <<'PY'
import sys
stock, out, line = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(stock).read().split("\n")
target = "        smooth_option = smth-desmth_special; smooth_passes=1"
assert text.count(target) == 1, "stock HGT_M smooth line not found once"
i = text.index(target)
if line:
    text[i] = "        " + line
else:
    del text[i]
blocks, cur = [], []
for row in text:
    if "=====" in row:
        blocks.append(cur)
        cur = []
    else:
        cur.append(row)
blocks.append(cur)
keep = [b for b in blocks
        if any(r.replace(" ", "") in ("name=HGT_M", "name=LANDUSEF") for r in b)]
assert len(keep) == 2, [b[:1] for b in keep]
sep = "==============================="
open(out, "w").write(sep + "\n" + ("\n" + sep + "\n").join("\n".join(b) for b in keep)
                     + "\n" + sep + "\n")
PY
}

run_geogrid() {  # dir
  (cd "$1" && ln -sf "$wps/geogrid/src/geogrid.exe" geogrid.exe \
     && ./geogrid.exe > geogrid.log 2>&1 \
     && grep -q "Successful completion of geogrid" geogrid.log) \
    || { echo "geogrid failed in $1" >&2; exit 1; }
}

# ref_x/ref_y at the mass-grid centre; the wide lattice shifts both by the
# 3-cell halo so its points are the small domain's, three more each side.
ref_x=$(python3 -c "print($e_we / 2)"); ref_y=$(python3 -c "print($e_sn / 2)")
for item in "${settings[@]}"; do
  name=${item%%|*}; line=${item#*|}
  d=$work/$name
  mkdir -p "$d"
  write_namelist "$d" "$e_we" "$e_sn" "$ref_x" "$ref_y"
  write_table "$d" "$line"
  run_geogrid "$d"
done
d=$work/wide-none
mkdir -p "$d"
write_namelist "$d" $((e_we + 6)) $((e_sn + 6)) \
  "$(python3 -c "print($e_we / 2 + 3)")" "$(python3 -c "print($e_sn / 2 + 3)")"
write_table "$d" ""
run_geogrid "$d"

{
  echo "wps_commit=${pinned_commit}"
  echo "sha256 $(sha256sum "$stock" | cut -d' ' -f1) geogrid/GEOGRID.TBL.ARW"
  echo "gfortran=$(gfortran --version | head -n 1)"
  echo "configure.wps $(grep -E '^(FFLAGS|CPPFLAGS)' "$wps/configure.wps" | tr -s ' ' | tr '
' ';')"
  echo "domain=lambert ${e_we}x${e_sn} dx=${dx} ref=${lat},${lon} truelat=45,48"
  echo "geog_topo=$(sha256sum "$geog"/topo_gmted2010_30s/index | cut -d' ' -f1) topo_gmted2010_30s/index"
} > "$work/provenance.txt"
cat "$work/provenance.txt"
