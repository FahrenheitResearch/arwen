#!/usr/bin/env bash
# Measure what one observation did to a WRF-format background: the analysis
# increment of each field GSI rewrote, its peak, and how far the temperature
# increment reaches from its peak. Runs on any host with NCO (ncdiff, ncks).
#
#   singleob_increments.sh <run dir of case/run_singleob.sh>
#
# Writes increments.nc and increments.txt in the run folder. Reach is the
# distance from the peak in each direction along its grid row and column,
# to the first |dT| below 1/e and 1/10 of the peak. A domain edge is a
# censored distance, never a zero-valued point beyond the available data.
# Distances use nominal DX, not a projection-adjusted physical distance.
set -euo pipefail
run="$1"; cd "$run"
fields="T U V W QVAPOR QCLOUD QRAIN QSNOW QICE QGRAUP MU PH"
avail=" $(ncdump -h wrf_inout | awk '/^\t[a-z]+ [A-Z_0-9]+\(Time,/ {sub(/\(.*/,"",$2); print $2}' | tr '\n' ' ') "
use=""
for v in $fields; do [[ "$avail" == *" $v "* ]] && use="${use:+$use,}$v"; done
rm -f increments.nc
ncdiff -O -v "$use" wrf_inout wrf_inout.background increments.nc
dx_km=$(ncdump -h wrf_inout | awk -F'= ' '/:DX =/ {gsub(/[ f;]/,"",$2); print $2/1000}')

# One line per value in NCO's traditional form:
#   Time[0] bottom_top[k] south_north[j] west_east[i] T[n]=value
peakline() {  # peakline <var>: the line of the largest |value|
  ncks --trd -H -C -v "$1" increments.nc | awk '
    /=/ { n=split($0, a, "="); val=a[n]+0; if (val<0) val=-val; if (val>m) {m=val; line=$0} }
    END { printf "%.6g|%s\n", m, line }'
}
index_of() {  # index_of <dimension> <line>
  sed -nE "s/.*[[:space:]]$1\[([0-9]+)\].*/\1/p" <<< "$2"
}
{
  printf '%-8s %-14s %s\n' field 'max |inc|' 'where (k, j, i)'
  for v in ${use//,/ }; do
    pl=$(peakline "$v"); m=${pl%%|*}; line=${pl#*|}
    k=$(index_of bottom_top "$line"); j=$(index_of south_north "$line"); i=$(index_of west_east "$line")
    [[ -n "$k" ]] || k=$(index_of bottom_top_stag "$line")
    [[ -n "$j" ]] || j=$(index_of south_north_stag "$line")
    [[ -n "$i" ]] || i=$(index_of west_east_stag "$line")
    printf '%-8s %-14s %s\n' "$v" "$m" "(${k:--}, ${j:--}, ${i:--})"
  done
} > increments.txt

reach() {  # reach <-d options for one line through the peak> <peak index on it>
  # shellcheck disable=SC2086
  ncks --trd -H -C -v T -d Time,0 $1 increments.nc | awk -v p="$2" -v dx="$dx_km" '
    /=/ { n2=split($0, a, "="); x[n]=a[n2]+0; n++ }
    function crossing(sign, threshold, label, d, i, v) {
      for (d=1; ; d++) {
        i=p+sign*d
        if (i<0 || i>=n) return sprintf("%s beyond edge (%d points, %.0f nominal km)", label, d-1, (d-1)*dx)
        v=x[i]; if (v<0) v=-v
        if (v<threshold) return sprintf("%s %d points (%.0f nominal km)", label, d, d*dx)
      }
    }
    END { pk=x[p]; if (pk<0) pk=-pk
          if (pk==0) {print "zero peak; reach undefined"; exit}
          printf "1/e: %s, %s; 1/10: %s, %s", crossing(-1,pk/exp(1),"negative"), crossing(1,pk/exp(1),"positive"), crossing(-1,pk/10,"negative"), crossing(1,pk/10,"positive") }'
}
pl=$(peakline T); line=${pl#*|}
if [[ -z "$line" ]]; then
  echo "T increment is zero; reach undefined" >> increments.txt
  cat increments.txt
  exit 0
fi
k=$(index_of bottom_top "$line"); j=$(index_of south_north "$line"); i=$(index_of west_east "$line")
{
  echo
  echo "T increment peak ${pl%%|*} K at bottom_top=$k south_north=$j west_east=$i; DX=${dx_km} km"
  echo "  along west_east:   $(reach "-d bottom_top,$k -d south_north,$j" "$i")"
  echo "  along south_north: $(reach "-d bottom_top,$k -d west_east,$i" "$j")"
} >> increments.txt
cat increments.txt
