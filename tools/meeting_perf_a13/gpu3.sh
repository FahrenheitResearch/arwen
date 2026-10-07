#!/usr/bin/env bash
set -u
W=/work/meeting-perf-a13; c=$W/case/c3; R=$W/r3
trap "date -u +%FT%TZ > $W/JOB3-DONE" EXIT
source $W/envE.sh
echo "cards=$GPU_MUTEX_CARDS start=$(date -u +%FT%TZ)"
PROOF=$(sha256sum $c/prepared/proof.json | cut -d' ' -f1)
SRC=$(python -c "import json;p=json.load(open('$c/prepared/proof.json'));print(p.get('input_manifest_sha256',''))")
CONT=$(python -c "import json;p=json.load(open('$c/prepared/proof.json'));print(p['prepared_cache']['content_sha256'])")
[ -n "$SRC" ] || SRC=$(sha256sum $c/prepared/source-evidence/input-manifest.json | cut -d" " -f1)
echo "SRC=$SRC CONT=$CONT"
run() {  # arm tree grid
  arm=$1 tree=$2 grid=$3; out=$R/$arm
  [ -e $out ] && { echo "$arm: $out exists, not overwriting"; return; }
  mkdir -p $out
  ( while :; do /usr/bin/nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i $GPU_MUTEX_CARDS | awk -v t=$(date +%s.%N) '{print t", "$0}'; sleep 0.2; done ) > $out/nvml.csv 2>/dev/null &
  smi=$!; t0=$(date +%s)
  PYTHONPATH=$W/sitex:$W/trees/$tree python -m gpuwm.prepared_single_domain_forecast --source rap-native --prepared-root $c/prepared \
     --proof-sha256 $PROOF --source-manifest-sha256 $SRC --prepared-content-sha256 $CONT \
     --experiment-config $c/experiment.toml --wps-namelist $c/namelist.wps --progress-format jsonl --io-mode history \
     --devices-table "{\"count\": 4, \"grid\": \"$grid\", \"ids\": [0, 1, 2, 3], \"transport\": \"auto\"}" \
     --outdir $out/out > $out/run.log 2>&1
  echo "$arm rc=$? wall=$(( $(date +%s) - t0 ))"; kill $smi 2>/dev/null
  ( cd $out/out && find . -name "wrfout*" -type f | sort | xargs -r sha256sum ) > $out/frames.sha256
  grep -E "ADMITTED|REFUSED" $out/run.log | head -8
}
cmp() { python $W/cmp.py $R/$1/out $R/$2/out > $R/cmp-$1-vs-$2.json 2> $R/cmp-$1-vs-$2.err; echo "cmp $1 vs $2 rc=$?"
  python -c "import json;d=json.load(open('$R/cmp-$1-vs-$2.json'));print({k:(sorted(v['differing_variables'])[:40],sorted(v['differing_global_attrs'])) for k,v in d['files'].items()})"; }
# The tip arms run the lane engine itself: since 5424bb19b (the rank pool byte bound) the 2x2 layout is admitted by
# the default gate, so the ADMITTED lines are the proof; the measurement copy is kept only for the base arms below.
run tip-2x2 tip 2x2
run tip-1x4 tip 1x4
cmp tip-2x2 tip-1x4
run base-2x2 base-meas 2x2
run base-1x4 base-meas 1x4
cmp base-2x2 base-1x4
cmp base-2x2 tip-2x2
echo "end=$(date -u +%FT%TZ)"
