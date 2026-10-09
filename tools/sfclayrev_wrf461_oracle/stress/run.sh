#!/usr/bin/env bash
# Run from the lane's src, on a sprint box. GPU checks require the card queue.
set -euo pipefail
export SFCLAYREV_CHECK_ROOT=${SFCLAYREV_CHECK_ROOT:?set the owned lane path}
export PYTHONPATH="$PWD:$PWD/tests"
export PYTHONDONTWRITEBYTECODE=1
export CUPY_CACHE_DIR="$SFCLAYREV_CHECK_ROOT/cupy-cache"
export TMPDIR="$SFCLAYREV_CHECK_ROOT/tmp"
mkdir -p "$TMPDIR"
tool=tools/sfclayrev_wrf461_oracle
case "$1" in
prepare)
    mkdir -p "$SFCLAYREV_CHECK_ROOT/results" "$SFCLAYREV_CHECK_ROOT/logs"
    bash "$tool/build.sh" "$WRF_SOURCE_ROOT" "$SFCLAYREV_CHECK_ROOT/oracle"
    python "$tool/stress/make-extra.py"
    python "$tool/stress/replay.py" build
    (
        cd "$SFCLAYREV_CHECK_ROOT/oracle"
        gfortran -O0 -ffree-form -ffree-line-length-none -I . \
            ../extra-driver.F90 ccpp_kind_types.o sf_sfclayrev_O0.o \
            module_sf_sfclayrev_O0.o -o run_extra
        ./run_extra ../extra/sfclayrev-inputs.hex ../extra/sfclayrev-outputs.hex
        ./run_sfclayrev "$OLDPWD/tests/data/oracles/sfclayrev/sfclayrev-inputs.hex" \
            ../results/seeded-outputs.hex --seed-exchange
    )
    ;;
strict|default)
    mode=$1
    if [ "$mode" = strict ]; then export GPUWM_WRF_EXACT=1; else unset GPUWM_WRF_EXACT; fi
    python -m pytest -q -rs --basetemp="$TMPDIR/pytest-$mode" tests/test_sfclayrev_flux_disable_replay.py \
        tests/test_sfclayrev_nonfinite.py tests/test_sfclayrev_wrf461_parity.py \
        tests/test_sfclay.py > "$SFCLAYREV_CHECK_ROOT/logs/after-tests-$mode.log" 2>&1
    python "$tool/stress/replay.py" "$mode" \
        > "$SFCLAYREV_CHECK_ROOT/logs/after-replay-$mode.log" 2>&1
    for kind in extra seeded baseline-audit; do
        # The legacy comparator reports the declared WRF NaN differences.
        # Only this gate, with a WRF-only finite mask, decides extra success.
        python "$tool/stress/extra-check.py" "$mode" "$kind" \
            > "$SFCLAYREV_CHECK_ROOT/logs/after-$kind-$mode.log" 2>&1 || {
            if [ "$kind" != extra ]; then exit 1; fi
        }
        python "$tool/stress/check.py" "$mode" "$kind" \
            >> "$SFCLAYREV_CHECK_ROOT/logs/after-$kind-$mode.log" 2>&1
    done
    ;;
*) echo 'usage: run.sh prepare|strict|default' >&2; exit 2 ;;
esac
