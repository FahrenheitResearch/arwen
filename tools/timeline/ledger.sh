#!/usr/bin/env bash
# Source this file, then ledger_emit STAGE [begin|end|mark] [ATTRS_JSON].
# GPUWM_TIME_LEDGER is the rental-wide destination. No Python package install
# is required. A failed emitter cannot stop a boot script using set -e.
_GPUWM_LEDGER_PY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/gpuwm/timeline/ledger.py"
ledger_emit() {
    local diagnostic attrs
    attrs="${3-}"
    [[ -n "$attrs" ]] || attrs='{}'
    diagnostic=$("${GPUWM_LEDGER_PYTHON:-python3}" "$_GPUWM_LEDGER_PY" \
        "${1-}" "${2:-mark}" --attrs "$attrs" 2>&1) || :
    if [[ -n "$diagnostic" && "${_GPUWM_LEDGER_WARNED_PATH:-}" != "${GPUWM_TIME_LEDGER:-<unset>}" ]]; then
        if [[ "$diagnostic" == 'time ledger:'* ]]; then
            printf '%s\n' "$diagnostic" >&2 || :
        else
            printf 'time ledger: %s: %s\n' "${GPUWM_TIME_LEDGER:-<unset>}" "$diagnostic" >&2 || :
        fi
        _GPUWM_LEDGER_WARNED_PATH="${GPUWM_TIME_LEDGER:-<unset>}"
    fi
    return 0
}
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    ledger_emit "$@"
fi
