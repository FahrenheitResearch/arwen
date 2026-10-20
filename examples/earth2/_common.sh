#!/usr/bin/env bash
# Shared argument handling for the Earth-2 recipes.
set -euo pipefail
earth2_hours=24
earth2_members=18
earth2_out=
earth2_model=
earth2_short=false
earth2_hours_override=
earth2_fetch_receipt=
while (($#)); do
    case "$1" in
        --short) earth2_hours=${earth2_short_hours:-12}; earth2_members=1; earth2_short=true; shift ;;
        --model)
            if (($# < 2)) || [[ -z "$2" ]]; then
                printf '%s\n' '--model needs an authority row ID' >&2; exit 2
            fi
            earth2_model=$2; shift 2 ;;
        --hours)
            if (($# < 2)) || [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                printf '%s\n' '--hours needs a positive integer' >&2; exit 2
            fi
            earth2_hours_override=$2; shift 2 ;;
        --fetch-receipt)
            if (($# < 2)) || [[ ! -f "$2" ]]; then
                printf '%s\n' '--fetch-receipt needs an existing earth2-run.json' >&2; exit 2
            fi
            earth2_fetch_receipt=$2; shift 2 ;;
        --out)
            if (($# < 2)) || [[ -z "$2" ]]; then
                printf '%s\n' '--out needs a directory' >&2; exit 2
            fi
            earth2_out=$2; shift 2 ;;
        --help|-h)
            printf '%s\n' "Usage: $0 [--short] [--model ID] [--hours N] --out DIR" \
                'Proof status and measured short settings: examples/earth2/README.md.' \
                '--short: UCast, 12 hours, one member per source; storm uses a cached FCN3 fetch for 6 hours.' \
                '--hours: override the duration, respecting the selected model cadence.' \
                '--fetch-receipt FILE: offer a completed fetch for receipt-checked reuse.' \
                'WOOF_BIN chooses the installed command (default: woof; engine tree: gpuwm).'
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if "$earth2_short" && [[ ${earth2_short_requires_fetch:-false} == true && -z "$earth2_fetch_receipt" ]]; then
    printf '%s\n' 'The storm short version requires --fetch-receipt FILE from a completed larger-card fetch.' >&2
    exit 2
fi
if [[ -n "$earth2_hours_override" ]]; then earth2_hours=$earth2_hours_override; fi
if [[ -n "$earth2_fetch_receipt" ]]; then
    export GPUWM_EARTH2_FETCH_RECEIPTS=$earth2_fetch_receipt
fi
if [[ -z "$earth2_model" ]]; then
    earth2_model=fcn3
    if "$earth2_short"; then earth2_model=${earth2_short_model:-ucast}; fi
fi
if [[ -z "$earth2_out" ]]; then
    printf '%s\n' '--out DIR is required' >&2; exit 2
fi
earth2_cli=${WOOF_BIN:-woof}
if ! command -v "$earth2_cli" >/dev/null 2>&1; then
    printf 'Command unavailable: %s (set WOOF_BIN=gpuwm for the engine spelling)\n' "$earth2_cli" >&2
    exit 127
fi
