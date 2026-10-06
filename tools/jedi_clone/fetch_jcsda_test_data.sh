#!/usr/bin/env bash
# Fetch tier-1 test data at one commit of one of JCSDA's public test-data
# repositories (mpas-jedi-data, ufo-data), without git-lfs, and check every
# file against the SHA-256 its LFS pointer records.
#
#   fetch_jcsda_test_data.sh <owner/repo> <commit> <output dir> [path...]
#
# With no paths, every file under testinput_tier_1/ is fetched; with paths
# (relative to the repository root), only those.
#
# Why a fetcher: RDASApp's own test-data links point into a fix tree that
# NOAA's machines provide (fix/.agent/jcsda/<repo>_<commit>_<date>), and
# NOAA's build notes skip git-lfs because JCSDA's LFS budget runs out. The
# data must match the code: mpas-jedi 20043d6 (the commit RDASApp 1bfecf33
# pins) reads x1.2562.invariant.nc, which mpas-jedi-data carries from
# 29a655c and neither its 3fff9f5 snapshot nor the 3.1.0 release tarball
# has; ufo 32d8f9e's aircraft test stops on the 1.10.0 release tarball's
# observation file (MEASURED 2026-10-03).
set -euo pipefail
repo="$1"; commit="$2"; out="$3"; shift 3
mkdir -p "$out"; cd "$out"
if [[ $# -gt 0 ]]; then
  printf '%s\n' "$@" > files.txt
else
  curl -fsSL "https://api.github.com/repos/$repo/git/trees/$commit?recursive=1" \
    | python3 -c 'import sys,json; [print(t["path"]) for t in json.load(sys.stdin)["tree"] if t["type"]=="blob" and t["path"].startswith("testinput_tier_1/")]' \
    > files.txt
fi
: > checks.txt
while read -r path; do
  mkdir -p "$(dirname "$path")"
  pointer="$(curl -fsSL "https://raw.githubusercontent.com/$repo/$commit/$path")"
  oid="$(sed -n 's/^oid sha256://p' <<< "$pointer")"
  if [[ -n "$oid" ]]; then
    curl -fsSL --retry 3 -o "$path" "https://media.githubusercontent.com/media/$repo/$commit/$path"
    got="$(sha256sum "$path" | cut -d' ' -f1)"
    [[ "$got" == "$oid" ]] || { echo "MISMATCH $path $got != $oid" >&2; exit 1; }
    echo "$oid  $path" >> checks.txt
  else
    printf '%s' "$pointer" > "$path"
    echo "$(sha256sum "$path" | cut -d' ' -f1)  $path (not LFS)" >> checks.txt
  fi
done < files.txt
echo "$repo $commit: $(wc -l < files.txt) files, every LFS object matches its pointer"
