#!/usr/bin/env bash
# Fetch every source tarball a GSI pin names (pins/<pin>.pin, PIN_SOURCES)
# into src-pins/<pin>/ and write SOURCES-<pin>.sha256 beside this script.
# Dockerfile.pin checks those hashes before it unpacks anything, so a moved or
# re-rolled upstream archive stops the build instead of changing a library.
#
#   ./fetch_pin_sources.sh pins/gsi-db90edf.pin [--check]
#
# With --check the hash file is not rewritten: the downloads must match the
# committed one.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
pinfile="$1"; mode="${2:-write}"
# shellcheck disable=SC1090
. "$pinfile"
dest="$here/src-pins/$PIN_NAME"
mkdir -p "$dest"
cd "$dest"
while IFS='|' read -r name version url; do
  [[ -z "$name" ]] && continue
  out="$name-$version.tar.gz"
  if [[ -s "$out" ]]; then echo "have $out"; continue; fi
  echo "get  $out <- $url"
  if [[ "$url" == git+* ]]; then
    # git+<repo>@<commit>#<path>,<path>...: the named paths at that commit.
    spec="${url#git+}"; repo="${spec%@*}"; rest="${spec##*@}"
    commit="${rest%%#*}"; paths="${rest#*#}"
    tmp="$(mktemp -d "$dest/.git-XXXXXX")"
    git -C "$tmp" init -q
    git -C "$tmp" fetch -q --depth 1 --filter=blob:none "$repo" "$commit"
    test "$(git -C "$tmp" rev-parse FETCH_HEAD)" = "$commit"
    # shellcheck disable=SC2086
    git -C "$tmp" archive --format=tar.gz --prefix="$name-$version/" -o "$dest/$out.part"       "$commit" -- ${paths//,/ }
    rm -rf "$tmp"
  else
    curl -fsSL --retry 4 --retry-delay 5 -o "$out.part" "$url"
  fi
  mv "$out.part" "$out"
done <<< "$PIN_SOURCES"
if [[ "$mode" == --check ]]; then
  sha256sum -c "$here/SOURCES-$PIN_NAME.sha256"
else
  sha256sum ./*.tar.gz | sed 's# \./# #' | sort -k2 > "$here/SOURCES-$PIN_NAME.sha256"
  echo "wrote $here/SOURCES-$PIN_NAME.sha256"
fi
