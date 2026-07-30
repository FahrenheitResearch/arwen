#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 /new/path/WRF-v4.6.1" >&2
  exit 2
fi

target=$(realpath -m "$1")
commit=d66e442fccc04111067e29274c9f9eaccc3cef28
source_sha=1eb1b138b75ff3b0cfe33c23779f4ec9b72e57a5455a53ef11c9e55ae0f42722

if [ -e "$target" ]; then
  echo "refusing to reuse source checkout: $target" >&2
  exit 2
fi

git clone --filter=blob:none --no-checkout \
  https://github.com/wrf-model/WRF.git "$target"
git -C "$target" fetch --depth=1 origin "$commit"
git -C "$target" checkout --detach "$commit"

actual_commit=$(git -C "$target" rev-parse HEAD)
actual_sha=$(sha256sum "$target/phys/module_mp_nssl_2mom.F" | awk '{print $1}')
test "$actual_commit" = "$commit"
test "$actual_sha" = "$source_sha"
git -C "$target" diff --quiet
git -C "$target" diff --cached --quiet

printf 'WRF_OFFICIAL_SOURCE_READY commit=%s source_sha256=%s path=%s\n' \
  "$actual_commit" "$actual_sha" "$target"
