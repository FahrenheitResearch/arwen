#!/usr/bin/env bash
# Verify pinned submodules and the exact source copies made by RDASApp's
# default workaround step. Reject any other source edit before compiling.
set -euo pipefail
src="$1"
status=$(git -C "$src" submodule status --recursive)
if grep -q '^[+-U]' <<< "$status"; then
  echo "a submodule differs from the RDASApp pin; refusing a mixed-source build" >&2
  exit 1
fi
while IFS= read -r path; do
  case "$path" in
    sorc/*|parm/jcb-algorithms) ;; # Submodules are checked separately below.
    bundle/test-data-release/*|fix/crtm/*) ;; # Local test-data links.
    *) echo "RDASApp source edit outside its default workarounds: $path" >&2; exit 1;;
  esac
done < <(git -C "$src" diff --name-only HEAD)
expected=$(mktemp -d "${TMPDIR:?set TMPDIR inside the build workspace}/source-check.XXXXXX")
trap 'rm -f "$expected/allowed" "$expected/actual"; rmdir "$expected"' EXIT
: > "$expected/allowed"
allow_copy() {
  local input="$1" submodule="$2" target="$3"
  cmp "$src/sorc/_workaround_/$input" "$src/$submodule/$target"
  printf '%s/%s\n' "$submodule" "$target" >> "$expected/allowed"
}
for f in CMakeLists.txt Fields/fv3jedi_field_mod.f90 Geometry/fv3jedi_geom_mod.f90 \
         IO/FV3Restart/IOFms.h IO/FV3Restart/IOFms.cc IO/FV3Restart/IOFms.interface.F90 \
         IO/FV3Restart/IOFms.interface.h IO/FV3Restart/fv3jedi_io_fms2_mod.f90 \
         IO/FV3Restart/module_fv3lam_stats.f90 IO/FV3Restart/m_TwoPhaseScatterGather.f90; do
  allow_copy "fv3-jedi-io/$f" sorc/fv3-jedi "src/fv3jedi/$f"
done
for f in "$src"/sorc/_workaround_/ufo/CMakeLists.txt "$src"/sorc/_workaround_/ufo/EvalSurface* "$src"/sorc/_workaround_/ufo/ObsSfcCorrected*; do
  name=$(basename "$f")
  allow_copy "ufo/$name" sorc/ufo "src/ufo/operators/sfccorrected/$name"
done
allow_copy fv3-jedi/fv3jedi_state_mod.F90 sorc/fv3-jedi src/fv3jedi/State/fv3jedi_state_mod.F90
allow_copy fv3-jedi/FieldsMetadataDefault.h sorc/fv3-jedi src/fv3jedi/FieldMetadata/FieldsMetadataDefault.h
allow_copy ufo/DuplicateThinning.cc sorc/ufo src/ufo/filters/DuplicateThinning.cc
sort -u -o "$expected/allowed" "$expected/allowed"
git -C "$src" submodule foreach --quiet --recursive '
  git diff --name-only HEAD | while IFS= read -r path; do printf "%s/%s\n" "$displaypath" "$path"; done
  git ls-files --others --exclude-standard | while IFS= read -r path; do printf "%s/%s\n" "$displaypath" "$path"; done
' | sort -u > "$expected/actual"
extra=$(comm -23 "$expected/actual" "$expected/allowed")
if [[ -n "$extra" ]]; then
  printf 'source edits beyond the default workaround copies:\n%s\n' "$extra" >&2
  exit 1
fi
echo 'submodule pins and default workaround copies PASS'
