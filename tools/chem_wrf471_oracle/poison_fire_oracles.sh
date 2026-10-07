#!/usr/bin/env bash
# Compile the pinned, unmodified source with poisoned unwritten locals.
# Explicit driver poisons additionally prove kpbl's above-column undefined read.
# Usage: bash poison_fire_oracles.sh SOURCE_ROOT OWNED_POISON_ROOT BASELINE_ROOT
set -euo pipefail
source_root=$(realpath "$1")
poison_root=$(realpath -m "$2")
baseline=$(realpath "$3")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$poison_root/compiler"
cat > "$poison_root/compiler/gfortran" <<'EOF'
#!/usr/bin/env bash
exec /usr/bin/gfortran -finit-real=snan -finit-integer=-98765 "$@"
EOF
chmod +x "$poison_root/compiler/gfortran"
PATH="$poison_root/compiler:$PATH" bash "$script_dir/build_fire_oracles.sh" "$source_root" "$poison_root/build"
count=0
for family in wrf gsl4 gsl8; do
    while IFS= read -r -d '' file; do
        relative=${file#"$baseline/$family/fixtures/"}
        cmp "$file" "$poison_root/build/$family/fixtures/$relative"
        count=$((count+1))
    done < <(find "$baseline/$family/fixtures" -type f -print0)
done
echo "Poisoned locals: $count fixture files byte-identical to baseline"
