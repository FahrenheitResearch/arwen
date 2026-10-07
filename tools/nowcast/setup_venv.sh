#!/usr/bin/env bash
# Build the StormScope runner's own venv and prefetch the pinned weights.
#
#   tools/nowcast/setup_venv.sh [VENV]        (default /work/nh/venv-e2s)
#
# Linux only.  Uses python3.12 unless PYTHON is set.  Writes beside the venv:
#   <VENV>/pip-freeze.txt        every installed version
#   <VENV>/weights-sha256.txt    sha256 of every cached weight and grid file
# and prints the weight hashes.  Re-running is safe: pip skips what is there
# and the weights come from the cache.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${1:-/work/nh/venv-e2s}"
PY="${PYTHON:-python3.12}"

if [ "$(uname -s)" != "Linux" ]; then
  echo "refused: setup_venv.sh is Linux only (NATTEN publishes Linux wheels only)" >&2
  exit 3
fi

# torch 2.13.0 publishes no CUDA 12.8 wheel, and its CUDA 12.6 wheel has no
# Blackwell kernels, so the pin is cu130: the box driver must support CUDA
# 13.0 (driver 580 or newer).  Checked first, so a box with an older driver
# is caught in seconds rather than after a 6 GB weight download, with
# "CUDA unavailable" on the first card step.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "refused: nvidia-smi not found; the runner needs an NVIDIA driver for CUDA 13.0" >&2
  exit 3
fi
DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
if [ -z "$DRIVER_CUDA" ] || [ "${DRIVER_CUDA%%.*}" -lt 13 ]; then
  echo "refused: the driver supports CUDA ${DRIVER_CUDA:-unknown}, the cu130 torch wheels need 13.0 or newer" >&2
  exit 3
fi

if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$HERE/requirements-stormscope.txt"
"$VENV/bin/python" -m pip freeze > "$VENV/pip-freeze.txt"

# Prove the stack on a card before the weights are fetched: torch sees the
# card, its build carries this card's architecture, and NATTEN loads its
# compiled library.
"$VENV/bin/python" - <<'PYEOF'
import sys
import torch
import natten

if not torch.cuda.is_available():
    sys.exit("refused: torch sees no CUDA device")
major, minor = torch.cuda.get_device_capability(0)
arch = f"sm_{major}{minor}"
if arch not in torch.cuda.get_arch_list():
    sys.exit(f"refused: this torch build has no {arch} kernels ({torch.cuda.get_arch_list()})")
if getattr(natten, "HAS_LIBNATTEN", True) is False:
    sys.exit("refused: NATTEN was installed without its compiled library")
print(f"torch {torch.__version__} cuda {torch.version.cuda} on {torch.cuda.get_device_name(0)} "
      f"({arch}); natten {natten.__version__}")
PYEOF

# Prefetch the pinned weights for the variant in sources.toml, and refuse a
# package root that is not the pin (the receipt would name other weights).
PYTHONPATH="$HERE/../.." "$VENV/bin/python" - "$HERE/sources.toml" "$VENV/weights-sha256.txt" <<'PYEOF'
import json, sys
from pathlib import Path

from tools.nowcast import frames as F
from earth2studio.models.auto import Package
from earth2studio.models.px.stormscope import StormScopeBase

row = F.source_row(F.load_sources(sys.argv[1]), "stormscope-3km-10min")
default_root = StormScopeBase.load_default_package().root
F.check_revision(default_root, row["revision"])
package = Package(f"{row['package']}@{row['revision']}",
                  cache_options={"cache_storage": Package.default_cache("stormscope")})
F.check_revision(package.root, row["revision"])

with open(package.resolve("registry.json")) as fh:
    registry = json.load(fh)
wanted = ["registry.json"]
for section in ("goes", "mrms"):
    entry = registry[section]["models"]
    name = registry[section].get("aliases", {}).get(row["variant"], row["variant"])
    for ck in entry[name]["checkpoints"]:
        wanted.append(ck["path"])

lines = []
for rel in wanted:
    local = Path(package.resolve(rel))
    digest, nbytes = F.sha256_file(local)
    lines.append(f"{digest}  {rel}  ({nbytes} bytes)")
# Grid, terrain, coverage and normalization files are fetched on first model
# load; load both models on the CPU once so they land in the cache too.
from earth2studio.models.px.stormscope import StormScopeGOES, StormScopeMRMS
StormScopeGOES.load_model(package=package, model_name=row["variant"], conditioning_data_source=None)
StormScopeMRMS.load_model(package=package, model_name=row["variant"], conditioning_data_source=None)
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
print(f"weights at revision {row['revision']}:")
print("\n".join(lines))
PYEOF

echo "venv ready: $VENV"
