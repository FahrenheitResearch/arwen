"""Read actual evolved C-grid winds for a nonzero vertical-motion probe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def evolved_flow(source,reader,output):
    """Decode evolved winds with Rust; retain their source receipt.

    These winds supplement an initialization-state diffusion fixture. They
    are an explicit operator probe, not a claimed complete evolved state.
    """
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    subprocess.run([str(reader),"dump","--raw",str(source),str(output),"U","V","W"],check=True)
    metadata = json.loads((output/"metadata.json").read_text())
    winds = {v["name"]:np.fromfile(output/v["filename"],dtype="<f8").reshape(v["shape"])[0].astype(np.float32)
             for v in metadata["variables"]}
    return winds,{"evolved_flow_source_sha256":hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                  "evolved_flow_fields":["U","V","W"],
                  "evolved_flow_note":"Actual 19Z C-grid winds on initialization-state thermodynamics; operator probe"}
