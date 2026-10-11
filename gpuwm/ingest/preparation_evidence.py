"""Keep measured execution policy outside immutable scientific proof values."""
from copy import deepcopy
import json
from pathlib import Path

from gpuwm.ingest.preprocess_backend import preprocess_identity, preprocess_measurements


RUNTIME_NAME = "preparation-runtime.json"


def scientific_preprocessing(receipt):
    # The existing prepared-cache contract already makes this distinction.
    # Copy now so subsequent vertical route recording cannot mutate a head.
    return deepcopy(preprocess_identity(receipt))


def write_runtime_preparation(directory, *, preprocessing, parallelism=None, forcing_stage_timings=None,
                              input_preparation_seconds=None):
    """Retain every execution measurement without binding it into a head hash."""
    document = {"schema": "gpuwm-preparation-runtime-v1",
                "preprocessing_measurements": preprocess_measurements(preprocessing),
                "preprocessing": preprocessing}
    if input_preparation_seconds is not None:
        document["input_preparation_seconds"] = float(input_preparation_seconds)
    if parallelism is not None:
        document["preparation_parallelism"] = parallelism
    if forcing_stage_timings is not None:
        document["forcing_stage_timings"] = forcing_stage_timings
    path = Path(directory) / RUNTIME_NAME
    temporary = path.with_suffix(".json.pending")
    temporary.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
