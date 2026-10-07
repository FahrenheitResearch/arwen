"""Orchestration facades for the native fire initializer text reader."""
from pathlib import Path
from gpuwm.static.sfire import (read_fire_ideal_inputs,fire_ideal_input_requests,
                                read_native_landuse_table)


def read_sounding(path):
    return read_fire_ideal_inputs(Path(path).parent,sounding=path)["sounding"]


def read_input_fields(cfg,directory):
    result=read_fire_ideal_inputs(directory,fire_ideal_input_requests(cfg))
    return result["fields"],result["_metadata"]["file_sha256"]
