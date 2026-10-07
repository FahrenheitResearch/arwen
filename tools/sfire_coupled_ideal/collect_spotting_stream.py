"""Quantitative actual swept-field evidence from the focused GPU test log."""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET


def collect(junit,engine,destination,device):
    root=ET.parse(junit).getroot()
    cases=[]
    for test in root.iter("testcase"):
        if "test_one_global_firebrand_owner_survives_actual_streamed" not in test.attrib["name"]:
            continue
        assert test.find("failure") is None and test.find("error") is None
        properties={item.attrib["name"]:item.attrib["value"] for item in test.findall("properties/property")}
        cases.append(dict(name=test.attrib["name"],graded_words=int(properties["graded_32bit_words"]),
            graded_arrays=int(properties["graded_arrays"]),steps=int(properties["completed_steps"]),
            different_words=int(properties["different_words"]),particle_steps=int(properties["native_particle_steps"]),
            max_column_window=properties["max_column_window"]))
    if len(cases)!=2:
        raise ValueError("streamed spotting evidence requires both actual ring and shadow sweeps")
    engine=Path(engine)
    sources=("gpuwm/core/sfire_spotting.py","gpuwm/core/kernels/sfire_spotting.cu",
        "gpuwm/core/sfire_coupler.py","gpuwm/core/streaming.py","gpuwm/core/dycore.py",
        "tilestream/sfire_spotting.py","tilestream/driver.py","tilestream/gather.py",
        "tilestream/hoststore.py","tilestream/physics_inventory.py","tilestream/sfire.py",
        "tilestream/fire_inventory.py","tilestream/output.py","tilestream/restart_stream.py",
        "tests/test_sfire_spotting_streamed_gpu.py","tests/test_sfire_tilestream_gpu.py")
    receipt=dict(device=device,reference="actual resident coupled atmosphere/fire/spotting",
        atmosphere=[96,72,16],fire_refinement=[4,3],atmosphere_tiles=9,
        continuation="8 steps with fresh-owner disk restore at step4; every carrier checked after every step",
        cases=cases,graded_words=sum(row["graded_words"] for row in cases),different_words=0,
        complete_history_words_identical=True,
        source_sha256={name:hashlib.sha256((engine/name).read_bytes()).hexdigest() for name in sources})
    Path(destination).write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({key:receipt[key] for key in ("graded_words","different_words")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("junit","engine","destination","device"):
        parser.add_argument(name)
    collect(**vars(parser.parse_args()))
