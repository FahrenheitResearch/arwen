"""Grade every column output and carried state word against a release oracle."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from gpuwm.verify.urban_ucm_host import HostTwin
from gpuwm.verify.urban_ucm_oracle import (
    UCM_VARIANTS, load_variant, port_params, _inputs, _state_from, _flatten, _reference,
)

def compare(port, ref, *, limit=24):
    result = {}
    for key, want in ref.items():
        a = np.asarray(port[key], np.float32).view(np.uint32)
        b = np.asarray(want, np.float32).view(np.uint32)
        bad = np.flatnonzero(a != b)
        if bad.size:
            shown = bad if limit is None else bad[:limit]
            result[key] = {"count":int(bad.size), "indices":shown.tolist(),
                           "got_words":a[shown].tolist(),"want_words":b[shown].tolist()}
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture",type=Path)
    parser.add_argument("receipt",type=Path)
    parser.add_argument("--host",type=Path)
    parser.add_argument("--generation")
    parser.add_argument("--canopy-generation",choices=("wrf_471","wrf_480"),default="wrf_471")
    parser.add_argument("--plant-bit",action="store_true")
    args = parser.parse_args()
    if args.host:
        if args.canopy_generation == "wrf_480":
            from gpuwm.verify.urban_ucm480_host import Host480Twin
            runner = Host480Twin(args.host).run_columns
        else:
            runner = HostTwin(args.host).run_columns
    else:
        from gpuwm.core.urban_ucm import run_columns
        runner = run_columns
    def run(params,inputs,state):
        kw = {} if args.generation is None else {"generation":args.generation}
        return runner(params,inputs,state,**kw)
    records = {}
    full_differences = {}
    words = 0
    for variant in UCM_VARIANTS:
        rows,table,switches = load_variant(variant,args.fixture)
        params = port_params(table,switches)
        if args.canopy_generation == "wrf_480":
            from dataclasses import replace
            from gpuwm.core.urban_tables import load_urban_params
            from gpuwm.core.urban_ucm import pack_params
            full = pack_params(load_urban_params(1,int("lcz" in variant),generation="wrf_480"))
            common = full.table.copy()
            common[:,:params.table.shape[1]] = params.table
            flags = full.switches.copy()
            flags[:11] = params.switches[:11]
            flags[11:13] = 0
            params = replace(full,table=common,globals_=params.globals_,switches=flags)
        allrows = np.arange(rows["ta"].size)
        first = np.flatnonzero(rows["step"]==1)
        for carry in (False,True):
            state = _state_from(rows,first,"_in")
            parts, refs = [],[]
            for step in range(1,7) if carry else (None,):
                index = first + step - 1 if carry else allrows
                incoming = state if carry else _state_from(rows,index,"_in")
                inputs = _inputs(rows,index)
                if args.canopy_generation == "wrf_480":
                    inputs.update({key:rows[key][index] for key in ("declin","cosz","xlat")})
                    inputs["ssgd"] = np.float32(.8)*inputs["ssg"]
                    inputs["ssgq"] = inputs["ssg"]-inputs["ssgd"]
                out,state,codes = run(params,inputs,incoming)
                assert not np.any(codes), (variant,codes)
                parts.append(_flatten(out,state)); refs.append(_reference(rows,index))
            port = {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
            ref = {k:np.concatenate([p[k] for p in refs]) for k in refs[0]}
            if args.plant_bit:
                port["after:drelr"] = port["after:drelr"].copy()
                port["after:drelr"].view(np.uint32)[0] ^= np.uint32(1)
            diff = compare(port,ref)
            full_differences[f"{variant}/{'carried' if carry else 'replay'}"] = compare(port,ref,limit=None)
            hashes = {k:hashlib.sha256(np.asarray(v,np.float32).tobytes()).hexdigest() for k,v in port.items()}
            records[f"{variant}/{'carried' if carry else 'replay'}"] = {"rows":int(rows["ta"].size),"differences":diff,"hashes":hashes}
            words += sum(v.size for v in port.values())
    failed = sum(sum(v["count"] for v in r["differences"].values()) for r in records.values())
    receipt = {"generation":args.generation or "bare default","canopy_generation":args.canopy_generation,"words":words,"differing_words":failed,"records":records}
    args.receipt.write_text(json.dumps(receipt,indent=2)+"\n")
    import gzip
    args.receipt.with_suffix(".cells.json.gz").write_bytes(gzip.compress(
        json.dumps(full_differences,separators=(",",":")).encode(),mtime=0))
    print(json.dumps({"words":words,"differing_words":failed,"receipt":str(args.receipt)}))
    return int(bool(failed))

if __name__ == "__main__":
    raise SystemExit(main())
