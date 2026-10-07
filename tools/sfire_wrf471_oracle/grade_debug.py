"""Drive the native ABI and compare WRF text bytes without field arithmetic."""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    library = ctypes.CDLL(str(args.library.resolve()))
    function = library.gpuwm_static_sfire_debug_array
    function.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    function.restype = ctypes.c_int
    results = []
    for case in json.loads((args.reference / "cases.json").read_text()):
        raw = (args.reference / (case["name"] + ".f32")).read_bytes()
        words = (ctypes.c_float * (len(raw) // 4)).from_buffer_copy(raw)
        request = dict(case, path=str(args.output.resolve()))
        metadata = json.dumps(request).encode()
        status = function(metadata, len(metadata), words, len(words))
        if status != 0:
            raise SystemExit(f"Native SFIRE debug writer failed: {case['name']}, status {status}")
        name = f"{case['name']}_{case['step']:08d}.txt"
        reference, actual = args.reference / name, args.output / name
        expected, produced = reference.read_bytes(), actual.read_bytes()
        if produced != expected:
            position = next((n for n, (a, b) in enumerate(zip(expected, produced)) if a != b),
                            min(len(expected), len(produced)))
            raise SystemExit(f"Native SFIRE text differs: {name}, first differing byte {position}")
        results.append(dict(name=name, words=len(words), records=len(words) + 6,
                            bytes=len(produced), input_sha256=digest(args.reference / (case["name"] + ".f32")),
                            text_sha256=digest(actual), status="bit-identical"))
    edge = json.loads((args.reference / "cases.json").read_text())[1]
    raw = (args.reference / "header_edges.f32").read_bytes()
    edge_words = (ctypes.c_float * 1).from_buffer_copy(raw)
    fixes = []
    for defect in ("long_step", "bound_overflow"):
        request = dict(edge, name=defect, path=str(args.output.resolve()))
        if defect == "long_step":
            request["step"] = 100000000
            reference = args.reference / "long_step_********.txt"
            reason = "WRF I8.8 filenames overflow to stars; retain the complete step"
        else:
            request["step"] = 1
            request["bounds"] = list(request["bounds"])
            request["bounds"][4:] = [2147483647, 2147483647]
            reference = args.reference / "header_edges_00000031.txt"
            reason = "WRF implied-DO counter overflows INT32_MAX; iterate the bounded slice"
        metadata = json.dumps(request).encode()
        if function(metadata, len(metadata), edge_words, 1) != 0:
            raise SystemExit(f"Native SFIRE debug corrected control failed: {defect}")
        actual = args.output / f"{defect}_{request['step']:08d}.txt"
        if actual.read_bytes() != reference.read_bytes():
            raise SystemExit(f"Native SFIRE debug corrected text differs: {defect}")
        fixes.append(dict(name=actual.name, status="fixed default-on", cause=reason,
                          text_sha256=digest(actual)))
    # Validate malformed metadata through the public ABI as well.
    malformed = dict(case, path=str(args.output.resolve()), bounds=[1, 2, 1, 1, 1, 1])
    metadata = json.dumps(malformed).encode()
    if function(metadata, len(metadata), words, len(words)) == 0:
        raise SystemExit("Native SFIRE debug ABI accepted mismatched bounds")
    receipt = dict(schema="sfire-debug-wrf471-v1", library_sha256=digest(args.library),
                   cases=results, corrected_wrf_defects=fixes,
                   wrf_bound_overflow_exit=int((args.reference / "bound-overflow.status").read_text()),
                   compiler=(args.reference / "compiler.txt").read_text().strip(),
                   source_sha256="ab9499f12b305257fd62a0c76f299ee308b7fabc98c6fae073ba25f132ec8d27",
                   malformed_bounds="refused", gpu_used=False)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
