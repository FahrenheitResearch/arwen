"""Regression checks for the GF top read and the staggered-W seam."""
from pathlib import Path
from types import SimpleNamespace as NS
import os
import json
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/gf_wrf461_oracle"))
import gf_regime_columns as gr
import gf_regime_gpu as rg
from gpuwm.core import gf
from gpuwm.core.kernels import module_source, module_options


def inputs():
    directory = Path(os.environ.get("GF_REGIME_ORACLE_DIR",
        ROOT / "tests/data/oracles/gf/regime"))
    return gr.read_inputs(directory / "gf-regime-inputs.bin")


@pytest.mark.gpu
def test_top_pbl_column_ignores_poison_above_top(monkeypatch):
    inp = inputs()
    inp["ii"][:, 0] = inp["nz"]
    inp["ii"][:, 1] = 1
    inp["ii"][:, 2] = 3
    source = module_source("gf")
    marker = "        xff1 = FMUL(K_P03, zws);"
    assert source.count(marker) == 1
    results = []
    for poison in ("-1.0e20f", "1.0e20f"):
        poisoned = source.replace(marker,
            marker + f"\n        po_cup[ktf + 1] = {poison};")
        module = cp.RawModule(code=f"#define GF_KMAX {inp['nz']}\n" + poisoned,
                              options=module_options("gf"))
        monkeypatch.setattr(gf, "_gf_module", lambda nz: module)
        results.append(rg.run_kernel(inp, 0))
    for key in ("lev", "sca", "isc"):
        assert np.isfinite(results[0][key]).all(), key
        assert np.isfinite(results[1][key]).all(), key
        assert np.array_equal(results[0][key].view(np.uint32),
                              results[1][key].view(np.uint32)), key


@pytest.mark.gpu
def test_shallow_closure_reads_only_real_pressure_levels():
    # Execute the production loop with a nonzero forcing at the top.
    # This witnesses the read even when a full GFDRV column rejects shallow.
    source = module_source("gf")
    start = source.index("        for (int k =", source.index("        xff1 = FMUL(K_P03, zws);"))
    end = source.index("        trash =", start)
    loop = source[start:end]
    probe = r'''
extern "C" __global__ void closure_probe(float sentinel, float *out) {
    const int ktf = 40, kpbl = ktf;
    float po_cup[42], dhdt[42];
    for (int k=0; k<42; ++k) { po_cup[k]=1000.0f-20.0f*k; dhdt[k]=0.001f; }
    po_cup[ktf+1] = sentinel;
    float blqe = K_ZERO;
    LOOP
    float xff2 = GMIN(K_ONE, GMAX(K_ZERO, FDIV(blqe, K_TEN)));
    out[0]=blqe; out[1]=xff2; out[2]=FDIV(xff2, K_THREE);
}
'''.replace("    LOOP", loop)
    fn = cp.RawModule(code=source + probe, options=module_options("gf")).get_function("closure_probe")
    outputs = []
    for sentinel in (-1.e20, 1.e20):
        out = cp.empty(3, cp.float32)
        fn((1,), (1,), (np.float32(sentinel), out))
        outputs.append(cp.asnumpy(out))
    assert np.isfinite(outputs).all()
    assert np.array_equal(outputs[0].view(np.uint32), outputs[1].view(np.uint32)), outputs


def adapter_outputs(inp):
    n, nz = inp["ncol"], inp["nz"]
    result = {f: np.zeros((n, nz), np.float32) for f in
              ("rthcuten", "rqvcuten", "rqccuten", "rqicuten")}
    result["raincv"] = np.zeros(n, np.float32)
    from gf_field_lists import DRV_LEV_FIELDS, DRV_SCA_FIELDS, DRV_ISCA_FIELDS
    boundary = dict(lev=np.zeros((n, len(DRV_LEV_FIELDS), nz), np.float32),
                    sca=np.zeros((n, len(DRV_SCA_FIELDS)), np.float32),
                    isc=np.zeros((n, len(DRV_ISCA_FIELDS)), np.int32))
    original_module = gf._gf_module
    groups = {}
    for i in range(n):
        key = (float(inp["sc"][i, gr.SCA_FIELDS.index("dt")]),
               int(inp["ii"][i, 1]), int(inp["ii"][i, 2]))
        groups.setdefault(key, []).append(i)
    def grid(a):
        return cp.asarray(a.T[:, None, :].copy())
    for (dt, ish, ich), ids in groups.items():
        ids = np.asarray(ids)
        v = {f: grid(inp["lev"][ids, gr.LEV_FIELDS.index(f), :]) for f in gr.LEV_FIELDS}
        faces = np.c_[inp["lev"][ids, gr.LEV_FIELDS.index("w"), :],
                      np.zeros(len(ids), np.float32)]
        state = NS(p=v["p"], ht=cp.asarray(inp["sc"][ids, gr.SCA_FIELDS.index("ht")][None, :]),
                   w=grid(faces), qi=cp.zeros_like(v["p"]))
        atmos = {dst: v[src] for dst, src in
                 (("u", "u"), ("v", "v"), ("temperature", "t"), ("qv", "qv"),
                  ("pressure", "p"), ("exner", "pi"), ("rho", "rho"), ("dz", "dz8w"))}
        atmos["p_interface"] = grid(inp["p8w"][ids])
        fields = {f: cp.asarray(inp["sc"][ids, gr.SCA_FIELDS.index(f)][None, :])
                  for f in ("hfx", "qfx", "xland")}
        fields["kpbl"] = cp.asarray(inp["ii"][ids, 0][None, :])
        driver = NS(rthratenlw=v["rthraten"], rthratensw=cp.zeros_like(v["rthraten"]),
            gf_rthdynten=v["rthften"], gf_rqvdynten=v["rqvften"],
            gf_rthblten=v["rthblten"], gf_rqvblten=v["rqvblten"],
            gf_dx_column=cp.asarray(inp["sc"][ids, gr.SCA_FIELDS.index("dx")][None, :]))
        adapter = gf.GrellFreitas()
        adapter.bind_driver(driver)
        launches = []
        class Capture:
            def __init__(self, fn):
                self.fn = fn
            def __getattr__(self, name):
                return getattr(self.fn, name)
            def __call__(self, grid, block, args):
                self.fn(grid, block, args)
                launches.append(tuple(cp.asnumpy(a) for a in args[3:6]))
        class Module:
            def get_function(self, name):
                return Capture(original_module(nz).get_function(name))
        with patch.object(gf, "_gf_module", lambda *a, **kw: Module()):
            got = adapter(atmosphere=atmos, fields=fields, state=state,
                          cfg=NS(dt=dt, dx=9000, ishallow=ish, clos_choice=ich,
                                 spp_conv=0, lightning_option=0))
        for j, key in enumerate(("lev", "sca", "isc")):
            boundary[key][ids] = np.concatenate([run[j] for run in launches], axis=0)
        for f in result:
            result[f][ids] = (cp.asnumpy(got.rainc)[0, :] if f == "raincv"
                             else cp.asnumpy(getattr(got, f))[:, 0, :].T)
    return result, len(groups), boundary


@pytest.mark.gpu
def test_adapter_passes_lower_face_w_word_for_word():
    inp = inputs()
    want = rg.run_kernel(inp, 0)
    got, groups, boundary = adapter_outputs(inp)
    from gf_field_lists import DRV_LEV_FIELDS, DRV_SCA_FIELDS
    differences = {}
    for field, values in got.items():
        ref = (want["sca"][:, DRV_SCA_FIELDS.index(field)] if field == "raincv"
               else want["lev"][:, DRV_LEV_FIELDS.index(field), :])
        differences[field] = int((values.view(np.uint32) != ref.view(np.uint32)).sum())
    assert groups == 26
    assert sum(differences.values()) == 0, differences
    directory = Path(os.environ.get("GF_REGIME_ORACLE_DIR",
        ROOT / "tests/data/oracles/gf/regime"))
    grade = rg.compare(boundary, rg.load_reference(directory, "corrected"))
    grade["groups"] = groups
    grade["returned_words_differ"] = differences
    target = os.environ.get("GF_ADAPTER_RECEIPT")
    if target:
        Path(target).write_text(json.dumps(grade, indent=2) + "\n")
    assert grade["words_differ"] == 0, grade


@pytest.mark.gpu
def test_cli_rejects_planted_bit_and_accepts_clean_kernel(tmp_path):
    directory = Path(os.environ.get("GF_REGIME_ORACLE_DIR",
        ROOT / "tests/data/oracles/gf/regime"))
    script = ROOT / "tools/gf_wrf461_oracle/gf_regime_gpu.py"
    clean = subprocess.run([sys.executable, str(script), str(directory),
                            str(tmp_path / "clean.json")], capture_output=True, text=True)
    assert clean.returncode == 0, clean.stdout + clean.stderr
    kernel = ROOT / "gpuwm/core/kernels/gf.cu"
    original = kernel.read_bytes()
    try:
        subprocess.run(["git", "apply", str(ROOT / "tools/gf_wrf461_oracle/planted-bit.patch")],
                       cwd=ROOT, check=True, capture_output=True, text=True)
        broken = subprocess.run([sys.executable, str(script), str(directory),
                                 str(tmp_path / "planted.json")],
                                capture_output=True, text=True)
        assert broken.returncode == 1, broken.stdout + broken.stderr
        from gpuwm import wrf_exact
        target = os.environ.get("GF_CLI_EXIT_RECEIPT")
        if target:
            Path(target).write_text(json.dumps(dict(
                strict=wrf_exact.ENABLED, clean_exit=clean.returncode,
                planted_exit=broken.returncode,
                clean=json.loads((tmp_path / "clean.json").read_text()),
                planted=json.loads((tmp_path / "planted.json").read_text())), indent=2)+"\n")
    finally:
        kernel.write_bytes(original)
    assert kernel.read_bytes() == original
