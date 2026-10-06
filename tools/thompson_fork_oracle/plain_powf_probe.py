"""Measure the two plain-powf additions from b0556bd76, lane/286-fork-thompson.

Run on a CUDA host under its GPU OWNER protocol:
  python plain_powf_probe.py FORK_SOURCE HEADER COLD_SOURCE SCRATCH RECEIPT [COLUMNS]

The authority is the SHA-pinned NOAA HRRR v4.1.21 source. This is a scalar
expression and helper-output probe, not a full microphysics qualification.
The Fortran expressions and mathematical routines are extracted unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np

SOURCE_SHA256 = "4d60011188443eb432294f7693beb64bdbc8f812541a15c7800013060c877283"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def function(text, name):
    start = text.index("__device__ __forceinline__ void " + name + "(")
    return text[start:text.index("\n}\n", start) + 3]


def main():
    source_path, header_path, cold_path, scratch_path, receipt_path = map(Path, sys.argv[1:6])
    source_bytes = source_path.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_SHA256
    source = source_bytes.decode("utf-8")
    lines = source.splitlines()
    header = header_path.read_text(encoding="utf-8")
    cold = cold_path.read_text(encoding="utf-8")
    helper = function(header, "thompson_aa_wrf39_bound_ice_number")
    coeff = re.search(r"const float t2_qg_sd = 0\.28f \* powf\(0\.632f, 0\.33333334326744080f\)\s*\* sqrtf\(442\.0f\) \* 1\.9021706581115723f;", cold).group()
    scratch_path.mkdir(parents=True, exist_ok=True)

    # Geometric inputs plus float32 neighbours of both source size bounds.
    f = np.float32
    density = [f(x) for x in (0.1, 0.5, 1.0, 1.3, 2.0)]
    mass = [f(x) for x in np.geomspace(1e-12, 1e-2, 37)]
    numbers = [f(x) for x in np.geomspace(1e-6, 1e8, 43)]
    rows = [(m, d, n) for d in density for m in mass for n in numbers]
    am_i = f(f(f(3.1415926536) * f(890.0)) / f(6.0))
    for d in density:
        for m in mass:
            for diameter in (f(5e-6), f(300e-6)):
                lam = f(f(4.0) / diameter)
                n = f(f(f(m / f(am_i * f(6))) * f(lam ** f(3))) / d)
                for _ in range(8):
                    rows.extend(((m, d, n), (m, d, f(np.nextafter(n, f(0.0))))))
                    n = f(np.nextafter(n, f(np.inf)))
    synthetic_rows = len(rows)
    columns_metadata = None
    if len(sys.argv) == 7:
        columns_path = Path(sys.argv[6])
        with np.load(columns_path) as columns:
            rho = np.asarray(columns["cp1_rho"], dtype=f)
            good = np.isfinite(rho) & (rho > 0)
            for prefix in ("col_", "wrf_"):
                ice_mass = np.maximum(f(1e-12), np.asarray(columns[prefix + "qi"], dtype=f) * rho)
                ice_number = np.asarray(columns[prefix + "ni"], dtype=f)
                valid = good & np.isfinite(ice_mass) & np.isfinite(ice_number)
                rows.extend(zip(ice_mass[valid], rho[valid], ice_number[valid]))
            columns_metadata = {"sha256": hashlib.sha256(columns_path.read_bytes()).hexdigest(),
                                "columns": int(rho.shape[0]), "levels": int(rho.shape[1]),
                                "added_rows": len(rows) - synthetic_rows,
                                "operands": "saved Fortran cp1_rho with entry col_qi/col_ni and final wrf_qi/wrf_ni"}
    inp = np.asarray(rows, dtype="<f4")
    (scratch_path / "inputs.bin").write_bytes(inp.tobytes())
    gamma = "\n".join(lines[4948:4972]) + "\n" + "\n".join(lines[4994:5002])
    # Variable declarations preserve the REAL and DOUBLE PRECISION source kinds.
    # The ice expression is source :2874-2876; the coefficient is :670/:769/:814.
    fortran = '''program probe
implicit none
real, parameter :: PI=3.1415926536, rho_i=890.0, am_i=PI*rho_i/6.0
real, parameter :: bm_i=3.0, mu_i=0.0, R1=1.E-12, R2=1.E-6
real, parameter :: Sc=0.632, av_g=442.0, bv_g=0.89, mu_g=0.0
real :: cie(7), cig(7), cge(12), cgg(12), oig1, oig2, obmi, Sc3, t2_qg_sd
real :: xri, xni, density, number, xDi, base, power
double precision :: lami, ilami
integer :: n, branch, ios
character(1024) :: input_path, output_path
call get_command_argument(1,input_path)
call get_command_argument(2,output_path)
''' + "\n".join(lines[693:695]) + '''
cig(1)=1.0
cig(2)=6.0
''' + "\n".join(lines[707:710]) + "\n" + lines[669] + "\n" + lines[768] + '''
cgg(11)=WGAMMA(cge(11))
''' + lines[813] + '''
open(20,file=trim(input_path),access='stream',form='unformatted',status='old')
open(21,file=trim(output_path),access='stream',form='unformatted',status='replace')
write(21) Sc3,t2_qg_sd,cgg(11)
do
 read(20,iostat=ios) xri,density,number
 if(ios.ne.0)exit
 xni=MAX(R2,number*density)
 base=am_i*cig(2)*oig1*xni/xri
 power=base**obmi
''' + "\n".join(lines[2873:2876]) + '''
 branch=0
 if(xDi.lt.5.E-6)branch=1
 if(xDi.gt.300.E-6)branch=2
 if(xri.le.R1)branch=-1
 write(21) base,power,xDi,real(branch)
enddo
close(20)
close(21)
contains
''' + gamma + "\nend program probe\n"
    fpath = scratch_path / "probe.F90"
    fpath.write_text(fortran, encoding="utf-8")
    binary = scratch_path / "probe"
    flags = ["-O2", "-fno-tree-vectorize", "-ffree-form", "-ffree-line-length-none"]
    compile_command = ["gfortran", *flags, str(fpath), "-o", str(binary)]
    subprocess.run(compile_command, check=True)
    subprocess.run([str(binary), str(scratch_path / "inputs.bin"), str(scratch_path / "oracle.bin")], check=True)
    oracle = np.fromfile(scratch_path / "oracle.bin", dtype="<f4")
    coeff_oracle = oracle[:3]
    oracle = oracle[3:].reshape(-1, 4)

    import cupy as cp
    corrected = helper.replace("thompson_aa_wrf39_bound_ice_number", "bounds_cr").replace("powf(", "thompson_aa_powf_cr(")
    wrappers = r'''
extern "C" __global__ void measure(const float* x, float* out, int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
 float mass=x[3*i],rho=x[3*i+1],number=x[3*i+2];
 float p=number,c=number;
 thompson_aa_wrf39_bound_ice_number(mass,rho,&p);
 bounds_cr(mass,rho,&c);
 float b=THOMPSON_AA_AM_I*6.0f*fmaxf(THOMPSON_AA_R2,number*rho)/mass;
 float pw=powf(b,0.33333334326744080f), cr=thompson_aa_powf_cr(b,0.33333334326744080f);
 out[7*i]=p; out[7*i+1]=c; out[7*i+2]=b; out[7*i+3]=pw; out[7*i+4]=cr;
 out[7*i+5]=(float)(4.0/(double)pw); out[7*i+6]=(float)(4.0/(double)cr);
}
extern "C" __global__ void coefficient(float* out) {
''' + coeff + r'''
 out[0]=t2_qg_sd;
 out[1]=0.28f*thompson_aa_powf_cr(0.632f,0.33333334326744080f)*sqrtf(442.0f)*1.9021706581115723f;
 out[2]=powf(0.632f,0.33333334326744080f);
 out[3]=thompson_aa_powf_cr(0.632f,0.33333334326744080f);
}
'''
    module = cp.RawModule(code=header + corrected + wrappers)
    x = cp.asarray(inp)
    out = cp.empty((len(inp), 7), dtype=cp.float32)
    cout = cp.empty(4, dtype=cp.float32)
    module.get_function("measure")(((len(inp)+127)//128,), (128,), (x, out, np.int32(len(inp))))
    module.get_function("coefficient")((1,), (1,), (cout,))
    cp.cuda.runtime.deviceSynchronize()
    result = cp.asnumpy(out)
    coefficient = cp.asnumpy(cout)
    word = lambda a: np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    differences = lambda a, b: int(np.count_nonzero(word(a) != word(b)))
    branch = lambda a: np.where(inp[:, 0] <= f(1e-12), -1, np.where(a < f(5e-6), 1, np.where(a > f(300e-6), 2, 0)))
    report = {
        "owner_commit": "b0556bd76b1f1276e7d52ece6d96242bf3f90f48",
        "owner_lane": "lane/286-fork-thompson",
        "authority_source_sha256": SOURCE_SHA256,
        "authority_source_lines": [670, 769, 814, "2874-2876", "4949-4971", "4995-5002"],
        "helper_sha256": digest(helper), "coefficient_statement_sha256": digest(coeff),
        "probe_fortran_sha256": digest(fortran), "probe_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compile_command": compile_command,
        "gfortran": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "libc": platform.libc_ver(), "cupy": cp.__version__,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "runtime": cp.cuda.runtime.runtimeGetVersion(),
        "ice": {"rows": len(inp), "synthetic_rows": synthetic_rows, "saved_real_columns": columns_metadata,
            "inputs_sha256": hashlib.sha256(inp.tobytes()).hexdigest(),
            "base_differences_vs_fortran": differences(result[:,2], oracle[:,0]),
            "plain_power_differences_vs_fortran": differences(result[:,3], oracle[:,1]),
            "cr_power_differences_vs_fortran": differences(result[:,4], oracle[:,1]),
            "plain_branch_differences_vs_fortran": int(np.count_nonzero(branch(result[:,5]) != oracle[:,3])),
            "cr_branch_differences_vs_fortran": int(np.count_nonzero(branch(result[:,6]) != oracle[:,3])),
            "complete_helper_output_changes_under_cr": differences(result[:,0], result[:,1]),
            "output_plain_sha256": hashlib.sha256(result[:,0].tobytes()).hexdigest(),
            "output_cr_sha256": hashlib.sha256(result[:,1].tobytes()).hexdigest()},
        "sublimation_coefficient": {"fortran_sc3_bits": int(word(coeff_oracle)[0]),
            "fortran_t2_bits": int(word(coeff_oracle)[1]), "fortran_cgg11_bits": int(word(coeff_oracle)[2]),
            "cuda_bits": [int(v) for v in word(coefficient)],
            "complete_coefficient_changes_under_cr": differences(coefficient[:1], coefficient[1:2]),
            "plain_coefficient_differences_vs_fortran": differences(coefficient[:1], coeff_oracle[1:2]),
            "cr_coefficient_differences_vs_fortran": differences(coefficient[1:2], coeff_oracle[1:2])},
        "scope": "scalar Fortran expression probes and actual CUDA helper/constant statement; no full network or forecast qualification",
    }
    Path(receipt_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
