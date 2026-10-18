//! Developer-only rebuild of the ONE checked-in kernel set.
//!
//! Normal builds include the CUBIN and PTX files under `kernels/` and never
//! call NVRTC or need a CUDA toolkit.  This example inlines every local
//! `#include "..."` of `kernels/woof_post.cu` (each header once, as
//! `#pragma once` asks), compiles the translation unit with NVRTC, and
//! rewrites `kernels/woof_post.manifest.json` with the source and artifact
//! hashes and each image's generator.  Two modes, two compilers:
//!
//! - `rebuild_kernels cubin`: the sm_100 and sm_120 CUBINs, with NVRTC 13
//!   (the first NVRTC line with both architectures and the one the CUBINs
//!   are certified from).
//! - `rebuild_kernels ptx`: the compute_60 PTX every other card
//!   JIT-compiles, with NVRTC 12.0 exactly.  NVRTC stamps its own PTX ISA
//!   version into the PTX and a driver refuses a newer ISA than it knows:
//!   NVRTC 13.4's `.version 9.4` PTX was refused by a CUDA 13.2 driver
//!   (595.91), so GRIB2 post fell back to the CPU on every card without a
//!   CUBIN.  NVRTC 12.0 writes ISA 8.0, which every driver from R525 on
//!   loads (`rw_post::gpu::PTX_ISA_CEILING`, pinned by `tests/gpu.rs`).
//!
//! Run each where that NVRTC is first on the loader path (`libnvrtc.so`,
//! for example from the `nvidia-cuda-nvrtc-cu12==12.0.76` wheel).

#[cfg(any(windows, target_os = "linux"))]
use std::{
    collections::BTreeSet,
    ffi::{CStr, CString, c_char},
    path::{Path, PathBuf},
};

#[cfg(any(windows, target_os = "linux"))]
use cudarc::nvrtc::{result, sys};
#[cfg(any(windows, target_os = "linux"))]
use serde_json::json;
#[cfg(any(windows, target_os = "linux"))]
use sha2::{Digest, Sha256};

#[cfg(any(windows, target_os = "linux"))]
// The two certified Blackwell architectures only; every other card JIT-compiles
// the compute_60 PTX.  Each CUBIN is embedded in every binary that links rw-post,
// and the twelve-architecture set put the 2.8.7 manylinux wheel over PyPI's
// 100,000,000-byte limit (src/gpu.rs, EMBEDDED_ARTIFACTS).
const ARCHITECTURES: &[u32] = &[100, 120];
#[cfg(any(windows, target_os = "linux"))]
const PTX_NVRTC: (i32, i32) = (12, 0);
#[cfg(any(windows, target_os = "linux"))]
const OPTIONS: &[&str] = &["--ftz=false", "--prec-sqrt=true", "--prec-div=true", "--fmad=false", "--std=c++17"];
#[cfg(any(windows, target_os = "linux"))]
const UNIT: &str = "woof_post.cu";

#[cfg(any(windows, target_os = "linux"))]
fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Inline local includes recursively, each file once; record every source.
#[cfg(any(windows, target_os = "linux"))]
fn inline(dir: &Path, file: &str, seen: &mut BTreeSet<String>, sources: &mut Vec<(String, String)>) -> std::io::Result<String> {
    let text = std::fs::read_to_string(dir.join(file))?;
    sources.push((file.to_owned(), sha256(text.as_bytes())));
    let mut out = String::new();
    for line in text.lines() {
        let t = line.trim();
        if t == "#pragma once" {
            continue;
        }
        if let Some(rest) = t.strip_prefix("#include \"") {
            let name = rest.trim_end_matches('"').to_owned();
            if seen.insert(name.clone()) {
                out.push_str(&format!("// ---- begin {name}\n"));
                out.push_str(&inline(dir, &name, seen, sources)?);
                out.push_str(&format!("// ---- end {name}\n"));
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    Ok(out)
}

#[cfg(any(windows, target_os = "linux"))]
fn compile(source: &CString, arch: &str, cubin: bool) -> Result<Vec<u8>, String> {
    let program = result::create_program(source, Some(c"woof_post.cu")).map_err(|e| e.to_string())?;
    let mut options: Vec<String> = OPTIONS.iter().map(|o| (*o).to_owned()).collect();
    options.push(format!("--gpu-architecture={arch}"));
    if let Err(error) = unsafe { result::compile_program(program, &options) } {
        let log = unsafe { result::get_program_log(program) }
            .ok()
            .and_then(|b| unsafe { CStr::from_ptr(b.as_ptr()) }.to_str().ok().map(str::to_owned))
            .unwrap_or_default();
        let _ = unsafe { result::destroy_program(program) };
        return Err(format!("{arch}: {error}: {log}"));
    }
    let out = if cubin {
        let mut size = 0usize;
        unsafe { sys::nvrtcGetCUBINSize(program, &mut size) }.result().map_err(|e| e.to_string())?;
        let mut bytes = vec![0u8; size];
        unsafe { sys::nvrtcGetCUBIN(program, bytes.as_mut_ptr().cast::<c_char>()) }
            .result()
            .map_err(|e| e.to_string())?;
        bytes
    } else {
        let raw = unsafe { result::get_ptx(program) }.map_err(|e| e.to_string())?;
        let mut bytes: Vec<u8> = raw.into_iter().map(|b| b as u8).collect();
        if bytes.last() == Some(&0) {
            bytes.pop();
        }
        bytes
    };
    unsafe { result::destroy_program(program) }.map_err(|e| e.to_string())?;
    Ok(out)
}

#[cfg(any(windows, target_os = "linux"))]
fn nvrtc_version() -> (i32, i32) {
    let (mut major, mut minor) = (0, 0);
    let _ = unsafe { sys::nvrtcVersion(&mut major, &mut minor) };
    (major, minor)
}

#[cfg(any(windows, target_os = "linux"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).unwrap_or_default();
    if mode != "cubin" && mode != "ptx" {
        return Err("usage: rebuild_kernels cubin|ptx (see the example's documentation)".into());
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let mut seen = BTreeSet::new();
    seen.insert(UNIT.to_owned());
    let mut sources = Vec::new();
    let unit = inline(&root, UNIT, &mut seen, &mut sources)?;
    let source = CString::new(unit.as_bytes())?;
    let (major, minor) = nvrtc_version();
    let generator = format!("NVIDIA NVRTC {major}.{minor}");
    let sources: Vec<_> = sources.into_iter().map(|(f, h)| json!({"file": f, "sha256": h})).collect();
    let old: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("woof_post.manifest.json"))?)?;
    let old_artifacts = old["artifacts"].as_array().cloned().unwrap_or_default();
    let mut artifacts = Vec::new();
    if mode == "cubin" {
        if major < 13 {
            return Err(format!("{generator}: the CUBINs are built with NVRTC 13").into());
        }
        for &arch in ARCHITECTURES {
            let label = format!("sm_{arch}");
            let bin = compile(&source, &label, true).map_err(std::io::Error::other)?;
            let file = format!("woof_post_sm{arch}.cubin");
            std::fs::write(root.join(&file), &bin)?;
            artifacts.push(json!({"architecture": label, "file": file, "bytes": bin.len(), "sha256": sha256(&bin), "generator": generator}));
        }
        artifacts.extend(old_artifacts.into_iter().filter(|a| a["file"] == "woof_post.ptx"));
    } else {
        if (major, minor) != PTX_NVRTC {
            return Err(format!(
                "{generator}: the PTX is built with NVRTC {}.{} so that its ISA version loads on R525 and newer drivers",
                PTX_NVRTC.0, PTX_NVRTC.1
            )
            .into());
        }
        // The CUBINs stay; they must come from the same sources.
        if old["sources"] != json!(sources) {
            return Err("kernel sources changed since the CUBINs were built: run `rebuild_kernels cubin` first".into());
        }
        let arch = rw_post::gpu::PTX_VIRTUAL_ARCHITECTURE;
        let ptx = compile(&source, arch, false).map_err(std::io::Error::other)?;
        std::fs::write(root.join("woof_post.ptx"), &ptx)?;
        artifacts.extend(old_artifacts.into_iter().filter(|a| a["file"] != "woof_post.ptx"));
        artifacts.push(json!({"architecture": format!("{arch} PTX fallback"), "file": "woof_post.ptx", "bytes": ptx.len(), "sha256": sha256(&ptx), "generator": generator}));
    }
    let manifest = json!({
        "abi_revision": 2,
        "kernel_set": "woof_post",
        "kernels": rw_post::gpu::KERNELS,
        "sources": sources,
        "translation_unit_sha256": sha256(unit.as_bytes()),
        "options": OPTIONS,
        "ptx_virtual_architecture": rw_post::gpu::PTX_VIRTUAL_ARCHITECTURE,
        "artifacts": artifacts,
    });
    std::fs::write(root.join("woof_post.manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    println!("rebuilt the {mode} image(s) with {generator}");
    Ok(())
}

#[cfg(not(any(windows, target_os = "linux")))]
fn main() {
    eprintln!("the kernel rebuild helper runs on Windows and Linux");
}
