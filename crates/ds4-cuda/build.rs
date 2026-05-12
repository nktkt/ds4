//! Build script: compile `ds4_cuda.cu` to PTX via `nvcc`.
//!
//! Behavior:
//! * On non-Linux hosts: silently exit (the crate's contents are also
//!   `#[cfg(target_os = "linux")]`-gated, so there's nothing to link).
//! * On Linux without a working `nvcc`: emit a warning but don't fail the
//!   build; downstream code already handles a missing `DS4_CUDA_PTX`.
//! * On Linux with `CUDA_HOME` (default `/usr/local/cuda`) pointing at a
//!   toolkit, invoke nvcc and export `DS4_CUDA_PTX=<path>` via
//!   `cargo:rustc-env` so `kernels.rs` can pick it up with `option_env!`.
//!
//! Flags mirror the upstream Makefile (`ds4/Makefile`):
//! `-O3 --use_fast_math -arch=native` with `sm_80` as a fallback when the
//! installed nvcc rejects `native` (older toolkits don't accept it).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const UPSTREAM_CU: &str = "../../../ds4/ds4_cuda.cu";

fn main() {
    // Re-run triggers — the .cu source and the env vars we read.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={UPSTREAM_CU}");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=DS4_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=DS4_CUDA_NVCC");

    // Non-Linux: nothing to do. The crate's modules are cfg-gated out and
    // `option_env!("DS4_CUDA_PTX")` will be `None`, which is handled at
    // runtime.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" {
        return;
    }

    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = env::var("DS4_CUDA_NVCC")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&cuda_home).join("bin").join("nvcc"));

    if !nvcc.exists() {
        println!(
            "cargo:warning=ds4-cuda: nvcc not found at {} (set CUDA_HOME or DS4_CUDA_NVCC); \
             skipping PTX compile — GPU runtime will refuse to start.",
            nvcc.display()
        );
        return;
    }

    let cu_src = match locate_cu_source() {
        Some(p) => p,
        None => {
            println!(
                "cargo:warning=ds4-cuda: upstream ds4_cuda.cu not found at {UPSTREAM_CU}; \
                 skipping PTX compile."
            );
            return;
        }
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let ptx_path = out_dir.join("ds4_cuda.ptx");

    // Architecture: env override > -arch=native > sm_80 fallback.
    let arch_override = env::var("DS4_CUDA_ARCH").ok();
    let candidates: Vec<String> = match arch_override.as_deref() {
        Some(a) if !a.is_empty() => vec![a.to_string()],
        _ => vec!["native".into(), "sm_80".into()],
    };

    let mut last_err: Option<String> = None;
    for arch in &candidates {
        match run_nvcc(&nvcc, &cu_src, &ptx_path, &cuda_home, arch) {
            Ok(()) => {
                println!(
                    "cargo:rustc-env=DS4_CUDA_PTX={}",
                    ptx_path.display()
                );
                return;
            }
            Err(e) => {
                last_err = Some(e);
                // Try the next candidate (e.g. older toolkits reject `native`).
            }
        }
    }

    // None of the architectures worked. Emit a warning and bail without
    // failing the build, so downstream code can still compile on hosts where
    // CUDA isn't fully usable.
    if let Some(err) = last_err {
        for line in err.lines() {
            println!("cargo:warning=ds4-cuda: {line}");
        }
    }
    println!(
        "cargo:warning=ds4-cuda: nvcc compile failed; runtime PTX load will be skipped."
    );
}

fn locate_cu_source() -> Option<PathBuf> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest_dir).join(UPSTREAM_CU);
    if p.exists() {
        // Canonicalize so `rerun-if-changed` and nvcc both see a stable path.
        std::fs::canonicalize(&p).ok().or(Some(p))
    } else {
        None
    }
}

fn run_nvcc(
    nvcc: &Path,
    cu_src: &Path,
    ptx_path: &Path,
    cuda_home: &str,
    arch: &str,
) -> Result<(), String> {
    let include_dir = PathBuf::from(cuda_home).join("include");

    let mut cmd = Command::new(nvcc);
    cmd.arg("-ptx")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg(format!("-arch={arch}"))
        .arg("-std=c++17")
        .arg("-I")
        .arg(&include_dir)
        // Match upstream defines / warnings layout.
        .arg("-Xcompiler")
        .arg("-Wno-unused-function")
        .arg("-o")
        .arg(ptx_path)
        .arg(cu_src);

    let output = cmd
        .output()
        .map_err(|e| format!("failed to spawn nvcc ({}): {e}", nvcc.display()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "nvcc (-arch={arch}) exited with {}: {}",
            output.status, stderr
        ));
    }
    Ok(())
}
