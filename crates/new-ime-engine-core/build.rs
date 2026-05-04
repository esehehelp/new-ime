//! Build script for `new-ime-engine-core`.
//!
//! On Windows/MSVC, compiles the KenLM C shim (`csrc/kenlm_shim.cpp`) and
//! links it against `kenlm.lib` / `kenlm_util.lib`. Two ways to point at
//! them, in priority order:
//!
//!   1. `NEW_IME_KENLM_SOURCE` + `NEW_IME_KENLM_LIB_DIR` env vars (any
//!      layout the user prefers).
//!   2. Repo-relative auto-detect: source at `references/kenlm/`, libs at
//!      `build/kenlm_win/lib/Release/`. Produced by `scripts/setup_kenlm_win.sh`
//!      followed by a `cmake --build` in `build/kenlm_win/`.
//!
//! If neither yields a usable layout, the shim build is skipped — the
//! workspace then doesn't link KenLM and TSF runs CTC-only. Other host /
//! target combinations always skip so `cargo check` stays usable.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=NEW_IME_KENLM_SOURCE");
    println!("cargo:rerun-if-env-changed=NEW_IME_KENLM_LIB_DIR");

    let host = env::var("HOST").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    if !host.contains("windows") || !target.contains("windows-msvc") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir
        .ancestors()
        .nth(2)
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| manifest_dir.clone());

    let kenlm_src = match env::var_os("NEW_IME_KENLM_SOURCE") {
        Some(v) => PathBuf::from(v),
        None => repo_root.join("references").join("kenlm"),
    };
    let lib_dir = match env::var_os("NEW_IME_KENLM_LIB_DIR") {
        Some(v) => PathBuf::from(v),
        None => repo_root.join("build").join("kenlm_win").join("lib").join("Release"),
    };

    if !kenlm_src.join("lm/model.hh").exists()
        || !lib_dir.join("kenlm.lib").exists()
        || !lib_dir.join("kenlm_util.lib").exists()
    {
        println!(
            "cargo:warning=KenLM artifacts missing (src={}, libs={}); skipping shim, TSF will link without shallow fusion. Run scripts/setup_kenlm_win.sh + cmake build to enable.",
            kenlm_src.display(),
            lib_dir.display()
        );
        return;
    }

    let shim_src = manifest_dir.join("csrc/kenlm_shim.cpp");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .static_crt(false)
        .file(&shim_src)
        .include(&kenlm_src)
        .define("KENLM_MAX_ORDER", "6")
        .define("NOMINMAX", None)
        .flag_if_supported("/utf-8")
        .flag_if_supported("/EHsc")
        .compile("kenlm_shim");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=kenlm");
    println!("cargo:rustc-link-lib=static=kenlm_util");

    println!("cargo:rerun-if-changed={}", shim_src.display());
    println!(
        "cargo:rerun-if-changed={}",
        kenlm_src.join("lm/model.hh").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        lib_dir.join("kenlm.lib").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        lib_dir.join("kenlm_util.lib").display()
    );
}
