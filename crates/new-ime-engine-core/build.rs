//! Build script for `new-ime-engine-core`.
//!
//! The active repository no longer carries the old C++ build tree, so the
//! KenLM shim is opt-in. On Windows/MSVC builds, set both
//! `NEW_IME_KENLM_SOURCE` and `NEW_IME_KENLM_LIB_DIR` to enable the shim.
//! All other hosts and target combinations skip the native step so
//! `cargo check` stays usable in the Rust-only workspace.

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

    let Some(kenlm_src) = env::var_os("NEW_IME_KENLM_SOURCE").map(PathBuf::from) else {
        println!(
            "cargo:warning=skipping KenLM shim build; set NEW_IME_KENLM_SOURCE and NEW_IME_KENLM_LIB_DIR to enable it"
        );
        return;
    };
    let Some(lib_dir) = env::var_os("NEW_IME_KENLM_LIB_DIR").map(PathBuf::from) else {
        println!(
            "cargo:warning=skipping KenLM shim build; set NEW_IME_KENLM_SOURCE and NEW_IME_KENLM_LIB_DIR to enable it"
        );
        return;
    };

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shim_src = manifest_dir.join("csrc/kenlm_shim.cpp");

    assert!(
        kenlm_src.join("lm/model.hh").exists(),
        "KenLM source missing at {}",
        kenlm_src.display()
    );
    assert!(
        lib_dir.join("kenlm.lib").exists(),
        "kenlm.lib not found at {}",
        lib_dir.display()
    );
    assert!(
        lib_dir.join("kenlm_util.lib").exists(),
        "kenlm_util.lib not found at {}",
        lib_dir.display()
    );

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
