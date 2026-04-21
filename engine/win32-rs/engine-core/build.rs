//! Build script for new-ime-engine-core.
//!
//! Compiles the KenLM C shim (`csrc/kenlm_shim.cpp`) and links it against
//! the pre-built `kenlm.lib` / `kenlm_util.lib` from the C++ build tree
//! (`build/kenlm_win/lib/Release/`). Those libs are produced by
//! `engine/win32/build.bat` the first time it is run — we reuse them
//! instead of rebuilding KenLM from scratch.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir
        .ancestors()
        .nth(3)
        .expect("repo root (../../.. from engine-core)")
        .to_path_buf();

    let kenlm_src = repo_root.join("engine/server/third_party/kenlm");
    let kenlm_build = repo_root.join("build/kenlm_win");
    let shim_src = manifest_dir.join("csrc/kenlm_shim.cpp");

    assert!(
        kenlm_src.join("lm/model.hh").exists(),
        "KenLM source missing at {}",
        kenlm_src.display()
    );
    assert!(
        kenlm_build.join("lib/Release/kenlm.lib").exists(),
        "kenlm.lib not found at {}; run engine/win32/build.bat first",
        kenlm_build.display()
    );

    // Compile the shim with the KenLM headers on the include path. Use the
    // dynamic CRT (/MD) to match Rust's default — then emit a NODEFAULTLIB
    // for the static CRT (LIBCMT) so the duplicate-symbol conflicts from
    // the existing kenlm.lib (which was built /MT) fall away: the dynamic
    // CRT wins the link.
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

    // KenLM must have been built with `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL`
    // (/MD) so the Runtime Library matches Rust's dynamic CRT default. If
    // the libs in `build/kenlm_win/` were built with /MT instead, delete
    // that directory and rerun the cmake from the README / plan.

    // Link order matters on MSVC: shim.obj → kenlm.lib → kenlm_util.lib.
    // zlib / bz2 are pulled in transitively by kenlm_util on Linux but the
    // Windows static build we're reusing already absorbed them into
    // kenlm_util.lib via FORCE_STATIC=ON (see engine/win32/build.bat).
    let lib_dir = kenlm_build.join("lib/Release");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=kenlm");
    println!("cargo:rustc-link-lib=static=kenlm_util");

    // Rebuild when the shim or the KenLM libs change.
    println!("cargo:rerun-if-changed={}", shim_src.display());
    println!(
        "cargo:rerun-if-changed={}",
        lib_dir.join("kenlm.lib").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        lib_dir.join("kenlm_util.lib").display()
    );

    // Silence dead-code warnings during the cc build.
    let _ = Path::new(&shim_src);
}
