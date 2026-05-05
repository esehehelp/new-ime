//! Build script for `new-ime-engine-core`.
//!
//! Compiles the KenLM C shim (`csrc/kenlm_shim.cpp`) and links it against
//! `kenlm` + `kenlm_util` static libraries. Two target platforms are
//! supported:
//!
//!   * Windows / MSVC: link `kenlm.lib` + `kenlm_util.lib` produced by
//!     `scripts/setup_kenlm_win.sh` + cmake build → `build/kenlm_win/lib/Release/`.
//!   * Linux / gnu:    link `libkenlm.a` + `libkenlm_util.a` produced by
//!     `scripts/setup_kenlm_linux.sh` → `build/kenlm_linux/lib/`.
//!
//! Lookup priority (both targets):
//!   1. `NEW_IME_KENLM_SOURCE` + `NEW_IME_KENLM_LIB_DIR` env vars (any
//!      layout the user prefers).
//!   2. Repo-relative auto-detect at the platform-specific default path.
//!
//! When the shim is built, `cargo:rustc-cfg=has_kenlm` is emitted so
//! `src/kenlm.rs` can switch its FFI declarations between the real extern
//! and a no-op stub via a single `#[cfg(has_kenlm)]` gate. Other host /
//! target combinations (e.g. macOS, Windows GNU) skip the shim build, the
//! cfg is not set, and the engine runs CTC-only.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_kenlm)");
    println!("cargo:rerun-if-env-changed=NEW_IME_KENLM_SOURCE");
    println!("cargo:rerun-if-env-changed=NEW_IME_KENLM_LIB_DIR");

    let target = env::var("TARGET").unwrap_or_default();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir
        .ancestors()
        .nth(2)
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| manifest_dir.clone());

    let plat = if target.contains("windows-msvc") {
        Some(Platform::WindowsMsvc)
    } else if target.contains("linux-gnu") || target.contains("linux-musl") {
        Some(Platform::LinuxGnu)
    } else {
        None
    };

    let Some(plat) = plat else {
        println!(
            "cargo:warning=KenLM shim skipped for target {target} (only windows-msvc / linux-gnu supported); engine runs CTC-only."
        );
        return;
    };

    let kenlm_src = match env::var_os("NEW_IME_KENLM_SOURCE") {
        Some(v) => PathBuf::from(v),
        None => repo_root.join("references").join("kenlm"),
    };
    let lib_dir = match env::var_os("NEW_IME_KENLM_LIB_DIR") {
        Some(v) => PathBuf::from(v),
        None => plat.default_lib_dir(&repo_root),
    };

    let (lm_lib, util_lib) = plat.lib_filenames();
    if !kenlm_src.join("lm/model.hh").exists()
        || !lib_dir.join(lm_lib).exists()
        || !lib_dir.join(util_lib).exists()
    {
        println!(
            "cargo:warning=KenLM artifacts missing (src={}, libs={}); skipping shim, engine will run without shallow fusion. Run scripts/setup_kenlm_{}.sh to build.",
            kenlm_src.display(),
            lib_dir.display(),
            plat.script_suffix(),
        );
        return;
    }

    let shim_src = manifest_dir.join("csrc/kenlm_shim.cpp");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(&shim_src)
        .include(&kenlm_src)
        .define("KENLM_MAX_ORDER", "6");

    match plat {
        Platform::WindowsMsvc => {
            build
                .static_crt(false)
                .define("NOMINMAX", None)
                .flag_if_supported("/utf-8")
                .flag_if_supported("/EHsc");
        }
        Platform::LinuxGnu => {
            // The static libs were built with -DKENLM_LIBS_ONLY=ON which
            // implies -fPIC (set by setup_kenlm_linux.sh via
            // CMAKE_POSITION_INDEPENDENT_CODE). The shim must agree.
            build.flag_if_supported("-fPIC");
        }
    }
    build.compile("kenlm_shim");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=kenlm");
    println!("cargo:rustc-link-lib=static=kenlm_util");
    if matches!(plat, Platform::LinuxGnu) {
        // KenLM uses C++ stdlib; cc crate links the shim's stdlib but the
        // static .a archives have unresolved C++ symbols that need stdc++.
        // Also pull in zlib/bz2/lzma which KenLM optionally uses for
        // compressed input — fine to link unconditionally on Linux.
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=z");
        println!("cargo:rustc-link-lib=bz2");
        println!("cargo:rustc-link-lib=lzma");
    }

    println!("cargo:rustc-cfg=has_kenlm");
    println!("cargo:rerun-if-changed={}", shim_src.display());
    println!(
        "cargo:rerun-if-changed={}",
        kenlm_src.join("lm/model.hh").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        lib_dir.join(lm_lib).display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        lib_dir.join(util_lib).display()
    );
}

#[derive(Copy, Clone, Debug)]
enum Platform {
    WindowsMsvc,
    LinuxGnu,
}

impl Platform {
    fn default_lib_dir(self, repo_root: &std::path::Path) -> PathBuf {
        match self {
            Platform::WindowsMsvc => repo_root.join("build").join("kenlm_win").join("lib").join("Release"),
            Platform::LinuxGnu => repo_root.join("build").join("kenlm_linux").join("lib"),
        }
    }
    fn lib_filenames(self) -> (&'static str, &'static str) {
        match self {
            Platform::WindowsMsvc => ("kenlm.lib", "kenlm_util.lib"),
            Platform::LinuxGnu => ("libkenlm.a", "libkenlm_util.a"),
        }
    }
    fn script_suffix(self) -> &'static str {
        match self {
            Platform::WindowsMsvc => "win",
            Platform::LinuxGnu => "linux",
        }
    }
}
