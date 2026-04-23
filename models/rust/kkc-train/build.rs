// Explicitly link torch_cuda and c10_cuda when the `cuda` feature is enabled.
// tch's own build.rs only links `-ltorch` and relies on GNU ld's
// --copy-dt-needed-entries to pull in CUDA libs transitively, which does not
// work on Windows (MSVC linker ignores those GNU ld flags).
fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_USE_PYTORCH");

    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        if let Ok(lib_path) = std::env::var("LIBTORCH") {
            let normalized = lib_path.replace('/', "\\");
            println!("cargo:rustc-link-search=native={}\\lib", normalized);
        }
        println!("cargo:rustc-link-lib=torch_cuda");
        println!("cargo:rustc-link-lib=c10_cuda");
    }
}
