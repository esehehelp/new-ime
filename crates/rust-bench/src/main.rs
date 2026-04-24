#[cfg(feature = "native-tch")]
#[path = "native.rs"]
mod native;

#[cfg(feature = "native-tch")]
fn main() -> anyhow::Result<()> {
    native::main()
}

#[cfg(not(feature = "native-tch"))]
use anyhow::{bail, Result};
#[cfg(not(feature = "native-tch"))]
use clap::Parser;
#[cfg(not(feature = "native-tch"))]
use std::path::PathBuf;

#[cfg(not(feature = "native-tch"))]
#[derive(Parser)]
#[command(
    name = "rust-bench",
    about = "Benchmark contract runner (build with --features native-tch for model execution)"
)]
struct Cli {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    run_dir: Option<PathBuf>,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long)]
    out_dir: Option<PathBuf>,
    #[arg(long)]
    markdown: Option<PathBuf>,
    #[arg(long)]
    model_name: Option<String>,
    #[arg(long)]
    probe_path: Option<PathBuf>,
    #[arg(long)]
    ajimee_path: Option<PathBuf>,
    #[arg(long)]
    general_path: Option<PathBuf>,
    #[arg(long)]
    benches: Option<String>,
    #[arg(long)]
    num_beams: Option<usize>,
    #[arg(long)]
    num_return: Option<usize>,
}

#[cfg(not(feature = "native-tch"))]
fn main() -> Result<()> {
    let _ = Cli::parse();
    bail!(
        "rust-bench was built without the `native-tch` feature — model execution is disabled.\n\
         Rebuild with:  cargo run -p rust-bench --features native-tch -- <args>\n\
         Requires LIBTORCH to point at a libtorch install (see docs/benchmark_comparison.md)."
    )
}
