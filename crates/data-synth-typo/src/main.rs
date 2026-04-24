//! data-synth-typo: qwerty romaji 経路の打鍵ミスノイズを既存 JSONL に注入する。
//!
//! 入出力は `data-core::Row` schema (reading / surface / context / writer? /
//! domain? / source?)。surface / context は不変、reading のみ変形、source は
//! `synth_typo` で上書き。bunsetsu schema (`left_context_*` 等) は初期版では
//! 非対応 (follow-up PR)。

mod qwerty;
mod romaji;
mod rules;

use anyhow::{Context, Result};
use clap::Parser;
use data_core::{open_output, write_row, JsonlLines, Row};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rules::{TypoConfig, TypoKind};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "data-synth-typo",
    about = "Inject qwerty-romaji typing-error noise into JSONL reading fields"
)]
struct Cli {
    /// 入力 JSONL (.zst/.xz/.gz 可、複数指定可)。
    #[arg(long = "input", required = true, num_args = 1..)]
    inputs: Vec<PathBuf>,
    /// 出力 JSONL (.zst で自動圧縮)。
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 2_000_000)]
    max_rows: usize,
    /// Phase A では 1 row あたりの edit は固定 1。Phase B で Poisson 化。
    #[arg(long, default_value_t = 1.0)]
    mean_edits: f64,
    #[arg(long, default_value_t = 0.45)]
    weight_adjacent: f64,
    /// 指定割合の row にのみノイズ適用 (それ以外は emit しない)。
    #[arg(long, default_value_t = 1.0)]
    augment_ratio: f64,
    /// 元 row を `synth_typo_clean` source で並行 emit する割合 (debug 用)。
    #[arg(long, default_value_t = 0.0)]
    keep_clean_ratio: f64,
    /// zstd 圧縮レベル (出力が .zst の場合のみ有効)。
    #[arg(long, default_value_t = 10)]
    compression_level: i32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = TypoConfig {
        mean_edits: cli.mean_edits,
        weights: vec![(TypoKind::AdjacentKey, cli.weight_adjacent)],
    };

    let mut rng = StdRng::seed_from_u64(cli.seed);
    let mut writer = open_output(&cli.output, None, cli.compression_level)
        .with_context(|| format!("open {}", cli.output.display()))?;

    let mut emitted = 0usize;
    let mut read = 0usize;
    let mut noise_fail = 0usize;

    'outer: for input in &cli.inputs {
        eprintln!("[synth-typo] reading {}", input.display());
        let lines = JsonlLines::open(input)
            .with_context(|| format!("open input {}", input.display()))?;
        for row in lines {
            let row: Row = match row {
                Ok(r) => r,
                Err(_) => continue,
            };
            read += 1;

            let apply_noise = rng.gen::<f64>() < cli.augment_ratio;
            if apply_noise {
                if let Some(noisy) = rules::apply(&row.reading, &cfg, &mut rng) {
                    let mut out = row.clone();
                    out.reading = noisy;
                    out.source = Some("synth_typo".into());
                    write_row(&mut writer, &out)?;
                    emitted += 1;
                    if emitted >= cli.max_rows {
                        break 'outer;
                    }
                } else {
                    noise_fail += 1;
                }
            }

            if cli.keep_clean_ratio > 0.0 && rng.gen::<f64>() < cli.keep_clean_ratio {
                let mut out = row.clone();
                out.source = Some("synth_typo_clean".into());
                write_row(&mut writer, &out)?;
                emitted += 1;
                if emitted >= cli.max_rows {
                    break 'outer;
                }
            }

            if read % 100_000 == 0 {
                eprintln!(
                    "[synth-typo] read={} emitted={} noise_fail={}",
                    read, emitted, noise_fail
                );
            }
        }
    }

    drop(writer);
    eprintln!(
        "[synth-typo] done read={} emitted={} noise_fail={} output={}",
        read,
        emitted,
        noise_fail,
        cli.output.display()
    );
    Ok(())
}
