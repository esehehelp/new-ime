//! data-synth-goji: かな直接レベルの音形/字形混同ノイズを JSONL reading に注入。
//!
//! 入出力は `data-core::Row`。surface / context 不変、reading のみ変形、source
//! を `synth_goji` で上書き。bunsetsu schema は初期版では非対応。

mod kana_tables;
mod rules;

use anyhow::{Context, Result};
use clap::Parser;
use data_core::{open_output, write_row, JsonlLines, Row};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rules::{GojiConfig, GojiKind};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "data-synth-goji",
    about = "Inject kana-level phonetic / orthographic confusion noise into JSONL reading fields"
)]
struct Cli {
    #[arg(long = "input", required = true, num_args = 1..)]
    inputs: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 2_000_000)]
    max_rows: usize,
    #[arg(long, default_value_t = 1.0)]
    mean_edits: f64,
    #[arg(long, default_value_t = 0.40)]
    weight_dakuten: f64,
    #[arg(long, default_value_t = 0.25)]
    weight_small_kana: f64,
    #[arg(long, default_value_t = 0.20)]
    weight_chouon: f64,
    #[arg(long, default_value_t = 0.10)]
    weight_hira_kata: f64,
    #[arg(long, default_value_t = 0.05)]
    weight_homophone_kana: f64,
    #[arg(long, default_value_t = 0.5)]
    length_drift_max: f64,
    #[arg(long, default_value_t = 1.0)]
    augment_ratio: f64,
    #[arg(long, default_value_t = 0.0)]
    keep_clean_ratio: f64,
    #[arg(long, default_value_t = 10)]
    compression_level: i32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = GojiConfig {
        mean_edits: cli.mean_edits,
        weights: vec![
            (GojiKind::DakutenFlip, cli.weight_dakuten),
            (GojiKind::SmallKana, cli.weight_small_kana),
            (GojiKind::ChouonConfusion, cli.weight_chouon),
            (GojiKind::HiraKataConfuse, cli.weight_hira_kata),
            (GojiKind::HomophoneKana, cli.weight_homophone_kana),
        ],
        length_drift_max: cli.length_drift_max,
    };

    let mut rng = StdRng::seed_from_u64(cli.seed);
    let mut writer = open_output(&cli.output, None, cli.compression_level)
        .with_context(|| format!("open {}", cli.output.display()))?;

    let mut emitted = 0usize;
    let mut read = 0usize;
    let mut noise_fail = 0usize;

    'outer: for input in &cli.inputs {
        eprintln!("[synth-goji] reading {}", input.display());
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
                    out.source = Some("synth_goji".into());
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
                out.source = Some("synth_goji_clean".into());
                write_row(&mut writer, &out)?;
                emitted += 1;
                if emitted >= cli.max_rows {
                    break 'outer;
                }
            }

            if read % 100_000 == 0 {
                eprintln!(
                    "[synth-goji] read={} emitted={} noise_fail={}",
                    read, emitted, noise_fail
                );
            }
        }
    }

    drop(writer);
    eprintln!(
        "[synth-goji] done read={} emitted={} noise_fail={} output={}",
        read,
        emitted,
        noise_fail,
        cli.output.display()
    );
    Ok(())
}
