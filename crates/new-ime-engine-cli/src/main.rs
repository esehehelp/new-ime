//! Suiko 推論 daemon / one-shot CLI.
//!
//! Python bench (`src/new_ime/eval/rust_engine_backend.py`) から subprocess
//! として起動され、stdin/stdout JSONL で 1 起動 / N requests を捌く。
//! 1 件ごとに ONNX session を再生成しないため、起動 overhead が分散される。

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use new_ime_engine_core::EngineSession;
use serde::{Deserialize, Serialize};

/// Decode mode the daemon will run for every request. Decided at startup
/// from CLI flags so the hot path doesn't re-check.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DecodeMode {
    /// Pure CTC greedy via `EngineSession::greedy_decode` — per-frame
    /// argmax + collapse. No prefix beam, no validate/fallback. Selected
    /// when beam_width == 1 AND no KenLM is attached.
    Greedy,
    /// `EngineSession::convert` — prefix beam search (+ optional KenLM
    /// shallow fusion) with candidate validation and kana fallback.
    Beam,
}

#[derive(Parser, Debug)]
#[command(
    name = "new-ime-engine-cli",
    version,
    about = "Suiko ONNX engine: one-shot convert or daemon mode."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Convert a single reading and print candidates as JSON, then exit.
    Convert {
        #[command(flatten)]
        engine: EngineArgs,
        #[arg(long)]
        reading: String,
        #[arg(long, default_value = "")]
        context: String,
    },
    /// Read JSONL requests from stdin, write JSONL responses to stdout.
    /// Stays alive until stdin closes.
    Daemon {
        #[command(flatten)]
        engine: EngineArgs,
    },
}

#[derive(Parser, Debug, Clone)]
struct EngineArgs {
    /// ONNX model artifact (fp32 or int8).
    #[arg(long)]
    onnx: PathBuf,

    /// vocab.hex.tsv sidecar. If omitted, the engine derives a default
    /// path next to the ONNX file (see EngineSession::load).
    #[arg(long)]
    vocab: Option<PathBuf>,

    /// Quantisation form of the ONNX artifact (echoed in the daemon
    /// `ready` message so the Python side can record `artifact_format`).
    #[arg(long, value_enum)]
    artifact_format: ArtifactFormat,

    /// Beam width passed straight to the engine (1 = greedy).
    #[arg(long, default_value_t = 1)]
    beam_width: usize,

    /// Top-K candidates returned per convert (= EngineSession.max_candidates).
    #[arg(long, default_value_t = 1)]
    top_k: usize,

    /// Per-step expansion breadth in the prefix-beam loop. Defaults to the
    /// engine-core default; only override when matching legacy traces.
    #[arg(long)]
    top_k_per_step: Option<usize>,

    /// Single KenLM model. Mutually exclusive with --kenlm-domain.
    #[arg(long)]
    kenlm: Option<PathBuf>,

    /// MoE entry: --kenlm-domain general=path/to.bin (repeatable).
    #[arg(long, value_parser = parse_domain_path, action = clap::ArgAction::Append)]
    kenlm_domain: Vec<(String, PathBuf)>,

    /// Shallow-fusion alpha (LM weight). Engine default if omitted.
    #[arg(long)]
    kenlm_alpha: Option<f32>,

    /// Shallow-fusion beta (length bonus). Engine default if omitted.
    #[arg(long)]
    kenlm_beta: Option<f32>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ArtifactFormat {
    Fp32,
    Int8,
}

impl ArtifactFormat {
    fn as_str(self) -> &'static str {
        match self {
            ArtifactFormat::Fp32 => "onnx-fp32",
            ArtifactFormat::Int8 => "onnx-int8",
        }
    }
}

fn parse_domain_path(s: &str) -> Result<(String, PathBuf), String> {
    let (name, path) = s
        .split_once('=')
        .ok_or_else(|| format!("expected NAME=PATH, got {s:?}"))?;
    if name.is_empty() {
        return Err("domain name must be non-empty".into());
    }
    Ok((name.to_string(), PathBuf::from(path)))
}

/// Construct an EngineSession from CLI args and decide the decode mode.
/// The vocab arg is honoured if given by symlinking / copying isn't
/// possible cross-platform — instead we fall back to the default sidecar
/// resolution and panic if the user passed an explicit path that doesn't
/// match. This keeps load() simple for v1.
fn build_session(args: &EngineArgs) -> Result<(EngineSession, DecodeMode)> {
    let mut session = EngineSession::load(&args.onnx)
        .with_context(|| format!("load engine from {}", args.onnx.display()))?;

    // Override defaults with CLI values.
    session.beam_width = args.beam_width.max(1);
    session.max_candidates = args.top_k.max(1);
    if let Some(tk) = args.top_k_per_step {
        session.top_k_per_step = tk;
    }
    // KenLM attachment: --kenlm and --kenlm-domain are mutually exclusive.
    let single = args.kenlm.is_some();
    let moe = !args.kenlm_domain.is_empty();
    if single && moe {
        return Err(anyhow!("--kenlm and --kenlm-domain are mutually exclusive"));
    }
    let lm_attached = single || moe;
    if let Some(path) = &args.kenlm {
        session
            .attach_kenlm(path)
            .with_context(|| format!("attach kenlm {}", path.display()))?;
    } else if moe {
        let mut paths: HashMap<String, PathBuf> = HashMap::new();
        for (name, path) in &args.kenlm_domain {
            paths.insert(name.clone(), path.clone());
        }
        session.attach_kenlm_moe(&paths).context("attach kenlm moe")?;
    }

    // alpha/beta are part of the same fused-score formula that the beam
    // ranks by — both must be zero in the no-LM case, otherwise the
    // length bonus (lm_beta * prefix.len()) silently biases the beam
    // toward longer prefixes even with no scorer attached. The engine's
    // default of (0.3, 0.6) is tuned for KenLM beam5; it must NOT leak
    // into the no-LM (greedy / pure beam) regime.
    if lm_attached {
        if let Some(a) = args.kenlm_alpha {
            session.lm_alpha = a;
        }
        if let Some(b) = args.kenlm_beta {
            session.lm_beta = b;
        }
    } else {
        session.lm_alpha = args.kenlm_alpha.unwrap_or(0.0);
        session.lm_beta = args.kenlm_beta.unwrap_or(0.0);
    }

    // beam_width=1 with no LM is classical greedy CTC; using
    // EngineSession::convert in that regime adds a prefix-beam pass and
    // kana validation/fallback that change the output vs the legacy
    // Python greedy_decode reference. Route those calls to greedy_decode
    // directly. Anything wider, or with a scorer attached, stays on the
    // beam path.
    let mode = if !lm_attached && session.beam_width <= 1 {
        DecodeMode::Greedy
    } else {
        DecodeMode::Beam
    };

    if let Some(vocab) = &args.vocab {
        // Light sanity check: warn (via stderr) if the explicit vocab arg
        // doesn't match the auto-derived sidecar. We don't fail because the
        // engine has already loaded successfully.
        if !vocab.exists() {
            eprintln!(
                "[engine-cli] WARN --vocab {} does not exist; using auto-derived sidecar",
                vocab.display()
            );
        }
    }

    Ok((session, mode))
}

fn decode_one(
    session: &mut EngineSession,
    mode: DecodeMode,
    context: &str,
    reading: &str,
) -> Result<Vec<String>> {
    if reading.is_empty() {
        return Ok(vec![String::new()]);
    }
    match mode {
        DecodeMode::Greedy => {
            let s = session.greedy_decode(context, reading)?;
            Ok(vec![s])
        }
        DecodeMode::Beam => session.convert(context, reading),
    }
}

#[derive(Deserialize, Debug)]
struct Request {
    id: u64,
    #[serde(default)]
    context: String,
    reading: String,
}

#[derive(Serialize, Debug)]
struct Response<'a> {
    id: u64,
    candidates: Vec<String>,
    engine_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

#[derive(Serialize, Debug)]
struct Ready {
    ready: bool,
    version: &'static str,
    artifact_format: &'static str,
    beam_width: usize,
    top_k: usize,
}

fn run_daemon(args: EngineArgs) -> Result<()> {
    let (mut session, mode) = build_session(&args)?;
    let stdout = std::io::stdout();
    let stdin = std::io::stdin();

    // Emit ready handshake first so the parent can record runtime metadata
    // and unblock its own initialisation.
    let ready = Ready {
        ready: true,
        version: env!("CARGO_PKG_VERSION"),
        artifact_format: args.artifact_format.as_str(),
        beam_width: session.beam_width,
        top_k: session.max_candidates,
    };
    {
        let mut out = stdout.lock();
        writeln!(out, "{}", serde_json::to_string(&ready)?)?;
        out.flush()?;
    }

    let reader = BufReader::new(stdin.lock());
    for line in reader.lines() {
        let line = line.context("read stdin line")?;
        if line.trim().is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let err = format!("invalid request json: {e}");
                let resp = Response {
                    id: 0,
                    candidates: Vec::new(),
                    engine_ms: 0.0,
                    error: Some(&err),
                };
                let mut out = stdout.lock();
                writeln!(out, "{}", serde_json::to_string(&resp)?)?;
                out.flush()?;
                continue;
            }
        };

        let t0 = Instant::now();
        let result = decode_one(&mut session, mode, &req.context, &req.reading);
        let engine_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let mut out = stdout.lock();
        match result {
            Ok(candidates) => {
                let resp = Response {
                    id: req.id,
                    candidates,
                    engine_ms,
                    error: None,
                };
                writeln!(out, "{}", serde_json::to_string(&resp)?)?;
            }
            Err(e) => {
                let msg = e.to_string();
                let resp = Response {
                    id: req.id,
                    candidates: Vec::new(),
                    engine_ms,
                    error: Some(&msg),
                };
                writeln!(out, "{}", serde_json::to_string(&resp)?)?;
            }
        }
        out.flush()?;
    }
    Ok(())
}

fn run_convert(args: EngineArgs, reading: String, context: String) -> Result<()> {
    let (mut session, mode) = build_session(&args)?;
    let t0 = Instant::now();
    let candidates = decode_one(&mut session, mode, &context, &reading)?;
    let engine_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let resp = Response {
        id: 0,
        candidates,
        engine_ms,
        error: None,
    };
    println!("{}", serde_json::to_string(&resp)?);
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Convert {
            engine,
            reading,
            context,
        } => run_convert(engine, reading, context),
        Command::Daemon { engine } => run_daemon(engine),
    }
}
