//! ONNX greedy-CTC probe_v3 benchmark for CTC-NAT checkpoints.
//!
//! Loads a fp32 or int8 ONNX export of `CTCNATInferenceWrapper`, runs greedy
//! CTC decode over every probe item on CPU, and reports EM1 / CharAcc / p50
//! latency. Companion to the Python `legacy dataset tools` pipeline;
//! this Rust version exists so int8 inference benchmarking does not depend on
//! a Python env / PyTorch install.
//!
//! Inference contract (matches models/tools/export/export_onnx_ctc_nat.py):
//!   inputs : input_ids (int64, 1xT), attention_mask (int64, 1xT)
//!   output : logits (float32, 1xTxV)
//!   T must equal the export-time --seq-len (default 128); shorter prompts
//!   are padded with [PAD]=0.

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use ndarray::Array2;
use ort::{inputs, session::Session, value::TensorRef};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// ort 2.0.0-rc errors are generic over R (a non-Send/Sync typestate marker),
/// which blocks automatic `?` conversion to `anyhow::Error`. Stringify in a
/// small helper so each call site can keep using `?`.
fn ort_err<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("ort: {}", e)
}

// Must match legacy Python reference constants.
const PAD_ID: i64 = 0;
const UNK_ID: i64 = 1;
const SEP_ID: i64 = 2;
const CLS_ID: i64 = 3;
const BLANK_ID: i64 = 4;
const MASK_ID: i64 = 5;

#[derive(Parser, Debug)]
#[command(about = "Greedy-CTC probe_v3 benchmark for ONNX-exported CTC-NAT models.")]
struct Args {
    /// ONNX model path (exported via export_onnx_ctc_nat.py).
    #[arg(long)]
    onnx: PathBuf,

    /// Tokenizer JSON (SharedCharTokenizer.save() format).
    #[arg(long)]
    tokenizer: PathBuf,

    /// probe_v3 JSON (AJIMEE-compatible items).
    #[arg(long, default_value = "datasets/eval/probe/probe.json")]
    probe: PathBuf,

    /// Must equal the ONNX export-time seq-len (default 128).
    #[arg(long, default_value_t = 128)]
    max_seq_len: usize,

    /// Max context chars fed into the encoder (matches Python backend default).
    #[arg(long, default_value_t = 40)]
    max_context: usize,

    /// Optional per-category EM1 breakdown dump (JSON path).
    #[arg(long)]
    output_json: Option<PathBuf>,

    /// Warm-up iterations before timing (excluded from p50/p95).
    #[arg(long, default_value_t = 3)]
    warmup: usize,
}

#[derive(Deserialize)]
struct TokenizerFile {
    #[allow(dead_code)]
    #[serde(default)]
    r#type: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    max_kanji: Option<usize>,
    token_to_id: BTreeMap<String, i64>,
}

struct Tokenizer {
    token_to_id: HashMap<String, i64>,
    id_to_token: HashMap<i64, String>,
}

impl Tokenizer {
    fn load(path: &PathBuf) -> Result<Self> {
        let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let file: TokenizerFile =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
        let mut t2i: HashMap<String, i64> = HashMap::with_capacity(file.token_to_id.len());
        let mut i2t: HashMap<i64, String> = HashMap::with_capacity(file.token_to_id.len());
        for (tok, id) in file.token_to_id.into_iter() {
            i2t.insert(id, tok.clone());
            t2i.insert(tok, id);
        }
        Ok(Self {
            token_to_id: t2i,
            id_to_token: i2t,
        })
    }

    /// Mirror SharedCharTokenizer.encode: char-by-char, byte-fallback on miss.
    fn encode(&self, text: &str) -> Vec<i64> {
        let mut out: Vec<i64> = Vec::with_capacity(text.chars().count());
        for ch in text.chars() {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            if let Some(&id) = self.token_to_id.get(s) {
                out.push(id);
                continue;
            }
            // byte fallback: emit <0xXX> for each UTF-8 byte.
            for &b in s.as_bytes() {
                let key = format!("<0x{:02X}>", b);
                let id = self.token_to_id.get(&key).copied().unwrap_or(UNK_ID);
                out.push(id);
            }
        }
        out
    }

    fn encode_with_special(&self, context: &str, reading: &str) -> Vec<i64> {
        let mut ids = Vec::with_capacity(context.chars().count() + reading.chars().count() + 2);
        ids.push(CLS_ID);
        ids.extend(self.encode(context));
        ids.push(SEP_ID);
        ids.extend(self.encode(reading));
        ids
    }

    /// Mirror SharedCharTokenizer.decode: skip specials, accumulate <0xXX>
    /// byte-fallbacks into a UTF-8 run, flush on any regular token.
    fn decode(&self, ids: &[i64]) -> String {
        let mut out = String::new();
        let mut byte_buf: Vec<u8> = Vec::new();
        let flush = |buf: &mut Vec<u8>, out: &mut String| {
            if buf.is_empty() {
                return;
            }
            match std::str::from_utf8(buf) {
                Ok(s) => out.push_str(s),
                Err(_) => out.push_str("<INVALID>"),
            }
            buf.clear();
        };
        for &id in ids {
            if matches!(id, PAD_ID | CLS_ID | SEP_ID | BLANK_ID | MASK_ID) {
                continue;
            }
            let tok = match self.id_to_token.get(&id) {
                Some(t) => t,
                None => continue,
            };
            if tok.starts_with("<0x") && tok.ends_with(">") && tok.len() == 6 {
                if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                    byte_buf.push(b);
                    continue;
                }
            }
            flush(&mut byte_buf, &mut out);
            // Skip other special-bracketed tokens.
            if !(tok.starts_with('[') && tok.ends_with(']')) {
                out.push_str(tok);
            }
        }
        flush(&mut byte_buf, &mut out);
        out
    }
}

#[derive(Deserialize)]
struct ProbeItem {
    #[serde(default)]
    category: String,
    #[serde(default)]
    context_text: String,
    input: String,
    expected_output: Vec<String>,
}

fn load_probe(path: &PathBuf) -> Result<Vec<ProbeItem>> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))
}

fn kata_to_hira(s: &str) -> String {
    s.chars()
        .map(|c| {
            let code = c as u32;
            if (0x30A1..=0x30F6).contains(&code) {
                char::from_u32(code - 0x60).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        s.to_string()
    } else {
        chars[chars.len() - max_chars..].iter().collect()
    }
}

/// Collapse CTC greedy path: drop blanks, dedup consecutive repeats.
fn ctc_collapse(ids: &[i64]) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::with_capacity(ids.len());
    let mut prev: i64 = -1;
    for &id in ids {
        if id == BLANK_ID {
            prev = BLANK_ID;
            continue;
        }
        if id != prev {
            out.push(id);
        }
        prev = id;
    }
    out
}

fn argmax_last_dim(logits: &[f32], time: usize, vocab: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(time);
    for t in 0..time {
        let base = t * vocab;
        let mut best_id = 0usize;
        let mut best_v = logits[base];
        for v in 1..vocab {
            let x = logits[base + v];
            if x > best_v {
                best_v = x;
                best_id = v;
            }
        }
        out.push(best_id as i64);
    }
    out
}

/// Character-level Levenshtein-derived char accuracy used by EvalResult.
/// `char_acc = 1 - edit_distance(ref, hyp) / max(len(ref), 1)`.
fn char_acc(reference: &str, hypothesis: &str) -> f64 {
    let r: Vec<char> = reference.chars().collect();
    let h: Vec<char> = hypothesis.chars().collect();
    if r.is_empty() && h.is_empty() {
        return 1.0;
    }
    if r.is_empty() {
        return 0.0;
    }
    let n = r.len();
    let m = h.len();
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur: Vec<usize> = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = if r[i - 1] == h[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let dist = prev[m] as f64;
    let max_len = n as f64;
    (1.0 - dist / max_len).max(0.0)
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("loading tokenizer: {}", args.tokenizer.display());
    let tokenizer = Tokenizer::load(&args.tokenizer)?;
    eprintln!("  vocab_size = {}", tokenizer.token_to_id.len());

    eprintln!("loading probe: {}", args.probe.display());
    let items = load_probe(&args.probe)?;
    eprintln!("  {} items", items.len());

    eprintln!("loading ONNX: {}", args.onnx.display());
    ort::init().commit();
    let mut session = Session::builder()
        .map_err(ort_err)?
        .with_intra_threads(4)
        .map_err(ort_err)?
        .commit_from_file(&args.onnx)
        .map_err(ort_err)?;

    let seq_len = args.max_seq_len;
    let mut input_ids = Array2::<i64>::zeros((1, seq_len));
    let mut attn_mask = Array2::<i64>::zeros((1, seq_len));

    // Warmup
    if args.warmup > 0 {
        eprintln!("warmup {} iterations...", args.warmup);
        for _ in 0..args.warmup {
            input_ids.fill(PAD_ID);
            attn_mask.fill(0);
            input_ids[(0, 0)] = CLS_ID;
            attn_mask[(0, 0)] = 1;
            let outs = session
                .run(inputs! {
                    "input_ids" => TensorRef::from_array_view(&input_ids).map_err(ort_err)?,
                    "attention_mask" => TensorRef::from_array_view(&attn_mask).map_err(ort_err)?,
                })
                .map_err(ort_err)?;
            let _ = outs["logits"].try_extract_array::<f32>().map_err(ort_err)?;
        }
    }

    // Eval loop
    let mut per_cat_hits: HashMap<String, (usize, usize)> = HashMap::new();
    let mut char_accs: Vec<f64> = Vec::with_capacity(items.len());
    let mut latencies_ms: Vec<f64> = Vec::with_capacity(items.len());
    let mut em1_hits: usize = 0;
    let mut sample_fails: Vec<(String, String, String)> = Vec::new();

    for (i, item) in items.iter().enumerate() {
        let ctx = truncate_chars(&item.context_text, args.max_context);
        let reading_hira = kata_to_hira(&item.input);
        let mut ids = tokenizer.encode_with_special(&ctx, &reading_hira);
        if ids.len() > seq_len {
            ids.truncate(seq_len);
        }
        input_ids.fill(PAD_ID);
        attn_mask.fill(0);
        for (k, &id) in ids.iter().enumerate() {
            input_ids[(0, k)] = id;
            attn_mask[(0, k)] = 1;
        }

        let t0 = Instant::now();
        let outs = session
            .run(inputs! {
                "input_ids" => TensorRef::from_array_view(&input_ids).map_err(ort_err)?,
                "attention_mask" => TensorRef::from_array_view(&attn_mask).map_err(ort_err)?,
            })
            .map_err(ort_err)?;
        let (shape, logits) = outs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        latencies_ms.push(ms);

        if shape.len() != 3 {
            return Err(anyhow!("unexpected logits shape: {:?}", shape));
        }
        let t = shape[1] as usize;
        let v = shape[2] as usize;
        // Only score the valid (non-pad) prefix of length ids.len().
        let active = ids.len().min(t);
        let argmax_ids = argmax_last_dim(&logits[..active * v], active, v);
        let collapsed = ctc_collapse(&argmax_ids);
        let hypothesis = tokenizer.decode(&collapsed);

        let hit = item.expected_output.iter().any(|s| s == &hypothesis);
        if hit {
            em1_hits += 1;
        } else if sample_fails.len() < 8 {
            sample_fails.push((
                reading_hira.clone(),
                item.expected_output.first().cloned().unwrap_or_default(),
                hypothesis.clone(),
            ));
        }

        let reference = item.expected_output.first().cloned().unwrap_or_default();
        char_accs.push(char_acc(&reference, &hypothesis));

        let entry = per_cat_hits.entry(item.category.clone()).or_insert((0, 0));
        entry.0 += 1;
        if hit {
            entry.1 += 1;
        }

        if (i + 1) % 50 == 0 {
            eprintln!("  [{}/{}]", i + 1, items.len());
        }
    }

    let n = items.len() as f64;
    let em1 = em1_hits as f64 / n.max(1.0);
    let char_acc_mean = if char_accs.is_empty() {
        0.0
    } else {
        char_accs.iter().sum::<f64>() / char_accs.len() as f64
    };
    let mut lat_sorted = latencies_ms.clone();
    lat_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ln = lat_sorted.len();
    let p50 = lat_sorted[ln / 2];
    let p95 = lat_sorted[(ln as f64 * 0.95) as usize];
    let mean_ms = lat_sorted.iter().sum::<f64>() / ln as f64;

    println!("\n=== RESULT ===");
    println!("  n              = {}", items.len());
    println!(
        "  EM1            = {:.4} ({}/{})",
        em1,
        em1_hits,
        items.len()
    );
    println!("  CharAcc (top1) = {:.4}", char_acc_mean);
    println!(
        "  latency ms     = p50 {:.1} / p95 {:.1} / mean {:.1}",
        p50, p95, mean_ms
    );
    println!("  per-category:");
    let mut cats: Vec<_> = per_cat_hits.iter().collect();
    cats.sort_by(|a, b| a.0.cmp(b.0));
    for (cat, (total, hits)) in cats {
        println!(
            "    {:<10} EM1={:.3}  ({}/{})",
            cat,
            *hits as f64 / *total as f64,
            hits,
            total
        );
    }
    if !sample_fails.is_empty() {
        println!("\n  sample failures:");
        for (r, refs, pred) in &sample_fails {
            println!("    reading={}  ref={}  pred={}", r, refs, pred);
        }
    }

    if let Some(out_path) = args.output_json {
        let per_cat_json: serde_json::Map<String, serde_json::Value> = per_cat_hits
            .into_iter()
            .map(|(c, (total, hits))| {
                (
                    c,
                    serde_json::json!({
                        "n": total,
                        "em1": hits as f64 / total.max(1) as f64,
                    }),
                )
            })
            .collect();
        let summary = serde_json::json!({
            "onnx_path": args.onnx.to_string_lossy(),
            "tokenizer_path": args.tokenizer.to_string_lossy(),
            "probe_path": args.probe.to_string_lossy(),
            "n": items.len(),
            "em1": em1,
            "char_acc_top1": char_acc_mean,
            "latency_ms": {
                "p50": p50,
                "p95": p95,
                "mean": mean_ms,
            },
            "per_category": per_cat_json,
        });
        fs::write(&out_path, serde_json::to_string_pretty(&summary)?)?;
        eprintln!("wrote summary: {}", out_path.display());
    }

    Ok(())
}
