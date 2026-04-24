#[path = "layers.rs"]
mod layers;
#[path = "model.rs"]
mod model;

use anyhow::{bail, Context, Result};
use clap::Parser;
use model::CtcNatModel;
use rust_tokenizer::SharedCharTokenizer;
use safetensors::tensor::Dtype;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tch::nn::VarStore;
use tch::{Device, Kind, Tensor};

const DEFAULT_PROBE_PATH: &str = "datasets/eval/probe/probe.json";
const DEFAULT_AJIMEE_PATH: &str = "references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json";

#[derive(Parser)]
#[command(
    name = "rust-bench",
    about = "Benchmark runner for self/external KKC models"
)]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    run_dir: Option<PathBuf>,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long, default_value = "results/eval_runs_rust")]
    out_dir: PathBuf,
    #[arg(long)]
    markdown: Option<PathBuf>,
    #[arg(long)]
    model_name: Option<String>,
    #[arg(long, default_value = DEFAULT_PROBE_PATH)]
    probe_path: PathBuf,
    #[arg(long, default_value = DEFAULT_AJIMEE_PATH)]
    ajimee_path: PathBuf,
    #[arg(long, default_value = "datasets/eval/general/dev.jsonl")]
    general_path: PathBuf,
    #[arg(long, default_value = "probe_v3,ajimee_jwtd_v2")]
    benches: String,
    #[arg(long, default_value_t = 5)]
    num_beams: usize,
    #[arg(long, default_value_t = 5)]
    num_return: usize,
}

#[derive(Debug, Deserialize)]
struct BenchTrainConfig {
    tokenizer: BenchTokenizerConfig,
    backend: BenchBackendConfig,
}

#[derive(Debug, Deserialize)]
struct BenchTokenizerConfig {
    path: Option<PathBuf>,
    #[serde(default = "default_max_kanji")]
    max_kanji: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BenchBackendConfig {
    hidden_size: usize,
    encoder_layers: usize,
    num_heads: usize,
    ffn_size: usize,
    decoder_layers: usize,
    decoder_heads: usize,
    decoder_ffn_size: usize,
    output_size: usize,
    blank_id: usize,
    max_positions: usize,
}

#[derive(Debug, Deserialize)]
struct TrainerStateLite {
    last_checkpoint: Option<String>,
}

#[derive(Debug, Clone)]
struct BenchItem {
    index: String,
    source: String,
    category: Option<String>,
    context: String,
    reading: String,
    references: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct FailureEntry {
    index: String,
    source: String,
    category: Option<String>,
    context: String,
    reading: String,
    references: Vec<String>,
    candidates: Vec<String>,
    top1: String,
    char_acc_top1: f64,
    exact_match_top1: bool,
    latency_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PatternCount {
    key: String,
    count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct FailurePatterns {
    reference_to_top1: Vec<PatternCount>,
    reading_to_top1: Vec<PatternCount>,
    by_category: Vec<PatternCount>,
    by_source: Vec<PatternCount>,
}

#[derive(Debug, Clone, Serialize)]
struct LatencySummary {
    p50_ms: f64,
    p95_ms: f64,
    mean_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BenchSummary {
    total: usize,
    exact_match_top1: f64,
    exact_match_top5: f64,
    char_acc_top1: f64,
    char_acc_top5: f64,
}

#[derive(Debug, Clone, Serialize)]
struct DecodingSummary {
    num_beams: usize,
    num_return: usize,
}

#[derive(Debug, Clone, Serialize)]
struct BenchResult {
    backend: String,
    model: String,
    params: String,
    bench: String,
    device: String,
    canonical: bool,
    decoding: DecodingSummary,
    total: usize,
    exact_match_top1: f64,
    exact_match_top5: f64,
    char_acc_top1: f64,
    char_acc_top5: f64,
    latency: LatencySummary,
    per_source: BTreeMap<String, BenchSummary>,
    per_category: BTreeMap<String, BenchSummary>,
    failures: Vec<FailureEntry>,
    failure_patterns: FailurePatterns,
    total_time_s: f64,
}

#[derive(Debug, Serialize)]
struct SummaryFile {
    canonical: bool,
    device: String,
    results: Vec<BenchResult>,
}

#[derive(Debug)]
struct Aggregate {
    total: usize,
    exact1: usize,
    exact5: usize,
    char1_sum: f64,
    char5_sum: f64,
}

impl Aggregate {
    fn update(&mut self, exact1: bool, exact5: bool, char1: f64, char5: f64) {
        self.total += 1;
        self.exact1 += usize::from(exact1);
        self.exact5 += usize::from(exact5);
        self.char1_sum += char1;
        self.char5_sum += char5;
    }

    fn summary(&self) -> BenchSummary {
        let total = self.total.max(1) as f64;
        BenchSummary {
            total: self.total,
            exact_match_top1: self.exact1 as f64 / total,
            exact_match_top5: self.exact5 as f64 / total,
            char_acc_top1: self.char1_sum / total,
            char_acc_top5: self.char5_sum / total,
        }
    }
}

struct NativeTchBenchBackend {
    name: String,
    tokenizer: SharedCharTokenizer,
    _vs: VarStore,
    model: CtcNatModel,
    max_positions: usize,
    vocab_size: usize,
}

impl NativeTchBenchBackend {
    fn prepare_input_ids(&self, context: &str, reading: &str) -> Result<Vec<u32>> {
        let mut ids = self.tokenizer.encode_with_special(context, reading);
        if ids.len() > self.max_positions {
            ids.truncate(self.max_positions);
        }
        if let Some((idx, bad)) = ids
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token)| *token as usize >= self.vocab_size)
        {
            bail!(
                "token id {} at position {} exceeds vocab_size {}",
                bad,
                idx,
                self.vocab_size
            );
        }
        Ok(ids)
    }

    fn predict(
        &mut self,
        context: &str,
        reading: &str,
        num_beams: usize,
        num_return: usize,
    ) -> Result<Vec<String>> {
        let ids = self.prepare_input_ids(context, reading)?;
        if ids.is_empty() {
            return Ok(vec![String::new()]);
        }
        let device = Device::Cpu;
        let input: Vec<i64> = ids.iter().map(|v| *v as i64).collect();
        let t = ids.len() as i64;
        let input_ids = Tensor::from_slice(&input).view([1, t]).to_device(device);
        let attention_mask = Tensor::ones([1, t], (Kind::Bool, device));
        let logits = tch::no_grad(|| {
            let enc = self.model.encode(&input_ids, &attention_mask);
            self.model.proposal(&enc, &attention_mask)
        });
        if num_beams <= 1 || num_return <= 1 {
            let argmax = logits.argmax(-1, false).to_device(Device::Cpu);
            let out = collapse_ctc_argmax(&argmax, self.model.blank_id);
            return Ok(out
                .into_iter()
                .take(num_return.max(1))
                .map(|row| self.tokenizer.decode(&row))
                .collect());
        }
        let log_probs = logits
            .log_softmax(-1, Kind::Float)
            .squeeze_dim(0)
            .to_device(Device::Cpu);
        let beam = prefix_beam_search(&log_probs, self.model.blank_id, num_beams, 16);
        Ok(beam
            .into_iter()
            .take(num_return)
            .map(|(tokens, _)| self.tokenizer.decode(&tokens))
            .collect())
    }
}

pub fn main() -> Result<()> {
    let cli = Cli::parse();
    bench_all(
        &cli.config,
        cli.run_dir.as_deref(),
        cli.checkpoint.as_deref(),
        &cli.out_dir,
        cli.markdown.as_deref(),
        cli.model_name.as_deref(),
        &cli.probe_path,
        &cli.ajimee_path,
        &cli.general_path,
        &cli.benches,
        cli.num_beams,
        cli.num_return,
    )
}

fn bench_all(
    config_path: &Path,
    run_dir: Option<&Path>,
    checkpoint: Option<&Path>,
    out_dir: &Path,
    markdown_path: Option<&Path>,
    model_name: Option<&str>,
    probe_path: &Path,
    ajimee_path: &Path,
    general_path: &Path,
    benches_spec: &str,
    num_beams: usize,
    num_return: usize,
) -> Result<()> {
    if num_beams == 0 || num_return == 0 {
        bail!("num_beams and num_return must be positive");
    }
    let selected_benches = parse_bench_list(benches_spec)?;
    let config = load_bench_config(config_path)?;
    let tokenizer = load_tokenizer(&config)?;
    let model_name = model_name.unwrap_or("Suiko-v2-small").to_string();
    let mut benches = Vec::new();
    for bench in selected_benches {
        match bench.as_str() {
            "probe_v3" => benches.push((bench, load_probe_items(probe_path)?)),
            "ajimee_jwtd_v2" => benches.push((bench, load_ajimee_items(ajimee_path)?)),
            "general_dev" => benches.push((bench, load_general_items(general_path)?)),
            other => bail!("unsupported bench `{other}`"),
        }
    }
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("create output dir {}", out_dir.display()))?;

    let checkpoint_source = resolve_checkpoint_source(run_dir, checkpoint)?;
    let (vs, model) = load_model_from_checkpoint(&config.backend, &checkpoint_source)?;
    let params = vs
        .trainable_variables()
        .iter()
        .map(|t| t.numel() as i64)
        .sum::<i64>();
    let params_label = format!("{:.1}M", params as f64 / 1_000_000.0);
    let mut native = NativeTchBenchBackend {
        name: "tch-ctc-nat".to_string(),
        tokenizer,
        _vs: vs,
        model,
        max_positions: config.backend.max_positions,
        vocab_size: config.backend.output_size,
    };

    let mut results = Vec::new();
    for (bench_name, items) in benches {
        eprintln!(
            "[rust-bench] bench={} samples={} model={}",
            bench_name,
            items.len(),
            model_name
        );
        let result = run_bench(
            &mut native,
            &model_name,
            &params_label,
            &bench_name,
            &items,
            num_beams,
            num_return,
        )?;
        let out_path = out_dir.join(format!("{}__{}.json", safe_name(&model_name), bench_name));
        std::fs::write(&out_path, serde_json::to_vec_pretty(&result)?)?;
        results.push(result);
    }
    let summary = SummaryFile {
        canonical: true,
        device: "CPU only".to_string(),
        results,
    };
    std::fs::write(
        out_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    if let Some(path) = markdown_path {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, render_markdown_tables(&summary.results))?;
    }
    Ok(())
}

fn default_max_kanji() -> u32 {
    6000
}

fn load_bench_config(path: &Path) -> Result<BenchTrainConfig> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn load_tokenizer(config: &BenchTrainConfig) -> Result<SharedCharTokenizer> {
    match &config.tokenizer.path {
        Some(path) => SharedCharTokenizer::load(path),
        None => Ok(SharedCharTokenizer::new_default(config.tokenizer.max_kanji)),
    }
}

fn resolve_checkpoint_source(run_dir: Option<&Path>, checkpoint: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = checkpoint {
        return Ok(path.to_path_buf());
    }
    let run_dir = run_dir.context("either --run-dir or --checkpoint is required")?;
    let state_path = run_dir.join("trainer_state.json");
    let bytes =
        std::fs::read(&state_path).with_context(|| format!("read {}", state_path.display()))?;
    let state: TrainerStateLite = serde_json::from_slice(&bytes)?;
    let checkpoint = state
        .last_checkpoint
        .context("trainer_state.last_checkpoint is missing")?;
    let ckpt_path = PathBuf::from(checkpoint);
    let name = ckpt_path
        .file_name()
        .and_then(|s| s.to_str())
        .context("checkpoint file name is not valid utf-8")?;
    let backend_name = name
        .strip_suffix(".ckpt.json")
        .map(|stem| format!("{stem}.backend.json"))
        .context("expected checkpoint path to end with .ckpt.json")?;
    Ok(ckpt_path.with_file_name(backend_name))
}

fn load_model_from_checkpoint(
    config: &BenchBackendConfig,
    checkpoint_source: &Path,
) -> Result<(VarStore, CtcNatModel)> {
    let mut vs = VarStore::new(Device::Cpu);
    let model = CtcNatModel::new(&vs.root(), config)?;
    match checkpoint_source
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
    {
        "pt" => {
            bail!(
                "legacy .pt checkpoints are not supported in the active rust-bench flow; \
                 use Rust-native weights.safetensors assets instead"
            );
        }
        _ => {
            let weights_path = checkpoint_source
                .with_file_name(sibling_name_for(checkpoint_source, "weights.safetensors"));
            load_var_store(&mut vs, &weights_path)?;
        }
    }
    Ok((vs, model))
}

fn collapse_ctc_argmax(argmax: &Tensor, blank_id: i64) -> Vec<Vec<u32>> {
    let sizes = argmax.size();
    let batch = sizes[0] as usize;
    let time = sizes[1] as usize;
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut row = Vec::new();
        let mut prev = None;
        for t in 0..time {
            let token = argmax.int64_value(&[b as i64, t as i64]);
            if token == blank_id {
                prev = None;
                continue;
            }
            if prev == Some(token) {
                continue;
            }
            row.push(token as u32);
            prev = Some(token);
        }
        out.push(row);
    }
    out
}

fn logsumexp_pair(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a.is_sign_negative() {
        return b;
    }
    if b.is_infinite() && b.is_sign_negative() {
        return a;
    }
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

fn prefix_beam_search(
    log_probs: &Tensor,
    blank_id: i64,
    beam_width: usize,
    top_k_per_step: usize,
) -> Vec<(Vec<u32>, f64)> {
    let sizes = log_probs.size();
    let time = sizes[0] as usize;
    let vocab = sizes[1] as usize;
    let mut beam: BTreeMap<Vec<i64>, (f64, f64)> =
        BTreeMap::from([(Vec::new(), (0.0, f64::NEG_INFINITY))]);
    for t in 0..time {
        let blank_logp = log_probs.double_value(&[t as i64, blank_id]);
        let mut top = Vec::with_capacity(vocab);
        for v in 0..vocab {
            top.push((v as i64, log_probs.double_value(&[t as i64, v as i64])));
        }
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        top.truncate(top_k_per_step.min(vocab));
        let mut next_beam: BTreeMap<Vec<i64>, (f64, f64)> = BTreeMap::new();
        let update =
            |map: &mut BTreeMap<Vec<i64>, (f64, f64)>, prefix: Vec<i64>, pb: f64, pnb: f64| {
                let entry = map
                    .entry(prefix)
                    .or_insert((f64::NEG_INFINITY, f64::NEG_INFINITY));
                entry.0 = logsumexp_pair(entry.0, pb);
                entry.1 = logsumexp_pair(entry.1, pnb);
            };
        for (prefix, (pb, pnb)) in &beam {
            update(
                &mut next_beam,
                prefix.clone(),
                logsumexp_pair(*pb, *pnb) + blank_logp,
                f64::NEG_INFINITY,
            );
            for (token, token_logp) in &top {
                if *token == blank_id {
                    continue;
                }
                if prefix.last().copied() == Some(*token) {
                    let mut extended = prefix.clone();
                    extended.push(*token);
                    update(
                        &mut next_beam,
                        extended,
                        f64::NEG_INFINITY,
                        *pb + *token_logp,
                    );
                    update(
                        &mut next_beam,
                        prefix.clone(),
                        f64::NEG_INFINITY,
                        *pnb + *token_logp,
                    );
                } else {
                    let mut extended = prefix.clone();
                    extended.push(*token);
                    update(
                        &mut next_beam,
                        extended,
                        f64::NEG_INFINITY,
                        logsumexp_pair(*pb, *pnb) + *token_logp,
                    );
                }
            }
        }
        let mut scored: Vec<_> = next_beam
            .into_iter()
            .map(|(prefix, (pb, pnb))| (prefix, (pb, pnb), logsumexp_pair(pb, pnb)))
            .collect();
        scored.sort_by(|a, b| b.2.total_cmp(&a.2));
        scored.truncate(beam_width.max(1));
        beam = scored
            .into_iter()
            .map(|(prefix, (pb, pnb), _)| (prefix, (pb, pnb)))
            .collect();
    }
    let mut final_beam: Vec<_> = beam
        .into_iter()
        .map(|(prefix, (pb, pnb))| {
            (
                prefix.into_iter().map(|t| t as u32).collect::<Vec<_>>(),
                logsumexp_pair(pb, pnb),
            )
        })
        .collect();
    final_beam.sort_by(|a, b| b.1.total_cmp(&a.1));
    final_beam
}

fn load_var_store(vs: &mut VarStore, path: &Path) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st =
        SafeTensors::deserialize(&bytes).with_context(|| format!("parse {}", path.display()))?;
    let mut vars = vs.variables();
    let mut seen = std::collections::HashSet::new();
    for name in st.names() {
        seen.insert(name.clone());
        let view = st.tensor(name)?;
        let kind = dtype_to_kind(view.dtype())?;
        let shape: Vec<i64> = view.shape().iter().map(|v| *v as i64).collect();
        let Some(var) = vars.get_mut(name) else {
            bail!("unknown variable `{name}`");
        };
        if var.size() != shape {
            bail!("shape mismatch for `{name}`");
        }
        let new_t = tensor_from_view_typed(view.data(), &shape, kind)?.to_device(var.device());
        tch::no_grad(|| {
            let _ = var.copy_(&new_t);
        });
    }
    for name in vars.keys() {
        if !seen.contains(name) {
            bail!("variable `{name}` missing from {}", path.display());
        }
    }
    Ok(())
}

fn tensor_from_view_typed(bytes: &[u8], shape: &[i64], kind: Kind) -> Result<Tensor> {
    let elem_bytes = match kind {
        Kind::Float | Kind::Int => 4usize,
        Kind::Double | Kind::Int64 => 8,
        Kind::Half | Kind::BFloat16 | Kind::Int16 => 2,
        Kind::Int8 | Kind::Uint8 | Kind::Bool => 1,
        other => bail!("unsupported dtype {:?}", other),
    };
    if bytes.len() % elem_bytes != 0 {
        bail!("payload length mismatch");
    }
    let expected: i64 = shape.iter().product();
    if expected as usize != bytes.len() / elem_bytes {
        bail!("payload numel does not match shape");
    }
    let dst = Tensor::zeros(shape, (kind, Device::Cpu));
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.data_ptr() as *mut u8, bytes.len());
    }
    Ok(dst)
}

fn dtype_to_kind(dtype: Dtype) -> Result<Kind> {
    Ok(match dtype {
        Dtype::F32 => Kind::Float,
        Dtype::F64 => Kind::Double,
        Dtype::F16 => Kind::Half,
        Dtype::BF16 => Kind::BFloat16,
        Dtype::I64 => Kind::Int64,
        Dtype::I32 => Kind::Int,
        Dtype::I16 => Kind::Int16,
        Dtype::I8 => Kind::Int8,
        Dtype::U8 => Kind::Uint8,
        Dtype::BOOL => Kind::Bool,
        other => bail!("unsupported safetensors dtype: {:?}", other),
    })
}

fn sibling_name_for(anchor: &Path, suffix: &str) -> String {
    let name = anchor.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let stem = name
        .strip_suffix(".backend.json")
        .or_else(|| name.strip_suffix(".json"))
        .unwrap_or(name);
    format!("{stem}.{suffix}")
}

fn load_probe_items(path: &Path) -> Result<Vec<BenchItem>> {
    #[derive(Deserialize)]
    struct ProbeRow {
        #[serde(default)]
        index: serde_json::Value,
        category: String,
        #[serde(default)]
        context_text: String,
        input: String,
        expected_output: Vec<String>,
    }
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let rows: Vec<ProbeRow> =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    Ok(rows
        .into_iter()
        .map(|row| BenchItem {
            index: json_index_to_string(&row.index),
            source: "probe".to_string(),
            category: Some(row.category),
            context: row.context_text,
            reading: kata_to_hira(&row.input),
            references: row.expected_output,
        })
        .collect())
}

fn load_ajimee_items(path: &Path) -> Result<Vec<BenchItem>> {
    #[derive(Deserialize)]
    struct AjimeeRow {
        #[serde(default)]
        index: serde_json::Value,
        #[serde(default)]
        context_text: String,
        input: String,
        expected_output: Vec<String>,
    }
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let rows: Vec<AjimeeRow> =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    Ok(rows
        .into_iter()
        .map(|row| BenchItem {
            index: json_index_to_string(&row.index),
            source: "ajimee_jwtd".to_string(),
            category: None,
            context: row.context_text,
            reading: kata_to_hira(&row.input),
            references: row.expected_output,
        })
        .collect())
}

fn load_general_items(path: &Path) -> Result<Vec<BenchItem>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut items = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("parse {}:{}", path.display(), line_no + 1))?;
        let reading = row
            .get("reading")
            .and_then(|v| v.as_str())
            .context("general row missing reading")?;
        let surface = row
            .get("surface")
            .and_then(|v| v.as_str())
            .context("general row missing surface")?;
        let context = row
            .get("context")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let source = row
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("general")
            .to_string();
        items.push(BenchItem {
            index: line_no.to_string(),
            source,
            category: None,
            context,
            reading: reading.to_string(),
            references: vec![surface.to_string()],
        });
    }
    Ok(items)
}

fn parse_bench_list(spec: &str) -> Result<Vec<String>> {
    let mut benches = Vec::new();
    for raw in spec.split(',') {
        let bench = raw.trim();
        if bench.is_empty() {
            continue;
        }
        benches.push(bench.to_string());
    }
    if benches.is_empty() {
        bail!("at least one bench must be selected");
    }
    Ok(benches)
}

fn json_index_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn kata_to_hira(text: &str) -> String {
    text.chars()
        .map(|ch| {
            let cp = ch as u32;
            if (0x30A1..=0x30F6).contains(&cp) {
                char::from_u32(cp - 0x60).unwrap_or(ch)
            } else {
                ch
            }
        })
        .collect()
}

fn run_bench(
    backend: &mut NativeTchBenchBackend,
    model_name: &str,
    params_label: &str,
    bench_name: &str,
    items: &[BenchItem],
    num_beams: usize,
    num_return: usize,
) -> Result<BenchResult> {
    let start = Instant::now();
    let mut latencies = Vec::with_capacity(items.len());
    let mut overall = Aggregate {
        total: 0,
        exact1: 0,
        exact5: 0,
        char1_sum: 0.0,
        char5_sum: 0.0,
    };
    let mut per_source: BTreeMap<String, Aggregate> = BTreeMap::new();
    let mut per_category: BTreeMap<String, Aggregate> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut ref_top1_counts = BTreeMap::new();
    let mut reading_top1_counts = BTreeMap::new();
    let mut category_counts = BTreeMap::new();
    let mut source_counts = BTreeMap::new();

    for (i, item) in items.iter().enumerate() {
        let t0 = Instant::now();
        let candidates = backend.predict(&item.context, &item.reading, num_beams, num_return)?;
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
        latencies.push(latency_ms);
        let top1 = candidates.first().cloned().unwrap_or_default();
        let exact1 = item.references.iter().any(|r| r == &top1);
        let exact5 = candidates
            .iter()
            .take(num_return)
            .any(|cand| item.references.iter().any(|r| r == cand));
        let char1 = best_char_acc(&item.references, std::iter::once(top1.as_str()));
        let char5 = best_char_acc(
            &item.references,
            candidates.iter().take(num_return).map(String::as_str),
        );
        overall.update(exact1, exact5, char1, char5);
        per_source
            .entry(item.source.clone())
            .or_insert(Aggregate {
                total: 0,
                exact1: 0,
                exact5: 0,
                char1_sum: 0.0,
                char5_sum: 0.0,
            })
            .update(exact1, exact5, char1, char5);
        if let Some(category) = &item.category {
            per_category
                .entry(category.clone())
                .or_insert(Aggregate {
                    total: 0,
                    exact1: 0,
                    exact5: 0,
                    char1_sum: 0.0,
                    char5_sum: 0.0,
                })
                .update(exact1, exact5, char1, char5);
        }
        if !exact1 {
            let first_ref = item.references.first().cloned().unwrap_or_default();
            failures.push(FailureEntry {
                index: if item.index.is_empty() {
                    i.to_string()
                } else {
                    item.index.clone()
                },
                source: item.source.clone(),
                category: item.category.clone(),
                context: item.context.clone(),
                reading: item.reading.clone(),
                references: item.references.clone(),
                candidates: candidates.clone(),
                top1: top1.clone(),
                char_acc_top1: char1,
                exact_match_top1: exact1,
                latency_ms,
            });
            *ref_top1_counts
                .entry(format!("{} => {}", first_ref, top1))
                .or_default() += 1;
            *reading_top1_counts
                .entry(format!("{} => {}", item.reading, top1))
                .or_default() += 1;
            if let Some(category) = &item.category {
                *category_counts.entry(category.clone()).or_default() += 1;
            }
            *source_counts.entry(item.source.clone()).or_default() += 1;
        }
    }

    Ok(BenchResult {
        backend: backend.name.clone(),
        model: model_name.to_string(),
        params: params_label.to_string(),
        bench: bench_name.to_string(),
        device: "CPU only".to_string(),
        canonical: true,
        decoding: DecodingSummary {
            num_beams,
            num_return,
        },
        total: overall.total,
        exact_match_top1: overall.summary().exact_match_top1,
        exact_match_top5: overall.summary().exact_match_top5,
        char_acc_top1: overall.summary().char_acc_top1,
        char_acc_top5: overall.summary().char_acc_top5,
        latency: summarize_latencies(&latencies),
        per_source: per_source
            .into_iter()
            .map(|(k, v)| (k, v.summary()))
            .collect(),
        per_category: per_category
            .into_iter()
            .map(|(k, v)| (k, v.summary()))
            .collect(),
        failures,
        failure_patterns: FailurePatterns {
            reference_to_top1: sort_pattern_counts(ref_top1_counts),
            reading_to_top1: sort_pattern_counts(reading_top1_counts),
            by_category: sort_pattern_counts(category_counts),
            by_source: sort_pattern_counts(source_counts),
        },
        total_time_s: start.elapsed().as_secs_f64(),
    })
}

fn sort_pattern_counts(counts: BTreeMap<String, usize>) -> Vec<PatternCount> {
    let mut items: Vec<_> = counts
        .into_iter()
        .map(|(key, count)| PatternCount { key, count })
        .collect();
    items.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.key.cmp(&b.key)));
    items
}

fn summarize_latencies(latencies: &[f64]) -> LatencySummary {
    if latencies.is_empty() {
        return LatencySummary {
            p50_ms: 0.0,
            p95_ms: 0.0,
            mean_ms: 0.0,
        };
    }
    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let p50 = sorted[n / 2];
    let p95 = sorted[((n as f64 * 0.95).floor() as usize).min(n - 1)];
    let mean = sorted.iter().sum::<f64>() / n as f64;
    LatencySummary {
        p50_ms: p50,
        p95_ms: p95,
        mean_ms: mean,
    }
}

fn best_char_acc<'a>(references: &[String], candidates: impl Iterator<Item = &'a str>) -> f64 {
    let mut best = 0.0_f64;
    for candidate in candidates {
        for reference in references {
            best = best.max(character_accuracy(reference, candidate));
        }
    }
    best
}

fn character_accuracy(reference: &str, prediction: &str) -> f64 {
    let ref_len = reference.chars().count();
    if ref_len == 0 {
        return f64::from(prediction.is_empty());
    }
    let distance = edit_distance(reference, prediction);
    ((ref_len as isize - distance as isize).max(0) as f64) / ref_len as f64
}

fn edit_distance(reference: &str, prediction: &str) -> usize {
    let ref_chars: Vec<char> = reference.chars().collect();
    let pred_chars: Vec<char> = prediction.chars().collect();
    let mut prev: Vec<usize> = (0..=pred_chars.len()).collect();
    let mut curr = vec![0usize; pred_chars.len() + 1];
    for (i, rc) in ref_chars.iter().enumerate() {
        curr[0] = i + 1;
        for (j, pc) in pred_chars.iter().enumerate() {
            let cost = usize::from(rc != pc);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[pred_chars.len()]
}

fn render_markdown_tables(results: &[BenchResult]) -> String {
    let mut out = String::new();
    out.push_str("## probe_v3\n\n");
    out.push_str("| model | params | EM1 | EM5 | CharAcc | p50 ms |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|\n");
    for row in results.iter().filter(|r| r.bench == "probe_v3") {
        out.push_str(&format!(
            "| {} | {} | {:.3} | {:.3} | {:.3} | {:.0} |\n",
            row.model,
            row.params,
            row.exact_match_top1,
            row.exact_match_top5,
            row.char_acc_top1,
            row.latency.p50_ms
        ));
    }
    out.push_str("\n## AJIMEE JWTD_v2\n\n");
    out.push_str("| model | EM1 | EM5 | CharAcc | p50 ms |\n");
    out.push_str("|---|---:|---:|---:|---:|\n");
    for row in results.iter().filter(|r| r.bench == "ajimee_jwtd_v2") {
        out.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.0} |\n",
            row.model,
            row.exact_match_top1,
            row.exact_match_top5,
            row.char_acc_top1,
            row.latency.p50_ms
        ));
    }
    out.push_str("\n## probe_v3 category 別 EM1\n\n");
    out.push_str("| model | edge | general | homo | names | numeric | particle | tech |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for row in results.iter().filter(|r| r.bench == "probe_v3") {
        let homo = row
            .per_category
            .get("homo")
            .or_else(|| row.per_category.get("homophone"));
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            row.model,
            fmt_category(row.per_category.get("edge")),
            fmt_category(row.per_category.get("general")),
            fmt_category(homo),
            fmt_category(row.per_category.get("names")),
            fmt_category(row.per_category.get("numeric")),
            fmt_category(row.per_category.get("particle")),
            fmt_category(row.per_category.get("tech")),
        ));
    }
    out
}

fn fmt_category(summary: Option<&BenchSummary>) -> String {
    match summary {
        Some(s) => format!("{:.3}", s.exact_match_top1),
        None => "-".to_string(),
    }
}

fn safe_name(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn katakana_is_normalized_to_hiragana() {
        assert_eq!(kata_to_hira("トウキョウ"), "とうきょう");
    }

    #[test]
    fn char_accuracy_matches_exact_for_identical_strings() {
        assert_eq!(character_accuracy("学校", "学校"), 1.0);
    }

    #[test]
    fn markdown_renderer_emits_expected_sections() {
        let result = BenchResult {
            backend: "tch-ctc-nat".to_string(),
            model: "Suiko-v2-small".to_string(),
            params: "30.0M".to_string(),
            bench: "probe_v3".to_string(),
            device: "CPU only".to_string(),
            canonical: true,
            decoding: DecodingSummary {
                num_beams: 5,
                num_return: 5,
            },
            total: 1,
            exact_match_top1: 0.5,
            exact_match_top5: 0.7,
            char_acc_top1: 0.8,
            char_acc_top5: 0.9,
            latency: LatencySummary {
                p50_ms: 12.0,
                p95_ms: 13.0,
                mean_ms: 12.5,
            },
            per_source: BTreeMap::new(),
            per_category: BTreeMap::from([
                (
                    "general".to_string(),
                    BenchSummary {
                        total: 1,
                        exact_match_top1: 0.5,
                        exact_match_top5: 0.7,
                        char_acc_top1: 0.8,
                        char_acc_top5: 0.9,
                    },
                ),
                (
                    "homophone".to_string(),
                    BenchSummary {
                        total: 1,
                        exact_match_top1: 0.4,
                        exact_match_top5: 0.6,
                        char_acc_top1: 0.7,
                        char_acc_top5: 0.8,
                    },
                ),
            ]),
            failures: Vec::new(),
            failure_patterns: FailurePatterns {
                reference_to_top1: Vec::new(),
                reading_to_top1: Vec::new(),
                by_category: Vec::new(),
                by_source: Vec::new(),
            },
            total_time_s: 1.0,
        };
        let markdown = render_markdown_tables(&[result]);
        assert!(markdown.contains("## probe_v3"));
        assert!(markdown.contains("## AJIMEE JWTD_v2"));
        assert!(markdown.contains("## probe_v3 category 別 EM1"));
        assert!(markdown.contains("| Suiko-v2-small | - | 0.500 | 0.400 | - | - | - | - |"));
    }
}
