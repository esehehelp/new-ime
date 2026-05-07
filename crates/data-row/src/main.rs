//! Row-level inspection / edit TUI for large kana-kanji JSONL mixes.
//!
//! Subcommands:
//!   index <jsonl>           — build <jsonl>.idx (binary u64 array of row
//!                             byte offsets). One-time, ~5 min for 23 GiB.
//!   stats <jsonl>           — per-source row count + length histogram.
//!   audit <jsonl> [--start N]
//!                           — TUI: scroll rows, accept / delete / edit,
//!                             write actions to <jsonl>.audit.jsonl on
//!                             save. data-mix can later --apply-audit.
//!
//! Audit log entries (one JSON object per line):
//!     {"row_id": N, "action": "delete"}
//!     {"row_id": N, "action": "replace", "new_row": {...}}
//!
//! "accept" produces no log entry — a row not present in the audit log is
//! kept verbatim by the mix consumer.

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use memmap2::Mmap;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{stdout, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Row-level JSONL inspect / audit TUI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Build a .idx file (u64 byte offsets per row) so the TUI can seek
    /// to any row in O(1).
    Index { jsonl: PathBuf },

    /// Print per-source row counts + reading-length histogram.
    Stats { jsonl: PathBuf },

    /// Open the row-level audit TUI.
    Audit {
        jsonl: PathBuf,
        /// Initial row id. Default: resume at (last audited row + 1), or 0
        /// if no prior audit log exists.
        #[arg(long)]
        start: Option<u64>,
        /// Review mode — start at the *first* replace edit instead of
        /// resuming at the next un-audited row. For checking edits a
        /// previous pass (manual or agent-driven) wrote to the audit log.
        #[arg(long, default_value_t = false)]
        review: bool,
    },
}

/// LLM connection config, sourced from env vars at startup. The TUI
/// presses `l` on a row → POSTs an OpenAI-compat chat completion request
/// to the configured endpoint and treats the response as a suggested
/// row replacement. HF Inference Endpoints expose this exact API shape
/// when "Container Type" is OpenAI-compatible.
struct LlmConfig {
    endpoint: String, // full URL incl. /v1/chat/completions
    token: String,
    model: String,
}

impl LlmConfig {
    fn from_env() -> Option<Self> {
        let endpoint = std::env::var("DATA_ROW_LLM_ENDPOINT").ok()?;
        let token = std::env::var("DATA_ROW_LLM_TOKEN").ok().unwrap_or_default();
        let model = std::env::var("DATA_ROW_LLM_MODEL")
            .unwrap_or_else(|_| "llm-jp/llm-jp-4-8b-thinking".into());
        Some(Self { endpoint, token, model })
    }
}

/// Whitelist of row fields that the LLM may inspect / suggest changes to.
/// Metadata fields (source, sentence_id, span_bunsetsu, *_id) are kept
/// out of the prompt — the model can't audit them, they consume tokens,
/// and exposing source tags has shown up as a hallucination trigger
/// (e.g. tatoeba metadata correlating with Chinese-character output on
/// some Qwen variants).
const LLM_FIELD_WHITELIST: &[&str] = &[
    "reading",
    "surface",
    "context",
    "left_context_surface",
    "left_context_reading",
];

fn project_for_llm(row: &Value) -> Value {
    let mut out = serde_json::Map::new();
    if let Value::Object(map) = row {
        for k in LLM_FIELD_WHITELIST {
            if let Some(v) = map.get(*k) {
                out.insert((*k).into(), v.clone());
            }
        }
    }
    Value::Object(out)
}

/// Apply a whitelist-projected suggestion back onto the full row,
/// preserving every metadata field the LLM didn't see.
fn merge_suggestion(original: &Value, suggestion: &Value) -> Value {
    let mut merged = original.clone();
    if let (Value::Object(merged_map), Value::Object(sugg_map)) = (&mut merged, suggestion) {
        for k in LLM_FIELD_WHITELIST {
            if let Some(v) = sugg_map.get(*k) {
                merged_map.insert((*k).into(), v.clone());
            }
        }
    }
    merged
}

/// Token counts + cost extracted from a chat-completion response.
#[derive(Default, Clone, Copy, Debug)]
struct LlmUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    /// Cost reported directly by the provider (DeepInfra fills this).
    /// `None` means the caller should fall back to its rate estimate.
    estimated_cost: Option<f64>,
}

impl LlmUsage {
    fn cost(&self, rate_in_per_m: f64, rate_out_per_m: f64) -> f64 {
        if let Some(c) = self.estimated_cost {
            return c;
        }
        (self.prompt_tokens as f64) * rate_in_per_m / 1_000_000.0
            + (self.completion_tokens as f64) * rate_out_per_m / 1_000_000.0
    }
}

fn parse_usage(resp: &Value) -> LlmUsage {
    let usage = resp.get("usage");
    LlmUsage {
        prompt_tokens: usage
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        completion_tokens: usage
            .and_then(|u| u.get("completion_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        estimated_cost: usage
            .and_then(|u| u.get("estimated_cost"))
            .and_then(|v| v.as_f64()),
    }
}

fn llm_suggest(cfg: &LlmConfig, row: &Value) -> Result<Option<Value>> {
    llm_suggest_with_usage(cfg, row).map(|(opt, _)| opt)
}

fn llm_suggest_with_usage(cfg: &LlmConfig, row: &Value) -> Result<(Option<Value>, LlmUsage)> {
    let projected = project_for_llm(row);
    let prompt = format!(
        "日本語かな漢字IME学習データ1行 (JSON) を監査し、より質の高い学習サンプルになるよう**補強**する。以下のいずれかに該当すれば修正版JSONを1行で返す。該当なしなら null のみ。\n\n\
         修正方針 (短縮ではなく**充実**させる方向):\n\
         1. reading が surface のかな読みと一致しない → 正しい全文 reading に拡張\n\
         2. surface が断片的・途中で切れている → 文として完結する自然な surface に拡張し、reading も対応して拡張\n\
         3. left_context_surface / context が短すぎ・無関係・右文脈 → surface の前文として自然に成立する日本語で**充実**させる (10-20 文字程度を目安、空にするのは最終手段)\n\
         4. left_context_reading が無い場合は left_context_surface に対応する全文かなを生成\n\n\
         例1 入力 {{\"reading\":\"きょう\",\"surface\":\"今日の天気は\"}}\n\
              → 出力 {{\"reading\":\"きょうのてんきははれですね\",\"surface\":\"今日の天気は晴れですね\"}}\n\
         例2 入力 {{\"reading\":\"わたしは\",\"surface\":\"私は\",\"left_context_surface\":\"明日もよろしく\"}}\n\
              → 出力 {{\"reading\":\"わたしは\",\"surface\":\"私は\",\"left_context_surface\":\"会議の冒頭で「\",\"left_context_reading\":\"かいぎのぼうとうで「\"}}\n\
         例3 入力 {{\"reading\":\"はい、わかりました\",\"surface\":\"はい、分かりました\",\"context\":\"課長の指示について部下が答えた。\"}}\n\
              → 出力 null\n\n\
         説明・コードフェンス禁止、出力は JSON 1 行か null のみ。\n\n\
         入力:\n{}\n\n出力:",
        serde_json::to_string(&projected)?
    );

    // Thinking handling per provider (one body covers both — unknown
    // fields are silently ignored by OpenAI-compat servers):
    //   * DeepSeek V4-Flash: thinking is required for usable JP audit
    //     quality. Measured: thinking-disabled returns null even on
    //     truncated reading; thinking-enabled fixes correctly. Cost
    //     ~$0.0006/row, of which most is reasoning_tokens.
    //   * Qwen3.6 (via DeepInfra / vLLM / HF): user-side preference is
    //     thinking-off. `chat_template_kwargs.enable_thinking=false` is
    //     the official toggle for the Qwen3.6 line (no /no_think soft
    //     switch). DeepSeek's server ignores this field.
    let body = serde_json::json!({
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": "日本語かな漢字IMEコーパスの監査+補強アシスタント。reading↔surface 整合を保ちつつ、断片的・短すぎる行は自然な日本語で内容を充実させる方向で修正する。"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": false},
    });

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;
    let mut req = client.post(&cfg.endpoint).json(&body);
    if !cfg.token.is_empty() {
        req = req.bearer_auth(&cfg.token);
    }
    let resp: Value = req.send()?.error_for_status()?.json()?;
    let usage = parse_usage(&resp);
    let content = resp
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let stripped = content
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    if stripped.eq_ignore_ascii_case("null") || stripped.is_empty() {
        return Ok((None, usage));
    }
    let suggestion: Value = serde_json::from_str(stripped)
        .with_context(|| format!("LLM response was not valid JSON: {stripped}"))?;
    let merged = merge_suggestion(row, &suggestion);
    if &merged == row {
        return Ok((None, usage));
    }
    Ok((Some(merged), usage))
}

/// Normalize fullwidth ASCII punctuation to ideographic equivalents
/// in every string value of the JSON tree. The input host writes 「，．」
/// (FULLWIDTH COMMA / FULL STOP, U+FF0C / U+FF0E) when the user types
/// 全角英数 punctuation, but the corpus convention is 和文記号
/// 「、。」(IDEOGRAPHIC COMMA / FULL STOP, U+3001 / U+3002). Apply on
/// edit so the dataset stays consistent.
fn normalize_jp_punct(v: &mut Value) {
    match v {
        Value::String(s) => {
            if s.contains('\u{FF0C}') || s.contains('\u{FF0E}') {
                *s = s.replace('\u{FF0C}', "\u{3001}").replace('\u{FF0E}', "\u{3002}");
            }
        }
        Value::Array(arr) => {
            for x in arr.iter_mut() {
                normalize_jp_punct(x);
            }
        }
        Value::Object(map) => {
            for (_, x) in map.iter_mut() {
                normalize_jp_punct(x);
            }
        }
        _ => {}
    }
}

fn idx_path(jsonl: &Path) -> PathBuf {
    let mut p = jsonl.to_path_buf();
    let new_ext = match p.extension().and_then(|s| s.to_str()) {
        Some(e) => format!("{e}.idx"),
        None => "idx".to_string(),
    };
    p.set_extension(new_ext);
    p
}

fn audit_log_path(jsonl: &Path) -> PathBuf {
    let mut p = jsonl.to_path_buf();
    let new_ext = match p.extension().and_then(|s| s.to_str()) {
        Some(e) => format!("{e}.audit.jsonl"),
        None => "audit.jsonl".to_string(),
    };
    p.set_extension(new_ext);
    p
}

/// Scan the JSONL once and write (offset, len) pairs as packed u64 pairs to .idx.
/// We only store offsets (not lengths) for v0; line ends are recoverable by
/// reading until '\n'.
fn build_index(jsonl: &Path) -> Result<()> {
    let started = Instant::now();
    let in_path = jsonl;
    let out_path = idx_path(jsonl);
    let total_size = std::fs::metadata(in_path)?.len();

    let f = File::open(in_path).with_context(|| format!("open {}", in_path.display()))?;
    let mmap = unsafe { Mmap::map(&f)? };
    let bytes: &[u8] = &mmap;

    let mut out = BufWriter::new(File::create(&out_path)?);
    let mut count: u64 = 0;
    let mut offset: u64 = 0;
    out.write_all(&offset.to_le_bytes())?;
    count += 1;
    let mut last_report = Instant::now();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\n' {
            let next = (i + 1) as u64;
            if next < total_size {
                out.write_all(&next.to_le_bytes())?;
                count += 1;
                offset = next;
            }
        }
        if last_report.elapsed().as_secs() >= 5 {
            eprintln!(
                "[index] {:.1}% ({}/{} bytes, rows={})",
                100.0 * (i as f64) / (total_size as f64),
                i,
                total_size,
                count
            );
            last_report = Instant::now();
        }
    }
    out.flush()?;
    eprintln!(
        "[index] done: rows={} -> {} ({:.1}s)",
        count,
        out_path.display(),
        started.elapsed().as_secs_f32()
    );
    Ok(())
}

fn load_index(jsonl: &Path) -> Result<Vec<u64>> {
    let p = idx_path(jsonl);
    if !p.exists() {
        return Err(anyhow!(
            "index not found at {}; run `data-row index {}` first",
            p.display(),
            jsonl.display()
        ));
    }
    let bytes = std::fs::read(&p).with_context(|| format!("read {}", p.display()))?;
    if bytes.len() % 8 != 0 {
        return Err(anyhow!(
            "index file size {} is not a multiple of 8",
            bytes.len()
        ));
    }
    let mut offsets = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        offsets.push(u64::from_le_bytes(buf));
    }
    Ok(offsets)
}

fn read_row(jsonl: &Path, offset: u64) -> Result<String> {
    let mut f = File::open(jsonl).with_context(|| format!("open {}", jsonl.display()))?;
    f.seek(SeekFrom::Start(offset))?;
    let mut br = BufReader::new(f);
    let mut line = String::new();
    let n = br.read_line(&mut line)?;
    if n == 0 {
        return Err(anyhow!("EOF at offset {}", offset));
    }
    if line.ends_with('\n') {
        line.pop();
        if line.ends_with('\r') {
            line.pop();
        }
    }
    Ok(line)
}

fn cmd_stats(jsonl: &Path) -> Result<()> {
    let f = File::open(jsonl)?;
    let br = BufReader::new(f);
    let mut by_source: HashMap<String, u64> = HashMap::new();
    let mut reading_lens: HashMap<usize, u64> = HashMap::new();
    let mut surface_lens: HashMap<usize, u64> = HashMap::new();
    let mut total: u64 = 0;
    let mut started = Instant::now();
    for line in br.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        total += 1;
        match serde_json::from_str::<Value>(&line) {
            Ok(v) => {
                let src = v
                    .get("source")
                    .and_then(|s| s.as_str())
                    .unwrap_or("?")
                    .to_string();
                *by_source.entry(src).or_default() += 1;
                if let Some(s) = v.get("reading").and_then(|s| s.as_str()) {
                    *reading_lens.entry(s.chars().count()).or_default() += 1;
                }
                if let Some(s) = v.get("surface").and_then(|s| s.as_str()) {
                    *surface_lens.entry(s.chars().count()).or_default() += 1;
                }
            }
            Err(_) => {}
        }
        if started.elapsed().as_secs() >= 5 {
            eprintln!("[stats] processed {} rows", total);
            started = Instant::now();
        }
    }

    println!("=== sources (n={}) ===", total);
    let mut srcs: Vec<_> = by_source.into_iter().collect();
    srcs.sort_by(|a, b| b.1.cmp(&a.1));
    for (s, n) in srcs.iter() {
        println!("  {:<28} {:>10}  {:.2}%", s, n, 100.0 * (*n as f64) / (total as f64));
    }

    println!("\n=== reading length histogram (chars) ===");
    let mut lens: Vec<_> = reading_lens.iter().collect();
    lens.sort_by_key(|x| x.0);
    for (l, n) in lens.iter() {
        let bar = "#".repeat(((*n.clone() as f64 / total as f64) * 200.0) as usize);
        println!("  {:>3}: {:>10}  {bar}", l, n);
    }
    Ok(())
}

#[derive(Default)]
struct AuditState {
    deleted: HashMap<u64, ()>,
    replaced: HashMap<u64, String>, // row_id -> new JSON line
    /// Rows the user explicitly reviewed and kept verbatim (`a`). Has no
    /// effect on the mix output but lets the TUI mark already-reviewed
    /// rows and resume at the last reviewed position next session.
    accepted: HashMap<u64, ()>,
    log_path: PathBuf,
    saved_count: usize,
}

impl AuditState {
    fn delete(&mut self, row_id: u64) {
        self.replaced.remove(&row_id);
        self.accepted.remove(&row_id);
        self.deleted.insert(row_id, ());
    }
    fn replace(&mut self, row_id: u64, new_line: String) {
        self.deleted.remove(&row_id);
        self.accepted.remove(&row_id);
        self.replaced.insert(row_id, new_line);
    }
    fn accept(&mut self, row_id: u64) {
        // delete / replace は accept より強い決定なので上書きしない。
        if self.deleted.contains_key(&row_id) || self.replaced.contains_key(&row_id) {
            return;
        }
        self.accepted.insert(row_id, ());
    }
    /// Highest row_id present in any of the three maps. Used to resume.
    fn last_audited(&self) -> Option<u64> {
        let mut max: Option<u64> = None;
        for &k in self
            .deleted
            .keys()
            .chain(self.replaced.keys())
            .chain(self.accepted.keys())
        {
            max = Some(max.map_or(k, |c| c.max(k)));
        }
        max
    }
    /// Read an existing audit log back into memory. Missing file = empty
    /// state. Malformed lines are skipped with a counter.
    fn load(&mut self) -> Result<usize> {
        if !self.log_path.exists() {
            return Ok(0);
        }
        let f = File::open(&self.log_path)
            .with_context(|| format!("open {}", self.log_path.display()))?;
        let mut bad = 0usize;
        for line in BufReader::new(f).lines() {
            let line = match line {
                Ok(l) if !l.trim().is_empty() => l,
                _ => continue,
            };
            let v: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => {
                    bad += 1;
                    continue;
                }
            };
            let row_id = match v.get("row_id").and_then(|x| x.as_u64()) {
                Some(n) => n,
                None => {
                    bad += 1;
                    continue;
                }
            };
            match v.get("action").and_then(|x| x.as_str()) {
                Some("delete") => {
                    self.deleted.insert(row_id, ());
                }
                Some("replace") => {
                    if let Some(nr) = v.get("new_row") {
                        self.replaced.insert(row_id, nr.to_string());
                    } else {
                        bad += 1;
                    }
                }
                Some("accept") => {
                    self.accepted.insert(row_id, ());
                }
                _ => bad += 1,
            }
        }
        Ok(bad)
    }

    fn save(&mut self) -> Result<()> {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.log_path)?;
        let mut total = 0;
        let mut keys: Vec<u64> = self
            .deleted
            .keys()
            .copied()
            .chain(self.replaced.keys().copied())
            .chain(self.accepted.keys().copied())
            .collect();
        keys.sort_unstable();
        keys.dedup();
        for row_id in keys {
            let entry = if self.replaced.contains_key(&row_id) {
                let new_line = self.replaced.get(&row_id).unwrap();
                let new_row: Value = serde_json::from_str(new_line)?;
                serde_json::json!({
                    "row_id": row_id,
                    "action": "replace",
                    "new_row": new_row,
                })
            } else if self.deleted.contains_key(&row_id) {
                serde_json::json!({"row_id": row_id, "action": "delete"})
            } else {
                serde_json::json!({"row_id": row_id, "action": "accept"})
            };
            writeln!(f, "{}", serde_json::to_string(&entry)?)?;
            total += 1;
        }
        f.flush()?;
        self.saved_count = total;
        Ok(())
    }
}

/// Modal input state. Normal-mode keys differ from text-edit keys, so we
/// dispatch on this rather than scattering `if pending_*` checks.
enum InputMode {
    Normal,
    Goto(String),
    /// Showing a numbered list of editable string fields, awaiting 0–9 (or
    /// Esc) to pick one to edit.
    PickField { fields: Vec<String> },
    /// Inline text edit on a single string field. Buffer is `Vec<char>` so
    /// the cursor index is character-based and handles JP multibyte cleanly.
    EditField { key: String, buffer: Vec<char>, cursor: usize },
    /// Reviewing an LLM suggestion: body shows diff(parsed, suggestion),
    /// `y` accepts (commits replace), `n` rejects.
    ReviewSuggestion,
}

impl Default for InputMode {
    fn default() -> Self { InputMode::Normal }
}

#[derive(Default)]
struct UiState {
    cur: u64,
    total: u64,
    raw_line: String,
    parsed: Option<Value>,
    /// Original row content as read directly from the source JSONL — only
    /// populated when the current row has a pending replace edit so the
    /// diff view can show "what changed" without a re-read.
    original: Option<Value>,
    /// Latest LLM suggestion for the current row. Set by pressing `l`,
    /// reviewed via the ReviewSuggestion mode.
    pending_suggestion: Option<Value>,
    parse_err: Option<String>,
    status: String,
    mode: InputMode,
    show_diff: bool,
    quit: bool,
}

/// Top-level string fields of the parsed JSON object — these are what the
/// user can pick to edit. Order matches `pretty_row`'s known list (so the
/// numbered picker matches what's on screen) plus any extras at the end.
fn editable_fields(parsed: &Value) -> Vec<String> {
    let known = [
        "reading",
        "surface",
        "context",
        "left_context_surface",
        "left_context_reading",
    ];
    let mut out: Vec<String> = Vec::new();
    if let Value::Object(map) = parsed {
        for k in known.iter() {
            if let Some(Value::String(_)) = map.get(*k) {
                out.push((*k).into());
            }
        }
        for (k, v) in map.iter() {
            if known.contains(&k.as_str()) {
                continue;
            }
            if matches!(v, Value::String(_)) {
                out.push(k.clone());
            }
        }
    }
    out
}

fn pretty_row(parsed: &Value, mode: &InputMode) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    let known = [
        "reading",
        "surface",
        "context",
        "left_context_surface",
        "left_context_reading",
        "source",
        "span_bunsetsu",
        "sentence_id",
        "writer_id",
        "domain_id",
        "source_id",
    ];
    let pick_fields: Option<&[String]> = match mode {
        InputMode::PickField { fields } => Some(fields.as_slice()),
        _ => None,
    };
    let editing: Option<(&str, &[char], usize)> = match mode {
        InputMode::EditField { key, buffer, cursor } => Some((key.as_str(), buffer.as_slice(), *cursor)),
        _ => None,
    };
    let mut emit = |k: &str, v: &Value, key_style: Style| {
        // Picker prefix: " [N] " if this key is in the picker list, else 5 spaces.
        let prefix = if let Some(list) = pick_fields {
            list.iter()
                .position(|x| x == k)
                .map(|i| format!(" [{}] ", i + 1))
                .unwrap_or_else(|| "     ".into())
        } else {
            "     ".into()
        };
        // Value rendering: inline-edit buffer with cursor marker if this is
        // the field being edited; else the JSON value verbatim.
        let key_label = format!("{prefix}{:<22}", k);
        if let Some((ek, buf, cur)) = editing {
            if ek == k {
                let before: String = buf.iter().take(cur).collect();
                let after: String = buf.iter().skip(cur).collect();
                out.push(Line::from(vec![
                    Span::styled(key_label, key_style.add_modifier(Modifier::REVERSED)),
                    Span::raw(before),
                    Span::styled("█", Style::default().fg(Color::Yellow).add_modifier(Modifier::SLOW_BLINK)),
                    Span::raw(after),
                ]));
                return;
            }
        }
        let val = match v {
            Value::String(s) => s.clone(),
            _ => v.to_string(),
        };
        out.push(Line::from(vec![
            Span::styled(key_label, key_style),
            Span::raw(val),
        ]));
    };
    for k in known.iter() {
        if let Some(v) = parsed.get(*k) {
            emit(k, v, Style::default().fg(Color::Cyan));
        }
    }
    if let Value::Object(map) = parsed {
        // Always render extras in alphabetical key order so the line layout
        // is stable across rows (mixed sources have different key sets and
        // the BTreeMap / preserve_order behaviour was making reading vs
        // context jump positions between rows).
        let mut extra_keys: Vec<&String> = map
            .keys()
            .filter(|k| !known.contains(&k.as_str()))
            .collect();
        extra_keys.sort();
        for k in extra_keys {
            emit(k, &map[k], Style::default().fg(Color::DarkGray));
        }
    }
    out
}

/// Field-level diff between original (from JSONL) and edited (from audit
/// log). For each key (union, in known order then alphabetic), emit:
///   ` =  key  value`           if unchanged
///   ` -  key  original`        if changed (red), then
///   ` +  key  edited`          (green) on next line
///   ` -  key  ...`             only present in original (deleted)
///   ` +  key  ...`             only present in edited (added)
fn pretty_row_diff(orig: &Value, edited: &Value) -> Vec<Line<'static>> {
    let known = [
        "reading",
        "surface",
        "context",
        "left_context_surface",
        "left_context_reading",
        "source",
        "span_bunsetsu",
        "sentence_id",
        "writer_id",
        "domain_id",
        "source_id",
    ];
    let mut out: Vec<Line<'static>> = Vec::new();
    let value_to_string = |v: &Value| match v {
        Value::String(s) => s.clone(),
        _ => v.to_string(),
    };
    let mut keys: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for k in known.iter() {
        if (orig.get(*k).is_some() || edited.get(*k).is_some()) && seen.insert(k.to_string()) {
            keys.push((*k).into());
        }
    }
    let mut extras: Vec<String> = Vec::new();
    for v in [orig, edited] {
        if let Value::Object(map) = v {
            for k in map.keys() {
                if !known.contains(&k.as_str()) && seen.insert(k.clone()) {
                    extras.push(k.clone());
                }
            }
        }
    }
    extras.sort();
    keys.extend(extras);

    let red = Style::default().fg(Color::Red);
    let green = Style::default().fg(Color::Green);
    let dim = Style::default().fg(Color::DarkGray);
    for k in keys {
        let o = orig.get(&k);
        let e = edited.get(&k);
        match (o, e) {
            (Some(ov), Some(ev)) if ov == ev => {
                out.push(Line::from(vec![
                    Span::styled(format!(" =  {:<22}", k), dim),
                    Span::styled(value_to_string(ov), dim),
                ]));
            }
            (Some(ov), Some(ev)) => {
                out.push(Line::from(vec![
                    Span::styled(format!(" -  {:<22}", k), red),
                    Span::styled(value_to_string(ov), red),
                ]));
                out.push(Line::from(vec![
                    Span::styled(format!(" +  {:<22}", k), green),
                    Span::styled(value_to_string(ev), green),
                ]));
            }
            (Some(ov), None) => {
                out.push(Line::from(vec![
                    Span::styled(format!(" -  {:<22}", k), red),
                    Span::styled(value_to_string(ov), red),
                ]));
            }
            (None, Some(ev)) => {
                out.push(Line::from(vec![
                    Span::styled(format!(" +  {:<22}", k), green),
                    Span::styled(value_to_string(ev), green),
                ]));
            }
            _ => {}
        }
    }
    out
}

fn render(
    f: &mut ratatui::Frame,
    ui: &UiState,
    audit: &AuditState,
    area: Rect,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header
            Constraint::Min(5),     // body
            Constraint::Length(3), // status
        ])
        .split(area);

    // Header
    let action_tag = if audit.deleted.contains_key(&ui.cur) {
        Span::styled(" DEL ", Style::default().bg(Color::Red).fg(Color::White))
    } else if audit.replaced.contains_key(&ui.cur) {
        Span::styled(" EDIT ", Style::default().bg(Color::Yellow).fg(Color::Black))
    } else if audit.accepted.contains_key(&ui.cur) {
        Span::styled(" OK ", Style::default().bg(Color::Green).fg(Color::Black))
    } else {
        Span::raw("")
    };
    let header = Paragraph::new(vec![Line::from(vec![
        Span::styled(
            format!("row {} / {}", ui.cur, ui.total.saturating_sub(1)),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw("    "),
        Span::raw(format!(
            "audited: {} ({} del, {} edit, {} ok)",
            audit.deleted.len() + audit.replaced.len() + audit.accepted.len(),
            audit.deleted.len(),
            audit.replaced.len(),
            audit.accepted.len()
        )),
        Span::raw("    "),
        action_tag,
    ])])
    .block(Block::default().borders(Borders::ALL).title("data-row audit"));
    f.render_widget(header, chunks[0]);

    // Body
    let body_lines: Vec<Line> = if matches!(ui.mode, InputMode::ReviewSuggestion)
        && ui.parsed.is_some()
        && ui.pending_suggestion.is_some()
    {
        pretty_row_diff(
            ui.parsed.as_ref().unwrap(),
            ui.pending_suggestion.as_ref().unwrap(),
        )
    } else if ui.show_diff && ui.original.is_some() && ui.parsed.is_some() {
        pretty_row_diff(ui.original.as_ref().unwrap(), ui.parsed.as_ref().unwrap())
    } else if let Some(parsed) = &ui.parsed {
        pretty_row(parsed, &ui.mode)
    } else if let Some(err) = &ui.parse_err {
        vec![
            Line::from(Span::styled(
                format!("  parse error: {}", err),
                Style::default().fg(Color::Red),
            )),
            Line::from(Span::raw(format!("  raw: {}", ui.raw_line))),
        ]
    } else {
        vec![Line::from("  (loading)")]
    };
    let body = Paragraph::new(body_lines)
        .block(Block::default().borders(Borders::ALL).title("row"))
        .wrap(Wrap { trim: false });
    f.render_widget(body, chunks[1]);

    let (status_text, hint) = match &ui.mode {
        InputMode::Normal => {
            let title = if ui.show_diff {
                "[j/k] move [g]oto [a]ccept [d]el [e]dit [u]ndo [c]ompare* [l]lm [s]ave [q]uit"
            } else {
                "[j/k] move [g]oto [a]ccept [d]el [e]dit [u]ndo [c]ompare [l]lm [s]ave [q]uit"
            };
            (ui.status.clone(), title)
        }
        InputMode::Goto(buf) => (
            format!("goto: {buf}"),
            "[Enter] confirm  [Esc] cancel",
        ),
        InputMode::PickField { fields } => (
            format!(
                "edit which field? {}",
                fields
                    .iter()
                    .enumerate()
                    .map(|(i, k)| format!("[{}]{}", i + 1, k))
                    .collect::<Vec<_>>()
                    .join("  ")
            ),
            "[1-9] pick  [Esc] cancel",
        ),
        InputMode::EditField { key, .. } => (
            format!("editing `{key}` (←→ Home End move, Backspace/Del erase)"),
            "[Enter] save  [Esc] cancel",
        ),
        InputMode::ReviewSuggestion => (
            "LLM suggestion (current → suggested)".into(),
            "[y] accept  [n] reject",
        ),
    };
    let status = Paragraph::new(Line::from(vec![Span::raw(status_text)]))
        .block(Block::default().borders(Borders::ALL).title(hint));
    f.render_widget(status, chunks[2]);
}

fn load_current(ui: &mut UiState, jsonl: &Path, offsets: &[u64], audit: &AuditState) {
    let i = ui.cur as usize;
    if i >= offsets.len() {
        ui.parse_err = Some(format!("row {} out of range (max {})", i, offsets.len() - 1));
        ui.parsed = None;
        ui.raw_line.clear();
        return;
    }
    // If this row has a pending edit, render the edited content instead of
    // re-reading the original JSONL — otherwise navigating away and back
    // makes prior edits look lost (they're still on disk in
    // <jsonl>.audit.jsonl, just not visible).
    if let Some(edited) = audit.replaced.get(&ui.cur) {
        ui.raw_line = edited.clone();
        match serde_json::from_str::<Value>(edited) {
            Ok(v) => {
                ui.parsed = Some(v);
                ui.parse_err = None;
            }
            Err(e) => {
                ui.parsed = None;
                ui.parse_err = Some(e.to_string());
            }
        }
        // Cache the unmodified original alongside so `c` (compare) shows
        // the field-level diff without re-reading on every render.
        ui.original = read_row(jsonl, offsets[i])
            .ok()
            .and_then(|line| serde_json::from_str::<Value>(&line).ok());
        return;
    }
    ui.original = None;
    match read_row(jsonl, offsets[i]) {
        Ok(line) => {
            ui.raw_line = line.clone();
            match serde_json::from_str::<Value>(&line) {
                Ok(v) => {
                    ui.parsed = Some(v);
                    ui.parse_err = None;
                }
                Err(e) => {
                    ui.parsed = None;
                    ui.parse_err = Some(e.to_string());
                }
            }
        }
        Err(e) => {
            ui.parse_err = Some(e.to_string());
            ui.parsed = None;
            ui.raw_line.clear();
        }
    }
}

/// Handle one key press while the inline EditField mode is active. Commits
/// on Enter (parsed JSON updated + autosaved to audit log), cancels on Esc.
fn handle_edit_key(ui: &mut UiState, audit: &mut AuditState, code: KeyCode) {
    let (key, buffer, cursor) = match &mut ui.mode {
        InputMode::EditField { key, buffer, cursor } => (key.clone(), buffer, cursor),
        _ => return,
    };
    match code {
        KeyCode::Esc => {
            // Cancel this field's edit but stay in the edit session — the
            // user pressed `e` because they want to fix multiple fields,
            // and reading/surface tend to come in pairs. PickField's Esc
            // is what leaves the session entirely.
            let fields = ui
                .parsed
                .as_ref()
                .map(editable_fields)
                .unwrap_or_default();
            ui.mode = if fields.is_empty() {
                InputMode::Normal
            } else {
                InputMode::PickField { fields }
            };
            ui.status = "edit cancelled (pick another field or Esc to exit)".into();
        }
        KeyCode::Enter => {
            let new_val: String = buffer.iter().collect();
            let mut new_parsed = ui.parsed.clone().unwrap_or(Value::Null);
            if let Value::Object(map) = &mut new_parsed {
                map.insert(key.clone(), Value::String(new_val));
            }
            normalize_jp_punct(&mut new_parsed);
            let commit_status = match serde_json::to_string(&new_parsed) {
                Ok(compact) => {
                    audit.replace(ui.cur, compact);
                    ui.parsed = Some(new_parsed);
                    ui.parse_err = None;
                    match audit.save() {
                        Ok(()) => format!(
                            "row {} `{}` edited (autosaved {} entries) — pick next field or Esc",
                            ui.cur, key, audit.saved_count
                        ),
                        Err(e) => format!("row {} `{}` edited (autosave failed: {e})", ui.cur, key),
                    }
                }
                Err(e) => format!("commit failed (serialize): {e}"),
            };
            ui.status = commit_status;
            // Stay in edit session: jump back to the field picker so the
            // paired field (e.g. reading after surface) is one keystroke
            // away.
            let fields = ui
                .parsed
                .as_ref()
                .map(editable_fields)
                .unwrap_or_default();
            ui.mode = if fields.is_empty() {
                InputMode::Normal
            } else {
                InputMode::PickField { fields }
            };
        }
        KeyCode::Backspace => {
            if *cursor > 0 {
                *cursor -= 1;
                buffer.remove(*cursor);
            }
        }
        KeyCode::Delete => {
            if *cursor < buffer.len() {
                buffer.remove(*cursor);
            }
        }
        KeyCode::Left => {
            if *cursor > 0 {
                *cursor -= 1;
            }
        }
        KeyCode::Right => {
            if *cursor < buffer.len() {
                *cursor += 1;
            }
        }
        KeyCode::Home => {
            *cursor = 0;
        }
        KeyCode::End => {
            *cursor = buffer.len();
        }
        KeyCode::Char(c) => {
            buffer.insert(*cursor, c);
            *cursor += 1;
        }
        _ => {}
    }
}

fn cmd_audit(jsonl: &Path, start: Option<u64>, review: bool) -> Result<()> {
    let offsets = load_index(jsonl)?;
    let total = offsets.len() as u64;
    if total == 0 {
        return Err(anyhow!("index empty"));
    }
    let mut audit = AuditState::default();
    audit.log_path = audit_log_path(jsonl);
    let prior_bad = audit.load().unwrap_or(0);
    let prior_count = audit.deleted.len() + audit.replaced.len() + audit.accepted.len();
    let llm = LlmConfig::from_env();
    let resolved_start: u64 = match (start, review) {
        (Some(n), _) => n,
        (None, true) => audit.replaced.keys().copied().min().unwrap_or(0),
        (None, false) => audit.last_audited().map(|n| n + 1).unwrap_or(0),
    };
    let mut ui = UiState {
        cur: resolved_start.min(total - 1),
        total,
        show_diff: review,
        ..UiState::default()
    };
    load_current(&mut ui, jsonl, &offsets, &audit);
    ui.status = format!(
        "loaded index: {} rows. {} prior audit entries{}. resume @ row {}. log → {}",
        total,
        prior_count,
        if prior_bad > 0 { format!(" ({} skipped)", prior_bad) } else { String::new() },
        ui.cur,
        audit.log_path.display()
    );

    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut term = Terminal::new(backend)?;

    while !ui.quit {
        term.draw(|f| render(f, &ui, &audit, f.area()))?;

        if let Event::Key(KeyEvent { code, modifiers, kind, .. }) = event::read()? {
            // Windows console emits Press + Release events for every key;
            // only act on Press so a single keystroke advances one row.
            if kind != KeyEventKind::Press {
                continue;
            }
            // Goto mode: numeric input + Enter/Esc.
            if let InputMode::Goto(buf) = &ui.mode {
                let mut buf = buf.clone();
                match code {
                    KeyCode::Esc => {
                        ui.mode = InputMode::Normal;
                    }
                    KeyCode::Enter => {
                        match buf.trim().parse::<u64>() {
                            Ok(n) if n < total => {
                                ui.cur = n;
                                load_current(&mut ui, jsonl, &offsets, &audit);
                                ui.status = format!("jumped to {}", n);
                            }
                            Ok(n) => ui.status = format!("out of range: {}", n),
                            Err(e) => ui.status = format!("bad number: {e}"),
                        }
                        ui.mode = InputMode::Normal;
                    }
                    KeyCode::Backspace => {
                        buf.pop();
                        ui.mode = InputMode::Goto(buf);
                    }
                    KeyCode::Char(c) if c.is_ascii_digit() => {
                        buf.push(c);
                        ui.mode = InputMode::Goto(buf);
                    }
                    _ => {}
                }
                continue;
            }
            // Field-pick mode: 1-9 picks, Esc cancels.
            if let InputMode::PickField { fields } = &ui.mode {
                let fields = fields.clone();
                match code {
                    KeyCode::Esc => {
                        ui.mode = InputMode::Normal;
                        ui.status = "edit cancelled".into();
                    }
                    KeyCode::Char(c) if c.is_ascii_digit() && c != '0' => {
                        let idx = (c as u8 - b'1') as usize;
                        if let Some(key) = fields.get(idx) {
                            let cur_val = ui
                                .parsed
                                .as_ref()
                                .and_then(|v| v.get(key))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let buffer: Vec<char> = cur_val.chars().collect();
                            let cursor = buffer.len();
                            ui.mode = InputMode::EditField { key: key.clone(), buffer, cursor };
                        } else {
                            ui.status = format!("no field at index {}", idx + 1);
                        }
                    }
                    _ => {}
                }
                continue;
            }
            // Inline text-edit mode.
            if let InputMode::EditField { .. } = &ui.mode {
                handle_edit_key(&mut ui, &mut audit, code);
                continue;
            }
            // LLM suggestion review mode: y to accept, n to reject.
            if let InputMode::ReviewSuggestion = &ui.mode {
                match code {
                    KeyCode::Char('y') | KeyCode::Enter => {
                        if let Some(mut sugg) = ui.pending_suggestion.take() {
                            normalize_jp_punct(&mut sugg);
                            match serde_json::to_string(&sugg) {
                                Ok(compact) => {
                                    audit.replace(ui.cur, compact);
                                    ui.parsed = Some(sugg);
                                    ui.parse_err = None;
                                    ui.status = match audit.save() {
                                        Ok(()) => format!(
                                            "row {} suggestion applied (autosaved {} entries)",
                                            ui.cur, audit.saved_count
                                        ),
                                        Err(e) => format!(
                                            "row {} suggestion applied (autosave failed: {e})",
                                            ui.cur
                                        ),
                                    };
                                }
                                Err(e) => {
                                    ui.status = format!("apply failed (serialize): {e}");
                                }
                            }
                        }
                        ui.mode = InputMode::Normal;
                    }
                    KeyCode::Char('n') | KeyCode::Esc => {
                        ui.pending_suggestion = None;
                        ui.mode = InputMode::Normal;
                        ui.status = "suggestion rejected".into();
                    }
                    _ => {}
                }
                continue;
            }

            match code {
                KeyCode::Char('q') => {
                    ui.quit = true;
                }
                KeyCode::Char('s') => match audit.save() {
                    Ok(()) => {
                        ui.status = format!(
                            "saved {} entries to {}",
                            audit.saved_count,
                            audit.log_path.display()
                        );
                    }
                    Err(e) => {
                        ui.status = format!("save failed: {e}");
                    }
                },
                KeyCode::Char('j') | KeyCode::Down => {
                    if ui.cur + 1 < total {
                        ui.cur += 1;
                        load_current(&mut ui, jsonl, &offsets, &audit);
                    }
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    if ui.cur > 0 {
                        ui.cur -= 1;
                        load_current(&mut ui, jsonl, &offsets, &audit);
                    }
                }
                KeyCode::PageDown => {
                    ui.cur = (ui.cur + 10).min(total - 1);
                    load_current(&mut ui, jsonl, &offsets, &audit);
                }
                KeyCode::PageUp => {
                    ui.cur = ui.cur.saturating_sub(10);
                    load_current(&mut ui, jsonl, &offsets, &audit);
                }
                KeyCode::Char('g') => {
                    ui.mode = InputMode::Goto(String::new());
                    ui.status = String::new();
                }
                KeyCode::Char('a') => {
                    // Accept marks "reviewed and kept verbatim" so the row
                    // gets an OK badge and counts toward audit progress.
                    // Does not override an existing delete/edit on the same
                    // row (those are stronger decisions).
                    let already_decided = audit.deleted.contains_key(&ui.cur)
                        || audit.replaced.contains_key(&ui.cur);
                    audit.accept(ui.cur);
                    let base = if already_decided {
                        format!("row {} already has a decision (kept)", ui.cur)
                    } else {
                        format!("row {} OK", ui.cur)
                    };
                    if !already_decided {
                        ui.status = match audit.save() {
                            Ok(()) => format!("{base} (autosaved {} entries)", audit.saved_count),
                            Err(e) => format!("{base} (autosave failed: {e})"),
                        };
                    } else {
                        ui.status = base;
                    }
                    if ui.cur + 1 < total {
                        ui.cur += 1;
                        load_current(&mut ui, jsonl, &offsets, &audit);
                    }
                }
                KeyCode::Char('u') => {
                    let had_del = audit.deleted.remove(&ui.cur).is_some();
                    let had_edit = audit.replaced.remove(&ui.cur).is_some();
                    let had_ack = audit.accepted.remove(&ui.cur).is_some();
                    let any = had_del || had_edit || had_ack;
                    let base = if any {
                        format!(
                            "row {} undone (was del={} edit={} ok={})",
                            ui.cur, had_del, had_edit, had_ack
                        )
                    } else {
                        format!("row {} had no pending change", ui.cur)
                    };
                    if any {
                        ui.status = match audit.save() {
                            Ok(()) => format!("{base} (autosaved {} entries)", audit.saved_count),
                            Err(e) => format!("{base} (autosave failed: {e})"),
                        };
                    } else {
                        ui.status = base;
                    }
                    load_current(&mut ui, jsonl, &offsets, &audit);
                }
                KeyCode::Char('d') => {
                    audit.delete(ui.cur);
                    if let Err(e) = audit.save() {
                        ui.status = format!("row {} DELETE (autosave failed: {e})", ui.cur);
                    } else {
                        ui.status = format!(
                            "row {} DELETE (autosaved {} entries)",
                            ui.cur, audit.saved_count
                        );
                    }
                    if ui.cur + 1 < total {
                        ui.cur += 1;
                        load_current(&mut ui, jsonl, &offsets, &audit);
                    }
                }
                KeyCode::Char('e') => {
                    // Enter field-picker. The picker shows numbered editable
                    // fields; the user presses 1-9 to start in-place edit.
                    let fields = ui
                        .parsed
                        .as_ref()
                        .map(editable_fields)
                        .unwrap_or_default();
                    if fields.is_empty() {
                        ui.status = "no editable string fields".into();
                    } else {
                        ui.mode = InputMode::PickField { fields };
                        ui.status.clear();
                    }
                }
                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                    ui.quit = true;
                }
                KeyCode::Char('l') => {
                    let cfg = match &llm {
                        Some(c) => c,
                        None => {
                            ui.status = "LLM not configured (set DATA_ROW_LLM_ENDPOINT, DATA_ROW_LLM_TOKEN)".into();
                            continue;
                        }
                    };
                    let row = match &ui.parsed {
                        Some(v) => v.clone(),
                        None => {
                            ui.status = "no parsed row to send".into();
                            continue;
                        }
                    };
                    ui.status = format!("calling LLM ({}) …", cfg.model);
                    // Force a redraw so the user sees "calling..." before
                    // we block on the network request.
                    term.draw(|f| render(f, &ui, &audit, f.area()))?;
                    match llm_suggest(cfg, &row) {
                        Ok(Some(sugg)) => {
                            if Value::Object(serde_json::Map::new()) == sugg
                                || sugg == row
                            {
                                ui.status = "LLM returned no change".into();
                            } else {
                                ui.pending_suggestion = Some(sugg);
                                ui.mode = InputMode::ReviewSuggestion;
                                ui.status = "review suggestion: y / n".into();
                            }
                        }
                        Ok(None) => {
                            ui.status = "LLM: no fix needed".into();
                        }
                        Err(e) => {
                            ui.status = format!("LLM error: {e}");
                        }
                    }
                }
                KeyCode::Char('c') => {
                    ui.show_diff = !ui.show_diff;
                    ui.status = if ui.show_diff {
                        if ui.original.is_some() {
                            format!("compare ON (orig vs edit)")
                        } else {
                            format!("compare ON (no edit on row {} — showing as-is)", ui.cur)
                        }
                    } else {
                        "compare OFF".into()
                    };
                }
                _ => {}
            }
        }
    }

    // Final save prompt — auto-save for v0.
    if audit.deleted.len() + audit.replaced.len() > 0 {
        if let Err(e) = audit.save() {
            eprintln!("[audit] save failed: {e}");
        } else {
            eprintln!(
                "[audit] saved {} entries to {}",
                audit.saved_count,
                audit.log_path.display()
            );
        }
    }

    disable_raw_mode()?;
    execute!(term.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    Ok(())
}

fn main() -> Result<()> {
    // Load .env from cwd (or any parent). Print where it was found so the
    // user can confirm — silent failure is otherwise indistinguishable
    // from "no env vars at all" when the LLM call later complains about
    // missing tokens.
    match dotenvy::dotenv() {
        Ok(path) => eprintln!("[data-row] loaded env from {}", path.display()),
        Err(e) => eprintln!(
            "[data-row] no .env loaded ({}). Using shell env only.",
            e
        ),
    }
    let cli = Cli::parse();
    match cli.command {
        Command::Index { jsonl } => build_index(&jsonl),
        Command::Stats { jsonl } => cmd_stats(&jsonl),
        Command::Audit { jsonl, start, review } => cmd_audit(&jsonl, start, review),
    }
}
