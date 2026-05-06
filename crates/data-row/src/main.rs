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
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers};
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
        /// Initial row id (default 0).
        #[arg(long, default_value_t = 0)]
        start: u64,
    },
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
    log_path: PathBuf,
    saved_count: usize,
}

impl AuditState {
    fn delete(&mut self, row_id: u64) {
        self.replaced.remove(&row_id);
        self.deleted.insert(row_id, ());
    }
    fn replace(&mut self, row_id: u64, new_line: String) {
        self.deleted.remove(&row_id);
        self.replaced.insert(row_id, new_line);
    }
    fn save(&mut self) -> Result<()> {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.log_path)?;
        let mut total = 0;
        let mut keys: Vec<u64> = self.deleted.keys().copied().chain(self.replaced.keys().copied()).collect();
        keys.sort_unstable();
        keys.dedup();
        for row_id in keys {
            if self.replaced.contains_key(&row_id) {
                let new_line = self.replaced.get(&row_id).unwrap();
                let new_row: Value = serde_json::from_str(new_line)?;
                let entry = serde_json::json!({
                    "row_id": row_id,
                    "action": "replace",
                    "new_row": new_row,
                });
                writeln!(f, "{}", serde_json::to_string(&entry)?)?;
            } else {
                let entry = serde_json::json!({"row_id": row_id, "action": "delete"});
                writeln!(f, "{}", serde_json::to_string(&entry)?)?;
            }
            total += 1;
        }
        f.flush()?;
        self.saved_count = total;
        Ok(())
    }
}

#[derive(Default)]
struct UiState {
    cur: u64,
    total: u64,
    raw_line: String,
    parsed: Option<Value>,
    parse_err: Option<String>,
    status: String,
    pending_goto: Option<String>,
    quit: bool,
    edited_for_current: bool,
}

fn pretty_row(parsed: &Value) -> Vec<Line<'static>> {
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
    for k in known.iter() {
        if let Some(v) = parsed.get(*k) {
            let val = match v {
                Value::String(s) => s.clone(),
                _ => v.to_string(),
            };
            out.push(Line::from(vec![
                Span::styled(
                    format!("  {:<22}", k),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(val),
            ]));
        }
    }
    // Show any extra fields not in the known set.
    if let Value::Object(map) = parsed {
        for (k, v) in map.iter() {
            if known.contains(&k.as_str()) {
                continue;
            }
            let val = match v {
                Value::String(s) => s.clone(),
                _ => v.to_string(),
            };
            out.push(Line::from(vec![
                Span::styled(
                    format!("  {:<22}", k),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::raw(val),
            ]));
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
            "audited: {} ({} del, {} edit)",
            audit.deleted.len() + audit.replaced.len(),
            audit.deleted.len(),
            audit.replaced.len()
        )),
        Span::raw("    "),
        action_tag,
    ])])
    .block(Block::default().borders(Borders::ALL).title("data-row audit"));
    f.render_widget(header, chunks[0]);

    // Body
    let body_lines: Vec<Line> = if let Some(parsed) = &ui.parsed {
        pretty_row(parsed)
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

    // Status
    let status_text = if let Some(buf) = &ui.pending_goto {
        format!("goto: {} (Enter to confirm, Esc to cancel)", buf)
    } else {
        ui.status.clone()
    };
    let status = Paragraph::new(Line::from(vec![
        Span::raw(status_text),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("[j/k] next/prev  [g]oto  [a]ccept  [d]elete  [e]dit  [s]ave  [q]uit"),
    );
    f.render_widget(status, chunks[2]);
}

fn load_current(ui: &mut UiState, jsonl: &Path, offsets: &[u64]) {
    let i = ui.cur as usize;
    if i >= offsets.len() {
        ui.parse_err = Some(format!("row {} out of range (max {})", i, offsets.len() - 1));
        ui.parsed = None;
        ui.raw_line.clear();
        return;
    }
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

fn cmd_audit(jsonl: &Path, start: u64) -> Result<()> {
    let offsets = load_index(jsonl)?;
    let total = offsets.len() as u64;
    if total == 0 {
        return Err(anyhow!("index empty"));
    }
    let mut ui = UiState {
        cur: start.min(total - 1),
        total,
        ..UiState::default()
    };
    let mut audit = AuditState::default();
    audit.log_path = audit_log_path(jsonl);
    load_current(&mut ui, jsonl, &offsets);
    ui.status = format!("loaded index: {} rows. log → {}", total, audit.log_path.display());

    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut term = Terminal::new(backend)?;

    while !ui.quit {
        term.draw(|f| render(f, &ui, &audit, f.area()))?;

        if let Event::Key(KeyEvent { code, modifiers, .. }) = event::read()? {
            if let Some(buf) = ui.pending_goto.clone() {
                match code {
                    KeyCode::Esc => {
                        ui.pending_goto = None;
                    }
                    KeyCode::Enter => {
                        match buf.trim().parse::<u64>() {
                            Ok(n) if n < total => {
                                ui.cur = n;
                                load_current(&mut ui, jsonl, &offsets);
                                ui.status = format!("jumped to {}", n);
                            }
                            Ok(n) => {
                                ui.status = format!("out of range: {}", n);
                            }
                            Err(e) => {
                                ui.status = format!("bad number: {e}");
                            }
                        }
                        ui.pending_goto = None;
                    }
                    KeyCode::Backspace => {
                        let mut s = buf;
                        s.pop();
                        ui.pending_goto = Some(s);
                    }
                    KeyCode::Char(c) if c.is_ascii_digit() => {
                        let mut s = buf;
                        s.push(c);
                        ui.pending_goto = Some(s);
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
                        load_current(&mut ui, jsonl, &offsets);
                    }
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    if ui.cur > 0 {
                        ui.cur -= 1;
                        load_current(&mut ui, jsonl, &offsets);
                    }
                }
                KeyCode::PageDown => {
                    ui.cur = (ui.cur + 10).min(total - 1);
                    load_current(&mut ui, jsonl, &offsets);
                }
                KeyCode::PageUp => {
                    ui.cur = ui.cur.saturating_sub(10);
                    load_current(&mut ui, jsonl, &offsets);
                }
                KeyCode::Char('g') => {
                    ui.pending_goto = Some(String::new());
                    ui.status = String::new();
                }
                KeyCode::Char('a') => {
                    audit.deleted.remove(&ui.cur);
                    audit.replaced.remove(&ui.cur);
                    ui.status = format!("row {} accepted", ui.cur);
                    if ui.cur + 1 < total {
                        ui.cur += 1;
                        load_current(&mut ui, jsonl, &offsets);
                    }
                }
                KeyCode::Char('d') => {
                    audit.delete(ui.cur);
                    ui.status = format!("row {} marked DELETE", ui.cur);
                    if ui.cur + 1 < total {
                        ui.cur += 1;
                        load_current(&mut ui, jsonl, &offsets);
                    }
                }
                KeyCode::Char('e') => {
                    // Edit current row in $EDITOR.
                    let editor = std::env::var("EDITOR").unwrap_or_else(|_| {
                        if cfg!(target_os = "windows") { "notepad".into() } else { "vi".into() }
                    });
                    let tmp = std::env::temp_dir().join(format!("data-row-{}.json", ui.cur));
                    let pretty = ui
                        .parsed
                        .as_ref()
                        .map(|v| serde_json::to_string_pretty(v).unwrap_or_default())
                        .unwrap_or_else(|| ui.raw_line.clone());
                    std::fs::write(&tmp, pretty)?;
                    // Suspend TUI, run editor, restore TUI.
                    disable_raw_mode()?;
                    execute!(term.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
                    let status = std::process::Command::new(&editor)
                        .arg(&tmp)
                        .status();
                    enable_raw_mode()?;
                    execute!(term.backend_mut(), EnterAlternateScreen, EnableMouseCapture)?;
                    term.clear()?;
                    match status {
                        Ok(_) => {
                            let new_text = std::fs::read_to_string(&tmp).unwrap_or_default();
                            // Try parse; if pretty, compact to one line.
                            match serde_json::from_str::<Value>(&new_text) {
                                Ok(v) => {
                                    let compact = serde_json::to_string(&v)?;
                                    audit.replace(ui.cur, compact);
                                    ui.parsed = Some(v);
                                    ui.parse_err = None;
                                    ui.status = format!("row {} edited", ui.cur);
                                }
                                Err(e) => {
                                    ui.status = format!("edit not saved (json error): {e}");
                                }
                            }
                        }
                        Err(e) => {
                            ui.status = format!("editor failed: {e}");
                        }
                    }
                    let _ = std::fs::remove_file(&tmp);
                }
                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                    ui.quit = true;
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
    let cli = Cli::parse();
    match cli.command {
        Command::Index { jsonl } => build_index(&jsonl),
        Command::Stats { jsonl } => cmd_stats(&jsonl),
        Command::Audit { jsonl, start } => cmd_audit(&jsonl, start),
    }
}
