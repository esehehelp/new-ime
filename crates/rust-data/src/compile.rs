use crate::shard::{MAGIC, VERSION};
use anyhow::{Context, Result};
use data_core::{JsonlLines, Row};
use rust_tokenizer::{SharedCharTokenizer, BLANK_ID};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub max_context_chars: usize,
    pub max_reading_tokens: usize,
    pub max_surface_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMetadata {
    pub shard_version: u32,
    pub row_count: usize,
    pub max_context_chars: usize,
    pub max_reading_tokens: usize,
    pub max_surface_tokens: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub writers: BTreeMap<String, u32>,
    #[serde(default)]
    pub domains: BTreeMap<String, u32>,
    #[serde(default)]
    pub sources: BTreeMap<String, u32>,
}

#[derive(Debug)]
struct RowJob {
    seq: u64,
    row: Row,
    writer_id: u32,
    domain_id: u32,
    source_id: u32,
}

#[derive(Debug)]
struct CompiledRow {
    seq: u64,
    bytes: Vec<u8>,
    row_bytes: u64,
}

enum WorkerMessage {
    Row(CompiledRow),
    Error(String),
    Done,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            max_context_chars: 40,
            max_reading_tokens: 128,
            max_surface_tokens: 128,
        }
    }
}

pub fn compile_jsonl_to_shard(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    tokenizer: &SharedCharTokenizer,
    options: &CompileOptions,
) -> Result<ShardMetadata> {
    const HEADER_LEN: u64 = 36;
    const LOG_EVERY_ROWS: u64 = 1_000_000;

    if let Some(parent) = output.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("mkdir {}", parent.display()))?;
        }
    }

    let index_tmp_path = output.as_ref().with_extension("kkc.index.tmp");
    let mut out = BufWriter::new(
        File::create(output.as_ref())
            .with_context(|| format!("create shard {}", output.as_ref().display()))?,
    );
    let mut index_tmp = BufWriter::new(
        File::create(&index_tmp_path)
            .with_context(|| format!("create temp index {}", index_tmp_path.display()))?,
    );
    out.write_all(&[0u8; HEADER_LEN as usize])
        .context("write placeholder header")?;

    let mut payload_len = 0u64;
    let mut row_count = 0u64;
    let start = Instant::now();
    let worker_count = thread::available_parallelism()
        .map(|n| n.get().clamp(2, 16))
        .unwrap_or(4);
    let input_path = input.as_ref().to_path_buf();
    let output_path = output.as_ref().to_path_buf();
    eprintln!(
        "compile-shard start: input={} output={} vocab_size={} workers={}",
        input_path.display(),
        output_path.display(),
        tokenizer.vocab_size(),
        worker_count
    );

    let (result_tx, result_rx) = mpsc::sync_channel::<WorkerMessage>(worker_count * 4);
    let mut writer_table = BTreeMap::<String, u32>::new();
    let mut domain_table = BTreeMap::<String, u32>::new();
    let mut source_table = BTreeMap::<String, u32>::new();

    thread::scope(|scope| -> Result<()> {
        let mut worker_handles = Vec::with_capacity(worker_count);
        let mut job_txs = Vec::with_capacity(worker_count);
        for worker_idx in 0..worker_count {
            let (job_tx, job_rx) = mpsc::sync_channel::<RowJob>(1024);
            job_txs.push(job_tx);
            let result_tx = result_tx.clone();
            let tokenizer = tokenizer.clone();
            let options = options.clone();
            worker_handles.push(
                scope.spawn(move || worker_loop(worker_idx, job_rx, result_tx, tokenizer, options)),
            );
        }
        drop(result_tx);

        let reader = scope.spawn(move || reader_loop(&input_path, job_txs));

        let mut pending = BTreeMap::<u64, CompiledRow>::new();
        let mut next_seq = 0u64;
        let mut workers_done = 0usize;
        while workers_done < worker_count {
            match result_rx.recv().context("receive compiled row")? {
                WorkerMessage::Row(compiled) => {
                    pending.insert(compiled.seq, compiled);
                    while let Some(compiled) = pending.remove(&next_seq) {
                        index_tmp
                            .write_all(&payload_len.to_le_bytes())
                            .context("write index entry")?;
                        out.write_all(&compiled.bytes)
                            .context("write row payload")?;
                        payload_len += compiled.row_bytes;
                        row_count += 1;
                        next_seq += 1;
                        if row_count % LOG_EVERY_ROWS == 0 {
                            let elapsed = start.elapsed().as_secs_f64().max(1e-9);
                            let rows_per_sec = row_count as f64 / elapsed;
                            let mib_written = (HEADER_LEN + payload_len) as f64 / (1024.0 * 1024.0);
                            eprintln!(
                                "compile-shard progress: rows={} elapsed_sec={:.1} rows_per_sec={:.0} written_mib={:.1}",
                                row_count, elapsed, rows_per_sec, mib_written
                            );
                        }
                    }
                }
                WorkerMessage::Error(err) => return Err(anyhow::anyhow!(err)),
                WorkerMessage::Done => workers_done += 1,
            }
        }

        let (reader_rows, writer_ids, domain_ids, source_ids) = reader
            .join()
            .map_err(|_| anyhow::anyhow!("reader thread panicked"))??;
        writer_table = writer_ids;
        domain_table = domain_ids;
        source_table = source_ids;
        if row_count != reader_rows {
            anyhow::bail!(
                "row count mismatch after parallel compile: reader={} writer={}",
                reader_rows,
                row_count
            );
        }
        if !pending.is_empty() {
            anyhow::bail!(
                "writer finished with {} pending out-of-order rows",
                pending.len()
            );
        }
        for handle in worker_handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("worker thread panicked"))??;
        }
        Ok(())
    })?;

    index_tmp.flush().context("flush temp index")?;
    drop(index_tmp);

    let index_offset = HEADER_LEN + payload_len;
    let mut index_reader = File::open(&index_tmp_path)
        .with_context(|| format!("open temp index {}", index_tmp_path.display()))?;
    std::io::copy(&mut index_reader, &mut out).context("append temp index")?;
    out.flush().context("flush shard body")?;
    out.seek(SeekFrom::Start(0)).context("seek shard header")?;
    out.write_all(&MAGIC).context("write magic")?;
    out.write_all(&VERSION.to_le_bytes())
        .context("write version")?;
    out.write_all(&row_count.to_le_bytes())
        .context("write row count")?;
    out.write_all(&HEADER_LEN.to_le_bytes())
        .context("write payload offset")?;
    out.write_all(&index_offset.to_le_bytes())
        .context("write index offset")?;
    out.flush().context("flush shard header")?;
    drop(out);
    let _ = std::fs::remove_file(&index_tmp_path);

    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    let rows_per_sec = row_count as f64 / elapsed;
    let shard_size = std::fs::metadata(output.as_ref())
        .map(|m| m.len())
        .unwrap_or(HEADER_LEN + payload_len + row_count * 8);
    eprintln!(
        "compile-shard done: rows={} elapsed_sec={:.1} rows_per_sec={:.0} shard_mib={:.1}",
        row_count,
        elapsed,
        rows_per_sec,
        shard_size as f64 / (1024.0 * 1024.0)
    );

    Ok(ShardMetadata {
        shard_version: VERSION,
        row_count: row_count as usize,
        max_context_chars: options.max_context_chars,
        max_reading_tokens: options.max_reading_tokens,
        max_surface_tokens: options.max_surface_tokens,
        vocab_size: tokenizer.vocab_size(),
        writers: writer_table,
        domains: domain_table,
        sources: source_table,
    })
}

fn reader_loop(
    input: &Path,
    job_txs: Vec<SyncSender<RowJob>>,
) -> Result<(
    u64,
    BTreeMap<String, u32>,
    BTreeMap<String, u32>,
    BTreeMap<String, u32>,
)> {
    let mut writer_table = BTreeMap::<String, u32>::new();
    let mut domain_table = BTreeMap::<String, u32>::new();
    let mut source_table = BTreeMap::<String, u32>::new();
    let mut row_count = 0u64;
    for row in JsonlLines::open(input).context("open jsonl")? {
        let row = row?;
        let writer_id = next_label_id(&mut writer_table, row.writer.as_ref());
        let domain_id = next_label_id(&mut domain_table, row.domain.as_ref());
        let source_id = next_label_id(&mut source_table, row.source.as_ref());
        let seq = row_count;
        let shard = (seq as usize) % job_txs.len();
        job_txs[shard]
            .send(RowJob {
                seq,
                row,
                writer_id,
                domain_id,
                source_id,
            })
            .context("send row job")?;
        row_count += 1;
    }
    drop(job_txs);
    Ok((row_count, writer_table, domain_table, source_table))
}

fn next_label_id(table: &mut BTreeMap<String, u32>, label: Option<&String>) -> u32 {
    match label {
        Some(value) => {
            let next = table.len() as u32 + 1;
            *table.entry(value.clone()).or_insert(next)
        }
        None => 0,
    }
}

fn worker_loop(
    _worker_idx: usize,
    job_rx: Receiver<RowJob>,
    result_tx: SyncSender<WorkerMessage>,
    tokenizer: SharedCharTokenizer,
    options: CompileOptions,
) -> Result<()> {
    for job in job_rx {
        match compile_row_job(job, &tokenizer, &options) {
            Ok(compiled) => {
                if result_tx.send(WorkerMessage::Row(compiled)).is_err() {
                    return Ok(());
                }
            }
            Err(err) => {
                let _ = result_tx.send(WorkerMessage::Error(err.to_string()));
                return Err(err);
            }
        }
    }
    let _ = result_tx.send(WorkerMessage::Done);
    Ok(())
}

fn compile_row_job(
    job: RowJob,
    tokenizer: &SharedCharTokenizer,
    options: &CompileOptions,
) -> Result<CompiledRow> {
    let context: String = job
        .row
        .context
        .chars()
        .rev()
        .take(options.max_context_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let mut reading = tokenizer.encode(&job.row.reading);
    let mut surface = tokenizer.encode(&job.row.surface);
    let mut context_ids = tokenizer.encode(&context);
    // Match `CTCCollator._encode_target` in train_ctc_nat.py: CTC targets
    // must not contain the blank token. Doing this at compile time keeps
    // the Python ShardReader collator a pure padding-to-tensor step.
    surface.retain(|&tid| tid != BLANK_ID);
    reading.truncate(options.max_reading_tokens);
    surface.truncate(options.max_surface_tokens);
    context_ids.truncate(options.max_context_chars.max(1));
    let row_bytes = 24u64 + ((reading.len() + surface.len() + context_ids.len()) as u64 * 4);
    let mut bytes = Vec::with_capacity(row_bytes as usize);
    write_row_bytes(
        &mut bytes,
        &reading,
        &surface,
        &context_ids,
        job.writer_id,
        job.domain_id,
        job.source_id,
    )?;
    Ok(CompiledRow {
        seq: job.seq,
        bytes,
        row_bytes,
    })
}

fn write_row_bytes(
    out: &mut impl Write,
    reading: &[u32],
    surface: &[u32],
    context: &[u32],
    writer_id: u32,
    domain_id: u32,
    source_id: u32,
) -> Result<u64> {
    let row_bytes = 24u64 + ((reading.len() + surface.len() + context.len()) as u64 * 4);
    out.write_all(&(reading.len() as u32).to_le_bytes())?;
    out.write_all(&(surface.len() as u32).to_le_bytes())?;
    out.write_all(&(context.len() as u32).to_le_bytes())?;
    out.write_all(&writer_id.to_le_bytes())?;
    out.write_all(&domain_id.to_le_bytes())?;
    out.write_all(&source_id.to_le_bytes())?;
    for token in reading.iter().chain(surface).chain(context) {
        out.write_all(&token.to_le_bytes())?;
    }
    Ok(row_bytes)
}
