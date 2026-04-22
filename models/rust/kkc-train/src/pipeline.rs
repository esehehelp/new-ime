//! Async-ish helpers for the training loop.
//!
//! Two components live here:
//! 1. [`AsyncCheckpointWriter`] owns a dedicated writer thread. The training
//!    loop hands it a serialized blob + destination path and immediately
//!    returns to the next step; the writer drains a bounded queue in order and
//!    flushes on drop.
//! 2. [`StagedBatchPipeline`] wraps `PrefetchedBatchIter` and (with the
//!    `cuda` feature) adds a second stage that runs H2D copy on a background
//!    thread, so the GPU compute step can consume a batch that was uploaded
//!    while the previous step was still computing.
//!
//! We deliberately use `std::thread` + `mpsc` rather than Tokio. The training
//! loop is a single long-running flow with a handful of producers; a full
//! runtime would add dependency weight and scheduling overhead without
//! buying anything the channel pair doesn't already provide.

use anyhow::{Context, Result};
use kkc_data::{PackedBatch, PrefetchedBatchIter};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

/// Work unit sent to [`AsyncCheckpointWriter`]. `bytes` is whatever the caller
/// wants persisted — typically a JSON blob for trainer state or a safetensors
/// payload for weights. The writer is agnostic; it just writes.
#[derive(Debug)]
pub struct CheckpointWrite {
    pub path: PathBuf,
    pub bytes: Vec<u8>,
    /// Optional sidecar file written alongside `path`. The training loop uses
    /// this for the per-step `*.ckpt.json` companion next to
    /// `*.backend.json` / `*.safetensors`.
    pub sidecar: Option<(PathBuf, Vec<u8>)>,
}

pub struct AsyncCheckpointWriter {
    tx: Option<SyncSender<CheckpointWrite>>,
    handle: Option<JoinHandle<Result<usize>>>,
}

impl AsyncCheckpointWriter {
    /// Spawn the writer. `queue_size` bounds how many writes can be in flight
    /// before `submit` blocks; values of 1-4 are typical. The training loop
    /// should still call [`AsyncCheckpointWriter::finish`] before exit so
    /// errors surface synchronously; otherwise `drop` swallows them.
    pub fn spawn(queue_size: usize) -> Self {
        let (tx, rx) = sync_channel::<CheckpointWrite>(queue_size.max(1));
        let handle = thread::Builder::new()
            .name("kkc-ckpt-writer".to_string())
            .spawn(move || writer_loop(rx))
            .expect("spawn checkpoint writer thread");
        Self {
            tx: Some(tx),
            handle: Some(handle),
        }
    }

    /// Submit a write. Blocks only if the queue is full. If the writer thread
    /// has already exited (poisoned), the send fails and the caller gets an
    /// error so the failure is not silently dropped.
    #[allow(dead_code)]
    pub fn submit(&self, work: CheckpointWrite) -> Result<()> {
        let tx = self
            .tx
            .as_ref()
            .context("checkpoint writer already finished")?;
        tx.send(work)
            .map_err(|_| anyhow::anyhow!("checkpoint writer thread is gone"))?;
        Ok(())
    }

    /// Borrow a clone-able sender so owners (e.g. `TchCtcNatBackend`)
    /// can drain bytes directly into the writer thread without having
    /// to hold a reference to the full `AsyncCheckpointWriter`.
    pub fn sender(&self) -> Option<SyncSender<CheckpointWrite>> {
        self.tx.as_ref().cloned()
    }

    /// Close the channel and join the writer thread. Returns the number of
    /// writes completed. Call this before program exit when possible so
    /// propagated I/O errors are not lost.
    pub fn finish(mut self) -> Result<usize> {
        drop(self.tx.take());
        match self.handle.take() {
            Some(h) => h
                .join()
                .map_err(|_| anyhow::anyhow!("checkpoint writer thread panicked"))?,
            None => Ok(0),
        }
    }
}

impl Drop for AsyncCheckpointWriter {
    fn drop(&mut self) {
        // Closing the channel is the signal for the writer to stop. We still
        // join so pending writes are flushed before this type disappears.
        drop(self.tx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn writer_loop(rx: Receiver<CheckpointWrite>) -> Result<usize> {
    let mut written = 0usize;
    while let Ok(work) = rx.recv() {
        persist_one(&work)
            .with_context(|| format!("checkpoint writer failed on {}", work.path.display()))?;
        written += 1;
    }
    Ok(written)
}

fn persist_one(work: &CheckpointWrite) -> Result<()> {
    if let Some(parent) = work.path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create dir {}", parent.display()))?;
        }
    }
    atomic_write(&work.path, &work.bytes)?;
    if let Some((sidecar_path, sidecar_bytes)) = &work.sidecar {
        atomic_write(sidecar_path, sidecar_bytes)?;
    }
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    // Write to `<path>.tmp` then rename. On Windows `rename` across drives
    // fails, but checkpoints always share the run_dir so same-volume.
    let tmp = path.with_extension(temp_suffix(path));
    std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

fn temp_suffix(path: &Path) -> String {
    let existing = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    if existing.is_empty() {
        "kkctmp".to_string()
    } else {
        format!("{existing}.kkctmp")
    }
}

/// Trait for anything that can hand the trainer its next batch. The GPU
/// path (`gpu::batch::StagedBatchPipeline`) implements this so the tch
/// backend can share the generic prefetch plumbing. In the default
/// (no-cuda) build the trait is unused but kept in the public API so
/// `gpu` module internals don't need to re-declare it behind a cfg.
#[allow(dead_code)]
pub trait BatchProducer {
    type Item;
    fn next_item(&mut self) -> Result<Option<Self::Item>>;
}

/// Thin wrapper so the existing `PrefetchedBatchIter` (which yields
/// `PackedBatch` on CPU) plugs into the staged pipeline on builds
/// without `cuda`. Used by integration tests; production build routes
/// through the cuda-gated pipeline.
#[allow(dead_code)]
pub struct CpuStagedPipeline {
    inner: PrefetchedBatchIter,
}

#[allow(dead_code)]
impl CpuStagedPipeline {
    pub fn new(inner: PrefetchedBatchIter) -> Self {
        Self { inner }
    }
}

impl BatchProducer for CpuStagedPipeline {
    type Item = PackedBatch;

    fn next_item(&mut self) -> Result<Option<PackedBatch>> {
        self.inner.next_batch()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tempfile::tempdir;

    #[test]
    fn writer_flushes_submitted_files() {
        let dir = tempdir().unwrap();
        let writer = AsyncCheckpointWriter::spawn(2);
        for step in 0..4usize {
            writer
                .submit(CheckpointWrite {
                    path: dir.path().join(format!("step_{step}.bin")),
                    bytes: vec![step as u8; 8],
                    sidecar: None,
                })
                .unwrap();
        }
        let written = writer.finish().unwrap();
        assert_eq!(written, 4);
        for step in 0..4usize {
            let got = std::fs::read(dir.path().join(format!("step_{step}.bin"))).unwrap();
            assert_eq!(got, vec![step as u8; 8]);
        }
    }

    #[test]
    fn writer_preserves_submission_order() {
        let dir = tempdir().unwrap();
        let writer = AsyncCheckpointWriter::spawn(4);
        // Overwrite the same path several times. Thanks to FIFO delivery the
        // final file must reflect the last submission.
        let target = dir.path().join("latest.bin");
        for step in 0u8..16 {
            writer
                .submit(CheckpointWrite {
                    path: target.clone(),
                    bytes: vec![step; 4],
                    sidecar: None,
                })
                .unwrap();
        }
        writer.finish().unwrap();
        let got = std::fs::read(&target).unwrap();
        assert_eq!(got, vec![15u8; 4]);
    }

    #[test]
    fn writer_writes_sidecar_alongside_main_payload() {
        let dir = tempdir().unwrap();
        let writer = AsyncCheckpointWriter::spawn(1);
        writer
            .submit(CheckpointWrite {
                path: dir.path().join("state.backend.json"),
                bytes: b"backend".to_vec(),
                sidecar: Some((dir.path().join("state.ckpt.json"), b"trainer".to_vec())),
            })
            .unwrap();
        writer.finish().unwrap();
        assert_eq!(
            std::fs::read(dir.path().join("state.backend.json")).unwrap(),
            b"backend"
        );
        assert_eq!(
            std::fs::read(dir.path().join("state.ckpt.json")).unwrap(),
            b"trainer"
        );
    }

    #[test]
    fn drop_flushes_without_finish() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("dropped.bin");
        {
            let writer = AsyncCheckpointWriter::spawn(1);
            writer
                .submit(CheckpointWrite {
                    path: target.clone(),
                    bytes: b"flushed-on-drop".to_vec(),
                    sidecar: None,
                })
                .unwrap();
            // explicit drop
        }
        assert_eq!(std::fs::read(&target).unwrap(), b"flushed-on-drop");
    }

    #[test]
    fn batch_producer_cpu_path_is_identity() {
        use kkc_data::{
            compile_jsonl_to_shard, BatchIter, BatchIterConfig, CompileOptions,
            PrefetchedBatchIter,
        };
        use kkc_tokenizer::SharedCharTokenizer;

        let dir = tempdir().unwrap();
        let jsonl = dir.path().join("t.jsonl");
        let mut body = String::new();
        for i in 0..4 {
            body.push_str(&format!(
                "{{\"reading\":\"あ{}\",\"surface\":\"亜{}\",\"context\":\"\"}}\n",
                i, i
            ));
        }
        std::fs::write(&jsonl, body).unwrap();
        let shard = dir.path().join("t.kkc");
        let tok = SharedCharTokenizer::new_default(64);
        compile_jsonl_to_shard(&jsonl, &shard, &tok, &CompileOptions::default()).unwrap();
        let iter = BatchIter::open(
            &shard,
            BatchIterConfig {
                batch_size: 2,
                block_rows: 4,
                ..BatchIterConfig::default()
            },
        )
        .unwrap();
        let prefetched = PrefetchedBatchIter::spawn(iter, 2);
        let mut pipeline = CpuStagedPipeline::new(prefetched);
        let mut seen = 0usize;
        while let Some(batch) = pipeline.next_item().unwrap() {
            assert!(batch.batch_size > 0);
            seen += batch.batch_size;
        }
        assert_eq!(seen, 4);
    }

    #[test]
    fn finish_on_empty_writer_reports_zero() {
        let writer = AsyncCheckpointWriter::spawn(1);
        assert_eq!(writer.finish().unwrap(), 0);
    }

    #[test]
    fn many_rapid_submits_do_not_lose_writes() {
        let dir = tempdir().unwrap();
        let writer = AsyncCheckpointWriter::spawn(2);
        let total = 64usize;
        let counter = Arc::new(AtomicUsize::new(0));
        for step in 0..total {
            writer
                .submit(CheckpointWrite {
                    path: dir.path().join(format!("rapid_{step}.bin")),
                    bytes: (step as u64).to_le_bytes().to_vec(),
                    sidecar: None,
                })
                .unwrap();
            counter.fetch_add(1, Ordering::Relaxed);
        }
        assert_eq!(counter.load(Ordering::Relaxed), total);
        assert_eq!(writer.finish().unwrap(), total);
        for step in 0..total {
            assert!(dir.path().join(format!("rapid_{step}.bin")).exists());
        }
    }
}
