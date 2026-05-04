//! Background prefetch for `BatchIter`.
//!
//! Owns the iterator on a worker thread, serves ready `PackedBatch`es via a
//! bounded channel. The caller's hot loop is GPU-side (build tensors, run
//! forward / backward), and this keeps batch construction off the critical
//! path. For a single trainer process this is a drop-in replacement for
//! "call `next_batch()` synchronously" without the per-step CPU wait.

use crate::iter::{BatchIter, PackedBatch};
use anyhow::Result;
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread::{self, JoinHandle};

pub struct PrefetchedBatchIter {
    rx: Receiver<Result<PackedBatch>>,
    handle: Option<JoinHandle<()>>,
}

impl PrefetchedBatchIter {
    /// Move `iter` onto a producer thread. `queue_size` bounds how far ahead
    /// the producer runs; typical values 2-8. A value of 0 disables
    /// prefetching entirely (the producer thread waits on each send).
    pub fn spawn(iter: BatchIter, queue_size: usize) -> Self {
        let (tx, rx) = sync_channel(queue_size.max(1));
        let handle = thread::spawn(move || {
            let mut iter = iter;
            loop {
                match iter.next_batch() {
                    Ok(Some(batch)) => {
                        if tx.send(Ok(batch)).is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        break;
                    }
                }
            }
        });
        Self {
            rx,
            handle: Some(handle),
        }
    }

    /// Block until the next batch is available. Returns `Ok(None)` when the
    /// underlying iterator is exhausted.
    pub fn next_batch(&mut self) -> Result<Option<PackedBatch>> {
        match self.rx.recv() {
            Ok(result) => result.map(Some),
            Err(_) => Ok(None),
        }
    }
}

impl Drop for PrefetchedBatchIter {
    fn drop(&mut self) {
        // Dropping `rx` closes the channel, so the producer thread's next
        // `send` fails and it exits the loop naturally. Still, join so we
        // don't orphan a worker on program exit.
        drop(std::mem::replace(
            &mut self.rx,
            sync_channel::<Result<PackedBatch>>(1).1,
        ));
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{compile_jsonl_to_shard, CompileOptions};
    use crate::iter::BatchIterConfig;
    use rust_tokenizer::SharedCharTokenizer;

    #[test]
    fn prefetched_iter_yields_same_sequence_as_synchronous() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl = dir.path().join("t.jsonl");
        // Emit 8 rows so several batches are produced.
        let mut content = String::new();
        for i in 0..8 {
            content.push_str(&format!(
                "{{\"reading\":\"かな{}\",\"surface\":\"仮名{}\",\"context\":\"\"}}\n",
                i, i
            ));
        }
        std::fs::write(&jsonl, content).unwrap();
        let shard = dir.path().join("t.kkc");
        let tok = SharedCharTokenizer::new_default(64);
        compile_jsonl_to_shard(&jsonl, &shard, &tok, &CompileOptions::default()).unwrap();

        let cfg = BatchIterConfig {
            batch_size: 2,
            block_rows: 2,
            seed: 123,
            ..BatchIterConfig::default()
        };
        let mut sync_iter = BatchIter::open(&shard, cfg.clone()).unwrap();
        let mut sync_batches = Vec::new();
        while let Some(batch) = sync_iter.next_batch().unwrap() {
            sync_batches.push(batch.input_lengths.clone());
        }

        let async_iter = BatchIter::open(&shard, cfg).unwrap();
        let mut prefetched = PrefetchedBatchIter::spawn(async_iter, 4);
        let mut async_batches = Vec::new();
        while let Some(batch) = prefetched.next_batch().unwrap() {
            async_batches.push(batch.input_lengths.clone());
        }

        assert_eq!(
            sync_batches, async_batches,
            "prefetched iter must produce identical batches in identical order"
        );
    }

    #[test]
    fn dropping_consumer_shuts_down_producer() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl = dir.path().join("t.jsonl");
        let mut content = String::new();
        for i in 0..32 {
            content.push_str(&format!(
                "{{\"reading\":\"かな{}\",\"surface\":\"仮名{}\",\"context\":\"\"}}\n",
                i, i
            ));
        }
        std::fs::write(&jsonl, content).unwrap();
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
        drop(prefetched);
        // If producer shutdown failed we'd leak the thread and the test
        // runner would block on join in Drop. Passing this assert means the
        // drop-join path works.
    }
}
