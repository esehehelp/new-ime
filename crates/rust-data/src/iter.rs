use crate::shard::ShardReader;
use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct BatchIterConfig {
    pub batch_size: usize,
    pub max_input_len: usize,
    pub max_target_len: usize,
    pub block_rows: usize,
    pub seed: u64,
    pub drop_last: bool,
}

impl Default for BatchIterConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_input_len: 128,
            max_target_len: 128,
            block_rows: 4096,
            seed: 42,
            drop_last: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PackedBatch {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub target_ids: Vec<u32>,
    pub input_lengths: Vec<u16>,
    pub target_lengths: Vec<u16>,
    pub writer_ids: Vec<u32>,
    pub domain_ids: Vec<u32>,
    pub source_ids: Vec<u32>,
    pub batch_size: usize,
    pub max_input_len: usize,
    pub max_target_len: usize,
    pub order_cursor: usize,
}

impl PackedBatch {
    pub fn bytes(&self) -> usize {
        self.input_ids.len() * std::mem::size_of::<u32>()
            + self.attention_mask.len()
            + self.target_ids.len() * std::mem::size_of::<u32>()
            + self.input_lengths.len() * std::mem::size_of::<u16>()
            + self.target_lengths.len() * std::mem::size_of::<u16>()
            + self.writer_ids.len() * std::mem::size_of::<u32>()
            + self.domain_ids.len() * std::mem::size_of::<u32>()
            + self.source_ids.len() * std::mem::size_of::<u32>()
    }

    pub fn non_padding_input_tokens(&self) -> usize {
        self.attention_mask.iter().map(|v| *v as usize).sum()
    }

    pub fn non_padding_target_tokens(&self) -> usize {
        self.target_lengths.iter().map(|v| *v as usize).sum()
    }
}

pub struct BatchIter {
    reader: ShardReader,
    config: BatchIterConfig,
    order: Vec<usize>,
    cursor: usize,
    /// When set, rows whose reading token count exceeds this are skipped.
    /// Used for the "short-sample warmup" that stabilizes CTC training
    /// during the first ~1000 steps (legacy Python parity).
    max_reading_tokens: Option<usize>,
}

impl BatchIter {
    pub fn open(path: impl AsRef<Path>, config: BatchIterConfig) -> Result<Self> {
        Self::open_at_cursor(path, config, 0)
    }

    pub fn open_at_cursor(
        path: impl AsRef<Path>,
        config: BatchIterConfig,
        cursor: usize,
    ) -> Result<Self> {
        let reader = ShardReader::open(path).context("open shard reader")?;
        let rows = reader.header().row_count as usize;
        let block = config.block_rows.max(config.batch_size).max(1);
        let mut order = Vec::with_capacity(rows);
        let mut starts = Vec::new();
        let mut start = 0usize;
        while start < rows {
            starts.push(start);
            start += block;
        }
        let mut rng = StdRng::seed_from_u64(config.seed);
        starts.shuffle(&mut rng);
        for block_start in starts {
            let mut row_ids = (block_start..(block_start + block).min(rows)).collect::<Vec<_>>();
            row_ids.shuffle(&mut rng);
            order.extend(row_ids);
        }
        Ok(Self {
            reader,
            config,
            order,
            cursor: cursor.min(rows),
            max_reading_tokens: None,
        })
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Enable/disable the short-sample filter. `Some(n)` skips rows whose
    /// `reading` token count exceeds `n`; `None` accepts every row. Safe
    /// to toggle between batches; the next call to `next_batch` honors
    /// the current setting.
    pub fn set_max_reading_tokens(&mut self, max: Option<usize>) {
        self.max_reading_tokens = max;
    }

    pub fn next_batch(&mut self) -> Result<Option<PackedBatch>> {
        if self.cursor >= self.order.len() {
            return Ok(None);
        }
        let batch_size = self.config.batch_size;
        let mut rows = Vec::with_capacity(batch_size);
        let mut max_input = 0usize;
        let mut max_target = 1usize;
        // With the short-sample filter active we advance `cursor` past
        // rejected rows until we have `batch_size` accepted rows (or the
        // shard is exhausted). Without it the loop exits after the first
        // block that fills the batch, matching the legacy behavior.
        while rows.len() < batch_size && self.cursor < self.order.len() {
            let idx = self.order[self.cursor];
            self.cursor += 1;
            let row = self.reader.row(idx)?;
            if let Some(max_tok) = self.max_reading_tokens {
                if row.reading.len() > max_tok {
                    continue;
                }
            }
            let context_budget = self.config.max_input_len.saturating_sub(2);
            let context_take = row.context.len().min(context_budget);
            let reading_take = row
                .reading
                .len()
                .min(context_budget.saturating_sub(context_take));
            let input_len = 2 + context_take + reading_take;
            let target_len = row.surface.len().min(self.config.max_target_len);
            max_input = max_input.max(input_len);
            max_target = max_target.max(target_len.max(1));
            rows.push((row, context_take, reading_take, target_len));
        }
        if rows.is_empty() {
            return Ok(None);
        }
        if rows.len() < batch_size && self.config.drop_last {
            return Ok(None);
        }
        let take = rows.len();

        let mut input_ids = vec![0u32; take * max_input];
        let mut attention_mask = vec![0u8; take * max_input];
        let mut target_ids = vec![0u32; take * max_target];
        let mut input_lengths = Vec::with_capacity(take);
        let mut target_lengths = Vec::with_capacity(take);
        let mut writer_ids = Vec::with_capacity(take);
        let mut domain_ids = Vec::with_capacity(take);
        let mut source_ids = Vec::with_capacity(take);

        for (row_idx, (row, context_take, reading_take, target_len)) in rows.into_iter().enumerate()
        {
            let input_base = row_idx * max_input;
            let target_base = row_idx * max_target;
            let mut pos = 0usize;
            input_ids[input_base + pos] = 3;
            attention_mask[input_base + pos] = 1;
            pos += 1;
            for token in row.context.iter().take(context_take) {
                input_ids[input_base + pos] = *token;
                attention_mask[input_base + pos] = 1;
                pos += 1;
            }
            input_ids[input_base + pos] = 2;
            attention_mask[input_base + pos] = 1;
            pos += 1;
            for token in row.reading.iter().take(reading_take) {
                input_ids[input_base + pos] = *token;
                attention_mask[input_base + pos] = 1;
                pos += 1;
            }
            for (i, token) in row.surface.iter().take(target_len).enumerate() {
                target_ids[target_base + i] = *token;
            }
            input_lengths.push(pos as u16);
            target_lengths.push(target_len as u16);
            writer_ids.push(row.writer_id);
            domain_ids.push(row.domain_id);
            source_ids.push(row.source_id);
        }

        Ok(Some(PackedBatch {
            input_ids,
            attention_mask,
            target_ids,
            input_lengths,
            target_lengths,
            writer_ids,
            domain_ids,
            source_ids,
            batch_size: take,
            max_input_len: max_input,
            max_target_len: max_target,
            order_cursor: self.cursor,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{compile_jsonl_to_shard, CompileOptions};
    use rust_tokenizer::SharedCharTokenizer;

    #[test]
    fn builds_packed_batches_from_shard() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl = dir.path().join("train.jsonl");
        std::fs::write(
            &jsonl,
            "{\"reading\": \"かな\", \"surface\": \"仮名\", \"context\": \"前\", \"source\": \"a\"}\n\
             {\"reading\": \"きょう\", \"surface\": \"今日\", \"context\": \"\", \"source\": \"b\"}\n",
        )
        .unwrap();
        let shard = dir.path().join("train.kkc");
        let tok = SharedCharTokenizer::new_default(128);
        compile_jsonl_to_shard(&jsonl, &shard, &tok, &CompileOptions::default()).unwrap();
        let mut iter = BatchIter::open(
            &shard,
            BatchIterConfig {
                batch_size: 2,
                block_rows: 2,
                ..BatchIterConfig::default()
            },
        )
        .unwrap();
        let batch = iter.next_batch().unwrap().unwrap();
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.input_lengths.len(), 2);
        assert_eq!(batch.target_lengths.len(), 2);
        assert!(batch.bytes() > 0);
        assert_eq!(batch.order_cursor, 2);
    }

    #[test]
    fn can_resume_from_saved_cursor() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl = dir.path().join("train.jsonl");
        std::fs::write(
            &jsonl,
            "{\"reading\": \"かな\", \"surface\": \"仮名\", \"context\": \"前\", \"source\": \"a\"}\n\
             {\"reading\": \"きょう\", \"surface\": \"今日\", \"context\": \"\", \"source\": \"b\"}\n\
             {\"reading\": \"あした\", \"surface\": \"明日\", \"context\": \"\", \"source\": \"c\"}\n",
        )
        .unwrap();
        let shard = dir.path().join("train.kkc");
        let tok = SharedCharTokenizer::new_default(128);
        compile_jsonl_to_shard(&jsonl, &shard, &tok, &CompileOptions::default()).unwrap();
        let cfg = BatchIterConfig {
            batch_size: 1,
            block_rows: 1,
            seed: 7,
            ..BatchIterConfig::default()
        };
        let mut iter = BatchIter::open(&shard, cfg.clone()).unwrap();
        let first = iter.next_batch().unwrap().unwrap();
        let saved_cursor = first.order_cursor;
        let second = iter.next_batch().unwrap().unwrap();

        let mut resumed = BatchIter::open_at_cursor(&shard, cfg, saved_cursor).unwrap();
        let resumed_second = resumed.next_batch().unwrap().unwrap();
        assert_eq!(second.input_ids, resumed_second.input_ids);
        assert_eq!(second.target_ids, resumed_second.target_ids);
    }
}
