use crate::iter::{BatchIter, BatchIterConfig};
use crate::shard::ShardReader;
use anyhow::Result;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct SequenceBudget {
    pub batch_size: usize,
    pub max_input_len: usize,
    pub max_target_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct BatchPlan {
    pub input_token_bytes: usize,
    pub target_token_bytes: usize,
    pub attention_bytes: usize,
    pub per_batch_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DatasetStats {
    pub rows: usize,
    pub mean_input_tokens: f64,
    pub mean_target_tokens: f64,
    pub max_input_tokens: usize,
    pub max_target_tokens: usize,
    pub sample_batch_bytes: usize,
}

impl SequenceBudget {
    pub fn estimate(&self) -> BatchPlan {
        let input_token_bytes = self.batch_size * self.max_input_len * std::mem::size_of::<u32>();
        let target_token_bytes = self.batch_size * self.max_target_len * std::mem::size_of::<u32>();
        let attention_bytes = self.batch_size * self.max_input_len;
        BatchPlan {
            input_token_bytes,
            target_token_bytes,
            attention_bytes,
            per_batch_bytes: input_token_bytes + target_token_bytes + attention_bytes,
        }
    }
}

pub fn inspect_shard(path: impl AsRef<Path>, sample_rows: usize) -> Result<DatasetStats> {
    let reader = ShardReader::open(path)?;
    let rows = reader.header().row_count as usize;
    let limit = sample_rows.min(rows).max(1);
    let mut total_input = 0usize;
    let mut total_target = 0usize;
    let mut max_input = 0usize;
    let mut max_target = 0usize;
    for idx in 0..limit {
        let row = reader.row(idx)?;
        let input = row.context.len() + row.reading.len() + 2;
        let target = row.surface.len();
        total_input += input;
        total_target += target;
        max_input = max_input.max(input);
        max_target = max_target.max(target);
    }
    Ok(DatasetStats {
        rows,
        mean_input_tokens: total_input as f64 / limit as f64,
        mean_target_tokens: total_target as f64 / limit as f64,
        max_input_tokens: max_input,
        max_target_tokens: max_target,
        sample_batch_bytes: 0,
    })
}

pub fn inspect_shard_batches(
    path: impl AsRef<Path>,
    config: BatchIterConfig,
    sample_batches: usize,
) -> Result<DatasetStats> {
    let mut stats = inspect_shard(&path, config.block_rows.max(config.batch_size))?;
    let mut iter = BatchIter::open(path, config)?;
    let mut total = 0usize;
    let mut seen = 0usize;
    while seen < sample_batches {
        let Some(batch) = iter.next_batch()? else {
            break;
        };
        total += batch.bytes();
        seen += 1;
    }
    if seen > 0 {
        stats.sample_batch_bytes = total / seen;
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_estimate_is_stable() {
        let plan = SequenceBudget {
            batch_size: 8,
            max_input_len: 128,
            max_target_len: 64,
        }
        .estimate();
        assert_eq!(plan.input_token_bytes, 4096);
        assert_eq!(plan.target_token_bytes, 2048);
        assert_eq!(plan.attention_bytes, 1024);
    }
}
