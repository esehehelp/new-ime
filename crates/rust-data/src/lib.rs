pub mod batch;
pub mod compile;
pub mod iter;
pub mod prefetch;
pub mod shard;

pub use batch::{inspect_shard, inspect_shard_batches, BatchPlan, DatasetStats, SequenceBudget};
pub use compile::{compile_jsonl_to_shard, CompileOptions, ShardMetadata};
pub use iter::{BatchIter, BatchIterConfig, PackedBatch};
pub use prefetch::PrefetchedBatchIter;
pub use shard::{ShardHeader, ShardReader, ShardRowRef, MAGIC, VERSION};
