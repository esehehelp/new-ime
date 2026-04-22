use crate::pipeline::CheckpointWrite;
use crate::{CheckpointEntry, TrainerState};
use anyhow::{Context, Result};
use kkc_data::PackedBatch;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::mpsc::SyncSender;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct TrainerLoopConfig {
    pub target_step: usize,
    pub checkpoint_every: usize,
    pub epoch_steps: usize,
    pub grad_accum: usize,
    pub checkpoint_keep_last: usize,
    /// If present, prune routes its deletes through this sender instead
    /// of running `std::fs::remove_file` inline. FIFO ordering with
    /// prior async saves is what prevents the "prune a file not yet
    /// flushed" race for the tch backend.
    pub ckpt_sender: Option<SyncSender<CheckpointWrite>>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct TrainerStep {
    pub loss: f64,
    pub rows: usize,
    pub bytes: usize,
    pub input_tokens: usize,
    pub target_tokens: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TrainerSummary {
    pub final_step: usize,
    pub final_epoch: usize,
    pub elapsed_sec: f64,
    pub steps_per_sec: f64,
    pub last_loss: Option<f64>,
}

pub trait BatchStream {
    fn next_batch(&mut self) -> Result<Option<PackedBatch>>;
}

pub fn run_training_loop<S, B>(
    source: &mut S,
    backend: &mut B,
    state: &mut TrainerState,
    run_dir: &Path,
    config: TrainerLoopConfig,
) -> Result<TrainerSummary>
where
    S: BatchStream,
    B: crate::backend::TrainBackend + ?Sized,
{
    let started = Instant::now();
    let mut last_loss = None;
    let start_step = state.step;
    while state.step < config.target_step {
        let Some(batch) = next_accumulated_batch(source, config.grad_accum)? else {
            break;
        };
        state.step += 1;
        state.epoch = state.step / config.epoch_steps.max(1);
        state.data_cursor = batch.order_cursor;
        let step_out = backend.step(state.step, &batch)?;
        last_loss = Some(step_out.loss);
        if config.checkpoint_every > 0 && state.step % config.checkpoint_every == 0 {
            write_backend_checkpoint(run_dir, state, step_out, backend)?;
            prune_checkpoints(
                run_dir,
                state,
                config.checkpoint_keep_last,
                config.ckpt_sender.as_ref(),
            )?;
        }
    }
    let elapsed_sec = started.elapsed().as_secs_f64().max(1e-9);
    let executed_steps = state.step.saturating_sub(start_step);
    Ok(TrainerSummary {
        final_step: state.step,
        final_epoch: state.epoch,
        elapsed_sec,
        steps_per_sec: executed_steps as f64 / elapsed_sec,
        last_loss,
    })
}

fn next_accumulated_batch<S>(source: &mut S, grad_accum: usize) -> Result<Option<PackedBatch>>
where
    S: BatchStream,
{
    let accum = grad_accum.max(1);
    let Some(mut merged) = source.next_batch()? else {
        return Ok(None);
    };
    for _ in 1..accum {
        let Some(next) = source.next_batch()? else {
            break;
        };
        merge_batches(&mut merged, next);
    }
    Ok(Some(merged))
}

fn merge_batches(dst: &mut PackedBatch, src: PackedBatch) {
    let merged_input_len = dst.max_input_len.max(src.max_input_len);
    let merged_target_len = dst.max_target_len.max(src.max_target_len);
    let total_rows = dst.batch_size + src.batch_size;

    let mut merged_input_ids = Vec::with_capacity(total_rows * merged_input_len);
    let mut merged_attention = Vec::with_capacity(total_rows * merged_input_len);
    repack_rows(
        &mut merged_input_ids,
        &dst.input_ids,
        dst.batch_size,
        dst.max_input_len,
        merged_input_len,
        0,
    );
    repack_rows(
        &mut merged_input_ids,
        &src.input_ids,
        src.batch_size,
        src.max_input_len,
        merged_input_len,
        0,
    );
    repack_rows(
        &mut merged_attention,
        &dst.attention_mask,
        dst.batch_size,
        dst.max_input_len,
        merged_input_len,
        0,
    );
    repack_rows(
        &mut merged_attention,
        &src.attention_mask,
        src.batch_size,
        src.max_input_len,
        merged_input_len,
        0,
    );

    let mut merged_target_ids = Vec::with_capacity(total_rows * merged_target_len);
    repack_rows(
        &mut merged_target_ids,
        &dst.target_ids,
        dst.batch_size,
        dst.max_target_len,
        merged_target_len,
        0,
    );
    repack_rows(
        &mut merged_target_ids,
        &src.target_ids,
        src.batch_size,
        src.max_target_len,
        merged_target_len,
        0,
    );

    dst.input_ids = merged_input_ids;
    dst.attention_mask = merged_attention;
    dst.target_ids = merged_target_ids;
    dst.input_lengths.extend(src.input_lengths);
    dst.target_lengths.extend(src.target_lengths);
    dst.source_ids.extend(src.source_ids);
    dst.batch_size += src.batch_size;
    dst.max_input_len = merged_input_len;
    dst.max_target_len = merged_target_len;
    dst.order_cursor = src.order_cursor;
}

fn repack_rows<T: Copy>(
    out: &mut Vec<T>,
    src: &[T],
    rows: usize,
    src_width: usize,
    dst_width: usize,
    pad: T,
) {
    for row_idx in 0..rows {
        let start = row_idx * src_width;
        let end = start + src_width;
        out.extend_from_slice(&src[start..end]);
        out.extend(std::iter::repeat_n(
            pad,
            dst_width.saturating_sub(src_width),
        ));
    }
}

fn write_backend_checkpoint<B: crate::backend::TrainBackend + ?Sized>(
    run_dir: &Path,
    state: &mut TrainerState,
    step: TrainerStep,
    backend: &mut B,
) -> Result<()> {
    let checkpoint_path = run_dir.join(format!("step_{:08}.ckpt.json", state.step));
    let backend_path = run_dir.join(format!("step_{:08}.backend.json", state.step));
    std::fs::write(
        &checkpoint_path,
        serde_json::to_vec_pretty(&step).context("serialize trainer checkpoint")?,
    )
    .with_context(|| format!("write {}", checkpoint_path.display()))?;
    backend.save_checkpoint(&backend_path)?;
    let checkpoint_str = checkpoint_path.display().to_string();
    state.last_checkpoint = Some(checkpoint_str.clone());
    state.checkpoints.push(CheckpointEntry {
        step: state.step,
        epoch: state.epoch,
        checkpoint: checkpoint_str,
        metric: Some(step.loss),
        kind: backend.kind().to_string(),
        metric_mode: "minimize".to_string(),
    });
    state.best_metric = match state.best_metric {
        Some(current) if current <= step.loss => Some(current),
        _ => {
            state.best_checkpoint = state.last_checkpoint.clone();
            Some(step.loss)
        }
    };
    Ok(())
}

fn prune_checkpoints(
    _run_dir: &Path,
    state: &mut TrainerState,
    keep_last: usize,
    ckpt_sender: Option<&SyncSender<CheckpointWrite>>,
) -> Result<()> {
    let keep_last = keep_last.max(1);
    if state.checkpoints.len() <= keep_last {
        return Ok(());
    }
    let best = state.best_checkpoint.clone();
    let latest_start = state.checkpoints.len().saturating_sub(keep_last);
    let mut retained = Vec::with_capacity(state.checkpoints.len());
    let mut pruned = Vec::new();
    for (idx, entry) in state.checkpoints.drain(..).enumerate() {
        let is_latest = idx >= latest_start;
        let is_best = best
            .as_deref()
            .map(|path| path == entry.checkpoint)
            .unwrap_or(false);
        if is_latest || is_best {
            retained.push(entry);
        } else {
            pruned.push(entry);
        }
    }
    state.checkpoints = retained;
    for entry in pruned {
        // Collect every file produced for this checkpoint step. The
        // GPU path emits `.weights.safetensors` alongside
        // `.backend.json`; retaining it would leak orphan weights
        // files across retention cycles.
        let mut paths: Vec<PathBuf> = vec![
            PathBuf::from(&entry.checkpoint),
            checkpoint_sidecar_path(&entry.checkpoint, ".ckpt.json", ".backend.json"),
            checkpoint_sidecar_path(
                &entry.checkpoint,
                ".ckpt.json",
                ".weights.safetensors",
            ),
            checkpoint_sidecar_path(&entry.checkpoint, ".ckpt.json", ".optim.safetensors"),
        ];
        // Drop duplicates if the above happens to collide on identical
        // paths (shouldn't today but defensive).
        paths.sort();
        paths.dedup();

        if let Some(sender) = ckpt_sender {
            // Enqueue as a Delete op so FIFO ordering with the async
            // save from the same step is preserved — otherwise prune
            // would race a still-queued safetensors write.
            sender
                .send(CheckpointWrite::Delete { paths })
                .map_err(|_| anyhow::anyhow!("checkpoint writer thread is gone"))?;
        } else {
            for p in paths {
                remove_if_exists(&p)?;
            }
        }
    }
    Ok(())
}

fn remove_if_exists(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err).with_context(|| format!("remove {}", path.display())),
    }
}

fn checkpoint_sidecar_path(
    checkpoint: &str,
    from_suffix: &str,
    to_suffix: &str,
) -> std::path::PathBuf {
    let path = std::path::PathBuf::from(checkpoint);
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("checkpoint");
    let replaced = file_name
        .strip_suffix(from_suffix)
        .map(|stem| format!("{}{}", stem, to_suffix))
        .unwrap_or_else(|| format!("{}{}", file_name, to_suffix));
    path.with_file_name(replaced)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn merge_batches_combines_micro_batches() {
        let mut a = PackedBatch {
            input_ids: vec![1, 2, 0, 0],
            attention_mask: vec![1, 1, 0, 0],
            target_ids: vec![3, 0],
            input_lengths: vec![2],
            target_lengths: vec![1],
            source_ids: vec![10],
            batch_size: 1,
            max_input_len: 4,
            max_target_len: 2,
            order_cursor: 1,
        };
        let b = PackedBatch {
            input_ids: vec![4, 5, 6, 0],
            attention_mask: vec![1, 1, 1, 0],
            target_ids: vec![7, 8],
            input_lengths: vec![3],
            target_lengths: vec![2],
            source_ids: vec![11],
            batch_size: 1,
            max_input_len: 4,
            max_target_len: 2,
            order_cursor: 2,
        };
        merge_batches(&mut a, b);
        assert_eq!(a.batch_size, 2);
        assert_eq!(a.input_ids.len(), 8);
        assert_eq!(a.target_ids.len(), 4);
        assert_eq!(a.order_cursor, 2);
    }

    #[test]
    fn merge_batches_repads_when_widths_differ() {
        let mut a = PackedBatch {
            input_ids: vec![1, 2],
            attention_mask: vec![1, 1],
            target_ids: vec![3],
            input_lengths: vec![2],
            target_lengths: vec![1],
            source_ids: vec![10],
            batch_size: 1,
            max_input_len: 2,
            max_target_len: 1,
            order_cursor: 1,
        };
        let b = PackedBatch {
            input_ids: vec![4, 5, 6],
            attention_mask: vec![1, 1, 1],
            target_ids: vec![7, 8],
            input_lengths: vec![3],
            target_lengths: vec![2],
            source_ids: vec![11],
            batch_size: 1,
            max_input_len: 3,
            max_target_len: 2,
            order_cursor: 2,
        };
        merge_batches(&mut a, b);
        assert_eq!(a.max_input_len, 3);
        assert_eq!(a.max_target_len, 2);
        assert_eq!(a.input_ids, vec![1, 2, 0, 4, 5, 6]);
        assert_eq!(a.target_ids, vec![3, 0, 7, 8]);
    }

    #[test]
    fn prune_keeps_latest_and_best_checkpoint() {
        let dir = tempdir().unwrap();
        let keep = |name: &str| dir.path().join(name);
        for name in [
            "step_00000001.ckpt.json",
            "step_00000001.backend.json",
            "step_00000001.weights.safetensors",
            "step_00000002.ckpt.json",
            "step_00000002.backend.json",
            "step_00000002.weights.safetensors",
            "step_00000003.ckpt.json",
            "step_00000003.backend.json",
            "step_00000003.weights.safetensors",
        ] {
            std::fs::write(keep(name), b"x").unwrap();
        }
        let mut state = TrainerState {
            step: 3,
            epoch: 0,
            data_cursor: 0,
            best_metric: Some(0.1),
            best_checkpoint: Some(keep("step_00000001.ckpt.json").display().to_string()),
            last_checkpoint: Some(keep("step_00000003.ckpt.json").display().to_string()),
            checkpoints: vec![
                CheckpointEntry {
                    step: 1,
                    epoch: 0,
                    checkpoint: keep("step_00000001.ckpt.json").display().to_string(),
                    metric: Some(0.1),
                    kind: "ctc".to_string(),
                    metric_mode: "minimize".to_string(),
                },
                CheckpointEntry {
                    step: 2,
                    epoch: 0,
                    checkpoint: keep("step_00000002.ckpt.json").display().to_string(),
                    metric: Some(0.2),
                    kind: "ctc".to_string(),
                    metric_mode: "minimize".to_string(),
                },
                CheckpointEntry {
                    step: 3,
                    epoch: 0,
                    checkpoint: keep("step_00000003.ckpt.json").display().to_string(),
                    metric: Some(0.3),
                    kind: "ctc".to_string(),
                    metric_mode: "minimize".to_string(),
                },
            ],
        };
        prune_checkpoints(dir.path(), &mut state, 1, None).unwrap();
        assert_eq!(state.checkpoints.len(), 2);
        assert!(keep("step_00000001.ckpt.json").exists());
        assert!(keep("step_00000003.ckpt.json").exists());
        assert!(!keep("step_00000002.ckpt.json").exists());
        assert!(!keep("step_00000002.backend.json").exists());
        // And the weights safetensors sidecar is cleaned up too — the
        // tch backend would otherwise leave orphans.
        assert!(
            !keep("step_00000002.weights.safetensors").exists(),
            "weights sidecar was not pruned"
        );
    }
}
