//! Host-to-device staging + double-buffered prefetch for the tch backend.
//!
//! The stage-2 worker lives here: it pulls `PackedBatch` off the CPU
//! prefetcher, copies scalars into contiguous i64/f32 buffers ready for
//! tch `Tensor::from_slice`, and queues them for the training thread.
//! The training thread finishes H2D via `GpuBatch::upload` so tch tensors
//! never cross thread boundaries (tch's Cuda stream handling is safest on
//! a single thread).

use crate::device::{resolve_tch_device, Device};
use crate::pipeline::BatchProducer;
use anyhow::{Context, Result};
use rust_data::PackedBatch;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use tch::{Device as TchDevice, Tensor};

/// Host-side scratch staged before handing to CUDA. Holds contiguous
/// i64/f32 buffers sized exactly for the tensor upload so the GPU thread
/// can allocate in one shot.
#[derive(Debug)]
pub struct StagedHostBatch {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<f32>,
    pub target_ids: Vec<i64>,
    pub input_lengths: Vec<i64>,
    pub target_lengths: Vec<i64>,
    pub batch_size: usize,
    pub max_input_len: usize,
    pub max_target_len: usize,
    pub order_cursor: usize,
    pub bytes: usize,
    pub non_padding_input_tokens: usize,
    pub non_padding_target_tokens: usize,
}

impl StagedHostBatch {
    pub fn from_packed(packed: &PackedBatch) -> Self {
        let input_ids = packed.input_ids.iter().map(|v| *v as i64).collect();
        let attention_mask = packed.attention_mask.iter().map(|v| *v as f32).collect();
        let target_ids = packed.target_ids.iter().map(|v| *v as i64).collect();
        let input_lengths = packed.input_lengths.iter().map(|v| *v as i64).collect();
        let target_lengths = packed.target_lengths.iter().map(|v| *v as i64).collect();
        Self {
            input_ids,
            attention_mask,
            target_ids,
            input_lengths,
            target_lengths,
            batch_size: packed.batch_size,
            max_input_len: packed.max_input_len,
            max_target_len: packed.max_target_len,
            order_cursor: packed.order_cursor,
            bytes: packed.bytes(),
            non_padding_input_tokens: packed.non_padding_input_tokens(),
            non_padding_target_tokens: packed.non_padding_target_tokens(),
        }
    }
}

/// GPU-resident batch. Tensors are pinned to the configured CUDA device
/// and ready for forward to consume directly.
#[derive(Debug)]
pub struct GpuBatch {
    pub input_ids: Tensor,
    pub attention_mask: Tensor,
    pub target_ids: Tensor,
    pub input_lengths: Tensor,
    pub target_lengths: Tensor,
    pub batch_size: usize,
    #[allow(dead_code)]
    pub max_input_len: usize,
    #[allow(dead_code)]
    pub max_target_len: usize,
    #[allow(dead_code)]
    pub order_cursor: usize,
    pub bytes: usize,
    pub non_padding_input_tokens: usize,
    pub non_padding_target_tokens: usize,
}

impl GpuBatch {
    pub fn upload(staged: StagedHostBatch, device: TchDevice) -> Self {
        let batch = staged.batch_size as i64;
        let in_len = staged.max_input_len as i64;
        let tgt_len = staged.max_target_len as i64;

        let input_ids = Tensor::from_slice(&staged.input_ids)
            .view([batch, in_len])
            .to_device(device);
        let attention_mask = Tensor::from_slice(&staged.attention_mask)
            .view([batch, in_len])
            .to_device(device);
        let target_ids = Tensor::from_slice(&staged.target_ids)
            .view([batch, tgt_len])
            .to_device(device);
        let input_lengths = Tensor::from_slice(&staged.input_lengths).to_device(device);
        let target_lengths = Tensor::from_slice(&staged.target_lengths).to_device(device);

        Self {
            input_ids,
            attention_mask,
            target_ids,
            input_lengths,
            target_lengths,
            batch_size: staged.batch_size,
            max_input_len: staged.max_input_len,
            max_target_len: staged.max_target_len,
            order_cursor: staged.order_cursor,
            bytes: staged.bytes,
            non_padding_input_tokens: staged.non_padding_input_tokens,
            non_padding_target_tokens: staged.non_padding_target_tokens,
        }
    }
}

/// Stage-2 prefetch: owns a worker thread that pulls `PackedBatch` off a
/// CPU-side producer, stages it on the host, and hands it back over a
/// bounded channel.
#[allow(dead_code)]
pub struct StagedBatchPipeline<P>
where
    P: BatchProducer<Item = PackedBatch> + Send + 'static,
{
    rx: Receiver<Result<StagedHostBatch>>,
    handle: Option<JoinHandle<()>>,
    device: TchDevice,
    _marker: std::marker::PhantomData<P>,
}

#[allow(dead_code)]
impl<P> StagedBatchPipeline<P>
where
    P: BatchProducer<Item = PackedBatch> + Send + 'static,
{
    pub fn spawn(mut producer: P, device: Device, queue_size: usize) -> Result<Self> {
        let tch_device = resolve_tch_device(device)?;
        let (tx, rx) = sync_channel::<Result<StagedHostBatch>>(queue_size.max(1));
        let handle = thread::Builder::new()
            .name("rust-stage-h2d".to_string())
            .spawn(move || stage_loop(tx, &mut producer))
            .context("spawn stage-h2d thread")?;
        Ok(Self {
            rx,
            handle: Some(handle),
            device: tch_device,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn next_gpu_batch(&mut self) -> Result<Option<GpuBatch>> {
        match self.rx.recv() {
            Ok(Ok(staged)) => Ok(Some(GpuBatch::upload(staged, self.device))),
            Ok(Err(err)) => Err(err),
            Err(_) => Ok(None),
        }
    }
}

impl<P> Drop for StagedBatchPipeline<P>
where
    P: BatchProducer<Item = PackedBatch> + Send + 'static,
{
    fn drop(&mut self) {
        drop(std::mem::replace(
            &mut self.rx,
            sync_channel::<Result<StagedHostBatch>>(1).1,
        ));
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[allow(dead_code)]
fn stage_loop<P>(tx: SyncSender<Result<StagedHostBatch>>, producer: &mut P)
where
    P: BatchProducer<Item = PackedBatch>,
{
    loop {
        match producer.next_item() {
            Ok(Some(packed)) => {
                let staged = StagedHostBatch::from_packed(&packed);
                if tx.send(Ok(staged)).is_err() {
                    break;
                }
            }
            Ok(None) => break,
            Err(err) => {
                let _ = tx.send(Err(err));
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_packed() -> PackedBatch {
        PackedBatch {
            input_ids: vec![1, 2, 0, 3, 4, 5],
            attention_mask: vec![1, 1, 0, 1, 1, 1],
            target_ids: vec![8, 9, 0, 7],
            input_lengths: vec![2, 3],
            target_lengths: vec![2, 1],
            source_ids: vec![0, 1],
            batch_size: 2,
            max_input_len: 3,
            max_target_len: 2,
            order_cursor: 2,
        }
    }

    #[test]
    fn staged_host_batch_preserves_non_padding_counts() {
        let packed = tiny_packed();
        let staged = StagedHostBatch::from_packed(&packed);
        assert_eq!(staged.batch_size, 2);
        assert_eq!(staged.non_padding_input_tokens, 5);
        assert_eq!(staged.non_padding_target_tokens, 3);
        assert_eq!(staged.input_ids.len(), 6);
    }

    #[test]
    fn upload_roundtrips_shapes_on_cpu() {
        let packed = tiny_packed();
        let staged = StagedHostBatch::from_packed(&packed);
        let gpu = GpuBatch::upload(staged, TchDevice::Cpu);
        assert_eq!(gpu.input_ids.size(), vec![2, 3]);
        assert_eq!(gpu.target_ids.size(), vec![2, 2]);
        assert_eq!(gpu.attention_mask.size(), vec![2, 3]);
        assert_eq!(gpu.input_lengths.size(), vec![2]);
        assert_eq!(gpu.target_lengths.size(), vec![2]);
    }
}
