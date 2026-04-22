//! CUDA backend built on `tch`. Feature-gated behind `cuda`.
//!
//! This module is intentionally scoped to the plumbing layer: device
//! resolution, host-to-device upload, a background stage-2 prefetch thread
//! that keeps a GPU-resident batch queue warm, and a `TrainBackend`
//! skeleton that satisfies the trait so the `fit` loop can drive it. The
//! actual transformer forward/backward / CTC loss lands in a follow-up —
//! this file makes the plumbing compilable and exercised by tests so the
//! next patch can focus on kernels without relitigating the pipeline.
//!
//! Build notes:
//! - Requires `LIBTORCH` env var pointing at a libtorch install (the
//!   project's `.venv/Lib/site-packages/torch` works).
//! - Windows: also add `LIBTORCH/lib` to `PATH` at runtime so the DLLs
//!   resolve. A `.cargo/config.toml` can pin this.

use crate::backend::{BackendConfig, TrainBackend};
use crate::device::{resolve_tch_device, Device};
use crate::pipeline::BatchProducer;
use crate::trainer::TrainerStep;
use anyhow::{Context, Result};
use kkc_data::PackedBatch;
use std::path::Path;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use tch::{Device as TchDevice, Kind, Tensor};

/// Host-side scratch staged before handing to CUDA. Holds contiguous i64/f32
/// buffers sized exactly for the tensor upload to avoid re-allocation on the
/// GPU thread.
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
        let bytes = packed.bytes();
        let non_padding_input_tokens = packed.non_padding_input_tokens();
        let non_padding_target_tokens = packed.non_padding_target_tokens();
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
            bytes,
            non_padding_input_tokens,
            non_padding_target_tokens,
        }
    }
}

/// GPU-resident batch. Tensors are pinned to the configured CUDA device and
/// ready for the forward pass to consume directly.
#[derive(Debug)]
pub struct GpuBatch {
    pub input_ids: Tensor,
    pub attention_mask: Tensor,
    pub target_ids: Tensor,
    pub input_lengths: Tensor,
    pub target_lengths: Tensor,
    pub batch_size: usize,
    pub max_input_len: usize,
    pub max_target_len: usize,
    pub order_cursor: usize,
    pub bytes: usize,
    pub non_padding_input_tokens: usize,
    pub non_padding_target_tokens: usize,
}

impl GpuBatch {
    pub fn upload(staged: StagedHostBatch, device: TchDevice) -> Self {
        // `Tensor::from_slice` copies to CPU first; `.to_device` then runs the
        // real H2D with whatever stream tch binds. For `non_blocking=true` we
        // rely on the default autograd stream which sequences after the
        // upload — good enough for correctness, and faster than device="cpu"
        // staging then `.to(cuda)` because we avoid a round trip through
        // an allocator-owned tensor we do not need.
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
        let input_lengths =
            Tensor::from_slice(&staged.input_lengths).to_device(device);
        let target_lengths =
            Tensor::from_slice(&staged.target_lengths).to_device(device);

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
/// CPU-side producer, stages it on the host (`StagedHostBatch`), and hands it
/// back over a bounded channel. The training loop finishes the upload to GPU
/// on its own thread so tch `Tensor` never needs to cross thread boundaries.
pub struct StagedBatchPipeline<P>
where
    P: BatchProducer<Item = PackedBatch> + Send + 'static,
{
    rx: Receiver<Result<StagedHostBatch>>,
    handle: Option<JoinHandle<()>>,
    device: TchDevice,
    _marker: std::marker::PhantomData<P>,
}

impl<P> StagedBatchPipeline<P>
where
    P: BatchProducer<Item = PackedBatch> + Send + 'static,
{
    pub fn spawn(mut producer: P, device: Device, queue_size: usize) -> Result<Self> {
        let tch_device = resolve_tch_device(device)?;
        let (tx, rx) = sync_channel::<Result<StagedHostBatch>>(queue_size.max(1));
        let handle = thread::Builder::new()
            .name("kkc-stage-h2d".to_string())
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

/// Skeleton CTC-NAT backend on CUDA. The real forward/backward is not wired
/// yet; this implementation tracks last-seen batch stats so the training loop
/// and checkpoint flow can be exercised end-to-end. It is a compile-time
/// proof that the trait + pipeline interface is sufficient, and a runtime
/// proof that H2D upload + `TrainerStep` accounting is correct.
pub struct TchCtcNatBackend {
    device: TchDevice,
    _config: BackendConfig,
    last_loss: Option<f64>,
    step_count: usize,
}

impl TchCtcNatBackend {
    pub fn new(config: &BackendConfig, device: Device) -> Result<Self> {
        let tch_device = resolve_tch_device(device)?;
        Ok(Self {
            device: tch_device,
            _config: config.clone(),
            last_loss: None,
            step_count: 0,
        })
    }

    pub fn device(&self) -> TchDevice {
        self.device
    }

    /// Consume a pre-uploaded [`GpuBatch`] directly. Preferred for the
    /// async pipeline since H2D has already happened on the stage thread.
    pub fn step_gpu(&mut self, step: usize, batch: &GpuBatch) -> Result<TrainerStep> {
        // Placeholder "loss": mean of attention_mask.sum over rows. Requires
        // a real forward; tracked in the next patch. Using a cheap tensor op
        // here validates the device is live and the tensors are usable.
        let token_mass = batch
            .attention_mask
            .sum_dim_intlist(
                [1i64].as_ref(),
                /*keepdim=*/ false,
                Kind::Float,
            )
            .mean(Kind::Float);
        // Pull the scalar back to host for logging. `double_value` on a 0-d
        // tensor is the tch-blessed way; `f64::from(&tensor)` works only on
        // certain builds.
        let loss = token_mass.double_value(&[]) / (step as f64 + 1.0);
        self.last_loss = Some(loss);
        self.step_count = step;
        Ok(TrainerStep {
            loss,
            rows: batch.batch_size,
            bytes: batch.bytes,
            input_tokens: batch.non_padding_input_tokens,
            target_tokens: batch.non_padding_target_tokens,
        })
    }
}

impl TrainBackend for TchCtcNatBackend {
    fn kind(&self) -> &'static str {
        "tch-ctc-nat"
    }

    fn step(&mut self, step: usize, batch: &PackedBatch) -> Result<TrainerStep> {
        let staged = StagedHostBatch::from_packed(batch);
        let gpu = GpuBatch::upload(staged, self.device);
        self.step_gpu(step, &gpu)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        // Weight snapshot lands in the next patch. For now we write a marker
        // file so the existing ckpt ledger keeps working end-to-end and the
        // async writer path is exercised.
        let body = serde_json::json!({
            "kind": "tch-ctc-nat",
            "step": self.step_count,
            "last_loss": self.last_loss,
            "note": "weights pending: transformer impl lands in follow-up",
        });
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&body).context("serialize tch backend marker")?,
        )
        .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let parsed: serde_json::Value =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        if let Some(step) = parsed.get("step").and_then(|v| v.as_u64()) {
            self.step_count = step as usize;
        }
        if let Some(loss) = parsed.get("last_loss").and_then(|v| v.as_f64()) {
            self.last_loss = Some(loss);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kkc_data::PackedBatch;

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
        // CPU tch device is always available — lets us validate the upload
        // plumbing (shape / dtype) without a CUDA runtime.
        let packed = tiny_packed();
        let staged = StagedHostBatch::from_packed(&packed);
        let gpu = GpuBatch::upload(staged, TchDevice::Cpu);
        assert_eq!(gpu.input_ids.size(), vec![2, 3]);
        assert_eq!(gpu.target_ids.size(), vec![2, 2]);
        assert_eq!(gpu.attention_mask.size(), vec![2, 3]);
        assert_eq!(gpu.input_lengths.size(), vec![2]);
        assert_eq!(gpu.target_lengths.size(), vec![2]);
    }

    #[test]
    fn tch_backend_step_produces_trainer_step_on_cpu() {
        let cfg = BackendConfig::default();
        let mut backend = TchCtcNatBackend::new(&cfg, Device::Cpu).unwrap();
        let packed = tiny_packed();
        let step = backend.step(1, &packed).unwrap();
        assert_eq!(step.rows, 2);
        assert!(step.loss.is_finite());
        assert_eq!(step.input_tokens, 5);
        assert_eq!(step.target_tokens, 3);
    }
}
