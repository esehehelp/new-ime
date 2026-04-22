//! CUDA backend built on `tch`. Feature-gated behind `cuda`.
//!
//! Layout:
//! - `batch`     — host staging + GPU upload pipeline
//! - `layers`    — transformer building blocks (MHA, encoder/decoder layers)
//! - `model`     — [`CtcNatModel`] = encoder + proposal decoder + CTC head
//!                 + refine decoder + refine/remask/stop heads. Mirrors the
//!                 Python `CTCNAT` in shape. Ports land step-by-step:
//!                   * Step 1 (this commit): forward + shapes + param count
//!                   * Step 2: losses + backward
//!                   * Step 3: AdamW + schedule + AMP
//!                   * Step 4: safetensors checkpoint
//! - `backend`   — `TrainBackend` wrapper bound to `CtcNatModel`

pub mod backend;
pub mod batch;
pub mod ckpt;
pub mod layers;
pub mod loss;
pub mod model;
pub mod optim;

pub use backend::TchCtcNatBackend;
pub use batch::{GpuBatch, StagedBatchPipeline, StagedHostBatch};
pub use model::{CtcNatForward, CtcNatModel};
pub use optim::TchOptimizer;
