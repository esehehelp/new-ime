//! Parity checks against Python reference fixtures.
//!
//! The fixtures are emitted by `tools/rust/emit_parity_fixture.py` and
//! land in `<repo>/parity-fixtures/`. They are NOT checked in — each
//! developer runs the Python script once after a tch API bump or a
//! schedule/loss refactor. Tests here `return` early when the fixture
//! is missing so a clean clone still passes CI.
//!
//! What we check today (Step 5 scope):
//! - `ctc_proposal_loss` matches `F.ctc_loss` within 1e-5 on a fixed
//!   random batch.
//! - `TchOptimizer::lr_at` matches the Python warmup-cosine schedule
//!   within 1e-9 across a handful of sampled steps.
//!
//! Full forward+backward architectural parity is deferred. It requires
//! a custom Python CTCNAT that mirrors this crate's exact layout (post-
//! norm encoder, pre-norm decoder, separate q/k/v projections, tied
//! embeddings). Worth building before we need weight-for-weight migration
//! from existing Python checkpoints; not required for a Rust-only run.

#[cfg(test)]
mod tests {
    use crate::backend::BackendConfig;
    use crate::gpu::loss::ctc_proposal_loss;
    use crate::gpu::optim::TchOptimizer;
    use safetensors::SafeTensors;
    use std::fs;
    use std::path::{Path, PathBuf};
    use tch::nn::VarStore;
    use tch::{Device, Kind, Tensor};

    fn fixture_dir() -> PathBuf {
        // CARGO_MANIFEST_DIR is models/rust/kkc-train; repo root is 3 up.
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .expect("repo root")
            .join("parity-fixtures")
    }

    fn skip_if_missing(path: &Path) -> bool {
        if path.exists() {
            return false;
        }
        eprintln!(
            "skip: parity fixture {} missing\n  hint: python tools/rust/emit_parity_fixture.py",
            path.display()
        );
        true
    }

    fn load_safetensor(path: &Path, name: &str) -> Tensor {
        let bytes = fs::read(path).expect("read safetensors");
        let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
        let view = st.tensor(name).expect("tensor in safetensors");
        let shape: Vec<i64> = view.shape().iter().map(|v| *v as i64).collect();
        let kind = match view.dtype() {
            safetensors::tensor::Dtype::F32 => Kind::Float,
            safetensors::tensor::Dtype::I64 => Kind::Int64,
            other => panic!("unsupported parity dtype {:?}", other),
        };
        let dst = Tensor::zeros(&shape, (kind, Device::Cpu));
        unsafe {
            std::ptr::copy_nonoverlapping(
                view.data().as_ptr(),
                dst.data_ptr() as *mut u8,
                view.data().len(),
            );
        }
        dst
    }

    #[test]
    fn ctc_loss_matches_python_reference_within_tolerance() {
        let dir = fixture_dir();
        let ctc_path = dir.join("ctc_random.safetensors");
        let meta_path = dir.join("ctc_random.json");
        if skip_if_missing(&ctc_path) || skip_if_missing(&meta_path) {
            return;
        }
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(&meta_path).unwrap()).unwrap();
        let blank_id = meta["blank_id"].as_i64().unwrap();
        let expected = meta["expected_loss"].as_f64().unwrap();

        let logits = load_safetensor(&ctc_path, "logits");
        let targets = load_safetensor(&ctc_path, "targets");
        let input_lengths = load_safetensor(&ctc_path, "input_lengths");
        let target_lengths = load_safetensor(&ctc_path, "target_lengths");

        let loss = ctc_proposal_loss(
            &logits,
            &targets,
            &input_lengths,
            &target_lengths,
            blank_id,
        );
        let got = loss.double_value(&[]);
        let diff = (got - expected).abs();
        assert!(
            diff < 1e-5,
            "CTC parity: rust={got:.9} python={expected:.9} diff={diff:.2e}"
        );
    }

    #[test]
    fn warmup_cosine_schedule_matches_python_reference() {
        let path = fixture_dir().join("lr_schedule.json");
        if skip_if_missing(&path) {
            return;
        }
        let payload: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let base_lr = payload["config"]["base_lr"].as_f64().unwrap();
        let warmup = payload["config"]["warmup_steps"].as_u64().unwrap() as usize;
        let total = payload["config"]["total_steps"].as_u64().unwrap() as usize;
        let min_scale = payload["config"]["min_lr_scale"].as_f64().unwrap();

        let cfg = BackendConfig {
            learning_rate: base_lr,
            warmup_steps: warmup,
            scheduler_total_steps: total,
            min_lr_scale: min_scale,
            ..BackendConfig::default()
        };
        let vs = VarStore::new(Device::Cpu);
        let _w: Tensor = vs.root().zeros("w", &[4]);
        let opt = TchOptimizer::from_config(&vs, &cfg, 0.0).unwrap();

        for sample in payload["samples"].as_array().unwrap() {
            let step = sample["step"].as_u64().unwrap() as usize;
            let expected = sample["lr"].as_f64().unwrap();
            let got = opt.lr_at(step);
            let diff = (got - expected).abs();
            assert!(
                diff < 1e-9,
                "LR parity at step {step}: rust={got:.12} python={expected:.12} diff={diff:.2e}"
            );
        }
    }
}
