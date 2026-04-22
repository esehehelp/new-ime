use crate::backend::BackendConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{Read, Write};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub kind: String,
    pub scheduler: String,
    pub base_learning_rate: f64,
    pub weight_decay: f64,
    pub momentum: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub warmup_steps: usize,
    pub scheduler_total_steps: usize,
    pub min_lr_scale: f64,
    pub momentum_buffers: BTreeMap<String, Vec<f64>>,
    pub exp_avg: BTreeMap<String, Vec<f64>>,
    pub exp_avg_sq: BTreeMap<String, Vec<f64>>,
}

impl OptimizerState {
    pub fn new(config: &BackendConfig) -> Result<Self> {
        match config.optimizer.as_str() {
            "sgd" | "adamw" => {}
            other => anyhow::bail!("unknown optimizer: {}", other),
        }
        match config.scheduler.as_str() {
            "constant" | "warmup_cosine" => {}
            other => anyhow::bail!("unknown scheduler: {}", other),
        }
        Ok(Self {
            kind: config.optimizer.clone(),
            scheduler: config.scheduler.clone(),
            base_learning_rate: config.learning_rate,
            weight_decay: config.weight_decay,
            momentum: config.momentum,
            beta1: config.beta1,
            beta2: config.beta2,
            epsilon: config.epsilon,
            warmup_steps: config.warmup_steps,
            scheduler_total_steps: config.scheduler_total_steps,
            min_lr_scale: config.min_lr_scale,
            momentum_buffers: BTreeMap::new(),
            exp_avg: BTreeMap::new(),
            exp_avg_sq: BTreeMap::new(),
        })
    }

    pub fn current_lr(&self, step: usize) -> f64 {
        let step = step.max(1);
        let warmup = self.warmup_steps.max(1);
        match self.scheduler.as_str() {
            "constant" => {
                if self.warmup_steps > 0 && step < self.warmup_steps {
                    self.base_learning_rate * step as f64 / warmup as f64
                } else {
                    self.base_learning_rate
                }
            }
            "warmup_cosine" => {
                if self.warmup_steps > 0 && step < self.warmup_steps {
                    return self.base_learning_rate * step as f64 / warmup as f64;
                }
                if self.scheduler_total_steps <= self.warmup_steps {
                    return self.base_learning_rate;
                }
                let progress = ((step.saturating_sub(self.warmup_steps)) as f64
                    / (self.scheduler_total_steps - self.warmup_steps) as f64)
                    .clamp(0.0, 1.0);
                let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                self.base_learning_rate * (self.min_lr_scale + (1.0 - self.min_lr_scale) * cosine)
            }
            _ => self.base_learning_rate,
        }
    }

    pub fn update(&mut self, step: usize, name: &str, weights: &mut [f64], grads: &[f64]) {
        if weights.len() != grads.len() {
            return;
        }
        let lr = self.current_lr(step);
        match self.kind.as_str() {
            "sgd" => self.update_sgd(lr, name, weights, grads),
            "adamw" => self.update_adamw(step, lr, name, weights, grads),
            _ => {}
        }
    }

    fn update_sgd(&mut self, lr: f64, name: &str, weights: &mut [f64], grads: &[f64]) {
        let momentum = self
            .momentum_buffers
            .entry(name.to_string())
            .or_insert_with(|| vec![0.0; weights.len()]);
        if momentum.len() != weights.len() {
            momentum.resize(weights.len(), 0.0);
        }
        for ((weight, grad), velocity) in weights
            .iter_mut()
            .zip(grads.iter().copied())
            .zip(momentum.iter_mut())
        {
            if self.weight_decay != 0.0 {
                *weight -= lr * self.weight_decay * *weight;
            }
            *velocity = self.momentum * *velocity + grad;
            *weight -= lr * *velocity;
        }
    }

    fn update_adamw(
        &mut self,
        step: usize,
        lr: f64,
        name: &str,
        weights: &mut [f64],
        grads: &[f64],
    ) {
        let exp_avg = self
            .exp_avg
            .entry(name.to_string())
            .or_insert_with(|| vec![0.0; weights.len()]);
        let exp_avg_sq = self
            .exp_avg_sq
            .entry(name.to_string())
            .or_insert_with(|| vec![0.0; weights.len()]);
        if exp_avg.len() != weights.len() {
            exp_avg.resize(weights.len(), 0.0);
        }
        if exp_avg_sq.len() != weights.len() {
            exp_avg_sq.resize(weights.len(), 0.0);
        }
        let beta1_correction = 1.0 - self.beta1.powi(step.max(1) as i32);
        let beta2_correction = 1.0 - self.beta2.powi(step.max(1) as i32);
        for (((weight, grad), m), v) in weights
            .iter_mut()
            .zip(grads.iter().copied())
            .zip(exp_avg.iter_mut())
            .zip(exp_avg_sq.iter_mut())
        {
            if self.weight_decay != 0.0 {
                *weight -= lr * self.weight_decay * *weight;
            }
            *m = self.beta1 * *m + (1.0 - self.beta1) * grad;
            *v = self.beta2 * *v + (1.0 - self.beta2) * grad * grad;
            let m_hat = *m / beta1_correction.max(1e-12);
            let v_hat = *v / beta2_correction.max(1e-12);
            *weight -= lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }

    pub fn write_binary<W: Write>(&self, writer: &mut W) -> Result<()> {
        write_string(writer, &self.kind)?;
        write_string(writer, &self.scheduler)?;
        write_f64(writer, self.base_learning_rate)?;
        write_f64(writer, self.weight_decay)?;
        write_f64(writer, self.momentum)?;
        write_f64(writer, self.beta1)?;
        write_f64(writer, self.beta2)?;
        write_f64(writer, self.epsilon)?;
        write_u64(writer, self.warmup_steps as u64)?;
        write_u64(writer, self.scheduler_total_steps as u64)?;
        write_f64(writer, self.min_lr_scale)?;
        write_map(writer, &self.momentum_buffers)?;
        write_map(writer, &self.exp_avg)?;
        write_map(writer, &self.exp_avg_sq)?;
        Ok(())
    }

    pub fn read_binary<R: Read>(reader: &mut R) -> Result<Self> {
        Ok(Self {
            kind: read_string(reader)?,
            scheduler: read_string(reader)?,
            base_learning_rate: read_f64(reader)?,
            weight_decay: read_f64(reader)?,
            momentum: read_f64(reader)?,
            beta1: read_f64(reader)?,
            beta2: read_f64(reader)?,
            epsilon: read_f64(reader)?,
            warmup_steps: read_u64(reader)? as usize,
            scheduler_total_steps: read_u64(reader)? as usize,
            min_lr_scale: read_f64(reader)?,
            momentum_buffers: read_map(reader)?,
            exp_avg: read_map(reader)?,
            exp_avg_sq: read_map(reader)?,
        })
    }
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn write_f64<W: Write>(writer: &mut W, value: f64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_f64<R: Read>(reader: &mut R) -> Result<f64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}

fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<()> {
    write_u64(writer, value.len() as u64)?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn read_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = read_u64(reader)? as usize;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    Ok(String::from_utf8(bytes)?)
}

fn write_vec_f32<W: Write>(writer: &mut W, values: &[f64]) -> Result<()> {
    write_u64(writer, values.len() as u64)?;
    for value in values {
        writer.write_all(&(*value as f32).to_le_bytes())?;
    }
    Ok(())
}

fn read_vec_f32<R: Read>(reader: &mut R) -> Result<Vec<f64>> {
    let len = read_u64(reader)? as usize;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        out.push(f32::from_le_bytes(bytes) as f64);
    }
    Ok(out)
}

fn write_map<W: Write>(writer: &mut W, map: &BTreeMap<String, Vec<f64>>) -> Result<()> {
    write_u64(writer, map.len() as u64)?;
    for (key, values) in map {
        write_string(writer, key)?;
        write_vec_f32(writer, values)?;
    }
    Ok(())
}

fn read_map<R: Read>(reader: &mut R) -> Result<BTreeMap<String, Vec<f64>>> {
    let len = read_u64(reader)? as usize;
    let mut map = BTreeMap::new();
    for _ in 0..len {
        let key = read_string(reader)?;
        let values = read_vec_f32(reader)?;
        map.insert(key, values);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_backend() -> BackendConfig {
        BackendConfig {
            optimizer: "adamw".to_string(),
            scheduler: "warmup_cosine".to_string(),
            learning_rate: 1e-3,
            weight_decay: 0.01,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            warmup_steps: 10,
            scheduler_total_steps: 100,
            min_lr_scale: 0.1,
            ..BackendConfig::default()
        }
    }

    #[test]
    fn warmup_cosine_changes_learning_rate() {
        let opt = OptimizerState::new(&sample_backend()).unwrap();
        assert!(opt.current_lr(1) < opt.current_lr(10));
        assert!(opt.current_lr(100) < opt.current_lr(10));
    }

    #[test]
    fn adamw_update_changes_weights() {
        let mut opt = OptimizerState::new(&sample_backend()).unwrap();
        let mut weights = vec![1.0, -1.0];
        opt.update(1, "w", &mut weights, &[0.5, -0.25]);
        assert_ne!(weights, vec![1.0, -1.0]);
    }
}
