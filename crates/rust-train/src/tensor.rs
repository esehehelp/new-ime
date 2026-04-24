use rust_data::PackedBatch;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TensorBatch {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<f32>,
    pub target_ids: Vec<u32>,
    pub writer_ids: Vec<u32>,
    pub domain_ids: Vec<u32>,
    pub source_ids: Vec<u32>,
    pub batch_size: usize,
    pub max_input_len: usize,
    pub max_target_len: usize,
}

impl TensorBatch {
    pub fn from_packed(batch: &PackedBatch) -> Self {
        Self {
            input_ids: batch.input_ids.clone(),
            attention_mask: batch.attention_mask.iter().map(|v| *v as f32).collect(),
            target_ids: batch.target_ids.clone(),
            writer_ids: batch.writer_ids.clone(),
            domain_ids: batch.domain_ids.clone(),
            source_ids: batch.source_ids.clone(),
            batch_size: batch.batch_size,
            max_input_len: batch.max_input_len,
            max_target_len: batch.max_target_len,
        }
    }

    pub fn input_rows(&self) -> impl Iterator<Item = &[u32]> {
        self.input_ids.chunks(self.max_input_len)
    }

    pub fn target_rows(&self) -> impl Iterator<Item = &[u32]> {
        self.target_ids.chunks(self.max_target_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_data::PackedBatch;

    #[test]
    fn converts_packed_batch_to_tensor_batch() {
        let batch = PackedBatch {
            input_ids: vec![1, 2, 0, 3, 4, 5],
            attention_mask: vec![1, 1, 0, 1, 1, 1],
            target_ids: vec![8, 9, 0, 7],
            input_lengths: vec![2, 3],
            target_lengths: vec![2, 1],
            writer_ids: vec![10, 11],
            domain_ids: vec![20, 21],
            source_ids: vec![0, 1],
            batch_size: 2,
            max_input_len: 3,
            max_target_len: 2,
            order_cursor: 2,
        };
        let tensor = TensorBatch::from_packed(&batch);
        assert_eq!(tensor.batch_size, 2);
        assert_eq!(tensor.input_ids.len(), 6);
        assert_eq!(tensor.target_ids.len(), 4);
        assert_eq!(tensor.input_rows().count(), 2);
        assert_eq!(tensor.target_rows().count(), 2);
    }
}
