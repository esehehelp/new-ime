use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub const MAGIC: [u8; 8] = *b"KKCSHRD1";
pub const VERSION: u32 = 2;
const VERSION_V1: u32 = 1;
const VERSION_V2: u32 = 2;
const HEADER_LEN: usize = 36;

#[derive(Debug, Clone, Copy)]
pub struct ShardHeader {
    pub version: u32,
    pub row_count: u64,
    pub payload_offset: u64,
    pub index_offset: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct ShardRowRef<'a> {
    pub reading: &'a [u32],
    pub surface: &'a [u32],
    pub context: &'a [u32],
    pub writer_id: u32,
    pub domain_id: u32,
    pub source_id: u32,
}

pub struct ShardReader {
    mmap: Mmap,
    header: ShardHeader,
}

impl ShardReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("open shard {}", path.as_ref().display()))?;
        let mmap = unsafe { Mmap::map(&file) }.context("mmap shard")?;
        if mmap.len() < HEADER_LEN {
            bail!("shard too small");
        }
        if mmap[0..8] != MAGIC {
            bail!("invalid shard magic");
        }
        let header = ShardHeader {
            version: read_u32(&mmap[8..12]),
            row_count: read_u64(&mmap[12..20]),
            payload_offset: read_u64(&mmap[20..28]),
            index_offset: read_u64(&mmap[28..36]),
        };
        if !matches!(header.version, VERSION_V1 | VERSION_V2) {
            bail!("unsupported shard version {}", header.version);
        }
        Ok(Self { mmap, header })
    }

    pub fn header(&self) -> ShardHeader {
        self.header
    }

    pub fn row(&self, idx: usize) -> Result<ShardRowRef<'_>> {
        if idx >= self.header.row_count as usize {
            bail!("row index out of range");
        }
        let index_pos = self.header.index_offset as usize + idx * 8;
        let row_offset = read_u64(&self.mmap[index_pos..index_pos + 8]) as usize;
        let base = self.header.payload_offset as usize + row_offset;
        let reading_len = read_u32(&self.mmap[base..base + 4]) as usize;
        let surface_len = read_u32(&self.mmap[base + 4..base + 8]) as usize;
        let context_len = read_u32(&self.mmap[base + 8..base + 12]) as usize;
        let (writer_id, domain_id, source_id, mut cursor) = match self.header.version {
            // V1 had only `source_id` at byte offset 12, 16-byte row header.
            // writer_id / domain_id were added in V2. See
            // legacy `kkc-data/src/shard.rs` (commit 30f102a) for the
            // original layout.
            VERSION_V1 => (0, 0, read_u32(&self.mmap[base + 12..base + 16]), base + 16),
            VERSION_V2 => (
                read_u32(&self.mmap[base + 12..base + 16]),
                read_u32(&self.mmap[base + 16..base + 20]),
                read_u32(&self.mmap[base + 20..base + 24]),
                base + 24,
            ),
            other => bail!("unsupported shard version {}", other),
        };
        let reading = as_u32_slice(&self.mmap[cursor..cursor + reading_len * 4]);
        cursor += reading_len * 4;
        let surface = as_u32_slice(&self.mmap[cursor..cursor + surface_len * 4]);
        cursor += surface_len * 4;
        let context = as_u32_slice(&self.mmap[cursor..cursor + context_len * 4]);
        Ok(ShardRowRef {
            reading,
            surface,
            context,
            writer_id,
            domain_id,
            source_id,
        })
    }
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes[0..4].try_into().unwrap())
}

fn read_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes[0..8].try_into().unwrap())
}

fn as_u32_slice(bytes: &[u8]) -> &[u32] {
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn opens_version1_rows_with_source_id() {
        // V1 row header is 16 bytes ending with source_id (writer / domain
        // were added in V2). Reference implementation: the legacy
        // `kkc-data/src/shard.rs` at commit 30f102a. Production shards
        // (datasets/mixes/*.kkc) were written with this layout.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v1.kkc");
        let mut file = File::create(&path).unwrap();
        let payload_offset = HEADER_LEN as u64;
        let index_offset = payload_offset + 16 + ((2 + 1 + 1) * 4) as u64;
        file.write_all(&MAGIC).unwrap();
        file.write_all(&VERSION_V1.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        file.write_all(&payload_offset.to_le_bytes()).unwrap();
        file.write_all(&index_offset.to_le_bytes()).unwrap();
        // row header (16 bytes): reading_len, surface_len, context_len, source_id
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&7u32.to_le_bytes()).unwrap();
        // body: reading, surface, context
        file.write_all(&10u32.to_le_bytes()).unwrap();
        file.write_all(&11u32.to_le_bytes()).unwrap();
        file.write_all(&20u32.to_le_bytes()).unwrap();
        file.write_all(&30u32.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let reader = ShardReader::open(&path).unwrap();
        let row = reader.row(0).unwrap();
        assert_eq!(reader.header().version, VERSION_V1);
        assert_eq!(row.reading, &[10, 11]);
        assert_eq!(row.surface, &[20]);
        assert_eq!(row.context, &[30]);
        assert_eq!(row.writer_id, 0);
        assert_eq!(row.domain_id, 0);
        assert_eq!(row.source_id, 7);
    }
}
