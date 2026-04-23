use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub const MAGIC: [u8; 8] = *b"KKCSHRD1";
pub const VERSION: u32 = 1;
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
        if header.version != VERSION {
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
        let source_id = read_u32(&self.mmap[base + 12..base + 16]);
        let mut cursor = base + 16;
        let reading = as_u32_slice(&self.mmap[cursor..cursor + reading_len * 4]);
        cursor += reading_len * 4;
        let surface = as_u32_slice(&self.mmap[cursor..cursor + surface_len * 4]);
        cursor += surface_len * 4;
        let context = as_u32_slice(&self.mmap[cursor..cursor + context_len * 4]);
        Ok(ShardRowRef {
            reading,
            surface,
            context,
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
