//! Build a `.offsets.npy` index for a JSONL file.
//!
//! One-time sequential scan records the byte offset where each non-empty line
//! starts. The Python `KanaKanjiDataset` then mmap's the resulting array and
//! does random-access reads by seeking to `offsets[idx]`.
//!
//! Rust exists here because this is a time-consuming bulk scan on 40+ GiB
//! input (memory `feedback_script_language_preference`: data pipelines are
//! Rust-first). A Python version of the same scan takes 3-5 min on a 46 GiB
//! file; this binary completes in 30-60 s.
//!
//! Output format matches `numpy.save(arr_uint64)` so the existing Python
//! loader can read it with `np.load(path, mmap_mode='r')`.

use anyhow::{Context, Result};
use clap::Parser;
use std::fs::{File, rename};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Build a <jsonl>.offsets.npy byte-offset index for random-access JSONL reads.")]
struct Args {
    /// Input JSONL path.
    #[arg(long)]
    input: PathBuf,

    /// Output .offsets.npy path. Defaults to "<input>.offsets.npy".
    #[arg(long)]
    output: Option<PathBuf>,

    /// Emit a progress line every N bytes scanned.
    #[arg(long, default_value_t = 1_073_741_824u64)]   // 1 GiB
    progress_bytes: u64,

    /// Read buffer size (bytes). Larger reduces syscall overhead.
    #[arg(long, default_value_t = 8 * 1024 * 1024)]
    buf_size: usize,
}

/// True if the byte slice is empty after trimming ASCII whitespace (matches
/// Python's `line.strip()` for JSONL purposes — full Unicode whitespace is
/// never present inside well-formed bytes-encoded JSON lines).
fn is_blank(line: &[u8]) -> bool {
    line.iter()
        .all(|&b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
}

/// Write a 1-D uint64 array in numpy .npy v1.0 format (little-endian).
///
/// Layout:
///   magic      = \x93NUMPY
///   version    = 0x01 0x00
///   header_len = u16 LE
///   header     = ASCII dict + trailing '\n', padded with spaces so the
///                total (magic + version + len + header) is 64-byte aligned.
///   body       = N * 8 bytes of u64 LE
fn write_npy_u64<W: Write>(mut out: W, data: &[u64]) -> std::io::Result<()> {
    let magic = b"\x93NUMPY";
    let version: [u8; 2] = [0x01, 0x00];
    let dict = format!(
        "{{'descr': '<u8', 'fortran_order': False, 'shape': ({},), }}",
        data.len()
    );
    let preamble_len = magic.len() + version.len() + 2; // + u16 header_len
    let unpadded = preamble_len + dict.len() + 1;       // + trailing '\n'
    let padded_to = ((unpadded + 63) / 64) * 64;
    let pad = padded_to - unpadded;
    let mut header = String::with_capacity(dict.len() + pad + 1);
    header.push_str(&dict);
    for _ in 0..pad {
        header.push(' ');
    }
    header.push('\n');
    assert_eq!((preamble_len + header.len()) % 64, 0);
    let header_len = u16::try_from(header.len())
        .expect("header fits in u16 for this shape size");

    out.write_all(magic)?;
    out.write_all(&version)?;
    out.write_all(&header_len.to_le_bytes())?;
    out.write_all(header.as_bytes())?;
    for v in data {
        out.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input = args.input;
    let output = args
        .output
        .unwrap_or_else(|| PathBuf::from(format!("{}.offsets.npy", input.display())));

    let file = File::open(&input).with_context(|| format!("open {}", input.display()))?;
    let total_bytes = file.metadata()?.len();
    eprintln!(
        "offset-index: input={} ({:.2} GiB), output={}",
        input.display(),
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        output.display()
    );

    let mut reader = BufReader::with_capacity(args.buf_size, file);
    let mut offsets: Vec<u64> = Vec::with_capacity(200_000_000);
    let mut line_start: u64 = 0;
    let mut next_report = args.progress_bytes.max(1);
    let t0 = Instant::now();
    let mut line_buf: Vec<u8> = Vec::with_capacity(4096);

    loop {
        line_buf.clear();
        let n = reader
            .read_until(b'\n', &mut line_buf)
            .context("read_until")?;
        if n == 0 {
            break;
        }
        if !is_blank(&line_buf) {
            offsets.push(line_start);
        }
        line_start += n as u64;
        if line_start >= next_report {
            let elapsed = t0.elapsed().as_secs_f64();
            let mib_per_s = (line_start as f64 / (1024.0 * 1024.0)) / elapsed.max(1e-6);
            eprintln!(
                "  scanned {:.1} GiB / {} lines, {:.1} MiB/s",
                line_start as f64 / (1024.0 * 1024.0 * 1024.0),
                offsets.len(),
                mib_per_s
            );
            next_report = line_start + args.progress_bytes;
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "scan complete: {} lines (non-empty), {:.2} GiB in {:.1}s ({:.1} MiB/s)",
        offsets.len(),
        line_start as f64 / (1024.0 * 1024.0 * 1024.0),
        elapsed,
        (line_start as f64 / (1024.0 * 1024.0)) / elapsed.max(1e-6)
    );

    // Atomic write: tmp then rename.
    let tmp = PathBuf::from(format!("{}.tmp", output.display()));
    let out_f = File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
    let mut buf_out = BufWriter::with_capacity(4 * 1024 * 1024, out_f);
    write_npy_u64(&mut buf_out, &offsets).context("write npy")?;
    buf_out.flush()?;
    drop(buf_out);
    rename(&tmp, &output).with_context(|| format!("rename {} → {}", tmp.display(), output.display()))?;

    let out_size = std::fs::metadata(&output)?.len();
    eprintln!(
        "wrote {} ({:.2} MiB)",
        output.display(),
        out_size as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn npy_header_alignment() {
        let mut buf: Vec<u8> = Vec::new();
        write_npy_u64(&mut buf, &[1u64, 2u64, 3u64]).unwrap();
        // magic + version + len + header_bytes should be 64-byte aligned.
        let header_len =
            u16::from_le_bytes([buf[8], buf[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
        // body starts right after
        let body_start = 10 + header_len;
        assert_eq!(buf.len(), body_start + 3 * 8);
        // first u64 == 1 LE
        assert_eq!(&buf[body_start..body_start + 8], &1u64.to_le_bytes());
    }

    #[test]
    fn blank_detection() {
        assert!(is_blank(b""));
        assert!(is_blank(b"\n"));
        assert!(is_blank(b" \t\r\n"));
        assert!(!is_blank(b"{\"reading\":\"a\"}"));
        assert!(!is_blank(b"x\n"));
    }
}
