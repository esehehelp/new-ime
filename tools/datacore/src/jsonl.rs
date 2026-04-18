//! Streaming JSONL readers and writers with optional compression.
//!
//! The Python pipeline uses `json.dumps(row, ensure_ascii=False) + "\n"` for
//! output. We go through `serde_json::to_string` which preserves UTF-8 by
//! default (no `ensure_ascii=True` equivalent). Field ordering on output is
//! fixed by the `Row` struct order, matching the Python writer.
//!
//! Compression is chosen per output path suffix: `.zst`, `.xz`, `.gz` are
//! auto-detected; anything else is raw.

use anyhow::{Context, Result};
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::Serialize;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use xz2::write::XzEncoder;

use crate::row::Row;

/// Auto-detected output encoding based on file suffix.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum OutputFormat {
    Raw,
    Zstd,
    Xz,
    Gzip,
}

impl OutputFormat {
    /// Infer the encoding from the path's extension. Defaults to `Raw`.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        let p = path.as_ref();
        match p.extension().and_then(|s| s.to_str()) {
            Some("zst") | Some("zstd") => OutputFormat::Zstd,
            Some("xz") => OutputFormat::Xz,
            Some("gz") | Some("gzip") => OutputFormat::Gzip,
            _ => OutputFormat::Raw,
        }
    }
}

/// A write handle that dispatches to the right compressor. Dropping it
/// finalises the compression stream.
pub enum Writer {
    Raw(BufWriter<File>),
    Zstd(zstd::stream::AutoFinishEncoder<'static, BufWriter<File>>),
    Xz(XzEncoder<BufWriter<File>>),
    Gzip(GzEncoder<BufWriter<File>>),
}

impl Write for Writer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Writer::Raw(w) => w.write(buf),
            Writer::Zstd(w) => w.write(buf),
            Writer::Xz(w) => w.write(buf),
            Writer::Gzip(w) => w.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Writer::Raw(w) => w.flush(),
            Writer::Zstd(w) => w.flush(),
            Writer::Xz(w) => w.flush(),
            Writer::Gzip(w) => w.flush(),
        }
    }
}

/// Open a JSONL output file, auto-detecting compression from the extension.
///
/// `compression_level` interpretations:
/// - zstd: 1..=22 (passed directly)
/// - xz: 0..=9 (clamped)
/// - gzip: 1..=9 (clamped to `Compression`)
pub fn open_output(
    path: &Path,
    format: Option<OutputFormat>,
    compression_level: i32,
) -> Result<Writer> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("mkdir parent for {}", path.display()))?;
        }
    }
    let fmt = format.unwrap_or_else(|| OutputFormat::from_path(path));
    let raw = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path).with_context(|| format!("create {}", path.display()))?);
    match fmt {
        OutputFormat::Raw => Ok(Writer::Raw(raw)),
        OutputFormat::Zstd => {
            let level = compression_level.clamp(1, 22);
            let enc = zstd::stream::Encoder::new(raw, level)
                .context("init zstd encoder")?;
            // Single-threaded zstd; multithreading requires a feature on the
            // zstd-sys crate we don't enable here. For 20GB+ outputs, the
            // bottleneck is typically downstream I/O anyway.
            Ok(Writer::Zstd(enc.auto_finish()))
        }
        OutputFormat::Xz => {
            let level = compression_level.clamp(0, 9) as u32;
            Ok(Writer::Xz(XzEncoder::new(raw, level)))
        }
        OutputFormat::Gzip => {
            let level = compression_level.clamp(1, 9) as u32;
            Ok(Writer::Gzip(GzEncoder::new(raw, Compression::new(level))))
        }
    }
}

/// A serde_json formatter that reproduces Python's `json.dumps` default
/// spacing: `", "` between key-value pairs, `": "` between key and value.
struct PythonCompatFormatter;

impl serde_json::ser::Formatter for PythonCompatFormatter {
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }

    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
}

/// Serialise one row and append a newline. Byte-identical to the Python
/// pipeline's `json.dumps(row, ensure_ascii=False) + "\n"` output.
pub fn write_row<W: Write>(mut writer: W, row: &Row) -> Result<()> {
    let mut ser = serde_json::Serializer::with_formatter(&mut writer, PythonCompatFormatter);
    row.serialize(&mut ser).context("serialize row")?;
    writer.write_all(b"\n").context("write newline")?;
    Ok(())
}

/// Streaming iterator over JSONL rows with automatic decompression.
///
/// Recognises `.zst`, `.xz`, `.gz` suffixes. Malformed lines are silently
/// skipped to match Python's try/except behaviour; empty lines are skipped.
pub struct JsonlLines {
    lines: Box<dyn Iterator<Item = std::io::Result<String>> + Send>,
}

impl JsonlLines {
    pub fn open(path: &Path) -> Result<Self> {
        let raw = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let reader: Box<dyn Read + Send> = match OutputFormat::from_path(path) {
            OutputFormat::Raw => Box::new(raw),
            OutputFormat::Zstd => Box::new(zstd::stream::Decoder::new(raw)?),
            OutputFormat::Xz => Box::new(xz2::read::XzDecoder::new(raw)),
            OutputFormat::Gzip => Box::new(flate2::read::GzDecoder::new(raw)),
        };
        let buf = BufReader::with_capacity(8 * 1024 * 1024, reader);
        Ok(Self {
            lines: Box::new(buf.lines()),
        })
    }
}

impl Iterator for JsonlLines {
    type Item = Result<Row>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let raw_line = match self.lines.next()? {
                Ok(s) => s,
                Err(e) => return Some(Err(anyhow::Error::from(e))),
            };
            let trimmed = raw_line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<Row>(trimmed) {
                Ok(row) => return Some(Ok(row)),
                // Match Python processors: drop malformed lines instead of
                // aborting the whole run.
                Err(_) => continue,
            }
        }
    }
}

/// Iterate JSONL rows in a fixed order over multiple files, cycling the list
/// indefinitely (used by the mixer to implement oversampling via reopen).
pub struct CyclingJsonl {
    paths: Vec<PathBuf>,
    current: Option<JsonlLines>,
    idx: usize,
    rows_this_cycle: usize,
}

impl CyclingJsonl {
    pub fn new(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            current: None,
            idx: 0,
            rows_this_cycle: 0,
        }
    }

    /// Advance to the next row across the file list, reopening files as we go
    /// around. Returns `None` only if a *complete* cycle through all files
    /// produced zero rows (file list empty or all files malformed).
    pub fn next_row(&mut self) -> Result<Option<Row>> {
        loop {
            if self.current.is_none() {
                if self.paths.is_empty() {
                    return Ok(None);
                }
                if self.idx >= self.paths.len() {
                    if self.rows_this_cycle == 0 {
                        return Ok(None);
                    }
                    self.idx = 0;
                    self.rows_this_cycle = 0;
                }
                let path = &self.paths[self.idx];
                self.current = Some(JsonlLines::open(path)?);
            }
            match self.current.as_mut().unwrap().next() {
                Some(Ok(row)) => {
                    self.rows_this_cycle += 1;
                    return Ok(Some(row));
                }
                Some(Err(e)) => return Err(e),
                None => {
                    self.current = None;
                    self.idx += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use tempfile::tempdir;

    fn write_jsonl(path: &Path, rows: &[Row]) {
        let mut f = std::fs::File::create(path).unwrap();
        for r in rows {
            let line = serde_json::to_string(r).unwrap();
            writeln!(f, "{}", line).unwrap();
        }
    }

    #[test]
    fn roundtrip_raw() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("t.jsonl");
        let rows = vec![
            Row::new("あい".into(), "愛".into(), "".into(), Some("src".into())),
            Row::new("かみ".into(), "神".into(), "前".into(), None),
        ];
        write_jsonl(&p, &rows);
        let back: Vec<_> = JsonlLines::open(&p).unwrap().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].reading, "あい");
        assert_eq!(back[1].context, "前");
    }

    #[test]
    fn cycling_iterates_until_budget() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("t.jsonl");
        let rows = vec![
            Row::new("a".into(), "A".into(), "".into(), None),
            Row::new("b".into(), "B".into(), "".into(), None),
        ];
        write_jsonl(&p, &rows);
        let mut cycler = CyclingJsonl::new(vec![p.clone()]);
        let mut surfaces = Vec::new();
        for _ in 0..5 {
            let row = cycler.next_row().unwrap().unwrap();
            surfaces.push(row.surface);
        }
        assert_eq!(surfaces, vec!["A", "B", "A", "B", "A"]);
    }
}
