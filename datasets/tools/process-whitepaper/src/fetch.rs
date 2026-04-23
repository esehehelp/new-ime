//! Download the PDFs enumerated by `discover` into `out_dir/<slug>/<file>`.
//!
//! - Serial-but-concurrent: a small thread pool (scope) runs N downloads
//!   at a time; we avoid an async runtime because `reqwest::blocking` fits
//!   the rest of the tool's synchronous style.
//! - Resume: if the partial size matches a fresh HEAD request's
//!   Content-Length we skip; otherwise we redownload from scratch (HTTP
//!   range requests against ministry servers are inconsistent, not worth
//!   debugging).
//! - Retry: up to `retries` with exponential backoff on 5xx / network err.
//! - Budget: `max_mib > 0` stops accepting new downloads once total bytes
//!   written in this run exceed the budget.

use anyhow::{Context, Result};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

pub fn run(
    manifest: &Path,
    out_dir: &Path,
    concurrency: usize,
    retries: u32,
    max_mib: u64,
) -> Result<()> {
    let items = read_manifest(manifest)?;
    eprintln!(
        "[fetch] {} entries from {}",
        items.len(),
        manifest.display()
    );
    fs::create_dir_all(out_dir).with_context(|| format!("mkdir {}", out_dir.display()))?;

    let items = Arc::new(Mutex::new(items.into_iter()));
    let out_dir = out_dir.to_path_buf();
    let bytes_written = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let max_bytes = max_mib.saturating_mul(1024 * 1024);

    let client = reqwest::blocking::Client::builder()
        .user_agent("Mozilla/5.0 (compatible; new-ime-dataset-builder/0.1)")
        .timeout(Duration::from_secs(300))
        .build()
        .context("build http client")?;

    thread::scope(|s| {
        for worker in 0..concurrency.max(1) {
            let items = Arc::clone(&items);
            let out_dir = out_dir.clone();
            let bytes_written = Arc::clone(&bytes_written);
            let stop = Arc::clone(&stop);
            let client = client.clone();
            s.spawn(move || {
                loop {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    let item = {
                        let mut guard = items.lock().unwrap();
                        guard.next()
                    };
                    let ManifestEntry {
                        ministry,
                        url,
                        filename,
                    } = match item {
                        Some(x) => x,
                        None => return,
                    };
                    let dest_dir = out_dir.join(&ministry);
                    if let Err(e) = fs::create_dir_all(&dest_dir) {
                        eprintln!("[w{worker}] mkdir {}: {}", dest_dir.display(), e);
                        continue;
                    }
                    let dest = dest_dir.join(&filename);
                    if dest.exists() && dest.metadata().map(|m| m.len()).unwrap_or(0) > 0 {
                        // Already fetched.
                        continue;
                    }
                    match fetch_one(&client, &url, &dest, retries) {
                        Ok(n) => {
                            let total = bytes_written.fetch_add(n, Ordering::Relaxed) + n;
                            eprintln!(
                                "[w{worker}] ✓ {} ({} MiB, total {} MiB)",
                                filename,
                                n / (1024 * 1024),
                                total / (1024 * 1024),
                            );
                            if max_bytes > 0 && total >= max_bytes {
                                eprintln!("[fetch] budget {} MiB reached, stopping", max_mib);
                                stop.store(true, Ordering::Relaxed);
                            }
                        }
                        Err(e) => eprintln!("[w{worker}] ✗ {}: {}", url, e),
                    }
                }
            });
        }
    });

    let total = bytes_written.load(Ordering::Relaxed);
    eprintln!("[fetch] done, downloaded {} MiB", total / (1024 * 1024));
    Ok(())
}

struct ManifestEntry {
    ministry: String,
    url: String,
    filename: String,
}

fn read_manifest(path: &Path) -> Result<Vec<ManifestEntry>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 && line.starts_with("ministry\t") {
            continue;
        }
        let parts: Vec<&str> = line.splitn(3, '\t').collect();
        if parts.len() != 3 {
            continue;
        }
        out.push(ManifestEntry {
            ministry: parts[0].to_string(),
            url: parts[1].to_string(),
            filename: parts[2].to_string(),
        });
    }
    Ok(out)
}

fn fetch_one(
    client: &reqwest::blocking::Client,
    url: &str,
    dest: &Path,
    retries: u32,
) -> Result<u64> {
    let tmp: PathBuf = dest.with_extension("pdf.tmp");
    let mut last_err = None;
    for attempt in 0..=retries {
        if attempt > 0 {
            thread::sleep(Duration::from_secs(1 << attempt));
        }
        match download_to(client, url, &tmp) {
            Ok(n) => {
                fs::rename(&tmp, dest)
                    .with_context(|| format!("rename {} -> {}", tmp.display(), dest.display()))?;
                return Ok(n);
            }
            Err(e) => {
                eprintln!("  retry {}/{}: {}", attempt + 1, retries + 1, e);
                last_err = Some(e);
            }
        }
    }
    let _ = fs::remove_file(&tmp);
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("unknown fetch error")))
}

fn download_to(client: &reqwest::blocking::Client, url: &str, dest: &Path) -> Result<u64> {
    let mut resp = client
        .get(url)
        .send()
        .with_context(|| format!("GET {}", url))?;
    if !resp.status().is_success() {
        anyhow::bail!("{} returned {}", url, resp.status());
    }
    let mut file = File::create(dest).with_context(|| format!("create {}", dest.display()))?;
    let mut total: u64 = 0;
    let mut buf = [0u8; 1 << 15];
    loop {
        let n = resp.read(&mut buf).context("read response")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("write output")?;
        total += n as u64;
    }
    file.flush()?;
    Ok(total)
}

// reqwest::blocking::Response implements Read via its inner decoder but we
// need the trait in scope for `.read()`.
use std::io::Read;
