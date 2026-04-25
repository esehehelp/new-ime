//! Subprocess `pdftotext -layout` over every downloaded PDF into
//! `text_dir/<ministry>/<stem>.txt`. Idempotent: already-converted files
//! are skipped. Concurrency limits the number of simultaneous pdftotext
//! subprocesses.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

pub fn run(pdf_dir: &Path, text_dir: &Path, concurrency: usize) -> Result<()> {
    let mut pdfs = Vec::new();
    for ministry_entry in
        fs::read_dir(pdf_dir).with_context(|| format!("read_dir {}", pdf_dir.display()))?
    {
        let ministry_entry = ministry_entry?;
        if !ministry_entry.file_type()?.is_dir() {
            continue;
        }
        let ministry = ministry_entry.file_name().to_string_lossy().to_string();
        for pdf_entry in fs::read_dir(ministry_entry.path())? {
            let pdf_entry = pdf_entry?;
            let path = pdf_entry.path();
            if path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("pdf"))
                == Some(true)
            {
                pdfs.push((ministry.clone(), path));
            }
        }
    }
    eprintln!(
        "[extract] {} PDFs found under {}",
        pdfs.len(),
        pdf_dir.display()
    );

    fs::create_dir_all(text_dir)?;
    let pdfs = Arc::new(Mutex::new(pdfs.into_iter()));
    let text_dir = text_dir.to_path_buf();
    let done = Arc::new(AtomicUsize::new(0));
    let skipped = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicUsize::new(0));

    thread::scope(|s| {
        for worker in 0..concurrency.max(1) {
            let pdfs = Arc::clone(&pdfs);
            let text_dir = text_dir.clone();
            let done = Arc::clone(&done);
            let skipped = Arc::clone(&skipped);
            let failed = Arc::clone(&failed);
            s.spawn(move || {
                loop {
                    let item = {
                        let mut guard = pdfs.lock().unwrap();
                        guard.next()
                    };
                    let (ministry, pdf_path) = match item {
                        Some(x) => x,
                        None => return,
                    };
                    let stem = pdf_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown");
                    let out_dir = text_dir.join(&ministry);
                    if let Err(e) = fs::create_dir_all(&out_dir) {
                        eprintln!("[w{worker}] mkdir {}: {}", out_dir.display(), e);
                        failed.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    let out_path: PathBuf = out_dir.join(format!("{}.txt", stem));
                    if out_path.exists() && out_path.metadata().map(|m| m.len()).unwrap_or(0) > 0 {
                        skipped.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    // `pdftotext -layout -enc UTF-8` keeps reading order roughly
                    // correct for 2-column white-paper layouts. `-nopgbrk`
                    // avoids embedding form-feed chars we'd have to strip.
                    let status = Command::new("pdftotext")
                        .arg("-layout")
                        .arg("-enc")
                        .arg("UTF-8")
                        .arg("-nopgbrk")
                        .arg(&pdf_path)
                        .arg(&out_path)
                        .status();
                    match status {
                        Ok(s) if s.success() => {
                            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                            if n % 100 == 0 {
                                eprintln!(
                                    "[extract] done={} skipped={} failed={}",
                                    n,
                                    skipped.load(Ordering::Relaxed),
                                    failed.load(Ordering::Relaxed)
                                );
                            }
                        }
                        Ok(s) => {
                            eprintln!("[w{worker}] pdftotext {} exit={}", pdf_path.display(), s);
                            let _ = fs::remove_file(&out_path);
                            failed.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            eprintln!(
                                "[w{worker}] spawn pdftotext for {}: {}",
                                pdf_path.display(),
                                e
                            );
                            failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            });
        }
    });

    eprintln!(
        "[extract] total done={} skipped={} failed={}",
        done.load(Ordering::Relaxed),
        skipped.load(Ordering::Relaxed),
        failed.load(Ordering::Relaxed)
    );
    Ok(())
}
