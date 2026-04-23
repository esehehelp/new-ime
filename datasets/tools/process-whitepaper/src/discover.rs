//! Ministry index scraping. Each ministry entry is a curated (name, url)
//! anchor. We follow the page, collect every `<a href="...pdf">`, and if
//! `follow_depth > 0` we recurse into same-host sub-pages that smell like
//! yearly archives (e.g. `r04`, `r05`, `h30`, `2022`, `2023`).
//!
//! Output is a TSV manifest: `ministry\turl\tfilename\n`.

use anyhow::{Context, Result};
use scraper::{Html, Selector};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Duration;
use url::Url;

/// One ministry / white paper series to crawl.
struct Source {
    /// Short ministry tag used in output filenames (ASCII, no spaces).
    slug: &'static str,
    /// Display name.
    name: &'static str,
    /// Entry pages — the crawl starts here.
    entries: &'static [&'static str],
    /// 0 = collect PDFs only from the entry page; 1 = also follow one level
    /// of same-host links that look like year archives.
    follow_depth: u8,
}

/// Curated source list. Adjust freely; scraping is resilient to broken
/// ministries (errors are reported per-entry and don't abort the run).
const SOURCES: &[Source] = &[
    Source {
        slug: "cao-keizai",
        name: "内閣府 経済財政白書",
        entries: &["https://www5.cao.go.jp/keizai3/keizaiwp-index.html"],
        follow_depth: 1,
    },
    Source {
        slug: "cao-gender",
        name: "内閣府 男女共同参画白書",
        entries: &["https://www.gender.go.jp/about_danjo/whitepaper/index.html"],
        follow_depth: 1,
    },
    Source {
        slug: "soumu-ict",
        name: "総務省 情報通信白書",
        entries: &["https://www.soumu.go.jp/johotsusintokei/whitepaper/index.html"],
        follow_depth: 1,
    },
    Source {
        slug: "meti",
        name: "経済産業省 白書",
        entries: &["https://www.meti.go.jp/report/whitepaper/index.html"],
        follow_depth: 1,
    },
    Source {
        slug: "mhlw",
        name: "厚生労働省 厚生労働白書",
        entries: &["https://www.mhlw.go.jp/wp/hakusyo/index.html"],
        follow_depth: 1,
    },
    Source {
        slug: "env",
        name: "環境省 環境白書",
        entries: &["https://www.env.go.jp/policy/hakusyo/"],
        follow_depth: 1,
    },
    Source {
        slug: "mlit",
        name: "国土交通省 国土交通白書",
        entries: &["https://www.mlit.go.jp/statistics/file000004.html"],
        follow_depth: 1,
    },
    Source {
        slug: "mext",
        name: "文部科学省 白書",
        entries: &["https://www.mext.go.jp/b_menu/hakusho/index.htm"],
        follow_depth: 1,
    },
    Source {
        slug: "mod",
        name: "防衛省 防衛白書",
        entries: &["https://www.mod.go.jp/j/publication/wp/"],
        follow_depth: 1,
    },
    Source {
        slug: "maff",
        name: "農林水産省 食料農業農村白書",
        entries: &["https://www.maff.go.jp/j/wpaper/index.html"],
        follow_depth: 1,
    },
    Source {
        slug: "npa",
        name: "警察庁 警察白書",
        entries: &["https://www.npa.go.jp/hakusyo/index.html"],
        follow_depth: 1,
    },
    Source {
        slug: "moj",
        name: "法務省 犯罪白書",
        entries: &["https://www.moj.go.jp/housouken/houso_hakusho2.html"],
        follow_depth: 1,
    },
];

pub fn run(out: &Path, only: Option<&str>) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
    }
    let filter: Option<BTreeSet<&str>> = only.map(|s| s.split(',').map(str::trim).collect());

    let client = reqwest::blocking::Client::builder()
        .user_agent("Mozilla/5.0 (compatible; new-ime-dataset-builder/0.1)")
        .timeout(Duration::from_secs(30))
        .build()
        .context("build http client")?;

    let mut writer =
        BufWriter::new(File::create(out).with_context(|| format!("create {}", out.display()))?);
    writeln!(writer, "ministry\turl\tfilename")?;

    let mut total_pdfs = 0usize;
    for src in SOURCES {
        if let Some(ref f) = filter {
            if !f.contains(src.slug) {
                continue;
            }
        }
        eprintln!("[discover] {} ({})", src.name, src.slug);
        let mut pdfs: BTreeSet<Url> = BTreeSet::new();
        for entry in src.entries {
            match crawl(&client, entry, src.follow_depth, &mut pdfs) {
                Ok(()) => {}
                Err(e) => eprintln!("  ! {}: {}", entry, e),
            }
        }
        eprintln!("  = {} PDFs", pdfs.len());
        for url in &pdfs {
            let fname = filename_for(url);
            writeln!(writer, "{}\t{}\t{}", src.slug, url, fname)?;
        }
        total_pdfs += pdfs.len();
    }
    writer.flush()?;
    eprintln!("[discover] wrote {} rows to {}", total_pdfs, out.display());
    Ok(())
}

/// Walk from `start` collecting every PDF link into `out`. When
/// `depth > 0`, also follow same-host anchor links whose URL or link text
/// looks like a year archive marker (e.g. "令和", "平成", "r04", "h30",
/// "2023"). Depth limits recursion to avoid crawling full ministry sites.
fn crawl(
    client: &reqwest::blocking::Client,
    start: &str,
    depth: u8,
    out: &mut BTreeSet<Url>,
) -> Result<()> {
    let start_url = Url::parse(start).context("parse start url")?;
    let html = fetch_html(client, &start_url)?;
    collect_pdfs(&html, &start_url, out);

    if depth == 0 {
        return Ok(());
    }

    let subpages = collect_yearly_subpages(&html, &start_url);
    for sub in subpages {
        // Skip cross-host to avoid escaping the ministry.
        if sub.host_str() != start_url.host_str() {
            continue;
        }
        match fetch_html(client, &sub) {
            Ok(sub_html) => collect_pdfs(&sub_html, &sub, out),
            Err(e) => eprintln!("  ~ {}: {}", sub, e),
        }
    }
    Ok(())
}

fn fetch_html(client: &reqwest::blocking::Client, url: &Url) -> Result<Html> {
    let resp = client
        .get(url.clone())
        .send()
        .with_context(|| format!("GET {}", url))?;
    if !resp.status().is_success() {
        anyhow::bail!("{} returned {}", url, resp.status());
    }
    let bytes = resp.bytes().with_context(|| format!("read body {}", url))?;
    // Ministry pages are usually UTF-8 these days; older ones may be
    // shift-jis. For robustness, try UTF-8 first, fall back to lossy.
    let body = match std::str::from_utf8(&bytes) {
        Ok(s) => s.to_string(),
        Err(_) => String::from_utf8_lossy(&bytes).to_string(),
    };
    Ok(Html::parse_document(&body))
}

fn collect_pdfs(html: &Html, base: &Url, out: &mut BTreeSet<Url>) {
    let sel = Selector::parse("a[href]").unwrap();
    for a in html.select(&sel) {
        let href = match a.value().attr("href") {
            Some(h) => h,
            None => continue,
        };
        if !href.to_ascii_lowercase().ends_with(".pdf") {
            continue;
        }
        if let Ok(resolved) = base.join(href) {
            out.insert(resolved);
        }
    }
}

fn collect_yearly_subpages(html: &Html, base: &Url) -> Vec<Url> {
    let sel = Selector::parse("a[href]").unwrap();
    let mut subs = BTreeSet::new();
    for a in html.select(&sel) {
        let href = match a.value().attr("href") {
            Some(h) => h,
            None => continue,
        };
        // Skip PDFs (we already collected them) and javascript / anchors.
        let lower = href.to_ascii_lowercase();
        if lower.starts_with("javascript:") || lower.starts_with('#') {
            continue;
        }
        if lower.ends_with(".pdf") {
            continue;
        }
        let text = a.text().collect::<String>();
        let looks_yearly = href_or_text_looks_yearly(href, &text);
        if !looks_yearly {
            continue;
        }
        if let Ok(resolved) = base.join(href) {
            subs.insert(resolved);
        }
    }
    subs.into_iter().collect()
}

fn href_or_text_looks_yearly(href: &str, text: &str) -> bool {
    let combined_lower = format!("{} {}", href.to_ascii_lowercase(), text);
    // Era markers and common year patterns.
    const ERA_HINTS: &[&str] = &["令和", "平成", "昭和"];
    for hint in ERA_HINTS {
        if text.contains(hint) {
            return true;
        }
    }
    // r04 / r05 / h30 style path segments.
    let bytes = combined_lower.as_bytes();
    for i in 0..bytes.len().saturating_sub(3) {
        let c = bytes[i];
        if (c == b'r' || c == b'h' || c == b's')
            && bytes[i + 1].is_ascii_digit()
            && bytes[i + 2].is_ascii_digit()
        {
            return true;
        }
    }
    // 20xx / 19xx path segments.
    for i in 0..bytes.len().saturating_sub(4) {
        if bytes[i] == b'2'
            && bytes[i + 1] == b'0'
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
        {
            return true;
        }
        if bytes[i] == b'1'
            && bytes[i + 1] == b'9'
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
        {
            return true;
        }
    }
    false
}

fn filename_for(url: &Url) -> String {
    let last = url
        .path_segments()
        .and_then(|s| s.filter(|x| !x.is_empty()).last())
        .unwrap_or("unknown.pdf")
        .to_string();
    // sanitize to a safe filesystem name
    last.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
