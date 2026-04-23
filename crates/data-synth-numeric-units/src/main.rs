//! data-synth-numeric-units: template-based synth for unit-bearing numeric phrases
//! that are *not* already covered by `legacy dataset tools`.
//!
//! The existing Python generators cover Japanese counter words (本, 個, 人,
//! 回, 時, 分, 秒, 円, 年, 月, 日, 歳, 枚, 冊, 匹, 頭, 羽, 軒, 台, 度, 番, 階,
//! 丁目, 世紀, センチ, メートル), time, date, decimal, currency, ordinal,
//! fraction. This tool *intentionally avoids duplicating* those and focuses on
//! gaps the student can't learn elsewhere:
//!
//!   - SI metric units (kg, g, mg, ml, l, km, cm, mm, Hz, Pa, W, J, V, A)
//!   - Imperial units (lb, oz, ft, in, yd, mi, gal, mph)
//!   - Temperature (°C, °F)
//!   - File sizes (KB, MB, GB, TB)
//!   - International currency (ドル, ユーロ, ポンド, 元, ウォン)
//!   - Percent / basis points (%, パーセント)
//!
//! Output schema matches the bunsetsu corpora (and existing synth pools):
//!   {"reading", "surface", "left_context_surface", "left_context_reading",
//!    "span_bunsetsu": 1, "source": "synth_<subtype>",
//!    "sentence_id": "synth_<subtype>:<idx>"}

use anyhow::{Context, Result};
use clap::Parser;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "data-synth-numeric-units",
    about = "Generate unit-bearing numeric synth rows"
)]
struct Cli {
    /// Output JSONL path (bunsetsu-schema compatible).
    #[arg(long)]
    output: PathBuf,
    /// Total rows to emit across all subtypes. The tool allocates the budget
    /// proportional to each subtype's fixed weight.
    #[arg(long, default_value_t = 1_500_000)]
    target_size: usize,
    /// PRNG seed. Deterministic output when the seed and version are fixed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Serialize)]
struct Row<'a> {
    reading: String,
    surface: String,
    left_context_surface: &'a str,
    left_context_reading: &'a str,
    span_bunsetsu: u32,
    source: &'a str,
    sentence_id: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(parent) = cli.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut out = BufWriter::with_capacity(
        8 * 1024 * 1024,
        File::create(&cli.output).with_context(|| format!("create {}", cli.output.display()))?,
    );

    // Each subtype gets a weight; the global target_size is split proportional.
    let subtypes: &[(
        &str,
        u32,
        fn(&mut StdRng, usize, &mut dyn Write, &str) -> Result<usize>,
    )] = &[
        ("synth_si", 20, gen_si),
        ("synth_imperial", 10, gen_imperial),
        ("synth_temp", 8, gen_temp),
        ("synth_filesize", 8, gen_filesize),
        ("synth_forex", 15, gen_forex),
        ("synth_percent", 10, gen_percent),
    ];
    let total_weight: u32 = subtypes.iter().map(|s| s.1).sum();

    let mut rng = StdRng::seed_from_u64(cli.seed);
    let mut emitted_total = 0usize;
    for (name, weight, gen) in subtypes {
        let budget = (cli.target_size * (*weight as usize)) / (total_weight as usize);
        eprintln!("[{}] budget={}", name, budget);
        let n = gen(&mut rng, budget, &mut out, name)?;
        eprintln!("  wrote {}", n);
        emitted_total += n;
    }
    out.flush()?;
    eprintln!(
        "[data-synth-numeric-units] wrote {} rows to {}",
        emitted_total,
        cli.output.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Number readings
// ---------------------------------------------------------------------------

/// Read a small-ish integer as a single Japanese noun phrase. Covers 0..9999
/// with compound reading construction. For the synth pool this is enough; we
/// rarely need arbitrary precision.
fn number_reading(n: u64) -> Option<String> {
    if n == 0 {
        return Some("ぜろ".into());
    }
    if n >= 10_000_000_000 {
        return None; // not worth handling; synth doesn't need it
    }
    // Break into 億, 万, ones chunks of 4 digits each.
    let oku = n / 100_000_000;
    let rem_after_oku = n % 100_000_000;
    let man = rem_after_oku / 10_000;
    let ones = rem_after_oku % 10_000;
    let mut out = String::new();
    if oku > 0 {
        out.push_str(&digits_1_9999(oku)?);
        out.push_str("おく");
    }
    if man > 0 {
        out.push_str(&digits_1_9999(man)?);
        out.push_str("まん");
    }
    if ones > 0 {
        out.push_str(&digits_1_9999(ones)?);
    }
    Some(out)
}

/// Reading of a number in the range 1..=9999.
fn digits_1_9999(n: u64) -> Option<String> {
    if !(1..=9999).contains(&n) {
        return None;
    }
    let thousand = n / 1000;
    let hundred = (n / 100) % 10;
    let ten = (n / 10) % 10;
    let one = n % 10;
    let mut r = String::new();
    // 千
    match thousand {
        0 => {}
        1 => r.push_str("せん"),
        2 => r.push_str("にせん"),
        3 => r.push_str("さんぜん"),
        4 => r.push_str("よんせん"),
        5 => r.push_str("ごせん"),
        6 => r.push_str("ろくせん"),
        7 => r.push_str("ななせん"),
        8 => r.push_str("はっせん"),
        9 => r.push_str("きゅうせん"),
        _ => {}
    }
    // 百
    match hundred {
        0 => {}
        1 => r.push_str("ひゃく"),
        2 => r.push_str("にひゃく"),
        3 => r.push_str("さんびゃく"),
        4 => r.push_str("よんひゃく"),
        5 => r.push_str("ごひゃく"),
        6 => r.push_str("ろっぴゃく"),
        7 => r.push_str("ななひゃく"),
        8 => r.push_str("はっぴゃく"),
        9 => r.push_str("きゅうひゃく"),
        _ => {}
    }
    // 十
    match ten {
        0 => {}
        1 => r.push_str("じゅう"),
        2 => r.push_str("にじゅう"),
        3 => r.push_str("さんじゅう"),
        4 => r.push_str("よんじゅう"),
        5 => r.push_str("ごじゅう"),
        6 => r.push_str("ろくじゅう"),
        7 => r.push_str("ななじゅう"),
        8 => r.push_str("はちじゅう"),
        9 => r.push_str("きゅうじゅう"),
        _ => {}
    }
    // 一の位
    match one {
        0 => {}
        1 => r.push_str("いち"),
        2 => r.push_str("に"),
        3 => r.push_str("さん"),
        4 => r.push_str("よん"),
        5 => r.push_str("ご"),
        6 => r.push_str("ろく"),
        7 => r.push_str("なな"),
        8 => r.push_str("はち"),
        9 => r.push_str("きゅう"),
        _ => {}
    }
    Some(r)
}

// ---------------------------------------------------------------------------
// Generators — each produces Row values for a specific subtype.
// ---------------------------------------------------------------------------

/// Pick a random integer bias toward small-ish values (1..=20, then tens).
fn pick_number(rng: &mut StdRng) -> u64 {
    let bucket: u32 = rng.gen_range(0..100);
    match bucket {
        0..=40 => rng.gen_range(1..=20),
        41..=70 => (rng.gen_range(1..=20) * 10) as u64,
        71..=85 => (rng.gen_range(1..=20) * 100) as u64,
        86..=95 => (rng.gen_range(1..=10) * 1000) as u64,
        96..=99 => (rng.gen_range(1..=100) * 10_000) as u64,
        _ => 1,
    }
}

/// Pick a decimal like 3.14, 0.5, 12.8. Returns `(surface, reading)`.
fn pick_decimal(rng: &mut StdRng) -> (String, String) {
    let integer = rng.gen_range(0..=999u64);
    let fraction_digits = rng.gen_range(1..=3);
    let fraction: u64 = rng.gen_range(1..10u64.pow(fraction_digits));
    let surface = format!(
        "{}.{:0width$}",
        integer,
        fraction,
        width = fraction_digits as usize
    );
    let int_reading = if integer == 0 {
        "れい".to_string()
    } else {
        number_reading(integer).unwrap_or_default()
    };
    let frac_reading = fraction_reading_digits(fraction, fraction_digits as usize);
    let reading = format!("{}てん{}", int_reading, frac_reading);
    (surface, reading)
}

fn fraction_reading_digits(frac: u64, width: usize) -> String {
    let formatted = format!("{:0width$}", frac, width = width);
    let mut r = String::new();
    for c in formatted.chars() {
        r.push_str(match c {
            '0' => "れい",
            '1' => "いち",
            '2' => "に",
            '3' => "さん",
            '4' => "よん",
            '5' => "ご",
            '6' => "ろく",
            '7' => "なな",
            '8' => "はち",
            '9' => "きゅう",
            _ => "",
        });
    }
    r
}

const PARTICLES: &[(&str, &str)] = &[
    ("", ""),
    ("の", "の"),
    ("を", "を"),
    ("が", "が"),
    ("に", "に"),
    ("で", "で"),
    ("と", "と"),
    ("は", "は"),
    ("から", "から"),
    ("まで", "まで"),
];

/// Write one row to `w`.
fn emit(
    w: &mut dyn Write,
    reading: String,
    surface: String,
    source: &str,
    idx: usize,
) -> Result<()> {
    let row = Row {
        reading,
        surface,
        left_context_surface: "",
        left_context_reading: "",
        span_bunsetsu: 1,
        source,
        sentence_id: format!("{}:{}", source, idx),
    };
    // Python-compat formatter.
    struct PyFmt;
    impl serde_json::ser::Formatter for PyFmt {
        fn begin_object_key<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                w.write_all(b", ")
            }
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
        ) -> std::io::Result<()> {
            w.write_all(b": ")
        }
        fn begin_array_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                w.write_all(b", ")
            }
        }
    }
    let mut buf = Vec::with_capacity(256);
    {
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyFmt);
        row.serialize(&mut ser).context("serialize row")?;
    }
    w.write_all(&buf)?;
    w.write_all(b"\n")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// SI units: kg, g, mg, ml, l, km, cm, mm, Hz, Pa, W, J, V, A
// ---------------------------------------------------------------------------
fn gen_si(rng: &mut StdRng, budget: usize, w: &mut dyn Write, source: &str) -> Result<usize> {
    // (surface_suffix, reading_suffix) pairs. Most Japanese technical writing
    // keeps the unit in katakana OR latin letters; we emit latin-letter form
    // with katakana reading, which is what IME users type.
    let units: &[(&str, &str)] = &[
        ("kg", "きろぐらむ"),
        ("g", "ぐらむ"),
        ("mg", "みりぐらむ"),
        ("ml", "みりりっとる"),
        ("l", "りっとる"),
        ("L", "りっとる"),
        ("km", "きろめーとる"),
        ("cm", "せんちめーとる"),
        ("mm", "みりめーとる"),
        ("Hz", "へるつ"),
        ("kHz", "きろへるつ"),
        ("MHz", "めがへるつ"),
        ("Pa", "ぱすかる"),
        ("kPa", "きろぱすかる"),
        ("MPa", "めがぱすかる"),
        ("W", "わっと"),
        ("kW", "きろわっと"),
        ("MW", "めがわっと"),
        ("J", "じゅーる"),
        ("kJ", "きろじゅーる"),
        ("V", "ぼると"),
        ("kV", "きろぼると"),
        ("A", "あんぺあ"),
        ("mA", "みりあんぺあ"),
        ("dB", "でしべる"),
    ];
    let mut written = 0;
    let mut idx = 0usize;
    while written < budget {
        let (u_surface, u_reading) = units.choose(rng).unwrap();
        let n_int = pick_number(rng);
        let n_reading = number_reading(n_int).unwrap_or_default();
        let use_decimal = rng.gen_bool(0.25);
        let (num_surface, num_reading) = if use_decimal {
            pick_decimal(rng)
        } else {
            (n_int.to_string(), n_reading.clone())
        };
        let (p_r, p_s) = PARTICLES.choose(rng).unwrap();
        let surface = format!("{}{}{}", num_surface, u_surface, p_s);
        let reading = format!("{}{}{}", num_reading, u_reading, p_r);
        emit(w, reading, surface, source, idx)?;
        idx += 1;
        written += 1;
    }
    Ok(written)
}

// ---------------------------------------------------------------------------
// Imperial: lb, oz, ft, in, yd, mi, gal, mph
// ---------------------------------------------------------------------------
fn gen_imperial(rng: &mut StdRng, budget: usize, w: &mut dyn Write, source: &str) -> Result<usize> {
    let units: &[(&str, &str)] = &[
        ("lb", "ぽんど"),
        ("oz", "おんす"),
        ("ft", "ふぃーと"),
        ("in", "いんち"),
        ("yd", "やーど"),
        ("mi", "まいる"),
        ("gal", "がろん"),
        ("mph", "まいるぱーあわー"),
    ];
    let mut written = 0;
    let mut idx = 0usize;
    while written < budget {
        let (u_surface, u_reading) = units.choose(rng).unwrap();
        let n = pick_number(rng);
        let n_reading = number_reading(n).unwrap_or_default();
        let (p_r, p_s) = PARTICLES.choose(rng).unwrap();
        let surface = format!("{}{}{}", n, u_surface, p_s);
        let reading = format!("{}{}{}", n_reading, u_reading, p_r);
        emit(w, reading, surface, source, idx)?;
        idx += 1;
        written += 1;
    }
    Ok(written)
}

// ---------------------------------------------------------------------------
// Temperature: °C, °F, 度
// ---------------------------------------------------------------------------
fn gen_temp(rng: &mut StdRng, budget: usize, w: &mut dyn Write, source: &str) -> Result<usize> {
    let mut written = 0;
    let mut idx = 0usize;
    while written < budget {
        let sign_positive = rng.gen_bool(0.85);
        let magnitude = rng.gen_range(0u64..=120);
        let use_decimal = rng.gen_bool(0.3);
        let (num_surface, num_reading) = if use_decimal {
            pick_decimal(rng)
        } else {
            (
                magnitude.to_string(),
                if magnitude == 0 {
                    "れい".to_string()
                } else {
                    number_reading(magnitude).unwrap_or_default()
                },
            )
        };
        let (sign_surface, sign_reading) = if sign_positive {
            ("", "")
        } else {
            ("マイナス", "まいなす")
        };
        // Pick unit spelling style.
        let unit_style: u8 = rng.gen_range(0..4);
        let (u_surface, u_reading) = match unit_style {
            0 => ("°C", "ど"),
            1 => ("℃", "ど"),
            2 => ("°F", "かし"),
            _ => ("度", "ど"),
        };
        let (p_r, p_s) = PARTICLES.choose(rng).unwrap();
        let surface = format!("{}{}{}{}", sign_surface, num_surface, u_surface, p_s);
        let reading = format!("{}{}{}{}", sign_reading, num_reading, u_reading, p_r);
        emit(w, reading, surface, source, idx)?;
        idx += 1;
        written += 1;
    }
    Ok(written)
}

// ---------------------------------------------------------------------------
// File sizes: 10KB, 3MB, 2GB, 512TB ...
// ---------------------------------------------------------------------------
fn gen_filesize(rng: &mut StdRng, budget: usize, w: &mut dyn Write, source: &str) -> Result<usize> {
    let units: &[(&str, &str)] = &[
        ("KB", "きろばいと"),
        ("MB", "めがばいと"),
        ("GB", "ぎがばいと"),
        ("TB", "てらばいと"),
        ("PB", "ぺたばいと"),
        ("bps", "びーぴーえす"),
        ("Mbps", "めがびーぴーえす"),
        ("Gbps", "ぎがびーぴーえす"),
    ];
    let mut written = 0;
    let mut idx = 0usize;
    while written < budget {
        let (u_surface, u_reading) = units.choose(rng).unwrap();
        let n = pick_number(rng);
        let n_reading = number_reading(n).unwrap_or_default();
        let (p_r, p_s) = PARTICLES.choose(rng).unwrap();
        let surface = format!("{}{}{}", n, u_surface, p_s);
        let reading = format!("{}{}{}", n_reading, u_reading, p_r);
        emit(w, reading, surface, source, idx)?;
        idx += 1;
        written += 1;
    }
    Ok(written)
}

// ---------------------------------------------------------------------------
// International currency: ドル, ユーロ, ポンド, 元, ウォン, 香港ドル, 台湾元
// ---------------------------------------------------------------------------
fn gen_forex(rng: &mut StdRng, budget: usize, w: &mut dyn Write, source: &str) -> Result<usize> {
    let units: &[(&str, &str)] = &[
        ("ドル", "どる"),
        ("ユーロ", "ゆーろ"),
        ("ポンド", "ぽんど"),
        ("元", "げん"),
        ("ウォン", "うぉん"),
        ("フラン", "ふらん"),
        ("ルーブル", "るーぶる"),
        ("ペソ", "ぺそ"),
        ("ルピー", "るぴー"),
        ("バーツ", "ばーつ"),
    ];
    let prefix_opts: &[(&str, &str)] = &[("", ""), ("約", "やく"), ("およそ", "およそ")];
    let mut written = 0;
    let mut idx = 0usize;
    while written < budget {
        let (u_surface, u_reading) = units.choose(rng).unwrap();
        let (pre_s, pre_r) = prefix_opts.choose(rng).unwrap();
        let n = pick_number(rng);
        let n_reading = number_reading(n).unwrap_or_default();
        // Large counts shift to 万/億 expressions which the Python pool already
        // has for yen. Keep this pool purely "latin-number + katakana unit".
        let (p_r, p_s) = PARTICLES.choose(rng).unwrap();
        let surface = format!("{}{}{}{}", pre_s, n, u_surface, p_s);
        let reading = format!("{}{}{}{}", pre_r, n_reading, u_reading, p_r);
        emit(w, reading, surface, source, idx)?;
        idx += 1;
        written += 1;
    }
    Ok(written)
}

// ---------------------------------------------------------------------------
// Percent, basis points, per-mille
// ---------------------------------------------------------------------------
fn gen_percent(rng: &mut StdRng, budget: usize, w: &mut dyn Write, source: &str) -> Result<usize> {
    let units: &[(&str, &str)] = &[
        ("%", "ぱーせんと"),
        ("パーセント", "ぱーせんと"),
        ("割", "わり"),
        ("分", "ぶ"),
        ("厘", "りん"),
    ];
    let mut written = 0;
    let mut idx = 0usize;
    while written < budget {
        let (u_surface, u_reading) = units.choose(rng).unwrap();
        let n = if rng.gen_bool(0.6) {
            rng.gen_range(1..=100)
        } else {
            rng.gen_range(1..=999)
        };
        let use_decimal = rng.gen_bool(0.2) && *u_surface != "割";
        let (num_surface, num_reading) = if use_decimal {
            pick_decimal(rng)
        } else {
            (n.to_string(), number_reading(n).unwrap_or_default())
        };
        let (p_r, p_s) = PARTICLES.choose(rng).unwrap();
        let surface = format!("{}{}{}", num_surface, u_surface, p_s);
        let reading = format!("{}{}{}", num_reading, u_reading, p_r);
        emit(w, reading, surface, source, idx)?;
        idx += 1;
        written += 1;
    }
    Ok(written)
}
