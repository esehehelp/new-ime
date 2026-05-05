//! KenLM character-level scorer + MoE mixture.
//!
//! Wraps the C shim defined in `csrc/kenlm_shim.cpp`. Matches the semantics
//! of the Python `KenLMCharScorer` (one KenLM "word" per UTF-8 codepoint,
//! BOS on, natural-log probability of the whole prefix) so fused CTC+LM
//! scores are directly comparable to the evaluation harness.
//!
//! The `KenLMMixture` variant keeps the archived evaluation semantics:
//! Σ_d w_d · logp_d(prefix) with per-(prefix, weight) cache.
//! `CategoryEstimator` computes the weights per input reading using the same
//! char-class heuristics as the historical evaluation path.

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CString};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Result};

// The KenLM C shim is built and linked when build.rs finds the static
// libraries (Windows MSVC: kenlm.lib + kenlm_util.lib, Linux gnu:
// libkenlm.a + libkenlm_util.a). On platforms / configurations where the
// libs are missing, build.rs skips the shim build and `has_kenlm` is not
// set; the stub variants below take over so the engine still compiles
// and runs CTC-only. Bench configs that depend on KenLM are silently
// downgraded (KenLM::load returns an error) in that case.
#[cfg(has_kenlm)]
unsafe extern "C" {
    fn kenlm_shim_load(path: *const c_char) -> *mut c_void;
    fn kenlm_shim_free(handle: *mut c_void);
    fn kenlm_shim_score(handle: *mut c_void, utf8: *const c_char, len: usize) -> f32;
}

#[cfg(not(has_kenlm))]
unsafe fn kenlm_shim_load(_path: *const c_char) -> *mut c_void {
    std::ptr::null_mut()
}

#[cfg(not(has_kenlm))]
unsafe fn kenlm_shim_free(_handle: *mut c_void) {}

#[cfg(not(has_kenlm))]
unsafe fn kenlm_shim_score(_handle: *mut c_void, _utf8: *const c_char, _len: usize) -> f32 {
    0.0
}

/// Trait shared by the single-model scorer and the MoE mixture so the beam
/// search code can hold `Option<&dyn LmScorer>` without caring which variant
/// is active.
pub trait LmScorer: Send {
    fn score_ids(&self, ids: &[u32]) -> f32;
    fn clear_cache(&self) {}
}

pub struct KenLM {
    handle: Mutex<*mut c_void>,
}

unsafe impl Send for KenLM {}
unsafe impl Sync for KenLM {}

impl KenLM {
    pub fn load(path: &Path) -> Result<Self> {
        let cstr = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| anyhow!("kenlm path contained NUL: {}", e))?;
        let handle = unsafe { kenlm_shim_load(cstr.as_ptr()) };
        if handle.is_null() {
            return Err(anyhow!("kenlm failed to load {}", path.display()));
        }
        Ok(Self {
            handle: Mutex::new(handle),
        })
    }

    /// Natural-log probability of the full prefix (BOS on, EOS off).
    pub fn score(&self, prefix_utf8: &str) -> f32 {
        if prefix_utf8.is_empty() {
            return 0.0;
        }
        let guard = self.handle.lock().expect("kenlm mutex poisoned");
        unsafe {
            kenlm_shim_score(
                *guard,
                prefix_utf8.as_ptr() as *const c_char,
                prefix_utf8.len(),
            )
        }
    }
}

impl Drop for KenLM {
    fn drop(&mut self) {
        if let Ok(guard) = self.handle.lock() {
            unsafe {
                kenlm_shim_free(*guard);
            }
            drop(guard);
        }
    }
}

/// Single-domain character scorer with a prefix cache.
pub struct KenLMCharScorer {
    model: KenLM,
    cache: Mutex<HashMap<Vec<u32>, f32>>,
    tokenizer: crate::SharedCharTokenizer,
}

impl KenLMCharScorer {
    pub fn new(model: KenLM, tokenizer: crate::SharedCharTokenizer) -> Self {
        Self {
            model,
            cache: Mutex::new(HashMap::new()),
            tokenizer,
        }
    }
}

impl LmScorer for KenLMCharScorer {
    fn score_ids(&self, ids: &[u32]) -> f32 {
        if ids.is_empty() {
            return 0.0;
        }
        {
            let cache = self.cache.lock().expect("cache mutex poisoned");
            if let Some(&v) = cache.get(ids) {
                return v;
            }
        }
        let text = self.tokenizer.decode(ids);
        let score = self.model.score(&text);
        let mut cache = self.cache.lock().expect("cache mutex poisoned");
        cache.insert(ids.to_vec(), score);
        score
    }

    fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }
}

// ---------------- MoE mixture ----------------

/// Weighted combination of multiple domain KenLMs. Direct port of
/// `KenLMMixtureScorer`. Weights are set before each `convert()` call via
/// `set_weights` (usually driven by `CategoryEstimator`); the per-prefix
/// cache is keyed on (ids, weight_fingerprint) so a weight change only
/// invalidates entries that had domain contributions affected by it.
pub struct KenLMMixture {
    /// Stable ordering of domains (matches the keys passed to `load`). The
    /// weight vector is indexed by position, avoiding per-call hashmap
    /// lookups in the hot path.
    domains: Vec<String>,
    models: Vec<KenLM>,
    weights: Mutex<Vec<f32>>,
    cache: Mutex<HashMap<(Vec<u32>, u64), f32>>,
    tokenizer: crate::SharedCharTokenizer,
}

impl KenLMMixture {
    /// `paths` maps domain name (e.g. "general" / "tech" / "entity") to
    /// KenLM `.bin` path. Domains load in iteration order; use a
    /// deterministic container (BTreeMap) if you need stable fingerprints
    /// across processes.
    pub fn load(
        paths: &HashMap<String, std::path::PathBuf>,
        tokenizer: crate::SharedCharTokenizer,
    ) -> Result<Self> {
        if paths.is_empty() {
            return Err(anyhow!("KenLMMixture needs at least one model"));
        }
        let mut domains: Vec<String> = paths.keys().cloned().collect();
        domains.sort();
        // Parallel load: each KenLM `.bin` is 1-3 GB and load is dominated
        // by mmap + initial page-in. Sequential load took 2-5s on cold
        // cache for the typical general+tech+entity set; spawning one
        // thread per file overlaps the page faults and brings wall time
        // close to the largest single file.
        let handles: Vec<_> = domains
            .iter()
            .map(|d| {
                let p = paths[d].clone();
                std::thread::spawn(move || KenLM::load(&p))
            })
            .collect();
        let mut models: Vec<KenLM> = Vec::with_capacity(domains.len());
        for h in handles {
            let model = h
                .join()
                .map_err(|_| anyhow!("kenlm load thread panicked"))??;
            models.push(model);
        }
        let n = domains.len();
        let uniform = vec![1.0 / n as f32; n];
        Ok(Self {
            domains,
            models,
            weights: Mutex::new(uniform),
            cache: Mutex::new(HashMap::new()),
            tokenizer,
        })
    }

    pub fn domains(&self) -> &[String] {
        &self.domains
    }

    /// Set active domain weights. Missing keys default to 0; unknown keys
    /// are ignored. The cache is NOT cleared — entries are keyed on a
    /// fingerprint of the weight vector, so old entries simply stop being
    /// hit after the swap.
    pub fn set_weights(&self, weights: &HashMap<String, f32>) {
        let mut w = self.weights.lock().expect("weights mutex poisoned");
        for (i, dom) in self.domains.iter().enumerate() {
            w[i] = weights.get(dom).copied().unwrap_or(0.0);
        }
    }

    fn weight_fingerprint(weights: &[f32]) -> u64 {
        // Quantise weights to 1e-4 grid so tiny float drift doesn't split
        // cache entries. Fast to compute, stable across runs.
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &w in weights {
            let q = (w * 10_000.0).round() as i32;
            q.hash(&mut hasher);
        }
        hasher.finish()
    }
}

impl LmScorer for KenLMMixture {
    fn score_ids(&self, ids: &[u32]) -> f32 {
        if ids.is_empty() {
            return 0.0;
        }
        let weights = self.weights.lock().expect("weights mutex poisoned").clone();
        let fp = Self::weight_fingerprint(&weights);

        {
            let cache = self.cache.lock().expect("mixture cache poisoned");
            if let Some(&v) = cache.get(&(ids.to_vec(), fp)) {
                return v;
            }
        }

        let text = self.tokenizer.decode(ids);
        let mut total = 0.0f32;
        for (i, w) in weights.iter().enumerate() {
            if *w == 0.0 {
                continue;
            }
            total += *w * self.models[i].score(&text);
        }

        let mut cache = self.cache.lock().expect("mixture cache poisoned");
        cache.insert((ids.to_vec(), fp), total);
        total
    }

    fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }
}

// ---------------- CategoryEstimator ----------------

/// Per-reading domain weights. Uses the same profiles, char-class signals,
/// and substring tech hints as the archived evaluation path so runtime and
/// offline comparison stay comparable.
pub struct CategoryEstimator {
    available: Vec<String>,
}

impl CategoryEstimator {
    pub fn new(available: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            available: available.into_iter().map(|s| s.into()).collect(),
        }
    }

    /// Convenience — assume {general, tech, entity} are all loaded.
    pub fn default_all() -> Self {
        Self::new(["general", "tech", "entity"])
    }

    pub fn estimate(&self, reading: &str, context: &str) -> HashMap<String, f32> {
        let (ascii_r, digit_r, kata_r, _hira_r) = char_class_ratios(reading);
        let has_tech_hint = KATA_TECH_HINTS.iter().any(|h| reading.contains(h))
            || HIRA_TECH_HINTS.iter().any(|h| reading.contains(h))
            || HIRA_LOAN_MARKERS.iter().any(|m| reading.contains(m));
        let tech_signal = ascii_r >= 0.05 || kata_r >= 0.30 || has_tech_hint;
        let _ = digit_r;

        let entity_signal = if context.is_empty() {
            false
        } else {
            ENTITY_MARKERS.iter().any(|m| context.contains(m))
        };

        let profile: &[(&str, f32)] = if tech_signal && entity_signal {
            &[("general", 0.2), ("tech", 0.5), ("entity", 0.3)]
        } else if tech_signal {
            &[("general", 0.3), ("tech", 0.7), ("entity", 0.0)]
        } else if entity_signal {
            &[("general", 0.4), ("tech", 0.0), ("entity", 0.6)]
        } else {
            &[("general", 1.0), ("tech", 0.0), ("entity", 0.0)]
        };

        let mut weights: HashMap<String, f32> = profile
            .iter()
            .map(|(d, w)| ((*d).to_string(), *w))
            .collect();

        // Redirect mass from unavailable domains to `general`.
        let mut residual = 0.0;
        let mut to_clear: Vec<String> = Vec::new();
        for (d, w) in weights.iter() {
            if !self.available.iter().any(|a| a == d) {
                residual += *w;
                to_clear.push(d.clone());
            }
        }
        for d in to_clear {
            weights.insert(d, 0.0);
        }
        if residual > 0.0 && self.available.iter().any(|a| a == "general") {
            *weights.entry("general".to_string()).or_insert(0.0) += residual;
        }
        weights
    }
}

fn char_class_ratios(s: &str) -> (f32, f32, f32, f32) {
    let n = s.chars().count();
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut ascii = 0usize;
    let mut digit = 0usize;
    let mut kata = 0usize;
    let mut hira = 0usize;
    for c in s.chars() {
        let code = c as u32;
        if c.is_ascii_alphabetic() {
            ascii += 1;
        } else if c.is_ascii_digit() {
            digit += 1;
        } else if (0x30A1..=0x30FA).contains(&code) || code == 0x30FC {
            kata += 1;
        } else if (0x3041..=0x3096).contains(&code) {
            hira += 1;
        }
    }
    let n = n as f32;
    (
        ascii as f32 / n,
        digit as f32 / n,
        kata as f32 / n,
        hira as f32 / n,
    )
}

static KATA_TECH_HINTS: &[&str] = &[
    "プログラム",
    "システム",
    "データ",
    "ソフト",
    "ハード",
    "デジタル",
    "コンピュータ",
    "ネットワーク",
    "アルゴリズム",
    "クラウド",
    "サーバ",
    "スマート",
    "アップデート",
    "セキュリティ",
    "インターネット",
];

static HIRA_LOAN_MARKERS: &[&str] = &[
    "ー", "てぃ", "でぃ", "ふぁ", "ふぃ", "ふぇ", "ふぉ", "うぃ", "うぇ", "うぉ", "しぇ", "じぇ",
    "ちぇ", "つぁ", "つぇ", "つぉ",
];

static HIRA_TECH_HINTS: &[&str] = &[
    "ぷろぐら",
    "しすて",
    "でーた",
    "そふと",
    "はーど",
    "でじた",
    "こんぴゅ",
    "ねっと",
    "あるごり",
    "くらうど",
    "さーば",
    "すまーと",
    "あっぷ",
    "せきゅ",
    "いんたー",
    "ぐらふ",
    "ろぼっ",
    "あぷり",
    "ふぁい",
    "ふぉる",
    "でーたべ",
    "あぷりけーしょん",
    "いんたふぇーす",
];

static ENTITY_MARKERS: &[&str] = &[
    "氏", "市", "区", "町", "村", "県", "府", "都", "駅", "線", "社", "部", "院", "党", "神", "寺",
    "藩", "郡", "省", "庁", "山", "川", "島",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_generic_default() {
        let est = CategoryEstimator::default_all();
        let w = est.estimate("こんにちは", "");
        assert!((w["general"] - 1.0).abs() < 1e-6);
        assert!((w["tech"]).abs() < 1e-6);
    }

    #[test]
    fn estimator_tech_from_katakana() {
        let est = CategoryEstimator::default_all();
        let w = est.estimate("プログラムを書く", "");
        assert!(w["tech"] > 0.0);
    }

    #[test]
    fn estimator_entity_from_context() {
        let est = CategoryEstimator::default_all();
        let w = est.estimate("やまだ", "田中氏は");
        assert!(w["entity"] > 0.0);
    }
}
