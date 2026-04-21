//! new-ime engine core (Rust port).
//!
//! Replaces the C++ `new-ime-engine.dll` as a regular Rust library used
//! directly by the TSF crate. Responsibilities:
//!   * SharedCharTokenizer (vocab.hex.tsv sidecar) encode/decode
//!   * ONNX CTC-NAT inference via `ort`
//!   * CTC prefix beam search
//!   * KenLM shallow fusion (FFI to existing kenlm.lib)
//!
//! The crate is platform-independent; ORT/KenLM linkage happens at runtime
//! via dynamic loading so tests on Linux don't require Windows artifacts.

pub mod beam;
pub mod kenlm;
pub mod session;
pub mod tokenizer;

pub use beam::{prefix_beam_search, BeamHypothesis};
pub use kenlm::{KenLM, KenLMCharScorer};
pub use session::EngineSession;
pub use tokenizer::SharedCharTokenizer;
