//! Public engine boundary — Rust mirror of `ds4.h`.
//!
//! The CLI and server treat `Engine` as the loaded model and `Session` as one
//! mutable inference timeline. A session owns the live KV cache and logits;
//! callers provide full token prefixes and let `Session::sync()` reuse, extend,
//! or rebuild the graph state. The original C header is intentionally narrow
//! and so is this module — HTTP/CLI code must not depend on tensor internals.

use std::path::PathBuf;
use thiserror::Error;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Backend {
    Metal,
    Cuda,
    Cpu,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Metal => "metal",
            Backend::Cuda => "cuda",
            Backend::Cpu => "cpu",
        }
    }

    pub fn parse(s: &str) -> Option<Backend> {
        match s.to_ascii_lowercase().as_str() {
            "metal" => Some(Backend::Metal),
            "cuda"  => Some(Backend::Cuda),
            "cpu"   => Some(Backend::Cpu),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ThinkMode {
    None,
    High,
    Max,
}

impl ThinkMode {
    pub fn enabled(self) -> bool { !matches!(self, ThinkMode::None) }
    pub fn name(self) -> &'static str {
        match self {
            ThinkMode::None => "none",
            ThinkMode::High => "high",
            ThinkMode::Max  => "max",
        }
    }
    /// Minimum context (in tokens) required to use ThinkMode::Max. Mirrors
    /// `ds4_think_max_min_context()` — the value below 32k context the engine
    /// auto-downgrades Max → High because the chain-of-thought scratch alone
    /// would dominate the window.
    pub fn max_min_context() -> u32 { 32 * 1024 }
    pub fn max_prefix() -> &'static str { "<|im_start|>think_max\n" }

    /// Downgrade `mode` if the supplied `ctx_size` cannot support it. Mirrors
    /// `ds4_think_mode_for_context()`.
    pub fn for_context(mode: ThinkMode, ctx_size: i32) -> ThinkMode {
        if matches!(mode, ThinkMode::Max) && (ctx_size as u32) < Self::max_min_context() {
            ThinkMode::High
        } else {
            mode
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LogType {
    Default,
    Prefill,
    Generation,
    KvCache,
    Tool,
    Warning,
    Timing,
    Ok,
    Error,
}

/// Mutable token sequence used at API boundaries. Mirrors the layout of the
/// C `ds4_tokens` (pointer + length + capacity) but the storage is just a
/// `Vec<i32>` underneath.
#[derive(Clone, Debug, Default)]
pub struct Tokens {
    v: Vec<i32>,
}

impl Tokens {
    pub fn new() -> Self { Self { v: Vec::new() } }
    pub fn from_vec(v: Vec<i32>) -> Self { Self { v } }
    pub fn as_slice(&self) -> &[i32] { &self.v }
    pub fn as_mut_slice(&mut self) -> &mut [i32] { &mut self.v }
    pub fn into_vec(self) -> Vec<i32> { self.v }
    pub fn len(&self) -> usize { self.v.len() }
    pub fn is_empty(&self) -> bool { self.v.is_empty() }
    pub fn push(&mut self, t: i32) { self.v.push(t); }
    pub fn clear(&mut self) { self.v.clear(); }
    pub fn reserve(&mut self, n: usize) { self.v.reserve(n); }
    pub fn extend(&mut self, slice: &[i32]) { self.v.extend_from_slice(slice); }
    pub fn copy_from(&mut self, src: &Tokens) {
        self.v.clear();
        self.v.extend_from_slice(&src.v);
    }
    pub fn starts_with(&self, prefix: &Tokens) -> bool {
        prefix.v.len() <= self.v.len() && self.v[..prefix.v.len()] == prefix.v[..]
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TokenScore {
    pub id: i32,
    pub logit: f32,
    pub logprob: f32,
}

#[derive(Clone, Debug, Default)]
pub struct EngineOptions {
    pub model_path: PathBuf,
    pub mtp_path: Option<PathBuf>,
    pub backend: Option<Backend>,
    pub n_threads: i32,
    pub mtp_draft_tokens: i32,
    pub mtp_margin: f32,
    pub directional_steering_file: Option<PathBuf>,
    pub directional_steering_attn: f32,
    pub directional_steering_ffn: f32,
    pub warm_weights: bool,
    pub quality: bool,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ContextMemory {
    pub total_bytes: u64,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    pub scratch_bytes: u64,
    pub prefill_cap: u32,
    pub raw_cap: u32,
    pub comp_cap: u32,
}

#[derive(Debug, Default)]
pub struct SessionSnapshot {
    pub buf: Vec<u8>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SessionRewriteResult {
    /// `DS4_SESSION_REWRITE_OK`
    Ok,
    /// `DS4_SESSION_REWRITE_REBUILD_NEEDED` — the live backend state cannot
    /// be rewritten safely in place. The caller should restore an older
    /// checkpoint, then sync to the prompt.
    RebuildNeeded,
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("gguf: {0}")]
    Gguf(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("backend ({0}): {1}")]
    Backend(&'static str, String),
    #[error("invalid arg: {0}")]
    Invalid(String),
    #[error("rebuild required")]
    RebuildRequired,
    #[error("not supported on this build: {0}")]
    Unsupported(&'static str),
    #[error("{0}")]
    Other(String),
}

impl From<anyhow::Error> for Error {
    fn from(e: anyhow::Error) -> Self { Error::Other(e.to_string()) }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Token emit callback. Replaces the C `ds4_token_emit_fn` /
/// `ds4_generation_done_fn` pair with a single closure that returns
/// `ControlFlow::Break` to stop early.
pub trait TokenSink: Send {
    fn emit(&mut self, token: i32);
    fn done(&mut self) {}
}

impl<F: FnMut(i32) + Send> TokenSink for F {
    fn emit(&mut self, token: i32) { (self)(token) }
}

/// Progress callback used by long-running session operations (prefill, KV
/// cache load, etc). Mirrors `ds4_session_progress_fn`.
pub trait ProgressSink: Send {
    fn report(&mut self, event: &str, current: i32, total: i32);
}

impl<F: FnMut(&str, i32, i32) + Send> ProgressSink for F {
    fn report(&mut self, event: &str, current: i32, total: i32) { (self)(event, current, total) }
}
