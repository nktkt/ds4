//! Engine — Rust mirror of `ds4_engine_*` from `ds4.c`.
//!
//! An engine wraps a loaded model and the backend handle that's been bound to
//! it. The engine is shared across sessions (immutable model weights live in
//! the mmap, so we just keep references). Sessions allocate their own KV
//! state on top.

use crate::api::{
    Backend, Tokens, EngineOptions, TokenSink, ProgressSink,
};
use crate::backend;
use crate::gguf::Gguf;
use crate::model::{Config, Tensors};
use crate::session::Session;
use crate::tokenizer::Tokenizer;
use anyhow::{anyhow, Result};
use std::sync::Arc;

pub struct Engine {
    #[allow(dead_code)]
    pub(crate) gguf: Arc<Gguf>,
    pub(crate) config: Config,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) backend: Backend,
    #[allow(dead_code)]
    pub(crate) mtp: Option<Arc<Gguf>>,
    pub(crate) mtp_draft_tokens: i32,
    pub(crate) routed_quant_bits: i32,
}

impl Engine {
    /// Borrow the underlying GGUF mmap. Used by debug tooling.
    pub fn gguf(&self) -> &Gguf { &self.gguf }
}

impl Engine {
    pub fn open(opt: &EngineOptions) -> Result<Self> {
        let gguf = Gguf::open(&opt.model_path)?;
        let config = Config::from_gguf(&gguf)?;
        // Resolve tensor table once so a missing tensor surfaces here.
        let _tensors = Tensors::resolve(&gguf, &config)?;
        let tokenizer = build_tokenizer(&gguf)?;
        let backend = opt.backend.unwrap_or(default_backend());
        let mtp = match &opt.mtp_path {
            Some(p) => Some(Arc::new(Gguf::open(p)?)),
            None => None,
        };
        let routed_quant_bits = guess_routed_quant_bits(&gguf, &_tensors);
        Ok(Engine {
            gguf: Arc::new(gguf),
            config,
            tokenizer,
            backend,
            mtp,
            mtp_draft_tokens: opt.mtp_draft_tokens,
            routed_quant_bits,
        })
    }

    pub fn backend(&self) -> Backend { self.backend }
    pub fn config(&self) -> &Config { &self.config }
    pub fn tokenizer(&self) -> &Tokenizer { &self.tokenizer }
    pub fn has_mtp(&self) -> bool { self.mtp.is_some() }
    pub fn mtp_draft_tokens(&self) -> i32 { self.mtp_draft_tokens }
    pub fn routed_quant_bits(&self) -> i32 { self.routed_quant_bits }

    pub fn summary(&self) {
        // Matches `ds4_engine_summary` line format so external tooling that
        // parses the CLI output keeps working.
        let cfg = &self.config;
        println!(
            "ds4: arch=ds4 backend={} layers={} d_model={} n_heads={} n_kv={} d_head={} d_ff={} experts={}/{} vocab={} ctx_max={}",
            self.backend.name(),
            cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_head,
            cfg.d_ff, cfg.n_experts_per_tok, cfg.n_experts,
            cfg.vocab_size, cfg.max_context,
        );
    }

    pub fn new_session(&self, ctx_size: u32) -> Result<Session> {
        let backend = backend::open(self.backend, &self.config, ctx_size)?;
        Ok(Session::new(backend, ctx_size))
    }

    /// Greedy generation helper. Mirrors `ds4_engine_generate_argmax`. Drives
    /// a temporary session, emits each predicted token to `emit`, and reports
    /// prefill / generation progress to `progress`.
    pub fn generate_argmax(
        &self,
        prompt: &Tokens,
        n_predict: i32,
        ctx_size: i32,
        emit: &mut dyn TokenSink,
        progress: Option<&mut dyn ProgressSink>,
    ) -> Result<i32> {
        let _ = progress;
        let mut session = self.new_session(ctx_size.max(0) as u32)?;
        session.sync(prompt)?;
        let mut emitted = 0;
        let eos = self.tokenizer.eos_id();
        for _ in 0..n_predict {
            let t = session.argmax();
            emit.emit(t);
            emitted += 1;
            if t == eos { break; }
            session.eval(t)?;
        }
        emit.done();
        Ok(emitted)
    }

    pub fn dump_tokens(&self, tokens: &Tokens) {
        for (i, &t) in tokens.as_slice().iter().enumerate() {
            let bytes = self.tokenizer.token_text(t).unwrap_or(b"<?>");
            println!("[{i:5}] {t:>6}  {}", String::from_utf8_lossy(bytes));
        }
    }
}

fn build_tokenizer(g: &Gguf) -> Result<Tokenizer> {
    use crate::gguf::Value;
    let vocab: Vec<Vec<u8>> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Value::Array(arr)) => arr.iter()
            .map(|v| match v {
                Value::String(s) => s.as_bytes().to_vec(),
                _ => Vec::new(),
            }).collect(),
        _ => return Err(anyhow!("tokenizer: no tokens array in gguf")),
    };
    let mut by_bytes = ahash::AHashMap::with_capacity(vocab.len());
    for (i, b) in vocab.iter().enumerate() {
        by_bytes.insert(b.clone(), i as i32);
    }
    // Merges: "<a> <b>" → rank index. Mirrors `vocab_load` (ds4.c).
    let mut merges: ahash::AHashMap<(Vec<u8>, Vec<u8>), i32> = ahash::AHashMap::new();
    if let Some(Value::Array(arr)) = g.metadata.get("tokenizer.ggml.merges") {
        for (rank, v) in arr.iter().enumerate() {
            if let Value::String(s) = v {
                if let Some(sp) = s.find(' ') {
                    let a = s[..sp].as_bytes().to_vec();
                    let b = s[sp + 1..].as_bytes().to_vec();
                    merges.insert((a, b), rank as i32);
                }
            }
        }
    }
    // Resolve DS4 special tokens by name. Names taken verbatim from
    // `vocab_load` in ds4.c. Compute everything *before* moving by_bytes
    // into the struct so the borrow checker stays happy.
    let lookup_str = |s: &str| -> i32 {
        by_bytes.get(s.as_bytes()).copied().unwrap_or(-1)
    };
    let lookup_meta = |k: &str| -> i32 {
        g.meta_u32(k).map(|v| v as i32).unwrap_or(-1)
    };
    let names = [
        "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>",
        "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
        "<\u{ff5c}User\u{ff5c}>",
        "<\u{ff5c}Assistant\u{ff5c}>",
        "<think>",
        "</think>",
        "\u{ff5c}DSML\u{ff5c}",
        "<|im_start|>",
        "<|im_end|>",
    ];
    let mut special: indexmap::IndexMap<String, i32> = indexmap::IndexMap::new();
    for name in names {
        let id = lookup_str(name);
        if id >= 0 { special.insert(name.to_string(), id); }
    }
    let bos_named = lookup_str(names[0]);
    let eos_named = lookup_str(names[1]);
    let user_role = lookup_str(names[2]);
    let assistant_role = lookup_str(names[3]);
    let think_start = lookup_str(names[4]);
    let think_end   = lookup_str(names[5]);
    let im_start    = lookup_str(names[7]);
    let im_end      = lookup_str(names[8]);
    let bos_meta = lookup_meta("tokenizer.ggml.bos_token_id");
    let eos_meta = lookup_meta("tokenizer.ggml.eos_token_id");

    Ok(Tokenizer {
        vocab,
        vocab_by_bytes: by_bytes,
        special,
        merges,
        bos: if bos_named >= 0 { bos_named } else { bos_meta },
        eos: if eos_named >= 0 { eos_named } else { eos_meta },
        im_start,
        im_end,
        user_role,
        assistant_role,
        system_role: -1,
        think_start,
        think_end,
    })
}

fn default_backend() -> Backend {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    { return Backend::Metal; }
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    { return Backend::Cuda; }
    Backend::Cpu
}

fn guess_routed_quant_bits(g: &Gguf, t: &Tensors<'_>) -> i32 {
    use crate::gguf::DType;
    let _ = g;
    // Routed experts dominate the model, so we look at the first layer's
    // expert tensors to characterize the mixed-precision build.
    let first = t.layers.first();
    let dt = first.and_then(|l| l.experts_down.map(|t| t.dtype));
    match dt {
        Some(DType::Q2_K) | Some(DType::IQ2_XXS) | Some(DType::IQ2_XS) | Some(DType::IQ2_S) => 2,
        Some(DType::Q3_K) | Some(DType::IQ3_XXS) | Some(DType::IQ3_S) => 3,
        Some(DType::Q4_K) | Some(DType::Q4_0) | Some(DType::Q4_1) | Some(DType::IQ4_NL) | Some(DType::IQ4_XS) => 4,
        Some(DType::Q5_K) | Some(DType::Q5_0) | Some(DType::Q5_1) => 5,
        Some(DType::Q6_K) => 6,
        Some(DType::Q8_K) | Some(DType::Q8_0) | Some(DType::Q8_1) => 8,
        Some(DType::F16) | Some(DType::BF16) => 16,
        Some(DType::F32) => 32,
        _ => 0,
    }
}
