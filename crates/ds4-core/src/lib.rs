//! ds4-core — model loading, tokenizer, sessions, backend dispatch.
//!
//! Public surface mirrors `ds4.h`: an opaque [`Engine`], an opaque [`Session`],
//! plus the supporting enums and option structs. Lifetimes here lean on Rust's
//! borrow checker instead of the C `ds4_engine_close` / `ds4_session_free`
//! handle dance — engines and sessions clean themselves up on drop.
//!
//! The actual inference graph implementation (which lives in the ~17k-line
//! `ds4.c`) is being ported incrementally into the modules below. Each
//! submodule annotates which range of the C source it corresponds to so the
//! port can be cross-checked.

pub mod api;        // public types analogous to ds4.h
pub mod log;        // ds4_log + ds4_log_is_tty
pub mod tokens;     // ds4_tokens helpers
pub mod tokenizer;  // BPE tokenizer + chat templating
pub mod gguf;       // GGUF mmap loader
pub mod quant;      // GGUF quant block layouts (Q2_K, Q4_K, IQ2_XXS, Q8_K)
pub mod iq2_tables; // IQ2_XXS lookup tables
pub mod dot;        // inner-loop integer dot helpers
pub mod half;       // f16 <-> f32 + E4M3FN dequant
pub mod nn;         // RMSNorm, SwiGLU, softmax, SiLU
pub mod rope;       // YaRN-style rotary position embedding
pub mod matvec;     // f16 / Q8_0 / Q2_K matrix-vector multiplies
pub mod matvec_q;   // Q2_K / Q4_K / IQ2_XXS row dots + matvecs
pub mod dequant;    // dequant of Q2_K, Q4_K, IQ2_XXS, Q8_K blocks
pub mod layer;      // per-layer parameters (compression ratio)
pub mod shape;      // fixed DeepSeek V4 Flash shape constants
pub mod hc;         // hyper-connection helpers (split, post, weighted-sum)
pub mod attn;       // MLA attention CPU reference
pub mod moe;        // MoE routing (sinkhorn, top-k, expert dispatch)
pub mod ffn;        // shared-expert SwiGLU FFN + layer_ffn_one
pub mod output_head; // HC collapse + final RMSNorm + vocab projection
pub mod layer_forward; // full per-layer composition (attn + MoE + FFN + HC)
pub mod weights;    // GGUF tensor binder → per-layer typed slices
pub mod cpu_generate; // end-to-end CPU prefill + decode loop
pub mod model;      // tensor metadata, weights, model config
pub mod sampler;    // top_k / top_p / min_p / temperature sampling
pub mod kv_cache;   // session KV cache and disk payload serialization
pub mod forward;    // CPU reference forward pass scaffold
pub mod session;    // ds4_session_*
pub mod engine;     // ds4_engine_*
pub mod chat;       // ds4_chat_* / ds4_encode_chat_prompt
pub mod backend;    // backend dispatch (metal / cuda / cpu)
pub mod think;      // think-mode helpers
pub mod context_memory; // ds4_context_memory_estimate

pub use api::{
    Backend, ThinkMode, LogType, EngineOptions, ContextMemory, SessionSnapshot,
    TokenScore, SessionRewriteResult, Tokens,
};
pub use engine::Engine;
pub use session::Session;
