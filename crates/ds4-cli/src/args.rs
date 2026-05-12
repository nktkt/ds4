//! CLI arguments. Mirrors the `argv` parsing block at the top of `ds4_cli.c`.
//!
//! Same long-flag names so existing scripts keep working. Some short flags
//! (e.g. `-t` for `--n-threads`) are kept too.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "ds4",
    about = "DwarfStar 4 — DeepSeek V4 Flash inference engine",
    version,
)]
pub struct Cli {
    /// Path to the GGUF file. Defaults to `./ds4flash.gguf`.
    #[arg(long, default_value = "./ds4flash.gguf")]
    pub model: PathBuf,

    /// Optional MTP (speculative decoding) GGUF.
    #[arg(long)]
    pub mtp: Option<PathBuf>,

    /// Backend choice (`metal`, `cuda`, `cpu`). Default is platform-native.
    #[arg(long)]
    pub backend: Option<String>,

    /// CPU thread count for the CPU backend.
    #[arg(long, short = 't', default_value_t = 0)]
    pub n_threads: i32,

    /// Context size in tokens for the live session.
    #[arg(long, default_value_t = 16384)]
    pub ctx_size: i32,

    /// Tokens to predict per turn.
    #[arg(long, short = 'n', default_value_t = 1024)]
    pub n_predict: i32,

    /// Think mode: `none`, `high`, `max`.
    #[arg(long, default_value = "high")]
    pub think: String,

    /// Run a smoke test instead of the REPL.
    #[arg(long)]
    pub smoke: bool,

    /// Tokenize stdin and dump tokens, then exit.
    #[arg(long)]
    pub dump_tokens: bool,

    /// One-shot prompt; bypasses the REPL.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Trace generation events to a file.
    #[arg(long)]
    pub trace: Option<PathBuf>,

    /// Warm-touch the GGUF mmap pages on startup.
    #[arg(long)]
    pub warm: bool,

    /// Sampling temperature. 0 = argmax.
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,
    #[arg(long, default_value_t = 0)]
    pub top_k: i32,
    #[arg(long, default_value_t = 1.0)]
    pub top_p: f32,
    #[arg(long, default_value_t = 0.0)]
    pub min_p: f32,
}

impl Cli {
    pub fn execute(self) -> anyhow::Result<()> {
        let backend = self.backend.as_deref()
            .map(|s| ds4_core::Backend::parse(s)
                .ok_or_else(|| anyhow::anyhow!("unknown backend `{s}`")))
            .transpose()?;
        let think = ds4_core::think::parse_mode(&self.think)
            .ok_or_else(|| anyhow::anyhow!("invalid --think value"))?;

        let opt = ds4_core::EngineOptions {
            model_path: self.model.clone(),
            mtp_path: self.mtp.clone(),
            backend,
            n_threads: self.n_threads,
            mtp_draft_tokens: 0,
            mtp_margin: 0.0,
            directional_steering_file: None,
            directional_steering_attn: 0.0,
            directional_steering_ffn: 0.0,
            warm_weights: self.warm,
            quality: false,
        };
        let engine = ds4_core::Engine::open(&opt)?;
        engine.summary();
        if self.dump_tokens {
            let mut input = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)?;
            let mut toks = ds4_core::Tokens::new();
            engine.tokenizer().encode(&input, &mut toks);
            engine.dump_tokens(&toks);
            return Ok(());
        }
        if let Some(prompt) = &self.prompt {
            return crate::repl::run_oneshot(&engine, prompt, &self, think);
        }
        crate::repl::run(&engine, &self, think)
    }
}
