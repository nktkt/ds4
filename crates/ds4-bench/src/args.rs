//! `ds4-bench` argument parser. Mirrors the long-form options of
//! `ds4_bench.c::parse_options`.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "ds4-bench",
    about = "DwarfStar 4 throughput benchmark — sweep context frontiers and \
             measure prefill + greedy decode tokens/sec at each.",
)]
pub struct BenchArgs {
    /// GGUF model path.
    #[arg(long, short = 'm', default_value = "ds4flash.gguf")]
    pub model: PathBuf,

    /// Raw benchmark text. Mutually exclusive with --chat-prompt-file.
    #[arg(long)]
    pub prompt_file: Option<PathBuf>,
    /// Render the file as one no-thinking chat user message, then slice.
    #[arg(long)]
    pub chat_prompt_file: Option<PathBuf>,
    /// System prompt for `--chat-prompt-file`.
    #[arg(long = "system", short = 's', default_value = "You are a helpful assistant.")]
    pub system: String,

    /// Backend: metal/cuda/cpu (defaults to platform native).
    #[arg(long)]
    pub backend: Option<String>,
    /// CPU helper threads.
    #[arg(long, short = 't', default_value_t = 0)]
    pub threads: i32,
    /// Prefer exact kernels where applicable.
    #[arg(long)]
    pub quality: bool,
    /// Touch mmap pages before benchmarking.
    #[arg(long)]
    pub warm_weights: bool,

    /// First measured frontier.
    #[arg(long, default_value_t = 2048)]
    pub ctx_start: i32,
    /// Last measured frontier.
    #[arg(long, default_value_t = 32768)]
    pub ctx_max: i32,
    /// Allocated context (default ctx_max + gen_tokens + 1).
    #[arg(long, default_value_t = 0)]
    pub ctx_alloc: i32,
    /// Multiplicative step. 1.0 ⇒ linear step.
    #[arg(long, default_value_t = 1.0)]
    pub step_mul: f64,
    /// Linear step when --step-mul is 1.
    #[arg(long, default_value_t = 2048)]
    pub step_incr: i32,
    /// Greedy decode tokens per frontier.
    #[arg(long = "gen-tokens", alias = "tokens", short = 'n', default_value_t = 128)]
    pub gen_tokens: i32,

    /// Write CSV to FILE instead of stdout.
    #[arg(long)]
    pub csv: Option<PathBuf>,
}

impl BenchArgs {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.prompt_file.is_some() == self.chat_prompt_file.is_some() {
            anyhow::bail!("specify exactly one of --prompt-file or --chat-prompt-file");
        }
        if self.ctx_start > self.ctx_max {
            anyhow::bail!("--ctx-start must be <= --ctx-max");
        }
        if self.step_mul < 1.0 {
            anyhow::bail!("--step-mul must be >= 1");
        }
        if self.step_mul == 1.0 && self.step_incr <= 0 {
            anyhow::bail!("--step-incr must be positive when --step-mul is 1");
        }
        Ok(())
    }

    /// `ctx-alloc` effective value (filling in the default if zero).
    pub fn effective_ctx_alloc(&self) -> i32 {
        if self.ctx_alloc == 0 {
            self.ctx_max + self.gen_tokens + 1
        } else {
            self.ctx_alloc
        }
    }
}

/// Compute the next frontier in the sweep. Mirrors `next_frontier`.
pub fn next_frontier(cfg: &BenchArgs, cur: i32) -> i32 {
    if cur >= cfg.ctx_max { return cfg.ctx_max; }
    let next = if cfg.step_mul == 1.0 {
        cur.saturating_add(cfg.step_incr)
    } else {
        let v = (cur as f64 * cfg.step_mul).ceil();
        if v > i32::MAX as f64 { cfg.ctx_max } else { v as i32 }
    };
    let next = if next <= cur { cur + 1 } else { next };
    next.min(cfg.ctx_max)
}
