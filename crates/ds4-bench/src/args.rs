use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(name = "ds4-bench", about = "DwarfStar 4 microbenchmark")]
pub struct BenchArgs {
    #[arg(long, default_value = "./ds4flash.gguf")]
    pub model: PathBuf,
    #[arg(long)]
    pub backend: Option<String>,
    /// Number of warmup tokens before timing starts.
    #[arg(long, default_value_t = 32)]
    pub warmup: i32,
    /// Number of tokens to time during decode.
    #[arg(long, default_value_t = 256)]
    pub decode: i32,
    /// Number of tokens to time during prefill.
    #[arg(long, default_value_t = 1024)]
    pub prefill: i32,
    /// Context size.
    #[arg(long, default_value_t = 4096)]
    pub ctx_size: i32,
}
