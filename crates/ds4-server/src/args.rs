use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "ds4-server",
    about = "DwarfStar 4 HTTP server (OpenAI/Anthropic-compatible API)",
)]
pub struct ServerArgs {
    #[arg(long, default_value = "./ds4flash.gguf")]
    pub model: PathBuf,
    #[arg(long)]
    pub mtp: Option<PathBuf>,
    #[arg(long)]
    pub backend: Option<String>,
    /// HTTP listen address.
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub bind: String,
    /// Context size per session.
    #[arg(long, default_value_t = 16384)]
    pub ctx_size: i32,
    /// On-disk KV cache directory.
    #[arg(long)]
    pub kv_cache_dir: Option<PathBuf>,
    /// Maximum on-disk KV cache size in bytes (0 = unlimited).
    #[arg(long, default_value_t = 0)]
    pub kv_cache_max_bytes: u64,
    /// Concurrent request slots. Each slot owns one engine session.
    #[arg(long, default_value_t = 1)]
    pub slots: usize,
    /// Bearer-token authentication.
    #[arg(long, env = "DS4_API_KEY")]
    pub api_key: Option<String>,
    /// Enable CORS for browser clients.
    #[arg(long)]
    pub cors: bool,
    #[arg(long)]
    pub trace: Option<PathBuf>,
}
