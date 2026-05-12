//! ds4-server library: HTTP layer, worker queue, streaming, tool-call mapping,
//! disk KV cache policy. The runtime is `tokio` with `hyper` because the
//! original blocking-thread + epoll/kqueue loop maps cleanly onto async.

pub mod args;
pub mod http;
pub mod queue;
pub mod stream;
pub mod stop;
pub mod tools;
pub mod openai;
pub mod anthropic;
pub mod disk_cache;

use args::ServerArgs;

pub fn run() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = <ServerArgs as clap::Parser>::parse();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(http::serve(args))
}
