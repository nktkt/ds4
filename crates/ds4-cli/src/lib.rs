//! ds4-cli library — separates `main()` from the CLI logic so tests can drive
//! the full argument parser and the REPL without spawning the binary.

pub mod args;
pub mod repl;
pub mod transcript;

pub use args::Cli;

pub fn run() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = <Cli as clap::Parser>::parse();
    cli.execute()
}
