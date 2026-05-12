//! ds4-bench library.

pub mod args;
pub mod runner;

pub fn run() -> anyhow::Result<()> {
    let args = <args::BenchArgs as clap::Parser>::parse();
    runner::run(args)
}
