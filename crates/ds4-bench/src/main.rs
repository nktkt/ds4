//! ds4-bench — DwarfStar 4 microbenchmark harness (port of `ds4_bench.c`).

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    ds4_bench::run()
}
