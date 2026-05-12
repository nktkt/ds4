//! Microbenchmark loop. Mirrors `ds4_bench.c::main` — prefill phase, then a
//! steady-state decode phase, then prints tokens/sec for both. Times come
//! from `std::time::Instant`; no monotonic-clock acrobatics needed because
//! `Instant` is monotonic on all our targets.

use crate::args::BenchArgs;
use anyhow::Result;
use ds4_core::{Backend, Engine, EngineOptions, Tokens};
use std::time::Instant;

pub fn run(args: BenchArgs) -> Result<()> {
    let backend = args.backend.as_deref()
        .map(|s| Backend::parse(s).ok_or_else(|| anyhow::anyhow!("unknown backend `{s}`")))
        .transpose()?;
    let opt = EngineOptions {
        model_path: args.model.clone(),
        backend,
        n_threads: 0,
        mtp_draft_tokens: 0,
        mtp_margin: 0.0,
        directional_steering_file: None,
        directional_steering_attn: 0.0,
        directional_steering_ffn: 0.0,
        warm_weights: true,
        quality: false,
        mtp_path: None,
    };
    let engine = Engine::open(&opt)?;
    engine.summary();
    // Build a synthetic prompt of `args.prefill` random valid tokens (the
    // exact contents don't matter for throughput numbers; we re-use token
    // id 1 which is reserved as <|im_start|> in DS4's vocab).
    let mut prefill = Tokens::new();
    for _ in 0..args.prefill { prefill.push(1); }
    let mut session = engine.new_session(args.ctx_size as u32)?;

    let t0 = Instant::now();
    session.sync(&prefill)?;
    let prefill_elapsed = t0.elapsed();
    eprintln!(
        "prefill: {} tokens in {:.3}s — {:.1} tok/s",
        prefill.len(),
        prefill_elapsed.as_secs_f64(),
        prefill.len() as f64 / prefill_elapsed.as_secs_f64().max(1e-9),
    );

    for _ in 0..args.warmup {
        let t = session.argmax();
        session.eval(t)?;
    }
    let t0 = Instant::now();
    for _ in 0..args.decode {
        let t = session.argmax();
        session.eval(t)?;
    }
    let decode_elapsed = t0.elapsed();
    eprintln!(
        "decode: {} tokens in {:.3}s — {:.1} tok/s",
        args.decode,
        decode_elapsed.as_secs_f64(),
        args.decode as f64 / decode_elapsed.as_secs_f64().max(1e-9),
    );
    Ok(())
}
