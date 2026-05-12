//! Benchmark runner — mirrors `ds4_bench.c::main`.

use crate::args::{next_frontier, BenchArgs};
use anyhow::{Context, Result};
use ds4_core::{Backend, Engine, EngineOptions, ThinkMode, Tokens};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

pub fn run(args: BenchArgs) -> Result<()> {
    args.validate()?;
    let ctx_alloc = args.effective_ctx_alloc();

    let backend = args.backend.as_deref()
        .map(|s| Backend::parse(s).ok_or_else(|| anyhow::anyhow!("invalid backend `{s}`")))
        .transpose()?;
    let opt = EngineOptions {
        model_path: args.model.clone(),
        mtp_path: None,
        backend,
        n_threads: args.threads,
        mtp_draft_tokens: 0,
        mtp_margin: 0.0,
        directional_steering_file: None,
        directional_steering_attn: 0.0,
        directional_steering_ffn: 0.0,
        warm_weights: args.warm_weights,
        quality: args.quality,
    };
    let engine = Engine::open(&opt)?;
    log_context_memory(&engine, ctx_alloc);

    // Read raw or chat-rendered prompt source.
    let prompt = build_prompt(&engine, &args)?;
    if (prompt.len() as i32) < args.ctx_max {
        anyhow::bail!(
            "prompt has {} tokens, need at least --ctx-max={}",
            prompt.len(), args.ctx_max,
        );
    }

    let mut session = engine.new_session(ctx_alloc as u32)
        .context("create session")?;

    let mut out: Box<dyn Write> = match &args.csv {
        Some(p) => Box::new(File::create(p).with_context(|| format!("create {}", p.display()))?),
        None => Box::new(std::io::stdout().lock()),
    };
    writeln!(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes")?;
    out.flush()?;

    let eos = engine.tokenizer().eos_id();
    let mut previous = 0i32;
    let mut frontier = args.ctx_start;
    loop {
        let prefix = Tokens::from_vec(prompt.as_slice()[..frontier as usize].to_vec());

        let t0 = Instant::now();
        session.sync(&prefix)?;
        let prefill_sec = t0.elapsed().as_secs_f64();
        let prefill_tokens = frontier - previous;

        // Snapshot before decode so we can restore.
        let snap = session.payload_bytes()?;

        let t0 = Instant::now();
        for _ in 0..args.gen_tokens {
            if (session.pos() as i32) + 1 >= session.ctx() as i32 {
                anyhow::bail!("generation would exceed allocated context at frontier {frontier}");
            }
            let token = session.argmax_excluding(eos);
            if token < 0 {
                anyhow::bail!("no non-EOS token at frontier {frontier}");
            }
            session.eval(token)?;
        }
        let gen_sec = t0.elapsed().as_secs_f64();

        session.load_payload(&snap)?;

        writeln!(out,
            "{},{},{:.2},{},{:.2},{}",
            frontier,
            prefill_tokens,
            if prefill_sec > 0.0 { prefill_tokens as f64 / prefill_sec } else { 0.0 },
            args.gen_tokens,
            if gen_sec > 0.0 { args.gen_tokens as f64 / gen_sec } else { 0.0 },
            snap.len(),
        )?;
        out.flush()?;

        previous = frontier;
        if frontier >= args.ctx_max { break; }
        frontier = next_frontier(&args, frontier);
    }
    Ok(())
}

fn build_prompt(engine: &Engine, args: &BenchArgs) -> Result<Tokens> {
    let mut tokens = Tokens::new();
    let path = args.prompt_file.as_ref().or(args.chat_prompt_file.as_ref()).unwrap();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    if args.chat_prompt_file.is_some() {
        let mut rendered = String::new();
        rendered.push_str("<|im_start|>system\n");
        rendered.push_str(&args.system);
        rendered.push_str("<|im_end|>\n<|im_start|>user\n");
        rendered.push_str(&text);
        rendered.push_str("<|im_end|>\n<|im_start|>assistant\n");
        let _ = ThinkMode::None;
        engine.tokenizer().encode_rendered_chat(&rendered, &mut tokens);
    } else {
        engine.tokenizer().encode(&text, &mut tokens);
    }
    Ok(tokens)
}

fn log_context_memory(engine: &Engine, ctx_size: i32) {
    let m = ds4_core::context_memory::estimate(engine.backend(), ctx_size);
    eprintln!(
        "ds4-bench: context buffers {:.2} MiB (ctx={ctx_size}, backend={}, prefill_chunk={}, raw_kv_rows={}, compressed_kv_rows={})",
        m.total_bytes as f64 / (1024.0 * 1024.0),
        engine.backend().name(),
        m.prefill_cap, m.raw_cap, m.comp_cap,
    );
}
