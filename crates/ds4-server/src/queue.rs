//! Worker queue. Each slot owns its own `Engine + Session`, and pulls Jobs
//! off a shared channel. Mirrors the producer/consumer split in
//! `ds4_server.c::worker_thread_*`.

use crate::args::ServerArgs;
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use ds4_core::{Backend, Engine, EngineOptions};
use parking_lot::Mutex;
use std::sync::Arc;
use std::thread;

pub struct Job {
    pub prompt: ds4_core::Tokens,
    pub n_predict: i32,
    pub eos_only: bool,
    pub sampler: SamplerCfg,
    pub on_token: Box<dyn Fn(i32) + Send + Sync>,
    pub on_done: Box<dyn FnOnce(Result<()>) + Send>,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplerCfg {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub rng_seed: u64,
}

#[derive(Clone)]
pub struct JobQueue {
    tx: Sender<Job>,
    pub engine: Arc<Mutex<Option<Arc<Engine>>>>, // present after first init
}

impl JobQueue {
    pub fn start(args: &ServerArgs) -> Result<Self> {
        let (tx, rx): (Sender<Job>, Receiver<Job>) = crossbeam_channel::unbounded();
        let backend = args.backend.as_deref()
            .map(|s| Backend::parse(s).ok_or_else(|| anyhow::anyhow!("unknown backend `{s}`")))
            .transpose()?;
        let opt = EngineOptions {
            model_path: args.model.clone(),
            mtp_path: args.mtp.clone(),
            backend,
            n_threads: 0,
            mtp_draft_tokens: 0,
            mtp_margin: 0.0,
            directional_steering_file: None,
            directional_steering_attn: 0.0,
            directional_steering_ffn: 0.0,
            warm_weights: false,
            quality: false,
        };
        let engine = Arc::new(Engine::open(&opt)?);
        let engine_slot = Arc::new(Mutex::new(Some(engine.clone())));
        for slot in 0..args.slots {
            let rx = rx.clone();
            let engine = engine.clone();
            let ctx = args.ctx_size.max(0) as u32;
            thread::Builder::new()
                .name(format!("ds4-worker-{slot}"))
                .spawn(move || worker(slot, engine, ctx, rx))?;
        }
        Ok(JobQueue { tx, engine: engine_slot })
    }

    pub fn submit(&self, job: Job) -> Result<()> {
        self.tx.send(job).map_err(|e| anyhow::anyhow!("submit: {e}"))
    }
}

fn worker(slot: usize, engine: Arc<Engine>, ctx: u32, rx: Receiver<Job>) {
    log::info!("ds4-worker-{slot}: starting");
    let mut session = match engine.new_session(ctx) {
        Ok(s) => s,
        Err(e) => { log::error!("ds4-worker-{slot}: session: {e}"); return; }
    };
    while let Ok(job) = rx.recv() {
        let result = run_job(&engine, &mut session, &job);
        (job.on_done)(result);
    }
}

fn run_job(engine: &Engine, session: &mut ds4_core::Session, job: &Job) -> Result<()> {
    session.sync(&job.prompt)?;
    let eos = engine.tokenizer().eos_id();
    let mut rng = job.sampler.rng_seed;
    for _ in 0..job.n_predict {
        let t = if job.sampler.temperature > 0.0 {
            session.sample(
                job.sampler.temperature, job.sampler.top_k,
                job.sampler.top_p, job.sampler.min_p, &mut rng,
            )
        } else {
            session.argmax()
        };
        if t == eos {
            if !job.eos_only { (job.on_token)(t); }
            break;
        }
        (job.on_token)(t);
        session.eval(t)?;
    }
    Ok(())
}
