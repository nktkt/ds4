//! OpenAI-compatible endpoints. Ports the high-level shape of
//! `ds4_server.c::openai_*`. Streaming uses SSE.

use crate::http::{json, AppState, BoxBod};
use crate::queue::{Job, SamplerCfg};
use anyhow::Result;
use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::{Request, Response, StatusCode};
use parking_lot::Mutex;
use serde::Deserialize;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::mpsc;

pub fn list_models(_st: &AppState) -> Response<BoxBod> {
    json(200, serde_json::json!({
        "object": "list",
        "data": [{
            "id": "ds4-flash",
            "object": "model",
            "owned_by": "deepseek",
        }]
    }))
}

#[derive(Deserialize, Debug)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: i32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: i32,
    #[serde(default)]
    pub min_p: f32,
    pub seed: Option<u64>,
    pub system: Option<String>,
}

fn default_max_tokens() -> i32 { 1024 }
fn default_temperature() -> f32 { 0.0 }
fn default_top_p() -> f32 { 1.0 }

#[derive(Deserialize, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub async fn chat_completions(req: Request<Incoming>, st: &Arc<AppState>) -> Result<Response<BoxBod>> {
    let bytes = req.collect().await?.to_bytes();
    let parsed: ChatRequest = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("bad json: {e}"))?;

    let engine = match st.queue.engine.lock().clone() {
        Some(e) => e,
        None => return Ok(json(503, serde_json::json!({"error":"engine not ready"}))),
    };
    let mut tokens = ds4_core::Tokens::new();
    let rendered = render_chat(&parsed);
    engine.tokenizer().encode_rendered_chat(&rendered, &mut tokens);
    let sampler = SamplerCfg {
        temperature: parsed.temperature,
        top_k: parsed.top_k,
        top_p: parsed.top_p,
        min_p: parsed.min_p,
        rng_seed: parsed.seed.unwrap_or_else(|| rand::random()),
    };

    if parsed.stream {
        Ok(streaming_response(st, engine, tokens, parsed.max_tokens, sampler))
    } else {
        non_streaming(st, engine, tokens, parsed.max_tokens, sampler).await
    }
}

fn render_chat(req: &ChatRequest) -> String {
    let mut s = String::new();
    if let Some(sys) = &req.system {
        s.push_str("<|im_start|>system\n");
        s.push_str(sys);
        s.push_str("<|im_end|>\n");
    }
    for m in &req.messages {
        s.push_str("<|im_start|>");
        s.push_str(&m.role);
        s.push('\n');
        s.push_str(&m.content);
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>assistant\n");
    s
}

async fn non_streaming(
    _st: &Arc<AppState>,
    engine: Arc<ds4_core::Engine>,
    prompt: ds4_core::Tokens,
    max_tokens: i32,
    sampler: SamplerCfg,
) -> Result<Response<BoxBod>> {
    // For the non-streaming path we still post a job to the queue, but we
    // collect every token first and then send a single JSON response.
    let (tx, mut rx) = mpsc::unbounded_channel::<i32>();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<anyhow::Result<()>>();
    let done_tx = Arc::new(Mutex::new(Some(done_tx)));
    let job = Job {
        prompt,
        n_predict: max_tokens,
        eos_only: false,
        sampler,
        on_token: Box::new(move |t| { let _ = tx.send(t); }),
        on_done: Box::new({
            let done_tx = done_tx.clone();
            move |r| { if let Some(tx) = done_tx.lock().take() { let _ = tx.send(r); } }
        }),
    };
    _st.queue.submit(job)?;
    // Drain emitted tokens into a buffer.
    let mut text = String::new();
    let eos = engine.tokenizer().eos_id();
    while let Some(t) = rx.recv().await {
        if t == eos { break; }
        if let Some(b) = engine.tokenizer().token_text(t) {
            text.push_str(&String::from_utf8_lossy(b));
        }
    }
    let _ = done_rx.await;
    Ok(json(200, serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "ds4-flash",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    })))
}

fn streaming_response(
    st: &Arc<AppState>,
    engine: Arc<ds4_core::Engine>,
    prompt: ds4_core::Tokens,
    max_tokens: i32,
    sampler: SamplerCfg,
) -> Response<BoxBod> {
    use futures::StreamExt;
    let (tx, rx) = mpsc::unbounded_channel::<i32>();
    let (done_tx, _done_rx) = tokio::sync::oneshot::channel::<anyhow::Result<()>>();
    let done_tx = Arc::new(Mutex::new(Some(done_tx)));
    let job = Job {
        prompt,
        n_predict: max_tokens,
        eos_only: false,
        sampler,
        on_token: Box::new(move |t| { let _ = tx.send(t); }),
        on_done: Box::new({
            let done_tx = done_tx.clone();
            move |r| { if let Some(tx) = done_tx.lock().take() { let _ = tx.send(r); } }
        }),
    };
    if let Err(e) = st.queue.submit(job) {
        return json(503, serde_json::json!({"error": format!("submit: {e}")}));
    }
    // Adapter from token stream to SSE bytes.
    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
        .map({
            let engine = engine.clone();
            move |t| {
                let mut text = String::new();
                if let Some(b) = engine.tokenizer().token_text(t) {
                    text.push_str(&String::from_utf8_lossy(b));
                }
                let payload = serde_json::json!({
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "model": "ds4-flash",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                    }],
                });
                Ok::<Frame<Bytes>, Infallible>(Frame::data(crate::stream::sse_chunk(
                    "", &payload.to_string(),
                )))
            }
        })
        .chain(futures::stream::once(async {
            Ok::<Frame<Bytes>, Infallible>(Frame::data(crate::stream::sse_done()))
        }));
    let body = BodyExt::boxed(StreamBody::new(stream));
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-store")
        .body(body)
        .unwrap()
}

pub async fn completions(req: Request<Incoming>, _st: &Arc<AppState>) -> Result<Response<BoxBod>> {
    let _bytes: Bytes = req.collect().await?.to_bytes();
    Ok(text_response(200, "(legacy /v1/completions not yet wired up)\n"))
}

fn text_response(status: u16, msg: &str) -> Response<BoxBod> {
    Response::builder()
        .status(StatusCode::from_u16(status).unwrap())
        .header("content-type", "text/plain; charset=utf-8")
        .body(Full::new(Bytes::from(msg.to_owned())).boxed())
        .unwrap()
}
