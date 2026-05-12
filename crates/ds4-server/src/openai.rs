//! OpenAI-compatible endpoints. Ported from `ds4_server.c::openai_*`.
//!
//! `/v1/models` returns a single-element list (this server hosts one model).
//! `/v1/chat/completions` and `/v1/completions` accept the OpenAI request
//! schema, convert it into a chat transcript, then enqueue a generation job.

use crate::http::{json, text, AppState, BoxBod};
use anyhow::Result;
use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::body::Incoming;
use hyper::{Request, Response};
use serde::Deserialize;
use std::sync::Arc;

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
    pub seed: Option<u64>,
}

fn default_max_tokens() -> i32 { 1024 }
fn default_temperature() -> f32 { 0.0 }
fn default_top_p() -> f32 { 1.0 }

#[derive(Deserialize, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub async fn chat_completions(req: Request<Incoming>, _st: &Arc<AppState>) -> Result<Response<BoxBod>> {
    let bytes = req.collect().await?.to_bytes();
    let _parsed: ChatRequest = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("bad json: {e}"))?;
    // TODO: actually enqueue a job through `_st.queue` and stream the result.
    // For now we return a placeholder so the route is reachable end-to-end.
    Ok(json(200, serde_json::json!({
        "id": "chatcmpl-placeholder",
        "object": "chat.completion",
        "model": "ds4-flash",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "(not implemented yet)" },
            "finish_reason": "stop",
        }],
    })))
}

pub async fn completions(req: Request<Incoming>, _st: &Arc<AppState>) -> Result<Response<BoxBod>> {
    let _bytes: Bytes = req.collect().await?.to_bytes();
    Ok(text(200, "(not implemented yet)\n"))
}
