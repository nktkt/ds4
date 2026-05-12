//! Anthropic-compatible `/v1/messages` endpoint. Ported from
//! `ds4_server.c::anthropic_*`. Same shape as the OpenAI path but the request
//! and response JSON encodings differ.

use crate::http::{json, AppState, BoxBod};
use anyhow::Result;
use http_body_util::BodyExt;
use hyper::body::Incoming;
use hyper::{Request, Response};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize, Debug)]
pub struct MessagesRequest {
    pub model: Option<String>,
    pub messages: Vec<AnthropicMsg>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max")]
    pub max_tokens: i32,
    pub system: Option<String>,
}

fn default_max() -> i32 { 1024 }

#[derive(Deserialize, Debug)]
pub struct AnthropicMsg {
    pub role: String,
    pub content: serde_json::Value,
}

pub async fn messages(req: Request<Incoming>, _st: &Arc<AppState>) -> Result<Response<BoxBod>> {
    let bytes = req.collect().await?.to_bytes();
    let _parsed: MessagesRequest = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("bad json: {e}"))?;
    Ok(json(200, serde_json::json!({
        "id": "msg_placeholder",
        "type": "message",
        "role": "assistant",
        "model": "ds4-flash",
        "content": [{ "type": "text", "text": "(not implemented yet)" }],
        "stop_reason": "end_turn",
    })))
}
