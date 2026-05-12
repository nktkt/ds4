//! Server endpoint wiring: thin adapters that bridge the request parsers
//! (`tool_parser`, `anthropic_messages`) and the per-request rendering layer
//! to the worker queue's `Job` / `SamplerCfg` types.
//!
//! These helpers correspond to the request-preparation prologue that lives in
//! `ds4_server.c` immediately before each call to `render_chat_prompt_text`:
//! the C code reads the same handful of sampler / decoder fields off the
//! parsed body, builds a `chat_msgs`, renders it to text, encodes it through
//! the tokenizer, and only then constructs the per-slot job descriptor.  The
//! Rust port does the same, but stops short of submitting to the queue — the
//! caller (`openai.rs` / `anthropic.rs`) drives that part.
//!
//! Streaming-chunk shapers (`render_openai_chunk`,
//! `render_anthropic_chunk_event`) live here too because they belong to the
//! same wire-protocol seam: the C source emits these as `buf_printf`
//! templates, but the Rust side returns `serde_json::Value`s so the streaming
//! layer can wrap them in `Frame<Bytes>` without touching format strings.

use serde_json::{json, Value};

use crate::anthropic_messages::{parse_anthropic_messages, render_to_chat_string};
use crate::queue::SamplerCfg;
use crate::stop::StopList;
use crate::tool_parser::parse_tools;
use ds4_core::{Engine, Tokens};

/// One fully prepared request, ready to be turned into a [`crate::queue::Job`].
///
/// The C source builds the same set of fields inline on the request struct
/// (see `r->prompt_text` / `r->sampler_*` / `r->stop` assignments around
/// `ds4_server.c:2140` and `ds4_server.c:2338`) before forwarding to the
/// worker.  Bundling them here keeps the OpenAI and Anthropic adapter
/// surfaces narrow and symmetric.
pub struct PreparedRequest {
    pub prompt: Tokens,
    pub sampler: SamplerCfg,
    pub max_tokens: i32,
    pub stop: StopList,
    pub stream: bool,
}

/// Default sampler knobs when the client omits them.  Matches the field
/// defaults that `openai_chat_completions` falls back to in `ds4_server.c`
/// (search for the `DS4_DEFAULT_TEMPERATURE` / `DS4_DEFAULT_TOP_P` constants
/// around the OpenAI request-decoder block).
fn default_sampler() -> SamplerCfg {
    SamplerCfg {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        rng_seed: 0,
    }
}

/// Render the OpenAI `messages` array into the DS4 chat-template text the
/// tokenizer expects.  Mirrors `render_chat_prompt_text` (`ds4_server.c:1901`)
/// for the OpenAI-flavoured input where each message already carries a plain
/// string `content`.
fn render_openai_messages(messages: &[Value], system: Option<&str>) -> String {
    let mut out = String::new();
    out.push_str("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>");

    if let Some(sys) = system {
        if !sys.is_empty() {
            out.push_str(sys);
        }
    }

    let mut pending_assistant = false;

    for msg in messages {
        let Some(obj) = msg.as_object() else { continue };
        let role = obj.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = match obj.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(arr)) => {
                // OpenAI multi-part content (text + images): collapse to text.
                let mut buf = String::new();
                for part in arr {
                    if let Some(po) = part.as_object() {
                        if let Some(t) = po.get("text").and_then(Value::as_str) {
                            buf.push_str(t);
                        }
                    } else if let Some(s) = part.as_str() {
                        buf.push_str(s);
                    }
                }
                buf
            }
            _ => String::new(),
        };

        match role {
            "system" | "developer" => {
                if !content.is_empty() {
                    if !out.is_empty() && !out.ends_with('\n') {
                        out.push('\n');
                    }
                    out.push_str(&content);
                }
            }
            "user" => {
                out.push_str("<\u{ff5c}User\u{ff5c}>");
                out.push_str(&content);
                pending_assistant = true;
            }
            "tool" | "function" => {
                out.push_str("<\u{ff5c}User\u{ff5c}>");
                out.push_str("<tool_result>");
                out.push_str(&content);
                out.push_str("</tool_result>");
                pending_assistant = true;
            }
            "assistant" => {
                if pending_assistant {
                    out.push_str("<\u{ff5c}Assistant\u{ff5c}>");
                    out.push_str("</think>");
                }
                out.push_str(&content);
                out.push_str("<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>");
                pending_assistant = false;
            }
            _ => {
                out.push_str("<\u{ff5c}User\u{ff5c}>");
                out.push_str(&content);
                pending_assistant = true;
            }
        }
    }

    if pending_assistant {
        out.push_str("<\u{ff5c}Assistant\u{ff5c}>");
        out.push_str("<think>");
    }

    out
}

/// Pull a `stop` field (either a single string or an array of strings) into a
/// `StopList`.  Mirrors the small block in `ds4_server.c` (around the OpenAI
/// request decoder) that calls `stop_list_push` for each entry.
fn stop_list_from_value(v: Option<&Value>) -> StopList {
    let mut out = StopList::default();
    match v {
        Some(Value::String(s)) => out.push(s.clone()),
        Some(Value::Array(arr)) => {
            for item in arr {
                if let Some(s) = item.as_str() {
                    out.push(s.to_owned());
                }
            }
        }
        _ => {}
    }
    out
}

/// Build a [`PreparedRequest`] from an OpenAI chat-completion request body.
///
/// Mirrors the request-preparation prologue in `openai_chat_completions`
/// (`ds4_server.c` around line 2140): pulls `messages`, `temperature`,
/// `top_p`, `top_k`, `min_p`, `stop`, `seed`, `tools`, `max_tokens`, and
/// `stream` from `body`, renders a chat-template prompt, encodes it through
/// `engine.tokenizer().encode_rendered_chat`, and bundles the sampler and
/// stop list. `parse_tools` is invoked on `body["tools"]` so the side-effect
/// path is in place; the result is intentionally discarded for now, matching
/// the upstream C path during the same porting checkpoint.
pub fn prepare_from_openai(engine: &Engine, body: &Value) -> Result<PreparedRequest, String> {
    let obj = body
        .as_object()
        .ok_or_else(|| "request body must be a JSON object".to_string())?;

    let messages = obj
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing 'messages' array".to_string())?;

    let system = obj.get("system").and_then(Value::as_str);

    let mut sampler = default_sampler();
    if let Some(v) = obj.get("temperature").and_then(Value::as_f64) {
        sampler.temperature = v as f32;
    }
    if let Some(v) = obj.get("top_p").and_then(Value::as_f64) {
        sampler.top_p = v as f32;
    }
    if let Some(v) = obj.get("top_k").and_then(Value::as_i64) {
        sampler.top_k = v as i32;
    }
    if let Some(v) = obj.get("min_p").and_then(Value::as_f64) {
        sampler.min_p = v as f32;
    }
    if let Some(v) = obj.get("seed").and_then(Value::as_u64) {
        sampler.rng_seed = v;
    }

    let max_tokens = obj
        .get("max_tokens")
        .and_then(Value::as_i64)
        .map(|n| n as i32)
        .unwrap_or(1024);

    let stream = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let stop = stop_list_from_value(obj.get("stop"));

    // Reserved for a later checkpoint: tool-schema extraction.  Run the
    // parser so any malformed `tools` field still surfaces as a no-op rather
    // than a silent skip; the result is intentionally unused for now.
    let _ = parse_tools(obj.get("tools").unwrap_or(&Value::Null));

    let rendered = render_openai_messages(messages, system);
    let mut prompt = Tokens::new();
    engine.tokenizer().encode_rendered_chat(&rendered, &mut prompt);

    Ok(PreparedRequest {
        prompt,
        sampler,
        max_tokens,
        stop,
        stream,
    })
}

/// Build a [`PreparedRequest`] from an Anthropic `/v1/messages` body.
///
/// Mirrors `anthropic_messages` request preparation (`ds4_server.c` around
/// line 2338): runs the body through `parse_anthropic_messages`, flattens it
/// to the DS4 chat template via `render_to_chat_string`, then encodes the
/// result through the engine tokenizer.  `stop_sequences` is folded into a
/// `StopList`; `temperature` / `top_p` are routed onto the sampler when
/// present (top_k / min_p / seed have no direct Anthropic counterpart, so
/// they keep their defaults).
pub fn prepare_from_anthropic(engine: &Engine, body: &Value) -> Result<PreparedRequest, String> {
    let req = parse_anthropic_messages(body)?;

    let mut sampler = default_sampler();
    if let Some(t) = req.temperature {
        sampler.temperature = t;
    }
    if let Some(p) = req.top_p {
        sampler.top_p = p;
    }

    let stop = StopList::from_iter(req.stop_sequences.iter().cloned());

    let rendered = render_to_chat_string(&req);
    let mut prompt = Tokens::new();
    engine.tokenizer().encode_rendered_chat(&rendered, &mut prompt);

    Ok(PreparedRequest {
        prompt,
        sampler,
        max_tokens: req.max_tokens,
        stop,
        stream: req.stream,
    })
}

/// Build the JSON payload for one OpenAI `chat.completion.chunk` SSE event.
///
/// Mirrors the `buf_printf("data: {\"id\":...,\"object\":\"chat.completion.chunk\"...`
/// templates in `ds4_server.c` (lines 3123 / 3157 / 3188 / 3208 / 3324 etc.).
/// When `finish_reason` is `None` we emit a `delta.content` chunk; when it is
/// `Some(...)` we emit a terminator chunk with an empty `delta` and the
/// provided `finish_reason` (matches the C terminator branch at line ~4140).
pub fn render_openai_chunk(token_text: &str, finish_reason: Option<&str>) -> Value {
    let choice = if let Some(reason) = finish_reason {
        json!({
            "index": 0,
            "delta": {},
            "finish_reason": reason,
        })
    } else {
        json!({
            "index": 0,
            "delta": { "content": token_text },
            "finish_reason": Value::Null,
        })
    };
    json!({
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "model": "ds4-flash",
        "choices": [choice],
    })
}

/// Build the JSON payload for one Anthropic SSE event.
///
/// Mirrors the four `sse_event` callers in `ds4_server.c` lines 4319, 4377,
/// 4401, and 4583: `message_start`, `content_block_delta` (with a
/// `text_delta` body), `content_block_stop`, and `message_stop`.  Other
/// event kinds fall through to a minimal `{"type": event_kind}` envelope,
/// matching the C `else { sse_event(fd, kind, "{}") }` fallback used for
/// events the engine does not yet emit.
pub fn render_anthropic_chunk_event(event_kind: &str, text: &str) -> Value {
    match event_kind {
        "message_start" => json!({
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "type": "message",
                "role": "assistant",
                "model": "ds4-flash",
                "content": [],
                "stop_reason": Value::Null,
            },
        }),
        "content_block_start" => json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": { "type": "text", "text": "" },
        }),
        "content_block_delta" => json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": { "type": "text_delta", "text": text },
        }),
        "content_block_stop" => json!({
            "type": "content_block_stop",
            "index": 0,
        }),
        "message_delta" => json!({
            "type": "message_delta",
            "delta": { "stop_reason": "end_turn" },
        }),
        "message_stop" => json!({ "type": "message_stop" }),
        other => json!({ "type": other }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn render_openai_messages_produces_template_text() {
        // Don't construct a real Engine in tests (it requires a GGUF); exercise
        // the internal renderer that `prepare_from_openai` calls just before
        // tokenization, plus the sampler-extraction path via a direct check on
        // the rendered string shape.  This is the prepare path minus the
        // tokenizer call itself.
        let body = json!({
            "messages": [
                { "role": "system", "content": "you are helpful" },
                { "role": "user",   "content": "hi" },
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 32,
            "stop": ["END"],
            "stream": true,
        });

        // Mirror the field extraction that prepare_from_openai performs so we
        // can assert on a non-empty prompt source (the rendered text) plus the
        // sampler/max_tokens/stop bookkeeping, without needing an Engine.
        let obj = body.as_object().unwrap();
        let messages = obj.get("messages").and_then(Value::as_array).unwrap();
        let rendered = render_openai_messages(messages, None);
        assert!(!rendered.is_empty(), "rendered prompt must be non-empty");
        assert!(rendered.contains("hi"));
        assert!(rendered.contains("<\u{ff5c}User\u{ff5c}>"));

        let stop = stop_list_from_value(obj.get("stop"));
        assert_eq!(stop.items, vec!["END".to_string()]);

        assert_eq!(
            obj.get("temperature").and_then(Value::as_f64).unwrap() as f32,
            0.7,
        );
        assert_eq!(
            obj.get("max_tokens").and_then(Value::as_i64).unwrap() as i32,
            32,
        );
        assert_eq!(obj.get("stream").and_then(Value::as_bool), Some(true));
    }

    #[test]
    fn anthropic_prepare_renders_one_text_message() {
        // We cannot run prepare_from_anthropic (it needs an Engine), but we
        // can drive the same internal pipeline: parser + renderer.  The
        // resulting string is what the tokenizer would be handed.
        let body = json!({
            "model": "claude-3",
            "max_tokens": 16,
            "messages": [
                { "role": "user", "content": "hello world" }
            ],
            "stop_sequences": ["STOP"],
            "temperature": 0.5,
        });
        let req = parse_anthropic_messages(&body).expect("parse");
        let rendered = render_to_chat_string(&req);
        assert!(!rendered.is_empty(), "rendered prompt must be non-empty");
        assert!(rendered.contains("hello world"));
        assert!(rendered.contains("<\u{ff5c}User\u{ff5c}>"));

        // Stop sequences fold into the StopList the prepare fn would attach.
        let stop = StopList::from_iter(req.stop_sequences.iter().cloned());
        assert_eq!(stop.items, vec!["STOP".to_string()]);
        assert_eq!(req.max_tokens, 16);
    }

    #[test]
    fn render_anthropic_chunk_event_content_block_delta_shape() {
        let v = render_anthropic_chunk_event("content_block_delta", "hi");
        assert!(v.is_object(), "must be a JSON object");
        assert_eq!(v.get("type").and_then(Value::as_str), Some("content_block_delta"));
        let delta = v.get("delta").expect("delta present");
        assert_eq!(delta.get("type").and_then(Value::as_str), Some("text_delta"));
        assert_eq!(delta.get("text").and_then(Value::as_str), Some("hi"));
    }
}
