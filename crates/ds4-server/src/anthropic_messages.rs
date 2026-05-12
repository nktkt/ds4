//! Anthropic `/v1/messages` request parser.
//!
//! Ported from the C implementation in
//! `ds4/ds4_server.c` (functions `parse_anthropic_messages`,
//! `parse_anthropic_content`, `parse_anthropic_content_block`,
//! `append_anthropic_block_content`, `anthropic_system_part_is_private`).
//!
//! The C parser walks a hand-rolled JSON tokenizer and emits one compact
//! `chat_msg` per role.  This Rust port assumes the caller has already parsed
//! the body into a `serde_json::Value` (the rest of the workspace uses serde),
//! but preserves the same shape transformations:
//!
//!   * Anthropic block-structured content is collapsed into a single string
//!     content per role.
//!   * `tool_use` blocks (assistant) become structured `ToolUse` entries.
//!   * `tool_result` blocks are kept as escaped text, matching how DS4 sees
//!     tool results in its chat template.
//!   * `image` blocks are preserved with base64 data and media type so the
//!     downstream stages can drop them if the model is text-only.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One content block inside an Anthropic message.
///
/// Mirrors the block discriminants handled by `parse_anthropic_content_block`
/// in `ds4_server.c` (around line 1334).  Unknown block types are skipped by
/// `parse_anthropic_content_block`, the same way the C parser routes them
/// through `json_skip_value`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind")]
pub enum AnthropicContent {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
    Image {
        data: String,
        media_type: String,
    },
}

/// One Anthropic chat message: a role plus an ordered list of content blocks.
///
/// In the C source this is materialised as a `chat_msg` (see the `typedef
/// struct { char *role; char *content; ... } chat_msg;` definition near
/// `ds4_server.c` line 537).  The Rust side keeps the blocks structured so
/// downstream code can decide how to render them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnthropicMsg {
    pub role: String,
    pub content: Vec<AnthropicContent>,
}

/// Top-level Anthropic `/v1/messages` request as needed by the DS4 engine.
///
/// Matches the subset of fields decoded by `parse_anthropic_request` in
/// `ds4_server.c` (starting around line 2154): the engine ignores extension
/// fields (e.g. `metadata`, `service_tier`) and only keeps what affects
/// model semantics, rendering, or streaming.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnthropicRequest {
    pub model: Option<String>,
    pub system: Option<String>,
    pub messages: Vec<AnthropicMsg>,
    pub max_tokens: i32,
    pub stream: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Vec<String>,
    pub tools: Vec<Value>,
    pub tool_choice: Option<Value>,
}

/// Returns true for system parts that should be hidden from the model.
///
/// Mirrors `anthropic_system_part_is_private` in `ds4_server.c` (line ~1555):
/// any system string starting with `"x-anthropic-"` is private and must not
/// be appended to the rendered system prompt.
fn anthropic_system_part_is_private(s: &str) -> bool {
    s.starts_with("x-anthropic-")
}

/// Append a non-empty, non-private text part to the system buffer.
///
/// Mirrors `append_anthropic_system_part` (`ds4_server.c` line ~1559).
fn append_anthropic_system_part(out: &mut String, s: &str) {
    if s.is_empty() || anthropic_system_part_is_private(s) {
        return;
    }
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(s);
}

/// Append plain text to a block buffer.
///
/// Mirrors `append_anthropic_block_content` (`ds4_server.c` line ~1324),
/// which the C version uses both to concatenate adjacent text blocks and
/// to forward thinking content into the reasoning channel.
fn append_anthropic_block_content(dst: &mut String, text: &str) {
    if !text.is_empty() {
        dst.push_str(text);
    }
}

/// Escape a string for inclusion inside a `<tool_result>...</tool_result>`
/// section of the DS4 chat template.
///
/// The C side uses `append_dsml_text_escaped`, which escapes `<` and `>` to
/// keep nested DSML tags from being interpreted.  The implementation here
/// is intentionally minimal: it covers the angle brackets the engine cares
/// about and leaves everything else verbatim.
fn append_dsml_text_escaped(dst: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '<' => dst.push_str("&lt;"),
            '>' => dst.push_str("&gt;"),
            '&' => dst.push_str("&amp;"),
            c => dst.push(c),
        }
    }
}

/// Parse one Anthropic content block (`{ "type": "...", ... }`).
///
/// Ported from `parse_anthropic_content_block` in `ds4_server.c` (line ~1334).
/// Unknown block types return `None` (matching the C behaviour of skipping
/// the value).  Note: the C version mutates a `chat_msg` in place; this
/// Rust version returns a structured `AnthropicContent`, leaving role-aware
/// flattening to `parse_anthropic_messages` / `render_to_chat_string`.
pub fn parse_anthropic_content_block(v: &Value) -> Option<AnthropicContent> {
    let obj = v.as_object()?;
    let ty = obj.get("type").and_then(Value::as_str).unwrap_or("");

    match ty {
        "text" => {
            let text = obj.get("text").and_then(Value::as_str).unwrap_or("");
            Some(AnthropicContent::Text(text.to_string()))
        }
        "thinking" => {
            // The C parser routes thinking into msg->reasoning rather than the
            // visible content; we preserve it as a Text block here so the
            // renderer can decide where to put it.  An empty thinking block is
            // still meaningful (signals the assistant reasoned), but we keep
            // the payload only.
            let text = obj.get("thinking").and_then(Value::as_str).unwrap_or("");
            Some(AnthropicContent::Text(text.to_string()))
        }
        "tool_use" => {
            let id = obj
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let name = obj
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let input = obj.get("input").cloned().unwrap_or(Value::Object(
                serde_json::Map::new(),
            ));
            Some(AnthropicContent::ToolUse { id, name, input })
        }
        "tool_result" => {
            let tool_use_id = obj
                .get("tool_use_id")
                .or_else(|| obj.get("id"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            // tool_result.content may be a plain string or a list of inner
            // blocks (per the Anthropic schema).  We flatten to a single
            // string the same way `parse_anthropic_content` does for the
            // outer content array.
            let content = match obj.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(arr)) => {
                    let mut buf = String::new();
                    for item in arr {
                        match item {
                            Value::String(s) => buf.push_str(s),
                            Value::Object(_) => {
                                if let Some(AnthropicContent::Text(t)) =
                                    parse_anthropic_content_block(item)
                                {
                                    buf.push_str(&t);
                                }
                            }
                            _ => {}
                        }
                    }
                    buf
                }
                _ => String::new(),
            };
            Some(AnthropicContent::ToolResult {
                tool_use_id,
                content,
            })
        }
        "image" => {
            // Anthropic image blocks come as { type: "image", source: {
            // type: "base64", media_type: "...", data: "..." } }.
            let source = obj.get("source")?.as_object()?;
            let data = source
                .get("data")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            Some(AnthropicContent::Image { data, media_type })
        }
        _ => None,
    }
}

/// Parse the `content` field of a single Anthropic message into structured
/// blocks.
///
/// Mirrors `parse_anthropic_content` (`ds4_server.c` line ~1459): a plain
/// string is wrapped as a single `Text` block, a `null` becomes an empty
/// vector, and an array is walked element by element.
fn parse_anthropic_content_value(v: &Value) -> Vec<AnthropicContent> {
    match v {
        Value::String(s) => vec![AnthropicContent::Text(s.clone())],
        Value::Null => Vec::new(),
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                match item {
                    Value::String(s) => out.push(AnthropicContent::Text(s.clone())),
                    Value::Object(_) => {
                        if let Some(block) = parse_anthropic_content_block(item) {
                            out.push(block);
                        }
                    }
                    _ => {}
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Parse the `system` field, which may be a plain string, an array of strings,
/// or an array of `{ "type": "text", "text": "..." }` objects.
///
/// Mirrors `parse_anthropic_system` and `parse_anthropic_system_object`
/// (`ds4_server.c` lines ~1565 and ~1600).  Private parts whose text begins
/// with `x-anthropic-` are dropped via `anthropic_system_part_is_private`.
fn parse_anthropic_system(v: &Value) -> Option<String> {
    let mut out = String::new();
    match v {
        Value::String(s) => {
            append_anthropic_system_part(&mut out, s);
        }
        Value::Array(arr) => {
            for item in arr {
                match item {
                    Value::String(s) => append_anthropic_system_part(&mut out, s),
                    Value::Object(obj) => {
                        if let Some(text) = obj.get("text").and_then(Value::as_str) {
                            append_anthropic_system_part(&mut out, text);
                        }
                    }
                    _ => {}
                }
            }
        }
        Value::Null => return None,
        _ => return None,
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Parse a full Anthropic `/v1/messages` request body.
///
/// Mirrors `parse_anthropic_messages` (`ds4_server.c` line ~1494) together
/// with the top-level field decoding done by `parse_anthropic_request`
/// (line ~2154).  Returns a structured `AnthropicRequest` or a human-readable
/// error string.  Missing optional fields use Anthropic-compatible defaults
/// (`stream=false`, `max_tokens=0`, empty tool list).
pub fn parse_anthropic_messages(v: &Value) -> Result<AnthropicRequest, String> {
    let obj = v
        .as_object()
        .ok_or_else(|| "request body must be a JSON object".to_string())?;

    // messages
    let messages_value = obj
        .get("messages")
        .ok_or_else(|| "missing 'messages' field".to_string())?;
    let messages_arr = messages_value
        .as_array()
        .ok_or_else(|| "'messages' must be an array".to_string())?;

    let mut messages = Vec::with_capacity(messages_arr.len());
    for (i, m) in messages_arr.iter().enumerate() {
        let m_obj = m
            .as_object()
            .ok_or_else(|| format!("messages[{}] must be an object", i))?;
        let role = m_obj
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user")
            .to_string();
        let content = match m_obj.get("content") {
            Some(c) => parse_anthropic_content_value(c),
            None => Vec::new(),
        };
        messages.push(AnthropicMsg { role, content });
    }

    // system
    let system = obj.get("system").and_then(parse_anthropic_system);

    // model
    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);

    // max_tokens (Anthropic requires it, but we tolerate absence with 0)
    let max_tokens = obj
        .get("max_tokens")
        .and_then(Value::as_i64)
        .map(|n| n as i32)
        .unwrap_or(0);

    let stream = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let temperature = obj
        .get("temperature")
        .and_then(Value::as_f64)
        .map(|n| n as f32);

    let top_p = obj.get("top_p").and_then(Value::as_f64).map(|n| n as f32);

    let stop_sequences = obj
        .get("stop_sequences")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|s| s.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let tools = obj
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let tool_choice = obj.get("tool_choice").cloned();

    Ok(AnthropicRequest {
        model,
        system,
        messages,
        max_tokens,
        stream,
        temperature,
        top_p,
        stop_sequences,
        tools,
        tool_choice,
    })
}

/// Flatten a parsed request into the DS4 chat template, ready for the
/// tokenizer.
///
/// The template tokens match `render_chat_prompt_text` in `ds4_server.c`
/// (line ~1901):
///
/// ```text
/// <｜begin▁of▁sentence｜>{system}
/// <｜User｜>{user content or <tool_result>...}
/// <｜Assistant｜><think>{reasoning}</think>{text}{tool_calls}<｜end▁of▁sentence｜>
/// ...
/// <｜Assistant｜><think>      <- trailing turn opener for generation
/// ```
///
/// Tool-use blocks are serialised as a lightweight `<tool_call>` envelope so
/// that the engine sees them as part of the assistant turn even though the
/// real C code uses the DSML helpers (`append_dsml_tool_calls_text`).  This
/// is sufficient for tokenizer consumption: the engine treats the entire
/// rendered string as raw text.
pub fn render_to_chat_string(req: &AnthropicRequest) -> String {
    let mut out = String::new();
    out.push_str("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>");

    if let Some(sys) = req.system.as_deref() {
        if !sys.is_empty() {
            out.push_str(sys);
        }
    }

    let mut pending_assistant = false;
    let mut pending_tool_result = false;

    for msg in &req.messages {
        match msg.role.as_str() {
            "system" | "developer" => {
                // The C renderer collects system messages into a separate
                // buffer that is prepended to `out` before this loop runs.
                // For request-shaped input the `system` field is the
                // canonical channel; inline system roles are concatenated
                // after the dedicated system block to preserve order.
                let text = collect_text(&msg.content);
                if !text.is_empty() {
                    if !out.ends_with('\n') && !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(&text);
                }
            }
            "user" => {
                out.push_str("<\u{ff5c}User\u{ff5c}>");
                render_user_blocks(&mut out, &msg.content);
                pending_assistant = true;
                pending_tool_result = false;
            }
            "tool" | "function" => {
                if !pending_tool_result {
                    out.push_str("<\u{ff5c}User\u{ff5c}>");
                }
                for block in &msg.content {
                    if let AnthropicContent::ToolResult { content, .. } = block {
                        out.push_str("<tool_result>");
                        append_dsml_text_escaped(&mut out, content);
                        out.push_str("</tool_result>");
                    } else if let AnthropicContent::Text(t) = block {
                        out.push_str("<tool_result>");
                        append_dsml_text_escaped(&mut out, t);
                        out.push_str("</tool_result>");
                    }
                }
                pending_assistant = true;
                pending_tool_result = true;
            }
            "assistant" => {
                // Anthropic puts tool_results on user turns, but tolerate the
                // assistant role carrying them as well to match the C parser.
                let has_tool_result = msg
                    .content
                    .iter()
                    .any(|b| matches!(b, AnthropicContent::ToolResult { .. }));
                if has_tool_result {
                    if !pending_tool_result {
                        out.push_str("<\u{ff5c}User\u{ff5c}>");
                    }
                    for block in &msg.content {
                        if let AnthropicContent::ToolResult { content, .. } = block {
                            out.push_str("<tool_result>");
                            append_dsml_text_escaped(&mut out, content);
                            out.push_str("</tool_result>");
                        }
                    }
                    pending_assistant = true;
                    pending_tool_result = true;
                    continue;
                }

                if pending_assistant {
                    out.push_str("<\u{ff5c}Assistant\u{ff5c}>");
                    // We have no reasoning channel in this slice; emit the
                    // closing `</think>` only, matching the C renderer's
                    // non-thinking branch for completed turns.
                    out.push_str("</think>");
                }
                render_assistant_blocks(&mut out, &msg.content);
                out.push_str("<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>");
                pending_assistant = false;
                pending_tool_result = false;
            }
            _ => {
                // Unknown role: treat like a user turn to avoid losing data.
                out.push_str("<\u{ff5c}User\u{ff5c}>");
                render_user_blocks(&mut out, &msg.content);
                pending_assistant = true;
                pending_tool_result = false;
            }
        }
    }

    if pending_assistant {
        out.push_str("<\u{ff5c}Assistant\u{ff5c}>");
        out.push_str("<think>");
    }

    out
}

fn collect_text(blocks: &[AnthropicContent]) -> String {
    let mut buf = String::new();
    for b in blocks {
        if let AnthropicContent::Text(t) = b {
            append_anthropic_block_content(&mut buf, t);
        }
    }
    buf
}

fn render_user_blocks(out: &mut String, blocks: &[AnthropicContent]) {
    for b in blocks {
        match b {
            AnthropicContent::Text(t) => append_anthropic_block_content(out, t),
            AnthropicContent::ToolResult { content, .. } => {
                out.push_str("<tool_result>");
                append_dsml_text_escaped(out, content);
                out.push_str("</tool_result>");
            }
            AnthropicContent::Image { .. } => {
                // Text-only tokenizer: drop image payloads.  The C renderer
                // never emitted image bytes either; the API just ignores
                // them at this stage.
            }
            AnthropicContent::ToolUse { .. } => {
                // tool_use is an assistant-only concept; ignore if a client
                // mis-routes it onto a user turn.
            }
        }
    }
}

fn render_assistant_blocks(out: &mut String, blocks: &[AnthropicContent]) {
    for b in blocks {
        match b {
            AnthropicContent::Text(t) => append_anthropic_block_content(out, t),
            AnthropicContent::ToolUse { id, name, input } => {
                out.push_str("<tool_call>");
                out.push_str("<id>");
                append_dsml_text_escaped(out, id);
                out.push_str("</id>");
                out.push_str("<name>");
                append_dsml_text_escaped(out, name);
                out.push_str("</name>");
                out.push_str("<arguments>");
                append_dsml_text_escaped(out, &input.to_string());
                out.push_str("</arguments>");
                out.push_str("</tool_call>");
            }
            AnthropicContent::ToolResult { .. } | AnthropicContent::Image { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn simple_text_message() {
        let body = json!({
            "model": "claude-3-opus",
            "max_tokens": 256,
            "messages": [
                { "role": "user", "content": "hello" }
            ]
        });
        let req = parse_anthropic_messages(&body).expect("parse ok");
        assert_eq!(req.model.as_deref(), Some("claude-3-opus"));
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(
            req.messages[0].content,
            vec![AnthropicContent::Text("hello".to_string())]
        );

        let rendered = render_to_chat_string(&req);
        assert!(rendered.contains("hello"));
        assert!(rendered.starts_with("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>"));
        assert!(rendered.contains("<\u{ff5c}User\u{ff5c}>"));
        // Trailing assistant turn opener (no completed assistant message).
        assert!(rendered.ends_with("<think>"));
    }

    #[test]
    fn message_with_tool_use() {
        let body = json!({
            "max_tokens": 64,
            "messages": [
                { "role": "user", "content": "what's the weather?" },
                {
                    "role": "assistant",
                    "content": [
                        { "type": "text", "text": "let me check" },
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "get_weather",
                            "input": { "city": "Tokyo" }
                        }
                    ]
                }
            ]
        });
        let req = parse_anthropic_messages(&body).expect("parse ok");
        assert_eq!(req.messages.len(), 2);
        let assistant = &req.messages[1];
        assert_eq!(assistant.role, "assistant");
        assert_eq!(assistant.content.len(), 2);
        match &assistant.content[1] {
            AnthropicContent::ToolUse { id, name, input } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(input.get("city").and_then(Value::as_str), Some("Tokyo"));
            }
            other => panic!("expected ToolUse, got {:?}", other),
        }

        let rendered = render_to_chat_string(&req);
        assert!(rendered.contains("let me check"));
        assert!(rendered.contains("<tool_call>"));
        assert!(rendered.contains("get_weather"));
        // Assistant turn was completed, so end-of-sentence is emitted.
        assert!(rendered.contains("<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>"));
    }

    #[test]
    fn message_with_tool_result() {
        let body = json!({
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "sunny <16C>"
                        }
                    ]
                }
            ]
        });
        let req = parse_anthropic_messages(&body).expect("parse ok");
        assert_eq!(req.messages.len(), 1);
        match &req.messages[0].content[0] {
            AnthropicContent::ToolResult {
                tool_use_id,
                content,
            } => {
                assert_eq!(tool_use_id, "call_1");
                assert_eq!(content, "sunny <16C>");
            }
            other => panic!("expected ToolResult, got {:?}", other),
        }

        let rendered = render_to_chat_string(&req);
        assert!(rendered.contains("<tool_result>"));
        // The angle brackets in the payload must be escaped.
        assert!(rendered.contains("sunny &lt;16C&gt;"));
        assert!(rendered.contains("</tool_result>"));
    }
}
