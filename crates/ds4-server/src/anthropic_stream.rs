//! Anthropic-style Server-Sent Events helpers.
//!
//! Ported from `ds4_server.c` — see the cluster of helpers around
//! `anthropic_sse_start_live`, `anthropic_sse_open_block`,
//! `anthropic_sse_delta_live`, `anthropic_sse_close_block_live`,
//! `anthropic_sse_tool_blocks_live`, and `anthropic_sse_stop_live`
//! (roughly lines 4312-4585 of the upstream C source).
//!
//! Each function returns a single SSE event already framed as
//! `event: <name>\ndata: <json>\n\n` via [`crate::stream::sse_chunk`].
//! Higher-level orchestration (the `anthropic_stream` state machine that
//! tracks open block index, think-mode masking and tool-call boundaries)
//! lives elsewhere; this module only emits the wire bytes for each
//! event the upstream code can produce.

use crate::stream::sse_chunk;
use bytes::Bytes;
use serde_json::json;

/// `event: message_start` with an empty assistant content block.
///
/// Mirrors `anthropic_sse_start_live` in `ds4_server.c` (~line 4312).
/// The upstream C also embeds an initial `usage.input_tokens`; we keep
/// `output_tokens: 0` and leave `input_tokens` to the caller via a
/// dedicated helper if needed — at this layer we expose only id+model
/// per the porting spec.
pub fn message_start(id: &str, model: &str) -> Bytes {
    let data = json!({
        "type": "message_start",
        "message": {
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": null,
            "stop_sequence": null,
            "usage": { "input_tokens": 0, "output_tokens": 0 }
        }
    });
    sse_chunk("message_start", &data.to_string())
}

/// `event: content_block_start` for a new block at `index`.
///
/// Mirrors `anthropic_sse_open_block` (~line 4333) for the text path
/// and the tool-block prelude in `anthropic_sse_tool_blocks_live`
/// (~line 4544). `block_type` is typically `"text"` or `"tool_use"`.
/// For `tool_use` the upstream C also fills `id`/`name`/`input`; here we
/// emit a minimal shell that the caller can override at a higher layer.
pub fn content_block_start(index: i32, block_type: &str) -> Bytes {
    let content_block = match block_type {
        "tool_use" => json!({
            "type": "tool_use",
            "id": "",
            "name": "",
            "input": {}
        }),
        "thinking" => json!({
            "type": "thinking",
            "thinking": "",
            "signature": ""
        }),
        _ => json!({
            "type": block_type,
            "text": ""
        }),
    };
    let data = json!({
        "type": "content_block_start",
        "index": index,
        "content_block": content_block,
    });
    sse_chunk("content_block_start", &data.to_string())
}

/// `event: content_block_delta` carrying an incremental `text_delta`.
///
/// Mirrors the text branch of `anthropic_sse_delta_live` (~line 4369).
pub fn content_block_delta_text(index: i32, text: &str) -> Bytes {
    let data = json!({
        "type": "content_block_delta",
        "index": index,
        "delta": {
            "type": "text_delta",
            "text": text,
        }
    });
    sse_chunk("content_block_delta", &data.to_string())
}

/// `event: content_block_delta` carrying an `input_json_delta` chunk
/// of partial tool-call arguments.
///
/// Mirrors the tool-call branch in `anthropic_sse_tool_blocks_live`
/// (~line 4556) which emits `delta.type = "input_json_delta"` with a
/// `partial_json` string payload.
pub fn content_block_delta_input_json(index: i32, partial_json: &str) -> Bytes {
    let data = json!({
        "type": "content_block_delta",
        "index": index,
        "delta": {
            "type": "input_json_delta",
            "partial_json": partial_json,
        }
    });
    sse_chunk("content_block_delta", &data.to_string())
}

/// `event: content_block_stop` for `index`.
///
/// Mirrors the `content_block_stop` emit at the tail of
/// `anthropic_sse_close_block_live` (~line 4399) and the matching
/// stop inside `anthropic_sse_tool_blocks_live` (~line 4565).
pub fn content_block_stop(index: i32) -> Bytes {
    let data = json!({
        "type": "content_block_stop",
        "index": index,
    });
    sse_chunk("content_block_stop", &data.to_string())
}

/// `event: message_delta` carrying the final `stop_reason` plus
/// cumulative `usage.output_tokens`.
///
/// Mirrors `anthropic_sse_stop_live` (~line 4574). Upstream wraps the
/// finish reason through `anthropic_stop_reason`; we trust the caller
/// to pass the already-normalised value (`"end_turn"`, `"tool_use"`,
/// `"max_tokens"`, …).
pub fn message_delta(stop_reason: &str, output_tokens: i32) -> Bytes {
    let data = json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": null,
        },
        "usage": { "output_tokens": output_tokens }
    });
    sse_chunk("message_delta", &data.to_string())
}

/// `event: message_stop`. The final structured event before the
/// transport-level `[DONE]` sentinel.
///
/// Mirrors the trailing `sse_event(fd, "message_stop", …)` call inside
/// `anthropic_sse_stop_live` (~line 4583).
pub fn message_stop() -> Bytes {
    let data = json!({ "type": "message_stop" });
    sse_chunk("message_stop", &data.to_string())
}

/// `event: ping` keep-alive. Not produced by the streaming helpers in
/// `ds4_server.c` directly, but part of the Anthropic SSE protocol that
/// long-lived `/v1/messages` streams emit to keep proxies from idling
/// the connection out. Kept here so the server can interleave them
/// alongside the data events from the upstream port.
pub fn ping() -> Bytes {
    let data = json!({ "type": "ping" });
    sse_chunk("ping", &data.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Strip the `event: <name>\ndata: ` envelope and the trailing
    /// `\n\n`, returning `(event_name, parsed_json)`.
    fn parse(bytes: &Bytes) -> (String, Value) {
        let s = std::str::from_utf8(bytes).expect("utf8");
        let mut lines = s.splitn(2, '\n');
        let event_line = lines.next().expect("event line");
        let rest = lines.next().expect("data line");
        let event = event_line
            .strip_prefix("event: ")
            .expect("event prefix")
            .to_string();
        let data = rest
            .strip_prefix("data: ")
            .expect("data prefix")
            .trim_end_matches('\n');
        let v: Value = serde_json::from_str(data).expect("json");
        (event, v)
    }

    #[test]
    fn message_start_event_shape() {
        let b = message_start("msg_123", "ds4-flash");
        let (event, v) = parse(&b);
        assert_eq!(event, "message_start");
        assert_eq!(v["type"], "message_start");
        assert_eq!(v["message"]["id"], "msg_123");
        assert_eq!(v["message"]["model"], "ds4-flash");
        assert_eq!(v["message"]["role"], "assistant");
        assert!(v["message"]["content"].is_array());
        assert!(v["message"]["usage"]["input_tokens"].is_number());
        assert!(v["message"]["usage"]["output_tokens"].is_number());
    }

    #[test]
    fn content_block_delta_text_and_input_json() {
        let text = content_block_delta_text(0, "hello");
        let (event, v) = parse(&text);
        assert_eq!(event, "content_block_delta");
        assert_eq!(v["type"], "content_block_delta");
        assert_eq!(v["index"], 0);
        assert_eq!(v["delta"]["type"], "text_delta");
        assert_eq!(v["delta"]["text"], "hello");

        let tool = content_block_delta_input_json(2, "{\"q\":\"a\"}");
        let (event, v) = parse(&tool);
        assert_eq!(event, "content_block_delta");
        assert_eq!(v["index"], 2);
        assert_eq!(v["delta"]["type"], "input_json_delta");
        assert_eq!(v["delta"]["partial_json"], "{\"q\":\"a\"}");
    }

    #[test]
    fn lifecycle_events_have_expected_names_and_keys() {
        let start = content_block_start(0, "text");
        let (event, v) = parse(&start);
        assert_eq!(event, "content_block_start");
        assert_eq!(v["type"], "content_block_start");
        assert_eq!(v["index"], 0);
        assert_eq!(v["content_block"]["type"], "text");

        let stop_block = content_block_stop(0);
        let (event, v) = parse(&stop_block);
        assert_eq!(event, "content_block_stop");
        assert_eq!(v["index"], 0);

        let mdelta = message_delta("end_turn", 42);
        let (event, v) = parse(&mdelta);
        assert_eq!(event, "message_delta");
        assert_eq!(v["delta"]["stop_reason"], "end_turn");
        assert!(v["delta"]["stop_sequence"].is_null());
        assert_eq!(v["usage"]["output_tokens"], 42);

        let mstop = message_stop();
        let (event, v) = parse(&mstop);
        assert_eq!(event, "message_stop");
        assert_eq!(v["type"], "message_stop");

        let p = ping();
        let (event, v) = parse(&p);
        assert_eq!(event, "ping");
        assert_eq!(v["type"], "ping");
    }
}
