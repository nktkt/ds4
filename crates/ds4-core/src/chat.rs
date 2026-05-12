//! Chat prompt encoding. Ports `ds4_encode_chat_prompt`,
//! `ds4_chat_begin`, `ds4_chat_append_*` from `ds4.c`. The DS4 chat template
//! is fixed:
//!
//! ```text
//! [bos]
//! [reasoning-effort-max-prefix?]
//! [system text]
//! [user-role]
//! [user prompt]
//! [assistant-role]
//! [<think>] or [</think>]
//! ```
//!
//! For ThinkMode::Max we prepend the long max-effort instructions; for
//! High we open with `<think>`; for None we close with `</think>` so the
//! model skips the chain-of-thought.

use crate::api::{ThinkMode, Tokens};
use crate::engine::Engine;
use crate::shape::REASONING_EFFORT_MAX_PREFIX;

pub fn begin(engine: &Engine, tokens: &mut Tokens) {
    let bos = engine.tokenizer().bos;
    if bos >= 0 { tokens.push(bos); }
}

/// Append the max-effort prompt prefix (`REASONING_EFFORT_MAX_PREFIX`)
/// already-tokenized. Mirrors `ds4_chat_append_max_effort_prefix`.
pub fn append_max_effort_prefix(engine: &Engine, tokens: &mut Tokens) {
    let mut buf = Tokens::new();
    engine.tokenizer().encode(REASONING_EFFORT_MAX_PREFIX, &mut buf);
    tokens.extend(buf.as_slice());
}

pub fn append_message(engine: &Engine, tokens: &mut Tokens, role: &str, content: &str) {
    let t = engine.tokenizer();
    let role_id = match role {
        "user" => t.user_role,
        "assistant" => t.assistant_role,
        "system" => t.system_role,
        _ => -1,
    };
    if role_id >= 0 { tokens.push(role_id); }
    engine.tokenizer().encode(content, tokens);
}

pub fn append_assistant_prefix(engine: &Engine, tokens: &mut Tokens, think_mode: ThinkMode) {
    let t = engine.tokenizer();
    if t.assistant_role >= 0 { tokens.push(t.assistant_role); }
    if think_mode.enabled() {
        if t.think_start >= 0 { tokens.push(t.think_start); }
    } else if t.think_end >= 0 {
        tokens.push(t.think_end);
    }
}

/// Full chat encoder. Mirrors `encode_chat_prompt`.
pub fn encode_chat_prompt(
    engine: &Engine,
    system: Option<&str>,
    prompt: &str,
    think_mode: ThinkMode,
    out: &mut Tokens,
) {
    begin(engine, out);
    if matches!(think_mode, ThinkMode::Max) {
        append_max_effort_prefix(engine, out);
    }
    if let Some(sys) = system {
        if !sys.is_empty() {
            engine.tokenizer().encode(sys, out);
        }
    }
    let t = engine.tokenizer();
    if t.user_role >= 0 { out.push(t.user_role); }
    t.encode(prompt, out);
    if t.assistant_role >= 0 { out.push(t.assistant_role); }
    if think_mode.enabled() {
        if t.think_start >= 0 { out.push(t.think_start); }
    } else if t.think_end >= 0 {
        out.push(t.think_end);
    }
}
