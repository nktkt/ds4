//! Chat transcript / template renderer.
//!
//! Ported from the `ds4_chat_*` helpers in `ds4.c`. Holds the list of
//! (role, content) turns and knows how to materialize them into a tokenized
//! prompt using the engine's tokenizer.

use ds4_core::{Engine, ThinkMode, Tokens};

#[derive(Debug, Clone)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Default)]
pub struct Transcript {
    pub turns: Vec<Turn>,
    pub system: Option<String>,
}

impl Transcript {
    pub fn new() -> Self { Self::default() }
    pub fn set_system(&mut self, s: impl Into<String>) { self.system = Some(s.into()); }
    pub fn push_user(&mut self, content: impl Into<String>) {
        self.turns.push(Turn { role: "user".into(), content: content.into() });
    }
    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.turns.push(Turn { role: "assistant".into(), content: content.into() });
    }
    pub fn render(&self, engine: &Engine, think: ThinkMode, out: &mut Tokens) {
        // Use the same template `ds4.c::encode_chat_prompt` uses, via the
        // ported helpers in ds4_core::chat. Walking the transcript turn-by-turn
        // keeps multi-turn KV-cache reuse working: prefix-sharing depends on
        // the token sequence not the rendered string.
        ds4_core::chat::begin(engine, out);
        if matches!(think, ThinkMode::Max) {
            ds4_core::chat::append_max_effort_prefix(engine, out);
        }
        if let Some(sys) = &self.system {
            engine.tokenizer().encode(sys, out);
        }
        for t in &self.turns {
            ds4_core::chat::append_message(engine, out, &t.role, &t.content);
        }
        ds4_core::chat::append_assistant_prefix(engine, out, think);
    }
}
