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
        // Round-trip through the rendered chat path. Matches
        // `ds4_encode_chat_prompt` / `ds4_chat_append_*` from the C side.
        let mut rendered = String::new();
        if let Some(sys) = &self.system {
            rendered.push_str("<|im_start|>system\n");
            rendered.push_str(sys);
            rendered.push_str("<|im_end|>\n");
        }
        for t in &self.turns {
            rendered.push_str("<|im_start|>");
            rendered.push_str(&t.role);
            rendered.push('\n');
            rendered.push_str(&t.content);
            rendered.push_str("<|im_end|>\n");
        }
        rendered.push_str("<|im_start|>assistant\n");
        if matches!(think, ThinkMode::Max) {
            rendered.push_str(ThinkMode::max_prefix());
        }
        engine.tokenizer().encode_rendered_chat(&rendered, out);
    }
}
