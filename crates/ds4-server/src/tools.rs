//! Tool-call mapping. Ported from `ds4_server.c::tool_*`.
//!
//! The DS4 model emits tool calls inside a structured `<tool>...</tool>` span.
//! This module hosts the parser that converts that span into the JSON shape
//! both the OpenAI and Anthropic APIs expect, plus a [`rax::Tree`] of
//! tool-name → metadata for fast routing.

use rax::Tree;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

pub struct ToolTable {
    inner: Tree<ToolSpec>,
}

impl Default for ToolTable {
    fn default() -> Self { Self::new() }
}

impl ToolTable {
    pub fn new() -> Self { Self { inner: Tree::new() } }
    pub fn insert(&mut self, spec: ToolSpec) {
        let name = spec.name.clone();
        self.inner.insert(name.as_bytes(), spec);
    }
    pub fn get(&self, name: &str) -> Option<&ToolSpec> {
        self.inner.find(name.as_bytes())
    }
    pub fn len(&self) -> u64 { self.inner.len() }
    pub fn is_empty(&self) -> bool { self.inner.is_empty() }
}

/// Parse a `<tool>…</tool>` span emitted by the model. Returns the parsed
/// JSON if the span is well-formed, otherwise `None` (the streamer keeps
/// emitting raw text in that case).
pub fn parse_tool_span(text: &str) -> Option<serde_json::Value> {
    let start = text.find("<tool>")?;
    let end   = text[start..].find("</tool>")? + start;
    let inner = text[start + "<tool>".len()..end].trim();
    serde_json::from_str(inner).ok()
}
