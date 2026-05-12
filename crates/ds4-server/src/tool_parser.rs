//! HTTP server tool-call JSON parsing. Ported from `ds4_server.c`.
//!
//! The C version walks the request body byte-by-byte with a hand-rolled JSON
//! tokenizer. We let `serde_json` do the lexing and then reach into the
//! resulting [`serde_json::Value`] tree, which keeps the surface area the same
//! while removing all of the manual `free(key)`/`goto bad` plumbing.
//!
//! The one place we still care about textual JSON is `arguments`: the OpenAI
//! wire format sometimes sends it as a JSON string ("{\"x\":1}") and sometimes
//! as a raw object ({"x":1}). The C code preserves whatever the client sent;
//! we do the same by always exposing `arguments` as a [`String`] that contains
//! valid JSON.
//!
//! Property ordering for tool schemas is preserved in declaration order via
//! `serde_json`'s `preserve_order` feature (enabled in the workspace) — the
//! resulting [`Vec<String>`] inside [`ToolSchemaOrder`] mirrors the order the
//! client sent, exactly like `tool_schema_order_prop_push` does in C.

use serde_json::Value;

use crate::tools::ToolSpec;

/// One entry in an assistant message's `tool_calls` array.
///
/// Mirrors the C `tool_call` struct (ds4_server.c:504).  `arguments` is kept
/// as a raw JSON string because the upstream model and downstream clients both
/// treat the argument blob opaquely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Records the property order of one tool's JSON schema so the model sees
/// fields in the same order the client declared them.
///
/// Mirrors the C `tool_schema_order` struct (ds4_server.c:524).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolSchemaOrder {
    pub name: String,
    pub properties: Vec<String>,
}

/// Parse one `{"name":..., "arguments":...}` object into a [`ToolCall`].
///
/// Ported from `parse_function_call` (ds4_server.c:966). The `id` field is
/// left empty — the C version of `parse_function_call` only fills `name` and
/// `arguments`; the caller (`parse_tool_calls_value`) is responsible for
/// supplying the `id`.
pub fn parse_function_call(v: &Value) -> Option<ToolCall> {
    let obj = v.as_object()?;
    let name = obj.get("name")?.as_str()?.to_owned();
    let arguments = match obj.get("arguments")? {
        // Already a string: pass through verbatim (matches `json_string` path).
        Value::String(s) => s.clone(),
        // Raw JSON value: re-serialize so downstream sees valid JSON text
        // (matches the `json_raw_value` path).
        other => serde_json::to_string(other).ok()?,
    };
    Some(ToolCall { id: String::new(), name, arguments })
}

/// Parse an array of `{"id":..., "function":{"name":..., "arguments":...}}`
/// objects into a vector of [`ToolCall`]s.
///
/// Ported from `parse_tool_calls_value` (ds4_server.c:1014). A `null` value
/// produces an empty vector (same as the C "no-op success" path). Entries
/// without both `name` and `arguments` are skipped, matching the C check
/// `if (tc.name && tc.arguments)`.
pub fn parse_tool_calls(v: &Value) -> Vec<ToolCall> {
    if v.is_null() {
        return Vec::new();
    }
    let Some(arr) = v.as_array() else { return Vec::new(); };
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let Some(obj) = entry.as_object() else { continue };
        let id = obj.get("id").and_then(Value::as_str).unwrap_or("").to_owned();
        let Some(func) = obj.get("function") else { continue };
        let Some(mut call) = parse_function_call(func) else { continue };
        if call.name.is_empty() {
            continue;
        }
        call.id = id;
        out.push(call);
    }
    out
}

/// Strip the OpenAI `{"type":"function","function":{...}}` wrapper, returning
/// just the inner function-schema JSON as a string.
///
/// Ported from `openai_function_schema_from_tool` (ds4_server.c:1080).
/// Returns `None` if the input is not a JSON object or does not contain a
/// `function` key — the caller then treats `raw` itself as the schema, which
/// matches the C code's `function ? function : raw` fallback.
pub fn openai_function_schema_from_tool(raw: &str) -> Option<String> {
    let v: Value = serde_json::from_str(raw).ok()?;
    let obj = v.as_object()?;
    let function = obj.get("function")?;
    serde_json::to_string(function).ok()
}

/// Pull the property names (in declaration order) out of a tool schema's
/// `properties` object, appending them onto `order`.
///
/// Ported from `parse_schema_properties` (ds4_server.c:1110). Operates on an
/// already-parsed [`Value`] rather than reparsing a string, but the externally
/// visible behavior — appending property names in the order they appear — is
/// identical because `serde_json` is built with `preserve_order` in this
/// workspace.
fn parse_schema_properties(schema: &Value, order: &mut ToolSchemaOrder) {
    let Some(obj) = schema.as_object() else { return };
    let Some(props) = obj.get("properties").and_then(Value::as_object) else { return };
    for (k, _) in props {
        order.properties.push(k.clone());
    }
}

/// Inspect one tool-schema object and, if it has both a `name` and a
/// non-empty property list, append a fresh [`ToolSchemaOrder`] to `orders`.
///
/// Ported from `tool_schema_orders_add_json` (ds4_server.c:1159). Accepts
/// either OpenAI's `parameters` or Anthropic's `input_schema` — the two
/// places a function's argument schema may live.
fn tool_schema_orders_add(orders: &mut Vec<ToolSchemaOrder>, schema: &Value) {
    let Some(obj) = schema.as_object() else { return };
    let mut order = ToolSchemaOrder::default();
    if let Some(name) = obj.get("name").and_then(Value::as_str) {
        order.name = name.to_owned();
    }
    if let Some(s) = obj.get("input_schema").or_else(|| obj.get("parameters")) {
        parse_schema_properties(s, &mut order);
    }
    if !order.name.is_empty() && !order.properties.is_empty() {
        orders.push(order);
    }
}

/// Parse the top-level `tools` array of a chat-completions request into
/// (a) a vector of [`ToolSpec`]s and (b) a vector of [`ToolSchemaOrder`]s
/// recording the per-tool property declaration order.
///
/// Ported from `parse_tools_value` (ds4_server.c:1211). OpenAI-style entries
/// (`{"type":"function","function":{...}}`) are unwrapped via
/// [`openai_function_schema_from_tool`]; Anthropic-style entries that already
/// look like `{"name":..., "input_schema":...}` are kept as-is. A `null`
/// value yields two empty vectors.
pub fn parse_tools(v: &Value) -> (Vec<ToolSpec>, Vec<ToolSchemaOrder>) {
    let mut specs = Vec::new();
    let mut orders = Vec::new();
    if v.is_null() {
        return (specs, orders);
    }
    let Some(arr) = v.as_array() else { return (specs, orders); };
    for entry in arr {
        // Unwrap OpenAI's {"type":"function","function":{...}} if present.
        let schema: &Value = entry
            .as_object()
            .and_then(|o| o.get("function"))
            .unwrap_or(entry);
        let Some(obj) = schema.as_object() else { continue };

        let name = match obj.get("name").and_then(Value::as_str) {
            Some(n) if !n.is_empty() => n.to_owned(),
            _ => continue,
        };
        let description = obj
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_owned);
        // Both OpenAI (`parameters`) and Anthropic (`input_schema`) variants.
        let parameters = obj
            .get("parameters")
            .or_else(|| obj.get("input_schema"))
            .cloned()
            .unwrap_or(Value::Null);

        specs.push(ToolSpec { name, description, parameters });
        tool_schema_orders_add(&mut orders, schema);
    }
    (specs, orders)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_function_call_single_object() {
        // Arguments as a JSON-encoded string — the OpenAI wire format.
        let v = json!({
            "name": "get_weather",
            "arguments": "{\"city\":\"Tokyo\"}",
        });
        let call = parse_function_call(&v).expect("must parse");
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, r#"{"city":"Tokyo"}"#);
        assert!(call.id.is_empty(), "id is filled by the array caller");
    }

    #[test]
    fn parse_function_call_raw_object_arguments() {
        // Some clients send `arguments` as a raw object rather than a string.
        // The C code accepts both via `json_raw_value`; we re-serialize so the
        // downstream consumer still sees a JSON string.
        let v = json!({
            "name": "echo",
            "arguments": { "msg": "hi" },
        });
        let call = parse_function_call(&v).expect("must parse");
        assert_eq!(call.name, "echo");
        let reparsed: Value = serde_json::from_str(&call.arguments).unwrap();
        assert_eq!(reparsed, json!({ "msg": "hi" }));
    }

    #[test]
    fn parse_tool_calls_array_of_two() {
        let v = json!([
            {
                "id": "call_1",
                "type": "function",
                "function": { "name": "a", "arguments": "{}" },
            },
            {
                "id": "call_2",
                "function": { "name": "b", "arguments": "{\"k\":1}" },
            },
        ]);
        let calls = parse_tool_calls(&v);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[0].arguments, "{}");
        assert_eq!(calls[1].id, "call_2");
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].arguments, r#"{"k":1}"#);
    }

    #[test]
    fn parse_tools_preserves_schema_property_order() {
        // OpenAI-style wrapper for `search`; Anthropic-style for `lookup`.
        // Property order is `query, limit, offset` — the model must see
        // exactly that order, not alphabetical.
        let v = json!([
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "full-text search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query":  { "type": "string" },
                            "limit":  { "type": "integer" },
                            "offset": { "type": "integer" },
                        },
                    },
                },
            },
            {
                "name": "lookup",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                    },
                },
            },
        ]);
        let (specs, orders) = parse_tools(&v);
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "search");
        assert_eq!(specs[0].description.as_deref(), Some("full-text search"));
        assert_eq!(specs[1].name, "lookup");

        assert_eq!(orders.len(), 2);
        assert_eq!(orders[0].name, "search");
        assert_eq!(orders[0].properties, vec!["query", "limit", "offset"]);
        assert_eq!(orders[1].name, "lookup");
        assert_eq!(orders[1].properties, vec!["id"]);
    }

    #[test]
    fn openai_function_schema_from_tool_unwraps_wrapper() {
        let raw = r#"{"type":"function","function":{"name":"x","parameters":{}}}"#;
        let inner = openai_function_schema_from_tool(raw).expect("unwrap");
        let parsed: Value = serde_json::from_str(&inner).unwrap();
        assert_eq!(parsed, json!({ "name": "x", "parameters": {} }));

        // No `function` key → None, caller falls back to the raw schema.
        let direct = r#"{"name":"y","input_schema":{}}"#;
        assert!(openai_function_schema_from_tool(direct).is_none());
    }
}
