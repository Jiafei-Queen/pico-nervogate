//! Shared primitives for protocol translators.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
static COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn now_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub fn gen_id(prefix: &str) -> String {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{prefix}{nanos:x}{n:x}")
}

/// Extract plain text from an OpenAI content field (string or array of parts).
pub fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
                    out.push_str(t);
                } else if let Some(r) = p.get("refusal").and_then(|x| x.as_str()) {
                    out.push_str(r);
                }
            }
            out
        }
        _ => String::new(),
    }
}

/// Assistant message text for Chat egresses. Handles string, array, and null
/// content, falling back to the top-level `refusal` field when content is
/// empty so refusal-only turns are not silently dropped.
pub fn chat_message_text(msg: &Value) -> String {
    let text = content_text(msg.get("content").unwrap_or(&Value::Null));
    if !text.is_empty() {
        return text;
    }
    msg.get("refusal")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string()
}

// ---------------------------------------------------------------------------
// SSE framing
// ---------------------------------------------------------------------------

/// One outbound SSE event. `event: None` emits `data:` only (Chat / Responses
/// framing); `event: Some(name)` emits an `event:` line first (Anthropic).
pub struct SseEvent {
    pub event: Option<String>,
    pub data: Value,
}

impl SseEvent {
    /// `data:`-only event (Chat chunks, Responses events).
    pub fn data(v: Value) -> Self {
        Self {
            event: None,
            data: v,
        }
    }
    /// `event:` + `data:` event (Anthropic).
    pub fn ev(name: &str, v: Value) -> Self {
        Self {
            event: Some(name.to_string()),
            data: v,
        }
    }
    pub fn frame(&self) -> String {
        match &self.event {
            None => frame_data(&self.data),
            Some(e) => frame_event(e, &self.data),
        }
    }
}

pub fn frame_data(v: &Value) -> String {
    let s = serde_json::to_string(v).unwrap_or_else(|_| "null".to_string());
    format!("data: {s}\n\n")
}

pub fn frame_event(name: &str, v: &Value) -> String {
    let s = serde_json::to_string(v).unwrap_or_else(|_| "null".to_string());
    format!("event: {name}\ndata: {s}\n\n")
}

// ---------------------------------------------------------------------------
// Usage shapes
// ---------------------------------------------------------------------------

/// Chat Completions `usage` object.
pub fn chat_usage(prompt: i64, completion: i64, cached: i64, reasoning: i64) -> Value {
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": reasoning}
    })
}

/// Responses API `usage` object.
pub fn responses_usage(input: i64, output: i64, cached: i64, reasoning: i64) -> Value {
    json!({
        "input_tokens": input,
        "output_tokens": output,
        "total_tokens": input + output,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens_details": {"reasoning_tokens": reasoning}
    })
}

/// Anthropic Messages `usage` object.
///
/// `cache_creation_input_tokens` must be carried, not zeroed: cache writes
/// are billed and dropping them understates cost on cache-heavy traffic.
pub fn anthropic_usage_full(
    input: i64,
    output: i64,
    cache_creation: i64,
    cache_read: i64,
) -> Value {
    json!({
        "input_tokens": input,
        "output_tokens": output,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read
    })
}

/// Anthropic Messages `usage` object, for callers that have no cache-write
/// count to report.
pub fn anthropic_usage(input: i64, output: i64, cache_read: i64) -> Value {
    anthropic_usage_full(input, output, 0, cache_read)
}

/// Chat `usage` -> (prompt, completion, cached, reasoning).
pub fn chat_usage_parts(u: Option<&Value>) -> (i64, i64, i64, i64) {
    let u = u.unwrap_or(&Value::Null);
    (
        u.pointer("/prompt_tokens")
            .and_then(|x| x.as_i64())
            .unwrap_or(0),
        u.pointer("/completion_tokens")
            .and_then(|x| x.as_i64())
            .unwrap_or(0),
        u.pointer("/prompt_tokens_details/cached_tokens")
            .and_then(|x| x.as_i64())
            .unwrap_or(0),
        u.pointer("/completion_tokens_details/reasoning_tokens")
            .and_then(|x| x.as_i64())
            .unwrap_or(0),
    )
}

// ---------------------------------------------------------------------------
// Content conversions
// ---------------------------------------------------------------------------

/// Log a content block the translation cannot represent. Silently skipping
/// multimodal payloads is very hard to debug from the client side.
pub fn warn_dropped_block(dir: &str, block_type: &str) {
    // Text-ish and empty blocks are expected no-ops, not data loss.
    if matches!(
        block_type,
        "" | "text" | "input_text" | "output_text" | "refusal"
    ) {
        return;
    }
    eprintln!("[gw] translate {dir}: dropped unsupported content block type `{block_type}`");
}

pub fn chunk(id: &str, model: &str, delta: Value, finish: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now_ts(),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
    })
}

/// `chat_to_responses_response` / `responses_to_openai` counterpart: turn a
/// canonical chat completion whose upstream reported `failed` back into a
/// Responses-shaped error object.
pub fn chat_failure(chat: &Value) -> Option<String> {
    chat.get("upstream_error")
        .and_then(|x| x.as_str())
        .map(String::from)
}

/// One in-flight Chat tool call, keyed by upstream `tool_calls[].index`.
#[derive(Default)]
pub struct PendingTool {
    pub id: String,
    pub name: String,
    pub args: String,
}

/// Slot identity for a streamed Chat tool call.
///
/// `Index` is the upstream's own `tool_calls[].index` and is authoritative
/// when present. Compatibility upstreams that omit `index` would otherwise
/// collapse every parallel call onto key 0 and concatenate their arguments
/// into one unparseable blob, so those fall back to `Synthetic`, one slot per
/// distinct `id`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ToolKey {
    Index(i64),
    Synthetic(usize),
}

/// Slot a tool-call delta belongs to.
///
/// - upstream sent `index` -> that slot, no ambiguity;
/// - no `index`, non-empty `id` we have not seen -> a brand new slot, so two
///   parallel calls stay separate;
/// - otherwise it is a continuation fragment -> the most recently opened
///   slot.
pub fn tool_slot(tc: &Value, tools: &[(ToolKey, PendingTool)]) -> Option<ToolKey> {
    if let Some(i) = tc.get("index").and_then(|x| x.as_i64()) {
        return Some(ToolKey::Index(i));
    }
    let id = tc.get("id").and_then(|x| x.as_str()).unwrap_or("");
    if !id.is_empty() {
        if tools.iter().any(|(_, t)| t.id == id) {
            return None; // known id: keep filling its existing slot
        }
        return Some(ToolKey::Synthetic(tools.len()));
    }
    tools.last().map(|(k, _)| *k)
}
