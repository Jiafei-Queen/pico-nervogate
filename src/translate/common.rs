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
// Reasoning provenance envelopes
// ---------------------------------------------------------------------------

/// Prefix for an Anthropic `thinking.signature` carried through the canonical
/// Chat shape.
///
/// The envelope is a provenance marker, not encryption: the gateway must be
/// able to tell an Anthropic signature from a Responses `encrypted_content`
/// blob before replaying either to an upstream that would reject it. It is
/// deterministic so the same history always produces the same bytes, which
/// Anthropic's prefix-binding check relies on.
pub const SIG_ANTHROPIC: &str = "nv1:a:";

/// Prefix for a Responses `encrypted_content` blob carried through the
/// canonical Chat shape.
pub const SIG_OPENAI: &str = "nv1:o:";

/// Prefix for an Anthropic `redacted_thinking` payload.
pub const SIG_REDACTED: &str = "nv1:r:";

pub fn wrap_anthropic_signature(sig: &str) -> String {
    if sig.starts_with("nv1:") {
        sig.to_string()
    } else {
        format!("{SIG_ANTHROPIC}{sig}")
    }
}

/// Wrap a Responses blob. Any existing `nv1:` prefix is kept so a client
/// echoing a gateway envelope does not get it double-wrapped.
pub fn wrap_openai_blob(blob: &str) -> String {
    if blob.starts_with("nv1:") {
        blob.to_string()
    } else {
        format!("{SIG_OPENAI}{blob}")
    }
}

pub fn wrap_redacted(data: &str) -> String {
    if data.starts_with(SIG_REDACTED) {
        data.to_string()
    } else {
        format!("{SIG_REDACTED}{data}")
    }
}

pub fn unwrap_anthropic_signature(sig: &str) -> Option<&str> {
    sig.strip_prefix(SIG_ANTHROPIC).filter(|s| !s.is_empty())
}

pub fn unwrap_openai_blob(blob: &str) -> Option<&str> {
    blob.strip_prefix(SIG_OPENAI).filter(|s| !s.is_empty())
}

pub fn unwrap_redacted(data: &str) -> Option<&str> {
    data.strip_prefix(SIG_REDACTED).filter(|s| !s.is_empty())
}

/// The value an Anthropic client should see in `thinking.signature`.
///
/// An `nv1:a:` envelope is stripped back to the upstream's raw signature.
/// An `nv1:o:` blob has no Anthropic equivalent, so it is passed through
/// verbatim as an opaque marker; replaying it to an Anthropic upstream is
/// prevented by the provenance check there.
pub fn anthropic_signature_for_wire(sig: &str) -> Option<&str> {
    if let Some(raw) = unwrap_anthropic_signature(sig) {
        return Some(raw);
    }
    sig.starts_with(SIG_OPENAI).then_some(sig)
}

/// A signature destined for a Responses client. Unknown provenance is
/// treated as an OpenAI blob: the client only echoes it, and the upstream
/// request path still refuses to send a foreign blob onward.
pub fn normalize_reasoning_signature(sig: &str) -> String {
    if sig.starts_with("nv1:") {
        sig.to_string()
    } else {
        wrap_openai_blob(sig)
    }
}

/// A canonical reasoning block, with the provenance envelope already applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningKind {
    Thinking,
    Redacted,
}

/// Classify a `reasoning_details` entry, accepting both the canonical
/// `thinking` / `redacted_thinking` names and OpenRouter's variants.
pub fn reasoning_kind(d: &Value) -> Option<ReasoningKind> {
    match d.get("type").and_then(|x| x.as_str()).unwrap_or("") {
        "thinking" | "reasoning.text" | "reasoning.summary" => Some(ReasoningKind::Thinking),
        "redacted_thinking" | "reasoning.encrypted" => Some(ReasoningKind::Redacted),
        _ => None,
    }
}

/// Visible text of a reasoning detail, across canonical and OpenRouter field
/// names. `reasoning.summary` stores it in `summary`.
pub fn reasoning_text(d: &Value) -> &str {
    d.get("thinking")
        .or_else(|| d.get("text"))
        .or_else(|| d.get("summary"))
        .and_then(|x| x.as_str())
        .unwrap_or("")
}

/// Opaque replay token of a reasoning detail. `redacted` blobs live in
/// `data`; ordinary blocks carry `signature`.
pub fn reasoning_signature(d: &Value) -> Option<&str> {
    d.get("signature")
        .and_then(|x| x.as_str())
        .or_else(|| d.get("data").and_then(|x| x.as_str()))
        .filter(|s| !s.is_empty())
}

/// The `id` an upstream attached to a reasoning block, if any.
pub fn reasoning_id(d: &Value) -> Option<&str> {
    d.get("id")
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
}

/// Normalize one `reasoning_details` entry to the canonical shape.
///
/// Chat upstreams disagree on the member names: this gateway and Anthropic
/// use `thinking` / `redacted_thinking`, while OpenRouter uses
/// `reasoning.text` / `reasoning.summary` / `reasoning.encrypted`. Both are
/// accepted here so a client that natively speaks either shape can replay.
/// A signature from a known provider family is wrapped with the matching
/// provenance prefix; an unknown signature is kept raw so neither upstream
/// family will replay it accidentally.
pub fn normalize_reasoning_detail(d: &Value) -> Option<Value> {
    let kind = reasoning_kind(d)?;
    let ty = d.get("type").and_then(|x| x.as_str()).unwrap_or("");
    let format = d.get("format").and_then(|x| x.as_str()).unwrap_or("");
    let id = reasoning_id(d);
    let mut out = match kind {
        ReasoningKind::Thinking => {
            let mut out = json!({
                "type": "thinking",
                "thinking": reasoning_text(d)
            });
            if let Some(sig) = reasoning_signature(d) {
                let wrapped = if sig.starts_with("nv1:") {
                    sig.to_string()
                } else if format.starts_with("anthropic") || ty == "thinking" {
                    wrap_anthropic_signature(sig)
                } else if format.contains("openai")
                    || format.contains("responses")
                    || format.contains("xai")
                {
                    wrap_openai_blob(sig)
                } else {
                    sig.to_string()
                };
                out["signature"] = json!(wrapped);
            }
            out
        }
        ReasoningKind::Redacted => {
            let data = d.get("data").and_then(|x| x.as_str()).unwrap_or("");
            if data.is_empty() {
                return None;
            }
            if ty == "reasoning.encrypted" {
                if format.contains("openai")
                    || format.contains("responses")
                    || format.contains("xai")
                {
                    // OpenRouter's generic encrypted entry may hold an
                    // OpenAI Responses blob; keep it replayable there.
                    json!({
                        "type": "thinking",
                        "thinking": "",
                        "signature": wrap_openai_blob(data)
                    })
                } else if format.is_empty() || format.starts_with("anthropic") {
                    json!({"type": "redacted_thinking", "data": data})
                } else {
                    // Google/Gemini or unknown provenance cannot be replayed
                    // to either family; dropping it is safer than a 400.
                    return None;
                }
            } else {
                json!({"type": "redacted_thinking", "data": data})
            }
        }
    };
    if let Some(id) = id {
        out["id"] = json!(id);
    }
    Some(out)
}

/// Render a canonical reasoning detail in OpenRouter's Chat Completions
/// schema. Used when forwarding to an upstream that understands the field.
pub fn reasoning_detail_to_openrouter(d: &Value) -> Option<Value> {
    let kind = reasoning_kind(d)?;
    let id = reasoning_id(d).unwrap_or("");
    let index = d.get("index").and_then(|x| x.as_i64());
    let out = match kind {
        ReasoningKind::Thinking => {
            let text = reasoning_text(d);
            let sig = d
                .get("signature")
                .and_then(|x| x.as_str())
                .filter(|s| !s.is_empty());
            if sig.is_none() && text.is_empty() {
                return None;
            }
            let mut out = match sig {
                Some(sig) if sig.starts_with(SIG_ANTHROPIC) => json!({
                    "type": "reasoning.text",
                    "text": text,
                    "signature": unwrap_anthropic_signature(sig).unwrap_or(""),
                    "format": "anthropic-claude-v1"
                }),
                Some(sig) if sig.starts_with(SIG_OPENAI) => json!({
                    "type": "reasoning.encrypted",
                    "data": unwrap_openai_blob(sig).unwrap_or(""),
                    "format": "openai-responses-v1"
                }),
                Some(sig) => json!({
                    "type": "reasoning.text",
                    "text": text,
                    "signature": sig,
                    "format": "unknown"
                }),
                None => json!({"type": "reasoning.summary", "summary": text}),
            };
            // Leave id out rather than inventing an identity the upstream
            // cannot resolve.
            if !id.is_empty() {
                out["id"] = json!(id);
            }
            if let Some(index) = index {
                out["index"] = json!(index);
            }
            out
        }
        ReasoningKind::Redacted => {
            let data = d.get("data").and_then(|x| x.as_str()).unwrap_or("");
            if data.is_empty() {
                return None;
            }
            let mut out = json!({
                "type": "reasoning.encrypted",
                "data": data,
                "format": "anthropic-claude-v1"
            });
            if !id.is_empty() {
                out["id"] = json!(id);
            }
            if let Some(index) = index {
                out["index"] = json!(index);
            }
            out
        }
    };
    Some(out)
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
