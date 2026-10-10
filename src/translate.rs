use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::config::ThinkingMode;

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
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
                    out.push_str(t);
                }
            }
            out
        }
        _ => String::new(),
    }
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
pub fn anthropic_usage(input: i64, output: i64, cache_read: i64) -> Value {
    json!({
        "input_tokens": input,
        "output_tokens": output,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cache_read
    })
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

fn openai_image_to_anthropic(p: &Value) -> Option<Value> {
    let url = p.pointer("/image_url/url").and_then(|x| x.as_str())?;
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',')?;
        let media = meta.split(';').next().unwrap_or("image/png");
        Some(json!({"type":"image","source":{"type":"base64","media_type":media,"data":data}}))
    } else {
        Some(json!({"type":"image","source":{"type":"url","url":url}}))
    }
}

fn anthropic_image_to_openai(b: &Value) -> Option<Value> {
    match b.pointer("/source/type").and_then(|x| x.as_str()) {
        Some("base64") => {
            let media = b
                .pointer("/source/media_type")
                .and_then(|x| x.as_str())
                .unwrap_or("image/png");
            let data = b.pointer("/source/data").and_then(|x| x.as_str())?;
            Some(
                json!({"type":"image_url","image_url":{"url": format!("data:{media};base64,{data}")}}),
            )
        }
        Some("url") => {
            let url = b.pointer("/source/url").and_then(|x| x.as_str())?;
            Some(json!({"type":"image_url","image_url":{"url": url}}))
        }
        _ => None,
    }
}

/// OpenAI user content -> Anthropic content blocks.
fn user_blocks(c: &Value) -> Vec<Value> {
    match c {
        Value::String(s) => {
            if s.is_empty() {
                vec![]
            } else {
                vec![json!({"type": "text", "text": s})]
            }
        }
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                let ty = p.get("type").and_then(|x| x.as_str()).unwrap_or("text");
                match ty {
                    "text" | "input_text" => p
                        .get("text")
                        .filter(|t| !t.as_str().unwrap_or("").is_empty())
                        .map(|t| json!({"type": "text", "text": t.clone()})),
                    "image_url" => openai_image_to_anthropic(p),
                    _ => None,
                }
            })
            .collect(),
        _ => vec![],
    }
}

fn finish_from_anthropic(stop: &str) -> &'static str {
    match stop {
        "max_tokens" | "max_output_tokens" => "length",
        "tool_use" => "tool_calls",
        "refusal" => "content_filter",
        // end_turn, stop_sequence, pause_turn, ...
        _ => "stop",
    }
}

/// Chat `finish_reason` -> Anthropic `stop_reason`.
fn anthropic_stop(finish: &str) -> &'static str {
    match finish {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    }
}

// ---------------------------------------------------------------------------
// OpenAI chat -> Anthropic Messages
// ---------------------------------------------------------------------------

/// Append blocks to the last message when it has the same role (Anthropic
/// wants alternating roles), else start a new message.
fn push_anthropic_blocks(msgs: &mut Vec<Value>, role: &str, blocks: Vec<Value>) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = msgs.last_mut() {
        if last.get("role").and_then(|r| r.as_str()) == Some(role) {
            if let Some(arr) = last.get_mut("content").and_then(|c| c.as_array_mut()) {
                arr.extend(blocks);
                return;
            }
        }
    }
    msgs.push(json!({"role": role, "content": blocks}));
}

/// Legacy `thinking.budget_tokens` for `ThinkingMode::Enabled`.
fn thinking_budget(effort: &str) -> i64 {
    match effort {
        "minimal" => 1024,
        "low" => 4096,
        "medium" => 8192,
        "high" => 16384,
        _ => 4096,
    }
}

pub fn openai_to_anthropic(req: &Value, thinking: ThinkingMode) -> Value {
    let mut out = serde_json::Map::new();

    let max_tokens = req
        .get("max_tokens")
        .or_else(|| req.get("max_completion_tokens"))
        .cloned()
        .unwrap_or(json!(4096));
    out.insert("max_tokens".into(), max_tokens);

    for k in ["temperature", "top_p"] {
        if let Some(v) = req.get(k) {
            out.insert(k.into(), v.clone());
        }
    }
    // `top_k` has no Chat Completions equivalent; it rides through the
    // reserved passthrough key.
    if let Some(k) = req.pointer("/x_nervogate/top_k") {
        out.insert("top_k".into(), k.clone());
    }
    if let Some(stop) = req.get("stop") {
        let arr = match stop {
            Value::Array(a) => a.clone(),
            Value::String(s) => vec![Value::String(s.clone())],
            _ => vec![],
        };
        if !arr.is_empty() {
            out.insert("stop_sequences".into(), Value::Array(arr));
        }
    }
    if let Some(md) = req.get("metadata") {
        if md.get("user_id").is_some() {
            out.insert("metadata".into(), md.clone());
        }
    }

    let mut system = String::new();
    let mut msgs: Vec<Value> = vec![];

    if let Some(list) = req.get("messages").and_then(|m| m.as_array()) {
        for m in list {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
            match role {
                "system" | "developer" => {
                    let t = content_text(m.get("content").unwrap_or(&Value::Null));
                    if t.is_empty() {
                        continue;
                    }
                    if !system.is_empty() {
                        system.push('\n');
                    }
                    system.push_str(&t);
                }
                "tool" => {
                    let id = m.get("tool_call_id").and_then(|x| x.as_str()).unwrap_or("");
                    let content = m.get("content").cloned().unwrap_or(json!(""));
                    let content = content_text(&content);
                    // Anthropic wants all tool_results for one assistant turn
                    // in a single user message, leading the content.
                    push_anthropic_blocks(
                        &mut msgs,
                        "user",
                        vec![json!({"type":"tool_result","tool_use_id":id,"content":content})],
                    );
                }
                "assistant" => {
                    let mut blocks: Vec<Value> = vec![];
                    if let Some(r) = m.get("reasoning_content").and_then(|x| x.as_str()) {
                        if !r.is_empty() {
                            blocks.push(json!({"type":"thinking","thinking":r,"signature":""}));
                        }
                    }
                    let text = content_text(m.get("content").unwrap_or(&Value::Null));
                    if !text.is_empty() {
                        blocks.push(json!({"type":"text","text":text}));
                    }
                    if let Some(tcs) = m.get("tool_calls").and_then(|t| t.as_array()) {
                        for tc in tcs {
                            let id = tc.get("id").and_then(|x| x.as_str()).unwrap_or("");
                            let name = tc
                                .pointer("/function/name")
                                .and_then(|x| x.as_str())
                                .unwrap_or("");
                            let args = tc
                                .pointer("/function/arguments")
                                .and_then(|x| x.as_str())
                                .unwrap_or("{}");
                            let input: Value = serde_json::from_str(args).unwrap_or(json!({}));
                            blocks
                                .push(json!({"type":"tool_use","id":id,"name":name,"input":input}));
                        }
                    }
                    push_anthropic_blocks(&mut msgs, "assistant", blocks);
                }
                _ => {
                    let c = m.get("content").cloned().unwrap_or(json!(""));
                    push_anthropic_blocks(&mut msgs, "user", user_blocks(&c));
                }
            }
        }
    }

    if !system.is_empty() {
        out.insert("system".into(), json!(system));
    }
    out.insert("messages".into(), Value::Array(msgs));

    let mut tools = req
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|tools| {
            tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.pointer("/function/name").and_then(|x| x.as_str()).unwrap_or(""),
                        "description": t.pointer("/function/description").cloned().unwrap_or(json!("")),
                        "input_schema": t.pointer("/function/parameters").cloned().unwrap_or(json!({"type":"object"}))
                    })
                })
                .collect::<Vec<Value>>()
        })
        .unwrap_or_default();

    if let Some(tc) = req.get("tool_choice") {
        match tc.as_str() {
            Some("auto") => {
                out.insert("tool_choice".into(), json!({"type":"auto"}));
            }
            Some("required") => {
                out.insert("tool_choice".into(), json!({"type":"any"}));
            }
            Some("none") => tools.clear(),
            _ => {
                if let Some(name) = tc.pointer("/function/name").and_then(|x| x.as_str()) {
                    out.insert("tool_choice".into(), json!({"type":"tool","name":name}));
                }
            }
        }
    }
    if !tools.is_empty() {
        out.insert("tools".into(), Value::Array(tools));
    }

    if let Some(e) = req.get("reasoning_effort").and_then(|x| x.as_str()) {
        if e != "none" {
            match thinking {
                // Newer upstream models reject `type: "enabled"` +
                // budget_tokens and require `type: "adaptive"` with
                // `output_config.effort`; the adaptive shape is accepted by
                // every Anthropic-protocol upstream tested so far.
                ThinkingMode::Adaptive => {
                    out.insert("thinking".into(), json!({"type":"adaptive"}));
                    out.insert("output_config".into(), json!({"effort": e}));
                }
                // Escape hatch for upstreams that only accept the legacy
                // `enabled` + budget_tokens shape.
                ThinkingMode::Enabled => {
                    out.insert(
                        "thinking".into(),
                        json!({"type":"enabled","budget_tokens": thinking_budget(e)}),
                    );
                }
            }
        }
    }

    Value::Object(out)
}

pub fn anthropic_to_openai(resp: &Value, model: &str) -> Value {
    let id = gen_id("chatcmpl-");
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: Vec<Value> = vec![];

    if let Some(blocks) = resp.get("content").and_then(|c| c.as_array()) {
        for b in blocks {
            match b.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(t) = b.get("text").and_then(|x| x.as_str()) {
                        text.push_str(t);
                    }
                }
                Some("thinking") => {
                    if let Some(t) = b.get("thinking").and_then(|x| x.as_str()) {
                        reasoning.push_str(t);
                    }
                }
                Some("tool_use") => {
                    let name = b.get("name").and_then(|x| x.as_str()).unwrap_or("");
                    let input = b.get("input").cloned().unwrap_or(json!({}));
                    let args = serde_json::to_string(&input).unwrap_or_else(|_| "{}".into());
                    let call_id = b
                        .get("id")
                        .and_then(|x| x.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| gen_id("call_"));
                    tool_calls.push(json!({
                        "id": call_id, "type": "function",
                        "function": {"name": name, "arguments": args}
                    }));
                }
                _ => {}
            }
        }
    }

    let stop = resp
        .get("stop_reason")
        .and_then(|s| s.as_str())
        .unwrap_or("end_turn");
    let finish = finish_from_anthropic(stop);
    let in_tok = resp
        .pointer("/usage/input_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let out_tok = resp
        .pointer("/usage/output_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let cached = resp
        .pointer("/usage/cache_read_input_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);

    let mut message = json!({"role": "assistant", "content": text});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
        message["content"] = Value::Null;
    }

    json!({
        "id": id,
        "object": "chat.completion",
        "created": now_ts(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": chat_usage(in_tok, out_tok, cached, 0)
    })
}

fn chunk(id: &str, model: &str, delta: Value, finish: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now_ts(),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
    })
}

/// Stateful translator for an Anthropic Messages SSE stream into OpenAI chunks.
pub struct AnthropicStream {
    id: String,
    model: String,
    role_sent: bool,
    tool_index: i64,
    block_ty: String,
    in_tok: i64,
    out_tok: i64,
    cached: i64,
    finish: Option<String>,
}

impl AnthropicStream {
    pub fn new(model: &str) -> Self {
        Self {
            id: gen_id("chatcmpl-"),
            model: model.to_string(),
            role_sent: false,
            tool_index: 0,
            block_ty: String::new(),
            in_tok: 0,
            out_tok: 0,
            cached: 0,
            finish: None,
        }
    }

    fn ensure_role(&mut self, out: &mut Vec<Value>) {
        if !self.role_sent {
            self.role_sent = true;
            out.push(chunk(
                &self.id,
                &self.model,
                json!({"role":"assistant"}),
                None,
            ));
        }
    }

    /// Handle one parsed Anthropic SSE `data:` payload. Returns OpenAI chunks.
    pub fn handle(&mut self, ev: &Value) -> Vec<Value> {
        let mut out = vec![];
        match ev.get("type").and_then(|t| t.as_str()).unwrap_or("") {
            "message_start" => {
                self.in_tok = ev
                    .pointer("/message/usage/input_tokens")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(0);
                self.cached = ev
                    .pointer("/message/usage/cache_read_input_tokens")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(0);
                self.ensure_role(&mut out);
            }
            "content_block_start" => {
                self.ensure_role(&mut out);
                let bt = ev
                    .pointer("/content_block/type")
                    .and_then(|x| x.as_str())
                    .unwrap_or("");
                self.block_ty = bt.to_string();
                if bt == "tool_use" {
                    let id = ev
                        .pointer("/content_block/id")
                        .and_then(|x| x.as_str())
                        .unwrap_or("");
                    let name = ev
                        .pointer("/content_block/name")
                        .and_then(|x| x.as_str())
                        .unwrap_or("");
                    out.push(chunk(
                        &self.id,
                        &self.model,
                        json!({"tool_calls":[{"index": self.tool_index, "id": id, "type": "function",
                            "function": {"name": name, "arguments": ""}}]}),
                        None,
                    ));
                }
            }
            "content_block_delta" => {
                self.ensure_role(&mut out);
                let dt = ev
                    .pointer("/delta/type")
                    .and_then(|x| x.as_str())
                    .unwrap_or("");
                if dt == "text_delta" {
                    if let Some(t) = ev.pointer("/delta/text").and_then(|x| x.as_str()) {
                        if !t.is_empty() {
                            out.push(chunk(&self.id, &self.model, json!({"content": t}), None));
                        }
                    }
                } else if dt == "thinking_delta" {
                    if let Some(t) = ev.pointer("/delta/thinking").and_then(|x| x.as_str()) {
                        if !t.is_empty() {
                            out.push(chunk(
                                &self.id,
                                &self.model,
                                json!({"reasoning_content": t}),
                                None,
                            ));
                        }
                    }
                } else if dt == "input_json_delta" {
                    if let Some(t) = ev.pointer("/delta/partial_json").and_then(|x| x.as_str()) {
                        out.push(chunk(
                            &self.id,
                            &self.model,
                            json!({"tool_calls":[{"index": self.tool_index, "function": {"arguments": t}}]}),
                            None,
                        ));
                    }
                }
            }
            "content_block_stop" => {
                // Only tool_use blocks consume a tool_calls index.
                if self.block_ty == "tool_use" {
                    self.tool_index += 1;
                }
                self.block_ty.clear();
            }
            "message_delta" => {
                if let Some(r) = ev.pointer("/delta/stop_reason").and_then(|x| x.as_str()) {
                    self.finish = Some(finish_from_anthropic(r).to_string());
                }
                if let Some(o) = ev.pointer("/usage/output_tokens").and_then(|x| x.as_i64()) {
                    self.out_tok = o;
                }
            }
            "message_stop" => {
                self.ensure_role(&mut out);
                let fr = self.finish.clone().unwrap_or_else(|| "stop".to_string());
                let mut c = chunk(&self.id, &self.model, json!({}), Some(&fr));
                c["usage"] = chat_usage(self.in_tok, self.out_tok, self.cached, 0);
                out.push(c);
            }
            _ => {}
        }
        out
    }
}

// ---------------------------------------------------------------------------
// OpenAI chat -> Responses API (egress)
// ---------------------------------------------------------------------------

fn responses_input_parts(c: &Value) -> Vec<Value> {
    let mut parts: Vec<Value> = vec![];
    match c {
        Value::String(s) => {
            if !s.is_empty() {
                parts.push(json!({"type": "input_text", "text": s}));
            }
        }
        Value::Array(arr) => {
            for p in arr {
                match p.get("type").and_then(|x| x.as_str()).unwrap_or("text") {
                    "text" | "input_text" => {
                        if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
                            if !t.is_empty() {
                                parts.push(json!({"type": "input_text", "text": t}));
                            }
                        }
                    }
                    "image_url" => {
                        if let Some(u) = p.pointer("/image_url/url").and_then(|x| x.as_str()) {
                            parts.push(json!({"type": "input_image", "image_url": u}));
                        }
                    }
                    "input_image" => parts.push(p.clone()),
                    _ => {}
                }
            }
        }
        _ => {}
    }
    parts
}

/// Assistant turns must use `output_text` (the upstream rejects `input_text`
/// with HTTP 400), and tool calls must be separate `function_call` items.
fn assistant_output_parts(c: &Value) -> Vec<Value> {
    responses_input_parts(c)
        .into_iter()
        .map(|mut p| {
            if p.get("type").and_then(|x| x.as_str()) == Some("input_text") {
                p["type"] = json!("output_text");
            }
            p
        })
        .collect()
}

pub fn chat_to_responses_request(req: &Value) -> Value {
    let mut input: Vec<Value> = vec![];
    let mut instructions = String::new();

    if let Some(list) = req.get("messages").and_then(|m| m.as_array()) {
        for m in list {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            match role {
                "system" | "developer" => {
                    let text = content_text(m.get("content").unwrap_or(&Value::Null));
                    if !text.is_empty() {
                        if !instructions.is_empty() {
                            instructions.push('\n');
                        }
                        instructions.push_str(&text);
                    }
                }
                "tool" => {
                    let call_id = m.get("tool_call_id").and_then(|x| x.as_str()).unwrap_or("");
                    let output = content_text(m.get("content").unwrap_or(&Value::Null));
                    input.push(json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output
                    }));
                }
                "assistant" => {
                    let parts = assistant_output_parts(m.get("content").unwrap_or(&Value::Null));
                    if !parts.is_empty() {
                        input.push(json!({"role": "assistant", "content": parts}));
                    }
                    if let Some(tcs) = m.get("tool_calls").and_then(|t| t.as_array()) {
                        for tc in tcs {
                            let call_id = tc.get("id").and_then(|x| x.as_str()).unwrap_or("");
                            let name = tc
                                .pointer("/function/name")
                                .and_then(|x| x.as_str())
                                .unwrap_or("");
                            let args = tc
                                .pointer("/function/arguments")
                                .and_then(|x| x.as_str())
                                .unwrap_or("{}");
                            input.push(json!({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": args
                            }));
                        }
                    }
                }
                _ => {
                    let parts = responses_input_parts(m.get("content").unwrap_or(&Value::Null));
                    if !parts.is_empty() {
                        input.push(json!({"role": role, "content": parts}));
                    }
                }
            }
        }
    }

    let mut out = serde_json::Map::new();
    out.insert("input".into(), Value::Array(input));
    if !instructions.is_empty() {
        out.insert("instructions".into(), json!(instructions));
    }
    if let Some(mt) = req
        .get("max_tokens")
        .or_else(|| req.get("max_completion_tokens"))
    {
        out.insert("max_output_tokens".into(), mt.clone());
    }
    for k in ["temperature", "top_p"] {
        if let Some(v) = req.get(k) {
            out.insert(k.into(), v.clone());
        }
    }
    if let Some(e) = req.get("reasoning_effort").and_then(|x| x.as_str()) {
        if e != "none" {
            out.insert("reasoning".into(), json!({"effort": e}));
        }
    }
    if let Some(rf) = req.get("response_format") {
        match rf.get("type").and_then(|x| x.as_str()) {
            Some("json_object") => {
                out.insert("text".into(), json!({"format": {"type": "json_object"}}));
            }
            Some("json_schema") => {
                let name = rf
                    .pointer("/json_schema/name")
                    .and_then(|x| x.as_str())
                    .unwrap_or("response");
                let schema = rf
                    .pointer("/json_schema/schema")
                    .cloned()
                    .unwrap_or(json!({"type": "object"}));
                let strict = rf
                    .pointer("/json_schema/strict")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(true);
                out.insert(
                    "text".into(),
                    json!({"format": {"type": "json_schema", "name": name, "schema": schema, "strict": strict}}),
                );
            }
            _ => {}
        }
    }
    if let Some(tools) = req.get("tools").and_then(|t| t.as_array()) {
        if !tools.is_empty() {
            let rt: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "name": t.pointer("/function/name").and_then(|x| x.as_str()).unwrap_or(""),
                        "description": t.pointer("/function/description").cloned().unwrap_or(json!("")),
                        "parameters": t.pointer("/function/parameters").cloned().unwrap_or(json!({"type":"object"}))
                    })
                })
                .collect();
            out.insert("tools".into(), Value::Array(rt));
        }
    }
    if let Some(tc) = req.get("tool_choice") {
        let mapped = match tc.as_str() {
            Some("auto") => Some(json!("auto")),
            Some("none") => Some(json!("none")),
            Some("required") => Some(json!("required")),
            _ => tc
                .pointer("/function/name")
                .and_then(|x| x.as_str())
                .map(|name| json!({"type": "function", "name": name})),
        };
        if let Some(m) = mapped {
            out.insert("tool_choice".into(), m);
        }
    }
    if let Some(p) = req.get("parallel_tool_calls") {
        out.insert("parallel_tool_calls".into(), p.clone());
    }
    if let Some(u) = req.get("user") {
        out.insert("user".into(), u.clone());
    }
    if let Some(md) = req.get("metadata") {
        out.insert("metadata".into(), md.clone());
    }
    // This gateway is stateless: never ask the upstream to keep a copy.
    out.insert("store".into(), json!(false));
    if let Some(inc) = req.pointer("/x_nervogate/include") {
        out.insert("include".into(), inc.clone());
    }
    Value::Object(out)
}

pub fn responses_to_openai(resp: &Value, model: &str) -> Value {
    let id = gen_id("chatcmpl-");
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: Vec<Value> = vec![];

    if let Some(output) = resp.get("output").and_then(|o| o.as_array()) {
        for item in output {
            match item.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                "message" => {
                    if let Some(content) = item.get("content").and_then(|c| c.as_array()) {
                        for c in content {
                            match c.get("type").and_then(|t| t.as_str()) {
                                Some("output_text") => {
                                    if let Some(t) = c.get("text").and_then(|x| x.as_str()) {
                                        text.push_str(t);
                                    }
                                }
                                Some("refusal") => {
                                    if let Some(t) = c.get("refusal").and_then(|x| x.as_str()) {
                                        text.push_str(t);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                "reasoning" => {
                    for s in item
                        .get("summary")
                        .and_then(|x| x.as_array())
                        .into_iter()
                        .flatten()
                    {
                        if let Some(t) = s.get("text").and_then(|x| x.as_str()) {
                            reasoning.push_str(t);
                        }
                    }
                    for s in item
                        .get("content")
                        .and_then(|x| x.as_array())
                        .into_iter()
                        .flatten()
                    {
                        if let Some(t) = s.get("text").and_then(|x| x.as_str()) {
                            reasoning.push_str(t);
                        }
                    }
                }
                "function_call" => {
                    let call_id = item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(|x| x.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| gen_id("call_"));
                    tool_calls.push(json!({
                        "id": call_id, "type": "function",
                        "function": {
                            "name": item.get("name").cloned().unwrap_or(json!("")),
                            "arguments": item.get("arguments").cloned().unwrap_or(json!("{}"))
                        }
                    }));
                }
                _ => {}
            }
        }
    }

    let status = resp
        .get("status")
        .and_then(|s| s.as_str())
        .unwrap_or("completed");
    let finish = if !tool_calls.is_empty() {
        "tool_calls"
    } else if status == "incomplete" {
        "length"
    } else {
        "stop"
    };
    let in_tok = resp
        .pointer("/usage/input_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let out_tok = resp
        .pointer("/usage/output_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let cached = resp
        .pointer("/usage/input_tokens_details/cached_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let reasoning_tok = resp
        .pointer("/usage/output_tokens_details/reasoning_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);

    let mut message = json!({"role": "assistant", "content": text});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
        message["content"] = Value::Null;
    }

    json!({
        "id": id,
        "object": "chat.completion",
        "created": now_ts(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": chat_usage(in_tok, out_tok, cached, reasoning_tok)
    })
}

/// Stateful translator for a Responses API SSE stream into OpenAI chunks.
pub struct ResponsesStream {
    id: String,
    model: String,
    role_sent: bool,
    tool_index: i64,
    cur_call: Option<(String, String)>, // (call_id, name) for streaming function args
    in_tok: i64,
    out_tok: i64,
    cached: i64,
    reasoning_tok: i64,
    finish: Option<String>,
}

impl ResponsesStream {
    pub fn new(model: &str) -> Self {
        Self {
            id: gen_id("chatcmpl-"),
            model: model.to_string(),
            role_sent: false,
            tool_index: 0,
            cur_call: None,
            in_tok: 0,
            out_tok: 0,
            cached: 0,
            reasoning_tok: 0,
            finish: None,
        }
    }

    fn ensure_role(&mut self, out: &mut Vec<Value>) {
        if !self.role_sent {
            self.role_sent = true;
            out.push(chunk(
                &self.id,
                &self.model,
                json!({"role":"assistant"}),
                None,
            ));
        }
    }

    pub fn handle(&mut self, ev: &Value) -> Vec<Value> {
        let mut out = vec![];
        match ev.get("type").and_then(|t| t.as_str()).unwrap_or("") {
            "response.output_item.added" => {
                self.ensure_role(&mut out);
                let item = ev.get("item").cloned().unwrap_or(json!({}));
                if item.get("type").and_then(|t| t.as_str()) == Some("function_call") {
                    let call_id = item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = item
                        .get("name")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                    self.cur_call = Some((call_id.clone(), name.clone()));
                    out.push(chunk(
                        &self.id,
                        &self.model,
                        json!({"tool_calls":[{"index": self.tool_index, "id": call_id, "type": "function",
                            "function": {"name": name, "arguments": ""}}]}),
                        None,
                    ));
                }
            }
            "response.output_text.delta" => {
                self.ensure_role(&mut out);
                if let Some(t) = ev.get("delta").and_then(|x| x.as_str()) {
                    if !t.is_empty() {
                        out.push(chunk(&self.id, &self.model, json!({"content": t}), None));
                    }
                }
            }
            "response.refusal.delta" => {
                self.ensure_role(&mut out);
                if let Some(t) = ev.get("delta").and_then(|x| x.as_str()) {
                    if !t.is_empty() {
                        out.push(chunk(&self.id, &self.model, json!({"content": t}), None));
                    }
                }
            }
            "response.reasoning_summary_text.delta" => {
                self.ensure_role(&mut out);
                if let Some(t) = ev.get("delta").and_then(|x| x.as_str()) {
                    if !t.is_empty() {
                        out.push(chunk(
                            &self.id,
                            &self.model,
                            json!({"reasoning_content": t}),
                            None,
                        ));
                    }
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(t) = ev.get("delta").and_then(|x| x.as_str()) {
                    out.push(chunk(
                        &self.id,
                        &self.model,
                        json!({"tool_calls":[{"index": self.tool_index, "function": {"arguments": t}}]}),
                        None,
                    ));
                }
            }
            "response.output_item.done" => {
                let item = ev.get("item").cloned().unwrap_or(json!({}));
                if item.get("type").and_then(|t| t.as_str()) == Some("function_call") {
                    self.tool_index += 1;
                    self.cur_call = None;
                }
            }
            "response.completed" | "response.incomplete" | "response.failed" => {
                self.ensure_role(&mut out);
                self.in_tok = ev
                    .pointer("/response/usage/input_tokens")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(0);
                self.out_tok = ev
                    .pointer("/response/usage/output_tokens")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(0);
                self.cached = ev
                    .pointer("/response/usage/input_tokens_details/cached_tokens")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(0);
                self.reasoning_tok = ev
                    .pointer("/response/usage/output_tokens_details/reasoning_tokens")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(0);
                let fr = match ev.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                    "response.incomplete" => "length",
                    "response.failed" => "stop",
                    _ => {
                        if self.tool_index > 0 {
                            "tool_calls"
                        } else {
                            "stop"
                        }
                    }
                };
                self.finish = Some(fr.to_string());
                let mut c = chunk(&self.id, &self.model, json!({}), Some(fr));
                c["usage"] = chat_usage(self.in_tok, self.out_tok, self.cached, self.reasoning_tok);
                out.push(c);
            }
            _ => {}
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Responses API ingress -> OpenAI chat request
// ---------------------------------------------------------------------------

/// Stateless gateways cannot serve stateful Responses features. Reject the
/// dangerous ones explicitly instead of silently dropping conversation state.
pub fn responses_stateful_error(req: &Value) -> Option<String> {
    if req
        .get("previous_response_id")
        .and_then(|x| x.as_str())
        .is_some_and(|s| !s.is_empty())
    {
        return Some(
            "previous_response_id is not supported: this gateway is stateless; \
             put the prior conversation into `input`"
                .to_string(),
        );
    }
    if req
        .get("background")
        .and_then(|x| x.as_bool())
        .unwrap_or(false)
    {
        return Some(
            "background is not supported: this gateway is stateless and has no \
             GET /v1/responses/{id} polling endpoint"
                .to_string(),
        );
    }
    None
}

fn responses_content_to_chat_parts(c: &Value) -> Vec<Value> {
    let mut parts: Vec<Value> = vec![];
    match c {
        Value::String(s) => {
            if !s.is_empty() {
                parts.push(json!({"type": "text", "text": s}));
            }
        }
        Value::Array(arr) => {
            for p in arr {
                match p.get("type").and_then(|x| x.as_str()).unwrap_or("text") {
                    "input_text" | "output_text" | "text" => {
                        if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
                            if !t.is_empty() {
                                parts.push(json!({"type": "text", "text": t}));
                            }
                        }
                    }
                    "refusal" => {
                        if let Some(t) = p.get("refusal").and_then(|x| x.as_str()) {
                            if !t.is_empty() {
                                parts.push(json!({"type": "text", "text": t}));
                            }
                        }
                    }
                    "input_image" => {
                        if let Some(u) = p.get("image_url").and_then(|x| x.as_str()) {
                            parts.push(json!({"type": "image_url", "image_url": {"url": u}}));
                        }
                    }
                    "image_url" => parts.push(p.clone()),
                    _ => {}
                }
            }
        }
        _ => {}
    }
    parts
}

/// Append a tool call to the trailing assistant message when possible, else
/// start a new one (Responses `function_call` items follow the message item
/// of the same turn and map onto the same Chat message).
fn push_chat_tool_call(messages: &mut Vec<Value>, id: &str, name: &str, args: &str) {
    let tc = json!({"id": id, "type": "function", "function": {"name": name, "arguments": args}});
    if let Some(last) = messages.last_mut() {
        if last.get("role").and_then(|r| r.as_str()) == Some("assistant") {
            match last.get_mut("tool_calls") {
                Some(Value::Array(arr)) => {
                    arr.push(tc);
                    return;
                }
                None => {
                    last["tool_calls"] = json!([tc]);
                    return;
                }
                _ => {}
            }
        }
    }
    messages.push(json!({"role": "assistant", "content": Value::Null, "tool_calls": [tc]}));
}

pub fn responses_to_chat_request(req: &Value) -> Value {
    let mut messages: Vec<Value> = vec![];
    let mut pending_reasoning = String::new();

    if let Some(inst) = req.get("instructions").and_then(|x| x.as_str()) {
        if !inst.is_empty() {
            messages.push(json!({"role": "system", "content": inst}));
        }
    }

    let flush_reasoning = |messages: &mut Vec<Value>, pending: &mut String| {
        if !pending.is_empty() {
            messages.push(
                json!({"role": "assistant", "content": "", "reasoning_content": pending.clone()}),
            );
            pending.clear();
        }
    };

    match req.get("input") {
        Some(Value::String(s)) => {
            flush_reasoning(&mut messages, &mut pending_reasoning);
            messages.push(json!({"role": "user", "content": s}));
        }
        Some(Value::Array(items)) => {
            for it in items {
                match it.get("type").and_then(|x| x.as_str()).unwrap_or("message") {
                    "message" => {
                        let role = it.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                        let c = it.get("content").unwrap_or(&Value::Null);
                        if role == "assistant" {
                            let mut msg = json!({"role": "assistant", "content": content_text(c)});
                            if !pending_reasoning.is_empty() {
                                msg["reasoning_content"] = json!(pending_reasoning.clone());
                                pending_reasoning.clear();
                            }
                            messages.push(msg);
                        } else if role == "system" || role == "developer" {
                            let text = content_text(c);
                            if !text.is_empty() {
                                messages.push(json!({"role": "system", "content": text}));
                            }
                        } else {
                            flush_reasoning(&mut messages, &mut pending_reasoning);
                            let parts = responses_content_to_chat_parts(c);
                            if !parts.is_empty() {
                                messages.push(json!({"role": "user", "content": parts}));
                            }
                        }
                    }
                    "function_call" => {
                        let call_id = it
                            .get("call_id")
                            .or_else(|| it.get("id"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        let name = it.get("name").and_then(|x| x.as_str()).unwrap_or("");
                        let args = it.get("arguments").and_then(|x| x.as_str()).unwrap_or("{}");
                        push_chat_tool_call(&mut messages, call_id, name, args);
                    }
                    "function_call_output" => {
                        let call_id = it.get("call_id").and_then(|x| x.as_str()).unwrap_or("");
                        let output = it.get("output").cloned().unwrap_or(json!(""));
                        messages.push(json!({
                            "role": "tool", "tool_call_id": call_id,
                            "content": content_text(&output)
                        }));
                    }
                    "reasoning" => {
                        for s in it
                            .get("summary")
                            .and_then(|x| x.as_array())
                            .into_iter()
                            .flatten()
                        {
                            if let Some(t) = s.get("text").and_then(|x| x.as_str()) {
                                pending_reasoning.push_str(t);
                            }
                        }
                        for s in it
                            .get("content")
                            .and_then(|x| x.as_array())
                            .into_iter()
                            .flatten()
                        {
                            if let Some(t) = s.get("text").and_then(|x| x.as_str()) {
                                pending_reasoning.push_str(t);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    flush_reasoning(&mut messages, &mut pending_reasoning);

    let mut out = json!({
        "model": req.get("model").cloned().unwrap_or(json!("")),
        "messages": messages
    });
    if let Some(mt) = req.get("max_output_tokens") {
        out["max_tokens"] = mt.clone();
    }
    for k in ["temperature", "top_p"] {
        if let Some(v) = req.get(k) {
            out[k] = v.clone();
        }
    }
    if let Some(rf) = req.pointer("/text/format") {
        match rf.get("type").and_then(|x| x.as_str()) {
            Some("json_object") => out["response_format"] = json!({"type": "json_object"}),
            Some("json_schema") => {
                let name = rf
                    .get("name")
                    .and_then(|x| x.as_str())
                    .unwrap_or("response");
                let schema = rf
                    .get("schema")
                    .cloned()
                    .unwrap_or(json!({"type": "object"}));
                let strict = rf.get("strict").and_then(|x| x.as_bool()).unwrap_or(true);
                out["response_format"] = json!({
                    "type": "json_schema",
                    "json_schema": {"name": name, "schema": schema, "strict": strict}
                });
            }
            _ => {}
        }
    }
    if let Some(tools) = req.get("tools").and_then(|t| t.as_array()) {
        if !tools.is_empty() {
            let ot: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({"type":"function","function":{
                        "name": t.get("name").cloned().unwrap_or(json!("")),
                        "description": t.get("description").cloned().unwrap_or(json!("")),
                        "parameters": t.get("parameters").cloned().unwrap_or(json!({"type":"object"}))
                    }})
                })
                .collect();
            out["tools"] = Value::Array(ot);
        }
    }
    if let Some(tc) = req.get("tool_choice") {
        let mapped = match tc.as_str() {
            Some(s @ ("auto" | "none" | "required")) => Some(json!(s)),
            _ => tc
                .get("name")
                .and_then(|x| x.as_str())
                .map(|name| json!({"type": "function", "function": {"name": name}})),
        };
        if let Some(m) = mapped {
            out["tool_choice"] = m;
        }
    }
    if let Some(p) = req.get("parallel_tool_calls") {
        out["parallel_tool_calls"] = p.clone();
    }
    if let Some(e) = req
        .get("reasoning")
        .and_then(|x| x.get("effort"))
        .and_then(|x| x.as_str())
    {
        out["reasoning_effort"] = json!(e);
    }
    if let Some(u) = req.get("user") {
        out["user"] = u.clone();
    }
    if let Some(md) = req.get("metadata") {
        out["metadata"] = md.clone();
    }
    if let Some(inc) = req.get("include") {
        out["x_nervogate"] = json!({"include": inc.clone()});
    }
    out
}

/// Chat completion -> Responses API response object.
pub fn chat_to_responses_response(chat: &Value, model: &str) -> Value {
    let msg = chat
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or(json!({}));
    let mut output: Vec<Value> = vec![];

    let reasoning = msg
        .get("reasoning_content")
        .and_then(|x| x.as_str())
        .unwrap_or("");
    if !reasoning.is_empty() {
        output.push(json!({
            "id": gen_id("rs_"),
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": reasoning}]
        }));
    }

    let text = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
    if !text.is_empty() {
        output.push(json!({
            "type": "message",
            "id": gen_id("msg_"),
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]
        }));
    }

    if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tcs {
            output.push(json!({
                "type": "function_call",
                "id": gen_id("fc_"),
                "call_id": tc.get("id").cloned().unwrap_or_else(|| json!(gen_id("call_"))),
                "name": tc.pointer("/function/name").cloned().unwrap_or(json!("")),
                "arguments": tc.pointer("/function/arguments").cloned().unwrap_or(json!("{}")),
                "status": "completed"
            }));
        }
    }

    let finish = chat
        .pointer("/choices/0/finish_reason")
        .and_then(|x| x.as_str())
        .unwrap_or("stop");
    let (status, incomplete) = match finish {
        "length" => ("incomplete", Some(json!({"reason": "max_output_tokens"}))),
        _ => ("completed", None),
    };
    let (p, c, cached, reasoning_tok) = chat_usage_parts(chat.get("usage"));

    json!({
        "id": gen_id("resp_"),
        "object": "response",
        "created_at": now_ts(),
        "model": model,
        "status": status,
        "incomplete_details": incomplete.unwrap_or(Value::Null),
        "output": output,
        "usage": responses_usage(p, c, cached, reasoning_tok),
        "error": Value::Null
    })
}

// ---------------------------------------------------------------------------
// Chat chunks -> Responses API SSE stream (egress)
// ---------------------------------------------------------------------------

enum OpenOut {
    Reasoning {
        index: usize,
        item_id: String,
        text: String,
    },
    Message {
        index: usize,
        item_id: String,
        text: String,
    },
    Tool {
        index: usize,
        item_id: String,
        call_id: String,
        name: String,
        args: String,
    },
}

/// Stateful translator for OpenAI chat chunks into Responses `response.*` SSE
/// events. Feed chunks with [`ChatToResponsesStream::handle`]; the terminal
/// `response.completed` / `response.incomplete` event is emitted when the
/// chunk carrying `finish_reason` arrives.
pub struct ChatToResponsesStream {
    resp_id: String,
    model: String,
    created: i64,
    started: bool,
    next_index: usize,
    open: Option<OpenOut>,
    items: Vec<Value>,
}

impl ChatToResponsesStream {
    pub fn new(model: &str) -> Self {
        Self {
            resp_id: gen_id("resp_"),
            model: model.to_string(),
            created: now_ts(),
            started: false,
            next_index: 0,
            open: None,
            items: vec![],
        }
    }

    fn skeleton(&self, status: &str) -> Value {
        json!({
            "id": self.resp_id,
            "object": "response",
            "created_at": self.created,
            "model": self.model,
            "status": status,
            "output": [],
            "usage": Value::Null,
            "error": Value::Null,
            "incomplete_details": Value::Null
        })
    }

    fn close_open(&mut self, out: &mut Vec<SseEvent>) {
        let open = match self.open.take() {
            Some(o) => o,
            None => return,
        };
        match open {
            OpenOut::Reasoning {
                index,
                item_id,
                text,
            } => {
                out.push(SseEvent::data(json!({
                    "type": "response.reasoning_summary_text.done",
                    "item_id": item_id, "output_index": index, "summary_index": 0, "text": text
                })));
                out.push(SseEvent::data(json!({
                    "type": "response.output_item.done", "output_index": index,
                    "item": {"id": item_id, "type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}
                })));
                self.items.push(json!({
                    "id": item_id, "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": text}]
                }));
            }
            OpenOut::Message {
                index,
                item_id,
                text,
            } => {
                out.push(SseEvent::data(json!({
                    "type": "response.output_text.done",
                    "item_id": item_id, "output_index": index, "content_index": 0,
                    "text": text, "logprobs": []
                })));
                out.push(SseEvent::data(json!({
                    "type": "response.content_part.done",
                    "item_id": item_id, "output_index": index, "content_index": 0,
                    "part": {"type": "output_text", "text": text, "annotations": []}
                })));
                out.push(SseEvent::data(json!({
                    "type": "response.output_item.done", "output_index": index,
                    "item": {
                        "id": item_id, "type": "message", "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]
                    }
                })));
                self.items.push(json!({
                    "id": item_id, "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]
                }));
            }
            OpenOut::Tool {
                index,
                item_id,
                call_id,
                name,
                args,
            } => {
                out.push(SseEvent::data(json!({
                    "type": "response.function_call_arguments.done",
                    "item_id": item_id, "output_index": index,
                    "name": name, "arguments": args
                })));
                out.push(SseEvent::data(json!({
                    "type": "response.output_item.done", "output_index": index,
                    "item": {
                        "id": item_id, "type": "function_call", "status": "completed",
                        "call_id": call_id, "name": name, "arguments": args
                    }
                })));
                self.items.push(json!({
                    "id": item_id, "type": "function_call", "status": "completed",
                    "call_id": call_id, "name": name, "arguments": args
                }));
            }
        }
    }

    /// Handle one canonical OpenAI chat chunk. Returns Responses SSE events.
    pub fn handle(&mut self, ch: &Value) -> Vec<SseEvent> {
        let mut out = vec![];
        if !self.started {
            self.started = true;
            out.push(SseEvent::data(
                json!({"type": "response.created", "response": self.skeleton("in_progress")}),
            ));
            out.push(SseEvent::data(
                json!({"type": "response.in_progress", "response": self.skeleton("in_progress")}),
            ));
        }

        let choice = ch.pointer("/choices/0").cloned().unwrap_or(json!({}));
        let delta = choice.get("delta").cloned().unwrap_or(json!({}));

        // Reasoning deltas open a reasoning output item.
        if let Some(t) = delta.get("reasoning_content").and_then(|x| x.as_str()) {
            if !t.is_empty() {
                let needs_open = !matches!(self.open, Some(OpenOut::Reasoning { .. }));
                if needs_open {
                    self.close_open(&mut out);
                    let index = self.next_index;
                    self.next_index += 1;
                    let item_id = gen_id("rs_");
                    out.push(SseEvent::data(json!({
                        "type": "response.output_item.added", "output_index": index,
                        "item": {"id": item_id, "type": "reasoning", "summary": []}
                    })));
                    self.open = Some(OpenOut::Reasoning {
                        index,
                        item_id,
                        text: String::new(),
                    });
                }
                if let Some(OpenOut::Reasoning {
                    index,
                    item_id,
                    text,
                }) = &mut self.open
                {
                    text.push_str(t);
                    out.push(SseEvent::data(json!({
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": item_id, "output_index": index, "summary_index": 0, "delta": t
                    })));
                }
            }
        }

        // Text deltas open a message output item.
        if let Some(t) = delta.get("content").and_then(|x| x.as_str()) {
            if !t.is_empty() {
                let needs_open = !matches!(self.open, Some(OpenOut::Message { .. }));
                if needs_open {
                    self.close_open(&mut out);
                    let index = self.next_index;
                    self.next_index += 1;
                    let item_id = gen_id("msg_");
                    out.push(SseEvent::data(json!({
                        "type": "response.output_item.added", "output_index": index,
                        "item": {
                            "id": item_id, "type": "message", "role": "assistant",
                            "status": "in_progress", "content": []
                        }
                    })));
                    out.push(SseEvent::data(json!({
                        "type": "response.content_part.added",
                        "item_id": item_id, "output_index": index, "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []}
                    })));
                    self.open = Some(OpenOut::Message {
                        index,
                        item_id,
                        text: String::new(),
                    });
                }
                if let Some(OpenOut::Message {
                    index,
                    item_id,
                    text,
                }) = &mut self.open
                {
                    text.push_str(t);
                    out.push(SseEvent::data(json!({
                        "type": "response.output_text.delta",
                        "item_id": item_id, "output_index": index, "content_index": 0,
                        "delta": t, "logprobs": []
                    })));
                }
            }
        }

        // Tool call deltas open function_call output items.
        if let Some(tcs) = delta.get("tool_calls").and_then(|x| x.as_array()) {
            for tc in tcs {
                let new_call = tc
                    .get("id")
                    .and_then(|x| x.as_str())
                    .is_some_and(|s| !s.is_empty())
                    || tc
                        .pointer("/function/name")
                        .and_then(|x| x.as_str())
                        .is_some_and(|s| !s.is_empty());
                if new_call {
                    self.close_open(&mut out);
                    let index = self.next_index;
                    self.next_index += 1;
                    let item_id = gen_id("fc_");
                    let call_id = tc
                        .get("id")
                        .and_then(|x| x.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| gen_id("call_"));
                    let name = tc
                        .pointer("/function/name")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                    out.push(SseEvent::data(json!({
                        "type": "response.output_item.added", "output_index": index,
                        "item": {
                            "id": item_id, "type": "function_call", "status": "in_progress",
                            "call_id": call_id, "name": name, "arguments": ""
                        }
                    })));
                    self.open = Some(OpenOut::Tool {
                        index,
                        item_id,
                        call_id,
                        name,
                        args: String::new(),
                    });
                }
                if let Some(args) = tc.pointer("/function/arguments").and_then(|x| x.as_str()) {
                    if !args.is_empty() {
                        if let Some(OpenOut::Tool {
                            index,
                            item_id,
                            args: acc,
                            ..
                        }) = &mut self.open
                        {
                            acc.push_str(args);
                            out.push(SseEvent::data(json!({
                                "type": "response.function_call_arguments.delta",
                                "item_id": item_id, "output_index": index, "delta": args
                            })));
                        }
                    }
                }
            }
        }

        // Terminal chunk carries finish_reason (and usage).
        if let Some(fr) = choice.get("finish_reason").and_then(|x| x.as_str()) {
            self.close_open(&mut out);
            let (p, c, cached, reasoning_tok) = chat_usage_parts(ch.get("usage"));
            let (status, incomplete) = match fr {
                "length" => ("incomplete", Some(json!({"reason": "max_output_tokens"}))),
                _ => ("completed", None),
            };
            let response = json!({
                "id": self.resp_id,
                "object": "response",
                "created_at": self.created,
                "model": self.model,
                "status": status,
                "incomplete_details": incomplete.unwrap_or(Value::Null),
                "output": self.items,
                "usage": responses_usage(p, c, cached, reasoning_tok),
                "error": Value::Null
            });
            let ty = if status == "incomplete" {
                "response.incomplete"
            } else {
                "response.completed"
            };
            out.push(SseEvent::data(json!({"type": ty, "response": response})));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Anthropic Messages ingress -> OpenAI chat request
// ---------------------------------------------------------------------------

pub fn anthropic_to_chat_request(req: &Value) -> Value {
    let mut messages: Vec<Value> = vec![];

    match req.get("system") {
        Some(Value::String(s)) => {
            if !s.is_empty() {
                messages.push(json!({"role": "system", "content": s}));
            }
        }
        Some(Value::Array(blocks)) => {
            let mut text = String::new();
            for b in blocks {
                if let Some(t) = b.get("text").and_then(|x| x.as_str()) {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(t);
                }
            }
            if !text.is_empty() {
                messages.push(json!({"role": "system", "content": text}));
            }
        }
        _ => {}
    }

    if let Some(list) = req.get("messages").and_then(|m| m.as_array()) {
        for m in list {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            match m.get("content") {
                Some(Value::String(s)) => {
                    messages.push(json!({"role": role, "content": s}));
                }
                Some(Value::Array(blocks)) => {
                    let mut parts: Vec<Value> = vec![];
                    let mut tool_calls: Vec<Value> = vec![];
                    let mut reasoning = String::new();
                    for b in blocks {
                        match b.get("type").and_then(|t| t.as_str()).unwrap_or("text") {
                            "text" => {
                                if let Some(t) = b.get("text").and_then(|x| x.as_str()) {
                                    if !t.is_empty() {
                                        parts.push(json!({"type": "text", "text": t}));
                                    }
                                }
                            }
                            "image" => {
                                if let Some(img) = anthropic_image_to_openai(b) {
                                    parts.push(img);
                                }
                            }
                            "tool_use" => {
                                let id = b.get("id").and_then(|x| x.as_str()).unwrap_or("");
                                let name = b.get("name").and_then(|x| x.as_str()).unwrap_or("");
                                let input = b.get("input").cloned().unwrap_or(json!({}));
                                let args =
                                    serde_json::to_string(&input).unwrap_or_else(|_| "{}".into());
                                tool_calls.push(json!({
                                    "id": id, "type": "function",
                                    "function": {"name": name, "arguments": args}
                                }));
                            }
                            "tool_result" => {
                                // tool_result answers an assistant tool_use and
                                // becomes a Chat `tool` message of its own.
                                let id =
                                    b.get("tool_use_id").and_then(|x| x.as_str()).unwrap_or("");
                                let content = b.get("content").cloned().unwrap_or(json!(""));
                                messages.push(json!({
                                    "role": "tool", "tool_call_id": id,
                                    "content": content_text(&content)
                                }));
                            }
                            "thinking" => {
                                if let Some(t) = b.get("thinking").and_then(|x| x.as_str()) {
                                    reasoning.push_str(t);
                                }
                            }
                            _ => {}
                        }
                    }
                    if !tool_calls.is_empty() {
                        let mut msg = json!({"role": "assistant", "content": Value::Null});
                        if !reasoning.is_empty() {
                            msg["reasoning_content"] = json!(reasoning);
                        }
                        if !parts.is_empty() {
                            // text alongside tool calls belongs in `content`
                            msg["content"] = json!(content_text(&Value::Array(parts.clone())));
                        }
                        msg["tool_calls"] = Value::Array(tool_calls);
                        messages.push(msg);
                    } else if !parts.is_empty() || !reasoning.is_empty() {
                        // Plain-text stays a string (best Chat compatibility);
                        // structured parts (images) keep their array form.
                        let content = if parts
                            .iter()
                            .all(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
                        {
                            json!(content_text(&Value::Array(parts.clone())))
                        } else {
                            Value::Array(parts.clone())
                        };
                        let mut msg = json!({"role": role, "content": content});
                        if !reasoning.is_empty() {
                            msg["reasoning_content"] = json!(reasoning);
                        }
                        messages.push(msg);
                    }
                }
                _ => {}
            }
        }
    }

    let mut out = json!({
        "model": req.get("model").cloned().unwrap_or(json!("")),
        "messages": messages
    });
    if let Some(mt) = req.get("max_tokens") {
        out["max_tokens"] = mt.clone();
    }
    for k in ["temperature", "top_p"] {
        if let Some(v) = req.get(k) {
            out[k] = v.clone();
        }
    }
    if let Some(k) = req.get("top_k") {
        out["x_nervogate"] = json!({"top_k": k.clone()});
    }
    if let Some(md) = req.get("metadata") {
        out["metadata"] = md.clone();
    }
    if let Some(seq) = req.get("stop_sequences") {
        out["stop"] = seq.clone();
    }
    if let Some(tools) = req.get("tools").and_then(|t| t.as_array()) {
        if !tools.is_empty() {
            let ot: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({"type":"function","function":{
                        "name": t.get("name").cloned().unwrap_or(json!("")),
                        "description": t.get("description").cloned().unwrap_or(json!("")),
                        "parameters": t.get("input_schema").cloned().unwrap_or(json!({"type":"object"}))
                    }})
                })
                .collect();
            out["tools"] = Value::Array(ot);
        }
    }
    if let Some(tc) = req.get("tool_choice") {
        let mapped = match tc.get("type").and_then(|x| x.as_str()).unwrap_or("auto") {
            "auto" => Some(json!("auto")),
            "any" => Some(json!("required")),
            "tool" => tc
                .get("name")
                .and_then(|x| x.as_str())
                .map(|name| json!({"type": "function", "function": {"name": name}})),
            "none" => Some(json!("none")),
            _ => None,
        };
        if let Some(m) = mapped {
            out["tool_choice"] = m;
        }
    }
    if let Some(e) = req
        .pointer("/output_config/effort")
        .and_then(|x| x.as_str())
    {
        // Adaptive round-trip: effort rides through `output_config`.
        out["reasoning_effort"] = json!(e);
    } else if let Some(budget) = req
        .pointer("/thinking/budget_tokens")
        .and_then(|x| x.as_i64())
    {
        let effort = match budget {
            0..=2047 => "low",
            2048..=8191 => "medium",
            _ => "high",
        };
        out["reasoning_effort"] = json!(effort);
    } else if req.pointer("/thinking/type").and_then(|x| x.as_str()) == Some("adaptive") {
        // Bare adaptive (no effort, no budget): keep reasoning on for
        // chat/responses upstreams at the middle level.
        out["reasoning_effort"] = json!("medium");
    }
    out
}

/// Chat completion -> Anthropic Messages response object.
pub fn chat_to_anthropic_response(chat: &Value, model: &str) -> Value {
    let msg = chat
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or(json!({}));
    let mut blocks: Vec<Value> = vec![];

    if let Some(r) = msg.get("reasoning_content").and_then(|x| x.as_str()) {
        if !r.is_empty() {
            blocks.push(json!({"type": "thinking", "thinking": r, "signature": ""}));
        }
    }
    let text = msg.get("content").and_then(|x| x.as_str()).unwrap_or("");
    if !text.is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }
    if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tcs {
            let id = tc.get("id").and_then(|x| x.as_str()).unwrap_or("");
            let name = tc
                .pointer("/function/name")
                .and_then(|x| x.as_str())
                .unwrap_or("");
            let args = tc
                .pointer("/function/arguments")
                .and_then(|x| x.as_str())
                .unwrap_or("{}");
            let input: Value = serde_json::from_str(args).unwrap_or(json!({}));
            blocks.push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
        }
    }
    if blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": ""}));
    }

    let finish = chat
        .pointer("/choices/0/finish_reason")
        .and_then(|x| x.as_str())
        .unwrap_or("stop");
    let (p, c, cached, _) = chat_usage_parts(chat.get("usage"));

    json!({
        "id": gen_id("msg_"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": anthropic_stop(finish),
        "stop_sequence": Value::Null,
        "usage": anthropic_usage(p, c, cached)
    })
}

// ---------------------------------------------------------------------------
// Chat chunks -> Anthropic Messages SSE stream (egress)
// ---------------------------------------------------------------------------

enum AnthBlock {
    Text,
    Thinking,
    Tool,
}

/// Stateful translator for OpenAI chat chunks into Anthropic Messages SSE
/// events (`message_start` ... `message_stop`). Each returned [`SseEvent`]
/// carries an `event:` name, which is mandatory for Anthropic clients.
pub struct ChatToAnthropicStream {
    msg_id: String,
    model: String,
    started: bool,
    index: usize,
    open: Option<AnthBlock>,
}

impl ChatToAnthropicStream {
    pub fn new(model: &str) -> Self {
        Self {
            msg_id: gen_id("msg_"),
            model: model.to_string(),
            started: false,
            index: 0,
            open: None,
        }
    }

    fn close_open(&mut self, out: &mut Vec<SseEvent>) {
        if self.open.take().is_some() {
            out.push(SseEvent::ev(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": self.index - 1}),
            ));
        }
    }

    /// Handle one canonical OpenAI chat chunk. Returns Anthropic SSE events.
    pub fn handle(&mut self, ch: &Value) -> Vec<SseEvent> {
        let mut out = vec![];
        if !self.started {
            self.started = true;
            out.push(SseEvent::ev(
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": self.msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": {"input_tokens": 0, "output_tokens": 0}
                    }
                }),
            ));
        }

        let choice = ch.pointer("/choices/0").cloned().unwrap_or(json!({}));
        let delta = choice.get("delta").cloned().unwrap_or(json!({}));

        if let Some(t) = delta.get("reasoning_content").and_then(|x| x.as_str()) {
            if !t.is_empty() {
                if !matches!(self.open, Some(AnthBlock::Thinking)) {
                    self.close_open(&mut out);
                    let index = self.index;
                    self.index += 1;
                    out.push(SseEvent::ev(
                        "content_block_start",
                        json!({
                            "type": "content_block_start", "index": index,
                            "content_block": {"type": "thinking", "thinking": "", "signature": ""}
                        }),
                    ));
                    self.open = Some(AnthBlock::Thinking);
                }
                out.push(SseEvent::ev(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta", "index": self.index - 1,
                        "delta": {"type": "thinking_delta", "thinking": t}
                    }),
                ));
            }
        }

        if let Some(t) = delta.get("content").and_then(|x| x.as_str()) {
            if !t.is_empty() {
                if !matches!(self.open, Some(AnthBlock::Text)) {
                    self.close_open(&mut out);
                    let index = self.index;
                    self.index += 1;
                    out.push(SseEvent::ev(
                        "content_block_start",
                        json!({
                            "type": "content_block_start", "index": index,
                            "content_block": {"type": "text", "text": ""}
                        }),
                    ));
                    self.open = Some(AnthBlock::Text);
                }
                out.push(SseEvent::ev(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta", "index": self.index - 1,
                        "delta": {"type": "text_delta", "text": t}
                    }),
                ));
            }
        }

        if let Some(tcs) = delta.get("tool_calls").and_then(|x| x.as_array()) {
            for tc in tcs {
                let new_call = tc
                    .get("id")
                    .and_then(|x| x.as_str())
                    .is_some_and(|s| !s.is_empty())
                    || tc
                        .pointer("/function/name")
                        .and_then(|x| x.as_str())
                        .is_some_and(|s| !s.is_empty());
                if new_call {
                    self.close_open(&mut out);
                    let index = self.index;
                    self.index += 1;
                    let id = tc.get("id").and_then(|x| x.as_str()).unwrap_or("");
                    let name = tc
                        .pointer("/function/name")
                        .and_then(|x| x.as_str())
                        .unwrap_or("");
                    out.push(SseEvent::ev(
                        "content_block_start",
                        json!({
                            "type": "content_block_start", "index": index,
                            "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}
                        }),
                    ));
                    self.open = Some(AnthBlock::Tool);
                }
                if let Some(args) = tc.pointer("/function/arguments").and_then(|x| x.as_str()) {
                    if !args.is_empty() && matches!(self.open, Some(AnthBlock::Tool)) {
                        out.push(SseEvent::ev(
                            "content_block_delta",
                            json!({
                                "type": "content_block_delta", "index": self.index - 1,
                                "delta": {"type": "input_json_delta", "partial_json": args}
                            }),
                        ));
                    }
                }
            }
        }

        if let Some(fr) = choice.get("finish_reason").and_then(|x| x.as_str()) {
            if self.open.is_none() {
                // Anthropic messages must have at least one content block.
                let index = self.index;
                self.index += 1;
                out.push(SseEvent::ev(
                    "content_block_start",
                    json!({
                        "type": "content_block_start", "index": index,
                        "content_block": {"type": "text", "text": ""}
                    }),
                ));
                self.open = Some(AnthBlock::Text);
            }
            self.close_open(&mut out);
            let (p, c, cached, _) = chat_usage_parts(ch.get("usage"));
            out.push(SseEvent::ev(
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": anthropic_stop(fr), "stop_sequence": Value::Null},
                    "usage": anthropic_usage(p, c, cached)
                }),
            ));
            out.push(SseEvent::ev(
                "message_stop",
                json!({"type": "message_stop"}),
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn types<'a>(events: &'a [SseEvent]) -> Vec<String> {
        events
            .iter()
            .map(|e| {
                e.data
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("?")
                    .to_string()
            })
            .collect()
    }

    // ---- framing ----------------------------------------------------------

    #[test]
    fn framing_data_only_vs_event_named() {
        assert_eq!(frame_data(&json!({"a": 1})), "data: {\"a\":1}\n\n");
        assert_eq!(
            frame_event("message_stop", &json!({"type": "message_stop"})),
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        );
        let ev = SseEvent::ev("x", json!({"type": "y"}));
        assert!(ev.frame().starts_with("event: x\n"));
    }

    // ---- chat -> anthropic ----------------------------------------------

    #[test]
    fn tool_results_merge_into_one_user_message() {
        let req = json!({
            "model": "m",
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "r1"},
                {"role": "tool", "tool_call_id": "c2", "content": "r2"},
                {"role": "user", "content": "next"}
            ]
        });
        let out = openai_to_anthropic(&req, ThinkingMode::Adaptive);
        let msgs = out.get("messages").unwrap().as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["content"].as_array().unwrap().len(), 2);
        assert_eq!(msgs[1]["role"], "user");
        let blocks = msgs[1]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["tool_use_id"], "c1");
        assert_eq!(blocks[1]["tool_use_id"], "c2");
        assert_eq!(blocks[2]["type"], "text");
    }

    #[test]
    fn tool_choice_function_form_and_none() {
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": {"type": "function", "function": {"name": "f"}}
        });
        let out = openai_to_anthropic(&req, ThinkingMode::Adaptive);
        assert_eq!(out["tool_choice"], json!({"type": "tool", "name": "f"}));
        assert!(out.get("tools").is_some());

        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": "none"
        });
        let out = openai_to_anthropic(&req, ThinkingMode::Adaptive);
        assert!(out.get("tools").is_none());
        assert!(out.get("tool_choice").is_none());
    }

    #[test]
    fn reasoning_effort_maps_to_thinking() {
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high"
        });
        let out = openai_to_anthropic(&req, ThinkingMode::Adaptive);
        assert_eq!(out["thinking"], json!({"type": "adaptive"}));
        assert_eq!(out["output_config"], json!({"effort": "high"}));
    }

    #[test]
    fn reasoning_effort_enabled_mode_keeps_legacy_shape() {
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high"
        });
        let out = openai_to_anthropic(&req, ThinkingMode::Enabled);
        assert_eq!(
            out["thinking"],
            json!({"type": "enabled", "budget_tokens": 16384})
        );
        assert!(out.get("output_config").is_none());

        // No effort -> no thinking key at all.
        let req = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        let out = openai_to_anthropic(&req, ThinkingMode::Enabled);
        assert!(out.get("thinking").is_none());
    }

    #[test]
    fn anthropic_thinking_shapes_map_to_reasoning_effort() {
        // Adaptive round-trip: effort passes through output_config.
        let out = anthropic_to_chat_request(&json!({
            "model": "m",
            "max_tokens": 10,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(out["reasoning_effort"], json!("low"));

        // Bare adaptive: keep reasoning on at the middle level.
        let out = anthropic_to_chat_request(&json!({
            "model": "m",
            "max_tokens": 10,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(out["reasoning_effort"], json!("medium"));

        // Legacy enabled + budget maps by thresholds.
        let out = anthropic_to_chat_request(&json!({
            "model": "m",
            "max_tokens": 10,
            "thinking": {"type": "enabled", "budget_tokens": 20000},
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(out["reasoning_effort"], json!("high"));

        // No thinking config -> no reasoning_effort.
        let out = anthropic_to_chat_request(&json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert!(out.get("reasoning_effort").is_none());
    }

    #[test]
    fn anthropic_to_openai_maps_thinking_and_cache_usage() {
        let resp = json!({
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "id": "tu1", "name": "f", "input": {"x": 1}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 7}
        });
        let out = anthropic_to_openai(&resp, "m");
        let msg = &out["choices"][0]["message"];
        assert_eq!(msg["content"], Value::Null);
        assert_eq!(msg["reasoning_content"], json!("hmm"));
        assert_eq!(msg["tool_calls"][0]["function"]["name"], json!("f"));
        assert_eq!(out["choices"][0]["finish_reason"], json!("tool_calls"));
        assert_eq!(out["usage"]["prompt_tokens"], json!(10));
        assert_eq!(
            out["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(7)
        );
        assert_eq!(out["usage"]["total_tokens"], json!(15));
    }

    #[test]
    fn anthropic_stream_tool_index_only_counts_tool_use() {
        let mut s = AnthropicStream::new("m");
        let evs = [
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 3}}}),
            json!({"type": "content_block_start", "content_block": {"type": "text"}}),
            json!({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "t"}}),
            json!({"type": "content_block_stop"}),
            json!({"type": "content_block_start", "content_block": {"type": "tool_use", "id": "c1", "name": "a"}}),
            json!({"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
            json!({"type": "content_block_stop"}),
            json!({"type": "content_block_start", "content_block": {"type": "tool_use", "id": "c2", "name": "b"}}),
            json!({"type": "content_block_stop"}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
            json!({"type": "message_stop"}),
        ];
        let mut chunks: Vec<Value> = vec![];
        for ev in &evs {
            chunks.extend(s.handle(ev));
        }
        // One start chunk per tool_use block (carries the call id).
        let starts: Vec<&Value> = chunks
            .iter()
            .filter(|c| c.pointer("/choices/0/delta/tool_calls/0/id").is_some())
            .collect();
        assert_eq!(starts.len(), 2);
        assert_eq!(
            starts[0]
                .pointer("/choices/0/delta/tool_calls/0/index")
                .unwrap(),
            0
        );
        assert_eq!(
            starts[1]
                .pointer("/choices/0/delta/tool_calls/0/index")
                .unwrap(),
            1
        );
        // thinking deltas surface as reasoning_content
        let last = chunks.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], json!("tool_calls"));
        assert_eq!(last["usage"]["prompt_tokens"], json!(3));
        assert_eq!(last["usage"]["completion_tokens"], json!(9));
    }

    // ---- responses ingress ------------------------------------------------

    #[test]
    fn responses_input_items_map_to_chat_messages() {
        let req = json!({
            "model": "m",
            "input": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "call"}]},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{\"x\":1}"},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": "http://x/i.png"}
                ]}
            ]
        });
        let out = responses_to_chat_request(&req);
        let msgs = out["messages"].as_array().unwrap();
        // reasoning folds into the assistant message that follows it
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["content"], json!("call"));
        assert_eq!(msgs[0]["reasoning_content"], json!("think"));
        assert_eq!(msgs[0]["tool_calls"][0]["function"]["name"], json!("f"));
        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[1]["tool_call_id"], json!("c1"));
        assert_eq!(msgs[1]["content"], json!("ok"));
        assert_eq!(msgs[2]["role"], "user");
        let parts = msgs[2]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
    }

    #[test]
    fn responses_params_map_to_chat() {
        let req = json!({
            "model": "m",
            "input": "hi",
            "max_output_tokens": 55,
            "top_p": 0.9,
            "reasoning": {"effort": "high"},
            "text": {"format": {"type": "json_schema", "name": "s", "schema": {"type": "object"}, "strict": true}},
            "tool_choice": {"type": "function", "name": "f"},
            "parallel_tool_calls": false,
            "include": ["reasoning.encrypted_content"]
        });
        let out = responses_to_chat_request(&req);
        assert_eq!(out["max_tokens"], json!(55));
        assert_eq!(out["top_p"], json!(0.9));
        assert_eq!(out["reasoning_effort"], json!("high"));
        assert_eq!(out["response_format"]["type"], json!("json_schema"));
        assert_eq!(out["response_format"]["json_schema"]["name"], json!("s"));
        assert_eq!(
            out["tool_choice"],
            json!({"type": "function", "function": {"name": "f"}})
        );
        assert_eq!(out["parallel_tool_calls"], json!(false));
        assert_eq!(
            out["x_nervogate"]["include"],
            json!(["reasoning.encrypted_content"])
        );
    }

    #[test]
    fn stateful_fields_policy() {
        assert!(responses_stateful_error(&json!({"previous_response_id": "resp_1"})).is_some());
        assert!(responses_stateful_error(&json!({"background": true})).is_some());
        assert!(responses_stateful_error(&json!({"store": true, "include": ["x"]})).is_none());
        assert!(responses_stateful_error(&json!({"input": "hi"})).is_none());
    }

    #[test]
    fn chat_to_responses_response_maps_tools_status_usage() {
        let chat = json!({
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": null,
                "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "f", "arguments": "{\"x\":1}"}}]
            }}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 6,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 1}}
        });
        let out = chat_to_responses_response(&chat, "m");
        assert_eq!(out["object"], json!("response"));
        assert_eq!(out["status"], json!("completed"));
        assert_eq!(out["output"][0]["type"], json!("function_call"));
        assert_eq!(out["output"][0]["call_id"], json!("c1"));
        assert_eq!(out["output"][0]["arguments"], json!("{\"x\":1}"));
        assert_eq!(out["usage"]["input_tokens"], json!(4));
        assert_eq!(out["usage"]["output_tokens"], json!(6));
        assert_eq!(
            out["usage"]["input_tokens_details"]["cached_tokens"],
            json!(2)
        );
        assert_eq!(
            out["usage"]["output_tokens_details"]["reasoning_tokens"],
            json!(1)
        );

        let chat = json!({
            "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": "x"}}]
        });
        let out = chat_to_responses_response(&chat, "m");
        assert_eq!(out["status"], json!("incomplete"));
        assert_eq!(
            out["incomplete_details"],
            json!({"reason": "max_output_tokens"})
        );
    }

    #[test]
    fn chat_to_responses_stream_emits_full_event_sequence() {
        let mut s = ChatToResponsesStream::new("m");
        let mut events: Vec<SseEvent> = vec![];
        events.extend(s.handle(&chunk("i", "m", json!({"role": "assistant"}), None)));
        events.extend(s.handle(&chunk("i", "m", json!({"content": "Hel"}), None)));
        events.extend(s.handle(&chunk("i", "m", json!({"content": "lo"}), None)));
        events.extend(s.handle(&chunk(
            "i",
            "m",
            json!({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                "function": {"name": "f", "arguments": ""}}]}),
            None,
        )));
        events.extend(s.handle(&chunk(
            "i",
            "m",
            json!({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}),
            None,
        )));
        let mut final_chunk = chunk("i", "m", json!({}), Some("tool_calls"));
        final_chunk["usage"] = chat_usage(1, 2, 0, 0);
        events.extend(s.handle(&final_chunk));

        let ts = types(&events);
        assert_eq!(ts[0], "response.created");
        assert_eq!(ts[1], "response.in_progress");
        assert_eq!(ts[2], "response.output_item.added"); // message
        assert_eq!(ts[3], "response.content_part.added");
        assert_eq!(ts[4], "response.output_text.delta");
        assert_eq!(ts[5], "response.output_text.delta");
        assert_eq!(ts[6], "response.output_text.done"); // message closes
        assert_eq!(ts[7], "response.content_part.done");
        assert_eq!(ts[8], "response.output_item.done");
        assert_eq!(ts[9], "response.output_item.added"); // function_call
        assert_eq!(ts[10], "response.function_call_arguments.delta");
        assert_eq!(ts[11], "response.function_call_arguments.done");
        assert_eq!(ts[12], "response.output_item.done");
        assert_eq!(ts[13], "response.completed");

        let last = events.last().unwrap();
        let output = last
            .data
            .pointer("/response/output")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[0]["content"][0]["text"], "Hello");
        assert_eq!(output[1]["type"], "function_call");
        assert_eq!(output[1]["arguments"], "{}");
        assert_eq!(last.data.pointer("/response/status").unwrap(), "completed");
        assert_eq!(
            last.data.pointer("/response/usage/input_tokens").unwrap(),
            1
        );
    }

    #[test]
    fn chat_to_responses_stream_incomplete_status() {
        let mut s = ChatToResponsesStream::new("m");
        let mut events = s.handle(&chunk("i", "m", json!({"content": "x"}), None));
        events.extend(s.handle(&chunk("i", "m", json!({}), Some("length"))));
        let last = events.last().unwrap();
        assert_eq!(last.data["type"], "response.incomplete");
        assert_eq!(
            last.data.pointer("/response/incomplete_details"),
            Some(&json!({"reason": "max_output_tokens"}))
        );
    }

    #[test]
    fn chat_to_anthropic_stream_emits_event_names() {
        let mut s = ChatToAnthropicStream::new("m");
        let mut events: Vec<SseEvent> = vec![];
        events.extend(s.handle(&chunk("i", "m", json!({"reasoning_content": "hmm"}), None)));
        events.extend(s.handle(&chunk("i", "m", json!({"content": "hi"}), None)));
        events.extend(s.handle(&chunk(
            "i",
            "m",
            json!({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                "function": {"name": "f", "arguments": ""}}]}),
            None,
        )));
        events.extend(s.handle(&chunk(
            "i",
            "m",
            json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"a\":1}"}}]}),
            None,
        )));
        let mut final_chunk = chunk("i", "m", json!({}), Some("tool_calls"));
        final_chunk["usage"] = chat_usage(3, 4, 1, 0);
        events.extend(s.handle(&final_chunk));

        assert!(events.iter().all(|e| e.event.is_some()));
        let names: Vec<&str> = events.iter().map(|e| e.event.as_deref().unwrap()).collect();
        assert_eq!(names[0], "message_start");
        assert_eq!(names[1], "content_block_start"); // thinking
        assert_eq!(names[2], "content_block_delta"); // thinking_delta
        assert_eq!(names[3], "content_block_stop");
        assert_eq!(names[4], "content_block_start"); // text
        assert_eq!(names[5], "content_block_delta"); // text_delta
        assert_eq!(names[6], "content_block_stop");
        assert_eq!(names[7], "content_block_start"); // tool_use
        assert_eq!(names[8], "content_block_delta"); // input_json_delta
        assert_eq!(names[9], "content_block_stop");
        assert_eq!(names[10], "message_delta");
        assert_eq!(names[11], "message_stop");

        let delta = &events[8].data;
        assert_eq!(delta["delta"]["type"], "input_json_delta");
        assert_eq!(delta["delta"]["partial_json"], "{\"a\":1}");
        let msg_delta = &events[10].data;
        assert_eq!(msg_delta["delta"]["stop_reason"], "tool_use");
        assert_eq!(msg_delta["usage"]["input_tokens"], 3);
        assert_eq!(msg_delta["usage"]["output_tokens"], 4);
    }

    // ---- anthropic ingress -----------------------------------------------

    #[test]
    fn anthropic_request_maps_to_chat_messages() {
        let req = json!({
            "model": "m",
            "max_tokens": 100,
            "system": [{"type": "text", "text": "be brief"}],
            "thinking": {"type": "enabled", "budget_tokens": 8192},
            "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "f"},
            "stop_sequences": ["stop!"],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "s"},
                    {"type": "text", "text": "calling"},
                    {"type": "tool_use", "id": "tu1", "name": "f", "input": {"x": 1}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}
                ]}
            ]
        });
        let out = anthropic_to_chat_request(&req);
        assert_eq!(out["max_tokens"], json!(100));
        assert_eq!(out["stop"], json!(["stop!"]));
        assert_eq!(out["reasoning_effort"], json!("high"));
        assert_eq!(
            out["tool_choice"],
            json!({"type": "function", "function": {"name": "f"}})
        );
        assert_eq!(out["tools"][0]["function"]["name"], json!("f"));
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "be brief");
        assert_eq!(msgs[1]["role"], "user");
        let parts = msgs[1]["content"].as_array().unwrap();
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,AA==");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["reasoning_content"], "hmm");
        assert_eq!(msgs[2]["content"], json!("calling"));
        assert_eq!(msgs[2]["tool_calls"][0]["function"]["name"], "f");
        assert_eq!(
            msgs[2]["tool_calls"][0]["function"]["arguments"],
            "{\"x\":1}"
        );
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["tool_call_id"], "tu1");
        assert_eq!(msgs[3]["content"], "ok");
    }

    #[test]
    fn chat_to_anthropic_response_with_thinking_and_tools() {
        let chat = json!({
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant",
                "content": "calling",
                "reasoning_content": "hmm",
                "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "f", "arguments": "{\"x\":1}"}}]
            }}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 1}}
        });
        let out = chat_to_anthropic_response(&chat, "m");
        assert_eq!(out["type"], "message");
        assert_eq!(out["role"], "assistant");
        assert_eq!(out["stop_reason"], "tool_use");
        let blocks = out["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["thinking"], "hmm");
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[2]["type"], "tool_use");
        assert_eq!(blocks[2]["input"], json!({"x": 1}));
        assert_eq!(out["usage"]["cache_read_input_tokens"], json!(1));
    }

    // ---- responses upstream -> chat --------------------------------------

    #[test]
    fn responses_to_openai_maps_details_refusal_reasoning() {
        let resp = json!({
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "hm"}]},
                {"type": "message", "content": [
                    {"type": "output_text", "text": "hi"},
                    {"type": "refusal", "refusal": "no"}
                ]},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 7,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 2}}
        });
        let out = responses_to_openai(&resp, "m");
        let msg = &out["choices"][0]["message"];
        assert_eq!(msg["reasoning_content"], json!("hm"));
        assert_eq!(msg["content"], Value::Null);
        assert_eq!(msg["tool_calls"][0]["id"], json!("c1"));
        assert_eq!(out["choices"][0]["finish_reason"], json!("tool_calls"));
        assert_eq!(
            out["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(3)
        );
        assert_eq!(
            out["usage"]["completion_tokens_details"]["reasoning_tokens"],
            json!(2)
        );
    }

    // ---- chat -> responses upstream --------------------------------------

    #[test]
    fn chat_to_responses_request_maps_params_and_passthrough() {
        let req = json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "calling", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}
            ],
            "max_tokens": 9,
            "top_p": 0.8,
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
            "tool_choice": {"type": "function", "function": {"name": "f"}},
            "x_nervogate": {"include": ["reasoning.encrypted_content"]}
        });
        let out = chat_to_responses_request(&req);
        assert_eq!(out["instructions"], json!("sys"));
        assert_eq!(out["max_output_tokens"], json!(9));
        assert_eq!(out["top_p"], json!(0.8));
        assert_eq!(out["reasoning"], json!({"effort": "low"}));
        assert_eq!(out["text"], json!({"format": {"type": "json_object"}}));
        assert_eq!(out["tool_choice"], json!({"type": "function", "name": "f"}));
        assert_eq!(out["store"], json!(false));
        assert_eq!(out["include"], json!(["reasoning.encrypted_content"]));
        let input = out["input"].as_array().unwrap();
        assert_eq!(input.len(), 4);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[3]["type"], "function_call_output");
    }
}
