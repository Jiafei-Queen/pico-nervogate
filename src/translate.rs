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

fn normalize_user_content(c: &Value) -> Value {
    match c {
        Value::String(_) => c.clone(),
        Value::Array(parts) => Value::Array(
            parts
                .iter()
                .filter_map(|p| {
                    let ty = p.get("type").and_then(|x| x.as_str()).unwrap_or("text");
                    match ty {
                        "text" | "input_text" => p
                            .get("text")
                            .map(|t| json!({"type": "text", "text": t.clone()})),
                        "image_url" => openai_image_to_anthropic(p),
                        _ => None,
                    }
                })
                .collect(),
        ),
        _ => json!(""),
    }
}

fn finish_from_anthropic(stop: &str) -> &'static str {
    match stop {
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        _ => "stop",
    }
}

// ---------------------------------------------------------------------------
// OpenAI chat -> Anthropic Messages
// ---------------------------------------------------------------------------

pub fn openai_to_anthropic(req: &Value) -> Value {
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
                    msgs.push(json!({"role":"user","content":[
                        {"type":"tool_result","tool_use_id":id,"content":content}
                    ]}));
                }
                "assistant" => {
                    let mut blocks: Vec<Value> = vec![];
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
                            blocks.push(json!({"type":"tool_use","id":id,"name":name,"input":input}));
                        }
                    }
                    if blocks.is_empty() {
                        blocks.push(json!({"type":"text","text":""}));
                    }
                    msgs.push(json!({"role":"assistant","content":blocks}));
                }
                _ => {
                    let c = m.get("content").cloned().unwrap_or(json!(""));
                    msgs.push(json!({"role":"user","content":normalize_user_content(&c)}));
                }
            }
        }
    }

    if !system.is_empty() {
        out.insert("system".into(), json!(system));
    }
    out.insert("messages".into(), Value::Array(msgs));

    if let Some(tools) = req.get("tools").and_then(|t| t.as_array()) {
        if !tools.is_empty() {
            let at: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.pointer("/function/name").and_then(|x| x.as_str()).unwrap_or(""),
                        "description": t.pointer("/function/description").cloned().unwrap_or(json!("")),
                        "input_schema": t.pointer("/function/parameters").cloned().unwrap_or(json!({"type":"object"}))
                    })
                })
                .collect();
            out.insert("tools".into(), Value::Array(at));
        }
    }
    if let Some(tc) = req.get("tool_choice") {
        if let Some(s) = tc.as_str() {
            let mapped = match s {
                "auto" => Some(json!({"type":"auto"})),
                "required" => Some(json!({"type":"any"})),
                "none" => None,
                _ => None,
            };
            if let Some(m) = mapped {
                out.insert("tool_choice".into(), m);
            }
        }
    }

    Value::Object(out)
}

pub fn anthropic_to_openai(resp: &Value, model: &str) -> Value {
    let id = gen_id("chatcmpl-");
    let mut text = String::new();
    let mut tool_calls: Vec<Value> = vec![];

    if let Some(blocks) = resp.get("content").and_then(|c| c.as_array()) {
        for b in blocks {
            match b.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(t) = b.get("text").and_then(|x| x.as_str()) {
                        text.push_str(t);
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

    let stop = resp.get("stop_reason").and_then(|s| s.as_str()).unwrap_or("end_turn");
    let finish = finish_from_anthropic(stop);
    let in_tok = resp.pointer("/usage/input_tokens").and_then(|x| x.as_i64()).unwrap_or(0);
    let out_tok = resp.pointer("/usage/output_tokens").and_then(|x| x.as_i64()).unwrap_or(0);

    let mut message = json!({"role": "assistant", "content": text});
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
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok, "total_tokens": in_tok + out_tok}
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
    in_tok: i64,
    out_tok: i64,
    finish: Option<String>,
}

impl AnthropicStream {
    pub fn new(model: &str) -> Self {
        Self {
            id: gen_id("chatcmpl-"),
            model: model.to_string(),
            role_sent: false,
            tool_index: 0,
            in_tok: 0,
            out_tok: 0,
            finish: None,
        }
    }

    fn ensure_role(&mut self, out: &mut Vec<Value>) {
        if !self.role_sent {
            self.role_sent = true;
            out.push(chunk(&self.id, &self.model, json!({"role":"assistant"}), None));
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
                self.ensure_role(&mut out);
            }
            "content_block_start" => {
                self.ensure_role(&mut out);
                let bt = ev.pointer("/content_block/type").and_then(|x| x.as_str()).unwrap_or("");
                if bt == "tool_use" {
                    let id = ev.pointer("/content_block/id").and_then(|x| x.as_str()).unwrap_or("");
                    let name = ev.pointer("/content_block/name").and_then(|x| x.as_str()).unwrap_or("");
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
                let dt = ev.pointer("/delta/type").and_then(|x| x.as_str()).unwrap_or("");
                if dt == "text_delta" {
                    if let Some(t) = ev.pointer("/delta/text").and_then(|x| x.as_str()) {
                        if !t.is_empty() {
                            out.push(chunk(&self.id, &self.model, json!({"content": t}), None));
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
                self.tool_index += 1;
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
                c["usage"] = json!({
                    "prompt_tokens": self.in_tok,
                    "completion_tokens": self.out_tok,
                    "total_tokens": self.in_tok + self.out_tok
                });
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
    if let Some(mt) = req.get("max_tokens").or_else(|| req.get("max_completion_tokens")) {
        out.insert("max_output_tokens".into(), mt.clone());
    }
    if let Some(t) = req.get("temperature") {
        out.insert("temperature".into(), t.clone());
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
    Value::Object(out)
}

pub fn responses_to_openai(resp: &Value, model: &str) -> Value {
    let id = gen_id("chatcmpl-");
    let mut text = String::new();
    let mut tool_calls: Vec<Value> = vec![];

    if let Some(output) = resp.get("output").and_then(|o| o.as_array()) {
        for item in output {
            match item.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                "message" => {
                    if let Some(content) = item.get("content").and_then(|c| c.as_array()) {
                        for c in content {
                            if c.get("type").and_then(|t| t.as_str()) == Some("output_text") {
                                if let Some(t) = c.get("text").and_then(|x| x.as_str()) {
                                    text.push_str(t);
                                }
                            }
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

    let status = resp.get("status").and_then(|s| s.as_str()).unwrap_or("completed");
    let finish = if !tool_calls.is_empty() {
        "tool_calls"
    } else if status == "incomplete" {
        "length"
    } else {
        "stop"
    };
    let in_tok = resp.pointer("/usage/input_tokens").and_then(|x| x.as_i64()).unwrap_or(0);
    let out_tok = resp.pointer("/usage/output_tokens").and_then(|x| x.as_i64()).unwrap_or(0);

    let mut message = json!({"role": "assistant", "content": text});
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
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok, "total_tokens": in_tok + out_tok}
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
        }
    }

    fn ensure_role(&mut self, out: &mut Vec<Value>) {
        if !self.role_sent {
            self.role_sent = true;
            out.push(chunk(&self.id, &self.model, json!({"role":"assistant"}), None));
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
                    let name = item.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string();
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
            "response.completed" => {
                self.ensure_role(&mut out);
                self.in_tok = ev
                    .pointer("/response/usage/input_tokens")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(0);
                self.out_tok = ev
                    .pointer("/response/usage/output_tokens")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(0);
                let fr = if self.tool_index > 0 { "tool_calls" } else { "stop" };
                let mut c = chunk(&self.id, &self.model, json!({}), Some(fr));
                c["usage"] = json!({
                    "prompt_tokens": self.in_tok,
                    "completion_tokens": self.out_tok,
                    "total_tokens": self.in_tok + self.out_tok
                });
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

pub fn responses_to_chat_request(req: &Value) -> Value {
    let mut messages: Vec<Value> = vec![];
    if let Some(inst) = req.get("instructions").and_then(|x| x.as_str()) {
        if !inst.is_empty() {
            messages.push(json!({"role": "system", "content": inst}));
        }
    }
    match req.get("input") {
        Some(Value::String(s)) => messages.push(json!({"role": "user", "content": s})),
        Some(Value::Array(items)) => {
            for it in items {
                let role = it.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                let text = content_text(it.get("content").unwrap_or(&Value::Null));
                messages.push(json!({"role": role, "content": text}));
            }
        }
        _ => {}
    }

    let mut out = json!({
        "model": req.get("model").cloned().unwrap_or(json!("")),
        "messages": messages
    });
    if let Some(mt) = req.get("max_output_tokens") {
        out["max_tokens"] = mt.clone();
    }
    if let Some(t) = req.get("temperature") {
        out["temperature"] = t.clone();
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
    out
}
