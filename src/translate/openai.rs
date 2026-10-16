//! OpenAI Responses <-> canonical Chat translation.

use serde_json::{json, Value};

use super::common::*;
pub fn responses_input_parts(c: &Value) -> Vec<Value> {
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
                    other => warn_dropped_block("chat_to_responses_request", other),
                }
            }
        }
        _ => {}
    }
    parts
}

/// Assistant turns must use `output_text` (the upstream rejects `input_text`
/// with HTTP 400), and tool calls must be separate `function_call` items.
pub fn assistant_output_parts(c: &Value) -> Vec<Value> {
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

/// Request parameters a Chat -> Responses translation cannot carry over.
pub fn chat_to_responses_loss(req: &Value) -> Vec<&'static str> {
    const LOST: &[&str] = &[
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "seed",
        "n",
        "stop",
        "logprobs",
    ];
    LOST.iter()
        .copied()
        .filter(|k| req.get(*k).is_some_and(|v| !v.is_null()))
        .collect()
}

/// `n > 1` would silently collapse to a single candidate.
pub fn chat_to_responses_fatal(req: &Value) -> Option<String> {
    if req.get("n").and_then(|x| x.as_i64()).unwrap_or(1) > 1 {
        return Some(
            "`n > 1` is not supported for Responses upstreams: the translation \
             yields a single candidate and `choices[1]` would be undefined"
                .to_string(),
        );
    }
    None
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
    // `failed` is an upstream fault, not a finish reason: mapping it onto
    // `stop` hands the client HTTP 200 with an empty "answer". Callers see
    // `status == "failed"` and turn it into a 502 / error event.
    let finish = if !tool_calls.is_empty() {
        "tool_calls"
    } else if status == "incomplete" {
        incomplete_reason(resp)
            .as_deref()
            .map(finish_from_incomplete)
            .unwrap_or("length")
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
        "usage": chat_usage(in_tok, out_tok, cached, reasoning_tok),
        // Internal hand-off read by `chat_failure`; `handle_ingress` strips it
        // before the object reaches a client.
        "upstream_error": responses_failure(resp)
    })
}

/// `incomplete_details.reason` — not always a token limit, it can be
/// `content_filter` or `max_output_tokens`.
pub fn incomplete_reason(resp: &Value) -> Option<String> {
    resp.pointer("/incomplete_details/reason")
        .and_then(|x| x.as_str())
        .map(String::from)
}

/// Chat `finish_reason` for a Responses `incomplete_details.reason`.
pub fn finish_from_incomplete(reason: &str) -> &'static str {
    match reason {
        "max_output_tokens" | "max_tokens" => "length",
        "content_filter" => "content_filter",
        _ => "length",
    }
}

/// Human-readable error text for a `failed` Responses response.
pub fn responses_failure(resp: &Value) -> Option<String> {
    if resp.get("status").and_then(|s| s.as_str()) != Some("failed") {
        return None;
    }
    Some(
        resp.pointer("/error/message")
            .and_then(|x| x.as_str())
            .map(String::from)
            .or_else(|| resp.get("error").and_then(|x| x.as_str()).map(String::from))
            .unwrap_or_else(|| "upstream response failed".to_string()),
    )
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
                let ty = ev.get("type").and_then(|t| t.as_str()).unwrap_or("");
                // `failed` is an upstream fault. Emitting `stop` here would
                // hand the client a normal-looking, truncated answer.
                let fr = match ty {
                    "response.incomplete" => {
                        incomplete_reason(ev.pointer("/response").unwrap_or(ev))
                            .as_deref()
                            .map(finish_from_incomplete)
                            .unwrap_or("length")
                    }
                    "response.failed" => "error",
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
    // `store: true` promises a retrievable response id, but there is no
    // GET endpoint behind it. Silently downgrading it to `store: false`
    // makes the client wait for an id that never resolves.
    if req.get("store").and_then(|x| x.as_bool()).unwrap_or(false) {
        return Some(
            "store: true is not supported: this gateway is stateless and keeps no \
             response store; send store: false"
                .to_string(),
        );
    }
    // An `item_reference` points into that same store. Dropping it would
    // silently remove earlier turns from the conversation.
    if let Some(Value::Array(items)) = req.get("input") {
        if items
            .iter()
            .any(|it| it.get("type").and_then(|t| t.as_str()) == Some("item_reference"))
        {
            return Some(
                "`item_reference` input items are not supported: they reference a \
                 server-side store this gateway does not keep; inline the item instead"
                    .to_string(),
            );
        }
    }
    None
}

pub fn responses_content_to_chat_parts(c: &Value) -> Vec<Value> {
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
                    other => warn_dropped_block("responses_to_chat_request", other),
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
pub fn push_chat_tool_call(messages: &mut Vec<Value>, id: &str, name: &str, args: &str) {
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
    // `refusal` carries the reply when `content` is empty; dropping it
    // yields an empty message the client cannot explain.
    let refusal = msg.get("refusal").and_then(|c| c.as_str()).unwrap_or("");
    if !text.is_empty() {
        output.push(json!({
            "type": "message",
            "id": gen_id("msg_"),
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]
        }));
    } else if !refusal.is_empty() {
        output.push(json!({
            "type": "message",
            "id": gen_id("msg_"),
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": refusal}]
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
        // Not always a token limit: see `finish_from_incomplete`.
        "content_filter" => ("incomplete", Some(json!({"reason": "content_filter"}))),
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

pub enum OpenOut {
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
    terminated: bool,
    next_index: usize,
    open: Option<OpenOut>,
    /// Tool calls buffered until the stream ends, keyed by upstream
    /// `tool_calls[].index` so parallel calls never mix arguments.
    tools: Vec<(ToolKey, PendingTool)>,
    items: Vec<Value>,
}

impl ChatToResponsesStream {
    pub fn new(model: &str) -> Self {
        Self {
            resp_id: gen_id("resp_"),
            model: model.to_string(),
            created: now_ts(),
            started: false,
            terminated: false,
            next_index: 0,
            open: None,
            tools: vec![],
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

    fn start(&mut self, out: &mut Vec<SseEvent>) {
        if !self.started {
            self.started = true;
            out.push(SseEvent::data(
                json!({"type": "response.created", "response": self.skeleton("in_progress")}),
            ));
            out.push(SseEvent::data(
                json!({"type": "response.in_progress", "response": self.skeleton("in_progress")}),
            ));
        }
    }

    fn terminal(&mut self, ty: &str, status: &str, incomplete: Value, out: &mut Vec<SseEvent>) {
        self.flush_tools(out);
        self.close_open(out);
        self.terminated = true;
        let response = json!({
            "id": self.resp_id,
            "object": "response",
            "created_at": self.created,
            "model": self.model,
            "status": status,
            "incomplete_details": incomplete,
            "output": self.items,
            "usage": responses_usage(0, 0, 0, 0),
            "error": Value::Null
        });
        out.push(SseEvent::data(json!({"type": ty, "response": response})));
    }

    /// Emit buffered tool calls as complete, contiguous `function_call`
    /// output items (added -> arguments -> done per `output_index`).
    fn flush_tools(&mut self, out: &mut Vec<SseEvent>) {
        for (_, t) in std::mem::take(&mut self.tools) {
            self.close_open(out);
            let index = self.next_index;
            self.next_index += 1;
            let item_id = gen_id("fc_");
            let call_id = if t.id.is_empty() {
                gen_id("call_")
            } else {
                t.id.clone()
            };
            let args = if t.args.trim().is_empty() {
                "{}".to_string()
            } else {
                t.args.clone()
            };
            out.push(SseEvent::data(json!({
                "type": "response.output_item.added", "output_index": index,
                "item": {
                    "id": item_id, "type": "function_call", "status": "in_progress",
                    "call_id": call_id, "name": t.name, "arguments": ""
                }
            })));
            out.push(SseEvent::data(json!({
                "type": "response.function_call_arguments.delta",
                "item_id": item_id, "output_index": index, "delta": args
            })));
            self.open = Some(OpenOut::Tool {
                index,
                item_id,
                call_id,
                name: t.name,
                args,
            });
        }
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
        self.start(&mut out);

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

        // Like the Anthropic egress, tool calls are buffered per upstream
        // `tool_calls[].index` and emitted as complete, contiguous output
        // items: Responses also requires added -> delta* -> done per
        // `output_index`, which interleaved parallel calls would violate.
        if let Some(tcs) = delta.get("tool_calls").and_then(|x| x.as_array()) {
            for tc in tcs {
                let pos = match tool_slot(tc, &self.tools) {
                    Some(key) => match self.tools.iter().position(|(k, _)| *k == key) {
                        Some(p) => p,
                        None => {
                            self.tools.push((key, PendingTool::default()));
                            self.tools.len() - 1
                        }
                    },
                    None => continue,
                };
                let entry = &mut self.tools[pos].1;
                if let Some(id) = tc
                    .get("id")
                    .and_then(|x| x.as_str())
                    .filter(|s| !s.is_empty())
                {
                    entry.id = id.to_string();
                }
                if let Some(n) = tc
                    .pointer("/function/name")
                    .and_then(|x| x.as_str())
                    .filter(|s| !s.is_empty())
                {
                    entry.name = n.to_string();
                }
                if let Some(a) = tc.pointer("/function/arguments").and_then(|x| x.as_str()) {
                    entry.args.push_str(a);
                }
            }
        }

        // Terminal chunk carries finish_reason (and usage).
        if let Some(fr) = choice.get("finish_reason").and_then(|x| x.as_str()) {
            self.flush_tools(&mut out);
            let (ty, status, incomplete) = match fr {
                "length" => (
                    "response.incomplete",
                    "incomplete",
                    json!({"reason": "max_output_tokens"}),
                ),
                // An `incomplete` Responses response can be a content-filter
                // stop, not a token limit. Reporting it as `completed` would
                // dress a truncated answer up as a finished one.
                "content_filter" => (
                    "response.incomplete",
                    "incomplete",
                    json!({"reason": "content_filter"}),
                ),
                _ => ("response.completed", "completed", Value::Null),
            };
            self.close_open(&mut out);
            self.terminated = true;
            let (p, c, cached, reasoning_tok) = chat_usage_parts(ch.get("usage"));
            let response = json!({
                "id": self.resp_id,
                "object": "response",
                "created_at": self.created,
                "model": self.model,
                "status": status,
                "incomplete_details": incomplete,
                "output": self.items,
                "usage": responses_usage(p, c, cached, reasoning_tok),
                "error": Value::Null
            });
            out.push(SseEvent::data(json!({"type": ty, "response": response})));
        }
        out
    }

    /// Terminal event for an upstream that closed without a finish chunk.
    /// Without it the client waits forever on a connection that is already
    /// gone. `None` when a finish chunk already produced the terminal event.
    pub fn finish(&mut self) -> Option<Vec<SseEvent>> {
        if self.terminated {
            return None;
        }
        let mut out = vec![];
        self.start(&mut out);
        self.terminal("response.completed", "completed", Value::Null, &mut out);
        Some(out)
    }

    /// Close any open output item before an error event, so clients see a
    /// well-formed stream instead of an abandoned open block.
    pub fn abort(&mut self) -> Vec<SseEvent> {
        if self.terminated {
            return vec![];
        }
        let mut out = vec![];
        self.start(&mut out);
        self.flush_tools(&mut out);
        self.close_open(&mut out);
        out
    }
}

// ---------------------------------------------------------------------------
// Anthropic Messages ingress -> OpenAI chat request
// ---------------------------------------------------------------------------
