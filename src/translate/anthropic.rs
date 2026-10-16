//! Anthropic Messages <-> canonical Chat translation.

use serde_json::{json, Value};

use crate::config::ThinkingMode;

use super::common::*;
pub fn openai_image_to_anthropic(p: &Value) -> Option<Value> {
    let url = p.pointer("/image_url/url").and_then(|x| x.as_str())?;
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',')?;
        let media = meta.split(';').next().unwrap_or("image/png");
        Some(json!({"type":"image","source":{"type":"base64","media_type":media,"data":data}}))
    } else {
        Some(json!({"type":"image","source":{"type":"url","url":url}}))
    }
}

pub fn anthropic_image_to_openai(b: &Value) -> Option<Value> {
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
///
/// Block types with no Anthropic equivalent (`input_audio`, `file`,
/// `document`, ...) are reported instead of silently dropped: the client
/// believes it sent the payload while the model never saw it.
pub fn user_blocks(c: &Value) -> Vec<Value> {
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
                    other => {
                        warn_dropped_block("openai_to_anthropic", other);
                        None
                    }
                }
            })
            .collect(),
        _ => vec![],
    }
}

pub fn finish_from_anthropic(stop: &str) -> &'static str {
    match stop {
        "max_tokens" | "max_output_tokens" => "length",
        "tool_use" => "tool_calls",
        "refusal" => "content_filter",
        // end_turn, stop_sequence, pause_turn, ...
        _ => "stop",
    }
}

/// Chat `finish_reason` -> Anthropic `stop_reason`.
pub fn anthropic_stop(finish: &str) -> &'static str {
    match finish {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        // Closes the round trip with `finish_from_anthropic`, which already
        // maps `refusal` -> `content_filter`.
        "content_filter" => "refusal",
        _ => "end_turn",
    }
}

// ---------------------------------------------------------------------------
// OpenAI chat -> Anthropic Messages
// ---------------------------------------------------------------------------

/// Append blocks to the last message when it has the same role (Anthropic
/// wants alternating roles), else start a new message.
pub fn push_anthropic_blocks(msgs: &mut Vec<Value>, role: &str, blocks: Vec<Value>) {
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
pub fn thinking_budget(effort: &str) -> i64 {
    match effort {
        "minimal" => 1024,
        "low" => 4096,
        "medium" => 8192,
        "high" => 16384,
        _ => 4096,
    }
}

/// Fallback `max_tokens` for Anthropic upstreams when the client omitted
/// it. Anthropic *requires* the field, Chat treats it as optional, so the
/// gateway must supply something; the value is configurable.
#[cfg(test)]
pub const ANTHROPIC_DEFAULT_MAX_TOKENS: i64 = 8192;

/// Translation with the built-in default. Production callers go through
/// `openai_to_anthropic_with_max` so the value comes from config.
#[cfg(test)]
pub fn openai_to_anthropic(req: &Value, thinking: ThinkingMode) -> Value {
    openai_to_anthropic_with_max(req, thinking, ANTHROPIC_DEFAULT_MAX_TOKENS)
}

pub fn openai_to_anthropic_with_max(
    req: &Value,
    thinking: ThinkingMode,
    default_max_tokens: i64,
) -> Value {
    let mut out = serde_json::Map::new();
    // `output_config` carries both `effort` (thinking) and `format`
    // (structured outputs); build it once so the two never clobber each other.
    let mut output_config: serde_json::Map<String, Value> = serde_json::Map::new();

    let max_tokens = req
        .get("max_tokens")
        .or_else(|| req.get("max_completion_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(default_max_tokens));
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
                    // Falls back to the top-level `refusal` when `content`
                    // is empty, so a refusal-only turn is not turned into a
                    // blank assistant message.
                    let text = chat_message_text(m);
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

    // Structured outputs: Chat `response_format` -> Anthropic
    // `output_config.format`. Dropping this silently returns free-form text
    // where the caller expects schema-valid JSON.
    if let Some(rf) = req.get("response_format") {
        match rf.get("type").and_then(|x| x.as_str()) {
            Some("json_object") => {
                // Anthropic's `output_config.format` only accepts
                // `json_schema`; sending `json_object` gets a 400 back.
                // The closest faithful translation is an unconstrained
                // object schema — the caller asked for "valid JSON object",
                // and that is exactly what an empty object schema means.
                output_config.insert(
                    "format".into(),
                    json!({"type": "json_schema", "schema": {"type": "object"}}),
                );
            }
            Some("json_schema") => {
                let schema = rf
                    .pointer("/json_schema/schema")
                    .cloned()
                    .unwrap_or(json!({"type": "object"}));
                output_config.insert(
                    "format".into(),
                    json!({"type": "json_schema", "schema": schema}),
                );
            }
            // `text` is Anthropic's native (unconstrained) mode; nothing to do.
            Some("text") | None => {}
            _ => {}
        }
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
                    output_config.insert("effort".into(), json!(e));
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
    if !output_config.is_empty() {
        out.insert("output_config".into(), Value::Object(output_config));
    }

    Value::Object(out)
}

/// Request parameters a Chat -> Anthropic translation cannot carry over.
/// Silently dropping them changes model behaviour, so the caller reports
/// them (stderr always, HTTP 400 when `strict_params` is on).
pub fn chat_to_anthropic_loss(req: &Value) -> Vec<&'static str> {
    const LOST: &[&str] = &[
        "n",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "seed",
        "logprobs",
        "top_logprobs",
        "user",
        // `stream_options` is deliberately absent: dropping it only means
        // "no OpenAI-style trailing usage chunk", which is the upstream
        // default anyway and `Inbound::Chat::usage` already covers the
        // ordering. Warning on it would fire on nearly every SDK request.
    ];
    LOST.iter()
        .copied()
        .filter(|k| req.get(*k).is_some_and(|v| !v.is_null()))
        .collect()
}

/// `n` changes the *shape* of the reply (one candidate vs many), so a
/// caller indexing `choices[1]` breaks silently. It gets a hard 400 under
/// `strict_params`. `parallel_tool_calls` is only warned about: OpenAI SDKs
/// send it by default, so rejecting it would break otherwise-working setups.
pub fn chat_to_anthropic_fatal(req: &Value) -> Option<String> {
    if req.get("n").and_then(|x| x.as_i64()).unwrap_or(1) > 1 {
        return Some(
            "`n > 1` is not supported for Anthropic upstreams: the translation \
             yields a single candidate and `choices[1]` would be undefined"
                .to_string(),
        );
    }
    None
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
    // Anthropic `input_tokens` excludes cache reads/writes, while OpenAI
    // `prompt_tokens` includes cached tokens. Adding them keeps
    // `prompt_tokens` comparable and `total_tokens` honest.
    let cached = resp
        .pointer("/usage/cache_read_input_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let cache_creation = resp
        .pointer("/usage/cache_creation_input_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let prompt_total = in_tok + cached + cache_creation;

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
        "usage": chat_usage(prompt_total, out_tok, cached, 0),
        "cache_creation_input_tokens": cache_creation
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
    cache_creation: i64,
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
            cache_creation: 0,
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
                self.cache_creation = ev
                    .pointer("/message/usage/cache_creation_input_tokens")
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
                // Anthropic's `input_tokens` excludes cache reads; OpenAI's
                // `prompt_tokens` includes them.
                c["usage"] = chat_usage(
                    self.in_tok + self.cached + self.cache_creation,
                    self.out_tok,
                    self.cached,
                    0,
                );
                c["cache_creation_input_tokens"] = json!(self.cache_creation);
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

/// Parameters an Anthropic upstream is likely to reject because thinking is
/// on.
///
/// Unlike [`chat_to_anthropic_loss`] these are **not** dropped — they are
/// forwarded. Removing `temperature: 0` would let the caller believe the
/// model is deterministic when it is not, which is a silent behaviour
/// change; making them an error would break the Anthropic-protocol proxies
/// that do accept them. So the gateway passes them through and says so.
///
/// Anthropic's docs state it as "you can't set a custom temperature while
/// thinking is on, so it uses the default of 1"; upstreams that enforce it
/// answer 400.
///
/// Only values differing from Anthropic's defaults (`temperature` 1,
/// `top_p` 1) are reported. Forwarding the default is accepted, and warning
/// on it would fire on essentially every request.
pub fn chat_to_anthropic_thinking_conflict(req: &Value) -> Vec<&'static str> {
    let thinking = req
        .get("reasoning_effort")
        .and_then(|x| x.as_str())
        .is_some_and(|e| e != "none");
    if !thinking {
        return vec![];
    }
    let mut out = vec![];
    for k in ["temperature", "top_p"] {
        if req
            .get(k)
            .and_then(|v| v.as_f64())
            .is_some_and(|v| v != 1.0)
        {
            out.push(k);
        }
    }
    out
}

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
                            other => warn_dropped_block("anthropic_to_chat_request", other),
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
    let refusal = msg.get("refusal").and_then(|x| x.as_str()).unwrap_or("");
    if !text.is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    } else if !refusal.is_empty() {
        // Anthropic reports a refusal through `stop_reason: "refusal"` with
        // the explanation in the text block; keeping the text preserves it.
        blocks.push(json!({"type": "text", "text": refusal}));
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
    // Canonical `prompt_tokens` already includes cached tokens; Anthropic
    // reports cache reads and writes separately.
    let (p, c, cached, _) = chat_usage_parts(chat.get("usage"));
    let cache_creation = chat
        .get("cache_creation_input_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let input = (p - cached - cache_creation).max(0);

    json!({
        "id": gen_id("msg_"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": anthropic_stop(finish),
        "stop_sequence": Value::Null,
        "usage": anthropic_usage_full(input, c, cache_creation, cached)
    })
}

// ---------------------------------------------------------------------------
// Chat chunks -> Anthropic Messages SSE stream (egress)
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
pub enum AnthBlock {
    Text,
    Thinking,
}

/// Stateful translator for OpenAI chat chunks into Anthropic Messages SSE
/// events (`message_start` ... `message_stop`). Each returned [`SseEvent`]
/// carries an `event:` name, which is mandatory for Anthropic clients.
pub struct ChatToAnthropicStream {
    msg_id: String,
    model: String,
    started: bool,
    terminated: bool,
    index: usize,
    /// The currently open block, tagged with the content index it was
    /// emitted at. `index - 1` is wrong once blocks interleave.
    open: Option<(AnthBlock, usize)>,
    /// Tool calls seen so far, in first-seen order so parallel calls keep
    /// their relative ordering in the Anthropic message.
    tools: Vec<(ToolKey, PendingTool)>,
}

impl ChatToAnthropicStream {
    pub fn new(model: &str) -> Self {
        Self {
            msg_id: gen_id("msg_"),
            model: model.to_string(),
            started: false,
            terminated: false,
            index: 0,
            open: None,
            tools: vec![],
        }
    }

    /// Emit every buffered tool call as a complete, contiguous Anthropic
    /// content block. Each call becomes exactly one `tool_use` block
    /// carrying its full argument JSON, which is what strict clients
    /// require (they reject interleaved and split blocks).
    ///
    /// Returns true when at least one block was emitted, so the caller can
    /// skip the "message needs a content block" fallback.
    fn flush_tools(&mut self, out: &mut Vec<SseEvent>) -> bool {
        let tools = std::mem::take(&mut self.tools);
        let emitted = !tools.is_empty();
        for (_, t) in tools {
            self.close_open(out);
            let ci = self.index;
            self.index += 1;
            let args = if t.args.trim().is_empty() {
                "{}".to_string()
            } else {
                t.args.clone()
            };
            out.push(SseEvent::ev(
                "content_block_start",
                json!({
                    "type": "content_block_start", "index": ci,
                    "content_block": {"type": "tool_use", "id": t.id, "name": t.name, "input": {}}
                }),
            ));
            out.push(SseEvent::ev(
                "content_block_delta",
                json!({
                    "type": "content_block_delta", "index": ci,
                    "delta": {"type": "input_json_delta", "partial_json": args}
                }),
            ));
            out.push(SseEvent::ev(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": ci}),
            ));
        }
        emitted
    }

    fn start(&mut self, out: &mut Vec<SseEvent>) {
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
    }

    /// Open a text block if no block is currently open.
    fn ensure_block(&mut self, out: &mut Vec<SseEvent>) {
        if self.open.is_none() {
            let index = self.index;
            self.index += 1;
            out.push(SseEvent::ev(
                "content_block_start",
                json!({
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "text", "text": ""}
                }),
            ));
            self.open = Some((AnthBlock::Text, index));
        }
    }

    fn close_open(&mut self, out: &mut Vec<SseEvent>) {
        if let Some((_, index)) = self.open.take() {
            out.push(SseEvent::ev(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": index}),
            ));
        }
    }

    fn terminal(&mut self, stop_reason: &str, out: &mut Vec<SseEvent>) {
        // Tool blocks already satisfy Anthropic's "at least one content
        // block" rule, so only add an empty text block when there are none.
        let had_tools = self.flush_tools(out);
        if !had_tools {
            self.ensure_block(out);
        }
        self.close_open(out);
        self.terminated = true;
        out.push(SseEvent::ev(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": Value::Null},
                "usage": anthropic_usage(0, 0, 0)
            }),
        ));
        out.push(SseEvent::ev(
            "message_stop",
            json!({"type": "message_stop"}),
        ));
    }

    /// Handle one canonical OpenAI chat chunk. Returns Anthropic SSE events.
    pub fn handle(&mut self, ch: &Value) -> Vec<SseEvent> {
        let mut out = vec![];
        self.start(&mut out);

        let choice = ch.pointer("/choices/0").cloned().unwrap_or(json!({}));
        let delta = choice.get("delta").cloned().unwrap_or(json!({}));

        if let Some(t) = delta.get("reasoning_content").and_then(|x| x.as_str()) {
            if !t.is_empty() {
                let ci = match &self.open {
                    Some((AnthBlock::Thinking, ci)) => *ci,
                    _ => {
                        self.close_open(&mut out);
                        let ci = self.index;
                        self.index += 1;
                        out.push(SseEvent::ev(
                            "content_block_start",
                            json!({
                                "type": "content_block_start", "index": ci,
                                "content_block": {"type": "thinking", "thinking": "", "signature": ""}
                            }),
                        ));
                        self.open = Some((AnthBlock::Thinking, ci));
                        ci
                    }
                };
                out.push(SseEvent::ev(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta", "index": ci,
                        "delta": {"type": "thinking_delta", "thinking": t}
                    }),
                ));
            }
        }

        if let Some(t) = delta.get("content").and_then(|x| x.as_str()) {
            if !t.is_empty() {
                let ci = match &self.open {
                    Some((AnthBlock::Text, ci)) => *ci,
                    _ => {
                        self.close_open(&mut out);
                        let ci = self.index;
                        self.index += 1;
                        out.push(SseEvent::ev(
                            "content_block_start",
                            json!({
                                "type": "content_block_start", "index": ci,
                                "content_block": {"type": "text", "text": ""}
                            }),
                        ));
                        self.open = Some((AnthBlock::Text, ci));
                        ci
                    }
                };
                out.push(SseEvent::ev(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta", "index": ci,
                        "delta": {"type": "text_delta", "text": t}
                    }),
                ));
            }
        }

        // Anthropic's SSE requires each content block to be started, fully
        // deltad and stopped before the next one opens, but Chat streams may
        // interleave arguments across parallel tool calls. Tool blocks are
        // therefore accumulated per upstream `tool_calls[].index` and
        // emitted contiguously at the end of the stream; text and thinking
        // still stream through live.
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
                // A later chunk may carry the field the first one omitted
                // (upstreams split `id` and `name` across chunks).
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

        if let Some(fr) = choice.get("finish_reason").and_then(|x| x.as_str()) {
            let had_tools = self.flush_tools(&mut out);
            if !had_tools {
                self.ensure_block(&mut out);
            }
            self.close_open(&mut out);
            let (p, c, cached, _) = chat_usage_parts(ch.get("usage"));
            let cache_creation = ch
                .get("cache_creation_input_tokens")
                .and_then(|x| x.as_i64())
                .unwrap_or(0);
            // Canonical `prompt_tokens` includes cached tokens; Anthropic
            // reports cache reads and writes separately.
            let input = (p - cached - cache_creation).max(0);
            self.terminated = true;
            out.push(SseEvent::ev(
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": anthropic_stop(fr), "stop_sequence": Value::Null},
                    "usage": anthropic_usage_full(input, c, cache_creation, cached)
                }),
            ));
            out.push(SseEvent::ev(
                "message_stop",
                json!({"type": "message_stop"}),
            ));
        }
        out
    }

    /// Terminal events for an upstream that closed without a finish chunk.
    /// Without them the client hangs on a connection that already closed.
    /// `None` when a finish chunk already emitted `message_stop`.
    pub fn finish(&mut self) -> Option<Vec<SseEvent>> {
        if self.terminated {
            return None;
        }
        let mut out = vec![];
        self.start(&mut out);
        self.terminal("end_turn", &mut out);
        Some(out)
    }

    /// Close the open content block before an error event so the client sees
    /// a well-formed stream instead of an abandoned block index.
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
