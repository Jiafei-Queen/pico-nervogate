//! Google Gemini (v1beta `generateContent`) translation.
//!
//! The gateway's canonical shape stays OpenAI Chat; this module only speaks
//! `chat <-> google`. All other pairs ride through it, exactly like the
//! Anthropic and Responses translators.
//!
//! Wire shapes (camelCase on the wire; snake_case accepted on ingress):
//! - request: `{system_instruction:{parts:[{text}]}, contents:[{role,parts}],
//!   tools:[{functionDeclarations}], toolConfig, generationConfig}`
//! - response: `{candidates:[{content:{role,parts},finishReason}],
//!   usageMetadata:{promptTokenCount,candidatesTokenCount,...}}`
//! - stream: `POST ...:streamGenerateContent?alt=sse`, each `data:` line is
//!   one `GenerateContentResponse` JSON object (no `event:` name).

use serde_json::{json, Value};

use super::common::*;

// ---------------------------------------------------------------------------
// Action parsing (ingress path)
// ---------------------------------------------------------------------------

/// Split `{model}:generateContent` / `{model}:streamGenerateContent`.
/// Returns `(model, stream)`.
pub fn parse_google_action(action: &str) -> Option<(String, bool)> {
    let (model, verb) = action.rsplit_once(':')?;
    if model.is_empty() {
        return None;
    }
    match verb {
        "generateContent" => Some((model.to_string(), false)),
        "streamGenerateContent" => Some((model.to_string(), true)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Shared part helpers
// ---------------------------------------------------------------------------

/// Thinking budget for `reasoning_effort`. Values mirror the Anthropic
/// legacy budgets so effort levels behave comparably across upstreams.
pub fn thinking_budget(effort: &str) -> i64 {
    match effort {
        "minimal" => 0,
        "low" => 4096,
        "medium" => 8192,
        "high" => 16384,
        _ => 4096,
    }
}

fn part_text(p: &Value) -> Option<&str> {
    p.get("text").and_then(|x| x.as_str())
}

fn is_thought_part(p: &Value) -> bool {
    p.get("thought").and_then(|x| x.as_bool()).unwrap_or(false)
}

fn get_inline_data(p: &Value) -> Option<(String, String)> {
    let d = p.get("inlineData").or_else(|| p.get("inline_data"))?;
    let mime = d
        .get("mimeType")
        .or_else(|| d.get("mime_type"))
        .and_then(|x| x.as_str())
        .unwrap_or("image/png")
        .to_string();
    let data = d.get("data").and_then(|x| x.as_str())?.to_string();
    Some((mime, data))
}

fn get_file_data(p: &Value) -> Option<(String, String)> {
    let d = p.get("fileData").or_else(|| p.get("file_data"))?;
    let mime = d
        .get("mimeType")
        .or_else(|| d.get("mime_type"))
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let uri = d
        .get("fileUri")
        .or_else(|| d.get("file_uri"))
        .or_else(|| d.get("fileUri"))
        .and_then(|x| x.as_str())?
        .to_string();
    Some((mime, uri))
}

fn get_function_call(p: &Value) -> Option<(&str, Value)> {
    let c = p.get("functionCall").or_else(|| p.get("function_call"))?;
    let name = c.get("name").and_then(|x| x.as_str())?;
    let args = c.get("args").cloned().unwrap_or(json!({}));
    Some((name, args))
}

/// Thought-signature round-trip for thinking models. Gemini attaches
/// `thoughtSignature` as a sibling of `functionCall` parts and requires it
/// back on multi-turn follow-ups; OpenAI-shaped clients (e.g. OpenWebUI)
/// carry it on the tool call as `extra_content.google.thought_signature`.
fn part_thought_signature(p: &Value) -> Option<&str> {
    p.get("thoughtSignature")
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
}

fn tool_call_thought_signature(tc: &Value) -> Option<&str> {
    tc.pointer("/extra_content/google/thought_signature")
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
}

fn get_function_response(p: &Value) -> Option<(&str, Value)> {
    let r = p
        .get("functionResponse")
        .or_else(|| p.get("function_response"))?;
    let name = r.get("name").and_then(|x| x.as_str())?;
    let resp = r.get("response").cloned().unwrap_or(json!({}));
    Some((name, resp))
}

/// Chat `image_url` value -> Google part. `data:` URLs become `inlineData`,
/// remote URLs become `fileData` (Gemini may not fetch arbitrary https;
/// documented in design.md).
fn image_url_to_google_part(url: &str) -> Value {
    if let Some(rest) = url.strip_prefix("data:") {
        if let Some((meta, data)) = rest.split_once(',') {
            let mime = meta.split(';').next().unwrap_or("image/png");
            return json!({"inlineData": {"mimeType": mime, "data": data}});
        }
    }
    json!({"fileData": {"fileUri": url}})
}

/// Google image-ish part -> Chat `image_url` part.
fn google_image_to_chat_part(p: &Value) -> Option<Value> {
    if let Some((mime, data)) = get_inline_data(p) {
        return Some(
            json!({"type":"image_url","image_url":{"url": format!("data:{mime};base64,{data}")}}),
        );
    }
    if let Some((_, uri)) = get_file_data(p) {
        return Some(json!({"type":"image_url","image_url":{"url": uri}}));
    }
    None
}

/// Function-response payload -> plain text for Chat `tool` messages.
/// `{result: "..."}` unwraps; otherwise the JSON itself.
fn function_response_text(resp: &Value) -> String {
    if let Some(s) = resp.as_str() {
        return s.to_string();
    }
    if let Some(s) = resp.get("result").and_then(|x| x.as_str()) {
        return s.to_string();
    }
    if let Some(s) = resp.get("content").and_then(|x| x.as_str()) {
        return s.to_string();
    }
    serde_json::to_string(resp).unwrap_or_else(|_| "{}".to_string())
}

/// Chat tool output text -> Google `functionResponse.response` object.
/// JSON objects pass through; anything else wraps as `{result: ...}`.
fn tool_output_to_response_object(text: &str) -> Value {
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(_)) => serde_json::from_str(text).unwrap_or(json!({"result": text})),
        Ok(other) => json!({"result": other}),
        Err(_) => json!({"result": text}),
    }
}

// ---------------------------------------------------------------------------
// Chat -> Google request
// ---------------------------------------------------------------------------

/// Google's function-declaration schemas are a restricted subset: unknown
/// keywords such as `additionalProperties` are rejected with a 400.
/// Strip that key at every depth of a parameter schema in place.
fn strip_additional_properties(schema: &mut Value) {
    match schema {
        Value::Object(map) => {
            map.remove("additionalProperties");
            for v in map.values_mut() {
                strip_additional_properties(v);
            }
        }
        Value::Array(items) => {
            for v in items {
                strip_additional_properties(v);
            }
        }
        _ => {}
    }
}

/// True when a parameter schema contains `additionalProperties` anywhere.
fn has_additional_properties(schema: &Value) -> bool {
    match schema {
        Value::Object(map) => {
            map.contains_key("additionalProperties")
                || map.values().any(has_additional_properties)
        }
        Value::Array(items) => items.iter().any(has_additional_properties),
        _ => false,
    }
}

/// Canonical chat request -> Gemini `generateContent` request body.
pub fn chat_to_google_request(req: &Value) -> Value {
    let mut out = serde_json::Map::new();
    let mut contents: Vec<Value> = vec![];
    let mut system_text = String::new();

    if let Some(list) = req.get("messages").and_then(|m| m.as_array()) {
        for m in list {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            match role {
                "system" | "developer" => {
                    let t = crate::translate::content_text(
                        m.get("content").unwrap_or(&Value::Null),
                    );
                    if !t.is_empty() {
                        if !system_text.is_empty() {
                            system_text.push('\n');
                        }
                        system_text.push_str(&t);
                    }
                }
                "tool" => {
                    let name = m
                        .get("name")
                        .and_then(|x| x.as_str())
                        .unwrap_or("");
                    // Chat `tool` messages carry `tool_call_id`; the function
                    // name is not repeated on the message, so recover it from
                    // the matching assistant tool_call when possible. Fall
                    // back to `name` / the call id.
                    let call_id =
                        m.get("tool_call_id").and_then(|x| x.as_str()).unwrap_or("");
                    let fname = if !name.is_empty() {
                        name.to_string()
                    } else {
                        lookup_tool_name(list, call_id).unwrap_or_else(|| call_id.to_string())
                    };
                    let content = m.get("content").cloned().unwrap_or(json!(""));
                    let text = crate::translate::content_text(&content);
                    contents.push(json!({
                        "role": "user",
                        "parts": [{
                            "functionResponse": {
                                "name": fname,
                                "response": tool_output_to_response_object(&text)
                            }
                        }]
                    }));
                }
                "assistant" => {
                    let mut parts: Vec<Value> = vec![];
                    if let Some(r) = m.get("reasoning_content").and_then(|x| x.as_str()) {
                        if !r.is_empty() {
                            parts.push(json!({"thought": true, "text": r}));
                        }
                    }
                    let content = m.get("content").unwrap_or(&Value::Null);
                    push_chat_content_parts(content, &mut parts);
                    if let Some(tcs) = m.get("tool_calls").and_then(|t| t.as_array()) {
                        for tc in tcs {
                            let name = tc
                                .pointer("/function/name")
                                .and_then(|x| x.as_str())
                                .unwrap_or("");
                            let args = tc
                                .pointer("/function/arguments")
                                .and_then(|x| x.as_str())
                                .unwrap_or("{}");
                            let args: Value =
                                serde_json::from_str(args).unwrap_or(json!({}));
                            let mut part =
                                json!({"functionCall": {"name": name, "args": args}});
                            if let Some(sig) = tool_call_thought_signature(tc) {
                                part["thoughtSignature"] = json!(sig);
                            }
                            parts.push(part);
                        }
                    }
                    if !parts.is_empty() {
                        contents.push(json!({"role": "model", "parts": parts}));
                    }
                }
                _ => {
                    let mut parts: Vec<Value> = vec![];
                    push_chat_content_parts(m.get("content").unwrap_or(&Value::Null), &mut parts);
                    if !parts.is_empty() {
                        contents.push(json!({"role": role_to_google(role), "parts": parts}));
                    }
                }
            }
        }
    }

    if !system_text.is_empty() {
        out.insert(
            "system_instruction".into(),
            json!({"parts": [{"text": system_text}]}),
        );
    }
    out.insert("contents".into(), Value::Array(contents));

    // Tools. Entries without a function name (e.g. OpenAI's
    // `{"type": "code_interpreter"}`, which has no Google equivalent) are
    // dropped: Gemini rejects empty declaration names with a 400.
    // Parameter schemas are sanitized for Google's restricted subset
    // (`additionalProperties` is rejected with a 400).
    if let Some(tools) = req.get("tools").and_then(|t| t.as_array()) {
        let decls: Vec<Value> = tools
            .iter()
            .filter_map(|t| {
                let name = t.pointer("/function/name").and_then(|x| x.as_str())?;
                if name.is_empty() {
                    return None;
                }
                let mut parameters = t
                    .pointer("/function/parameters")
                    .cloned()
                    .unwrap_or(json!({"type":"object"}));
                strip_additional_properties(&mut parameters);
                Some(json!({
                    "name": name,
                    "description": t.pointer("/function/description").cloned().unwrap_or(json!("")),
                    "parameters": parameters
                }))
            })
            .collect();
        if !decls.is_empty() {
            out.insert(
                "tools".into(),
                json!([{"functionDeclarations": decls}]),
            );
        }
    }
    if let Some(tc) = req.get("tool_choice") {
        let (mode, names): (&str, Option<Vec<Value>>) = match tc.as_str() {
            Some("auto") => ("AUTO", None),
            Some("required") => ("ANY", None),
            Some("none") => ("NONE", None),
            _ => tc
                .pointer("/function/name")
                .and_then(|x| x.as_str())
                .map(|n| ("ANY", Some(vec![json!(n)])))
                .unwrap_or(("AUTO", None)),
        };
        let mut fcc = serde_json::Map::new();
        fcc.insert("mode".into(), json!(mode));
        if let Some(n) = names {
            fcc.insert("allowedFunctionNames".into(), Value::Array(n));
        }
        out.insert(
            "toolConfig".into(),
            json!({"functionCallingConfig": fcc}),
        );
    }

    // Sampling / limits.
    let mut gen = serde_json::Map::new();
    if let Some(v) = req.get("temperature") {
        gen.insert("temperature".into(), v.clone());
    }
    if let Some(v) = req.get("top_p") {
        gen.insert("topP".into(), v.clone());
    }
    if let Some(k) = req.pointer("/x_nervogate/top_k") {
        gen.insert("topK".into(), k.clone());
    }
    if let Some(mt) = req
        .get("max_tokens")
        .or_else(|| req.get("max_completion_tokens"))
    {
        gen.insert("maxOutputTokens".into(), mt.clone());
    }
    if let Some(stop) = req.get("stop") {
        let arr = match stop {
            Value::Array(a) => a.clone(),
            Value::String(s) => vec![Value::String(s.clone())],
            _ => vec![],
        };
        if !arr.is_empty() {
            gen.insert("stopSequences".into(), Value::Array(arr));
        }
    }
    if let Some(s) = req.get("seed") {
        gen.insert("seed".into(), s.clone());
    }
    if let Some(v) = req.get("presence_penalty") {
        gen.insert("presencePenalty".into(), v.clone());
    }
    if let Some(v) = req.get("frequency_penalty") {
        gen.insert("frequencyPenalty".into(), v.clone());
    }
    if let Some(e) = req.get("reasoning_effort").and_then(|x| x.as_str()) {
        if e != "none" {
            let budget = thinking_budget(e);
            if budget > 0 {
                gen.insert(
                    "thinkingConfig".into(),
                    json!({"thinkingBudget": budget, "includeThoughts": true}),
                );
            }
        }
    }
    // Structured output.
    if let Some(rf) = req.get("response_format") {
        match rf.get("type").and_then(|x| x.as_str()) {
            Some("json_object") => {
                gen.insert("responseMimeType".into(), json!("application/json"));
                gen.insert(
                    "responseSchema".into(),
                    json!({"type": "OBJECT", "properties": {}}),
                );
            }
            Some("json_schema") => {
                let schema = rf
                    .pointer("/json_schema/schema")
                    .cloned()
                    .unwrap_or(json!({"type": "object"}));
                gen.insert("responseMimeType".into(), json!("application/json"));
                gen.insert("responseSchema".into(), uppercase_schema(schema));
            }
            _ => {}
        }
    }
    if !gen.is_empty() {
        out.insert("generationConfig".into(), Value::Object(gen));
    }
    // Safety settings ride the reserved key so unknown keys never leak.
    if let Some(s) = req.pointer("/x_nervogate/safetySettings") {
        out.insert("safetySettings".into(), s.clone());
    }
    Value::Object(out)
}

fn role_to_google(role: &str) -> &'static str {
    match role {
        "assistant" => "model",
        "system" | "developer" => "user",
        _ => "user",
    }
}

/// Find the function name for `tool_call_id` by scanning assistant messages.
fn lookup_tool_name(list: &[Value], call_id: &str) -> Option<String> {
    if call_id.is_empty() {
        return None;
    }
    for m in list {
        if m.get("role").and_then(|r| r.as_str()) != Some("assistant") {
            continue;
        }
        if let Some(tcs) = m.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tcs {
                if tc.get("id").and_then(|x| x.as_str()) == Some(call_id) {
                    if let Some(n) = tc.pointer("/function/name").and_then(|x| x.as_str()) {
                        return Some(n.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Chat message `content` (string or parts array) -> Google parts.
fn push_chat_content_parts(content: &Value, parts: &mut Vec<Value>) {
    match content {
        Value::String(s) => {
            if !s.is_empty() {
                parts.push(json!({"text": s}));
            }
        }
        Value::Array(arr) => {
            for p in arr {
                match p.get("type").and_then(|x| x.as_str()).unwrap_or("text") {
                    "text" | "input_text" | "output_text" => {
                        if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
                            if !t.is_empty() {
                                parts.push(json!({"text": t}));
                            }
                        }
                    }
                    "image_url" => {
                        if let Some(u) = p.pointer("/image_url/url").and_then(|x| x.as_str()) {
                            parts.push(image_url_to_google_part(u));
                        }
                    }
                    "input_image" => {
                        if let Some(u) = p.get("image_url").and_then(|x| x.as_str()) {
                            parts.push(image_url_to_google_part(u));
                        }
                    }
                    other => warn_dropped_block("chat_to_google_request", other),
                }
            }
        }
        _ => {}
    }
}

/// Uppercase JSON Schema `type` strings for Gemini (`object` -> `OBJECT`).
fn uppercase_schema(v: Value) -> Value {
    match v {
        Value::Object(o) => {
            let mut out = serde_json::Map::new();
            for (k, val) in o {
                if k == "type" {
                    if let Some(s) = val.as_str() {
                        out.insert(k, json!(s.to_ascii_uppercase()));
                    } else {
                        out.insert(k, val);
                    }
                } else {
                    out.insert(k, uppercase_schema(val));
                }
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.into_iter().map(uppercase_schema).collect()),
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Request params the translation cannot carry
// ---------------------------------------------------------------------------

/// Parameters a Chat -> Google translation cannot represent.
pub fn chat_to_google_loss(req: &Value) -> Vec<&'static str> {
    const LOST: &[&str] = &[
        "n",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "user",
        "parallel_tool_calls",
        "metadata",
    ];
    let mut out: Vec<&'static str> = LOST
        .iter()
        .copied()
        .filter(|k| req.get(*k).is_some_and(|v| !v.is_null()))
        .collect();
    // Non-function tools (e.g. `{"type": "code_interpreter"}`) have no
    // Google equivalent and are dropped from the request.
    if req
        .get("tools")
        .and_then(|t| t.as_array())
        .is_some_and(|ts| {
            ts.iter().any(|t| {
                t.pointer("/function/name")
                    .and_then(|x| x.as_str())
                    .map_or(true, str::is_empty)
            })
        })
    {
        out.push("nameless tools (dropped: not function tools)");
    }
    // `additionalProperties` is stripped from tool schemas (see above).
    if req
        .get("tools")
        .and_then(|t| t.as_array())
        .is_some_and(|ts| {
            ts.iter().any(|t| {
                t.pointer("/function/parameters")
                    .is_some_and(has_additional_properties)
            })
        })
    {
        out.push("tool additionalProperties stripped (unsupported by Google)");
    }
    out
}

/// `n > 1` would silently collapse to a single candidate.
pub fn chat_to_google_fatal(req: &Value) -> Option<String> {
    if req.get("n").and_then(|x| x.as_i64()).unwrap_or(1) > 1 {
        return Some(
            "`n > 1` is not supported for Google upstreams: the translation \
             yields a single candidate and `choices[1]` would be undefined"
                .to_string(),
        );
    }
    None
}

// ---------------------------------------------------------------------------
// Google response -> Chat
// ---------------------------------------------------------------------------

fn google_finish(finish: &str, has_tools: bool) -> &'static str {
    // A malformed call is an upstream fault even when call parts are present:
    // handing a broken call to the client looks like a normal tool turn.
    if finish == "MALFORMED_FUNCTION_CALL" {
        return "error";
    }
    if has_tools {
        return "tool_calls";
    }
    match finish {
        "STOP" | "" | "FINISH_REASON_UNSPECIFIED" => "stop",
        "MAX_TOKENS" => "length",
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII"
        | "IMAGE_SAFETY" | "LANGUAGE" => "content_filter",
        _ => "stop",
    }
}

fn first_candidate(resp: &Value) -> Value {
    resp.get("candidates")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .cloned()
        .unwrap_or(json!({}))
}

fn candidate_parts(cand: &Value) -> Vec<Value> {
    cand.pointer("/content/parts")
        .and_then(|p| p.as_array())
        .cloned()
        .unwrap_or_default()
}

/// Gemini `GenerateContentResponse` -> canonical chat completion.
pub fn google_to_chat(resp: &Value, model: &str) -> Value {
    let cand = first_candidate(resp);
    let finish_raw = cand
        .get("finishReason")
        .or_else(|| cand.get("finish_reason"))
        .and_then(|x| x.as_str())
        .unwrap_or("STOP");

    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: Vec<Value> = vec![];

    for p in candidate_parts(&cand) {
        if let Some((name, args)) = get_function_call(&p) {
            let args_str = match &args {
                Value::String(s) => s.clone(),
                _ => serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string()),
            };
            let call_id = crate::translate::gen_id("call_");
            let mut call = json!({
                "id": call_id, "type": "function",
                "function": {"name": name, "arguments": args_str}
            });
            if let Some(sig) = part_thought_signature(&p) {
                call["extra_content"] = json!({"google": {"thought_signature": sig}});
            }
            tool_calls.push(call);
        } else if get_function_response(&p).is_some() {
            // Echoed function results are conversation history, not output.
        } else if is_thought_part(&p) {
            if let Some(t) = part_text(&p) {
                reasoning.push_str(t);
            }
        } else if let Some(t) = part_text(&p) {
            text.push_str(t);
        } else if google_image_to_chat_part(&p).is_some() {
            // Image output has no Chat equivalent; keep the text path.
        }
    }

    let finish = google_finish(finish_raw, !tool_calls.is_empty());
    let (p_tok, c_tok, cached, reasoning_tok) = google_usage_parts(resp);

    let mut message = json!({"role": "assistant", "content": text});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
        message["content"] = Value::Null;
    }

    let mut out = json!({
        "id": crate::translate::gen_id("chatcmpl-"),
        "object": "chat.completion",
        "created": crate::translate::now_ts(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": chat_usage(p_tok, c_tok, cached, reasoning_tok),
    });
    if finish == "error" {
        out["upstream_error"] = json!(format!(
            "upstream response failed (finishReason={finish_raw})"
        ));
    } else {
        out["upstream_error"] = Value::Null;
    }
    out
}

/// `usageMetadata` -> (prompt, completion, cached, reasoning).
pub fn google_usage_parts(resp: &Value) -> (i64, i64, i64, i64) {
    let u = resp.get("usageMetadata").unwrap_or(&Value::Null);
    let prompt = u
        .get("promptTokenCount")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let cand_count = u
        .get("candidatesTokenCount")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let cached = u
        .get("cachedContentTokenCount")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let reasoning = u
        .get("thoughtsTokenCount")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    (prompt, cand_count, cached, reasoning)
}

// ---------------------------------------------------------------------------
// Google ingress -> Chat request
// ---------------------------------------------------------------------------

/// Gemini ingress body -> canonical chat request. `model` comes from the URL.
pub fn google_to_chat_request(req: &Value, model: &str) -> Value {
    let mut messages: Vec<Value> = vec![];

    if let Some(sys) = req.get("system_instruction").or_else(|| req.get("systemInstruction")) {
        let mut t = String::new();
        if let Some(parts) = sys.get("parts").and_then(|p| p.as_array()) {
            for p in parts {
                if let Some(s) = part_text(p) {
                    if !t.is_empty() {
                        t.push('\n');
                    }
                    t.push_str(s);
                }
            }
        } else if let Some(s) = sys.as_str() {
            t.push_str(s);
        }
        if !t.is_empty() {
            messages.push(json!({"role": "system", "content": t}));
        }
    }

    if let Some(list) = req.get("contents").and_then(|c| c.as_array()) {
        for c in list {
            let role = c.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let parts = c.get("parts").and_then(|p| p.as_array());
            let parts: &[Value] = parts.map(|v| v.as_slice()).unwrap_or(&[]);
            match role {
                "model" => {
                    let mut text = String::new();
                    let mut reasoning = String::new();
                    let mut tool_calls: Vec<Value> = vec![];
                    for p in parts {
                        if let Some((name, args)) = get_function_call(p) {
                            let args_str = match &args {
                                Value::String(s) => s.clone(),
                                _ => serde_json::to_string(&args)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            };
                            tool_calls.push(json!({
                                "id": crate::translate::gen_id("call_"),
                                "type": "function",
                                "function": {"name": name, "arguments": args_str}
                            }));
                        } else if is_thought_part(p) {
                            if let Some(t) = part_text(p) {
                                reasoning.push_str(t);
                            }
                        } else if let Some(t) = part_text(p) {
                            text.push_str(t);
                        }
                        // Image parts are collected below so vision survives.
                    }
                    // Collect image parts separately to preserve arrays.
                    let mut img_parts: Vec<Value> = vec![];
                    for p in parts {
                        if part_text(p).is_none()
                            && get_function_call(p).is_none()
                            && !is_thought_part(p)
                        {
                            if let Some(img) = google_image_to_chat_part(p) {
                                img_parts.push(img);
                            }
                        }
                    }
                    let mut msg = if img_parts.is_empty() {
                        json!({"role": "assistant", "content": text})
                    } else if text.is_empty() {
                        json!({"role": "assistant", "content": Value::Array(img_parts)})
                    } else {
                        let mut arr =
                            vec![json!({"type": "text", "text": text})];
                        arr.extend(img_parts);
                        json!({"role": "assistant", "content": Value::Array(arr)})
                    };
                    if !reasoning.is_empty() {
                        msg["reasoning_content"] = json!(reasoning);
                    }
                    if !tool_calls.is_empty() {
                        msg["tool_calls"] = Value::Array(tool_calls);
                        if msg["content"].as_str().is_some_and(|s| s.is_empty()) {
                            msg["content"] = Value::Null;
                        }
                    }
                    if msg["content"].as_str().is_some_and(|s| s.is_empty())
                        && msg.get("tool_calls").is_none()
                        && reasoning.is_empty()
                    {
                        continue;
                    }
                    messages.push(msg);
                }
                _ => {
                    // user / function: text + images become Chat parts;
                    // functionResponse becomes a `tool` message each.
                    let mut text_parts: Vec<Value> = vec![];
                    for p in parts {
                        if let Some((name, resp)) = get_function_response(p) {
                            messages.push(json!({
                                "role": "tool",
                                "tool_call_id": name,
                                "content": function_response_text(&resp)
                            }));
                        } else if let Some(t) = part_text(p) {
                            if !t.is_empty() {
                                text_parts.push(json!({"type": "text", "text": t}));
                            }
                        } else if let Some(img) = google_image_to_chat_part(p) {
                            text_parts.push(img);
                        } else if get_function_call(p).is_some() {
                            // Stray model part in a user turn; ignore.
                        } else if google_image_to_chat_part(p).is_none() {
                            warn_dropped_block("google_to_chat_request", "unknown");
                        }
                    }
                    if !text_parts.is_empty() {
                        let content = if text_parts
                            .iter()
                            .all(|q| q.get("type").and_then(|t| t.as_str()) == Some("text"))
                        {
                            let mut s = String::new();
                            for q in &text_parts {
                                if let Some(t) =
                                    q.get("text").and_then(|x| x.as_str())
                                {
                                    s.push_str(t);
                                }
                            }
                            json!(s)
                        } else {
                            Value::Array(text_parts)
                        };
                        messages.push(json!({"role": "user", "content": content}));
                    }
                }
            }
        }
    }

    let mut out = json!({"model": model, "messages": messages});
    if let Some(gen) = req.get("generationConfig").or_else(|| req.get("generation_config")) {
        if let Some(v) = gen.get("temperature") {
            out["temperature"] = v.clone();
        }
        if let Some(v) = gen.get("topP").or_else(|| gen.get("top_p")) {
            out["top_p"] = v.clone();
        }
        if let Some(v) = gen.get("topK").or_else(|| gen.get("top_k")) {
            out["x_nervogate"] = json!({"top_k": v.clone()});
        }
        if let Some(v) = gen
            .get("maxOutputTokens")
            .or_else(|| gen.get("max_output_tokens"))
        {
            out["max_tokens"] = v.clone();
        }
        if let Some(v) = gen
            .get("stopSequences")
            .or_else(|| gen.get("stop_sequences"))
        {
            out["stop"] = v.clone();
        }
        if let Some(v) = gen.get("seed") {
            out["seed"] = v.clone();
        }
        if let Some(mime) = gen
            .get("responseMimeType")
            .or_else(|| gen.get("response_mime_type"))
            .and_then(|x| x.as_str())
        {
            if mime == "application/json" {
                if let Some(schema) = gen
                    .get("responseSchema")
                    .or_else(|| gen.get("response_schema"))
                {
                    out["response_format"] = json!({
                        "type": "json_schema",
                        "json_schema": {"name": "response", "schema": lowercase_schema(schema.clone()), "strict": false}
                    });
                } else {
                    out["response_format"] = json!({"type": "json_object"});
                }
            }
        }
        if let Some(budget) = gen
            .pointer("/thinkingConfig/thinkingBudget")
            .or_else(|| gen.pointer("/thinking_config/thinking_budget"))
            .and_then(|x| x.as_i64())
        {
            out["reasoning_effort"] = json!(budget_to_effort(budget));
        }
    }
    if let Some(tools) = req.get("tools").and_then(|t| t.as_array()) {
        let mut ot: Vec<Value> = vec![];
        for t in tools {
            let decls = t
                .get("functionDeclarations")
                .or_else(|| t.get("function_declarations"))
                .and_then(|d| d.as_array());
            if let Some(decls) = decls {
                for d in decls {
                    ot.push(json!({"type":"function","function":{
                        "name": d.get("name").cloned().unwrap_or(json!("")),
                        "description": d.get("description").cloned().unwrap_or(json!("")),
                        "parameters": d.get("parameters").cloned().unwrap_or(json!({"type":"object"}))
                    }}));
                }
            }
        }
        if !ot.is_empty() {
            out["tools"] = Value::Array(ot);
        }
    }
    if let Some(fcc) = req
        .pointer("/toolConfig/functionCallingConfig")
        .or_else(|| req.pointer("/tool_config/function_calling_config"))
    {
        let mode = fcc.get("mode").and_then(|x| x.as_str()).unwrap_or("AUTO");
        let mapped = match mode {
            "ANY" => {
                if let Some(names) = fcc
                    .get("allowedFunctionNames")
                    .or_else(|| fcc.get("allowed_function_names"))
                    .and_then(|n| n.as_array())
                    .and_then(|a| a.first())
                    .and_then(|n| n.as_str())
                {
                    json!({"type": "function", "function": {"name": names}})
                } else {
                    json!("required")
                }
            }
            "NONE" => json!("none"),
            _ => json!("auto"),
        };
        out["tool_choice"] = mapped;
    }
    out
}

/// Map a Gemini thinking budget back to an effort level.
fn budget_to_effort(budget: i64) -> &'static str {
    match budget {
        ..=0 => "none",
        1..=4096 => "low",
        4097..=8192 => "medium",
        _ => "high",
    }
}

fn lowercase_schema(v: Value) -> Value {
    match v {
        Value::Object(o) => {
            let mut out = serde_json::Map::new();
            for (k, val) in o {
                if k == "type" {
                    if let Some(s) = val.as_str() {
                        out.insert(k, json!(s.to_ascii_lowercase()));
                    } else {
                        out.insert(k, val);
                    }
                } else {
                    out.insert(k, lowercase_schema(val));
                }
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.into_iter().map(lowercase_schema).collect()),
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Chat -> Google response (ingress Google)
// ---------------------------------------------------------------------------

/// Canonical chat completion -> Gemini `generateContent` response.
pub fn chat_to_google_response(chat: &Value, model: &str) -> Value {
    let msg = chat
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or(json!({}));
    let mut parts: Vec<Value> = vec![];

    if let Some(r) = msg.get("reasoning_content").and_then(|x| x.as_str()) {
        if !r.is_empty() {
            parts.push(json!({"thought": true, "text": r}));
        }
    }
    match msg.get("content") {
        Some(Value::String(s)) if !s.is_empty() => parts.push(json!({"text": s})),
        Some(Value::Array(arr)) => {
            for p in arr {
                if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
                    parts.push(json!({"text": t}));
                } else if let Some(u) = p.pointer("/image_url/url").and_then(|x| x.as_str())
                {
                    parts.push(image_url_to_google_part(u));
                }
            }
        }
        _ => {}
    }
    if let Some(refusal) = msg.get("refusal").and_then(|x| x.as_str()) {
        if !refusal.is_empty() && parts.is_empty() {
            parts.push(json!({"text": refusal}));
        }
    }
    if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tcs {
            let name = tc
                .pointer("/function/name")
                .and_then(|x| x.as_str())
                .unwrap_or("");
            let args = tc
                .pointer("/function/arguments")
                .and_then(|x| x.as_str())
                .unwrap_or("{}");
            let args: Value = serde_json::from_str(args).unwrap_or(json!({}));
            let mut part = json!({"functionCall": {"name": name, "args": args}});
            if let Some(sig) = tool_call_thought_signature(tc) {
                part["thoughtSignature"] = json!(sig);
            }
            parts.push(part);
        }
    }
    if parts.is_empty() {
        parts.push(json!({"text": ""}));
    }

    let finish = chat
        .pointer("/choices/0/finish_reason")
        .and_then(|x| x.as_str())
        .unwrap_or("stop");
    let finish_reason = match finish {
        "length" => "MAX_TOKENS",
        "content_filter" => "SAFETY",
        "tool_calls" => "STOP",
        _ => "STOP",
    };
    let (p, c, cached, reasoning_tok) = crate::translate::chat_usage_parts(chat.get("usage"));

    json!({
        "candidates": [{
            "content": {"role": "model", "parts": parts},
            "finishReason": finish_reason,
            "index": 0
        }],
        "modelVersion": model,
        "usageMetadata": {
            "promptTokenCount": p,
            "candidatesTokenCount": c,
            "totalTokenCount": p + c,
            "cachedContentTokenCount": cached,
            "thoughtsTokenCount": reasoning_tok
        }
    })
}

// ---------------------------------------------------------------------------
// Streaming: Google SSE -> chat chunks (inbound)
// ---------------------------------------------------------------------------

/// Stateful translator for a Gemini `streamGenerateContent` SSE stream into
/// canonical OpenAI chunks. Each `data:` payload is one
/// `GenerateContentResponse`.
pub struct GoogleStream {
    id: String,
    model: String,
    role_sent: bool,
    tool_index: i64,
    in_tok: i64,
    out_tok: i64,
    cached: i64,
    reasoning_tok: i64,
    finish: Option<String>,
}

impl GoogleStream {
    pub fn new(model: &str) -> Self {
        Self {
            id: crate::translate::gen_id("chatcmpl-"),
            model: model.to_string(),
            role_sent: false,
            tool_index: 0,
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
            out.push(role_chunk(&self.id, &self.model));
        }
    }

    pub fn handle(&mut self, ev: &Value) -> Vec<Value> {
        let mut out = vec![];
        let cand = first_candidate(ev);
        if cand.is_null() {
            // Usage-only or empty chunk.
            let (p, c, cached, r) = google_usage_parts(ev);
            if ev.get("usageMetadata").is_some() {
                self.in_tok = p;
                self.out_tok = c;
                self.cached = cached;
                self.reasoning_tok = r;
            }
            return out;
        }
        self.ensure_role(&mut out);
        for p in candidate_parts(&cand) {
            if let Some((name, args)) = get_function_call(&p) {
                let args_str = match &args {
                    Value::String(s) => s.clone(),
                    _ => serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string()),
                };
                let idx = self.tool_index;
                self.tool_index += 1;
                let mut call = json!({"index": idx, "id": crate::translate::gen_id("call_"),
                    "type": "function", "function": {"name": name, "arguments": ""}});
                if let Some(sig) = part_thought_signature(&p) {
                    call["extra_content"] =
                        json!({"google": {"thought_signature": sig}});
                }
                out.push(json!({
                    "id": self.id, "object": "chat.completion.chunk",
                    "created": crate::translate::now_ts(), "model": self.model,
                    "choices": [{"index": 0,
                        "delta": {"tool_calls": [call]},
                        "finish_reason": Value::Null}]
                }));
                // Args arrive complete (object), so emit as one delta.
                out.push(json!({
                    "id": self.id, "object": "chat.completion.chunk",
                    "created": crate::translate::now_ts(), "model": self.model,
                    "choices": [{"index": 0,
                        "delta": {"tool_calls": [{"index": idx, "function": {"arguments": args_str}}]},
                        "finish_reason": Value::Null}]
                }));
            } else if is_thought_part(&p) {
                if let Some(t) = part_text(&p) {
                    if !t.is_empty() {
                        out.push(json!({
                            "id": self.id, "object": "chat.completion.chunk",
                            "created": crate::translate::now_ts(), "model": self.model,
                            "choices": [{"index": 0,
                                "delta": {"reasoning_content": t},
                                "finish_reason": Value::Null}]
                        }));
                    }
                }
            } else if let Some(t) = part_text(&p) {
                if !t.is_empty() {
                    out.push(json!({
                        "id": self.id, "object": "chat.completion.chunk",
                        "created": crate::translate::now_ts(), "model": self.model,
                        "choices": [{"index": 0,
                            "delta": {"content": t},
                            "finish_reason": Value::Null}]
                    }));
                }
            }
        }
        if let Some(fr) = cand
            .get("finishReason")
            .or_else(|| cand.get("finish_reason"))
            .and_then(|x| x.as_str())
        {
            // Gemini repeats the finish reason on trailing chunks; only the
            // first terminal signal closes the stream.
            if self.finish.is_none() {
                let (p, c, cached, r) = google_usage_parts(ev);
                if ev.get("usageMetadata").is_some() {
                    self.in_tok = p;
                    self.out_tok = c;
                    self.cached = cached;
                    self.reasoning_tok = r;
                }
                let has_tools = self.tool_index > 0;
                let finish = google_finish(fr, has_tools).to_string();
                self.finish = Some(finish.clone());
                let mut chunk = json!({
                    "id": self.id, "object": "chat.completion.chunk",
                    "created": crate::translate::now_ts(), "model": self.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]
                });
                chunk["usage"] = chat_usage(self.in_tok, self.out_tok, self.cached, self.reasoning_tok);
                out.push(chunk);
            } else if ev.get("usageMetadata").is_some() {
                let (p, c, cached, r) = google_usage_parts(ev);
                self.in_tok = p;
                self.out_tok = c;
                self.cached = cached;
                self.reasoning_tok = r;
            }
        } else if ev.get("usageMetadata").is_some() {
            let (p, c, cached, r) = google_usage_parts(ev);
            self.in_tok = p;
            self.out_tok = c;
            self.cached = cached;
            self.reasoning_tok = r;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Streaming: chat chunks -> Google SSE (egress)
// ---------------------------------------------------------------------------

/// Stateful translator for canonical chat chunks into Gemini
/// `GenerateContentResponse` SSE events (`data:` only, no `event:` name).
pub struct ChatToGoogleStream {
    model: String,
    tools: Vec<PendingEgressTool>,
    terminated: bool,
}

impl ChatToGoogleStream {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            tools: vec![],
            terminated: false,
        }
    }

    fn response(&self, parts: Vec<Value>, finish: Option<&str>) -> SseEvent {
        let mut cand = json!({
            "content": {"role": "model", "parts": parts},
            "index": 0
        });
        if let Some(f) = finish {
            cand["finishReason"] = json!(f);
        }
        SseEvent::data(json!({"candidates": [cand], "modelVersion": self.model}))
    }

    pub fn handle(&mut self, ch: &Value) -> Vec<SseEvent> {
        let mut out = vec![];
        let choice = ch.pointer("/choices/0").cloned().unwrap_or(json!({}));
        let delta = choice.get("delta").cloned().unwrap_or(json!({}));

        if let Some(t) = delta.get("reasoning_content").and_then(|x| x.as_str()) {
            if !t.is_empty() {
                out.push(self.response(vec![json!({"thought": true, "text": t})], None));
            }
        }
        if let Some(t) = delta.get("content").and_then(|x| x.as_str()) {
            if !t.is_empty() {
                out.push(self.response(vec![json!({"text": t})], None));
            }
        }
        if let Some(tcs) = delta.get("tool_calls").and_then(|x| x.as_array()) {
            for tc in tcs {
                accumulate_egress_tool(&mut self.tools, tc);
            }
        }
        if let Some(fr) = choice.get("finish_reason").and_then(|x| x.as_str()) {
            let mut parts: Vec<Value> = vec![];
            let flushed = std::mem::take(&mut self.tools);
            for t in flushed {
                let args: Value =
                    serde_json::from_str(&t.args).unwrap_or(json!({}));
                parts.push(json!({"functionCall": {"name": t.name, "args": args}}));
            }
            let finish_reason = match fr {
                "length" => "MAX_TOKENS",
                "content_filter" => "SAFETY",
                "error" => "MALFORMED_FUNCTION_CALL",
                _ => "STOP",
            };
            if parts.is_empty() {
                // Terminal carries usage; empty parts keep the shape valid.
                let mut ev = self.response(vec![], Some(finish_reason));
                attach_usage(&mut ev, ch);
                out.push(ev);
            } else {
                let mut ev = self.response(parts, Some(finish_reason));
                attach_usage(&mut ev, ch);
                out.push(ev);
            }
            self.terminated = true;
        }
        out
    }

    pub fn finish(&mut self) -> Option<Vec<SseEvent>> {
        if self.terminated {
            return None;
        }
        let mut out = vec![];
        let mut parts: Vec<Value> = vec![];
        let flushed = std::mem::take(&mut self.tools);
        for t in flushed {
            let args: Value = serde_json::from_str(&t.args).unwrap_or(json!({}));
            parts.push(json!({"functionCall": {"name": t.name, "args": args}}));
        }
        self.terminated = true;
        out.push(self.response(parts, Some("STOP")));
        Some(out)
    }

    pub fn abort(&mut self) -> Vec<SseEvent> {
        if self.terminated {
            return vec![];
        }
        let _ = std::mem::take(&mut self.tools);
        vec![]
    }
}

fn attach_usage(ev: &mut SseEvent, ch: &Value) {
    let (p, c, cached, reasoning) = crate::translate::chat_usage_parts(ch.get("usage"));
    ev.data["usageMetadata"] = json!({
        "promptTokenCount": p,
        "candidatesTokenCount": c,
        "totalTokenCount": p + c,
        "cachedContentTokenCount": cached,
        "thoughtsTokenCount": reasoning
    });
}

// ---------------------------------------------------------------------------
// Local stream helpers (kept here so translate.rs stays untouched)
// ---------------------------------------------------------------------------

fn role_chunk(id: &str, model: &str) -> Value {
    json!({
        "id": id, "object": "chat.completion.chunk",
        "created": crate::translate::now_ts(), "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
    })
}

#[derive(Default)]
struct PendingEgressTool {
    id: String,
    name: String,
    args: String,
}

fn accumulate_egress_tool(tools: &mut Vec<PendingEgressTool>, tc: &Value) {
    // Prefer the upstream `index` when present; otherwise append to the
    // last open slot (matches the Anthropic/Responses egress policy).
    if let Some(i) = tc.get("index").and_then(|x| x.as_i64()) {
        let idx = i as usize;
        while tools.len() <= idx {
            tools.push(PendingEgressTool::default());
        }
        let e = &mut tools[idx];
        if let Some(id) = tc.get("id").and_then(|x| x.as_str()).filter(|s| !s.is_empty()) {
            e.id = id.to_string();
        }
        if let Some(n) = tc.pointer("/function/name").and_then(|x| x.as_str()).filter(|s| !s.is_empty()) {
            e.name = n.to_string();
        }
        if let Some(a) = tc.pointer("/function/arguments").and_then(|x| x.as_str()) {
            e.args.push_str(a);
        }
        return;
    }
    let id = tc.get("id").and_then(|x| x.as_str()).unwrap_or("");
    if !id.is_empty() && !tools.iter().any(|t| t.id == id) {
        tools.push(PendingEgressTool::default());
    } else if tools.is_empty() {
        tools.push(PendingEgressTool::default());
    }
    let e = tools.last_mut().unwrap();
    if !id.is_empty() {
        e.id = id.to_string();
    }
    if let Some(n) = tc.pointer("/function/name").and_then(|x| x.as_str()).filter(|s| !s.is_empty()) {
        e.name = n.to_string();
    }
    if let Some(a) = tc.pointer("/function/arguments").and_then(|x| x.as_str()) {
        e.args.push_str(a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::state::{config_for, test_state};
    use crate::upstream::{upstream_headers, upstream_url};

    fn chat_req() -> Value {
        json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"}
            ],
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "description": "weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
            }}],
            "tool_choice": "auto",
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 100,
            "stop": ["done"],
            "reasoning_effort": "high"
        })
    }

    #[test]
    fn action_parsing_splits_model_and_verb() {
        assert_eq!(
            parse_google_action("gemini-2.5-flash:generateContent"),
            Some(("gemini-2.5-flash".to_string(), false))
        );
        assert_eq!(
            parse_google_action("gemini-2.5-flash:streamGenerateContent"),
            Some(("gemini-2.5-flash".to_string(), true))
        );
        assert_eq!(parse_google_action("no-verb"), None);
        assert_eq!(parse_google_action("m:countTokens"), None);
        assert_eq!(parse_google_action(":generateContent"), None);
    }

    #[test]
    fn chat_request_maps_system_tools_sampling_and_thinking() {
        let out = chat_to_google_request(&chat_req());
        assert_eq!(
            out["system_instruction"],
            json!({"parts": [{"text": "be brief"}]})
        );
        let contents = out["contents"].as_array().unwrap();
        // user + assistant(functionCall) + tool(functionResponse)
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(
            contents[1]["parts"][0]["functionCall"]["name"],
            json!("get_weather")
        );
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["name"],
            json!("get_weather")
        );
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["response"],
            json!({"result": "sunny"})
        );
        let decls = out["tools"][0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(decls[0]["name"], json!("get_weather"));
        assert_eq!(
            out["toolConfig"]["functionCallingConfig"]["mode"],
            json!("AUTO")
        );
        let gen = &out["generationConfig"];
        assert_eq!(gen["temperature"], json!(0.5));
        assert_eq!(gen["topP"], json!(0.9));
        assert_eq!(gen["maxOutputTokens"], json!(100));
        assert_eq!(gen["stopSequences"], json!(["done"]));
        assert_eq!(gen["thinkingConfig"]["thinkingBudget"], json!(16384));
        assert!(chat_to_google_loss(&chat_req()).is_empty());
        assert!(chat_to_google_fatal(&chat_req()).is_none());
    }

    #[test]
    fn chat_request_maps_vision_and_structured_output() {
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "image_url", "image_url": {"url": "https://x/i.png"}}
            ]}],
            "response_format": {"type": "json_schema",
                "json_schema": {"name": "s", "schema": {"type": "object"}, "strict": true}}
        });
        let out = chat_to_google_request(&req);
        let parts = out["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts[0], json!({"text": "look"}));
        assert_eq!(
            parts[1]["inlineData"],
            json!({"mimeType": "image/png", "data": "AA=="})
        );
        assert_eq!(parts[2]["fileData"]["fileUri"], json!("https://x/i.png"));
        let gen = &out["generationConfig"];
        assert_eq!(gen["responseMimeType"], json!("application/json"));
        assert_eq!(gen["responseSchema"]["type"], json!("OBJECT"));
    }

    #[test]
    fn named_tool_choice_maps_to_any_with_allowlist() {
        let mut req = chat_req();
        req["tool_choice"] = json!({"type": "function", "function": {"name": "get_weather"}});
        let out = chat_to_google_request(&req);
        assert_eq!(
            out["toolConfig"]["functionCallingConfig"],
            json!({"mode": "ANY", "allowedFunctionNames": ["get_weather"]})
        );
    }

    #[test]
    fn loss_and_fatal_report_untranslatable_params() {
        let req = json!({"model": "m", "messages": [], "n": 2, "user": "u", "logprobs": true});
        assert!(chat_to_google_loss(&req).contains(&"user"));
        assert!(chat_to_google_loss(&req).contains(&"logprobs"));
        assert!(chat_to_google_fatal(&req).is_some());
    }

    #[test]
    fn nameless_tools_are_dropped_not_emptied() {
        // Open WebUI's code interpreter arrives as a bare non-function
        // tool; translating it to an empty-name declaration makes the
        // upstream reject the request with a 400.
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "say hi"}],
            "tools": [{"type": "code_interpreter"}]
        });
        let out = chat_to_google_request(&req);
        assert!(out.get("tools").is_none());
        assert!(
            chat_to_google_loss(&req)
                .contains(&"nameless tools (dropped: not function tools)")
        );

        // ... while real function tools alongside it still pass through.
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "say hi"}],
            "tools": [
                {"type": "code_interpreter"},
                {"type": "function", "function": {"name": "get_time", "description": "t"}}
            ]
        });
        let out = chat_to_google_request(&req);
        let decls = out["tools"][0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0]["name"], json!("get_time"));
    }

    #[test]
    fn additional_properties_stripped_from_tool_schemas() {
        // Google rejects `additionalProperties` in function declarations
        // with a 400; it must be stripped at every depth, keeping the rest.
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "say hi"}],
            "tools": [{"type": "function", "function": {
                "name": "ask_user",
                "description": "ask",
                "parameters": {"type": "object",
                    "properties": {"items": {"type": "array",
                        "items": {"type": "object", "additionalProperties": true}}}}
            }}]
        });
        let out = chat_to_google_request(&req);
        let params = &out["tools"][0]["functionDeclarations"][0]["parameters"];
        assert_eq!(
            params["properties"]["items"]["items"],
            json!({"type": "object"})
        );
        assert!(!has_additional_properties(params));
        assert!(
            chat_to_google_loss(&req)
                .contains(&"tool additionalProperties stripped (unsupported by Google)")
        );

        // Schemas without it are untouched and report no loss.
        let clean = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "say hi"}],
            "tools": [{"type": "function", "function": {
                "name": "get_time",
                "parameters": {"type": "object", "properties": {}}
            }}]
        });
        let out = chat_to_google_request(&clean);
        assert_eq!(
            out["tools"][0]["functionDeclarations"][0]["parameters"],
            json!({"type": "object", "properties": {}})
        );
        assert!(
            !chat_to_google_loss(&clean)
                .contains(&"tool additionalProperties stripped (unsupported by Google)")
        );
    }

    #[test]
    fn thought_signature_round_trips_for_tool_calls() {
        // Thinking models attach thoughtSignature as a sibling of
        // functionCall parts and require it back on follow-ups.
        let resp = json!({"candidates": [{
            "content": {"role": "model", "parts": [
                {"thoughtSignature": "sig123",
                 "functionCall": {"name": "f", "args": {"x": 1}}}
            ]},
            "finishReason": "STOP", "index": 0
        }]});

        // Outbound non-stream: signature rides extra_content.
        let out = google_to_chat(&resp, "m");
        assert_eq!(
            out["choices"][0]["message"]["tool_calls"][0]["extra_content"],
            json!({"google": {"thought_signature": "sig123"}})
        );

        // Outbound SSE: same contract on the tool-call delta.
        let mut inbound = GoogleStream::new("m");
        let chunks = inbound.handle(&resp);
        let delta = chunks
            .iter()
            .find(|c| c.pointer("/choices/0/delta/tool_calls").is_some())
            .unwrap();
        assert_eq!(
            delta.pointer("/choices/0/delta/tool_calls/0/extra_content"),
            Some(&json!({"google": {"thought_signature": "sig123"}}))
        );

        // Inbound: extra_content becomes the sibling thoughtSignature.
        let req = json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "f", "arguments": "{}"},
                     "extra_content": {"google": {"thought_signature": "sig123"}}}
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}
            ]
        });
        let out = chat_to_google_request(&req);
        let parts = out["contents"][1]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionCall"]["name"], json!("f"));
        assert_eq!(parts[0]["thoughtSignature"], json!("sig123"));

        // Absent signature stays absent (no empty field sent upstream).
        let req = json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "f", "arguments": "{}"}}
                ]}
            ]
        });
        let out = chat_to_google_request(&req);
        assert!(
            out["contents"][1]["parts"][0]
                .get("thoughtSignature")
                .is_none()
        );
    }

    #[test]
    fn google_response_maps_text_tools_thought_and_usage() {
        let resp = json!({
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"thought": true, "text": "hmm"},
                    {"text": "hi"},
                    {"functionCall": {"name": "f", "args": {"x": 1}}}
                ]},
                "finishReason": "STOP", "index": 0
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5,
                "totalTokenCount": 15, "cachedContentTokenCount": 2, "thoughtsTokenCount": 3}
        });
        let out = google_to_chat(&resp, "m");
        let msg = &out["choices"][0]["message"];
        assert_eq!(msg["reasoning_content"], json!("hmm"));
        assert_eq!(msg["content"], Value::Null);
        assert_eq!(msg["tool_calls"][0]["function"]["name"], json!("f"));
        assert_eq!(msg["tool_calls"][0]["function"]["arguments"], json!("{\"x\":1}"));
        assert_eq!(out["choices"][0]["finish_reason"], json!("tool_calls"));
        assert_eq!(out["usage"]["prompt_tokens"], json!(10));
        assert_eq!(out["usage"]["completion_tokens"], json!(5));
        assert_eq!(out["usage"]["completion_tokens_details"]["reasoning_tokens"], json!(3));
        assert!(out["upstream_error"].is_null());

        for (fr, want) in [
            ("MAX_TOKENS", "length"),
            ("SAFETY", "content_filter"),
            ("RECITATION", "content_filter"),
        ] {
            let mut r = resp.clone();
            r["candidates"][0]["finishReason"] = json!(fr);
            r["candidates"][0]["content"]["parts"] = json!([{"text": "x"}]);
            let o = google_to_chat(&r, "m");
            assert_eq!(o["choices"][0]["finish_reason"], json!(want), "{fr}");
        }

        let mut bad = resp.clone();
        bad["candidates"][0]["finishReason"] = json!("MALFORMED_FUNCTION_CALL");
        let o = google_to_chat(&bad, "m");
        assert_eq!(o["choices"][0]["finish_reason"], json!("error"));
        assert!(o["upstream_error"].is_string());
    }

    #[test]
    fn google_ingress_request_maps_to_chat() {
        let req = json!({
            "system_instruction": {"parts": [{"text": "sys"}]},
            "contents": [
                {"role": "user", "parts": [{"text": "hi"}]},
                {"role": "model", "parts": [
                    {"thought": true, "text": "t"},
                    {"functionCall": {"name": "f", "args": {"x": 1}}}
                ]},
                {"role": "user", "parts": [
                    {"functionResponse": {"name": "f", "response": {"result": "ok"}}}
                ]}
            ],
            "generationConfig": {"temperature": 0.2, "topP": 0.8,
                "maxOutputTokens": 7, "stopSequences": ["s"],
                "responseMimeType": "application/json",
                "responseSchema": {"type": "OBJECT", "properties": {}},
                "thinkingConfig": {"thinkingBudget": 16384}},
            "tools": [{"functionDeclarations": [
                {"name": "f", "description": "d", "parameters": {"type": "OBJECT"}}
            ]}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}
        });
        let out = google_to_chat_request(&req, "m");
        assert_eq!(out["model"], json!("m"));
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[0], json!({"role": "system", "content": "sys"}));
        assert_eq!(msgs[1]["content"], json!("hi"));
        assert_eq!(msgs[2]["tool_calls"][0]["function"]["name"], json!("f"));
        assert_eq!(msgs[2]["reasoning_content"], json!("t"));
        assert_eq!(msgs[3]["role"], json!("tool"));
        assert_eq!(msgs[3]["content"], json!("ok"));
        assert_eq!(out["temperature"], json!(0.2));
        assert_eq!(out["max_tokens"], json!(7));
        assert_eq!(out["reasoning_effort"], json!("high"));
        assert_eq!(out["response_format"]["type"], json!("json_schema"));
        assert_eq!(out["tool_choice"], json!("required"));
        assert_eq!(out["tools"][0]["function"]["name"], json!("f"));
    }

    #[test]
    fn chat_completion_maps_to_google_response() {
        let chat = json!({
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": null,
                "reasoning_content": "hmm",
                "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "f", "arguments": "{\"x\":1}"}}]
            }}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 6,
                "prompt_tokens_details": {"cached_tokens": 1},
                "completion_tokens_details": {"reasoning_tokens": 2}}
        });
        let out = chat_to_google_response(&chat, "m");
        let parts = out["candidates"][0]["content"]["parts"].as_array().unwrap();
        assert_eq!(parts[0], json!({"thought": true, "text": "hmm"}));
        assert_eq!(parts[1]["functionCall"]["name"], json!("f"));
        assert_eq!(parts[1]["functionCall"]["args"], json!({"x": 1}));
        assert_eq!(out["candidates"][0]["finishReason"], json!("STOP"));
        assert_eq!(out["usageMetadata"]["promptTokenCount"], json!(4));
        assert_eq!(out["usageMetadata"]["thoughtsTokenCount"], json!(2));
    }

    #[test]
    fn stream_round_trips_text_thought_and_tools() {
        // Upstream Google SSE -> chat chunks.
        let mut inbound = GoogleStream::new("m");
        let ev = json!({"candidates": [{
            "content": {"role": "model", "parts": [
                {"thought": true, "text": "h"},
                {"text": "hi"},
                {"functionCall": {"name": "f", "args": {"x": 1}}}
            ]},
            "finishReason": "STOP", "index": 0
        }],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2,
            "totalTokenCount": 5}});
        let chunks = inbound.handle(&ev);
        assert!(chunks.iter().any(|c| c.pointer("/choices/0/delta/content")
            .and_then(|x| x.as_str())
            == Some("hi")));
        assert!(chunks.iter().any(|c| c.pointer("/choices/0/delta/reasoning_content")
            .and_then(|x| x.as_str())
            == Some("h")));
        let last = chunks.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], json!("tool_calls"));
        assert_eq!(last["usage"]["prompt_tokens"], json!(3));

        // Chat chunks -> Google SSE.
        let mut egress = ChatToGoogleStream::new("m");
        let mut events: Vec<SseEvent> = vec![];
        events.extend(egress.handle(&json!({"choices": [{"index": 0,
            "delta": {"content": "he"}, "finish_reason": Value::Null}]})));
        events.extend(egress.handle(&json!({"choices": [{"index": 0,
            "delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                "function": {"name": "f", "arguments": "{\"x\":"}}]},
            "finish_reason": Value::Null}]})));
        events.extend(egress.handle(&json!({"choices": [{"index": 0,
            "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]},
            "finish_reason": Value::Null}]})));
        events.extend(egress.handle(&json!({"choices": [{"index": 0,
            "delta": {}, "finish_reason": "stop"}]})));
        assert!(events.iter().all(|e| e.event.is_none()));
        let texts: String = events
            .iter()
            .filter_map(|e| {
                e.data.pointer("/candidates/0/content/parts/0/text")
                    .and_then(|x| x.as_str())
            })
            .collect();
        assert_eq!(texts, "he");
        let last = events.last().unwrap();
        assert_eq!(
            last.data.pointer("/candidates/0/finishReason"),
            Some(&json!("STOP"))
        );
        assert_eq!(
            last.data.pointer("/candidates/0/content/parts/0/functionCall/name"),
            Some(&json!("f"))
        );
        assert!(egress.finish().is_none());
    }

    // ---- auth + URL ------------------------------------------------------

    fn google_model(base: &str, auth: crate::config::GoogleAuth) -> crate::config::ModelCfg {
        let mut m = crate::config::test_model("m", crate::config::Protocol::Google, base);
        m.google_auth = auth;
        m
    }

    #[test]
    fn api_key_auth_sends_x_goog_api_key_without_bearer() {
        let st = test_state(config_for(
            "https://generativelanguage.googleapis.com/v1beta",
            vec![google_model(
                "https://generativelanguage.googleapis.com/v1beta",
                crate::config::GoogleAuth::Auto,
            )],
        ));
        let m = google_model(
            "https://generativelanguage.googleapis.com/v1beta",
            crate::config::GoogleAuth::Auto,
        );
        let h = upstream_headers(&st, &m);
        assert!(h.contains_key("x-goog-api-key"));
        assert!(!h.contains_key(axum::http::header::AUTHORIZATION));
    }

    #[test]
    fn vertex_base_url_sniffs_bearer_and_override_wins() {
        let vertex = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models";
        let st = test_state(config_for(
            vertex,
            vec![google_model(vertex, crate::config::GoogleAuth::Auto)],
        ));
        let auto = google_model(vertex, crate::config::GoogleAuth::Auto);
        let h = upstream_headers(&st, &auto);
        assert!(h.contains_key(axum::http::header::AUTHORIZATION));
        assert!(!h.contains_key("x-goog-api-key"));

        let forced = google_model(vertex, crate::config::GoogleAuth::ApiKey);
        let h = upstream_headers(&st, &forced);
        assert!(h.contains_key("x-goog-api-key"));
        assert!(!h.contains_key(axum::http::header::AUTHORIZATION));
    }

    #[test]
    fn upstream_url_embeds_model_and_stream_verb() {
        let m = google_model("https://generativelanguage.googleapis.com/v1beta/", crate::config::GoogleAuth::Auto);
        assert_eq!(
            upstream_url("https://generativelanguage.googleapis.com/v1beta/", &m, false),
            "https://generativelanguage.googleapis.com/v1beta/models/m:generateContent"
        );
        assert_eq!(
            upstream_url("https://generativelanguage.googleapis.com/v1beta/", &m, true),
            "https://generativelanguage.googleapis.com/v1beta/models/m:streamGenerateContent?alt=sse"
        );
    }
}
