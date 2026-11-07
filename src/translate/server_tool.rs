//! Server-side tools: web search and code execution.
//!
//! Unlike function tools these run at the upstream provider, not in the
//! client. The canonical chat hub represents them as top-level `tools`
//! entries and reports their results on the assistant message:
//!
//! ```jsonc
//! // request
//! "tools": [
//!   {"type": "web_search", "allowed_domains": ["example.com"]},
//!   {"type": "code_execution"}
//! ]
//! // response message
//! "server_tool_calls": [
//!   {"id": "srv_...", "type": "web_search", "status": "completed",
//!    "query": "...", "results": [{"url": "...", "title": "..."}]},
//!   {"id": "srv_...", "type": "code_execution", "status": "completed",
//!    "code": "...", "output": "..."}
//! ],
//! "citations": [
//!   {"url": "...", "title": "...", "cited_text": "...",
//!    "start_index": 10, "end_index": 20}
//! ]
//! ```
//!
//! Streams carry the same objects one at a time in `delta.server_tool_calls`
//! and `delta.citations`. Providers execute these tools themselves, so the
//! canonical finish reason stays `stop`: the client has nothing to run.

use serde_json::{json, Value};

use super::common::gen_id;

pub const WEB_SEARCH: &str = "web_search";
pub const CODE_EXECUTION: &str = "code_execution";

/// Canonical server-tool kind for a canonical `tools` entry.
///
/// The alias names are accepted because a Chat-shaped client may ask for the
/// provider's hosted tool directly (`web_search_preview`,
/// `code_interpreter`); they mean the same thing at the provider.
pub fn kind_of(tool: &Value) -> Option<&'static str> {
    match tool.get("type").and_then(|x| x.as_str())? {
        WEB_SEARCH | "web_search_preview" => Some(WEB_SEARCH),
        CODE_EXECUTION | "code_interpreter" => Some(CODE_EXECUTION),
        _ => None,
    }
}

/// A canonical server-tool entry for the request `tools` array.
pub fn request_entry(kind: &str, allowed_domains: Vec<Value>) -> Value {
    let mut v = json!({"type": kind});
    if !allowed_domains.is_empty() {
        v["allowed_domains"] = Value::Array(allowed_domains);
    }
    v
}

/// `allowed_domains` as a JSON string array (empty when absent or malformed).
pub fn allowed_domains(v: &Value) -> Vec<Value> {
    v.get("allowed_domains")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter(|d| d.is_string()).cloned().collect())
        .unwrap_or_default()
}

/// A completed web-search call for the canonical message.
pub fn web_search(query: &str, results: Vec<Value>) -> Value {
    json!({
        "id": gen_id("srv_"),
        "type": WEB_SEARCH,
        "status": "completed",
        "query": query,
        "results": results
    })
}

/// A completed code-execution call for the canonical message.
pub fn code_execution(code: &str, output: &str) -> Value {
    json!({
        "id": gen_id("srv_"),
        "type": CODE_EXECUTION,
        "status": "completed",
        "code": code,
        "output": output
    })
}

/// One canonical citation. `start_index`/`end_index` are filled in by
/// [`locate_citations`] once the full answer text is known.
pub fn citation(url: &str, title: &str, cited_text: &str) -> Value {
    json!({"url": url, "title": title, "cited_text": cited_text})
}

/// Normalize citation offsets to character offsets into the assembled text.
///
/// Providers that report spans keep them; providers that only report the
/// cited snippet (Anthropic, Responses when a client rebuilds) get offsets
/// from the first occurrence.
pub fn locate_citations(citations: &mut [Value], text: &str) {
    for c in citations {
        if c.get("start_index").is_some() {
            continue;
        }
        let cited = c.get("cited_text").and_then(|x| x.as_str()).unwrap_or("");
        if cited.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(cited) {
            let start = text[..pos].chars().count();
            let end = start + cited.chars().count();
            c["start_index"] = json!(start);
            c["end_index"] = json!(end);
        }
    }
}

/// Search results derived from citations, for providers that report no
/// explicit result list (OpenAI Responses works this way).
pub fn results_from_citations(citations: &[Value]) -> Vec<Value> {
    let mut out: Vec<Value> = vec![];
    for c in citations {
        let url = c.get("url").and_then(|x| x.as_str()).unwrap_or("");
        if url.is_empty()
            || out
                .iter()
                .any(|r| r.get("url").and_then(|x| x.as_str()) == Some(url))
        {
            continue;
        }
        out.push(json!({
            "url": url,
            "title": c.get("title").cloned().unwrap_or(json!(""))
        }));
    }
    out
}
