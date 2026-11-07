# Tool-call translation limits

When a request crosses from one protocol to another, the gateway rebuilds
tool declarations, tool calls and tool results from the subset that the
canonical Chat shape can express. Anything outside that subset is dropped.
The client still receives a successful response, so the loss is silent from
the client's point of view; some drops are logged to stderr, but not all.

Same-protocol passthrough is exempt. When the client's protocol matches the
model's `protocol`, the body is forwarded verbatim, so `strict`,
`cache_control`, `tool_result.is_error`, provider built-in tools and native
call ids all survive. Every loss described here applies only to translated
paths.

## What is carried

| Object | Fields carried |
| --- | --- |
| Function tool declaration | `name`, `description`, `parameters` (JSON Schema) |
| Function call | `id`, `name`, `arguments` |
| Function result | text only |
| `tool_choice` | `auto`, `required` / `any`, `none`, a named function tool |
| Server tools | `web_search`, `code_execution` |
| Server-tool output | query/code plus synthesized results, output and citations |

Server tools run at the provider. `web_search` carries an `allowed_domains`
filter to Anthropic and Responses upstreams; Gemini has no equivalent filter.
`code_execution` has no configurable options on any translated path.

## Dropped tool entries

Tool entries that have no name and no supported server-tool mapping are
dropped. These drops are logged as `dropped unsupported tools` (or as part of
the translated-path loss list).

| Source surface | Entry | Result |
| --- | --- | --- |
| Responses | `web_search` / `web_search_preview` | carried as `web_search` |
| Responses | `code_interpreter` | carried as `code_execution` |
| Responses | named `function` | carried as a function tool |
| Responses | named `custom` | **coerced to a function tool**; the `format` / grammar is dropped and `parameters` falls back to `{"type":"object"}` |
| Responses | `file_search`, `image_generation`, `mcp`, `computer_use_preview`, `local_shell`, … | dropped |
| Anthropic | `web_search_*`, `code_execution_*` | carried as server tools |
| Anthropic | other named tool types (`bash_*`, `text_editor_*`, custom types) | **coerced to a function tool** from `name` + `input_schema`; provider semantics are not preserved |
| Gemini | `functionDeclarations` | carried as function tools |
| Gemini | `googleSearch`, `codeExecution` | carried as server tools |
| Gemini | `fileSearch`, `urlContext`, `googleSearchRetrieval`, `computerUse`, `mcpServers`, … | dropped |
| Chat | any entry without a non-empty `function.name` | dropped on every translated egress |

A Responses `custom` tool is the most misleading case: it has a name, so it is
not filtered. The upstream then receives an ordinary JSON-schema function
where the client asked for a free-form tool, and the model is no longer
constrained by the custom format.

Gemini entries can combine several tool kinds, but the gateway reads one kind
per entry. `googleSearch` / `codeExecution` take precedence; otherwise the
entry is read as `functionDeclarations`. Any sibling kind is ignored without a
separate warning.

## Dropped declaration fields

Every translated egress rebuilds a function tool from `name`, `description`
and `parameters` only. These fields are therefore lost:

| Field | Where it is lost | Consequence |
| --- | --- | --- |
| `strict` | Chat → Anthropic, Chat → Responses, Chat → Gemini | schema conformance is no longer enforced by the upstream |
| vendor extensions on the tool object | all translated paths | unknown fields are not forwarded |
| Anthropic `cache_control` | Anthropic → Chat → any upstream | prompt-cache breakpoints on tools are lost |
| Anthropic `defer_loading`, `input_examples`, `eager_input_streaming` | Anthropic → Chat | Anthropic-specific tool behavior is lost |
| Responses `filters` beyond `allowed_domains` | Responses → Chat → any upstream | `search_context_size`, `user_location`, … are lost |
| Responses `container` on `code_interpreter` | Responses → Chat → any upstream | the gateway always emits `{"type":"auto"}` |

Gemini additionally rejects a large part of JSON Schema. The gateway strips
these keys at every depth of a tool parameter schema:

```
additionalProperties  $schema  $id  $defs  $anchor  $comment
$dynamicRef  $recursiveRef  definitions  $ref  strict  examples
const  exclusiveMinimum  exclusiveMaximum  multipleOf  uniqueItems
readOnly  writeOnly  contentMediaType  contentEncoding  prefixItems
contains  propertyNames  unevaluatedProperties  if  then  else
dependentRequired
```

This is not cosmetic. Removing `additionalProperties: false` widens the
accepted argument shape, and removing `$ref` / `$defs` can flatten or break a
recursive schema. The strip is reported as `tool schema keys stripped`.

## `tool_choice` and parallel calls

`auto`, `required` / `any`, `none` and a named function tool map across all
four protocols. Two limits remain:

- **Forced choices are checked after filtering.** If every function tool was
  dropped, `required` is rejected with HTTP 400 instead of being forwarded as
  a contradictory request. A named `tool_choice` that points at a dropped
  tool is rejected the same way. `auto` / `none` still pass.
- **`parallel_tool_calls` is not universal.** It is forwarded to Responses
  upstreams, dropped for Google (and logged), and dropped for Anthropic
  without a warning. Anthropic's native equivalent,
  `tool_choice.disable_parallel_tool_use`, is never emitted, and is dropped
  when it arrives from an Anthropic client.

A forced `tool_choice` with only server tools is rejected too: `required`
counts function tools, not provider tools.

## Dropped call and result features

| Feature | Behavior on translated paths |
| --- | --- |
| Tool-result images, documents, audio | dropped; only text is concatenated |
| Anthropic `tool_result.is_error` | dropped; the error flag is not represented in canonical Chat |
| Structured tool output | reduced to text; Chat → Gemini wraps a non-object string as `{"result": ...}`, an object passes through |
| Responses built-in calls other than web search / code interpreter | dropped from the response (`file_search_call`, `image_generation_call`, `mcp_call`, `computer_call`, `local_shell_call`, `custom_tool_call`) |
| Anthropic server tools other than web search / code execution | dropped from the response |
| Gemini call ids | none exist; the gateway synthesizes `call_*` ids on the way to Chat |
| Gemini `thoughtSignature` | carried only through Chat `extra_content.google.thought_signature`; lost when either side is Anthropic or Responses |

The Gemini id synthesis has a matching limit. `functionResponse` carries only
a function name, so two parallel calls to the same function cannot be paired
reliably when the history is sent back. On the way to Gemini the gateway looks
up the name from the preceding assistant `tool_call_id`; if the lookup fails,
it falls back to the call id as the function name.

`thoughtSignature` is a hard requirement in the follow-up turn of thinking
Gemini models. A Chat client can echo it, but an Anthropic or Responses client
has nowhere to store it, so a Gemini thinking tool loop that crosses protocols
may be rejected by the upstream on the next turn.

## Observability

- Whole-entry drops and unsatisfiable forced choices are logged, and the
  latter also return HTTP 400.
- Field-level losses (`strict`, `cache_control`, Gemini schema keys,
  tool-result media) are not logged individually.
- `strict_params = true` escalates selected request-shape parameters such as
  `n > 1`; it does not restore or reject lost tool features.

## Avoiding the loss

- Keep the client protocol equal to the model's upstream protocol when the
  full tool surface matters.
- Prefer function tools with `name` / `description` / `parameters` only.
- Do not rely on `strict`, `additionalProperties: false` or recursive schemas
  when the upstream is Gemini.
- Do not force a tool call when only built-in tools are configured.
- Keep Gemini `thoughtSignature` on the same Chat path that received it, or
  avoid cross-protocol tool loops with thinking Gemini models.
