# Reasoning-state replay

Anthropic and OpenAI Responses both return an opaque, encrypted token that
lets the model resume its own reasoning on a later turn. Anthropic puts it in
`thinking.signature`; Responses puts it in a reasoning item's
`encrypted_content`. The two tokens are not interchangeable, and neither
upstream accepts a forged or re-encoded one.

This gateway carries each token across a protocol change without decoding it,
so a client that only speaks one protocol can still run a multi-turn agent
loop against an upstream that speaks another. The token is wrapped in a
provenance envelope so the gateway can return it only to the family that
issued it.

## Where the token lives

| Surface | Visible reasoning | Opaque token |
| --- | --- | --- |
| Anthropic Messages | `thinking.thinking` | `thinking.signature` |
| Anthropic Messages | — | `redacted_thinking.data` |
| OpenAI Responses | reasoning item `summary` | reasoning item `encrypted_content` |
| Canonical Chat | `reasoning_content` | `reasoning_details[]` |
| Gemini | thought parts (`{"thought": true}`) | — (no equivalent opaque token) |

Canonical `reasoning_details[]` holds one entry per upstream reasoning block:

```jsonc
{"type": "thinking", "thinking": "…", "signature": "nv1:a:…"}
{"type": "thinking", "thinking": "", "signature": "nv1:o:…"}
{"type": "redacted_thinking", "data": "…"}
```

An entry may also carry `id` (the Responses reasoning item id) and
`tool_calls_before` (how many assistant tool calls preceded the block) so
interleaved thinking and tool calls can be rebuilt in their original order.

## The provenance envelope

The envelope is a prefix plus the untouched upstream token:

| Prefix | Meaning | Replayed to |
| --- | --- | --- |
| `nv1:a:` | Anthropic `thinking.signature` | Anthropic upstreams only |
| `nv1:o:` | Responses `encrypted_content` | Responses upstreams only |
| `nv1:r:` | Anthropic `redacted_thinking.data` | Anthropic upstreams only |

The envelope is a provenance marker, not encryption and not a MAC. Its only
job is to answer "which upstream family produced this blob?" before the
gateway puts it back on the wire. The token inside is passed byte-for-byte:
no decoding, no re-encoding, no re-encryption.

A bare, unprefixed signature in canonical `thinking` is assumed to be an
Anthropic signature and wrapped as `nv1:a:`. Clients that carry tokens from
another gateway should preserve the envelope to keep provenance intact.

## Direction by direction

### Anthropic upstream → Responses client

- `thinking.signature` → `encrypted_content: "nv1:a:<signature>"`.
- `thinking.thinking` → the reasoning item's `summary[].text` (may be empty).
- `redacted_thinking` → `encrypted_content: "nv1:r:<data>"`.
- On replay, only `nv1:a:` / `nv1:r:` are unwrapped and sent back to the
  Anthropic upstream. An `nv1:o:` blob is left untouched for the client to
  echo, but is never sent to Anthropic.

### Responses upstream → Anthropic client

- `encrypted_content` → `thinking.signature: "nv1:o:<blob>"`.
- `summary[].text` → `thinking.thinking` (an empty summary is valid).
- On replay, only `nv1:o:` is unwrapped and put back into a Responses
  `reasoning` input item. An `nv1:a:` signature is dropped from that request.

### Gemini upstream

Gemini has no opaque reasoning-replay token. Thought text maps to
`reasoning_content` and back, but there is nothing for the gateway to carry
as a signature. Tool calls do carry a separate `thoughtSignature`, documented
in [tool-calls.md](tool-calls.md).

### Same-protocol passthrough

When the client protocol matches the model's protocol, the body is forwarded
verbatim. The gateway does not add or remove envelopes on that path.

## Why the envelope is stateless and deterministic

The gateway keeps no registry, TTL or in-memory map of issued tokens. The
envelope is computed from the token alone, so the same conversation history
always produces the same bytes after a restart, a config reload or a second
gateway instance.

That is a hard requirement on the Anthropic side. Some deployments bind a
thinking signature to the exact conversation prefix and return
`Invalid signature in thinking block. The block is bound to a different
conversation.` when the rebuilt prefix differs from the one that produced it.
Any nondeterministic wrapping — a timestamp, a random nonce, a randomized MAC
or a TTL-evicted registry entry — could change that prefix between turns and
invalidate a previously valid signature. Stateless determinism is the only
safe default when the deployment's binding policy cannot be probed from the
request side.

## Advantages

- **Long agent loops survive a protocol change.** Tool-call turns keep the
  reasoning block and its token, so the model resumes with its own state
  instead of a summary the gateway invented.
- **No server state.** Restarting the gateway, reloading config or running
  several replicas does not break replay.
- **Family-safe.** A Responses blob is never sent to Anthropic and an
  Anthropic signature is never sent to Responses, where either would be
  rejected as invalid ciphertext.
- **No plaintext dependency.** The probe showed that visible thinking text
  does not participate in verification. The gateway may carry a summary, an
  empty string or no visible text at all; only the opaque token has to be
  exact.
- **Cheap degradation.** If a token is missing, foreign or invalid, dropping
  the whole block is accepted by both upstreams. The gateway never emits
  `signature: ""`, which Anthropic rejects.
- **One-shot recovery.** If an Anthropic or Responses upstream answers 400
  with a signature/decryption error, the gateway retries once with every
  reasoning block removed. The retry is logged and never amplifies traffic.

## Limits

### The token cannot be verified or repaired

Both upstreams perform real cryptographic validation. A changed byte, a
foreign blob or a mismatched item id is a hard 400, not a silent drop. The
gateway never inspects the contents; it can only replay the token verbatim or
omit the block. The one-shot retry converts the failure into a degraded
success, not into a corrected turn.

### Prefix binding is deployment-dependent

Some Anthropic deployments enforce conversation-prefix binding; others do
not. The gateway cannot detect which one it is talking to, so it assumes the
strictest case:

- rebuilt history is append-only and deterministic;
- reasoning blocks are kept 1:1, never merged or duplicated;
- a block kept on one turn is kept on every later turn, and a block dropped
  on one turn stays dropped.

A client that rewrites, truncates or reorders its own history can still break
the prefix. The upstream 400 plus the one-shot retry is the fallback.

### Model, account and route affinity are not enforced ahead of time

The probe found that a signature from one model in a family was accepted by
another model in the same family, and that account affinity was not testable
with a single account. The envelope records only the family, not the
`base_url`, model or API key. Switching routes between turns is therefore
caught by the upstream's 400 and the retry, not by a pre-emptive drop.

### Responses item ids

A Responses reasoning item has both an `id` and its `encrypted_content`. The
probe showed that the id must match the blob when both are present, and that
omitting the id is accepted. The gateway preserves an id when the client
surface can carry one. When the intermediate protocol has no id field (for
example Anthropic), it synthesizes a schema-valid `rs_` id for the replay
request. The real-client β test exercised this path, but the id semantics are
the least portable part of that direction; omitting the id would be the more
conservative wire shape if an upstream rejects it.

### Streaming ordering

Anthropic normally sends `signature_delta` immediately before
`content_block_stop`, and the gateway waits for it before emitting a thinking
block. If a signature only arrives after visible text has already started, the
block cannot be placed before that text any more, so the gateway drops it
rather than emit out-of-order content. A redacted block in the same position
is dropped for the same reason.

### Gemini cannot participate in opaque replay

Gemini returns thought text but no reusable encrypted reasoning token. A
thinking block that crosses to or from Gemini is therefore text-only. This
does not affect the tool-call `thoughtSignature`, which is a separate
mechanism.

### Chat upstreams must opt in

Canonical `reasoning_details` is stripped before a request reaches an
`openai-compatible` Chat upstream, because vanilla Chat APIs may reject the
unknown field. Set `forward_reasoning_details = true` only for upstreams that
understand the extension (for example OpenRouter).

### Client behavior is part of the contract

The token only survives if the client echoes it back on the next turn. A
client that stores only visible text, drops metadata after an error, or
switches provider/model between turns will break the chain. Two clients were
verified end-to-end on 2026-11-07: Claude Code 2.1.319 (β direction) and Codex
0.167.0 (α direction). OpenCode was not part of that verification.

### Length and secret handling

The envelope adds six bytes (`nv1:a:` / `nv1:o:` / `nv1:r:`). No upstream
length limit was established by the probe. The token is not a secret in the
credential sense, but it is model state: request bodies and tokens are not
logged unless `GATEWAY_DEBUG_BODY=1` is set, and that switch is for local
debugging only.

## Degradation rules

| Situation | Behavior |
| --- | --- |
| Reasoning has text but no token | thinking block is omitted; text may still appear as normal assistant text on Chat |
| Token has the wrong family | block is omitted from that upstream request |
| Anthropic signature is empty | never emitted; the block is omitted |
| Upstream rejects the token with a reasoning 400 | retry once without reasoning blocks, then log `[gw] retry … without reasoning blocks` |
| Retry leaves an assistant turn with no content | the empty turn is dropped |
| Client omits the token entirely | upstream runs without replay; no error is synthesized |

## Probe evidence

The design was validated on 2026-11-07 against an Anthropic-protocol upstream
and a real Responses upstream. The full local probe report is not part of this
repository; the table summarizes the conclusions that shaped the rules above.

| Finding | Anthropic | Responses |
| --- | --- | --- |
| Tampered token | 400 `Invalid signature` | 400 `invalid_encrypted_content` |
| Visible plaintext altered | accepted | accepted |
| Whole block omitted | accepted | accepted |
| Empty signature / blob placeholder | rejected | rejected |
| Prefix binding | deployment-dependent | none observed |
| Duplicate block | rejected on binding deployments | accepted |
| Block order | not enforced | not enforced |
| Token replay across models | same family accepted | not fully verified |
| Streaming token | `signature_delta` before block stop | final value in the terminal item |

The remaining unverified areas are cross-account affinity, Anthropic's
`drop_block` beta switch, `redacted_thinking` under a real safety response,
the exact `include` behavior of the first-party OpenAI API, token length
limits, and non-verified clients.

## Related configuration

| Key | Effect |
| --- | --- |
| `thinking_type` (`adaptive` / `enabled`) | shapes the Anthropic request that produces the signature |
| `reasoning_output` (`reasoning_content` / `think_tag` / `both`) | visible rendering on Chat egress only; does not change the token |
| `forward_reasoning_details` | forwards canonical reasoning details to an `openai-compatible` upstream in OpenRouter's schema |

See [../design.md](../design.md) for the full translation policy and
[../configuration.md](../configuration.md) for the per-model keys.
