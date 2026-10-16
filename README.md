# pico-nervogate

A tiny multi-protocol LLM gateway. It sits between your client and an
OpenAI-compatible / Anthropic-compatible upstream, **translates between
protocols** so a client that only speaks one wire format can reach an upstream
that expects another, and **passes configured HTTP headers upstream**.

Built for people who want the routing/conversion behavior of a full proxy
(LiteLLM & friends) without the footprint. The release binary runs in well
under 1 MB of resident memory.

## Features

- **Four protocol surfaces**, one upstream, translated both ways:
  `POST /v1/chat/completions` (OpenAI Chat),
  `POST /v1/responses` (OpenAI Responses),
  `POST /v1/messages` (Anthropic Messages),
  `POST /v1beta/models/{model}:generateContent` /
  `:streamGenerateContent` (Google Gemini).
  Each model's `protocol` picks the upstream wire format (`chat` /
  `anthropic` / `responses` / `google`), so any client surface can reach any upstream.
- **Same-protocol passthrough**: when the client's protocol already matches
  the model's, the request and response are forwarded verbatim — no round trip
  through the canonical chat shape, so `cache_control`, thinking signatures and
  `tool_result.is_error` survive, and the upstream's SSE `event:` names are
  preserved.
- **Error shape**: an upstream failure is rewrapped in the _client's_ protocol
  regardless of what the upstream speaks, keeping its status code (a 429 stays
  a 429) and error object. Gateway-originated failures use the same shape.
- **No prompt logging**: request bodies are never written to the logs.
  `GATEWAY_DEBUG_BODY=1` restores them for local debugging only — see
  [docs/design.md](docs/design.md#logging).
- **Headers**: upstream headers configurable globally (`[extra_headers]`) and
  per model; `Authorization` / `x-api-key` / `anthropic-version` are derived
  from the configured key. Client headers are **not** forwarded, and upstream
  `x-ratelimit-*` / `request-id` / `retry-after` are not relayed.
- **Streaming**: SSE parsed and re-emitted in the client's own framing for all
  four protocols, with errors surfaced in the client's error shape.
- **Config-driven models**: each model declares a `protocol`, optional
  `base_url`, `upstream` id and `vision` flag.
- **models.dev enrichment** (opt-in `[models_dev]`): vision, prices and context
  limits inferred into `/v1/models`; explicit config always wins.
- **Hot-reload** (`[reload]`, on by default): SIGHUP or config mtime change
  re-reads the file without restart; broken files are rejected.
- **Auto-discovery** (opt-in `[discovery]`): upstream `/models` ids merge under
  explicit entries, with a protocol probe so a wrong `protocol` is caught at
  refresh time instead of failing every request.
- **Tiny**: `opt-level = "z"`, LTO, `panic = "abort"`, static musl builds.

## Endpoints

| Method | Path                                              | Description                                      |
| ------ | ------------------------------------------------- | ------------------------------------------------ |
| GET    | `/healthz`                                        | liveness check                                   |
| GET    | `/v1/models`                                      | list configured models (OpenAI model list shape) |
| POST   | `/v1/chat/completions`                            | OpenAI Chat Completions in/out                   |
| POST   | `/v1/responses`                                   | OpenAI Responses in → Chat up → Responses out    |
| POST   | `/v1/messages`                                    | Anthropic Messages in → Chat up → Anthropic out  |
| POST   | `/v1beta/models/{model}:generateContent`         | Gemini generateContent in/out                   |
| POST   | `/v1beta/models/{model}:streamGenerateContent`    | Gemini streaming in/out (SSE)                   |

## Quick start

```sh
# 1. Get a config and an API key
cp examples/gateway.min.toml gateway.local.toml
cp .env.example .env
$EDITOR gateway.local.toml   # set models + base_url
$EDITOR .env                 # set MY_API_KEY=...

# 2. Run (reads GATEWAY_CONFIG, defaulting to ./gateway.toml)
GATEWAY_CONFIG=gateway.local.toml cargo run --release

# 3. Try it
curl localhost:8787/healthz
curl localhost:8787/v1/models
```

Then point any OpenAI-compatible client at `http://localhost:8787/v1`.

## Configuration

`GATEWAY_CONFIG` selects the TOML file (default `gateway.toml`). See
`examples/gateway.min.toml` for a minimal config and
`examples/opencode-go.toml` for a real-world example.

Top-level keys:

| Key                             | Default                    | Description                                        |
| ------------------------------- | -------------------------- | -------------------------------------------------- |
| `listen`                        | `0.0.0.0:8787`             | bind address                                       |
| `api_key_env`                   | —                          | env var holding the upstream Bearer key            |
| `api_key`                       | —                          | inline key (prefer the env var)                    |
| `session_id` / `session_header` | —                          | optional stable session header sent upstream       |
| `user_agent`                    | `pico-nervogate/<version>` | User-Agent sent upstream                           |
| `default_base_url`              | —                          | base URL for models without their own              |
| `owned_by` / `provider`         | binary name                | fields advertised by `/v1/models`                  |
| `extra_headers`                 | —                          | headers added to every upstream request            |
| `anthropic_max_tokens`          | `8192`                     | `max_tokens` sent to Anthropic upstreams on omission |
| `strict_params`                 | `false`                    | reject untranslatable parameters instead of warning |
| `reasoning_output`              | `reasoning_content`        | Chat egress thinking render: `reasoning_content` / `think_tag` / `both` |
| `max_body_bytes`                | `33554432` (32 MiB)        | largest client request body; `0` disables the limit |
| `[models_dev]`                  | disabled                   | models.dev enrichment for `/v1/models`             |
| `[reload]`                      | enabled                    | SIGHUP + mtime config hot-reload                   |
| `[discovery]`                   | disabled                   | upstream `/models` auto-discovery                  |
| `[[models]]`                    | —                          | one or more model definitions                      |

Full reference — the `[models_dev]` / `[reload]` / `[discovery]` sections,
per-model fields, `thinking_type` and the models.dev override keys — is in
[docs/configuration.md](docs/configuration.md).

## Docker

```sh
docker build -t pico-nervogate:arm64 .
cp examples/gateway.min.toml gateway.local.toml
cp .env.example .env
docker compose up -d
```

`docker-compose.yml` mounts `gateway.local.toml` (git-ignored) and reads keys
from `.env`.

## Design notes

Protocol translation semantics — tool calls, thinking, structured output, the
stateless Responses policy, usage mapping, parameter handling, streaming
failures, parallel tool calls — are documented in
[docs/design.md](docs/design.md).

## Tests

```sh
cargo test
```

Unit tests cover the translators and the SSE parser in isolation; `src/e2e.rs`
drives a real mock upstream behind the real router over real sockets, across
all sixteen client-surface × upstream protocol combinations, and asserts the
bytes a client would actually consume.

## Disclaimer

Unofficial and unaffiliated. `pico-nervogate` is a generic client-side gateway;
example configs may reference third-party services, which are the property of
their respective owners. You are responsible for complying with the terms of any
service you point it at.

## License

MIT. See `LICENSE`.

## Performance

Two datasets, recorded 2026-10-11 and not comparable with each other: a
**Docker-vs-Docker comparison against LiteLLM** (same harness, both gateways
in containers), and a **bare-metal baseline for pico-nervogate only** (direct
process on the host).

### pico-nervogate vs LiteLLM (Docker, 2026-10-11)

Both gateways run in containers on the same Colima VM (6 CPUs / 8 GiB), the
load generator and mock upstream run on the host, and every case uses a fresh
container. Full environment, configuration, sampling method, and caveats are
in [docs/performance-comparison.md](docs/performance-comparison.md).

Non-streaming Chat ingress, 4 s after a 1 s warmup — throughput (req/s) with
p95 latency:

| Upstream path | Concurrency | pico-nervogate | LiteLLM |
| --- | ---: | ---: | ---: |
| Chat passthrough | 1 | 2,123 (0.60 ms) | 206 (5.91 ms) |
| Chat passthrough | 16 | 12,448 (1.83 ms) | 253 (67.7 ms) |
| Chat passthrough | 64 | 17,057 (5.50 ms) | 238 (347 ms) |
| Chat → Anthropic | 16 | 10,623 (2.10 ms) | 284 (60.7 ms) |
| Chat → Responses | 16 | 10,792 (2.07 ms) | 205 (85.0 ms) |

Short SSE responses, concurrency 16:

| Upstream protocol | pico-nervogate | LiteLLM |
| --- | ---: | ---: |
| Chat passthrough | 11,844 (1.93 ms) | 160 (143 ms) |
| Chat → Anthropic translation | 7,359 (2.97 ms) | 186 (113 ms) |
| Chat → Responses translation | 6,305 (3.46 ms) | 112 (171 ms) |

Memory and CPU, same runs:

| Metric | pico-nervogate | LiteLLM |
| --- | ---: | ---: |
| Idle memory | 0.7 MiB | 281.3 MiB |
| Peak memory (1 MiB body, concurrency 4) | 8.4 MiB | 352.3 MiB |
| Peak memory (1 MiB body, concurrency 16) | 5.8 MiB | 419.3 MiB |
| CPU per request (concurrency 1, Chat) | ~0.14 ms | ~4.5 ms |

Both targets completed all 15 cases with zero failed requests. Caveats that
apply only to this dataset (VM resources, cgroup-based memory accounting,
single LiteLLM worker, background load, upload-path-bound large-body cases)
are listed in the comparison document — in particular these container
numbers must not be mixed with the bare-metal baseline below.

### Bare-metal baseline — pico-nervogate only (2026-10-11)

#### Environment and method

- Machine: Apple M4 Max MacBook Pro, ARM64, macOS 26.6.2
- Logical CPUs: 14; physical memory: 36 GiB
- Release executable: 2,049,648 bytes
- Load generator: Node.js 26.7.0, HTTP/1.1 keep-alive; local mock upstream on
  `127.0.0.1`
- Gateway request logs were discarded during measurement
- Small-request and SSE measurements ran for four seconds after a one-second
  warmup; large-body measurements ran cold for three seconds and sampled RSS
  again five seconds later. Each case used a fresh gateway process
- All recorded load cases completed with zero failed requests

The gateway had about **3.2 MiB idle RSS**. CPU below is reported as
logical-core equivalents, so 370% means approximately 3.7 cores. Percentile
latency is end-to-end over the local client → gateway → mock-upstream path and
does not include a real model provider.

#### Small non-streaming requests

All cases use Chat ingress with a short prompt and response. The model
protocol selects the upstream protocol, so `bench-chat` measures
same-protocol passthrough and the other models measure translation.

| Upstream path | Concurrency | Throughput (req/s) | p95 (ms) | p99 (ms) | Gateway CPU | Gateway peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| Chat passthrough | 1 | 9,446 | 0.119 | 0.147 | 0.6 cores | 4.0 MiB |
| Chat passthrough | 16 | 47,044 | 0.475 | 0.610 | 3.7 cores | 6.2 MiB |
| Chat passthrough | 64 | 47,140 | 2.399 | 2.659 | 3.7 cores | 10.6 MiB |
| Chat → Anthropic | 1 | 8,898 | 0.127 | 0.160 | 0.6 cores | 4.1 MiB |
| Chat → Anthropic | 16 | 46,546 | 0.463 | 0.601 | 4.1 cores | 6.6 MiB |
| Chat → Anthropic | 64 | 47,512 | 2.353 | 2.640 | 4.4 cores | 11.2 MiB |
| Chat → Responses | 1 | 8,932 | 0.125 | 0.153 | 0.6 cores | 4.0 MiB |
| Chat → Responses | 16 | 46,974 | 0.453 | 0.602 | 4.4 cores | 6.4 MiB |
| Chat → Responses | 64 | 47,329 | 2.364 | 2.659 | 4.3 cores | 10.9 MiB |

Throughput was close to flat between concurrency 16 and 64 while tail latency
increased. This indicates the local request path was near a host-side ceiling;
it does not isolate a hard gateway-only limit.

#### Short SSE responses

| Upstream protocol | Concurrency | Throughput (req/s) | p95 (ms) | p99 (ms) | Gateway CPU | Gateway peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| Chat passthrough | 16 | 37,391 | 0.587 | 0.720 | 3.2 cores | 6.2 MiB |
| Chat → Anthropic translation | 16 | 36,497 | 0.576 | 0.712 | 4.0 cores | 6.7 MiB |
| Chat → Responses translation | 16 | 36,604 | 0.569 | 0.709 | 4.4 cores | 6.8 MiB |

#### Large request-body memory profile

The request contains approximately 1 MiB of text. Each concurrency level ran
in a fresh process; RSS was sampled during the run and five seconds after
load stopped.

| Concurrency | Throughput (req/s) | p95 (ms) | Peak RSS | RSS 5 s after load |
|---:|---:|---:|---:|---:|
| 1 | 872 | 1.302 | 20.5 MiB | 20.5 MiB |
| 4 | 2,103 | 3.005 | 51.1 MiB | 51.1 MiB |
| 16 | 1,924 | 14.441 | 166.2 MiB | 166.2 MiB |

At concurrency 16, the cold memory run measured 166.2 MiB peak and retained
RSS five seconds later. Large bodies increase memory sharply because each
in-flight request is read and parsed before it is sent upstream. The allocator
retained most of the RSS high-water mark for at least five seconds; this
observation alone does not prove a leak.

#### Scope and caveats

These figures describe this Mac and this local mock setup. They are not
Raspberry Pi numbers and do not include external DNS/TLS, upstream latency,
model generation, provider throttling, logging collection, or long-duration
soak behavior. The Node load generator and mock server consume host CPU too.

### Running the suite

```sh
# Bare metal: release binary on the host
python3 scripts/perf/run.py

# Docker: gateway in a container (builds images; --skip-build reuses them)
python3 scripts/perf/run-docker.py --target pico
python3 scripts/perf/run-docker.py --target litellm
```

Both variants use a loopback mock upstream and write a JSON report under
`target/` (git-ignored). Requirements, options, and measured metrics are
documented in [docs/performance-testing.md](docs/performance-testing.md);
the Docker-vs-Docker method and results are documented in
[docs/performance-comparison.md](docs/performance-comparison.md).
