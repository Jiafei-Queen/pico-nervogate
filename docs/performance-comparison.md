# pico-nervogate vs LiteLLM — Docker comparison (2026-10-11)

Both gateways measured with the same load-test harness, both running in
containers on the same machine, back to back. This document records the
environment, the exact configuration of each target, full results, and the
caveats that apply. The [README](../README.md#performance) carries a summary
table; the bare-metal pico-nervogate baseline is documented there as well and
is a separate, non-comparable dataset.

## Headline numbers

Non-streaming Chat ingress at concurrency 16 (4 s measurement after 1 s
warmup), throughput in req/s with p95 latency:

| Upstream path | pico-nervogate | LiteLLM | Throughput ratio |
| --- | ---: | ---: | ---: |
| Chat passthrough | 12,448 (1.8 ms) | 253 (67.7 ms) | 49x |
| Chat → Anthropic | 10,623 (2.1 ms) | 284 (60.7 ms) | 37x |
| Chat → Responses | 10,792 (2.1 ms) | 205 (85.0 ms) | 53x |

Memory: **0.7 MiB vs 281.3 MiB idle**, 8.4 MiB vs 352.3 MiB peak under the
1 MiB-body load (concurrency 4). At concurrency 1, LiteLLM spends ~4.5 ms of
CPU per request where pico-nervogate spends ~0.14 ms (~32x).

## Environment

- Host: Apple M4 Max MacBook Pro, ARM64, macOS 26.6.2; 14 logical CPUs,
  36 GiB memory
- Container runtime: Colima (aarch64, macOS Virtualization.Framework),
  Docker server 28.4.0; VM limited to **6 CPUs / 8 GiB**
- Images:
  - `pico-nervogate:perf` — `sha256:29edd218…`, built from the repository
    `Dockerfile` (static musl binary on `scratch`)
  - `litellm-perf:1.104.2` — `sha256:c1ddedf9…`, built from
    `scripts/perf/litellm.Dockerfile` (`python:3.12-slim` +
    `litellm[proxy]==1.104.2`)
- Load generator: Node.js 26.7.0, HTTP/1.1 keep-alive, running on the host;
  mock upstream also on the host
- Gateway request logs discarded during measurement (container stdout/stderr
  dropped)
- Git revision `234358b` (worktree dirty: this change set)
- Other containers (`open-webui`, `pi-searxng`, a `pico-nervogate` instance)
  were running in the VM during both runs; both targets faced the same
  background load

## Topology

```
loadgen (host) ── 127.0.0.1 published port ──▶ gateway container (Colima VM)
gateway container ── host.docker.internal (192.168.5.2) ──▶ mock upstream (host)
```

Only the gateway under test runs in a container; the load generator and the
mock upstream are identical host-side processes for both targets. Every case
uses a fresh container.

## Method

The case matrix, durations, warmup, settle rules, and pass criterion (zero
failed requests) are identical to the bare-metal suite
([performance-testing.md](performance-testing.md)) and share its code
(`loadgen.js`, `mock-upstream.js`, helpers imported from `run.py`):

- Non-streaming: `bench-chat` / `bench-anthropic` / `bench-responses`
  at concurrency 1, 16, and 64 — 4 s measured after 1 s warmup
- Short SSE: the three models at concurrency 16 — 4 s after 1 s warmup
- Large body: 1 MiB request at concurrency 1, 4, and 16 — 3 s cold, RSS
  sampled again 5 s after load stops

That is 15 cases per target, each in a fresh container.

### Resource sampling

Metrics come from the Docker Engine API over the Colima unix socket
(`containers/{id}/stats`), so the same numbers are produced for both targets
regardless of what is inside the image:

- **CPU** (`gateway_cpu_one_core_pct`): delta of cumulative
  `cpu_stats.cpu_usage.total_usage` across the measurement window, divided by
  wall time and multiplied by 100 — the same "logical-core equivalents"
  formula the bare-metal suite uses for process CPU.
- **Memory** (`gateway_rss_*`): cgroup memory usage minus inactive file
  cache, sampled every ~20 ms (large-body cases) and ~80 ms (small cases).
  This is an RSS approximation: it excludes clean file-backed pages, so
  pico-nervogate's idle figure (0.7 MiB) reads lower than the 3.2 MiB
  process-RSS value in the bare-metal baseline. Fields: `idle` after
  readiness, `peak` during load, `end` right after load, and `after_settle`
  5 s later (large-body cases only).

### Per-target configuration

pico-nervogate — generated TOML identical to the bare-metal suite, except
`listen = "0.0.0.0:8787"` and `default_base_url` pointing at
`http://host.docker.internal:{mock}/v1`; the three models (`bench-chat`,
`bench-anthropic`, `bench-responses`) select the upstream protocol.

LiteLLM — generated YAML mounted at `/config/config.yaml`, started as
`litellm --config /config/config.yaml --host 0.0.0.0 --port 4000
--num_workers 1` (single worker so the whole container runs one process):

- `bench-chat` → `model: openai/bench-upstream`,
  `api_base: …/v1` (Chat passthrough)
- `bench-anthropic` → `model: anthropic/bench-upstream`,
  `api_base: …/v1/messages`
- `bench-responses` → `model: openai/responses/bench-upstream`,
  `api_base: …/v1` — LiteLLM's chat→Responses bridge. Note
  `openai_responses/…` is not a valid provider prefix in 1.104.2; the
  `openai/responses/…` form is what routes Chat ingress to a Responses
  upstream.
- `LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY=true` — starts
  without a master key so, like pico-nervogate, requests need no client auth
  and `loadgen.js` stays unchanged.

Readiness: `/healthz` (pico) vs `/health/liveliness` (LiteLLM).

## Results

All cases completed with zero failed requests.

### Non-streaming, short request (4 s + 1 s warmup)

| Upstream path | Conc | pico rps | pico p95 (ms) | pico CPU (cores) | pico peak RSS (MiB) | LiteLLM rps | LiteLLM p95 (ms) | LiteLLM CPU (cores) | LiteLLM peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chat passthrough | 1 | 2,123 | 0.597 | 0.30 | 1.2 | 206 | 5.909 | 0.93 | 316.7 |
| Chat passthrough | 16 | 12,448 | 1.826 | 2.33 | 3.1 | 253 | 67.694 | 0.67 | 320.4 |
| Chat passthrough | 64 | 17,057 | 5.502 | 2.51 | 6.7 | 238 | 347.303 | 0.69 | 326.7 |
| Chat → Anthropic | 1 | 2,074 | 0.600 | 0.31 | 1.7 | 232 | 5.524 | 0.93 | 339.2 |
| Chat → Anthropic | 16 | 10,623 | 2.100 | 2.67 | 3.1 | 284 | 60.702 | 0.67 | 343.3 |
| Chat → Anthropic | 64 | 12,417 | 7.557 | 2.73 | 7.1 | 287 | 333.311 | 0.68 | 348.5 |
| Chat → Responses | 1 | 1,991 | 0.632 | 0.33 | 1.2 | 179 | 6.657 | 0.94 | 339.2 |
| Chat → Responses | 16 | 10,792 | 2.068 | 2.75 | 2.7 | 205 | 84.987 | 0.67 | 344.7 |
| Chat → Responses | 64 | 12,523 | 7.474 | 2.78 | 7.3 | 204 | 388.965 | 0.69 | 351.8 |

### Short SSE (concurrency 16, 4 s + 1 s warmup)

| Upstream protocol | pico rps | pico p95 (ms) | pico CPU (cores) | LiteLLM rps | LiteLLM p95 (ms) | LiteLLM CPU (cores) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chat passthrough | 11,844 | 1.933 | 2.42 | 160 | 143.286 | 0.68 |
| Chat → Anthropic translation | 7,359 | 2.973 | 2.99 | 186 | 113.166 | 0.67 |
| Chat → Responses translation | 6,305 | 3.460 | 3.15 | 112 | 171.391 | 0.67 |

### Large request body (1 MiB, 3 s cold + 5 s settle)

| Conc | pico rps | pico p95 (ms) | pico peak / settle RSS (MiB) | LiteLLM rps | LiteLLM p95 (ms) | LiteLLM peak / settle RSS (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 132 | 8.246 | 2.7 / 0.9 | 68 | 14.246 | 341.2 / 341.2 |
| 4 | 213 | 24.146 | 8.4 / 1.1 | 102 | 40.217 | 352.3 / 350.0 |
| 16 | 198 | 93.507 | 5.8 / 1.4 | 107 | 147.777 | 419.3 / 419.2 |

### Observations

- **Memory**: LiteLLM idles at 281.3 MiB (Python + framework) and peaks at
  419 MiB on the large-body load, retaining it after settle.
  pico-nervogate idles at ~0.7 MiB (cgroup approximation) and returns to
  ~1 MiB after the large-body runs.
- **CPU per request**: at concurrency 1, LiteLLM uses ~4.5 ms of CPU per
  request vs ~0.14 ms for pico-nervogate.
- **LiteLLM plateaus near 250–290 req/s** at every concurrency while using
  only ~0.7 cores — its ceiling here is neither CPU nor the 6-core VM limit;
  the single-worker bottleneck was not diagnosed further. Tail latency grows
  with concurrency (queuing), not throughput.
- **pico-nervogate** scales with concurrency up to the host-side ceiling
  (~17k req/s at 64 connections); CPU stays at 2.3–2.8 cores of the 6
  available, so the remaining cost is the cross-VM network path, not the
  gateway.
- **Large-body cases are upload-path-bound**: 198 req/s of 1 MiB bodies at
  concurrency 16 is ~200 MB/s through the published-port forwarding, ~10x
  below loopback in the bare-metal run. Bodies therefore queue in the proxy
  and VM socket buffers outside the container, so gateway-side buffering —
  and the measured peak memory — stays far below the bare-metal run's
  166 MiB at the same concurrency. The pico-vs-LiteLLM rows remain
  comparable (identical path for both), but this row must not be read as an
  absolute large-body memory figure for either gateway.

## TCP_NODELAY fix applied before this run

While validating the harness, SSE through the container path measured ~45 ms
per request for pico-nervogate (keep-alive connections) while bare-metal
measured sub-millisecond. Diagnosis: pico-nervogate never set `TCP_NODELAY`,
so Nagle held multi-segment SSE responses until the peer's delayed ACK
(~40 ms) landed — the first request on a connection was fast (quickack
window), every subsequent one stalled. LiteLLM's stack (asyncio/anyio) sets
`TCP_NODELAY=1` and was unaffected; bare-metal loopback hid the problem,
which is why the 2026-10-11 baseline did not show it.

The fix (axum `ListenerExt::tap_io` + `set_nodelay(true)` on accepted
sockets) landed before these measurements. Measured on the same container
path, keep-alive SSE requests went from ~45 ms to 0.7–2.4 ms, and a 1 s
single-connection loadgen run went from 22 req/s to 2,035 req/s. Without the
fix the SSE rows above would have measured a TCP artifact, not the gateway.

## Reproduce

```sh
# Images (run-docker.py builds them too; --skip-build reuses them)
docker build -t pico-nervogate:perf .
docker build -t litellm-perf:1.104.2 -f scripts/perf/litellm.Dockerfile scripts/perf

# The two measured runs (back to back)
python3 scripts/perf/run-docker.py --target pico --output target/perf-results-pico-docker.json
python3 scripts/perf/run-docker.py --target litellm --output target/perf-results-litellm-docker.json
```

Reports land in the git-ignored `target/` directory. Requirements are the
same as the bare-metal suite plus Docker (Colima on macOS) — see
[performance-testing.md](performance-testing.md).

## Caveats

- The container numbers are **not comparable with the bare-metal baseline**
  in the README: the VM has 6 CPUs (not 14), all gateway traffic crosses the
  VM boundary through published-port proxies, and the load generator, mock
  and host kernel share the remaining cores. This is most visible in the
  large-body cases, where the upload path — not the gateway — is the
  bottleneck.
- Memory figures are cgroup usage minus inactive file cache, not `/proc`
  RSS; pico-nervogate's idle number therefore reads lower than its
  process-RSS figure elsewhere.
- One run per target, executed back to back; there are no error bars.
- Background containers in the VM (see Environment) add noise equally to
  both targets.
- The mock returns tiny static payloads on loopback: no real model
  generation, TLS, DNS, provider throttling, logging collection, or
  long-duration soak behavior is measured. The Node load generator and mock
  consume host CPU too.
- LiteLLM ran as a single worker (`--num_workers 1`); multi-worker
  deployments would trade memory for throughput and change these numbers.
- These are not Raspberry Pi numbers.
