# Performance and load testing

The local load-test suite measures the release gateway separately from model generation and external network latency. It starts a deterministic mock upstream on loopback, runs the release binary as a separate process, and drives it with a Node.js HTTP/1.1 keep-alive client.

## Requirements

- Rust toolchain used by the project (`cargo` and `rustc`)
- Python 3.10 or later (standard library only)
- Node.js 18 or later (built-in `http` module only)
- macOS or Linux; process RSS and CPU sampling use the native process APIs for those systems

The test binds two temporary free ports on `127.0.0.1`. It does not call an LLM provider. Gateway request logs are discarded during the measurement so terminal/file logging does not become part of the result. Do not point this suite at a production or billable upstream.

To measure containerized gateways instead (including the LiteLLM comparison),
use `scripts/perf/run-docker.py`, which shares the same load generator, mock
upstream, case matrix and metrics — see
[performance-comparison.md](performance-comparison.md).

## Run

From the repository root:

```sh
python3 scripts/perf/run.py
```

The default run builds `target/release/pico-nervogate`, then measures:

- Non-streaming Chat ingress against Chat, Anthropic, and Responses mock upstreams, at concurrency 1, 16, and 64.
- Short SSE responses from each of those upstream protocols, at concurrency 16.
- A 1 MiB Chat request at concurrency 1, 4, and 16, with RSS sampled during load and again five seconds after load ends.

Each case starts a fresh gateway process. This keeps allocator high-water memory from a large-body test from affecting a later case. Small-request cases receive a one-second warmup; the large-body memory cases start cold so their RSS curve is easier to interpret. The suite prints one JSON object per case and writes the full report to `target/perf-results.json` by default. `target/` is ignored by Git.

Useful options:

```sh
# Reuse the current release binary; shorter smoke run
python3 scripts/perf/run.py --skip-build --duration 1 --memory-duration 1 \
  --concurrency 1 8 --memory-concurrency 1 4

# Save the report elsewhere
python3 scripts/perf/run.py --output /tmp/pico-nervogate-perf.json

# Change concurrency and measurement length
python3 scripts/perf/run.py --duration 10 --memory-duration 5 \
  --concurrency 1 16 32 64 128 --memory-concurrency 1 2 4 8 16
```

## Metrics and interpretation

Every case reports completed and failed requests, throughput in requests per second, mean latency and p50/p90/p95/p99/max latency. Gateway resource fields include idle RSS, sampled peak RSS, end-of-load RSS, and process CPU as a percentage of one logical core. For example, `370%` means about 3.7 logical CPU cores were busy on behalf of the gateway. This is process CPU time divided by wall time; it is not a percentage of all CPU cores in the machine.

RSS sampling is periodic, so a very short memory spike between samples may not be captured. `gateway_rss_after_settle_mib` is taken five seconds after a large-body run. Allocators can keep freed pages mapped, so retained RSS alone does not establish a memory leak. macOS reports resident bytes through `proc_pidinfo`; Linux uses `/proc/<pid>/status` and `/proc/<pid>/stat`.

The load generator, mock upstream, kernel loopback stack, and gateway all run on the same host. At high request rates, the generator or mock can become the bottleneck. Treat the results as a repeatable local comparison, not as a guaranteed production capacity number. Use the same machine, release build, options, and background-load conditions when comparing changes.

## Latest recorded run

The bare-metal baseline and the Docker-vs-Docker comparison against LiteLLM
are documented in the [Performance section of the README](../README.md#performance),
with full tables, environment details and caveats in
[performance-comparison.md](performance-comparison.md). Repeat the suite on
the machine and revision you intend to deploy.
