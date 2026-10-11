const http = require('http');
const fs = require('fs');
const { performance } = require('perf_hooks');

const [urlText, bodyPath, secondsText, concurrencyText] = process.argv.slice(2);
const url = new URL(urlText);
const body = fs.readFileSync(bodyPath);
const seconds = Number(secondsText);
const concurrency = Number(concurrencyText);
const agent = new http.Agent({ keepAlive: true, maxSockets: concurrency, maxFreeSockets: concurrency });
const deadline = performance.now() + seconds * 1000;
const started = performance.now();
let completed = 0;
let failed = 0;
let responseBytes = 0;
const latencies = [];

function requestOnce() {
  const start = performance.now();
  return new Promise(resolve => {
    let finished = false;
    const finish = (success, bytes) => {
      if (finished) return;
      finished = true;
      latencies.push(performance.now() - start);
      if (success) { completed++; responseBytes += bytes; } else failed++;
      resolve();
    };
    const req = http.request({
      hostname: url.hostname, port: url.port, path: url.pathname + url.search,
      method: 'POST', agent,
      headers: { 'content-type': 'application/json', 'content-length': body.length },
    }, res => {
      let size = 0;
      res.on('data', chunk => { size += chunk.length; });
      res.on('end', () => finish(res.statusCode >= 200 && res.statusCode < 300, size));
      res.on('error', () => finish(false, 0));
    });
    req.setTimeout(30000, () => req.destroy(new Error('request timeout')));
    req.on('error', () => finish(false, 0));
    req.end(body);
  });
}

async function worker() {
  while (performance.now() < deadline) await requestOnce();
}

(async () => {
  await Promise.all(Array.from({ length: concurrency }, worker));
  agent.destroy();
  const duration = (performance.now() - started) / 1000;
  latencies.sort((a, b) => a - b);
  const percentile = p => latencies.length
    ? latencies[Math.min(latencies.length - 1, Math.round((latencies.length - 1) * p))]
    : 0;
  console.log(JSON.stringify({
    duration_s: +duration.toFixed(3), concurrency, completed, failed,
    rps: +(completed / duration).toFixed(1), response_bytes: responseBytes,
    mean_ms: +(latencies.reduce((sum, value) => sum + value, 0) / Math.max(1, latencies.length)).toFixed(3),
    p50_ms: +percentile(0.50).toFixed(3), p90_ms: +percentile(0.90).toFixed(3),
    p95_ms: +percentile(0.95).toFixed(3), p99_ms: +percentile(0.99).toFixed(3),
    max_ms: +percentile(1).toFixed(3),
  }));
})().catch(error => { console.error(error); process.exitCode = 1; });
