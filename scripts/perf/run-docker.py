#!/usr/bin/env python3
"""Load-test pico-nervogate or LiteLLM running in a Docker container on localhost."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import run as base

ROOT = base.ROOT
LOADGEN = base.LOADGEN
MIB = base.MIB

LITELLM_VERSION = "1.104.2"  # keep in sync with scripts/perf/litellm.Dockerfile
PICO_IMAGE = "pico-nervogate:perf"
LITELLM_IMAGE = f"litellm-perf:{LITELLM_VERSION}"


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self.unix_path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.unix_path)


class Docker:
    """Minimal Docker Engine API client over the active context's unix socket."""

    def __init__(self) -> None:
        host = os.environ.get("DOCKER_HOST") or subprocess.check_output(
            ["docker", "context", "inspect", "--format",
             '{{(index .Endpoints "docker").Host}}'], text=True).strip()
        if not host.startswith("unix://"):
            raise RuntimeError(f"only unix docker sockets are supported, got {host}")
        self.socket_path = host.removeprefix("unix://")

    def request(self, method: str, path: str, payload: dict | None = None,
                timeout: float = 30) -> Any:
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if body else {}
        conn = UnixHTTPConnection(self.socket_path, timeout)
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            data = response.read()
            if response.status >= 400:
                raise RuntimeError(
                    f"docker {method} {path} -> {response.status}: "
                    f"{data[:300].decode(errors='replace')}")
            return json.loads(data) if data else {}
        finally:
            conn.close()

    def server_version(self) -> str:
        return self.request("GET", "/version").get("Version", "unknown")

    def resources(self) -> dict[str, Any]:
        info = self.request("GET", "/info")
        return {"cpus": info.get("NCPU"), "memory_mib": round(info.get("MemTotal", 0) / MIB)}

    def image_id(self, ref: str) -> str:
        return self.request("GET", f"/images/{ref}/json").get("Id", "unknown")

    def create(self, image: str, name: str, *, host_port: int, container_port: int,
               env: list[str], cmd: list[str] | None, binds: list[str]) -> str:
        payload: dict[str, Any] = {
            "Image": image,
            "Env": env,
            "ExposedPorts": {f"{container_port}/tcp": {}},
            "HostConfig": {
                "PortBindings": {f"{container_port}/tcp": [
                    {"HostIp": "127.0.0.1", "HostPort": str(host_port)}]},
                "Binds": binds,
            },
        }
        if cmd:
            payload["Cmd"] = cmd
        return self.request(
            "POST", f"/containers/create?name={name}", payload)["Id"]

    def start(self, cid: str) -> None:
        self.request("POST", f"/containers/{cid}/start")

    def remove(self, cid: str) -> None:
        try:
            self.request("DELETE", f"/containers/{cid}?force=true")
        except RuntimeError:
            pass

    def state(self, cid: str) -> dict[str, Any]:
        return self.request("GET", f"/containers/{cid}/json").get("State", {})

    def logs(self, cid: str) -> str:
        proc = subprocess.run(["docker", "logs", "--tail", "30", cid],
                              capture_output=True, text=True)
        return (proc.stdout + proc.stderr).strip()[-2000:]

    def usage(self, cid: str) -> dict[str, int]:
        """Cumulative CPU nanoseconds and approximate RSS bytes for the container."""
        stats = self.request("GET", f"/containers/{cid}/stats?stream=false", timeout=15)
        memory = stats.get("memory_stats", {})
        buckets = memory.get("stats") or {}
        inactive = buckets.get("total_inactive_file", buckets.get("inactive_file", 0))
        return {
            "cpu_ns": stats.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0),
            "rss_bytes": max(memory.get("usage", 0) - inactive, 0),
        }


class ContainerHandle:
    """Duck-typed like a Popen for base.wait_for_http: poll(), pid, returncode."""

    def __init__(self, docker: Docker, cid: str):
        self.docker = docker
        self.cid = cid
        self.returncode: int | None = None

    @property
    def pid(self) -> str:
        return self.cid[:12]

    def poll(self) -> int | None:
        if self.returncode is None:
            state = self.docker.state(self.cid)
            if state.get("Running"):
                return None
            self.returncode = state.get("ExitCode") or 1
        return self.returncode


class DockerServices:
    """Mock upstream on the host plus one containerized gateway per case."""

    def __init__(self, target: str, gateway_port: int, mock_port: int):
        self.target = target
        self.gateway_port = gateway_port
        self.mock_port = mock_port
        self.docker = Docker()
        self.mock: subprocess.Popen[bytes] | None = None
        self.container: str | None = None
        # The container runtime only sees the project directory, not $TMPDIR,
        # so generated configs must live under the git-ignored target/.
        (ROOT / "target").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="perf-", dir=ROOT / "target")
        if target == "pico":
            self.image = PICO_IMAGE
            self.container_port = 8787
            self.health_path = "/healthz"
            self.ready_timeout = 15
            self.env = ["GATEWAY_CONFIG=/gateway.toml"]
            self.cmd = None
            config = Path(self.temp.name) / "gateway.toml"
            config.write_text(f'''listen = "0.0.0.0:8787"
api_key = "benchmark-only"
default_base_url = "http://host.docker.internal:{mock_port}/v1"
[reload]
enable = false
watch_interval_secs = 0
[[models]]
name = "bench-chat"
protocol = "openai-compatible"
[[models]]
name = "bench-anthropic"
protocol = "anthropic"
[[models]]
name = "bench-responses"
protocol = "openai"
''')
            self.binds = [f"{config}:/gateway.toml:ro"]
        else:
            self.image = LITELLM_IMAGE
            self.container_port = 4000
            self.health_path = "/health/liveliness"
            self.ready_timeout = 60
            # No client auth, matching pico: LiteLLM refuses to start without
            # a master key unless this local-development flag is set.
            self.env = ["LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY=true"]
            self.cmd = ["--config", "/config/config.yaml", "--host", "0.0.0.0",
                        "--port", "4000", "--num_workers", "1"]
            config = Path(self.temp.name) / "config.yaml"
            config.write_text(f'''model_list:
  - model_name: bench-chat
    litellm_params:
      model: openai/bench-upstream
      api_base: http://host.docker.internal:{mock_port}/v1
      api_key: benchmark-only
  - model_name: bench-anthropic
    litellm_params:
      model: anthropic/bench-upstream
      api_base: http://host.docker.internal:{mock_port}/v1/messages
      api_key: benchmark-only
  - model_name: bench-responses
    litellm_params:
      model: openai/responses/bench-upstream
      api_base: http://host.docker.internal:{mock_port}/v1
      api_key: benchmark-only
''')
            self.binds = [f"{config}:/config/config.yaml:ro"]

    def build_image(self) -> None:
        if self.target == "pico":
            subprocess.run(["docker", "build", "-q", "-t", PICO_IMAGE, "."],
                           cwd=ROOT, check=True)
        else:
            subprocess.run(
                ["docker", "build", "-t", LITELLM_IMAGE, "-f",
                 "scripts/perf/litellm.Dockerfile", "scripts/perf"],
                cwd=ROOT, check=True)

    def start_mock(self) -> None:
        if self.mock is None:
            env = os.environ.copy()
            env["MOCK_PORT"] = str(self.mock_port)
            self.mock = subprocess.Popen(
                ["node", str(ROOT / "scripts" / "perf" / "mock-upstream.js")], env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            base.wait_for_http(
                f"http://127.0.0.1:{self.mock_port}/healthz", FakeProcess())

    def start_gateway(self) -> str:
        self.stop_gateway()
        name = f"perf-{self.target}-{uuid.uuid4().hex[:8]}"
        cid = self.docker.create(
            self.image, name, host_port=self.gateway_port,
            container_port=self.container_port, env=self.env, cmd=self.cmd,
            binds=self.binds)
        self.container = cid
        self.docker.start(cid)
        try:
            base.wait_for_http(
                f"http://127.0.0.1:{self.gateway_port}{self.health_path}",
                ContainerHandle(self.docker, cid), timeout=self.ready_timeout)
        except (RuntimeError, TimeoutError) as error:
            logs = self.docker.logs(cid)
            self.stop_gateway()
            raise RuntimeError(f"{error}\ncontainer logs:\n{logs}") from error
        return cid

    def stop_gateway(self) -> None:
        if self.container is not None:
            self.docker.remove(self.container)
            self.container = None

    def close(self) -> None:
        self.stop_gateway()
        if self.mock is not None:
            self.mock.terminate()
            try:
                self.mock.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.mock.kill()
                self.mock.wait()
        self.temp.cleanup()


class FakeProcess:
    pid = 0

    def poll(self) -> None:
        return None


def run_case(services: DockerServices, label: str, model: str, duration: float,
             concurrency: int, stream: bool = False, large_bytes: int = 0,
             warm_seconds: float = 1, settle_seconds: float = 0) -> dict[str, Any]:
    body = Path(services.temp.name) / f"{label}.json"
    base.write_body(body, model, stream, large_bytes)
    cid = services.start_gateway()
    url = f"http://127.0.0.1:{services.gateway_port}/v1/chat/completions"
    try:
        idle = services.docker.usage(cid)
        if warm_seconds:
            warmup = base.invoke_loadgen(url, body, warm_seconds, concurrency)
            if warmup["failed"]:
                raise RuntimeError(f"{label} warmup had {warmup['failed']} failed requests")
        before = services.docker.usage(cid)
        start = time.monotonic()
        load = subprocess.Popen(["node", str(LOADGEN), url, str(body), str(duration),
                                 str(concurrency)], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        peak_rss = before["rss_bytes"]
        while load.poll() is None:
            sample = services.docker.usage(cid)
            peak_rss = max(peak_rss, sample["rss_bytes"])
            time.sleep(0.02 if large_bytes else 0.08)
        stdout, stderr = load.communicate()
        if load.returncode:
            raise RuntimeError(f"load generator failed: {stderr or stdout}")
        elapsed = time.monotonic() - start
        after = services.docker.usage(cid)
        peak_rss = max(peak_rss, after["rss_bytes"])
        result = json.loads(stdout.strip().splitlines()[-1])
        result.update({
            "case": label,
            "gateway_cpu_one_core_pct": round(
                (after["cpu_ns"] - before["cpu_ns"]) / 1e9 / elapsed * 100, 1),
            "gateway_rss_idle_mib": round(idle["rss_bytes"] / MIB, 1),
            "gateway_rss_peak_mib": round(peak_rss / MIB, 1),
            "gateway_rss_end_mib": round(after["rss_bytes"] / MIB, 1),
        })
        if settle_seconds:
            time.sleep(settle_seconds)
            settled = services.docker.usage(cid)
            result["gateway_rss_after_settle_mib"] = round(settled["rss_bytes"] / MIB, 1)
        print(json.dumps(result), flush=True)
        if result["failed"]:
            raise RuntimeError(f"{label}: {result['failed']} requests failed")
        return result
    finally:
        services.stop_gateway()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    positive_float = lambda value: float(value) if float(value) > 0 else (_ for _ in ()).throw(
        argparse.ArgumentTypeError("must be greater than zero"))
    positive_int = lambda value: int(value) if int(value) > 0 else (_ for _ in ()).throw(
        argparse.ArgumentTypeError("must be greater than zero"))
    parser.add_argument("--target", required=True, choices=("pico", "litellm"),
                        help="which gateway image to measure")
    parser.add_argument("--duration", type=positive_float, default=4,
                        help="seconds per small-request and SSE case (default: 4)")
    parser.add_argument("--memory-duration", type=positive_float, default=3,
                        help="seconds per large-body memory case (default: 3)")
    parser.add_argument("--concurrency", type=positive_int, nargs="+", default=[1, 16, 64],
                        help="concurrency levels for non-streaming cases")
    parser.add_argument("--memory-concurrency", type=positive_int, nargs="+",
                        default=[1, 4, 16],
                        help="concurrency levels for 1 MiB body cases")
    parser.add_argument("--output", type=Path, default=None,
                        help="write full machine-readable results to this path")
    parser.add_argument("--skip-build", action="store_true",
                        help="reuse existing images instead of rebuilding")
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "target" / f"perf-results-{args.target}-docker.json"
    args.output = Path(args.output)

    for command in ("docker", "node"):
        if not shutil.which(command):
            parser.error(f"required command not found: {command}")

    gateway_port, mock_port = base.free_port(), base.free_port()
    while mock_port == gateway_port:
        mock_port = base.free_port()
    services = DockerServices(args.target, gateway_port, mock_port)
    try:
        if not args.skip_build:
            services.build_image()
        services.docker.image_id(services.image)
        services.start_mock()
        environment: dict[str, Any] = {
            "os": platform.platform(),
            **base.host_details(),
            "architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "node": subprocess.check_output(["node", "--version"], text=True).strip(),
            "runtime": "docker",
            "docker_server": services.docker.server_version(),
            **{f"container_runtime_{k}": v
               for k, v in services.docker.resources().items()},
            "target": args.target,
            "gateway_image": services.docker.image_id(services.image),
            "gateway_log_output": "discarded during measurement",
        }
        if args.target == "litellm":
            environment["litellm_version"] = LITELLM_VERSION
        try:
            environment["git_revision"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                stderr=subprocess.DEVNULL).strip()
            environment["git_worktree_dirty"] = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True,
                stderr=subprocess.DEVNULL).strip())
        except subprocess.CalledProcessError:
            pass
        report: dict[str, Any] = {"environment": environment, "cases": []}
        print(json.dumps({"environment": environment}), flush=True)

        for model in ("bench-chat", "bench-anthropic", "bench-responses"):
            for concurrency in args.concurrency:
                report["cases"].append(run_case(
                    services, f"nonstream-{model}-c{concurrency}", model,
                    args.duration, concurrency))
        for model in ("bench-chat", "bench-anthropic", "bench-responses"):
            report["cases"].append(run_case(
                services, f"sse-{model}", model, args.duration,
                args.concurrency[len(args.concurrency) // 2], stream=True))
        for concurrency in args.memory_concurrency:
            report["cases"].append(run_case(
                services, f"large-1mib-c{concurrency}", "bench-chat",
                args.memory_duration, concurrency, large_bytes=MIB,
                warm_seconds=0, settle_seconds=5))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"results_file": str(args.output.resolve())}), flush=True)
    finally:
        services.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError, TimeoutError) as error:
        print(f"performance test failed: {error}", file=sys.stderr)
        raise SystemExit(1)
