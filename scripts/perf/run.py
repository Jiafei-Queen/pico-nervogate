#!/usr/bin/env python3
"""Build and load-test the pico-nervogate release binary on localhost."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MOCK = ROOT / "scripts" / "perf" / "mock-upstream.js"
LOADGEN = ROOT / "scripts" / "perf" / "loadgen.js"
MIB = 1024 * 1024


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def host_details() -> dict[str, Any]:
    details: dict[str, Any] = {}
    if sys.platform == "darwin":
        try:
            details["cpu_model"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    elif sys.platform.startswith("linux"):
        try:
            details["cpu_model"] = next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name"))
        except (OSError, StopIteration):
            pass
    try:
        details["physical_memory_mib"] = round(
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / MIB)
    except (ValueError, OSError):
        pass
    return details


def process_usage(pid: int) -> dict[str, int] | None:
    """Return process CPU nanoseconds and RSS bytes on macOS or Linux."""
    if sys.platform == "darwin":
        class ProcTaskInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("total_user", ctypes.c_uint64),
                ("total_system", ctypes.c_uint64),
                ("threads_user", ctypes.c_uint64),
                ("threads_system", ctypes.c_uint64),
                ("policy", ctypes.c_int32),
                ("faults", ctypes.c_int32),
                ("pageins", ctypes.c_int32),
                ("cow_faults", ctypes.c_int32),
                ("messages_sent", ctypes.c_int32),
                ("messages_received", ctypes.c_int32),
                ("mach_syscalls", ctypes.c_int32),
                ("unix_syscalls", ctypes.c_int32),
                ("csw", ctypes.c_int32),
                ("thread_count", ctypes.c_int32),
                ("running_count", ctypes.c_int32),
                ("priority", ctypes.c_int32),
            ]

        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                 ctypes.c_void_p, ctypes.c_int]
        proc_pidinfo.restype = ctypes.c_int
        info = ProcTaskInfo()
        size = ctypes.sizeof(info)
        if proc_pidinfo(pid, 4, 0, ctypes.byref(info), size) != size:
            return None

        class Timebase(ctypes.Structure):
            _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]

        timebase = Timebase()
        libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        mach_timebase_info = libsystem.mach_timebase_info
        mach_timebase_info.argtypes = [ctypes.POINTER(Timebase)]
        mach_timebase_info.restype = ctypes.c_int
        mach_timebase_info(ctypes.byref(timebase))
        cpu_ns = (info.total_user + info.total_system) * timebase.numer // timebase.denom
        return {"cpu_ns": cpu_ns, "rss_bytes": info.resident_size}

    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()  # first item is field 3
            clock_ticks = os.sysconf("SC_CLK_TCK")
            cpu_ticks = int(fields[11]) + int(fields[12])  # fields 14 and 15
            status = Path(f"/proc/{pid}/status").read_text()
            rss_kib = int(next(line for line in status.splitlines()
                               if line.startswith("VmRSS:")).split()[1])
            return {"cpu_ns": cpu_ticks * 1_000_000_000 // clock_ticks,
                    "rss_bytes": rss_kib * 1024}
        except (FileNotFoundError, StopIteration, ValueError, IndexError):
            return None
    raise RuntimeError("process CPU/RSS sampling is supported on macOS and Linux")


def wait_for_http(url: str, process: subprocess.Popen[bytes], timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early (pid {process.pid}, code {process.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=0.5):
                return
        except Exception:
            time.sleep(0.05)
    raise TimeoutError(f"server did not respond at {url}")


class LocalServices:
    def __init__(self, binary: Path, gateway_port: int, mock_port: int):
        self.binary = binary
        self.gateway_port = gateway_port
        self.mock_port = mock_port
        self.mock: subprocess.Popen[bytes] | None = None
        self.gateway: subprocess.Popen[bytes] | None = None
        self.temp = tempfile.TemporaryDirectory(prefix="pico-nervogate-perf-")
        self.config = Path(self.temp.name) / "gateway.toml"
        self.config.write_text(f'''listen = "127.0.0.1:{gateway_port}"
api_key = "benchmark-only"
default_base_url = "http://127.0.0.1:{mock_port}/v1"
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

    def start_mock(self) -> None:
        if self.mock is None:
            env = os.environ.copy()
            env["MOCK_PORT"] = str(self.mock_port)
            self.mock = subprocess.Popen(["node", str(MOCK)], env=env,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wait_for_http(f"http://127.0.0.1:{self.mock_port}/healthz", self.mock)

    def start_gateway(self) -> int:
        self.stop_gateway()
        env = os.environ.copy()
        env["GATEWAY_CONFIG"] = str(self.config)
        self.gateway = subprocess.Popen([str(self.binary)], env=env,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_for_http(f"http://127.0.0.1:{self.gateway_port}/healthz", self.gateway)
        return self.gateway.pid

    def stop_gateway(self) -> None:
        if self.gateway is not None:
            self.gateway.terminate()
            try:
                self.gateway.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.gateway.kill()
                self.gateway.wait()
            self.gateway = None

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


def write_body(path: Path, model: str, stream: bool = False, large_bytes: int = 0) -> None:
    content = "x" * large_bytes if large_bytes else "Say ok"
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    if stream:
        body["stream"] = True
    path.write_text(json.dumps(body, separators=(",", ":")))


def invoke_loadgen(url: str, body: Path, duration: float, concurrency: int) -> dict[str, Any]:
    proc = subprocess.run(["node", str(LOADGEN), url, str(body), str(duration), str(concurrency)],
                          capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_case(services: LocalServices, label: str, model: str, duration: float,
             concurrency: int, stream: bool = False, large_bytes: int = 0,
             warm_seconds: float = 1, settle_seconds: float = 0) -> dict[str, Any]:
    body = Path(services.temp.name) / f"{label}.json"
    write_body(body, model, stream, large_bytes)
    pid = services.start_gateway()
    url = f"http://127.0.0.1:{services.gateway_port}/v1/chat/completions"
    try:
        idle = process_usage(pid)
        if warm_seconds:
            warmup = invoke_loadgen(url, body, warm_seconds, concurrency)
            if warmup["failed"]:
                raise RuntimeError(f"{label} warmup had {warmup['failed']} failed requests")
        before = process_usage(pid)
        start = time.monotonic()
        load = subprocess.Popen(["node", str(LOADGEN), url, str(body), str(duration),
                                 str(concurrency)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
        peak_rss = (before or idle or {}).get("rss_bytes", 0)
        while load.poll() is None:
            sample = process_usage(pid)
            if sample:
                peak_rss = max(peak_rss, sample["rss_bytes"])
            time.sleep(0.02 if large_bytes else 0.08)
        stdout, stderr = load.communicate()
        if load.returncode:
            raise RuntimeError(f"load generator failed: {stderr or stdout}")
        elapsed = time.monotonic() - start
        after = process_usage(pid)
        if after:
            peak_rss = max(peak_rss, after["rss_bytes"])
        result = json.loads(stdout.strip().splitlines()[-1])
        result.update({
            "case": label,
            "gateway_cpu_one_core_pct": round(
                ((after["cpu_ns"] - before["cpu_ns"]) / 1e9 / elapsed * 100), 1)
                if after and before else None,
            "gateway_rss_idle_mib": round(idle["rss_bytes"] / MIB, 1) if idle else None,
            "gateway_rss_peak_mib": round(peak_rss / MIB, 1),
            "gateway_rss_end_mib": round(after["rss_bytes"] / MIB, 1) if after else None,
        })
        if settle_seconds:
            time.sleep(settle_seconds)
            settled = process_usage(pid)
            result["gateway_rss_after_settle_mib"] = (
                round(settled["rss_bytes"] / MIB, 1) if settled else None)
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
    parser.add_argument("--duration", type=positive_float, default=4,
                        help="seconds per small-request and SSE case (default: 4)")
    parser.add_argument("--memory-duration", type=positive_float, default=3,
                        help="seconds per large-body memory case (default: 3)")
    parser.add_argument("--concurrency", type=positive_int, nargs="+", default=[1, 16, 64],
                        help="concurrency levels for non-streaming cases")
    parser.add_argument("--memory-concurrency", type=positive_int, nargs="+", default=[1, 4, 16],
                        help="concurrency levels for 1 MiB body cases")
    parser.add_argument("--output", type=Path, default=ROOT / "target" / "perf-results.json",
                        help="write full machine-readable results to this path")
    parser.add_argument("--skip-build", action="store_true",
                        help="reuse target/release/pico-nervogate instead of building")
    args = parser.parse_args()

    for command in ("cargo", "node"):
        if not shutil.which(command):
            parser.error(f"required command not found: {command}")
    binary = ROOT / "target" / "release" / "pico-nervogate"
    if not args.skip_build:
        subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True)
    if not binary.is_file():
        parser.error(f"release binary not found: {binary}")

    gateway_port, mock_port = free_port(), free_port()
    while mock_port == gateway_port:
        mock_port = free_port()
    services = LocalServices(binary, gateway_port, mock_port)
    try:
        services.start_mock()
        environment = {
            "os": platform.platform(),
            **host_details(),
            "architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "node": subprocess.check_output(["node", "--version"], text=True).strip(),
            "release_binary_bytes": binary.stat().st_size,
            "benchmark_upstream": "local mock on 127.0.0.1",
            "gateway_log_output": "discarded during measurement",
        }
        try:
            git_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                stderr=subprocess.DEVNULL).strip()
            environment["git_revision"] = git_revision
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
