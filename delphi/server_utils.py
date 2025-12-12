import os
import signal
import subprocess
import time

import requests
import torch

from delphi.config import RunConfig


def _resolve_metrics_port(server_port: int | None, run_cfg: RunConfig) -> int | None:
    """
    Determine which metrics port to use for vLLM.
    Only attach a metrics port if explicitly configured.
    """
    _ = server_port  # unused but kept for signature compatibility
    return getattr(run_cfg, "server_metrics_port", None)


def check_other_vllm_servers():
    """Check for other vLLM servers running and warn user"""
    try:
        # Use pgrep to find vLLM processes
        result = subprocess.run(
            ["pgrep", "-f", "vllm.*serve"], capture_output=True, text=True, check=False
        )

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            print(
                f"⚠️  WARNING: Found {len(pids)} other vLLM server process(es) running:"
            )
            for pid in pids:
                if pid.strip():
                    # Get more details about the process
                    try:
                        ps_result = subprocess.run(
                            ["ps", "-p", pid.strip(), "-o", "pid,ppid,cmd"],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        if ps_result.returncode == 0:
                            cmd_line = ps_result.stdout.strip().split("\n")[-1]
                            print(f"   PID {pid.strip()}: {cmd_line}")
                    except Exception:
                        print(f"   PID {pid.strip()}")
            print("   Consider stopping other servers to avoid resource conflicts.")
            print()
    except Exception as e:
        print(f"Could not check for other vLLM servers: {e}")


def start_server_if_not_running(server_port: int, run_cfg: RunConfig):
    # Check for other vLLM servers first
    check_other_vllm_servers()

    metrics_port = _resolve_metrics_port(server_port, run_cfg)

    try:
        response = requests.get(f"http://localhost:{server_port}/v1/models")
        if response.status_code == 200:
            print(
                f"[server_utils.py:start_server_if_not_running] Server is already running on port {server_port}"
            )
            return
    except requests.exceptions.RequestException:
        print(
            f"[server_utils.py:start_server_if_not_running] Server is not running on port {server_port}; attempting to start server"
        )
        # Build command with conditional boolean flags
        cmd = [
            "vllm",
            "serve",
            getattr(
                run_cfg,
                "explainer_model",
                "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            ),  # positional model argument
            "--host",
            "0.0.0.0",
            "--max-model-len",
            str(getattr(run_cfg, "explainer_model_max_len", 5120)),
            "--tensor-parallel-size",
            str(getattr(run_cfg, "num_gpus", torch.cuda.device_count())),
            "--gpu-memory-utilization",
            str(getattr(run_cfg, "max_memory_utilization", 0.9)),
            "--port",
            str(getattr(run_cfg, "server_port", 8000)),
            "--uvicorn-log-level",
            "warning",
            # "--disable-log-requests",
            # "--no-access-log",
        ]

        if metrics_port is not None:
            cmd.extend(["--metrics-port", str(metrics_port)])

        # Add boolean flags only if True (don't pass False values)
        if getattr(run_cfg, "enable_prefix_caching", True):
            cmd.append("--enable-prefix-caching")
        if getattr(run_cfg, "enforce_eager", True):
            cmd.append("--enforce-eager")

        model_name_lower = str(getattr(run_cfg, "explainer_model", "")).lower()
        if "qwen" in model_name_lower:
            # cmd.append("--disable-think")
            pass

        # Reduce vLLM logging verbosity
        os.environ["VLLM_LOGGING_LEVEL"] = os.environ.get(
            "VLLM_LOGGING_LEVEL", "WARNING"
        )
        os.environ["VLLM_CONFIGURE_LOGGING"] = os.environ.get(
            "VLLM_CONFIGURE_LOGGING", "1"
        )

        env = os.environ.copy()
        if getattr(run_cfg, "cuda_visible_devices", None):
            env["CUDA_VISIBLE_DEVICES"] = str(run_cfg.cuda_visible_devices)

        server_process = subprocess.Popen(
            cmd,
            # Don't redirect stdout/stderr initially - let server logs show
            start_new_session=True,
            env=env,
        )
        # server_process is the process - return so we can shut down later
    with open(f"/tmp/vllm_{run_cfg.server_port}.pid", "w") as f:
        f.write(str(server_process.pid))
    # kill with:
    # pid = int(open("/tmp/vllm_8000.pid").read())
    # os.killpg(pid, signal.SIGTERM)

    for i in range(10):  # 10 * 30 seconds = 5 minutes
        try:
            response = requests.get(f"http://localhost:{server_port}/v1/models")
            if response.status_code == 200:
                print(
                    f"[server_utils.py:start_server_if_not_running] Server is running on port {server_port}"
                )
                # Now detach the server process so it continues after script ends
                print(
                    "[server_utils.py:start_server_if_not_running] Detaching server process..."
                )
                return server_process
        except requests.exceptions.RequestException:
            # print(f"Server is not running on port {server_port}")
            print(
                f"[server_utils.py:start_server_if_not_running] \
                Have waited {i / 2} minutes.  Will wait another {5 - i / 2} minutes"
            )
            time.sleep(30)  # 30 seconds

    print(
        f"Server did not start on port {server_port} after 5 minutes; giving up; terminating + killing"
    )
    server_process.terminate()
    server_process.kill()

    # Wait a moment for process to actually terminate
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Force kill if it doesn't terminate gracefully
        server_process.kill()
        server_process.wait(timeout=5)

    # Assert that the server process was successfully aborted
    assert server_process.poll() is not None, (
        f"Failed to abort server process on port {server_port}; pid: {server_process.pid}"
    )
    print("Server process successfully aborted")
    return server_process


def report_vllm_gpu_utilization(run_cfg: RunConfig, context: str = "post-run") -> None:
    """
    Fetch and print the vLLM GPU utilization metric (max across GPUs) if available.
    """
    server_port = getattr(run_cfg, "server_port", None)
    if server_port is None:
        return  # Local/offline LLM without a persistent server; nothing to query.

    metrics_port = _resolve_metrics_port(server_port, run_cfg)
    if metrics_port is None:
        return

    try:
        resp = requests.get(
            f"http://localhost:{metrics_port}/metrics",
            timeout=2,
        )
        resp.raise_for_status()
    except requests.RequestException as err:
        print(
            f"[server_utils] Unable to fetch vLLM metrics on port {metrics_port}: {err}"
        )
        return

    gpu_utils: list[float] = []
    for line in resp.text.splitlines():
        if not line.startswith("vllm_gpu_utilization"):
            continue
        try:
            value = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        gpu_utils.append(value)

    if not gpu_utils:
        print(
            f"[server_utils] vLLM metrics endpoint responded but no gpu util entries were found (port {metrics_port})"
        )
        return

    max_util_pct = max(gpu_utils) * 100
    print(
        f"[server_utils] vLLM GPU util ({context}): {max_util_pct:.1f}% (metrics port {metrics_port})"
    )

