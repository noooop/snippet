import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import argparse
import signal
import socket
import subprocess
import time

import requests
import numpy as np
from concurrent.futures import ProcessPoolExecutor

PORT = 8000
RENDER_URL = f"http://localhost:{PORT}/v1/completions/render"


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #
def run_server(args):
    serve_cmd = [
        "vllm",
        "serve",
        args.model,
        "--renderer-num-workers",
        str(args.renderer_num_workers),
        "--api-server-count",
        str(args.api_server_count),
        "--disable-uvicorn-access-log",
        "--enable-scale-out",
        "--port",
        str(PORT),
    ]

    # start_new_session=True -> dedicated session/process group, so we can
    # killpg the whole tree (launcher + api workers + renderer workers)
    # and avoid orphaned processes squatting on port 8000.
    process = subprocess.Popen(serve_cmd, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def wait(args, proc, timeout=600.0):
    """Poll the render endpoint until ready, bailing out if the server dies."""
    payload = {"model": args.model, "prompt": "hello"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        # If the launcher has already exited (e.g. failed to start),
        # fail fast instead of polling forever.
        if proc.poll() is not None:
            raise RuntimeError(
                f"render server exited early with code {proc.returncode}"
            )
        try:
            r = requests.post(RENDER_URL, json=payload, timeout=5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("render server did not become ready in time")


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def _benchmark(args):
    from gevent.pool import Pool
    from gevent import monkey

    monkey.patch_socket()

    prompt = args.prompt
    n_tasks = args.num_tasks

    def worker(_):
        payload = {"model": args.model, "prompt": prompt}
        # Belt-and-suspenders: explicitly ask the server to close the TCP
        # connection after sending the response.
        headers = {"Connection": "close"}
        start = time.perf_counter()
        # Create a fresh Session per request; on exit, `with` calls
        # Session.close(), which clears that Session's connection pool
        # and closes its idle connections.
        with requests.Session() as session:
            response = session.post(RENDER_URL, json=payload, headers=headers)
            assert response.status_code == 200
            len(response.content)
        end = time.perf_counter()
        return end - start

    for n_clients in args.n_clients_list:
        latency_list = []
        p = Pool(n_clients)
        start = time.perf_counter()
        for metrics in p.imap_unordered(worker, range(n_tasks)):
            latency_list.append(metrics)
        end = time.perf_counter()

        elapsed_time = end - start
        latency = float(np.mean(latency_list))
        print(
            f"renderer_num_workers: {args.renderer_num_workers}, "
            f"api_server_count: {args.api_server_count}, "
            f"n_clients: {n_clients}, "
            f"Throughput: {n_tasks / elapsed_time:.4f} requests/s, "
            f"Latency {latency * 1000:0.4f} ms",
            flush=True,
        )

def benchmark(args):
    # Run the gevent benchmark in a separate process so that
    # monkey.patch_socket() does not leak into the parent process.
    with ProcessPoolExecutor(1) as executor:
        f = executor.submit(_benchmark, args)
        f.result()


# --------------------------------------------------------------------------- #
# Shutdown / sweep driver
# --------------------------------------------------------------------------- #
def _kill_process_group(proc, sig):
    """Send `sig` to the whole process group led by `proc` (best-effort)."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _shutdown_server(proc, term_timeout=15.0, kill_timeout=10.0):
    """Gracefully terminate the vllm launcher *tree* and reap the launcher.

    - SIGTERM to the process group so SIGTERM handlers inside the launcher
      and workers fire, tearing children down on their own.
    - proc.wait() MUST be called; otherwise the launcher becomes a zombie.
    - SIGKILL fallback if SIGTERM is ignored.
    """
    _kill_process_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=term_timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    _kill_process_group(proc, signal.SIGKILL)
    try:
        proc.wait(timeout=kill_timeout)
    except subprocess.TimeoutExpired:
        print(
            f"[warn] launcher pid={proc.pid} still alive after SIGKILL",
            flush=True,
        )


def run(args):
    # ---- Sweep ----------------------------------------------------------- #
    for num_workers in args.renderer_num_workers_list:
        for api_server_count in args.api_server_count_list:
            print(
                f"renderer_num_workers = {num_workers}, "
                f"api_server_count = {api_server_count}",
                flush=True,
            )
            proc = None
            try:
                args.renderer_num_workers = num_workers
                args.api_server_count = api_server_count
                proc = run_server(args)
                wait(args, proc)
                time.sleep(10)
                benchmark(args)
            finally:
                if proc is not None:
                    _shutdown_server(proc)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM renderer sweeping over "
        "--renderer-num-workers and --api-server-count."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path passed to vllm launch render.",
    )
    parser.add_argument(
        "--renderer-num-workers-list",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Values of --renderer-num-workers to sweep.",
    )
    parser.add_argument(
        "--api-server-count-list",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Values of --api-server-count to sweep.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="hello " * 10000,
        help="Prompt sent to the render endpoint.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=1024,
        help="Total number of requests per (num_workers, api_server_count) run.",
    )
    parser.add_argument(
        "--n-clients-list",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="Values of concurrent client count to sweep.",
    )
    parser.add_argument(
        "--kill-leftovers",
        action="store_true",
        help="Before the sweep, SIGKILL any render-server processes left "
        "over from previous runs.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)