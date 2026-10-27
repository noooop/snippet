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
# Process inspection helpers (Linux /proc only)
# --------------------------------------------------------------------------- #
def _iter_proc():
    """Yield (pid, argv) for every other process on the system."""
    self_pid = os.getpid()
    try:
        entries = os.listdir("/proc")
    except FileNotFoundError:
        return
    for name in entries:
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == self_pid:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        argv = [p for p in raw.split(b"\x00") if p]
        yield pid, argv


def _read_pgid(pid):
    """Return pgid of `pid` from /proc/<pid>/stat, or None if gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    rp = data.rfind(b")")
    if rp < 0:
        return None
    try:
        # stat fields after ") ": state ppid pgrp ...  -> pgrp is index 2
        return int(data[rp + 2:].split()[2])
    except (IndexError, ValueError):
        return None


def _looks_like_render(argv):
    """Heuristic: does this argv belong to our render-server tree?

    Matches:
      * `vllm launch render ...`                (launcher)
      * `... vllm.entrypoints.launchers.render` (renderer entrypoint)
      * processes whose argv carries the "render" service tag after
        `set_process_title("APIServer", rank)` rewrote /proc/<pid>/cmdline
    """
    if not argv:
        return False
    joined = " ".join(p.decode(errors="replace") for p in argv)
    if "render" not in joined:
        return False
    if "vllm" in joined:
        return True
    if "APIServer" in joined or "ApiServer" in joined:
        return True
    if "launchers.render" in joined:
        return True
    return False


def find_render_processes(pgids=None):
    """Return [(pid, pgid, cmdline)] for render-server-looking processes.

    If `pgids` is provided, only processes whose pgid is in that set are
    returned (used to detect orphans left in a known launcher's group).
    """
    hits = []
    for pid, argv in _iter_proc():
        if not _looks_like_render(argv):
            continue
        pgid = _read_pgid(pid)
        if pgids is not None and pgid not in pgids:
            continue
        cmd = " ".join(p.decode(errors="replace") for p in argv)
        hits.append((pid, pgid, cmd))
    return hits


def _is_port_free(port):
    """Probe `port` by attempting to bind; True means it is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _report_and_maybe_kill(label, procs, kill=False):
    """Print a table of `procs` and, if `kill`, SIGKILL their process groups."""
    if not procs:
        return
    print(
        f"[warn] {label}: {len(procs)} leftover render process(es):",
        flush=True,
    )
    for pid, pgid, cmd in procs:
        print(f"    pid={pid} pgid={pgid} cmd={cmd[:200]}", flush=True)
    if not kill:
        return
    pgids = {pgid for _, pgid, _ in procs if pgid is not None}
    for g in pgids:
        try:
            os.killpg(g, signal.SIGKILL)
        except ProcessLookupError:
            pass


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #
def run_server(args):
    serve_cmd = [
        "vllm",
        "launch",
        "render",
        args.model,
        "--renderer-num-workers",
        str(args.renderer_num_workers),
        "--api-server-count",
        str(args.api_server_count),
        "--disable-uvicorn-access-log",
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
    # ---- Pre-flight: clean up a dirty environment ------------------------- #
    pre = find_render_processes()
    if pre:
        _report_and_maybe_kill(
            "pre-flight leftover render process(es)",
            pre,
            kill=args.kill_leftovers,
        )
        if args.kill_leftovers:
            time.sleep(1)
            remaining = find_render_processes()
            if remaining:
                print(
                    f"[warn] pre-flight: {len(remaining)} process(es) survived SIGKILL",
                    flush=True,
                )
            else:
                print("[info] pre-flight: leftover render processes cleaned", flush=True)

    if not _is_port_free(PORT):
        raise RuntimeError(
            f"port {PORT} is already in use; an orphaned render server may be "
            f"holding it (rerun with --kill-leftovers to auto-clean)"
        )

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
                benchmark(args)
            finally:
                if proc is not None:
                    # start_new_session=True -> pgid == launcher pid.
                    # After _shutdown_server() reaps the launcher, any process
                    # still alive in this pgid is by definition an orphan.
                    pgid = proc.pid
                    _shutdown_server(proc)
                    orphans = find_render_processes(pgids={pgid})
                    _report_and_maybe_kill(
                        f"orphans after sweep (pgid={pgid})",
                        orphans,
                        kill=True,
                    )
                    if not _is_port_free(PORT):
                        print(
                            f"[warn] port {PORT} still busy after sweep "
                            f"(possible surviving connection/process)",
                            flush=True,
                        )


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