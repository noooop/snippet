#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark: shared listen fd vs per-process SO_REUSEPORT bind.

Two server architectures are compared on localhost:

1. ``shared_fd``: the parent binds+listens once and passes the same socket
   to every worker (this is what vLLM's APIServerProcessManager does).
2. ``reuseport``: each worker creates its own socket, sets SO_REUSEPORT and
   binds the same port independently.

Both run a short-connection workload (connection setup heavy) and a
long-connection workload (request heavy).

Run:
    python bench_listen_modes.py

Notes:
  * Loopback only; absolute numbers are machine-specific. Focus on the
    shared_fd / reuseport ratio, not the raw RPS.
  * The short-conn test creates many TCP connections; if you run it
    repeatedly you may hit TIME_WAIT. Keep --short-conns modest.
  * Client threads and server workers share the same machine, so they
    compete for CPU. More cores => cleaner signal.
"""

import argparse
import multiprocessing as mp
import socket
import threading
import time

HOST = "127.0.0.1"
REQUEST = b"ping"
RESPONSE = b"pong"
BACKLOG = 4096
ACCEPT_TIMEOUT = 0.2


# ── server ──────────────────────────────────────────────────────────────────


def _serve_loop(lsock, n_requests_per_conn, ready, stop, counter):
    lsock.settimeout(ACCEPT_TIMEOUT)
    local = 0
    with ready.get_lock():
        ready.value += 1
    while not stop.is_set():
        try:
            conn, _ = lsock.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with conn:
            served = 0
            while n_requests_per_conn is None or served < n_requests_per_conn:
                buf = b""
                while len(buf) < len(REQUEST):
                    chunk = conn.recv(len(REQUEST) - len(buf))
                    if not chunk:
                        break
                    buf += chunk
                if len(buf) < len(REQUEST):
                    break
                conn.sendall(RESPONSE)
                served += 1
                local += 1
    with counter.get_lock():
        counter.value += local


def _serve_shared(lsock, n_requests_per_conn, ready, stop, counter):
    _serve_loop(lsock, n_requests_per_conn, ready, stop, counter)


def _serve_reuseport(port, n_requests_per_conn, ready, stop, counter):
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    lsock.bind((HOST, port))
    lsock.listen(BACKLOG)
    try:
        _serve_loop(lsock, n_requests_per_conn, ready, stop, counter)
    finally:
        lsock.close()


# ── client ──────────────────────────────────────────────────────────────────


def _client(port, n_conns, n_reqs, latencies, errors, stop):
    for _ in range(n_conns):
        if stop.is_set():
            return
        try:
            s = socket.create_connection((HOST, port), timeout=10)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                for _ in range(n_reqs):
                    t0 = time.perf_counter_ns()
                    s.sendall(REQUEST)
                    buf = b""
                    while len(buf) < len(RESPONSE):
                        chunk = s.recv(len(RESPONSE) - len(buf))
                        if not chunk:
                            raise ConnectionError("peer closed")
                        buf += chunk
                    t1 = time.perf_counter_ns()
                    latencies.append(t1 - t0)
            finally:
                s.close()
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))


def _run_clients(port, n_threads, n_conns_per_thread, n_reqs, stop):
    local_lat = [[] for _ in range(n_threads)]
    local_err = [[] for _ in range(n_threads)]
    threads = [
        threading.Thread(
            target=_client,
            args=(port, n_conns_per_thread, n_reqs,
                  local_lat[i], local_err[i], stop),
            daemon=True,
        )
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lat = [x for sub in local_lat for x in sub]
    err = [e for sub in local_err for e in sub]
    return lat, err


# ── orchestration ───────────────────────────────────────────────────────────


def _start_servers(mode, n_workers, n_requests_per_conn):
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    ready = ctx.Value("i", 0)
    counters = [ctx.Value("q", 0) for _ in range(n_workers)]
    procs = []
    placeholder = None
    lsock = None

    if mode == "shared_fd":
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        lsock.bind((HOST, 0))
        lsock.listen(BACKLOG)
        port = lsock.getsockname()[1]
        for i in range(n_workers):
            p = ctx.Process(
                target=_serve_shared,
                args=(lsock, n_requests_per_conn, ready, stop, counters[i]),
            )
            p.start()
            procs.append(p)
    elif mode == "reuseport":
        # Hold the port so it cannot be stolen, but do NOT listen: a
        # non-listening socket is not part of the kernel's reuseport
        # dispatch group, so it will never receive connections.
        placeholder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        placeholder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        placeholder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        placeholder.bind((HOST, 0))
        port = placeholder.getsockname()[1]
        for i in range(n_workers):
            p = ctx.Process(
                target=_serve_reuseport,
                args=(port, n_requests_per_conn, ready, stop, counters[i]),
            )
            p.start()
            procs.append(p)
    else:
        raise ValueError(f"unknown mode: {mode}")

    while ready.value < n_workers:
        time.sleep(0.005)

    if lsock is not None:
        # Workers have their own dup'd fd now; parent can release its copy.
        lsock.close()

    return port, procs, stop, counters, placeholder


def _stop_servers(procs, stop, placeholder):
    stop.set()
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)
    if placeholder is not None:
        placeholder.close()


def _pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = min(int(len(sorted_vals) * p / 100.0), len(sorted_vals) - 1)
    return sorted_vals[idx]


def run_one(mode, n_workers, n_threads, n_conns_per_thread, n_reqs_per_conn,
            warmup_conns=0):
    port, procs, stop, counters, placeholder = _start_servers(
        mode, n_workers, n_reqs_per_conn
    )
    try:
        if warmup_conns:
            _run_clients(port, min(n_threads, 2), warmup_conns,
                         n_reqs_per_conn, stop)

        t0 = time.perf_counter()
        lat, err = _run_clients(port, n_threads, n_conns_per_thread,
                                n_reqs_per_conn, stop)
        elapsed = time.perf_counter() - t0
    finally:
        _stop_servers(procs, stop, placeholder)

    lat_ns = sorted(lat)
    total = len(lat)
    rps = total / elapsed if elapsed > 0 else 0.0

    return {
        "mode": mode,
        "rps": rps,
        "total": total,
        "errors": len(err),
        "elapsed": elapsed,
        "p50_us": _pct(lat_ns, 50) / 1e3,
        "p95_us": _pct(lat_ns, 95) / 1e3,
        "p99_us": _pct(lat_ns, 99) / 1e3,
        "max_us": (lat_ns[-1] / 1e3) if lat_ns else 0.0,
        "worker_counts": [c.value for c in counters],
    }


def _print_result(label, r):
    print(f"--- {label} ---")
    print(
        f"  rps={r['rps']:10.1f}  total={r['total']:7d}  "
        f"errors={r['errors']:4d}  elapsed={r['elapsed']:.2f}s"
    )
    print(
        f"  latency (us): p50={r['p50_us']:8.1f}  p95={r['p95_us']:8.1f}  "
        f"p99={r['p99_us']:8.1f}  max={r['max_us']:9.1f}"
    )
    counts = r["worker_counts"]
    total_w = sum(counts) or 1
    shares = [f"{100.0 * c / total_w:.1f}%" for c in counts]
    print(f"  worker requests: {counts}  shares: {shares}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--short-conns", type=int, default=500,
                    help="connections per client thread, short-conn test")
    ap.add_argument("--long-conns", type=int, default=16,
                    help="connections per client thread, long-conn test")
    ap.add_argument("--long-reqs", type=int, default=200,
                    help="requests per connection, long-conn test")
    ap.add_argument("--warmup-conns", type=int, default=20)
    args = ap.parse_args()

    print(
        f"workers={args.workers} threads={args.threads} host={HOST}\n"
        f"(loopback, absolute numbers are machine-specific; compare ratios)\n"
    )

    short_cfg = dict(
        n_workers=args.workers,
        n_threads=args.threads,
        n_conns_per_thread=args.short_conns,
        n_reqs_per_conn=1,
        warmup_conns=args.warmup_conns,
    )
    long_cfg = dict(
        n_workers=args.workers,
        n_threads=args.threads,
        n_conns_per_thread=args.long_conns,
        n_reqs_per_conn=args.long_reqs,
        warmup_conns=0,
    )

    for label, cfg in [
        ("short connection (1 req/conn)", short_cfg),
        (f"long connection ({args.long_reqs} reqs/conn)", long_cfg),
    ]:
        results = {}
        for mode in ("shared_fd", "reuseport"):
            results[mode] = run_one(mode, **cfg)
            _print_result(f"{label}, mode={mode}", results[mode])

        a, b = results["shared_fd"], results["reuseport"]
        if b["rps"] > 0:
            ratio = a["rps"] / b["rps"]
            who = "shared_fd faster" if ratio > 1 else "reuseport faster"
            print(f"  => shared_fd / reuseport rps ratio: {ratio:.3f} ({who})\n")


if __name__ == "__main__":
    main()


"""
python bench_listen_modes.py \
  --workers 4 --threads 8 \
  --short-conns 5000 \
  --long-conns 64 --long-reqs 5000
workers=4 threads=8 host=127.0.0.1
(loopback, absolute numbers are machine-specific; compare ratios)

--- short connection (1 req/conn), mode=shared_fd ---
  rps=   65731.1  total=  40000  errors=   0  elapsed=0.61s
  latency (us): p50=    38.9  p95=    91.0  p99=   121.6  max=    265.3
  worker requests: [9752, 10103, 10001, 10184]  shares: ['24.4%', '25.2%', '25.0%', '25.4%']

--- short connection (1 req/conn), mode=reuseport ---
  rps=   73402.6  total=  40000  errors=   0  elapsed=0.54s
  latency (us): p50=    37.6  p95=    84.1  p99=   109.1  max=    234.4
  worker requests: [9874, 10507, 9715, 9944]  shares: ['24.7%', '26.2%', '24.3%', '24.8%']

  => shared_fd / reuseport rps ratio: 0.895 (reuseport faster)

--- long connection (5000 reqs/conn), mode=shared_fd ---
  rps=  324701.3  total=2560000  errors=   0  elapsed=7.88s
  latency (us): p50=    11.2  p95=    21.1  p99=    27.1  max=  68488.4
  worker requests: [640000, 645000, 640000, 635000]  shares: ['25.0%', '25.2%', '25.0%', '24.8%']

--- long connection (5000 reqs/conn), mode=reuseport ---
  rps=  352869.4  total=2560000  errors=   0  elapsed=7.25s
  latency (us): p50=     8.3  p95=    18.0  p99=    24.0  max= 223067.2
  worker requests: [665000, 510000, 705000, 680000]  shares: ['26.0%', '19.9%', '27.5%', '26.6%']

  => shared_fd / reuseport rps ratio: 0.920 (reuseport faster)
"""