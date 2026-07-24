#!/usr/bin/env python3
"""
Benchmark CPU block copy bandwidth for ordinary CPU tensor random read + write (Python loop)
and compare with ZMQ IPC transfer (zero-copy vs server-side copy).
Results are printed as a Markdown table and a bandwidth plot is generated.
"""

import argparse
import platform
import random
import time
import queue
import struct
import multiprocessing as mp
from multiprocessing.synchronize import Event

import numpy as np
import torch

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

# ---------------------------------------------------------------------------
# matplotlib check
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available – plot will be skipped.")

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_size(
    num_bytes: int,
    decimal_places: int = 4,
    use_binary: bool = True,
    target_unit: str = None,
) -> str:
    """Format a byte count as a human-readable string."""
    if num_bytes == 0:
        return f"0 {target_unit or 'B'}"
    units = ["B", "KiB", "MiB", "GiB"] if use_binary else ["B", "KB", "MB", "GB"]
    base = 1024 if use_binary else 1000
    if target_unit is not None:
        target_exp = units.index(target_unit)
        size = num_bytes / (base ** target_exp)
        return f"{size:.{decimal_places}f} {target_unit}"
    exponent = 0
    size = num_bytes
    while size >= base and exponent < len(units) - 1:
        size /= base
        exponent += 1
    return f"{size:.{decimal_places}f} {units[exponent]}"


def format_bandwidth(bytes_per_sec: float, decimal_places: int = 4) -> str:
    """Format bandwidth in GiB/s."""
    return (
        format_size(
            int(bytes_per_sec), decimal_places=decimal_places, target_unit="GiB"
        )
        + "/s"
    )


# ---------------------------------------------------------------------------
# System info helper
# ---------------------------------------------------------------------------
def get_system_info() -> str:
    """Return a string describing CPU model and total memory."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        cpu_model = platform.machine()

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    mem_kb = int(parts[1])
                    mem_str = f"{mem_kb / (1024**2):.1f} GiB"
                    break
    except Exception:
        mem_str = "unknown"

    return f"CPU: {cpu_model}, Memory: {mem_str}"


# ---------------------------------------------------------------------------
# Random index generation
# ---------------------------------------------------------------------------
def generate_random_indices(num_blocks: int, n_iters: int) -> tuple:
    """Return a pair of numpy arrays (src_indices, dst_indices) of shape (n_iters,)."""
    tasks = [
        (random.randint(0, num_blocks - 1), random.randint(0, num_blocks - 1))
        for _ in range(n_iters)
    ]
    src_idx = np.array([t[0] for t in tasks], dtype=np.intp)
    dst_idx = np.array([t[1] for t in tasks], dtype=np.intp)
    return src_idx, dst_idx

def generate_random_src_indices(num_blocks: int, n_iters: int) -> np.ndarray:
    """Return a numpy array of source indices only."""
    return np.random.randint(0, num_blocks, size=n_iters, dtype=np.intp)


# ---------------------------------------------------------------------------
# ZMQ server / client (adapted from provided code)
# ---------------------------------------------------------------------------
POLL_INTERVAL = 1000
EMPTY = b''
OK = b'OK'

def get_open_zmq_ipc_path():
    """Return an available IPC path for ZMQ binding."""
    import tempfile
    import uuid
    return f"ipc://{tempfile.gettempdir()}/zmq-{uuid.uuid4().hex}.sock"


def _zmq_server(conn, stop_event: Event):
    context = zmq.Context()

    # Bind to an available IPC path
    address = get_open_zmq_ipc_path()
    socket = context.socket(zmq.ROUTER)
    socket.bind(address)

    # Notify parent process of the address
    conn.send(address.encode())
    conn.close()

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    while not stop_event.is_set():
        try:
            socks = dict(poller.poll(POLL_INTERVAL))
        except (zmq.ZMQError, KeyboardInterrupt, EOFError):
            break

        if socket not in socks or socks[socket] != zmq.POLLIN:
            continue

        try:
            frames = socket.recv_multipart(copy=False)
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                break
            continue

        # Expect at least 5 frames: identity, delimiter, copy_flag, size_frame, payload
        if len(frames) < 5:
            continue

        identity, delimiter, copy_flag, size_frame, payload = frames[:5]

        # ----- Size verification -----
        expected_size = struct.unpack("<I", size_frame.buffer)[0]
        assert expected_size == len(payload.buffer)

        if copy_flag == b'c':
            arr = np.frombuffer(payload.buffer, dtype=np.uint8)
            arr_copy = arr.copy()
            # Assert that the copy allocated a new memory region
            assert arr.ctypes.data != arr_copy.ctypes.data, \
                "ZMQ copy mode: data was NOT copied to a new memory region!"

        # Send OK response
        socket.send_multipart([identity, EMPTY, OK])


class ZmqServerProc:
    def __init__(self, stop_event: Event):
        # Use the same context as the stop_event to avoid fork/spawn issues
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self.proc = ctx.Process(
            target=_zmq_server,
            args=(child_conn, stop_event),
        )
        self.stop_event = stop_event
        self.parent_conn = parent_conn
        self.address = ""

    def start(self):
        self.proc.start()
        self.address = self.parent_conn.recv().decode()
        self.parent_conn.close()

    def close(self):
        self.stop_event.set()
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join()


class Client:
    def __init__(self, address: str, init_pool_size: int = 4):
        self._address = address
        self._ctx = zmq.Context()
        self._pool: queue.Queue = queue.Queue()

        for _ in range(init_pool_size):
            sock = self._init_sock()
            self._pool.put(sock)

    def _init_sock(self) -> zmq.Socket:
        sock = self._ctx.socket(zmq.REQ)
        sock.connect(self._address)
        return sock

    def request(self, copy: bool, payload: memoryview):
        try:
            sock = self._pool.get_nowait()
        except queue.Empty:
            sock = self._init_sock()

        flag = b"c" if copy else b""
        size_frame = struct.pack("<I", len(payload))
        sock.send_multipart([flag, size_frame, payload], copy=False)
        sock.recv_multipart(copy=False)
        self._pool.put(sock)


# ---------------------------------------------------------------------------
# Core measurement function
# ---------------------------------------------------------------------------
def measure_bandwidth(
    copy_func, total_bytes: int, block_sizes: list, n_iters: int, label: str = ""
) -> list:
    """
    Measure bandwidth for a given copy_func over a range of block sizes.
    copy_func(src_indices, dst_indices, block_bytes) performs the actual copy.
    Returns a list of bandwidths in bytes/s, one per block_size.
    """
    bandwidths = []
    for bs_bytes in block_sizes:
        num_blocks = total_bytes // bs_bytes

        # Warm-up
        src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
        copy_func(src_indices, dst_indices, bs_bytes)

        # Timed run
        src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
        start = time.perf_counter()
        copy_func(src_indices, dst_indices, bs_bytes)
        elapsed = time.perf_counter() - start

        bw = (bs_bytes * n_iters) / elapsed
        bandwidths.append(bw)
        print(
            f"[{label}] size: {format_size(bs_bytes)}, Bandwidth: {format_bandwidth(bw)}"
        )

    return bandwidths


# ---------------------------------------------------------------------------
# Benchmark scenario – ordinary CPU tensor (memcpy)
# ---------------------------------------------------------------------------
def run_memcpy(total_bytes, block_sizes, n_iters):
    """Ordinary CPU tensor copy using Python loop."""
    dtype = torch.uint8
    src = torch.randn(total_bytes // 4, dtype=torch.float32, device="cpu").view(dtype)
    dst = torch.randn(total_bytes // 4, dtype=torch.float32, device="cpu").view(dtype)
    print(f"Allocated {format_size(src.nelement() * src.element_size())} for memcpy")

    def copy_func(src_indices, dst_indices, block_bytes):
        # Views to enable 2D indexing
        s_view = src.view(-1, block_bytes)
        d_view = dst.view(-1, block_bytes)
        for i, j in zip(src_indices, dst_indices):
            d_view[i] = s_view[j]

    with torch.inference_mode():
        return measure_bandwidth(copy_func, total_bytes, block_sizes, n_iters, "memcpy")


# ---------------------------------------------------------------------------
# Benchmark scenario – ZMQ transfer (with optional server-side copy)
# ---------------------------------------------------------------------------
def run_zmq(total_bytes, block_sizes, n_iters, copy_mode=False):
    """
    ZMQ transfer using a separate server process.
    If copy_mode is True, the server will explicitly copy the received data.
    """
    if not HAS_ZMQ:
        print("ZMQ not installed – skipping ZMQ benchmark.")
        return [None] * len(block_sizes)

    dtype = torch.uint8
    src = torch.randn(total_bytes // 4, dtype=torch.float32, device="cpu").view(dtype)
    src_np = src.numpy()

    mode_str = "copy" if copy_mode else "no_copy"
    print(f"Allocated {format_size(src.nelement() * src.element_size())} for ZMQ source (mode={mode_str})")

    # Use spawn context to avoid fork/spawn SemLock issues
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    server = ZmqServerProc(stop_event)
    server.start()
    print(f"ZMQ server bound to {server.address}")

    # Create client (single socket for sequential requests)
    client = Client(server.address, init_pool_size=1)

    bandwidths = []
    for bs_bytes in block_sizes:
        num_blocks = total_bytes // bs_bytes
        src_block_view = src_np.reshape(num_blocks, bs_bytes)

        # Warm-up
        src_indices = generate_random_src_indices(num_blocks, n_iters)
        for idx in src_indices:
            data_view = memoryview(src_block_view[idx])
            client.request(copy=copy_mode, payload=data_view)

        # Timed run
        src_indices = generate_random_src_indices(num_blocks, n_iters)
        start = time.perf_counter()
        for idx in src_indices:
            data_view = memoryview(src_block_view[idx])
            client.request(copy=copy_mode, payload=data_view)
        elapsed = time.perf_counter() - start

        bw = (bs_bytes * n_iters) / elapsed
        bandwidths.append(bw)
        print(
            f"[zmq_{mode_str}] size: {format_size(bs_bytes)}, Bandwidth: {format_bandwidth(bw)}"
        )

    # Cleanup
    server.close()
    return bandwidths


# ---------------------------------------------------------------------------
# Output functions
# ---------------------------------------------------------------------------
def print_results_table(block_sizes: list, results: dict):
    """
    Print a Markdown table of bandwidths (GiB/s).
    System information is printed before the table.
    """
    sys_info = get_system_info()
    print(f"\n**System:** {sys_info}\n")

    bench_names = list(results.keys())
    header = ["Block Size"] + bench_names
    print("### Bandwidth Results (GiB/s)")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join([" --- " for _ in header]) + "|")

    for i, bs in enumerate(block_sizes):
        row = [format_size(bs, decimal_places=0)]
        for name in bench_names:
            bw = results[name][i]
            if bw is None:
                row.append("N/A")
            else:
                gib_s = bw / (1024 ** 3)
                row.append(f"{gib_s:.4f}")
        print("| " + " | ".join(row) + " |")


def plot_results(block_sizes: list, results: dict, output_file: str = None):
    """
    Bandwidth plot (GiB/s vs block size).
    X‑axis: log2, human‑readable ticks.
    Y‑axis: linear, starting at 0.
    System information is included in the plot title.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available.")
        return

    sys_info = get_system_info()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (name, bw_list) in enumerate(results.items()):
        x_vals, y_vals = [], []
        for bs, bw in zip(block_sizes, bw_list):
            if bw is not None:
                x_vals.append(bs)
                y_vals.append(bw / (1024 ** 3))
        if not x_vals:
            continue
        ax.plot(
            x_vals,
            y_vals,
            label=name,
            color=colors[idx % len(colors)],
            marker="o",
            markersize=4,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(block_sizes)
    ax.set_xticklabels(
        [format_size(bs, decimal_places=0) for bs in block_sizes],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Block Size")
    ax.set_ylabel("Bandwidth (GiB/s)")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
    ax.set_title(f"CPU Block Copy Bandwidth (memcpy vs ZMQ no-copy vs ZMQ copy)\n({sys_info})")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU block copy bandwidth (memcpy vs ZMQ no-copy vs ZMQ copy)."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2 ** 32,
        help="Total memory allocated in bytes (must be divisible by block sizes). Default: 2**32 (4 GiB).",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=100,
        help="Number of timed iterations per block size. Default: 100.",
    )
    parser.add_argument(
        "--min-block-exp",
        type=int,
        default=8,
        help="Exponent for the smallest block size (2**exp bytes). Default: 8 (256 B).",
    )
    parser.add_argument(
        "--max-block-exp",
        type=int,
        default=30,
        help="Exponent for the largest block size (2**exp bytes). Default: 30 (1 GiB).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display or save the bandwidth plot.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="benchmark_cpu_block_copy.png",
        help="Save the plot to the given file path.",
    )
    parser.add_argument(
        "--skip-zmq",
        action="store_true",
        help="Skip ZMQ benchmarks (run only memcpy).",
    )
    args = parser.parse_args()

    total_bytes = args.size
    n_iters = args.n_iters
    block_sizes = [2 ** n for n in range(args.min_block_exp, args.max_block_exp + 1)]

    max_block = block_sizes[-1]
    if total_bytes % max_block != 0:
        print(
            f"Warning: total_bytes ({total_bytes}) is not a multiple of the largest "
            f"block size ({max_block}). Adjusting total_bytes to the next multiple."
        )
        total_bytes = ((total_bytes // max_block) + 1) * max_block
        print(f"New total_bytes: {total_bytes}")

    results = {}

    # Run memcpy benchmark
    print("\n=== Benchmark: memcpy ===")
    results["memcpy"] = run_memcpy(total_bytes, block_sizes, n_iters)

    if not args.skip_zmq:
        print("\n=== Benchmark: ZMQ no-copy ===")
        results["zmq_no_copy"] = run_zmq(total_bytes, block_sizes, n_iters, copy_mode=False)
        print("\n=== Benchmark: ZMQ copy ===")
        results["zmq_copy"] = run_zmq(total_bytes, block_sizes, n_iters, copy_mode=True)
    else:
        print("\nSkipping ZMQ benchmarks as requested.")

    if results:
        print_results_table(block_sizes, results)

    if (not args.no_plot or args.save_plot) and results:
        plot_results(block_sizes, results, output_file=args.save_plot)


if __name__ == "__main__":
    main()