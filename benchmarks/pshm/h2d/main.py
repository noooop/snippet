#!/usr/bin/env python3
"""
Benchmark: compare CPU‑to‑GPU copy bandwidth for two scenarios:
  1. zmq         – send data blocks via ZMQ; server copies to GPU (no extra CPU copy)
  2. zmq+shm     – send block indices and size via ZMQ; server reads from shared memory
                   and copies to GPU.
                   Two variants are tested: copy enabled and copy disabled.

Results are printed as a Markdown table and a bandwidth plot is generated.
"""

import argparse
import contextlib
import multiprocessing as mp
import platform
import random
import struct
import tempfile
import time
import uuid
from multiprocessing import shared_memory
from multiprocessing.synchronize import Event
from unittest.mock import patch

import numpy as np
import torch

# Optional dependency for fast GPU copy (assumes vllm is installed)
try:
    from vllm import _custom_ops as ops
    from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor
except ImportError:
    print("Warning: vllm not found – GPU copy will be disabled.")
    ops = None
    def pin_tensor(tensor):
        return tensor

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

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
        format_size(int(bytes_per_sec), decimal_places=decimal_places, target_unit="GiB")
        + "/s"
    )


# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------
def get_system_info() -> str:
    """Return a string describing CPU model, total memory and GPU info if available."""
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
                    mem_kb = int(line.split()[1])
                    mem_str = f"{mem_kb / (1024**2):.1f} GiB"
                    break
    except Exception:
        mem_str = "unknown"

    gpu_info = ""
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_info = f", GPU: {gpu_name})"
        except Exception:
            gpu_info = ", GPU: detected but details unavailable"

    return f"CPU: {cpu_model}, Memory: {mem_str}{gpu_info}"


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
# Shared memory helpers (for zmq+shm)
# ---------------------------------------------------------------------------
def _create_shm_tensor(size):
    parent_conn, child_conn = mp.Pipe()
    stop_event = mp.Event()
    process = mp.Process(target=_shm_worker, args=(size, child_conn, stop_event))
    process.start()
    shm_name = parent_conn.recv()
    parent_conn.close()
    with patch("multiprocessing.resource_tracker.register", lambda *args, **kwargs: None):
        shm = shared_memory.SharedMemory(name=shm_name)
    tensor = torch.from_numpy(np.ndarray(size, dtype=np.uint8, buffer=shm.buf))
    return shm, tensor, process, stop_event


@contextlib.contextmanager
def shared_tensor_pair(total_bytes: int):
    """
    Context manager that yields (src_tensor, dst_tensor) backed by shared memory.
    Used for the zmq+shm benchmark.
    """
    shm_src, src_t, proc_src, stop_src = _create_shm_tensor(total_bytes)
    shm_dst, dst_t, proc_dst, stop_dst = _create_shm_tensor(total_bytes)
    try:
        yield src_t, dst_t, shm_src, shm_dst
    finally:
        del src_t
        shm_src.close()
        stop_src.set()
        proc_src.join()
        del dst_t
        shm_dst.close()
        stop_dst.set()
        proc_dst.join()


def _shm_worker(size: int, conn: mp.connection.Connection, stop_event: mp.Event):
    """Worker that creates a shared memory segment and waits for stop."""
    shm = shared_memory.SharedMemory(size=size, create=True)
    try:
        conn.send(shm.name)
        conn.close()
        stop_event.wait()
    finally:
        shm.close()
        shm.unlink()


# ---------------------------------------------------------------------------
# GPU copy helper (CPU tensor to GPU tensor)
# ---------------------------------------------------------------------------
def h2d(src: torch.Tensor, dst: torch.Tensor, block_size: int):
    """
    Copy a CPU tensor `src` to a GPU tensor `dst` using a fast CUDA kernel.
    Falls back to a normal copy if vllm/custom_ops is not available.
    """
    if ops is None:
        dst.copy_(src)
        return
    src_addrs = torch.tensor([src.data_ptr()], dtype=torch.int64)
    dst_addrs = torch.tensor([dst.data_ptr()], dtype=torch.int64)
    sizes = torch.tensor([block_size], dtype=torch.int64)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ops.swap_blocks_batch(src_addrs, dst_addrs, sizes)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# ZMQ server and client (used for both 'zmq' and 'zmq+shm')
# ---------------------------------------------------------------------------
POLL_INTERVAL = 1000
EMPTY = b""
OK = b"OK"


def get_open_zmq_ipc_path():
    """Return an available IPC path for ZMQ binding."""
    return f"ipc://{tempfile.gettempdir()}/zmq-{uuid.uuid4().hex}.sock"


class ZmqServerProc:
    """
    Manage a ZMQ server process.
    mode: 'zmq' or 'zmq+shm'
    total_bytes: total shared memory size (only used for 'zmq+shm')
    shm_name: name of shared memory segment (only for 'zmq+shm')
    """

    def __init__(self, stop_event: Event, mode: str, shm_name: str, total_bytes: int):
        self.mode = mode
        self.total_bytes = total_bytes
        self.shm_name = shm_name
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self.proc = ctx.Process(
            target=_zmq_server_worker,
            args=(child_conn, stop_event, mode, shm_name, total_bytes),
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


def _zmq_server_worker(conn, stop_event, mode, shm_name, total_bytes):
    """
    ZMQ server that handles two protocols:
      - 'zmq': receive data block (zero-copy on CPU side) and copy to GPU.
      - 'zmq+shm': receive index (8 bytes), block size (4 bytes), and a copy flag (1 byte).
                   Read from shared memory, optionally copy within CPU, then copy to GPU.
    """
    context = zmq.Context()
    address = get_open_zmq_ipc_path()
    socket = context.socket(zmq.ROUTER)
    socket.bind(address)
    conn.send(address.encode())
    conn.close()

    dtype = torch.uint8
    # Destination tensor on GPU
    dst = torch.randn(total_bytes // 4, dtype=torch.float32, device="cuda:0").view(dtype)

    shm_np = None
    shm_tensor = None
    if mode == "zmq+shm" and shm_name:
        with patch("multiprocessing.resource_tracker.register", lambda *args, **kwargs: None):
            shm = shared_memory.SharedMemory(name=shm_name)
        shm_np = np.ndarray(total_bytes, dtype=np.uint8, buffer=shm.buf)
        shm_tensor = torch.from_numpy(shm_np)
        if pin_tensor is not None:
            pin_tensor(shm_tensor)

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

        identity = frames[0]
        if len(frames) < 3:
            continue

        if mode == "zmq":
            # Format: identity, delimiter, flag, size_frame, payload, d_idx_frame
            if len(frames) < 6:
                continue
            flag, size_frame, payload, d_idx_frame = frames[2:6]
            expected_size = struct.unpack("<I", size_frame.buffer)[0]
            d_idx = struct.unpack("<I", d_idx_frame.buffer)[0]
            assert expected_size == len(payload.buffer)
            start = d_idx * expected_size

            arr = np.frombuffer(memoryview(payload), dtype=np.uint8)
            payload_tensor = torch.from_numpy(arr)
            payload_tensor = payload_tensor.pin_memory()
            # Copy to GPU
            h2d(src=payload_tensor, dst=dst[start: start + expected_size], block_size=expected_size)
        else:  # 'zmq+shm'
            # Format: identity, delimiter, idx_frame (8 bytes), size_frame (4 bytes), copy_flag (1 byte)
            if len(frames) < 5:
                continue
            idx_frame, size_frame, copy_flag_frame = frames[2:5]
            idx = struct.unpack("<Q", idx_frame.buffer)[0]
            expected_size = struct.unpack("<I", size_frame.buffer)[0]
            copy_data = struct.unpack("<?", copy_flag_frame.buffer)[0]
            start = idx * expected_size
            if start + expected_size > total_bytes:
                socket.send_multipart([identity, EMPTY, b"ERROR"])
                continue

            # Read from shared memory and optionally copy within CPU (not needed for GPU path)
            if copy_data:
                payload_tensor = shm_tensor[start: start + expected_size].clone()
                payload_tensor = payload_tensor.pin_memory()
            else:
                payload_tensor = shm_tensor[start: start + expected_size]
            # Copy to GPU
            h2d(src=payload_tensor, dst=dst[start: start + expected_size], block_size=expected_size)

        socket.send_multipart([identity, EMPTY, OK])

    if shm_np is not None:
        shm.close()


class ZmqClient:
    """
    ZMQ client with a small pool of REQ sockets.
    Provides two request methods:
      - request_data(payload)        for 'zmq' mode
      - request_index(idx, size, copy_flag) for 'zmq+shm' mode
    """

    def __init__(self, address: str, init_pool_size: int = 4):
        self._address = address
        self._ctx = zmq.Context()
        self._pool = []
        for _ in range(init_pool_size):
            sock = self._init_sock()
            self._pool.append(sock)

    def _init_sock(self) -> zmq.Socket:
        sock = self._ctx.socket(zmq.REQ)
        sock.connect(self._address)
        return sock

    def request_data(self, payload: memoryview, d_idx: int):
        """Send a data block (for 'zmq' mode)."""
        sock = self._pool.pop() if self._pool else self._init_sock()
        size_frame = struct.pack("<I", len(payload))
        d_idx_frame = struct.pack("<I", d_idx)
        sock.send_multipart([b"", size_frame, payload, d_idx_frame], copy=False)
        sock.recv_multipart(copy=False)
        self._pool.append(sock)

    def request_index(self, idx: int, block_bytes: int, copy_data: bool = True):
        """
        Send an index (block number), block size, and a copy flag (for 'zmq+shm' mode).
        """
        sock = self._pool.pop() if self._pool else self._init_sock()
        idx_frame = struct.pack("<Q", idx)
        size_frame = struct.pack("<I", block_bytes)
        copy_flag = struct.pack("<?", copy_data)
        sock.send_multipart([idx_frame, size_frame, copy_flag], copy=False)
        sock.recv_multipart(copy=False)
        self._pool.append(sock)

    def close(self):
        for sock in self._pool:
            sock.close()
        self._ctx.term()


# ---------------------------------------------------------------------------
# Core measurement function
# ---------------------------------------------------------------------------
def measure_bandwidth(
    copy_func, total_bytes: int, block_sizes: list, n_iters: int, label: str = ""
) -> list:
    """
    Measure bandwidth for a given copy_func over a range of block sizes.
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
# Benchmark implementations (only zmq and zmq+shm)
# ---------------------------------------------------------------------------
def run_zmq(total_bytes, block_sizes, n_iters):
    """ZMQ transfer: client sends data blocks, server copies them to GPU."""
    if not HAS_ZMQ:
        print("ZMQ not installed – skipping zmq benchmark.")
        return [None] * len(block_sizes)

    dtype = torch.uint8
    src = torch.randn(total_bytes // 4, dtype=torch.float32, device="cpu").view(dtype)
    src_np = src.numpy()
    print(f"Allocated {format_size(total_bytes)} for zmq source on CPU")

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    server = ZmqServerProc(stop_event, mode="zmq", total_bytes=total_bytes, shm_name="")
    server.start()
    print(f"ZMQ server (zmq) bound to {server.address}")

    client = ZmqClient(server.address, init_pool_size=1)
    bandwidths = []

    for bs_bytes in block_sizes:
        num_blocks = total_bytes // bs_bytes
        src_block_view = src_np.reshape(num_blocks, bs_bytes)

        # Warm-up
        src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
        for s_idx, d_idx in zip(src_indices, dst_indices):
            data_view = memoryview(src_block_view[s_idx])
            client.request_data(data_view, d_idx)

        # Timed run
        src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
        start = time.perf_counter()
        for s_idx, d_idx in zip(src_indices, dst_indices):
            data_view = memoryview(src_block_view[s_idx])
            client.request_data(data_view, d_idx)
        elapsed = time.perf_counter() - start

        bw = (bs_bytes * n_iters) / elapsed
        bandwidths.append(bw)
        print(f"[zmq] size: {format_size(bs_bytes)}, Bandwidth: {format_bandwidth(bw)}")

    server.close()
    return bandwidths


def run_zmq_shm(total_bytes, block_sizes, n_iters, copy_data=True):
    """
    ZMQ + shared memory: client sends indices and sizes; server reads from shared memory
    and copies to GPU. The `copy_data` flag controls whether an intermediate CPU copy
    is made on the server side before GPU transfer.
    """
    if not HAS_ZMQ:
        print("ZMQ not installed – skipping zmq+shm benchmark.")
        return [None] * len(block_sizes)

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    bandwidths = []

    with shared_tensor_pair(total_bytes) as (src, dst, shm_src, shm_dst):
        src[:] = torch.randn(total_bytes // 4, dtype=torch.float32).view(torch.uint8)
        dst[:] = torch.randn(total_bytes // 4, dtype=torch.float32).view(torch.uint8)
        print(f"Allocated {format_size(total_bytes)} for shm (source and dummy destination)")

        # Start server once; use the source shared memory for reading
        server = ZmqServerProc(
            stop_event, mode="zmq+shm", total_bytes=total_bytes, shm_name=shm_src.name
        )
        server.start()
        print(f"ZMQ server (zmq+shm) bound to {server.address}")

        client = ZmqClient(server.address, init_pool_size=1)

        # We still do a CPU-side copy from src to dst to simulate the cost on the client side
        # (this matches the original benchmark and can be removed if not needed)
        def copy_func(src_indices, dst_indices, block_bytes):
            s_view = src.view(-1, block_bytes)
            d_view = dst.view(-1, block_bytes)
            for i, j in zip(src_indices, dst_indices):
                d_view[i] = s_view[j]

        label = "zmq+shm (copy)" if copy_data else "zmq+shm (no-copy)"
        for bs_bytes in block_sizes:
            num_blocks = total_bytes // bs_bytes

            # Warm-up
            src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
            copy_func(src_indices, dst_indices, bs_bytes)
            for idx in src_indices:
                client.request_index(idx, bs_bytes, copy_data=copy_data)

            # Timed run
            src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
            start = time.perf_counter()
            copy_func(src_indices, dst_indices, bs_bytes)
            for idx in src_indices:
                client.request_index(idx, bs_bytes, copy_data=copy_data)
            elapsed = time.perf_counter() - start

            bw = (bs_bytes * n_iters) / elapsed
            bandwidths.append(bw)
            print(
                f"[{label}] size: {format_size(bs_bytes)}, Bandwidth: {format_bandwidth(bw)}"
            )

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
                row.append(f"{bw / (1024**3):.4f}")
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
                y_vals.append(bw / (1024**3))
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
    ax.set_title(f"CPU-to-GPU Block Copy Bandwidth\n({sys_info})")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU-to-GPU block copy: zmq and zmq+shm (with/without intermediate copy)."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2**32,
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
        "--bench",
        nargs="+",
        choices=["zmq", "zmq+shm"],
        default=["zmq", "zmq+shm"],
        help="Which benchmarks to run. Default: all.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display or save the bandwidth plot.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="cpu_to_gpu_bandwidth.png",
        help="Save the plot to the given file path.",
    )
    args = parser.parse_args()

    total_bytes = args.size
    n_iters = args.n_iters
    block_sizes = [2**n for n in range(args.min_block_exp, args.max_block_exp + 1)]

    max_block = block_sizes[-1]
    if total_bytes % max_block != 0:
        print(
            f"Warning: total_bytes ({total_bytes}) is not a multiple of the largest "
            f"block size ({max_block}). Adjusting total_bytes to the next multiple."
        )
        total_bytes = ((total_bytes // max_block) + 1) * max_block
        print(f"New total_bytes: {total_bytes}")

    if not torch.cuda.is_available():
        print("Warning: CUDA not available – GPU copy will not work properly.")

    # Map benchmark names to functions
    benchmarks = {
        "zmq": run_zmq,
        "zmq+shm": run_zmq_shm,
    }

    results = {}
    for name in args.bench:
        if name == "zmq+shm":
            # Run both copy and no-copy variants
            results["zmq+shm (copy)"] = run_zmq_shm(
                total_bytes, block_sizes, n_iters, copy_data=True
            )
            results["zmq+shm (no-copy)"] = run_zmq_shm(
                total_bytes, block_sizes, n_iters, copy_data=False
            )
        else:
            results[name] = benchmarks[name](total_bytes, block_sizes, n_iters)

    if results:
        print_results_table(block_sizes, results)

    if (not args.no_plot or args.save_plot) and results:
        plot_results(block_sizes, results, output_file=args.save_plot)


if __name__ == "__main__":
    main()