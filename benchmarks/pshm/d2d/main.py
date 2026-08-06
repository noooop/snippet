#!/usr/bin/env python3
"""
Integrated benchmark: compare GPU‑to‑GPU block‑transfer bandwidth for
three scenarios:
  1. zmq         – GPU data copied to CPU pinned memory, then sent via ZMQ;
                   server uploads it to GPU.
  2. zmq+shm     – GPU data written directly to CPU shared memory via an optimized
                   kernel; server is notified via ZMQ and reads from shared memory,
                   optionally performing an extra CPU copy before uploading to GPU.
                   Two variants: copy enabled and copy disabled.
  3. zmq+queue   – GPU data transferred via CUDA IPC through a
                   torch.multiprocessing.Queue; server is notified via ZMQ.
                   No CPU copy is involved.
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

from vllm import _custom_ops as ops
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor

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
    """Format a byte count as a human‑readable string."""
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
# System information
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
                    mem_kb = int(line.split()[1])
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


# ---------------------------------------------------------------------------
# CPU shared memory helpers (used only by zmq+shm)
# ---------------------------------------------------------------------------

def _create_shm_tensor(size):
    """Create a CPU shared‑memory tensor in a separate process."""
    parent_conn, child_conn = mp.Pipe()
    stop_event = mp.Event()
    process = mp.Process(target=_shm_worker, args=(size, child_conn, stop_event))
    process.start()
    shm_name = parent_conn.recv()
    parent_conn.close()
    with patch(
        "multiprocessing.resource_tracker.register", lambda *args, **kwargs: None
    ):
        shm = shared_memory.SharedMemory(name=shm_name)
    tensor = torch.from_numpy(np.ndarray(size, dtype=np.uint8, buffer=shm.buf))
    return shm, tensor, process, stop_event


@contextlib.contextmanager
def shared_tensor_pair(total_bytes: int):
    """
    Context manager that yields (src_tensor, dst_tensor) backed by CPU shared memory.
    Used only for zmq+shm benchmarks.
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


def blocks_batch(src: torch.Tensor, dst: torch.Tensor, block_size: int):
    """
    Copy a tensor `src` to `dst` using vLLM’s fast CUDA kernel if available.
    Works for both GPU↔CPU and GPU↔GPU copies.
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
# ZMQ server and client (used for all modes)
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

    mode: 'zmq'       – receives data blocks (already copied to CPU by client),
                        then uploads them to GPU.
          'zmq+shm'   – receives index, block size and a copy flag;
                        reads from CPU shared memory, optionally copies within CPU,
                        then uploads to GPU.
          'zmq+queue' – receives index, block size; reads the tensor from a
                        multiprocessing.Queue (GPU tensor via CUDA IPC),
                        then copies it to its own GPU buffer.
    """

    def __init__(self, stop_event: Event, mode: str, shm_name: str, total_bytes: int,
                 queue=None):
        self.mode = mode
        self.total_bytes = total_bytes
        self.shm_name = shm_name
        self.queue = queue
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self.proc = ctx.Process(
            target=_zmq_server_worker,
            args=(child_conn, stop_event, mode, shm_name, total_bytes, queue),
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


def _zmq_server_worker(conn, stop_event, mode, shm_name, total_bytes, queue=None):
    """
    ZMQ server that handles three GPU‑to‑GPU (via CPU or CUDA IPC) protocols.
    """
    context = zmq.Context()
    address = get_open_zmq_ipc_path()
    socket = context.socket(zmq.ROUTER)
    socket.bind(address)

    conn.send(address.encode())
    conn.close()

    dtype = torch.uint8
    # Destination GPU buffer where final data is collected
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
        # Expect at least identity + delimiter
        if len(frames) < 3:
            continue

        if mode == "zmq":
            # Format: identity, delimiter, size_frame, payload, d_idx_frame
            if len(frames) < 5:
                continue
            _, size_frame, payload, d_idx_frame = frames[2:6]
            expected_size = struct.unpack("<I", size_frame.buffer)[0]
            d_idx = struct.unpack("<I", d_idx_frame.buffer)[0]
            assert expected_size == len(payload.buffer)
            start = d_idx * expected_size

            # Upload from CPU to GPU
            arr = np.frombuffer(memoryview(payload), dtype=np.uint8)
            payload_tensor = torch.from_numpy(arr)
            payload_tensor = payload_tensor.pin_memory()
            blocks_batch(src=payload_tensor, dst=dst[start: start + expected_size],
                         block_size=expected_size)

        elif mode == "zmq+shm":
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

            # Read from CPU shared memory, optionally copy, then upload to GPU
            if copy_data:
                payload_tensor = shm_tensor[start: start + expected_size].clone()
                payload_tensor = payload_tensor.pin_memory()
            else:
                payload_tensor = shm_tensor[start: start + expected_size]
            blocks_batch(src=payload_tensor, dst=dst[start: start + expected_size],
                         block_size=expected_size)

        elif mode == "zmq+queue":
            # Format: identity, delimiter, idx_frame (8 bytes), size_frame (4 bytes)
            if len(frames) < 4:
                continue
            idx_frame, size_frame = frames[2:4]
            idx = struct.unpack("<Q", idx_frame.buffer)[0]
            expected_size = struct.unpack("<I", size_frame.buffer)[0]
            start = idx * expected_size

            # Retrieve GPU tensor from queue (passed via CUDA IPC)
            try:
                d_idx, tensor = queue.get(timeout=5)
            except Exception:
                socket.send_multipart([identity, EMPTY, b"ERROR"])
                continue

            assert d_idx == idx, f"Mismatched index: queue {d_idx} vs request {idx}"

            # GPU -> GPU copy within server's context
            blocks_batch(src=tensor, dst=dst[start: start + expected_size],
                         block_size=expected_size)

        socket.send_multipart([identity, EMPTY, OK])

    if shm_np is not None:
        shm.close()


class ZmqClient:
    """
    ZMQ client with a small pool of REQ sockets.
    Provides two request methods:
      - request_data(payload, d_idx)       for 'zmq' mode
      - request_index(idx, size, copy_flag) for 'zmq+shm' and 'zmq+queue'
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

    def request_data(self, payload: memoryview, d_idx):
        """Send a data block (GPU data already moved to CPU) for 'zmq' mode."""
        sock = self._pool.pop() if self._pool else self._init_sock()
        size_frame = struct.pack("<I", len(payload))
        d_idx_frame = struct.pack("<I", d_idx)
        sock.send_multipart([b"", size_frame, payload, d_idx_frame], copy=False)
        sock.recv_multipart(copy=False)
        self._pool.append(sock)

    def request_index(self, idx: int, block_bytes: int, copy_data: bool = True):
        """
        Send an index (block number), block size, and a copy flag.
        Used for 'zmq+shm' (copy flag matters) and 'zmq+queue' (flag ignored).
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
    copy_func(src_indices, dst_indices, block_bytes) performs the actual transfer.
    Returns a list of bandwidths in bytes/s, one per block_size.
    """
    bandwidths = []
    for bs_bytes in block_sizes:
        num_blocks = total_bytes // bs_bytes

        # Warm‑up
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
# Benchmark implementations
# ---------------------------------------------------------------------------
def run_zmq(total_bytes, block_sizes, n_iters):
    """
    GPU‑to‑GPU transfer via ZMQ (through CPU):
    The client copies a GPU block to CPU, then sends it via ZMQ.
    The server receives it and uploads to its GPU buffer.
    """
    if not HAS_ZMQ:
        print("ZMQ not installed – skipping zmq benchmark.")
        return [None] * len(block_sizes)

    dtype = torch.uint8
    # Source tensor resides on GPU
    src = torch.randn(total_bytes // 4, dtype=torch.float32, device="cuda:0").view(dtype)
    print(f"Allocated {format_size(total_bytes)} GPU source tensor (zmq)")

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    server = ZmqServerProc(stop_event, mode="zmq", shm_name="", total_bytes=total_bytes)
    server.start()
    print(f"ZMQ server (zmq) bound to {server.address}")

    client = ZmqClient(server.address, init_pool_size=1)
    bandwidths = []

    for bs_bytes in block_sizes:
        num_blocks = total_bytes // bs_bytes
        src_block_view = src.view(num_blocks, bs_bytes)

        # Warm‑up
        src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
        for s_idx, d_idx, in zip(src_indices, dst_indices):
            src_block_view_cpu = src_block_view[s_idx].cpu()
            data_view = memoryview(src_block_view_cpu.numpy())
            client.request_data(data_view, d_idx)

        # Timed run
        src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
        start = time.perf_counter()
        for s_idx, d_idx, in zip(src_indices, dst_indices):
            src_block_view_cpu = src_block_view[s_idx].cpu()
            data_view = memoryview(src_block_view_cpu.numpy())
            client.request_data(data_view, d_idx)
        elapsed = time.perf_counter() - start

        bw = (bs_bytes * n_iters) / elapsed
        bandwidths.append(bw)
        print(f"[zmq] size: {format_size(bs_bytes)}, Bandwidth: {format_bandwidth(bw)}")

    server.close()
    return bandwidths


def run_zmq_shm(total_bytes, block_sizes, n_iters, copy_data=True):
    """
    GPU‑to‑GPU transfer via ZMQ + CPU shared memory:
    The client uses a GPU kernel to write a block directly into CPU shared memory,
    then notifies the server via ZMQ.  The server reads from shared memory,
    optionally copies within CPU, then uploads to GPU.
    """
    if not HAS_ZMQ:
        print("ZMQ not installed – skipping zmq+shm benchmark.")
        return [None] * len(block_sizes)

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()

    bandwidths = []

    with shared_tensor_pair(total_bytes) as (_, dst, shm_src, shm_dst):
        # GPU source tensor
        src = torch.randn(total_bytes // 4, dtype=torch.float32, device="cuda:0").view(torch.uint8)
        # Initialize the CPU‑side shared memory destination (used as scratchpad)
        dst[:] = torch.randn(total_bytes // 4, dtype=torch.float32).view(torch.uint8)
        print(f"Allocated CPU shared memory segment of {format_size(total_bytes)} for zmq+shm")

        # Start server once; it reads from the same shared memory (shm_dst)
        server = ZmqServerProc(
            stop_event, mode="zmq+shm", shm_name=shm_dst.name, total_bytes=total_bytes
        )
        server.start()
        print(f"ZMQ server (zmq+shm) bound to {server.address}")

        client = ZmqClient(server.address, init_pool_size=1)

        label = "zmq+shm (copy)" if copy_data else "zmq+shm (no-copy)"
        for bs_bytes in block_sizes:
            num_blocks = total_bytes // bs_bytes

            src_view = src.view(num_blocks, bs_bytes)
            dst_view = dst.view(num_blocks, bs_bytes)

            # Warm‑up
            src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
            for i, j in zip(src_indices, dst_indices):
                # GPU -> CPU shared memory via optimized kernel
                blocks_batch(src_view[i], dst_view[j], bs_bytes)
                # Notify server (read from shared memory, upload to GPU)
                client.request_index(j, bs_bytes, copy_data=copy_data)

            # Timed run
            src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
            start = time.perf_counter()
            for i, j in zip(src_indices, dst_indices):
                blocks_batch(src_view[i], dst_view[j], bs_bytes)
                client.request_index(j, bs_bytes, copy_data=copy_data)
            elapsed = time.perf_counter() - start

            bw = (bs_bytes * n_iters) / elapsed
            bandwidths.append(bw)
            print(
                f"[{label}] size: {format_size(bs_bytes)}, Bandwidth: {format_bandwidth(bw)}"
            )

    server.close()
    return bandwidths


def run_zmq_queue(total_bytes, block_sizes, n_iters):
    """
    GPU‑to‑GPU transfer via ZMQ + torch.multiprocessing.Queue:
    Client places a GPU tensor slice into a Queue (CUDA IPC),
    then notifies the server via ZMQ.
    Server pops the slice and copies it to its own GPU buffer using blocks_batch.
    No CPU memory is involved.
    """
    if not HAS_ZMQ:
        print("ZMQ not installed – skipping zmq+queue benchmark.")
        return [None] * len(block_sizes)

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()

    # Queue holds references to GPU tensors (CUDA IPC handles)
    queue = ctx.Queue(maxsize=n_iters * 2)

    # GPU source tensor
    dtype = torch.uint8
    src = torch.randn(total_bytes // 4, dtype=torch.float32,
                      device="cuda:0").view(dtype)

    print(f"Allocated {format_size(total_bytes)} GPU source tensor (zmq+queue)")

    # Start ZMQ server (it will use its own internal dst GPU buffer)
    server = ZmqServerProc(
        stop_event,
        mode="zmq+queue",
        shm_name="",            # not used in this mode
        total_bytes=total_bytes,
        queue=queue,
    )
    server.start()
    print(f"ZMQ server (zmq+queue) bound to {server.address}")

    client = ZmqClient(server.address, init_pool_size=1)
    bandwidths = []

    for bs_bytes in block_sizes:
        num_blocks = total_bytes // bs_bytes
        src_view = src.view(num_blocks, bs_bytes)

        # Warm‑up
        src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
        for i, j in zip(src_indices, dst_indices):
            # Put a slice of the GPU tensor into the Queue (CUDA IPC)
            data = src_view[i].clone()
            data.share_memory_()   # enable CUDA IPC for this tensor

            queue.put((j, data))
            client.request_index(j, bs_bytes, copy_data=False)

        # Timed run
        src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
        start = time.perf_counter()
        for i, j in zip(src_indices, dst_indices):
            queue.put((j, src_view[i]))
            client.request_index(j, bs_bytes, copy_data=False)
        elapsed = time.perf_counter() - start

        bw = (bs_bytes * n_iters) / elapsed
        bandwidths.append(bw)
        print(f"[zmq+queue] size: {format_size(bs_bytes)}, Bandwidth: {format_bandwidth(bw)}")

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
    print("### GPU‑to‑GPU Bandwidth Results (GiB/s)")
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
    ax.set_title(f"GPU‑to‑GPU Block Transfer Bandwidth (via CPU)\n({sys_info})")
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
        description="Benchmark GPU‑to‑GPU block transfer using ZMQ, ZMQ+SHM, and ZMQ+Queue."
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
        choices=["zmq", "zmq+shm", "zmq+queue"],
        default=["zmq", "zmq+shm", "zmq+queue"],
        help="Which benchmarks to run. Default: all three.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display or save the bandwidth plot.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="gpu_to_gpu_bandwidth.png",
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

    # Map benchmark names to functions
    benchmarks = {
        "zmq": run_zmq,
        "zmq+shm": run_zmq_shm,
        "zmq+queue": run_zmq_queue,
    }

    results = {}
    for name in args.bench:
        if name == "zmq+shm":
            # Run both copy and no‑copy variants
            results["zmq+shm (copy)"] = run_zmq_shm(
                total_bytes, block_sizes, n_iters, copy_data=True
            )
            results["zmq+shm (no-copy)"] = run_zmq_shm(
                total_bytes, block_sizes, n_iters, copy_data=False
            )
        else:
            if name in benchmarks:
                print(f"\n=== Benchmark: {name} ===")
                results[name] = benchmarks[name](total_bytes, block_sizes, n_iters)

    if results:
        print_results_table(block_sizes, results)

    if (not args.no_plot or args.save_plot) and results:
        plot_results(block_sizes, results, output_file=args.save_plot)


if __name__ == "__main__":
    main()