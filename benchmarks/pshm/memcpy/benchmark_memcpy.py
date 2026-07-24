#!/usr/bin/env python3
"""
Benchmark CPU block copy bandwidth in four scenarios:
  1. memcpy     – ordinary CPU tensor random read + write (Python loop)
  2. shm        – same operation on tensors backed by shared memory (Python loop)
  3. cython     – Cython‑accelerated memcpy on ordinary CPU tensors
  4. cython_shm – Cython‑accelerated memcpy on shared memory tensors

Results are printed as a Markdown table and a bandwidth plot is generated.
"""

import argparse
import contextlib
import multiprocessing as mp
import os
import platform
import random
import sys
import time
from multiprocessing import shared_memory
from unittest.mock import patch

import numpy as np
import torch

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
# Cython integration
# ---------------------------------------------------------------------------
try:
    import pyximport

    pyximport.install()

    _cython_code = """
import cython
from libc.string cimport memcpy

@cython.cdivision
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _cython_block_copy(
    const unsigned char* src,
    unsigned char* dst,
    const Py_ssize_t* src_indices,
    const Py_ssize_t* dst_indices,
    Py_ssize_t n_blocks,
    Py_ssize_t block_bytes) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n_blocks):
        memcpy(dst + dst_indices[i] * block_bytes,
               src + src_indices[i] * block_bytes,
               block_bytes)

@cython.cdivision
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void cython_block_copy(
    const unsigned char[:] src,
    unsigned char[:] dst,
    const Py_ssize_t[:] src_indices,
    const Py_ssize_t[:] dst_indices,
    Py_ssize_t block_bytes):
    cdef Py_ssize_t n = src_indices.shape[0]
    _cython_block_copy(&src[0], &dst[0], &src_indices[0], &dst_indices[0],
                       n, block_bytes)
"""
    _tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pyx_tmp")
    os.makedirs(_tmp_dir, exist_ok=True)
    _pyx_path = os.path.join(_tmp_dir, "_block_copy.pyx")
    with open(_pyx_path, "w") as f:
        f.write(_cython_code)
    sys.path.append(_tmp_dir)
    from _block_copy import cython_block_copy

    HAS_CYTHON = True
    print("Cython block copy module loaded successfully.")
except Exception as e:
    HAS_CYTHON = False
    cython_block_copy = None
    print(f"Cython not available ({e}) – cython benchmarks disabled.")


# ---------------------------------------------------------------------------
# formatting helpers
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
        size = num_bytes / (base**target_exp)
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
# Shared memory helper
# ---------------------------------------------------------------------------
class SharedMemoryTensor:
    """Manages a single shared memory segment wrapped as a torch tensor."""

    def __init__(self, size: int):
        self.size = size
        self._shm = None
        self._process = None
        self._stop_event = mp.Event()
        self._tensor = None

    def __enter__(self):
        """Create the shared memory segment in a worker process and map it."""
        parent_conn, child_conn = mp.Pipe()
        self._process = mp.Process(
            target=self._worker, args=(child_conn, self._stop_event)
        )
        self._process.start()
        shm_name = parent_conn.recv()
        parent_conn.close()

        # Avoid double registration in resource tracker
        with patch(
            "multiprocessing.resource_tracker.register", lambda *args, **kwargs: None
        ):
            self._shm = shared_memory.SharedMemory(name=shm_name)

        np_dtype = np.dtype(np.uint8)
        arr = np.ndarray(self.size, dtype=np_dtype, buffer=self._shm.buf)
        self._tensor = torch.from_numpy(arr)
        return self._tensor

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Signal the worker and clean up resources."""
        # Release the tensor and numpy array to allow shm cleanup
        del self._tensor
        self._shm.close()
        self._stop_event.set()
        self._process.join()
        self._shm.unlink()
        return False

    @staticmethod
    def _worker(conn, stop_event):
        """Worker that creates a shared memory segment and waits for stop."""
        conn.recv()
        pass


# We'll use a simpler factory function with a context manager over two segments.
@contextlib.contextmanager
def shared_tensor_pair(total_bytes: int):
    """Context manager that yields (src_tensor, dst_tensor) backed by shared memory."""

    # Create two shared memory segments each in its own worker.
    # We adapt the original get_shm logic into a small helper.
    def _create_shm_tensor(size):
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

    shm_src, src_t, proc_src, stop_src = _create_shm_tensor(total_bytes)
    shm_dst, dst_t, proc_dst, stop_dst = _create_shm_tensor(total_bytes)
    try:
        yield src_t, dst_t
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
    """Creates a shared memory segment, sends its name, then waits for stop."""
    shm = shared_memory.SharedMemory(size=size, create=True)
    try:
        conn.send(shm.name)
        conn.close()
        stop_event.wait()
    finally:
        shm.close()
        shm.unlink()


# ---------------------------------------------------------------------------
# random indices generation
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
# core measurement function
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
# benchmark scenario constructors
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


def run_shm(total_bytes, block_sizes, n_iters):
    """Shared memory tensor copy using Python loop."""
    with shared_tensor_pair(total_bytes) as (src, dst):
        # Fill with random data
        src[:] = torch.randn(total_bytes // 4, dtype=torch.float32).view(torch.uint8)
        dst[:] = torch.randn(total_bytes // 4, dtype=torch.float32).view(torch.uint8)
        print(f"Allocated {format_size(total_bytes)} for shm")

        def copy_func(src_indices, dst_indices, block_bytes):
            s_view = src.view(-1, block_bytes)
            d_view = dst.view(-1, block_bytes)
            for i, j in zip(src_indices, dst_indices):
                d_view[i] = s_view[j]

        with torch.inference_mode():
            return measure_bandwidth(
                copy_func, total_bytes, block_sizes, n_iters, "shm"
            )


def run_cython(total_bytes, block_sizes, n_iters):
    """Cython-accelerated block copy on ordinary CPU tensors."""
    if not HAS_CYTHON:
        print("Cython not available, skipping benchmark.")
        return [None] * len(block_sizes)

    dtype = torch.uint8
    src = torch.randn(total_bytes // 4, dtype=torch.float32).view(dtype)
    dst = torch.randn(total_bytes // 4, dtype=torch.float32).view(dtype)
    print(f"Allocated {format_size(src.nelement() * src.element_size())} for cython")

    src_flat = src.numpy()
    dst_flat = dst.numpy()

    def copy_func(src_indices, dst_indices, block_bytes):
        cython_block_copy(src_flat, dst_flat, src_indices, dst_indices, block_bytes)

    return measure_bandwidth(copy_func, total_bytes, block_sizes, n_iters, "cython")


def run_cython_shm(total_bytes, block_sizes, n_iters):
    """Cython-accelerated block copy on shared memory tensors."""
    if not HAS_CYTHON:
        print("Cython not available, skipping benchmark.")
        return [None] * len(block_sizes)

    with shared_tensor_pair(total_bytes) as (src, dst):
        src[:] = torch.randn(total_bytes // 4, dtype=torch.float32).view(torch.uint8)
        dst[:] = torch.randn(total_bytes // 4, dtype=torch.float32).view(torch.uint8)
        print(f"Allocated {format_size(total_bytes)} for cython_shm")

        src_flat = src.numpy()
        dst_flat = dst.numpy()

        def copy_func(src_indices, dst_indices, block_bytes):
            cython_block_copy(src_flat, dst_flat, src_indices, dst_indices, block_bytes)

        return measure_bandwidth(
            copy_func, total_bytes, block_sizes, n_iters, "cython_shm"
        )


# ---------------------------------------------------------------------------
# output functions
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
                gib_s = bw / (1024**3)
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
    ax.set_title(f"CPU Block Copy Bandwidth\n({sys_info})")
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
        description="Benchmark CPU block copy bandwidth (memcpy / shm / cython / cython_shm)."
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
        choices=["memcpy", "shm", "cython", "cython_shm"],
        default=["memcpy", "shm", "cython", "cython_shm"],
        help="Which benchmarks to run. Default: memcpy shm cython cython_shm.",
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
        "memcpy": run_memcpy,
        "shm": run_shm,
        "cython": run_cython,
        "cython_shm": run_cython_shm,
    }

    results = {}
    for name in args.bench:
        if name in benchmarks:
            print(f"\n=== Benchmark: {name} ===")
            results[name] = benchmarks[name](total_bytes, block_sizes, n_iters)

    if results:
        print_results_table(block_sizes, results)

    if (not args.no_plot or args.save_plot) and results:
        plot_results(block_sizes, results, output_file=args.save_plot)


if __name__ == "__main__":
    main()
