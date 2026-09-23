#!/usr/bin/env python3
"""
Benchmark CPU block copy bandwidth in four scenarios:
  1. memcpy     – ordinary CPU tensor random read + write (Python loop)
  2. shm        – same operation on tensors backed by shared memory (Python loop)
  3. cython_shm – Cython + OpenMP intra-block copy on shared memory tensors
  4. cython_mt  – Cython + OpenMP intra-block copy on ordinary CPU tensors

The Cython path supports three copy backends via --copy-mode:
  simd      – AVX-512 / AVX2 load/store (default)
  simd_nt   – AVX-512 / AVX2 non-temporal store
  memcpy    – libc memcpy (baseline)

Results are printed as a Markdown table and a bandwidth plot is generated.
"""

import argparse
import contextlib
import multiprocessing as mp
import os
import platform
import random
import sys
import tempfile
import time
from multiprocessing import shared_memory
from pathlib import Path
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
# Manual cythonize + Extension build (proven to work with OpenMP).
# Includes SIMD (AVX-512/AVX2) and non-temporal store copy backends.
# ---------------------------------------------------------------------------
CYTHON_SOURCE = r'''
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
from libc.string cimport memcpy
from cython.parallel cimport prange

cdef extern from "omp.h" nogil:
    void omp_set_num_threads(int num_threads)
    int  omp_get_max_threads()
    int  omp_get_num_threads()


cdef extern from *:
    """
    #include <immintrin.h>
    #include <stddef.h>
    #include <string.h>

    /* SIMD copy: AVX-512 if available, else AVX2, else scalar. */
    static inline void simd_copy(void* dst, const void* src, size_t n) {
    #if defined(__AVX512F__)
        size_t i = 0;
        for (; i + 64 <= n; i += 64) {
            __m512i v = _mm512_loadu_si512((const __m512i*)((const char*)src + i));
            _mm512_storeu_si512((__m512i*)((char*)dst + i), v);
        }
        for (; i < n; i++) ((char*)dst)[i] = ((const char*)src)[i];
    #elif defined(__AVX2__)
        size_t i = 0;
        for (; i + 32 <= n; i += 32) {
            __m256i v = _mm256_loadu_si256((const __m256i*)((const char*)src + i));
            _mm256_storeu_si256((__m256i*)((char*)dst + i), v);
        }
        for (; i < n; i++) ((char*)dst)[i] = ((const char*)src)[i];
    #else
        memcpy(dst, src, n);
    #endif
    }

    /* Non-temporal store version (bypasses cache). */
    static inline void simd_copy_nt(void* dst, const void* src, size_t n) {
    #if defined(__AVX512F__)
        size_t i = 0;
        for (; i + 64 <= n; i += 64) {
            __m512i v = _mm512_loadu_si512((const __m512i*)((const char*)src + i));
            _mm512_stream_si512((__m512i*)((char*)dst + i), v);
        }
        _mm_sfence();
        for (; i < n; i++) ((char*)dst)[i] = ((const char*)src)[i];
    #elif defined(__AVX2__)
        size_t i = 0;
        for (; i + 32 <= n; i += 32) {
            __m256i v = _mm256_loadu_si256((const __m256i*)((const char*)src + i));
            _mm256_stream_si256((__m256i*)((char*)dst + i), v);
        }
        _mm_sfence();
        for (; i < n; i++) ((char*)dst)[i] = ((const char*)src)[i];
    #else
        memcpy(dst, src, n);
    #endif
    }
    """
    void simd_copy(void* dst, const void* src, size_t n) nogil
    void simd_copy_nt(void* dst, const void* src, size_t n) nogil


# ---------------------------------------------------------------------------
# Intra-block parallel copy (OpenMP).
# copy_mode: 0 = simd, 1 = simd_nt, 2 = memcpy
# ---------------------------------------------------------------------------
cpdef void cython_block_copy_mt(
    unsigned char[:] src,
    unsigned char[:] dst,
    const Py_ssize_t[:] src_indices,
    const Py_ssize_t[:] dst_indices,
    Py_ssize_t block_bytes,
    int n_threads=0,
    int copy_mode=0,
):
    cdef Py_ssize_t n = src_indices.shape[0]
    if n == 0:
        return

    cdef unsigned char* s = &src[0]
    cdef unsigned char* d = &dst[0]
    cdef const Py_ssize_t* si = &src_indices[0]
    cdef const Py_ssize_t* di = &dst_indices[0]

    cdef int T
    if n_threads > 0:
        with nogil:
            omp_set_num_threads(n_threads)
        T = n_threads
    else:
        with nogil:
            T = omp_get_max_threads()
    if T < 1:
        T = 1

    cdef Py_ssize_t subchunk = (block_bytes + T - 1) // T
    cdef Py_ssize_t total = n * T
    cdef Py_ssize_t t, blk, tid, offset, length

    with nogil:
        for t in prange(total, schedule='static'):
            blk = t // T
            tid = t % T
            offset = tid * subchunk
            length = subchunk
            if offset < block_bytes:
                if offset + length > block_bytes:
                    length = block_bytes - offset
                if copy_mode == 0:
                    simd_copy(d + di[blk] * block_bytes + offset,
                              s + si[blk] * block_bytes + offset,
                              length)
                elif copy_mode == 1:
                    simd_copy_nt(d + di[blk] * block_bytes + offset,
                                 s + si[blk] * block_bytes + offset,
                                 length)
                else:
                    memcpy(d + di[blk] * block_bytes + offset,
                           s + si[blk] * block_bytes + offset,
                           length)
'''


def _build_cython_module(verbose: bool = True):
    """Compile the Cython source with OpenMP + AVX-512 and return the module."""
    import numpy as np
    from Cython.Build import cythonize
    from setuptools import Extension
    from setuptools.dist import Distribution

    build_dir = Path(tempfile.mkdtemp(prefix="block_copy_mt_"))
    pyx_path = build_dir / "_block_copy.pyx"
    pyx_path.write_text(CYTHON_SOURCE)

    common = {
        "name": "_block_copy",
        "sources": [str(pyx_path)],
        "include_dirs": [np.get_include()],
        "extra_link_args": ["-fopenmp"],
    }

    # Prefer AVX-512; fall back to AVX2; then plain.
    flags_list = [
        ["-O3", "-fopenmp", "-march=native", "-funroll-loops", "-ftree-vectorize"],
        ["-O3", "-fopenmp", "-mavx512f", "-mavx512bw", "-mavx512vl"],
        ["-O3", "-fopenmp", "-mavx2"],
        ["-O3", "-fopenmp"],
    ]

    last_err = None
    for flags in flags_list:
        try:
            ext = Extension(extra_compile_args=flags, **common)
            dist = Distribution({
                "ext_modules": cythonize(
                    [ext],
                    compiler_directives={
                        "boundscheck": False,
                        "wraparound": False,
                        "cdivision": True,
                        "language_level": 3,
                    },
                    quiet=not verbose,
                )
            })
            cmd = dist.get_command_obj("build_ext")
            cmd.inplace = 1
            cmd.build_lib = str(build_dir)
            cmd.build_temp = str(build_dir / "build")
            cmd.ensure_finalized()
            if not verbose:
                cmd.verbose = 0
            cmd.run()
            if verbose:
                print(f"[build] compiled with flags={flags}")
            sys.path.insert(0, str(build_dir))
            return __import__("_block_copy")
        except Exception as e:
            last_err = e
            if verbose:
                print(f"[build] flags={flags} failed: {e}")
            continue

    raise RuntimeError(f"Failed to compile Cython module: {last_err}")


try:
    _mod = _build_cython_module()
    cython_block_copy_mt = _mod.cython_block_copy_mt
    HAS_CYTHON = True
    print("Cython block copy module loaded successfully (SIMD + OpenMP).")
except Exception as e:
    HAS_CYTHON = False
    cython_block_copy_mt = None
    print(f"Cython not available ({e}) – cython benchmarks disabled.")

# Copy mode mapping
COPY_MODES = {"simd": 0, "simd_nt": 1, "memcpy": 2}


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def format_size(num_bytes, decimal_places=4, use_binary=True, target_unit=None):
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


def format_bandwidth(bytes_per_sec, decimal_places=4):
    return (
        format_size(int(bytes_per_sec), decimal_places=decimal_places, target_unit="GiB")
        + "/s"
    )


def get_system_info():
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
@contextlib.contextmanager
def shared_tensor_pair(total_bytes):
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


def _shm_worker(size, conn, stop_event):
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
def generate_random_indices(num_blocks, n_iters):
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
def measure_bandwidth(copy_func, total_bytes, block_sizes, n_iters, label=""):
    bandwidths = []
    for bs_bytes in block_sizes:
        num_blocks = total_bytes // bs_bytes

        src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
        copy_func(src_indices, dst_indices, bs_bytes)

        src_indices, dst_indices = generate_random_indices(num_blocks, n_iters)
        start = time.perf_counter()
        copy_func(src_indices, dst_indices, bs_bytes)
        elapsed = time.perf_counter() - start

        bw = (bs_bytes * n_iters) / elapsed
        bandwidths.append(bw)
        print(f"[{label}] size: {format_size(bs_bytes)}, "
              f"Bandwidth: {format_bandwidth(bw)}")

    return bandwidths


# ---------------------------------------------------------------------------
# benchmark scenario constructors
# ---------------------------------------------------------------------------
def run_memcpy(total_bytes, block_sizes, n_iters):
    dtype = torch.uint8
    src = torch.randn(total_bytes // 4, dtype=torch.float32, device="cpu").view(dtype)
    dst = torch.randn(total_bytes // 4, dtype=torch.float32, device="cpu").view(dtype)
    print(f"Allocated {format_size(src.nelement() * src.element_size())} for memcpy")

    def copy_func(src_indices, dst_indices, block_bytes):
        s_view = src.view(-1, block_bytes)
        d_view = dst.view(-1, block_bytes)
        for i, j in zip(src_indices, dst_indices):
            d_view[i] = s_view[j]

    with torch.inference_mode():
        return measure_bandwidth(copy_func, total_bytes, block_sizes, n_iters, "memcpy")


def run_shm(total_bytes, block_sizes, n_iters):
    with shared_tensor_pair(total_bytes) as (src, dst):
        src[:] = torch.randn(total_bytes // 4, dtype=torch.float32).view(torch.uint8)
        dst[:] = torch.randn(total_bytes // 4, dtype=torch.float32).view(torch.uint8)
        print(f"Allocated {format_size(total_bytes)} for shm")

        def copy_func(src_indices, dst_indices, block_bytes):
            s_view = src.view(-1, block_bytes)
            d_view = dst.view(-1, block_bytes)
            for i, j in zip(src_indices, dst_indices):
                d_view[i] = s_view[j]

        with torch.inference_mode():
            return measure_bandwidth(copy_func, total_bytes, block_sizes, n_iters, "shm")


def _run_cython_common(src_t, dst_t, total_bytes, block_sizes, n_iters,
                       label, n_threads, copy_mode):
    src_flat = src_t.numpy()
    dst_flat = dst_t.numpy()

    def copy_func(src_indices, dst_indices, block_bytes):
        cython_block_copy_mt(
            src_flat, dst_flat, src_indices, dst_indices,
            block_bytes, n_threads, copy_mode,
        )

    return measure_bandwidth(copy_func, total_bytes, block_sizes, n_iters, label)


def run_cython_shm(total_bytes, block_sizes, n_iters, n_threads, copy_mode):
    if not HAS_CYTHON:
        print("Cython not available, skipping benchmark.")
        return [None] * len(block_sizes)

    mode_name = [k for k, v in COPY_MODES.items() if v == copy_mode][0]
    label = f"cython_shm_{n_threads}_{mode_name}"
    with shared_tensor_pair(total_bytes) as (src, dst):
        src[:] = torch.randn(total_bytes // 4, dtype=torch.float32).view(torch.uint8)
        dst[:] = torch.randn(total_bytes // 4, dtype=torch.float32).view(torch.uint8)
        print(f"Allocated {format_size(total_bytes)} for {label}")
        return _run_cython_common(src, dst, total_bytes, block_sizes,
                                  n_iters, label, n_threads, copy_mode)


def run_cython_mt(total_bytes, block_sizes, n_iters, n_threads, copy_mode):
    if not HAS_CYTHON:
        print("Cython not available, skipping benchmark.")
        return [None] * len(block_sizes)

    dtype = torch.uint8
    src = torch.randn(total_bytes // 4, dtype=torch.float32).view(dtype)
    dst = torch.randn(total_bytes // 4, dtype=torch.float32).view(dtype)

    mode_name = [k for k, v in COPY_MODES.items() if v == copy_mode][0]
    label = f"cython_mt_{n_threads}_{mode_name}"
    print(f"Allocated {format_size(src.nelement() * src.element_size())} for {label}")

    return _run_cython_common(src, dst, total_bytes, block_sizes,
                              n_iters, label, n_threads, copy_mode)


# ---------------------------------------------------------------------------
# output functions
# ---------------------------------------------------------------------------
def print_results_table(block_sizes, results):
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
            row.append("N/A" if bw is None else f"{bw / (1024**3):.4f}")
        print("| " + " | ".join(row) + " |")


def plot_results(block_sizes, results, output_file=None):
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
        ax.plot(x_vals, y_vals, label=name,
                color=colors[idx % len(colors)], marker="o", markersize=4)

    ax.set_xscale("log", base=2)
    ax.set_xticks(block_sizes)
    ax.set_xticklabels([format_size(bs, decimal_places=0) for bs in block_sizes],
                       rotation=45, ha="right")
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
        description="Benchmark CPU block copy bandwidth "
                    "(memcpy / shm / cython_shm / cython_mt)."
    )
    parser.add_argument("--size", type=int, default=2**32,
                        help="Total memory allocated in bytes. Default: 4 GiB.")
    parser.add_argument("--n-iters", type=int, default=100,
                        help="Timed iterations per block size. Default: 100.")
    parser.add_argument("--min-block-exp", type=int, default=8,
                        help="Smallest block size exponent (default: 8 = 256 B).")
    parser.add_argument("--max-block-exp", type=int, default=30,
                        help="Largest block size exponent (default: 30 = 1 GiB).")
    parser.add_argument("--bench", nargs="+",
                        choices=["memcpy", "shm", "cython_shm", "cython_mt"],
                        default=["memcpy", "shm", "cython_shm", "cython_mt"],
                        help="Which benchmarks to run.")
    parser.add_argument("--n-threads", type=int, nargs="+", default=[8],
                        help="Thread counts for cython benchmarks. Default: 8.")
    parser.add_argument("--copy-mode", nargs="+",
                        choices=["simd", "simd_nt", "memcpy"],
                        default=["simd"],
                        help="Copy backend(s) for the Cython kernels. Default: simd.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Do not display or save the bandwidth plot.")
    parser.add_argument("--save-plot", type=str,
                        default="benchmark_cpu_block_copy.png",
                        help="Save plot to this file.")
    args = parser.parse_args()

    total_bytes = args.size
    n_iters = args.n_iters
    block_sizes = [2**n for n in range(args.min_block_exp, args.max_block_exp + 1)]

    max_block = block_sizes[-1]
    if total_bytes % max_block != 0:
        print(f"Warning: total_bytes ({total_bytes}) not a multiple of "
              f"largest block ({max_block}). Adjusting.")
        total_bytes = ((total_bytes // max_block) + 1) * max_block
        print(f"New total_bytes: {total_bytes}")

    simple_benchmarks = {
        "memcpy": run_memcpy,
        "shm": run_shm,
    }
    threaded_benchmarks = {
        "cython_shm": run_cython_shm,
        "cython_mt": run_cython_mt,
    }

    results = {}
    for name in args.bench:
        if name in simple_benchmarks:
            print(f"\n=== Benchmark: {name} ===")
            results[name] = simple_benchmarks[name](total_bytes, block_sizes, n_iters)
        elif name in threaded_benchmarks:
            for nt in args.n_threads:
                for mode in args.copy_mode:
                    cm = COPY_MODES[mode]
                    key = f"{name}_{nt}_{mode}"
                    print(f"\n=== Benchmark: {key} ===")
                    results[key] = threaded_benchmarks[name](
                        total_bytes, block_sizes, n_iters, nt, cm
                    )

    if results:
        print_results_table(block_sizes, results)

    if (not args.no_plot or args.save_plot) and results:
        plot_results(block_sizes, results, output_file=args.save_plot)


if __name__ == "__main__":
    main()