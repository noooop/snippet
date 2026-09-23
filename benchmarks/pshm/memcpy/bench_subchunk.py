#!/usr/bin/env python3
"""
Scan sub-chunk granularity for block copy bandwidth.

Test setup (fixed by design):
  1. 4 GiB total buffer, filled with random data (working set >> L3)
  2. Buffer is divided into 1 MiB blocks
  3. Each block is split into subchunk_bytes-sized tasks
  4. Block indices are randomly shuffled across the buffer

Scanning subchunk_bytes finds the minimum granularity that still
saturates memory bandwidth. Multiple threads may process the same
block concurrently when subchunk_bytes < block_bytes / n_threads.

Backends:
  memcpy    – libc memcpy
  simd      – AVX-512 / AVX2 load + storeu (cached)
  simd_nt   – AVX-512 / AVX2 load + NT store (bypasses L3)

Dependencies:
    pip install cython numpy setuptools
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Cython source
# ---------------------------------------------------------------------------
CYTHON_SOURCE = r'''
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
from libc.string cimport memcpy
from cython.parallel cimport prange

cdef extern from "omp.h" nogil:
    void omp_set_num_threads(int num_threads)
    int  omp_get_max_threads()


cdef extern from *:
    """
    #include <immintrin.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <string.h>

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

    static inline void simd_copy_nt(void* dst, const void* src, size_t n) {
    #if defined(__AVX512F__)
        if ((((uintptr_t)dst) & 63) == 0) {
            size_t i = 0;
            for (; i + 64 <= n; i += 64) {
                __m512i v = _mm512_loadu_si512((const __m512i*)((const char*)src + i));
                _mm512_stream_si512((__m512i*)((char*)dst + i), v);
            }
            _mm_sfence();
            for (; i < n; i++) ((char*)dst)[i] = ((const char*)src)[i];
        } else {
            size_t i = 0;
            for (; i + 64 <= n; i += 64) {
                __m512i v = _mm512_loadu_si512((const __m512i*)((const char*)src + i));
                _mm512_storeu_si512((__m512i*)((char*)dst + i), v);
            }
            for (; i < n; i++) ((char*)dst)[i] = ((const char*)src)[i];
        }
    #elif defined(__AVX2__)
        if ((((uintptr_t)dst) & 31) == 0) {
            size_t i = 0;
            for (; i + 32 <= n; i += 32) {
                __m256i v = _mm256_loadu_si256((const __m256i*)((const char*)src + i));
                _mm256_stream_si256((__m256i*)((char*)dst + i), v);
            }
            _mm_sfence();
            for (; i < n; i++) ((char*)dst)[i] = ((const char*)src)[i];
        } else {
            size_t i = 0;
            for (; i + 32 <= n; i += 32) {
                __m256i v = _mm256_loadu_si256((const __m256i*)((const char*)src + i));
                _mm256_storeu_si256((__m256i*)((char*)dst + i), v);
            }
            for (; i < n; i++) ((char*)dst)[i] = ((const char*)src)[i];
        }
    #else
        memcpy(dst, src, n);
    #endif
    }
    """
    void simd_copy(void* dst, const void* src, size_t n) nogil
    void simd_copy_nt(void* dst, const void* src, size_t n) nogil


cpdef void cython_block_copy_sub(
    unsigned char[:] src,
    unsigned char[:] dst,
    const Py_ssize_t[:] src_indices,
    const Py_ssize_t[:] dst_indices,
    Py_ssize_t block_bytes,
    Py_ssize_t subchunk_bytes,
    int n_threads=0,
    int copy_mode=0,
):
    """Split each block into subchunk_bytes-sized tasks and copy in parallel.

    copy_mode: 0 = simd, 1 = simd_nt, 2 = memcpy
    """
    cdef Py_ssize_t n = src_indices.shape[0]
    if n == 0 or subchunk_bytes <= 0 or block_bytes <= 0:
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

    cdef Py_ssize_t splits = (block_bytes + subchunk_bytes - 1) // subchunk_bytes
    if splits < 1:
        splits = 1
    cdef Py_ssize_t total = n * splits
    cdef Py_ssize_t t, blk, sidx, offset, length, base_src, base_dst

    with nogil:
        for t in prange(total, schedule='static'):
            blk = t // splits
            sidx = t % splits
            offset = sidx * subchunk_bytes
            if offset < block_bytes:
                length = subchunk_bytes
                if offset + length > block_bytes:
                    length = block_bytes - offset
                base_src = si[blk] * block_bytes + offset
                base_dst = di[blk] * block_bytes + offset
                if copy_mode == 0:
                    simd_copy(d + base_dst, s + base_src, length)
                elif copy_mode == 1:
                    simd_copy_nt(d + base_dst, s + base_src, length)
                else:
                    memcpy(d + base_dst, s + base_src, length)


def get_omp_max_threads():
    cdef int n
    with nogil:
        n = omp_get_max_threads()
    return n
'''

COPY_MODES = {"simd": 0, "simd_nt": 1, "memcpy": 2}


# ---------------------------------------------------------------------------
# Build module
# ---------------------------------------------------------------------------
def build_module(verbose=True):
    import numpy as np
    from Cython.Build import cythonize
    from setuptools import Extension
    from setuptools.dist import Distribution

    build_dir = Path(tempfile.mkdtemp(prefix="block_copy_sub_"))
    pyx_path = build_dir / "_block_copy_sub.pyx"
    pyx_path.write_text(CYTHON_SOURCE)

    common = {
        "name": "_block_copy_sub",
        "sources": [str(pyx_path)],
        "include_dirs": [np.get_include()],
        "extra_link_args": ["-fopenmp"],
    }
    flags_list = [
        ["-O3", "-fopenmp", "-mavx512f", "-mavx512bw", "-mavx512vl",
         "-funroll-loops", "-ftree-vectorize"],
        ["-O3", "-fopenmp", "-march=native"],
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
                print(f"[build] flags={flags}")
            sys.path.insert(0, str(build_dir))
            return __import__("_block_copy_sub")
        except Exception as e:
            last_err = e
            if verbose:
                print(f"[build] flags={flags} failed: {e}")
            continue
    raise RuntimeError(f"compile failed: {last_err}")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_size(num_bytes, decimal_places=2):
    if num_bytes == 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB"]
    base = 1024
    size = float(num_bytes)
    e = 0
    while size >= base and e < len(units) - 1:
        size /= base
        e += 1
    return f"{size:.{decimal_places}f} {units[e]}"


# ---------------------------------------------------------------------------
# Random block index generation (numpy, C-speed)
# ---------------------------------------------------------------------------
def generate_random_indices(num_blocks, n_iters):
    """Return (src_indices, dst_indices) of shape (n_iters,) with random values.

    Uses numpy's vectorized randint to avoid the Python-level loop that
    would otherwise dominate for large n_iters.
    """
    src_idx = np.random.randint(0, num_blocks, size=n_iters, dtype=np.intp)
    dst_idx = np.random.randint(0, num_blocks, size=n_iters, dtype=np.intp)
    return src_idx, dst_idx


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def bench(module, src, dst, block_bytes, subchunk_bytes,
          n_threads, copy_mode, n_iters=100, n_runs=5, warn_short=True):
    """Return best bandwidth (GiB/s) across n_runs random-index runs.

    Each run copies n_iters randomly-selected blocks. Choose n_iters large
    enough that a single run takes > 4 ms for stable measurement.
    """
    total_bytes = src.shape[0]
    n_blocks = total_bytes // block_bytes
    if n_blocks == 0:
        raise ValueError("block_bytes > total_bytes")

    fn = module.cython_block_copy_sub

    # Warm-up (prime caches/TLB with a full workload)
    si, di = generate_random_indices(n_blocks, n_iters)
    fn(src, dst, si, di, block_bytes, subchunk_bytes, n_threads, copy_mode)

    best = 0.0
    for _ in range(n_runs):
        si, di = generate_random_indices(n_blocks, n_iters)
        t0 = time.perf_counter()
        fn(src, dst, si, di, block_bytes, subchunk_bytes, n_threads, copy_mode)
        elapsed = time.perf_counter() - t0
        if warn_short and elapsed < 0.004:
            print(f"  [warn] run took only {elapsed*1000:.2f} ms; "
                  f"increase --iters for stable results")
            warn_short = False
        bw = (block_bytes * n_iters) / elapsed / (1024 ** 3)
        best = max(best, bw)
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Scan sub-chunk granularity for block copy bandwidth."
    )
    parser.add_argument("--size", type=int, default=4 * 1024 ** 3,
                        help="Total buffer size in bytes. Default: 4 GiB.")
    parser.add_argument("--block", type=int, default=1 * 1024 ** 2,
                        help="Block size in bytes. Default: 1 MiB.")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of OpenMP threads. Default: 8.")
    parser.add_argument("--iters", type=int, default=100,
                        help="Blocks copied per timed run. Default: 100. "
                             "Increase for lower measurement noise.")
    parser.add_argument("--runs", type=int, default=5,
                        help="Timed runs per config; best is reported. Default: 5.")
    parser.add_argument("--subchunk-exp", type=int, nargs="+",
                        default=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        help="Sub-chunk sizes as 2**exp bytes. "
                             "Default: 10..20 (1 KiB .. 1 MiB).")
    parser.add_argument("--copy-mode", nargs="+",
                        choices=["simd", "simd_nt", "memcpy"],
                        default=["memcpy", "simd", "simd_nt"],
                        help="Copy backends to test. Default: all three.")
    parser.add_argument("--quiet-build", action="store_true",
                        help="Suppress Cython build output.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible runs.")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    block_bytes = args.block
    n_blocks = args.size // block_bytes

    print(f"Buffer:        {format_size(args.size)}")
    print(f"Block:         {format_size(block_bytes)}  ({n_blocks} blocks)")
    print(f"Threads:       {args.threads}")
    print(f"Iters:         {args.iters} blocks per run  "
          f"({format_size(args.iters * block_bytes)} copied per run)")
    print(f"Runs:          {args.runs} (best taken)")
    print(f"Copy modes:    {', '.join(args.copy_mode)}")

    module = build_module(verbose=not args.quiet_build)
    print(f"OMP max threads: {module.get_omp_max_threads()}")

    # Random-content source and zero destination. Random content avoids
    # kernel zero-page / compression optimizations that would otherwise
    # inflate apparent bandwidth.
    print(f"Allocating {format_size(args.size)} random source + destination...")
    src = np.random.randint(0, 256, size=args.size, dtype=np.uint8)
    dst = np.zeros(args.size, dtype=np.uint8)
    print(f"  src.data % 64 = {src.ctypes.data % 64}")
    print(f"  dst.data % 64 = {dst.ctypes.data % 64}")
    print()

    header = ["Sub-chunk"] + [f"{m} (GiB/s)" for m in args.copy_mode] + ["tasks/block"]
    print(" | ".join(f"{h:>16}" for h in header))
    print("-" * (18 * len(header)))

    for sc in [2**e for e in args.subchunk_exp]:
        splits = (block_bytes + sc - 1) // sc
        row = [f"{format_size(sc):>16}"]
        for mode in args.copy_mode:
            cm = COPY_MODES[mode]
            bw = bench(module, src, dst, block_bytes, sc,
                       args.threads, cm,
                       n_iters=args.iters, n_runs=args.runs)
            row.append(f"{bw:>16.2f}")
        row.append(f"{splits:>16}")
        print(" | ".join(row))

    print()
    print("Notes:")
    print(f"  - Random block indices; {args.iters} blocks copied per run.")
    print("  - Each run should take > 4 ms for stable measurement.")
    print("  - simd_nt bypasses L3; simd and memcpy go through L3.")


if __name__ == "__main__":
    main()