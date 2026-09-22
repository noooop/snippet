#!/usr/bin/env python3
"""
Single-file Cython + OpenMP block copy benchmark.

Intra-block parallelism: each block is split into n_threads sub-chunks,
and multiple threads copy different parts of the same block concurrently.

Dependencies:
    pip install cython numpy setuptools

Recommended run (NUMA binding on dual-socket machines):
    OMP_PROC_BIND=close OMP_PLACES=cores \
    numactl --cpunodebind=0 --membind=0 \
    python block_copy_mt.py

Result Xeon Gold 6554S:
 threads     bandwidth (GiB/s)
------------------------------------------------------------
  1 (st)                  7.91
       2                 15.55
       4                 27.43
       8                 42.62
      12                 52.78
      16                 64.32
      20                 72.90
      24                 78.79
      28                 84.09
      32                 41.16
      36                 40.93
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Embedded Cython source
# ---------------------------------------------------------------------------
PYX_SOURCE = r'''
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
from libc.string cimport memcpy
from cython.parallel cimport prange

cdef extern from "omp.h" nogil:
    void omp_set_num_threads(int num_threads)
    int  omp_get_max_threads()
    int  omp_get_num_threads()


# ---------------------------------------------------------------------------
# Intra-block parallel copy.
# Each block is divided into T sub-chunks; multiple threads copy different
# parts of the same block concurrently. Total tasks = n_blocks * T.
# ---------------------------------------------------------------------------
cpdef void cython_block_copy_split(
    unsigned char[:] src,
    unsigned char[:] dst,
    const Py_ssize_t[:] src_indices,
    const Py_ssize_t[:] dst_indices,
    Py_ssize_t block_bytes,
    int n_threads=0,
):
    cdef Py_ssize_t n = src_indices.shape[0]
    if n == 0:
        return

    cdef unsigned char* s = &src[0]
    cdef unsigned char* d = &dst[0]
    cdef const Py_ssize_t* si = &src_indices[0]
    cdef const Py_ssize_t* di = &dst_indices[0]

    # Resolve thread count: use n_threads if > 0, otherwise OpenMP default.
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

    # Sub-chunk size (rounded up to cover the whole block).
    cdef Py_ssize_t subchunk = (block_bytes + T - 1) // T
    cdef Py_ssize_t total = n * T
    cdef Py_ssize_t t, blk, tid, offset, length

    with nogil:
        for t in prange(total, schedule='static'):
            blk = t // T          # block index
            tid = t % T           # sub-chunk index within the block
            offset = tid * subchunk
            if offset >= block_bytes:
                continue
            length = subchunk
            if offset + length > block_bytes:
                length = block_bytes - offset
            memcpy(d + di[blk] * block_bytes + offset,
                   s + si[blk] * block_bytes + offset,
                   length)


# ---------------------------------------------------------------------------
# Single-threaded baseline (whole block per memcpy).
# ---------------------------------------------------------------------------
cpdef void cython_block_copy_st(
    unsigned char[:] src,
    unsigned char[:] dst,
    const Py_ssize_t[:] src_indices,
    const Py_ssize_t[:] dst_indices,
    Py_ssize_t block_bytes,
):
    cdef Py_ssize_t n = src_indices.shape[0]
    if n == 0:
        return

    cdef Py_ssize_t i
    cdef unsigned char* s = &src[0]
    cdef unsigned char* d = &dst[0]
    cdef const Py_ssize_t* si = &src_indices[0]
    cdef const Py_ssize_t* di = &dst_indices[0]

    with nogil:
        for i in range(n):
            memcpy(d + di[i] * block_bytes,
                   s + si[i] * block_bytes,
                   block_bytes)


def get_omp_max_threads():
    """Return OpenMP's max thread count."""
    cdef int n
    with nogil:
        n = omp_get_max_threads()
    return n
'''


# ---------------------------------------------------------------------------
# Runtime compilation of the Cython module
# ---------------------------------------------------------------------------
def build_module(verbose: bool = True):
    """Compile the embedded Cython source and return the imported module."""
    import numpy as np
    from Cython.Build import cythonize
    from setuptools import Extension
    from setuptools.dist import Distribution

    build_dir = Path(tempfile.mkdtemp(prefix="block_copy_mt_"))
    pyx_path = build_dir / "block_copy_mt.pyx"
    pyx_path.write_text(PYX_SOURCE)

    common = {
        "name": "block_copy_mt",
        "sources": [str(pyx_path)],
        "include_dirs": [np.get_include()],
        "extra_link_args": ["-fopenmp"],
    }

    # Try increasingly conservative flag sets; -march=native may fail on VMs.
    flags_list = [
        ["-O3", "-fopenmp", "-march=native", "-funroll-loops", "-ftree-vectorize"],
        ["-O3", "-fopenmp", "-mavx2"],
        ["-O3", "-fopenmp"],
    ]

    last_err = None
    success = False
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
            success = True
            if verbose:
                print(f"[build] compiled with flags={flags}")
            break
        except Exception as e:
            last_err = e
            if verbose:
                print(f"[build] flags={flags} failed: {e}")
            continue

    if not success:
        raise RuntimeError(f"Failed to compile Cython module: {last_err}")

    sys.path.insert(0, str(build_dir))
    import block_copy_mt
    if verbose:
        print(f"[build] module built at {build_dir}")
    return block_copy_mt


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def bench(module, total_bytes, block_bytes, n_threads, n_iters=5):
    """Run the intra-block parallel benchmark; return best bandwidth (GiB/s)."""
    import numpy as np

    n_blocks = total_bytes // block_bytes
    if n_blocks == 0:
        raise ValueError("block_bytes > total_bytes")

    src = np.random.randint(0, 256, size=total_bytes, dtype=np.uint8)
    dst = np.zeros(total_bytes, dtype=np.uint8)

    src_idx = np.arange(n_blocks, dtype=np.intp)
    dst_idx = np.arange(n_blocks, dtype=np.intp)

    copy_fn = module.cython_block_copy_split

    # Warm-up
    copy_fn(src, dst, src_idx, dst_idx, block_bytes, n_threads)

    best = 0.0
    for _ in range(n_iters):
        t0 = time.perf_counter()
        copy_fn(src, dst, src_idx, dst_idx, block_bytes, n_threads)
        elapsed = time.perf_counter() - t0
        bw = total_bytes / elapsed / (1024 ** 3)
        best = max(best, bw)
    return best


def bench_st(module, total_bytes, block_bytes, n_iters=5):
    """Single-threaded baseline for reference."""
    import numpy as np

    n_blocks = total_bytes // block_bytes
    if n_blocks == 0:
        raise ValueError("block_bytes > total_bytes")

    src = np.random.randint(0, 256, size=total_bytes, dtype=np.uint8)
    dst = np.zeros(total_bytes, dtype=np.uint8)

    src_idx = np.arange(n_blocks, dtype=np.intp)
    dst_idx = np.arange(n_blocks, dtype=np.intp)

    copy_fn = module.cython_block_copy_st
    copy_fn(src, dst, src_idx, dst_idx, block_bytes)

    best = 0.0
    for _ in range(n_iters):
        t0 = time.perf_counter()
        copy_fn(src, dst, src_idx, dst_idx, block_bytes)
        elapsed = time.perf_counter() - t0
        bw = total_bytes / elapsed / (1024 ** 3)
        best = max(best, bw)
    return best


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cython + OpenMP intra-block (split) copy benchmark."
    )
    parser.add_argument("--size", type=int, default=4 * 1024 ** 3,
                        help="Total bytes per buffer (default 4 GiB).")
    parser.add_argument("--block", type=int, default=1 * 1024 ** 2,
                        help="Block size in bytes (default 1 MiB).")
    parser.add_argument("--iters", type=int, default=5,
                        help="Timed iterations per thread count (default 5).")
    parser.add_argument("--threads", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 128],
                        help="Thread counts to test.")
    parser.add_argument("--quiet-build", action="store_true",
                        help="Suppress Cython build output.")
    args = parser.parse_args()

    n_blocks = args.size // args.block
    print(f"PID: {os.getpid()}")
    print(f"Total: {args.size / 1024**3:.2f} GiB   "
          f"Block: {args.block / 1024**2:.2f} MiB   "
          f"Blocks: {n_blocks}   "
          f"Iters: {args.iters}")
    print("-" * 60)

    module = build_module(verbose=not args.quiet_build)

    print(f"\n{'threads':>8}  {'bandwidth (GiB/s)':>20}")
    print("-" * 60)

    # Single-threaded baseline
    st_bw = bench_st(module, args.size, args.block, n_iters=args.iters)
    print(f"{'1 (st)':>8}  {st_bw:>20.2f}")

    # Intra-block parallel
    for nt in args.threads:
        if nt == 1:
            continue
        bw = bench(module, args.size, args.block, nt, n_iters=args.iters)
        print(f"{nt:>8}  {bw:>20.2f}")


if __name__ == "__main__":
    main()