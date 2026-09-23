#!/usr/bin/env python3
"""
Pure Python + numpy block copy benchmark (no Cython).

Same test setup as bench_subchunk.py:
  1. 4 GiB total buffer, filled with random data
  2. Buffer divided into 1 MiB blocks
  3. Each block is split into subchunk_bytes-sized tasks
  4. Random block indices across the buffer

Compares:
  np_block    – single-threaded, one numpy slice per whole block (upper bound)
  np_single   – single-threaded, one numpy slice per sub-chunk
  np_thread_N – ThreadPoolExecutor created inside the copy function,
                one task per sub-chunk

The point is to see how far pure Python + numpy can go relative to the
Cython + OpenMP version, and where the Python/GIL overhead starts to bite.

Dependencies: numpy
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


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


def generate_random_indices(num_blocks, n_iters):
    src_idx = np.random.randint(0, num_blocks, size=n_iters, dtype=np.intp)
    dst_idx = np.random.randint(0, num_blocks, size=n_iters, dtype=np.intp)
    return src_idx, dst_idx


# ---------------------------------------------------------------------------
# Benchmark kernels
# ---------------------------------------------------------------------------
def np_block(src, dst, block_bytes, src_idx, dst_idx):
    """One numpy slice copy per whole block. Upper bound for numpy."""
    for k in range(len(src_idx)):
        sbase = int(src_idx[k]) * block_bytes
        dbase = int(dst_idx[k]) * block_bytes
        dst[dbase:dbase + block_bytes] = src[sbase:sbase + block_bytes]


def np_single(src, dst, block_bytes, subchunk_bytes, src_idx, dst_idx):
    """One numpy slice copy per sub-chunk, single thread."""
    splits = (block_bytes + subchunk_bytes - 1) // subchunk_bytes
    for k in range(len(src_idx)):
        sbase = int(src_idx[k]) * block_bytes
        dbase = int(dst_idx[k]) * block_bytes
        off = 0
        for _ in range(splits):
            n = subchunk_bytes
            if off + n > block_bytes:
                n = block_bytes - off
            dst[dbase + off:dbase + off + n] = src[sbase + off:sbase + off + n]
            off += subchunk_bytes


def np_thread(src, dst, block_bytes, subchunk_bytes, src_idx, dst_idx,
              n_threads):
    """Multi-threaded with numpy slice copies.

    Creates a fresh ThreadPoolExecutor inside this call. Each worker pulls
    a contiguous range of the global task space to minimize overhead.
    numpy releases the GIL for large slice copies; for tiny sub-chunks GIL
    contention can make this slower than np_single.
    """
    splits = (block_bytes + subchunk_bytes - 1) // subchunk_bytes
    n_blocks = len(src_idx)
    total_tasks = n_blocks * splits

    def worker(lo_hi):
        lo, hi = lo_hi
        for t in range(lo, hi):
            k = t // splits
            j = t % splits
            off = j * subchunk_bytes
            if off >= block_bytes:
                continue
            n = subchunk_bytes
            if off + n > block_bytes:
                n = block_bytes - off
            sbase = int(src_idx[k]) * block_bytes + off
            dbase = int(dst_idx[k]) * block_bytes + off
            dst[dbase:dbase + n] = src[sbase:sbase + n]

    chunk = (total_tasks + n_threads - 1) // n_threads
    ranges = [
        (i * chunk, min((i + 1) * chunk, total_tasks))
        for i in range(n_threads)
    ]
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(worker, ranges))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def bench(fn, bytes_per_run, n_runs, *args, **kwargs):
    fn(*args, **kwargs)  # warm-up
    best = 0.0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        bw = bytes_per_run / elapsed / (1024 ** 3)
        best = max(best, bw)
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Pure Python + numpy block copy benchmark."
    )
    parser.add_argument("--size", type=int, default=4 * 1024 ** 3,
                        help="Total buffer size in bytes. Default: 4 GiB.")
    parser.add_argument("--block", type=int, default=1 * 1024 ** 2,
                        help="Block size in bytes. Default: 1 MiB.")
    parser.add_argument("--threads", type=int, default=8,
                        help="Threads for np_thread. Default: 8.")
    parser.add_argument("--iters", type=int, default=100,
                        help="Blocks copied per timed run. Default: 100.")
    parser.add_argument("--runs", type=int, default=5,
                        help="Timed runs per config; best is reported. Default: 5.")
    parser.add_argument("--subchunk-exp", type=int, nargs="+",
                        default=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        help="Sub-chunk sizes as 2**exp bytes. Default: 10..20 "
                             "(1 KiB .. 1 MiB).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible runs.")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    block_bytes = args.block
    n_blocks = args.size // block_bytes
    bytes_per_run = args.iters * block_bytes

    print(f"Buffer:      {format_size(args.size)}")
    print(f"Block:       {format_size(block_bytes)}  ({n_blocks} blocks)")
    print(f"Iters:       {args.iters} blocks per run  "
          f"({format_size(bytes_per_run)} copied per run)")
    print(f"Runs:        {args.runs} (best taken)")
    print(f"Threads:     {args.threads} (for np_thread)")
    print()

    print(f"Allocating {format_size(args.size)} random source + destination...")
    src = np.random.randint(0, 256, size=args.size, dtype=np.uint8)
    dst = np.zeros(args.size, dtype=np.uint8)
    print()

    header = [
        "Sub-chunk",
        "np_block (GiB/s)",
        "np_single (GiB/s)",
        f"np_thread_{args.threads} (GiB/s)",
        "tasks/block",
    ]
    print(" | ".join(f"{h:>22}" for h in header))
    print("-" * (24 * len(header)))

    for exp in args.subchunk_exp:
        sc = 2 ** exp
        splits = (block_bytes + sc - 1) // sc

        # Same random index set for all three kernels in this row.
        src_idx, dst_idx = generate_random_indices(n_blocks, args.iters)

        bw_block = bench(
            np_block, bytes_per_run, args.runs,
            src, dst, block_bytes, src_idx, dst_idx,
        )
        bw_single = bench(
            np_single, bytes_per_run, args.runs,
            src, dst, block_bytes, sc, src_idx, dst_idx,
        )
        bw_thread = bench(
            np_thread, bytes_per_run, args.runs,
            src, dst, block_bytes, sc, src_idx, dst_idx,
            args.threads,
        )

        row = [
            f"{format_size(sc):>22}",
            f"{bw_block:>22.2f}",
            f"{bw_single:>22.2f}",
            f"{bw_thread:>22.2f}",
            f"{splits:>22}",
        ]
        print(" | ".join(row))

    print()
    print("Notes:")
    print("  - np_block: one numpy slice per whole 1 MiB block (upper bound).")
    print("  - np_single: one numpy slice per sub-chunk, single thread.")
    print("  - np_thread: ThreadPoolExecutor created inside the copy function,")
    print("    one task per sub-chunk.")
    print("  - numpy releases the GIL for large slice copies (>~1 KB);")
    print("    for tiny sub-chunks, GIL contention can make np_thread slower")
    print("    than np_single.")


if __name__ == "__main__":
    main()