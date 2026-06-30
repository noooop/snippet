from multiprocessing import shared_memory
import multiprocessing as mp
from unittest.mock import patch

import numpy as np
import torch
from vllm.triton_utils import tl, triton


def format_size(size, decimal_places=4, use_binary=True):
    if size == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    base = 1024 if use_binary else 1000
    exponent = 0
    while size >= base and exponent < len(units) - 1:
        size /= base
        exponent += 1
    return f"{size:.{decimal_places}f} {units[exponent]}"


def format_size_gb(size, decimal_places=4, use_binary=True):
    if size == 0:
        return "0 GB"
    units = ["B", "KB", "MB", "GB"]
    base = 1024 if use_binary else 1000
    exponent = 0
    while exponent < len(units) - 1:
        size /= base
        exponent += 1
    return f"{size:.{decimal_places}f} {units[exponent]}"


def worker(size, conn, stop_event):
    shm = shared_memory.SharedMemory(size=size, create=True)
    try:
        conn.send(shm.name)
        conn.close()
        stop_event.wait()
    finally:
        shm.close()
        shm.unlink()


def get_shm(size):
    parent_conn, child_conn = mp.Pipe()
    stop_event = mp.Event()
    process = mp.Process(target=worker, args=(size, child_conn, stop_event))
    process.start()
    shm_name = parent_conn.recv()
    parent_conn.close()
    with patch(
        "multiprocessing.resource_tracker.register",
        lambda *args, **kwargs: None,
    ):
        shm = shared_memory.SharedMemory(name=shm_name)

    return shm, process, stop_event


size = 2**34  # 16G
dtype = torch.uint8
np_dtype = np.uint8
n_iters = 100
NUM_SMS = 12


@triton.jit
def _swap_blocks_kernel(
    src_addrs,
    dst_addrs,
    sizes,
    n_jobs,  # type: ignore[name-defined]
    BYTES_PER_CHUNK: tl.constexpr,  # type: ignore[name-defined]
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    WORDS_PER_CHUNK: tl.constexpr = BYTES_PER_CHUNK // 8
    offsets = tl.arange(0, WORDS_PER_CHUNK)
    job = pid
    while job < n_jobs:
        src = tl.load(src_addrs + job).to(tl.pointer_type(tl.int64))
        dst = tl.load(dst_addrs + job).to(tl.pointer_type(tl.int64))
        words = tl.load(sizes + job) // 8
        for start in range(0, words, WORDS_PER_CHUNK):
            idx = start + offsets
            mask = idx < words
            data = tl.load(src + idx, mask=mask, other=0)
            tl.store(dst + idx, data, mask=mask)
        job += num_progs


def swap_blocks_batch(
    src_addrs: torch.Tensor,
    dst_addrs: torch.Tensor,
    sizes: torch.Tensor,
    *,
    bytes_per_chunk: int,
) -> None:
    n = src_addrs.numel()

    _swap_blocks_kernel[(min(NUM_SMS, n),)](
        src_addrs,
        dst_addrs,
        sizes,
        n,
        BYTES_PER_CHUNK=bytes_per_chunk,
    )
