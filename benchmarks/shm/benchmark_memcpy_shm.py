import multiprocessing as mp
import random
import time
from multiprocessing import shared_memory
from unittest.mock import patch
import numpy as np
import torch


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


size = 2**32  # 4G
dtype = np.uint8


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


shm_src, process_src, stop_event_src = get_shm(size)
shm_dst, process_dst, stop_event_dst = get_shm(size)

src_raw = torch.from_numpy(np.ndarray(size, dtype=dtype, buffer=shm_src.buf))
dst_raw = torch.from_numpy(np.ndarray(size, dtype=dtype, buffer=shm_dst.buf))

src_raw[:] = torch.randn(size // 4, dtype=torch.float32, device="cpu").view(torch.uint8)
dst_raw[:] = torch.randn(size // 4, dtype=torch.float32, device="cpu").view(torch.uint8)


n_iters = 100

print("random read+write")
with torch.inference_mode():
    for n in range(8, 32):
        block_size = 2**n

        src = src_raw.view(-1, block_size)
        dst = dst_raw.view(-1, block_size)
        bs, _ = src.size()

        tasks = [
            (random.randint(0, bs - 1), random.randint(0, bs - 1))
            for _ in range(n_iters)
        ]

        start = time.perf_counter()

        for i, j in tasks:
            dst[i] = src[j]

        end = time.perf_counter()
        elapsed_time_s = end - start

        bw = block_size * n_iters / elapsed_time_s
        print(f"size: {format_size(block_size)}, Bandwidth: {format_size_gb(bw)}/s")

del src, dst
time.sleep(1)

stop_event_src.set()
process_src.join()
shm_src.close()


stop_event_dst.set()
process_dst.join()
shm_dst.close()
