import multiprocessing as mp
import random
import time
from multiprocessing import shared_memory
from unittest.mock import patch
import numpy as np
import torch

from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor


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


size = 2**34  # 16G
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

host_raw = torch.from_numpy(np.ndarray(size, dtype=dtype, buffer=shm_src.buf))
device_raw = torch.randn(size // 4, dtype=torch.float32, device="cuda").view(
    torch.uint8
)

host_raw[:] = torch.randn(size // 4, dtype=torch.float32, device="cpu").view(
    torch.uint8
)

pin_tensor(host_raw)


n_iters = 100


print("random H2D")
with torch.inference_mode():
    for n in range(8, 32):
        block_size = 2**n

        host = host_raw.view(-1, block_size)
        device = device_raw.view(-1, block_size)
        bs, _ = host.size()

        tasks = [
            (random.randint(0, bs - 1), random.randint(0, bs - 1))
            for _ in range(n_iters)
        ]

        torch.accelerator.synchronize()

        start = time.perf_counter()

        for i, j in tasks:
            device[i] = host[j]

        torch.accelerator.synchronize()

        end = time.perf_counter()
        elapsed_time_s = end - start

        bw = block_size * n_iters / elapsed_time_s
        print(f"size: {format_size(block_size)}, Bandwidth: {format_size_gb(bw)}/s")


print("random D2H")
with torch.inference_mode():
    for n in range(8, 32):
        block_size = 2**n

        host = host_raw.view(-1, block_size)
        device = device_raw.view(-1, block_size)
        bs, _ = host.size()

        tasks = [
            (random.randint(0, bs - 1), random.randint(0, bs - 1))
            for _ in range(n_iters)
        ]

        torch.accelerator.synchronize()

        start = time.perf_counter()

        for i, j in tasks:
            host[i] = device[j]

        torch.accelerator.synchronize()

        end = time.perf_counter()
        elapsed_time_s = end - start

        bw = block_size * n_iters / elapsed_time_s
        print(f"size: {format_size(block_size)}, Bandwidth: {format_size_gb(bw)}/s")


del host, device
time.sleep(1)

stop_event_src.set()
process_src.join()
shm_src.close()
