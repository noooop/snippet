import random
import time
import numpy as np
import torch

from util import (
    format_size,
    format_size_gb,
    size,
    n_iters,
    get_shm,
    np_dtype,
)

shm_src, process_src, stop_event_src = get_shm(size)
shm_dst, process_dst, stop_event_dst = get_shm(size)

src_raw = torch.from_numpy(np.ndarray(size, dtype=np_dtype, buffer=shm_src.buf))
dst_raw = torch.from_numpy(np.ndarray(size, dtype=np_dtype, buffer=shm_dst.buf))

src_raw[:] = torch.randn(size // 4, dtype=torch.float32, device="cpu").view(torch.uint8)
dst_raw[:] = torch.randn(size // 4, dtype=torch.float32, device="cpu").view(torch.uint8)


print("random read+write")
with torch.inference_mode():
    for n in range(8, 32):
        block_size = 2**n

        src = src_raw.view(-1, block_size)
        dst = dst_raw.view(-1, block_size)
        bs, _ = src.size()

        def test(n_iters):

            tasks = [
                (random.randint(0, bs - 1), random.randint(0, bs - 1))
                for _ in range(n_iters)
            ]

            start = time.perf_counter()

            for i, j in tasks:
                dst[i] = src[j]

            end = time.perf_counter()
            elapsed_time_s = end - start
            return elapsed_time_s

        test(n_iters=3)
        elapsed_time_s = test(n_iters)
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
