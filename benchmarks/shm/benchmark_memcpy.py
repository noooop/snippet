import random
import time

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
dtype = torch.uint8

src_raw = torch.randn(size // 4, dtype=torch.float32, device="cpu").view(dtype)
dst_raw = torch.randn(size // 4, dtype=torch.float32, device="cpu").view(dtype)

print(format_size(src_raw.nelement() * src_raw.element_size()))

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
