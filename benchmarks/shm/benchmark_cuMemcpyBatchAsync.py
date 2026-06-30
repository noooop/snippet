import random
import time

import torch
from util import format_size, format_size_gb, size, dtype, n_iters
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor
from vllm import _custom_ops as ops

host_raw = torch.randn(size // 4, dtype=torch.float32, device="cpu").view(dtype)
device_raw = torch.randn(size // 4, dtype=torch.float32, device="cuda").view(dtype)
pin_tensor(host_raw)

print(format_size(host_raw.nelement() * host_raw.element_size()))


print("random H2D")
with torch.inference_mode():
    for n in range(8, 32):
        block_size = 2**n

        host = host_raw.view(-1, block_size)
        device = device_raw.view(-1, block_size)
        bs, _ = host.size()

        def test(n_iters):

            tasks = [
                (random.randint(0, bs - 1), random.randint(0, bs - 1))
                for _ in range(n_iters)
            ]

            src_addrs = torch.tensor(
                [host[i].data_ptr() for i, j in tasks], dtype=torch.int64
            )
            dst_addrs = torch.tensor(
                [device[j].data_ptr() for i, j in tasks], dtype=torch.int64
            )
            sizes = torch.tensor([block_size] * n_iters, dtype=torch.int64)

            torch.accelerator.synchronize()

            start = time.perf_counter()

            with torch.cuda.stream(torch.cuda.Stream()) as stream:
                ops.swap_blocks_batch(
                    src_addrs,
                    dst_addrs,
                    sizes,
                )

            torch.accelerator.synchronize()

            end = time.perf_counter()
            elapsed_time_s = end - start
            return elapsed_time_s

        test(n_iters=3)
        elapsed_time_s = test(n_iters)

        bw = block_size * n_iters / elapsed_time_s
        print(f"size: {format_size(block_size)}, Bandwidth: {format_size_gb(bw)}/s")


print("random D2H")
with torch.inference_mode():
    for n in range(8, 32):
        block_size = 2**n

        host = host_raw.view(-1, block_size)
        device = device_raw.view(-1, block_size)
        bs, _ = host.size()

        def test(n_iters):
            tasks = [
                (random.randint(0, bs - 1), random.randint(0, bs - 1))
                for _ in range(n_iters)
            ]

            src_addrs = torch.tensor(
                [device[i].data_ptr() for i, j in tasks], dtype=torch.int64
            )
            dst_addrs = torch.tensor(
                [host[j].data_ptr() for i, j in tasks], dtype=torch.int64
            )
            sizes = torch.tensor([block_size] * n_iters, dtype=torch.int64)

            torch.accelerator.synchronize()

            start = time.perf_counter()

            with torch.cuda.stream(torch.cuda.Stream()) as stream:
                ops.swap_blocks_batch(
                    src_addrs,
                    dst_addrs,
                    sizes,
                )

            torch.accelerator.synchronize()

            end = time.perf_counter()
            elapsed_time_s = end - start
            return elapsed_time_s

        test(n_iters=3)
        elapsed_time_s = test(n_iters)

        bw = block_size * n_iters / elapsed_time_s
        print(f"size: {format_size(block_size)}, Bandwidth: {format_size_gb(bw)}/s")
