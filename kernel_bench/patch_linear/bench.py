# SPDX-License-Identifier: Apache-2.0
"""Three-way benchmark for SigLIP patch embedding:
    A) F.conv2d          (Conv2dLayer._forward_conv, current CUDA path)
    B) unfold + F.linear (Conv2dLayer._forward_mulmat, enable_linear path)
    C) pre-patchify + nn.Linear (external patchify, no unfold in forward)

All three produce [B, N, D]. Weights are transferred once, inputs live on GPU.
"""

import gc

import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.conv import Conv2dLayer


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    C, H, W, P, D = 3, 896, 896, 14, 1152
    Gh, Gw = H // P, W // P
    N = Gh * Gw
    device, dtype = "cuda", torch.bfloat16

    with set_current_vllm_config(VllmConfig()):
        conv = Conv2dLayer(
            in_channels=C, out_channels=D,
            kernel_size=P, stride=P, padding="valid",
        ).to(device, dtype)

    # enable_linear must be True for _forward_mulmat to be valid.
    assert conv.enable_linear, "kernel != stride, padding, or groups != 1"

    with torch.no_grad():
        torch.nn.init.kaiming_uniform_(conv.weight, a=5 ** 0.5)
        if conv.bias is not None:
            torch.nn.init.zeros_(conv.bias)
    assert not torch.isnan(conv.weight).any()

    lin = nn.Linear(C * P * P, D).to(device, dtype)
    with torch.no_grad():
        lin.weight.copy_(conv.weight.reshape(D, -1))
        if conv.bias is not None:
            lin.bias.copy_(conv.bias)
        else:
            lin.bias.zero_()

    batches = [1, 2, 4, 8, 16, 32, 64]
    header = (
        f"{'B':>4}  {'A:conv':>9}  {'B:mulmat':>9}  {'C:linear':>9}  "
        f"{'A/B':>6}  {'A/C':>6}  {'diff':>9}"
    )
    print(header)
    print("-" * len(header))

    for B in batches:
        if B <= 4:
            warmup, iters = 20, 200
        elif B <= 16:
            warmup, iters = 20, 100
        else:
            warmup, iters = 10, 50

        img = torch.randn(B, C, H, W, device=device, dtype=dtype)
        # Pre-patchify for path C. Not part of any timed region.
        patches = (
            img.unfold(2, P, P).unfold(3, P, P)
               .permute(0, 2, 3, 1, 4, 5)
               .reshape(B, Gh, Gw, C * P * P)
               .contiguous()
        )

        def path_conv():
            return conv._forward_conv(img).flatten(2).transpose(1, 2)

        def path_mulmat():
            return conv._forward_mulmat(img).flatten(2).transpose(1, 2)

        def path_linear():
            return lin(patches).reshape(B, N, D)

        # Numerics check.
        with torch.no_grad():
            y_a = path_conv()
            y_b = path_mulmat()
            y_c = path_linear()
        diff = max(
            (y_a - y_b).abs().max().item(),
            (y_a - y_c).abs().max().item(),
        )

        t_a = bench(path_conv, warmup, iters)
        t_b = bench(path_mulmat, warmup, iters)
        t_c = bench(path_linear, warmup, iters)

        print(
            f"{B:>4}  {t_a:>9.3f}  {t_b:>9.3f}  {t_c:>9.3f}  "
            f"{t_a / t_b:>5.2f}x  {t_a / t_c:>5.2f}x  {diff:>9.3e}"
        )

        del img, patches, y_a, y_b, y_c
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()