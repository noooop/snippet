# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Eval-mode ``nn.BatchNorm1d`` baseline for the fused input-norm kernel.

In eval mode, ``nn.BatchNorm1d`` applies the affine transform derived from its
running statistics:

    y = (x - running_mean) / sqrt(running_var + eps) * weight + bias

which is exactly the computation the fused kernel performs. Two roles:

* **performance baseline** — BatchNorm1d dispatches to cuDNN/cuBLAS kernels
  and materializes an fp32 intermediate, so it is the honest "generic library
  kernel" bar the fused path must beat;
* **numerical cross-check** — its output can be compared against the fused
  kernel via the inherited ``reference()``, which recomputes the affine
  transform with plain elementwise ops (independent of the BN module).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .naive import Naive
from .utils import benchmark


class BatchNorm1d(Naive):
    """Eval-mode ``nn.BatchNorm1d`` over a ``(N, C, L)`` input.

    Inherits tensor construction and size accounting from :class:`Naive`; only
    replaces the forward op and provides a plain-op reference for the affine
    transform derived from the running statistics.
    """

    # BN's forward is faster than the eager broadcast path but slower than a
    # fully fused kernel; these defaults balance stability against sweep cost.
    warmup_iters = 30
    bench_iters = 100

    # Capture the BN launch into a graph to compare pure device time.
    use_cuda_graph = True

    # BN *is* one of the reference implementations for the affine transform;
    # verifying it against itself would only confirm the trivial equivalence.
    verify = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.bn = nn.BatchNorm1d(
            self.channel, eps=1e-5, momentum=0.1, affine=True
        )
        self.bn = self.bn.to(device="cuda:0", dtype=self.compute_dtype)
        self.bn.eval()

        # Randomize running statistics and affine params so the transform is
        # non-trivial. With the defaults (mean=0, var=1, weight=1, bias=0) the
        # mapping would be close to identity and could mask numerical bugs.
        with torch.no_grad():
            self.bn.running_mean.normal_(0.0, 0.1)
            self.bn.running_var.uniform_(0.5, 1.5)
            self.bn.weight.normal_(1.0, 0.05)
            self.bn.bias.normal_(0.0, 0.05)

    # ----- Test interface --------------------------------------------------
    @torch.inference_mode()
    def function_under_test(self) -> None:
        # ``inputs.to(compute_dtype)`` mirrors what the fused kernel does when
        # the input is uint8; BatchNorm1d then materializes an fp32 output
        # internally, which ``copy_`` casts down to ``outputs_dtype``.
        out = self.bn(self.inputs.to(self.compute_dtype))
        self._outputs.copy_(out)

    def reference(self) -> torch.Tensor:
        """Recompute the eval-mode affine transform from running statistics.

        Uses only plain elementwise ops so the result is independent of
        whatever kernel ``nn.BatchNorm1d`` dispatches to. The affine form is

            scale = weight / sqrt(running_var + eps)
            shift = bias - running_mean * scale
            y = x * scale + shift
        """
        with torch.no_grad():
            inv_std = torch.rsqrt(self.bn.running_var + self.bn.eps)
            scale = self.bn.weight * inv_std
            shift = self.bn.bias - self.bn.running_mean * scale

        x = self.inputs.to(self.compute_dtype)
        out = x * scale.view(1, self.channel, 1) + shift.view(1, self.channel, 1)
        return out.to(self.outputs_dtype)


if __name__ == "__main__":
    print("Test batch_norm1d!")

    for compute_dtype in [torch.float32, torch.bfloat16]:
        for n in range(8, 19):
            try:
                benchmark(
                    BatchNorm1d,
                    compute_dtype,
                    inputs_dtype=torch.uint8,
                    outputs_dtype=torch.bfloat16,
                    patches=2**n,
                    channel=3,
                    embed_size=1024,
                )
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OOM at compute={compute_dtype}, patches=2**{n}; skipping."
                )
                torch.cuda.empty_cache()
                continue
        torch.cuda.empty_cache()