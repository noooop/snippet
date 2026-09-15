# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure memory-bandwidth baseline: uint8 -> bf16 copy via Triton.

No arithmetic, no per-channel affine — just read uint8 and write bf16. This
isolates the DRAM bandwidth ceiling for the ``(1 byte read, 2 bytes write)``
traffic pattern used by the fused input-norm kernel, so the fused kernel's
efficiency can be measured against it::

    efficiency = fused_bw / copy_bw

A value close to 1.0 means the fused kernel is saturating DRAM and its
arithmetic is effectively free. Anything significantly below 1.0 indicates
overhead that could still be removed.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from .utils import Test, benchmark


@triton.jit
def _u8_to_bf16_kernel(
    x_ptr,
    y_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(y_ptr + offs, x.to(tl.bfloat16), mask=mask)


class U8ToBf16Copy(Test):
    """Read uint8, write bf16, nothing else.

    Inherits directly from :class:`Test` rather than :class:`Naive` because
    there is no per-channel affine parameter and no eager reference that would
    be meaningful — the point is to measure pure memory throughput.
    """

    use_cuda_graph = True
    clear_l2_cache = True
    warmup_iters = 20
    bench_iters = 200

    # Nothing to verify against; this is a bandwidth probe, not a correctness
    # target.
    verify = False

    def __init__(
        self,
        patches: int,
        channel: int,
        embed_size: int,
        inputs_dtype: torch.dtype = torch.uint8,
        outputs_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype = torch.bfloat16,
        block: int = 4096,
    ) -> None:
        super().__init__(compute_dtype=compute_dtype)

        assert inputs_dtype == torch.uint8, "baseline only supports uint8 input"
        assert outputs_dtype == torch.bfloat16, "baseline only supports bf16 output"

        self.patches = patches
        self.channel = channel
        self.embed_size = embed_size
        self.numel = patches * channel * embed_size
        self.block = block

        # Flat 1-D buffers so index arithmetic cannot interfere with the
        # bandwidth measurement.
        self.inputs = torch.randint(
            0, 256, (self.numel,), dtype=torch.uint8, device="cuda:0"
        )
        self._outputs = torch.empty(
            (self.numel,), dtype=torch.bfloat16, device="cuda:0"
        )

    # ----- Test interface --------------------------------------------------
    def function_under_test(self) -> None:
        grid = (triton.cdiv(self.numel, self.block),)
        _u8_to_bf16_kernel[grid](
            self.inputs,
            self._outputs,
            self.numel,
            BLOCK=self.block,
        )

    def reference(self) -> torch.Tensor:
        raise NotImplementedError("baseline has no reference output")

    def output_tensor(self) -> torch.Tensor:
        return self._outputs

    def nelement(self) -> int:
        return self.numel

    def size(self) -> int:
        # 1 byte read (uint8) + 2 bytes write (bf16) per element.
        return self.numel * 3


if __name__ == "__main__":
    print("Test uint8 -> bf16 copy (bandwidth ceiling)!")

    # Sweep the same shapes as the fused kernel benchmark so the two sets of
    # numbers are directly comparable.
    for n in range(8, 19):
        try:
            benchmark(
                U8ToBf16Copy,
                torch.bfloat16,
                patches=2**n,
                channel=3,
                embed_size=1024,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at patches=2**{n}; skipping.")
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()