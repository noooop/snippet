# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton fused input-norm baseline.

Routes the workload through ``vllm.model_executor.layers.fusion.mm_input_norm
.fused_input_norm_kernel``, which performs the per-channel affine transform in
a single pass over ``(N, C, L)``. This is the implementation the benchmark is
ultimately validating; the eager / BN / ``torch.compile`` baselines exist to
quantify how much the fused kernel actually buys.
"""
from typing import Any

import torch

from .naive import Naive
from .utils import benchmark

import vllm.model_executor.layers.fusion.mm_input_norm as mm_input_norm


class UseTriton(Naive):
    """Triton fused kernel adapter.

    The kernel requires contiguous inputs and takes the affine parameters as
    1-D ``(C,)`` tensors; ``Naive`` already provides both in the right shape.
    ``output_tensor()`` is inherited from ``Naive`` and points at the same
    buffer the kernel writes into.
    """

    use_cuda_graph = True
    verify = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # The kernel signature expects contiguous inputs. ``Naive`` produces
        # contiguous tensors, but assert explicitly so a future change to
        # ``Naive`` cannot silently break this path.
        assert self.inputs.is_contiguous(), "UseTriton requires contiguous inputs"
        assert self._outputs.is_contiguous(), "UseTriton requires contiguous outputs"
        assert self.weight.is_contiguous() and self.bias.is_contiguous(), (
            "UseTriton requires contiguous weight/bias"
        )

    # ----- Test interface --------------------------------------------------
    def function_under_test(self) -> None:
        mm_input_norm.fused_mm_input_norm_triton(
            self.inputs,
            self._outputs,
            self.weight,
            self.bias,
        )


if __name__ == "__main__":
    print("Test Triton!")

    for inputs_dtype in [torch.uint8]:
        for outputs_dtype in [torch.bfloat16]:
            for embed_size in [1024]:
                for n in range(8, 19):
                    try:
                        benchmark(
                            UseTriton,
                            compute_dtype=torch.float32,
                            inputs_dtype=inputs_dtype,
                            outputs_dtype=outputs_dtype,
                            patches=2**n,
                            channel=3,
                            embed_size=embed_size,
                            label=(
                                f"UseTriton[in={str(inputs_dtype).replace('torch.', '')},"
                                f"out={str(outputs_dtype).replace('torch.', '')},"
                                f"L={embed_size}]"
                            ),
                        )
                    except torch.cuda.OutOfMemoryError:
                        print(
                            f"OOM at in={inputs_dtype}, out={outputs_dtype}, "
                            f"L={embed_size}, patches=2**{n}; skipping."
                        )
                        torch.cuda.empty_cache()
                        continue
                    torch.cuda.empty_cache()
        torch.cuda.empty_cache()