# SPDX-License-Identifier: Apache-2.0
"""Benchmark: 1D flat-index vs 2D tiled fused input-norm kernels."""

import torch
import triton
import triton.language as tl

from .utils import Test, benchmark


# ===========================================================================
# Kernel A: 1D flat index (the "naive parallel" variant you pasted)
# ===========================================================================
@triton.jit
def _fused_input_norm_1d_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr,
    numel, L,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    c = (offs // L) % C
    x = tl.load(
        x_ptr + offs, mask=mask, other=0, eviction_policy="evict_first"
    ).to(COMPUTE_DTYPE)
    w = tl.load(w_ptr + c, mask=mask, other=0).to(COMPUTE_DTYPE)
    b = tl.load(b_ptr + c, mask=mask, other=0).to(COMPUTE_DTYPE)
    tl.store(
        y_ptr + offs, x * w + b, mask=mask, eviction_policy="evict_first"
    )


# ===========================================================================
# Kernel B: 2D (N, L-block) grid, C unrolled per program
# ===========================================================================
@triton.jit
def _fused_input_norm_2d_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr,
    L,
    stride_xn, stride_xc,
    stride_yn, stride_yc,
    C: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_L: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    n = tl.program_id(0)
    lb = tl.program_id(1)
    offs = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_L), BLOCK_L)
    x_base = x_ptr + n * stride_xn + offs
    y_base = y_ptr + n * stride_yn + offs

    if HAS_MASK:
        mask = offs < L
        for c in tl.static_range(C):
            w = tl.load(w_ptr + c, eviction_policy="evict_last").to(COMPUTE_DTYPE)
            b = tl.load(b_ptr + c, eviction_policy="evict_last").to(COMPUTE_DTYPE)
            x = tl.load(x_base + c * stride_xc, mask=mask, other=0,
                        eviction_policy="evict_first").to(COMPUTE_DTYPE)
            tl.store(y_base + c * stride_yc, x * w + b, mask=mask,
                     eviction_policy="evict_first")
    else:
        for c in tl.static_range(C):
            w = tl.load(w_ptr + c, eviction_policy="evict_last").to(COMPUTE_DTYPE)
            b = tl.load(b_ptr + c, eviction_policy="evict_last").to(COMPUTE_DTYPE)
            x = tl.load(x_base + c * stride_xc,
                        eviction_policy="evict_first").to(COMPUTE_DTYPE)
            tl.store(y_base + c * stride_yc, x * w + b,
                     eviction_policy="evict_first")


_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


# ===========================================================================
# Shared base: tensor setup + accounting (mirrors Naive)
# ===========================================================================
class _FusedNormBase(Test):
    use_cuda_graph = True
    verify = True
    clear_l2_cache = True

    # 1D block size / 2D tile size, overridable via subclass
    BLOCK = 4096
    BLOCK_L = 2048

    def __init__(
        self,
        patches: int,
        channel: int,
        embed_size: int,
        inputs_dtype: torch.dtype,
        outputs_dtype: torch.dtype,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__(compute_dtype=compute_dtype)
        self.patches = patches
        self.channel = channel
        self.embed_size = embed_size
        self.inputs_dtype = inputs_dtype
        self.outputs_dtype = outputs_dtype

        self.weight = torch.randn(channel, dtype=compute_dtype, device="cuda:0")
        self.bias = torch.randn(channel, dtype=compute_dtype, device="cuda:0")

        if inputs_dtype == torch.uint8:
            self.inputs = torch.randint(
                0, 256, (patches, channel, embed_size),
                dtype=inputs_dtype, device="cuda:0",
            )
        else:
            self.inputs = torch.randn(
                (patches, channel, embed_size),
                dtype=inputs_dtype, device="cuda:0",
            )

        self._outputs = torch.empty(
            (patches, channel, embed_size),
            dtype=outputs_dtype, device="cuda:0",
        )

    def reference(self) -> torch.Tensor:
        ref = (
            self.inputs.to(self.compute_dtype)
            * self.weight.view(1, self.channel, 1)
            + self.bias.view(1, self.channel, 1)
        )
        return ref.to(self.outputs_dtype)

    def output_tensor(self) -> torch.Tensor:
        return self._outputs

    def nelement(self) -> int:
        return self.inputs.nelement()

    def size(self) -> int:
        return (
            self.inputs.nelement() * self.inputs.element_size()
            + self._outputs.nelement() * self._outputs.element_size()
        )


# ===========================================================================
# Benchmark target A: 1D flat kernel
# ===========================================================================
class FusedNorm1D(_FusedNormBase):
    """1D flat-index Triton kernel."""

    def function_under_test(self) -> None:
        numel = self.inputs.numel()
        grid = (triton.cdiv(numel, self.BLOCK),)
        _fused_input_norm_1d_kernel[grid](
            self.inputs, self._outputs,
            self.weight, self.bias,
            numel,
            self.embed_size,
            C=self.channel,
            BLOCK=self.BLOCK,
            COMPUTE_DTYPE=_TL_DTYPE[self.compute_dtype],
            num_warps=4,
        )


# ===========================================================================
# Benchmark target B: 2D tiled kernel
# ===========================================================================
class FusedNorm2D(_FusedNormBase):
    """2D (N, L-block) Triton kernel with C unrolled per program."""

    def function_under_test(self) -> None:
        N, C, L = self.inputs.shape
        block_l = self.BLOCK_L
        has_mask = (L % block_l) != 0
        grid = (N, triton.cdiv(L, block_l))
        _fused_input_norm_2d_kernel[grid](
            self.inputs, self._outputs,
            self.weight, self.bias,
            L,
            self.inputs.stride(0), self.inputs.stride(1),
            self._outputs.stride(0), self._outputs.stride(1),
            C=C,
            HAS_MASK=has_mask,
            BLOCK_L=block_l,
            COMPUTE_DTYPE=_TL_DTYPE[self.compute_dtype],
            num_warps=4,
        )


if __name__ == "__main__":
    # Fixed shape: (patches, channel, embed_size) = (2^n, 3, 1024).
    # Sweep patches so the L2-flush effect becomes visible once inputs
    # exceed the cache.
    for cls in (FusedNorm1D, FusedNorm2D):
        for compute_dtype in [torch.float32]:
            for n in range(8, 19):
                kwargs = dict(
                    inputs_dtype=torch.uint8,
                    outputs_dtype=torch.bfloat16,
                    patches=2**n,
                    channel=3,
                    embed_size=1024,
                )
                try:
                    benchmark(cls, compute_dtype, label=cls.__name__, **kwargs)
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM: {cls.__name__} @ patches=2**{n}")
                    torch.cuda.empty_cache()
                    break
        torch.cuda.empty_cache()