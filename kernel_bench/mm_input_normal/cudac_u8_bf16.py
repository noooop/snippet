# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA C benchmark: C=3, uint8 -> bf16, vectorised along L.

Only exercises the specialised path: input ``(patches, 3, embed_size)`` in
``uint8``, output in ``bfloat16``, per-channel affine computed in fp32. Each
thread processes 4 consecutive elements along L per channel, using one
``uint32`` load and two ``uint32`` stores.
"""

from __future__ import annotations

import torch

from .utils import Test, benchmark


class CudaC3U8ToBf16(Test):
    """Fused per-channel affine, C=3, uint8 -> bf16, vec4 along L."""

    use_cuda_graph = True
    clear_l2_cache = True
    warmup_iters = 20
    bench_iters = 200

    verify = True

    # Each thread handles VEC=4 consecutive elements along L.
    VEC = 4
    THREADS = 64  # -> 256 L-elements per block

    # ----- CUDA source ----------------------------------------------------
    cuda_src = r"""
    #include <torch/extension.h>
    #include <c10/cuda/CUDAStream.h>
    #include <cuda_runtime.h>
    #include <cuda_bf16.h>
    #include <cstdint>

    // -----------------------------------------------------------------
    // Specialised C=3, uint8 -> bf16, vectorised along L.
    //
    // Each thread processes VEC=4 consecutive elements along L for each of
    // the three channels:
    //   - 1x uint32 load  == 4x uint8  (4 bytes read)
    //   - 2x uint32 store == 4x bf16   (8 bytes written)
    // This matches the natural 1:2 read/write width and cuts the per-element
    // instruction count by ~4x versus the scalar version.
    //
    // Layout assumptions:
    //   x, y are contiguous (N, 3, L) tensors.
    //   A masked tail handles L not being a multiple of VEC.
    // -----------------------------------------------------------------

    constexpr int VEC = 4;

    template <bool HAS_MASK>
    __global__ void fused_input_norm_c3_u8_to_bf16_vec4_kernel(
        const unsigned char* __restrict__ x,
        __nv_bfloat16*       __restrict__ y,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        int       L,
        long long stride_xn,
        long long stride_yn)
    {
        const int n  = blockIdx.x;
        const int lb = blockIdx.y;

        // Base l for this thread (4 consecutive elements).
        const int l0 = lb * (blockDim.x * VEC) + threadIdx.x * VEC;

        const unsigned char* x_n = x + (long long)n * stride_xn;
        __nv_bfloat16*       y_n = y + (long long)n * stride_yn;

        // Hoist per-channel scalars; uniform across the block, broadcast
        // from L1/constant cache.
        const float w0 = weight[0], w1 = weight[1], w2 = weight[2];
        const float b0 = bias[0],   b1 = bias[1],   b2 = bias[2];

        if (!HAS_MASK || (l0 + VEC - 1 < L)) {
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                const uint32_t packed = *reinterpret_cast<const uint32_t*>(
                    x_n + (long long)c * L + l0);

                const unsigned int u0 = (packed      ) & 0xFFu;
                const unsigned int u1 = (packed >>  8) & 0xFFu;
                const unsigned int u2 = (packed >> 16) & 0xFFu;
                const unsigned int u3 = (packed >> 24) & 0xFFu;

                const float w = (c == 0) ? w0 : (c == 1 ? w1 : w2);
                const float b = (c == 0) ? b0 : (c == 1 ? b1 : b2);

                const __nv_bfloat16 r0 = __float2bfloat16((float)u0 * w + b);
                const __nv_bfloat16 r1 = __float2bfloat16((float)u1 * w + b);
                const __nv_bfloat16 r2 = __float2bfloat16((float)u2 * w + b);
                const __nv_bfloat16 r3 = __float2bfloat16((float)u3 * w + b);

                const uint32_t p0 = (uint32_t)__bfloat16_as_ushort(r0)
                                  | ((uint32_t)__bfloat16_as_ushort(r1) << 16);
                const uint32_t p1 = (uint32_t)__bfloat16_as_ushort(r2)
                                  | ((uint32_t)__bfloat16_as_ushort(r3) << 16);

                uint32_t* y_dst = reinterpret_cast<uint32_t*>(
                    y_n + (long long)c * L + l0);
                y_dst[0] = p0;
                y_dst[1] = p1;
            }
        } else {
            // Masked tail: element-wise, only executed by the last block(s).
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                const float w = (c == 0) ? w0 : (c == 1 ? w1 : w2);
                const float b = (c == 0) ? b0 : (c == 1 ? b1 : b2);
                #pragma unroll
                for (int k = 0; k < VEC; ++k) {
                    const int l = l0 + k;
                    if (l < L) {
                        const float xv = (float)x_n[(long long)c * L + l];
                        y_n[(long long)c * L + l] =
                            __float2bfloat16(xv * w + b);
                    }
                }
            }
        }
    }

    // ---- host-side launcher -----------------------------------------
    torch::Tensor fused_input_norm_c3_u8_to_bf16(
        torch::Tensor x, torch::Tensor y,
        torch::Tensor w, torch::Tensor b,
        int64_t threads)
    {
        TORCH_CHECK(x.is_cuda() && y.is_cuda(), "tensors must be on CUDA");
        TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
                    "tensors must be contiguous");
        TORCH_CHECK(x.dim() == 3 && y.dim() == 3, "tensors must be (N, 3, L)");
        TORCH_CHECK(x.size(1) == 3, "C must be 3");
        TORCH_CHECK(y.sizes() == x.sizes(), "y must match x");
        TORCH_CHECK(x.scalar_type() == torch::kUInt8, "x must be uint8");
        TORCH_CHECK(y.scalar_type() == torch::kBFloat16, "y must be bfloat16");
        TORCH_CHECK(w.scalar_type() == torch::kFloat32 &&
                    b.scalar_type() == torch::kFloat32,
                    "weight/bias must be float32");
        TORCH_CHECK(w.is_contiguous() && b.is_contiguous(),
                    "weight/bias must be contiguous");
        TORCH_CHECK(w.numel() == 3 && b.numel() == 3,
                    "weight/bias must have 3 elements");

        const int N = (int)x.size(0);
        const int L = (int)x.size(2);
        if (N == 0 || L == 0) {
            return y;
        }

        const int THREADS = (int)threads;
        const int BLOCK_L = THREADS * VEC;             // L-elements per block
        const int num_l_blocks = (L + BLOCK_L - 1) / BLOCK_L;
        const bool has_mask = (L % BLOCK_L) != 0;

        dim3 grid(N, num_l_blocks);
        dim3 block(THREADS);

        auto stream = c10::cuda::getCurrentCUDAStream();

        const unsigned char* xp =
            reinterpret_cast<const unsigned char*>(x.data_ptr());
        __nv_bfloat16* yp =
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
        const float* wp = w.data_ptr<float>();
        const float* bp = b.data_ptr<float>();

        const long long stride_xn = (long long)x.stride(0);
        const long long stride_yn = (long long)y.stride(0);

        if (has_mask) {
            fused_input_norm_c3_u8_to_bf16_vec4_kernel<true>
                <<<grid, block, 0, stream>>>(
                    xp, yp, wp, bp, L, stride_xn, stride_yn);
        } else {
            fused_input_norm_c3_u8_to_bf16_vec4_kernel<false>
                <<<grid, block, 0, stream>>>(
                    xp, yp, wp, bp, L, stride_xn, stride_yn);
        }
        return y;
    }
    """

    cpp_src = r"""
    torch::Tensor fused_input_norm_c3_u8_to_bf16(
        torch::Tensor x, torch::Tensor y,
        torch::Tensor w, torch::Tensor b,
        int64_t threads);
    """

    def __init__(
        self,
        patches: int,
        channel: int,
        embed_size: int,
        inputs_dtype: torch.dtype = torch.uint8,
        outputs_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(compute_dtype=compute_dtype)

        assert channel == 3, "specialised kernel only supports C == 3"
        assert inputs_dtype == torch.uint8, "specialised kernel only supports uint8 input"
        assert outputs_dtype == torch.bfloat16, "specialised kernel only supports bf16 output"
        assert compute_dtype == torch.float32, "specialised kernel computes in fp32"

        self.patches = patches
        self.channel = channel
        self.embed_size = embed_size

        self.weight = torch.randn(3, dtype=torch.float32, device="cuda:0")
        self.bias = torch.randn(3, dtype=torch.float32, device="cuda:0")

        self.inputs = torch.randint(
            0, 256, (patches, 3, embed_size),
            dtype=torch.uint8, device="cuda:0",
        )
        self._outputs = torch.empty(
            (patches, 3, embed_size),
            dtype=torch.bfloat16, device="cuda:0",
        )

        from torch.utils.cpp_extension import load_inline

        self._module = load_inline(
            name="fused_input_norm_c3_u8_to_bf16_ext",
            cpp_sources=self.cpp_src,
            cuda_sources=self.cuda_src,
            functions=["fused_input_norm_c3_u8_to_bf16"],
            verbose=False,
        )

    # ----- Test interface --------------------------------------------------
    def function_under_test(self) -> None:
        self._module.fused_input_norm_c3_u8_to_bf16(
            self.inputs, self._outputs,
            self.weight, self.bias,
            self.THREADS,
        )

    def reference(self) -> torch.Tensor:
        ref = (
            self.inputs.to(torch.float32)
            * self.weight.view(1, 3, 1)
            + self.bias.view(1, 3, 1)
        )
        return ref.to(torch.bfloat16)

    def output_tensor(self) -> torch.Tensor:
        return self._outputs

    def nelement(self) -> int:
        return self.inputs.nelement()

    def size(self) -> int:
        # 1 byte read (uint8) + 2 bytes write (bf16) per element.
        return self.inputs.nelement() * 3


if __name__ == "__main__":
    print("Test CudaC3U8ToBf16 (specialised C=3, uint8 -> bf16, vec4 along L)!")

    for n in range(8, 19):
        try:
            benchmark(
                CudaC3U8ToBf16,
                torch.float32,
                patches=2**n,
                channel=3,
                embed_size=1024,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at patches=2**{n}; skipping.")
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()