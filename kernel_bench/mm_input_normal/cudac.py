# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA C implementation of the fused per-channel affine transform.

Computes ``y = x * weight[c] + bias[c]`` over a ``(patches, channel,
embed_size)`` tensor, with the arithmetic performed in fp32 inside the kernel
regardless of the I/O dtypes.
"""

from __future__ import annotations

import torch

from .utils import Test, benchmark


class UseCudaC(Test):
    """CUDA C kernel implementing the same computation as ``FusedInputNorm``.

    Mirrors the Triton tiling: a 2D grid ``(patches, cdiv(L, BLOCK_L))`` with
    one thread per element along L, and the C loop fully unrolled so the
    per-channel weight/bias scalars stay in registers.
    """

    use_cuda_graph = True
    clear_l2_cache = True
    warmup_iters = 20
    bench_iters = 200

    verify = True

    # Tile size along L (equals the CUDA block size, one element per thread).
    BLOCK_L = 256

    # ----- CUDA source ----------------------------------------------------
    cuda_src = r"""
    #include <torch/extension.h>
    #include <c10/cuda/CUDAStream.h>
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
    #include <cstdint>

    // ---- dtype conversion helpers ----------------------------------------
    __device__ __forceinline__ float to_f32(uint8_t v)        { return (float)v; }
    __device__ __forceinline__ float to_f32(__half v)         { return __half2float(v); }
    __device__ __forceinline__ float to_f32(__nv_bfloat16 v)  { return __bfloat162float(v); }
    __device__ __forceinline__ float to_f32(float v)          { return v; }

    __device__ __forceinline__ void from_f32(float v, __half& o)        { o = __float2half_rn(v); }
    __device__ __forceinline__ void from_f32(float v, __nv_bfloat16& o) { o = __float2bfloat16_rn(v); }
    __device__ __forceinline__ void from_f32(float v, float& o)         { o = v; }

    // ---- kernel ----------------------------------------------------------
    template <typename T_IN, typename T_OUT, int C, bool HAS_MASK>
    __global__ void fused_input_norm_cuda_kernel(
        const T_IN*  __restrict__ x,
        T_OUT*       __restrict__ y,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        int       L,
        long long stride_xn, long long stride_xc,
        long long stride_yn, long long stride_yc)
    {
        const int n  = blockIdx.x;
        const int lb = blockIdx.y;
        const long long l = (long long)lb * blockDim.x + threadIdx.x;
        const bool in_bounds = !HAS_MASK || (l < L);

        // Hoist per-patch base pointers out of the channel loop.
        const T_IN* x_n = x + (long long)n * stride_xn;
        T_OUT*      y_n = y + (long long)n * stride_yn;

        // C is a compile-time constant; fully unrolled.
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            const float w = weight[c];
            const float b = bias[c];
            if (in_bounds) {
                const float xv = to_f32(x_n[c * stride_xc + l]);
                const float yv = xv * w + b;   // fp32 compute
                from_f32(yv, y_n[c * stride_yc + l]);
            }
        }
    }

    // ---- dispatch --------------------------------------------------------
    template <typename T_IN, typename T_OUT, int C>
    void launch_fused(
        const torch::Tensor& x, torch::Tensor& y,
        const torch::Tensor& w, const torch::Tensor& b,
        int N, int L, int BLOCK_L)
    {
        dim3 grid(N, (L + BLOCK_L - 1) / BLOCK_L);
        dim3 block(BLOCK_L);
        const bool has_mask = (L % BLOCK_L) != 0;

        const T_IN*  xp = reinterpret_cast<const T_IN*>(x.data_ptr());
        T_OUT*       yp = reinterpret_cast<T_OUT*>(y.data_ptr());
        const float* wp = w.data_ptr<float>();
        const float* bp = b.data_ptr<float>();

        // Use the current PyTorch stream so CUDA-graph capture picks up the
        // kernel instead of silently launching on the legacy default stream
        // (which would result in an empty graph).
        auto stream = c10::cuda::getCurrentCUDAStream();

        if (has_mask) {
            fused_input_norm_cuda_kernel<T_IN, T_OUT, C, true>
                <<<grid, block, 0, stream>>>(
                    xp, yp, wp, bp, L,
                    x.stride(0), x.stride(1),
                    y.stride(0), y.stride(1));
        } else {
            fused_input_norm_cuda_kernel<T_IN, T_OUT, C, false>
                <<<grid, block, 0, stream>>>(
                    xp, yp, wp, bp, L,
                    x.stride(0), x.stride(1),
                    y.stride(0), y.stride(1));
        }
    }

    torch::Tensor fused_input_norm_cuda(
        torch::Tensor x, torch::Tensor y,
        torch::Tensor w, torch::Tensor b,
        int64_t block_l)
    {
        TORCH_CHECK(x.is_cuda() && y.is_cuda(), "tensors must be on CUDA");
        TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
                    "tensors must be contiguous");
        TORCH_CHECK(x.dim() == 3, "x must be (N, C, L)");
        TORCH_CHECK(y.sizes() == x.sizes(), "y must match x");
        TORCH_CHECK(w.scalar_type() == torch::kFloat32 &&
                    b.scalar_type() == torch::kFloat32,
                    "weight/bias must be float32");
        TORCH_CHECK(w.is_contiguous() && b.is_contiguous(),
                    "weight/bias must be contiguous");

        const int N = (int)x.size(0);
        const int C = (int)x.size(1);
        const int L = (int)x.size(2);
        TORCH_CHECK(C == 3, "this baseline is specialised for C == 3");
        TORCH_CHECK(w.numel() == C && b.numel() == C,
                    "weight/bias must have C elements");

        const auto in_dt  = x.scalar_type();
        const auto out_dt = y.scalar_type();

        // Input x output dtype dispatch. Only the combinations that the
        // fused path actually supports are instantiated.
        #define DISPATCH_OUT(IN_T)                                              \
            do {                                                                \
                if (out_dt == torch::kFloat16) {                                \
                    launch_fused<IN_T, __half, 3>(x, y, w, b, N, L, (int)block_l); \
                } else if (out_dt == torch::kBFloat16) {                        \
                    launch_fused<IN_T, __nv_bfloat16, 3>(x, y, w, b, N, L, (int)block_l); \
                } else if (out_dt == torch::kFloat32) {                         \
                    launch_fused<IN_T, float, 3>(x, y, w, b, N, L, (int)block_l); \
                } else {                                                        \
                    TORCH_CHECK(false, "unsupported output dtype");             \
                }                                                               \
            } while (0)

        if (in_dt == torch::kUInt8) {
            DISPATCH_OUT(uint8_t);
        } else if (in_dt == torch::kFloat16) {
            DISPATCH_OUT(__half);
        } else if (in_dt == torch::kBFloat16) {
            DISPATCH_OUT(__nv_bfloat16);
        } else if (in_dt == torch::kFloat32) {
            DISPATCH_OUT(float);
        } else {
            TORCH_CHECK(false, "unsupported input dtype");
        }
        #undef DISPATCH_OUT

        return y;
    }
    """

    cpp_src = r"""
    torch::Tensor fused_input_norm_cuda(
        torch::Tensor x, torch::Tensor y,
        torch::Tensor w, torch::Tensor b,
        int64_t block_l);
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

        # The CUDA kernel always computes in fp32 internally; the
        # ``compute_dtype`` knob therefore only affects the reference.
        assert compute_dtype == torch.float32, (
            "UseCudaC computes in fp32; pass compute_dtype=torch.float32"
        )
        assert inputs_dtype in (
            torch.uint8, torch.float16, torch.bfloat16, torch.float32
        ), f"unsupported input dtype: {inputs_dtype}"
        assert outputs_dtype in (
            torch.float16, torch.bfloat16, torch.float32
        ), f"unsupported output dtype: {outputs_dtype}"

        self.patches = patches
        self.channel = channel
        self.embed_size = embed_size
        self.inputs_dtype = inputs_dtype
        self.outputs_dtype = outputs_dtype

        # Per-channel affine parameters, kept in fp32 to match the kernel.
        self.weight = torch.randn(
            channel, dtype=torch.float32, device="cuda:0"
        )
        self.bias = torch.randn(
            channel, dtype=torch.float32, device="cuda:0"
        )

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

        # Compile the extension once. ``load_inline`` caches the build under
        # ``TORCH_EXTENSIONS_DIR`` (defaults to ``~/.cache/torch_extensions``),
        # so subsequent runs reuse the compiled .so without recompiling.
        from torch.utils.cpp_extension import load_inline

        self._module = load_inline(
            name="fused_input_norm_cuda_ext",
            cpp_sources=self.cpp_src,
            cuda_sources=self.cuda_src,
            functions=["fused_input_norm_cuda"],
            verbose=False,
        )

    # ----- Test interface --------------------------------------------------
    def function_under_test(self) -> None:
        self._module.fused_input_norm_cuda(
            self.inputs, self._outputs,
            self.weight, self.bias,
            self.BLOCK_L,
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
        # Input read + output write; the fp32 intermediate lives only in
        # registers and is not counted, matching the other implementations.
        return (
            self.inputs.nelement() * self.inputs.element_size()
            + self._outputs.nelement() * self._outputs.element_size()
        )


if __name__ == "__main__":
    print("Test UseCudaC!")

    for n in range(8, 19):
        try:
            benchmark(
                UseCudaC,
                compute_dtype=torch.float32,
                patches=2**n,
                channel=3,
                embed_size=1024,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at patches=2**{n}; skipping.")
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()