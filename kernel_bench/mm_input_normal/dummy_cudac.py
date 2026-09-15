# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA C baseline: uint8 -> bf16 copy."""

from __future__ import annotations

import torch

from .utils import Test, benchmark


class CudaU8ToBf16Copy(Test):
    """uint8 -> bf16 copy implemented as a raw CUDA kernel."""

    use_cuda_graph = False
    clear_l2_cache = True
    warmup_iters = 20
    bench_iters = 200

    verify = False

    # ----- CUDA source ----------------------------------------------------
    cuda_src = r"""
    #include <cuda_runtime.h>
    #include <cuda_bf16.h>
    #include <cstdint>

    // Each thread processes 4 contiguous elements: one uchar4 read (4 B) and
    // two uint32 writes (8 B of bf16). The 1:2 read/write ratio matches the
    // natural width of the traffic pattern.
    __global__ void u8_to_bf16_vec4_kernel(
        const unsigned char* __restrict__ x,
        __nv_bfloat16* __restrict__ y,
        long long n
    ) {
        long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 4;

        if (i + 3 < n) {
            uchar4 v = *reinterpret_cast<const uchar4*>(x + i);

            __nv_bfloat16 r0 = __float2bfloat16((float)v.x);
            __nv_bfloat16 r1 = __float2bfloat16((float)v.y);
            __nv_bfloat16 r2 = __float2bfloat16((float)v.z);
            __nv_bfloat16 r3 = __float2bfloat16((float)v.w);

            uint32_t p0 = (uint32_t)__bfloat16_as_ushort(r0)
                        | ((uint32_t)__bfloat16_as_ushort(r1) << 16);
            uint32_t p1 = (uint32_t)__bfloat16_as_ushort(r2)
                        | ((uint32_t)__bfloat16_as_ushort(r3) << 16);

            reinterpret_cast<uint32_t*>(y + i)[0] = p0;
            reinterpret_cast<uint32_t*>(y + i)[1] = p1;
        } else {
            // Tail: scalar fallback. Also covers the case where n is not a
            // multiple of 4.
            for (long long j = i; j < n; ++j) {
                y[j] = __float2bfloat16((float)x[j]);
            }
        }
    }

    torch::Tensor u8_to_bf16_cuda(torch::Tensor x, torch::Tensor y) {
        TORCH_CHECK(x.is_cuda() && y.is_cuda(), "tensors must be on CUDA");
        TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
                    "tensors must be contiguous");
        TORCH_CHECK(x.scalar_type() == torch::kUInt8,
                    "x must be uint8");
        TORCH_CHECK(y.scalar_type() == torch::kBFloat16,
                    "y must be bfloat16");
        TORCH_CHECK(x.numel() == y.numel(), "size mismatch");

        long long n = (long long)x.numel();
        if (n == 0) {
            return y;
        }

        const int threads = 256;
        // One thread per 4 elements, rounded up.
        int blocks = (int)((n / 4 + threads - 1) / threads);

        u8_to_bf16_vec4_kernel<<<blocks, threads>>>(
            x.data_ptr<unsigned char>(),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
            n);

        return y;
    }
    """

    cpp_src = r"""
    torch::Tensor u8_to_bf16_cuda(torch::Tensor x, torch::Tensor y);
    """

    def __init__(
        self,
        patches: int,
        channel: int,
        embed_size: int,
        inputs_dtype: torch.dtype = torch.uint8,
        outputs_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(compute_dtype=compute_dtype)

        assert inputs_dtype == torch.uint8, "baseline only supports uint8 input"
        assert outputs_dtype == torch.bfloat16, "baseline only supports bf16 output"

        self.patches = patches
        self.channel = channel
        self.embed_size = embed_size
        self.numel = patches * channel * embed_size

        self.inputs = torch.randint(
            0, 256, (self.numel,), dtype=torch.uint8, device="cuda:0"
        )
        self._outputs = torch.empty(
            (self.numel,), dtype=torch.bfloat16, device="cuda:0"
        )

        # Compile the extension once. ``load_inline`` caches the build under
        # ``TORCH_EXTENSIONS_DIR`` (defaults to ``~/.cache/torch_extensions``),
        # so subsequent runs reuse the compiled .so without recompiling.
        from torch.utils.cpp_extension import load_inline

        self._module = load_inline(
            name="u8_to_bf16_copy_ext",
            cpp_sources=self.cpp_src,
            cuda_sources=self.cuda_src,
            functions=["u8_to_bf16_cuda"],
            verbose=False,
        )

    # ----- Test interface --------------------------------------------------
    def function_under_test(self) -> None:
        self._module.u8_to_bf16_cuda(self.inputs, self._outputs)

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
    print("Test uint8 -> bf16 copy (CUDA C baseline)!")

    for n in range(8, 19):
        try:
            benchmark(
                CudaU8ToBf16Copy,
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