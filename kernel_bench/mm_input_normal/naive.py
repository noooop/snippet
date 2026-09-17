"""Naive eager reference for the fused input-normalization kernel.

Computes ``y = (x * weight[c] + bias[c]).to(outputs_dtype)`` with plain
broadcasted tensor ops. Serves three roles:

* the **numerical reference** that other implementations are verified against
  (``reference()``),
* the **performance floor** for the same computation (eager path materializes
  an fp32 intermediate between input and output), and
* the **base class** for other baselines (``BatchNorm1d``, ``TorchCompile``,
  ``UseTriton``) that want the same tensor setup and accounting.
"""

import torch

from .utils import Test, benchmark


class Naive(Test):
    """Eager broadcasted multiply-add over ``(N, C, L)``.

    This is the computation the fused kernel performs; expressing it in plain
    PyTorch makes it an independent reference that does not share code with the
    kernel under test.
    """

    use_cuda_graph = False
    verify = False

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

        # Per-channel affine parameters, kept in the compute dtype so the eager
        # path matches the fused kernel's internal precision.
        self.weight = torch.randn(
            channel, dtype=compute_dtype, device="cuda:0"
        )
        self.bias = torch.randn(
            channel, dtype=compute_dtype, device="cuda:0"
        )

        # Input tensor. ``uint8`` mirrors quantized pixel_values; the wider
        # dtypes exercise the pre-normalized path.
        if inputs_dtype == torch.uint8:
            self.inputs = torch.randint(
                0,
                256,
                (patches, channel, embed_size),
                dtype=inputs_dtype,
                device="cuda:0",
            )
        else:
            self.inputs = torch.randn(
                (patches, channel, embed_size),
                dtype=inputs_dtype,
                device="cuda:0",
            )

        # Output buffer is fully overwritten by every implementation; using
        # ``empty`` avoids an unnecessary RNG pass over a potentially large
        # tensor.
        self._outputs = torch.empty(
            (patches, channel, embed_size),
            dtype=outputs_dtype,
            device="cuda:0",
        )

    # ----- Core op ---------------------------------------------------------
    @torch.inference_mode()
    def op(
        self, inputs: torch.Tensor, outputs: torch.Tensor
    ) -> None:
        """Write ``inputs * weight + bias`` into ``outputs`` in place.

        Kept as a standalone method so subclasses can reuse it without
        duplicating the broadcast / cast sequence.
        """

        outputs.copy_(
            (
                inputs * self.weight.view(1, self.channel, 1)
                + self.bias.view(1, self.channel, 1)
            ).to(self.outputs_dtype)
        )

    # ----- Test interface --------------------------------------------------
    def function_under_test(self) -> None:
        self.op(self.inputs, self._outputs)

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


class GC(Naive):
    use_cuda_graph = True


if __name__ == "__main__":
    print("Test Naive!")

    for compute_dtype in [torch.float32, torch.bfloat16]:
        for n in range(8, 19):
            try:
                benchmark(
                    Naive,
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

    print("Test GC!")
    for compute_dtype in [torch.float32, torch.bfloat16]:
        for n in range(8, 19):
            try:
                benchmark(
                    GC,
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