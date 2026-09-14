# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""``torch.compile`` baseline for the fused input-normalization kernel.

Wraps the eager broadcasted multiply-add (see :mod:`naive`) into a compiled
callable. Depending on the chosen mode, Inductor will either fuse the whole
expression into a single pass or fall back to a multi-kernel schedule with an
intermediate fp32 materialization. Either way, this baseline quantifies what a
general-purpose compiler achieves on the same problem, which is the honest bar
the hand-written Triton kernel has to beat.
"""

from __future__ import annotations

from typing import Any

import torch

from .naive import Naive
from .utils import benchmark


class TorchCompile(Naive):
    """``torch.compile``-wrapped eager multiply-add.

    ``reference()`` is inherited from :class:`Naive` and computed with plain
    eager ops, so the compiled path is verified against an implementation that
    does not depend on the compiler. This catches Inductor-specific numerical
    issues (accumulation order changes, constant folding, dtype promotion).
    """

    use_cuda_graph = True

    # First call into the compiled function triggers Inductor compilation,
    # which can take seconds. The driver's warmup loop covers it, but keep the
    # warmup window modest so the sweep stays responsive.
    warmup_iters = 10
    bench_iters = 200

    # This is an alternative implementation of the same computation, so verify
    # it against the eager reference.
    verify = True

    # ``max-autotune-no-cudagraphs`` lets Inductor autotune each subgraph while
    # leaving CUDA-graph management to the benchmark driver. Using plain
    # ``"max-autotune"`` here would make Inductor capture its own CUDA graph,
    # and the driver's outer capture would then fail with
    # "CUDA graphs cannot be nested".
    compile_mode: str = "max-autotune-no-cudagraphs"
    compile_fullgraph: bool = False

    # Dynamic shapes are essential for this sweep. With ``dynamic=False``,
    # each new ``patches`` value is a fresh (shape-specialized) compilation;
    # after ``torch._dynamo.config.recompile_limit`` (default 8) recompiles,
    # Dynamo gives up and silently falls back to eager, so the tail of the
    # sweep stops measuring the compiled kernel at all.
    #
    # ``dynamic=True`` compiles once with symbolic shapes so a single artifact
    # handles every ``patches`` value. The cost is that Inductor cannot rely on
    # exact sizes for unrolling / vectorization choices, but for a simple
    # elementwise op that is negligible compared to the benefit of not
    # recompiling per shape.
    compile_dynamic: bool = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Allow per-instance overrides of the compiler configuration through
        # constructor kwargs so callers can sweep modes without subclassing.
        compile_mode = kwargs.pop("compile_mode", self.compile_mode)
        compile_fullgraph = kwargs.pop("compile_fullgraph", self.compile_fullgraph)
        compile_dynamic = kwargs.pop("compile_dynamic", self.compile_dynamic)

        super().__init__(*args, **kwargs)

        # Pre-reshape the affine parameters so ``torch.compile`` treats them as
        # ordinary tensor inputs rather than tracing a ``view`` on a Python
        # attribute. Without this, changing ``self.weight`` would invalidate
        # the compiled graph.
        self._weight_3d = self.weight.view(1, self.channel, 1)
        self._bias_3d = self.bias.view(1, self.channel, 1)

        # Compile eagerly inside ``__init__`` so the compilation cost is
        # attributed to the object's lifetime rather than to the first warmup
        # iteration. The actual compilation is lazy (first call), but holding
        # the compiled callable here makes the intent explicit.
        self._compiled = torch.compile(
            self._compiled_fn,
            mode=compile_mode,
            fullgraph=compile_fullgraph,
            dynamic=compile_dynamic,
        )

    # ----- Compiled core ---------------------------------------------------
    def _compiled_fn(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-function form of the affine transform.

        Returns a fresh tensor instead of writing into ``self._outputs`` in
        place: Inductor generates cleaner code for functional returns than for
        aliased writes, and the extra copy is charged uniformly across all
        baselines via ``size()`` in :class:`Naive`.
        """
        x = inputs.to(self.compute_dtype)
        return (x * weight + bias).to(self.outputs_dtype)

    # ----- Test interface --------------------------------------------------
    @torch.inference_mode()
    def function_under_test(self) -> None:
        out = self._compiled(self.inputs, self._weight_3d, self._bias_3d)
        self._outputs.copy_(out)


if __name__ == "__main__":
    # Raise Dynamo's recompilation budget as a safety net. Even with
    # ``dynamic=True``, a benchmark sweep can accumulate a few legitimate
    # recompiles (e.g. one per dtype combination). Hitting the limit causes
    # Dynamo to silently fall back to eager, which would invalidate the
    # bandwidth numbers for the tail of the sweep.
    torch._dynamo.config.cache_size_limit = 64

    print("Test torch.compile!")

    # Sweep the same grid as the other baselines so results are comparable.
    # With ``compile_dynamic=True`` the whole sweep needs only one compiled
    # artifact per ``compute_dtype`` (the dtype is captured as a constant by
    # the compiler and thus triggers a recompile when it changes).
    for compute_dtype in [torch.float32, torch.bfloat16]:
        for n in range(8, 19):
            try:
                benchmark(
                    TorchCompile,
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