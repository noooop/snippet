# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark harness for fused input-normalization kernels.

The harness is intentionally small and self-contained:

* A ``Test`` base class defines the interface every benchmarked implementation
  must implement (setup, one forward pass, reference output, size accounting).
* ``benchmark()`` drives a single class through warmup, optional correctness
  verification, optional CUDA-graph capture, and a timed loop that flushes L2
  between iterations.
* ``format_size()`` renders byte counts and bandwidths in human-readable units.

Design notes
------------

* **Warmup vs. timed iterations are separate.** ``warmup_iters`` triggers
  Triton JIT compilation and any lazy initialization; ``bench_iters`` is the
  timed window. Raising the timing count does not force more warmup work.

* **Verification is opt-in per class.** Implementations that are themselves the
  reference set ``verify = False``; everything else should leave it ``True``.

* **CUDA-graph capture happens after verification.** Reference tensors can be
  large; releasing them before capture keeps the graph's private memory pool
  small.

* **L2 flush buffer is allocated once.** Reallocating per iteration would
  pollute the caching allocator's steady state and cannot guarantee eviction of
  the same physical pages.
"""

from __future__ import annotations

from typing import Any

import torch


__all__ = ["Test", "benchmark", "format_size", "format_gb"]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def format_size(
    size: float, decimal_places: int = 2, use_binary: bool = True
) -> str:
    """Format a byte count as a human-readable string (B/KB/MB/GB/TB)."""
    if size == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    base = 1024 if use_binary else 1000
    exponent = 0
    while size >= base and exponent < len(units) - 1:
        size /= base
        exponent += 1
    return f"{size:.{decimal_places}f} {units[exponent]}"


def format_gb(
    size: float, decimal_places: int = 2, use_binary: bool = True
) -> str:
    """Format a byte count always in GB (GiB when use_binary=True)."""
    base = 1024 if use_binary else 1000
    return f"{size / (base ** 3):.{decimal_places}f} GB"


# ---------------------------------------------------------------------------
# L2 flusher
# ---------------------------------------------------------------------------
class _L2Flusher:
    """Evict the contents of L2 between timed iterations.

    The dummy buffer is allocated once. Each ``flush()`` performs an in-place
    add, which is enough to replace L2's working set with the dummy's pages.
    A subsequent synchronize guarantees the write has drained before the next
    timed region begins.
    """

    def __init__(self, cache_size_mb: int = 256, device: str = "cuda") -> None:
        n_elements = int(cache_size_mb * 1024 * 1024 / 4)
        self._dummy = torch.ones(n_elements, dtype=torch.float32, device=device)

    def flush(self) -> None:
        self._dummy += 1
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Test interface
# ---------------------------------------------------------------------------
class Test:
    """Interface implemented by every benchmarked workload.

    Subclasses must override the four ``@abstractmethod``-like methods below.
    The class attributes control how ``benchmark()`` drives the workload.
    """

    # Number of warmup iterations performed before verification and capture.
    warmup_iters: int = 20

    # Number of timed iterations.
    bench_iters: int = 200

    # Whether to flush L2 between timed iterations.
    clear_l2_cache: bool = True

    # Whether to capture the workload into a CUDA graph.
    use_cuda_graph: bool = True

    # Whether to compare ``output_tensor()`` against ``reference()`` before
    # timing. Implementations that *are* the reference should set this to False.
    verify: bool = True

    def __init__(self, compute_dtype: torch.dtype, **kwargs: Any) -> None:
        # Subclasses normally store their own state via ``super().__init__``
        # followed by their own setup.
        self.compute_dtype = compute_dtype

    # ----- Workload --------------------------------------------------------
    def function_under_test(self) -> None:
        """Execute the workload exactly once.

        Must be side-effect-free across invocations apart from writing to the
        output tensor, and must not synchronize (which would break CUDA-graph
        capture).
        """
        raise NotImplementedError

    def reference(self) -> torch.Tensor:
        """Return the expected output as a fresh tensor.

        The reference is computed with plain PyTorch ops so it does not depend
        on the implementation being tested.
        """
        raise NotImplementedError

    def output_tensor(self) -> torch.Tensor:
        """Return the buffer written by ``function_under_test``."""
        raise NotImplementedError

    # ----- Accounting ------------------------------------------------------
    def nelement(self) -> int:
        """Number of input elements processed."""
        raise NotImplementedError

    def size(self) -> int:
        """Number of bytes moved (input read + output write) per invocation.

        Implementations may count only the primary input and output buffers;
        internal intermediates (e.g. an fp32 cast inside the kernel) are
        deliberately excluded so all implementations are scored the same way.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------
def benchmark(
    cls: type[Test],
    compute_dtype: torch.dtype,
    *,
    warmup: int | None = None,
    iters: int | None = None,
    verify: bool | None = None,
    clear_l2_cache: bool | None = None,
    use_cuda_graph: bool | None = None,
    label: str | None = None,
    **kwargs: Any,
) -> float:
    """Benchmark ``cls`` with the given ``compute_dtype`` and constructor kwargs.

    Keyword overrides (``warmup``, ``iters``, ``verify``, ``clear_l2_cache``,
    ``use_cuda_graph``) take precedence over the class attributes, which lets
    callers sweep those knobs without subclassing.

    Returns:
        Effective bandwidth in bytes per second: ``size * iters / elapsed_s``.
    """
    c = cls(compute_dtype=compute_dtype, **kwargs)

    warmup_iters = warmup if warmup is not None else c.warmup_iters
    bench_iters = iters if iters is not None else c.bench_iters
    do_verify = verify if verify is not None else c.verify
    do_flush = (
        clear_l2_cache if clear_l2_cache is not None else c.clear_l2_cache
    )
    do_capture = (
        use_cuda_graph if use_cuda_graph is not None else c.use_cuda_graph
    )

    # ----- 1. Warmup --------------------------------------------------------
    # Triggers Triton JIT compilation and any lazy module initialization.
    for _ in range(warmup_iters):
        c.function_under_test()
    torch.cuda.synchronize()

    # ----- 2. Correctness verification -------------------------------------
    if do_verify:
        ref = c.reference()
        c.function_under_test()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c.output_tensor(), ref, rtol=1e-2, atol=1e-2
        )
        del ref
        torch.cuda.empty_cache()

    # ----- 3. CUDA-graph capture -------------------------------------------
    if do_capture:
        g = torch.cuda.CUDAGraph()
        # Capture on the current (default) stream so the L2 flusher below
        # operates on the same stream and correctly evicts the graph's pages.
        with torch.cuda.graph(g):
            c.function_under_test()
        torch.cuda.synchronize()
        run = g.replay
    else:
        run = c.function_under_test

    # Warm up the chosen execution path (graph replay or direct launch).
    for _ in range(warmup_iters):
        run()
    torch.cuda.synchronize()

    # ----- 4. Timed loop ----------------------------------------------------
    flusher = _L2Flusher() if do_flush else None
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    elapsed_s = 0.0
    for _ in range(bench_iters):
        if flusher is not None:
            flusher.flush()

        start_event.record()
        run()
        end_event.record()

        torch.cuda.synchronize()
        elapsed_s += start_event.elapsed_time(end_event) / 1000.0

    # ----- 5. Report --------------------------------------------------------
    size = c.size()
    nelement = c.nelement()
    bandwidth = size * bench_iters / elapsed_s

    name = label if label is not None else cls.__name__
    print(
        f"[{name}] compute={str(compute_dtype).replace('torch.', '')}, "
        f"nelement: {format_size(nelement)}, "
        f"size: {format_size(size)}, "
        f"bw: {format_gb(bandwidth)}/s"
    )

    return bandwidth