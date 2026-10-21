#!/usr/bin/env python3
"""
Benchmark JSON encoding for /v1/embeddings responses at fixed dim=1024.

Paths compared:
  1. baseline_tolist : .tolist() + orjson.dumps
  2. pr_numpy        : ndarray  + orjson.dumps(OPT_SERIALIZE_NUMPY)
  3. pydantic_json   : response_model -> model_dump(mode="json") -> orjson
  4. pydantic_numpy  : model -> model_dump(mode="python") -> orjson(SERIALIZE_NUMPY)

A second experiment isolates the batch=512 uptick by rerunning the
benchmark with garbage collection disabled.
"""

import gc
import random
import statistics
import time
from typing import Any

import numpy as np
import orjson
from pydantic import BaseModel, ConfigDict, Field, field_serializer


# --------------------------------------------------------------------------- #
# Pydantic response model                                                     #
# --------------------------------------------------------------------------- #

class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    object: str = "list"
    data: list[dict[str, Any]]
    model: str
    usage: dict[str, int] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "total_tokens": 0}
    )

    @field_serializer("data")
    def _ser_data(self, data):
        # Pydantic v2 cannot serialize ndarray under mode="json"; convert manually.
        return [
            {
                **item,
                "embedding": item["embedding"].tolist()
                if isinstance(item.get("embedding"), np.ndarray)
                else item.get("embedding"),
            }
            for item in data
        ]


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

def make_fixture(batch: int, dim: int, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    emb = np.ascontiguousarray(rng.standard_normal((batch, dim), dtype=np.float32))
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": emb[i]}
            for i in range(batch)
        ],
        "model": "intfloat/multilingual-e5-small",
        "usage": {"prompt_tokens": batch, "total_tokens": batch},
    }


# --------------------------------------------------------------------------- #
# Encoders                                                                    #
# --------------------------------------------------------------------------- #

def baseline_tolist(p: dict[str, Any]) -> bytes:
    data = [{**it, "embedding": it["embedding"].tolist()} for it in p["data"]]
    return orjson.dumps({**p, "data": data})


def pr_numpy(p: dict[str, Any]) -> bytes:
    return orjson.dumps(p, option=orjson.OPT_SERIALIZE_NUMPY)


def pydantic_json(p: dict[str, Any]) -> bytes:
    resp = EmbeddingResponse(**p)
    return orjson.dumps(resp.model_dump(mode="json"))


def pydantic_numpy(p: dict[str, Any]) -> bytes:
    resp = EmbeddingResponse(**p)
    return orjson.dumps(
        resp.model_dump(mode="python"), option=orjson.OPT_SERIALIZE_NUMPY
    )


ENCODERS = [
    ("baseline_tolist", baseline_tolist),
    ("pr_numpy",        pr_numpy),
    ("pydantic_json",   pydantic_json),
    ("pydantic_numpy",  pydantic_numpy),
]


# --------------------------------------------------------------------------- #
# Timing                                                                      #
# --------------------------------------------------------------------------- #

def bench_one(
    fixture, warmup: int, groups: int, gc_enabled: bool = True
) -> dict[str, float]:
    for _, fn in ENCODERS:
        for _ in range(warmup):
            fn(fixture)

    results = {name: [] for name, _ in ENCODERS}
    rng = random.Random(42)

    # Toggle GC state only around the timed region to isolate its effect.
    if not gc_enabled:
        gc.collect()
        gc.disable()
    try:
        for _ in range(groups):
            order = list(ENCODERS)
            rng.shuffle(order)
            for name, fn in order:
                t0 = time.perf_counter_ns()
                fn(fixture)
                results[name].append((time.perf_counter_ns() - t0) / 1e6)
    finally:
        if not gc_enabled:
            gc.enable()

    return {name: statistics.median(v) for name, v in results.items()}


def auto_sched(batch: int, dim: int) -> tuple[int, int]:
    n_elem = batch * dim
    if n_elem <= 2_000:
        return 10, 60
    if n_elem <= 50_000:
        return 5, 40
    if n_elem <= 500_000:
        return 3, 20
    return 2, 10


# --------------------------------------------------------------------------- #
# Report helpers                                                              #
# --------------------------------------------------------------------------- #

def print_ratio_table(rows):
    header = (
        f"{'batch':<7}{'bytes(MB)':<11}"
        f"{'baseline':<11}{'pr_numpy':<11}{'pyd_json':<11}{'pyd_numpy':<11}"
        f"{'pr/baseline':<13}{'pr/pyd_json':<13}"
    )
    print(header)
    print("-" * len(header))
    for b, nbytes, base, pr, pj, pn in rows:
        print(
            f"{b:<7}{nbytes/1e6:<11.2f}"
            f"{base:<11.3f}{pr:<11.3f}{pj:<11.3f}{pn:<11.3f}"
            f"{base/pr:<13.2f}{pj/pr:<13.2f}"
        )


def print_per_element_table(rows, dim):
    header = (
        f"{'batch':<7}{'baseline':<13}{'pr_numpy':<13}"
        f"{'pyd_json':<13}{'pyd_numpy':<13}"
    )
    print(header)
    print("-" * len(header))
    for b, _nbytes, base, pr, pj, pn in rows:
        n = b * dim
        print(
            f"{b:<7}{base*1e6/n:<13.2f}{pr*1e6/n:<13.2f}"
            f"{pj*1e6/n:<13.2f}{pn*1e6/n:<13.2f}"
        )


def print_throughput_table(rows):
    header = (
        f"{'batch':<7}{'baseline':<13}{'pr_numpy':<13}"
        f"{'pyd_json':<13}{'pyd_numpy':<13}"
    )
    print(header)
    print("-" * len(header))
    for b, nbytes, base, pr, pj, pn in rows:
        mb = nbytes / 1e6
        print(
            f"{b:<7}{mb/(base/1e3):<13.1f}{mb/(pr/1e3):<13.1f}"
            f"{mb/(pj/1e3):<13.1f}{mb/(pn/1e3):<13.1f}"
        )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    dim = 1024
    batches = [1, 8, 32, 128, 512]

    print(f"Fixed dim={dim}. All timings are milliseconds (median of per-group medians).\n")

    rows = []
    for b in batches:
        fx = make_fixture(b, dim)
        warmup, groups = auto_sched(b, dim)
        med = bench_one(fx, warmup, groups, gc_enabled=True)
        nbytes = len(pr_numpy(fx))
        rows.append(
            (b, nbytes, med["baseline_tolist"], med["pr_numpy"],
             med["pydantic_json"], med["pydantic_numpy"])
        )

    print("Main table: end-to-end encode time per path, plus how much faster pr_numpy runs than each alternative.")
    print_ratio_table(rows)

    print("\nPer-element cost normalizes batch away; a flat row indicates strict linear scaling with batch * dim.")
    print_per_element_table(rows, dim)

    print("\nThroughput shows the asymptotic bandwidth each path approaches; pr_numpy should plateau near its steady-state MB/s.")
    print_throughput_table(rows)

    # -----------------------------------------------------------------------
    # GC attribution for the batch=512 uptick
    # -----------------------------------------------------------------------

    print("\nGC attribution experiment. batch=512 pushes pr/baseline above its 1.31x plateau.")
    print("The benchmark is rerun with GC disabled and compared side-by-side;")
    print("if the uptick disappears with GC off it is GC-attributable, otherwise")
    print("the residual points to allocator or cache effects instead.\n")

    gc_batches = [32, 128, 512]
    gc_header = (
        f"{'batch':<7}{'gc':<10}"
        f"{'baseline':<11}{'pr_numpy':<11}{'pr/baseline':<13}"
        f"{'baseline_ns/el':<17}{'pr_numpy_ns/el':<17}"
    )
    print(gc_header)
    print("-" * len(gc_header))

    for b in gc_batches:
        fx = make_fixture(b, dim)
        warmup, groups = auto_sched(b, dim)
        n = b * dim
        for label, enabled in (("on", True), ("off", False)):
            med = bench_one(fx, warmup, groups, gc_enabled=enabled)
            base = med["baseline_tolist"]
            pr = med["pr_numpy"]
            print(
                f"{b:<7}{label:<10}"
                f"{base:<11.3f}{pr:<11.3f}{base/pr:<13.2f}"
                f"{base*1e6/n:<17.2f}{pr*1e6/n:<17.2f}"
            )

    print("\nReading the GC experiment:")
    print("  - If baseline drops toward pr_numpy's ~16.2 ns/el with GC off at batch=512,")
    print("    the uptick is GC-attributable and the PR's advantage widens with batch size.")
    print("  - If baseline stays above ~21 ns/el with GC off, the residual is allocator")
    print("    or cache pressure, and GC is not the dominant cause.")


if __name__ == "__main__":
    main()