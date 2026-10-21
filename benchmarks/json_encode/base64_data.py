#!/usr/bin/env python3
"""Benchmark JSON-encoding paths for vLLM-style GenerateRequest payloads.

Encoders:
  - stdlib json.dumps  (what Starlette JSONResponse uses)
  - Pydantic model_dump_json()
  - orjson.dumps(model_dump(mode="json"))
  - cached TypeAdapter.dump_json()

Fixtures are deterministic synthetic tensors encoded with vLLM's real
multimodal serde (dtype / shape / base64 data).

Usage:
    python bench_json_encode.py
    python bench_json_encode.py --repeats 5 --groups 35 --warmup 5
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import random
import statistics
import time
from typing import Any

import orjson
import torch
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

def tensor_to_b64(t: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(t.dtype).split(".")[-1],
        "shape": list(t.shape),
        "data": base64.b64encode(t.numpy().tobytes()).decode("ascii"),
    }


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt: Any
    multi_modal_data: dict | None = None
    mm_processor_kwargs: dict | None = None
    sampling_params: dict = Field(default_factory=dict)
    model: str | None = None
    request_id: str | None = None


REQ_TA = TypeAdapter(GenerateRequest)
REQ_LIST_TA = TypeAdapter(list[GenerateRequest])


def _tensor(nbytes: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=g)


def make_request(raw_bytes: int, seed: int) -> GenerateRequest:
    return GenerateRequest(
        request_id=f"req-{seed}",
        prompt=f"Describe the image. (seed={seed})",
        multi_modal_data={
            "pixel_values": tensor_to_b64(_tensor(raw_bytes, seed)),
            "image_grid_thw": [[1, 1, 1]],
        },
        sampling_params={"max_tokens": 16, "temperature": 0.0},
        model="Qwen3-VL-7B-Instruct",
    )


def target_raw_bytes(target_mb: float) -> int:
    # base64 inflates by 4/3
    return int(target_mb * 1024 * 1024 * 3 / 4)


def build_cases() -> list[tuple[str, Any]]:
    half = target_raw_bytes(28.9) // 2
    return [
        ("14.4 MB singleton", make_request(target_raw_bytes(14.4), 1)),
        ("57.7 MB singleton", make_request(target_raw_bytes(57.7), 2)),
        ("28.9 MB list of two", [make_request(half, 3), make_request(half, 4)]),
    ]


# --------------------------------------------------------------------------- #
# Encoders                                                                    #
# --------------------------------------------------------------------------- #

def enc_stdlib(reqs) -> bytes:
    content = (
        [r.model_dump(mode="json") for r in reqs]
        if isinstance(reqs, list)
        else reqs.model_dump(mode="json")
    )
    return json.dumps(
        content, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def enc_pydantic(reqs) -> bytes:
    return (
        REQ_LIST_TA.dump_json(reqs)
        if isinstance(reqs, list)
        else reqs.model_dump_json()
    )


def enc_orjson(reqs) -> bytes:
    content = (
        [r.model_dump(mode="json") for r in reqs]
        if isinstance(reqs, list)
        else reqs.model_dump(mode="json")
    )
    return orjson.dumps(content)


def enc_typeadapter(reqs) -> bytes:
    return (
        REQ_LIST_TA.dump_json(reqs)
        if isinstance(reqs, list)
        else REQ_TA.dump_json(reqs)
    )


METHODS = [
    ("stdlib json",        enc_stdlib),
    ("Pydantic",           enc_pydantic),
    ("orjson",             enc_orjson),
    ("TypeAdapter bytes",  enc_typeadapter),
]


# --------------------------------------------------------------------------- #
# Timing                                                                      #
# --------------------------------------------------------------------------- #

def bench_interleaved(fixture, methods, warmup, groups, rng):
    for _, fn in methods:
        for _ in range(warmup):
            fn(fixture)

    out = {name: [] for name, _ in methods}
    for _ in range(groups):
        order = list(methods)
        rng.shuffle(order)
        for name, fn in order:
            t0 = time.perf_counter_ns()
            fn(fixture)
            out[name].append((time.perf_counter_ns() - t0) / 1e6)
    return out


def run_once(cases, args, seed_offset: int = 0):
    rng = random.Random(args.seed + seed_offset)
    results = {m: {} for m, _ in METHODS}
    for case_name, fixture in cases:
        times = bench_interleaved(fixture, METHODS, args.warmup, args.groups, rng)
        for mname, vals in times.items():
            results[mname][case_name] = statistics.median(vals)
        del fixture, times
        gc.collect()
    return results


# --------------------------------------------------------------------------- #
# Correctness                                                                 #
# --------------------------------------------------------------------------- #

def verify(cases) -> None:
    print("Verifying equivalence...")
    for case_name, fixture in cases:
        parsed, sizes = {}, {}
        for mname, fn in METHODS:
            body = fn(fixture)
            sizes[mname] = len(body)
            parsed[mname] = orjson.loads(body)

        names = list(parsed)
        ref = parsed[names[0]]
        for n in names[1:]:
            assert parsed[n] == ref, f"{case_name}: {n} differs from {names[0]}"
        print(f"  {case_name}: OK (~{sizes[names[0]]:,} bytes)")
    print("Verification passed.\n")


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #

def report(runs) -> None:
    cases = list(runs[0][METHODS[0][0]].keys())
    labels = [m for m, _ in METHODS]

    print("=" * 100)
    print("Median of per-run medians (ms)  [lower is better]")
    print("=" * 100)
    print(f"{'Payload':<22}" + "".join(f"{lbl:<22}" for lbl in labels))
    for case in cases:
        line = f"{case:<22}"
        for lbl in labels:
            med = statistics.median(r[lbl][case] for r in runs)
            line += f"{med:>8.3f}  {'':<12}"
        print(line)
    print()

    print("Speedup vs stdlib json (x):")
    for case in cases:
        base = statistics.median(r[labels[0]][case] for r in runs)
        parts = [
            f"{lbl}={base / statistics.median(r[lbl][case] for r in runs):5.2f}x"
            for lbl in labels[1:]
        ]
        print(f"  {case:<22} " + ", ".join(parts))
    print()

    print("Run-to-run spread (min–max ms):")
    for case in cases:
        for lbl in labels:
            vals = [r[lbl][case] for r in runs]
            print(
                f"  {case:<22} {lbl:<18} "
                f"median={statistics.median(vals):7.3f}  "
                f"range=[{min(vals):7.3f}, {max(vals):7.3f}]"
            )
        print()


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--groups", type=int, default=35)
    p.add_argument("--repeats", type=int, default=3,
                   help="Repeats the whole bench with a fresh RNG; reports "
                        "median of per-run medians.")
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--skip-verify", action="store_true")
    args = p.parse_args()

    cases = build_cases()
    if not args.skip_verify:
        verify(cases)

    runs = []
    for i in range(args.repeats):
        print(f"[run {i + 1}/{args.repeats}]  warmup={args.warmup} groups={args.groups}")
        runs.append(run_once(cases, args, seed_offset=i))

    report(runs)


if __name__ == "__main__":
    main()