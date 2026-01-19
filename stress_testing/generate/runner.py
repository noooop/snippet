import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path

import requests
import re


@dataclass
class EngineArgs:
    model: str
    load_format: str
    max_num_seqs: int
    max_num_batched_tokens: int | None
    all2all_backend: str
    async_scheduling: bool
    tp: int
    dp: int
    enable_expert_parallel: bool
    moe_simulation: bool

    log_file: str
    output_len: int
    n_clients_list: list[int]


def run_server(args: EngineArgs):
    if args.moe_simulation:
        os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = "uniform_random"

    serve_cmd = [
        "vllm",
        "serve",
        args.model,
        "--load_format",
        str(args.load_format),
        "--max_num_seqs",
        str(args.max_num_seqs),
        "--async_scheduling" if args.async_scheduling else "--no-async_scheduling",
        "--tensor_parallel_size",
        str(args.tp),
        "--data_parallel_size",
        str(args.dp),
        "--no-enable_prefix_caching",
    ]

    if args.max_num_batched_tokens is not None:
        serve_cmd.extend(["--max_num_batched_tokens", str(args.max_num_batched_tokens)])

    serve_cmd.extend(
        [
            "--enable_expert_parallel"
            if args.enable_expert_parallel
            else "--no-enable_expert_parallel",
        ]
    )

    if args.enable_expert_parallel:
        serve_cmd.extend(
            [
                "--all2all_backend",
                str(args.all2all_backend),
            ]
        )

    process = subprocess.Popen(serve_cmd, stdout=subprocess.PIPE)

    return process


def wait():
    while True:
        try:
            models_url = "http://localhost:8000/v1/models"
            response = requests.get(models_url)
            if response.status_code == 200:
                break
        except Exception:
            pass

        time.sleep(5)


def get_kv_cache_size(out):
    model = re.findall(r"Model loading took (.+) GiB memory", out)
    kv_cache = re.findall(r"Available KV cache memory: (.+) GiB", out)
    kv_cache_size = re.findall(r"GPU KV cache size: (.+) tokens", out)

    print(f"Model loading took {model[0]} GiB memory")
    print(f"Available KV cache memory: {kv_cache[0]} GiB")
    print(f"GPU KV cache size: {kv_cache_size[0]} tokens")

    return int(kv_cache_size[0].replace(",", ""))


def benchmark(
    args: EngineArgs, n_clients: int, available_kv_cache: int, instance_id: str
):
    serve_cmd = [
        "python3",
        "-u",
        "decode.py",
        "--model",
        args.model,
        "--filename",
        str(args.log_file),
        "--output-len",
        str(args.output_len),
        "--n-clients",
        str(n_clients),
        "--available-kv-cache",
        str(available_kv_cache),
        "--instance_id",
        instance_id,
    ]

    process = subprocess.Popen(serve_cmd, stdout=subprocess.PIPE)

    with open(f"{args.log_file}.{n_clients}.log", "w") as f:
        for line in process.stdout:
            line = line.decode("utf-8")
            f.write(line)
            f.flush()


def read_log(proc):
    for line in proc.stdout:
        line = line.decode("utf-8")
        print(line, end="")


def run(args):
    instance_id = f"{time.time_ns()}"

    if args.log_file is not None:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    for n_clients in args.n_clients_list:
        with open(f"{args.log_file}.{n_clients}.log", "w") as f:
            f.write("")

        with open(f"{args.log_file}.{n_clients}.{instance_id}.txt", "w") as f:
            f.write("")

    proc = run_server(args)
    try:
        out = ""
        for line in proc.stdout:
            line = line.decode("utf-8")
            print(line, end="")
            if "Starting vLLM API server" in line:
                break
            out += line
        pool = ThreadPoolExecutor(max_workers=1)
        pool.submit(read_log, proc)

        wait()

        kv_cache_size = get_kv_cache_size(out)
        available_kv_cache = kv_cache_size * args.dp

        for n_clients in args.n_clients_list:
            benchmark(
                args,
                n_clients=n_clients,
                available_kv_cache=available_kv_cache,
                instance_id=instance_id,
            )

    finally:
        proc.terminate()
        pool.shutdown()


def parse_engine_args():
    parser = argparse.ArgumentParser(description="Engine configuration arguments")

    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-V3.1",
        help="Model name/path to load",
    )

    parser.add_argument(
        "--load-format",
        type=str,
        default="auto",
        choices=["auto", "dummy"],
        help="Format to load the model weights",
    )

    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences to process simultaneously",
    )

    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to batch together (None for auto)",
    )

    parser.add_argument(
        "--all2all-backend",
        type=str,
        default="pplx",
        choices=[
            "naive",
            "pplx",
            "deepep_high_throughput",
            "deepep_low_latency",
            "allgather_reducescatter",
            "flashinfer_all2allv",
        ],
        help="Backend for all-to-all communication",
    )

    parser.add_argument(
        "--async-scheduling",
        action="store_true",
        default=True,
        help="Enable asynchronous scheduling",
    )

    # Parallelism arguments
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")

    parser.add_argument("--dp", type=int, default=1, help="Data parallelism degree")

    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        default=False,
        help="Enable expert parallelism for MoE models",
    )

    parser.add_argument(
        "--moe-simulation",
        action="store_true",
        default=True,
        help="Enable MoE simulation mode",
    )

    # Logging and testing arguments
    parser.add_argument(
        "--log-file", type=str, default="./engine.log", help="File to write engine logs"
    )

    parser.add_argument(
        "--output-len",
        type=int,
        default=160000,
        help="Maximum tokens to generate per request",
    )

    parser.add_argument(
        "--n-clients-list",
        type=lambda s: [int(item) for item in s.split(",")],
        default=[32, 64, 128, 256, 384, 512, 640, 768, 896, 1024],
        help="List of client counts to test (comma-separated)",
    )

    args = parser.parse_args()

    # Create EngineArgs instance
    engine_args = EngineArgs(
        model=args.model,
        load_format=args.load_format,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        all2all_backend=args.all2all_backend,
        async_scheduling=args.async_scheduling,
        tp=args.tp,
        dp=args.dp,
        enable_expert_parallel=args.enable_expert_parallel,
        moe_simulation=args.moe_simulation,
        log_file=args.log_file,
        output_len=args.output_len,
        n_clients_list=args.n_clients_list,
    )

    return engine_args


if __name__ == "__main__":
    args = parse_engine_args()
    run(args)
