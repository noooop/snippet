"""
vLLM vs HuggingFace Embedding Benchmark

This script benchmarks the inference performance of two embedding backends:
- HuggingFace
- vLLM

For vLLM, additional sub‑tests are run for all combinations of:
- async_scheduling (False / True)
- VLLM_USE_V2_MODEL_RUNNER environment variable ("0" / "1")

Each combination is submitted as a separate task to the ProcessPoolExecutor,
ensuring a single configuration is tested per process invocation.
"""

import argparse
import gc
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def get_system_info():
    import vllm

    info = []
    info.append(f"PyTorch {torch.__version__}")
    info.append(f"vLLM {vllm.__version__}")
    if torch.cuda.is_available():
        info.append(f"CUDA {torch.version.cuda}")
        info.append(f"GPU {torch.cuda.get_device_name(0)}")
    return ", ".join(info)


def check_dtype_support(dtype):
    """Check if the given dtype is supported on the current GPU."""
    if dtype == "bfloat16":
        if not torch.cuda.is_available():
            return False
        # bfloat16 requires compute capability >= 8.0
        cap = torch.cuda.get_device_capability()
        if cap[0] < 8:
            print("Warning: bfloat16 not supported on this GPU.")
            return False
    return True


def benchmark_hf(args):
    """Benchmark HuggingFace/SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer

    results = {}

    for dtype in args.dtypes:
        if not check_dtype_support(dtype):
            print(f"Skipping dtype {dtype} for HF due to lack of hardware support.")
            continue

        print(f"\n=== Benchmarking HF with dtype: {dtype} ===")
        model = SentenceTransformer(
            args.model,
            model_kwargs={"torch_dtype": getattr(torch, dtype)},
            trust_remote_code=True,
        )

        dtype_results = []

        with torch.no_grad():
            for batchsize in args.batchsize:
                batch_results = {}
                for input_len in args.input_len:
                    prompt = "hello " * (input_len // 2 - 1)
                    requests = [prompt for _ in range(args.num_prompts)]

                    inputs_batch = model.tokenizer(prompt)
                    assert len(inputs_batch["input_ids"]) == input_len

                    # Warmup
                    model.encode(requests[:10], batch_size=batchsize)
                    torch.cuda.synchronize()

                    start = time.perf_counter()

                    n_step = 0
                    for i in range(0, len(requests), batchsize):
                        batch = requests[i : i + batchsize]
                        model.encode(batch, batch_size=batchsize)
                        n_step += 1

                    torch.cuda.synchronize()
                    end = time.perf_counter()

                    elapsed_time = end - start
                    delay = elapsed_time / n_step * 1000
                    throughput_req = len(requests) / elapsed_time
                    throughput_tokens = (len(requests) * input_len) / elapsed_time

                    batch_results[input_len] = {
                        "throughput_req": throughput_req,
                        "throughput_tokens": throughput_tokens,
                        "latency": delay,
                        "n_step": n_step,
                        "elapsed_time": elapsed_time,
                    }

                    print(
                        f"  Batchsize {batchsize}, Input_len {input_len}: "
                        f"Throughput: {throughput_req:.4f} req/s, "
                        f"{throughput_tokens:.4f} tokens/s, "
                        f"Latency (batch): {delay:.2f} ms"
                    )

                dtype_results.append(batch_results)

            results[dtype] = dtype_results

        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


def benchmark_vllm_single_config(args, config):
    """
    Benchmark vLLM for a single configuration.
    config is a dict with keys: dtype, async_scheduling (bool), v2_runner ("0" or "1").
    Returns a dict mapping config_name -> list of per-batchsize results.
    """
    dtype = config["dtype"]
    async_scheduling = config["async_scheduling"]
    v2_runner = config["v2_runner"]
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = str(int(v2_runner))

    from vllm import LLM
    from vllm.distributed import cleanup_dist_env_and_memory

    if not check_dtype_support(dtype):
        print(f"Skipping dtype {dtype} for vLLM due to lack of hardware support.")
        return {}

    config_name = f"vLLM_{dtype}_async{int(async_scheduling)}_v2{v2_runner}"
    print(f"\n=== Benchmarking {config_name} ===")
    dtype_results = []

    for batchsize in args.batchsize:
        batch_results = {}
        try:
            # Save original env var to restore later
            orig_v2 = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = v2_runner

            llm = LLM(
                model=args.model,
                dtype=dtype,
                max_model_len=args.max_model_len,
                max_num_seqs=batchsize,
                max_num_batched_tokens=batchsize * args.max_model_len * 2,
                disable_log_stats=False,
                async_scheduling=async_scheduling,
            )

            for input_len in args.input_len:
                prompt = "hello " * (input_len // 2 - 1)
                prompts = [prompt for _ in range(args.num_prompts)]

                # Warmup
                time.sleep(2)
                outputs = llm.embed(prompts[:10], use_tqdm=False)
                assert len(outputs[0].prompt_token_ids) == input_len

                # Benchmark run
                time.sleep(2)
                start = time.perf_counter()
                outputs = llm.embed(prompts, use_tqdm=False)
                end = time.perf_counter()
                assert len(outputs[-1].prompt_token_ids) == input_len

                elapsed_time = end - start
                throughput_req = len(prompts) / elapsed_time
                throughput_tokens = (len(prompts) * input_len) / elapsed_time

                latency = (
                    float(
                        np.mean(
                            [
                                o.metrics.first_token_ts - o.metrics.scheduled_ts
                                for o in outputs
                            ]
                        )
                    )
                    * 1000
                )

                batch_results[input_len] = {
                    "throughput_req": throughput_req,
                    "throughput_tokens": throughput_tokens,
                    "latency": latency,
                    "elapsed_time": elapsed_time,
                }

                print(
                    f"  Batchsize {batchsize}, Input_len {input_len}: "
                    f"Throughput: {throughput_req:.4f} req/s, "
                    f"{throughput_tokens:.4f} tokens/s, "
                    f"Latency (batch): {latency:.2f} ms"
                )

            dtype_results.append(batch_results)

        except Exception as e:
            print(f"  Error with {config_name}, batchsize {batchsize}: {e}")
            for input_len in args.input_len:
                batch_results[input_len] = None
            dtype_results.append(batch_results)

        finally:
            # Clean up LLM instance
            try:
                del llm
            except NameError:
                pass
            gc.collect()
            cleanup_dist_env_and_memory()

    return {config_name: dtype_results}


def print_perf_table(batchsizes, perf_data, input_len):
    """Print a Markdown table showing throughput and latency for each configuration."""
    sys_info = get_system_info()
    print(f"\n**System:** {sys_info}\n")
    print(
        f"### Throughput (tokens/s) and Latency (ms/batch) — Input Length = {input_len}"
    )

    config_names = list(perf_data.keys())
    # Header: Batch Size, then each config: Throughput, Latency
    header = ["Batch Size"]
    for name in config_names:
        header.append(f"{name} Tok/s")
        header.append(f"{name} Latency(ms)")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join([" --- " for _ in header]) + "|")

    for i, bs in enumerate(batchsizes):
        row = [str(bs)]
        for name in config_names:
            entry = perf_data[name][i]
            if entry is not None:
                row.append(f"{entry['throughput_tokens']:.2f}")
                row.append(f"{entry['latency']:.2f}")
            else:
                row.append("N/A")
                row.append("N/A")
        print("| " + " | ".join(row) + " |")


def plot_latency_vs_throughput(batchsizes, perf_data, input_len, output_file=None):
    """Plot Throughput (tokens/s) vs Latency (ms/batch) with scatter + lines, Y log scale."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available.")
        return

    sys_info = get_system_info()
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (name, entries) in enumerate(perf_data.items()):
        x_vals, y_vals = [], []
        for entry in entries:
            if entry is not None:
                x_vals.append(entry["throughput_tokens"])
                y_vals.append(entry["latency"])
        if not x_vals:
            continue

        ax.plot(
            x_vals,
            y_vals,
            color=colors[idx % len(colors)],
            marker="o",
            markersize=6,
            linestyle="-",
            linewidth=2,
            label=name,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Throughput (tokens/s)")
    ax.set_ylabel("Latency (ms per batch) [log scale]")
    ax.set_title(f"Throughput vs Latency\n({sys_info})")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark comparison between vLLM and HuggingFace/Transformers."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-m3",
        help="Model name or path. Default: BAAI/bge-m3",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to benchmark. Default: 1000",
    )
    parser.add_argument(
        "--batchsize",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="Batch sizes to test. Default: 1 2 4 8 16 32 64 128",
    )
    parser.add_argument(
        "--input-len",
        nargs="+",
        type=int,
        default=[512],
        help="Input lengths to test. Default: 512",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="Maximum model length. Default: 1024",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float16", "bfloat16", "float32"],
        default=["float16"],
        help="Data types to test. Default: float16",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        choices=["hf", "vllm", "both"],
        default=["both"],
        help="Which benchmarks to run. Default: both",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Do not display or save the plot."
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="benchmark_dtype.png",
        help="Save the plot to the given file path.",
    )
    # --metric argument removed; table and plot now always use throughput_tokens and latency
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Benchmarks will run on CPU (may be slow).")

    # Prepare tasks for ProcessPoolExecutor
    # Each task is a tuple: (backend_name, function, args, config_dict_or_None)
    tasks = []

    if "hf" in args.benchmark or "both" in args.benchmark:
        try:
            import sentence_transformers

            tasks.append(("HF", benchmark_hf, args, None))
        except ImportError:
            print(
                "Error: sentence-transformers is required for HF benchmark. Install with: pip install sentence-transformers"
            )
            return

    if "vllm" in args.benchmark or "both" in args.benchmark:
        try:
            from mteb.models.vllm_wrapper import VllmEncoderWrapper
        except ImportError:
            print(
                "Error: mteb is required for vLLM benchmark. Install with: pip install mteb"
            )
            return

        # Generate all combinations of async_scheduling and V2 runner
        async_options = [False, True]
        v2_options = ["0", "1"]
        for dtype in args.dtypes:
            for async_sched in async_options:
                for v2 in v2_options:
                    config = {
                        "dtype": dtype,
                        "async_scheduling": async_sched,
                        "v2_runner": v2,
                    }
                    # Use config_name as task name for logging
                    config_name = f"vLLM_{dtype}_async{int(async_sched)}_v2{v2}"
                    tasks.append(
                        (config_name, benchmark_vllm_single_config, args, config)
                    )

    if not tasks:
        print("No valid backends selected. Exiting.")
        return

    print("=" * 60)

    hf_results = None
    vllm_results = {}  # Will aggregate all vLLM sub-results

    with ProcessPoolExecutor(
        max_workers=1, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = {}
        for name, func, func_args, config in tasks:
            if config is None:
                future = executor.submit(func, func_args)
            else:
                future = executor.submit(func, func_args, config)
            futures[future] = name

        for future in futures:
            name = futures[future]
            try:
                result = future.result()
                if name == "HF":
                    hf_results = result
                else:
                    # vLLM results: merge the single config dict into vllm_results
                    vllm_results.update(result)
            except Exception as e:
                print(f"Benchmark {name} failed with error: {e}")

    # Build unified performance data structure for all configs
    perf_data = {}
    target_input_len = args.input_len[
        0
    ]  # only first input length used for tables/plots

    if hf_results:
        for dtype, dtype_results in hf_results.items():
            key = f"HF_{dtype}"
            perf_data[key] = []
            for i, bs in enumerate(args.batchsize):
                if (
                    i < len(dtype_results)
                    and dtype_results[i]
                    and target_input_len in dtype_results[i]
                ):
                    metrics = dtype_results[i][target_input_len]
                    perf_data[key].append(
                        {
                            "throughput_tokens": metrics["throughput_tokens"],
                            "latency": metrics["latency"],
                        }
                    )
                else:
                    perf_data[key].append(None)

    if vllm_results:
        for config_name, dtype_results in vllm_results.items():
            perf_data[config_name] = []
            for i, bs in enumerate(args.batchsize):
                if (
                    i < len(dtype_results)
                    and dtype_results[i]
                    and target_input_len in dtype_results[i]
                ):
                    metrics = dtype_results[i][target_input_len]
                    perf_data[config_name].append(
                        {
                            "throughput_tokens": metrics["throughput_tokens"],
                            "latency": metrics["latency"],
                        }
                    )
                else:
                    perf_data[config_name].append(None)

    if perf_data:
        print_perf_table(args.batchsize, perf_data, target_input_len)

        if not args.no_plot:
            plot_latency_vs_throughput(
                args.batchsize, perf_data, target_input_len, output_file=args.save_plot
            )

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
