import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
# lmcache config
os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "20.0"


def get_requests(args):
    from benchmarks.kv_cache.utils import TokenSampler

    token_sampler = TokenSampler(args.tokenizer)

    prefix_len = int(args.hit_rate * args.input_len)
    unique_len = args.input_len - prefix_len
    prefix_token_ids = token_sampler.random_sample(prefix_len)
    num_prompts = args.num_prompts + args.num_warmup

    requests = []
    for _ in range(num_prompts):
        unique_part_token_ids = token_sampler.random_sample(unique_len)

        prompt_token_ids = prefix_token_ids + unique_part_token_ids
        requests.append(prompt_token_ids)

    return requests


def benchmark(args):
    import time
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.config import KVTransferConfig

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        ignore_eos=True,
        max_tokens=1,
    )

    lmcache_connector = "LMCacheConnectorV1"
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        kv_transfer_config=ktc,
    )

    requests = get_requests(args)

    prompts = []
    for prompt_token_ids in requests:
        inputs = TokensPrompt(prompt_token_ids=prompt_token_ids)
        prompts.append(inputs)

    warmup_prompts = prompts[: args.num_warmup]
    prompts = prompts[args.num_warmup :]

    # warmup
    outputs = llm.generate(warmup_prompts, sampling_params, use_tqdm=False)
    for output in outputs:
        pass

    ## benchmark
    for n in range(args.get("repeat", 1)):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for output in outputs:
            pass
        end = time.perf_counter()

        elapsed_time = end - start
        print(
            f"Hit rate: {args.hit_rate}, Throughput: {len(prompts) / elapsed_time:.4f} requests/s"
        )

    del llm
    cleanup_dist_env_and_memory()


def process_warp(fn, /, *args, **kwargs):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(1, mp.get_context("spawn")) as executor:
        f = executor.submit(fn, *args, **kwargs)
        return f.result()


def exception_handling(fn, /, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        import traceback

        traceback.print_exc()


def process_warp_with_exc(fn, /, *args, **kwargs):
    return process_warp(exception_handling, fn, *args, **kwargs)


if __name__ == "__main__":
    from easydict import EasyDict as edict

    args = edict()

    args.input_len = 1000
    args.output_len = 1
    args.num_prompts = 100
    args.num_warmup = 3

    args.model = "Qwen/Qwen3-4B"
    args.max_model_len = 2000
    args.tokenizer = args.model
    args.gpu_memory_utilization = 0.9

    def test_vary_hit_rate(args):
        for hit_rate in [0.1 * x for x in range(0, 11)] + [0.99]:
            args.hit_rate = hit_rate
            process_warp_with_exc(benchmark, args)

    def test_vary_enable_prefix_caching(args):
        for enable_prefix_caching in [False, True]:
            args.enable_prefix_caching = enable_prefix_caching

            test_vary_hit_rate(args)

    test_vary_enable_prefix_caching(args)
