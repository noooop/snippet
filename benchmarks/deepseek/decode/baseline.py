import time
from concurrent.futures import ProcessPoolExecutor


def _benchmark_vllm(args):
    from vllm import LLM, TokensPrompt

    llm = LLM(
        model=args.model,
        async_scheduling=args.async_scheduling,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        all2all_backend=args.all2all_backend,
        trust_remote_code=True,
        enable_prefix_caching=False,
        load_format="dummy"
    )

    llm.n_step = 0
    llm_engine_step = llm.llm_engine.step

    def step():
        start = time.perf_counter()
        out = llm_engine_step()
        end = time.perf_counter()
        elapsed_time = end - start
        if llm.n_step % 50 == 1:
            print(llm.n_step-1, elapsed_time*1000)
        llm.n_step += 1
        return out

    llm.llm_engine.step = step

    def run(batched_tokens):
        time.sleep(1)
        print("batched_tokens:", batched_tokens)
        llm.n_step = 0
        prompts = [TokensPrompt(prompt_token_ids=[100])] * batched_tokens

        sampling_params = llm.get_default_sampling_params()
        sampling_params.max_tokens = min(args.max_tokens, 390720 / batched_tokens)
        sampling_params.ignore_eos = True

        llm.generate(prompts, sampling_params)

    for batched_tokens in args.batched_tokens_list:
        run(batched_tokens)


def benchmark_vllm(args):
    _benchmark_vllm(args)

    from vllm.distributed import cleanup_dist_env_and_memory
    cleanup_dist_env_and_memory()

if __name__ == "__main__":
    from easydict import EasyDict as edict

    args = edict()

    args.model = "deepseek-ai/DeepSeek-V3.1/"
    args.tokenizer = args.model
    args.max_num_seqs = 4096
    args.max_num_batched_tokens = 4096
    args.max_tokens = 2101
    args.async_scheduling = True
    args.batched_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    def run(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_vllm, args)
            f.result()

    for all2all_backend in [
            "naive",
            "pplx",
            "deepep_high_throughput",
            "deepep_low_latency",
            "allgather_reducescatter",
            "flashinfer_all2allv",
        ]:
        args.all2all_backend = all2all_backend
        run(args)