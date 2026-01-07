import time
from concurrent.futures import ProcessPoolExecutor


def _benchmark_vllm(args):
    from vllm import LLM, TokensPrompt

    llm = LLM(
        model=args.model,
        async_scheduling=args.async_scheduling,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_expert_parallel=True,
        tensor_parallel_size=8,
        trust_remote_code=True,
        enable_prefix_caching=False,
        load_format="dummy"
    )

    llm.n_step = 0

    llm_engine_step = llm.llm_engine.step

    elapsed_time_list = []

    def step():
        start = time.perf_counter()
        out = llm_engine_step()
        end = time.perf_counter()
        elapsed_time = end - start
        elapsed_time_list.append((llm.n_step, elapsed_time))

        if llm.n_step % 1000 == 1:
            print(llm.n_step-1, elapsed_time*1000)

        llm.n_step += 1

        return out

    llm.llm_engine.step = step

    # i1o3
    prompts = TokensPrompt(prompt_token_ids=[100])
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = 3
    sampling_params.ignore_eos = True

    llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    llm.start_profile()


def benchmark_vllm(args):
    _benchmark_vllm(args)

    from vllm.distributed import cleanup_dist_env_and_memory
    cleanup_dist_env_and_memory()

def run(args):
    with ProcessPoolExecutor(1) as executor:
        f = executor.submit(benchmark_vllm, args)
        f.result()

if __name__ == "__main__":
    from easydict import EasyDict as edict

    args = edict()
    args.max_num_seqs = 128
    args.max_num_batched_tokens = 128
    args.async_scheduling = True

    for model in [
        "deepseek-ai/DeepSeek-V3.1/",
        "deepseek-ai/DeepSeek-V3.2/",
    ]:
        args.model = model
        args.tokenizer = args.model
        run(args)

# VLLM_TORCH_PROFILER_DIR=/share/profiles python profile.py