import os
import time
from pathlib import Path

from easydict import EasyDict as edict


def benchmark_vllm(args):
    from vllm import LLM, TokensPrompt
    from vllm.distributed import cleanup_dist_env_and_memory

    vllm_extra_kwargs = {}
    if args.model == "Alibaba-NLP/gte-multilingual-base":
        hf_overrides = {"architectures": ["GteNewModel"]}
        vllm_extra_kwargs["hf_overrides"] = hf_overrides

    llm = LLM(
        model=args.model,
        max_num_seqs=args.batchsize,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        trust_remote_code=True,
        enable_prefix_caching=False,
        max_num_batched_tokens=args.batchsize * args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        async_scheduling=args.async_scheduling,
        **vllm_extra_kwargs,
    )

    prompt = TokensPrompt(prompt_token_ids=[1024] * args.input_len)
    prompts = [prompt for _ in range(args.batchsize * args.n_batch)]

    for i in range(2):
        llm.embed(prompts, use_tqdm=False)

    time.sleep(10)

    llm.start_profile()
    outputs = llm.embed(prompts, use_tqdm=False)
    for prompt, output in zip(prompts, outputs):
        pass
    llm.stop_profile()

    time.sleep(10)

    del llm
    cleanup_dist_env_and_memory()


def main(model_name: str, async_scheduling: bool, dtype: str = "auto", tp: int = 1):
    profiles_dir = (
        Path("/share/profiles") / f"{dtype}_tp_{tp}" / model_name.replace("/", "_")
    )
    profiles_dir.mkdir(exist_ok=True, parents=True)
    os.environ["VLLM_TORCH_PROFILER_DIR"] = str(profiles_dir)

    args = edict()
    args.model = model_name
    args.tokenizer = args.model
    args.max_model_len = 512

    args.n_batch = 8
    args.batchsize = 32
    args.input_len = 128
    args.dtype = dtype
    args.enforce_eager = True
    args.tensor_parallel_size = tp
    args.async_scheduling = async_scheduling

    benchmark_vllm(args)

    return profiles_dir


if __name__ == "__main__":
    main("BAAI/bge-base-en-v1.5", async_scheduling=False)
    main("BAAI/bge-base-en-v1.5", async_scheduling=True)
