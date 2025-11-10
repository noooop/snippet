import os
import time
from pathlib import Path

from easydict import EasyDict as edict


def benchmark_vllm(args):
    from vllm import LLM, TokensPrompt
    from vllm.distributed import cleanup_dist_env_and_memory

    llm = LLM(
        model=args.model,
        max_num_seqs=args.batchsize,
        async_scheduling=args.async_scheduling,
        trust_remote_code=True,
        enable_prefix_caching=False,
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 10

    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = 10


    time.sleep(10)

    llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
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

    args.batchsize = 4
    args.async_scheduling = async_scheduling

    benchmark_vllm(args)

    return profiles_dir


if __name__ == "__main__":
    main("Qwen/Qwen3-0.6B", async_scheduling=False)
    main("Qwen/Qwen3-0.6B", async_scheduling=True)
