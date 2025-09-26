import os
from pathlib import Path

from easydict import EasyDict as edict


def benchmark_vllm(args):
    from vllm import LLM, TokensPrompt
    from vllm.distributed import cleanup_dist_env_and_memory

    vllm_extra_kwargs = {}
    if args.model == "Alibaba-NLP/gte-multilingual-base":
        hf_overrides = {"architectures": ["GteNewModel"]}
        vllm_extra_kwargs["hf_overrides"] = hf_overrides

    for batchsize in args.batchsize:
        llm = LLM(model=args.model,
                  max_num_seqs=batchsize,
                  enforce_eager=args.enforce_eager,
                  trust_remote_code=True,
                  **vllm_extra_kwargs)

        for input_len in args.input_len:
            prompt = TokensPrompt(prompt_token_ids=[1024] * input_len)
            prompts = [prompt for _ in range(args.num_prompts)]

            for i in range(10):
                llm.embed(prompts, use_tqdm=False)

            llm.start_profile()
            outputs = llm.embed(prompts, use_tqdm=False)
            for prompt, output in zip(prompts, outputs):
                pass
            llm.stop_profile()

    del llm
    cleanup_dist_env_and_memory()


def main(model_name: str):
    profiles_dir = Path("/share/profiles") / model_name.replace("/", "_")
    profiles_dir.mkdir(exist_ok=True)
    os.environ["VLLM_TORCH_PROFILER_DIR"] = str(profiles_dir)

    args = edict()
    args.model = model_name
    args.tokenizer = args.model
    args.max_model_len = None
    batchsize = 64
    args.num_prompts = batchsize * 4
    args.batchsize = [batchsize]
    args.input_len = [32]

    args.enforce_eager = True
    benchmark_vllm(args)


if __name__ == '__main__':
    main('BAAI/bge-m3')
