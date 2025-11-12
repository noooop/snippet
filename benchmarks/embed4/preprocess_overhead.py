import os

import torch

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
import time
from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
import gc


def benchmark_vllm(args):
    for batchsize in args.batchsize:
        llm = LLM(
            model=args.model,
            max_model_len=args.max_model_len,
            max_num_seqs=batchsize,
            max_num_batched_tokens=batchsize * args.max_model_len * 2,
            enforce_eager=args.enforce_eager,
        )

        def dummy():
            return 0

        llm.llm_engine.has_unfinished_requests = dummy
        llm.llm_engine.get_num_unfinished_requests = dummy

        def step():
            return []

        llm.llm_engine.step = step

        for input_len in args.input_len:
            prompt = "你" * (input_len - 2)
            prompts = [prompt for _ in range(args.num_prompts)]

            start = time.perf_counter()
            outputs = llm.embed(prompts, use_tqdm=False)
            for prompt, output in zip(prompts, outputs):
                pass
            end = time.perf_counter()

            elapsed_time = end - start

            print(elapsed_time)

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    from easydict import EasyDict as edict

    args = edict()

    args.model = "BAAI/bge-base-en-v1.5"

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.max_model_len = 512
    args.num_prompts = 10000
    args.batchsize = [128]
    args.input_len = [512]

    args.enforce_eager = True
    benchmark_vllm(args)

