import dataclasses
import os
import random
import time
from typing import List, Union

from easydict import EasyDict as edict


class TokenSampler:

    def __init__(self, tokenizer):
        if isinstance(tokenizer, str):
            from vllm.transformers_utils.tokenizer import get_tokenizer
            tokenizer = get_tokenizer(tokenizer)

        self.tokenizer = tokenizer
        vocab = tokenizer.get_vocab()
        vocab = {
            k: v
            for k, v in vocab.items() if k not in tokenizer.all_special_ids
        }
        vocab = list(vocab.values())

        self.vocab = vocab

    def random_sample(self,
                      length: int,
                      decode: bool = False) -> Union[List[int], str]:
        prompt_token_ids = random.choices(self.vocab, k=length)

        if not decode:
            return prompt_token_ids

        prompt = self.tokenizer.decode(prompt_token_ids)
        return prompt

    @classmethod
    def get_random_sample(cls, args):
        token_sampler = cls(
            args.tokenizer if hasattr(args, "tokenizer") else args.model)
        input_len = args.input_len
        num_prompts = args.num_prompts

        prompts = []
        for i in range(num_prompts):
            prompt = token_sampler.random_sample(input_len, decode=True)
            prompts.append(prompt)
        return prompts


def benchmark_vllm(args):
    os.environ["VLLM_NO_USAGE_STATS"] = "True"

    if hasattr(args, "environs"):
        for k, v in args.environs.items():
            os.environ[k] = v

    import vllm
    from vllm import LLM, EngineArgs, SamplingParams

    print("vllm version: ", vllm.__version__)

    engine_args = EngineArgs(
        model=args.model,
        seed=args.seed,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        kv_cache_dtype=args.kv_cache_dtype,
        device=args.device,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
    )

    llm = LLM(**dataclasses.asdict(engine_args))

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    for i, prompt in enumerate(args.prompts):
        start_time = time.perf_counter()
        llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
        end_time = time.perf_counter()
        latency = end_time - start_time
        print(i, latency)


if __name__ == '__main__':
    args = edict()

    args.input_len = 8000
    args.output_len = 16
    args.num_prompts = 1
    args.max_model_len = 9000

    args.seed = 0
    args.model = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    args.dtype = 'auto'
    args.kv_cache_dtype = "auto"
    args.device = "cuda"
    args.prompts = TokenSampler.get_random_sample(args)

    args.enable_chunked_prefill = True
    args.max_num_seqs = 4
    args.gpu_memory_utilization = 0.95
    args.enforce_eager = False

    args.quantization = "gptq_marlin"
    args.max_num_batched_tokens = 1024
    benchmark_vllm(args)
