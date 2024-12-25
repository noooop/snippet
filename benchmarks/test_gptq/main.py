import dataclasses
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Union

import numpy as np
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

    latencies = []

    for i, prompt in enumerate(args.prompts):
        start_time = time.perf_counter()
        llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
        end_time = time.perf_counter()
        latency = end_time - start_time
        print(i, latency)
        latencies.append(latency)

    print("avg_latency", np.mean(latencies[1:]))


def benchmark_sgl(args):
    random.seed(args.seed)

    if hasattr(args, "environs"):
        for k, v in args.environs.items():
            os.environ[k] = v

    import sglang as sgl

    print("sglang version:", sgl.__version__)

    llm = sgl.Engine(model_path=args.model,
                     dtype=args.dtype,
                     kv_cache_dtype=args.kv_cache_dtype,
                     device=args.device,
                     context_length=args.max_model_len,
                     mem_fraction_static=args.gpu_memory_utilization,
                     chunked_prefill_size=args.max_num_batched_tokens,
                     max_running_requests=args.max_num_seqs,
                     disable_radix_cache=True,
                     enforce_eager=args.enforce_eager)

    sampling_params = dict(
        n=1,
        temperature=0.8,
        top_p=0.95,
        ignore_eos=True,
        max_new_tokens=args.output_len,
    )

    latencies = []

    for i, prompt in enumerate(args.prompts):
        start_time = time.perf_counter()

        prompts = [prompt]
        outputs = llm.generate(prompts, sampling_params)

        for prompt, output in zip(prompts, outputs):
            pass

        end_time = time.perf_counter()
        latency = end_time - start_time
        print(i, latency)
        latencies.append(latency)

    print("avg_latency", np.mean(latencies[1:]))

    llm.shutdown()


def run_vllm(args):
    try:
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_vllm, args)
            f.result()
    except Exception:
        import traceback
        traceback.print_exc()


def run_sgl(args):
    try:
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_sgl, args)
            f.result()
    except Exception:
        import traceback
        traceback.print_exc()


def vllm_for_prefills():
    args = edict()

    args.input_len = 8000
    args.output_len = 16
    args.num_prompts = 11
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
    print("=" * 80)
    print("use MarlinLinearKernel")
    for max_num_batched_tokens in [1024, 512, 256, 128, 64, 32]:
        args.environs = {
            "VLLM_DISABLED_KERNELS":
            "GPTQMarlinLinearMethod,MacheteLinearKernel"
        }
        args.max_num_batched_tokens = max_num_batched_tokens
        run_vllm(args)

    print("=" * 80)
    print("Using ExllamaLinearKernel")
    for max_num_batched_tokens in [1024, 512, 256, 128, 64, 32]:
        args.environs = {
            "VLLM_DISABLED_KERNELS": "MarlinLinearKernel,MacheteLinearKernel"
        }
        args.max_num_batched_tokens = max_num_batched_tokens
        run_vllm(args)

    print("=" * 80)
    print("Using MacheteLinearKernel")
    # MacheteLinearKernel requires capability 90, current (4090) compute capability is 89
    for max_num_batched_tokens in [1024, 512, 256, 128, 64, 32]:
        args.environs = {
            "VLLM_DISABLED_KERNELS": "MarlinLinearKernel,ExllamaLinearKernel"
        }
        args.max_num_batched_tokens = max_num_batched_tokens
        run_vllm(args)

    args.quantization = "gptq"

    print("gptq")
    print("=" * 80)
    for max_num_batched_tokens in [1024, 512, 256, 128, 64, 32]:
        args.max_num_batched_tokens = max_num_batched_tokens
        run_vllm(args)


def vllm_for_decoding():
    args = edict()

    args.input_len = 8000
    args.output_len = 512
    args.num_prompts = 11
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
    print("=" * 80)
    print("use MarlinLinearKernel")
    args.environs = {
        "VLLM_DISABLED_KERNELS": "GPTQMarlinLinearMethod,MacheteLinearKernel"
    }
    args.max_num_batched_tokens = 1024
    run_vllm(args)

    print("=" * 80)
    print("use MarlinLinearKernel+flashinfer")
    args.environs = {
        "VLLM_DISABLED_KERNELS": "GPTQMarlinLinearMethod,MacheteLinearKernel",
        "VLLM_ATTENTION_BACKEND": "FLASHINFER"
    }
    args.max_num_batched_tokens = 1024
    run_vllm(args)


def sgl_for_prefills():
    args = edict()

    args.input_len = 8000
    args.output_len = 16
    args.num_prompts = 11
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

    print("=" * 80)
    for max_num_batched_tokens in [1024, 512, 256, 128, 64, 32]:
        args.max_num_batched_tokens = max_num_batched_tokens
        run_sgl(args)


def sgl_for_decoding():
    args = edict()

    args.input_len = 8000
    args.output_len = 512
    args.num_prompts = 11
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

    print("=" * 80)
    args.max_num_batched_tokens = 1024
    run_sgl(args)


def vllm_default_for_prefills():
    args = edict()

    args.input_len = 8000
    args.output_len = 16
    args.num_prompts = 11
    args.max_model_len = 9000

    args.seed = 0
    args.model = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    args.dtype = 'auto'
    args.kv_cache_dtype = "auto"
    args.device = "cuda"
    args.prompts = TokenSampler.get_random_sample(args)

    args.max_num_seqs = 4
    args.gpu_memory_utilization = 0.95

    args.enable_chunked_prefill = False
    args.max_num_batched_tokens = None

    for enforce_eager in [False, True]:
        args.enforce_eager = enforce_eager

        args.quantization = "gptq_marlin"
        run_vllm(args)

        args.quantization = "gptq"
        run_vllm(args)


def vllm_default_for_decoding():
    args = edict()

    args.input_len = 8000
    args.output_len = 512
    args.num_prompts = 11
    args.max_model_len = 9000

    args.seed = 0
    args.model = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    args.dtype = 'auto'
    args.kv_cache_dtype = "auto"
    args.device = "cuda"
    args.prompts = TokenSampler.get_random_sample(args)

    args.enable_chunked_prefill = False
    args.max_num_seqs = 4
    args.gpu_memory_utilization = 0.95

    args.quantization = "gptq_marlin"
    args.max_num_batched_tokens = None

    for enforce_eager in [False, True]:
        args.enforce_eager = enforce_eager

        run_vllm(args)

        args.environs = {"VLLM_ATTENTION_BACKEND": "FLASHINFER"}
        run_vllm(args)


if __name__ == '__main__':
    print("For prefills")
    print("=" * 80)
    vllm_for_prefills()
    sgl_for_prefills()
    vllm_default_for_prefills()

    print("For decoding")
    print("=" * 80)
    vllm_for_decoding()
    sgl_for_decoding()
    vllm_default_for_decoding()
