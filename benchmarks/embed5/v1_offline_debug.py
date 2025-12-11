import torch

# os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
import time
from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.utils.counter import Counter

import gc


def benchmark_vllm(args):
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batchsize[0],
        max_num_batched_tokens=args.batchsize[0] * args.max_model_len * 2,
        enforce_eager=args.enforce_eager,
    )

    llm.n_step = 0
    llm_engine_step = llm.llm_engine.step

    def step():
        llm.n_step += 1
        start = time.perf_counter()
        o = llm_engine_step()
        end = time.perf_counter()

        print(llm.n_step, f"{(end - start) * 1000:0.2f} ms")

        return o

    llm.llm_engine.step = step

    def warmup(prompts):
        time.sleep(2)
        outputs = llm.embed(prompts[:10], use_tqdm=False)
        assert len(outputs[0].prompt_token_ids) == input_len

    def run(prompts):
        time.sleep(2)

        llm.n_step = 0
        llm.request_counter = Counter()
        start = time.perf_counter()
        outputs = llm.embed(prompts, use_tqdm=False)
        end = time.perf_counter()
        assert len(outputs[-1].prompt_token_ids) == input_len

        n_step = llm.n_step
        elapsed_time = end - start
        delay = elapsed_time / n_step

        print(
            f"Batchsize {batchsize}, Input_len {input_len} Throughput: "
            f"{len(prompts) / elapsed_time:.4f} requests/s, "
            f"{len(prompts * input_len) / elapsed_time:.4f} tokens/s, "
            f"Latency {delay * 1000:0.2f} ms, n_step {n_step}"
        )

    for batchsize in args.batchsize:
        llm.reconfigure(
            max_num_seqs=batchsize,
            max_num_batched_tokens=batchsize * args.max_model_len * 2,
        )

        if batchsize == 128:
            continue

        for input_len in args.input_len:
            if input_len == 512:
                continue

            prompt = "你" * (input_len - 2)
            prompts = [prompt for _ in range(args.num_prompts)]

            warmup(prompts)
            run(prompts)

    del llm, llm_engine_step
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
    args.num_prompts = 3
    args.batchsize = [128, 1]
    args.input_len = [32]

    from concurrent.futures import ProcessPoolExecutor

    def run(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_vllm, args)
            f.result()

    args.enforce_eager = False
    run(args)

    args.batchsize = [1]
    args.input_len = [32]

    run(args)
