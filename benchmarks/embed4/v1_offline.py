import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
import time


def benchmark_vllm(args):
    from vllm import LLM
    from vllm import PoolingRequestOutput, RequestOutput
    from vllm.distributed import cleanup_dist_env_and_memory

    for batchsize in args.batchsize:
        llm = LLM(
            model=args.model,
            max_model_len=args.max_model_len,
            max_num_seqs=batchsize,
            max_num_batched_tokens=batchsize * args.max_model_len,
            enforce_eager=args.enforce_eager,
        )

        for input_len in args.input_len:
            prompt = "你" * (input_len - 2)
            prompts = [prompt for _ in range(args.num_prompts)]

            outputs = llm.embed(prompt, use_tqdm=False)
            assert len(outputs[0].prompt_token_ids) == input_len

            start = time.perf_counter()
            outputs = llm.embed(prompts, use_tqdm=False)
            for prompt, output in zip(prompts, outputs):
                pass
            end = time.perf_counter()

            n_step = llm.n_step
            elapsed_time = end - start
            delay = elapsed_time / n_step

            print(
                f"Batchsize {batchsize}, Throughput: "
                f"{len(prompts) / elapsed_time:.4f} requests/s, "
                f"{len(prompts * input_len) / elapsed_time:.4f} tokens/s, "
                f"Latency {delay * 1000:0.2f} ms, n_step {n_step}"
            )

        del llm
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    from easydict import EasyDict as edict

    args = edict()

    args.model = "BAAI/bge-base-en-v1.5"

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.max_model_len = 512
    args.num_prompts = 10000
    args.batchsize = [1, 2, 4, 8, 16, 32, 64, 128]
    args.input_len = [32, 64, 128, 256, 512]

    from concurrent.futures import ProcessPoolExecutor

    def run(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_vllm, args)
            f.result()

    args.enforce_eager = True
    run(args)

    args.enforce_eager = False
    run(args)
