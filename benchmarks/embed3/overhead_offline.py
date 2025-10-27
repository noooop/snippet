
# os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import time

from concurrent.futures import ProcessPoolExecutor


def _benchmark(args):
    from vllm import LLM
    from vllm.distributed import cleanup_dist_env_and_memory

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batchsize,
        max_num_batched_tokens=args.batchsize * args.max_model_len,
        enforce_eager=args.enforce_eager,
    )

    prompt = "你" * (args.input_len[0] - 2)

    def worker(prompt):
        start = time.perf_counter()
        outputs = llm.embed([prompt], use_tqdm=False)
        for prompt, output in zip([prompt], outputs):
            pass
        end = time.perf_counter()
        e2e = end - start
        print(e2e * 1000)

    for i in range(10):
        print("=" * 80)
        worker(prompt)

    cleanup_dist_env_and_memory()


def benchmark(args):
    with ProcessPoolExecutor(1) as executor:
        f = executor.submit(_benchmark, args)
        f.result()


def run(args):
    for batchsize in args.batchsizes:
        args.batchsize = batchsize
        benchmark(args)


if __name__ == "__main__":
    from easydict import EasyDict as edict

    args = edict()

    args.model = "BAAI/bge-base-en-v1.5"

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.max_model_len = 512
    args.num_prompts = 10000
    args.batchsizes = [1]
    args.input_len = [32]
    args.n_clients_list = [1]

    args.enforce_eager = False

    run(args)
