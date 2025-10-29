import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import subprocess
import time

import requests
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def run_server(args):
    serve_cmd = [
        "vllm",
        "serve",
        args.model,
        "--max_num_seqs",
        str(args.batchsize),
        "--max_num_batched_tokens",
        str(args.batchsize * args.max_model_len),
        "--api-server-count",
        str(args.api_server_count),
        "--disable-uvicorn-access-log",
    ]

    if args.enforce_eager:
        serve_cmd.append("--enforce_eager")

    process = subprocess.Popen(serve_cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    return process


def wait():
    while True:
        try:
            api_url = "http://localhost:8000/v1/embeddings"
            prompt = {
                "model": args.model,
                "input": "vLLM is great!",
            }
            response = requests.post(api_url, json=prompt)
            if response.status_code == 200:
                break
        except Exception:
            pass

        time.sleep(5)


def _benchmark(args):
    from gevent.pool import Pool
    from gevent import monkey

    monkey.patch_socket()

    for input_len in args.input_len:
        prompt = "你" * (input_len - 2)
        prompts = [prompt for _ in range(args.num_prompts)]

        def worker(prompt):
            api_url = "http://localhost:8000/v1/embeddings"
            prompt = {
                "model": args.model,
                "input": prompt,
            }
            start = time.perf_counter()
            response = requests.post(api_url, json=prompt)
            assert response.status_code == 200
            len(response.content)
            end = time.perf_counter()
            e2e = end - start
            return e2e

        for n_clients in args.n_clients_list:
            metrics_list = []
            p = Pool(n_clients)
            start = time.perf_counter()
            for metrics in p.imap_unordered(worker, prompts):
                metrics_list.append(metrics)
            end = time.perf_counter()
            elapsed_time = end - start
            e2e = np.mean(metrics_list)

            print(
                f"n_clients {n_clients}, Batchsize {args.batchsize}, Throughput: "
                f"{len(prompts) / elapsed_time:.4f} requests/s, "
                f"{len(prompts * input_len) / elapsed_time:.4f} tokens/s, "
                f"Latency {e2e * 1000:0.2f} ms"
            )


def benchmark(args):
    with ProcessPoolExecutor(1) as executor:
        f = executor.submit(_benchmark, args)
        f.result()


def run(args):
    for batchsize in args.batchsizes:
        try:
            args.batchsize = batchsize
            proc = run_server(args)
            wait()

            benchmark(args)
        finally:
            proc.terminate()
            out, err = proc.communicate()
            print("=" * 80)
            print("out")
            print(out)
            print("err")
            print(err)
            print("=" * 80)


if __name__ == "__main__":
    from easydict import EasyDict as edict

    args = edict()

    args.model = "BAAI/bge-base-en-v1.5"

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.max_model_len = 512
    args.num_prompts = 10000
    args.batchsizes = [128]
    args.input_len = [512]
    args.n_clients_list = [1, 2, 4, 8, 16, 32]
    args.api_server_count = 1
    args.enforce_eager = False

    run(args)
