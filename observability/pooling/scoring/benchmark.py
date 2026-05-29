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

    process = subprocess.Popen(
        serve_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    return process


get_input_len = lambda output_len: (output_len - 4) // 2
api_url = "http://localhost:8000/score"


def wait():
    while True:
        try:
            queries = "What is the capital of France?"
            documents = [
                "The capital of Brazil is Brasilia.",
            ]

            payload = {"model": args.model, "queries": queries, "documents": documents}
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                break
        except Exception:
            pass

        time.sleep(5)


def _benchmark(args):
    from gevent.pool import Pool
    from gevent import monkey

    monkey.patch_socket()

    def worker(pair):
        payload = {"model": args.model, "queries": pair[0], "documents": pair[1]}
        start = time.perf_counter()
        response = requests.post(api_url, json=payload)
        assert response.status_code == 200
        end = time.perf_counter()
        e2e = end - start
        return e2e

    for input_len in args.input_len:
        _input_len = get_input_len(input_len)

        queries = "你" * _input_len
        documents = "你" * _input_len

        pair = (queries, documents)

        prompts = [pair for _ in range(args.num_prompts)]

        for i in range(10):
            worker(pair)

        for n_clients in args.n_clients_list:
            metrics_list = []
            p = Pool(n_clients)
            start = time.perf_counter()
            for metrics in p.imap_unordered(worker, prompts):
                metrics_list.append(metrics)
            end = time.perf_counter()
            elapsed_time = end - start
            e2e = np.mean(metrics_list)
            std = np.std(metrics_list)

            print(
                f"n_clients {n_clients}, Batchsize {args.batchsize}, Throughput: "
                f"{len(prompts) / elapsed_time:.4f} requests/s, "
                f"{len(prompts * input_len) / elapsed_time:.4f} tokens/s, "
                f"Latency {e2e * 1000:0.2f} ms, std {std * 1000:0.2f} ms"
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

    args.model = "BAAI/bge-reranker-v2-m3"

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.max_model_len = 512
    args.num_prompts = 10000
    args.batchsizes = [128]
    args.input_len = [512]
    args.n_clients_list = [1, 2, 4, 8, 16, 32, 64, 128]
    args.api_server_count = 1
    args.enforce_eager = False

    run(args)
