import os
import sys

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["VLLM_SERVER_DEV_MODE"] = "1"

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
        str(args.batchsize[0]),
        "--max_num_batched_tokens",
        str(args.batchsize[0] * args.max_model_len * 2),
        "--api-server-count",
        str(args.api_server_count),
        "--disable-uvicorn-access-log",
    ]

    process = subprocess.Popen(
        serve_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

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


def _benchmark(args, batchsize):
    from gevent.pool import Pool
    from gevent import monkey

    monkey.patch_socket()

    def reconfigure():
        api_url = "http://localhost:8000/reconfigure"
        response = requests.post(
            api_url,
            json={
                "max_num_seqs": batchsize,
                "max_num_batched_tokens": batchsize * args.max_model_len * 2,
            },
        )
        assert response.status_code == 200

    reconfigure()
    time.sleep(2)

    for input_len in args.input_len:
        prompt = "你" * (input_len - 2)
        prompts = [prompt for _ in range(args.num_prompts)]

        def worker(prompt):
            api_url = "http://localhost:8000/v1/embeddings"
            prompt = {
                "model": args.model,
                "input": prompt,
                "encoding_format": "bytes",
                "embed_dtype": "float16",
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
                f"n_clients {n_clients}, Batchsize {batchsize}, input_len {input_len}, Throughput: "
                f"{len(prompts) / elapsed_time:.4f} requests/s, "
                f"{len(prompts * input_len) / elapsed_time:.4f} tokens/s, "
                f"Latency {e2e * 1000:0.2f} ms"
            )


def benchmark(*args):
    with ProcessPoolExecutor(1) as executor:
        f = executor.submit(_benchmark, *args)
        f.result()


def run(args):
    proc = run_server(args)
    wait()

    for batchsize in args.batchsize:
        try:
            benchmark(args, batchsize)
        finally:
            proc.terminate()
            out, err = proc.communicate()
            print("=" * 80, file=sys.stderr)
            print("out", file=sys.stderr)
            print(out, file=sys.stderr)
            print("err", file=sys.stderr)
            print(err, file=sys.stderr)
            print("=" * 80, file=sys.stderr)


if __name__ == "__main__":
    from easydict import EasyDict as edict

    args = edict()

    args.model = "BAAI/bge-base-en-v1.5"

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.max_model_len = 512
    args.num_prompts = 10000
    args.batchsize = [128, 64, 32, 16, 8, 4, 2, 1]
    args.input_len = [512, 256, 128, 64, 32]
    args.n_clients_list = [128, 64, 32, 16, 8, 4, 2, 1]
    args.api_server_count = 4

    run(args)
