import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import argparse
import base64
import io
import json
import random
import signal
import subprocess
import time

import requests
import numpy as np
from concurrent.futures import ProcessPoolExecutor

PORT = 8000
RENDER_URL = f"http://localhost:{PORT}/v1/chat/completions/render"

TEXT_PROMPT = "What's in this image?"


def build_image_data_url(width=512, height=512):
    from PIL import Image

    color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def build_payload(model, image_data_url):
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": TEXT_PROMPT},
                ],
            }
        ],
    }


def run_server(args):
    serve_cmd = [
        "vllm",
        "launch",
        "render",
        args.model,
        "--renderer-num-workers",
        str(args.renderer_num_workers),
        "--api-server-count",
        str(args.api_server_count),
        "--disable-uvicorn-access-log",
        "--port",
        str(PORT),
    ]
    return subprocess.Popen(
        serve_cmd,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def wait(args, proc, timeout=900.0):
    # Use a multimodal request so the first probe also warms up the processor.
    payload = build_payload(args.model, args.image_data_url)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"render server exited early with code {proc.returncode}")
        try:
            r = requests.post(RENDER_URL, json=payload, timeout=30)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("render server did not become ready in time")


def _benchmark(args):
    from gevent.pool import Pool
    from gevent import monkey

    monkey.patch_socket()

    n_tasks = args.num_tasks

    # Pre-generate every request body so image creation never lands on the
    # measured path. Each request uses a distinct random image.
    gen_start = time.perf_counter()
    bodies = [
        json.dumps(build_payload(args.model, build_image_data_url())).encode()
        for _ in range(n_tasks)
    ]
    print(
        f"pre-generated {n_tasks} payloads in "
        f"{time.perf_counter() - gen_start:.3f}s",
        flush=True,
    )

    headers = {
        "Content-Type": "application/json",
        "Connection": "close",
    }

    def worker(i):
        body = bodies[i]
        start = time.perf_counter()
        with requests.Session() as session:
            response = session.post(RENDER_URL, data=body, headers=headers, timeout=120)
            assert response.status_code == 200
            # Fully consume the body; otherwise latency is underestimated.
            len(response.content)
        end = time.perf_counter()
        return end - start

    for n_clients in args.n_clients_list:
        latency_list = []
        p = Pool(n_clients)
        start = time.perf_counter()
        for metrics in p.imap_unordered(worker, range(n_tasks)):
            latency_list.append(metrics)
        end = time.perf_counter()

        elapsed_time = end - start
        latency = float(np.mean(latency_list))
        print(
            f"renderer_num_workers: {args.renderer_num_workers}, "
            f"api_server_count: {args.api_server_count}, "
            f"n_clients: {n_clients}, "
            f"Throughput: {n_tasks / elapsed_time:.4f} requests/s, "
            f"Latency {latency * 1000:0.4f} ms",
            flush=True,
        )


def benchmark(args):
    # Isolate the gevent monkey-patch from the parent process.
    with ProcessPoolExecutor(1) as executor:
        f = executor.submit(_benchmark, args)
        f.result()


def _kill_process_group(proc, sig):
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _shutdown_server(proc, term_timeout=15.0, kill_timeout=10.0):
    _kill_process_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=term_timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    _kill_process_group(proc, signal.SIGKILL)
    try:
        proc.wait(timeout=kill_timeout)
    except subprocess.TimeoutExpired:
        print(f"[warn] launcher pid={proc.pid} still alive after SIGKILL", flush=True)


def run(args):
    # Single random image used only for the readiness probe.
    args.image_data_url = build_image_data_url()

    for num_workers in args.renderer_num_workers_list:
        for api_server_count in args.api_server_count_list:
            print(
                f"renderer_num_workers = {num_workers}, "
                f"api_server_count = {api_server_count}",
                flush=True,
            )
            proc = None
            try:
                args.renderer_num_workers = num_workers
                args.api_server_count = api_server_count
                proc = run_server(args)
                wait(args, proc)
                benchmark(args)
            finally:
                if proc is not None:
                    _shutdown_server(proc)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM multimodal render endpoint sweeping over "
        "--renderer-num-workers and --api-server-count."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Multimodal model passed to vllm launch render.",
    )
    parser.add_argument(
        "--renderer-num-workers-list",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Values of --renderer-num-workers to sweep.",
    )
    parser.add_argument(
        "--api-server-count-list",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Values of --api-server-count to sweep.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=1024,
        help="Total number of requests per (num_workers, api_server_count) run.",
    )
    parser.add_argument(
        "--n-clients-list",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="Values of concurrent client count to sweep.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)