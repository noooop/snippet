import argparse
import math
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import requests


@dataclass
class Task:
    model: str
    req_id: int
    total: int
    max_tokens: int
    log_interval: int


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def worker(task: Task):
    api_url = "http://localhost:8000/v1/completions"
    headers = {"User-Agent": "Test Client"}
    pload = {
        "model": task.model,
        "prompt": [[100]],
        "max_tokens": task.max_tokens,
        "ignore_eos": True,
        "stream": True,
    }

    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    metrics = []

    n_step = 0
    t = time.perf_counter()
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\n"):
        if not chunk:
            continue

        tt = time.perf_counter()
        n_step += 1
        elapsed_time = tt - t
        metrics.append(elapsed_time)
        t = tt

        if (
            task.req_id == 0
            and task.log_interval > 0
            and (task.log_interval == 1 or n_step % task.log_interval == 1)
        ):
            print(
                f"req-{task.req_id} total-{task.total}", n_step - 1, elapsed_time * 1000
            )
    return task.req_id, metrics


def benchmark(tasks: list[Task]):
    from gevent.pool import Pool
    from gevent import monkey

    monkey.patch_socket()

    p = Pool(len(tasks))
    outputs_list = []
    for outputs in p.imap_unordered(worker, tasks):
        outputs_list.append(outputs)
    return outputs_list


def least_squares(index, metrics):
    b, n = metrics.shape

    X = np.ones((b, n, 2), dtype=np.float32)
    X[:, :, 1] = index
    X = X.reshape((-1, 2))
    y = metrics.reshape(-1)

    (a, b), res, rank, s = np.linalg.lstsq(X, y, rcond=None)
    a = float(a)
    b = float(b)
    res = float(res[0])

    return a * 1000, (1 / b) / 1000, res


def linear_fitting(metrics):
    b, n = metrics.shape
    o = int(n * 0.1)
    metrics = metrics[:, o:-o]
    index = np.arange(o, n - o)
    return least_squares(index, metrics)


def hr(metrics):
    b, n = metrics.shape
    sorted_metrics = np.sort(metrics, axis=0)

    b10 = int(b * 0.1)
    b20 = int(b * 0.2)
    b25 = int(b * 0.25)
    b50 = int(b * 0.5)
    b75 = int(b * 0.75)
    b80 = int(b * 0.8)
    b90 = int(b * 0.9)
    b_max = -1
    bb = [b10, b25, b50, b75, b90, b_max]

    outputs = []
    for b in bb:
        outputs.append(linear_fitting(sorted_metrics[b].reshape((1, -1))))

    outputs.append(linear_fitting(sorted_metrics[b20:b80]))

    return outputs


HEAD = """
==============================================
Decode stress testing for LLM inference server
==============================================
Config:
----------------------------------------------
instance_id: {instance_id}
model: {model}
max_tokens: {max_tokens}
n_clients: {n_clients}
filename: {filename}
n_gevent_clients: {n_gevent_clients}
log_interval: {log_interval}
"""

LR = """
----------------------------------------------
Result
----------------------------------------------
MAIN TPOT:{a:.4f}ms +1ms:{b:.0f} std:{res:.4f}ms
==============================================
"""

HR = """
----------------------------------------------
Result
----------------------------------------------
MAIN TPOT:{a:.4f}ms +1ms:{b:.0f} std:{res:.4f}ms
R10 TPOT:{o[0][0]:.4f}ms +1ms:{o[0][1]:.0f} std:{o[0][2]:.4f}ms
R25 TPOT:{o[1][0]:.4f}ms +1ms:{o[1][1]:.0f} std:{o[1][2]:.4f}mss
R50 TPOT:{o[2][0]:.4f}ms +1ms:{o[2][1]:.0f} std:{o[2][2]:.4f}ms
R75 TPOT:{o[3][0]:.4f}ms +1ms:{o[3][1]:.0f} std:{o[3][2]:.4f}ms
R90 TPOT:{o[3][0]:.4f}ms +1ms:{o[4][1]:.0f} std:{o[4][2]:.4f}ms
Rmax TPOT:{o[5][0]:.4f}ms +1ms:{o[5][1]:.0f} std:{o[5][2]:.4f}ms
R20-80 TPOT:{o[6][0]:.4f}ms +1ms:{o[6][1]:.0f} std:{o[6][2]:.4f}ms
==============================================
"""


def main(
    model: str,
    max_tokens: int,
    n_clients: int,
    filename: str | None = None,
    n_gevent_clients: int = 16,
    available_kv_cache: int | None = None,
    log_interval: int | None = None,
):
    if available_kv_cache is not None:
        max_tokens = min(max_tokens, available_kv_cache // n_clients)

    if log_interval is None:
        l = math.log10(max_tokens)
        log_interval = 10 ** max(int(l) - 2, 0)

    instance_id = f"{time.time_ns()}"

    log = HEAD.format(
        instance_id=instance_id,
        model=model,
        max_tokens=max_tokens,
        n_clients=n_clients,
        filename=filename,
        n_gevent_clients=n_gevent_clients,
        available_kv_cache=available_kv_cache,
        log_interval=log_interval,
    )

    print(log)

    tasks = [
        Task(
            model=model,
            req_id=req_id,
            total=n_clients,
            max_tokens=max_tokens,
            log_interval=log_interval,
        )
        for req_id in range(n_clients)
    ]

    tasks_list = list(chunks(tasks, n_gevent_clients))

    p = Pool(len(tasks_list))
    outputs_list = []
    for outputs in p.imap_unordered(benchmark, tasks_list):
        outputs_list.extend(outputs)

    outputs_list.sort(key=lambda x: x[0])
    metrics = [x[1] for x in outputs_list]
    min_n = min(len(x) for x in metrics)
    metrics = [x[:min_n] for x in metrics]
    metrics = np.array(metrics)

    if n_clients < 10:
        a, b, res = linear_fitting(metrics)
        log2 = LR.format(a=a, b=b, res=res**0.5)
    else:
        a, b, res = linear_fitting(metrics)
        o = hr(metrics)
        log2 = HR.format(a=a, b=b, res=res**0.5, o=o)

    print(log2)

    if filename is not None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        np.save(filename + f".{max_tokens}.{instance_id}", metrics)
        with open(filename + f".{max_tokens}.{instance_id}.txt", "w") as f:
            f.write(log)
            f.write(log2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode stress testing for LLM inference server"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-V3.1",
        help="Model name to test",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=160000,
        help="Maximum tokens to generate per request",
    )

    parser.add_argument(
        "--n-clients", type=int, default=256, help="Number of concurrent clients"
    )

    parser.add_argument(
        "--filename",
        type=str,
        default="./decode",
        help="Filename to save metrics (optional)",
    )

    parser.add_argument(
        "--n-gevent-clients",
        type=int,
        default=16,
        help="Number of concurrent gevent clients per process",
    )

    parser.add_argument(
        "--available-kv-cache",
        type=int,
        default=None,
        help="Available KV cache size (if provided, adjusts max_tokens per client)",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Log interval for request 0 (default: auto-calculated based on max_tokens)",
    )

    args = parser.parse_args()

    main(
        model=args.model,
        max_tokens=args.max_tokens,
        n_clients=args.n_clients,
        filename=args.filename,
        n_gevent_clients=args.n_gevent_clients,
        available_kv_cache=args.available_kv_cache,
        log_interval=args.log_interval,
    )
