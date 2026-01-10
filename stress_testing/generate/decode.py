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
            and n_step % task.log_interval == 1
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


def linear_fitting(metrics):
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

    b, n = metrics.shape

    o = int(n * 0.1)
    metrics = metrics[:, o:-o]
    index = np.arange(o, n - o)
    return least_squares(index, metrics)


def main(
    model: str,
    max_tokens: int,
    n_clients: int,
    filename: str | None = None,
    n_gevent_clients: int = 16,
    log_interval: int = 50,
):
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

    a, b, res = linear_fitting(metrics)
    print(f"TPOT:{a:.4f}ms +1ms:{b:.0f} std:{res**0.5:.4f}ms")

    if filename is not None:
        instance_id = f"{time.time_ns()}"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        np.save(filename + f".{instance_id}", metrics)


if __name__ == "__main__":
    main(
        model="deepseek-ai/DeepSeek-V3.1/",
        max_tokens=160000,
        n_clients=1,
        filename="./mla/dp8",
        log_interval=100,
    )
