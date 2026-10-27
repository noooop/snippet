
import time

from vllm.tokenizers import get_tokenizer

model_name = "Qwen/Qwen3-0.6B"
tokenizer = get_tokenizer(model_name)
prompt = "hello " * 10000
n_tasks = 1024

for i in range(3):
    tokens = tokenizer.encode([prompt])
    print(len(tokens[0]))


start = time.perf_counter()
for i in range(n_tasks):
    tokenizer.encode([prompt])
end = time.perf_counter()
elapsed_time = end - start

print(f"Throughput: {n_tasks / elapsed_time:.4f} requests/s, Latency {elapsed_time * 1000 / n_tasks :0.4f} ms")


def worker(_):
    tokenizer.encode([prompt])


for n in [1, 2, 4, 8, 16]:
    from concurrent.futures import ThreadPoolExecutor as PoolExecutor
    pool = PoolExecutor(max_workers=n)
    start = time.perf_counter()
    for _ in pool.map(worker, range(n_tasks)):
        pass
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"ThreadPoolExecutor n_workers: {n}, Throughput: {n_tasks / elapsed_time:.4f} requests/s")


for n in [1, 2, 4, 8, 16]:
    from concurrent.futures import ProcessPoolExecutor as PoolExecutor
    pool = PoolExecutor(max_workers=n)
    start = time.perf_counter()
    for _ in pool.map(worker, range(n_tasks)):
        pass
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"ProcessPoolExecutor n_workers: {n}, Throughput: {n_tasks / elapsed_time:.4f} requests/s")