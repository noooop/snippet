
import os

os.environ["FASTOKENS_BPE_THREADS"] = "0"

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