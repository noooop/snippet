import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time

from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
prompt = "hello " * 10000

n_tasks = 1024

for bs in [1, 2, 4, 8, 16,32]:
    start = time.perf_counter()

    for _ in range(n_tasks // bs):
        tokenizer.encode([prompt] * bs)

    end = time.perf_counter()

    e2e = end - start
    print(f"tokenizer_encode batch size: {bs}, e2e: {e2e} ms")


"""
tokenizer_encode batch size: 1, e2e: 8.075238061002892 ms
tokenizer_encode batch size: 2, e2e: 4.354615904998354 ms
tokenizer_encode batch size: 4, e2e: 2.6432245899995905 ms
tokenizer_encode batch size: 8, e2e: 1.986106992000714 ms
tokenizer_encode batch size: 16, e2e: 1.711663033998775 ms
tokenizer_encode batch size: 32, e2e: 1.7132872410002165 ms


tokenizer_encode batch size: 1, e2e: 7.97959671100034 ms
tokenizer_encode batch size: 2, e2e: 8.035094329999993 ms
tokenizer_encode batch size: 4, e2e: 8.010271022998495 ms
tokenizer_encode batch size: 8, e2e: 8.016586410001764 ms
tokenizer_encode batch size: 16, e2e: 8.042091774001165 ms
tokenizer_encode batch size: 32, e2e: 8.068442815998424 ms

"""