import time

from transformers import AutoTokenizer

model_name = "BAAI/bge-base-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

N = 1000

for input_len in range(128, 8192, 128):
    prompt = "你" * (input_len - 2)

    start = time.perf_counter()
    for i in range(N):
        tokenizer.encode(prompt)

    end = time.perf_counter()
    e2e = end - start
    print(f"input_len: {input_len}, encode: {e2e / N * 1000:0.2f} ms")
