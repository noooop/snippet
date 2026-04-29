import os

os.environ["OMP_NUM_THREADS"] = "1"
import time
from copy import deepcopy


import torch
torch.set_num_threads(1)

from concurrent.futures import ThreadPoolExecutor as PoolExecutor
#from concurrent.futures import ProcessPoolExecutor as PoolExecutor

from transformers import AutoTokenizer

model_name = "deepseek-ai/DeepSeek-V3.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
prompt = "hello " * 100000


def tokenizer_encode(id):
    tokenizer_copyed = deepcopy(tokenizer)
    tokenizer_copyed.encode(prompt)


def test_tokenizer_encode():
    for n in [1, 2, 4, 8, 16]:
        tasks = list(range(128))
        pool = PoolExecutor(max_workers=n)
        start = time.perf_counter()
        for _ in pool.map(tokenizer_encode, tasks):
            pass
        end = time.perf_counter()
        e2e = end - start
        print(f"tokenizer_encode n_workers: {n}, e2e: {e2e}")



test_tokenizer_encode()

"""
from concurrent.futures import ProcessPoolExecutor
tokenizer_encode n_workers: 1, e2e: 12.22839603600005
tokenizer_encode n_workers: 2, e2e: 6.675321902999713
tokenizer_encode n_workers: 4, e2e: 3.770936963999702
tokenizer_encode n_workers: 8, e2e: 2.363701287999902
tokenizer_encode n_workers: 16, e2e: 2.231783658000495

from concurrent.futures import ThreadPoolExecutor
tokenizer_encode n_workers: 1, e2e: 11.007459864000339
tokenizer_encode n_workers: 2, e2e: 5.999560164000286
tokenizer_encode n_workers: 4, e2e: 3.4470128889997795
tokenizer_encode n_workers: 8, e2e: 2.0952387249999447
tokenizer_encode n_workers: 16, e2e: 1.9648324209993007


ThreadPoolExecutor + deepcopy
tokenizer_encode n_workers: 1, e2e: 29.1953536609999
tokenizer_encode n_workers: 2, e2e: 18.48894192500029
tokenizer_encode n_workers: 4, e2e: 18.999746747000245
tokenizer_encode n_workers: 8, e2e: 18.900613595999857
tokenizer_encode n_workers: 16, e2e: 19.041412347000005
"""