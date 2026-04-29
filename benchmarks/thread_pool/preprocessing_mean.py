import os

os.environ["OMP_NUM_THREADS"] = "1"
import time


import torch
torch.set_num_threads(1)


import cv2
import numpy as np

# from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from concurrent.futures import ProcessPoolExecutor as PoolExecutor



@torch.inference_mode()
def _test_torch(id):
    tensor = torch.rand((256, 256, 3), device="cpu")
    m = tensor.mean()
    s = m.shape


def test_torch():
    for n in [1, 2, 4, 8, 16]:
        tasks = list(range(128 * 128))
        pool = PoolExecutor(max_workers=n)
        start = time.perf_counter()
        for _ in pool.map(_test_torch, tasks):
            pass
        end = time.perf_counter()
        e2e = end - start
        print(f"test_torch n_workers: {n}, e2e: {e2e}")



@torch.inference_mode()
def _test_numpy(id):
    tensor = np.random.rand(256, 256, 3)
    m = tensor.mean()
    s = m.shape


def test_numpy():
    for n in [1, 2, 4, 8, 16]:
        tasks = list(range(128 * 128))
        pool = PoolExecutor(max_workers=n)
        start = time.perf_counter()
        for _ in pool.map(_test_numpy, tasks):
            pass
        end = time.perf_counter()
        e2e = end - start
        print(f"test_numpy n_workers: {n}, e2e: {e2e}")


test_torch()
test_numpy()

"""
from concurrent.futures import ThreadPoolExecutor
test_torch n_workers: 1, e2e: 5.049105689000498
test_torch n_workers: 2, e2e: 4.90301097000156
test_torch n_workers: 4, e2e: 4.941402203001417
test_torch n_workers: 8, e2e: 4.9559328140003345
test_torch n_workers: 16, e2e: 5.080274697000277
test_numpy n_workers: 1, e2e: 10.57447092000075
test_numpy n_workers: 2, e2e: 10.28184652299933
test_numpy n_workers: 4, e2e: 10.56365339200056
test_numpy n_workers: 8, e2e: 10.47741941700042
test_numpy n_workers: 16, e2e: 10.837958570000416


from concurrent.futures import ProcessPoolExecutor
test_torch n_workers: 1, e2e: 5.226045836000026
test_torch n_workers: 2, e2e: 2.680683584999997
test_torch n_workers: 4, e2e: 1.4195794900000465
test_torch n_workers: 8, e2e: 0.8375435970000353
test_torch n_workers: 16, e2e: 0.8535018989999799
test_numpy n_workers: 1, e2e: 10.782182759999955
test_numpy n_workers: 2, e2e: 5.472388928999976
test_numpy n_workers: 4, e2e: 2.809420111999998
test_numpy n_workers: 8, e2e: 1.5256661380000196
test_numpy n_workers: 16, e2e: 1.4067537720000018
"""