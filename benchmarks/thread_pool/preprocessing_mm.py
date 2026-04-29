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
    a = torch.rand((1024, 1024), device="cpu")
    b = torch.rand((1024, 1024), device="cpu")
    m = torch.mm(a, b)
    s = m.shape


def test_torch():
    for n in [1, 2, 4, 8, 16]:
        tasks = list(range(1024))
        pool = PoolExecutor(max_workers=n)
        start = time.perf_counter()
        for _ in pool.map(_test_torch, tasks):
            pass
        end = time.perf_counter()
        e2e = end - start
        print(f"test_torch n_workers: {n}, e2e: {e2e}")



def _test_numpy(id):
    a = np.random.rand(1024, 1024)
    b = np.random.rand(1024, 1024)
    m = a@b
    s = m.shape


def test_numpy():
    for n in [1, 2, 4, 8, 16]:
        tasks = list(range(1024))
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
test_torch n_workers: 1, e2e: 19.948702640999954
test_torch n_workers: 2, e2e: 9.97323515200003
test_torch n_workers: 4, e2e: 5.014321293999956
test_torch n_workers: 8, e2e: 4.336058832999925
test_torch n_workers: 16, e2e: 4.581200295000031
test_numpy n_workers: 1, e2e: 23.456164654999952
test_numpy n_workers: 2, e2e: 11.749719010000035
test_numpy n_workers: 4, e2e: 7.251445292999961
test_numpy n_workers: 8, e2e: 7.3015678929999694
test_numpy n_workers: 16, e2e: 7.628783422999959

from concurrent.futures import ProcessPoolExecutor
test_torch n_workers: 1, e2e: 19.918051543000047
test_torch n_workers: 2, e2e: 9.977503864000028
test_torch n_workers: 4, e2e: 5.032261247999941
test_torch n_workers: 8, e2e: 2.586272206999979
test_torch n_workers: 16, e2e: 2.1502063479999833
test_numpy n_workers: 1, e2e: 23.895369596000023
test_numpy n_workers: 2, e2e: 12.013010236000014
test_numpy n_workers: 4, e2e: 6.211122473000046
test_numpy n_workers: 8, e2e: 3.2980710719999706
test_numpy n_workers: 16, e2e: 3.0448952600000894
"""