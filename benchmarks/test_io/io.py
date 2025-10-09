# ruff: noqa: F841, E402

import time

SIZE = 1 << 29 # float64 -> 4G
GB = 4.


def test_kvikio(path):
    import cupy
    import kvikio

    a = cupy.random.rand(SIZE)

    start = time.perf_counter()
    f = kvikio.CuFile(path, "w")

    f.write(a)
    f.close()
    end = time.perf_counter()

    elapsed_time = end - start

    print(f"kvikio: ssd to gpu {GB / elapsed_time} GB/s")

    c = cupy.empty_like(a)
    start = time.perf_counter()
    with kvikio.CuFile(path, "r") as f:
        f.read(c)

    end = time.perf_counter()

    elapsed_time = end - start

    print(f"kvikio: gpu to ssd {GB / elapsed_time} GB/s")


def test_numpy(path):
    import numpy as np

    a = np.random.rand(SIZE)

    start = time.perf_counter()
    np.save(path, a)
    end = time.perf_counter()

    elapsed_time = end - start

    print(f"numpy: cpu to ssd {GB / elapsed_time} GB/s")

    start = time.perf_counter()
    b = np.load(path + ".npy")

    end = time.perf_counter()
    elapsed_time = end - start
    print(f"numpy: ssd to cpu {GB / elapsed_time} GB/s")

    start = time.perf_counter()
    b[:] = a[:]
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"numpy: cpu to cpu {GB / elapsed_time} GB/s")


def test_torch_cpu(path):
    import torch

    a = torch.rand(SIZE, dtype=torch.float64)

    start = time.perf_counter()
    torch.save(a, path + ".pt")
    end = time.perf_counter()

    elapsed_time = end - start

    print(f"torch: cpu to ssd {GB / elapsed_time} GB/s")

    start = time.perf_counter()
    b = torch.load(path + ".pt")

    end = time.perf_counter()
    elapsed_time = end - start
    print(f"torch: ssd to cpu {GB / elapsed_time} GB/s")

    start = time.perf_counter()
    b[:] = a[:]
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"torch: cpu to cpu {GB / elapsed_time} GB/s")


def test_torch_gpu(path):
    import gc
    import torch

    a = torch.rand(SIZE, dtype=torch.float64, device="cuda")

    start = time.perf_counter()
    torch.save(a, path + ".pt")
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed_time = end - start

    print(f"torch: gpu to ssd {GB / elapsed_time} GB/s")

    start = time.perf_counter()
    b = torch.load(path + ".pt", weights_only=False)
    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"torch: ssd to gpu {GB / elapsed_time} GB/s")

    start = time.perf_counter()
    b[:] = a[:]
    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"torch: gpu to gpu {GB / elapsed_time} GB/s")

    c = torch.randn(a.shape, dtype=torch.float64, device="cpu")

    start = time.perf_counter()
    c[:] = a
    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"torch: gpu to cpu {GB / elapsed_time} GB/s")

    del a, b

    start = time.perf_counter()
    e = c.to("cuda")
    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"torch: cpu to gpu {GB / elapsed_time} GB/s")


if __name__ == "__main__":
    def process_warp(fn, /, *args, **kwargs):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(1, mp.get_context("spawn")) as executor:
            f = executor.submit(fn, *args, **kwargs)
            return f.result()
        return None


    process_warp(test_kvikio, path="/share/test_kvikio")
    process_warp(test_numpy, path="/share/test_numpy")
    process_warp(test_torch_cpu, path="/share/test_torch_cpu")
    process_warp(test_torch_gpu, path="/share/test_torch_gpu")
