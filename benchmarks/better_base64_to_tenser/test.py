import io
import time

import torch
import base64
import binascii


def tenser2base64(data, torch_dtype)->bytes:
    embedding_bytes = (
        data.to(torch_dtype)
        .flatten()
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return base64.b64encode(embedding_bytes)


def base64_to_tenser1(data, torch_dtype):
    return torch.frombuffer(
        binascii.a2b_base64(data), dtype=torch_dtype
    )


def base64_to_tenser2(data: bytes, torch_dtype):
    return torch.frombuffer(
        bytearray(binascii.a2b_base64(data)), dtype=torch_dtype
    )

def base64_to_tenser3(data: bytes, torch_dtype):
    # Decode base64 into a writable BytesIO buffer for zero-copy tensor creation
    bio_in = io.BytesIO(data)
    bio_out = io.BytesIO()
    base64.decode(bio_in, bio_out)

    # Get writable memoryview and create tensor with zero-copy semantics
    mv = bio_out.getbuffer()
    tensor = torch.frombuffer(mv, dtype=torch_dtype)

    # Keep buffer alive to ensure memory validity
    tensor._buffer_owner = bio_out

    return tensor


def base64_to_tenser4(data: bytes, torch_dtype):
    bio_in = io.BytesIO(data)
    size = ((4 * len(data) // 3) + 3) & ~3
    tensor = torch.empty((1, size), dtype=torch_dtype, device="cpu")
    buffer = tensor.view(torch.uint8).numpy().tobytes()
    bio_out = io.BytesIO(buffer)
    base64.decode(bio_in, bio_out)
    return tensor


float32_tenser = torch.rand(1024, device="cpu", dtype=torch.float32)

float32_base64 = tenser2base64(float32_tenser, torch.float32)
float16_base64 = tenser2base64(float32_tenser, torch.float16)
float8_base64 = tenser2base64(float32_tenser, torch.float8_e5m2)

methods = [
    base64_to_tenser1,
    base64_to_tenser2,
    base64_to_tenser3,
    base64_to_tenser4,
]

N = 10000
for method in methods:
    start = time.perf_counter()
    for i in range(N):
        method(float32_base64, torch.float32)
    end = time.perf_counter()

    print(N / (end-start))