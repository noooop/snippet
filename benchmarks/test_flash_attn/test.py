import time

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

hidden_size = 8192
head_size = 64
num_heads = hidden_size // 2


@torch.inference_mode()
def test_prefill(num_batched_tokens):
    q = torch.randn(num_batched_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(num_batched_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(num_batched_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    output = torch.randn(num_batched_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")

    seq_lens = [num_batched_tokens//2, num_batched_tokens//2]
    seq_lens_tensor = torch.tensor(seq_lens,
                                   dtype=torch.long,
                                   device="cpu")
    seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                dtype=torch.int32,
                                device="cpu")
    torch.cumsum(seq_lens_tensor,
                 dim=0,
                 dtype=seq_start_loc.dtype,
                 out=seq_start_loc[1:])

    seq_lens_tensor = seq_lens_tensor.cuda()
    seq_start_loc = seq_start_loc.cuda()

    scaling = hidden_size**-0.5

    max_seqlen = max(seq_lens)


    def function_under_test():
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            out=output,
            cu_seqlens_q=seq_start_loc,
            cu_seqlens_k=seq_start_loc,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=scaling,
            causal=True,
        )


    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        function_under_test()
    torch.accelerator.synchronize()
    function_under_test = lambda: g.replay()

    n_iters = 100

    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        function_under_test()
        torch.accelerator.synchronize()
    end = time.perf_counter()
    print("num_batched_tokens:", num_batched_tokens, (end - start) / n_iters)


for i in range(2, 14):
    test_prefill(2**i)



@torch.inference_mode()
def test_decode(num_batched_tokens):
    seqlen_k = 1024

    q = torch.randn(num_batched_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    key_cache = torch.randn(num_batched_tokens * seqlen_k, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    value_cache = torch.randn(num_batched_tokens * seqlen_k, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    output = torch.randn(num_batched_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")


    seqlens_q = [1] * num_batched_tokens
    seqlens_q_tensor = torch.tensor(seqlens_q,
                                   dtype=torch.long,
                                   device="cpu")
    seqlens_q_start_loc = torch.zeros(seqlens_q_tensor.shape[0] + 1,
                                dtype=torch.int32,
                                device="cpu")
    torch.cumsum(seqlens_q_tensor,
                 dim=0,
                 dtype=seqlens_q_start_loc.dtype,
                 out=seqlens_q_start_loc[1:])

    seqlens_q_tensor = seqlens_q_tensor.cuda()
    seqlens_q_start_loc = seqlens_q_start_loc.cuda()


    seqlens_k = [seqlen_k] * num_batched_tokens
    seqlens_k_tensor = torch.tensor(seqlens_k,
                                   dtype=torch.long,
                                   device="cpu")
    seqlens_k_start_loc = torch.zeros(seqlens_k_tensor.shape[0] + 1,
                                dtype=torch.int32,
                                device="cpu")
    torch.cumsum(seqlens_k_tensor,
                 dim=0,
                 dtype=seqlens_k_start_loc.dtype,
                 out=seqlens_k_start_loc[1:])

    seqlens_k_tensor = seqlens_k_tensor.cuda()
    seqlens_k_start_loc = seqlens_k_start_loc.cuda()

    scaling = hidden_size**-0.5

    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)

    def function_under_test():
        flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=seqlens_q_start_loc,
            cu_seqlens_k=seqlens_k_start_loc,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=scaling,
            causal=True,
        )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        function_under_test()
    torch.accelerator.synchronize()
    function_under_test = lambda: g.replay()

    n_iters = 100

    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        function_under_test()
        torch.accelerator.synchronize()
    end = time.perf_counter()
    print("num_batched_tokens:", num_batched_tokens, (end - start) / n_iters)


for i in range(0, 5):
    test_decode(2 ** i)


@torch.inference_mode()
def test_prefill_decode(n_prefill_tokens, n_decode_tokens):
    seqlen_k = 1024

    decode_q = torch.randn(n_decode_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    decode_key_cache = torch.randn(n_decode_tokens * seqlen_k, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    decode_value_cache = torch.randn(n_decode_tokens * seqlen_k, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    decode_output = torch.randn(n_decode_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")

    prefill_q = torch.randn(n_prefill_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    prefill_k = torch.randn(n_prefill_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    prefill_v = torch.randn(n_prefill_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    prefill_output = torch.randn(n_prefill_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")

    q = torch.cat([decode_q, prefill_q])
    k = torch.cat([decode_key_cache, prefill_k])
    v = torch.cat([decode_value_cache, prefill_v])
    output = torch.cat([decode_output, prefill_output])


    seqlens_q = [n_prefill_tokens] + [1] * n_decode_tokens
    seqlens_q_tensor = torch.tensor(seqlens_q,
                                   dtype=torch.long,
                                   device="cpu")
    seqlens_q_start_loc = torch.zeros(seqlens_q_tensor.shape[0] + 1,
                                dtype=torch.int32,
                                device="cpu")
    torch.cumsum(seqlens_q_tensor,
                 dim=0,
                 dtype=seqlens_q_start_loc.dtype,
                 out=seqlens_q_start_loc[1:])

    seqlens_q_tensor = seqlens_q_tensor.cuda()
    seqlens_q_start_loc = seqlens_q_start_loc.cuda()


    seqlens_k = [n_prefill_tokens] + [seqlen_k] * n_decode_tokens
    seqlens_k_tensor = torch.tensor(seqlens_k,
                                   dtype=torch.long,
                                   device="cpu")
    seqlens_k_start_loc = torch.zeros(seqlens_k_tensor.shape[0] + 1,
                                dtype=torch.int32,
                                device="cpu")
    torch.cumsum(seqlens_k_tensor,
                 dim=0,
                 dtype=seqlens_k_start_loc.dtype,
                 out=seqlens_k_start_loc[1:])

    seqlens_k_tensor = seqlens_k_tensor.cuda()
    seqlens_k_start_loc = seqlens_k_start_loc.cuda()

    scaling = hidden_size**-0.5

    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)

    def function_under_test():
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            out=output,
            cu_seqlens_q=seqlens_q_start_loc,
            cu_seqlens_k=seqlens_k_start_loc,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=scaling,
            causal=True,
        )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        function_under_test()
    torch.accelerator.synchronize()
    function_under_test = lambda: g.replay()

    n_iters = 100

    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        function_under_test()
        torch.accelerator.synchronize()
    end = time.perf_counter()
    print("n_prefill_tokens:", n_prefill_tokens, "n_decode_tokens:", n_decode_tokens, (end - start) / n_iters)

for n_prefill_tokens in [128, 256, 512]:
    for n_decode_tokens in [1, 2, 4]:
        test_prefill_decode(n_prefill_tokens, n_decode_tokens)


@torch.inference_mode()
def test_decode_prefill(n_prefill_tokens, n_decode_tokens):
    seqlen_k = 1024

    decode_q = torch.randn(n_decode_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    decode_key_cache = torch.randn(n_decode_tokens * seqlen_k, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    decode_value_cache = torch.randn(n_decode_tokens * seqlen_k, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    decode_output = torch.randn(n_decode_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")

    prefill_q = torch.randn(n_prefill_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    prefill_k = torch.randn(n_prefill_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    prefill_v = torch.randn(n_prefill_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    prefill_output = torch.randn(n_prefill_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")

    q = torch.cat([decode_q, prefill_q])
    k = torch.cat([decode_key_cache, prefill_k])
    v = torch.cat([decode_value_cache, prefill_v])
    output = torch.cat([decode_output, prefill_output])


    seqlens_q = [1] * n_decode_tokens + [n_prefill_tokens]
    seqlens_q_tensor = torch.tensor(seqlens_q,
                                   dtype=torch.long,
                                   device="cpu")
    seqlens_q_start_loc = torch.zeros(seqlens_q_tensor.shape[0] + 1,
                                dtype=torch.int32,
                                device="cpu")
    torch.cumsum(seqlens_q_tensor,
                 dim=0,
                 dtype=seqlens_q_start_loc.dtype,
                 out=seqlens_q_start_loc[1:])

    seqlens_q_tensor = seqlens_q_tensor.cuda()
    seqlens_q_start_loc = seqlens_q_start_loc.cuda()


    seqlens_k = [seqlen_k] * n_decode_tokens + [n_prefill_tokens]
    seqlens_k_tensor = torch.tensor(seqlens_k,
                                   dtype=torch.long,
                                   device="cpu")
    seqlens_k_start_loc = torch.zeros(seqlens_k_tensor.shape[0] + 1,
                                dtype=torch.int32,
                                device="cpu")
    torch.cumsum(seqlens_k_tensor,
                 dim=0,
                 dtype=seqlens_k_start_loc.dtype,
                 out=seqlens_k_start_loc[1:])

    seqlens_k_tensor = seqlens_k_tensor.cuda()
    seqlens_k_start_loc = seqlens_k_start_loc.cuda()

    scaling = hidden_size**-0.5

    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)

    def function_under_test():
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            out=output,
            cu_seqlens_q=seqlens_q_start_loc,
            cu_seqlens_k=seqlens_k_start_loc,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=scaling,
            causal=True,
        )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        function_under_test()
    torch.accelerator.synchronize()
    function_under_test = lambda: g.replay()

    n_iters = 100

    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        function_under_test()
        torch.accelerator.synchronize()
    end = time.perf_counter()
    print("n_prefill_tokens:", n_prefill_tokens, "n_decode_tokens:", n_decode_tokens, (end - start) / n_iters)

for n_prefill_tokens in [128, 256, 512]:
    for n_decode_tokens in [1, 2, 4]:
        test_decode_prefill(n_prefill_tokens, n_decode_tokens)