# gptq speed test

PTAL [#11286](https://github.com/vllm-project/vllm/issues/11286)

## setting
- hardware 4090*1
- [anaconda](https://github.com/noooop/snippet/blob/main/benchmarks/test_gptq/environment_linux.yml)

## prefills

- input_len = 8000
- output_len = 16
- num_prompts = 11

| max_num_batched_tokens | vllm 0.6.4 + gptq_marlin | vllm 0.6.4 + gptq | sglang 0.4.0.post2 |
|------------------------|--------------------------|-------------------|--------------------|
| 1024                   | 4.15                     | 3.67              | 4.16               |
| 512                    | 4.19                     | 4.58              | 4.27               |
| 256                    | 4.34                     | 6.58              | 4.26               |
| 128                    | 4.53                     | 11.52             | 4.50               |
| 64                     | 5.53                     | 21.95             | 5.21               |
| 32                     | 8.59                     | 18.12             | 7.72               |


## decoding

- input_len = 8000
- output_len = 512
- num_prompts = 11

|                              | decoding    |
|------------------------------|-------------|
| vllm 0.6.4 + flash attention | 16.74334423 |
| vllm 0.6.4 + flashinfer      | 16.76823786 |
| sglang 0.4.0.post2           | 16.09388748 |


## conclusion
1. Offline inference, using chunked prefill, vllm and sglang are almost the same speed.
2. marlin (MarlinLinearKernel) work well Almost all max_num_batched_tokens.
3. gptq (ExllamaLinearKernel) Probably works well at >1024, but not much better.
4. There is almost no difference in speed between flashinfer and flash attention.

## Situations not tested
1. vllm default scheduler (not using chunked prefill) oom on my 4090
2. MacheteLinearKernel requires capability 90, current (4090) compute capability is 89
3. Maybe vllm and sglang webserver have different speeds.
4. Maybe vllm and sglang have different output lengths.