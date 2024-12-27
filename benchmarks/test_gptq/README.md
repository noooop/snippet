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

# Summarize @Flynn-Zh
> modify main.by and run offline test again, the result is：
> [result.txt](https://github.com/user-attachments/files/18251709/result.txt)

hardware L40*1

# Offline inference

## prefills

- input_len = 8000
- output_len = 16
- num_prompts = 11

using chunked prefill

| batchsize | vllm gptq_marlin | vllm gptq_marlin + gptq | sglang 0.4.0.post2 |
|-----------|------------------|-------------------------|--------------------|
| 1024      | 2.41             | 3.01                    | 2.33               | 
| 512       | 2.47             | 3.43                    | 2.35               | 
| 256       | 2.57             | 4.14                    | 2.49               | 
| 128       | 2.80             | 6.51                    | 2.79               | 
| 64        | 3.83             | 11.97                   | 3.82               | 
| 32        | 7.10             | 13.10                   | 7.21               | 

vllm default scheduler

| method                       |      |
|------------------------------|------|
| gptq_marlin                  | 2.33 |
| gptq                         | 2.35 |
| gptq_marlin  + enforce_eager | 2.49 |
| gptq + enforce_eager         | 2.79 |



## decoding

- input_len = 8000
- output_len = 512
- num_prompts = 11

|                                                      | decoding |
|------------------------------------------------------|----------|
| vllm flash attention                                 | 15.50    |  
| vllm flashinfer                                      | 15.86    |  
| vllm default scheduler + gptq_marlin                 | 16.24    |  
| vllm default scheduler + gptq                        | 15.88    |  
| vllm default scheduler + gptq_marlin + enforce_eager | 16.07    |  
| vllm default scheduler + gptq + enforce_eager        | 16.07    |  
| sglang 0.4.0.post2                                   | 10.41    |  


---

conclusion
1. for prefills: sglang is similar to vllm
2. for decoding: sglang  10.41 vs vllm (under all configurations) 15 ~ 16. Really faster. 

3. for vllm

L40 864GB/s 
4090 1008 GB/s 

So 4090 prefill is slower than L40, but decoding is almost the same. very reasonable

5. I don't know why，but **In the decoding stage, sglang is indeed faster than vllm**