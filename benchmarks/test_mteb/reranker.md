


| model_name                                        | float32 st | float16 diff            | bfloat16 diff           | float32 diff |
|---------------------------------------------------|------------|-------------------------|-------------------------|--------------|
| bm25s retriever only                              | 0.32102    | -                       | -                       | -            |
| random reranker                                   | 0.25306    | -                       | -                       | -            |
| cross-encoder/ms-marco-TinyBERT-L-2-v2            | 0.3288     | 4.0000000000040004e-05  | 0.00017000000000000348  | 0.0          |
| cross-encoder/ms-marco-MiniLM-L-6-v2              | 0.33437    | 0.0                     | 0.00010999999999999899  | 0.0          |
| tomaarsen/Qwen3-Reranker-0.6B-seq-cls wo/template | 0.25782    | 3.999999999998449e-05   | -0.0007499999999999729  | 0.0          |
| tomaarsen/Qwen3-Reranker-0.6B-seq-cls w/template  | 0.33699    | 0.00040000000000001146  | -0.0012600000000000389  | 0.0          |
| jinaai/jina-reranker-v2-base-multilingual         | 0.33623    | 0.0011400000000000299   | 0.0010100000000000109   | 0.0          |
| BAAI/bge-reranker-base                            | 0.32379    | -1.0000000000010001e-05 | 0.0                     | 0.0          |
| BAAI/bge-reranker-large                           | 0.33321    | -7.00000000000145e-05   | -0.00031000000000003247 | 0.0          |
| BAAI/bge-reranker-v2-m3                           | 0.32803    | 7.00000000000145e-05    | -0.00046000000000001595 | 0.0          |


```commandline
 python benchmarks/test_mteb/test_mteb_rerank.py model_name
```
