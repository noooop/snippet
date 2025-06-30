


## Qwen3-Reranker

converting:

```commandline
python convert_model_to_seq_cls.py --model_name Qwen/Qwen3-Reranker-0.6B --classifier_from_tokens '["no", "yes"]' --method from_2_way_softmax --path ./Qwen3-Reranker-0.6B-seq-cls
```

testing:

```commandline
pytest -s -vvv test_qwen3_reranker.py
```



## BAAI/bge-reranker-v2-gemma

converting:

```commandline
python convert_model_to_seq_cls.py --model_name BAAI/bge-reranker-v2-gemma --classifier_from_tokens '["Yes"]' --method no_post_processing --path ./bge-reranker-v2-gemma-seq-cls
```

testing:

```commandline
pytest -s -vvv test_bge-reranker-v2-gemma.py
```


## mxbai-rerank-v2

converting:

```commandline
python convert_model_to_seq_cls.py --model_name mixedbread-ai/mxbai-rerank-base-v2 --classifier_from_tokens '["0", "1"]' --method from_2_way_softmax --path ./mxbai-rerank-base-v2-seq-cls
python convert_model_to_seq_cls.py --model_name mixedbread-ai/mxbai-rerank-large-v2 --classifier_from_tokens '["0", "1"]' --method from_2_way_softmax --path ./mxbai-rerank-large-v2-seq-cls
```

testing:

```commandline
pytest -s -vvv test_mxbai_rerank_v2.py
```
