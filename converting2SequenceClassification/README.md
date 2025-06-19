


## Qwen3-Reranker

converting:

```commandline
python converting.py --model_name Qwen/Qwen3-Reranker-0.6B --classifier_from_tokens '["no", "yes"]' --method from_2_way_softmax --path ./Qwen3-Reranker-0.6B-seq-cls
```

testing:

```commandline
pytest -s -vvv test_qwen3_reranker.py
```