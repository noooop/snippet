# Refer to https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/pooling/qwen3_reranker.py

import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory

queries = ["中国首都在哪"]
documents = ["北京", "西京", "南京", "东京", "面筋"]


def test_reranker_official(model_name: str):
    llm = LLM(
        model=model_name,
        runner="pooling",
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    )

    outputs = llm.score(queries, documents)

    print("-" * 30)
    print(model_name)
    for i, output in enumerate(outputs):
        print(i, output.outputs.score)
    print("-" * 30)

    del llm
    cleanup_dist_env_and_memory()


def test_reranker_seq_cls(model_name: str):
    llm = LLM(model=model_name, runner="pooling")

    outputs = llm.score(queries, documents)

    print("-" * 30)
    print(model_name)
    for i, output in enumerate(outputs):
        print(i, output.outputs.score)
    print("-" * 30)

    del llm
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    for model_name in [
        "Qwen/Qwen3-Reranker-0.6B",
        "Qwen/Qwen3-Reranker-4B",
        "Qwen/Qwen3-Reranker-8B",
    ]:
        test_reranker_official(model_name)

    for model_name in [
        "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
        "tomaarsen/Qwen3-Reranker-4B-seq-cls",
        "tomaarsen/Qwen3-Reranker-8B-seq-cls",
    ]:
        test_reranker_seq_cls(model_name)
