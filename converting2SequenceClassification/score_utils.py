import math

import pytest

from .mteb_utils import ModelInfo


def ping_pong_test_score_models(hf_runner,
                                vllm_runner,
                                model_info: ModelInfo,
                                dtype="float32"):
    sentences = []

    vllm_model = vllm_runner(model_info.converted_model_name, dtype=dtype)

    max_model_len = vllm_model.model.llm_engine.model_config.max_model_len

    for i in range(0, int(math.log2(max_model_len - 1)) - 1):
        sentences.append(("ping", "pong" * 2**i))

    vllm_scores, vllm_n_tokens = vllm_model.predict(sentences,
                                                    return_n_tokens=True)

    hf_model = hf_runner(model_info.original_model_name, dtype=dtype)

    hf_scores, hf_n_tokens = hf_model.predict(sentences, return_n_tokens=True)

    for i in range(len(sentences)):
        assert vllm_n_tokens[i] == hf_n_tokens[i], (
            f"Test failed at #{i}, vllm: {vllm_n_tokens[i]}, st: {hf_n_tokens[i]}"
        )

        assert float(vllm_scores[i]) == pytest.approx(
            float(hf_scores[i]), rel=0.01
        ), (f"Test failed at #{i}, vllm: {vllm_scores[i]}, st: {hf_scores[i]}")
