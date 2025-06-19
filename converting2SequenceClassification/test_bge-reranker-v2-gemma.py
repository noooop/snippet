# ruff: noqa: E501
from typing import Optional

import numpy as np
import pytest
import torch

from .mteb_utils import ModelInfo, VllmRunner, mteb_test_rerank_models
from .score_utils import ping_pong_test_score_models

RERANK_MODELS = [
    ModelInfo(
        original_model_name="BAAI/bge-reranker-v2-gemma",
        converted_model_name="./bge-reranker-v2-gemma-seq-cls",
    )
]

max_model_len = 4096


class BgeGemmaRerankerHfRunner:

    def __init__(self, model_name: str, dtype: str = "auto") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype == "float32":
            torch_dtype = torch.float32
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            assert False, "Unknown dtype: {}".format(dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype).eval()
        self.model.to("cuda")

        self.max_length = max_model_len
        self.yes_loc = self.tokenizer("Yes",
                                      add_special_tokens=False)["input_ids"][0]

    @torch.no_grad()
    def predict(
        self,
        sentences: list[tuple[str, str,
                              Optional[str]]],  # query, corpus, prompt
        return_n_tokens=False,
        **kwargs,
    ):

        def get_inputs(pairs, tokenizer, prompt=None):
            if prompt is None:
                prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
            sep = "\n"
            prompt_inputs = tokenizer(prompt,
                                      return_tensors=None,
                                      add_special_tokens=False)["input_ids"]
            sep_inputs = tokenizer(sep,
                                   return_tensors=None,
                                   add_special_tokens=False)["input_ids"]
            inputs = []
            for query, passage in pairs:
                query_inputs = tokenizer(
                    f"A: {query}",
                    return_tensors=None,
                    add_special_tokens=False,
                    truncation=True,
                )
                passage_inputs = tokenizer(
                    f"B: {passage}",
                    return_tensors=None,
                    add_special_tokens=False,
                    truncation=True,
                )
                item = tokenizer.prepare_for_model(
                    [tokenizer.bos_token_id] + query_inputs["input_ids"],
                    sep_inputs + passage_inputs["input_ids"],
                    truncation="only_second",
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    add_special_tokens=False,
                )
                item["input_ids"] = item[
                    "input_ids"] + sep_inputs + prompt_inputs
                item["attention_mask"] = [1] * len(item["input_ids"])
                inputs.append(item)
            return tokenizer.pad(
                inputs,
                padding=True,
                return_tensors="pt",
            )

        scores = []
        n_tokens = []
        for query, doc, *_ in sentences:
            pairs = [(query, doc)]
            inputs = get_inputs(pairs, self.tokenizer)
            inputs = inputs.to(self.model.device)
            _n_tokens = inputs["input_ids"].shape[1]
            logits = self.model(**inputs, return_dict=True).logits
            _scores = (logits[:, -1,
                              self.yes_loc].view(-1, ).float().sigmoid())
            scores.append(_scores[0].item())
            n_tokens.append(_n_tokens)

        if return_n_tokens:
            return np.array(scores), np.array(n_tokens)
        else:
            return np.array(scores)


class BgeGemmaVllmRunner(VllmRunner):

    def __init__(self, model_name: str, dtype: str = "auto") -> None:
        super().__init__(model_name, dtype, max_model_len=max_model_len)

        self.prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        self.query_template = "A: {query}\n"
        self.document_template = "B: {doc}\n{prompt}"

    def predict(
        self,
        sentences: list[tuple[str, str,
                              Optional[str]]],  # query, corpus, prompt
        return_n_tokens=False,
        **kwargs,
    ):
        querys = [self.query_template.format(query=s[0]) for s in sentences]
        corpus = [
            self.document_template.format(doc=s[1], prompt=self.prompt)
            for s in sentences
        ]
        return super().predict(querys, corpus, return_n_tokens)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(model_info: ModelInfo) -> None:
    mteb_test_rerank_models(BgeGemmaRerankerHfRunner, BgeGemmaVllmRunner,
                            model_info)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_correctness(model_info: ModelInfo) -> None:
    ping_pong_test_score_models(BgeGemmaRerankerHfRunner, BgeGemmaVllmRunner,
                                model_info)
