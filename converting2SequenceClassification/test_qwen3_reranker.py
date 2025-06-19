# ruff: noqa: E501
from typing import Optional

import numpy as np
import pytest
import torch

from .mteb_utils import ModelInfo, VllmRunner, mteb_test_rerank_models
from .score_utils import ping_pong_test_score_models

RERANK_MODELS = [
    ModelInfo(original_model_name="Qwen/Qwen3-Reranker-0.6B",
              converted_model_name="./Qwen3-Reranker-0.6B-seq-cls")
]


class Qwen3RerankerHfRunner:

    def __init__(self, model_name: str, dtype: str = "auto") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype == "float32":
            torch_dtype = torch.float32
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            assert False, "Unknown dtype: {}".format(dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype).eval()
        self.model.to("cuda")

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 40000

        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix,
                                                   add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix,
                                                   add_special_tokens=False)
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'

    def predict(
            self,
            sentences: list[tuple[str, str,
                                  Optional[str]]],  # query, corpus, prompt
            return_n_tokens=False,
            **kwargs):

        def format_instruction(instruction, query, doc):
            if instruction is None:
                instruction = 'Given a web search query, retrieve relevant passages that answer the query'
            output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc)
            return output

        def process_inputs(pairs):
            inputs = self.tokenizer(pairs,
                                    padding=False,
                                    truncation='longest_first',
                                    return_attention_mask=False,
                                    max_length=self.max_length -
                                    len(self.prefix_tokens) -
                                    len(self.suffix_tokens))
            for i, ele in enumerate(inputs['input_ids']):
                inputs['input_ids'][
                    i] = self.prefix_tokens + ele + self.suffix_tokens
            inputs = self.tokenizer.pad(inputs,
                                        padding=True,
                                        return_tensors="pt",
                                        max_length=self.max_length)
            for key in inputs:
                inputs[key] = inputs[key].to(self.model.device)
            return inputs

        @torch.no_grad()
        def compute_logits(inputs):
            n_tokens = len(inputs['input_ids'][0])
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp()
            return scores, n_tokens

        scores = []
        n_tokens = []
        for query, doc, *_ in sentences:
            pairs = [format_instruction(self.task, query, doc)]
            inputs = process_inputs(pairs)
            _scores, _n_tokens = compute_logits(inputs)
            scores.append(_scores[0].item())
            n_tokens.append(_n_tokens)

        if return_n_tokens:
            return np.array(scores), np.array(n_tokens)
        else:
            return np.array(scores)


class Qwen3RerankerVllmRunner(VllmRunner):

    def __init__(self, model_name: str, dtype: str = "auto") -> None:
        super().__init__(model_name, dtype)

        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        self.document_template = "<Document>: {doc}{suffix}"

        self.instruction = "Given a web search query, retrieve relevant passages that answer the query"

    def predict(
            self,
            sentences: list[tuple[str, str,
                                  Optional[str]]],  # query, corpus, prompt
            return_n_tokens=False,
            **kwargs):

        querys = [
            self.query_template.format(prefix=self.prefix,
                                       instruction=self.instruction,
                                       query=s[0]) for s in sentences
        ]
        corpus = [
            self.document_template.format(doc=s[1], suffix=self.suffix)
            for s in sentences
        ]
        return super().predict(querys, corpus, return_n_tokens)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(model_info: ModelInfo) -> None:
    mteb_test_rerank_models(Qwen3RerankerHfRunner, Qwen3RerankerVllmRunner,
                            model_info)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_correctness(model_info: ModelInfo) -> None:
    ping_pong_test_score_models(Qwen3RerankerHfRunner, Qwen3RerankerVllmRunner,
                                model_info)
