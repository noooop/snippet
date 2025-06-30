# ruff: noqa: E501
# Adapted from https://github.com/mixedbread-ai/mxbai-rerank/blob/main/mxbai_rerank/mxbai_rerank_v2.py
from typing import ClassVar, Dict, List, Optional

import numpy as np
import pytest
import torch

from .mteb_utils import ModelInfo, VllmRunner, mteb_test_rerank_models
from .score_utils import ping_pong_test_score_models

RERANK_MODELS = [
    ModelInfo(
        original_model_name="mixedbread-ai/mxbai-rerank-base-v2",
        converted_model_name="./mxbai-rerank-base-v2-seq-cls",
    ),
    ModelInfo(
        original_model_name="mixedbread-ai/mxbai-rerank-large-v2",
        converted_model_name="./mxbai-rerank-large-v2-seq-cls",
    )
]


class MxbaiRerankV2HfRunner:
    sep = "\n"
    instruction_prompt = "instruction: {instruction}"
    query_prompt = "query: {query}"
    doc_prompt = "document: {document}"
    task_prompt = "You are a search relevance expert who evaluates how well documents match search queries. For each query-document pair, carefully analyze the semantic relationship between them, then provide your binary relevance judgment (0 for not relevant, 1 for relevant).\nRelevance:"  # noqa: E501
    chat_template: ClassVar[Dict[str, str]] = {
        "prefix":
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",  # noqa: E501
        "suffix": "<|im_end|>\n<|im_start|>assistant\n",
    }

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

        self.device = "cuda"
        self.model.to(self.device)
        self.max_length = 40000
        self.model_max_length = 80000
        self.prepare_predefined_inputs()

    def prepare_predefined_inputs(self):
        """Pre-tokenize static prompts and templates for efficiency."""

        def get_input_ids(x):
            return self.tokenizer(x,
                                  return_tensors=None,
                                  add_special_tokens=False)["input_ids"]

        self.yes_loc = get_input_ids("1")[0]
        self.no_loc = get_input_ids("0")[0]

        self.task_prompt_inputs = get_input_ids(self.task_prompt)
        self.sep_inputs = get_input_ids(self.sep)
        self.chat_template_prefix_inputs = get_input_ids(
            self.chat_template["prefix"])
        self.chat_template_suffix_inputs = get_input_ids(
            self.chat_template["suffix"])

        # Calculate total length of static tokens
        self.predefined_length = (len(self.chat_template_prefix_inputs) +
                                  len(self.task_prompt_inputs) +
                                  len(self.chat_template_suffix_inputs) +
                                  len(self.sep_inputs))

    def concat_input_ids(self, input_ids: List[int]) -> List[int]:
        """Concatenate input IDs with prompt templates."""
        return (self.chat_template_prefix_inputs + input_ids +
                self.sep_inputs + self.task_prompt_inputs +
                self.chat_template_suffix_inputs)

    def prepare_inputs(self,
                       queries: List[str],
                       documents: List[str],
                       *,
                       instruction: Optional[str] = None) -> dict:
        """Prepare model inputs from query-document pairs.

        Args:
            queries: List of queries
            documents: List of documents
            instruction: Optional instruction

        Returns:
            dict: Tokenized and padded inputs
        """
        inputs = []
        instruction_prompt = self.instruction_prompt.format(
            instruction=instruction) if instruction else None

        for query, document in zip(queries, documents):
            query_prompt = self.query_prompt.format(query=query)
            if instruction_prompt:
                query_prompt = "".join(
                    [instruction_prompt, self.sep, query_prompt])

            # Tokenize query with length limit
            query_inputs = self.tokenizer(
                query_prompt,
                return_tensors=None,
                add_special_tokens=False,
                truncation=True,
            )

            available_tokens = self.model_max_length - len(
                query_inputs["input_ids"]) - self.predefined_length
            doc_maxlen = min(available_tokens, self.max_length)

            # Tokenize document
            document_inputs = self.tokenizer(
                self.doc_prompt.format(document=document),
                return_tensors=None,
                add_special_tokens=False,
                truncation=True,
            )

            # Combine query and document
            item = self.tokenizer.prepare_for_model(
                query_inputs["input_ids"],
                self.sep_inputs + document_inputs["input_ids"],
                truncation="only_second",
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )

            # Add prompt templates
            item["input_ids"] = self.concat_input_ids(item["input_ids"])
            item["attention_mask"] = [1] * len(item["input_ids"])
            inputs.append(item)

        return self.tokenizer.pad(
            inputs,
            padding="longest",
            return_tensors="pt",
        )

    @torch.no_grad()
    def predict(
        self,
        sentences: list[tuple[str, str,
                              Optional[str]]],  # query, corpus, prompt
        return_n_tokens=False,
        **kwargs,
    ):
        scores = []
        n_tokens = []
        for query, doc, *_ in sentences:
            inputs = self.prepare_inputs(queries=[query], documents=[doc])
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            _n_tokens = len(inputs["input_ids"][0])

            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            yes_logits = outputs.logits[:, -1, self.yes_loc]
            no_logits = outputs.logits[:, -1, self.no_loc]
            logits = yes_logits - no_logits

            n_tokens.append(_n_tokens)
            scores.append(logits.float().sigmoid().item())

        if return_n_tokens:
            return np.array(scores), np.array(n_tokens)
        else:
            return np.array(scores)


class MxbaiRerankV2VllmRunner(VllmRunner):

    def __init__(self, model_name: str, dtype: str = "auto") -> None:
        super().__init__(model_name, dtype)

        self.prefix = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n"

        self.query_template = "{prefix}query: {query}\n"
        self.document_template = "document: {doc}\n{instruction}{suffix}"

        self.instruction = "You are a search relevance expert who evaluates how well documents match search queries. For each query-document pair, carefully analyze the semantic relationship between them, then provide your binary relevance judgment (0 for not relevant, 1 for relevant).\nRelevance:"

    def predict(
        self,
        sentences: list[tuple[str, str,
                              Optional[str]]],  # query, corpus, prompt
        return_n_tokens=False,
        **kwargs,
    ):
        querys = [
            self.query_template.format(prefix=self.prefix, query=s[0])
            for s in sentences
        ]
        corpus = [
            self.document_template.format(
                doc=s[1],
                suffix=self.suffix,
                instruction=self.instruction,
            ) for s in sentences
        ]
        return super().predict(querys, corpus, return_n_tokens)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(model_info: ModelInfo) -> None:
    mteb_test_rerank_models(MxbaiRerankV2HfRunner,
                            MxbaiRerankV2VllmRunner,
                            model_info)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_correctness(model_info: ModelInfo) -> None:
    ping_pong_test_score_models(MxbaiRerankV2HfRunner, MxbaiRerankV2VllmRunner,
                                model_info)
