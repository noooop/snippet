import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

from copy import deepcopy
import mteb
from typing import Sequence
import numpy as np
import torch
from vllm import LLM
from vllm.entrypoints.openai.protocol import EMBED_DTYPE_TO_TORCH_DTYPE
from vllm.distributed import cleanup_dist_env_and_memory

TASKS = ["STS12"]


class VllmEncoder(mteb.Encoder):
    def __init__(self, model_name):
        super().__init__()

        vllm_extra_kwargs = {}
        if model_name == "Alibaba-NLP/gte-multilingual-base":
            hf_overrides = {"architectures": ["GteNewModel"]}
            vllm_extra_kwargs["hf_overrides"] = hf_overrides

        self.model = LLM(
            model=model_name,
            runner="pooling",
            dtype="float32",
            trust_remote_code=True,
            enforce_eager=True,
            **vllm_extra_kwargs,
        )
        self.embeds_np1 = None
        self.embeds_np2 = None

    def encode(self, sentences: Sequence[str], **kwargs) -> np.ndarray:
        kwargs.pop("task_name", None)
        outputs = self.model.embed(sentences, use_tqdm=False)
        embeds_np = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
        if self.embeds_np1 is None:
            self.embeds_np1 = deepcopy(embeds_np)
        else:
            self.embeds_np2 = deepcopy(embeds_np)
        return embeds_np


class EmbedDtypeEncoder(mteb.Encoder):
    def __init__(self, embeds_np1, embeds_np2, embed_dtype):
        super().__init__()
        torch_dtype = EMBED_DTYPE_TO_TORCH_DTYPE[embed_dtype]

        self.embeds_np1 = (
            torch.from_numpy(embeds_np1).to(torch_dtype).to(torch.float32).numpy()
        )
        self.embeds_np2 = (
            torch.from_numpy(embeds_np2).to(torch_dtype).to(torch.float32).numpy()
        )
        self.flag = True

    def encode(self, sentences: Sequence[str], **kwargs) -> np.ndarray:
        if self.flag:
            self.flag = False
            return self.embeds_np1
        return self.embeds_np2


def run_and_get_main_score(model):
    tasks = mteb.get_tasks(tasks=TASKS)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, verbosity=0, output_folder=None)

    main_score = results[0].scores["test"][0]["main_score"]
    return main_score


def run(model_name):
    print("model_name", model_name)
    encoder = VllmEncoder(model_name)
    main_score = run_and_get_main_score(encoder)
    print("float32", main_score)

    embeds_np1, embeds_np2 = encoder.embeds_np1, encoder.embeds_np2

    del encoder
    cleanup_dist_env_and_memory()
    return embeds_np1, embeds_np2


@torch.inference_mode
def main(model_name):
    embeds_np1, embeds_np2 = run(model_name)

    for embed_dtype in EMBED_DTYPE_TO_TORCH_DTYPE:
        xx = EmbedDtypeEncoder(embeds_np1, embeds_np2, embed_dtype)
        main_score = run_and_get_main_score(xx)
        print(embed_dtype, main_score)


if __name__ == "__main__":
    for model_name in [
        "jinaai/jina-embeddings-v3",
        "BAAI/bge-m3",
        "intfloat/multilingual-e5-base",
        "BAAI/bge-base-en",
        "Alibaba-NLP/gte-multilingual-base",
        "Qwen/Qwen3-Embedding-0.6B",
        "thenlper/gte-large",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "BAAI/bge-code-v1",
        "Alibaba-NLP/gte-modernbert-base",
        "google/embeddinggemma-300m",
        "intfloat/e5-small",
        "nomic-ai/nomic-embed-text-v1",
        "nomic-ai/nomic-embed-text-v2-moe",
        "Snowflake/snowflake-arctic-embed-xs",
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        "TencentBAC/Conan-embedding-v1",
        "Snowflake/snowflake-arctic-embed-m-long",
        "Snowflake/snowflake-arctic-embed-m-v1.5",
    ]:
        main(model_name)
