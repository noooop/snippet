# SPDX-License-Identifier: Apache-2.0
import gc
import os
from collections.abc import Sequence
from functools import partial

import mteb
import numpy as np
import torch
from mteb.encoder_interface import PromptType

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"


TASKS = ["STS12"]
rng = np.random.default_rng(seed=42)


class VllmEncoder(mteb.Encoder):

    def __init__(self, model, trust_remote_code: bool = True, **kwargs):
        super().__init__()
        from vllm import LLM

        if model == "Alibaba-NLP/gte-Qwen2-1.5B-instruct":
            kwargs["hf_overrides"] = {"is_causal": True}

        if model in ["jinaai/jina-embeddings-v3", "intfloat/multilingual-e5-small"]:
            kwargs["dtype"] = "float32"

        if model == "intfloat/multilingual-e5-large-instruct":
            kwargs["dtype"] = "float32"

        self.model = LLM(model=model,
                         task="embed",
                         trust_remote_code=trust_remote_code,
                         **kwargs)

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
    ) -> np.ndarray:
        r = rng.permutation(len(sentences))
        sentences = [sentences[i] for i in r]
        outputs = self.model.embed(sentences, use_tqdm=False)
        embeds = np.array([o.outputs.embedding for o in outputs])
        embeds = embeds[np.argsort(r)]
        return embeds



def run_and_get_main_score(encoder):
    tasks = mteb.get_tasks(tasks=TASKS)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(encoder, verbosity=0, output_folder=None)

    main_score = results[0].scores["test"][0]["main_score"]
    return main_score



def get_st_main_score(model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, trust_remote_code=True)

    if model_name == "jinaai/jina-embeddings-v3":
        model.encode = partial(model.encode, task="text-matching")

    main_score = run_and_get_main_score(model)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return main_score


def run(model_name, times=10):
    st_main_score = get_st_main_score(model_name)
    encoder = VllmEncoder(model_name)

    scores = []
    for i in range(times):
        main_score = run_and_get_main_score(encoder)
        scores.append(main_score

    )

    print("main_score", model_name, st_main_score, np.mean(scores) - st_main_score, np.std(scores))


def process_warp(fn, /, *args, **kwargs):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(1, mp.get_context("spawn")) as executor:
        f = executor.submit(fn, *args, **kwargs)
        return f.result()



if __name__ == "__main__":
    MODELS = [
        "BAAI/bge-m3",
        "Snowflake/snowflake-arctic-embed-xs",
        "Snowflake/snowflake-arctic-embed-s",
        "Snowflake/snowflake-arctic-embed-m",
        "Snowflake/snowflake-arctic-embed-l",
        "Snowflake/snowflake-arctic-embed-m-v1.5",
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        "BAAI/bge-base-en-v1.5",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",

        "intfloat/multilingual-e5-large-instruct"
        "intfloat/multilingual-e5-small",
        "jinaai/jina-embeddings-v3",
        "Snowflake/snowflake-arctic-embed-m-long",
    ]

    for model_name in MODELS:
        try:
            process_warp(run, model_name)
        except Exception as e:
            print(model_name, e)
