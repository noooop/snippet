# SPDX-License-Identifier: Apache-2.0
import gc
import os
from collections.abc import Sequence
from functools import partial

import mteb
import numpy as np
import torch
from mteb.encoder_interface import PromptType
from vllm.transformers_utils.utils import maybe_model_redirect

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"


TASKS = ["STS12"]
rng = np.random.default_rng(seed=42)


class VllmEncoder(mteb.Encoder):

    def __init__(self, model, dtype, trust_remote_code: bool = True, **kwargs):
        super().__init__()
        from vllm import LLM

        if model == "Alibaba-NLP/gte-Qwen2-1.5B-instruct":
            kwargs["hf_overrides"] = {"is_causal": True}

        self.model = LLM(model=model,
                         task="embed",
                         dtype=dtype,
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
    model = SentenceTransformer(maybe_model_redirect(model_name), trust_remote_code=True)

    if model_name == "jinaai/jina-embeddings-v3":
        model.encode = partial(model.encode, task="text-matching")

    main_score = run_and_get_main_score(model)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return main_score


def run(model_name, times=10):
    st_main_score = get_st_main_score(model_name)
    print(model_name, st_main_score)

    for dtype in ["float16", "bfloat16", "float32"]:
        encoder = VllmEncoder(model_name, dtype=dtype)

        scores = []
        for i in range(times):
            main_score = run_and_get_main_score(encoder)
            scores.append(main_score)

        print(dtype, model_name, st_main_score, np.mean(scores) - st_main_score, np.std(scores))


def process_warp(fn, /, *args, **kwargs):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(1, mp.get_context("spawn")) as executor:
        f = executor.submit(fn, *args, **kwargs)
        return f.result()

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1]
    run(model_name)
