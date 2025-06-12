# SPDX-License-Identifier: Apache-2.0
import gc
import os
import shutil
from pathlib import Path
from typing import Optional

import mteb
import numpy as np
import torch

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

from vllm.transformers_utils.utils import maybe_model_redirect

MTEB_RERANK_TASKS = ["NFCorpus"]
MTEB_RERANK_LANGS = ["en"]
MTEB_RERANK_TOL = 1e-4
rng = np.random.default_rng(seed=42)


class VllmEncoder(mteb.Encoder):

    def __init__(self, model, dtype, trust_remote_code: bool = True, **kwargs):
        super().__init__()
        from vllm import LLM

        self.model = LLM(model=model,
                         task="score",
                         dtype=dtype,
                         trust_remote_code=trust_remote_code,
                         **kwargs)

    def predict(
        self,
        sentences: list[tuple[str, str,
                              Optional[str]]],  # query, corpus, prompt
        *args,
        **kwargs,
    ) -> np.ndarray:
        querys = [s[0] for s in sentences]
        corpus = [s[1] for s in sentences]

        outputs = self.model.score(querys,
                                   corpus,
                                   truncate_prompt_tokens=-1,
                                   use_tqdm=False)
        outputs = [output.outputs.score for output in outputs]
        scores = np.array(outputs)
        return scores


def _run_mteb_rerank_stage1(evaluation, eval_splits, results_folder_retriever):
    retriever = mteb.get_model("bm25s")

    results = evaluation.run(
        retriever,
        verbosity=0,
        eval_splits=eval_splits,
        save_predictions=True,
        output_folder=str(results_folder_retriever),
        encode_kwargs={"show_progress_bar": False},
    )

    main_score = results[0].scores["test"][0]["main_score"]
    print("mteb_rerank_stage1", main_score)


def run_mteb_rerank(cross_encoder, tasks, languages):
    results_folder = "tmp_mteb_results"
    subset = "default"
    eval_splits = ["test"]

    results_folder_retriever = Path(__file__).parent
    previous_results = results_folder_retriever / f"{tasks[0]}_{subset}_predictions.json"

    shutil.rmtree(results_folder, ignore_errors=True)

    tasks = mteb.get_tasks(tasks=tasks, languages=languages)
    evaluation = mteb.MTEB(tasks=tasks)

    if not (results_folder_retriever / previous_results).exists():
        _run_mteb_rerank_stage1(evaluation, eval_splits,
                                results_folder_retriever)

    results = evaluation.run(
        cross_encoder,
        verbosity=0,
        eval_splits=eval_splits,
        top_k=10,
        save_predictions=True,
        output_folder=f"{results_folder}/stage2",
        previous_results=str(previous_results),
        encode_kwargs={"show_progress_bar": False},
    )

    main_score = results[0].scores["test"][0]["main_score"]
    shutil.rmtree(results_folder, ignore_errors=True)
    return main_score


@torch.no_grad()
def get_st_main_score(model_name):
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(maybe_model_redirect(model_name),
                         model_kwargs={"torch_dtype": torch.float32},
                         trust_remote_code=True)
    model_dtype = next(model.model.parameters()).dtype

    model_predict = model.predict

    def _predict(
        sentences: list[tuple[str, str,
                              Optional[str]]],  # query, corpus, prompt
        *args,
        **kwargs,
    ):
        # vllm and st both remove the prompt, fair comparison.
        prompts = [(s[0], s[1]) for s in sentences]
        return model_predict(prompts, *args, **kwargs, batch_size=8)

    model.predict = _predict

    st_main_score = run_mteb_rerank(model,
                                    tasks=MTEB_RERANK_TASKS,
                                    languages=MTEB_RERANK_LANGS)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return st_main_score, model_dtype


def run(model_name):
    st_main_score, model_dtype = get_st_main_score(model_name)
    print(model_name, model_dtype, st_main_score)

    for dtype in ["float16", "bfloat16", "float32"]:
        encoder = VllmEncoder(model_name, dtype=dtype)
        main_score = run_mteb_rerank(encoder,
                                     tasks=MTEB_RERANK_TASKS,
                                     languages=MTEB_RERANK_LANGS)

        print(dtype, model_name, st_main_score, main_score,
              main_score - st_main_score)


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1]
    run(model_name)
