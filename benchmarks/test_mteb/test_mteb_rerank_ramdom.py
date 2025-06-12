# SPDX-License-Identifier: Apache-2.0
import os
import shutil
from pathlib import Path
from typing import Optional

import mteb
import numpy as np

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

MTEB_RERANK_TASKS = ["NFCorpus"]
MTEB_RERANK_LANGS = ["en"]
MTEB_RERANK_TOL = 1e-4
rng = np.random.default_rng(seed=42)


class RandomEncoder(mteb.Encoder):

    def __init__(self):
        super().__init__()

    def predict(
        self,
        sentences: list[tuple[str, str,
                              Optional[str]]],  # query, corpus, prompt
        *args,
        **kwargs,
    ) -> np.ndarray:
        scores = np.random.rand(len(sentences))
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


def run_mteb_rerank(tasks, languages):
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
        RandomEncoder(),
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


def get_st_main_score_random():
    main_score = run_mteb_rerank(tasks=MTEB_RERANK_TASKS,
                                 languages=MTEB_RERANK_LANGS)
    print("random", main_score)


if __name__ == "__main__":
    get_st_main_score_random()
