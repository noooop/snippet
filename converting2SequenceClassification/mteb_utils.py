import os

os.environ["VLLM_USE_V1"] = "0"

import tempfile
from typing import NamedTuple

import mteb
import numpy as np
import pytest


class ModelInfo(NamedTuple):
    original_model_name: str
    converted_model_name: str


MTEB_RERANK_TASKS = ["NFCorpus"]
MTEB_RERANK_LANGS = ["en"]
MTEB_RERANK_TOL = 5e-4


class VllmRunner:

    def __init__(self,
                 model_name,
                 dtype,
                 trust_remote_code: bool = True,
                 **kwargs):
        super().__init__()
        from vllm import LLM

        self.model_name = model_name

        self.model = LLM(
            model=model_name,
            task="score",
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def predict(
        self,
        querys,
        corpus,
        return_n_tokens=False,
    ):
        outputs = self.model.score(querys,
                                   corpus,
                                   truncate_prompt_tokens=-1,
                                   use_tqdm=False)
        scores = [output.outputs.score for output in outputs]
        n_tokens = [len(output.prompt_token_ids) for output in outputs]

        if return_n_tokens:
            return np.array(scores), np.array(n_tokens)
        else:
            return np.array(scores)


def run_mteb_rerank(cross_encoder, tasks, languages):
    with tempfile.TemporaryDirectory() as results_folder:
        bm25s = mteb.get_model("bm25s")
        tasks = mteb.get_tasks(tasks=tasks, languages=languages)

        subset = "default"
        eval_splits = ["test"]

        evaluation = mteb.MTEB(tasks=tasks)
        evaluation.run(
            bm25s,
            verbosity=0,
            eval_splits=eval_splits,
            save_predictions=True,
            output_folder=f"{results_folder}/stage1",
            encode_kwargs={"show_progress_bar": False},
        )

        results = evaluation.run(
            cross_encoder,
            verbosity=0,
            eval_splits=eval_splits,
            top_k=10,
            save_predictions=True,
            output_folder=f"{results_folder}/stage2",
            previous_results=
            f"{results_folder}/stage1/NFCorpus_{subset}_predictions.json",
            encode_kwargs={"show_progress_bar": False},
        )
        main_score = results[0].scores["test"][0]["main_score"]
    return main_score


def mteb_test_rerank_models(hf_runner,
                            vllm_runner,
                            model_info: ModelInfo,
                            dtype="float16"):
    vllm_model = vllm_runner(model_info.converted_model_name, dtype=dtype)
    vllm_main_score = run_mteb_rerank(vllm_model,
                                      tasks=MTEB_RERANK_TASKS,
                                      languages=MTEB_RERANK_LANGS)
    print("VLLM:", vllm_main_score)

    hf_model = hf_runner(model_info.original_model_name, dtype=dtype)
    st_main_score = run_mteb_rerank(hf_model,
                                    tasks=MTEB_RERANK_TASKS,
                                    languages=MTEB_RERANK_LANGS)
    print("SentenceTransformers:", st_main_score)
    print("Difference:", st_main_score - vllm_main_score)

    assert st_main_score == pytest.approx(vllm_main_score, abs=MTEB_RERANK_TOL)
