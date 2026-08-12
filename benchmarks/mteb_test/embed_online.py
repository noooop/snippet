import os
import subprocess
import time
import argparse
import contextlib
import numpy as np
import requests
from torch.utils.data import DataLoader
from openai import OpenAI
import mteb
from mteb.models import ModelMeta
from mteb.types import Array

#os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

MTEB_RERANK_TASKS = ["NFCorpus"]
MTEB_RERANK_LANGS = ["eng"]

_empty_model_meta = ModelMeta(
    loader=None,
    name="vllm/model",
    revision="1",
    release_date=None,
    languages=None,
    framework=[],
    similarity_fn_name=None,
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    open_weights=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=None,
    training_datasets=None,
    modalities=["text"],
)


class MtebEmbedMixin(mteb.EncoderProtocol):
    mteb_model_meta = _empty_model_meta

    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        return np.dot(embeddings1, embeddings2.T) / (norm1 * norm2.T)

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        return np.sum(embeddings1 * embeddings2, axis=1) / (norm1.flatten() * norm2.flatten())


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class OpenAIClientMtebEncoder(MtebEmbedMixin):
    def __init__(self, model_name: str, batchsize: int, seed: int = 0):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="http://localhost:8000/v1/",
            api_key="DUMMY"
        )
        self.rng = np.random.default_rng(seed=seed)
        self.batchsize = batchsize

    def encode(self, inputs: DataLoader[mteb.types.BatchedInput], *args, **kwargs) -> np.ndarray:
        sentences = [text for batch in inputs for text in batch["text"]]
        idx = self.rng.permutation(len(sentences))
        shuffled = [sentences[i] for i in idx]

        embeddings = []
        for chunk in chunk_list(shuffled, self.batchsize):
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=chunk,
                extra_body={"truncate_prompt_tokens": -1}
            )
            embeddings.extend([d.embedding for d in resp.data])

        embeds = np.array(embeddings)
        # Restore original order
        embeds = embeds[np.argsort(idx)]
        return embeds


def run_mteb_task(encoder: mteb.EncoderProtocol) -> float:
    tasks = mteb.get_tasks(
        tasks=MTEB_RERANK_TASKS,
        languages=MTEB_RERANK_LANGS,
        eval_splits=["test"]
    )
    results = mteb.evaluate(encoder, tasks, cache=None, show_progress_bar=False)
    return results[0].scores["test"][0]["main_score"]


@contextlib.contextmanager
def run_server(model: str, max_model_len: int = None):
    cmd = ["vllm", "serve", model, "--disable-uvicorn-access-log"]
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])

    proc = subprocess.Popen(cmd)

    try:
        # Wait for server to become ready (max 300 seconds)
        start = time.time()
        timeout = 300
        while True:
            try:
                resp = requests.post(
                    "http://localhost:8000/v1/embeddings",
                    json={"model": model, "input": "ping"},
                    timeout=5
                )
                if resp.status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.time() - start > timeout:
                raise TimeoutError("vLLM server did not start within timeout.")
            time.sleep(5)
        yield proc
    finally:
        proc.terminate()


def main():
    parser = argparse.ArgumentParser(description="MTEB evaluation with vLLM embedding server")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-large-instruct",
                        help="Model name or path")
    parser.add_argument("--max-model-len", type=int, default=512,
                        help="Maximum model context length")
    parser.add_argument("--batchsizes", type=int, nargs="+", default=[1, 16, 32, 128],
                        help="List of batch sizes to test")
    parser.add_argument("--retry", type=int, default=3,
                        help="Number of repeated runs per batch size (each with a different random seed)")
    args = parser.parse_args()

    with run_server(args.model, args.max_model_len):
        for bs in args.batchsizes:
            for retry_idx in range(args.retry):
                encoder = OpenAIClientMtebEncoder(args.model, bs, seed=retry_idx)
                score = run_mteb_task(encoder)
                print(f"batchsize={bs}, retry={retry_idx+1}, main_score={score:.6f}")


if __name__ == "__main__":
    main()