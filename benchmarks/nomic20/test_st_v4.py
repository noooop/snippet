import mteb
from sentence_transformers import SentenceTransformer
import numpy as np
from mteb.types import Array
from mteb.models import ModelMeta
from torch.utils.data import DataLoader

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
    modalities=["text"],  # 'image' can be added to evaluate multimodal models
)

class MtebEmbedMixin(mteb.EncoderProtocol):
    mteb_model_meta = _empty_model_meta

    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        # Cosine similarity
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        sim = np.dot(embeddings1, embeddings2.T) / (norm1 * norm2.T)
        return sim

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        # Cosine similarity
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        sim = np.sum(embeddings1 * embeddings2, axis=1) / (
            norm1.flatten() * norm2.flatten()
        )
        return sim


class HfMtebEncoder(MtebEmbedMixin):
    def __init__(self, model_name:str, revision:str):
        self.model = SentenceTransformer(model_name, revision=revision, trust_remote_code=True)

    def encode(
        self,
        inputs: DataLoader[mteb.types.BatchedInput],
        *args,
        **kwargs,
    ) -> np.ndarray:
        sentences = [text * 20 for batch in inputs for text in batch["text"]]
        return self.model.encode(sentences)


if __name__ == "__main__":
    model_name = "nomic-ai/nomic-embed-text-v1"
    revision = "720244025c1a7e15661a174c63cce63c8218e52b"

    tasks = mteb.get_tasks(tasks=["STS12"])

    evaluation = mteb.MTEB(tasks=tasks)
    encoder = HfMtebEncoder(model_name=model_name, revision=revision)
    results = evaluation.run(encoder, verbosity=0, output_folder=None)

    main_score = results[0].scores["test"][0]["main_score"]

    print(main_score)