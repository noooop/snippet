from typing import Sequence

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import mteb
import numpy as np
from vllm import LLM


class VllmMtebEncoder(mteb.Encoder):

    def __init__(self,
                 model_name_or_path,
                 dtype,
                 trust_remote_code=True,
                 batchsize=128,
                 truncate_prompt_tokens=-1,
                 **kwargs):
        super().__init__()

        rope_theta = 1000
        factor = 4.0
        original_max_position_embeddings = 2048

        # Use yarn to extend context
        hf_overrides = {
            "rope_theta": rope_theta,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": original_max_position_embeddings
            },
            "max_model_len": int(original_max_position_embeddings * factor)
        }

        self.model = LLM(model=model_name_or_path,
                         dtype=dtype,
                         trust_remote_code=trust_remote_code,
                         max_num_seqs=batchsize,
                         hf_overrides=hf_overrides,
                         **kwargs)
        print("max_model_len", self.model.llm_engine.model_config.max_model_len)
        self.truncate_prompt_tokens = truncate_prompt_tokens

    def encode(
        self,
        sentences: Sequence[str],
        *args,
        **kwargs,
    ):
        outputs = self.model.embed(sentences,
                                   use_tqdm=True,
                                   truncate_prompt_tokens=self.truncate_prompt_tokens)
        embeds = np.array([o.outputs.embedding for o in outputs])
        return embeds


def run(model_name, dtype, truncate_prompt_tokens=-1):
    tasks = mteb.get_tasks(tasks=["T2Reranking"])
    evaluator = mteb.MTEB(tasks=tasks)

    results = evaluator.run(VllmMtebEncoder(model_name,
                                            dtype=dtype,
                                            truncate_prompt_tokens=truncate_prompt_tokens),
                            verbosity=0,
                            output_folder=None)

    print("=" * 80)
    print(model_name, dtype)
    print(results[0].scores['dev'][0]["main_score"])


if __name__ == "__main__":
    import sys
    model_name, dtype = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        truncate_prompt_tokens = int(sys.argv[3])
    else:
        truncate_prompt_tokens = -1

    run(model_name, dtype, truncate_prompt_tokens)
