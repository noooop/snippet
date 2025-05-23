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
                 **kwargs):
        super().__init__()
        self.model = LLM(model=model_name_or_path,
                         dtype=dtype,
                         trust_remote_code=trust_remote_code,
                         max_num_seqs=batchsize,
                         **kwargs)

    def encode(
        self,
        sentences: Sequence[str],
        *args,
        **kwargs,
    ):
        outputs = self.model.embed(sentences,
                                   use_tqdm=True,
                                   truncate_prompt_tokens=-1)
        embeds = np.array([o.outputs.embedding for o in outputs])
        return embeds


def run(model_name, dtype):
    tasks = mteb.get_tasks(tasks=["T2Reranking"])
    evaluator = mteb.MTEB(tasks=tasks)

    results = evaluator.run(VllmMtebEncoder(model_name, dtype=dtype),
                            verbosity=0,
                            output_folder=None)

    print("=" * 80)
    print(model_name, dtype)
    print(results)


if __name__ == "__main__":
    import sys
    model_name, dtype = sys.argv[1], sys.argv[2]
    run(model_name, dtype)
