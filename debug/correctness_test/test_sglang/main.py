import dataclasses
import multiprocessing as mp
import unittest
from typing import List

import torch

from debug.correctness_test.test_sglang.utils import HFRunner, SRTRunner, check_close_model_outputs


@dataclasses.dataclass
class ModelCase:
    model_path: str
    tp_size: int = 1
    prefill_tolerance: float = 5e-2
    decode_tolerance: float = 5e-2
    rouge_l_tolerance: float = 0
    skip_long_prompt: bool = False
    trust_remote_code: bool = False


MODELS = [
    ModelCase("Qwen/Qwen2.5-7B-Instruct"),
    #   ModelCase("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"),
]

TORCH_DTYPES = [torch.bfloat16]


DEFAULT_PROMPTS = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]


class TestGenerationModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def assert_close_logits_and_output_strs(
        self,
        prompts: List[str],
        model_case: ModelCase,
        torch_dtype: torch.dtype,
    ) -> None:
        model_path = model_case.model_path
        prefill_tolerance, decode_tolerance, rouge_l_tolerance = (
            model_case.prefill_tolerance,
            model_case.decode_tolerance,
            model_case.rouge_l_tolerance,
        )
        max_new_tokens = 32

        with HFRunner(
                model_path,
                torch_dtype=torch_dtype,
                model_type="generation",
                trust_remote_code=model_case.trust_remote_code,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(prompts,
                                           max_new_tokens=max_new_tokens)

        with SRTRunner(model_path,
                       tp_size=model_case.tp_size,
                       torch_dtype=torch_dtype,
                       model_type="generation",
                       trust_remote_code=model_case.trust_remote_code,
                       disable_radix_cache=True) as srt_runner:
            srt_outputs = srt_runner.forward(prompts,
                                             max_new_tokens=max_new_tokens)

        check_close_model_outputs(
            hf_outputs=hf_outputs,
            srt_outputs=srt_outputs,
            prefill_tolerance=prefill_tolerance,
            decode_tolerance=decode_tolerance,
            rouge_l_tolerance=rouge_l_tolerance,
            debug_text=f"model_path={model_path} prompts={prompts}",
        )

    def test_models(self):
        for model_case in MODELS:
            for torch_dtype in TORCH_DTYPES:

                # Skip long prompts for models that do not have a long context
                prompts = DEFAULT_PROMPTS
                if model_case.skip_long_prompt:
                    prompts = [p for p in DEFAULT_PROMPTS if len(p) < 1000]

                # Assert the logits and output strs are close
                self.assert_close_logits_and_output_strs(
                    prompts, model_case, torch_dtype)


if __name__ == "__main__":
    unittest.main()
