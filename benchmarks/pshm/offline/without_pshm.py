import time
import os

#os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import numpy as np
from PIL import Image

from vllm import LLM, SamplingParams, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

def make_image(H, W, seed=42):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8))

questions = ["What is the content of this image?"]

image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"

modality = "image"
placeholder = image_placeholder

prompts = [
    (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    for question in questions
]

sampling_params = SamplingParams(temperature=0.2, max_tokens=1)

# size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}

def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        model="/share/cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c/",
        #max_model_len=100000,
        max_num_seqs=1,
        enforce_eager=False,
        load_format="dummy",
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 100},
        mm_processor_cache_gb=0,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    llm = LLM(**vars(args))


    def generate(n_images=1, seed=0):
        start = time.perf_counter()
        images = [
            make_image(H=1920, W=1080, seed=seed+i)
            for i in range(n_images)
        ]

        inputs = {
            "prompt": prompts[0] + placeholder * n_images,
            "multi_modal_data": {modality: images},
        }
        llm.generate(
            inputs,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        end = time.perf_counter()
        elapsed_time = end - start
        return elapsed_time

    generate(seed=1)
    elapsed_time = generate(seed=2,n_images=10)
    print(f"elapsed_time: {elapsed_time * 1000} ms",  )
