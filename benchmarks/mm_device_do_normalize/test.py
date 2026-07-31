import time

import numpy as np
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from vllm import LLM, SamplingParams

org_preprocess = Qwen2VLImageProcessor.preprocess

def preprocess(self, images, **kwargs):
    print("Qwen2VLImageProcessor preprocess")
    print(images)
    print(kwargs)

    start = time.perf_counter()
    out = org_preprocess(self, images, **kwargs)
    end = time.perf_counter()
    elapsed_time = end - start
    print("preprocess", elapsed_time * 1000, "ms")
    return out

Qwen2VLImageProcessor.preprocess = preprocess


def make_image(size, seed=42):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


questions = ["What is the content of this image?"]

image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"

modality = "image"
placeholder = image_placeholder

prompts = [
    (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{placeholder}{placeholder}"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    for question in questions
]

sampling_params = SamplingParams(temperature=0.2, max_tokens=1)


if __name__ == "__main__":
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=10000,
        max_num_seqs=1,
        enforce_eager=True,
        load_format="dummy",
        limit_mm_per_prompt={"image": 2},
    )

    print("+" * 90)

    image = make_image(1024, seed=1)
    inputs = {
        "prompt": prompts[0],
        "multi_modal_data": {modality: [image]},
    }

    start = time.perf_counter()
    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params,
    )
    end = time.perf_counter()
    elapsed_time = end - start
    print("1", elapsed_time)

    print("+" * 90)

    image = make_image(1024, seed=2)
    inputs = {
        "prompt": prompts[0],
        "multi_modal_data": {modality: [image]},
    }
    start = time.perf_counter()
    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params,
    )
    end = time.perf_counter()
    elapsed_time = end - start
    print("2", elapsed_time)
