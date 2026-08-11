import time
from vllm import LLM, SamplingParams

import numpy as np
from PIL import Image

def make_image(seed=42):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (1920, 1080, 3), dtype=np.uint8))

question = "What is the content of this image?"
image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"

modality = "image"
placeholder = image_placeholder


sampling_params = SamplingParams(temperature=0.2, max_tokens=64)


if __name__ == "__main__":
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    GB = 1024*1024*1024
    MB = 1024*1024

    llm = LLM(
        model=model_name,
        max_num_seqs=4,
        gpu_memory_utilization=0.7,
        enforce_eager=False,
        load_format="dummy",
        enable_prefix_caching=False,
        paged_shm_size=GB*10,
        paged_shm_block_size=MB*4,
    )

    def generate(seed, n_images=1):
        images = [make_image(seed=seed + i * 10) for  i in range(n_images)]

        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{placeholder * n_images}"
        )

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {modality: images},
        }
        start = time.perf_counter()
        llm.generate(
            inputs,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        end = time.perf_counter()
        elapsed_time = end - start
        return elapsed_time


    generate(seed=1)


    for n_images in range(1, 11):
        tt = []
        for i in range(100):
            elapsed_time = generate(seed=n_images *100000 + i * 1000, n_images=n_images)
            tt.append(elapsed_time)

        print(n_images, np.mean(tt))

