import time


from vllm import EngineArgs
from vllm.entrypoints.chat_utils import ChatTemplateConfig
from vllm.entrypoints.pooling.embed.io_processor import TokenEmbedIOProcessor
from vllm.entrypoints.pooling.embed.protocol import EmbeddingCompletionRequest
from vllm.entrypoints.pooling.typing import PoolingServeContext
from vllm.renderers import renderer_from_config
import vllm.entrypoints.pooling.embed.protocol as protocol


model_name = "BAAI/bge-base-en-v1.5"


def dummy_get_max_total_output_tokens(*args, **kwargs):
    return 8192, 0


protocol._get_max_total_output_tokens = dummy_get_max_total_output_tokens

engine_args = EngineArgs(model=model_name, runner="pooling")
vllm_config = engine_args.create_engine_config()
renderer = renderer_from_config(vllm_config)
chat_template_config = ChatTemplateConfig()

io_processor = TokenEmbedIOProcessor(
    vllm_config=vllm_config,
    renderer=renderer,
    chat_template_config=chat_template_config,
)


def preprocessing(prompt):
    request = EmbeddingCompletionRequest(input=prompt)

    pooling_params = io_processor.create_pooling_params(request)
    ctx = PoolingServeContext(
        request=request,
        model_name=model_name,
        pooling_params=pooling_params,
        request_id="0",
        trace_headers={},
    )

    io_processor.pre_process_online(ctx)


N = 1000

for input_len in range(128, 8192, 128):
    prompt = "你" * (input_len - 2)

    start = time.perf_counter()
    for i in range(N):
        preprocessing(prompt)

    end = time.perf_counter()
    e2e = end - start
    print(f"input_len: {input_len}, encode: {e2e / N * 1000:0.2f} ms")
