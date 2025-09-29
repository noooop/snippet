import traceback

from benchmarks.embed2.profile import main
from benchmarks.embed2.merge import main as merger

for model_name in [
        "jinaai/jina-embeddings-v3",
        "BAAI/bge-m3",
        "intfloat/multilingual-e5-base",
        "BAAI/bge-base-en",
        "Alibaba-NLP/gte-multilingual-base",
        "Qwen/Qwen3-Embedding-0.6B",
        "thenlper/gte-large",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "BAAI/bge-code-v1",
        "Alibaba-NLP/gte-modernbert-base",
        "google/embeddinggemma-300m",
        "intfloat/e5-small",
        "nomic-ai/nomic-embed-text-v1",
        "nomic-ai/nomic-embed-text-v2-moe",
        "Snowflake/snowflake-arctic-embed-xs",
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        "TencentBAC/Conan-embedding-v1",
        "Snowflake/snowflake-arctic-embed-m-long",
        "Snowflake/snowflake-arctic-embed-m-v1.5",
]:
    try:
        main(model_name, dtype="float16", tp=1)
    except Exception:
        traceback.print_exc()

    try:
        profiles_dir = main(model_name, dtype="float16", tp=2)
        merger(dir_data=profiles_dir)
    except Exception:
        traceback.print_exc()