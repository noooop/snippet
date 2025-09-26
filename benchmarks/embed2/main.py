import traceback

from benchmarks.embed2.profile import main

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
        main(model_name)
    except Exception:
        traceback.print_exc()
