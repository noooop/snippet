from transformers import AutoTokenizer

model_name = "BAAI/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

"""
for input_len in range(1, 512, 16):

    queries = "你" * input_len
    documents = "你" * input_len

    prompt_inputs = tokenizer.encode(
        text=queries, text_pair=documents
    )

    print(input_len, len(prompt_inputs))

# y = 2x + 4
"""

get_input_len = lambda output_len: (output_len - 4) // 2


for output_len in range(32, 512, 16):
    input_len = get_input_len(output_len)

    queries = "你" * input_len
    documents = "你" * input_len

    prompt_inputs = tokenizer.encode(text=queries, text_pair=documents)

    print(input_len, len(prompt_inputs), output_len)
    assert len(prompt_inputs) == output_len
