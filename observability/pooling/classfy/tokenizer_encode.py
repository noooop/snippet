from transformers import AutoTokenizer

model_name = "BAAI/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


for output_len in range(32, 512, 16):
    input_len = output_len - 2

    prompt = "你" * input_len
    prompt_inputs = tokenizer.encode(text=prompt)

    print(input_len, len(prompt_inputs), output_len)
    assert len(prompt_inputs) == output_len
