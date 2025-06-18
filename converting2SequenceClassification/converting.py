# refer to https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3

import torch


def converting(model_name, classifier_from_token, path, device="cpu", ):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    causal_lm = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

    lm_head_weights = causal_lm.lm_head.weight


    a = tokenizer.convert_tokens_to_ids(classifier_from_token[0])
    b = tokenizer.convert_tokens_to_ids(classifier_from_token[1])


    score_weight = lm_head_weights[b].to(torch.float32).to(
    device).to(torch.float32)- lm_head_weights[a].to(device)

    seq_cls_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
        device_map=device
    )

    with torch.no_grad():
        seq_cls_model.score.weight.copy_(score_weight.unsqueeze(0))
        if seq_cls_model.score.bias is not None:
            seq_cls_model.score.bias.zero_()

    seq_cls_model.save_pretrained(path)
    tokenizer.save_pretrained(path)


if __name__ == "__main__":
    converting("Qwen/Qwen3-Reranker-0.6B", ["no", "yes"], "./Qwen3-Reranker-0.6B-seq-cls")
