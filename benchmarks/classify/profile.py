

def benchmark_vllm(args):
    from vllm import LLM

    for batchsize in args.batchsize:

        llm = LLM(model=args.model,
                  max_num_seqs=batchsize,
                  enforce_eager=args.enforce_eager,
                  hf_overrides=args.hf_overrides,
                  trust_remote_code=True,)

        for input_len in args.input_len:
            prompt = "if" * (input_len-2)
            prompts = [prompt for _ in range(args.num_prompts)]

            outputs = llm.classify(prompt, use_tqdm=False)
            assert len(outputs[0].prompt_token_ids) == input_len

            for i in range(10):
                llm.classify(prompts, use_tqdm=False)

            llm.start_profile()
            outputs = llm.classify(prompts, use_tqdm=False)
            for prompt, output in zip(prompts, outputs):
                pass
            llm.stop_profile()


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.model = 'Alibaba-NLP/gte-multilingual-reranker-base'
    args.hf_overrides = {"architectures": ["GteNewForSequenceClassification"]}

    args.tokenizer = args.model
    args.max_model_len = None
    batchsize = 64
    args.num_prompts = batchsize * 4
    args.batchsize = [batchsize]
    args.input_len = [32]

    args.enforce_eager =True
    benchmark_vllm(args)

    #args.enforce_eager = False
    #benchmark_vllm(args)