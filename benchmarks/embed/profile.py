def benchmark_vllm(args):
    from vllm import LLM

    for batchsize in args.batchsize:

        llm = LLM(model=args.model,
                  max_num_seqs=batchsize,
                  enforce_eager=args.enforce_eager)

        for input_len in args.input_len:
            prompt = "if" * (input_len - 2)
            prompts = [prompt for _ in range(args.num_prompts)]

            for i in range(10):
                llm.embed(prompts, use_tqdm=False)

            llm.start_profile()
            outputs = llm.embed(prompts, use_tqdm=False)
            for prompt, output in zip(prompts, outputs):
                pass
            llm.stop_profile()


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.model = 'BAAI/bge-m3'

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.max_model_len = None
    batchsize = 64
    args.num_prompts = batchsize * 4
    args.batchsize = [batchsize]
    args.input_len = [32]

    args.enforce_eager = True
    benchmark_vllm(args)

    #args.enforce_eager = False
    #benchmark_vllm(args)
