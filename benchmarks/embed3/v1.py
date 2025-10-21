import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
import time
from typing import Callable, Union


def benchmark_vllm(args):
    from tqdm import tqdm
    from vllm import LLM as BaseLLM
    from vllm import PoolingRequestOutput, RequestOutput
    from vllm.distributed import cleanup_dist_env_and_memory

    class LLM(BaseLLM):
        def _run_engine(
            self, *, use_tqdm: Union[bool, Callable[..., tqdm]] = False
        ) -> list[Union[RequestOutput, PoolingRequestOutput]]:
            # Initialize tqdm.
            if use_tqdm:
                num_requests = self.llm_engine.get_num_unfinished_requests()
                tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
                pbar = tqdm_func(
                    total=num_requests,
                    desc="Processed prompts",
                    dynamic_ncols=True,
                    postfix=(
                        f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"
                    ),
                )

            # Run the engine.
            self.n_step = 0
            outputs: list[Union[RequestOutput, PoolingRequestOutput]] = []
            total_in_toks = 0
            total_out_toks = 0
            while self.llm_engine.has_unfinished_requests():
                step_outputs = self.llm_engine.step()
                self.n_step += 1
                for output in step_outputs:
                    if output.finished:
                        outputs.append(output)
                        if use_tqdm:
                            if isinstance(output, RequestOutput):
                                # Calculate tokens only for RequestOutput
                                n = len(output.outputs)
                                assert output.prompt_token_ids is not None
                                total_in_toks += len(output.prompt_token_ids) * n
                                in_spd = total_in_toks / pbar.format_dict["elapsed"]
                                total_out_toks += sum(
                                    len(stp.token_ids) for stp in output.outputs
                                )
                                out_spd = total_out_toks / pbar.format_dict["elapsed"]
                                pbar.postfix = (
                                    f"est. speed input: {in_spd:.2f} toks/s, "
                                    f"output: {out_spd:.2f} toks/s"
                                )
                                pbar.update(n)
                            else:
                                pbar.update(1)
                            if pbar.n == num_requests:
                                pbar.refresh()

            if use_tqdm:
                pbar.close()
            # Sort the outputs by request ID.
            # This is necessary because some requests may be finished earlier than
            # its previous requests.
            return sorted(outputs, key=lambda x: int(x.request_id))

    for batchsize in args.batchsize:
        llm = LLM(
            model=args.model,
            max_model_len=args.max_model_len,
            max_num_seqs=batchsize,
            max_num_batched_tokens=batchsize * args.max_model_len,
            enforce_eager=args.enforce_eager,
        )

        for input_len in args.input_len:
            prompt = "你" * (input_len - 2)
            prompts = [prompt for _ in range(args.num_prompts)]

            outputs = llm.embed(prompt, use_tqdm=False)
            assert len(outputs[0].prompt_token_ids) == input_len

            start = time.perf_counter()
            outputs = llm.embed(prompts, use_tqdm=False)
            for prompt, output in zip(prompts, outputs):
                pass
            end = time.perf_counter()

            n_step = llm.n_step
            elapsed_time = end - start
            delay = elapsed_time / n_step

            print(
                f"Batchsize {batchsize}, Throughput: "
                f"{len(prompts) / elapsed_time:.4f} requests/s, "
                f"{len(prompts * input_len) / elapsed_time:.4f} tokens/s, "
                f"Latency {delay * 1000:0.2f} ms, n_step {n_step}"
            )

        del llm
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    from easydict import EasyDict as edict

    args = edict()

    args.model = "BAAI/bge-base-en-v1.5"

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.max_model_len = 512
    args.num_prompts = 10000
    args.batchsize = [1, 2, 4, 8, 16, 32, 64]
    args.input_len = [32, 64, 128, 256]

    from concurrent.futures import ProcessPoolExecutor

    def run(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_vllm, args)
            f.result()

    args.enforce_eager = True
    run(args)

    args.enforce_eager = False
    run(args)
