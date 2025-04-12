import torch
import torch.nn.functional as F
from torch import nn

x = torch.randn(256, 196, 768).cuda()
linear = nn.Linear(768, 768).cuda()
N = 10


@torch.inference_mode()
def test(run):
    print("=" * 80)
    print(run.__name__)

    linear.weight.normal_()
    linear.bias.normal_()

    torch.cuda.synchronize()
    # warmup
    for _ in range(5):
        run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(N):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    num_iters = 100

    for r in range(10):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        latencies: list[float] = []
        for i in range(num_iters):
            torch.cuda.synchronize()

            start_event.record()
            graph.replay()
            end_event.record()
            end_event.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
        avg = sum(latencies) / (num_iters * N) * 1000  # us

        print(f"Round {r}, average latency: {avg:.2f} us")

    graph.reset()


def a():
    y = linear(x)


def b():
    y = F.linear(x, linear.weight, linear.bias)


def c():
    y = x @ linear.weight.t() + linear.bias


if __name__ == "__main__":
    test(a)
    test(b)
    test(c)