def client(N):
    import asyncio
    import time

    import zmq.asyncio

    context = zmq.asyncio.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    async def main_loop():
        start = time.perf_counter()
        for request in range(N):
            await socket.send(b"Hello")
            await socket.recv()
        end = time.perf_counter()
        elapsed_time = end - start

        print(
            f"Latency: {1000 * elapsed_time/N:0.4f} ms, QPS: {N / elapsed_time:0.2f}"
        )

    asyncio.run(main_loop())
