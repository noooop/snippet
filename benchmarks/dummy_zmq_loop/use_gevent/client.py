def client(N):
    import time

    import zmq.green as zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    start = time.perf_counter()
    for request in range(N):
        socket.send(b"Hello")
        socket.recv()
    end = time.perf_counter()
    elapsed_time = end - start

    print(
        f"Latency: {1000 * elapsed_time/N:0.4f} ms, QPS: {N / elapsed_time:0.2f}"
    )
