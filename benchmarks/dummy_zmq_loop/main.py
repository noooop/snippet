import time
from multiprocessing import Process


def lazy_import(module):
    import importlib
    module_name, class_name = module.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


servers = [
    "benchmarks.dummy_zmq_loop.use_naive.server:server",
    "benchmarks.dummy_zmq_loop.use_gevent.server:server",
    "benchmarks.dummy_zmq_loop.use_asyncio.server:server",
    "benchmarks.dummy_zmq_loop.use_uvloop.server:server",
]

clients = [
    "benchmarks.dummy_zmq_loop.use_naive.client:client",
    "benchmarks.dummy_zmq_loop.use_gevent.client:client",
    "benchmarks.dummy_zmq_loop.use_asyncio.client:client",
    "benchmarks.dummy_zmq_loop.use_uvloop.client:client",
]

N = 100000

for server_impl in servers:
    for client_impl in clients:
        server_impl_name = server_impl.split(".")[2]
        client_impl_name = client_impl.split(".")[2]
        print(f"server: {server_impl_name}, client: {client_impl_name}")

        server = lazy_import(server_impl)
        client = lazy_import(client_impl)

        s = Process(target=server)
        c = Process(target=client, args=(N, ))

        s.start()
        time.sleep(1)

        c.start()
        c.join()
        s.terminate()

        time.sleep(1)
