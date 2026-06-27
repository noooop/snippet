import multiprocessing as mp
import time
from multiprocessing import resource_tracker, shared_memory

import numpy as np

size = 2**32
dtype = np.uint8


def worker(size, conn, stop_event):
    shm = shared_memory.SharedMemory(size=size, create=True)
    try:
        conn.send(shm.name)
        conn.close()
        stop_event.wait()
    finally:
        shm.close()
        shm.unlink()


def get_shm(size):
    parent_conn, child_conn = mp.Pipe()
    stop_event = mp.Event()
    process = mp.Process(target=worker, args=(size, child_conn, stop_event))
    process.start()
    shm_name = parent_conn.recv()
    parent_conn.close()
    shm = shared_memory.SharedMemory(name=shm_name)
    resource_tracker.unregister(shm._name, "shared_memory")

    return shm, process, stop_event


if __name__ == "__main__":
    shm_src, process_src, stop_event_src = get_shm(size)
    shm_dst, process_dst, stop_event_dst = get_shm(size)

    src = np.ndarray(size, dtype=dtype, buffer=shm_src.buf)
    dst = np.ndarray(size, dtype=dtype, buffer=shm_dst.buf)

    del src, dst
    time.sleep(1)

    stop_event_src.set()
    process_src.join()
    stop_event_dst.set()
    process_dst.join()

    time.sleep(1)

    shm_src.close()
    shm_dst.close()
