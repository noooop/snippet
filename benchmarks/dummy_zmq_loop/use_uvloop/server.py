def server():
    import uvloop
    import zmq.asyncio

    context = zmq.asyncio.Context()

    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    async def main_loop():
        while True:
            await socket.recv()
            await socket.send(b"World")

    uvloop.run(main_loop())
