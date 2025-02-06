def server():
    import zmq.green as zmq

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    while True:
        socket.recv()
        socket.send(b"World")
