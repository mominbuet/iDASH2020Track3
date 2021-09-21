import struct


def connectSocket(sock, ip, port):
    sock.connect((ip, port))
    return sock


def sendData(sock, data):
    if type(data) is str:
        data = data.encode()
    sock.sendall(data)


def recvData(sock):
    data = sock.recv(4096)
    return data


def sendLargemsg(sock, msg):
    if type(msg) is str:
        msg = msg.encode()
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def recvLargeMsg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvLargeMsgInternal(sock, 4)
    if not raw_msglen:
        return None
    msglen = int(struct.unpack('>I', raw_msglen)[0])
    # Read the message data
    return recvLargeMsgInternal(sock, msglen)


def recvLargeMsgInternal(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def recvall(sock):
    BUFF_SIZE = 4096  # 4 KiB
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break
    return data


def closeSocket(sock):
    if sock != None:
        sock.close()
