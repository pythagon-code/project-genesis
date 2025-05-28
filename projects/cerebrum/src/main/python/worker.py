from sys import argv
from socket import socket, AF_INET, SOCK_STREAM, error
import json

import transformers


def get_message(client: socket, buffer: bytearray) -> dict | None:
    while b"\n" not in buffer:
        try:
            chunk = client.recv(bufsize=1024)
        except error as e:
            return {"Operation": "Shutdown"}
        if len(chunk) == 0:
            return {"Operation": "Shutdown"}

        buffer.extend(chunk)

    message, buffer = buffer.split(sep=b"\n", maxsplit=1)
    message = message.decode().strip()
    return json.loads(message)


def send_message(client: socket, response: dict) -> None:
    message = json.dumps(response) + "\n"
    client.sendall(message.encode())


def main() -> None:
    port = int(argv[1])
    client = socket(family=AF_INET, type=SOCK_STREAM)
    client.connect(address=("localhost", port))
    buffer = bytearray()

    print(get_message(client=client, buffer=buffer), flush=True)

    client.close()


if __name__ == "__main__":
    main()
