import argparse
import json
import logging
import os
import socket as sock
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image  # type: ignore

from fluid_ai.base import UiElement
from fluid_ai.utils import plot_ui_elements

from src.utils import parse_results

HOSTNAME = "143.248.136.109"
PORT = 8440


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action="store", required=True)
    parser.add_argument("-j", "--json", action="store", required=True)
    parser.add_argument(
        "--config", action="store", default="config.json", help="Path to a config file"
    )
    parser.add_argument(
        "--chunk-size",
        action="store",
        default=1024,
        type=int,
        help="Chunk size to read",
    )
    parser.add_argument(
        "--scale",
        action="store",
        default=1,
        type=float,
        help="Scale of the result image to display",
    )
    return parser.parse_args()


def readall(
    socket: sock.socket,
    num_bytes: int,
    chunk_size: int,
    timeout: Optional[float] = None,
) -> bytes:
    deadline = old_timeout = None
    if timeout is not None:
        deadline = time.time() + timeout
        old_timeout = socket.gettimeout()

    buffer = bytearray(num_bytes)
    curr = 0
    while curr < num_bytes:
        if deadline is not None:
            socket.settimeout(deadline - time.time())
        if chunk_size < 0:
            data = socket.recv(num_bytes)
        else:
            data = socket.recv(min(chunk_size, num_bytes - curr))
        if len(data) == 0:
            raise ConnectionError()
        buffer[curr : curr + len(data)] = data
        curr += len(data)

    if old_timeout is not None:
        socket.settimeout(old_timeout)
    return bytes(buffer)


def print_ui_info(elems: List[UiElement], fname: Optional[str] = None):
    outfile = None if fname is None else open(fname, "w")

    for i, e in enumerate(elems, 1):
        info = []
        info.append(f'class: "{e.name}"')
        for k, v in e.info.items():
            info.append(f'{k}: "{v}"')
        for k, v in e.relation.items():
            info.append(f"{k}: {v}")
        print(f"({i}) {', '.join(info)}", file=outfile)

    if outfile is not None:
        outfile.close()


def request(
    hostname: str,
    port: int,
    fname: str,
    json_file: str,
    chunk_size: int,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    s = sock.socket(sock.AF_INET, sock.SOCK_STREAM)

    try:
        s.connect((hostname, port))

        n = os.path.getsize(fname)
        s.sendall(n.to_bytes(4, "big", signed=False))
        with open(fname, "rb") as f:
            s.sendfile(f)
        n = os.path.getsize(json_file)
        s.sendall(n.to_bytes(4, "big", signed=False))
        with open(json_file, "rb") as f:
            s.sendfile(f)

        n = int.from_bytes(readall(s, 4, chunk_size, timeout), "big", signed=False)
        data = readall(s, n, chunk_size, timeout)
        s.close()

        return json.loads(data.decode("utf-8"))
    except Exception as e:
        s.close()
        raise e


def main(args: argparse.Namespace):
    fname = args.file
    chunk_size = args.chunk_size

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
        hostname, port = config["server"]["hostname"], config["server"]["port"]
    else:
        logging.warn("Configuration file not found. Use the default configuration.")
        hostname, port = HOSTNAME, PORT

    start = time.time()
    results = request(hostname, port, fname, args.json, chunk_size)
    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed} seconds")

    img = np.asarray(Image.open(fname))
    elems = parse_results(img, results)
    plot_ui_elements(img, elems, scale=args.scale)
    print_ui_info(elems)


if __name__ == "__main__":
    logging.basicConfig()
    main(parse_args())
