import argparse
import json
import logging
import os
import socket as sock
import sys
import time
from typing import Dict, List

import numpy as np
from PIL import Image  # type: ignore

sys.path.append(os.path.abspath(".."))

from fluid_ai.base import UiElement
from fluid_ai.utils import plot_ui_elements

from src.utils import parse_results

HOSTNAME = "143.248.136.109"
PORT = 8440


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action="store", required=True)
    parser.add_argument("-t", "--test", action="store_true", default=False)
    parser.add_argument("-j", "--json", action="store", default=None)
    parser.add_argument(
        "--config", action="store", default="config.json", help="Path to a config file"
    )
    parser.add_argument(
        "--chunk_size",
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


def readall(socket: sock.socket, num_bytes: int, chunk_size: int) -> bytes:
    buffer = bytearray(num_bytes)
    curr = 0
    while curr < num_bytes:
        if chunk_size < 0:
            data = socket.recv(num_bytes)
        else:
            data = socket.recv(min(chunk_size, num_bytes - curr))
        buffer[curr : curr + len(data)] = data
        curr += len(data)
    return bytes(buffer)


def print_ui_info(elems: List[UiElement]):
    for i, e in enumerate(elems, 1):
        info = []
        info.append(f'class: "{e.name}"')
        if "icon_label" in e.info:
            info.append(f'label: "{e.info["icon_label"]}"')
        if "text" in e.info:
            info.append(f'text: "{e.info["text"]}"')
        print(f"({i}) {', '.join(info)}")


def test(args: argparse.Namespace):
    raise NotImplementedError()


def main(args: argparse.Namespace):
    fname = args.file
    chunk_size = args.chunk_size

    s = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
        s.connect((config["server"]["hostname"], config["server"]["port"]))
    else:
        logging.warn("Configuration file not found. Use the default configuration.")
        s.connect((HOSTNAME, PORT))

    start = time.time()
    n = os.path.getsize(fname)
    s.sendall(n.to_bytes(4, "big", signed=False))
    with open(fname, "rb") as f:
        s.sendfile(f)
    n = os.path.getsize(args.json)
    s.sendall(n.to_bytes(4, "big", signed=False))
    with open(args.json, "rb") as f:
        s.sendfile(f)

    n = int.from_bytes(readall(s, 4, chunk_size), "big", signed=False)
    data = readall(s, n, chunk_size)
    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed} seconds")

    # Visualize the results
    results = json.loads(data.decode("utf-8"))
    img = np.asarray(Image.open(fname))
    elems = parse_results(img, results)
    plot_ui_elements(img, elems, scale=args.scale)
    print_ui_info(elems)


if __name__ == "__main__":
    logging.basicConfig()
    args = parse_args()
    if args.test:
        test(args)
    else:
        main(args)
