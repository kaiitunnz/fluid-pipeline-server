import json
import logging
import multiprocessing
import socket as sock
from io import BytesIO
from multiprocessing.connection import Connection
from multiprocessing.pool import Pool
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
from fluid_ai.base import UiElement
from fluid_ai.pipeline import UiDetectionPipeline
from PIL import Image

from src._multiprocessing.manager import PipelineManager, Screenshot

logger = logging.getLogger("pipeline_server")


def _handle_connection(
    conn: sock.socket,
    detector_pipe: Connection,
    text_recognizer_pipe: Connection,
    icon_labeller_pipe: Connection,
    textual_elements: List[str],
    icon_elements: List[str],
    chunk_size: int,
    max_image_size: int,
    verbose: bool,
):
    packet_size = int.from_bytes(_readall(conn, 4, chunk_size), "big", signed=False)
    if packet_size > max_image_size:
        conn.close()

    if verbose:
        print("Receiving a packet of size:", packet_size)
    packet = BytesIO(conn.recv(packet_size))
    screenshot_img = np.asarray(Image.open(packet), dtype=np.double)
    if verbose:
        print("Received an image of shape:", screenshot_img.shape)
    screenshot = Screenshot(screenshot_img)

    # Detect UI elements.
    if verbose:
        print("Detecting UI elements...")
    receiver, sender = multiprocessing.Pipe(False)
    print("Pipe created...")
    detector_pipe.send((sender, (screenshot,)))
    print("Message sent...")
    detected: List[UiElement] = receiver.recv()
    if verbose:
        print(f"Found {len(detected)} UI elements.")

    # Partition the result.
    text_elems = []
    icon_elems = []
    results: List[UiElement] = []
    for e in detected:
        if e.name in textual_elements:
            text_elems.append(e)
        elif e.name in icon_elements:
            icon_elems.append(e)
        else:
            results.append(e)

    # Extract UI info.
    text_receiver, text_sender = multiprocessing.Pipe(False)
    text_recognizer_pipe.send((text_sender, (text_elems, screenshot)))
    icon_receiver, icon_sender = multiprocessing.Pipe(False)
    icon_labeller_pipe.send((icon_sender, (icon_elems, screenshot)))
    results.extend(text_receiver.recv())
    results.extend(icon_receiver.recv())

    results_json = _ui_to_json(screenshot_img, results).encode("utf-8")

    if verbose:
        print("Sending back the results:", results_json)

    conn.send(len(results_json).to_bytes(4, "big", signed=False))
    conn.send(results_json)


def _readall(socket: sock.socket, num_bytes: int, chunk_size: int) -> bytes:
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


def _ui_to_json(screenshot_img: np.ndarray, elems: List[UiElement]) -> str:
    h, w, *_ = screenshot_img.shape
    data = {"img_shape": [w, h], "elements": [_elem_to_dict(e) for e in elems]}
    return json.dumps(data)


def _elem_to_dict(elem: UiElement) -> Dict[str, Any]:
    (x0, y0), (x1, y1) = elem.bbox
    return {
        "class": elem.name,
        "position": {
            "x_min": x0,
            "y_min": y0,
            "x_max": x1,
            "y_max": y1,
        },
        "info": elem.info,
    }


class PipelineServer:
    hostname: str
    port: str
    pipeline: UiDetectionPipeline
    chunk_size: int
    max_image_size: int
    num_workers: int
    socket: Optional[sock.socket] = None
    manager: PipelineManager
    verbose: bool

    def __init__(
        self,
        *,
        hostname: str,
        port: str,
        pipeline: UiDetectionPipeline,
        chunk_size: int = -1,
        max_image_size: int = -1,
        num_workers: int = 4,
        verbose: bool = True,
    ):
        self.hostname = hostname
        self.port = port
        self.pipeline = pipeline
        self.chunk_size = chunk_size
        self.max_image_size = max_image_size
        self.num_workers = num_workers
        self.manager = PipelineManager(pipeline)
        self.verbose = verbose

        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        multiprocessing.set_start_method("forkserver")

    def start(self):
        self.socket = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
        self.socket.bind((self.hostname, self.port))
        self.socket.listen(1)
        self.manager.start()

        if self.verbose:
            print("Pipeline server started...")

        with Pool(processes=self.num_workers) as pool:
            while True:
                conn, addr = self.socket.accept()
                if self.verbose:
                    print(f"Got connection from {addr}")
                pool.apply_async(
                    _handle_connection,
                    args=(
                        conn,
                        self.manager.detector_pipe,
                        self.manager.text_recognizer_pipe,
                        self.manager.icon_labeller_pipe,
                        self.manager.pipeline.textual_elements,
                        self.manager.pipeline.icon_elements,
                        self.chunk_size,
                        self.max_image_size,
                        self.verbose,
                    ),
                )
