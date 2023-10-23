import logging
import socket as sock
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from fluid_ai.base import UiElement


def parse_results(img: np.ndarray, results: Dict[str, Any]) -> List[UiElement]:
    elems = []
    for elem in results["elements"]:
        name = elem["class"]
        position = elem["position"]
        bbox = (
            (position["x_min"], position["y_min"]),
            (position["x_max"], position["y_max"]),
        )
        info = elem["info"]
        ui_element = UiElement(name, bbox, img)
        ui_element.info = info
        elems.append(ui_element)
    return elems


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


def _log(log_func: Callable[[str], None], addr: Tuple[str, int], msg: str):
    ip, port = addr
    log_func(f"({ip}:{port}) {msg}")


def log_debug(logger: logging.Logger, addr: Tuple[str, int], msg: str):
    _log(logger.debug, addr, msg)


def log_info(logger: logging.Logger, addr: Tuple[str, int], msg: str):
    _log(logger.info, addr, msg)


def log_warning(logger: logging.Logger, addr: Tuple[str, int], msg: str):
    _log(logger.warn, addr, msg)


def log_error(logger: logging.Logger, addr: Tuple[str, int], msg: str):
    _log(logger.error, addr, msg)


def log_critical(logger: logging.Logger, addr: Tuple[str, int], msg: str):
    _log(logger.critical, addr, msg)
