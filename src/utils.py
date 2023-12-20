import json
import logging
import socket as sock
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from fluid_ai.base import Array, BBox, NormalizedBBox, UiElement, array_get_size


def parse_results(img: np.ndarray, results: Dict[str, Any]) -> List[UiElement]:
    elems = []
    for elem in results["elements"]:
        name = elem["class"]
        position = elem["position"]
        nbox = NormalizedBBox.new(
            (position["x_min"], position["y_min"]),
            (position["x_max"], position["y_max"]),
            unchecked=True,
        )
        info = elem["info"]
        ui_element = UiElement(name, nbox, img)
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


def json_to_ui(json_elements: str, screenshot: Array) -> List[UiElement]:
    elements: List[Dict[str, Any]] = json.loads(json_elements)
    result = []
    w, h = array_get_size(screenshot)
    for elem in elements:
        name = elem["class"]
        position = elem["position"]
        info = elem["info"]
        nbox = BBox(
            (position["x_min"], position["y_min"]),
            (position["x_max"], position["y_max"]),
        ).to_normalized(w, h, unchecked=True)
        assert nbox is not None
        result.append(
            UiElement(
                name=name,
                bbox=nbox,
                screenshot=screenshot,
                info=info,
            )
        )
    return result


def ui_to_json(screenshot: Array, elems: List[UiElement], **kwargs) -> str:
    h, w, *_ = screenshot.shape
    data = {"img_shape": [w, h], "elements": [_elem_to_dict(e) for e in elems]}
    data.update(kwargs)
    return json.dumps(data)


def _elem_to_dict(elem: UiElement) -> Dict[str, Any]:
    w, h = array_get_size(elem.get_screenshot())
    (x0, y0), (x1, y1) = elem.bbox.to_bbox(w, h)
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
