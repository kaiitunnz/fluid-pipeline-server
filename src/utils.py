import json
import logging
import socket as sock
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from fluid_ai.base import Array, BBox, NormalizedBBox, UiElement, array_get_size


def parse_results(img: np.ndarray, results: Dict[str, Any]) -> List[UiElement]:
    """Parses the results of the UI detection pipeline server and initializes detected
    UI elements

    Parameters
    ----------
    img : ndarray
        Image input to the server.
    results : Dict[str, Any]
        Results returned from the server.

    Returns
    -------
    List[UiElement]
        List of UI elements detected by the server.
    """
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
    """Reads the specified number of bytes from the socket

    Parameters
    ----------
    socket : socket
        Socket to read from.
    num_bytes : int
        Total number of bytes to be read.
    chunk_size : int
        Maximum number of bytes to be read at a time.

    Returns
    -------
    bytes
        Read bytes whose length is guaranteed to equal `num_bytes`.
    """
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
    """Initializes UI elements from serialized objects in JSON format

    Parameters
    ----------
    json_elements : str
        Serialized UI elements in JSON format.
    screenshot : Array
        Screenshot associated with the UI elements.

    Returns
    -------
    List[UiElement]
        List of the initialized UI elements.
    """
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
        ).to_normalized_unchecked(w, h)
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
    """Serializes UI elements in JSON format

    Parameters
    ----------
    screenshot : Array
        Screenshot associated with the UI elements.
    elems : List[UiElement]
        List of UI elements.
    **kwargs
        Additional fields to be added to the JSON object.

    Returns
    -------
    str
        Resulting JSON string.
    """
    h, w, *_ = screenshot.shape
    data = {"img_shape": [w, h], "elements": [_elem_to_dict(e) for e in elems]}
    data.update(kwargs)
    return json.dumps(data)


def _elem_to_dict(elem: UiElement) -> Dict[str, Any]:
    """Converts a UI element to a dictionary

    Parameters
    ----------
    elem : UiElement
        UI element to be converted.

    Returns
    -------
    Dict[str, Any]
        Resulting dictionary.
    """
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
    """Formats a message with the associated client's IP and port and logs it with
    the given logging function

    Parameters
    ----------
    log_func : Callable[[str], None]
        Logging function that takes a string to be logged.
    addr : Tuple[str, int]
        `(IP address, port)` of the associated client.
    msg : str
        Actual message to be logged.
    """
    ip, port = addr
    log_func(f"({ip}:{port}) {msg}")


def log_debug(logger: logging.Logger, addr: Tuple[str, int], msg: str):
    """Formats a message with the associated client's IP and port and logs it in
    the debug level

    Parameters
    ----------
    logger : Logger
        Logger.
    addr : Tuple[str, int]
        `(IP address, port)` of the associated client.
    msg : str
        Actual message to be logged.
    """
    _log(logger.debug, addr, msg)


def log_info(logger: logging.Logger, addr: Tuple[str, int], msg: str):
    """Formats a message with the associated client's IP and port and logs it in
    the info level

    Parameters
    ----------
    logger : Logger
        Logger.
    addr : Tuple[str, int]
        `(IP address, port)` of the associated client.
    msg : str
        Actual message to be logged.
    """
    _log(logger.info, addr, msg)


def log_warning(logger: logging.Logger, addr: Tuple[str, int], msg: str):
    """Formats a message with the associated client's IP and port and logs it in
    the warning level

    Parameters
    ----------
    logger : Logger
        Logger.
    addr : Tuple[str, int]
        `(IP address, port)` of the associated client.
    msg : str
        Actual message to be logged.
    """
    _log(logger.warn, addr, msg)


def log_error(logger: logging.Logger, addr: Tuple[str, int], msg: str):
    """Formats a message with the associated client's IP and port and logs it in
    the error level

    Parameters
    ----------
    logger : Logger
        Logger.
    addr : Tuple[str, int]
        `(IP address, port)` of the associated client.
    msg : str
        Actual message to be logged.
    """
    _log(logger.error, addr, msg)


def log_critical(logger: logging.Logger, addr: Tuple[str, int], msg: str):
    """Formats a message with the associated client's IP and port and logs it in
    the critical level

    Parameters
    ----------
    logger : Logger
        Logger.
    addr : Tuple[str, int]
        `(IP address, port)` of the associated client.
    msg : str
        Actual message to be logged.
    """
    _log(logger.critical, addr, msg)
