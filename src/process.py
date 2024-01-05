import logging
import pickle
import socket as sock
import time
from PIL import Image
from io import BytesIO
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import numpy as np
from fluid_ai.base import UiElement

from src.pipeline import IPipelineHelper, PipelineModule
from src.utils import json_to_ui, readall, ui_to_json

SAVE_IMG_DIR = "res"
SAVE_IMG = False
SOCKET_TIMEOUT = 5


class UiDetectionArgs(NamedTuple):
    """
    Arguments to a UI detection function performing the UI detection process.

    Parameters
    ----------
    helper : PipelineHelper
        Helper for accessing the UI detection pipeline modules.
    job_no : int
        Job number, used to identify the job.
    waiting_time : float
        Time for which the client waits until its request gets handled.
    addr : Tuple[str, int]
        Client's IP address.
    screenshot_img : ndarray
        Screenshot to be processed.
    base_elements : Optional[List[UiElement]]
        Base elements, aka additional UI elements.
    test_mode : bool
        Whether to handle connections in test mode.
    """

    helper: IPipelineHelper
    job_no: int
    waiting_time: float
    addr: Tuple[str, int]
    screenshot_img: np.ndarray
    base_elements: Optional[List[UiElement]]
    test_mode: bool


UiDetectionProcess = Callable[[UiDetectionArgs], bytes]
"""UI detection function performing the UI detection process.

Parameters
----------
args : UiDetectionArgs
    Arguments. See `UiDetectionArgs` for more details.

Returns
-------
bytes
    Result of the process, serialized into UTF-8-encoded JSON format.
"""


def ui_detection_process(args: UiDetectionArgs) -> bytes:
    """Performs the UI detection process

    This is the default implementation of the UI detection process.

    Parameters
    ----------
    args : UiDetectionArgs
        Arguments. See `UiDetectionArgs` for more details.

    Returns
    -------
    bytes
        Result of the process, serialized into UTF-8-encoded JSON format.
    """
    helper, job_no, waiting_time, addr, screenshot_img, base_elements, test_mode = args

    detection_start = time.time()  # bench
    if helper.benchmarker is None:
        # Detect UI elements and filter additional UI elements.
        helper.log_debug(
            addr, "Detecting UI elements and filtering additional UI elements."
        )
        helper.send(PipelineModule.DETECTOR, job_no, screenshot_img)
        helper.send(PipelineModule.FILTER, job_no, base_elements)
        detected = helper.wait(PipelineModule.DETECTOR)
        filtered = helper.wait(PipelineModule.FILTER)
        helper.log_debug(addr, f"Found {len(detected)} UI elements.")
        helper.log_debug(addr, f"Filtered in {len(filtered)} additional UI elements.")
    else:
        # Detect UI elements.
        helper.log_debug(addr, "Detecting UI elements.")
        helper.send(PipelineModule.DETECTOR, job_no, screenshot_img)
        detected = helper.wait(PipelineModule.DETECTOR)
        detection_time = time.time() - detection_start  # bench
        helper.log_debug(addr, f"Found {len(detected)} UI elements.")

        # Filter additional UI elements.
        helper.log_debug(addr, "Filtering additional UI elements.")
        filter_start = time.time()  # bench
        helper.send(PipelineModule.FILTER, job_no, base_elements)
        filtered = helper.wait(PipelineModule.FILTER)
        filter_time = time.time() - filter_start  # bench
        helper.log_debug(addr, f"Filtered in {len(filtered)} additional UI elements.")

    if test_mode:
        with open(f"{SAVE_IMG_DIR}/detected_elements{job_no}.pkl", "wb") as f:
            pickle.dump(detected, f)

    # Match UI elements
    helper.log_debug(addr, "Matching UI elements.")
    matching_start = time.time()  # bench
    helper.send(PipelineModule.MATCHER, job_no, filtered, detected)
    matched = helper.wait(PipelineModule.MATCHER)
    matching_time = time.time() - matching_start  # bench
    helper.log_debug(addr, f"Matched UI elements. {len(matched)} UI elements left.")

    ui_processing_time = time.time() - detection_start  # bench

    # Partition the result.
    text_elems = []
    icon_elems = []
    results: List[UiElement] = []
    for e in matched:
        if e.name in helper.textual_elements:
            text_elems.append(e)
        elif e.name in helper.icon_elements:
            icon_elems.append(e)
        else:
            results.append(e)

    # Extract UI info.
    helper.log_debug(addr, "Extracting UI info.")
    if helper.benchmarker is None:
        helper.send(PipelineModule.TEXT_RECOGNIZER, job_no, text_elems)
        helper.send(PipelineModule.ICON_LABELER, job_no, icon_elems)
        results.extend(helper.wait(PipelineModule.TEXT_RECOGNIZER))
        results.extend(helper.wait(PipelineModule.ICON_LABELER))
    else:
        text_start = time.time()  # bench
        helper.send(PipelineModule.TEXT_RECOGNIZER, job_no, text_elems)
        results.extend(helper.wait(PipelineModule.TEXT_RECOGNIZER))
        text_time = time.time() - text_start  # bench
        icon_start = time.time()  # bench
        helper.send(PipelineModule.ICON_LABELER, job_no, icon_elems)
        results.extend(helper.wait(PipelineModule.ICON_LABELER))
        icon_time = time.time() - icon_start  # bench

    processing_time = time.time() - detection_start  # bench
    if helper.benchmarker is None:
        results_json = ui_to_json(screenshot_img, results).encode("utf-8")
    else:
        entry = [waiting_time, detection_time, filter_time, matching_time, ui_processing_time, text_time, icon_time, processing_time]  # type: ignore
        helper.benchmarker.add(entry)
        metrics = {"keys": helper.benchmarker.metrics, "values": entry}
        results_json = ui_to_json(screenshot_img, results, metrics=metrics).encode(
            "utf-8"
        )

    return results_json


def ui_detection_serve(
    helper: IPipelineHelper,
    job_no: int,
    start_time: float,
    conn: sock.socket,
    chunk_size: int,
    max_image_size: int,
    test_mode: bool,
    process: UiDetectionProcess = ui_detection_process,
):
    """Handles a job/connection, essentially serving the UI detection pipeline

    Parameters
    ----------
    helper : PipelineHelper
        Helper for accessing the UI detection pipeline modules.
    job_no : int
        Job number, used to identify the job.
    start_time : float
        Time at which the connection is accepted.
    conn : socket
        Socket for an accepted connection.
    chunk_size : int
        Chunk size for reading bytes from the sockets.
    max_image_size : int
        Maximum size of an image from the client.
    test_mode : bool
        Whether to handle connections in test mode.
    process : UiDetectionProcess
        Function implementing the UI detection process.
    """
    waiting_time = time.time() - start_time  # bench

    try:
        addr = conn.getpeername()
        conn.settimeout(SOCKET_TIMEOUT)
    except Exception as e:
        helper.logger.log(
            logging.ERROR,
            f"The following exception occurred while attempting to connect: {e}",
        )
        return

    try:
        # Receive a screenshot.
        payload_size = int.from_bytes(readall(conn, 4, chunk_size), "big", signed=False)
        helper.log_debug(
            addr, f"Receiving a screenshot payload of size {payload_size} bytes."
        )
        if payload_size > max_image_size:
            helper.log_error(
                addr, f"The payload size exceeds the maximum allowable size."
            )
            conn.close()
            return
        screenshot_payload = readall(conn, payload_size, chunk_size)
        helper.log_debug(
            addr,
            f"Received a screenshot payload of size {len(screenshot_payload)} bytes.",
        )

        # Receive additional UI elements.
        payload_size = int.from_bytes(readall(conn, 4, chunk_size), "big", signed=False)
        if payload_size > 0:
            helper.log_debug(
                addr,
                f"Receiving a UI-element payload of size {payload_size} bytes.",
            )
            element_payload = readall(conn, payload_size, chunk_size)
            helper.log_debug(
                addr,
                f"Received a UI-element payload of size {len(element_payload)} bytes.",
            )
        else:
            element_payload = None

        # Parse the screenshot payload.
        screenshot_bytes = BytesIO(screenshot_payload)
        img = Image.open(screenshot_bytes)
        if SAVE_IMG:
            helper.__class__.save_image_i(img, job_no, SAVE_IMG_DIR)
        screenshot_img = np.asarray(img)
        helper.log_debug(addr, f"Received an image of shape {screenshot_img.shape}.")

        # Parse the UI-element payload.
        base_elements = (
            None
            if element_payload is None
            else json_to_ui(element_payload.decode(encoding="utf-8"), screenshot_img)
        )
        if base_elements is not None:
            helper.log_debug(
                addr, f"Received {len(base_elements)} additional UI elements."
            )
            if test_mode:
                with open(f"{SAVE_IMG_DIR}/base_elements{job_no}.pkl", "wb") as f:
                    pickle.dump(base_elements, f)

        # Process the screenshot and additional UI elements.
        results_json = process(
            UiDetectionArgs(
                helper,
                job_no,
                waiting_time,
                addr,
                screenshot_img,
                base_elements,
                test_mode,
            )
        )

        helper.log_debug(
            addr,
            f"Sending back the response of size {len(results_json)} bytes.",
        )

        conn.sendall(len(results_json).to_bytes(4, "big", signed=False))
        helper.log_debug(addr, f"Sent response size: {len(results_json)}")
        conn.sendall(results_json)
        helper.log_info(addr, "Response sent.")
    except Exception as e:
        helper.log_error(addr, f"The following exception occurred: {e}")
    finally:
        helper.log_info(addr, "Connection closed.")
        conn.close()


def ui_detection_warmup(
    helper: IPipelineHelper,
    warmup_image: str,
    name: str = "Pipeline",
    on_error: Optional[Callable[[str, Any], None]] = None,
) -> bool:
    """Warms up the UI detection pipeline and performs initial testing

    Parameters
    ----------
    helper : PipelineHelper
        Helper used to access the UI detection pipeline modules.
    warmup_image : str
        Path to the image for warming up the UI detection pipeline and performing
        initial testing.
    name : str
        Name to identify the process in the server log.
    on_error : Optional[Callable[[str, Any], None]]
        Function to be called on error events. It should expect two arguments: an
        error message, `message`, and additional information about the error, `info`.

    Returns
    -------
    bool
        Whether the warming up process is successful.
    """
    success = True
    job_no = 0

    helper.logger.debug(f"[{name}] Warming up the pipeline...")
    img = np.asarray(Image.open(warmup_image))

    # Detect UI elements.
    helper.send(PipelineModule.DETECTOR, job_no, img)
    detected = helper.wait(PipelineModule.DETECTOR)
    helper.logger.debug(f"[{name}] ({PipelineModule.DETECTOR.value}) PASSED.")

    # Filter UI elements.
    helper.send(PipelineModule.FILTER, job_no, detected)
    filtered = helper.wait(PipelineModule.FILTER)
    helper.logger.debug(f"[{name}] ({PipelineModule.FILTER.value}) PASSED.")

    # Match UI elements.
    helper.send(PipelineModule.MATCHER, job_no, filtered, filtered)
    matched = helper.wait(PipelineModule.MATCHER)
    if len(matched) == len(filtered):
        helper.logger.debug(f"[{name}] ({PipelineModule.MATCHER.value}) PASSED.")
    else:
        success = False
        helper.logger.debug(f"[{name}] ({PipelineModule.MATCHER.value}) FAILED.")
        if on_error is not None:
            on_error(
                "Failed to initialize the pipeline.",
                {
                    "detected": len(detected),
                    "filtered": len(filtered),
                    "matched": len(matched),
                },
            )

    # Partition the result.
    text_elems = []
    icon_elems = []
    for e in matched:
        if e.name in helper.textual_elements:
            text_elems.append(e)
        elif e.name in helper.icon_elements:
            icon_elems.append(e)

    # Extract UI info.
    helper.send(PipelineModule.TEXT_RECOGNIZER, job_no, text_elems)
    helper.send(PipelineModule.ICON_LABELER, job_no, icon_elems)
    helper.wait(PipelineModule.TEXT_RECOGNIZER)
    helper.wait(PipelineModule.ICON_LABELER)
    helper.logger.debug(f"[{name}] ({PipelineModule.TEXT_RECOGNIZER.value}) PASSED.")
    helper.logger.debug(f"[{name}] ({PipelineModule.ICON_LABELER.value}) PASSED.")

    return success
