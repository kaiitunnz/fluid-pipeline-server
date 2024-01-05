import logging
import os
import pickle
import socket as sock
import time
from abc import abstractmethod
from PIL import Image
from io import BytesIO
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from fluid_ai.base import UiElement

from src.benchmark import IBenchmarker
from src.logger import ILogger
from src.pipeline import PipelineModule
from src.utils import json_to_ui, readall, ui_to_json

SAVE_IMG_DIR = "res"
SAVE_IMG = False
SOCKET_TIMEOUT = 5


class IPipelineHelper:
    """
    Helper for accessing the UI detection pipeline modules.

    Attributes
    ----------
    logger : ILogger
        Logger to log the UI detection process.
    benchmarker : Optional[IBenchmarker]
        Benchmarker to benchmark the UI detection pipeline server. `None` to not
        benchmark the server.
    textual_elements : List[str]
        List of textual UI class names.
    icon_elements : List[str]
        List of icon UI class names.
    """

    logger: ILogger
    benchmarker: Optional[IBenchmarker]
    textual_elements: List[str]
    icon_elements: List[str]

    def __init__(
        self,
        logger: ILogger,
        benchmarker: Optional[IBenchmarker],
        textual_elements: List[str],
        icon_elements: List[str],
    ):
        """
        Attributes
        ----------
        logger : ILogger
            Logger to log the UI detection process.
        benchmarker : Optional[IBenchmarker]
            Benchmarker to benchmark the UI detection pipeline server. `None` to not
            benchmark the server.
        textual_elements : List[str]
            List of textual UI class names.
        icon_elements : List[str]
            List of icon UI class names.
        """
        self.logger = logger
        self.benchmarker = benchmarker
        self.textual_elements = textual_elements
        self.icon_elements = icon_elements

    @abstractmethod
    def send(self, target: PipelineModule, *args):
        """Sends data to the target UI detection pipeline module

        It does not immediately return the result. Otherwise, `PipelineHelper.wait()`
        must be called to retrieve the result.

        Parameters
        ----------
        target : PipelineModule
            Target UI detection pipeline module.
        *args
            Data to be sent to the module.
        """
        raise NotImplementedError()

    @abstractmethod
    def wait(self, target: PipelineModule) -> List[UiElement]:
        """Waits and gets the result from the target UI detection pipeline module

        This method must be called only after an associated call to the `PipelineHelper.send()`
        method. Otherwise, it will block indefinitely.

        Parameters
        ----------
        target : PipelineModule
            Target UI detection pipeline module.

        Returns
        -------
        List[UiElement]
            Resulting list of UI elements returned from the module.
        """
        raise NotImplementedError()

    @staticmethod
    def save_image_i(img: Image.Image, i: int, save_dir: str, prefix: str = "img"):
        """Saves an image with a filename containing `prefix` and `i`

        Parameters
        ----------
        img : Image
            The image to be saved.
        i : int
            Integer to be included in the file name.
        save_dir : str
            Path to the directory where the image will be saved.
        prefix : str
            File name's prefix. The saved file name will be `prefix` followed by `i`.
        """
        img.save(os.path.join(save_dir, f"{prefix}{i}.jpg"))

    @staticmethod
    def _log(log_func: Callable[[str], None], addr: Tuple[str, int], msg: str):
        """Formats a message with the associated client's IP and port and logs it
        with the given logging function

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

    def log_debug(self, addr: Tuple[str, int], msg: str):
        """Formats a message with the associated client's IP and port and logs it in
        the debug level

        Parameters
        ----------
        addr : Tuple[str, int]
            `(IP address, port)` of the associated client.
        msg : str
            Actual message to be logged.
        """
        self._log(self.logger.debug, addr, msg)

    def log_info(self, addr: Tuple[str, int], msg: str):
        """Formats a message with the associated client's IP and port and logs it in
        the info level

        Parameters
        ----------
        addr : Tuple[str, int]
            `(IP address, port)` of the associated client.
        msg : str
            Actual message to be logged.
        """
        self._log(self.logger.info, addr, msg)

    def log_warning(self, addr: Tuple[str, int], msg: str):
        """Formats a message with the associated client's IP and port and logs it in
        the warning level

        Parameters
        ----------
        addr : Tuple[str, int]
            `(IP address, port)` of the associated client.
        msg : str
            Actual message to be logged.
        """
        self._log(self.logger.warn, addr, msg)

    def log_error(self, addr: Tuple[str, int], msg: str):
        """Formats a message with the associated client's IP and port and logs it in
        the error level

        Parameters
        ----------
        addr : Tuple[str, int]
            `(IP address, port)` of the associated client.
        msg : str
            Actual message to be logged.
        """
        self._log(self.logger.error, addr, msg)

    def log_critical(self, addr: Tuple[str, int], msg: str):
        """Formats a message with the associated client's IP and port and logs it in
        the critical level

        Parameters
        ----------
        addr : Tuple[str, int]
            `(IP address, port)` of the associated client.
        msg : str
            Actual message to be logged.
        """
        self._log(self.logger.critical, addr, msg)

    def process(
        self,
        job_no: int,
        waiting_time: float,
        addr: Tuple[str, int],
        screenshot_img: np.ndarray,
        base_elements: Optional[List[UiElement]],
        test_mode: bool,
    ) -> bytes:
        """Executes the UI detection process

        Parameters
        ----------
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

        Returns
        -------
        bytes
            Result of the process, serialized into UTF-8-encoded JSON format.
        """
        detection_start = time.time()  # bench
        if self.benchmarker is None:
            # Detect UI elements and filter additional UI elements.
            self.log_debug(
                addr, "Detecting UI elements and filtering additional UI elements."
            )
            self.send(PipelineModule.DETECTOR, job_no, screenshot_img)
            self.send(PipelineModule.FILTER, job_no, base_elements)
            detected = self.wait(PipelineModule.DETECTOR)
            filtered = self.wait(PipelineModule.FILTER)
            self.log_debug(addr, f"Found {len(detected)} UI elements.")
            self.log_debug(addr, f"Filtered in {len(filtered)} additional UI elements.")
        else:
            # Detect UI elements.
            self.log_debug(addr, "Detecting UI elements.")
            self.send(PipelineModule.DETECTOR, job_no, screenshot_img)
            detected = self.wait(PipelineModule.DETECTOR)
            detection_time = time.time() - detection_start  # bench
            self.log_debug(addr, f"Found {len(detected)} UI elements.")

            # Filter additional UI elements.
            self.log_debug(addr, "Filtering additional UI elements.")
            filter_start = time.time()  # bench
            self.send(PipelineModule.FILTER, job_no, base_elements)
            filtered = self.wait(PipelineModule.FILTER)
            filter_time = time.time() - filter_start  # bench
            self.log_debug(addr, f"Filtered in {len(filtered)} additional UI elements.")

        if test_mode:
            with open(f"{SAVE_IMG_DIR}/detected_elements{job_no}.pkl", "wb") as f:
                pickle.dump(detected, f)

        # Match UI elements
        self.log_debug(addr, "Matching UI elements.")
        matching_start = time.time()  # bench
        self.send(PipelineModule.MATCHER, job_no, filtered, detected)
        matched = self.wait(PipelineModule.MATCHER)
        matching_time = time.time() - matching_start  # bench
        self.log_debug(addr, f"Matched UI elements. {len(matched)} UI elements left.")

        ui_processing_time = time.time() - detection_start  # bench

        # Partition the result.
        text_elems = []
        icon_elems = []
        results: List[UiElement] = []
        for e in matched:
            if e.name in self.textual_elements:
                text_elems.append(e)
            elif e.name in self.icon_elements:
                icon_elems.append(e)
            else:
                results.append(e)

        # Extract UI info.
        self.log_debug(addr, "Extracting UI info.")
        if self.benchmarker is None:
            self.send(PipelineModule.TEXT_RECOGNIZER, job_no, text_elems)
            self.send(PipelineModule.ICON_LABELER, job_no, icon_elems)
            results.extend(self.wait(PipelineModule.TEXT_RECOGNIZER))
            results.extend(self.wait(PipelineModule.ICON_LABELER))
        else:
            text_start = time.time()  # bench
            self.send(PipelineModule.TEXT_RECOGNIZER, job_no, text_elems)
            results.extend(self.wait(PipelineModule.TEXT_RECOGNIZER))
            text_time = time.time() - text_start  # bench
            icon_start = time.time()  # bench
            self.send(PipelineModule.ICON_LABELER, job_no, icon_elems)
            results.extend(self.wait(PipelineModule.ICON_LABELER))
            icon_time = time.time() - icon_start  # bench

        processing_time = time.time() - detection_start  # bench
        if self.benchmarker is None:
            results_json = ui_to_json(screenshot_img, results).encode("utf-8")
        else:
            entry = [waiting_time, detection_time, filter_time, matching_time, ui_processing_time, text_time, icon_time, processing_time]  # type: ignore
            self.benchmarker.add(entry)
            metrics = {"keys": self.benchmarker.metrics, "values": entry}
            results_json = ui_to_json(screenshot_img, results, metrics=metrics).encode(
                "utf-8"
            )

        return results_json

    def serve(
        self,
        job_no: int,
        start_time: float,
        conn: sock.socket,
        chunk_size: int,
        max_image_size: int,
        test_mode: bool,
    ):
        """Handles a job/connection, essentially serving the UI detection pipeline

        Parameters
        ----------
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
            self.logger.log(
                logging.ERROR,
                f"The following exception occurred while attempting to connect: {e}",
            )
            return

        try:
            # Receive a screenshot.
            payload_size = int.from_bytes(
                readall(conn, 4, chunk_size), "big", signed=False
            )
            self.log_debug(
                addr, f"Receiving a screenshot payload of size {payload_size} bytes."
            )
            if payload_size > max_image_size:
                self.log_error(
                    addr, f"The payload size exceeds the maximum allowable size."
                )
                conn.close()
                return
            screenshot_payload = readall(conn, payload_size, chunk_size)
            self.log_debug(
                addr,
                f"Received a screenshot payload of size {len(screenshot_payload)} bytes.",
            )

            # Receive additional UI elements.
            payload_size = int.from_bytes(
                readall(conn, 4, chunk_size), "big", signed=False
            )
            if payload_size > 0:
                self.log_debug(
                    addr,
                    f"Receiving a UI-element payload of size {payload_size} bytes.",
                )
                element_payload = readall(conn, payload_size, chunk_size)
                self.log_debug(
                    addr,
                    f"Received a UI-element payload of size {len(element_payload)} bytes.",
                )
            else:
                element_payload = None

            # Parse the screenshot payload.
            screenshot_bytes = BytesIO(screenshot_payload)
            img = Image.open(screenshot_bytes)
            if SAVE_IMG:
                self.__class__.save_image_i(img, job_no, SAVE_IMG_DIR)
            screenshot_img = np.asarray(img)
            self.log_debug(addr, f"Received an image of shape {screenshot_img.shape}.")

            # Parse the UI-element payload.
            base_elements = (
                None
                if element_payload is None
                else json_to_ui(
                    element_payload.decode(encoding="utf-8"), screenshot_img
                )
            )
            if base_elements is not None:
                self.log_debug(
                    addr, f"Received {len(base_elements)} additional UI elements."
                )
                if test_mode:
                    with open(f"{SAVE_IMG_DIR}/base_elements{job_no}.pkl", "wb") as f:
                        pickle.dump(base_elements, f)

            # Process the screenshot and additional UI elements.
            results_json = self.process(
                job_no,
                waiting_time,
                addr,
                screenshot_img,
                base_elements,
                test_mode,
            )

            self.log_debug(
                addr,
                f"Sending back the response of size {len(results_json)} bytes.",
            )

            conn.sendall(len(results_json).to_bytes(4, "big", signed=False))
            self.log_debug(addr, f"Sent response size: {len(results_json)}")
            conn.sendall(results_json)
            self.log_info(addr, "Response sent.")
        except Exception as e:
            self.log_error(addr, f"The following exception occurred: {e}")
        finally:
            self.log_info(addr, "Connection closed.")
            conn.close()

    def warmup(
        self,
        warmup_image: str,
        name: str = "Pipeline",
        on_error: Optional[Callable[[str, Any], None]] = None,
    ) -> bool:
        """Warms up the UI detection pipeline and performs initial testing

        Parameters
        ----------
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

        self.logger.debug(f"[{name}] Warming up the pipeline...")
        img = np.asarray(Image.open(warmup_image))

        # Detect UI elements.
        self.send(PipelineModule.DETECTOR, job_no, img)
        detected = self.wait(PipelineModule.DETECTOR)
        self.logger.debug(f"[{name}] ({PipelineModule.DETECTOR.value}) PASSED.")

        # Filter UI elements.
        self.send(PipelineModule.FILTER, job_no, detected)
        filtered = self.wait(PipelineModule.FILTER)
        self.logger.debug(f"[{name}] ({PipelineModule.FILTER.value}) PASSED.")

        # Match UI elements.
        self.send(PipelineModule.MATCHER, job_no, filtered, filtered)
        matched = self.wait(PipelineModule.MATCHER)
        if len(matched) == len(filtered):
            self.logger.debug(f"[{name}] ({PipelineModule.MATCHER.value}) PASSED.")
        else:
            success = False
            self.logger.debug(f"[{name}] ({PipelineModule.MATCHER.value}) FAILED.")
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
            if e.name in self.textual_elements:
                text_elems.append(e)
            elif e.name in self.icon_elements:
                icon_elems.append(e)

        # Extract UI info.
        self.send(PipelineModule.TEXT_RECOGNIZER, job_no, text_elems)
        self.send(PipelineModule.ICON_LABELER, job_no, icon_elems)
        self.wait(PipelineModule.TEXT_RECOGNIZER)
        self.wait(PipelineModule.ICON_LABELER)
        self.logger.debug(f"[{name}] ({PipelineModule.TEXT_RECOGNIZER.value}) PASSED.")
        self.logger.debug(f"[{name}] ({PipelineModule.ICON_LABELER.value}) PASSED.")

        return success
