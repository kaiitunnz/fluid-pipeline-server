import multiprocessing as mp
import numpy as np
import os
import pickle
import signal
import socket as sock
import sys
import threading
import time
from PIL import Image  # type: ignore
from io import BytesIO
from multiprocessing.queues import SimpleQueue
from multiprocessing.synchronize import Semaphore
from queue import Queue
from threading import Thread
from typing import Any, List, Optional

from fluid_ai.base import UiElement

from src.constructor import PipelineConstructor
from src.hybrid.benchmark import Benchmarker
from src.hybrid.helper import PipelineHelper
from src.hybrid.logging import Logger
from src.hybrid.manager import PipelineManager
from src.utils import json_to_ui, readall, ui_to_json

SAVE_IMG_DIR = "res"
SAVE_IMG = False


class _HandlerHelper:
    helper: PipelineHelper
    chunk_size: int
    max_image_size: int
    job_queue: Queue
    logger: Logger
    test_mode: bool

    _workers: List[Thread]

    def __init__(
        self,
        helper: PipelineHelper,
        chunk_size: int,
        max_image_size: int,
        num_workers: int,
        logger: Logger,
        name: str,
        test_mode: bool,
    ):
        self.helper = helper
        self.chunk_size = chunk_size
        self.max_image_size = max_image_size
        self.job_queue = Queue(num_workers)
        self.logger = logger
        self.name = name
        self.test_mode = test_mode

        self._workers = [
            Thread(target=self._serve, args=(self.job_queue,), daemon=False)
            for _ in range(num_workers)
        ]

    def start(self):
        for worker in self._workers:
            worker.start()

    def _serve(self, job_queue: Queue):
        while True:
            job = job_queue.get()
            if job is None:
                break
            self._handle_connection(*job)

    def _handle_connection(
        self,
        job_no: int,
        start_time: float,
        conn: sock.socket,
    ):
        waiting_time = time.time() - start_time  # bench

        try:
            addr = conn.getpeername()
        except Exception as e:
            self.helper.logger.error(
                f"The following exception occurred while attempting to connect: {e}"
            )
            return

        try:
            # Receive a screenshot.
            payload_size = int.from_bytes(
                readall(conn, 4, self.chunk_size), "big", signed=False
            )
            self.helper.log_debug(
                addr, f"Receiving a screenshot payload of size {payload_size} bytes."
            )
            if payload_size > self.max_image_size:
                self.helper.log_error(
                    addr, f"The payload size exceeds the maximum allowable size."
                )
                conn.close()
                return
            screenshot_payload = readall(conn, payload_size, self.chunk_size)
            self.helper.log_debug(
                addr,
                f"Received a screenshot payload of size {len(screenshot_payload)} bytes.",
            )

            # Receive additional UI elements.
            payload_size = int.from_bytes(
                readall(conn, 4, self.chunk_size), "big", signed=False
            )
            if payload_size > 0:
                self.helper.log_debug(
                    addr,
                    f"Receiving a UI-element payload of size {payload_size} bytes.",
                )
                element_payload = readall(conn, payload_size, self.chunk_size)
                self.helper.log_debug(
                    addr,
                    f"Received a UI-element payload of size {len(element_payload)} bytes.",
                )
            else:
                element_payload = None

            # Parse the screenshot payload.
            screenshot_bytes = BytesIO(screenshot_payload)
            img = Image.open(screenshot_bytes)
            if SAVE_IMG:
                self.helper.save_image_i(img, job_no, SAVE_IMG_DIR)
            screenshot_img = np.asarray(img)
            self.helper.log_debug(
                addr, f"Received an image of shape {screenshot_img.shape}."
            )

            # Parse the UI-element payload.
            base_elements = (
                None
                if element_payload is None
                else json_to_ui(
                    element_payload.decode(encoding="utf-8"), screenshot_img
                )
            )
            if base_elements is not None:
                self.helper.log_debug(
                    addr, f"Received {len(base_elements)} additional UI elements."
                )
                if self.test_mode:
                    with open(f"{SAVE_IMG_DIR}/base_elements{job_no}.pkl", "wb") as f:
                        pickle.dump(base_elements, f)

            # Detect UI elements.
            self.helper.log_debug(addr, "Detecting UI elements.")
            detection_start = time.time()  # bench
            self.helper.send(PipelineConstructor.DETECTOR, job_no, screenshot_img)
            detected = self.helper.wait_result()
            detection_time = time.time() - detection_start  # bench
            self.helper.log_debug(addr, f"Found {len(detected)} UI elements.")

            if self.test_mode:
                with open(f"{SAVE_IMG_DIR}/detected_elements{job_no}.pkl", "wb") as f:
                    pickle.dump(detected, f)

            # Match UI elements
            self.helper.log_debug(addr, "Matching UI elements.")
            matching_start = time.time()  # bench
            self.helper.send(
                PipelineConstructor.MATCHER, job_no, base_elements, detected
            )
            matched = self.helper.wait_result()
            matching_time = time.time() - matching_start  # bench
            self.helper.log_debug(
                addr, f"Matched UI elements. {len(matched)} UI elements left."
            )

            # Partition the result.
            text_elems = []
            icon_elems = []
            results: List[UiElement] = []
            for e in matched:
                if e.name in self.helper.textual_elements:
                    text_elems.append(e)
                elif e.name in self.helper.icon_elements:
                    icon_elems.append(e)
                else:
                    results.append(e)

            # Extract UI info.
            self.helper.log_debug(addr, "Extracting UI info.")
            if self.helper.benchmarker is None:
                self.helper.send(
                    PipelineConstructor.TEXT_RECOGNIZER, job_no, text_elems
                )
                self.helper.send(PipelineConstructor.ICON_LABELLER, job_no, icon_elems)
                results.extend(self.helper.wait_result())
                results.extend(self.helper.wait_result())
            else:
                text_start = time.time()  # bench
                self.helper.send(
                    PipelineConstructor.TEXT_RECOGNIZER, job_no, text_elems
                )
                results.extend(self.helper.wait_result())
                text_time = time.time() - text_start  # bench
                icon_start = time.time()  # bench
                self.helper.send(PipelineConstructor.ICON_LABELLER, job_no, icon_elems)
                results.extend(self.helper.wait_result())
                icon_time = time.time() - icon_start  # bench

            processing_time = time.time() - detection_start  # bench
            if self.helper.benchmarker is None:
                results_json = ui_to_json(screenshot_img, results).encode("utf-8")
            else:
                entry = [waiting_time, detection_time, matching_time, text_time, icon_time, processing_time]  # type: ignore
                self.helper.benchmarker.add(entry)
                metrics = {"keys": self.helper.benchmarker.metrics, "values": entry}
                results_json = ui_to_json(
                    screenshot_img, results, metrics=metrics
                ).encode("utf-8")

            self.helper.log_debug(
                addr,
                f"Sending back the response of size {len(results_json)} bytes.",
            )

            conn.sendall(len(results_json).to_bytes(4, "big", signed=False))
            self.helper.log_debug(addr, f"Sent response size: {len(results_json)}")
            conn.sendall(results_json)
            self.helper.log_info(addr, "Response sent.")
        except Exception as e:
            self.helper.log_error(addr, f"The following exception occurred: {e}")
        finally:
            self.helper.log_info(addr, "Connection closed.")
            conn.close()

    def terminate(self, _force: bool = False):
        for _ in range(len(self._workers)):
            self.job_queue.put(None)
        for i, worker in enumerate(self._workers):
            worker.join()
            self.logger.debug(f"[{self.name}] Worker{i} terminated.")


class ConnectionHandler:
    CLS: str = "connection_handler"

    key: int
    name: str
    constructor: PipelineConstructor
    num_workers: int
    logger: Logger
    benchmarker: Optional[Benchmarker]

    chunk_size: int
    max_image_size: int

    _job_queue: SimpleQueue
    _process: Optional[mp.Process]
    _ready_sema: Semaphore = mp.Semaphore(0)
    _is_ready = mp.Value("i", 0, lock=False)
    # To handle terminating signals only once
    _exit_sema: threading.Semaphore = threading.Semaphore(1)

    def __init__(
        self,
        key: int,
        constructor: PipelineConstructor,
        job_queue: SimpleQueue,
        num_workers: int,
        logger: Logger,
        benchmarker: Optional[Benchmarker],
        chunk_size: int,
        max_image_size: int,
        name: str = CLS,
    ):
        self.key = key
        self.constructor = constructor
        self.num_workers = num_workers
        self.logger = logger
        self.benchmarker = benchmarker
        self.name = name
        self.chunk_size = chunk_size
        self.max_image_size = max_image_size

        self._job_queue = job_queue
        self._process = None

    def start(self, warmup_image: Optional[str] = None):
        self._process = mp.Process(
            target=self._serve,
            name=self.name,
            args=(warmup_image,),
            daemon=False,
        )
        self._process.start()

    def _serve(self, warmup_image: Optional[str] = None):
        self.logger.debug(f"[{self.get_name()}] Started serving.")

        manager = PipelineManager(
            self.key,
            self.constructor,
            self.logger,
            self.benchmarker,
        )
        handler_helper = _HandlerHelper(
            manager.get_helper(),
            self.chunk_size,
            self.max_image_size,
            self.num_workers,
            self.logger,
            self.get_name(),
            self.constructor.test_mode,
        )

        manager.start()
        handler_helper.start()

        self._register_signal_handlers(handler_helper, manager)
        if warmup_image is not None:
            self._warmup(manager.helper.get_helper(), warmup_image)
        self._ready()

        while True:
            job = self._job_queue.get()
            if job is None:
                break
            handler_helper.job_queue.put(job, block=True)

    def _ready(self):
        self._is_ready.value = 1
        self._ready_sema.release()

    def wait_ready(self) -> bool:
        self._ready_sema.acquire()
        return bool(self._is_ready.value)

    @staticmethod
    def send(
        job_queue: SimpleQueue,
        job_no: int,
        start_time: float,
        conn: sock.socket,
    ):
        job_queue.put((job_no, start_time, conn))

    def _error(self, message: str, info: Any):
        self.logger.error(message)
        self.logger.debug("Cause:")
        self.logger.debug(str(info))
        self._is_ready.value = 0

    def _warmup(self, helper: PipelineHelper, warmup_image: str, kill: bool = True):
        success = True
        job_no = 0

        self.logger.debug(f"[{self.get_name()}] Warming up the pipeline...")
        img = np.asarray(Image.open(warmup_image))

        # Detect UI elements.
        helper.send(PipelineConstructor.DETECTOR, job_no, img)
        detected = helper.wait_result()
        self.logger.debug(
            f"[{self.get_name()}] ({PipelineConstructor.DETECTOR}) PASSED."
        )

        # Match UI elements.
        helper.send(PipelineConstructor.MATCHER, job_no, detected, detected)
        matched = helper.wait_result()
        if len(matched) == len(detected):
            self.logger.debug(
                f"[{self.get_name()}] ({PipelineConstructor.MATCHER}) PASSED."
            )
        else:
            success = False
            self.logger.debug(
                f"[{self.get_name()}] ({PipelineConstructor.MATCHER}) FAILED."
            )
            self._error(
                "Failed to initialize the pipeline.",
                {"detected": len(detected), "matched": len(matched)},
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
        helper.send(PipelineConstructor.TEXT_RECOGNIZER, job_no, text_elems)
        helper.send(PipelineConstructor.ICON_LABELLER, job_no, icon_elems)
        helper.wait_result()
        helper.wait_result()
        self.logger.debug(
            f"[{self.get_name()}] ({PipelineConstructor.TEXT_RECOGNIZER}) PASSED."
        )
        self.logger.debug(
            f"[{self.get_name()}] ({PipelineConstructor.ICON_LABELLER}) PASSED."
        )

        if success:
            self.logger.debug(f"[{self.get_name()}] Warm-up complete.")
        else:
            self.logger.debug(f"[{self.get_name()}] Sending termination signal.")
            self._ready_sema.release()
            if kill:
                os.kill(os.getpid(), signal.SIGTERM)

    def _register_signal_handlers(
        self, helper: _HandlerHelper, manager: PipelineManager
    ):
        term_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)

        def _exit(signum: int, _):
            if not self._exit_sema.acquire(blocking=False):
                return
            self.logger.debug(
                f"[{self.get_name()}] Termination signal received: {signal.Signals(signum).name}"
            )
            helper.terminate(True)
            manager.terminate(True)
            sys.exit(0)

        for sig in term_signals:
            signal.signal(sig, _exit)

    def get_name(self) -> str:
        return self.name + str(self.key)

    def terminate(self, force: bool = False):
        if self._process is None:
            raise ValueError("The handler process has not started.")
        if force:
            self._process.terminate()
        else:
            self._job_queue.put(None)
        self._process.join()
        self.logger.debug(f"[{self.get_name()}] Terminated.")
