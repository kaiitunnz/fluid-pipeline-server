import json
import multiprocessing as mp
import numpy as np
import signal
import socket as sock
import sys
import threading
import time
from PIL import Image
from io import BytesIO
from multiprocessing.pool import ThreadPool
from multiprocessing.queues import SimpleQueue
from multiprocessing.synchronize import Semaphore
from typing import Any, Dict, List, Optional

from fluid_ai.base import UiElement

from src.constructor import PipelineConstructor
from src.hybrid.benchmark import Benchmarker
from src.hybrid.helper import PipelineHelper
from src.hybrid.logging import Logger
from src.hybrid.manager import PipelineManager
from src.utils import readall


SAVE_IMG_DIR = "res"
SAVE_IMG = False


def _handle_connection(
    conn: sock.socket,
    helper: PipelineHelper,
    chunk_size: int,
    max_image_size: int,
    start_time: float,
):
    waiting_time = time.time() - start_time  # bench

    try:
        addr = conn.getpeername()
    except Exception as e:
        helper.logger.error(
            f"The following exception occurred while attempting to connect: {e}"
        )
        return

    try:
        packet_size = int.from_bytes(readall(conn, 4, chunk_size), "big", signed=False)
        helper.log_debug(addr, f"Receiving a packet of size {packet_size} bytes.")
        if packet_size > max_image_size:
            helper.log_error(
                addr, f"The packet size exceeds the maximum allowable size."
            )
            conn.close()
            return

        data = readall(conn, packet_size, chunk_size)
        helper.log_debug(addr, f"Received a packet of size {len(data)} bytes.")
        packet = BytesIO(data)
        img = Image.open(packet)
        if SAVE_IMG:
            helper.save_image(img, SAVE_IMG_DIR)
        screenshot_img = np.asarray(img)
        helper.log_debug(addr, f"Received an image of shape {screenshot_img.shape}.")

        # Detect UI elements.
        helper.log_debug(addr, "Detecting UI elements.")
        detection_start = time.time()  # bench
        helper.send(PipelineConstructor.DETECTOR, screenshot_img)
        detected = helper.wait_result()
        detection_time = time.time() - detection_start  # bench
        helper.log_debug(addr, f"Found {len(detected)} UI elements.")

        # Partition the result.
        text_elems = []
        icon_elems = []
        results: List[UiElement] = []
        for e in detected:
            if e.name in helper.textual_elements:
                text_elems.append(e)
            elif e.name in helper.icon_elements:
                icon_elems.append(e)
            else:
                results.append(e)

        # Extract UI info.
        helper.log_debug(addr, "Extracting UI info.")
        if helper.benchmarker is None:
            helper.send(PipelineConstructor.TEXT_RECOGNIZER, text_elems)
            helper.send(PipelineConstructor.ICON_LABELLER, icon_elems)
            results.extend(helper.wait_result())
            results.extend(helper.wait_result())
        else:
            text_start = time.time()  # bench
            helper.send(PipelineConstructor.TEXT_RECOGNIZER, text_elems)
            results.extend(helper.wait_result())
            text_time = time.time() - text_start  # bench
            icon_start = time.time()  # bench
            helper.send(PipelineConstructor.ICON_LABELLER, icon_elems)
            results.extend(helper.wait_result())
            icon_time = time.time() - icon_start  # bench

        processing_time = time.time() - detection_start  # bench
        if helper.benchmarker is None:
            results_json = _ui_to_json(screenshot_img, results).encode("utf-8")
        else:
            entry = [waiting_time, detection_time, text_time, icon_time, processing_time]  # type: ignore
            helper.benchmarker.add(entry)
            metrics = {"keys": helper.benchmarker.metrics, "values": entry}
            results_json = _ui_to_json(screenshot_img, results, metrics=metrics).encode(
                "utf-8"
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


def _ui_to_json(screenshot_img: np.ndarray, elems: List[UiElement], **kwargs) -> str:
    h, w, *_ = screenshot_img.shape
    data = {"img_shape": [w, h], "elements": [_elem_to_dict(e) for e in elems]}
    data.update(kwargs)
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


class ConnectionHandler:
    CLS: str = "connection_handler"

    key: int
    name: str
    constructor: PipelineConstructor
    num_workers: int
    logger: Logger
    benchmarker: Optional[Benchmarker]
    job_queue: SimpleQueue

    chunk_size: int
    max_image_size: int

    _process: Optional[mp.Process]
    _ready_sema: Semaphore = mp.Semaphore(0)
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
        self.job_queue = job_queue
        self.num_workers = num_workers
        self.logger = logger
        self.benchmarker = benchmarker
        self.name = name
        self.chunk_size = chunk_size
        self.max_image_size = max_image_size

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
        manager.start()

        with ThreadPool(processes=self.num_workers) as pool:
            self._register_signal_handlers(pool, manager)
            if warmup_image is not None:
                self._warmup(manager.helper.get_helper(), warmup_image)
            self._ready()
            while True:
                job = self.job_queue.get()
                if job is None:
                    break
                if not isinstance(job, sock.socket):
                    raise ValueError(
                        f"Invalid job. Expected a socket connection. Got {type(job)} instead."
                    )
                conn = job
                pool.apply_async(
                    _handle_connection,
                    args=(
                        conn,
                        manager.get_helper(),
                        self.chunk_size,
                        self.max_image_size,
                        time.time(),
                    ),
                )

    def _ready(self):
        self._ready_sema.release()

    def wait_ready(self):
        self._ready_sema.acquire()

    def _warmup(self, helper: PipelineHelper, warmup_image: str):
        self.logger.debug(f"[{self.get_name()}] Warming up the pipeline...")
        img = np.asarray(Image.open(warmup_image))

        # Detect UI elements.
        helper.send(PipelineConstructor.DETECTOR, img)
        detected = helper.wait_result()

        # Partition the result.
        text_elems = []
        icon_elems = []
        for e in detected:
            if e.name in helper.textual_elements:
                text_elems.append(e)
            elif e.name in helper.icon_elements:
                icon_elems.append(e)

        # Extract UI info.
        helper.send(PipelineConstructor.TEXT_RECOGNIZER, text_elems)
        helper.send(PipelineConstructor.ICON_LABELLER, icon_elems)
        helper.wait_result()
        helper.wait_result()

        self.logger.debug(f"[{self.get_name()}] Warm-up complete.")

    def _register_signal_handlers(self, pool: ThreadPool, manager: PipelineManager):
        term_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)

        def _exit(signum: int, _):
            if not self._exit_sema.acquire(blocking=False):
                return
            self.logger.debug(
                f"[{self.get_name()}] Termination signal received: {signal.Signals(signum).name}"
            )
            pool.close()
            pool.join()
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
            self.job_queue.put(None)
        self._process.join()
        self.logger.debug(f"[{self.get_name()}] Terminated.")
