import logging
import os
import signal
import socket as sock
import sys
import time
from io import BytesIO
from multiprocessing.pool import Pool, ThreadPool
from typing import List, Optional

import numpy as np
from fluid_ai.base import UiElement
from PIL import Image, ImageFile  # type: ignore

from src.benchmark import Benchmarker
from src.constructor import PipelineConstructor
from src.pipeline import PipelineServerInterface
from src.threading.helper import PipelineHelper
from src.threading.manager import PipelineManager
from src.utils import readall, ui_to_json

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_BENCHMARK_FILE = "benchmark.csv"
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
        for elem in detected:
            if elem.name in helper.textual_elements:
                text_elems.append(elem)
            elif elem.name in helper.icon_elements:
                icon_elems.append(elem)
            else:
                results.append(elem)

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
            results_json = ui_to_json(screenshot_img, results).encode("utf-8")
        else:
            entry = [waiting_time, detection_time, text_time, icon_time, processing_time]  # type: ignore
            helper.benchmarker.add(entry)
            metrics = {"keys": helper.benchmarker.metrics, "values": entry}
            results_json = ui_to_json(screenshot_img, results, metrics=metrics).encode(
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


class PipelineServer(PipelineServerInterface):
    hostname: str
    port: str
    pipeline: PipelineConstructor
    chunk_size: int
    max_image_size: int
    num_workers: int
    socket: Optional[sock.socket] = None
    verbose: bool
    logger: logging.Logger
    benchmarker: Optional[Benchmarker]

    def __init__(
        self,
        *,
        hostname: str,
        port: str,
        pipeline: PipelineConstructor,
        chunk_size: int = -1,
        max_image_size: int = -1,
        num_workers: int = 4,
        num_instances: int = 1,
        verbose: bool = True,
        benchmark: bool = False,
        benchmark_file: Optional[str] = None,
    ):
        self.hostname = hostname
        self.port = port
        self.pipeline = pipeline
        self.chunk_size = chunk_size
        self.max_image_size = max_image_size
        self.num_workers = num_workers
        self.num_instances = num_instances
        self.verbose = verbose
        self.logger = PipelineServer._init_logger(verbose)

        benchmark_metrics = [
            "Waiting time",
            "UI detection time",
            "Text recognition time",
            "Icon labeling time",
            "Processing time",
        ]
        self.benchmarker = (
            Benchmarker(
                benchmark_metrics,
                benchmark_file or DEFAULT_BENCHMARK_FILE,
            )
            if benchmark
            else None
        )

    def start(self, warmup_image: Optional[str] = None):
        self.logger.info("Starting the pipeline server...")

        self.socket = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
        self.socket.bind((self.hostname, self.port))
        self.socket.listen(1)

        manager = PipelineManager(
            self.pipeline, self.logger, self.benchmarker, self.num_instances
        )
        manager.start()

        with ThreadPool(processes=self.num_workers) as pool:
            self._register_signal_handlers(pool, manager)
            if warmup_image is not None:
                self._warmup(manager.helper.get_helper(), warmup_image)
            self.logger.info(
                f'Pipeline server started serving at "{self.hostname}:{self.port} (PID={os.getpid()})".'
            )
            while True:
                conn, addr = self.socket.accept()
                self.logger.info(f'Got connection from "{addr[0]}:{addr[1]}"')
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

    def _warmup(self, helper: PipelineHelper, warmup_image: str):
        self.logger.info("Warming up the pipeline...")
        img = np.asarray(Image.open(warmup_image))

        for i in range(self.num_instances):
            # Detect UI elements.
            helper.sendi(PipelineConstructor.DETECTOR, i, img)
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
            helper.sendi(PipelineConstructor.TEXT_RECOGNIZER, i, text_elems)
            helper.sendi(PipelineConstructor.ICON_LABELLER, i, icon_elems)
            helper.wait_result()
            helper.wait_result()

    def _register_signal_handlers(self, pool: Pool, manager: PipelineManager):
        term_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)

        def _exit(signum: int, _):
            self.logger.info(
                f"Termination signal received: {signal.Signals(signum).name}"
            )
            pool.close()
            pool.join()
            manager.terminate(force=True)
            self.logger.info("Server successfully exited.")
            sys.exit(0)

        for sig in term_signals:
            signal.signal(sig, _exit)

    @classmethod
    def _init_logger(cls, verbose: bool = True) -> logging.Logger:
        fmt = "[%(asctime)s | %(name)s] [%(levelname)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        logging.basicConfig(format=fmt, datefmt=datefmt)
        logger = logging.getLogger(cls.__name__)
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        return logger
