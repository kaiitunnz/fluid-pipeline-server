import json
import logging
import os
import signal
import socket as sock
import sys
import time
from io import BytesIO
from multiprocessing.pool import Pool, ThreadPool
from queue import SimpleQueue
from typing import Any, Dict, List, Optional

import numpy as np
from fluid_ai.base import UiElement
from fluid_ai.pipeline import UiDetectionPipeline
from PIL import Image, ImageFile

from src.benchmark import Benchmarker
from src.sequential.manager import PipelineManager
from src.utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_BENCHMARK_FILE = "benchmark.csv"


def _handle_connection(
    conn: sock.socket,
    pipeline_ch: SimpleQueue,
    chunk_size: int,
    max_image_size: int,
    logger: logging.Logger,
    benchmarker: Optional[Benchmarker],
    start_time: float,
):
    waiting_time = time.time() - start_time  # bench

    try:
        addr = conn.getpeername()
    except Exception as e:
        logger.error(
            f"The following exception occurred while attempting to connect: {e}"
        )
        return

    result_queue = SimpleQueue()

    try:
        packet_size = int.from_bytes(readall(conn, 4, chunk_size), "big", signed=False)
        log_debug(logger, addr, f"Receiving a packet of size {packet_size} bytes.")
        if packet_size > max_image_size:
            log_error(
                logger, addr, f"The packet size exceeds the maximum allowable size."
            )
            conn.close()
            return

        data = readall(conn, packet_size, chunk_size)
        log_debug(logger, addr, f"Received a packet of size {len(data)} bytes.")
        packet = BytesIO(data)
        screenshot_img = np.asarray(Image.open(packet))
        log_debug(logger, addr, f"Received an image of shape {screenshot_img.shape}.")

        # Process the screenshot.
        log_debug(logger, addr, "Processing UI elements.")
        processing_start = time.time()  # bench
        pipeline_ch.put((result_queue, (screenshot_img,)))
        results: List[UiElement] = result_queue.get()
        processing_time = time.time() - processing_start  # bench
        log_debug(logger, addr, f"Found {len(results)} UI elements.")

        if benchmarker is None:
            results_json = _ui_to_json(screenshot_img, results).encode("utf-8")
        else:
            entry = [waiting_time, processing_time]  # type: ignore
            benchmarker.add(entry)
            metrics = {"keys": benchmarker.metrics, "values": entry}
            results_json = _ui_to_json(screenshot_img, results, metrics=metrics).encode(
                "utf-8"
            )

        log_debug(
            logger,
            addr,
            f"Sending back the response of size {len(results_json)} bytes.",
        )

        conn.sendall(len(results_json).to_bytes(4, "big", signed=False))
        log_debug(logger, addr, f"Sent response size: {len(results_json)}")
        conn.sendall(results_json)
        log_info(logger, addr, "Response sent.")
    except Exception as e:
        log_error(logger, addr, f"The following exception occurred: {e}")
    finally:
        log_info(logger, addr, "Connection closed.")
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


class PipelineServer:
    hostname: str
    port: str
    pipeline: UiDetectionPipeline
    chunk_size: int
    max_image_size: int
    num_workers: int
    socket: Optional[sock.socket] = None
    verbose: bool
    logger: logging.Logger
    benchmarker: Optional[Benchmarker]
    manager: PipelineManager

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
        benchmark: bool = False,
        benchmark_file: Optional[str] = None,
    ):
        self.hostname = hostname
        self.port = port
        self.pipeline = pipeline
        self.chunk_size = chunk_size
        self.max_image_size = max_image_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.logger = PipelineServer._init_logger(verbose)
        self.manager = PipelineManager(pipeline, self.logger)

        benchmark_metrics = [
            "Waiting time",
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

        if warmup_image is not None:
            self.warmup(warmup_image)

        self.socket = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
        self.socket.bind((self.hostname, self.port))
        self.socket.listen(1)
        self.manager.start()

        with ThreadPool(processes=self.num_workers) as pool:
            self._register_signal_handlers(pool)
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
                        self.manager.pipeline_ch,
                        self.chunk_size,
                        self.max_image_size,
                        self.logger,
                        self.benchmarker,
                        time.time(),
                    ),
                )

    def warmup(self, sample_file: str):
        self.logger.info("Warming up the pipeline...")
        sample = np.asarray(Image.open(sample_file))
        self.pipeline.detect([sample])

    def _register_signal_handlers(self, pool: Pool):
        term_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)

        def _exit(signum: int, _):
            self.logger.info(
                f"Termination signal received: {signal.Signals(signum).name}"
            )
            pool.close()
            pool.join()
            self.manager.terminate(force=True)
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
