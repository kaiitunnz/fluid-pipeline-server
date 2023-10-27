import os
from PIL import Image
from logging import Logger
from queue import SimpleQueue
from typing import Callable, List, Optional, Tuple

import numpy as np
from fluid_ai.base import UiElement
from fluid_ai.pipeline import UiDetectionPipeline

from src.benchmark import Benchmarker


class PipelineManagerHelper:
    detector_ch: SimpleQueue
    text_recognizer_ch: SimpleQueue
    icon_labeller_ch: SimpleQueue

    textual_elements: List[str]
    icon_elements: List[str]

    logger: Logger
    benchmarker: Optional[Benchmarker]

    _count: int

    def __init__(
        self,
        pipeline: UiDetectionPipeline,
        logger: Logger,
        benchmarker: Optional[Benchmarker],
    ):
        self.detector_ch = SimpleQueue()
        self.text_recognizer_ch = SimpleQueue()
        self.icon_labeller_ch = SimpleQueue()

        self.textual_elements = pipeline.textual_elements
        self.icon_elements = pipeline.icon_elements

        self.logger = logger
        self.benchmarker = benchmarker

        self._count = 0

    def get_helper(self) -> "PipelineHelper":
        key = self._count
        self._count += 1
        return PipelineHelper(
            key,
            self.detector_ch,
            self.text_recognizer_ch,
            self.icon_labeller_ch,
            self.textual_elements,
            self.icon_elements,
            self.logger,
            self.benchmarker,
        )


class PipelineHelper:
    key: int

    detector_ch: SimpleQueue
    text_recognizer_ch: SimpleQueue
    icon_labeller_ch: SimpleQueue

    textual_elements: List[str]
    icon_elements: List[str]

    logger: Logger
    benchmarker: Optional[Benchmarker]

    _channel: SimpleQueue

    def __init__(
        self,
        key: int,
        detector_ch: SimpleQueue,
        text_recognizer_ch: SimpleQueue,
        icon_labeller_ch: SimpleQueue,
        textual_elements: List[str],
        icon_elements: List[str],
        logger: Logger,
        benchmarker: Optional[Benchmarker],
    ):
        self.key = key
        self.detector_ch = detector_ch
        self.text_recognizer_ch = text_recognizer_ch
        self.icon_labeller_ch = icon_labeller_ch
        self.textual_elements = textual_elements
        self.icon_elements = icon_elements
        self.logger = logger
        self.benchmarker = benchmarker
        self._channel = SimpleQueue()

    def detect(self, image: np.ndarray):
        self.detector_ch.put((self._channel, (image,)))

    def recognize_text(self, elements: List[UiElement]):
        self.text_recognizer_ch.put((self._channel, (elements,)))

    def label_icons(self, elements: List[UiElement]):
        self.icon_labeller_ch.put((self._channel, (elements,)))

    def wait_result(self) -> List[UiElement]:
        return self._channel.get()

    def save_image(self, img: Image.Image, save_dir: str, prefix: str = "img"):
        img.save(os.path.join(save_dir, f"{prefix}{self.key}.jpg"))

    @staticmethod
    def _log(log_func: Callable[[str], None], addr: Tuple[str, int], msg: str):
        ip, port = addr
        log_func(f"({ip}:{port}) {msg}")

    def log_debug(self, addr: Tuple[str, int], msg: str):
        PipelineHelper._log(self.logger.debug, addr, msg)

    def log_info(self, addr: Tuple[str, int], msg: str):
        PipelineHelper._log(self.logger.info, addr, msg)

    def log_warning(self, addr: Tuple[str, int], msg: str):
        PipelineHelper._log(self.logger.warn, addr, msg)

    def log_error(self, addr: Tuple[str, int], msg: str):
        PipelineHelper._log(self.logger.error, addr, msg)

    def log_critical(self, addr: Tuple[str, int], msg: str):
        PipelineHelper._log(self.logger.critical, addr, msg)
