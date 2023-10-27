import logging
from queue import SimpleQueue
from typing import Any, List

import numpy as np
from fluid_ai.base import UiElement
from fluid_ai.icon import BaseIconLabeller
from fluid_ai.ocr import BaseOCR
from fluid_ai.pipeline import UiDetectionPipeline
from fluid_ai.ui.detection import BaseUiDetector

from src.threading.worker import Worker


def detect(screenshot: np.ndarray, module: Any) -> List[UiElement]:
    assert isinstance(module, BaseUiDetector)
    return next(module.detect([screenshot]))


def recognize_texts(
    elements: List[UiElement],
    module: Any,
) -> List[UiElement]:
    assert isinstance(module, BaseOCR)
    module.process(elements)
    return elements


def label_icons(
    elements: List[UiElement],
    module: Any,
) -> List[UiElement]:
    assert isinstance(module, BaseIconLabeller)
    module.process(elements)
    return elements


class PipelineManager:
    pipeline: UiDetectionPipeline
    detector: BaseUiDetector
    text_recognizer: BaseOCR
    icon_labeller: BaseIconLabeller

    detector_ch: SimpleQueue
    text_recognizer_ch: SimpleQueue
    icon_labeller_ch: SimpleQueue

    detector_worker: Worker
    text_recognizer_worker: Worker
    icon_labeller_worker: Worker

    logger: logging.Logger

    def __init__(self, pipeline: UiDetectionPipeline, logger: logging.Logger):
        self.pipeline = pipeline
        self.detector = pipeline.detector
        self.text_recognizer = pipeline.text_recognizer
        self.icon_labeller = pipeline.icon_labeller
        self.logger = logger

    def start(self):
        self.detector_ch = SimpleQueue()
        self.detector_worker = Worker(
            detect, self.detector_ch, self.detector, self.logger, "detector"
        )
        self.text_recognizer_ch = SimpleQueue()
        self.text_recognizer_worker = Worker(
            recognize_texts,
            self.text_recognizer_ch,
            self.text_recognizer,
            self.logger,
            "text_recognizer",
        )
        self.icon_labeller_ch = SimpleQueue()
        self.icon_labeller_worker = Worker(
            label_icons,
            self.icon_labeller_ch,
            self.icon_labeller,
            self.logger,
            "icon_labeller",
        )

        self.detector_worker.start()
        self.text_recognizer_worker.start()
        self.icon_labeller_worker.start()

    def terminate(self, force: bool = False):
        self.logger.info("Terminating the worker processes...")
        self.detector_worker.terminate(force)
        self.text_recognizer_worker.terminate(force)
        self.icon_labeller_worker.terminate(force)
        self.logger.info("Done.")
