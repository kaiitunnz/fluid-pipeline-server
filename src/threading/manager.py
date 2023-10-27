import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
from fluid_ai.base import UiElement
from fluid_ai.icon import BaseIconLabeller
from fluid_ai.ocr import BaseOCR
from fluid_ai.pipeline import UiDetectionPipeline
from fluid_ai.ui.detection import BaseUiDetector

from src.benchmark import Benchmarker
from src.threading.helper import PipelineHelper, PipelineManagerHelper
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
    helper: PipelineManagerHelper
    workers: Dict[str, Worker]
    logger: logging.Logger

    def __init__(
        self,
        pipeline: UiDetectionPipeline,
        logger: logging.Logger,
        benchmarker: Optional[Benchmarker],
    ):
        self.pipeline = pipeline
        self.helper = PipelineManagerHelper(pipeline, logger, benchmarker)
        self.workers = OrderedDict()
        self.logger = logger

    def start(self):
        name = "detector"
        self.workers[name] = Worker(
            detect,
            self.helper.detector_ch,
            self.pipeline.detector,
            self.logger,
            name,
        )

        name = "text_recognizer"
        self.workers[name] = Worker(
            recognize_texts,
            self.helper.text_recognizer_ch,
            self.pipeline.text_recognizer,
            self.logger,
            name,
        )

        name = "icon_labeller"
        self.workers[name] = Worker(
            label_icons,
            self.helper.icon_labeller_ch,
            self.pipeline.icon_labeller,
            self.logger,
            name,
        )

        for worker in self.workers.values():
            worker.start()

    def get_helper(self) -> PipelineHelper:
        return self.helper.get_helper()

    def terminate(self, force: bool = False):
        self.logger.info("Terminating the worker processes...")
        for worker in self.workers.values():
            worker.terminate(force)
            self.logger.info(f"'{worker.name}' worker has terminated.")
