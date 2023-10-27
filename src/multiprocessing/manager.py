import logging
from typing import Any, List

from fluid_ai.base import UiElement
from fluid_ai.icon import BaseIconLabeller
from fluid_ai.ocr import BaseOCR
from fluid_ai.ui.detection import BaseUiDetector
from multiprocessing.managers import SyncManager
from torch import Tensor

from src.multiprocessing.constructor import PipelineConstructor
from src.multiprocessing.helper import PipelineHelper, PipelineManagerHelper
from src.multiprocessing.logging import LogListener
from src.multiprocessing.worker import Worker


def detect(screenshot: Tensor, module: Any) -> List[UiElement]:
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
    pipeline: PipelineConstructor

    detector_worker: Worker
    text_recognizer_worker: Worker
    icon_labeller_worker: Worker

    _helper: PipelineManagerHelper

    log_listener: LogListener
    logger: logging.Logger

    def __init__(
        self,
        pipeline: PipelineConstructor,
        manager: SyncManager,
        logger: logging.Logger,
    ):
        self.pipeline = pipeline
        self.log_listener = LogListener(logger, manager.Queue())
        self._helper = PipelineManagerHelper(pipeline, manager, self.log_listener)
        self.logger = logger

    def start(self):
        logger = self.log_listener.get_logger()

        self.detector_worker = Worker(
            detect,
            self.pipeline.detector,
            self._helper.detector_ch,
            self._helper.detector_pool,
            logger,
            "detector",
        )
        self.text_recognizer_worker = Worker(
            recognize_texts,
            self.pipeline.text_recognizer,
            self._helper.text_recognizer_ch,
            self._helper.text_recognizer_pool,
            logger,
            "text_recognizer",
        )
        self.icon_labeller_worker = Worker(
            label_icons,
            self.pipeline.icon_labeller,
            self._helper.icon_labeller_ch,
            self._helper.icon_labeller_pool,
            logger,
            "icon_labeller",
        )

        self.log_listener.start()
        self.detector_worker.start()
        self.text_recognizer_worker.start()
        self.icon_labeller_worker.start()

    def get_helper(self) -> PipelineHelper:
        return self._helper.clone()

    def terminate(self, force: bool = False):
        self.logger.info("Terminating the worker processes...")
        workers = (
            self.detector_worker,
            self.text_recognizer_worker,
            self.icon_labeller_worker,
            self.log_listener,
        )
        for worker in workers:
            worker.terminate(force)
            self.logger.info(f"'{worker.name}' worker has terminated.")
