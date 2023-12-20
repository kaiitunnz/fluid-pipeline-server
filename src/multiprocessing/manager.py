import logging
from typing import Any, List, Optional

from fluid_ai.base import UiElement
from fluid_ai.icon import BaseIconLabeller
from fluid_ai.ocr import BaseOCR
from fluid_ai.ui.detection import BaseUiDetector
from multiprocessing.managers import SyncManager
from torch import Tensor

import src.benchmark as bench
from src.multiprocessing.benchmark import BenchmarkListener
from src.constructor import PipelineConstructor
from src.multiprocessing.helper import PipelineHelper, PipelineManagerHelper
from src.multiprocessing.logging import LogListener
from src.multiprocessing.worker import Worker
from src.pipeline import PipelineModule


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

    benchmark_listener: Optional[BenchmarkListener]

    def __init__(
        self,
        pipeline: PipelineConstructor,
        manager: SyncManager,
        logger: logging.Logger,
        benchmarker: Optional[bench.Benchmarker],
    ):
        self.pipeline = pipeline
        self.log_listener = LogListener(logger, manager.Queue())
        self.logger = logger
        self.benchmark_listener = (
            None
            if benchmarker is None
            else BenchmarkListener(benchmarker, manager.Queue(), logger)
        )
        self._helper = PipelineManagerHelper(
            pipeline, manager, self.log_listener, self.benchmark_listener
        )

    def start(self):
        logger = self.log_listener.get_logger()

        self.detector_worker = Worker(
            detect,
            self.pipeline.modules[PipelineModule.DETECTOR],
            self._helper.detector_ch,
            self._helper.detector_pool,
            logger,
            PipelineModule.DETECTOR,
        )
        self.text_recognizer_worker = Worker(
            recognize_texts,
            self.pipeline.modules[PipelineModule.TEXT_RECOGNIZER],
            self._helper.text_recognizer_ch,
            self._helper.text_recognizer_pool,
            logger,
            PipelineModule.TEXT_RECOGNIZER,
        )
        self.icon_labeller_worker = Worker(
            label_icons,
            self.pipeline.modules[PipelineModule.ICON_LABELLER],
            self._helper.icon_labeller_ch,
            self._helper.icon_labeller_pool,
            logger,
            PipelineModule.ICON_LABELLER,
        )

        self.log_listener.start()
        if self.benchmark_listener is not None:
            self.benchmark_listener.start()
        self.detector_worker.start()
        self.text_recognizer_worker.start()
        self.icon_labeller_worker.start()

    def get_helper(self) -> PipelineHelper:
        return self._helper.get_helper()

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

        optional_workers = (self.benchmark_listener,)
        for worker in optional_workers:
            if worker is not None:
                worker.terminate(force)
                self.logger.info(f"'{worker.name}' worker has terminated.")
