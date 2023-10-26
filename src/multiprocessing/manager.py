import logging
from queue import Queue
from typing import Any, List

from fluid_ai.base import UiElement
from fluid_ai.icon import BaseIconLabeller
from fluid_ai.ocr import BaseOCR
from fluid_ai.pipeline import UiDetectionPipeline
from fluid_ai.ui.detection import BaseUiDetector
from multiprocessing.managers import DictProxy, SyncManager
from threading import Condition
from torch import Tensor

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


class PipelineManagerHelper:
    manager: SyncManager

    detector_ch: Queue
    text_recognizer_ch: Queue
    icon_labeller_ch: Queue

    detector_pool: DictProxy
    text_recognizer_pool: DictProxy
    icon_labeller_pool: DictProxy

    _count: int
    _conditions: DictProxy

    def __init__(self, manager: SyncManager):
        self.manager = manager

        self.detector_ch = manager.Queue()
        self.text_recognizer_ch = manager.Queue()
        self.icon_labeller_ch = manager.Queue()

        self.detector_pool = manager.dict()
        self.text_recognizer_pool = manager.dict()
        self.icon_labeller_pool = manager.dict()

        self._count = 0
        self._conditions = manager.dict()

    def clone(self) -> "PipelineHelper":
        key = self._count
        self._count += 1

        condition = self.manager.Condition()
        self._conditions[key] = condition

        return PipelineHelper(
            key,
            self.detector_ch,
            self.text_recognizer_ch,
            self.icon_labeller_ch,
            self.detector_pool,
            self.text_recognizer_pool,
            self.icon_labeller_pool,
            condition,
            self._conditions,
        )


class PipelineHelper:
    key: int
    detector_ch: Queue
    text_recognizer_ch: Queue
    icon_labeller_ch: Queue

    detector_pool: DictProxy
    text_recognizer_pool: DictProxy
    icon_labeller_pool: DictProxy

    condition: Condition
    manager_conditions: DictProxy

    def __init__(
        self,
        key: int,
        detector_ch: Queue,
        text_recognizer_ch: Queue,
        icon_labeller_ch: Queue,
        detector_pool: DictProxy,
        text_recognizer_pool: DictProxy,
        icon_labeller_pool: DictProxy,
        condition: Condition,
        manager_conditions: DictProxy,
    ):
        self.key = key
        self.detector_ch = detector_ch
        self.text_recognizer_ch = text_recognizer_ch
        self.icon_labeller_ch = icon_labeller_ch
        self.detector_pool = detector_pool
        self.text_recognizer_pool = text_recognizer_pool
        self.icon_labeller_pool = icon_labeller_pool
        self.condition = condition
        self.manager_conditions = manager_conditions

    def detect(self, screenshot: Tensor):
        self.detector_ch.put((self.key, self.condition, (screenshot,)))

    def wait_detect(self) -> List[UiElement]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.detector_pool.get(self.key, None) is not None
            )
            return self.detector_pool.pop(self.key)

    def recognize_text(self, elements: List[UiElement]):
        self.text_recognizer_ch.put((self.key, self.condition, (elements,)))

    def wait_recognize_text(self) -> List[UiElement]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.text_recognizer_pool.get(self.key, None) is not None
            )
            return self.text_recognizer_pool.pop(self.key)

    def label_icons(self, elements: List[UiElement]):
        self.icon_labeller_ch.put((self.key, self.condition, (elements,)))

    def wait_label_icons(self) -> List[UiElement]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.icon_labeller_pool.get(self.key, None) is not None
            )
            return self.icon_labeller_pool.pop(self.key)

    def __del__(self):
        self.manager_conditions.pop(self.key, None)


class PipelineManager:
    pipeline: UiDetectionPipeline
    detector: BaseUiDetector
    text_recognizer: BaseOCR
    icon_labeller: BaseIconLabeller

    detector_worker: Worker
    text_recognizer_worker: Worker
    icon_labeller_worker: Worker

    _helper: PipelineManagerHelper

    logger: logging.Logger

    def __init__(
        self,
        pipeline: UiDetectionPipeline,
        manager: SyncManager,
        logger: logging.Logger,
    ):
        self.pipeline = pipeline
        self.detector = pipeline.detector
        self.text_recognizer = pipeline.text_recognizer
        self.icon_labeller = pipeline.icon_labeller
        self._helper = PipelineManagerHelper(manager)
        self.logger = logger

    def start(self):
        self.detector_worker = Worker(
            detect,
            self.detector,
            self._helper.detector_ch,
            self._helper.detector_pool,
            self.logger,
            "detector",
        )
        self.text_recognizer_worker = Worker(
            recognize_texts,
            self.text_recognizer,
            self._helper.text_recognizer_ch,
            self._helper.text_recognizer_pool,
            self.logger,
            "text_recognizer",
        )
        self.icon_labeller_worker = Worker(
            label_icons,
            self.icon_labeller,
            self._helper.icon_labeller_ch,
            self._helper.icon_labeller_pool,
            self.logger,
            "icon_labeller",
        )

        self.detector_worker.start()
        self.text_recognizer_worker.start()
        self.icon_labeller_worker.start()

    def get_helper(self) -> PipelineHelper:
        return self._helper.clone()

    def terminate(self, force: bool = False):
        self.detector_worker.terminate(force)
        self.text_recognizer_worker.terminate(force)
        self.icon_labeller_worker.terminate(force)

    def __del__(self):
        self.terminate(force=True)
